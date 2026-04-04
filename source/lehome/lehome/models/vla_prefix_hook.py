import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

_SUBMISSION_DIR = Path(__file__).resolve().parents[4] / "submission" / "source_code" / "lerobot_policies_smolvla"
if str(_SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(_SUBMISSION_DIR))

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import (
    VLAFlowMatching,
    make_att_2d_masks,
    pad_vector,
    resize_with_pad,
)
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel


class VLAPrefixHook:
    def __init__(
        self,
        pretrained_path: str | None = None,
        device: str = "cpu",
        task_description: str = "fold the garment",
        image_keys: list[str] | None = None,
        state_dim: int = 12,
    ):
        self.device = torch.device(device)
        self.task_description = task_description
        self.image_keys = image_keys or [
            "observation.images.top_rgb",
            "observation.images.left_rgb",
            "observation.images.right_rgb",
        ]
        self.state_dim = state_dim

        self.config = self._build_config()
        self.model = VLAFlowMatching(self.config)
        self.model.eval()

        if pretrained_path is not None:
            self._load_checkpoint(pretrained_path)

        self.model.to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False

        self._lang_tokens, self._lang_masks = self._tokenize_language()
        self._num_lang_tokens = int(self._lang_masks.sum().item())

    def _build_config(self) -> SmolVLAConfig:
        input_features = {
            "observation.state": {"type": "STATE", "shape": [self.state_dim]},
        }
        for key in self.image_keys:
            input_features[key] = {"type": "VISUAL", "shape": [3, 480, 640]}
        return SmolVLAConfig(
            input_features=input_features,
            output_features={"action": {"type": "ACTION", "shape": [self.state_dim]}},
            device="cpu",
            n_action_steps=self.state_dim,
            chunk_size=50,
            resize_imgs_with_padding=(512, 512),
            max_state_dim=32,
            max_action_dim=32,
            train_state_proj=False,
            freeze_vision_encoder=True,
            train_expert_only=True,
        )

    def _load_checkpoint(self, pretrained_path: str):
        path = Path(pretrained_path)
        if path.is_dir():
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

            policy = SmolVLAPolicy.from_pretrained(str(path))
            state_dict = policy.model.state_dict()
            result = self.model.load_state_dict(state_dict, strict=False)
            if result.missing_keys or result.unexpected_keys:
                print(f"  ⚠️ load_state_dict: missing={result.missing_keys}, unexpected={result.unexpected_keys}")
            else:
                print(f"  ✅ All checkpoint weights loaded (no missing/unexpected keys)")
        elif path.suffix in (".pt", ".safetensors"):
            state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            result = self.model.load_state_dict(state_dict, strict=False)
            if result.missing_keys or result.unexpected_keys:
                print(f"  ⚠️ load_state_dict: missing={result.missing_keys}, unexpected={result.unexpected_keys}")
            else:
                print(f"  ✅ All checkpoint weights loaded (no missing/unexpected keys)")

    def _tokenize_language(self):
        processor = self.model.vlm_with_expert.processor
        tokenized = processor.tokenizer(self.task_description, return_tensors="pt", padding=True)
        lang_tokens = tokenized["input_ids"]
        lang_masks = tokenized["attention_mask"].bool()
        return lang_tokens, lang_masks

    def prepare_images(self, images_dict: dict[str, torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        images = []
        img_masks = []
        for key in self.image_keys:
            img = images_dict.get(key)
            if img is None:
                raise ValueError(f"Missing image key: {key}")
            if img.ndim == 5:
                img = img[:, -1, :, :, :]
            img = resize_with_pad(img, 512, 512, pad_value=0)
            img = img * 2.0 - 1.0
            mask = torch.ones(img.shape[0], dtype=torch.bool, device=img.device)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    def prepare_state(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim > 2:
            state = state[:, -1, :]
        return pad_vector(state, self.config.max_state_dim)

    @torch.no_grad()
    def extract_prefix(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch["observation.state"])
        B = state.shape[0]
        lang_tokens = self._lang_tokens.expand(B, -1).to(self.device)
        lang_masks = self._lang_masks.expand(B, -1).to(self.device)
        images = [img.to(self.device) for img in images]
        img_masks = [m.to(self.device) for m in img_masks]
        state = state.to(self.device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )

        att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        outputs, _ = self.model.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
            fill_kv_cache=True,
        )

        z_vlm = outputs[0]
        return z_vlm

    @property
    def num_lang_tokens(self) -> int:
        return self._num_lang_tokens

    @property
    def num_image_tokens_per_camera(self) -> int:
        return 64

    @property
    def num_image_tokens(self) -> int:
        return 64 * len(self.image_keys)

    @property
    def num_state_tokens(self) -> int:
        return 1


class PrefixEmbeddingDataset(Dataset):
    def __init__(self, data: dict[int, torch.Tensor] | torch.Tensor, indices: list[int] | None = None):
        self.data = data
        if indices is not None:
            self.indices = indices
        elif isinstance(data, dict):
            self.indices = sorted(data.keys())
        else:
            self.indices = list(range(data.shape[0]))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[self.indices[idx]]

    @staticmethod
    def collate_fn(batch: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(batch, dim=0)


def precompute_prefix_embeddings(
    hook: VLAPrefixHook,
    lerobot_dataset,
    output_path: str,
    batch_size: int = 1,
    save_every: int = 5000,
):
    from tqdm import tqdm

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cache: dict[int, torch.Tensor] = {}
    total_frames = len(lerobot_dataset)

    for idx in tqdm(range(total_frames), desc="Precomputing prefix embeddings"):
        frame = lerobot_dataset[idx]
        batch: dict[str, torch.Tensor] = {}
        for key in hook.image_keys:
            img_key = f"observation.images.{key.split('.')[-1].replace('_rgb', '')}_rgb"
            if img_key in frame:
                tensor = frame[img_key]
                if not isinstance(tensor, torch.Tensor):
                    tensor = torch.from_numpy(tensor).float()
                if tensor.ndim == 3:
                    tensor = tensor.unsqueeze(0)
                batch[key] = tensor
            else:
                for k in frame:
                    if k.startswith("observation.images.") and k.endswith("_rgb"):
                        tensor = frame[k]
                        if not isinstance(tensor, torch.Tensor):
                            tensor = torch.from_numpy(tensor).float()
                        if tensor.ndim == 3:
                            tensor = tensor.unsqueeze(0)
                        batch[k] = tensor
                        break

        state = frame.get("observation.state", frame.get("state"))
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float()
        if state.ndim == 1:
            state = state.unsqueeze(0)
        batch["observation.state"] = state

        z_vlm = hook.extract_prefix(batch)
        cache[idx] = z_vlm.squeeze(0).cpu().half()

        if (idx + 1) % save_every == 0:
            torch.save(cache, output_path)
            print(f"  Saved {idx + 1}/{total_frames} embeddings to {output_path}")

    torch.save(cache, output_path)
    print(f"Done. Saved {total_frames} embeddings to {output_path}")
    return cache
