"""
LeHome Challenge — MoE-SmolVLA Policy.

Mixture-of-Experts SmolVLA with garment-type routing.
Sticky routing: router runs once per episode, then locks the expert.
"""

import copy
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from server import BasePolicyServer

# Garment type definitions
GARMENT_TYPES = ["top_short", "top_long", "pant_short", "pant_long"]


class GarmentRouter(nn.Module):
    """MLP classifier for garment type routing."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [512, 256, 128],
        num_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout if i < len(hidden_dims) - 1 else dropout * 0.7),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class MoEPolicy(BasePolicyServer):
    """
    MoE-SmolVLA Policy with garment-type sticky routing.

    Architecture:
        - Shared VLM backbone (frozen)
        - Shared state projection (frozen)
        - Independent lm_expert per garment type
        - Independent action projections per garment type
        - Router selects expert on first frame, locks for episode
    """

    def __init__(self):
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, resize_with_pad
        from lerobot.policies.factory import make_policy, make_pre_post_processors
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        self.resize_with_pad = resize_with_pad

        # Configuration from environment
        checkpoint_dir = Path(os.environ.get("CHECKPOINT_DIR", "/app/checkpoints"))
        dataset_root = os.environ.get("DATASET_ROOT", "/app/datasets")
        device_str = os.environ.get("DEVICE", "cuda")
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        self.task_description = "fold the garment on the table"

        # Default expert checkpoint paths
        self.expert_paths = {
            "pant_short": checkpoint_dir / "smolvla_moe_expert_pant_short_no_st_proj/checkpoints/003000/pretrained_model",
            "pant_long": checkpoint_dir / "smolvla_moe_expert_pant_long_no_st_proj/checkpoints/004000/pretrained_model",
            "top_short": checkpoint_dir / "smolvla_moe_expert_top_short_no_st_proj/checkpoints/008000/pretrained_model",
            "top_long": checkpoint_dir / "smolvla_moe_expert_top_long_no_st_proj/checkpoints/008000/pretrained_model",
        }
        router_path = checkpoint_dir / "router_all_frames/checkpoints/best/router.pt"

        # Discover available experts
        self.available_experts = {}
        self.missing_experts = []
        for gtype, path in self.expert_paths.items():
            if path.exists():
                self.available_experts[gtype] = str(path)
                print(f"  [OK] Expert for {gtype}: {path}")
            else:
                self.missing_experts.append(gtype)
                print(f"  [MISSING] Expert for {gtype}: {path}")

        if not self.available_experts:
            raise RuntimeError("No expert checkpoints found!")

        # Use first available expert as base VLM
        vlm_model_path = list(self.available_experts.values())[0]
        print(f"Loading base VLM from {vlm_model_path}")

        # Load dataset metadata for normalization stats
        try:
            meta = LeRobotDatasetMetadata(repo_id="lehome", root=dataset_root)
        except Exception as e:
            print(f"Warning: Failed to load dataset metadata: {e}. Using minimal metadata.")
            meta = None

        # Load base policy with preprocessor
        policy_cfg = PreTrainedConfig.from_pretrained(vlm_model_path, cli_overrides={})
        policy_cfg.pretrained_path = vlm_model_path

        if meta is not None:
            self.base_policy = make_policy(policy_cfg, ds_meta=meta)
        else:
            self.base_policy = SmolVLAPolicy.from_pretrained(vlm_model_path)

        self.base_policy.eval()
        self.base_policy.to(self.device)

        # Create preprocessor and postprocessor
        preprocessor_overrides = {"device_processor": {"device": str(self.device)}}
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=vlm_model_path,
            preprocessor_overrides=preprocessor_overrides,
        )
        self.base_policy.preprocessor = preprocessor
        self.base_policy.postprocessor = postprocessor

        # Extract shared VLM
        self.vlm_with_expert = self.base_policy.model.vlm_with_expert
        self.resize_hw = self.base_policy.config.resize_imgs_with_padding

        # Load expert components
        print("Loading expert components...")
        self.experts = {}
        for gtype, ckpt_path in self.available_experts.items():
            full_policy = SmolVLAPolicy.from_pretrained(ckpt_path)
            full_policy.eval()
            self.experts[gtype] = {
                "lm_expert": copy.deepcopy(full_policy.model.vlm_with_expert.lm_expert),
                "action_in_proj": copy.deepcopy(full_policy.model.action_in_proj),
                "action_out_proj": copy.deepcopy(full_policy.model.action_out_proj),
            }
            for comp in self.experts[gtype].values():
                comp.to(self.device)
                for p in comp.parameters():
                    p.requires_grad = False
            print(f"  [OK] Loaded expert for {gtype}")

        # Load router
        print(f"Loading router from {router_path}")
        if not router_path.exists():
            raise FileNotFoundError(f"Router checkpoint not found: {router_path}")
        checkpoint = torch.load(str(router_path), map_location=self.device, weights_only=False)
        config = checkpoint["config"]
        self.router = GarmentRouter(
            input_dim=config["input_dim"],
            hidden_dims=config["hidden_dims"],
            num_classes=config["num_classes"],
        ).to(self.device)
        self.router.load_state_dict(checkpoint["router_state_dict"])

        # Match VLM dtype
        vlm_dtype = next(self.vlm_with_expert.parameters()).dtype
        self.router = self.router.to(vlm_dtype)
        self.router.eval()

        # Episode state (sticky routing)
        self._locked_expert = None
        self._selected_expert = None

        print(f"MoEPolicy initialized on {self.device}")
        print(f"Available experts: {list(self.available_experts.keys())}")
        if self.missing_experts:
            print(f"Missing experts: {self.missing_experts}")

    def reset(self):
        self._locked_expert = None
        self._selected_expert = None
        print("  Episode reset (sticky routing unlocked)")

    def infer(self, observation: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """Route observation to expert and return action."""
        # Sticky routing: lock expert on first frame
        if self._locked_expert is None:
            image = observation.get("observation.images.top_rgb")
            if image is not None:
                garment_type, confidence = self._route_image(image)
                self._locked_expert = garment_type
                self._selected_expert = garment_type
                print(f"  Sticky Routing: locked {garment_type} (confidence: {confidence:.3f})")

        # Run inference with locked expert
        action = self._infer_with_expert(observation, self._locked_expert)
        return [action]

    def _route_image(self, image: np.ndarray):
        """Route image to appropriate expert using router."""
        img_tensor = torch.from_numpy(image.copy()).float()
        if img_tensor.max() > 1.0:
            img_tensor = img_tensor / 255.0
        # HWC -> CHW
        if img_tensor.ndim == 3 and img_tensor.shape[-1] == 3:
            img_tensor = img_tensor.permute(2, 0, 1)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Resize and normalize
        if self.resize_hw is not None:
            img_tensor = self.resize_with_pad(img_tensor, *self.resize_hw, pad_value=0)
        img_tensor = img_tensor * 2.0 - 1.0

        # Extract VLM features
        with torch.no_grad():
            img_emb = self.vlm_with_expert.embed_image(img_tensor)
            if img_emb.dim() == 2:
                img_emb = img_emb.unsqueeze(0)

            # Rich features: mean + std + max pooling
            mean_f = img_emb.mean(dim=1)
            std_f = img_emb.std(dim=1)
            max_f = img_emb.max(dim=1).values
            rich_features = torch.cat([mean_f, std_f, max_f], dim=-1)
            rich_features = torch.nn.functional.normalize(rich_features, p=2, dim=-1)

            logits = self.router(rich_features)
            probs = torch.softmax(logits, dim=-1)
            predicted_class = probs.argmax(dim=-1).item()
            confidence = probs[0, predicted_class].item()
            garment_type = GARMENT_TYPES[predicted_class]

        # Smart fallback for missing experts
        if garment_type in self.missing_experts:
            print(f"  Warning: router predicted '{garment_type}' but expert not available")
            best_type = None
            best_prob = -1
            for idx, gt in enumerate(GARMENT_TYPES):
                if gt in self.available_experts:
                    p = probs[0, idx].item()
                    if p > best_prob:
                        best_prob = p
                        best_type = gt
            if best_type:
                print(f"  Fallback: '{garment_type}' -> '{best_type}' (prob: {best_prob:.3f})")
                garment_type = best_type
                confidence = best_prob
            else:
                raise RuntimeError("No experts available!")

        return garment_type, confidence

    def _infer_with_expert(self, observation: Dict[str, np.ndarray], garment_type: str) -> np.ndarray:
        """Run inference using the specified expert."""
        if garment_type not in self.experts:
            raise ValueError(f"Expert for {garment_type} not available!")

        expert = self.experts[garment_type]

        # Save original components
        orig_lm = self.base_policy.model.vlm_with_expert.lm_expert
        orig_in = self.base_policy.model.action_in_proj
        orig_out = self.base_policy.model.action_out_proj

        # Swap in expert components
        self.base_policy.model.vlm_with_expert.lm_expert = expert["lm_expert"]
        self.base_policy.model.action_in_proj = expert["action_in_proj"]
        self.base_policy.model.action_out_proj = expert["action_out_proj"]

        try:
            # Prepare observation for base policy
            batch_obs = self._prepare_batch(observation)

            with torch.no_grad():
                batch_action = self.base_policy.select_action(batch_obs)
                if hasattr(self.base_policy, "postprocessor") and self.base_policy.postprocessor is not None:
                    batch_action = self.base_policy.postprocessor(batch_action)

            action = batch_action.squeeze(0).cpu().numpy()
            print(f"  Action stats: min={action.min():.4f} max={action.max():.4f} mean={action.mean():.4f} norm={np.linalg.norm(action):.4f}")
        finally:
            # Restore original components
            self.base_policy.model.vlm_with_expert.lm_expert = orig_lm
            self.base_policy.model.action_in_proj = orig_in
            self.base_policy.model.action_out_proj = orig_out

        return action.astype(np.float32)

    def _prepare_batch(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Prepare observation dict for base policy's preprocessor."""
        try:
            from lerobot.processor.core import TransitionKey
        except ImportError:
            from lerobot.types import TransitionKey

        obs_for_preproc = {}
        for key, value in obs.items():
            if not key.startswith("observation."):
                continue
            if isinstance(value, np.ndarray):
                value_tensor = torch.from_numpy(value).float()
                if value.ndim == 3 and value.shape[-1] == 3:
                    value_tensor = value_tensor.permute(2, 0, 1).to(self.device) / 255.0
                obs_for_preproc[key] = value_tensor.unsqueeze(0)

        # Camera key mapping: lehome -> lerobot
        camera_mapping = {
            "observation.images.top_rgb": "observation.images.camera1",
            "observation.images.left_rgb": "observation.images.camera2",
            "observation.images.right_rgb": "observation.images.camera3",
        }
        for old_key, new_key in camera_mapping.items():
            if old_key in obs_for_preproc:
                obs_for_preproc[new_key] = obs_for_preproc[old_key]

        # Create transition with task description
        dummy_action = torch.zeros(1, 12, dtype=torch.float32, device=self.device)
        transition = {
            TransitionKey.OBSERVATION: obs_for_preproc,
            TransitionKey.ACTION: dummy_action,
            TransitionKey.COMPLEMENTARY_DATA: {"task": self.task_description},
        }

        # Use preprocessor (handles language tokens, normalization)
        if hasattr(self.base_policy, "preprocessor") and self.base_policy.preprocessor is not None:
            transformed = self.base_policy.preprocessor._forward(transition)
            return self.base_policy.preprocessor.to_output(transformed)
        return obs_for_preproc


if __name__ == "__main__":
    MoEPolicy().run()
