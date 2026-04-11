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

from server import BasePolicyServer

# Garment type definitions
GARMENT_TYPES = ["top_short", "top_long", "pant_short", "pant_long"]


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
        """Route image using Qwen3-VL zero-shot VQA."""

        # Lazy-load Qwen on first call
        if not hasattr(self, '_qwen_model'):
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
            print("  Loading Qwen3-VL-8B for routing...")
            self._qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen3-VL-8B-Instruct",
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self._qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
            print("  Qwen3-VL loaded")

        from PIL import Image
        from qwen_vl_utils import process_vision_info

        VALID_TYPES = {"top_short", "top_long", "pant_short", "pant_long"}

        # Build VQA prompt
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": Image.fromarray(image)},
                {"type": "text", "text": (
                    "Look at this garment on a table from a top-down camera view. "
                    "Classify it as exactly one of: top_short, top_long, pant_short, pant_long. "
                    "Reply with only the label, nothing else."
                )},
            ],
        }]

        text = self._qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._qwen_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self._qwen_model.generate(**inputs, max_new_tokens=10)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self._qwen_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True
            )[0].strip().lower()

        print(f"  Qwen response: '{response}'")

        # Parse response to garment type
        garment_type = None
        confidence = 1.0
        for gt in GARMENT_TYPES:
            if gt in response or gt.replace("_", " ") in response:
                garment_type = gt
                break

        if garment_type is None:
            # Try broader matching
            if "top" in response or "shirt" in response:
                if "short" in response or "t-shirt" in response:
                    garment_type = "top_short"
                else:
                    garment_type = "top_long"
            elif "pant" in response or "trouser" in response or "jean" in response:
                if "short" in response or "shorts" in response:
                    garment_type = "pant_short"
                else:
                    garment_type = "pant_long"
            else:
                # Default fallback
                garment_type = GARMENT_TYPES[0]
                confidence = 0.0
                print(f"  Warning: couldn't parse Qwen response, defaulting to {garment_type}")

        # Fallback for missing experts
        if garment_type in self.missing_experts:
            print(f"  Warning: Qwen predicted '{garment_type}' but expert not available")
            best_type = None
            for gt in GARMENT_TYPES:
                if gt in self.available_experts:
                    best_type = gt
                    break
            if best_type:
                garment_type = best_type
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
