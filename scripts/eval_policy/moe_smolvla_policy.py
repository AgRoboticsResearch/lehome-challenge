"""
MoE-SmolVLA Policy for LeHome Challenge

Implements a Mixture-of-Experts system for SmolVLA with garment-type routing.
"""

import copy
import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

# Import to register smolvla_new_line_processor in ProcessorStepRegistry
# This is required for loading expert preprocessors with normalization stats
from lerobot.policies.smolvla.processor_smolvla import SmolVLANewLineProcessor  # noqa: F401

from lehome.utils.logger import get_logger
from scripts.eval_policy.base_policy import BasePolicy
from scripts.eval_policy.registry import PolicyRegistry

logger = get_logger(__name__)


@PolicyRegistry.register("moe_smolvla")
class MoESmolVLAPolicy(BasePolicy):
    """
    MoE-SmolVLA Policy with garment-type routing.

    Architecture:
        - Shared VLM (vision encoder + text embeddings)
        - Shared state_proj (frozen)
        - Independent lm_expert per garment type
        - Independent action projections (action_in_proj, action_out_proj) per garment type
        - Independent action_time_mlp (in/out) per garment type (Flow Matching time encoding)
        - Router selects expert based on visual features
    """

    # Garment type definitions
    GARMENT_TYPES = ["top_short", "top_long", "pant_short", "pant_long"]
    TYPE_TO_LABEL = {name: i for i, name in enumerate(GARMENT_TYPES)}

    def __init__(
        self,
        vlm_model_path: str = None,
        expert_checkpoints: Dict[str, str] = None,
        router_checkpoint: str = None,
        device: str = "cuda",
        task_description: str = "fold the garment on the table",
        dataset_root: str = None,
        model_path: str = None,  # Alias for vlm_model_path (for eval framework compatibility)
    ):
        """
        Initialize MoE-SmolVLA Policy.

        Args:
            vlm_model_path: Path to base VLM model (for shared components).
                If None, uses the first available expert checkpoint as base.
            expert_checkpoints: Dict mapping garment_type -> checkpoint path
                e.g., {"pant_short": "outputs/train/.../pretrained_model"}
            router_checkpoint: Path to trained router checkpoint
            device: Device to run on ("cuda" or "cpu")
            task_description: Task description for VLA models
            dataset_root: Path to dataset root (for metadata/normalization stats)
            model_path: Alias for vlm_model_path (for eval framework compatibility)
        """
        super().__init__()

        # Default policy_path from eval framework (should be ignored for MoE)
        DEFAULT_POLICY_PATH = (
            "outputs/train/diffusion_fold_1/checkpoints/100000/pretrained_model"
        )

        # Normalize empty strings to None
        if model_path == "":
            model_path = None
        if vlm_model_path == "":
            vlm_model_path = None

        # Ignore the default policy_path from eval framework
        # MoE policy uses its own expert checkpoints, not the default diffusion path
        if model_path == DEFAULT_POLICY_PATH:
            logger.info(f"⚠️ Ignoring default policy_path: {DEFAULT_POLICY_PATH}")
            logger.info("   MoE policy will use expert checkpoints instead")
            model_path = None
        if vlm_model_path == DEFAULT_POLICY_PATH:
            vlm_model_path = None

        # Handle model_path alias for compatibility with eval framework
        if model_path is not None and vlm_model_path is None:
            vlm_model_path = model_path

        # DEBUG: Log normalized values
        logger.info(f"🐛 DEBUG - After normalization:")
        logger.info(f"  model_path = {repr(model_path)}")
        logger.info(f"  vlm_model_path = {repr(vlm_model_path)}")

        self.device = torch.device(device)
        self.task_description = task_description

        # Default expert checkpoints (relative to outputs/moe_train/)
        if expert_checkpoints is None:
            expert_checkpoints = self._get_default_expert_checkpoints()

        # Validate expert checkpoints
        self.available_experts = {}
        self.missing_experts = []
        for garment_type, checkpoint_path in expert_checkpoints.items():
            full_path = Path(checkpoint_path)
            if full_path.exists():
                self.available_experts[garment_type] = str(full_path)
                logger.info(f"✅ Found expert for {garment_type}: {full_path}")
            else:
                self.missing_experts.append(garment_type)
                logger.warning(f"❌ Missing expert for {garment_type}: {full_path}")

        if len(self.available_experts) == 0:
            raise ValueError("No expert checkpoints found!")

        logger.info(f"Available experts: {list(self.available_experts.keys())}")
        if self.missing_experts:
            logger.warning(f"Missing experts: {self.missing_experts}")

        # Use first available expert as base VLM if not specified
        # (Important: Experts are dual-arm trained, lerobot/smolvla_base is single-arm)
        if vlm_model_path is None:
            vlm_model_path = list(self.available_experts.values())[0]
            logger.info(f"Using first expert checkpoint as base VLM: {vlm_model_path}")

        # Load metadata for preprocessor (required for normalization stats)
        if dataset_root is None:
            # Try to infer from first expert checkpoint
            first_expert_path = list(self.available_experts.values())[0]
            # Look for dataset info in the expert's config
            dataset_root = "Datasets/example"  # Default fallback
            logger.info(f"Using default dataset_root: {dataset_root}")

        logger.info(f"Loading metadata from {dataset_root}")
        try:
            meta = LeRobotDatasetMetadata(repo_id="lehome", root=dataset_root)
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}. Using minimal metadata.")
            meta = None

        # Load shared components (VLM + state_proj) with preprocessor
        logger.info(f"Loading shared VLM from {vlm_model_path}")
        self.base_policy = self._load_base_policy_with_preprocessor(
            vlm_model_path, meta
        )

        original_n_action_steps = self.base_policy.config.n_action_steps
        self.base_policy.config.n_action_steps = 12
        self.base_policy.reset()
        logger.info(
            f"Overridden n_action_steps: {original_n_action_steps} → 12 (re-plan every 12 steps / 100ms)"
        )

        # Extract shared VLM model
        self.vlm_with_expert = self.base_policy.model.vlm_with_expert

        # Store config
        self.vlm_hidden_size = self.vlm_with_expert.config.text_config.hidden_size
        logger.info(f"VLM hidden size: {self.vlm_hidden_size}")

        # Load experts (independent components)
        logger.info("Loading expert models...")
        self.experts = {}
        for garment_type, checkpoint_path in self.available_experts.items():
            expert = self._load_expert_checkpoint(checkpoint_path, garment_type)
            self.experts[garment_type] = expert
            logger.info(f"✅ Loaded expert for {garment_type}")

        # Load router
        if router_checkpoint is None:
            router_checkpoint = "outputs/moe_train/router/checkpoints/best/router.pt"

        logger.info(f"Loading router from {router_checkpoint}")
        self.router = self._load_router(router_checkpoint)

        # Preprocessor for image normalization
        self.resize_hw = self.base_policy.config.resize_imgs_with_padding

        # Episode state management (for sticky routing)
        self.current_episode = None
        self.selected_expert = None
        self._locked_expert = None  # Sticky Routing: Initialize as None
        self.route_confidence_history = []

        # Log expert availability summary
        self._log_expert_status()

        logger.info("MoESmolVLAPolicy initialization complete!")

    def _get_default_expert_checkpoints(self) -> Dict[str, str]:
        """Get default expert checkpoint paths from moe_train directory."""
        base_path = Path("outputs/moe_train")
        return {
            "pant_short": base_path
            / "smolvla_moe_expert_pant_short_no_st_proj/checkpoints/010000/pretrained_model",
            "pant_long": base_path
            / "smolvla_moe_expert_pant_long_no_st_proj/checkpoints/009000/pretrained_model",
            "top_short": base_path
            / "smolvla_moe_expert_top_short_no_st_proj/checkpoints/014000/pretrained_model",
            "top_long": base_path
            / "smolvla_moe_expert_top_long_no_st_proj/checkpoints/011000/pretrained_model",
        }

    def _log_expert_status(self):
        """Log expert availability status for debugging."""
        total_experts = len(self.GARMENT_TYPES)
        available_count = len(self.available_experts)
        missing_count = len(self.missing_experts)

        logger.info("=" * 60)
        logger.info("📊 MoE Expert Status Summary")
        logger.info("=" * 60)
        logger.info(f"Total experts: {available_count}/{total_experts} available")

        for gtype in self.GARMENT_TYPES:
            status = "✅" if gtype in self.available_experts else "❌"
            logger.info(f"  {status} {gtype}")

        if self.missing_experts:
            logger.warning("-" * 60)
            logger.warning("⚠️ Some experts are missing!")
            logger.warning(
                "Smart fallback will be used when Router predicts missing types."
            )
            logger.warning(f"Missing: {self.missing_experts}")
        logger.info("=" * 60)

    def _load_base_policy_with_preprocessor(
        self, vlm_model_path: str, meta: Optional[LeRobotDatasetMetadata]
    ) -> SmolVLAPolicy:
        """Load base policy with preprocessor and postprocessor initialized.

        This is necessary for proper handling of language tokens and normalization.
        """
        from lerobot.policies.factory import make_policy, make_pre_post_processors
        from lerobot.configs.policies import PreTrainedConfig

        # Load config
        policy_cfg = PreTrainedConfig.from_pretrained(vlm_model_path, cli_overrides={})
        policy_cfg.pretrained_path = vlm_model_path

        # Create policy with metadata
        if meta is not None:
            policy = make_policy(policy_cfg, ds_meta=meta)
        else:
            # Fallback: create policy without metadata
            policy = SmolVLAPolicy.from_pretrained(vlm_model_path)

        policy.eval()
        policy.to(self.device)

        # Create preprocessor and postprocessor
        preprocessor_overrides = {
            "device_processor": {"device": str(self.device)},
        }
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=vlm_model_path,
            preprocessor_overrides=preprocessor_overrides,
        )

        # Attach to policy (SmolVLAPolicy doesn't have these by default)
        policy.preprocessor = preprocessor
        policy.postprocessor = postprocessor

        logger.info("✅ Loaded base policy with preprocessor and postprocessor")

        return policy

    def _load_expert_checkpoint(
        self, checkpoint_path: str, garment_type: str
    ) -> Dict[str, nn.Module]:
        """Load expert-specific components (lm_expert, action projections, action_time_mlp).

        Following MoE design: each expert has independent:
        - lm_expert: action planning transformer
        - action_in_proj: noisy action encoding
        - action_out_proj: action output projection
        - action_time_mlp_in/out: time encoding for Flow Matching

        Shared components (loaded once in base_policy):
        - VLM (SigLIP + text backbone)
        - state_proj

        Also loads expert-specific preprocessor and postprocessor for correct
        normalization/denormalization stats.
        """
        checkpoint_path = Path(checkpoint_path)

        # Load the full checkpoint to extract expert components
        logger.info(f"Loading expert components from {checkpoint_path}")
        full_policy = SmolVLAPolicy.from_pretrained(str(checkpoint_path))
        full_policy.eval()

        # Extract expert-specific components
        # Note: action_time_mlp_in/out are always trainable (no config flag to freeze them)
        # so each expert checkpoint contains different weights for them
        expert_components = {
            "lm_expert": copy.deepcopy(full_policy.model.vlm_with_expert.lm_expert),
            "action_in_proj": copy.deepcopy(full_policy.model.action_in_proj),
            "action_out_proj": copy.deepcopy(full_policy.model.action_out_proj),
            "action_time_mlp_in": copy.deepcopy(full_policy.model.action_time_mlp_in),
            "action_time_mlp_out": copy.deepcopy(full_policy.model.action_time_mlp_out),
        }

        # Move to device and freeze
        for component in expert_components.values():
            component.to(self.device)
            for param in component.parameters():
                param.requires_grad = False

        # Load expert-specific preprocessor and postprocessor (for normalization stats)
        expert_components["preprocessor"] = self._load_expert_preprocessor(
            checkpoint_path
        )
        expert_components["postprocessor"] = self._load_expert_postprocessor(
            checkpoint_path
        )

        logger.info(f"✅ Loaded expert components for {garment_type}")

        return expert_components

    def _load_expert_preprocessor(self, checkpoint_path: Path):
        """Load preprocessor with expert-specific normalization stats."""
        from lerobot.processor import PolicyProcessorPipeline

        try:
            preprocessor = PolicyProcessorPipeline.from_pretrained(
                pretrained_model_name_or_path=str(checkpoint_path),
                config_filename="policy_preprocessor.json",
                overrides={"device_processor": {"device": str(self.device)}},
            )
            logger.info(f"  ✅ Loaded preprocessor for {checkpoint_path.name}")
            return preprocessor
        except Exception as e:
            logger.warning(
                f"  ⚠️ Failed to load preprocessor: {e}, using base policy's preprocessor"
            )
            return None

    def _load_expert_postprocessor(self, checkpoint_path: Path):
        """Load postprocessor with expert-specific denormalization stats."""
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.converters import (
            policy_action_to_transition,
            transition_to_policy_action,
        )

        try:
            postprocessor = PolicyProcessorPipeline.from_pretrained(
                pretrained_model_name_or_path=str(checkpoint_path),
                config_filename="policy_postprocessor.json",
                overrides={"device_processor": {"device": "cpu"}},
                to_transition=policy_action_to_transition,
                to_output=transition_to_policy_action,
            )
            logger.info(f"  ✅ Loaded postprocessor for {checkpoint_path.name}")
            return postprocessor
        except Exception as e:
            logger.warning(
                f"  ⚠️ Failed to load postprocessor: {e}, using base policy's postprocessor"
            )
            return None

    def _load_router(self, checkpoint_path: str) -> nn.Module:
        """Load trained router classifier."""
        from scripts.train_router import GarmentRouter

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Router checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)

        # Reconstruct router
        config = checkpoint["config"]
        router = GarmentRouter(
            input_dim=config["input_dim"],
            hidden_dims=config["hidden_dims"],
            num_classes=config["num_classes"],
        ).to(self.device)
        router.load_state_dict(checkpoint["router_state_dict"])

        # Convert router to match VLM dtype (BFloat16) to avoid dtype mismatch
        vlm_dtype = next(self.vlm_with_expert.parameters()).dtype
        router = router.to(vlm_dtype)

        router.eval()

        logger.info(
            f"Router loaded with {config['num_classes']} classes: {config['type_names']}"
        )
        logger.info(f"Router dtype: {vlm_dtype}")

        return router

    def reset(self):
        """Reset policy state (called at episode start)."""
        self.current_episode = None
        self.selected_expert = None
        self._locked_expert = None  # Sticky Routing: 清除锁定状态
        self.route_confidence_history = []
        self.base_policy.reset()

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate action using MoE routing with Sticky Routing.

        Sticky Routing: Route on first frame, then lock for entire episode.

        Args:
            observation: Dictionary with keys:
                - observation.images.top_rgb: RGB image [H, W, C]
                - observation.state: Joint positions [12]
                - (optional) observation.images.{left,right}_rgb

        Returns:
            action: Joint actions [12]
        """
        # Preprocess observation
        observation_dict = self._prepare_observation(observation)

        # === Sticky Routing: 首帧路由并锁定 ===
        if self._locked_expert is None:
            # Extract image for routing
            image = observation_dict["observation.images.top_rgb"]

            # Route to appropriate expert
            garment_type, confidence = self._route_image(image)

            # Lock expert for this episode
            self._locked_expert = garment_type
            self.selected_expert = garment_type

            logger.info(
                f"🔒 Sticky Routing: Locked Expert {garment_type} (confidence: {confidence:.3f})"
            )

        # === 使用锁定的Expert（不再调用Router）===
        return self._select_action_with_expert(observation_dict, self._locked_expert)

    def _prepare_observation(
        self, observation: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Prepare observation for model input (keep as numpy arrays)."""
        obs_dict = {}

        for key, value in observation.items():
            if not key.startswith("observation."):
                continue
            # Keep images as numpy arrays - LeRobot policy will handle preprocessing
            obs_dict[key] = value

        return obs_dict

    def _route_image(self, image: np.ndarray) -> tuple[str, float]:
        """
        Route image to appropriate expert using router.

        Returns:
            (garment_type, confidence)

        Note: If the predicted expert is not available, will intelligently
        fall back to the best available expert based on router probabilities.
        """
        from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad

        # Preprocess image
        if isinstance(image, np.ndarray):
            img_tensor = torch.from_numpy(image).float()
        else:
            img_tensor = image

        # Normalize to [0, 1]
        if img_tensor.max() > 1.0:
            img_tensor = img_tensor / 255.0

        # HWC -> CHW
        if img_tensor.ndim == 3 and img_tensor.shape[-1] == 3:
            img_tensor = img_tensor.permute(2, 0, 1)

        img_tensor = img_tensor.unsqueeze(0).to(self.device)  # [1, 3, H, W]

        # Resize and normalize (same as SmolVLA preprocessing)
        if self.resize_hw is not None:
            img_tensor = resize_with_pad(img_tensor, *self.resize_hw, pad_value=0)
        img_tensor = img_tensor * 2.0 - 1.0  # [0, 1] -> [-1, 1]

        # Extract VLM features
        with torch.no_grad():
            img_emb = self.vlm_with_expert.embed_image(img_tensor)
            if img_emb.dim() == 2:
                img_emb = img_emb.unsqueeze(0)

            # Extract routing features (mean + std + max pooling)
            mean_f = img_emb.mean(dim=1)
            std_f = img_emb.std(dim=1)
            max_f = img_emb.max(dim=1).values
            rich_features = torch.cat([mean_f, std_f, max_f], dim=-1)
            rich_features = torch.nn.functional.normalize(rich_features, p=2, dim=-1)

            # Get router predictions (probabilities for all classes)
            logits = self.router(rich_features)
            probs = torch.softmax(logits, dim=-1)

            # Get top prediction
            predicted_class = probs.argmax(dim=-1).item()
            confidence = probs[0, predicted_class].item()

            garment_type = self.GARMENT_TYPES[predicted_class]

            # === Smart Fallback for Missing Experts ===
            if garment_type in self.missing_experts:
                logger.warning(
                    f"⚠️ Router predicted '{garment_type}' but expert not available"
                )

                # Find the best available expert based on router probabilities
                best_available_type = None
                best_available_prob = -1

                for idx, gtype in enumerate(self.GARMENT_TYPES):
                    if gtype in self.available_experts:
                        prob = probs[0, idx].item()
                        if prob > best_available_prob:
                            best_available_prob = prob
                            best_available_type = gtype

                if best_available_type is not None:
                    logger.warning(
                        f"🔄 Smart fallback: '{garment_type}' → '{best_available_type}' "
                        f"(confidence: {best_available_prob:.3f})"
                    )
                    garment_type = best_available_type
                    confidence = best_available_prob
                else:
                    raise ValueError("No experts available!")

            return garment_type, confidence

    def _select_action_with_expert(
        self, observation_dict: Dict[str, np.ndarray], garment_type: str
    ) -> np.ndarray:
        """Select action using the specified expert.

        Uses expert-specific preprocessor/postprocessor for correct normalization,
        and swaps in the expert's lm_expert, action projections, and action_time_mlp.
        """
        if garment_type not in self.experts:
            raise ValueError(f"Expert for {garment_type} not available!")

        expert_components = self.experts[garment_type]

        # Save original components
        original_lm_expert = self.base_policy.model.vlm_with_expert.lm_expert
        original_action_in_proj = self.base_policy.model.action_in_proj
        original_action_out_proj = self.base_policy.model.action_out_proj
        original_action_time_mlp_in = self.base_policy.model.action_time_mlp_in
        original_action_time_mlp_out = self.base_policy.model.action_time_mlp_out

        # Temporarily swap in expert components
        self.base_policy.model.vlm_with_expert.lm_expert = expert_components[
            "lm_expert"
        ]
        self.base_policy.model.action_in_proj = expert_components["action_in_proj"]
        self.base_policy.model.action_out_proj = expert_components["action_out_proj"]
        self.base_policy.model.action_time_mlp_in = expert_components[
            "action_time_mlp_in"
        ]
        self.base_policy.model.action_time_mlp_out = expert_components[
            "action_time_mlp_out"
        ]

        # CRITICAL: Each expert was trained with different dataset stats.
        # We MUST use the expert's preprocessor/postprocessor for correct normalization.
        # Falling back to base_policy's preprocessor would use WRONG stats!
        expert_preprocessor = expert_components.get("preprocessor")
        expert_postprocessor = expert_components.get("postprocessor")

        # Validate that expert has preprocessor - without it, normalization will be wrong!
        if expert_preprocessor is None:
            logger.error(f"❌ CRITICAL: Expert {garment_type} has NO preprocessor!")
            logger.error(
                f"   Cannot normalize observation correctly - this will cause wrong actions!"
            )
            raise ValueError(
                f"Expert {garment_type} missing preprocessor. "
                f"Each expert must have its own preprocessor with correct normalization stats. "
                f"Check if 'policy_preprocessor.json' exists in the expert checkpoint."
            )

        if expert_postprocessor is None:
            logger.error(f"❌ CRITICAL: Expert {garment_type} has NO postprocessor!")
            logger.error(
                f"   Cannot denormalize action correctly - this will cause wrong actions!"
            )
            raise ValueError(
                f"Expert {garment_type} missing postprocessor. "
                f"Each expert must have its own postprocessor with correct denormalization stats. "
                f"Check if 'policy_postprocessor.json' exists in the expert checkpoint."
            )

        logger.info(
            f"✅ [{garment_type}] Using expert preprocessor/postprocessor with correct normalization stats"
        )

        try:
            with torch.no_grad():
                # Always use expert's preprocessor (validated above)
                batch_obs = self._prepare_batch_with_preprocessor(
                    observation_dict, expert_preprocessor
                )

                batch_action = self.base_policy.select_action(batch_obs)

                # Always use expert's postprocessor (validated above)
                batch_action = expert_postprocessor(batch_action)

            # Convert to numpy
            action = batch_action.squeeze(0).cpu().numpy()

        finally:
            # Restore original components
            self.base_policy.model.vlm_with_expert.lm_expert = original_lm_expert
            self.base_policy.model.action_in_proj = original_action_in_proj
            self.base_policy.model.action_out_proj = original_action_out_proj
            self.base_policy.model.action_time_mlp_in = original_action_time_mlp_in
            self.base_policy.model.action_time_mlp_out = original_action_time_mlp_out

        return action

    def _prepare_batch_with_preprocessor(
        self, obs_dict: Dict[str, np.ndarray], preprocessor
    ) -> Dict[str, torch.Tensor]:
        """Prepare observation using a specific preprocessor (handles language tokens)."""
        from lerobot.processor.core import TransitionKey

        # Prepare tensors for preprocessor
        obs_for_preproc = {}
        for key, value in obs_dict.items():
            if not key.startswith("observation."):
                continue

            if isinstance(value, np.ndarray):
                value_tensor = torch.from_numpy(value).float()
                if value.ndim == 3 and value.shape[-1] == 3:  # Image: (H, W, C)
                    value_tensor = value_tensor.permute(2, 0, 1).to(self.device) / 255.0
                    obs_for_preproc[key] = value_tensor.unsqueeze(0)
                else:
                    obs_for_preproc[key] = value_tensor.unsqueeze(0)

        # Map camera keys: LeHome uses top_rgb/left_rgb/right_rgb,
        # but preprocessor expects camera1/camera2/camera3
        camera_key_mapping = {
            "observation.images.top_rgb": "observation.images.camera1",
            "observation.images.left_rgb": "observation.images.camera2",
            "observation.images.right_rgb": "observation.images.camera3",
        }

        # Apply mapping if source keys exist
        for old_key, new_key in camera_key_mapping.items():
            if old_key in obs_for_preproc:
                obs_for_preproc[new_key] = obs_for_preproc[old_key]

        # Create transition with task description
        dummy_action = torch.zeros(1, 12, dtype=torch.float32, device=self.device)
        transition = {
            TransitionKey.OBSERVATION: obs_for_preproc,
            TransitionKey.ACTION: dummy_action,
            TransitionKey.COMPLEMENTARY_DATA: {"task": self.task_description},
        }

        # Use the provided preprocessor (handles language tokens)
        transformed = preprocessor._forward(transition)
        batch_obs = preprocessor.to_output(transformed)

        return batch_obs

    def _prepare_transition(self, obs_dict: Dict[str, np.ndarray]) -> Dict:
        """Prepare observation in LeRobot transition format."""
        from lerobot.processor.core import TransitionKey

        obs_for_preproc = {}
        for key, value in obs_dict.items():
            if not key.startswith("observation."):
                continue

            if isinstance(value, np.ndarray):
                value_tensor = torch.from_numpy(value).float()
                if value.ndim == 3 and value.shape[-1] == 3:  # Image: (H, W, C)
                    # (H, W, C) -> (C, H, W), [0, 1] normalization
                    value_tensor = value_tensor.permute(2, 0, 1).to(self.device) / 255.0
                    obs_for_preproc[key] = value_tensor.unsqueeze(0)  # Add batch dim
                else:
                    obs_for_preproc[key] = value_tensor.unsqueeze(0)  # Add batch dim

        # Create transition format with complementary_data for VLA models
        dummy_action = torch.zeros(1, 12, dtype=torch.float32, device=self.device)
        transition = {
            TransitionKey.OBSERVATION: obs_for_preproc,
            TransitionKey.ACTION: dummy_action,
            TransitionKey.COMPLEMENTARY_DATA: {"task": self.task_description},
        }
        return transition

    def _prepare_batch_for_base_policy(
        self, obs_dict: Dict[str, np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        """Prepare observation using base_policy's preprocessor (handles language tokens)."""
        from lerobot.processor.core import TransitionKey

        # Prepare tensors for preprocessor
        obs_for_preproc = {}
        for key, value in obs_dict.items():
            if not key.startswith("observation."):
                continue

            if isinstance(value, np.ndarray):
                value_tensor = torch.from_numpy(value).float()
                if value.ndim == 3 and value.shape[-1] == 3:  # Image: (H, W, C)
                    value_tensor = value_tensor.permute(2, 0, 1).to(self.device) / 255.0
                    obs_for_preproc[key] = value_tensor.unsqueeze(0)
                else:
                    obs_for_preproc[key] = value_tensor.unsqueeze(0)

        # Map camera keys: LeHome uses top_rgb/left_rgb/right_rgb,
        # but base_policy expects camera1/camera2/camera3
        camera_key_mapping = {
            "observation.images.top_rgb": "observation.images.camera1",
            "observation.images.left_rgb": "observation.images.camera2",
            "observation.images.right_rgb": "observation.images.camera3",
        }

        # Apply mapping if source keys exist
        for old_key, new_key in camera_key_mapping.items():
            if old_key in obs_for_preproc:
                obs_for_preproc[new_key] = obs_for_preproc[old_key]
                # Keep both keys for compatibility

        # Create transition with task description
        dummy_action = torch.zeros(1, 12, dtype=torch.float32, device=self.device)
        transition = {
            TransitionKey.OBSERVATION: obs_for_preproc,
            TransitionKey.ACTION: dummy_action,
            TransitionKey.COMPLEMENTARY_DATA: {"task": self.task_description},
        }

        # Use base_policy's preprocessor (handles language tokens)
        if (
            hasattr(self.base_policy, "preprocessor")
            and self.base_policy.preprocessor is not None
        ):
            transformed = self.base_policy.preprocessor._forward(transition)
            batch_obs = self.base_policy.preprocessor.to_output(transformed)
        else:
            # Fallback if no preprocessor
            batch_obs = obs_for_preproc

        return batch_obs
