# SmolVLA Value Function Implementation Plan

> **Created**: 2026-03-21
> **Goal**: Implement a value function based on SmolVLA architecture for LeHome Challenge
> **Approach**: Include robot joint state (Option 1 - Single Embodiment)

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Design](#architecture-design)
3. [Implementation Steps](#implementation-steps)
4. [File Structure](#file-structure)
5. [Configuration](#configuration)
6. [Training Pipeline](#training-pipeline)
7. [Integration with Evo-RL](#integration-with-evo-rl)
8. [Testing & Validation](#testing--validation)

---

## Overview

### Goal

Create a value function that:
- Reuses SmolVLA's VLM backbone (frozen)
- Includes robot joint state for better value estimation
- Predicts value distribution over 201 bins [-1, 0]
- Can be used for RECAP-style advantage-conditioned policy improvement

### Why Include State?

| Factor | Reasoning |
|--------|-----------|
| **Single Embodiment** | LeHome Challenge uses fixed SO101 dual-arm |
| **Better Value Estimation** | Joint positions indicate feasible actions |
| **Reuse Infrastructure** | SmolVLA already has `state_proj` |
| **No Cross-Robot Need** | Not training for generalization |

### Comparison with Pistar06

| Aspect | Pistar06 | Our SmolVLA Value |
|--------|----------|-------------------|
| Vision Encoder | SigLIP (1152 dim) | SmolVLM Vision (768 dim) |
| Language Model | Gemma 3 270M (2304 dim) | VLlama3 (960 dim) |
| State Input | ❌ No | ✅ Yes |
| Fusion | Concat projectors | VLM internal attention |
| Output | 201 bins [-1, 0] | 201 bins [-1, 0] |

---

## Architecture Design

### Full Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SmolVLA Value Function (with State)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUTS:                                                                    │
│  ┌──────────────┬──────────────┬──────────────┐                            │
│  │ Images       │ Language     │ State        │                            │
│  │ [B,N,C,H,W]  │ tokens [B,T] │ [B, 12]      │                            │
│  └──────┬───────┴──────┬───────┴──────┬───────┘                            │
│         │              │              │                                     │
│         ▼              ▼              ▼                                     │
│  ┌──────────────────────────────────────────────────────┐                  │
│  │                    FROZEN VLM                         │                  │
│  │                                                       │                  │
│  │  ┌─────────────┐   ┌─────────────┐   ┌────────────┐ │                  │
│  │  │ Vision Enc  │   │ Text Embed  │   │ State Proj │ │                  │
│  │  │ (SmolVLM)   │   │ (VLlama3)   │   │ Linear     │ │                  │
│  │  │ 768 dim     │   │ 960 dim     │   │ 12 → 960   │ │                  │
│  │  └──────┬──────┘   └──────┬──────┘   └──────┬─────┘ │                  │
│  │         │                 │                 │       │                  │
│  │         ▼                 ▼                 ▼       │                  │
│  │  ┌─────────────────────────────────────────────────┐│                  │
│  │  │ Concat: [IMG_START + imgs + IMG_END + lang + st]││                  │
│  │  │ Shape: [B, prefix_len, 960]                     ││                  │
│  │  └───────────────────────┬─────────────────────────┘│                  │
│  │                          │                          │                  │
│  │                          ▼                          │                  │
│  │  ┌─────────────────────────────────────────────────┐│                  │
│  │  │         VLM Transformer (Frozen)                ││                  │
│  │  │         Multi-layer Self-Attention              ││                  │
│  │  └───────────────────────┬─────────────────────────┘│                  │
│  └──────────────────────────┼──────────────────────────┘                  │
│                             │                                               │
│                             ▼                                               │
│                    [B, prefix_len, 960]                                     │
│                             │                                               │
│  ┌──────────────────────────┼──────────────────────────────────────────┐  │
│  │                    TRAINABLE HEAD                                    │  │
│  │                          │                                          │  │
│  │                          ▼                                          │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │              Attention Pooling                               │   │  │
│  │  │  - Learnable query token [1, 1, 960]                        │   │  │
│  │  │  - Multi-head attention over prefix tokens                   │   │  │
│  │  │  - Output: [B, 960]                                         │   │  │
│  │  └─────────────────────────────┬───────────────────────────────┘   │  │
│  │                                │                                    │  │
│  │                                ▼                                    │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │                    Value Head                                │   │  │
│  │  │                                                              │   │  │
│  │  │   Linear(960 → 512)                                         │   │  │
│  │  │   LayerNorm(512)                                            │   │  │
│  │  │   GELU                                                      │   │  │
│  │  │   Dropout(0.1)                                              │   │  │
│  │  │   Linear(512 → 256)                                         │   │  │
│  │  │   GELU                                                      │   │  │
│  │  │   Dropout(0.1)                                              │   │  │
│  │  │   Linear(256 → 201)  ← Value bins                           │   │  │
│  │  │                                                              │   │  │
│  │  └─────────────────────────────┬───────────────────────────────┘   │  │
│  └────────────────────────────────┼────────────────────────────────────┘  │
│                                   │                                         │
│                                   ▼                                         │
│                          Output: [B, 201]                                   │
│                          (Value distribution logits)                        │
│                                   │                                         │
│                                   ▼                                         │
│                    Expected Value = Σ(softmax × bin_centers)               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Value Head Details

```python
# Value head structure
value_head = nn.Sequential(
    nn.Linear(960, 512),      # Project VLM output
    nn.LayerNorm(512),        # Normalize
    nn.GELU(),                # Activation
    nn.Dropout(0.1),          # Regularization
    nn.Linear(512, 256),      # Second layer
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(256, 201),      # Output: 201 bins for [-1, 0]
)
```

### Attention Pooling vs Mean Pooling

| Method | Pros | Cons |
|--------|------|------|
| **Mean Pooling** | Simple, no params | Treats all tokens equally |
| **Attention Pooling** | Learns what matters | Extra parameters, complexity |

**Recommendation**: Use Attention Pooling for better feature extraction.

---

## Implementation Steps

### Phase 1: Core Model Implementation

#### Step 1.1: Create Configuration

```python
# src/lerobot/values/smolvla_value/configuration_smolvla_value.py

from dataclasses import dataclass, field
from typing import Optional, List
from lerobot.configs.policies import PreTrainedConfig


@PreTrainedConfig.register_subclass("smolvla_value")
@dataclass
class SmolVLAValueConfig(PreTrainedConfig):
    """Configuration for SmolVLA Value Function

    This configuration defines the SmolVLA-based value function that:
    - Reuses SmolVLA's VLM backbone (frozen)
    - Includes robot joint state for better value estimation
    - Predicts value distribution over discrete bins
    """

    # Override name for registration
    name: str = "smolvla_value"

    # Base model
    smolvla_model_path: str = ""  # Path to pretrained SmolVLA policy
    freeze_vlm: bool = True       # Freeze VLM backbone

    # Value head architecture
    value_hidden_dim: int = 512
    value_hidden_dim_2: int = 256
    dropout: float = 0.1

    # Value distribution (same as Pistar06 for consistency)
    num_bins: int = 201
    bin_min: float = -1.0
    bin_max: float = 0.0

    # Pooling strategy
    pooling_type: str = "attention"  # "attention" or "mean"
    num_attention_heads: int = 8

    # Training configuration
    loss_weight_key: Optional[str] = None  # Key for sample weights in batch
    target_key: str = "value_target"       # Key for value targets in batch

    # Input features (inherited from SmolVLA)
    image_features: List[str] = field(default_factory=lambda: [
        "observation.images.top_rgb",
        "observation.images.left_rgb",
        "observation.images.right_rgb",
    ])
    state_feature: str = "observation.state"
```

#### Step 1.2: Create Model

```python
# source/lerobot/lerobot/values/smolvla_value/modeling_smolvla_value.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from pathlib import Path

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching
from .configuration_smolvla_value import SmolVLAValueConfig


class SmolVLAValueFunction(nn.Module):
    """
    SmolVLA-based Value Function

    Reuses SmolVLA's VLM backbone with a new value prediction head.
    Includes robot joint state for better value estimation.
    """

    def __init__(self, config: SmolVLAValueConfig):
        super().__init__()
        self.config = config

        # Load SmolVLA base model
        self._load_smolvla_base(config.smolvla_model_path)

        # Build value head
        self._build_value_head(config)

        # Register bin centers
        self.register_buffer(
            'bin_centers',
            torch.linspace(config.bin_min, config.bin_max, config.num_bins)
        )

    def _load_smolvla_base(self, model_path: str):
        """Load and freeze SmolVLA VLM backbone"""
        # Load pretrained SmolVLA
        self.smolvla_base = SmolVLAPolicy.from_pretrained(model_path)
        self.vlm_with_expert = self.smolvla_base.model.vlm_with_expert

        # Get VLM components we need
        self.vlm = self.vlm_with_expert.vlm
        self.state_proj = self.smolvla_base.model.state_proj

        # Freeze VLM
        if self.config.freeze_vlm:
            for param in self.vlm.parameters():
                param.requires_grad = False
            # Keep state_proj trainable (optional)
            # for param in self.state_proj.parameters():
            #     param.requires_grad = False

    def _build_value_head(self, config: SmolVLAValueConfig):
        """Build the value prediction head"""
        vlm_hidden_size = 960  # VLlama3 hidden size

        # Pooling layer
        if config.pooling_type == "attention":
            self.value_query = nn.Parameter(
                torch.randn(1, 1, vlm_hidden_size) * 0.02
            )
            self.attention_pool = nn.MultiheadAttention(
                embed_dim=vlm_hidden_size,
                num_heads=config.num_attention_heads,
                batch_first=True
            )
        else:
            self.value_query = None
            self.attention_pool = None

        # Value head MLP
        self.value_head = nn.Sequential(
            nn.Linear(vlm_hidden_size, config.value_hidden_dim),
            nn.LayerNorm(config.value_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.value_hidden_dim, config.value_hidden_dim_2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.value_hidden_dim_2, config.num_bins),
        )

    def _extract_prefix_features(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract VLM features using SmolVLA's embed_prefix"""
        # Use SmolVLA's existing prefix embedding
        prefix_embs, pad_masks, att_masks = self.smolvla_base.model.embed_prefix(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
        )
        return prefix_embs, pad_masks, att_masks

    def forward(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass to predict value distribution

        Args:
            images: List of image tensors [B, C, H, W] per camera
            img_masks: List of masks [B] per camera
            lang_tokens: Language token IDs [B, T]
            lang_masks: Language attention masks [B, T]
            state: Robot joint state [B, state_dim]

        Returns:
            Value logits [B, num_bins]
        """
        # 1. Extract prefix embeddings (includes state)
        prefix_embs, pad_masks, att_masks = self._extract_prefix_features(
            images, img_masks, lang_tokens, lang_masks, state
        )

        # 2. VLM forward (frozen)
        with torch.no_grad() if self.config.freeze_vlm else torch.enable_grad():
            vlm_output = self.vlm.model(
                inputs_embeds=prefix_embs,
                attention_mask=att_masks,
                return_dict=True,
            )
            features = vlm_output.last_hidden_state  # [B, T, 960]

        # 3. Pooling
        if self.config.pooling_type == "attention":
            pooled = self._attention_pooling(features, pad_masks)
        else:
            pooled = self._mean_pooling(features, pad_masks)

        # 4. Value prediction
        value_logits = self.value_head(pooled)  # [B, num_bins]

        return value_logits

    def _attention_pooling(
        self, features: torch.Tensor, pad_masks: torch.Tensor
    ) -> torch.Tensor:
        """Attention-based pooling over sequence dimension"""
        bsize = features.shape[0]
        device = features.device

        # Expand query to batch size
        query = self.value_query.expand(bsize, -1, -1)  # [B, 1, 960]

        # Key padding mask (True = ignore)
        key_padding_mask = ~pad_masks.bool()

        # Attention pooling
        pooled, _ = self.attention_pool(
            query, features, features,
            key_padding_mask=key_padding_mask
        )
        return pooled.squeeze(1)  # [B, 960]

    def _mean_pooling(
        self, features: torch.Tensor, pad_masks: torch.Tensor
    ) -> torch.Tensor:
        """Mean pooling over valid tokens"""
        mask = pad_masks.float().unsqueeze(-1)  # [B, T, 1]
        pooled = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled  # [B, 960]

    def predict_value(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict expected value for inference

        Returns:
            Expected values [B]
        """
        logits = self.forward(images, img_masks, lang_tokens, lang_masks, state)
        probs = F.softmax(logits, dim=-1)
        expected_value = (probs * self.bin_centers).sum(dim=-1)
        return expected_value
```

#### Step 1.3: Create Policy Wrapper (Required by Evo-RL)

```python
# src/lerobot/values/smolvla_value/modeling_smolvla_value.py
# (Add SmolVLAValuePolicy class to the same file)

from lerobot.policies.pretrained import PreTrainedPolicy
from typing import Any


class SmolVLAValuePolicy(PreTrainedPolicy):
    """
    Policy wrapper for SmolVLA Value Function

    This class implements PreTrainedPolicy interface required by Evo-RL's
    lerobot-value-train pipeline. Required methods:
    - forward(batch, reduction) -> (loss, loss_dict)
    - predict_value(batch) -> Tensor
    - build_training_raw_batch_hook(dataset, targets_cfg) -> Callable
    """

    config_class = SmolVLAValueConfig
    name = "smolvla_value"

    def __init__(self, config: SmolVLAValueConfig, dataset_meta=None, **kwargs):
        super().__init__(config)
        self.config = config
        self.model = SmolVLAValueFunction(config)
        self.dataset_meta = dataset_meta

        # Register bin centers for value computation
        self.register_buffer(
            "bin_centers",
            torch.linspace(config.bin_min, config.bin_max, config.num_bins),
            persistent=False,
        )

    def reset(self):
        """Reset policy state (no-op for value function)"""
        pass

    def get_optim_params(self):
        """Return parameters to optimize"""
        return self.parameters()

    def predict_action_chunk(self, batch, **kwargs):
        """Not applicable - value function does not predict actions"""
        raise RuntimeError(
            "SmolVLAValuePolicy is a value function and does not support action prediction. "
            "Use predict_value() instead."
        )

    def select_action(self, batch, **kwargs):
        """Not applicable - value function does not select actions"""
        raise RuntimeError(
            "SmolVLAValuePolicy is a value function and does not support action selection. "
            "Use predict_value() instead."
        )

    def predict_value(self, batch: dict) -> torch.Tensor:
        """
        Predict expected value for a batch.

        This is the main inference method for the value function.

        Args:
            batch: Dictionary containing:
                - observation.images.*: Camera images
                - observation.language_tokens: Task description tokens
                - observation.language_attention_mask: Language mask
                - observation.state: Robot joint state

        Returns:
            Expected value tensor [B]
        """
        # Prepare inputs using SmolVLA's preprocessing
        images, img_masks = self._prepare_images(batch)
        lang_tokens = batch["observation.language_tokens"]
        lang_masks = batch["observation.language_attention_mask"]
        state = self._prepare_state(batch)

        return self.model.predict_value(
            images, img_masks, lang_tokens, lang_masks, state
        )

    def _prepare_images(self, batch: dict) -> tuple:
        """Prepare images using SmolVLA's preprocessing"""
        # Reuse SmolVLA's image preparation
        return self.model.smolvla_base.prepare_images(batch)

    def _prepare_state(self, batch: dict) -> torch.Tensor:
        """Prepare state tensor"""
        state = batch.get("observation.state")
        if state is None:
            raise ValueError("observation.state is required for SmolVLA value function")
        # Handle temporal dimension if present
        if state.ndim == 3:
            state = state[:, -1, :]  # Take last timestep
        return state

    def forward(self, batch: dict, reduction: str = "mean"):
        """
        Training forward pass to compute loss.

        Required by lerobot-value-train pipeline.

        Args:
            batch: Dictionary containing observations and value targets
            reduction: "mean" returns scalar loss, "none" returns per-sample losses

        Returns:
            Tuple of (loss, loss_dict) where:
            - loss: Scalar loss tensor (or per-sample if reduction="none")
            - loss_dict: Dictionary of metrics for logging
        """
        # Get inputs
        images, img_masks = self._prepare_images(batch)
        lang_tokens = batch["observation.language_tokens"]
        lang_masks = batch["observation.language_attention_mask"]
        state = self._prepare_state(batch)

        # Get value prediction
        logits = self.model(images, img_masks, lang_tokens, lang_masks, state)

        # Get target
        if self.config.target_key not in batch:
            raise KeyError(
                f"Missing target key '{self.config.target_key}' in batch. "
                "Make sure build_training_raw_batch_hook is registered."
            )
        value_target = batch[self.config.target_key]
        if value_target.ndim == 2:
            value_target = value_target.squeeze(-1)

        # Project targets to soft labels (same as Pistar06)
        from lerobot.values.pistar06.modeling_pistar06 import (
            project_values_to_bins,
            expected_value_from_logits,
        )
        soft_target = project_values_to_bins(value_target, self.bin_centers)

        # Cross-entropy loss with soft targets
        log_probs = F.log_softmax(logits, dim=-1)
        per_sample_loss = -(soft_target * log_probs).sum(dim=-1)

        # Optional sample weighting
        if self.config.loss_weight_key and self.config.loss_weight_key in batch:
            weights = batch[self.config.loss_weight_key]
            if weights.ndim == 2:
                weights = weights.squeeze(-1)
            per_sample_loss = per_sample_loss * weights

        # Compute metrics
        with torch.no_grad():
            pred_value = expected_value_from_logits(logits, self.bin_centers)
            value_mae = (pred_value - value_target).abs().mean()

        # Apply reduction
        loss = per_sample_loss if reduction == "none" else per_sample_loss.mean()

        loss_dict = {
            "loss": float(loss.mean().item() if reduction == "none" else loss.item()),
            "value_mae": float(value_mae.item()),
        }

        return loss, loss_dict

    def build_training_raw_batch_hook(self, dataset, targets_cfg):
        """
        Build hook to compute value targets during training.

        Required by lerobot-value-train pipeline.
        This hook is called for each batch to add value targets.

        Reuses Pistar06's target computation logic.

        Args:
            dataset: LeRobot dataset
            targets_cfg: Target configuration (success_field, c_fail_coef, etc.)

        Returns:
            Callable hook that adds value targets to batch
        """
        from lerobot.values.pistar06.modeling_pistar06 import (
            compute_normalized_value_targets,
            EpisodeTargetInfo,
        )
        from lerobot.utils.recording_annotations import (
            EPISODE_SUCCESS,
            resolve_episode_success_label,
        )
        import numpy as np

        # Extract episode and frame info
        raw_frames = dataset.hf_dataset.with_format(None)
        episode_indices = np.asarray(raw_frames["episode_index"], dtype=np.int64)
        frame_indices = np.asarray(raw_frames["frame_index"], dtype=np.int64)
        absolute_indices = np.asarray(raw_frames["index"], dtype=np.int64)

        # Get episode metadata
        episodes_ds = dataset.meta.episodes.with_format(None)
        episodes = episodes_ds[:]
        n_episodes = len(episodes_ds)
        has_success = targets_cfg.success_field in episodes_ds.column_names

        # Build episode info lookup
        episode_info: dict[int, EpisodeTargetInfo] = {}
        task_max_length: dict[int, int] = {}

        for i in range(n_episodes):
            ep_idx = int(episodes["episode_index"][i])
            ep_length = int(episodes["length"][i])
            tasks = episodes["tasks"][i]
            task_name = tasks[0] if isinstance(tasks, list) else tasks

            if task_name not in dataset.meta.tasks.index:
                raise KeyError(f"Episode {ep_idx} references unknown task '{task_name}'")
            task_index = int(dataset.meta.tasks.loc[task_name].task_index)

            # Resolve success label
            explicit_success = episodes[targets_cfg.success_field][i] if has_success else None
            resolved = resolve_episode_success_label(
                explicit_success,
                default_label=targets_cfg.default_success,
                require_label=True,
            )
            ep_success = resolved == EPISODE_SUCCESS

            episode_info[ep_idx] = EpisodeTargetInfo(
                episode_index=ep_idx,
                task_index=task_index,
                length=ep_length,
                success=ep_success,
            )
            task_max_length[task_index] = max(
                task_max_length.get(task_index, 0), ep_length
            )

        # Compute value targets for all frames
        value_targets = compute_normalized_value_targets(
            episode_indices=episode_indices,
            frame_indices=frame_indices,
            episode_info=episode_info,
            task_max_lengths=task_max_length,
            c_fail_coef=targets_cfg.c_fail_coef,
            clip_min=self.config.bin_min,
            clip_max=self.config.bin_max,
        )

        # Build lookup table
        max_index = int(np.max(absolute_indices))
        value_target_lookup = np.zeros(max_index + 1, dtype=np.float32)
        value_target_lookup[absolute_indices] = value_targets

        target_key = targets_cfg.target_field

        def value_target_hook(batch: dict, step: int) -> dict:
            """Hook to add value targets to batch"""
            batch_indices = batch.get("index")
            if batch_indices is None:
                raise KeyError("Missing 'index' in batch while building value targets")

            if not isinstance(batch_indices, torch.Tensor):
                batch_indices = torch.as_tensor(batch_indices)

            indices_np = batch_indices.detach().cpu().numpy().astype(np.int64).reshape(-1)
            target_values = torch.from_numpy(value_target_lookup[indices_np]).float()
            batch[target_key] = target_values
            return batch

        return value_target_hook
```

#### Step 1.4: Create `__init__.py`

```python
# src/lerobot/values/smolvla_value/__init__.py

from .configuration_smolvla_value import SmolVLAValueConfig
from .modeling_smolvla_value import SmolVLAValueFunction, SmolVLAValuePolicy
from .processor_smolvla_value import make_smolvla_value_pre_post_processors

__all__ = [
    "SmolVLAValueConfig",
    "SmolVLAValueFunction",
    "SmolVLAValuePolicy",
    "make_smolvla_value_pre_post_processors",
]
```

---

## File Structure (Following Evo-RL Convention)

Based on Evo-RL's suggestion for plugging in a new value function:

```
third_party/Evo-RL/src/lerobot/values/
├── pistar06/
│   ├── __init__.py
│   ├── configuration_pistar06.py
│   ├── modeling_pistar06.py
│   └── processor_pistar06.py
└── smolvla_value/                    # NEW - Add here
    ├── __init__.py
    ├── configuration_smolvla_value.py  # @PreTrainedConfig.register_subclass("smolvla_value")
    ├── modeling_smolvla_value.py       # SmolVLAValuePolicy(PreTrainedPolicy)
    └── processor_smolvla_value.py      # make_smolvla_value_pre_post_processors(...)
```

### Required Files

| File | Purpose |
|------|---------|
| `configuration_smolvla_value.py` | Config class with `@PreTrainedConfig.register_subclass("smolvla_value")` |
| `modeling_smolvla_value.py` | `SmolVLAValuePolicy(PreTrainedPolicy)` implementing `forward`, `predict_value`, `build_training_raw_batch_hook` |
| `processor_smolvla_value.py` | `make_smolvla_value_pre_post_processors(...)` for data preprocessing |

### Files to Modify

| File | Change |
|------|--------|
| `src/lerobot/configs/value_train.py` | Remove/replace pistar06-only type checks |
| `src/lerobot/scripts/lerobot_value_infer.py` | Remove/replace pistar06-only type checks |

### Step 1.4: Create Processor

```python
# src/lerobot/values/smolvla_value/processor_smolvla_value.py

from dataclasses import dataclass
from typing import Any, Callable
import torch
from torch import Tensor

from lerobot.configs.train import TrainConfig


@dataclass
class SmolVLAValuePreprocessedBatch:
    """Preprocessed batch for SmolVLA value function"""
    images: list[Tensor]
    image_masks: list[Tensor]
    language_tokens: Tensor
    language_attention_mask: Tensor
    state: Tensor
    index: Tensor
    episode_index: Tensor
    frame_index: Tensor


def make_smolvla_value_pre_post_processors(
    train_cfg: TrainConfig,
    dataset_meta: Any,
) -> tuple[Callable, Callable]:
    """
    Create pre and post processors for SmolVLA value function training.

    This follows the Evo-RL convention for value function processors.

    Args:
        train_cfg: Training configuration
        dataset_meta: Dataset metadata

    Returns:
        Tuple of (pre_processor, post_processor) functions
    """
    from lerobot.policies.smolvla.processor_smolvla import (
        SmolVLAProcessor,
        make_smolvla_pre_post_processors as make_smolvla_policy_processors,
    )

    # Reuse SmolVLA policy's image/text preprocessing
    # since value function uses the same VLM backbone
    policy_pre, policy_post = make_smolvla_policy_processors(train_cfg, dataset_meta)

    def pre_processor(batch: dict) -> dict:
        """
        Preprocess batch for value function training.

        This reuses SmolVLA's preprocessing for images and language,
        and keeps state as-is.
        """
        # Apply SmolVLA's preprocessing (handles images, language tokens)
        batch = policy_pre(batch)

        # Ensure state is present
        if "observation.state" not in batch:
            raise ValueError("observation.state is required for SmolVLA value function")

        return batch

    def post_processor(batch: dict) -> dict:
        """
        Post-process batch (e.g., move to device, apply transforms).
        """
        # Apply SmolVLA's post-processing
        batch = policy_post(batch)
        return batch

    return pre_processor, post_processor
```

---

## Files to Modify in Evo-RL

### Modify `src/lerobot/configs/value_train.py`

```python
# Find and update the pistar06-only type checks

# BEFORE (pistar06 only):
if value_type != "pistar06":
    raise ValueError(f"Unknown value type: {value_type}")

# AFTER (support multiple value types):
VALID_VALUE_TYPES = ["pistar06", "smolvla_value"]

if value_type not in VALID_VALUE_TYPES:
    raise ValueError(f"Unknown value type: {value_type}. Valid types: {VALID_VALUE_TYPES}")
```

### Modify `src/lerobot/scripts/lerobot_value_infer.py`

```python
# Find and update the pistar06-only imports and checks

# BEFORE:
from lerobot.values.pistar06 import Pistar06Policy

# AFTER:
from lerobot.values.pistar06 import Pistar06Policy
from lerobot.values.smolvla_value import SmolVLAValuePolicy

VALUE_POLICY_MAP = {
    "pistar06": Pistar06Policy,
    "smolvla_value": SmolVLAValuePolicy,
}

def load_value_policy(value_type: str, policy_path: str, **kwargs):
    """Load value policy based on type"""
    if value_type not in VALUE_POLICY_MAP:
        raise ValueError(f"Unknown value type: {value_type}")
    policy_cls = VALUE_POLICY_MAP[value_type]
    return policy_cls.from_pretrained(policy_path, **kwargs)
```

---

## Configuration

### Training Config YAML

```yaml
# configs/train_smolvla_value.yaml

# Policy configuration
policy:
  type: smolvla_value
  smolvla_model_path: outputs/train/smolvla_base/checkpoints/last/pretrained_model
  freeze_vlm: true
  value_hidden_dim: 512
  value_hidden_dim_2: 256
  dropout: 0.1
  num_bins: 201
  bin_min: -1.0
  bin_max: 0.0
  pooling_type: attention
  num_attention_heads: 8
  target_key: value_target

# Dataset configuration
dataset:
  repo_id: lehome/hil_pant_short
  root: Datasets/hil/pant_short
  image_transforms:
    enable: false  # SmolVLA handles its own transforms

# Training configuration
training:
  batch_size: 32
  learning_rate: 1e-4
  weight_decay: 0.01
  num_workers: 4
  steps: 10000
  warmup_steps: 500
  save_every_n_steps: 1000
  eval_every_n_steps: 500

# Value targets
targets:
  success_field: episode_success
  default_success: failure
  c_fail_coef: 1.0
  target_field: value_target

# Hardware
device: cuda
seed: 42

# Logging
wandb:
  enable: true
  project: lehome-value-function
  name: smolvla_value_pant_short
```

---

## Training Pipeline

### Training Command

```bash
# Using lerobot-value-train (from Evo-RL)
lerobot-value-train \
  --config.path=configs/train_smolvla_value.yaml \
  --policy.type=smolvla_value \
  --policy.smolvla_model_path=outputs/train/smolvla_base/pretrained_model \
  --dataset.repo_id=lehome/hil_pant_short \
  --output_dir=outputs/value_train/smolvla_pant_short \
  --training.batch_size=32 \
  --training.steps=10000
```

### Custom Training Script (Optional)

```python
# scripts/train_smolvla_value.py

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from lerobot.values.smolvla_value import SmolVLAValuePolicy, SmolVLAValueConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import wandb

def train_value_function(config):
    # Initialize wandb
    if config.wandb.enable:
        wandb.init(project=config.wandb.project, name=config.wandb.name)

    # Load dataset
    dataset = LeRobotDataset(
        repo_id=config.dataset.repo_id,
        root=config.dataset.root,
    )

    # Load policy
    policy_config = SmolVLAValueConfig(**config.policy)
    policy = SmolVLAValuePolicy(config=policy_config)

    # Build target hook
    target_hook = policy.build_training_raw_batch_hook(dataset, config.targets)

    # Optimizer
    optimizer = AdamW(
        policy.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.training.learning_rate,
        total_steps=config.training.steps,
        pct_start=0.05,
    )

    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=dataset.collate_fn,
    )

    # Training loop
    policy.train()
    global_step = 0

    while global_step < config.training.steps:
        for batch in dataloader:
            # Apply target hook
            batch = target_hook(batch, global_step)

            # Move to device
            batch = {k: v.to(config.device) if hasattr(v, 'to') else v
                     for k, v in batch.items()}

            # Forward pass
            loss, loss_dict = policy(batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1

            # Logging
            if global_step % 10 == 0:
                print(f"Step {global_step}: loss={loss_dict['loss']:.4f}, mae={loss_dict['value_mae']:.4f}")
                if config.wandb.enable:
                    wandb.log(loss_dict, step=global_step)

            # Save checkpoint
            if global_step % config.training.save_every_n_steps == 0:
                policy.save_pretrained(f"{config.output_dir}/checkpoint_{global_step}")

            if global_step >= config.training.steps:
                break

    # Final save
    policy.save_pretrained(f"{config.output_dir}/final")

if __name__ == "__main__":
    import yaml
    from types import SimpleNamespace

    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        return d

    with open("configs/train_smolvla_value.yaml") as f:
        config = yaml.safe_load(f)
    config = dict_to_namespace(config)

    train_value_function(config)
```

---

## Integration with Evo-RL

### For RECAP Advantage-Conditioned Training

```python
# Example: Using value function for advantage computation

from lerobot.values.smolvla_value import SmolVLAValuePolicy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

class RECAPTrainer:
    def __init__(self, policy_path, value_path):
        # Load expert policy
        self.policy = SmolVLAPolicy.from_pretrained(policy_path)

        # Load value function
        self.value_fn = SmolVLAValuePolicy.from_pretrained(value_path)

    def compute_advantage(self, batch):
        """
        Compute advantage: A(s,a) = Q(s,a) - V(s)
        For RECAP: advantage = normalized_to_goal
        """
        # Get value prediction
        with torch.no_grad():
            value = self.value_fn.predict_value(batch)

        # Advantage is how much better we can do than expected
        # For goal-reaching: advantage = -remaining_steps_normalized - value
        # This encourages actions that reduce distance to goal

        return advantage

    def train_step(self, batch):
        # 1. Compute advantage
        advantage = self.compute_advantage(batch)

        # 2. Weight policy loss by advantage
        action_loss = self.policy.forward(batch)

        # 3. Higher advantage → higher weight
        weighted_loss = action_loss * F.relu(advantage).exp()

        return weighted_loss.mean()
```

---

## Testing & Validation

### Unit Tests

```python
# tests/test_smolvla_value.py

import torch
import pytest
from lerobot.values.smolvla_value import SmolVLAValueConfig, SmolVLAValueFunction

def test_smolvla_value_forward():
    """Test basic forward pass"""
    config = SmolVLAValueConfig(
        smolvla_model_path="lerobot/smolvla_base",
        num_bins=201,
        pooling_type="attention",
    )

    model = SmolVLAValueFunction(config)

    # Create dummy inputs
    bsize = 2
    images = [torch.randn(bsize, 3, 224, 224) for _ in range(3)]  # 3 cameras
    img_masks = [torch.ones(bsize, dtype=torch.bool) for _ in range(3)]
    lang_tokens = torch.randint(0, 1000, (bsize, 32))
    lang_masks = torch.ones(bsize, 32, dtype=torch.bool)
    state = torch.randn(bsize, 12)  # Dual-arm: 12 DOF

    # Forward pass
    logits = model(images, img_masks, lang_tokens, lang_masks, state)

    assert logits.shape == (bsize, 201)
    assert not torch.isnan(logits).any()

def test_smolvla_value_predict():
    """Test value prediction"""
    config = SmolVLAValueConfig(
        smolvla_model_path="lerobot/smolvla_base",
        num_bins=201,
        bin_min=-1.0,
        bin_max=0.0,
    )

    model = SmolVLAValueFunction(config)

    # Create dummy inputs
    bsize = 4
    images = [torch.randn(bsize, 3, 224, 224) for _ in range(3)]
    img_masks = [torch.ones(bsize, dtype=torch.bool) for _ in range(3)]
    lang_tokens = torch.randint(0, 1000, (bsize, 32))
    lang_masks = torch.ones(bsize, 32, dtype=torch.bool)
    state = torch.randn(bsize, 12)

    # Predict value
    values = model.predict_value(images, img_masks, lang_tokens, lang_masks, state)

    assert values.shape == (bsize,)
    assert (values >= -1.0).all() and (values <= 0.0).all()

def test_bin_projection():
    """Test value to bin projection"""
    from lerobot.values.pistar06.modeling_pistar06 import (
        project_values_to_bins,
        expected_value_from_logits,
    )

    bin_centers = torch.linspace(-1.0, 0.0, 201)
    values = torch.tensor([-0.5, -0.25, -0.75])

    soft_targets = project_values_to_bins(values, bin_centers)

    assert soft_targets.shape == (3, 201)
    assert torch.allclose(soft_targets.sum(dim=-1), torch.ones(3))

    # Test reconstruction
    logits = torch.log(soft_targets + 1e-10)
    reconstructed = expected_value_from_logits(logits, bin_centers)

    assert torch.allclose(reconstructed, values, atol=0.01)
```

### Integration Test

```bash
# Test with real dataset
python -m scripts.test_value_function \
  --policy_path outputs/value_train/smolvla_pant_short/final \
  --dataset_root Datasets/hil/pant_short \
  --num_samples 100
```

---

## Summary

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Include State** | ✅ Yes | Better value estimation for single embodiment |
| **Freeze VLM** | ✅ Yes | Reduce overfitting, leverage pretrained features |
| **Pooling** | Attention | Learn what matters for value prediction |
| **Value Head** | 3-layer MLP | Similar to Pistar06, proven effective |
| **Bins** | 201 bins [-1, 0] | Same as Pistar06 for consistency |

### Estimated Trainable Parameters

| Component | Parameters |
|-----------|------------|
| Attention Pooling | ~2.4M |
| Value Head | ~0.7M |
| **Total** | **~3.1M** |

### Next Steps

1. [ ] Create `configuration_smolvla_value.py` with `@PreTrainedConfig.register_subclass("smolvla_value")`
2. [ ] Implement `modeling_smolvla_value.py` with `SmolVLAValuePolicy(PreTrainedPolicy)`
3. [ ] Create `processor_smolvla_value.py` with `make_smolvla_value_pre_post_processors(...)`
4. [ ] Modify `src/lerobot/configs/value_train.py` to support `smolvla_value` type
5. [ ] Modify `src/lerobot/scripts/lerobot_value_infer.py` to support `smolvla_value` type
6. [ ] Create `__init__.py` for the `smolvla_value` module
7. [ ] Write unit tests
8. [ ] Create training config YAML
9. [ ] Test with sample dataset
10. [ ] Full training run
11. [ ] Integration with RECAP

---

*Created: 2026-03-21*
