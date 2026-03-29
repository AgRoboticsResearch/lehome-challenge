# Source Code: MoE-SmolVLA Submission

## Overview

This directory contains modified source files needed to run `--policy_type moe_smolvla`:

1. **lerobot SmolVLA modifications** - Required to load expert-trained checkpoints
2. **eval_policy module** - Self-contained MoE routing policy with inline GarmentRouter

**Note:** Checkpoints must be downloaded first (see main README.md).

## Files

### lerobot_policies_smolvla/
Modified lerobot SmolVLA files that add MoE Expert layer support:

| File | Description |
|---|---|
| `modeling_smolvla.py` | Modified: uses `SmolVLMWithExpertModel` instead of standard SmolVLM |
| `configuration_smolvla.py` | Modified: adds `train_expert_only`, `expert_width_multiplier`, `num_expert_layers` config |
| `smolvlm_with_expert.py` | NEW: adds `lm_expert` expert layers alongside base VLM |

### scripts/eval_policy/
Policy implementation for MoE routing:

| File | Description |
|---|---|
| `__init__.py` | Registers `MoESmolVLAPolicy` as `"moe_smolvla"` policy type |
| `moe_smolvla_policy.py` | Self-contained MoE policy with inline `GarmentRouter`, sticky routing, expert swapping |

## How It Works

- **Sticky routing:** The garment type is classified on the first frame using a trained router, then locked for the entire episode
- **Expert swapping:** The correct expert checkpoint is loaded based on the classified garment type
- **No dataset required:** Unlike `--policy_type lerobot`, the MoE policy doesn't need `--dataset_root`
