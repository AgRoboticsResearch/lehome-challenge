# Source Code: MoE-SmolVLA Submission

## Overview

This directory contains the modified source files needed to run `--policy_type moe_smolvla`:

1. **lerobot SmolVLA modifications** - Required to load expert-trained checkpoints
2. **eval_policy module** - Self-contained MoE routing policy with inline GarmentRouter

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

## Installation

After setting up lehome-challenge with Isaac Sim, Isaac Lab, and lerobot 0.4.3:

```bash
# 1. Copy lerobot SmolVLA modifications
LEROBOT_SMOLVLA=$(python -c "import lerobot.policies.smolvla; import os; print(os.path.dirname(lerobot.policies.smolvla.__file__))")
cp lerobot_policies_smolvla/* "$LEROBOT_SMOLVLA/"

# 2. Copy eval_policy files (replaces existing files in scripts/eval_policy/)
EVAL_POLICY="scripts/eval_policy"
cp scripts/eval_policy/__init__.py "$EVAL_POLICY/__init__.py"
cp scripts/eval_policy/moe_smolvla_policy.py "$EVAL_POLICY/moe_smolvla_policy.py"
```

## Verification

```bash
# Verify lerobot modifications
python -c "from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel; print('OK')"

# Verify MoE policy registration
python -c "from scripts.eval_policy import MoESmolVLAPolicy; print('OK')"
```

## Evaluation

Use `--policy_type moe_smolvla` to automatically route to the correct garment expert:

```bash
python -m scripts.eval \
    --policy_type moe_smolvla \
    --garment_type "pant_short" \
    --num_episodes 5 --max_steps 600 \
    --enable_cameras --device cpu --headless \
    --task_description "fold the garment on the table"
```

The policy automatically:
- Loads all 4 expert checkpoints from `outputs/moe_train/`
- Routes each episode to the correct expert using the trained router
- Locks the expert for the duration of each episode (sticky routing)
