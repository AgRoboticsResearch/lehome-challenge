# Source Code: Modified SmolVLA with Expert Layers

## Overview
These files are modified versions of the `lerobot` SmolVLA policy package. They add support for **MoE Expert training** - each checkpoint contains a shared VLM backbone plus independent expert layers (`lm_expert`, `action_in_proj`, `action_out_proj`).

Without these modifications, the trained checkpoints **cannot be loaded** by the standard lerobot 0.4.3 package.

## Files
| File | Description |
|---|---|
| `modeling_smolvla.py` | Modified: uses `SmolVLMWithExpertModel` instead of standard SmolVLM |
| `configuration_smolvla.py` | Modified: adds `train_expert_only`, `expert_width_multiplier`, `num_expert_layers` config |
| `processor_smolvla.py` | Standard SmolVLA preprocessing |
| `smolvlm_with_expert.py` | NEW: adds `lm_expert` expert layers alongside base VLM |

## Installation

After setting up lehome-challenge with Isaac Sim, Isaac Lab, and lerobot 0.4.3, copy these files into the lerobot SmolVLA package directory:

```bash
# Find your lerobot installation path
LEROBOT_SMOLVLA_PATH=$(python -c "import lerobot.policies.smolvla; print(lerobot.policies.smolvla.__file__)" | xargs dirname)

# Back up original files (recommended)
cp -r "$LEROBOT_SMOLVLA_PATH" "${LEROBOT_SMOLVLA_PATH}_backup"

# Copy modified files
cp lerobot_policies_smolvla/* "$LEROBOT_SMOLVLA_PATH/"
```

## Verification
After copying, verify the installation:
```bash
python -c "from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel; print('OK')"
```

## Evaluation
Use `--policy_type lerobot` with the provided checkpoints. Each garment type has a separate model:
```bash
python -m scripts.eval \
    --policy_type lerobot \
    --policy_path <checkpoint_path>/pretrained_model \
    --dataset_root Datasets/example/<garment_type>_merged \
    --garment_type <garment_type> \
    --num_episodes 5 --max_steps 600 \
    --enable_cameras --device cpu --headless \
    --task_description "fold the garment on the table"
```
