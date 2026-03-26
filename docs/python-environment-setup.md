# LeHome Challenge - Python Environment Setup

> **Last Updated**: 2026-03-22
> **Status**: ✅ Tested and working with IsaacLab + Evo-RL LeRobot

---

## Overview

This environment combines:
- **IsaacSim 5.1.0** + **IsaacLab 2.3.1** - Physics simulation
- **Evo-RL LeRobot 0.4.4** - Enhanced LeRobot with value function training
- **LeHome** - Garment manipulation tasks

### Key Configuration: NumPy Override

⚠️ **Critical**: This setup uses a NumPy version override to maintain IsaacLab compatibility.

```toml
override-dependencies = [
    "packaging==23.0",
    "numpy==1.26.0",  # Required by IsaacLab (NOT 2.x)
]
```

- **IsaacLab requires**: `numpy==1.26.0`
- **Evo-RL LeRobot wants**: `numpy>=2.x` (but works with 1.26.0!)
- **Solution**: Force `numpy==1.26.0` via `uv` override

---

## Current Versions

| Component | Version | Source |
|-----------|---------|--------|
| Python | 3.11.14 | System/uv |
| NumPy | 1.26.0 | Override (critical!) |
| LeRobot | 0.4.4 | Evo-RL fork |
| IsaacSim | 5.1.0 | NVIDIA |
| IsaacLab | 2.3.1 | third_party/IsaacLab |
| Torch | 2.7.0 | PyTorch |
| Transformers | >=4.57.6 | HuggingFace |

---

## Installation Steps

### 1. Install Dependencies with uv

```bash
# From project root
uv sync

# This installs:
# - IsaacSim 5.1.0
# - LeRobot (initially official 0.4.3)
# - All Python dependencies
```

### 2. Install IsaacLab

```bash
cd third_party && git clone https://github.com/lehome-official/IsaacLab.git && cd ..

source .venv/bin/activate
./third_party/IsaacLab/isaaclab.sh -i none
```

### 3. Replace Official LeRobot with Evo-RL Version

```bash
# Uninstall official LeRobot
uv pip uninstall lerobot -y

# Install Evo-RL LeRobot in editable mode
cd /path/to/lehome-challenge/third_party/Evo-RL
uv pip install -e .

# Force numpy back to 1.26.0 (Evo-RL may upgrade it)
uv pip install "numpy==1.26.0" --reinstall --no-deps
```

### 4. Update pyproject.toml

```bash
# Edit pyproject.toml to use Evo-RL LeRobot
sed -i 's|"lerobot==0.4.3",|"lerobot @ file://${PWD}/third_party/Evo-RL",|' pyproject.toml
```

### 5. Verify Installation

```bash
# Check versions
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"  # Should be 1.26.0
python -c "import lerobot; print(f'LeRobot: {lerobot.__version__}')"  # Should be 0.4.4

# Test IsaacLab
python -m scripts.eval --help

# Test new Evo-RL entry points
lerobot-value-train --help
lerobot-value-infer --help
```

---

## New Evo-RL Features

With Evo-RL LeRobot, you now have access to:

### 1. Value Function Training

```bash
lerobot-value-train \
  --dataset.repo_id=lehome_eval \
  --dataset.root=Datasets/eval_with_failures \
  --value.type=pistar06 \
  --targets.success_field=episode_success \
  --targets.default_success=failure \
  --batch_size=64 \
  --steps=10000 \
  --output_dir=outputs/value_train/pistar06
```

### 2. ACP Label Generation

```bash
lerobot-value-infer \
  --dataset.root=Datasets/eval_with_failures \
  --inference.checkpoint_path=outputs/value_train/pistar06/checkpoints/best \
  --acp.enable=true \
  --acp.n_step=50 \
  --acp.positive_ratio=0.3 \
  --output_dir=outputs/acp_inference
```

### 3. Enhanced Dataset Fields

Evo-RL datasets support additional metadata:

```python
# Episode-level
{
    "episode_index": 0,
    "episode_success": "success",  # ← NEW: Required for value training
}

# Frame-level
{
    "action": [...],
    "observation.state": [...],
    "task": "Fold the garment",

    # NEW: Complementary info for HIL
    "complementary_info.policy_action": [...],     # What policy suggested
    "complementary_info.is_intervention": [...],    # Human took over
    "complementary_info.value": [...],              # Predicted value
    "complementary_info.advantage": [...],           # Calculated advantage
    "complementary_info.acp_indicator": [...],       # 0/1 ACP label
}
```

---

## Troubleshooting

### Issue: NumPy version conflicts

**Symptom**: `ImportError: numpy version mismatch`

**Solution**:
```bash
# Force numpy 1.26.0
uv pip install "numpy==1.26.0" --reinstall --no-deps

# Verify
python -c "import numpy; print(numpy.__version__)"  # Should be 1.26.0
```

### Issue: LeRobot entry points not found

**Symptom**: `lerobot-value-train: command not found`

**Solution**:
```bash
# Reinstall Evo-RL LeRobot
cd third_party/Evo-RL
uv pip install -e .

# Or use python module directly
python -m lerobot.scripts.lerobot_value_train --help
```

### Issue: IsaacLab import errors

**Symptom**: `ModuleNotFoundError: No module named 'omni'`

**Explanation**: This is normal! `omni` modules are only available when running IsaacSim scripts.

**Test properly**:
```bash
# Don't test with "import omni"
# Instead, run an actual IsaacLab script:
python -m scripts.eval --help
```

---

## Running Evaluation

```bash
# Standard evaluation (IsaacLab + Evo-RL LeRobot)
python -m scripts.eval \
  --policy_type moe_smolvla \
  --policy_path outputs/moe_train/... \
  --garment_type pant_short \
  --num_episodes 10 \
  --enable_cameras \
  --device cpu

# With dataset saving (includes episode_success)
python -m scripts.eval \
  --policy_type moe_smolvla \
  --policy_path outputs/moe_train/... \
  --garment_type pant_short \
  --num_episodes 50 \
  --save_datasets \
  --eval_dataset_path Datasets/eval_with_failures \
  --task_description "Fold the garment"
```

---

## Next Steps

### 1. Collect Data with Episode Success

```bash
# Run evaluation to collect success + failure data
python -m scripts.eval \
  --policy_type moe_smolvla \
  --policy_path outputs/moe_train/... \
  --garment_type pant_short \
  --num_episodes 50 \
  --save_datasets \
  --eval_dataset_path Datasets/eval_with_failures/pant_short
```

### 2. Train Value Function

```bash
# Train Pistar06 on collected data
lerobot-value-train \
  --dataset.repo_id=lehome_eval_pant_short \
  --dataset.root=Datasets/eval_with_failures/pant_short \
  --value.type=pistar06 \
  --targets.success_field=episode_success \
  --output_dir=outputs/value_train/pistar06_pant_short
```

### 3. Generate ACP Labels

```bash
# Generate advantage labels for policy training
lerobot-value-infer \
  --dataset.root=Datasets/eval_with_failures/pant_short \
  --inference.checkpoint_path=outputs/value_train/pistar06_pant_short/checkpoints/best \
  --acp.enable=true \
  --output_dir=outputs/acp_inference/pant_short
```

### 4. Train Improved Policy

```bash
# Train policy with ACP
lerobot-train \
  --dataset.root=Datasets/eval_with_failures/pant_short \
  --policy.type=smolvla \
  --acp.enable=true \
  --acp.indicator_field=complementary_info.acp_indicator \
  --output_dir=outputs/moe_train_v2/acp_improved
```

---

## File Structure

```
lehome-challenge/
├── .venv/                          # Python virtual environment
├── third_party/
│   ├── Evo-RL/                     # Evo-RL enhanced LeRobot ← NEW
│   │   └── src/lerobot/
│   │       ├── values/             # Value function (Pistar06)
│   │       ├── rl/                 # ACP implementation
│   │       └── scripts/            # lerobot-value-train, etc.
│   └── IsaacLab/                   # Isaac Gym environments
├── scripts/
│   ├── eval.py                     # Evaluation script
│   └── utils/evaluation.py         # Modified to save episode_success
├── pyproject.toml                  # Project dependencies
└── docs/
    └── python-environment-setup.md # This file
```

---

## Summary

✅ **IsaacLab** works with `numpy==1.26.0`
✅ **Evo-RL LeRobot 0.4.4** works with `numpy==1.26.0`
✅ **Override** ensures compatibility
✅ **New entry points** available for value training

**Critical**: Always verify numpy is 1.26.0 after any pip install operations:

```bash
python -c "import numpy; assert numpy.__version__.startswith('1.26'), f'Wrong numpy: {numpy.__version__}'"
```
