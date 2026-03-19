# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working Style

**IMPORTANT: Before modifying any code or implementing any plan, always confirm with the user first.**

Our conversations are primarily focused on **research direction and solution design**, NOT direct code implementation:
- Researching approaches and comparing alternatives
- Discussing design and architecture trade-offs
- Exploring integration strategies
- Planning implementation steps
- Understanding how different systems work together

When presenting a plan:
1. Summarize what needs to be done
2. List the files/changes involved
3. Wait for user confirmation
4. Only proceed with implementation after approval

Do NOT directly modify code files without explicit user approval.

## Project Overview

LeHome Challenge 2026 is an ICRA competition for garment manipulation skill learning in household scenarios. It's built on:
- **Isaac Sim 5.1.0** - Physics simulation
- **Isaac Lab 2.3.1** - Robot learning environments
- **LeRobot 0.4.3** - Imitation learning framework
- **Python 3.11** (strict requirement)

The simulation currently only supports **CPU devices** to avoid garment physics issues.

## Installation

```bash
# 1. Install dependencies with uv
uv sync

# 2. Clone IsaacLab into third_party
cd third_party && git clone https://github.com/lehome-official/IsaacLab.git && cd ..

# 3. Install IsaacLab
source .venv/bin/activate
./third_party/IsaacLab/isaaclab.sh -i none

# 4. Install LeHome package
uv pip install -e ./source/lehome
```

## Common Commands

### Asset & Data Preparation
```bash
# Download simulation assets
hf download lehome/asset_challenge --repo-type dataset --local-dir Assets

# Download example dataset
hf download lehome/dataset_challenge_merged --repo-type dataset --local-dir Datasets/example
```

### Training
```bash
# Train with pre-configured policies (ACT, DP, SmolVLA)
lerobot-train --config_path=configs/train_act.yaml
lerobot-train --config_path=configs/train_dp.yaml
lerobot-train --config_path=configs/train_smolvla.yaml
```

### Evaluation
```bash
# Evaluate LeRobot policy
python -m scripts.eval \
    --policy_type lerobot \
    --policy_path outputs/train/act_top_long/checkpoints/last/pretrained_model \
    --garment_type "top_long" \
    --dataset_root Datasets/example/top_long_merged \
    --num_episodes 5 \
    --enable_cameras \
    --device cpu

# Evaluate custom policy
python -m scripts.eval \
    --policy_type custom \
    --garment_type "top_long" \
    --num_episodes 5 \
    --enable_cameras \
    --device cpu
```

### Dataset Operations (No Isaac Sim Required)
```bash
# Inspect dataset
python -m scripts.dataset inspect --dataset_root Datasets/example/top_long_merged

# Read dataset states
python -m scripts.dataset read --dataset_root Datasets/example/top_long_merged

# Add end-effector pose to dataset
python -m scripts.dataset augment \
    --dataset_root Datasets/record/001 \
    --urdf_path Assets/robots/so101_new_calib.urdf \
    --state_unit rad

# Merge datasets (via Python)
python -c "
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets
datasets = [LeRobotDataset(f'ds_{i:03d}', root=Path('path/to/dataset')) for i in range(1, 4)]
merged = merge_datasets(datasets, 'merged', Path('output/path'))
"
```

### Dataset Operations (Requires Isaac Sim)
```bash
# Record teleoperation data (SO101 Leader Arms)
python -m scripts.dataset_sim record \
    --teleop_device bi-so101leader \
    --garment_name Top_Long_Unseen_0 \
    --enable_record \
    --num_episode 10 \
    --device cpu \
    --enable_cameras

# Replay dataset in simulation
python -m scripts.dataset_sim replay \
    --dataset_root Datasets/example/pant_long_merged \
    --device cpu \
    --enable_cameras
```

## Architecture

### Directory Structure
- `source/lehome/lehome/` - Core LeHome package
  - `tasks/bedroom/` - Gym environment registrations and garment manipulation tasks
  - `devices/` - Input devices (keyboard, SO101 Leader Arms)
  - `assets/` - Robots, objects, and scene definitions
  - `utils/` - IK solver, logging, kinematics, success checking
- `scripts/` - Entry points for evaluation and dataset management
  - `eval.py` - Policy evaluation entry point
  - `dataset.py` - Dataset operations (no Isaac Sim)
  - `dataset_sim.py` - Dataset operations requiring Isaac Sim
  - `eval_policy/` - Policy registry and base classes
  - `utils/` - Helper functions for evaluation, replay, recording
- `configs/` - Training configuration files (YAML)
- `Assets/` - Simulation assets (downloaded separately)
- `Datasets/` - Training datasets

### Gym Environments
Registered in `source/lehome/lehome/tasks/bedroom/__init__.py`:
- `LeHome-BiSO101-Direct-Garment-v2` - Dual-arm garment manipulation (primary task)
- `LeHome-SO101-Direct-Garment-v0` - Single-arm variant
- `LeHome-BiSO101-Direct-Garment-fling-v0` - Fling motion variant

### Policy System
Custom policies must:
1. Inherit from `scripts.eval_policy.base_policy.BasePolicy`
2. Implement `select_action(observation: Dict) -> np.ndarray`
3. Register via `@PolicyRegistry.register("policy_name")` decorator
4. Import in `scripts/eval_policy/__init__.py`

### Dataset Format
LeRobot format with these key features:
- `observation.state`: Joint positions (12D for dual-arm)
- `action`: Joint actions (12D for dual-arm)
- `observation.images.{top,left,right}_rgb`: Camera images (480x640)
- `observation.top_depth`: Depth map (optional)
- `observation.ee_pose`: End-effector poses (optional, not recommended)

## Important Notes

### Action Space
- **Dual-arm**: 12 dimensions (6 per arm)
- Dimension order: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
- Use **joint-space control** (`observation.state`/`action`) rather than end-effector poses due to IK solver limitations with SO101 hardware

### Garment Types
- `top_long`, `top_short`, `pant_long`, `pant_short`, `custom`
- Garment definitions in `Assets/objects/Challenge_Garment/Release/`

### Training Configuration
- Specify features explicitly in `input_features`/`output_features` sections
- RGB images use `type: VISUAL`
- Depth maps use `type: STATE`
- Use `rename_map` to bypass visual feature consistency check when using partial cameras

### Evaluation Parameters
| Parameter | Required | Description |
|-----------|----------|-------------|
| `--policy_type` | Yes | `lerobot` or `custom` |
| `--policy_path` | LeRobot | Model checkpoint path |
| `--dataset_root` | LeRobot | Dataset for metadata |
| `--garment_type` | Yes | Garment category |
| `--device` | No | Use `cpu` (required) |
| `--enable_cameras` | No | Enable camera rendering |
