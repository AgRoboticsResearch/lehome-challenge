# LeHome Challenge 2026 - Submission

## Team
[BRL-SROI(r97)]

## Checkpoints

4 separate models, one per garment category:

| Garment Type | Best Checkpoint | Success Rate |
|---|---|---|
| pant_short | 011000 | 90.00% |
| top_long | 015000 | 78.33% |
| pant_long | 019000 | 58.33% |
| top_short | 020000 | 51.67% |
| **Average** | | **69.58%** |

## Evaluation Instructions

### Prerequisites
- Isaac Sim 5.1.0
- Isaac Lab 2.3.1
- Python 3.11
- LeRobot 0.4.3

### Setup
```bash
# Clone and install
git clone https://github.com/lehome-official/lehome-challenge.git
cd lehome-challenge
uv sync

# Install IsaacLab
cd third_party && git clone https://github.com/lehome-official/IsaacLab.git && cd ..
source .venv/bin/activate
./third_party/IsaacLab/isaaclab.sh -i none

# Install LeHome package
uv pip install -e ./source/lehome

# Download assets
hf download lehome/asset_challenge --repo-type dataset --local-dir Assets

# Download example datasets (for metadata)
hf download lehome/dataset_challenge_merged --repo-type dataset --local-dir Datasets/example
```

### Download Checkpoints
Download the checkpoint folders from the provided HuggingFace repo and place them under `outputs/moe_train/`:
```
outputs/moe_train/
  smolvla_moe_expert_pant_short_no_st_proj/checkpoints/011000/pretrained_model/
  smolvla_moe_expert_top_long_no_st_proj/checkpoints/015000/pretrained_model/
  smolvla_moe_expert_pant_long_no_st_proj/checkpoints/019000/pretrained_model/
  smolvla_moe_expert_top_short_no_st_proj/checkpoints/020000/pretrained_model/
```

### Run Evaluation

Evaluate each garment category separately:

```bash
# pant_short (90.00%)
python -m scripts.eval \
    --policy_type lerobot \
    --policy_path outputs/moe_train/smolvla_moe_expert_pant_short_no_st_proj/checkpoints/011000/pretrained_model \
    --dataset_root Datasets/example/pant_short_merged \
    --garment_type "pant_short" \
    --num_episodes 5 \
    --max_steps 600 \
    --enable_cameras \
    --device cpu \
    --headless \
    --task_description "fold the garment on the table"

# top_long (78.33%)
python -m scripts.eval \
    --policy_type lerobot \
    --policy_path outputs/moe_train/smolvla_moe_expert_top_long_no_st_proj/checkpoints/015000/pretrained_model \
    --dataset_root Datasets/example/top_long_merged \
    --garment_type "top_long" \
    --num_episodes 5 \
    --max_steps 600 \
    --enable_cameras \
    --device cpu \
    --headless \
    --task_description "fold the garment on the table"

# pant_long (58.33%)
python -m scripts.eval \
    --policy_type lerobot \
    --policy_path outputs/moe_train/smolvla_moe_expert_pant_long_no_st_proj/checkpoints/019000/pretrained_model \
    --dataset_root Datasets/example/pant_long_merged \
    --garment_type "pant_long" \
    --num_episodes 5 \
    --max_steps 600 \
    --enable_cameras \
    --device cpu \
    --headless \
    --task_description "fold the garment on the table"

# top_short (51.67%)
python -m scripts.eval \
    --policy_type lerobot \
    --policy_path outputs/moe_train/smolvla_moe_expert_top_short_no_st_proj/checkpoints/020000/pretrained_model \
    --dataset_root Datasets/example/top_short_merged \
    --garment_type "top_short" \
    --num_episodes 5 \
    --max_steps 600 \
    --enable_cameras \
    --device cpu \
    --headless \
    --task_description "fold the garment on the table"
```

### Expected Output
Each command prints per-garment and overall success rates. With 5 episodes per garment, total episodes per type = 60 (12 garments x 5 episodes).

## Source Code

The `source_code/` directory contains modified lerobot SmolVLA files required to load our checkpoints. These must be copied into the lerobot package before evaluation:

```bash
# After installing lehome-challenge, copy modified files:
LEROBOT_SMOLVLA=$(python -c "import lerobot.policies.smolvla; import os; print(os.path.dirname(lerobot.policies.smolvla.__file__))")
cp source_code/lerobot_policies_smolvla/* "$LEROBOT_SMOLVLA/"
```

See `source_code/README.md` for details.

## Notes
- Device must be `cpu` to avoid garment physics issues
- Each garment type uses a separate specialist model
- Evaluation uses fixed seed by default (use `--use_random_seed` for random initialization)
