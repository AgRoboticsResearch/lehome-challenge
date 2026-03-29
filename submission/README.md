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

# Install submission source code
LEROBOT_SMOLVLA=$(python -c "import lerobot.policies.smolvla; import os; print(os.path.dirname(lerobot.policies.smolvla.__file__))")
cp submission/source_code/lerobot_policies_smolvla/* "$LEROBOT_SMOLVLA/"

cp submission/source_code/scripts/eval_policy/__init__.py scripts/eval_policy/__init__.py
cp submission/source_code/scripts/eval_policy/moe_smolvla_policy.py scripts/eval_policy/moe_smolvla_policy.py

# Verify installation
python -c "from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel; print('lerobot OK')"
python -c "from scripts.eval_policy import MoESmolVLAPolicy; print('eval_policy OK')"
```

### Download Checkpoints
Download from [HuggingFace](https://huggingface.co/linsheng888/lehome-challenge-brl-sroi):

```bash
hf download linsheng888/lehome-challenge-brl-sroi --local-dir outputs/moe_train
```

Expected directory structure:
```
outputs/moe_train/
  smolvla_moe_expert_pant_short_no_st_proj/checkpoints/011000/pretrained_model/
  smolvla_moe_expert_top_long_no_st_proj/checkpoints/015000/pretrained_model/
  smolvla_moe_expert_pant_long_no_st_proj/checkpoints/019000/pretrained_model/
  smolvla_moe_expert_top_short_no_st_proj/checkpoints/020000/pretrained_model/
  router/checkpoints/best/router.pt
```

### Run Evaluation

The MoE policy automatically routes each garment to the correct expert model. Simply run with `--policy_type moe_smolvla`:

```bash
# Evaluate all garment types
for garment in pant_short top_long pant_long top_short; do
    python -m scripts.eval \
        --policy_type moe_smolvla \
        --garment_type "$garment" \
        --num_episodes 5 \
        --max_steps 600 \
        --enable_cameras \
        --device cpu \
        --headless \
        --task_description "fold the garment on the table"
done
```

Or evaluate a single garment type:

```bash
python -m scripts.eval \
    --policy_type moe_smolvla \
    --garment_type "pant_short" \
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

See `source_code/README.md` for details on the modified lerobot files and MoE policy implementation.

## Notes
- Device must be `cpu` to avoid garment physics issues
- The MoE policy uses a trained router to classify garment type and routes to the appropriate specialist model
- Sticky routing: the expert is selected on the first frame and locked for the entire episode
- Evaluation uses fixed seed by default (use `--use_random_seed` for random initialization)
