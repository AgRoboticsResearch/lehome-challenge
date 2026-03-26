#!/bin/bash
# Multi-GPU value function training for LeHome Challenge
# Usage: bash scripts/train_value_lehome_multi_gpu.sh [NUM_GPUS] [OUTPUT_DIR]

set -e

# Configuration
NUM_GPUS="${1:-2}"
DATASET_PATH="/home/hls/codes/lehome-challenge/third_party/Evo-RL/Datasets/official_evo_shuffled"
OUTPUT_DIR="${2:-/home/hls/codes/lehome-challenge/outputs/value_train}"
RUN_NAME="lehome_pistar06_multigpu_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "LeHome Value Function Training (Multi-GPU)"
echo "=========================================="
echo "GPUs: $NUM_GPUS"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR/$RUN_NAME"
echo "=========================================="

# Calculate per-GPU batch size
PER_GPU_BATCH_SIZE=$((32 / NUM_GPUS))

# Navigate to Evo-RL directory
cd /home/hls/codes/lehome-challenge/third_party/Evo-RL

# Check if dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    exit 1
fi

# Run value training with accelerate
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --multi_gpu \
  --num_processes "$NUM_GPUS" \
  --mixed_precision bf16 \
  $(which lerobot-value-train) \
  --dataset.repo_id "$DATASET_PATH" \
  --value.type pistar06 \
  --value.vision_repo_id google/siglip-so400m-patch14-384 \
  --value.language_repo_id google/gemma-3-270m \
  --value.camera_features "[observation.images.top_rgb,observation.images.left_rgb,observation.images.right_rgb]" \
  --value.state_feature observation.state \
  --value.task_field task \
  --value.task_index_feature task_index \
  --value.max_state_dim 32 \
  --value.num_bins 201 \
  --value.bin_min -1.0 \
  --value.bin_max 0.0 \
  --value.dropout 0.1 \
  --value.dtype bfloat16 \
  --value.optimizer_lr 5.0e-5 \
  --value.optimizer_weight_decay 1.0e-5 \
  --value.scheduler_warmup_steps 500 \
  --value.scheduler_decay_steps 8000 \
  --value.device cuda \
  --targets.success_field episode_success \
  --targets.default_success failure \
  --targets.c_fail_coef 1.0 \
  --batch_size "$PER_GPU_BATCH_SIZE" \
  --steps 8000 \
  --num_workers 4 \
  --log_freq 200 \
  --save_freq 4000 \
  --output_dir "$OUTPUT_DIR" \
  --job_name "$RUN_NAME" \
  --wandb.enable true \
  --wandb.project lehome-evo-rl \
  --seed 1000

echo "=========================================="
echo "Training complete!"
echo "Results saved to: $OUTPUT_DIR/$RUN_NAME"
echo "=========================================="
