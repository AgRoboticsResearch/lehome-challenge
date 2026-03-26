#!/bin/bash
# Value function training script for LeHome on remote server
# Usage: bash scripts/train_value_remote.sh

set -e

# Configuration
CONFIG_PATH="${1:-configs/train_value_lehome.yaml}"
DATASET_PATH="${2:-/home/hls/codes/lehome-challenge/third_party/Evo-RL/Datasets/official_evo_shuffled}"
OUTPUT_DIR="${3:-/home/hls/codes/lehome-challenge/outputs/value_train}"

# Navigate to Evo-RL directory
cd /home/hls/codes/lehome-challenge/third_party/Evo-RL

echo "=========================================="
echo "LeHome Value Function Training"
echo "=========================================="
echo "Config: $CONFIG_PATH"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Check if dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    echo "Please sync the dataset first using:"
    echo "  rsync -avz /local/path/to/dataset hls@192.168.3.103:/home/hls/codes/lehome-challenge/third_party/Evo-RL/Datasets/"
    exit 1
fi

# Activate conda environment (adjust env name as needed)
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate evo-rl

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi || echo "Warning: nvidia-smi not found. Training might run on CPU."

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run value training
echo "Starting value function training..."
python -m lerobot.scripts.lerobot_value_train \
    --config_path "$CONFIG_PATH" \
    --dataset.root "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --value.device cuda

echo "=========================================="
echo "Training complete!"
echo "Check results at: $OUTPUT_DIR"
echo "=========================================="

# Optional: Display training curve if wandb was used
echo "View training progress at: https://wandb.ai/"
