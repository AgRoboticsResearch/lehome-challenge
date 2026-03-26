#!/bin/bash
# Value inference and ACP generation for LeHome Challenge
# Usage: bash scripts/value_infer_lehome.sh [CHECKPOINT_PATH] [OUTPUT_DIR]

set -e

# Configuration
DATASET_PATH="/home/hls/codes/lehome-challenge/third_party/Evo-RL/Datasets/official_evo_shuffled"
CHECKPOINT_PATH="${1}"
OUTPUT_DIR="${2:-/home/hls/codes/lehome-challenge/outputs/value_infer}"
RUN_NAME="lehome_acp_$(date +%Y%m%d_%H%M%S)"
TAG="lehome"  # Tag name for ACP fields

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint path is required"
    echo "Usage: $0 <CHECKPOINT_PATH> [OUTPUT_DIR]"
    exit 1
fi

echo "=========================================="
echo "LeHome Value Inference & ACP Generation"
echo "=========================================="
echo "Dataset: $DATASET_PATH"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output: $OUTPUT_DIR/$RUN_NAME"
echo "Tag: $TAG"
echo "=========================================="

# Navigate to Evo-RL directory
cd /home/hls/codes/lehome-challenge/third_party/Evo-RL

# Check if dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    exit 1
fi

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT_PATH"
    exit 1
fi

# Run value inference with ACP
lerobot-value-infer \
  --dataset.repo_id "$DATASET_PATH" \
  --inference.checkpoint_path "$CHECKPOINT_PATH" \
  --runtime.device cuda \
  --runtime.batch_size 64 \
  --acp.enable true \
  --acp.n_step 50 \
  --acp.positive_ratio 0.3 \
  --acp.value_field "complementary_info.value_${TAG}" \
  --acp.advantage_field "complementary_info.advantage_${TAG}" \
  --acp.indicator_field "complementary_info.acp_indicator_${TAG}" \
  --output_dir "$OUTPUT_DIR" \
  --job_name "$RUN_NAME" \
  --viz.enable true \
  --viz.episodes "0-9" \
  --viz.video_key observation.images.top_rgb

echo "=========================================="
echo "Value inference complete!"
echo ""
echo "New fields added to dataset:"
echo "  - complementary_info.value_${TAG}"
echo "  - complementary_info.advantage_${TAG}"
echo "  - complementary_info.acp_indicator_${TAG}"
echo ""
echo "Visualization videos saved to:"
echo "  $OUTPUT_DIR/$RUN_NAME/value/viz/"
echo "=========================================="
