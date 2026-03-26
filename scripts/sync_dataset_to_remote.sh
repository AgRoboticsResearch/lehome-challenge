#!/bin/bash
# Sync dataset to remote server, excluding macOS metadata files
# Usage: bash scripts/sync_dataset_to_remote.sh [SOURCE_PATH] [DEST_PATH] [REMOTE_USER@HOST]

set -e

SOURCE_PATH="${1:-/Volumes/MK-ssd-mini/lehome_datasets/official_evo_shuffled}"
DEST_PATH="${2:-/home/hls/codes/lehome-challenge/third_party/Evo-RL/Datasets/}"
REMOTE="${3:-hls@192.168.3.102}"

echo "=========================================="
echo "Syncing Dataset to Remote Server"
echo "=========================================="
echo "Source: $SOURCE_PATH"
echo "Destination: $REMOTE:$DEST_PATH"
echo "=========================================="

# Check if source exists
if [ ! -d "$SOURCE_PATH" ]; then
    echo "ERROR: Source directory not found: $SOURCE_PATH"
    exit 1
fi

# Sync with exclusions for macOS metadata files
rsync -avz --progress \
    --exclude='._*' \
    --exclude='.DS_Store' \
    --exclude='.Spotlight-V100' \
    --exclude='.Trashes' \
    --exclude='.fseventsd' \
    "$SOURCE_PATH" "$REMOTE:$DEST_PATH"

echo "=========================================="
echo "Sync complete!"
echo "=========================================="

# Clean up any existing metadata files on remote
echo "Cleaning up metadata files on remote..."
ssh "$REMOTE" "find $DEST_PATH -name '._*' -type f -delete 2>/dev/null || true"
echo "Cleanup complete!"
