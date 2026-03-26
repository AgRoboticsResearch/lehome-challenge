#!/bin/bash

# Watch wandb directory size and auto-clean when exceeding limit
# Usage: bash scripts/watch_wandb.sh [MAX_SIZE_GB] [CHECK_INTERVAL_SECONDS]

# Auto-detect wandb directory
detect_wandb_dir() {
    if [ -n "$WANDB_DIR" ]; then
        echo "$WANDB_DIR"
        return
    fi

    # Check common locations in order
    for dir in \
        "$(pwd)/wandb" \
        "$HOME/.local/share/wandb" \
        "$HOME/.wandb"
    do
        if [ -d "$dir" ]; then
            echo "$dir"
            return
        fi
    done

    # Fallback to default
    echo "$HOME/.local/share/wandb"
}

# Artifact cache directory (separate from run directory)
ARTIFACT_CACHE_DIR="$HOME/.cache/wandb/artifacts"

WANDB_DIR=$(detect_wandb_dir)
MAX_SIZE_GB=${1:-1}  # 默认 1GB
CHECK_INTERVAL=${2:-300}  # 默认每5分钟检查一次

echo "Watching wandb directory: $WANDB_DIR"
echo "Artifact cache: $ARTIFACT_CACHE_DIR"
echo "Max size: ${MAX_SIZE_GB} GB"
echo "Check interval: ${CHECK_INTERVAL}s"
echo "Press Ctrl+C to stop"
echo "---"

while true; do
    if [ -d "$WANDB_DIR" ]; then
        # Get size in KB, convert to GB
        size_kb=$(du -sk "$WANDB_DIR" 2>/dev/null | cut -f1)
        current_size=$((size_kb / 1024 / 1024))

        if [ "$current_size" -ge "$MAX_SIZE_GB" ]; then
            echo "[$(date)] wandb size ($current_size GB) > ${MAX_SIZE_GB} GB, syncing and cleaning..."

            # Sync all runs to cloud
            wandb sync --sync-all --sync-tensorboard 2>/dev/null

            # Clean staging artifacts (temp files from failed uploads)
            staging_dir="$WANDB_DIR/artifacts/staging"
            if [ -d "$staging_dir" ]; then
                staging_size=$(du -sh "$staging_dir" 2>/dev/null | cut -f1)
                echo "Cleaning staging artifacts: $staging_size"
                rm -rf "$staging_dir"/*
            fi

            # Clean artifact cache (downloaded model checkpoints, etc.)
            if [ -d "$ARTIFACT_CACHE_DIR" ]; then
                cache_size=$(du -sh "$ARTIFACT_CACHE_DIR" 2>/dev/null | cut -f1)
                echo "Cleaning artifact cache: $cache_size"
                wandb artifact cache cleanup 1GB 2>/dev/null || rm -rf "$ARTIFACT_CACHE_DIR"/obj/*
            fi

            # Find and delete old runs, keep the current one (latest-run symlink)
            latest_run=$(readlink -f "$WANDB_DIR/latest-run" 2>/dev/null)

            if [ -n "$latest_run" ]; then
                find "$WANDB_DIR" -maxdepth 1 -type d -name "run-*" | while read dir; do
                    if [ "$dir" != "$latest_run" ]; then
                        echo "Removing old run: $dir"
                        rm -rf "$dir"
                    fi
                done
            fi

            # Recalculate size after cleaning
            size_kb=$(du -sk "$WANDB_DIR" 2>/dev/null | cut -f1)
            current_size=$((size_kb / 1024 / 1024))
            echo "[$(date)] Done. Size now: ${current_size} GB"
        else
            echo "[$(date)] wandb size: ${current_size} GB - OK"
        fi
    else
        echo "[$(date)] wandb directory not found, waiting..."
    fi

    sleep $CHECK_INTERVAL
done
