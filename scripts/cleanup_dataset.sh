#!/bin/bash
# Clean up macOS metadata files from dataset directories
# Usage: bash scripts/cleanup_dataset.sh [DATASET_PATH]

set -e

DATASET_PATH="${1:-.}"

echo "=========================================="
echo "Cleaning up macOS metadata files"
echo "=========================================="
echo "Path: $DATASET_PATH"
echo "=========================================="

# Count files before
BEFORE=$(find "$DATASET_PATH" -name '._*' -type f 2>/dev/null | wc -l)
DS_STORE=$(find "$DATASET_PATH" -name '.DS_Store' -type f 2>/dev/null | wc -l)
echo "Found $BEFORE '._*' files"
echo "Found $DS_STORE '.DS_Store' files"

# Delete macOS metadata files
echo "Deleting metadata files..."
find "$DATASET_PATH" -name '._*' -type f -delete 2>/dev/null || true
find "$DATASET_PATH" -name '.DS_Store' -type f -delete 2>/dev/null || true
find "$DATASET_PATH" -name '.Spotlight-V100' -type d -exec rm -rf {} + 2>/dev/null || true
find "$DATASET_PATH" -name '.Trashes' -type d -exec rm -rf {} + 2>/dev/null || true
find "$DATASET_PATH" -name '.fseventsd' -type d -exec rm -rf {} + 2>/dev/null || true

# Count files after
AFTER=$(find "$DATASET_PATH" -name '._*' -type f 2>/dev/null | wc -l)
DS_STORE_AFTER=$(find "$DATASET_PATH" -name '.DS_Store' -type f 2>/dev/null | wc -l)

echo "=========================================="
echo "Cleanup complete!"
echo "Remaining '._*' files: $AFTER"
echo "Remaining '.DS_Store' files: $DS_STORE_AFTER"
echo "=========================================="
