#!/usr/bin/env python3
"""
Merge multiple LeRobot datasets into a single dataset.

Features:
1. Merge all sub-datasets under a folder
2. Filter by suffix pattern (e.g., *_no_depth)
3. Exclude specific directories
4. Dry-run mode to preview

Usage:
    # Preview what will be merged
    python scripts/utils/batch_merge_datasets.py \
        --source Datasets/pant_long_newdata_0329 \
        --suffix _no_depth \
        --dry_run

    # Merge all *_no_depth datasets
    python scripts/utils/batch_merge_datasets.py \
        --source Datasets/pant_long_newdata_0329 \
        --suffix _no_depth \
        --output Datasets/pant_long_merged_no_depth

    # Merge specific datasets
    python scripts/utils/batch_merge_datasets.py \
        --datasets "['Datasets/001_no_depth', 'Datasets/002_no_depth']" \
        --output Datasets/merged

    # Exclude some directories
    python scripts/utils/batch_merge_datasets.py \
        --source Datasets/pant_long_newdata_0329 \
        --suffix _no_depth \
        --exclude merged failure \
        --output Datasets/pant_long_merged_no_depth
"""

import argparse
import json
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.utils.utils import init_logging


def find_datasets(
    source_dir: Path,
    suffix: str | None = None,
    exclude_patterns: list[str] | None = None
) -> list[Path]:
    """Find all LeRobot datasets under a directory."""
    exclude_patterns = exclude_patterns or []
    datasets = []

    for item in sorted(source_dir.iterdir()):
        if not item.is_dir():
            continue
        if item.name == "meta":
            continue
        if not (item / "meta" / "info.json").exists():
            continue
        if suffix and not item.name.endswith(suffix):
            continue
        if any(excl in item.name for excl in exclude_patterns):
            continue
        datasets.append(item)

    return datasets


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple LeRobot datasets"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Source directory containing sub-datasets",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="JSON list of specific dataset paths to merge",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Only merge directories ending with this suffix",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=[],
        help="Directory name patterns to exclude",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for merged dataset",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="merged_dataset",
        help="Repository ID for merged dataset",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview what will be merged without making changes",
    )
    args = parser.parse_args()

    init_logging()

    # Collect dataset paths
    if args.datasets:
        import json as json_mod
        dataset_paths = [Path(p) for p in json_mod.loads(args.datasets)]
    elif args.source:
        source_dir = Path(args.source).resolve()
        dataset_paths = find_datasets(source_dir, args.suffix, args.exclude)
    else:
        parser.error("Either --source or --datasets is required")

    if not dataset_paths:
        print("No datasets found to merge.")
        return

    # Print info
    print("=" * 60)
    print("MERGE DATASETS")
    print("=" * 60)
    print(f"Found {len(dataset_paths)} datasets:")
    print()

    total_episodes = 0
    total_frames = 0
    for p in dataset_paths:
        info_path = p / "meta" / "info.json"
        with open(info_path) as f:
            info = json.load(f)
        eps = info.get("total_episodes", 0)
        frames = info.get("total_frames", 0)
        total_episodes += eps
        total_frames += frames
        print(f"  {p.name}: {eps} eps, {frames} frames")

    print()
    print(f"Total: {total_episodes} episodes, {total_frames} frames")
    print(f"Output: {args.output}")
    print(f"Repo ID: {args.repo_id}")
    print()

    if args.dry_run:
        print("DRY RUN - No changes will be made")
        return

    # Load datasets
    print("Loading datasets...")
    datasets = [
        LeRobotDataset(f"ds_{i:03d}", root=p)
        for i, p in enumerate(dataset_paths, 1)
    ]

    # Merge
    output_dir = Path(args.output)
    print(f"\nMerging to {output_dir}...")
    merged = merge_datasets(datasets, args.repo_id, output_dir)

    print()
    print("=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)
    print(f"Episodes: {merged.meta.total_episodes}")
    print(f"Frames: {merged.meta.total_frames}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
