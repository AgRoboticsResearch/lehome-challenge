#!/usr/bin/env python
"""
Merge all datasets from pant_long_evo_01 into a single dataset.

This script uses LeRobot's official merge_datasets tool to combine
all 60 datasets (5 subdirectories × 12 datasets each) into one.
"""

import argparse
import shutil
from pathlib import Path
from typing import List

from lerobot.datasets.dataset_tools import merge_datasets as lerobot_merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def find_all_datasets(evo_root: Path) -> List[Path]:
    """Find all LeRobot datasets within the evo_root directory.

    Handles both flat and nested structures:
    - Flat: evo_root/001_0319_10/, evo_root/002_0319_10/, ...
    - Nested: evo_root/pant_long_0319_10/001_0319_10/, ...

    Args:
        evo_root: Root directory containing subdirectories of datasets

    Returns:
        List of dataset root directories (each is a complete LeRobot dataset)
    """
    dataset_paths = []

    # First pass: check for flat structure (datasets directly in root)
    for item in evo_root.iterdir():
        if not item.is_dir():
            continue
        if item.name.startswith('.'):
            continue

        meta_dir = item / "meta"
        if meta_dir.exists():
            dataset_paths.append(item)

    # If we found datasets in flat structure, return them
    if dataset_paths:
        return sorted(dataset_paths)

    # Otherwise, assume nested structure (datasets in subdirectories)
    for subdir in evo_root.iterdir():
        if not subdir.is_dir():
            continue
        if subdir.name.startswith('.'):
            continue

        # Find all datasets within this subdirectory
        for dataset_dir in sorted(subdir.iterdir()):
            if not dataset_dir.is_dir():
                continue

            meta_dir = dataset_dir / "meta"
            if meta_dir.exists():
                dataset_paths.append(dataset_dir)

    return sorted(dataset_paths)


def merge_evo_datasets(
    evo_root: Path,
    output_root: Path,
    output_repo_id: str = "pant_long_evo_merged",
    test_mode: bool = False,
    test_limit: int = 3
) -> None:
    """Merge all datasets from evo_root into a single output dataset.

    Args:
        evo_root: Root directory containing subdirectories of datasets
        output_root: Output directory for the merged dataset
        output_repo_id: Repository ID for the merged dataset
        test_mode: If True, only merge first N datasets for testing
        test_limit: Number of datasets to merge in test mode
    """
    print("=" * 80)
    print("MERGE PANT_LONG_EVO_01 DATASETS")
    print("=" * 80)
    print(f"Source: {evo_root}")
    print(f"Output: {output_root}")
    print(f"Repo ID: {output_repo_id}")
    if test_mode:
        print(f"TEST MODE: Merging only first {test_limit} datasets")
    print()

    # Find all datasets
    print("Scanning for datasets...")
    dataset_paths = find_all_datasets(evo_root)
    print(f"Found {len(dataset_paths)} datasets")

    if test_mode:
        dataset_paths = dataset_paths[:test_limit]
        print(f"Test mode: using first {len(dataset_paths)} datasets")

    print()
    for i, path in enumerate(dataset_paths, 1):
        rel_path = path.relative_to(evo_root)
        print(f"  {i:2d}. {rel_path}")

    print()

    # Check if output directory exists
    if output_root.exists():
        print(f"Output directory already exists: {output_root}")
        response = input("Delete and recreate? (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(output_root)
            print("Deleted existing output directory")
        else:
            print("Aborted")
            return

    # Note: Don't create output_root - LeRobot's merge_datasets will create it

    # Load all datasets
    print()
    print("Loading datasets...")
    datasets = []
    for dataset_path in dataset_paths:
        # Each numbered directory IS a complete LeRobot dataset
        # The root should be the dataset_path itself
        repo_id = dataset_path.name
        try:
            dataset = LeRobotDataset(
                repo_id=repo_id,
                root=dataset_path,
            )
            datasets.append(dataset)
            print(f"  ✓ Loaded: {repo_id} ({dataset.meta.total_episodes} episodes, "
                  f"{dataset.meta.total_frames} frames)")
        except Exception as e:
            print(f"  ✗ Failed to load {repo_id}: {e}")
            raise

    print()
    print(f"Loaded {len(datasets)} datasets successfully")
    print(f"Total episodes to merge: {sum(d.meta.total_episodes for d in datasets)}")
    print(f"Total frames to merge: {sum(d.meta.total_frames for d in datasets)}")
    print()

    # Merge datasets using LeRobot's official merge tool
    print("Merging datasets...")
    merged_dataset = lerobot_merge_datasets(
        datasets=datasets,
        output_repo_id=output_repo_id,
        output_dir=output_root,
    )

    print()
    print("=" * 80)
    print("MERGE COMPLETE")
    print("=" * 80)
    print(f"Output: {output_root}")
    print(f"Total episodes: {merged_dataset.meta.total_episodes}")
    print(f"Total frames: {merged_dataset.meta.total_frames}")
    print(f"Video size: {merged_dataset.meta.info['video_files_size_in_mb']} MB")
    print(f"Data size: {merged_dataset.meta.info['data_files_size_in_mb']} MB")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Merge all datasets from pant_long_evo_01"
    )
    parser.add_argument(
        "--evo_root",
        type=str,
        default="Datasets/pant_long_evo_01",
        help="Root directory containing EVO datasets"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="Datasets/pant_long_evo_merged",
        help="Output directory for merged dataset"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="pant_long_evo_merged",
        help="Repository ID for merged dataset"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: only merge first few datasets"
    )
    parser.add_argument(
        "--test_limit",
        type=int,
        default=3,
        help="Number of datasets to merge in test mode (default: 3)"
    )

    args = parser.parse_args()

    merge_evo_datasets(
        evo_root=Path(args.evo_root),
        output_root=Path(args.output),
        output_repo_id=args.repo_id,
        test_mode=args.test,
        test_limit=args.test_limit
    )


if __name__ == "__main__":
    main()
