#!/usr/bin/env python
"""
Merge datasets by first stripping problematic features, then using LeRobot API.

This approach properly handles video concatenation by letting LeRobot's
merge_datasets do the heavy lifting after we've normalized the features.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.utils.utils import init_logging


def strip_features_from_dataset(source_dir: Path, dest_dir: Path, drop_features: list[str]) -> Path:
    """
    Create a copy of dataset with specified features removed.

    Returns path to the stripped dataset.
    """
    import json

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Copy meta directory with modified info.json
    shutil.copytree(source_dir / "meta", dest_dir / "meta", dirs_exist_ok=True)

    # Modify info.json to remove dropped features
    info_path = dest_dir / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    for feat in drop_features:
        info["features"].pop(feat, None)

    # Update totals
    info["total_frames"] = info.get("total_frames", 0)

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    # Copy and modify data files
    dest_data = dest_dir / "data"
    dest_data.mkdir(exist_ok=True)

    for chunk_dir in sorted((source_dir / "data").glob("chunk-*")):
        dest_chunk = dest_data / chunk_dir.name
        dest_chunk.mkdir(exist_ok=True)

        for parquet_file in sorted(chunk_dir.glob("*.parquet")):
            table = pq.read_table(parquet_file)
            # Drop specified columns
            cols_to_keep = [c for c in table.column_names if c not in drop_features]
            table = table.select(cols_to_keep)
            pq.write_table(table, dest_chunk / parquet_file.name)

    # Copy videos (symlink to save space)
    if (source_dir / "videos").exists():
        shutil.copytree(source_dir / "videos", dest_dir / "videos", dirs_exist_ok=True)

    return dest_dir


def merge_with_feature_drop(
    source_base: Path, output_root: Path, repo_id: str, drop_features: list[str], exclude: list[str]
):
    """
    Merge datasets by first stripping features, then using LeRobot merge.
    """
    init_logging()

    print("=" * 80)
    print(f"MERGING SUBDATASETS FROM: {source_base}")
    print(f"Dropping features: {drop_features}")
    print("=" * 80)
    print()

    # Find subdirectories with valid LeRobot datasets
    all_subdirs = sorted(
        [
            d
            for d in source_base.glob("*")
            if d.is_dir() and d.name != "meta" and (d / "meta" / "info.json").exists()
        ]
    )

    # Apply exclusions
    subdirs = []
    for d in all_subdirs:
        excluded = any(excl in d.name for excl in exclude)
        if excluded:
            print(f"  [EXCLUDED] {d.name}")
        else:
            subdirs.append(d)

    if not subdirs:
        print("No valid datasets found to merge")
        return

    print(f"Found {len(subdirs)} datasets to merge")
    for d in subdirs:
        print(f"  - {d.name}")
    print()

    # Create temp directory for stripped datasets
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        stripped_dirs = []

        print("Stripping features from datasets...")
        for subdir in subdirs:
            stripped_dir = tmpdir / subdir.name
            strip_features_from_dataset(subdir, stripped_dir, drop_features)
            stripped_dirs.append(stripped_dir)
            print(f"  Stripped {subdir.name}")

        print()
        print("Loading stripped datasets...")
        datasets = []
        for stripped_dir in stripped_dirs:
            ds = LeRobotDataset(repo_id=stripped_dir.name, root=stripped_dir)
            datasets.append(ds)
            print(f"  Loaded {stripped_dir.name}: {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames")

        print()
        print("Merging datasets using LeRobot API...")
        merged = merge_datasets(datasets, repo_id, output_root)

    print()
    print("=" * 80)
    print("MERGE COMPLETE")
    print("=" * 80)
    print(f"Merged dataset saved to: {output_root}")
    print(f"  Total episodes: {merged.meta.total_episodes}")
    print(f"  Total frames: {merged.meta.total_frames}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Merge datasets with feature dropping")
    parser.add_argument("--source", required=True, help="Source directory with subdatasets")
    parser.add_argument("--output", required=True, help="Output directory for merged dataset")
    parser.add_argument("--repo_id", default="merged_dataset", help="Repository ID for merged dataset")
    parser.add_argument(
        "--drop_features", nargs="*", default=["observation.top_depth"], help="Features to drop"
    )
    parser.add_argument("--exclude", nargs="*", default=[], help="Directory patterns to exclude")

    args = parser.parse_args()

    merge_with_feature_drop(
        source_base=Path(args.source),
        output_root=Path(args.output),
        repo_id=args.repo_id,
        drop_features=args.drop_features,
        exclude=args.exclude,
    )


if __name__ == "__main__":
    main()
