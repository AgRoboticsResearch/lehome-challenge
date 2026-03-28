#!/usr/bin/env python
"""
Merge multiple sub-datasets into a single dataset using LeRobot API.

This script merges all subdirectories (e.g., 002, 003, ...) into one dataset.
Uses LeRobot's official merge_datasets API when possible, with fallback to
manual merge for datasets with 2D array features (like depth maps).
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.utils.utils import init_logging


def get_feature_hash(info_path: Path, drop_features: list[str] | None = None) -> str:
    """Get a hash of the features to group compatible datasets."""
    drop_features = drop_features or []
    with open(info_path) as f:
        info = json.load(f)
    features = info.get("features", {}).copy()
    # Remove dropped features before hashing
    for feat in drop_features:
        features.pop(feat, None)
    # Create deterministic string representation
    feature_str = json.dumps(features, sort_keys=True)
    return hashlib.md5(feature_str.encode()).hexdigest()[:8]


def has_2d_array_features(info_path: Path) -> bool:
    """Check if dataset has 2D array features that cause LeRobot merge issues."""
    with open(info_path) as f:
        info = json.load(f)
    features = info.get("features", {})
    for name, feat in features.items():
        shape = feat.get("shape", ())
        if len(shape) == 2:  # 2D array like depth map (480, 640)
            return True
    return False


def manual_merge_datasets(
    subdirs: list[Path], output_root: Path, repo_id: str, drop_features: list[str] | None = None
):
    """
    Manually merge datasets with proper episode/frame renumbering.

    This is used as fallback when LeRobot's merge_datasets fails (e.g., with 2D array features).
    Uses pyarrow directly to preserve schema for complex types like 2D arrays.

    Args:
        subdirs: List of dataset directories to merge
        output_root: Output directory for merged dataset
        repo_id: Repository ID for merged dataset
        drop_features: List of feature columns to drop (e.g., ['observation.top_depth'])
    """
    drop_features = drop_features or []
    if drop_features:
        print(f"Using manual merge (dropping features: {drop_features})...")
    else:
        print("Using manual merge (LeRobot API not compatible with dataset features)...")

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "data").mkdir(exist_ok=True)
    (output_root / "videos").mkdir(exist_ok=True)
    (output_root / "meta").mkdir(exist_ok=True)

    # Collect metadata
    total_episodes = 0
    total_frames = 0
    fps = 30
    features = {}

    for subdir in subdirs:
        info_file = subdir / "meta" / "info.json"
        with open(info_file) as f:
            info = json.load(f)
        total_episodes += info.get("total_episodes", 0)
        total_frames += info.get("total_frames", 0)
        fps = info.get("fps", 30)
        if not features:
            features = info.get("features", {})
            # Remove dropped features from metadata
            for feat in drop_features:
                features.pop(feat, None)

    # Create merged info.json
    merged_info = {
        "codebase_version": "v3.0",
        "robot_type": None,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100 * len(subdirs),
        "video_files_size_in_mb": 200 * len(subdirs),
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }

    with open(output_root / "meta" / "info.json", "w") as f:
        json.dump(merged_info, f, indent=2)

    print(f"  Created info.json: {total_episodes} episodes, {total_frames} frames")

    # Merge data with proper renumbering using pyarrow directly
    chunk_idx = 0
    file_idx = 0
    frames_in_chunk = 0
    max_frames_per_chunk = 1000
    global_frame_idx = 0
    episode_offset = 0

    current_chunk_dir = output_root / "data" / f"chunk-{chunk_idx:03d}"
    current_chunk_dir.mkdir(exist_ok=True)

    print("\nMerging data files with renumbering...")

    for subdir in subdirs:
        print(f"  Processing {subdir.name}...")
        data_dir = subdir / "data"
        if not data_dir.exists():
            continue

        # Get episode count for this dataset
        info_file = subdir / "meta" / "info.json"
        with open(info_file) as f:
            info = json.load(f)
        local_episode_count = info.get("total_episodes", 0)

        for chunk_dir in sorted(data_dir.glob("chunk-*")):
            for data_file in sorted(chunk_dir.glob("*.parquet")):
                # Read with pyarrow to preserve schema
                table = pq.read_table(data_file)
                num_rows = table.num_rows

                # Create new columns for renumbering (directly in pyarrow)
                new_columns = {}
                if "episode_index" in table.column_names:
                    old_episode = table.column("episode_index").to_pylist()
                    new_episode = [x + episode_offset for x in old_episode]
                    new_columns["episode_index"] = pa.array(new_episode, type=pa.int64())

                if "index" in table.column_names:
                    # Global frame index
                    new_index = list(range(global_frame_idx, global_frame_idx + num_rows))
                    new_columns["index"] = pa.array(new_index, type=pa.int64())
                    global_frame_idx += num_rows

                # Build new table with updated columns
                if new_columns:
                    # Replace columns in the table
                    for col_name, new_col in new_columns.items():
                        idx = table.column_names.index(col_name)
                        table = table.set_column(idx, col_name, new_col)

                # Drop specified features
                if drop_features:
                    cols_to_keep = [c for c in table.column_names if c not in drop_features]
                    table = table.select(cols_to_keep)

                # Check if we need a new chunk
                if frames_in_chunk + num_rows > max_frames_per_chunk:
                    chunk_idx += 1
                    file_idx = 0
                    frames_in_chunk = 0
                    current_chunk_dir = output_root / "data" / f"chunk-{chunk_idx:03d}"
                    current_chunk_dir.mkdir(exist_ok=True)

                output_file = current_chunk_dir / f"file-{file_idx:03d}.parquet"
                pq.write_table(table, output_file)

                frames_in_chunk += num_rows
                file_idx += 1

        episode_offset += local_episode_count

        # Copy videos
        videos_dir = subdir / "videos"
        if videos_dir.exists():
            output_videos = output_root / "videos"
            for video_key in videos_dir.glob("*"):
                if video_key.is_dir():
                    dest_video_key = output_videos / video_key.name
                    dest_video_key.mkdir(exist_ok=True)

                    for video_chunk in sorted(video_key.glob("chunk-*")):
                        dest_video_chunk = dest_video_key / video_chunk.name
                        dest_video_chunk.mkdir(exist_ok=True)

                        for video_file in sorted(video_chunk.glob("*.mp4")):
                            dest_file = dest_video_chunk / video_file.name
                            if not dest_file.exists():
                                shutil.copy(video_file, dest_file)

    print(f"\n  Data files merged into {chunk_idx + 1} chunks")

    # Merge episodes.parquet
    print("\nMerging episodes metadata...")
    episodes_dfs = []
    episode_offset = 0

    for subdir in subdirs:
        episodes_dir = subdir / "meta" / "episodes"
        if episodes_dir.exists():
            for episodes_chunk in sorted(episodes_dir.glob("chunk-*")):
                for episodes_file in sorted(episodes_chunk.glob("*.parquet")):
                    df = pd.read_parquet(episodes_file)
                    if "episode_index" in df.columns:
                        df["episode_index"] = df["episode_index"] + episode_offset
                    episodes_dfs.append(df)

        # Update offset for next dataset
        info_file = subdir / "meta" / "info.json"
        with open(info_file) as f:
            info = json.load(f)
        episode_offset += info.get("total_episodes", 0)

    if episodes_dfs:
        merged_episodes = pd.concat(episodes_dfs, ignore_index=True)
        merged_episodes_dir = output_root / "meta" / "episodes" / "chunk-000"
        merged_episodes_dir.mkdir(parents=True, exist_ok=True)
        merged_episodes.to_parquet(merged_episodes_dir / "file-000.parquet", index=False)
        print(f"  Merged {len(merged_episodes)} episode records")

    # Copy tasks.parquet and stats.json from first subdir
    tasks_src = subdirs[0] / "meta" / "tasks.parquet"
    if tasks_src.exists():
        shutil.copy(tasks_src, output_root / "meta" / "tasks.parquet")
        print(f"  Copied tasks.parquet")

    stats_src = subdirs[0] / "meta" / "stats.json"
    if stats_src.exists():
        shutil.copy(stats_src, output_root / "meta" / "stats.json")
        print(f"  Copied stats.json")

    # Load and return merged dataset
    return LeRobotDataset(repo_id=repo_id, root=output_root)


def merge_subdirectories(
    source_base: Path,
    output_root: Path,
    repo_id: str,
    exclude: list[str] | None = None,
    drop_features: list[str] | None = None,
):
    """
    Merge all subdirectories into a single dataset using LeRobot API.

    Args:
        source_base: Base directory containing subdirectories (e.g., Datasets/record/xxx)
        output_root: Path for output merged dataset
        repo_id: Repository ID for the merged dataset
        exclude: List of directory name patterns to exclude
        drop_features: List of feature columns to drop (e.g., ['observation.top_depth'])
    """
    init_logging()

    source_base = Path(source_base)
    output_root = Path(output_root)
    exclude = exclude or []
    drop_features = drop_features or []

    print("=" * 80)
    print(f"MERGING SUBDATASETS FROM: {source_base}")
    print("=" * 80)
    print()

    # Check source exists
    if not source_base.exists():
        raise FileNotFoundError(f"Source base not found: {source_base}")

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

    if exclude:
        print()

    if not subdirs:
        print("No valid LeRobot datasets found to merge")
        return

    # Group datasets by feature compatibility (after dropping specified features)
    feature_groups: dict[str, list[Path]] = {}
    for subdir in subdirs:
        info_path = subdir / "meta" / "info.json"
        feature_hash = get_feature_hash(info_path, drop_features)
        if feature_hash not in feature_groups:
            feature_groups[feature_hash] = []
        feature_groups[feature_hash].append(subdir)

    if len(feature_groups) > 1:
        print(f"WARNING: Found {len(feature_groups)} different feature sets.")
        print("Only datasets with matching features can be merged together.")
        print()
        for i, (fhash, group) in enumerate(feature_groups.items(), 1):
            print(f"  Group {i} ({len(group)} datasets):")
            for d in group[:5]:
                print(f"    - {d.name}")
            if len(group) > 5:
                print(f"    ... and {len(group) - 5} more")
        print()

        # Use the largest group
        largest_group = max(feature_groups.values(), key=len)
        print(f"Using largest group with {len(largest_group)} datasets.")
        print("Use --exclude to skip unwanted directories.")
        print()
        subdirs = largest_group

    print(f"Found {len(subdirs)} subdirectories to merge:")
    for subdir in subdirs:
        info_file = subdir / "meta" / "info.json"
        if info_file.exists():
            with open(info_file) as f:
                info = json.load(f)
            print(
                f"  {subdir.name}: {info.get('total_episodes', 0)} episodes, {info.get('total_frames', 0)} frames"
            )
    print()

    # Check if any dataset has 2D array features
    has_problematic_features = any(has_2d_array_features(d / "meta" / "info.json") for d in subdirs)

    # Load datasets using LeRobot API
    print("Loading datasets...")
    datasets = []
    for subdir in subdirs:
        ds = LeRobotDataset(repo_id=subdir.name, root=subdir)
        datasets.append(ds)
        print(f"  Loaded {subdir.name}: {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames")

    print()

    # Try LeRobot API first, fall back to manual merge
    merged = None
    if not has_problematic_features:
        print("Merging datasets using LeRobot API...")
        try:
            merged = merge_datasets(datasets, repo_id, output_root)
        except Exception as e:
            print(f"LeRobot API failed: {e}")
            print("Falling back to manual merge...")
            merged = None

    if merged is None:
        merged = manual_merge_datasets(subdirs, output_root, repo_id, drop_features)

    print()
    print("=" * 80)
    print("MERGE COMPLETE")
    print("=" * 80)
    print(f"Merged dataset saved to: {output_root}")
    print(f"  Total episodes: {merged.meta.total_episodes}")
    print(f"  Total frames: {merged.meta.total_frames}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Merge subdirectories into single dataset using LeRobot API"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Base directory containing subdirectories to merge",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path for merged output dataset",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="merged_dataset",
        help="Repository ID for merged dataset",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=[],
        help="Directory name patterns to exclude (e.g., --exclude merged old_data)",
    )
    parser.add_argument(
        "--drop_features",
        type=str,
        nargs="*",
        default=[],
        help="Feature columns to drop (e.g., --drop_features observation.top_depth)",
    )

    args = parser.parse_args()

    merge_subdirectories(
        source_base=Path(args.source),
        output_root=Path(args.output),
        repo_id=args.repo_id,
        exclude=args.exclude,
        drop_features=args.drop_features,
    )


if __name__ == "__main__":
    main()
