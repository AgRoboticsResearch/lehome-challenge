#!/usr/bin/env python3
"""
Fix dataset format to comply with LeRobot/Evo-RL standards:

1. Move `episode_success` from data frames to episodes table
2. Change scalar field shapes from [] to [1] in info.json

Usage:
    python scripts/fix_dataset_format.py --dataset_root /path/to/dataset
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


def get_episode_success_from_frames(dataset_root: Path) -> dict[int, float]:
    """
    Extract episode_success from data frames.
    Assumes all frames in an episode have the same success value.
    """
    data_dir = dataset_root / "data"
    episodes_with_success = {}

    # Find all parquet files
    parquet_files = sorted(data_dir.glob("*/file-*.parquet"))

    print(f"Found {len(parquet_files)} data files to process...")

    for parquet_path in tqdm(parquet_files, desc="Processing data files"):
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        if "episode_success" not in df.columns:
            continue

        # Group by episode_index and get the unique success value
        for ep_idx, group in df.groupby("episode_index"):
            success_values = group["episode_success"].unique()
            if len(success_values) > 1:
                raise ValueError(
                    f"Episode {ep_idx} has multiple success values: {success_values}. "
                    "This is unexpected - all frames in an episode should have the same success value."
                )

            success_val = float(success_values[0])
            if ep_idx in episodes_with_success and episodes_with_success[ep_idx] != success_val:
                raise ValueError(
                    f"Episode {ep_idx} has conflicting success values across files: "
                    f"{episodes_with_success[ep_idx]} vs {success_val}"
                )
            episodes_with_success[ep_idx] = success_val

    print(f"Extracted episode_success for {len(episodes_with_success)} episodes")
    return episodes_with_success


def update_episodes_table(dataset_root: Path, episode_success_map: dict[int, float]) -> None:
    """Update the episodes table with episode_success column."""
    episodes_dir = dataset_root / "meta" / "episodes"
    output_dir = dataset_root / "meta" / "episodes_fixed"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert float to string ("success"/"failure")
    episode_success_str_map = {
        k: ("success" if v == 1.0 else "failure")
        for k, v in episode_success_map.items()
    }

    # Find all episode parquet files
    episode_files = sorted(episodes_dir.glob("*/file-*.parquet"))

    print(f"Found {len(episode_files)} episode files to update...")

    for parquet_path in tqdm(episode_files, desc="Updating episodes table"):
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        # Add episode_success column as string
        df["episode_success"] = df["episode_index"].map(episode_success_str_map)

        # Reorder columns to put episode_success after length
        cols = list(df.columns)
        if "episode_success" in cols:
            cols.remove("episode_success")
            length_idx = cols.index("length")
            cols.insert(length_idx + 1, "episode_success")
            df = df[cols]

        # Write to new location
        rel_path = parquet_path.relative_to(episodes_dir)
        output_path = output_dir / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)

    print(f"Updated episodes table written to {output_dir}")


def remove_non_dataframe_columns_from_data(dataset_root: Path, output_root: Path) -> None:
    """Remove task and episode_success columns from all data frames.

    These fields should NOT be in the data frames:
    - task is added dynamically during __getitem__ based on task_index
    - episode_success is only in the episodes table, not in data frames
    """
    data_dir = dataset_root / "data"
    output_data_dir = output_root / "data"

    output_data_dir.mkdir(parents=True, exist_ok=True)

    # Find all parquet files
    parquet_files = sorted(data_dir.glob("*/file-*.parquet"))

    print(f"Removing 'task' and 'episode_success' from {len(parquet_files)} data files...")

    columns_to_remove = ["task", "episode_success"]
    removed_count = {"task": 0, "episode_success": 0}

    for parquet_path in tqdm(parquet_files, desc="Removing columns from data"):
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        for col in columns_to_remove:
            if col in df.columns:
                df = df.drop(columns=[col])
                removed_count[col] += 1

        # Write to new location
        rel_path = parquet_path.relative_to(data_dir)
        output_path = output_data_dir / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)

    print(f"Data files written to {output_data_dir}")
    print(f"  Removed 'task' from {removed_count['task']} files")
    print(f"  Removed 'episode_success' from {removed_count['episode_success']} files")


def update_info_json(dataset_root: Path, output_root: Path) -> None:
    """Update info.json to change scalar field shapes from [] to [1]."""
    info_path = dataset_root / "meta" / "info.json"

    with open(info_path) as f:
        info = json.load(f)

    # IMPORTANT: task and episode_success should NOT be in features!
    # - task is added dynamically during __getitem__ based on task_index
    # - episode_success is only in episodes table, not in data frames
    # Official LeRobot datasets do NOT have these in features.

    # Remove task from features - it's added dynamically during loading
    if "task" in info["features"]:
        del info["features"]["task"]
        print(f"Removed 'task' from features (added dynamically during loading)")

    # Remove episode_success from features since it's only in episodes table, not data frames
    if "episode_success" in info["features"]:
        del info["features"]["episode_success"]
        print(f"Removed 'episode_success' from features (it's only in episodes table)")

    # Write updated info.json
    output_info_path = output_root / "meta" / "info.json"
    output_info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_info_path, "w") as f:
        json.dump(info, f, indent=4)

    print(f"Updated info.json written to {output_info_path}")


def copy_other_files(dataset_root: Path, output_root: Path) -> None:
    """Copy other files that don't need modification."""
    # Copy videos directory (if exists)
    videos_src = dataset_root / "videos"
    if videos_src.exists():
        videos_dst = output_root / "videos"
        import shutil
        shutil.copytree(videos_src, videos_dst)
        print(f"Copied videos to {videos_dst}")

    # Copy stats.json
    stats_src = dataset_root / "meta" / "stats.json"
    if stats_src.exists():
        stats_dst = output_root / "meta" / "stats.json"
        stats_dst.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(stats_src, stats_dst)
        print(f"Copied stats.json to {stats_dst}")

    # Copy tasks.parquet
    tasks_src = dataset_root / "meta" / "tasks.parquet"
    if tasks_src.exists():
        tasks_dst = output_root / "meta" / "tasks.parquet"
        import shutil
        shutil.copy2(tasks_src, tasks_dst)
        print(f"Copied tasks.parquet to {tasks_dst}")


def main():
    parser = argparse.ArgumentParser(description="Fix dataset format for LeRobot/Evo-RL")
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Path to the dataset directory"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Path to the output directory (default: <dataset_root>_fixed)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only analyze without making changes"
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    if args.output_root:
        output_root = Path(args.output_root)
    else:
        output_root = Path(f"{dataset_root}_fixed")

    print(f"Input dataset: {dataset_root}")
    print(f"Output dataset: {output_root}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Step 1: Extract episode_success from data frames
    print("=" * 60)
    print("Step 1: Extracting episode_success from data frames...")
    print("=" * 60)
    episode_success_map = get_episode_success_from_frames(dataset_root)

    # Show statistics
    success_count = sum(1 for v in episode_success_map.values() if v == 1.0)
    failure_count = sum(1 for v in episode_success_map.values() if v == 0.0)
    print(f"\nEpisode success statistics:")
    print(f"  Success: {success_count}")
    print(f"  Failure: {failure_count}")
    print(f"  Total: {len(episode_success_map)}")

    if args.dry_run:
        print("\nDry run - stopping here.")
        return

    # Step 2: Update episodes table
    print("\n" + "=" * 60)
    print("Step 2: Updating episodes table...")
    print("=" * 60)
    update_episodes_table(dataset_root, episode_success_map)

    # Step 3: Remove task and episode_success from data frames
    print("\n" + "=" * 60)
    print("Step 3: Removing 'task' and 'episode_success' from data frames...")
    print("=" * 60)
    remove_non_dataframe_columns_from_data(dataset_root, output_root)

    # Step 4: Update info.json
    print("\n" + "=" * 60)
    print("Step 4: Updating info.json...")
    print("=" * 60)
    update_info_json(dataset_root, output_root)

    # Step 5: Copy other files
    print("\n" + "=" * 60)
    print("Step 5: Copying other files...")
    print("=" * 60)
    copy_other_files(dataset_root, output_root)

    # Copy the fixed episodes table to output
    import shutil
    episodes_fixed_src = dataset_root / "meta" / "episodes_fixed"
    episodes_fixed_dst = output_root / "meta" / "episodes"
    shutil.copytree(episodes_fixed_src, episodes_fixed_dst)
    print(f"Copied fixed episodes table to {episodes_fixed_dst}")

    print("\n" + "=" * 60)
    print("✅ Dataset format fix completed!")
    print("=" * 60)
    print(f"Fixed dataset saved to: {output_root}")
    print("\nChanges made:")
    print("  1. Moved episode_success from data frames to episodes table")
    print("  2. Removed 'task' from data frames (added dynamically during loading)")
    print("  3. Removed 'task' from features (not a data frame feature)")
    print("  4. Removed 'episode_success' from features (only in episodes table)")
    print("\nYou can now use the fixed dataset with lerobot-value-train:")
    print(f"  lerobot-value-train --dataset.repo_id={output_root}")


if __name__ == "__main__":
    main()
