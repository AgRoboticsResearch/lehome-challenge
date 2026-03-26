#!/usr/bin/env python3
"""
Add episode_success column to episodes table in a LeRobot dataset.

This script is for datasets that are otherwise correctly formatted but
are missing the episode_success column in the episodes table.

Usage:
    # Label all episodes as success
    python scripts/add_episode_success.py --dataset_root /path/to/dataset --all_success

    # Label all episodes as failure
    python scripts/add_episode_success.py --dataset_root /path/to/dataset --all_failure

    # Label specific episodes
    python scripts/add_episode_success.py --dataset_root /path/to/dataset --success_episodes 0-199,201-249
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


def parse_episode_list(episodes_str: str) -> set[int]:
    """Parse episode list like '0-10,20,30-35' into a set of integers."""
    episodes = set()
    for part in episodes_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            episodes.update(range(int(start), int(end) + 1))
        else:
            episodes.add(int(part))
    return episodes


def add_episode_success(
    dataset_root: Path,
    success_episodes: set[int] | None = None,
    all_success: bool = False,
    all_failure: bool = False,
    inplace: bool = False,
) -> Path:
    """
    Add episode_success column to episodes table.

    Returns path to the updated dataset.
    """
    if all_success:
        success_episodes = None  # Will mark all as success
    elif all_failure:
        success_episodes = set()  # Empty set = no successes
    elif success_episodes is None:
        raise ValueError("Must specify --all_success, --all_failure, or --success_episodes")

    # Determine output path
    if inplace:
        output_root = dataset_root
    else:
        output_root = Path(f"{dataset_root}_with_success")

    # Copy dataset structure if not inplace
    if not inplace:
        import shutil
        print(f"Copying dataset to {output_root}...")
        if output_root.exists():
            shutil.rmtree(output_root)
        shutil.copytree(dataset_root, output_root)

    # Update episodes table
    episodes_dir = output_root / "meta" / "episodes"
    episode_files = sorted(episodes_dir.glob("*/file-*.parquet"))

    print(f"Found {len(episode_files)} episode files to update...")

    total_episodes = 0
    for parquet_path in tqdm(episode_files, desc="Adding episode_success"):
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        # Add episode_success column
        if success_episodes is None:  # all_success
            df["episode_success"] = "success"
        else:
            df["episode_success"] = df["episode_index"].apply(
                lambda x: "success" if x in success_episodes else "failure"
            )

        # Reorder columns to put episode_success after length
        cols = list(df.columns)
        if "episode_success" in cols:
            cols.remove("episode_success")
            # Find length column index
            try:
                length_idx = cols.index("length")
                cols.insert(length_idx + 1, "episode_success")
            except ValueError:
                # length column not found, just append
                cols.append("episode_success")
            df = df[cols]

        # Write back
        df.to_parquet(parquet_path, index=False)
        total_episodes += len(df)

    # Update info.json to add episode_success to the info
    info_path = output_root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    # Note: episode_success is NOT in features dict - it's only in episodes table
    # This is correct according to LeRobot format

    # Write back info.json (unchanged, but ensuring format is correct)
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n✅ Added episode_success to {total_episodes} episodes")
    print(f"Output: {output_root}")

    # Show statistics
    if success_episodes is None:  # all_success
        print(f"  Success: {total_episodes}")
        print(f"  Failure: 0")
    else:
        success_count = len(success_episodes)
        print(f"  Success: {success_count}")
        print(f"  Failure: {total_episodes - success_count}")

    return output_root


def main():
    parser = argparse.ArgumentParser(
        description="Add episode_success column to episodes table"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--all_success",
        action="store_true",
        help="Label all episodes as success",
    )
    parser.add_argument(
        "--all_failure",
        action="store_true",
        help="Label all episodes as failure",
    )
    parser.add_argument(
        "--success_episodes",
        type=str,
        default=None,
        help="Comma-separated list of successful episodes, e.g., '0-10,20,30-35'",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Modify dataset in place (default: create copy with '_with_success' suffix)",
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    success_episodes = None
    if args.success_episodes:
        success_episodes = parse_episode_list(args.success_episodes)

    add_episode_success(
        dataset_root=dataset_root,
        success_episodes=success_episodes,
        all_success=args.all_success,
        all_failure=args.all_failure,
        inplace=args.inplace,
    )


if __name__ == "__main__":
    main()
