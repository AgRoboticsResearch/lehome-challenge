#!/usr/bin/env python
"""
Check episode_success metadata in LeRobot v3.0 dataset format.
"""

import json
import pandas as pd
from pathlib import Path


def check_episode_metadata(dataset_path: str):
    """Check episode metadata in a LeRobot v3.0 dataset."""

    dataset_path = Path(dataset_path)

    print("=" * 60)
    print("Episode Metadata Check (LeRobot v3.0)")
    print("=" * 60)
    print(f"\nDataset path: {dataset_path}")

    # Handle get_next_experiment_path_with_gap() structure
    if dataset_path.is_dir():
        subdirs = sorted([d for d in dataset_path.iterdir() if d.is_dir() and d.name.isdigit()])
        if subdirs:
            latest_dir = subdirs[-1]
            print(f"Found multiple experiment directories, using latest: {latest_dir.name}")
            dataset_path = latest_dir

    print(f"Checking: {dataset_path}\n")

    # Check parquet files for episode data
    data_dir = dataset_path / "data"
    if not data_dir.exists():
        print("❌ No data directory found")
        return

    parquet_files = list(data_dir.glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        print("❌ No parquet files found")
        return

    print(f"📦 Found {len(parquet_files)} parquet file(s)\n")

    # Load the first parquet file
    df = pd.read_parquet(parquet_files[0])

    print("-" * 60)
    print("Episode Information:")
    print("-" * 60)

    # Group by episode_index
    for ep_idx in sorted(df['episode_index'].unique()):
        ep_frames = df[df['episode_index'] == ep_idx]
        num_frames = len(ep_frames)
        start_idx = int(ep_frames['index'].min())
        end_idx = int(ep_frames['index'].max())

        # Check for intervention
        has_intervention = (ep_frames['complementary_info.is_intervention'] > 0.5).any()
        intervention_frames = (ep_frames['complementary_info.is_intervention'] > 0.5).sum()

        print(f"\n  Episode {int(ep_idx)}:")
        print(f"    Frames: {num_frames}")
        print(f"    Index range: {start_idx} to {end_idx}")
        print(f"    Intervention: {'Yes' if has_intervention else 'No'} ({intervention_frames} frames)")

    # Check for episode metadata files
    print("\n" + "-" * 60)
    print("Episode Metadata Files:")
    print("-" * 60)

    meta_dir = dataset_path / "meta"
    if meta_dir.exists():
        meta_files = list(meta_dir.glob("*.json"))
        for f in meta_files:
            print(f"  📄 {f.name}")

            # Check if it contains episode_success
            with open(f) as fp:
                data = json.load(fp)

            # Look for episode_success in the file
            if isinstance(data, dict):
                if 'episode_success' in data:
                    print(f"     ✅ Contains 'episode_success': {data['episode_success']}")

                # Check for episodes key (v3.0 format)
                if 'episodes' in data:
                    print(f"     ✅ Contains 'episodes' key with {len(data['episodes'])} entries")
                    for ep in data['episodes'][:3]:  # Show first 3
                        if 'episode_success' in ep:
                            print(f"        Episode {ep.get('episode_index')}: episode_success = {ep['episode_success']}")

    # Check if episode_success is in the frame data (it shouldn't be, but let's verify)
    print("\n" + "-" * 60)
    print("Frame Data Columns:")
    print("-" * 60)

    if 'episode_success' in df.columns:
        print("  ⚠️  episode_success found in frame data (unusual)")
        print(f"     Values: {df['episode_success'].unique()}")
    else:
        print("  ✅ episode_success NOT in frame data (correct)")
        print("     (Should be in episode metadata, not frame data)")

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print("\nIn LeRobot v3.0, episode_success metadata should be stored")
    print("in the episode metadata, not in frame data.")
    print("\nIf episode_success is missing, it means the evaluation.py")
    print("may not be saving it correctly via extra_episode_metadata.")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python scripts/check_episode_metadata.py <dataset_path>")
        print("\nExample:")
        print("  python scripts/check_episode_metadata.py Datasets/test_hil_manual/pant_long")
        sys.exit(1)

    dataset_path = sys.argv[1]
    check_episode_metadata(dataset_path)
