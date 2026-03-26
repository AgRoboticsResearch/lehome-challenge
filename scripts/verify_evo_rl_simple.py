#!/usr/bin/env python
"""
Simple Evo-RL dataset verification that directly reads JSON files.
Avoids LeRobotDataset initialization issues with local datasets.
"""

import json
from pathlib import Path


def verify_dataset_simple(dataset_path: str):
    """Verify Evo-RL dataset format by directly reading JSON files."""

    print("=" * 60)
    print("Evo-RL Dataset Verification (Simple)")
    print("=" * 60)
    print(f"\nDataset path: {dataset_path}")

    dataset_path = Path(dataset_path)

    # Handle get_next_experiment_path_with_gap() structure
    if dataset_path.is_dir():
        subdirs = sorted([d for d in dataset_path.iterdir() if d.is_dir() and d.name.isdigit()])
        if subdirs:
            latest_dir = subdirs[-1]
            print(f"Found multiple experiment directories, using latest: {latest_dir.name}")
            dataset_path = latest_dir

    print(f"Checking: {dataset_path}\n")

    # Check required files
    meta_dir = dataset_path / "meta"
    info_path = meta_dir / "info.json"
    episodes_path = meta_dir / "episodes.json"

    if not info_path.exists():
        print(f"❌ info.json not found at: {info_path}")
        return

    # Load info.json
    with open(info_path) as f:
        info = json.load(f)

    print("📋 Dataset Info:")
    print(f"  repo_id: {info.get('repo_id', 'N/A')}")
    print(f"  total_episodes: {info.get('total_episodes', 0)}")
    print(f"  total_frames: {info.get('total_frames', 0)}")
    print(f"  fps: {info.get('fps', 'N/A')}")

    # Check Evo-RL features
    print("\n🔍 Checking Evo-RL Features:")
    features = info.get("features", {})

    evo_rl_features = {
        "complementary_info.policy_action": {"dtype": "float32"},
        "complementary_info.is_intervention": {"dtype": "float32", "shape": [1]},
        "complementary_info.state": {"dtype": "float32", "shape": [1]},
        "complementary_info.collector_policy_id": {"dtype": "string", "shape": [1]},
    }

    all_present = True
    for feature_name, expected in evo_rl_features.items():
        if feature_name in features:
            actual = features[feature_name]
            match = True
            if "dtype" in expected:
                match = match and actual.get("dtype") == expected["dtype"]
            if "shape" in expected:
                match = match and actual.get("shape") == expected["shape"]

            status = "✅" if match else "⚠️"
            print(f"  {status} {feature_name}")
            print(f"     dtype: {actual.get('dtype')}, shape: {actual.get('shape', 'N/A')}")
            if not match:
                print(f"     Expected: dtype={expected.get('dtype')}, shape={expected.get('shape')}")
                all_present = False
        else:
            print(f"  ❌ {feature_name} - MISSING")
            all_present = False

    # Check episodes metadata (parquet format in v3.0)
    episodes_parquet_path = dataset_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if episodes_parquet_path.exists():
        import pandas as pd
        episodes_df = pd.read_parquet(episodes_parquet_path)

        print(f"\n📺 Episodes ({len(episodes_df)} total):")

        has_episode_success = "episode_success" in episodes_df.columns
        if has_episode_success:
            print(f"  episode_success metadata: ✅ Present")
            for i, row in episodes_df.iterrows():
                ep_idx = int(row['episode_index'])
                length = int(row['length'])
                success = row['episode_success']
                print(f"    Episode {ep_idx}: {length} frames, success={success}")
        else:
            print(f"  episode_success metadata: ❌ Missing")
    else:
        print(f"\n⚠️  Episodes parquet not found (dataset may not be finalized yet)")

    # Check for data files
    data_dir = dataset_path / "data"
    if data_dir.exists():
        chunks = list(data_dir.glob("chunk-*"))
        print(f"\n📦 Data: {len(chunks)} chunk directories")
    else:
        print(f"\n⚠️  No data directory found")

    # Check for video files
    videos_dir = dataset_path / "videos"
    if videos_dir.exists():
        video_keys = [d.name for d in videos_dir.iterdir() if d.is_dir()]
        print(f"🎬 Videos: {len(video_keys)} video keys - {', '.join(video_keys[:3])}{'...' if len(video_keys) > 3 else ''}")

    # Final verdict
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)

    total_episodes = info.get('total_episodes', 0)
    # Check for episode_success in parquet file
    has_success_metadata = False
    episodes_parquet_path = dataset_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if episodes_parquet_path.exists():
        import pandas as pd
        episodes_df = pd.read_parquet(episodes_parquet_path)
        has_success_metadata = "episode_success" in episodes_df.columns

    if all_present and total_episodes > 0:
        print("✅ Dataset appears to be Evo-RL compatible!")
        print("\nYou can use this dataset for:")
        print("  1. Value function training: lerobot-value-train")
        print("  2. ACP label generation: lerobot-value-infer")

        if has_success_metadata:
            print("  3. episode_success metadata is present ✅")
        else:
            print("  ⚠️  Warning: episode_success metadata may be missing")

    else:
        print("❌ Dataset is NOT fully Evo-RL compatible")
        if not all_present:
            print("   - Some complementary_info features are missing")
        if total_episodes == 0:
            print("   - No episodes saved (evaluation may not have completed)")

    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python scripts/verify_evo_rl_simple.py <dataset_path>")
        print("\nExample:")
        print("  python scripts/verify_evo_rl_simple.py Datasets/test_evo_rl/pant_long")
        sys.exit(1)

    dataset_path = sys.argv[1]
    verify_dataset_simple(dataset_path)
