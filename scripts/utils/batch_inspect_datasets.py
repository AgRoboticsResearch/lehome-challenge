#!/usr/bin/env python3
"""
Batch inspect all LeRobot datasets under a directory.

Shows a summary table of episodes, frames, features, and garment types,
then optionally runs detailed inspection on each dataset.

Usage:
    # Summary only
    python scripts/utils/batch_inspect_datasets.py \
        --source Datasets/pant_long_0331

    # Summary with detailed inspection of each dataset
    python scripts/utils/batch_inspect_datasets.py \
        --source Datasets/pant_long_0331 \
        --detailed

    # Detailed inspection with sample frames and stats
    python scripts/utils/batch_inspect_datasets.py \
        --source Datasets/pant_long_0331 \
        --detailed --show_frames 3 --show_stats

    # Exclude specific directories
    python scripts/utils/batch_inspect_datasets.py \
        --source Datasets/pant_long_0331 \
        --exclude pant_long_og merged
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from .dataset_inspection import inspect


def find_datasets(
    source_dir: Path,
    exclude_patterns: list[str] | None = None,
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
        if any(excl in item.name for excl in exclude_patterns):
            continue
        datasets.append(item)

    return datasets


def print_summary(datasets: list[Path]) -> None:
    """Print a summary table of all datasets."""
    print("=" * 80)
    print("BATCH DATASET INSPECTION")
    print("=" * 80)
    print(f"Found {len(datasets)} datasets\n")

    # Header
    print(f"{'Dataset':<50} {'Eps':>5} {'Frames':>7} {'Features'}")
    print("-" * 80)

    total_eps = 0
    total_frames = 0
    feature_sets: dict[str, int] = {}

    for ds in datasets:
        info_path = ds / "meta" / "info.json"
        with open(info_path) as f:
            info = json.load(f)

        eps = info.get("total_episodes", 0)
        frames = info.get("total_frames", 0)
        total_eps += eps
        total_frames += frames

        features = info.get("features", {})
        obs_keys = sorted(
            [k for k in features if k.startswith("observation.")]
        )
        feat_label = ", ".join(
            k.replace("observation.", "") for k in obs_keys
        )
        # Track unique feature sets
        feat_sig = ",".join(sorted(features.keys()))
        feature_sets[feat_sig] = feature_sets.get(feat_sig, 0) + 1

        # Garment info
        garment_path = ds / "meta" / "garment_info.json"
        garment_label = ""
        if garment_path.exists():
            try:
                gi = json.loads(garment_path.read_text())
                garment_label = ", ".join(gi.keys())
            except Exception:
                pass

        name = ds.name
        if len(name) > 48:
            name = name[:45] + "..."
        print(f"{name:<50} {eps:>5} {frames:>7}   {feat_label}")
        if garment_label:
            print(f"{'':50} {'':5} {'':7}   garments: {garment_label}")

    print("-" * 80)
    print(f"{'TOTAL':<50} {total_eps:>5} {total_frames:>7}")
    print()

    # Feature schema groups
    if len(feature_sets) > 1:
        print("⚠️  Schema mismatch detected — datasets have different features:")
        for sig, count in sorted(feature_sets.items(), key=lambda x: -x[1]):
            keys = sig.split(",")
            comp_cols = [k for k in keys if k.startswith("complementary_info.")]
            base_cols = [k for k in keys if not k.startswith("complementary_info.")]
            label = f"{count} datasets"
            if comp_cols:
                label += f" (+ {len(comp_cols)} complementary_info cols)"
            print(f"  {label}")
        print()
    else:
        print("✅ All datasets share the same feature schema")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Batch inspect all LeRobot datasets under a directory"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Directory containing sub-datasets",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=[],
        help="Directory name patterns to exclude",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Run detailed inspection on each dataset",
    )
    parser.add_argument(
        "--show_frames",
        type=int,
        default=None,
        help="Number of sample frames to show (requires --detailed)",
    )
    parser.add_argument(
        "--show_stats",
        action="store_true",
        help="Show column statistics (requires --detailed)",
    )
    args = parser.parse_args()

    source_dir = Path(args.source).resolve()
    datasets = find_datasets(source_dir, args.exclude)

    if not datasets:
        print("No datasets found.")
        return

    print_summary(datasets)

    if not args.detailed:
        print("Tip: Use --detailed to inspect each dataset individually")
        return

    # Detailed inspection
    for ds in datasets:
        print()
        inspect(ds, show_frames=args.show_frames, show_stats=args.show_stats)
        print()


if __name__ == "__main__":
    main()
