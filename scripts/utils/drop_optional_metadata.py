#!/usr/bin/env python3
"""
Drop complementary_info metadata from LeRobot datasets so they can be merged.

Works in-place: rewrites parquet data files and meta files directly.
Useful when some datasets were recorded with policy-assisted collection
and have extra complementary_info.* columns that others lack.

Usage:
    # Preview which datasets have complementary_info
    python scripts/utils/drop_optional_metadata.py \
        --source Datasets/pant_long_0331 \
        --dry_run

    # Drop complementary_info from all datasets (in-place)
    python scripts/utils/drop_optional_metadata.py \
        --source Datasets/pant_long_0331

    # Exclude specific directories
    python scripts/utils/drop_optional_metadata.py \
        --source Datasets/pant_long_0331 \
        --exclude pant_long_og merged
"""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

COL_PREFIX = "complementary_info."


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


def get_complementary_cols(features: dict) -> list[str]:
    """Return feature keys that are complementary_info columns."""
    return [k for k in features if k.startswith(COL_PREFIX)]


def drop_complementary_info(dataset_path: Path, dry_run: bool = False) -> dict:
    """Drop complementary_info columns from a dataset in-place."""
    info_path = dataset_path / "meta" / "info.json"

    with open(info_path) as f:
        info = json.load(f)

    cols = get_complementary_cols(info.get("features", {}))
    episodes = info.get("total_episodes", 0)
    frames = info.get("total_frames", 0)

    if not cols:
        return {"success": True, "skipped": True, "episodes": episodes, "frames": frames}

    if dry_run:
        return {"success": True, "skipped": False, "episodes": episodes, "frames": frames, "cols": cols}

    # 1. Rewrite data parquet files
    data_files = sorted((dataset_path / "data").rglob("*.parquet"))
    for pf in data_files:
        table = pq.read_table(pf)
        drop_cols = [c for c in cols if c in table.column_names]
        if drop_cols:
            table = table.drop(drop_cols)
            pq.write_table(table, pf)

    # 2. Clean info.json
    for c in cols:
        del info["features"][c]
    info_path.write_text(json.dumps(info, indent=4))

    # 3. Clean stats.json if present
    stats_path = dataset_path / "meta" / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        changed = False
        for c in cols:
            if c in stats:
                del stats[c]
                changed = True
        if changed:
            stats_path.write_text(json.dumps(stats, indent=4))

    return {"success": True, "skipped": False, "episodes": episodes, "frames": frames, "cols": cols}


def main():
    parser = argparse.ArgumentParser(
        description="Drop complementary_info metadata from LeRobot datasets (in-place)"
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
        "--dry_run",
        action="store_true",
        help="Preview without modifying files",
    )
    args = parser.parse_args()

    source_dir = Path(args.source).resolve()
    datasets = find_datasets(source_dir, args.exclude)

    print("=" * 60)
    print("DROP OPTIONAL METADATA (complementary_info)")
    print("=" * 60)
    print(f"Source: {source_dir}")
    print()

    if not datasets:
        print("No datasets found.")
        return

    # Scan and report
    has_meta = []
    clean = []
    for ds in datasets:
        info_path = ds / "meta" / "info.json"
        with open(info_path) as f:
            info = json.load(f)
        cols = get_complementary_cols(info.get("features", {}))
        eps = info.get("total_episodes", 0)
        frames = info.get("total_frames", 0)
        if cols:
            has_meta.append((ds, eps, frames, cols))
        else:
            clean.append((ds, eps, frames))

    print(f"Datasets with complementary_info: {len(has_meta)}")
    for ds, eps, frames, cols in has_meta:
        print(f"  {ds.name}: {eps} eps, {frames} frames  [{', '.join(cols)}]")

    print(f"\nClean datasets (no action needed): {len(clean)}")
    print()

    if args.dry_run:
        print(f"DRY RUN - Would modify {len(has_meta)} datasets")
        return

    # Process
    print(f"Processing {len(has_meta)} datasets...")
    print("-" * 60)

    modified = 0
    for ds, _, _, _ in tqdm(has_meta, desc="Dropping metadata"):
        result = drop_complementary_info(ds)
        if result["success"] and not result.get("skipped"):
            print(f"  {ds.name}: dropped {len(result['cols'])} columns")
            modified += 1

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Modified: {modified} datasets")
    print(f"Clean (unchanged): {len(clean)}")

    if modified > 0:
        print(f"\nTo merge all datasets, run:")
        print(f"  python scripts/utils/batch_merge_datasets.py \\")
        print(f"      --source {source_dir} \\")
        print(f"      --output <output_path>")


if __name__ == "__main__":
    main()
