#!/usr/bin/env python3
"""
Batch process multiple LeRobot datasets to remove depth column.

Features:
1. Process all sub-datasets under a folder
2. Remove depth column and related statistics
3. Support dry-run mode to preview changes

Usage:
    # Preview what will be processed
    python scripts/utils/batch_remove_depth.py \
        --source Datasets/pant_long_newdata_0329 \
        --dry_run

    # Process all datasets (keep originals)
    python scripts/utils/batch_remove_depth.py \
        --source Datasets/pant_long_newdata_0329

    # Exclude specific directories
    python scripts/utils/batch_remove_depth.py \
        --source Datasets/pant_long_newdata_0329 \
        --exclude merged failure

    # After processing, merge with batch_merge_datasets.py:
    python scripts/utils/batch_merge_datasets.py \
        --source Datasets/pant_long_newdata_0329 \
        --suffix _no_depth \
        --output Datasets/merged_no_depth
"""

import argparse
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm


def clean_episodes_table(table: pa.Table, rm_col_name: str) -> pa.Table:
    """Remove depth columns and reset file indices in episodes table."""
    # Remove all columns containing rm_col_name
    cols_to_drop = [c for c in table.column_names if rm_col_name in c]
    if cols_to_drop:
        table = table.drop(cols_to_drop)

    # Reset data/meta file index columns to 0 (NOT video indices)
    # Video file indices should be preserved for multi-file video datasets
    index_cols = [
        c for c in table.column_names
        if (c.endswith("/file_index") or c.endswith("/chunk_index"))
        and not c.startswith("videos/")
    ]
    for col in index_cols:
        col_idx = table.column_names.index(col)
        table = table.remove_column(col_idx)
        zero_array = pa.array([0] * table.num_rows, type=pa.int64())
        table = table.add_column(col_idx, col, zero_array)

    return table


def process_single_dataset(
    src_root: Path,
    dst_root: Path,
    rm_col: str = "observation.top_depth",
    verbose: bool = True
) -> dict:
    """Process a single dataset to remove depth column."""
    result = {
        "source": str(src_root),
        "output": str(dst_root),
        "success": False,
        "episodes": 0,
        "frames": 0,
        "error": None
    }

    try:
        # Check if valid LeRobot dataset
        info_path = src_root / "meta" / "info.json"
        if not info_path.exists():
            result["error"] = "Not a valid LeRobot dataset (missing meta/info.json)"
            return result

        # Read source info
        with open(info_path) as f:
            src_info = json.load(f)
        result["episodes"] = src_info.get("total_episodes", 0)
        result["frames"] = src_info.get("total_frames", 0)

        # Check if depth column exists
        has_depth = rm_col in src_info.get("features", {})
        if not has_depth:
            if verbose:
                print(f"   ⏭️  No depth column found, skipping...")
            result["success"] = True
            result["skipped"] = True
            return result

        # Create output directory
        if dst_root.exists():
            shutil.rmtree(dst_root)
        dst_root.mkdir(parents=True)

        # 1. Process DATA
        dst_data_chunk = dst_root / "data" / "chunk-000"
        dst_data_chunk.mkdir(parents=True)

        data_files = sorted((src_root / "data").rglob("*.parquet"))
        tables = []
        for f in data_files:
            t = pq.read_table(f)
            if rm_col in t.column_names:
                t = t.drop([rm_col])
            tables.append(t)

        if not tables:
            result["error"] = "No data parquet files found"
            return result

        full_table = pa.concat_tables(tables)
        pq.write_table(full_table, dst_data_chunk / "file-000.parquet")
        total_rows = full_table.num_rows

        # 2. Process EPISODES
        dst_ep_chunk = dst_root / "meta" / "episodes" / "chunk-000"
        dst_ep_chunk.mkdir(parents=True)

        ep_files = sorted((src_root / "meta" / "episodes").rglob("*.parquet"))
        if ep_files:
            ep_tables = [pq.read_table(f) for f in ep_files]
            full_ep_table = pa.concat_tables(ep_tables)
            full_ep_table = clean_episodes_table(full_ep_table, rm_col)
            pq.write_table(full_ep_table, dst_ep_chunk / "file-000.parquet")

        # 3. Process META
        for item in (src_root / "meta").glob("*"):
            if item.name == "episodes":
                continue
            dst_item = dst_root / "meta" / item.name

            if item.name == "info.json":
                info = json.loads(item.read_text())
                if "features" in info and rm_col in info["features"]:
                    del info["features"][rm_col]
                info["chunks"] = 1
                info["total_frames"] = total_rows
                dst_item.write_text(json.dumps(info, indent=4))
            elif item.name == "stats.json":
                stats = json.loads(item.read_text())
                if rm_col in stats:
                    del stats[rm_col]
                dst_item.write_text(json.dumps(stats, indent=4))
            elif item.is_dir():
                shutil.copytree(item, dst_item)
            else:
                shutil.copy2(item, dst_item)

        # 4. Copy VIDEOS (exclude depth video)
        if (src_root / "videos").exists():
            shutil.copytree(
                src_root / "videos",
                dst_root / "videos",
                ignore=shutil.ignore_patterns(f"{rm_col}.mp4"),
            )

        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


def find_datasets(source_dir: Path, exclude_patterns: list[str] | None = None) -> list[Path]:
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


def main():
    parser = argparse.ArgumentParser(
        description="Batch process datasets to remove depth column"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source directory containing sub-datasets",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_no_depth",
        help="Suffix for output directories (default: _no_depth)",
    )
    parser.add_argument(
        "--column",
        type=str,
        default="observation.top_depth",
        help="Column to remove (default: observation.top_depth)",
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
        help="Preview what will be processed without making changes",
    )
    args = parser.parse_args()

    source_dir = Path(args.source).resolve()
    exclude_patterns = args.exclude + [args.output_suffix]  # Exclude already processed

    print("=" * 60)
    print("BATCH REMOVE DEPTH COLUMN")
    print("=" * 60)
    print(f"Source: {source_dir}")
    print(f"Column to remove: {args.column}")
    print(f"Output suffix: {args.output_suffix}")
    print(f"Exclude patterns: {exclude_patterns}")
    print()

    # Find datasets
    datasets = find_datasets(source_dir, exclude_patterns)

    if not datasets:
        print("No datasets found to process.")
        return

    print(f"Found {len(datasets)} datasets:")
    total_episodes = 0
    total_frames = 0
    for ds in datasets:
        info_path = ds / "meta" / "info.json"
        with open(info_path) as f:
            info = json.load(f)
        eps = info.get("total_episodes", 0)
        frames = info.get("total_frames", 0)
        has_depth = args.column in info.get("features", {})
        total_episodes += eps
        total_frames += frames
        depth_status = "📐 has depth" if has_depth else "⏭️  no depth"
        print(f"  {ds.name}: {eps} eps, {frames} frames [{depth_status}]")

    print(f"\nTotal: {total_episodes} episodes, {total_frames} frames")
    print()

    if args.dry_run:
        print("DRY RUN - No changes will be made")
        return

    # Process each dataset
    print("Processing datasets...")
    print("-" * 60)

    results = []
    processed_paths = []

    for ds in tqdm(datasets, desc="Processing"):
        output_dir = source_dir / f"{ds.name}{args.output_suffix}"

        # Check if already processed
        if output_dir.exists():
            print(f"\n  ⏭️  {ds.name} already processed, skipping...")
            continue

        print(f"\n📦 {ds.name}")
        result = process_single_dataset(ds, output_dir, args.column)
        results.append(result)

        if result["success"]:
            if result.get("skipped"):
                print(f"   ⏭️  Skipped (no depth column)")
            else:
                print(f"   ✅ Done: {result['episodes']} eps, {result['frames']} frames")
                processed_paths.append(output_dir)
        else:
            print(f"   ❌ Error: {result['error']}")

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    success_count = sum(1 for r in results if r["success"])
    skipped_count = sum(1 for r in results if r.get("skipped"))
    error_count = sum(1 for r in results if not r["success"])

    print(f"Processed: {success_count - skipped_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Errors: {error_count}")

    if error_count > 0:
        print("\nErrors:")
        for r in results:
            if not r["success"]:
                print(f"  - {Path(r['source']).name}: {r['error']}")

    # Hint for merging
    if processed_paths:
        print(f"\n💡 To merge all processed datasets, run:")
        print(f"   python scripts/utils/batch_merge_datasets.py \\")
        print(f"       --source {source_dir} \\")
        print(f"       --suffix {args.output_suffix} \\")
        print(f"       --output <output_path>")


if __name__ == "__main__":
    main()
