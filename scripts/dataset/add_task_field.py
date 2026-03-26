#!/usr/bin/env python3
"""
Add task field to LeHome dataset for Pistar06 value training.
Pistar06 requires text task descriptions for the language model.
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd
from tqdm import tqdm


def get_task_mapping() -> Dict[int, str]:
    """Define task descriptions for LeHome garment manipulation."""
    return {
        0: "fold long pants garment on table",  # Specific task for pant_long
        # Add more specific tasks if needed:
        # 1: "fold short pants garment on table",
        # 2: "fold long top garment on table",
        # 3: "fold short top garment on table",
    }


def add_task_field_to_parquet(
    parquet_path: Path,
    task_mapping: Dict[int, str],
    backup: bool = True
) -> None:
    """Add task field to a single parquet file."""
    try:
        # Read parquet file
        df = pd.read_parquet(parquet_path)

        # Skip if task column already exists
        if 'task' in df.columns:
            return

        # Add task column based on task_index
        if 'task_index' in df.columns:
            df['task'] = df['task_index'].map(task_mapping)
            # Fill any unmapped indices with default
            df['task'] = df['task'].fillna(f"task {df['task_index']}")
        else:
            # No task_index, use default task
            df['task'] = "manipulate garment on table"

        # Backup original file if requested
        if backup:
            backup_path = str(parquet_path) + '.backup'
            df_without_task = df.drop(columns=['task'])
            df_without_task.to_parquet(backup_path, index=False)

        # Write updated file
        df.to_parquet(parquet_path, index=False)
        return True

    except Exception as e:
        print(f"Error processing {parquet_path}: {e}")
        return False


def update_info_json(info_path: Path) -> None:
    """Update info.json to include task field in features."""
    with open(info_path, 'r') as f:
        info = json.load(f)

    # Add task field to features
    if 'features' not in info:
        info['features'] = {}

    info['features']['task'] = {
        "dtype": "string",
        "shape": [],
    }

    # Write back
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Add task field to LeHome dataset")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/Volumes/MK-ssd-mini/lehome_datasets/official_evo_shuffled",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Run on remote server (use ssh)"
    )
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="Don't create backup files"
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    task_mapping = get_task_mapping()

    print(f"Adding task field to dataset: {dataset_path}")
    print(f"Task mapping: {task_mapping}")

    # Find all parquet files
    parquet_files = list(dataset_path.glob("data/chunk-*/file-*.parquet"))
    # Filter out backup and metadata files
    parquet_files = [f for f in parquet_files if not f.name.startswith('._')]

    print(f"Found {len(parquet_files)} parquet files")

    # Process each parquet file
    success_count = 0
    for parquet_file in tqdm(parquet_files, desc="Processing parquet files"):
        if add_task_field_to_parquet(parquet_file, task_mapping, backup=not args.no_backup):
            success_count += 1

    print(f"Successfully processed {success_count}/{len(parquet_files)} files")

    # Update info.json
    info_path = dataset_path / "meta" / "info.json"
    if info_path.exists():
        print(f"Updating {info_path}")
        update_info_json(info_path)
    else:
        print(f"Warning: {info_path} not found")

    print("Done!")


if __name__ == "__main__":
    main()
