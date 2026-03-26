#!/usr/bin/env python
"""
Rename subdirectories in pant_long_evo_01 to date-based format.

Parent directories: pant_long_MMDD_HH
- MM: month (03 = March)
- DD: day
- HH: hour (24-hour format)

Sub-folders: ###_MMDD_HH (inherits parent date)
"""

import shutil
from pathlib import Path
from typing import Dict, List


def get_rename_map() -> Dict[str, str]:
    """Get mapping of old to new directory names."""
    return {
        "pant_long": "pant_long_0320_18",  # Modified March 20, 18:02
        "pant_long_03191024": "pant_long_0319_10",  # March 19, 10:24
        "pant_long_03191140": "pant_long_0319_11",  # March 19, 11:40
        "pant_long_03200059": "pant_long_0320_00",  # March 20, 00:59
        "pant_long_0320759": "pant_long_0320_07",   # March 20, 07:59
    }


def extract_date_suffix(new_name: str) -> str:
    """Extract date suffix from new directory name (e.g., '0320_18' from 'pant_long_0320_18')."""
    return new_name.split("_")[-2] + "_" + new_name.split("_")[-1]


def rename_subdirs(evo_root: Path, dry_run: bool = True) -> None:
    """Rename subdirectories and their sub-folders to date-based format.

    Args:
        evo_root: Root directory containing subdirectories to rename
        dry_run: If True, only print what would be done without actually renaming
    """
    rename_map = get_rename_map()

    print("=" * 80)
    if dry_run:
        print("DRY RUN - No actual renaming will be performed")
    else:
        print("RENAME SUBDIRECTORIES - This will modify directory names!")
    print("=" * 80)
    print(f"Root: {evo_root}")
    print()

    # Collect all rename operations (parent dirs first, then sub-folders)
    operations: List[tuple] = []

    for old_parent, new_parent in rename_map.items():
        old_parent_path = evo_root / old_parent
        if not old_parent_path.exists():
            print(f"✗ Source not found: {old_parent}")
            return

        date_suffix = extract_date_suffix(new_parent)

        # First, rename sub-folders within the parent directory
        for sub_folder in sorted(old_parent_path.iterdir()):
            if not sub_folder.is_dir():
                continue
            if sub_folder.name.startswith('.'):
                continue

            # Extract the base name (number) and add date suffix
            old_sub_name = sub_folder.name
            new_sub_name = f"{old_sub_name}_{date_suffix}"
            operations.append((
                "sub_folder",
                sub_folder,
                evo_root / new_parent / new_sub_name,
                f"  {old_parent}/{old_sub_name} → {new_parent}/{new_sub_name}"
            ))

        # Then rename the parent directory itself (this must happen AFTER sub-folders)
        operations.append((
            "parent",
            old_parent_path,
            evo_root / new_parent,
            f"  {old_parent} → {new_parent}"
        ))

    # Check for conflicts in target names
    parent_targets = [rename_map[k] for k in rename_map.keys()]
    if len(parent_targets) != len(set(parent_targets)):
        print("✗ Duplicate target names detected!")
        return

    # Check if any target directories already exist
    conflicts = []
    for op_type, old_path, new_path, _ in operations:
        if new_path.exists() and old_path != new_path:
            conflicts.append(f"  Target already exists: {new_path}")

    if conflicts:
        print("✗ Cannot proceed - conflicts detected:")
        for conflict in conflicts:
            print(conflict)
        return

    # Display rename plan
    print("Rename plan:")
    print()
    parent_ops = [op for op in operations if op[0] == "parent"]
    sub_ops = [op for op in operations if op[0] == "sub_folder"]

    print("Parent directories:")
    for i, (_, _, _, desc) in enumerate(parent_ops, 1):
        print(f"  {i}. {desc}")
    print()

    print(f"Sub-folders ({len(sub_ops)} total):")
    for i, (_, _, _, desc) in enumerate(sub_ops[:10], 1):
        print(f"  {i}. {desc}")
    if len(sub_ops) > 10:
        print(f"  ... and {len(sub_ops) - 10} more")
    print()

    if dry_run:
        print("Run with --execute to perform the actual renaming.")
        return

    # Perform renaming - sub-folders first, then parents
    print("Renaming directories...")
    total = len(operations)

    for i, (op_type, old_path, new_path, desc) in enumerate(operations, 1):
        try:
            shutil.move(str(old_path), str(new_path))
            if op_type == "parent":
                print(f"  [{i}/{total}] {desc}")
            elif i % 20 == 0 or i == len(operations):
                print(f"  [{i}/{total}] Renamed sub-folders...")
        except Exception as e:
            print(f"  ✗ Failed to rename {old_path} to {new_path}: {e}")
            return

    print()
    print("=" * 80)
    print("RENAME COMPLETE")
    print("=" * 80)
    print(f"Renamed {len(parent_ops)} parent directories")
    print(f"Renamed {len(sub_ops)} sub-folders")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Rename subdirectories in pant_long_evo_01 to date-based format"
    )
    parser.add_argument(
        "--evo_root",
        type=str,
        default="Datasets/pant_long_evo_01",
        help="Root directory containing subdirectories to rename"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the renaming (default: dry run)"
    )

    args = parser.parse_args()

    rename_subdirs(
        evo_root=Path(args.evo_root),
        dry_run=not args.execute
    )


if __name__ == "__main__":
    main()
