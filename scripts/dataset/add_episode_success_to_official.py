#!/usr/bin/env python
"""
Add episode_success field to official datasets for Evo-RL value function training.

This script reads official LeRobot datasets (all success) and adds the episode_success
field with value 1.0 to all frames, making them compatible with Evo-RL value training.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import shutil

def add_episode_success_to_dataset(
    source_root: Path,
    output_root: Path,
    success_value: float = 1.0,
    videos: bool = True
):
    """
    Add episode_success field to official dataset.

    Args:
        source_root: Path to official dataset (e.g., Datasets/example/pant_long_merged)
        output_root: Path for output dataset
        success_value: Value to assign (1.0 for all official/success data)
        videos: Whether to copy video files
    """
    source_root = Path(source_root)
    output_root = Path(output_root)

    print('='*80)
    print('ADDING episode_success TO OFFICIAL DATASET')
    print('='*80)
    print()

    # Check source exists
    if not source_root.exists():
        raise FileNotFoundError(f'Source dataset not found: {source_root}')

    # Read info.json
    info_file = source_root / 'meta' / 'info.json'
    if not info_file.exists():
        raise FileNotFoundError(f'info.json not found: {info_file}')

    import json
    with open(info_file) as f:
        info = json.load(f)

    print(f'Source dataset: {source_root}')
    print(f'  Episodes: {info.get("total_episodes", 0)}')
    print(f'  Frames: {info.get("total_frames", 0)}')
    print(f'  FPS: {info.get("fps", 30)}')
    print()

    # Update features to include episode_success
    features = info.get('features', {})

    if 'episode_success' in features:
        print('⚠️  episode_success already exists in features')
        response = input('Continue anyway? (y/n): ')
        if response.lower() != 'y':
            print('Aborted')
            return

    # Add episode_success to features
    features['episode_success'] = {
        'dtype': 'float32',
        'shape': [1],
        'names': None
    }

    print(f'Adding episode_success field with value {success_value}')
    print()

    # Create output directory
    output_root = Path(output_root)
    if output_root.exists():
        print(f'⚠️  Output directory exists: {output_root}')
        response = input('Delete and recreate? (y/n): ')
        if response.lower() == 'y':
            shutil.rmtree(output_root)
        else:
            print('Aborted')
            return

    output_root.mkdir(parents=True, exist_ok=True)

    # Copy and update meta files
    meta_output = output_root / 'meta'
    meta_output.mkdir(exist_ok=True)

    # Copy info.json with updated features
    info['features'] = features
    with open(meta_output / 'info.json', 'w') as f:
        json.dump(info, f, indent=2)

    # Copy other meta files
    for meta_file in ['stats.json']:
        src = source_root / 'meta' / meta_file
        if src.exists():
            shutil.copy(src, meta_output / meta_file)

    # Process data files
    data_dir = source_root / 'data'
    output_data_dir = output_root / 'data'
    output_data_dir.mkdir(exist_ok=True)

    total_frames_processed = 0
    total_episodes = 0

    for chunk_dir in sorted(data_dir.glob('chunk-*')):
        chunk_name = chunk_dir.name
        output_chunk_dir = output_data_dir / chunk_name
        output_chunk_dir.mkdir(exist_ok=True)

        print(f'Processing {chunk_name}...')

        for data_file in sorted(chunk_dir.glob('*.parquet')):
            print(f'  Reading {data_file.name}...')

            # Read data
            df = pd.read_parquet(data_file)

            # Add episode_success column as scalar float32
            # Each frame gets episode_success = success_value (scalar, not list)
            df['episode_success'] = np.array([success_value] * len(df), dtype=np.float32)

            # Write updated data
            output_file = output_chunk_dir / data_file.name
            df.to_parquet(output_file, index=False)

            total_frames_processed += len(df)

            # Count episodes in this file
            if 'episode_index' in df.columns:
                episodes_in_file = df['episode_index'].nunique()
                total_episodes += episodes_in_file
                print(f'    Episodes: {episodes_in_file}, Frames: {len(df)}')

    print()
    print('='*80)
    print('DATA PROCESSING COMPLETE')
    print('='*80)
    print(f'Total frames processed: {total_frames_processed}')
    print(f'Total episodes: {total_episodes}')
    print()

    # Handle videos
    if videos:
        print('Copying video files...')
        videos_src = source_root / 'videos'
        videos_dst = output_root / 'videos'

        if videos_src.exists():
            shutil.copytree(videos_src, videos_dst)
            print(f'Videos copied to {videos_dst}')
        else:
            print('No videos directory found')

    # Handle tasks.parquet if exists
    tasks_src = source_root / 'meta' / 'tasks.parquet'
    if tasks_src.exists():
        shutil.copy(tasks_src, meta_output / 'tasks.parquet')
        print('tasks.parquet copied')

    # Handle episodes directory if exists
    episodes_src = source_root / 'meta' / 'episodes'
    if episodes_src.exists():
        episodes_dst = meta_output / 'episodes'
        shutil.copytree(episodes_src, episodes_dst)
        print('episodes directory copied')

    print()
    print('='*80)
    print('SUCCESS!')
    print('='*80)
    print(f'Dataset with episode_success saved to: {output_root}')
    print()
    print('Next steps:')
    print('1. Verify the dataset can be loaded')
    print('2. Merge with evo datasets for value function training')


def main():
    parser = argparse.ArgumentParser(
        description='Add episode_success field to official LeRobot datasets'
    )
    parser.add_argument(
        '--source',
        type=str,
        default='Datasets/example/pant_long_merged',
        help='Path to official dataset'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='Datasets/pant_long_value_training/official_with_success',
        help='Path for output dataset'
    )
    parser.add_argument(
        '--success_value',
        type=float,
        default=1.0,
        help='Value to assign to episode_success (default: 1.0 for all success data)'
    )
    parser.add_argument(
        '--no_videos',
        action='store_true',
        help='Skip copying video files'
    )

    args = parser.parse_args()

    add_episode_success_to_dataset(
        source_root=Path(args.source),
        output_root=Path(args.output),
        success_value=args.success_value,
        videos=not args.no_videos
    )


if __name__ == '__main__':
    main()
