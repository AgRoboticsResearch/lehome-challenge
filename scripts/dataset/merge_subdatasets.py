#!/usr/bin/env python
"""
Merge multiple sub-datasets into a single dataset for Evo-RL training.

This script merges all subdirectories (e.g., 002, 003, ...) into one dataset.
"""

import argparse
import shutil
from pathlib import Path
from typing import List

def merge_subdirectories(
    source_base: Path,
    output_root: Path,
    repo_id: str = 'pant_long_evo_merged'
):
    """
    Merge all subdirectories into a single dataset.

    Args:
        source_base: Base directory containing subdirectories (e.g., Datasets/pant_long_evo_01/pant_long)
        output_root: Path for output merged dataset
        repo_id: Repository ID for the merged dataset
    """
    source_base = Path(source_base)
    output_root = Path(output_root)

    print('='*80)
    print(f'MERGING SUBDATASETS FROM: {source_base}')
    print('='*80)
    print()

    # Check source exists
    if not source_base.exists():
        raise FileNotFoundError(f'Source base not found: {source_base}')

    # Get all subdirectories (exclude meta)
    subdirs = sorted([d for d in source_base.glob('*') if d.is_dir() and d.name != 'meta'])

    if not subdirs:
        print('No subdirectories found to merge')
        return

    print(f'Found {len(subdirs)} subdirectories to merge:')
    for subdir in subdirs:
        info_file = subdir / 'meta' / 'info.json'
        if info_file.exists():
            import json
            with open(info_file) as f:
                info = json.load(f)
            print(f'  {subdir.name}: {info.get("total_episodes", 0)} episodes, {info.get("total_frames", 0)} frames')
        else:
            print(f'  {subdir.name}: (no info.json)')
    print()

    # Strategy: Use symlinks to create a unified view
    # We'll create a new dataset structure with symlinks to original data

    output_root.mkdir(parents=True, exist_ok=True)

    # Create output directories
    (output_root / 'data').mkdir(exist_ok=True)
    (output_root / 'videos').mkdir(exist_ok=True)
    (output_root / 'meta').mkdir(exist_ok=True)

    print('Creating merged dataset structure...')

    # Collect all info.json files and compute aggregated info
    total_episodes = 0
    total_frames = 0
    fps = 30
    features = {}

    for subdir in subdirs:
        info_file = subdir / 'meta' / 'info.json'
        if info_file.exists():
            import json
            with open(info_file) as f:
                info = json.load(f)

            total_episodes += info.get('total_episodes', 0)
            total_frames += info.get('total_frames', 0)
            fps = info.get('fps', 30)

            # Use features from first dataset
            if not features:
                features = info.get('features', {})

    # Create aggregated info.json
    merged_info = {
        'codebase_version': 'v3.0',
        'robot_type': None,
        'total_episodes': total_episodes,
        'total_frames': total_frames,
        'total_tasks': 1,
        'chunks_size': 1000,
        'data_files_size_in_mb': 100 * len(subdirs),
        'video_files_size_in_mb': 200 * len(subdirs),
        'fps': fps,
        'splits': {
            'train': f'0:{total_episodes}'
        },
        'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
        'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4',
        'features': features
    }

    import json
    with open(output_root / 'meta' / 'info.json', 'w') as f:
        json.dump(merged_info, f, indent=2)

    print(f'  Created info.json: {total_episodes} episodes, {total_frames} frames')

    # Merge data files by copying and renumbering chunks
    chunk_idx = 0
    file_idx = 0
    frames_in_current_chunk = 0
    max_frames_per_chunk = 1000

    current_chunk_dir = output_root / 'data' / f'chunk-{chunk_idx:03d}'
    current_chunk_dir.mkdir(exist_ok=True)

    print(f'\nMerging data files...')

    for subdir in subdirs:
        print(f'  Processing {subdir.name}...')

        data_dir = subdir / 'data'
        if not data_dir.exists():
            continue

        for chunk_dir in sorted(data_dir.glob('chunk-*')):
            for data_file in sorted(chunk_dir.glob('*.parquet')):
                # Read original data
                import pandas as pd
                df = pd.read_parquet(data_file)

                # Check if we need a new chunk
                if frames_in_current_chunk + len(df) > max_frames_per_chunk:
                    chunk_idx += 1
                    file_idx = 0
                    frames_in_current_chunk = 0
                    current_chunk_dir = output_root / 'data' / f'chunk-{chunk_idx:03d}'
                    current_chunk_dir.mkdir(exist_ok=True)

                # Write to new location with renumbered filename
                output_file = current_chunk_dir / f'file-{file_idx:03d}.parquet'
                shutil.copy(data_file, output_file)

                frames_in_current_chunk += len(df)
                file_idx += 1

        # Copy videos
        videos_dir = subdir / 'videos'
        if videos_dir.exists():
            output_videos = output_root / 'videos'
            for video_key in videos_dir.glob('*'):
                if video_key.is_dir():
                    dest_video_key = output_videos / video_key.name
                    dest_video_key.mkdir(exist_ok=True)

                    for video_chunk in sorted(video_key.glob('chunk-*')):
                        dest_video_chunk = dest_video_key / video_chunk.name
                        dest_video_chunk.mkdir(exist_ok=True)

                        for video_file in sorted(video_chunk.glob('*.mp4')):
                            dest_file = dest_video_chunk / video_file.name
                            shutil.copy(video_file, dest_file)

    print(f'\n  Data files merged into {chunk_idx + 1} chunks')

    # Copy tasks.parquet if exists
    tasks_src = subdirs[0] / 'meta' / 'tasks.parquet'
    if tasks_src.exists():
        shutil.copy(tasks_src, output_root / 'meta' / 'tasks.parquet')
        print(f'  Copied tasks.parquet')

    # Merge episodes.parquet files
    print(f'\nMerging episodes metadata...')
    episodes_dfs = []
    for subdir in subdirs:
        episodes_dir = subdir / 'meta' / 'episodes'
        if episodes_dir.exists():
            for episodes_chunk in sorted(episodes_dir.glob('chunk-*')):
                episodes_file = episodes_chunk / 'file-000.parquet'
                if episodes_file.exists():
                    import pandas as pd
                    df = pd.read_parquet(episodes_file)
                    # Renumber episodes to be sequential
                    if 'episode_index' in df.columns:
                        df['episode_index'] = range(len(episodes_dfs) * 5, (len(episodes_dfs) + 1) * 5)
                    episodes_dfs.append(df)

    if episodes_dfs:
        import pandas as pd
        merged_episodes = pd.concat(episodes_dfs, ignore_index=True)

        # Save merged episodes
        merged_episodes_dir = output_root / 'meta' / 'episodes' / 'chunk-000'
        merged_episodes_dir.mkdir(parents=True, exist_ok=True)
        merged_episodes.to_parquet(merged_episodes_dir / 'file-000.parquet', index=False)

        print(f'  Merged {len(merged_episodes)} episode records')

    # Copy stats.json from first subdir (or create new one)
    stats_src = subdirs[0] / 'meta' / 'stats.json'
    if stats_src.exists():
        shutil.copy(stats_src, output_root / 'meta' / 'stats.json')
        print(f'  Copied stats.json')

    print()
    print('='*80)
    print('MERGE COMPLETE')
    print('='*80)
    print(f'Merged dataset saved to: {output_root}')
    print(f'  Total episodes: {total_episodes}')
    print(f'  Total frames: {total_frames}')
    print()
    print('This merged dataset can now be used with:')
    print('  - Official data (with episode_success=1.0)')
    print('  - For value function training')


def main():
    parser = argparse.ArgumentParser(
        description='Merge subdirectories into single dataset'
    )
    parser.add_argument(
        '--source',
        type=str,
        default='Datasets/pant_long_evo_01/pant_long',
        help='Base directory containing subdirectories to merge'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='Datasets/pant_long_value_training/evo_merged',
        help='Path for merged output dataset'
    )
    parser.add_argument(
        '--repo_id',
        type=str,
        default='pant_long_evo_merged',
        help='Repository ID for merged dataset'
    )

    args = parser.parse_args()

    merge_subdirectories(
        source_base=Path(args.source),
        output_root=Path(args.output),
        repo_id=args.repo_id
    )


if __name__ == '__main__':
    main()
