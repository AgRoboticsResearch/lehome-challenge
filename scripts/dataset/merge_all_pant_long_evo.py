#!/usr/bin/env python
"""
Merge all pant_long_evo_01 datasets into a single training dataset.

Merges 5 timestamped directories (60 episodes each) into one dataset of 300 episodes.
"""

import argparse
import shutil
from pathlib import Path
import json
import pandas as pd

def merge_all_pant_long_evo_datasets(
    source_base: Path,
    output_root: Path,
    repo_id: str = 'pant_long_evo_full'
):
    """
    Merge all pant_long_evo_01 timestamped directories.

    Args:
        source_base: Base directory containing timestamped dirs (e.g., Datasets/pant_long_evo_01)
        output_root: Path for output merged dataset
        repo_id: Repository ID for the merged dataset
    """
    source_base = Path(source_base)
    output_root = Path(output_root)

    print('='*100)
    print(f'MERGING ALL PANT_LONG EVO DATASETS FROM: {source_base}')
    print('='*100)
    print()

    # Get all timestamped pant_long directories (exclude merged)
    timestamped_dirs = sorted([
        d for d in source_base.glob('pant_long*')
        if d.is_dir() and d.name != 'pant_long_merged'
    ])

    if not timestamped_dirs:
        print('No pant_long directories found to merge')
        return

    print(f'Found {len(timestamped_dirs)} timestamped directories:')

    total_episodes = 0
    total_frames = 0
    fps = 30
    features = {}

    # First pass: collect statistics and verify structure
    for base in timestamped_dirs:
        subdirs = sorted([d for d in base.glob('*') if d.is_dir()])
        base_episodes = 0
        base_frames = 0

        for subdir in subdirs:
            info_file = subdir / 'meta' / 'info.json'
            if info_file.exists():
                with open(info_file) as f:
                    info = json.load(f)
                base_episodes += info.get('total_episodes', 0)
                base_frames += info.get('total_frames', 0)

                # Use features from first dataset
                if not features and info.get('features'):
                    features = info.get('features', {})
                    fps = info.get('fps', 30)

        total_episodes += base_episodes
        total_frames += base_frames

        print(f'  {base.name}: {len(subdirs)} subdirs, {base_episodes} episodes, {base_frames} frames')

    print()
    print(f'Total to merge: {total_episodes} episodes, {total_frames} frames')
    print()

    # Create output directory
    if output_root.exists():
        print(f'Output directory exists: {output_root}')
        response = input('Delete and recreate? (y/n): ')
        if response.lower() == 'y':
            shutil.rmtree(output_root)
        else:
            print('Aborted')
            return

    output_root.mkdir(parents=True, exist_ok=True)

    # Create output directories
    (output_root / 'data').mkdir(exist_ok=True)
    (output_root / 'videos').mkdir(exist_ok=True)
    (output_root / 'meta').mkdir(exist_ok=True)

    # Create aggregated info.json
    merged_info = {
        'codebase_version': 'v3.0',
        'robot_type': None,
        'total_episodes': total_episodes,
        'total_frames': total_frames,
        'total_tasks': 1,
        'chunks_size': 1000,
        'data_files_size_in_mb': 100 * len(timestamped_dirs),
        'video_files_size_in_mb': 200 * len(timestamped_dirs),
        'fps': fps,
        'splits': {
            'train': f'0:{total_episodes}'
        },
        'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
        'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4',
        'features': features
    }

    with open(output_root / 'meta' / 'info.json', 'w') as f:
        json.dump(merged_info, f, indent=2)

    print('Created aggregated info.json')
    print()

    # Merge data files with proper episode renumbering
    print('Merging data files...')

    chunk_idx = 0
    file_idx = 0
    frames_in_current_chunk = 0
    max_frames_per_chunk = 1000
    episode_offset = 0

    current_chunk_dir = output_root / 'data' / f'chunk-{chunk_idx:03d}'
    current_chunk_dir.mkdir(exist_ok=True)

    for base_idx, base in enumerate(timestamped_dirs):
        print(f'  Processing {base.name}...')

        # Process each subdirectory in this base directory
        subdirs = sorted([d for d in base.glob('*') if d.is_dir()])

        for subdir in subdirs:
            data_dir = subdir / 'data'
            if not data_dir.exists():
                continue

            for chunk_dir in sorted(data_dir.glob('chunk-*')):
                for data_file in sorted(chunk_dir.glob('*.parquet')):
                    # Read and modify data
                    df = pd.read_parquet(data_file)

                    # Renumber episodes to be sequential across merged dataset
                    if 'episode_index' in df.columns:
                        unique_episodes = sorted(df['episode_index'].unique())
                        episode_mapping = {old_idx: old_idx + episode_offset for old_idx in unique_episodes}
                        df['episode_index'] = df['episode_index'].map(episode_mapping)

                        # Also update index if present
                        if 'index' in df.columns:
                            df['index'] = df['episode_index'] * max_frames_per_chunk + df['frame_index']

                    # Check chunk size
                    if frames_in_current_chunk + len(df) > max_frames_per_chunk:
                        chunk_idx += 1
                        file_idx = 0
                        frames_in_current_chunk = 0
                        current_chunk_dir = output_root / 'data' / f'chunk-{chunk_idx:03d}'
                        current_chunk_dir.mkdir(exist_ok=True)

                    # Write to merged dataset
                    output_file = current_chunk_dir / f'file-{file_idx:03d}.parquet'
                    df.to_parquet(output_file, index=False)

                    frames_in_current_chunk += len(df)
                    file_idx += 1

            # Update episode offset after each subdirectory (each has 5 episodes)
            episode_offset += 5

    print(f'  Data merged into {chunk_idx + 1} chunks')
    print()

    # Merge videos
    print('Merging video files...')
    video_keys = ['observation.images.top_rgb', 'observation.images.left_rgb', 'observation.images.right_rgb']

    # Track video file indices per video_key to avoid overwrites
    video_file_indices = {key: 0 for key in video_keys}

    for video_key in video_keys:
        video_output_dir = output_root / 'videos' / video_key
        video_output_dir.mkdir(parents=True, exist_ok=True)

        video_chunk_idx = 0
        current_video_chunk_dir = video_output_dir / f'chunk-{video_chunk_idx:03d}'
        current_video_chunk_dir.mkdir(parents=True, exist_ok=True)
        videos_in_current_chunk = 0

        for base_idx, base in enumerate(timestamped_dirs):
            # Videos are in base/{subdir}/videos/{video_key}, not base/videos/{video_key}
            for subdir in sorted(base.glob('*')):
                if not subdir.is_dir():
                    continue
                subdir_videos = subdir / 'videos' / video_key
                if subdir_videos.exists():
                    for video_chunk in sorted(subdir_videos.glob('chunk-*')):
                        for video_file in sorted(video_chunk.glob('*.mp4')):
                            # Check if we need a new chunk (max ~1000 frames per video file)
                            if videos_in_current_chunk >= 100:
                                video_chunk_idx += 1
                                current_video_chunk_dir = video_output_dir / f'chunk-{video_chunk_idx:03d}'
                                current_video_chunk_dir.mkdir(parents=True, exist_ok=True)
                                videos_in_current_chunk = 0

                            # Use running counter for unique file numbering
                            new_file_num = video_file_indices[video_key]
                            dest_file = current_video_chunk_dir / f'file-{new_file_num:03d}.mp4'
                            shutil.copy2(video_file, dest_file)

                            video_file_indices[video_key] += 1
                            videos_in_current_chunk += 1

    print(f'  Videos merged: {", ".join(f"{k}={v} files" for k, v in video_file_indices.items())}')
    print()

    # Copy tasks.parquet (from first directory)
    tasks_src = timestamped_dirs[0] / 'pant_long' / '002' / 'meta' / 'tasks.parquet'
    if tasks_src.exists():
        shutil.copy(tasks_src, output_root / 'meta' / 'tasks.parquet')
        print('Copied tasks.parquet')

    # Merge episodes metadata
    print('Merging episodes metadata...')
    all_episodes_dfs = []
    episode_offset = 0  # Track offset across all subdirectories (not per base)

    # Track video file indices per video_key (must match video merging logic)
    video_file_indices = {key: 0 for key in ['observation.images.top_rgb', 'observation.images.left_rgb', 'observation.images.right_rgb']}

    for base_idx, base in enumerate(timestamped_dirs):
        subdirs = sorted([d for d in base.glob('*') if d.is_dir()])

        for subdir in subdirs:
            episodes_dir = subdir / 'meta' / 'episodes'
            if episodes_dir.exists():
                for episodes_chunk in sorted(episodes_dir.glob('chunk-*')):
                    episodes_file = episodes_chunk / 'file-000.parquet'
                    if episodes_file.exists():
                        df = pd.read_parquet(episodes_file)
                        # Renumber episodes
                        if 'episode_index' in df.columns:
                            df['episode_index'] = df['episode_index'] + episode_offset

                        # Renumber video file_index using running counter (must match video merging)
                        for video_key in video_file_indices.keys():
                            col_name = f'videos/{video_key}/file_index'
                            if col_name in df.columns:
                                # Each subdir has 1 video file per camera, increment after processing
                                df[col_name] = df[col_name] + video_file_indices[video_key]
                                video_file_indices[video_key] += 1

                        all_episodes_dfs.append(df)

            # Update episode offset after each subdirectory (each has 5 episodes)
            episode_offset += 5

    if all_episodes_dfs:
        merged_episodes = pd.concat(all_episodes_dfs, ignore_index=True)

        # Save merged episodes
        merged_episodes_dir = output_root / 'meta' / 'episodes' / 'chunk-000'
        merged_episodes_dir.mkdir(parents=True, exist_ok=True)
        merged_episodes.to_parquet(merged_episodes_dir / 'file-000.parquet', index=False)

        print(f'  Merged {len(merged_episodes)} episode records')

    # Copy stats.json
    stats_src = timestamped_dirs[0] / 'pant_long' / '002' / 'meta' / 'stats.json'
    if stats_src.exists():
        shutil.copy(stats_src, output_root / 'meta' / 'stats.json')
        print('Copied stats.json')

    print()
    print('='*100)
    print('MERGE COMPLETE')
    print('='*100)
    print(f'Merged dataset saved to: {output_root}')
    print(f'  Total episodes: {total_episodes}')
    print(f'  Total frames: {total_frames}')
    print()
    print('Ready for value function training with official data!')


def main():
    parser = argparse.ArgumentParser(
        description='Merge all pant_long_evo_01 datasets'
    )
    parser.add_argument(
        '--source',
        type=str,
        default='Datasets/pant_long_evo_01',
        help='Base directory containing timestamped datasets'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='Datasets/pant_long_value_training/evo_all_merged',
        help='Path for merged output dataset'
    )
    parser.add_argument(
        '--repo_id',
        type=str,
        default='pant_long_evo_all',
        help='Repository ID for merged dataset'
    )

    args = parser.parse_args()

    merge_all_pant_long_evo_datasets(
        source_base=Path(args.source),
        output_root=Path(args.output),
        repo_id=args.repo_id
    )


if __name__ == '__main__':
    main()
