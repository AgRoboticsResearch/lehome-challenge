#!/usr/bin/env python
"""
Merge official and EVO datasets for Pistar06 value function training.

This script combines two datasets with episode-level shuffling to ensure
proper mixing of success/failure examples for value function training.
"""

import argparse
import shutil
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict

def load_episode_mapping(data_root: Path) -> Dict[int, pd.DataFrame]:
    """
    Load episode metadata and create mapping from episode_index to frames.

    Returns:
        Dict mapping episode_index to DataFrame of all frames in that episode
    """
    data_dir = data_root / 'data'
    episodes_data = {}

    for chunk_dir in sorted(data_dir.glob('chunk-*')):
        for data_file in sorted(chunk_dir.glob('*.parquet')):
            df = pd.read_parquet(data_file)

            # Group by episode_index
            for episode_idx, episode_df in df.groupby('episode_index'):
                if episode_idx not in episodes_data:
                    episodes_data[episode_idx] = []
                episodes_data[episode_idx].append(episode_df)

    # Concatenate frames for each episode
    for episode_idx in episodes_data:
        episodes_data[episode_idx] = pd.concat(episodes_data[episode_idx], ignore_index=True)

    return episodes_data


def load_episode_metadata(data_root: Path) -> pd.DataFrame:
    """
    Load episode metadata from episodes.parquet file.

    Returns:
        DataFrame with episode metadata including video references
    """
    episodes_file = data_root / 'meta' / 'episodes' / 'chunk-000' / 'file-000.parquet'
    if episodes_file.exists():
        return pd.read_parquet(episodes_file)
    return pd.DataFrame()


def shuffle_and_merge_datasets(
    official_root: Path,
    evo_root: Path,
    output_root: Path,
    seed: int = 42
):
    """
    Merge official and EVO datasets with episode-level shuffling.

    Args:
        official_root: Path to official dataset (all success)
        evo_root: Path to EVO dataset (mixed success/failure)
        output_root: Path for merged output dataset
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)

    print('='*100)
    print('MERGING OFFICIAL + EVO DATASETS FOR VALUE FUNCTION TRAINING')
    print('='*100)
    print()
    print(f'Official dataset: {official_root}')
    print(f'EVO dataset: {evo_root}')
    print(f'Output: {output_root}')
    print(f'Shuffle seed: {seed}')
    print()

    # Read info.json files
    with open(official_root / 'meta' / 'info.json') as f:
        official_info = json.load(f)

    with open(evo_root / 'meta' / 'info.json') as f:
        evo_info = json.load(f)

    official_episodes = official_info['total_episodes']
    evo_episodes = evo_info['total_episodes']
    total_episodes = official_episodes + evo_episodes

    print(f'Official: {official_episodes} episodes, {official_info["total_frames"]} frames')
    print(f'EVO: {evo_episodes} episodes, {evo_info["total_frames"]} frames')
    print(f'Total: {total_episodes} episodes, {official_info["total_frames"] + evo_info["total_frames"]} frames')
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

    # Load episodes from both datasets
    print('Loading episode data...')
    official_episodes_data = load_episode_mapping(official_root)
    evo_episodes_data = load_episode_mapping(evo_root)
    print(f'  Loaded {len(official_episodes_data)} official episodes')
    print(f'  Loaded {len(evo_episodes_data)} EVO episodes')

    # Load episode metadata (for video references)
    print('Loading episode metadata...')
    official_episodes_meta = load_episode_metadata(official_root)
    evo_episodes_meta = load_episode_metadata(evo_root)
    print(f'  Loaded metadata for {len(official_episodes_meta)} official episodes')
    print(f'  Loaded metadata for {len(evo_episodes_meta)} EVO episodes')
    print()

    # Create shuffled episode order with video references
    # Format: (source_name, original_episode_index, video_refs_dict)
    episode_sources = []

    for orig_idx in range(official_episodes):
        video_refs = {}
        if not official_episodes_meta.empty and orig_idx < len(official_episodes_meta):
            meta_row = official_episodes_meta.iloc[orig_idx]
            for video_key in ['observation.images.top_rgb', 'observation.images.left_rgb', 'observation.images.right_rgb']:
                chunk_col = f'videos/{video_key}/chunk_index'
                file_col = f'videos/{video_key}/file_index'
                if chunk_col in meta_row and file_col in meta_row:
                    video_refs[video_key] = (meta_row[chunk_col], meta_row[file_col])
        episode_sources.append(('official', orig_idx, video_refs))

    for orig_idx in range(evo_episodes):
        video_refs = {}
        if not evo_episodes_meta.empty and orig_idx < len(evo_episodes_meta):
            meta_row = evo_episodes_meta.iloc[orig_idx]
            for video_key in ['observation.images.top_rgb', 'observation.images.left_rgb', 'observation.images.right_rgb']:
                chunk_col = f'videos/{video_key}/chunk_index'
                file_col = f'videos/{video_key}/file_index'
                if chunk_col in meta_row and file_col in meta_row:
                    video_refs[video_key] = (meta_row[chunk_col], meta_row[file_col])
        episode_sources.append(('evo', orig_idx, video_refs))

    # Shuffle the episode order
    print('Shuffling episode order...')
    shuffled_indices = np.random.permutation(len(episode_sources))
    episode_sources = [episode_sources[i] for i in shuffled_indices]
    print(f'  Shuffled {len(episode_sources)} episodes')
    print()

    # Create episode mapping: new_episode_index -> (source, old_episode_index, video_refs)
    episode_mapping = {
        new_idx: (source, old_idx, video_refs)
        for new_idx, (source, old_idx, video_refs) in enumerate(episode_sources)
    }

    # Merge data with renumbered episodes
    print('Merging data files with shuffled episode order...')

    chunk_idx = 0
    file_idx = 0
    frames_in_current_chunk = 0
    max_frames_per_chunk = 1000

    current_chunk_dir = output_root / 'data' / f'chunk-{chunk_idx:03d}'
    current_chunk_dir.mkdir(exist_ok=True)

    total_frames_processed = 0

    # Track video references for each new episode
    episode_video_refs = {}

    for new_episode_idx, (source, old_episode_idx, video_refs) in episode_mapping.items():
        # Get episode data from source
        if source == 'official':
            episode_df = official_episodes_data[old_episode_idx].copy()
            source_root = official_root
        else:
            episode_df = evo_episodes_data[old_episode_idx].copy()
            source_root = evo_root

        # Store video references for this episode
        episode_video_refs[new_episode_idx] = video_refs

        # Renumber episode_index
        episode_df['episode_index'] = new_episode_idx

        # Update index column
        if 'index' in episode_df.columns:
            episode_df['index'] = episode_df['episode_index'] * max_frames_per_chunk + episode_df['frame_index']

        # Check if we need a new chunk
        if frames_in_current_chunk + len(episode_df) > max_frames_per_chunk and frames_in_current_chunk > 0:
            chunk_idx += 1
            file_idx = 0
            frames_in_current_chunk = 0
            current_chunk_dir = output_root / 'data' / f'chunk-{chunk_idx:03d}'
            current_chunk_dir.mkdir(exist_ok=True)

        # Write to merged dataset
        output_file = current_chunk_dir / f'file-{file_idx:03d}.parquet'
        episode_df.to_parquet(output_file, index=False)

        frames_in_current_chunk += len(episode_df)
        file_idx += 1
        total_frames_processed += len(episode_df)

        if (new_episode_idx + 1) % 50 == 0:
            print(f'  Processed {new_episode_idx + 1}/{total_episodes} episodes...')

    print(f'  Data merged into {chunk_idx + 1} chunks, {total_frames_processed} frames')
    print()

    # Create aggregated info.json
    merged_info = {
        'codebase_version': 'v3.0',
        'robot_type': None,
        'total_episodes': total_episodes,
        'total_frames': official_info['total_frames'] + evo_info['total_frames'],
        'total_tasks': 1,
        'chunks_size': 1000,
        'data_files_size_in_mb': official_info['data_files_size_in_mb'] + evo_info['data_files_size_in_mb'],
        'video_files_size_in_mb': official_info['video_files_size_in_mb'] + evo_info['video_files_size_in_mb'],
        'fps': official_info.get('fps', 30),
        'splits': {
            'train': f'0:{total_episodes}'
        },
        'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
        'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4',
        'features': official_info.get('features', {})
    }

    # Ensure episode_success is in features
    if 'episode_success' not in merged_info['features']:
        merged_info['features']['episode_success'] = {
            'dtype': 'float32',
            'shape': [1],
            'names': None
        }

    with open(output_root / 'meta' / 'info.json', 'w') as f:
        json.dump(merged_info, f, indent=2)

    print('Created aggregated info.json')
    print()

    # Create episode metadata from actual merged data
    print('Creating episode metadata from merged data...')
    merged_episodes_list = []

    # Track video file mapping: (source, old_chunk, old_file) -> (new_chunk, new_file)
    video_file_mapping = {}

    # Iterate through all data files to build accurate episode metadata
    data_dir = output_root / 'data'
    for chunk_dir in sorted(data_dir.glob('chunk-*')):
        chunk_idx = int(chunk_dir.name.split('-')[-1])
        for file_idx, data_file in enumerate(sorted(chunk_dir.glob('*.parquet'))):
            # Skip macOS metadata files
            if data_file.name.startswith('._'):
                continue
            try:
                df = pd.read_parquet(data_file)
            except Exception as e:
                print(f'  Warning: Could not read {data_file}: {e}')
                continue

            # Group by episode_index
            for episode_idx, episode_df in df.groupby('episode_index'):
                # Create episode record
                episode_record = {
                    'episode_index': episode_idx,
                    'length': len(episode_df),
                    'data/chunk_index': chunk_idx,
                    'data/file_index': file_idx,
                    'tasks': [0]  # Assuming single task
                }

                # Get video references from original episode
                video_refs = episode_video_refs.get(episode_idx, {})

                # Add video references (will be updated after video merge)
                for video_key in ['observation.images.top_rgb', 'observation.images.left_rgb', 'observation.images.right_rgb']:
                    if video_key in video_refs:
                        old_chunk, old_file = video_refs[video_key]
                        # Store mapping for later update
                        video_file_mapping[(episode_idx, video_key)] = (old_chunk, old_file)
                        # Temporary placeholder
                        episode_record[f'videos/{video_key}/chunk_index'] = old_chunk
                        episode_record[f'videos/{video_key}/file_index'] = old_file
                    else:
                        # No video reference for this episode
                        episode_record[f'videos/{video_key}/chunk_index'] = 0
                        episode_record[f'videos/{video_key}/file_index'] = 0
                    episode_record[f'videos/{video_key}/from_timestamp'] = 0
                    episode_record[f'videos/{video_key}/to_timestamp'] = len(episode_df) / 30  # Approximate duration

                # Add stats placeholders (skip array columns)
                for col in ['timestamp', 'frame_index', 'episode_index', 'index', 'task_index', 'episode_success']:
                    if col in episode_df.columns:
                        try:
                            episode_record[f'stats/{col}/min'] = float(episode_df[col].min())
                            episode_record[f'stats/{col}/max'] = float(episode_df[col].max())
                            episode_record[f'stats/{col}/mean'] = float(episode_df[col].mean())
                            episode_record[f'stats/{col}/std'] = float(episode_df[col].std())
                            episode_record[f'stats/{col}/count'] = len(episode_df)
                        except (ValueError, TypeError):
                            # Skip array columns
                            episode_record[f'stats/{col}/min'] = 0.0
                            episode_record[f'stats/{col}/max'] = 0.0
                            episode_record[f'stats/{col}/mean'] = 0.0
                            episode_record[f'stats/{col}/std'] = 0.0
                            episode_record[f'stats/{col}/count'] = len(episode_df)

                merged_episodes_list.append(episode_record)

    # Convert to DataFrame
    merged_episodes = pd.DataFrame(merged_episodes_list)

    # Save temporary episodes (will update after video merge)
    merged_episodes_dir = output_root / 'meta' / 'episodes' / 'chunk-000'
    merged_episodes_dir.mkdir(parents=True, exist_ok=True)
    merged_episodes.to_parquet(merged_episodes_dir / 'file-000.parquet', index=False)

    print(f'  Created {len(merged_episodes)} episode records from actual data')
    print()

    # Merge videos by copying entire video directories from both sources
    # Videos are stored separately and referenced through the video_path template
    # We copy all videos to ensure complete dataset, even though shuffling
    # affects episode ordering in data files
    print('Merging video files...')
    video_keys = ['observation.images.top_rgb', 'observation.images.left_rgb', 'observation.images.right_rgb']

    # Track video file mapping: (source_name, old_chunk, old_file) -> (new_chunk, new_file)
    video_mapping = {}

    for video_key in video_keys:
        video_output_dir = output_root / 'videos' / video_key
        video_output_dir.mkdir(parents=True, exist_ok=True)

        # Track which video files we've already copied to avoid duplicates
        copied_videos = set()
        video_chunk_idx = 0
        video_file_idx = 0
        frames_in_current_video_chunk = 0
        max_frames_per_video_chunk = 1000

        current_video_chunk_dir = video_output_dir / f'chunk-{video_chunk_idx:03d}'
        current_video_chunk_dir.mkdir(exist_ok=True)

        # Copy videos from both sources, avoiding duplicates
        for source_root in [official_root, evo_root]:
            source_video_dir = source_root / 'videos' / video_key

            if not source_video_dir.exists():
                continue

            # Determine source name for mapping
            if source_root == official_root:
                source_name = 'official'
            else:
                source_name = 'evo'

            for video_chunk in sorted(source_video_dir.glob('chunk-*')):
                old_chunk_idx = int(video_chunk.name.split('-')[-1])
                for video_file in sorted(video_chunk.glob('*.mp4')):
                    # Skip macOS metadata files
                    if video_file.name.startswith('._'):
                        continue

                    old_file_idx = int(video_file.stem.split('-')[-1])

                    # Create unique identifier for this video file
                    video_id = f"{source_root.name}/{video_chunk.name}/{video_file.name}"

                    if video_id in copied_videos:
                        continue

                    copied_videos.add(video_id)

                    # Check if we need a new chunk
                    if frames_in_current_video_chunk > max_frames_per_video_chunk:
                        video_chunk_idx += 1
                        video_file_idx = 0
                        frames_in_current_video_chunk = 0
                        current_video_chunk_dir = video_output_dir / f'chunk-{video_chunk_idx:03d}'
                        current_video_chunk_dir.mkdir(exist_ok=True)

                    # Copy video file with new number
                    dest_file = current_video_chunk_dir / f'file-{video_file_idx:03d}.mp4'
                    try:
                        shutil.copy2(video_file, dest_file)
                        # Track mapping for updating episode metadata
                        video_mapping[(source_name, video_key, old_chunk_idx, old_file_idx)] = (video_chunk_idx, video_file_idx)
                        video_file_idx += 1
                        frames_in_current_video_chunk += 1000  # Approximate
                    except Exception as e:
                        print(f'  Warning: Failed to copy {video_file}: {e}')

    print(f'  Videos merged into {video_chunk_idx + 1} chunks')
    print()

    # Update episode metadata with correct video references
    print('Updating episode metadata with video references...')
    for episode_idx in range(len(merged_episodes)):
        video_refs = episode_video_refs.get(episode_idx, {})
        for video_key in ['observation.images.top_rgb', 'observation.images.left_rgb', 'observation.images.right_rgb']:
            if video_key in video_refs:
                old_chunk, old_file = video_refs[video_key]
                # Determine source for this episode
                source = episode_mapping[episode_idx][0]  # ('official' or 'evo')
                # Look up new video file location
                mapping_key = (source, video_key, old_chunk, old_file)
                if mapping_key in video_mapping:
                    new_chunk, new_file = video_mapping[mapping_key]
                    merged_episodes.at[episode_idx, f'videos/{video_key}/chunk_index'] = new_chunk
                    merged_episodes.at[episode_idx, f'videos/{video_key}/file_index'] = new_file

    # Save updated episodes metadata
    merged_episodes.to_parquet(merged_episodes_dir / 'file-000.parquet', index=False)
    print(f'  Updated video references for {len(merged_episodes)} episodes')
    print()

    # Copy stats.json (use official as base)
    stats_src = official_root / 'meta' / 'stats.json'
    if stats_src.exists():
        shutil.copy(stats_src, output_root / 'meta' / 'stats.json')
        print('Copied stats.json from official dataset')

    # Copy tasks.parquet if exists
    tasks_src = official_root / 'meta' / 'tasks.parquet'
    if tasks_src.exists():
        shutil.copy(tasks_src, output_root / 'meta' / 'tasks.parquet')
        print('Copied tasks.parquet from official dataset')

    print()
    print('='*100)
    print('MERGE COMPLETE')
    print('='*100)
    print(f'Merged dataset saved to: {output_root}')
    print(f'  Total episodes: {total_episodes}')
    print(f'  Total frames: {total_frames_processed}')
    print(f'  Episodes shuffled: Yes (seed={seed})')
    print()

    # Verify success distribution
    print('Verifying episode_success distribution...')
    success_count = 0
    failure_count = 0

    for chunk_dir in sorted((output_root / 'data').glob('chunk-*')):
        for data_file in chunk_dir.glob('*.parquet'):
            # Skip macOS metadata files
            if data_file.name.startswith('._'):
                continue
            df = pd.read_parquet(data_file)
            success_count += (df['episode_success'] == 1.0).sum()
            failure_count += (df['episode_success'] == 0.0).sum()

    total_frames = success_count + failure_count
    print(f'  Success frames: {success_count} ({100*success_count/total_frames:.1f}%)')
    print(f'  Failure frames: {failure_count} ({100*failure_count/total_frames:.1f}%)')
    print()
    print('Ready for Pistar06 value function training!')


def main():
    parser = argparse.ArgumentParser(
        description='Merge official and EVO datasets with shuffling'
    )
    parser.add_argument(
        '--official',
        type=str,
        default='Datasets/pant_long_value_training/official_with_success',
        help='Path to official dataset'
    )
    parser.add_argument(
        '--evo',
        type=str,
        default='Datasets/pant_long_value_training/evo_all_merged',
        help='Path to EVO dataset'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='Datasets/pant_long_value_training/official_evo_shuffled',
        help='Path for merged output dataset'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for shuffling (default: 42)'
    )

    args = parser.parse_args()

    shuffle_and_merge_datasets(
        official_root=Path(args.official),
        evo_root=Path(args.evo),
        output_root=Path(args.output),
        seed=args.seed
    )


if __name__ == '__main__':
    main()
