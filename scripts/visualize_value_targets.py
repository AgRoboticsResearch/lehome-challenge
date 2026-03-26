#!/usr/bin/env python3
"""
Visualize value targets for success vs failure episodes.

This script computes value targets using the same formula as Pistar06:
- For success: g = -remaining_steps
- For failure: g = -remaining_steps - c_fail

And shows how they differ between success and failure episodes.
"""

import pyarrow.parquet as pq
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Dict


@dataclass
class EpisodeTargetInfo:
    episode_index: int
    task_index: int
    length: int
    success: bool


def compute_normalized_value_targets(
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    episode_info: Dict[int, EpisodeTargetInfo],
    task_max_lengths: Dict[int, int],
    c_fail_coef: float,
    clip_min: float = -1.0,
    clip_max: float = 0.0,
) -> np.ndarray:
    """Compute value targets (same as Pistar06)."""
    targets = np.zeros(episode_indices.shape[0], dtype=np.float32)

    for i in range(episode_indices.shape[0]):
        ep_idx = int(episode_indices[i])
        ep = episode_info[ep_idx]
        task_max = task_max_lengths.get(ep.task_index)

        remaining_steps = ep.length - int(frame_indices[i]) - 1
        c_fail = float(task_max) * c_fail_coef
        g = -float(remaining_steps)

        if not ep.success:
            g -= c_fail  # Penalty for failure

        denom = float(task_max) + c_fail
        g_norm = g / denom
        targets[i] = np.clip(g_norm, clip_min, clip_max)

    return targets


def load_dataset(dataset_path: str):
    """Load dataset metadata."""
    dataset_path = Path(dataset_path)

    # Load episode metadata
    episodes_files = list(dataset_path.glob("meta/episodes/chunk-*/file-*.parquet"))
    if not episodes_files:
        raise FileNotFoundError(f"No episode files found in {dataset_path}")

    episodes_table = pq.read_table(episodes_files[0])

    # Load frame data
    data_files = list(dataset_path.glob("data/chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No data files found in {dataset_path}")

    data_table = pq.read_table(data_files[0])

    return episodes_table, data_table


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize value targets")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/Users/moky/codes/lehome-challenge/Datasets/pant_long_evo_rl_smol_mix_merged",
        help="Path to dataset",
    )
    parser.add_argument(
        "--c_fail_coef",
        type=float,
        default=1.0,
        help="Failure penalty coefficient",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots",
    )
    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset_path}...")
    episodes_table, data_table = load_dataset(args.dataset_path)

    # Extract episode info
    episode_indices_data = data_table["episode_index"].to_pylist()
    frame_indices_data = data_table["frame_index"].to_pylist()

    # Build episode info dict
    episode_info = {}
    task_max_length = {}

    ep_indices = episodes_table["episode_index"].to_pylist()
    ep_lengths = episodes_table["length"].to_pylist()
    ep_tasks = episodes_table["tasks"].to_pylist()
    ep_success = episodes_table["episode_success"].to_pylist()

    for i in range(len(ep_indices)):
        ep_idx = int(ep_indices[i])
        ep_length = int(ep_lengths[i])
        task_name = ep_tasks[i][0] if isinstance(ep_tasks[i], list) else ep_tasks[i]
        task_index = hash(task_name) % 1000  # Simple task index

        success = ep_success[i] == "success"

        episode_info[ep_idx] = EpisodeTargetInfo(
            episode_index=ep_idx,
            task_index=task_index,
            length=ep_length,
            success=success,
        )
        task_max_length[task_index] = max(task_max_length.get(task_index, 0), ep_length)

    print(f"Loaded {len(episode_info)} episodes")
    print(f"  Success: {sum(1 for ep in episode_info.values() if ep.success)}")
    print(f"  Failure: {sum(1 for ep in episode_info.values() if not ep.success)}")

    # Compute value targets
    episode_indices = np.array(episode_indices_data, dtype=np.int64)
    frame_indices = np.array(frame_indices_data, dtype=np.int64)

    value_targets = compute_normalized_value_targets(
        episode_indices=episode_indices,
        frame_indices=frame_indices,
        episode_info=episode_info,
        task_max_lengths=task_max_length,
        c_fail_coef=args.c_fail_coef,
    )

    print(f"Computed {len(value_targets)} value targets")
    print(f"  Min: {value_targets.min():.4f}")
    print(f"  Max: {value_targets.max():.4f}")
    print(f"  Mean: {value_targets.mean():.4f}")

    # Separate by success/failure
    success_mask = np.array([episode_info[int(ep_idx)].success for ep_idx in episode_indices])
    success_values = value_targets[success_mask]
    failure_values = value_targets[~success_mask]

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Histogram of value targets by success/failure
    ax1 = axes[0, 0]
    ax1.hist(success_values, bins=50, alpha=0.7, label=f'Success (n={len(success_values)})', color='green')
    ax1.hist(failure_values, bins=50, alpha=0.7, label=f'Failure (n={len(failure_values)})', color='red')
    ax1.set_xlabel('Value Target')
    ax1.set_ylabel('Count')
    ax1.set_title('Value Target Distribution: Success vs Failure')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Value targets over episode progress (success)
    ax2 = axes[0, 1]
    success_episodes = [ep_idx for ep_idx in sorted(set(episode_indices)) if episode_info[int(ep_idx)].success]
    for ep_idx in success_episodes[:10]:  # Plot first 10 success episodes
        mask = episode_indices == ep_idx
        frames = frame_indices[mask]
        values = value_targets[mask]
        ep_len = episode_info[int(ep_idx)].length
        progress = frames / ep_len
        ax2.plot(progress, values, alpha=0.5, linewidth=1)
    ax2.set_xlabel('Episode Progress (frame / length)')
    ax2.set_ylabel('Value Target')
    ax2.set_title('Value Trajectory: Success Episodes (first 10)')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Value targets over episode progress (failure)
    ax3 = axes[1, 0]
    failure_episodes = [ep_idx for ep_idx in sorted(set(episode_indices)) if not episode_info[int(ep_idx)].success]
    for ep_idx in failure_episodes[:10]:  # Plot first 10 failure episodes
        mask = episode_indices == ep_idx
        frames = frame_indices[mask]
        values = value_targets[mask]
        ep_len = episode_info[int(ep_idx)].length
        progress = frames / ep_len
        ax3.plot(progress, values, alpha=0.5, linewidth=1, color='red')
    ax3.set_xlabel('Episode Progress (frame / length)')
    ax3.set_ylabel('Value Target')
    ax3.set_title('Value Trajectory: Failure Episodes (first 10)')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Box plot comparison
    ax4 = axes[1, 1]
    data_to_plot = [success_values, failure_values]
    bp = ax4.boxplot(data_to_plot, tick_labels=['Success', 'Failure'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax4.set_ylabel('Value Target')
    ax4.set_title('Value Target Comparison')
    ax4.grid(True, alpha=0.3)

    # Add statistics
    stats_text = f"Success: mean={success_values.mean():.3f}, std={success_values.std():.3f}\n"
    stats_text += f"Failure: mean={failure_values.mean():.3f}, std={failure_values.std():.3f}\n"
    stats_text += f"c_fail_coef={args.c_fail_coef}"
    fig.text(0.02, 0.02, stats_text, fontsize=10, family='monospace')

    plt.tight_layout()

    # Save or show
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.dataset_path)
    output_path = output_dir / "value_targets_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to {output_path}")

    plt.show()

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"\nSuccess Episodes ({len(success_values)} frames):")
    print(f"  Mean value: {success_values.mean():.4f}")
    print(f"  Std value:  {success_values.std():.4f}")
    print(f"  Min value:  {success_values.min():.4f}")
    print(f"  Max value:  {success_values.max():.4f}")

    print(f"\nFailure Episodes ({len(failure_values)} frames):")
    print(f"  Mean value: {failure_values.mean():.4f}")
    print(f"  Std value:  {failure_values.std():.4f}")
    print(f"  Min value:  {failure_values.min():.4f}")
    print(f"  Max value:  {failure_values.max():.4f}")

    print(f"\nDifference (Success - Failure):")
    print(f"  Mean diff: {success_values.mean() - failure_values.mean():.4f}")


if __name__ == "__main__":
    main()
