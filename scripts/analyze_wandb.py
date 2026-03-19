#!/usr/bin/env python3
"""
Analyze wandb training data for lehome-challenge SmolVLA models.

Usage:
    python scripts/analyze_wandb.py
"""

import os
import wandb
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def get_wandb_runs(project="lehome-challenge"):
    """Get all runs from wandb project."""
    api = wandb.Api()
    runs = api.runs(f"{project}")
    return runs


def analyze_run(run):
    """Analyze a single run."""
    print(f"\n{'='*60}")
    print(f"Run: {run.name} (ID: {run.id})")
    print(f"State: {run.state}")
    print(f"Created: {run.created_at}")

    # Get config
    config = run.config
    if config:
        print(f"\nConfig:")
        for key in ['batch_size', 'learning_rate', 'epochs', 'steps']:
            if key in config:
                print(f"  {key}: {config[key]}")

    # Get history
    try:
        history = run.history()
        if history:
            df = pd.DataFrame(history)
            print(f"\nHistory samples: {len(df)}")

            # Check for loss column
            loss_cols = [c for c in df.columns if 'loss' in c.lower()]
            if loss_cols:
                print(f"\nLoss columns: {loss_cols}")
                for col in loss_cols[:3]:
                    print(f"\n{col} statistics:")
                    print(df[col].describe())

            return df, loss_cols
    except Exception as e:
        print(f"Error getting history: {e}")
        return None, None


def main():
    print("=" * 60)
    print("LeHome-Challenge Wandb Analysis")
    print("=" * 60)

    # Get all runs
    runs = get_wandb_runs()

    print(f"\nFound {len(runs)} runs")

    # Filter SmolVLA runs
    smolvla_runs = [r for r in runs if 'smolvla' in r.name.lower()]
    print(f"\nSmolVLA runs: {len(smolvla_runs)}")

    # Analyze each run
    all_data = []
    for run in smolvla_runs:
        df, loss_cols = analyze_run(run)
        if df is not None:
            df['run_name'] = run.name
            all_data.append(df)

    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\n{'='*60}")
        print("Combined Data Summary")
        print("=" * 60)
        print(combined_df.describe())

        # Save to CSV
        output_path = Path("wandb_analysis.csv")
        combined_df.to_csv(output_path, index=False)
        print(f"\nData saved to {output_path}")


if __name__ == "__main__":
    main()
