#!/usr/bin/env python3
"""
Analyze wandb training data for lehome-challenge SmolVLA models.

Usage:
    python scripts/analyze_wandb_simple.py
"""

import wandb
import pandas as pd
from pathlib import Path


def main():
    print("=" * 60)
    print("LeHome-Challenge Wandb Analysis")
    print("=" * 60)

    api = wandb.Api()

    # Get all runs
    runs = api.runs("lehome-challenge")
    print(f"\nFound {len(runs)} total runs")

    # Filter SmolVLA runs
    smolvla_runs = [r for r in runs if 'smolvla' in r.name.lower()]
    print(f"\nSmolVLA runs: {len(smolvla_runs)}")

    # Analyze each run
    for run in smolvla_runs:
        print(f"\n{'='*60}")
        print(f"Run: {run.name}")
        print(f"State: {run.state}")
        print(f"Created: {run.created_at}")

        # Get config
        config = run.config
        print(f"\nConfig:")
        for key in ['batch_size', 'learning_rate', 'epochs', 'steps', 'device', 'seed']:
            if key in config:
                print(f"  {key}: {config.get(key)}")

        # Get summary metrics
        summary = run.summary_metrics
        if summary:
            print(f"\nSummary metrics:")
            for key in ['train/loss', 'eval/loss', 'epoch']:
                if key in summary:
                    print(f"  {key}: {summary.get(key)}")

        # Try to get history
        try:
            history = run.history()
            if history:
                df = pd.DataFrame(history)
                print(f"\nHistory samples: {len(df)}")
                print(f"Columns: {list(df.columns)[:10]}")

                # Check for loss column
                loss_cols = [c for c in df.columns if 'loss' in c.lower()]
                if loss_cols:
                    print(f"\nLoss columns found: {loss_cols}")

                    # Show first and last few loss values
                    for col in loss_cols[:3]:
                        valid_losses = df[col].dropna()
                        if not valid_losses.empty:
                            print(f"\n  {col}:")
                            print(valid_losses.head())

                        # Plot final loss
                        final_loss = valid_losses.iloc[-1]
                        print(f"  Final loss: {final_loss:.6f}")

        except Exception as e:
            print(f"Error getting history: {e}")
