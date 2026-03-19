#!/usr/bin/env python3
"""
数据一致性分析脚本
分析四种服装类型的数据质量和一致性

用法:
    python scripts/analyze_data_consistency.py --dataset pant_long_merged
    python scripts/analyze_data_consistency.py --all
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from typing import Dict, List, Any, Optional


def load_dataset(dataset_name: str, base_path: Path = Path("Datasets/example")) -> tuple:
    """加载数据集"""
    ds_path = base_path / dataset_name / "data" / "chunk-000"
    parquet_files = sorted(ds_path.glob("*.parquet"))

    all_actions = []
    all_episodes = []

    for pf in parquet_files:
        table = pq.read_table(pf)
        df = table.to_pandas()
        all_actions.append(np.stack(df["action"].values))
        all_episodes.append(df["episode_index"].values)

    actions = np.concatenate(all_actions)
    episodes = np.concatenate(all_episodes)

    return actions, episodes


def compute_episode_metrics(ep_actions: np.ndarray) -> Dict[str, float]:
    """计算单个 episode 的指标"""
    if len(ep_actions) < 10:
        return {}

    metrics = {}

    # 1. 协调性 (左右手动作相关性)
    left = ep_actions[:, :6]
    right = ep_actions[:, 6:]
    left_diff = np.diff(left, axis=0).flatten()
    right_diff = np.diff(right, axis=0).flatten()

    if np.std(left_diff) > 1e-6 and np.std(right_diff) > 1e-6:
        metrics["correlation"] = np.corrcoef(left_diff, right_diff)[0, 1]
    else:
        metrics["correlation"] = 0.0

    # 2. 动作幅度
    diffs = np.diff(ep_actions, axis=0)
    mags = np.sqrt(np.sum(diffs**2, axis=1))
    metrics["avg_magnitude"] = np.mean(mags)
    metrics["std_magnitude"] = np.std(mags)

    # 3. Gripper 变化次数
    left_grip = ep_actions[:, 5]
    right_grip = ep_actions[:, 11]
    left_grip_changes = np.sum(np.abs(np.diff(left_grip)) > 0.05)
    right_grip_changes = np.sum(np.abs(np.diff(right_grip)) > 0.05)
    metrics["grip_changes"] = left_grip_changes + right_grip_changes

    # 4. Episode 长度
    metrics["length"] = len(ep_actions)

    return metrics


def analyze_garment(actions: np.ndarray, episodes: np.ndarray,
                   garment_idx: int, eps_per_garment: int = 25) -> Dict[str, Any]:
    """分析单个 garment 的数据质量"""
    start_ep = garment_idx * eps_per_garment
    end_ep = (garment_idx + 1) * eps_per_garment

    episode_metrics = []

    for ep in range(start_ep, end_ep):
        mask = episodes == ep
        ep_actions = actions[mask]

        metrics = compute_episode_metrics(ep_actions)
        if metrics:
            episode_metrics.append(metrics)

    if not episode_metrics:
        return {}

    # 计算聚合指标
    correlations = [m["correlation"] for m in episode_metrics]
    magnitudes = [m["avg_magnitude"] for m in episode_metrics]
    grip_changes = [m["grip_changes"] for m in episode_metrics]
    lengths = [m["length"] for m in episode_metrics]

    return {
        "garment_idx": garment_idx,
        "num_episodes": len(episode_metrics),
        "correlation_mean": np.mean(correlations),
        "correlation_std": np.std(correlations),
        "magnitude_mean": np.mean(magnitudes),
        "magnitude_std": np.std(magnitudes),
        "grip_changes_mean": np.mean(grip_changes),
        "grip_changes_std": np.std(grip_changes),
        "length_mean": np.mean(lengths),
        "length_std": np.std(lengths),
    }


def analyze_dataset(dataset_name: str, base_path: Path = Path("Datasets/example")) -> Dict[str, Any]:
    """分析整个数据集的一致性"""
    print(f"\\n{'='*80}")
    print(f"分析数据集: {dataset_name}")
    print(f"{'='*80}")

    # 加载数据
    actions, episodes = load_dataset(dataset_name, base_path)
    print(f"总帧数: {len(actions)}, 总 episodes: {len(np.unique(episodes))}")

    # 分析每个 garment
    garment_results = []
    for garment_idx in range(10):
        result = analyze_garment(actions, episodes, garment_idx)
        if result:
            garment_results.append(result)

    # 计算 garment 间的 CV (变异系数)
    corr_means = [g["correlation_mean"] for g in garment_results]
    mag_means = [g["magnitude_mean"] for g in garment_results]
    grip_means = [g["grip_changes_mean"] for g in garment_results]

    # CV = 标准差 / 均值
    corr_cv = np.std(corr_means) / np.mean(corr_means) if np.mean(corr_means) > 0 else 0
    mag_cv = np.std(mag_means) / np.mean(mag_means) if np.mean(mag_means) > 0 else 0
    grip_cv = np.std(grip_means) / np.mean(grip_means) if np.mean(grip_means) > 0 else 0

    # 计算平均内部标准差
    avg_internal_corr_std = np.mean([g["correlation_std"] for g in garment_results])

    # 打印结果
    print(f"\\n{'Garment':>10} {'Corr':>8} {'Mag':>8} {'Grip':>8} {'内部Std':>10}")
    print("-" * 60)

    problem_garments = []
    for g in garment_results:
        is_problem = g["correlation_std"] > 0.06 or (g["correlation_std"] > 0.05 and g["correlation_mean"] < 0.35)
        marker = " ⚠️" if is_problem else ""
        print(f"Seen_{g['garment_idx']:<5} {g['correlation_mean']:>8.4f} {g['magnitude_mean']:>8.4f} {g['grip_changes_mean']:>8.1f} {g['correlation_std']:>10.4f} {marker}")
        if is_problem:
            problem_garments.append(g["garment_idx"])

    # 汇总
    print(f"\\n数据一致性汇总:")
    print(f"  协调性 CV (Garment 间): {corr_cv:.4f}")
    print(f"  动作幅度 CV (Garment 间): {mag_cv:.4f}")
    print(f"  Gripper CV (Garment 间): {grip_cv:.4f}")
    print(f"  平均内部标准差: {avg_internal_corr_std:.4f}")
    print(f"  问题 Garments: {len(problem_garments)}/10")

    return {
        "dataset": dataset_name,
        "total_frames": len(actions),
        "total_episodes": len(np.unique(episodes)),
        "corr_cv": corr_cv,
        "mag_cv": mag_cv,
        "grip_cv": grip_cv,
        "avg_internal_std": avg_internal_corr_std,
        "problem_garments": problem_garments,
        "garment_results": garment_results,
    }


def main():
    parser = argparse.ArgumentParser(description="数据一致性分析")
    parser.add_argument("--dataset", type=str, help="数据集名称")
    parser.add_argument("--all", action="store_true", help="分析所有数据集")
    parser.add_argument("--output", type=str, help="输出文件路径")
    args = parser.parse_args()

    datasets = [
        "pant_short_merged",
        "pant_long_merged",
        "top_long_merged",
        "top_short_merged",
    ]

    success_rates = {
        "pant_short_merged": 88.3,
        "pant_long_merged": 48.3,
        "top_long_merged": 73.3,
        "top_short_merged": 41.67,
    }

    if args.all:
        print("=" * 80)
        print("四种服装类型数据一致性分析")
        print("=" * 80)

        results = []
        for ds in datasets:
            result = analyze_dataset(ds)
            result["success_rate"] = success_rates.get(ds, 0)
            results.append(result)

        # 打印汇总表格
        print(f"\\n\\n{'='*80}")
        print("汇总表格")
        print(f"{'='*80}")
        print(f"\\n{'数据集':<20} {'成功率':>8} {'Corr CV':>10} {'Mag CV':>10} {'Grip CV':>10} {'问题Gar':>10}")
        print("-" * 80)

        for r in results:
                print(f"{r['dataset']:<20} {r['success_rate']:>7.1f}% {r['corr_cv']:>10.4f} {r['mag_cv']:>10.4f} {r['grip_cv']:>10.4f} {len(r['problem_garments']):>10}")

        # 保存结果
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\\n结果已保存到: {args.output}")

    elif args.dataset:
        analyze_dataset(args.dataset)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
