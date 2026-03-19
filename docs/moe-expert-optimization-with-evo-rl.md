# MoE Expert Optimization with Evo-RL

> **Status**: Design Phase
> **Created**: 2026-03-19
> **Author**: Claude Code + User Collaboration

---

## Executive Summary

This document outlines a progressive approach to optimize individual experts in our MoE-SmolVLA system by incorporating concepts from Evo-RL. The optimization is divided into three stages, from simple data curation to full iterative refinement.

### Key Insights from Evo-RL

| Evo-RL Concept | Application to MoE Experts |
|----------------|---------------------------|
| **Value Function** | Learn to distinguish high-quality vs low-quality states for each garment type |
| **ACP (Advantage-Conditioned Policy)** | Train experts to prefer efficient execution patterns |
| **Iterative Loop** | Continuously improve experts through evaluation → value training → retraining cycles |
| **Data Curation** | Filter demonstrations based on quality metrics |

---

## Current State Analysis

### MoE System Status ✅

```
┌─────────────────────────────────────────────────────────────────┐
│                    Current MoE Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Image                                                    │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────┐                                          │
│  │  VLM (Frozen)    │ ← Shared vision encoder                  │
│  └────────┬─────────┘                                          │
│           │ img_emb                                             │
│           ▼                                                     │
│  ┌──────────────────┐                                          │
│  │  Router (100%)   │ ← Garment type classifier                │
│  └────────┬─────────┘                                          │
│           │ argmax (sticky)                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Selected Expert (e.g., top_long)            │   │
│  │                                                          │   │
│  │    lm_expert (independent) + action_proj (independent)  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Expert Performance (Baseline)

| Expert | Training Data | Checkpoint | Status |
|--------|--------------|------------|--------|
| `pant_short` | Datasets/example/pant_short_merged | outputs/moe_train/.../003000 | ✅ Trained |
| `pant_long` | Datasets/example/pant_long_merged | outputs/moe_train/.../004000 | ✅ Trained |
| `top_short` | Datasets/example/top_short_merged | outputs/moe_train/.../008000 | ✅ Trained |
| `top_long` | Datasets/example/top_long_merged | outputs/moe_train/.../008000 | ✅ Trained |

### Bottlenecks Identified ⚠️

1. **Data Quality**: All demonstrations treated equally, no quality filtering
2. **No Value Learning**: Experts cannot distinguish efficient vs inefficient patterns
3. **Static Training**: No iterative improvement mechanism
4. **Failure Data**: Currently discarded, could be valuable for learning

---

## Three-Stage Optimization Strategy

### Stage 1: Data Curation (Simplest, Immediate Impact)

**Goal**: Filter training data to retain only high-quality demonstrations

**Core Idea**: Not all demonstrations are equal. Some are more efficient, smoother, and more successful than others.

#### Implementation

```python
# scripts/optimize/expert_data_curation.py

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

class DataQualityAnalyzer:
    """Analyze and score demonstration quality"""

    def __init__(self, dataset_root: Path, garment_type: str):
        self.dataset = LeRobotDataset(
            repo_id=f"lehome_{garment_type}",
            root=dataset_root / f"{garment_type}_merged"
        )
        self.garment_type = garment_type

    def calculate_episode_metrics(self, episode_idx: int) -> Dict[str, float]:
        """Calculate quality metrics for a single episode"""
        # Get episode data
        frames = self.dataset.get_episode_data(episode_idx)
        actions = frames["action"]
        states = frames["observation.state"]

        metrics = {}

        # 1. Efficiency Score (0-1) - Shorter is better
        episode_length = len(actions)
        max_expected_length = {
            "pant_short": 200,
            "pant_long": 350,
            "top_short": 250,
            "top_long": 400
        }.get(self.garment_type, 300)
        metrics["efficiency"] = 1 - min(episode_length / max_expected_length, 1.0)

        # 2. Action Smoothness (0-1) - Less jerk is better
        action_diffs = np.diff(actions, axis=0)
        jerk = np.mean(np.abs(action_diffs))
        # Normalize: typical jerk range [0.01, 0.1]
        metrics["smoothness"] = 1 - min((jerk - 0.01) / 0.09, 1.0)
        metrics["smoothness"] = max(metrics["smoothness"], 0.0)

        # 3. State Consistency (0-1) - Less state fluctuation
        state_std = np.std(states, axis=0).mean()
        # Normalize: typical std range [0.1, 1.0]
        metrics["consistency"] = 1 - min((state_std - 0.1) / 0.9, 1.0)
        metrics["consistency"] = max(metrics["consistency"], 0.0)

        # 4. Success Flag (if available)
        if "episode_success" in frames:
            metrics["success"] = frames["episode_success"][0]
        else:
            metrics["success"] = 1.0  # Assume success if not labeled

        return metrics

    def score_episode(self, episode_idx: int,
                     weights: Dict[str, float] = None) -> float:
        """Calculate overall quality score for an episode"""
        if weights is None:
            weights = {
                "efficiency": 0.3,
                "smoothness": 0.3,
                "consistency": 0.2,
                "success": 0.2
            }

        metrics = self.calculate_episode_metrics(episode_idx)

        # Weighted sum
        score = sum(metrics[k] * weights[k] for k in weights)
        return score

    def curate_dataset(self, output_dir: Path,
                      keep_ratio: float = 0.7,
                      weights: Dict[str, float] = None) -> Tuple[List[int], np.ndarray]:
        """Curate dataset by filtering low-quality episodes

        Returns:
            (kept_episodes, scores): Indices of kept episodes and their scores
        """
        all_scores = []

        # Score all episodes
        for ep_idx in range(self.dataset.num_episodes):
            score = self.score_episode(ep_idx, weights)
            all_scores.append(score)

        all_scores = np.array(all_scores)

        # Determine threshold
        threshold = np.percentile(all_scores, keep_ratio * 100)

        # Filter episodes
        kept_episodes = np.where(all_scores >= threshold)[0].tolist()

        # Save curation info
        output_dir.mkdir(parents=True, exist_ok=True)

        curation_info = {
            "garment_type": self.garment_type,
            "total_episodes": self.dataset.num_episodes,
            "kept_episodes": len(kept_episodes),
            "keep_ratio": keep_ratio,
            "threshold": float(threshold),
            "score_distribution": {
                "mean": float(np.mean(all_scores)),
                "std": float(np.std(all_scores)),
                "min": float(np.min(all_scores)),
                "max": float(np.max(all_scores))
            },
            "weights": weights or {}
        }

        import json
        with open(output_dir / "curation_info.json", "w") as f:
            json.dump(curation_info, f, indent=2)

        # Save kept episode indices
        np.save(output_dir / "kept_episodes.npy", kept_episodes)
        np.save(output_dir / "episode_scores.npy", all_scores)

        print(f"[{self.garment_type}] Curated {len(kept_episodes)}/{self.dataset.num_episodes} episodes")
        print(f"  Threshold: {threshold:.3f}")
        print(f"  Score stats: mean={np.mean(all_scores):.3f}, std={np.std(all_scores):.3f}")

        return kept_episodes, all_scores

def curate_all_experts(
    dataset_root: Path,
    output_base_dir: Path,
    keep_ratio: float = 0.7
):
    """Curate training data for all experts"""
    garment_types = ["pant_short", "pant_long", "top_short", "top_long"]

    for garment_type in garment_types:
        print(f"\n{'='*60}")
        print(f"Curating data for {garment_type}")
        print(f"{'='*60}")

        analyzer = DataQualityAnalyzer(dataset_root, garment_type)
        output_dir = output_base_dir / garment_type

        kept_episodes, scores = analyzer.curate_dataset(
            output_dir=output_dir,
            keep_ratio=keep_ratio
        )

        # Optionally: Create filtered dataset
        # This would require implementing a filtered dataset view

if __name__ == "__main__":
    curate_all_experts(
        dataset_root=Path("Datasets/example"),
        output_base_dir=Path("outputs/data_curation"),
        keep_ratio=0.7  # Keep top 70%
    )
```

#### Training with Curated Data

After curation, retrain each expert on filtered data:

```bash
# Example: Retrain pant_short expert on curated data
lerobot-train \
  --dataset.repo_id=lehome_pant_short_curated \
  --dataset.root=outputs/data_curation/pant_short \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --batch_size=16 \
  --steps=30000 \
  --output_dir=outputs/moe_train_v2/smolvla_moe_expert_pant_short_curated
```

**Expected Benefits**:
- ✅ Immediate quality improvement
- ✅ No additional model training required
- ✅ Faster convergence (less noise)
- ✅ Better generalization

---

### Stage 2: Value-Guided Expert Training (Medium Complexity)

**Goal**: Train experts to distinguish and prefer efficient execution patterns

**Core Idea**: Use value functions (from Evo-RL) to learn which states lead to successful outcomes.

#### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                 Value-Guided Expert Training                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Step 1: Train Value Function (per garment type)                   │
│  ────────────────────────────────────────────                       │
│  pistar06(image, state, task) → predicted_value                    │
│                                                                     │
│  Value targets:                                                    │
│    Success episode: V[t] = -remaining_steps / max_length           │
│    Failure episode:  V[t] = -(remaining_steps + c_fail) / max_len  │
│                                                                     │
│  Step 2: Generate ACP Labels                                        │
│  ──────────────────────                                            │
│  advantage[t] = (reward[t:t+n] + bootstrap) - value[t]             │
│  indicator[t] = 1 if advantage >= top_30% else 0                   │
│                                                                     │
│  Step 3: Train Expert with ACP                                      │
│  ──────────────────────────                                        │
│  task_text = "Fold the shirt\nAdvantage: positive"                 │
│  SmolVLA(image, state, task_text) → action                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Implementation

```python
# scripts/optimize/train_value_for_expert.py

import argparse
from pathlib import Path

def train_expert_value_function(
    garment_type: str,
    dataset_root: Path,
    output_dir: Path,
    pretrained_value: Path = None
):
    """Train value function for a specific garment type"""

    # Import Evo-RL components
    from lerobot.scripts.lerobot_value_train import main as value_train_main
    import sys

    # Construct arguments
    args = [
        "--dataset.repo_id", f"lehome_{garment_type}",
        "--dataset.root", str(dataset_root / f"{garment_type}_merged"),
        "--value.type", "pistar06",
        "--value.dtype", "bfloat16",
        "--value.camera_features", "observation.images.top_rgb",
        "--value.camera_features", "observation.images.left_rgb",
        "--value.camera_features", "observation.images.right_rgb",
        "--targets.success_field", "episode_success",
        "--targets.default_success", "failure",
        "--targets.c_fail_coef", "1.0",
        "--batch_size", "64",
        "--steps", "10000",
        "--output_dir", str(output_dir),
        "--job_name", f"pistar06_{garment_type}",
        "--wandb.enable", "true"
    ]

    if pretrained_value is not None:
        args.extend(["--value.pretrained_path", str(pretrained_value)])

    sys.argv = ["lerobot-value-train"] + args
    value_train_main()


# scripts/optimize/generate_acp_labels.py

def generate_acp_labels_for_expert(
    garment_type: str,
    dataset_root: Path,
    value_checkpoint: Path,
    output_dir: Path,
    n_step: int = 50,
    positive_ratio: float = 0.3
):
    """Generate ACP labels for expert training"""

    from lerobot.scripts.lerobot_value_infer import main as value_infer_main
    import sys

    args = [
        "--dataset.repo_id", f"lehome_{garment_type}",
        "--dataset.root", str(dataset_root / f"{garment_type}_merged"),
        "--inference.checkpoint_path", str(value_checkpoint),
        "--runtime.device", "cuda",
        "--runtime.batch_size", "64",
        "--acp.enable", "true",
        "--acp.n_step", str(n_step),
        "--acp.positive_ratio", str(positive_ratio),
        "--acp.value_field", "complementary_info.value",
        "--acp.advantage_field", "complementary_info.advantage",
        "--acp.indicator_field", "complementary_info.acp_indicator",
        "--output_dir", str(output_dir),
        "--job_name", f"acp_{garment_type}"
    ]

    sys.argv = ["lerobot-value-infer"] + args
    value_infer_main()


# scripts/optimize/train_acp_expert.py

def train_expert_with_acp(
    garment_type: str,
    dataset_root: Path,
    pretrained_expert: Path,
    output_dir: Path,
    acp_indicator_dropout: float = 0.3
):
    """Train expert with ACP labels"""

    from lerobot.scripts.lerobot_train import main as train_main
    import sys

    args = [
        "--dataset.repo_id", f"lehome_{garment_type}_acp",
        "--dataset.root", str(dataset_root / f"{garment_type}_merged"),
        "--policy.type", "smolvla",
        "--policy.pretrained_path", str(pretrained_expert),
        "--policy.device", "cuda",
        "--policy.dtype", "bfloat16",
        "--batch_size", "16",
        "--steps", "30000",
        "--acp.enable", "true",
        "--acp.indicator_field", "complementary_info.acp_indicator",
        "--acp.indicator_dropout_prob", str(acp_indicator_dropout),
        "--output_dir", str(output_dir),
        "--job_name", f"smolvla_moe_expert_{garment_type}_acp",
        "--wandb.enable", "true"
    ]

    sys.argv = ["lerobot-train"] + args
    train_main()
```

#### Usage Pipeline

```bash
# Step 1: Train value functions (can be parallelized)
for garment in pant_short pant_long top_short top_long; do
  python -m scripts.optimize.train_value_for_expert \
    --garment_type $garment \
    --dataset_root Datasets/example \
    --output_dir outputs/value_train_v2
done &

# Step 2: Generate ACP labels
for garment in pant_short pant_long top_short top_long; do
  python -m scripts.optimize.generate_acp_labels_for_expert \
    --garment_type $garment \
    --dataset_root Datasets/example \
    --value_checkpoint outputs/value_train_v2/pistar06_$garment/checkpoints/best/pretrained_model \
    --output_dir outputs/acp_inference_v2
done

# Step 3: Train experts with ACP
for garment in pant_short pant_long top_short top_long; do
  python -m scripts.optimize.train_expert_with_acp \
    --garment_type $garment \
    --dataset_root Datasets/example \
    --pretrained_expert outputs/moe_train/smolvla_moe_expert_${garment}_no_st_proj/checkpoints/last/pretrained_model \
    --output_dir outputs/moe_train_v3/acp_expert_${garment}
done &
```

**Expected Benefits**:
- ✅ Experts learn to prefer efficient states
- ✅ Better generalization to unseen scenarios
- ✅ Automatic quality assessment via value function
- ✅ Can be iterated for continuous improvement

---

### Stage 3: Iterative Expert Refinement (Full Pipeline)

**Goal**: Establish a complete closed-loop optimization system

**Core Idea**: Similar to Evo-RL's iterative loop, but applied per-expert

#### Iteration Loop

```
┌─────────────────────────────────────────────────────────────────────┐
│                 Iterative Expert Refinement Loop                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  For iteration = 1 to N:                                            │
│                                                                     │
│  1. Evaluate Current MoE Policy                                     │
│     ├── Deploy experts in simulation                                │
│     ├── Collect success/failure data                                │
│     └── Save to Datasets/eval/iteration_{k}                         │
│                                                                     │
│  2. Merge Data for Each Expert                                     │
│     ├── Original training data                                      │
│     ├── Previous iteration data                                     │
│     └── New evaluation data (with success labels)                   │
│                                                                     │
│  3. Update Value Functions                                         │
│     ├── Train pistar06 on merged data                               │
│     └── Save to outputs/iteration_{k}/value_{expert}                │
│                                                                     │
│  4. Generate ACP Labels                                            │
│     ├── Run value inference on merged data                          │
│     └── Write acp_indicator to dataset                              │
│                                                                     │
│  5. Retrain Experts                                                │
│     ├── Fine-tune from previous checkpoint                          │
│     ├── Use ACP labels in task text                                 │
│     └── Save to outputs/iteration_{k}/expert_{expert}               │
│                                                                     │
│  6. Evaluate Improvement                                            │
│     ├── Compare success rate vs baseline                            │
│     ├── Log metrics to wandb                                        │
│     └── Decide whether to continue                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Implementation

```python
# scripts/optimize/iterative_expert_refinement.py

import argparse
import json
from pathlib import Path
from typing import Dict, List
import subprocess

class IterativeRefinement:
    """Manage iterative expert refinement loop"""

    def __init__(
        self,
        base_output_dir: Path,
        garment_types: List[str],
        num_iterations: int = 3
    ):
        self.base_output_dir = Path(base_output_dir)
        self.garment_types = garment_types
        self.num_iterations = num_iterations

    def run_iteration(self, iteration: int, expert_checkpoints: Dict[str, Path]):
        """Run a single refinement iteration"""

        iter_dir = self.base_output_dir / f"iteration_{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"Iteration {iteration}/{self.num_iterations}")
        print(f"{'='*80}")

        # Step 1: Evaluate current MoE policy
        print("\n[Step 1/5] Evaluating current MoE policy...")
        eval_output = self._evaluate_moe_policy(iteration, expert_checkpoints)

        # Step 2: Merge datasets
        print("\n[Step 2/5] Merging datasets...")
        merged_datasets = self._merge_datasets(iteration)

        # Step 3: Update value functions
        print("\n[Step 3/5] Updating value functions...")
        value_checkpoints = self._train_value_functions(iteration, merged_datasets)

        # Step 4: Generate ACP labels
        print("\n[Step 4/5] Generating ACP labels...")
        self._generate_acp_labels(iteration, merged_datasets, value_checkpoints)

        # Step 5: Retrain experts
        print("\n[Step 5/5] Retraining experts...")
        new_expert_checkpoints = self._retrain_experts(
            iteration,
            merged_datasets,
            expert_checkpoints
        )

        # Step 6: Evaluate improvement
        print("\n[Evaluation] Measuring improvement...")
        improvement = self._evaluate_improvement(
            iteration,
            eval_output,
            new_expert_checkpoints
        )

        # Save iteration summary
        self._save_iteration_summary(iteration, improvement)

        return new_expert_checkpoints, improvement

    def _evaluate_moe_policy(
        self,
        iteration: int,
        expert_checkpoints: Dict[str, Path]
    ) -> Path:
        """Evaluate current MoE policy and save results"""
        eval_output_dir = self.base_output_dir / f"iteration_{iteration}" / "eval_data"

        # Run evaluation
        cmd = [
            "python", "-m", "scripts.eval",
            "--policy_type", "moe_smolvla",
            "--num_episodes", "50",
            "--save_datasets",
            "--eval_dataset_path", str(eval_output_dir),
            "--enable_cameras",
            "--device", "cpu"
        ]

        # Add expert checkpoints
        for garment_type, checkpoint in expert_checkpoints.items():
            cmd.extend([f"--expert_{garment_type}", str(checkpoint)])

        subprocess.run(cmd, check=True)

        return eval_output_dir

    def _merge_datasets(self, iteration: int) -> Dict[str, Path]:
        """Merge datasets for each expert"""
        merged = {}

        for garment_type in self.garment_types:
            # Paths to merge
            original_data = Path(f"Datasets/example/{garment_type}_merged")
            eval_data = self.base_output_dir / f"iteration_{iteration}" / "eval_data" / garment_type

            # Create merged directory
            merged_dir = self.base_output_dir / f"iteration_{iteration}" / "merged_data" / garment_type
            merged_dir.mkdir(parents=True, exist_ok=True)

            # Use LeRobot's merge functionality
            from lerobot.datasets.dataset_tools import merge_datasets
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            datasets_to_merge = [LeRobotDataset(f"{garment_type}_original", root=original_data)]

            if eval_data.exists():
                datasets_to_merge.append(LeRobotDataset(f"{garment_type}_eval", root=eval_data))

            # Add previous iteration data if available
            if iteration > 1:
                prev_data = self.base_output_dir / f"iteration_{iteration-1}" / "merged_data" / garment_type
                if prev_data.exists():
                    datasets_to_merge.append(LeRobotDataset(f"{garment_type}_prev", root=prev_data))

            # Merge
            merged_dataset = merge_datasets(
                datasets=datasets_to_merge,
                repo_id=f"{garment_type}_iter{iteration}",
                root=merged_dir
            )

            merged[garment_type] = merged_dir

        return merged

    def _train_value_functions(
        self,
        iteration: int,
        merged_datasets: Dict[str, Path]
    ) -> Dict[str, Path]:
        """Train value functions for each expert"""
        value_checkpoints = {}

        for garment_type in self.garment_types:
            output_dir = self.base_output_dir / f"iteration_{iteration}" / "value_functions" / garment_type

            # Load previous iteration's value function if available
            pretrained = None
            if iteration > 1:
                pretrained = self.base_output_dir / f"iteration_{iteration-1}" / "value_functions" / garment_type / "checkpoints" / "best" / "pretrained_model"

            train_expert_value_function(
                garment_type=garment_type,
                dataset_root=merged_datasets[garment_type].parent,
                output_dir=output_dir,
                pretrained_value=pretrained
            )

            value_checkpoints[garment_type] = output_dir / "checkpoints" / "best" / "pretrained_model"

        return value_checkpoints

    def _generate_acp_labels(
        self,
        iteration: int,
        merged_datasets: Dict[str, Path],
        value_checkpoints: Dict[str, Path]
    ):
        """Generate ACP labels for each expert"""

        for garment_type in self.garment_types:
            output_dir = self.base_output_dir / f"iteration_{iteration}" / "acp_labels"

            generate_acp_labels_for_expert(
                garment_type=garment_type,
                dataset_root=merged_datasets[garment_type].parent,
                value_checkpoint=value_checkpoints[garment_type],
                output_dir=output_dir
            )

    def _retrain_experts(
        self,
        iteration: int,
        merged_datasets: Dict[str, Path],
        expert_checkpoints: Dict[str, Path]
    ) -> Dict[str, Path]:
        """Retrain experts with ACP labels"""
        new_checkpoints = {}

        for garment_type in self.garment_types:
            output_dir = self.base_output_dir / f"iteration_{iteration}" / "experts"

            new_checkpoint = train_expert_with_acp(
                garment_type=garment_type,
                dataset_root=merged_datasets[garment_type].parent,
                pretrained_expert=expert_checkpoints[garment_type],
                output_dir=output_dir / garment_type
            )

            new_checkpoints[garment_type] = new_checkpoint

        return new_checkpoints

    def _evaluate_improvement(
        self,
        iteration: int,
        eval_output: Path,
        new_expert_checkpoints: Dict[str, Path]
    ) -> Dict:
        """Evaluate improvement compared to baseline"""

        # Load evaluation results
        results_file = eval_output / "evaluation_results.json"

        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)

            # Calculate overall success rate
            total_success = sum(1 for ep in results if ep.get("success", False))
            total_episodes = len(results)
            success_rate = total_success / total_episodes if total_episodes > 0 else 0

            improvement = {
                "iteration": iteration,
                "success_rate": success_rate,
                "total_episodes": total_episodes,
                "successful_episodes": total_success
            }

            # Load baseline if available
            if iteration > 1:
                baseline_file = self.base_output_dir / f"iteration_{iteration-1}" / "improvement.json"
                if baseline_file.exists():
                    with open(baseline_file) as f:
                        baseline = json.load(f)
                    improvement["delta"] = success_rate - baseline.get("success_rate", 0)

            return improvement

        return {"iteration": iteration, "success_rate": 0}

    def _save_iteration_summary(self, iteration: int, improvement: Dict):
        """Save iteration summary"""

        summary_file = self.base_output_dir / f"iteration_{iteration}" / "improvement.json"

        with open(summary_file, "w") as f:
            json.dump(improvement, f, indent=2)

        print(f"\n{'='*80}")
        print(f"Iteration {iteration} Summary")
        print(f"{'='*80}")
        print(f"Success Rate: {improvement.get('success_rate', 0):.2%}")
        if "delta" in improvement:
            print(f"Improvement: {improvement['delta']:+.2%}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Iterative Expert Refinement")
    parser.add_argument("--base_output_dir", type=str, default="outputs/iterative_refinement")
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--initial_experts", type=str, nargs=4,
                       default=["outputs/moe_train/.../pant_short",
                                "outputs/moe_train/.../pant_long",
                                "outputs/moe_train/.../top_short",
                                "outputs/moe_train/.../top_long"])

    args = parser.parse_args()

    # Initialize
    garment_types = ["pant_short", "pant_long", "top_short", "top_long"]

    initial_experts = {
        "pant_short": Path(args.initial_experts[0]),
        "pant_long": Path(args.initial_experts[1]),
        "top_short": Path(args.initial_experts[2]),
        "top_long": Path(args.initial_experts[3])
    }

    refinement = IterativeRefinement(
        base_output_dir=Path(args.base_output_dir),
        garment_types=garment_types,
        num_iterations=args.num_iterations
    )

    # Run iterations
    current_experts = initial_experts

    for iteration in range(1, args.num_iterations + 1):
        current_experts, improvement = refinement.run_iteration(
            iteration=iteration,
            expert_checkpoints=current_experts
        )

        # Check if we should continue
        if iteration < args.num_iterations:
            # You could add logic here to stop early if improvement plateaus
            pass

    print("\n" + "="*80)
    print("Iterative Refinement Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
```

**Expected Benefits**:
- ✅ Continuous improvement over iterations
- ✅ Automatic data quality management
- ✅ Adaptation to edge cases through failure data
- ✅ Full reproduction of Evo-RL's success

---

## Integration with Current MoE System

### Minimal Changes Required

| Component | Change | Impact |
|-----------|--------|--------|
| `MoESmolVLAPolicy` | None - already supports independent experts | ✅ Compatible |
| Expert Checkpoints | Replace with ACP-trained versions | ✅ Drop-in replacement |
| Router | No changes needed | ✅ Unaffected |
| Evaluation | Add `--save_failures` flag | ✅ Minor addition |

### File Structure

```
lehome-challenge/
├── scripts/
│   ├── eval_policy/
│   │   └── moe_smolvla_policy.py          # Existing, no changes needed
│   └── optimize/                          # NEW: Expert optimization scripts
│       ├── expert_data_curation.py
│       ├── train_value_for_expert.py
│       ├── generate_acp_labels_for_expert.py
│       ├── train_acp_expert.py
│       └── iterative_expert_refinement.py
├── third_party/lerobot/                    # Modified LeRobot
│   └── src/lerobot/
│       ├── values/                        # NEW: From Evo-RL
│       │   └── pistar06/
│       ├── rl/                            # NEW: From Evo-RL
│       │   ├── acp_hook.py
│       │   ├── acp_tags.py
│       │   └── acp_dataset_stats.py
│       └── configs/
│           ├── value.py                   # NEW
│           └── train.py                   # MODIFIED: Add ACPConfig
└── docs/
    └── moe-expert-optimization-with-evo-rl.md  # This document
```

---

## Expected Performance Improvements

### Baseline (Current)

| Expert | Success Rate | Notes |
|--------|-------------|-------|
| pant_short | ~88% | From training data |
| pant_long | ~48% | Challenging task |
| top_short | ~42% | Needs improvement |
| top_long | ~73% | Relatively good |

### Projected Improvements

| Stage | pant_short | pant_long | top_short | top_long |
|-------|-----------|-----------|-----------|----------|
| Stage 1: Data Curation | +5% | +8% | +10% | +5% |
| Stage 2: Value-Guided | +10% | +15% | +15% | +10% |
| Stage 3: Iterative (3 rounds) | +15% | +20% | +20% | +15% |

**Note**: These are projections based on Evo-RL paper results. Actual improvements may vary.

---

## Comparison with Alternative Approaches

### Why Not Just Train All Experts Together?

| Aspect | Joint Training | MoE + Expert Optimization |
|--------|---------------|--------------------------|
| Gradient interference | ❌ Severe | ✅ Eliminated |
| Specialization | ❌ None | ✅ Per-expert |
| Data efficiency | ❌ Wasted | ✅ Focused |
| Performance (paper) | 1.6-7.7% | 41-88% |

### Why Not Use Ensemble?

| Aspect | Ensemble | MoE (Sticky Routing) |
|---------|----------|---------------------|
| Inference cost | N× experts | 1× expert |
| Latency | High | Low |
| Determinism | None | Full |
| Interpretability | Low | High |

---

## Timeline and Resources

### Stage 1: Data Curation (1-2 days)

**Tasks**:
- Implement `DataQualityAnalyzer`
- Run analysis on all 4 garment types
- Create curated datasets
- Retrain experts

**Resources**:
- Compute: Minimal (just analysis)
- Storage: ~2GB for curated datasets

### Stage 2: Value-Guided (3-5 days)

**Tasks**:
- Integrate Evo-RL pistar06 component
- Train 4 value functions
- Generate ACP labels
- Retrain experts with ACP

**Resources**:
- Compute: 4× GPU-days for value training
- Storage: ~5GB for value checkpoints

### Stage 3: Iterative Refinement (1-2 weeks)

**Tasks**:
- Implement iteration manager
- Run 3 iterations
- Monitor and log improvements

**Resources**:
- Compute: ~12 GPU-days per iteration
- Storage: ~20GB for all iterations

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Value function overfits | Poor generalization | Use strong regularization, validate on held-out episodes |
| ACP labels too sparse | Limited learning | Adjust `positive_ratio`, use label smoothing |
| Iteration plateaus early | Wasted compute | Monitor improvement, stop early if <2% gain |
| Failure data quality issues | Degrades performance | Manual review of first iteration failures |

---

## Next Steps

1. **Confirm direction**: Which stage to start with?
2. **Resource check**: GPU availability for value training?
3. **Data preparation**: Ensure `episode_success` field in datasets
4. **Baseline measurement**: Run current MoE policy to establish baseline

---

## References

- Evo-RL Repository: `third_party/Evo-RL/`
- Pi\*06 Paper: https://www.pi.website/blog/pistar06
- Current MoE Design: `docs/moe_design_v2.md`
- Evo-RL Integration Plan: `docs/evo-rl-integration-plan.md`

---

## Appendix: Quick Start Commands

### Stage 1: Data Curation

```bash
# Analyze and curate data
python -m scripts.optimize.expert_data_curation \
  --dataset_root Datasets/example \
  --output_dir outputs/data_curation \
  --keep_ratio 0.7

# Retrain experts (example for pant_short)
lerobot-train \
  --config_path configs/train_smolvla_pant_short_curated.yaml
```

### Stage 2: Value-Guided Training

```bash
# Train all value functions (parallel)
for garment in pant_short pant_long top_short top_long; do
  python -m scripts.optimize.train_value_for_expert \
    --garment_type $garment \
    --dataset_root Datasets/example \
    --output_dir outputs/value_train &
done
wait

# Generate ACP labels
for garment in pant_short pant_long top_short top_long; do
  python -m scripts.optimize.generate_acp_labels_for_expert \
    --garment_type $garment \
    --dataset_root Datasets/example \
    --value_checkpoint outputs/value_train/pistar06_$garment/checkpoints/best/pretrained_model \
    --output_dir outputs/acp_inference
done

# Train experts with ACP
for garment in pant_short pant_long top_short top_long; do
  python -m scripts.optimize.train_expert_with_acp \
    --garment_type $garment \
    --dataset_root Datasets/example \
    --pretrained_expert outputs/moe_train/smolvla_moe_expert_${garment}_no_st_proj/checkpoints/last/pretrained_model \
    --output_dir outputs/moe_train_acp/expert_${garment}
done &
```

### Stage 3: Iterative Refinement

```bash
# Run 3 iterations
python -m scripts.optimize.iterative_expert_refinement \
  --base_output_dir outputs/iterative_refinement \
  --num_iterations 3 \
  --initial_experts \
    outputs/moe_train_acp/expert_pant_short/checkpoints/last/pretrained_model \
    outputs/moe_train_acp/expert_pant_long/checkpoints/last/pretrained_model \
    outputs/moe_train_acp/expert_top_short/checkpoints/last/pretrained_model \
    outputs/moe_train_acp/expert_top_long/checkpoints/last/pretrained_model
```

---

*End of Document*
