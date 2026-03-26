# LeHome项目Evo-RL完整流程指南

> **创建日期**: 2026-03-19
> **目的**: 详细说明如何在LeHome项目中运行Evo-RL完整流程
> **状态**: 规划阶段，待用户确认后实施

---

## 目录

1. [概述](#概述)
2. [Evo-RL核心概念](#evo-rl核心概念)
3. [完整流程](#完整流程)
4. [数据格式要求](#数据格式要求)
5. [关键代码解析](#关键代码解析)
6. [待确认问题](#待确认问题)
7. [实施步骤](#实施步骤)

---

## 概述

### 目标

使用Evo-RL框架通过价值函数和ACP（Advantage-Conditioned Policy）方法改进LeHome的Expert模型性能。

### 当前LeHome状态

```
✅ 官方数据集已下载 (Datasets/example/)
✅ 训练好的Expert模型 (MoE-SmolVLA)
✅ 评估脚本 (scripts/eval.py)
✅ 四种garment类型的Expert性能数据:
   - pant_short:  88% success
   - pant_long: 48% success
   - top_short:  42% success
   - top_long:   73% success

❌ 缺少：episode_success字段（需要修改评估脚本）
❓ 缺少：task_index字段（Evo-RL需要）
❓ 缺少：complementary_info数据结构
❌ 缺少：价值函数模型
❌ 缺少：ACP标签数据
```

### 预期效果（基于PI的RECAP数据）

- **低成功率任务** (pant_long 48%, top_short 42%): 预期提升15-25%
- **高成功率任务** (pant_short 88%): 预期提升5-10%
- **处理效率**: 可能提升30-50%

---

## Evo-RL核心概念

### 架构概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Evo-RL 核心组件                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  1. 价值函数 (Pistar06)                                                  │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  输入: observation (images, state) + task                     │        │
│  │  输出: V(s) ∈ [-1, 0] (状态价值预测)                          │        │
│  │  架构: SigLIP SO400M + Gemma 3 270M                           │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                                                                           │
│  2. ACP (Advantage-Conditioned Policy)                                  │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  Advantage = Return - V(s)                                   │        │
│  │  指标二值化: positive/top-30% vs negative/bottom-70%           │        │
│  │  Prompt注入: "Fold shirt\nAdvantage: positive/negative"      │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                                                                           │
│  3. HIL (Human-in-the-Loop)                                            │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  S0: 策略控制 → 标记为positive                                │        │
│  │  S1: 人工干预 → 标记为negative                                │        │
│  │  S2: 释放过渡 → 人工动作标记为positive                        │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### 关键代码位置

| 组件 | 文件路径 | 说明 |
|------|---------|------|
| 价值推理 | `src/lerobot/scripts/lerobot_value_infer.py` | 核心脚本：价值预测+ACP标签生成 |
| ACP标签 | `src/lerobot/rl/acp_tags.py` | ACP标签常量和工具函数 |
| ACP Hook | `src/lerobot/rl/acp_hook.py` | 训练时注入ACP标签 |
| 价值训练 | `src/lerobot/scripts/lerobot_value_train.py` | 价值函数训练脚本 |
| HIL示例 | `examples/tutorial/rl/hilserl_example.py` | 完整HIL-SERL示例 |

---

## 完整流程

### 流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Evo-RL 完整流程                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  阶段0: 数据准备                                                          │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  ✓ 修改evaluation.py添加episode_success字段                   │        │
│  │  ✓ 确认数据集包含task和task_index字段                         │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                              ↓                                          │
│  阶段1: 收集初始数据                                                      │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  使用当前Expert评估 → 收集成功+失败episodes                   │        │
│  │  输出: Datasets/evo_rl_round1/ (含episode_success)            │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                              ↓                                          │
│  阶段2: 训练价值函数                                                      │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  lerobot-value-train                                         │        │
│  │  输入: Datasets/evo_rl_round1/                                │        │
│  │  输出: outputs/value_train_round1/ (Pistar06模型)             │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                              ↓                                          │
│  阶段3: 价值推理 + ACP标签生成                                          │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  lerobot-value-infer                                         │        │
│  │  输入: 数据集 + 价值函数checkpoint                            │        │
│  │  输出: 添加complementary_info字段到原数据集                   │        │
│  │       (value, advantage, acp_indicator)                       │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                              ↓                                          │
│  阶段4: 使用ACP标签训练Expert                                           │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  lerobot-train --acp.enable=true                             │        │
│  │  训练时自动注入ACP标签到task prompt                           │        │
│  │  输出: outputs/train_acp_round1/ (改进后的Expert)             │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                              ↓                                          │
│  阶段5: 迭代改进                                                          │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  评估新Expert → 收集更多数据 → 重复阶段2-4                    │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 数据格式要求

### 完整字段列表

```
LeRobot数据集需要的字段：

══════════════════════════════════════════════════════════════════════════════
必需字段（现有）:
├── observation.state (12D float) - 关节位置
├── observation.images.top_rgb (480x640x3 uint8)
├── observation.images.left_rgb (480x640x3 uint8)
├── observation.images.right_rgb (480x640x3 uint8)
├── action (12D float) - 关节动作
├── episode_index (int64)
├── frame_index (int64)
└── timestamp (float64)

══════════════════════════════════════════════════════════════════════════════
需要添加/确认:
├── task (string) ← 任务描述，如 "fold the pant_long"
├── task_index (int64) ← 任务类型索引 (0=pant_long, 1=pant_short, ...)
└── episode_success (bool) ← episode是否成功

══════════════════════════════════════════════════════════════════════════════
ACP推理后自动添加:
└── complementary_info
    ├── value (float32) ← 价值预测 V(s) ∈ [-1, 0]
    ├── advantage (float32) ← 优势 A = Return - V(s)
    ├── acp_indicator (int64) ← ACP标签 (0=negative, 1=positive)
    └── is_intervention (bool, optional) ← 是否人工干预
```

### episode_success字段添加

**文件**: `scripts/utils/evaluation.py` (约206-217行)

**原代码**（只保存成功）:
```python
# Save Datasets
if args.save_datasets:
    if success_flag:
        eval_dataset.save_episode()
        append_episode_initial_pose(
            json_path,
            episode_index,
            object_initial_pose,
            garment_name=garment_name,
        )
        episode_index += 1
    else:
        eval_dataset.clear_episode_buffer()  # 丢弃失败数据
```

**修改后**（保存所有，带success标签）:
```python
# Save Datasets
if args.save_datasets:
    # 添加episode_success字段到每一帧
    for frame in eval_dataset.frames:
        frame["episode_success"] = is_success

    eval_dataset.save_episode()
    append_episode_initial_pose(
        json_path,
        episode_index,
        object_initial_pose,
        garment_name=garment_name,
    )
    episode_index += 1
```

---

## 关键代码解析

### 1. 价值推理脚本 (lerobot_value_infer.py)

#### 核心函数分析

**价值预测** (第387-410行):
```python
with torch.no_grad():
    for raw_batch in eval_loader:
        processed_batch = preprocessor(raw_batch)
        predicted_value = value_policy.predict_value(processed_batch)
```

**计算价值目标** (第417行):
```python
value_targets = compute_normalized_value_targets(
    episode_indices=episode_indices,
    frame_indices=frame_indices,
    episode_info=episode_info,
    task_max_lengths=task_max_lengths,
    c_fail_coef=cfg.acp.c_fail_coef,  # 失败惩罚系数
    clip_min=value_cfg.bin_min,      # -1.0
    clip_max=value_cfg.bin_max,      # 0.0
)
# 成功episode的目标值: 从0递增到接近0
# 失败episode的目标值: 从-c_fail_coef递增到接近-1
```

**计算奖励** (第127行):
```python
def _compute_dense_rewards_from_targets(targets, episode_indices, frame_indices):
    rewards = np.zeros_like(targets, dtype=np.float32)
    for i in range(n):
        is_next_in_episode = (
            i + 1 < n
            and episode_indices[i + 1] == episode_indices[i]
            and frame_indices[i + 1] == frame_indices[i] + 1
        )
        if is_next_in_episode:
            rewards[i] = float(targets[i] - targets[i + 1])
        else:
            rewards[i] = float(targets[i])  # 最后一帧
    return rewards
```

**计算N-step Advantage** (第132行):
```python
def _compute_n_step_advantages(rewards, values, episode_indices, frame_indices, n_step):
    advantages = np.zeros(n, dtype=np.float32)
    for i in range(n):
        discounted_sum = sum(rewards[i:i+n_step])
        if can_bootstrap:
            bootstrap = values[i + n_step]
        else:
            bootstrap = 0.0
        advantages[i] = discounted_sum + bootstrap - values[i]
    return advantages
```

**计算阈值并二值化** (第147-154行):
```python
# 按task计算分位数阈值
thresholds = {}
for task_idx in np.unique(task_indices):
    task_adv = advantages[task_indices == task_idx]
    quantile = 1.0 - positive_ratio  # 默认0.3 → 0.7分位数
    thresholds[task_idx] = np.quantile(task_adv, quantile)

# 二值化
indicators = np.zeros_like(advantages, dtype=np.int64)
for i in range(len(advantages)):
    task_idx = task_indices[i]
    indicators[i] = 1 if advantages[i] >= thresholds[task_idx] else 0
```

**写回数据集** (第425行):
```python
_write_columns_in_place(
    dataset_root=Path(dataset.root),
    absolute_indices=absolute_indices,
    columns={
        cfg.acp.value_field: predicted_values,      # V(s)
        cfg.acp.advantage_field: advantages,        # A
        cfg.acp.indicator_field: indicators,        # 0/1标签
    },
    feature_infos=feature_infos,
)
```

### 2. ACP Hook (acp_hook.py)

训练时自动修改task prompt:

```python
class ACPPromptHook:
    def __call__(self, batch, step):
        tasks = batch["task"]  # ["Fold pant_long", ...]
        indicators = batch["complementary_info.acp_indicator"]  # [1, 0, 1, ...]

        conditioned_tasks = []
        for task, is_positive in zip(tasks, indicators):
            if dropout and rng.random() < dropout_prob:
                conditioned_tasks.append(task)  # 不添加标签
            else:
                tag = "Advantage: positive" if is_positive else "Advantage: negative"
                conditioned_tasks.append(f"{task}\n{tag}")

        batch["task"] = conditioned_tasks
        return batch
```

**效果示例**:
```
原始:  "Fold the pant_long"
Positive: "Fold the pant_long\nAdvantage: positive"
Negative: "Fold the pant_long\nAdvantage: negative"
```

### 3. HIL干预状态机 (hilserl_example.py)

```python
# 人工干预检测
is_intervention = teleop_events.get(TeleopEvents.IS_INTERVENTION, False)

# 存储带干预标签的transition
transition = {
    "state": obs,
    "action": action,
    "reward": reward,
    "complementary_info": {
        "is_intervention": is_intervention,  # 关键！
    },
}

# 在ACP推理时，干预帧强制标记为positive
if force_intervention_positive:
    indicators[intervention_mask] = 1
```

---

## 待确认问题

### 问题1: task和task_index字段

Evo-RL要求数据集必须有这些字段：

```python
# lerobot_value_infer.py 第345行
if value_cfg.task_index_feature not in raw_frames.column_names:
    raise KeyError(f"Missing task feature '{value_cfg.task_index_feature}'")
```

**检查方法**:
```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("Datasets/example/pant_long_merged")
print("Columns:", ds.hf_dataset.column_names)

# 需要确认是否包含:
# - "task" (string)
# - "task_index" (int64)
```

**如果没有，需要添加**:
```python
# 添加task字段
task_name = f"fold the {garment_type}"

# 添加task_index字段
task_to_index = {
    "pant_long": 0,
    "pant_short": 1,
    "top_long": 2,
    "top_short": 3,
}
task_index = task_to_index[garment_type]
```

### 问题2: complementary_info数据结构

Evo-RL使用嵌套的`complementary_info`结构：

```python
complementary_info:
    ├── value (float32)
    ├── advantage (float32)
    ├── acp_indicator (int64)
    └── is_intervention (bool)
```

**检查你的数据集**:
- 是否使用这种嵌套结构？
- 还是字段平铺在顶层？

**如果是平铺结构**，需要配置rename_map:
```python
# 或者修改Evo-RL代码适配平铺结构
```

### 问题3: Expert模型是否支持text输入

ACP标签注入到task中，Expert必须支持text输入：

```python
# 训练时的输入
{
    "observation.images.top_rgb": ...,
    "observation.state": ...,
    "task": "Fold the pant_long\nAdvantage: positive",  # 必须支持
}
```

**SmolVLA应该支持**，但需要确认：
- task输入在哪里？
- 如何传递给VLM？

### 问题4: 图像格式兼容性

Pistar06使用SigLIP SO400M，期望：
- 图像尺寸: 480x640 (RGB)
- 数据类型: uint8 [0, 255]

**确认LeRobot数据集的图像格式是否兼容**。

### 问题5: 选择哪个garment类型开始

建议：
1. **从成功率低的开始** (pant_long 48%, top_short 42%)
2. 提升空间更大
3. 更容易看到效果

### 问题6: GPU资源

- 价值函数训练: 需要GPU
- 价值推理: 需要GPU (可用CPU但很慢)
- Expert训练: 需要GPU

**确认可用GPU显存**（建议16GB+）。

---

## 实施步骤

### 第一步：数据集格式检查

```bash
# 检查当前数据集字段
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pprint import pprint

ds = LeRobotDataset('Datasets/example/pant_long_merged')
print('=== Dataset Columns ===')
pprint(ds.hf_dataset.column_names)
print()
print('=== Dataset Info ===')
pprint(ds.info)
"
```

**期望输出应包含**:
- `task` (string)
- `task_index` (int64)
- 或者需要确认如何添加

### 第二步：修改评估脚本

见上方"episode_success字段添加"部分。

### 第三步：收集初始数据

```bash
# 收集50个episodes (包含成功和失败)
python -m scripts.eval \
    --policy_type custom \
    --policy_path outputs/train/expert/checkpoints/last/pretrained_model \
    --garment_type "pant_long" \
    --dataset_root Datasets/example/pant_long_merged \
    --num_episodes 50 \
    --save_datasets \
    --dataset_output_dir Datasets/evo_rl_round1 \
    --device cpu
```

**验证输出**:
```bash
# 检查新数据集
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset('Datasets/evo_rl_round1')
# 确认episode_success字段存在
print('episode_success' in ds.hf_dataset.column_names)

# 统计成功率
import numpy as np
success = ds.hf_dataset['episode_success']
print(f'Success rate: {np.mean(success):.2%}')
"
```

### 第四步：安装Evo-RL

```bash
# 克隆Evo-RL仓库
git clone https://github.com/MINT-SJTU/Evo-RL.git third_party/Evo-RL
cd third_party/Evo-RL

# 安装（在LeHome虚拟环境中）
pip install -e .
```

### 第五步：训练价值函数

```bash
# 配置参数
DATASET_REPO="Datasets/evo_rl_round1"
DATASET_ROOT="/path/to/lehome/Datasets/evo_rl_round1"
OUTPUT_DIR="outputs/value_train_round1"

# 训练
lerobot-value-train \
  --dataset.repo_id=${DATASET_REPO} \
  --dataset.root=${DATASET_ROOT} \
  --dataset.success_field=episode_success \
  --dataset.default_success=false \
  --value.type=pistar06 \
  --value.dtype=bfloat16 \
  --value.device=cuda \
  --value.batch_size=64 \
  --value.steps=8000 \
  --value.learning_rate=1e-4 \
  --wandb.enable=true \
  --wandb.project=lehome-evo-rl \
  --wandb.run.name=value_round1 \
  --output_dir=${OUTPUT_DIR}
```

### 第六步：价值推理 + ACP标签生成

```bash
# 配置参数
CHECKPOINT_PATH="outputs/value_train_round1"
OUTPUT_DIR="outputs/value_infer_round1"

# 价值推理和ACP标签生成
lerobot-value-infer \
  --dataset.repo_id=${DATASET_REPO} \
  --dataset.root=${DATASET_ROOT} \
  --dataset.success_field=episode_success \
  --dataset.default_success=false \
  --inference.checkpoint_path=${CHECKPOINT_PATH} \
  --inference.checkpoint_ref=last \
  --runtime.device=cuda \
  --runtime.batch_size=64 \
  --acp.enable=true \
  --acp.n_step=50 \
  --acp.positive_ratio=0.3 \
  --acp.c_fail_coef=1.0 \
  --acp.force_intervention_positive=true \
  --acp.value_field=complementary_info.value \
  --acp.advantage_field=complementary_info.advantage \
  --acp.indicator_field=complementary_info.acp_indicator \
  --output_dir=${OUTPUT_DIR} \
  --viz.enable=true
```

**验证输出**:
```bash
# 检查ACP标签是否添加成功
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np

ds = LeRobotDataset('Datasets/evo_rl_round1')

# 检查新字段
print('complementary_info.value' in ds.hf_dataset.column_names)
print('complementary_info.advantage' in ds.hf_dataset.column_names)
print('complementary_info.acp_indicator' in ds.hf_dataset.column_names)

# 统计positive比例
indicators = ds.hf_dataset['complementary_info.acp_indicator']
print(f'Positive ratio: {np.mean(indicators):.2%}')

# 查看价值分布
values = ds.hf_dataset['complementary_info.value']
print(f'Value range: [{np.min(values):.3f}, {np.max(values):.3f}]')

# 查看advantage分布
advantages = ds.hf_dataset['complementary_info.advantage']
print(f'Advantage range: [{np.min(advantages):.3f}, {np.max(advantages):.3f}]')
"
```

### 第七步：使用ACP标签训练Expert

```bash
# 配置参数
EXPERT_PATH="outputs/train/expert/checkpoints/last/pretrained_model"
OUTPUT_DIR="outputs/train_acp_round1"

# 训练
lerobot-train \
  --dataset.repo_id=${DATASET_REPO} \
  --policy.type=smolvla \
  --policy.pretrained_path=${EXPERT_PATH} \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --batch_size=32 \
  --steps=30000 \
  --acp.enable=true \
  --acp.indicator_field=complementary_info.acp_indicator \
  --acp.indicator_dropout_prob=0.3 \
  --wandb.enable=true \
  --wandb.project=lehome-evo-rl \
  --wandb.run.name=acp_round1 \
  --output_dir=${OUTPUT_DIR}
```

### 第八步：评估改进后的Expert

```bash
# 评估新训练的Expert
python -m scripts.eval \
    --policy_type custom \
    --policy_path outputs/train_acp_round1/checkpoints/last/pretrained_model \
    --garment_type "pant_long" \
    --dataset_root Datasets/example/pant_long_merged \
    --num_episodes 50 \
    --device cpu

# 对比原始Expert性能
# 预期: 成功率从48%提升到60-70%
```

### 第九步：迭代改进

```bash
# 使用改进后的Expert收集更多数据
python -m scripts.eval \
    --policy_type custom \
    --policy_path outputs/train_acp_round1/checkpoints/last/pretrained_model \
    --garment_type "pant_long" \
    --dataset_root Datasets/example/pant_long_merged \
    --num_episodes 50 \
    --save_datasets \
    --dataset_output_dir Datasets/evo_rl_round2 \
    --device cpu

# 合并数据集
# 然后重复第五步到第八步
```

---

## 参考命令汇总

### Evo-RL核心命令

| 命令 | 用途 |
|------|------|
| `lerobot-value-train` | 训练价值函数 |
| `lerobot-value-infer` | 价值推理 + ACP标签生成 |
| `lerobot-train` | 训练策略（支持ACP） |
| `lerobot-eval` | 评估策略 |

### 参数说明

#### 价值训练关键参数

```
--value.type=pistar06           # 价值函数类型
--value.dtype=bfloat16          # 数据类型
--value.device=cuda             # 设备
--value.batch_size=64           # 批大小
--value.steps=8000              # 训练步数
--value.learning_rate=1e-4      # 学习率
```

#### 价值推理关键参数

```
--acp.enable=true                        # 启用ACP
--acp.n_step=50                          # N-step advantage
--acp.positive_ratio=0.3                 # Positive比例
--acp.c_fail_coef=1.0                    # 失败惩罚系数
--acp.value_field=complementary_info.value
--acp.advantage_field=complementary_info.advantage
--acp.indicator_field=complementary_info.acp_indicator
```

#### ACP训练关键参数

```
--acp.enable=true                               # 启用ACP
--acp.indicator_field=complementary_info.acp_indicator
--acp.indicator_dropout_prob=0.3                # Dropout概率
```

---

## 常见问题

### Q1: 如果数据集没有task字段怎么办？

需要手动添加。可以在数据收集时添加，或者后处理：

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import pyarrow.parquet as pq

ds = LeRobotDataset("Datasets/evo_rl_round1")

# 定义task映射
task_name = "fold the pant_long"
task_index = 0

# 遍历所有parquet文件添加字段
data_files = sorted((Path(ds.root) / "data").glob("chunk-*/file-*.parquet"))
for parquet_path in data_files:
    table = pq.read_table(parquet_path)
    # 添加task和task_index列
    ...
```

### Q2: 价值函数训练多久？

根据Evo-RL论文：
- 简单任务: ~5000 steps
- 复杂任务: ~8000-10000 steps

建议先从5000步开始，检查loss收敛情况。

### Q3: ACP标签的positive_ratio如何设置？

默认0.3（30%标记为positive）：
- 太高（如0.5）: 策略可能学不到区别
- 太低（如0.1）: positive样本太少
- 建议: 0.2-0.4

### Q4: 每轮迭代需要多少数据？

根据PI的经验：
- 简单任务: ~600 episodes/轮
- 复杂任务: ~600-1000 episodes/轮

LeHome建议：
- 第一轮: 50 episodes（测试流程）
- 后续: 根据效果决定是否增加

### Q5: 如何监控训练进度？

```bash
# WandB自动记录
--wandb.enable=true \
--wandb.project=lehome-evo-rl \
--wandb.run-name=experiment_name

# 或查看tensorboard
tensorboard --logdir outputs/value_train_round1
```

---

## 下一步

请确认以下问题后，我们可以开始实施：

1. [ ] 数据集当前字段情况（task, task_index是否存在）
2. [ ] complementary_info结构确认
3. [ ] Expert模型支持text输入确认
4. [ ] GPU资源确认
5. [ ] 选择起始garment类型
6. [ ] 是否需要实现HIL（人工干预）功能

---

*创建于2026-03-19*
*基于Evo-RL代码库 (MINT-SJTU/Evo-RL) 分析*
