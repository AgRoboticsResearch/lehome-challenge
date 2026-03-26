# Evo-RL 集成计划

> 创建日期: 2026-03-14
> 状态: 设计阶段，待用户确认后实施

---

## 1. 背景与目标

### 1.1 当前状态

| 组件 | 状态 | 说明 |
|------|------|------|
| SmolVLA 模型 | ✅ 已训练 | 在全成功数据上 fine-tuned |
| 仿真环境 | ✅ 可用 | Isaac Lab，支持自动成功判断 |
| 人工演示数据 | ✅ 可用 | 全成功轨迹 |
| 失败数据 | ❌ 缺失 | 当前评估丢弃失败数据 |
| 价值函数 | ❌ 未集成 | pistar06 组件未添加 |
| ACP 训练 | ❌ 未集成 | 优势条件策略未添加 |

### 1.2 目标

将 Evo-RL 的价值函数引导策略学习集成到 LeHome 项目中，实现：
1. 收集成功+失败混合数据
2. 训练价值函数区分成功/失败
3. 生成 ACP 标签改进策略训练
4. 迭代优化策略性能

---

## 2. Evo-RL 核心概念

### 2.1 架构对比

```
┌────────────────┬─────────────────────┬─────────────────────┐
│     组件       │   pistar06 (价值)   │   SmolVLA (策略)    │
├────────────────┼─────────────────────┼─────────────────────┤
│ 视觉编码器     │ SigLIP (384×384)    │ SmolVLM2 (512×512)  │
│ 语言模型       │ Gemma-3-270M        │ SmolVLM2-500M       │
│ 输出           │ 价值分布 (201 bins) │ 动作序列 (50 steps) │
│ 状态归一化     │ QUANTILES           │ MEAN_STD            │
│ 参数量         │ ~270M               │ ~500M               │
└────────────────┴─────────────────────┴─────────────────────┘
```

### 2.2 价值函数设计

**核心公式：**
```
价值 = -(剩余代价) / 最大可能代价

成功轨迹: 价值从负数逐渐上升到 0
失败轨迹: 价值始终更负 (有 c_fail 惩罚)
```

**代码实现 (modeling_pistar06.py:100-122):**
```python
def compute_normalized_value_targets(...):
    remaining_steps = ep.length - frame_index - 1

    if ep.success:
        g = -remaining_steps           # 成功：代价 = 剩余步数
    else:
        g = -remaining_steps - c_fail  # 失败：额外惩罚

    g_norm = g / (task_max_length + c_fail)
    targets[i] = np.clip(g_norm, -1.0, 0.0)
```

**可视化示例：**
```
成功轨迹 (5步完成):
帧:        [0]     [1]     [2]     [3]     [4] ← 成功
价值目标:  -0.8   -0.6    -0.4    -0.2     0.0
         起点           逐渐接近成功      终点

失败轨迹 (5步后失败):
帧:        [0]     [1]     [2]     [3]     [4] ← 失败
价值目标:  -1.0   -0.8    -0.6    -0.4    -0.2
         起点更低                        永远到不了0
```

### 2.3 ACP (Advantage-Conditioned Policy) 流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 1: 价值函数预测                                                        │
│  predicted_value = pistar06(image, state, task)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Step 2: 计算 dense rewards                                                 │
│  reward[i] = target[i] - target[i+1]  # "这一步进步了多少"                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  Step 3: 计算 n-step advantage                                              │
│  advantage[i] = Σ reward[i:i+n] + bootstrap - value[i]                      │
│  含义: "实际表现 vs 预期表现"                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  Step 4: 二值化                                                              │
│  threshold = quantile(advantages, 1 - positive_ratio)  # e.g., top 30%      │
│  indicator = 1 if advantage >= threshold else 0                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Step 5: ACP 标签注入                                                        │
│  task_text = "Fold the shirt\nAdvantage: positive"  (或 negative)           │
└─────────────────────────────────────────────────────────────────────────────┘
```

**ACP 标签含义：**
- `positive`: 这个状态比预期更快接近成功 (高效执行)
- `negative`: 这个状态比预期更慢接近成功 (低效执行)

---

## 3. 集成方案

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LeHome + Evo-RL 完整流程                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 1: 数据收集                                                          │
│  ─────────────────                                                          │
│  SmolVLA 评估 → 仿真环境 → 自动判断成功/失败 → 保存到数据集                    │
│                                                                             │
│  Phase 2: 价值函数训练                                                       │
│  ─────────────────────                                                      │
│  混合数据集 → lerobot-value-train → pistar06 模型                            │
│                                                                             │
│  Phase 3: ACP 标签生成                                                       │
│  ─────────────────────                                                      │
│  pistar06 推理 → lerobot-value-infer → 数据集添加 ACP 字段                   │
│                                                                             │
│  Phase 4: ACP 策略训练                                                       │
│  ─────────────────────                                                      │
│  增强数据集 → lerobot-train (acp.enable=true) → 改进的 SmolVLA              │
│                                                                             │
│  Phase 5: 迭代优化                                                          │
│  ─────────────────                                                          │
│  改进的策略 → 评估 → 收集新数据 → 重新训练 → ...                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 代码修改范围

#### Phase 1: 数据收集

**文件: `scripts/utils/evaluation.py`**

| 位置 | 当前行为 | 修改内容 |
|------|----------|----------|
| 创建数据集 | 无 `episode_success` | 添加 `episode_success` 到 features |
| 录制帧 | 无 success 标记 | 添加 `frame["episode_success"]` |
| 保存逻辑 | 只保存成功，丢弃失败 | 保存成功和失败 |

**修改示意：**
```python
# 改前:
if success_flag:
    eval_dataset.save_episode()
else:
    eval_dataset.clear_episode_buffer()  # 丢弃失败

# 改后:
# 保存成功和失败
frame["episode_success"] = 1.0 if success_flag else 0.0
eval_dataset.save_episode()  # 都保存
```

**文件: `scripts/utils/parser.py`**
- 添加 `--save_failures` 参数

#### Phase 2-4: 添加 Evo-RL 组件

**复制文件清单：**

```
从 Evo-RL 复制到 third_party/lerobot/src/lerobot/:

values/                              # 【新建目录】
└── pistar06/
    ├── __init__.py                  # 【复制】
    ├── configuration_pistar06.py    # 【复制】
    ├── modeling_pistar06.py         # 【复制】
    └── processor_pistar06.py        # 【复制】

rl/                                  # 【新建目录】
├── __init__.py                      # 【新建】
├── acp_hook.py                      # 【复制】
├── acp_tags.py                      # 【复制】
└── acp_dataset_stats.py             # 【复制】

configs/
├── value.py                         # 【新建】价值推理配置
└── value_train.py                   # 【新建】价值训练配置

scripts/
├── lerobot_value_train.py           # 【复制】
└── lerobot_value_infer.py           # 【复制】
```

**修改文件：**

```
third_party/lerobot/src/lerobot/configs/train.py
  - 添加 ACPConfig 类

third_party/lerobot/pyproject.toml
  - 添加入口点: lerobot-value-train, lerobot-value-infer
```

### 3.3 配置文件

**新建: `configs/value_train_top_long.yaml`**
```yaml
dataset:
  repo_id: lehome_top_long
  root: Datasets/eval/top_long  # 包含成功+失败的数据

value:
  type: pistar06
  dtype: bfloat16
  camera_features:
    - observation.images.top_rgb
    - observation.images.left_rgb
    - observation.images.right_rgb

targets:
  success_field: episode_success
  default_success: failure  # 默认失败，除非标记成功
  c_fail_coef: 1.0

output_dir: outputs/value_train/pistar06_top_long
batch_size: 64
steps: 8000
```

**新建: `configs/value_infer_top_long.yaml`**
```yaml
dataset:
  repo_id: lehome_top_long
  root: Datasets/example/top_long_merged

inference:
  checkpoint_path: outputs/value_train/pistar06_top_long

acp:
  enable: true
  n_step: 50
  positive_ratio: 0.3
  value_field: complementary_info.value
  advantage_field: complementary_info.advantage
  indicator_field: complementary_info.acp_indicator
```

**新建: `configs/train_smolvla_acp_top_long.yaml`**
```yaml
dataset:
  repo_id: lehome_top_long_acp
  root: Datasets/example/top_long_merged  # 已包含 ACP 标签

policy:
  type: smolvla
  pretrained_path: lerobot/smolvla_base
  # ... 其他配置 ...

acp:
  enable: true
  indicator_field: complementary_info.acp_indicator
  indicator_dropout_prob: 0.3
```

---

## 4. 实施步骤

### Step 1: 修改数据收集 (预计 1-2 天)

1. 修改 `scripts/utils/evaluation.py`
   - 添加 `episode_success` 到数据集 features
   - 修改保存逻辑，保存成功和失败数据

2. 修改 `scripts/utils/parser.py`
   - 添加 `--save_failures` 参数

3. 运行评估收集数据
   ```bash
   python -m scripts.eval \
     --policy_type lerobot \
     --policy_path outputs/train/smolvla_top_long/checkpoints/last/pretrained_model \
     --garment_type "top_long" \
     --dataset_root Datasets/example/top_long_merged \
     --save_datasets \
     --eval_dataset_path Datasets/eval/top_long \
     --num_episodes 50
   ```

### Step 2: 添加 Evo-RL 组件 (预计 1 天)

1. 复制 pistar06 文件
2. 复制 ACP 相关文件
3. 复制配置和脚本文件
4. 修改 train.py 添加 ACPConfig
5. 修改 pyproject.toml 添加入口点
6. 本地安装: `pip install -e third_party/lerobot`

### Step 3: 训练价值函数 (预计 1-2 天)

```bash
lerobot-value-train --config_path configs/value_train_top_long.yaml
```

### Step 4: 生成 ACP 标签 (预计 0.5 天)

```bash
lerobot-value-infer --config_path configs/value_infer_top_long.yaml
```

### Step 5: ACP 策略训练 (预计 1-2 天)

```bash
lerobot-train --config_path configs/train_smolvla_acp_top_long.yaml
```

### Step 6: 评估对比 (预计 0.5 天)

对比 baseline vs ACP 成功率。

---

## 5. 全成功数据 vs 混合数据

### 5.1 全成功数据的 ACP 效果

| 方面 | 全成功数据 | 有失败数据 |
|------|-----------|-----------|
| 价值区分 | 只能区分"快/慢成功" | 能区分"成功/失败" |
| Advantage 范围 | 较窄 | 更宽 |
| 学习信号 | 弱但有意义 | 更强 |
| 策略改进 | 主要优化效率 | 避免失败 + 更高效 |

### 5.2 建议

**推荐：先收集失败数据，用混合数据训练**

理由：
1. 价值函数的核心是区分成功和失败
2. 全成功数据下，ACP 只能学习"高效执行"
3. 有失败数据后，策略能学习"避免失败"

---

## 6. 文件清单汇总

| 操作 | 文件路径 | 说明 |
|------|----------|------|
| **修改** | `scripts/utils/evaluation.py` | 保存成功+失败数据 |
| **修改** | `scripts/utils/parser.py` | 添加参数 |
| **新建** | `third_party/lerobot/.../values/__init__.py` | 模块入口 |
| **复制** | `third_party/lerobot/.../values/pistar06/*.py` | 4个文件 |
| **新建** | `third_party/lerobot/.../rl/__init__.py` | 模块入口 |
| **复制** | `third_party/lerobot/.../rl/acp_*.py` | 3个文件 |
| **新建** | `third_party/lerobot/.../configs/value.py` | 配置类 |
| **新建** | `third_party/lerobot/.../configs/value_train.py` | 配置类 |
| **修改** | `third_party/lerobot/.../configs/train.py` | 添加 ACPConfig |
| **复制** | `third_party/lerobot/.../scripts/lerobot_value_train.py` | 训练脚本 |
| **复制** | `third_party/lerobot/.../scripts/lerobot_value_infer.py` | 推理脚本 |
| **修改** | `third_party/lerobot/pyproject.toml` | 添加入口点 |
| **新建** | `configs/value_train_*.yaml` | 训练配置 |
| **新建** | `configs/value_infer_*.yaml` | 推理配置 |
| **新建** | `configs/train_smolvla_acp_*.yaml` | ACP训练配置 |

---

## 7. evaluation.py 详细修改方案

### 7.1 修改概述

**目标**: 保存成功和失败数据，添加 `episode_success` 字段

**核心原则**: 使用缓存方式 (方式 A) - 先缓存所有帧，episode 结束后用最终状态统一标记

### 7.2 数据流对比

```
修改前:
┌─────────────────────────────────────────────────────────────────────────────┐
│  for step in episode:                                                        │
│      frame = {...observation, "task": task}                                  │
│      eval_dataset.add_frame(frame)   ← 立即添加，无 success 标记              │
│                                                                             │
│  if success:                                                                 │
│      save_episode()          ← 只保存成功                                    │
│  else:                                                                       │
│      clear_episode_buffer()  ← 丢弃失败！❌                                   │
└─────────────────────────────────────────────────────────────────────────────┘

修改后:
┌─────────────────────────────────────────────────────────────────────────────┐
│  episode_frames_buffer = []   ← 新增：帧缓存                                 │
│                                                                             │
│  for step in episode:                                                        │
│      frame = {...observation, "task": task}                                  │
│      episode_frames_buffer.append(frame)  ← 缓存帧                           │
│                                                                             │
│  # Episode 结束，确定最终成功状态                                             │
│  final_success = success_flag                                                │
│  success_value = np.array([1.0 if final_success else 0.0])                   │
│                                                                             │
│  # 用最终状态统一标记所有帧                                                   │
│  for frame in episode_frames_buffer:                                         │
│      frame["episode_success"] = success_value  ← 统一标记                    │
│      eval_dataset.add_frame(frame)                                           │
│                                                                             │
│  save_episode()  ← 保存成功和失败                                            │
│  episode_frames_buffer = []  ← 清空缓存                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.3 具体修改点

#### 修改点 1: 添加 episode_success 到 features

**文件**: `scripts/utils/evaluation.py`
**函数**: `run_evaluation_loop()`
**位置**: 约 Line 69-97 (创建数据集 features 后)

**找到这段代码**:
```python
if args.save_datasets:
    features = None
    if args.dataset_root and Path(args.dataset_root).exists():
        source_dataset = LeRobotDataset(repo_id="collected_dataset", root=Path(args.dataset_root))
        features = dict(source_dataset.meta.features)
        fps = source_dataset.fps
    else:
        fps = 30
        # ... features 定义 ...
        for key in image_keys:
            features[f"observation.images.{key}"] = {...}
```

**在这段代码后面添加**:
```python
    # 添加 episode_success 字段（用于价值函数训练）
    features["episode_success"] = {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
    }
```

---

#### 修改点 2: 添加 episode 帧缓存变量

**文件**: `scripts/utils/evaluation.py`
**函数**: `run_evaluation_loop()`
**位置**: 约 Line 100-103 (初始化变量处)

**找到这段代码**:
```python
    all_episode_metrics = []
    logger.info(f"Starting evaluation: {args.num_episodes} episodes")
    rate_limiter = RateLimiter(args.step_hz)
```

**改为**:
```python
    all_episode_metrics = []
    logger.info(f"Starting evaluation: {args.num_episodes} episodes")
    rate_limiter = RateLimiter(args.step_hz)

    # 缓存 episode 帧，用于最后统一标记 success
    episode_frames_buffer = []
```

---

#### 修改点 3: 修改帧录制逻辑（缓存而非立即添加）

**文件**: `scripts/utils/evaluation.py`
**函数**: `run_evaluation_loop()`
**位置**: 约 Line 183-190 (录制帧处)

**找到这段代码**:
```python
            # Recording
            if args.save_datasets:
                frame = {
                    k: v
                    for k, v in observation_dict.items()
                    if k != "observation.top_depth"
                }
                frame["task"] = args.task_description
                eval_dataset.add_frame(frame)
```

**改为**:
```python
            # Recording - 缓存帧，等 episode 结束后统一标记 success
            if args.save_datasets:
                frame = {
                    k: v
                    for k, v in observation_dict.items()
                    if k != "observation.top_depth"
                }
                frame["task"] = args.task_description
                episode_frames_buffer.append(frame)  # 缓存而不是直接 add_frame
```

---

#### 修改点 4: 修改保存逻辑（统一标记并保存成功+失败）

**文件**: `scripts/utils/evaluation.py`
**函数**: `run_evaluation_loop()`
**位置**: 约 Line 202-234 (episode 结束处理处)

**找到这段代码**:
```python
        # --- End of Episode Handling ---
        is_success = success.item() if success_flag else False

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
                eval_dataset.clear_episode_buffer()
```

**改为**:
```python
        # --- End of Episode Handling ---
        is_success = success.item() if success_flag else False

        # Save Datasets - 保存成功和失败数据
        if args.save_datasets:
            # 用最终成功状态统一标记所有帧
            success_value = np.array([1.0 if is_success else 0.0], dtype=np.float32)

            for frame in episode_frames_buffer:
                frame["episode_success"] = success_value
                eval_dataset.add_frame(frame)

            # 保存 episode（无论成功或失败）
            eval_dataset.save_episode()
            append_episode_initial_pose(
                json_path,
                episode_index,
                object_initial_pose,
                garment_name=garment_name,
            )
            episode_index += 1

            # 清空缓存，准备下一个 episode
            episode_frames_buffer = []

            # 日志
            status = "SUCCESS" if is_success else "FAILURE"
            logger.info(f"Episode {episode_index} saved with status: {status}")
```

### 7.4 需要添加的 import

确保文件顶部有：
```python
import numpy as np
```

### 7.5 对 SmolVLA 训练的影响

**结论: 不影响 SmolVLA 训练**

原因:
1. SmolVLA 的 `forward()` 只访问配置中定义的 features (`input_features`, `output_features`)
2. `episode_success` 是额外的 feature，会被自动忽略
3. LeRobot 数据集设计支持任意额外的 features

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  数据集 batch 包含:                                                          │
│  {                                                                          │
│    "observation.state": ...,       ◄── SmolVLA 使用                        │
│    "observation.images.*": ...,    ◄── SmolVLA 使用                        │
│    "action": ...,                  ◄── SmolVLA 使用                        │
│    "episode_success": [1.0],       ◄── 额外 feature，自动忽略               │
│    ...                                                                      │
│  }                                                                          │
│                                                                             │
│  SmolVLA 只访问配置中定义的 keys，额外的 episode_success 被忽略              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.6 数据集统计信息

**好消息: 数据集会自动计算所有需要的统计信息**

当 `save_episode()` 被调用时，LeRobot 会自动计算:
- `mean`, `std` → SmolVLA 使用 (MEAN_STD 归一化)
- `q01`, `q10`, `q50`, `q90`, `q99` → pistar06 使用 (QUANTILES 归一化)

这些统计信息存储在 `meta/stats.json` 中。

### 7.7 验证方法

修改完成后，运行评估：

```bash
python -m scripts.eval \
  --policy_type lerobot \
  --policy_path outputs/train/smolvla_top_long/checkpoints/last/pretrained_model \
  --garment_type "top_long" \
  --dataset_root Datasets/example/top_long_merged \
  --save_datasets \
  --eval_dataset_path Datasets/eval/top_long \
  --num_episodes 10 \
  --enable_cameras \
  --device cpu
```

**检查结果**:

```bash
# 1. 检查 features 是否包含 episode_success
cat Datasets/eval/top_long/001/meta/info.json | grep -A3 episode_success

# 2. 检查是否有成功和失败数据
ls Datasets/eval/top_long/001/meta/episodes/

# 3. 检查 stats.json 是否包含 episode_success 的统计
cat Datasets/eval/top_long/001/meta/stats.json | grep -A10 episode_success
```

---

## 8. 待确认事项

1. **数据收集顺序**: 先收集失败数据，还是先用全成功数据验证流程？
2. **服装类型**: 从哪个类型开始？(top_long / pant_long / etc.)
3. **评估规模**: 每个类型收集多少成功/失败数据？
4. **计算资源**: 价值函数训练使用几卡？

---

## 9. 参考资料

- Evo-RL 仓库: `third_party/Evo-RL/`
- pistar06 论文: https://www.pi.website/blog/pistar06
- LeRobot 文档: https://huggingface.co/docs/lerobot
