# LeHome Evo-RL 完整流程指南

> **创建日期**: 2026-03-19
> **目标**: 从官方数据集开始，完整跑通Evo-RL流程

---

## 概览

Evo-RL 是一个基于价值函数引导的策略改进框架，核心思想是：
1. **收集成功+失败混合数据**
2. **训练价值函数区分成功/失败**
3. **生成ACP (Advantage-Conditioned Policy) 标签**
4. **用ACP标签训练改进的策略**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LeHome + Evo-RL 完整流程                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 1: 数据收集 → 保存成功+失败数据                                       │
│      ↓                                                                      │
│  Phase 2: 价值函数训练 → pistar06模型                                        │
│      ↓                                                                      │
│  Phase 3: ACP标签生成 → 添加acp_indicator字段                                 │
│      ↓                                                                      │
│  Phase 4: ACP策略训练 → 改进的SmolVLA/MoE                                    │
│      ↓                                                                      │
│  Phase 5: 评估与迭代 → 性能提升则继续                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: 数据收集

### 1.1 修改评估脚本保存失败数据

**文件**: `scripts/utils/evaluation.py`

**状态**: ✅ 已实施 (2026-03-19)
**更新**: ✅ 已修复 finalize() 和 task 字段格式问题 (2026-03-19)
**最终修复**: ✅ 已修复 task 字段为字符串格式 (2026-03-19)

#### 修改点 1: 添加 episode_success 到 features 定义

**位置**: 第51-93行

**原代码**:
```python
if args.dataset_root and Path(args.dataset_root).exists():
    source_dataset = LeRobotDataset(repo_id="collected_dataset", root=Path(args.dataset_root))
    features = dict(source_dataset.meta.features)
    fps = source_dataset.fps
else:
    # ... 创建默认 features ...
    features = {
        "observation.state": {...},
        "action": {...},
    }
```

**修改后**:
```python
if args.dataset_root and Path(args.dataset_root).exists():
    source_dataset = LeRobotDataset(repo_id="collected_dataset", root=Path(args.dataset_root))
    features = dict(source_dataset.meta.features)
    fps = source_dataset.fps

    # ✅ 新增：确保包含 episode_success 字段
    if "episode_success" not in features:
        features["episode_success"] = {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        }
    # 注意：task 是 LeRobot 的特殊字段，不在 features 中定义
else:
    # ... 创建默认 features ...
    features = {
        "observation.state": {...},
        "action": {...},
        # ✅ 新增：episode_success 字段
        "episode_success": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
        # 注意：task 是 LeRobot 的特殊字段，不在 features 中定义
    }
```

**重要说明**:
- `episode_success`: 常规 feature，需要在 features 中定义
- `task`: **LeRobot 特殊字段**，不需要在 features 中定义！它会在 `save_episode()` 时自动从 episode_buffer 中提取并保存到 meta/tasks.parquet

#### 修改点 2: 修改保存逻辑保存所有 episodes

**位置**: 第205-217行

**原代码**:
```python
# Save Datasets
if args.save_datasets:
    if success_flag:
        eval_dataset.save_episode()
        append_episode_initial_pose(...)
        episode_index += 1
    else:
        eval_dataset.clear_episode_buffer()  # ❌ 丢弃失败数据
```

**修改后**:
```python
# Save Datasets
if args.save_datasets:
    # ✅ 在保存前，为所有帧添加 episode_success 标签
    num_frames = eval_dataset.episode_buffer.get("size", 0)
    success_value = np.array([1.0 if is_success else 0.0], dtype=np.float32)

    # 为 episode_buffer 中的所有帧添加 episode_success
    if "episode_success" not in eval_dataset.episode_buffer:
        eval_dataset.episode_buffer["episode_success"] = []
    eval_dataset.episode_buffer["episode_success"].extend([success_value] * num_frames)

    # ✅ 保存所有 episode（成功和失败都保存）
    eval_dataset.save_episode()
    append_episode_initial_pose(
        json_path,
        episode_index,
        object_initial_pose,
        garment_name=garment_name,
    )
    episode_index += 1
    # ❌ 删除了 else 分支，不再丢弃失败数据
```

#### 修改点 3: 添加 finalize() 调用

**位置**: 第256行 (evaluation.py 结尾处)

**原代码**:
```python
    return all_episode_metrics
```

**修改后**:
```python
    # Finalize dataset to flush metadata buffers and close writers
    if args.save_datasets and eval_dataset is not None:
        logger.info("Finalizing dataset...")
        eval_dataset.finalize()

    return all_episode_metrics
```

**重要说明**: `finalize()` 调用至关重要！它会：
- 刷新 metadata buffer，将 episodes 写入 `meta/episodes/` 目录
- 关闭 parquet writers，写入 footer 元数据
- 没有 `finalize()`，数据集无法被 LeRobotDataset 加载

#### 重要说明

1. **task 字段格式**: 第189行设置 `frame["task"] = args.task_description`，必须是**字符串**格式
   - ❌ 错误: `frame["task"] = [args.task_description]` (列表)
   - ✅ 正确: `frame["task"] = args.task_description` (字符串)

2. **兼容性保证**:
   - ✅ 不影响现有训练（新字段被未配置的脚本忽略）
   - ✅ 现有数据集仍可正常使用
   - ✅ 只有使用新数据集 + 明确配置才会使用新字段

3. **数据格式**:
   ```python
   # Episode 级别
   {
       "episode_index": 0,
       "task": ["Fold the pant_long"],
       "task_index": [0],  # 从官方数据集继承
   }

   # Frame 级别
   {
       "observation.state": [...],
       "observation.images.top_rgb": [...],
       "action": [...],
       "task": "Fold the pant_long",
       "episode_success": [1.0],  # 1.0=成功, 0.0=失败
   }
   ```

### 为什么这样修改可以工作

#### Task 字段的数据流

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Task 字段在 LeRobot 中的完整流程                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. 数据收集 (evaluation.py:189)                                             │
│     frame["task"] = "Fold the garment"  # ✅ 字符串格式                       │
│                                                                             │
│  2. add_frame() 调用 (lerobot_dataset.py:1170)                              │
│     self.episode_buffer["task"].append(frame.pop("task"))                   │
│     # episode_buffer["task"] = ["Fold the garment", "Fold the garment", ...] │
│                                                                             │
│  3. save_episode() 调用 (lerobot_dataset.py:1221)                           │
│     tasks = episode_buffer.pop("task")  # 提取任务列表                       │
│     episode_tasks = list(set(tasks))   # ✅ 去重: ["Fold the garment"]       │
│     # 保存到 meta/tasks.parquet                                             │
│                                                                             │
│  4. finalize() 调用 (lerobot_dataset.py:1111)                               │
│     # 刷新 metadata，写入 parquet footer                                    │
│                                                                             │
│  5. __getitem__() 加载 (lerobot_dataset.py:1089-1091)                       │
│     task_idx = item["task_index"].item()                                   │
│     item["task"] = self.meta.tasks.iloc[task_idx].name  # 自动加载 task     │
│                                                                             │
│  6. Value Training 使用 (lerobot_value_train.py:252-253)                   │
│     task_batch = batch[cfg.value.task_field]  # 读取 batch["task"]          │
│     # Pistar06 将 task 用于 prompt 构建                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 关键修复总结

| 修复点 | 问题 | 解决方案 | 影响 |
|-------|------|----------|------|
| **Task 格式** | `frame["task"] = [description]` (列表) | 改为字符串: `frame["task"] = description` | 修复 `unhashable type: 'list'` 错误 |
| **episode_success 字段** | 数据集缺少此字段 | 在 features 中定义 | 价值函数训练所需 |
| **保存失败数据** | 原代码只保存成功 episodes | 移除条件判断，保存所有 episodes | Evo-RL 需要成功+失败数据 |
| **finalize() 调用** | 缺少 finalize() | 在循环结束后调用 | 确保 meta/episodes 正确保存 |
| **Task 特殊字段** | 误认为 task 是常规 feature | 不在 features 中定义 | LeRobot 自动处理 task |

#### Task 在 Evo-RL 中的作用

**为什么 Task 对价值函数很重要：**

1. **Prompt 上下文**: Pistar06 价值模型使用 task 描述构建 prompt
   ```python
   # Pistar06 processor 中:
   prompt = f"{task} {state_description}"
   # 例如: "Fold the garment [12, 45, 67, ...]"
   ```

2. **价值预测**: V(s) 预测 "此状态对于完成该任务有多好"
   - 不同任务 → 不同价值函数
   - 相同任务 + 不同状态 → 不同价值

3. **通用任务策略**: 对于我们的场景：
   - ✅ 使用通用任务: `"Fold the garment"`
   - ✅ 状态和图像提供衣服类型信息
   - ✅ 价值函数学习预测"折叠进度"，与具体衣服类型无关

**如果是特定任务会怎样：**
```python
# 如果使用特定任务:
frame["task"] = "Fold the long_pant"  # 特定于长裤

# 结果:
# - 需要为每种衣服训练单独的价值函数，或
# - 需要更大的价值模型来处理多种任务
```

**我们选择通用任务的原因：**
- 单一价值函数适用于所有衣服类型
- 状态/图像已包含衣服特定信息
- 减少模型复杂度和训练成本

### 1.2 运行评估收集数据

```bash
python -m scripts.eval \
  --policy_type moe_smolvla \
  --num_episodes 50 \
  --save_datasets \
  --eval_dataset_path Datasets/eval_with_failures \
  --enable_cameras \
  --device cpu
```

**输出**: `Datasets/eval_with_failures/` 包含成功和失败的数据

---

## Phase 2: 价值函数训练

### 2.1 Evo-RL核心组件

**位置**: `third_party/Evo-RL/src/lerobot/values/pistar06/`

| 文件 | 功能 |
|------|------|
| `configuration_pistar06.py` | Pistar06配置类 |
| `modeling_pistar06.py` | Pistar06模型实现 |
| `processor_pistar06.py` | 数据预处理 |

**价值目标计算公式** (`modeling_pistar06.py:85-122`):
```python
# 成功轨迹: V[t] = -remaining_steps / (max_length + c_fail)
# 失败轨迹: V[t] = -(remaining_steps + c_fail*max_length) / (max_length + c_fail)

# 例如: max_length=200, c_fail=1.0
# 成功(150步完成): V[0] = -150/400 = -0.375, V[149] = -1/400 = -0.0025, V[150] = 0.0
# 失败(200步未完成): V[0] = -(200+200)/400 = -1.0, V[199] = -201/400 ≈ -0.5
```

### 2.2 训练命令

```bash
lerobot-value-train \
  --dataset.repo_id=lehome_eval \
  --dataset.root=Datasets/eval_with_failures \
  --value.type=pistar06 \
  --value.dtype=bfloat16 \
  --value.camera_features=observation.images.top_rgb \
  --value.camera_features=observation.images.left_rgb \
  --value.camera_features=observation.images.right_rgb \
  --targets.success_field=episode_success \
  --targets.default_success=failure \
  --targets.c_fail_coef=1.0 \
  --batch_size=64 \
  --steps=10000 \
  --output_dir=outputs/value_train/pistar06
```

**输出**: `outputs/value_train/pistar06/checkpoints/best/pretrained_model/`

---

## Phase 3: ACP标签生成

### 3.1 ACP计算原理

**文件**: `third_party/Evo-RL/src/lerobot/scripts/lerobot_value_infer.py`

```
Step 1: 价值函数预测 → predicted_value[t]

Step 2: 计算dense rewards
  reward[t] = target[t+1] - target[t]

Step 3: 计算n-step advantage
  advantage[t] = Σ(reward[t:t+n]) + bootstrap_value - predicted_value[t]

Step 4: 二值化
  threshold = quantile(advantages, 1 - positive_ratio)  # top 30%
  indicator[t] = 1 if advantage[t] >= threshold else 0
```

### 3.2 推理命令

```bash
lerobot-value-infer \
  --dataset.repo_id=lehome_eval \
  --dataset.root=Datasets/eval_with_failures \
  --inference.checkpoint_path=outputs/value_train/pistar06/checkpoints/best \
  --runtime.device=cuda \
  --runtime.batch_size=64 \
  --acp.enable=true \
  --acp.n_step=50 \
  --acp.positive_ratio=0.3 \
  --acp.value_field=complementary_info.value \
  --acp.advantage_field=complementary_info.advantage \
  --acp.indicator_field=complementary_info.acp_indicator \
  --output_dir=outputs/acp_inference
```

**输出**: 数据集添加三个字段:
- `complementary_info.value`: 预测的价值
- `complementary_info.advantage`: 计算的advantage
- `complementary_info.acp_indicator`: 0/1标签

---

## Phase 4: ACP策略训练

### 4.1 ACP Hook实现

**文件**: `third_party/Evo-RL/src/lerobot/rl/acp_hook.py`

```python
# 训练时自动修改task文本
def __call__(self, batch, step):
    for task, is_positive in zip(batch["task"], indicators):
        if dropout and random() < dropout_prob:
            new_task = task  # dropout，不加标签
        else:
            new_task = f"{task}\nAdvantage: {'positive' if is_positive else 'negative'}"
    batch["task"] = new_tasks
    return batch
```

### 4.2 训练命令

```bash
lerobot-train \
  --dataset.repo_id=lehome_eval_acp \
  --dataset.root=Datasets/eval_with_failures \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --acp.enable=true \
  --acp.indicator_field=complementary_info.acp_indicator \
  --acp.indicator_dropout_prob=0.3 \
  --batch_size=16 \
  --steps=30000 \
  --output_dir=outputs/moe_train_v2/acp_improved
```

**关键参数**:
- `acp.enable=true`: 启用ACP
- `acp.indicator_field`: ACP标签字段
- `acp.indicator_dropout_prob=0.3`: 30%概率不加标签（学习无标签情况）

---

## Phase 5: 评估与迭代

### 5.1 评估改进后的策略

```bash
python -m scripts.eval \
  --policy_type moe_smolvla \
  --num_episodes 20 \
  --enable_cameras \
  --device cpu
```

### 5.2 迭代决策

| 成功率提升 | 动作 |
|-----------|------|
| >5% | 继续下一轮迭代 |
| 2-5% | 可选择继续或调整参数 |
| <2% | 停止迭代，分析原因 |

---

## 关键文件清单

### 需要添加到LeRobot的文件

```
third_party/lerobot/src/lerobot/
├── values/
│   └── pistar06/
│       ├── __init__.py
│       ├── configuration_pistar06.py
│       ├── modeling_pistar06.py
│       └── processor_pistar06.py
├── rl/
│   ├── __init__.py
│   ├── acp_hook.py
│   ├── acp_tags.py
│   └── acp_dataset_stats.py
├── scripts/
│   ├── lerobot_value_train.py
│   └── lerobot_value_infer.py
└── configs/
    ├── value.py
    └── value_train.py
```

### 需要修改的文件

```
third_party/lerobot/src/lerobot/configs/train.py
  → 添加 ACPConfig 类

third_party/lerobot/pyproject.toml
  → 添加入口点: lerobot-value-train, lerobot-value-infer
```

---

## 数据格式要求

### Episode级别

```json
{
  "episode_index": 0,
  "length": 200,
  "tasks": ["Fold the shirt"],
  "episode_success": "success"  // 必需！
}
```

### Frame级别

```json
{
  "observation.images.top_rgb": [...],
  "observation.state": [...],
  "action": [...],
  "episode_success": [1.0],  // 1.0=成功, 0.0=失败
  "complementary_info.acp_indicator": [1.0]  // ACP标签（推理后添加）
}
```

---

## 常见问题

1. **Q: 当前数据集没有episode_success字段怎么办？**
   A: Phase 1的评估脚本修改会自动添加

2. **Q: Pistar06和SmolVLA的视觉编码器不同怎么办？**
   A: 短期独立训练Pistar06没问题，长期可以考虑替换视觉编码器

3. **Q: ACP标签的positive_ratio怎么设置？**
   A: 默认0.3（top 30%），可根据数据分布调整

4. **Q: CPU环境能运行吗？**
   A: 价值训练建议用GPU，推理和策略训练可用CPU但会很慢

---

## 预期性能提升

根据Evo-RL论文和设计文档：

| 阶段 | 预期提升 |
|------|----------|
| Stage 1: 数据筛选 | +5-10% |
| Stage 2: Value-Guided | +10-15% |
| Stage 3: 迭代优化 (3轮) | +15-25% |

---

## 快速开始检查清单

### 准备工作

- [ ] 确认有Evo-RL代码 (`third_party/Evo-RL`)
- [ ] 确认有当前Expert checkpoint
- [ ] 确认有评估数据集

### Step 1: 修改评估脚本

- [ ] 编辑 `scripts/utils/evaluation.py`
- [ ] 修改第206-217行，保存失败数据
- [ ] 添加`episode_success`字段

### Step 2: 收集数据

- [ ] 运行评估脚本收集50 episodes
- [ ] 验证数据集包含`episode_success`字段
- [ ] 检查成功/失败比例

### Step 3: 训练价值函数

- [ ] 安装Evo-RL依赖
- [ ] 配置Pistar06训练参数
- [ ] 运行`lerobot-value-train`
- [ ] 验证价值MAE下降

### Step 4: 生成ACP标签

- [ ] 配置ACP推理参数
- [ ] 运行`lerobot-value-infer`
- [ ] 验证ACP标签分布

### Step 5: 训练ACP策略

- [ ] 配置ACP训练参数
- [ ] 运行`lerobot-train` with `--acp.enable`
- [ ] 验证训练损失下降

### Step 6: 评估改进

- [ ] 运行评估脚本测试新策略
- [ ] 对比原始策略和ACP策略的成功率
- [ ] 决定是否继续迭代

---

## 参考资料

- Evo-RL 仓库: `third_party/Evo-RL/`
- Pi*06 论文: https://www.pi.website/blog/pistar06
- LeRobot 文档: https://huggingface.co/docs/lerobot
- 相关设计文档:
  - `docs/evo-rl-integration-plan.md`
  - `docs/evo-rl-components-summary.md`
  - `docs/moe-expert-optimization-with-evo-rl.md`
