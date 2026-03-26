# Evo-RL完整组件清单与实施指南

> **创建日期**: 2026-03-19
> **主题**: 使用Pistar06在LeHome上实施Evo-RL

---

## 目录

1. [为什么叫Evo-RL？](#为什么叫evo-rl)
2. [核心组件清单](#核心组件清单)
3. [HIL实施要点](#hil实施要点)
4. [完整工作流程](#完整工作流程)
5. [潜在问题与解决方案](#潜在问题与解决方案)

---

## 为什么叫Evo-RL？

### Evo-RL ≠ 传统RL

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    传统RL vs Evo-RL                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  传统RL (如PPO, SAC):                                                    │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  Environment ──→ Agent ──→ Action ──→ Reward ──→ Update     │        │
│  │       ↑                                                    │        │
│  │       └────────────────────────────────────────────────────  │        │
│  │                                                             │        │
│  │  • 需要环境reward signal                                    │        │
│  │  • 需要大量环境交互                                         │        │
│  │  • 样本效率低                                               │        │
│  │  • 难以处理sparse reward                                   │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                                                                           │
│  Evo-RL:                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  Human Labels ──→ Value Function ──→ ACP ──→ Policy        │        │
│  │       ↑                                                    │        │
│  │       └─ Human Intervention Loop (HIL)                     │        │
│  │                                                             │        │
│  │  • 使用人类标签 (success/failure)                           │        │
│  │  • 不需要环境reward                                         │        │
│  │  • 基于已有的demonstration数据                              │        │
│  │  • 样本效率高                                               │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### 为什么叫RL？

**Evo-RL的"RL"来自价值函数的学习机制**，而非环境交互：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Evo-RL中的RL元素                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  1. Value Function Learning (RL核心)                                     │
│     ┌──────────────────────────────────────────────────────────────┐    │
│     │  状态 s → 价值 V(s)                                            │    │
│     │                                                               │    │
│     │  训练目标: 最小化价值预测误差                                   │    │
│     │  Loss = KL(V_pred(s), V_target(s))                            │    │
│     │                                                               │    │
│     │  这本质上是RL中的value learning!                               │    │
│     └──────────────────────────────────────────────────────────────┘    │
│                                                                           │
│  2. Advantage Calculation (RL核心)                                      │
│     ┌──────────────────────────────────────────────────────────────┐    │
│     │  A(s,a) = R(s,a) + γV(s') - V(s)                              │    │
│     │                                                               │    │
│     │  这里用n-step return作为R，                                    │    │
│     │  用Pistar06预测V(s)                                           │    │
│     │                                                               │    │
│     │  这就是RL中的advantage estimation!                            │    │
│     └──────────────────────────────────────────────────────────────┘    │
│                                                                           │
│  3. Policy Improvement with ACP (RL核心)                                │
│     ┌──────────────────────────────────────────────────────────────┐    │
│     │  条件策略: π(a|s, advantage="positive/negative")               │    │
│     │                                                               │    │
│     │  通过advantage标签指导策略学习，                                │    │
│     │  类似于Actor-Critic中的advantage function                     │    │
│     └──────────────────────────────────────────────────────────────┘    │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### RL的具体体现

| RL概念 | Evo-RL中的对应 | 说明 |
|--------|---------------|------|
| **State Value V(s)** | Pistar06输出 | 预测状态的长期价值 |
| **Advantage A(s,a)** | n-step advantage | 实际收益vs价值预测的差异 |
| **Return** | 成功轨迹的负长度 | 成功越快价值越高 |
| **Policy Gradient** | 通过ACP条件化 | 根据advantage调整策略 |
| **Critic** | Pistar06价值函数 | 评估状态价值 |
| **Actor** | 策略模型 | 生成动作 |

### 与传统RL的关键区别

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 传统RL vs Evo-RL的数据流                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  传统PPO:                                                                │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  State → Policy → Action → Environment → Reward → Update    │        │
│  │    ↓                                                        │        │
│  │  Value Function learns from Environment Rewards             │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                                                                           │
│  Evo-RL:                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  Human Labels → Value Function → Advantage → ACP → Policy  │        │
│  │    ↓                                                        │        │
│  │  Value Function learns from Human Success/Failure Labels   │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                                                                           │
│  关键: Evo-RL用Human Feedback替代Environment Reward!                     │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 核心组件清单

### 必需组件

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Evo-RL必需组件                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  1. Pistar06价值函数                                                     │
│     ┌──────────────────────────────────────────────────────────────┐    │
│     │  位置: third_party/Evo-RL/src/lerobot/values/pistar06/      │    │
│     │  功能: 预测状态价值 V(s)                                       │    │
│     │  输入: 图像 + 任务文本 + state                                 │    │
│     │  输出: 201 bins的价值分布 [-1, 0]                             │    │
│     │  预训练: SigLIP (vision) + Gemma 3 270M (language)            │    │
│     └──────────────────────────────────────────────────────────────┘    │
│                                                                           │
│  2. 价值目标计算                                                         │
│     ┌──────────────────────────────────────────────────────────────┐    │
│     │  位置: modeling_pistar06.py:85-122                          │    │
│     │  函数: compute_normalized_value_targets()                   │    │
│     │  输入: episode_info (success, length), task_max_lengths      │    │
│     │  输出: 每帧的价值目标 [-1, 0]                                  │    │
│     │                                                               │    │
│     │  公式:                                                        │    │
│     │  g = -(remaining_steps)                                      │    │
│     │  if failure: g -= c_fail * max_length                        │    │
│     │  target = g / (max_length + c_fail * max_length)             │    │
│     └──────────────────────────────────────────────────────────────┘    │
│                                                                           │
│  3. ACP标签生成                                                          │
│     ┌──────────────────────────────────────────────────────────────┐    │
│     │  位置: third_party/Evo-RL/src/lerobot/scripts/               │    │
│     │        lerobot_value_infer.py                                 │    │
│     │  功能: 计算advantage并生成ACP标签                             │    │
│     │  输入: Pistar06价值预测, episode info                         │    │
│     │  输出: complementary_info.acp_indicator (0/1)                 │    │
│     │                                                               │    │
│     │  步骤:                                                        │    │
│     │  1. 计算n-step return: R = Σ(从t到t+n-1的reward)              │    │
│     │     (这里reward是成功=0, 失败=-1, 实际用负长度近似)             │    │
│     │  2. 计算advantage: A = R - V(s)                              │    │
│     │  3. 按分位数二值化: top 30% → 1, others → 0                   │    │
│     │  4. 可选: 强制干预帧为positive                                 │    │
│     └──────────────────────────────────────────────────────────────┘    │
│                                                                           │
│  4. ACP训练Hook                                                          │
│     ┌──────────────────────────────────────────────────────────────┐    │
│     │  位置: third_party/Evo-RL/src/lerobot/rl/acp_hook.py        │    │
│     │  功能: 在训练时将ACP标签注入到任务文本中                       │    │
│     │  输入: batch, acp_indicator字段                               │    │
│     │  输出: 修改后的batch["task"]                                  │    │
│     │                                                               │    │
│     │  格式: "Fold the shirt\nAdvantage: positive/negative"        │    │
│     │                                                               │    │
│     │  特性:                                                        │    │
│     │  - 随机dropout 30%标签 (学习无标签情况)                       │    │
│     │  - 支持CFG (Classifier-Free Guidance)                        │    │
│     └──────────────────────────────────────────────────────────────┘    │
│                                                                           │
│  5. 训练脚本                                                             │
│     ┌──────────────────────────────────────────────────────────────┐    │
│     │  价值训练: lerobot-value-train                               │    │
│     │  价值推理: lerobot-value-infer                               │    │
│     │  策略训练: lerobot-train (with --acp.enable)                 │    │
│     └──────────────────────────────────────────────────────────────┘    │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### 可选但推荐组件

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 HIL相关组件 (可选但推荐)                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  1. HIL数据收集                                                          │
│     ┌──────────────────────────────────────────────────────────────┐    │
│     │  脚本: lerobot-human-inloop-record                            │    │
│     │  功能: 策略执行时允许人工干预                                   │    │
│     │  快捷键:                                                      │    │
│     │    - 'i': 切换干预模式                                        │    │
│     │    - 's': 标记成功                                           │    │
│     │    - 'f': 标记失败                                           │    │
│     │                                                               │    │
│     │  收集的数据:                                                  │    │
│     │    - action: 执行的动作 (策略或人工)                          │    │
│     │    - complementary_info.policy_action: 策略动作                │    │
│     │    - complementary_info.is_intervention: 是否干预              │    │
│     │    - episode_success: episode结果                             │    │
│     └──────────────────────────────────────────────────────────────┘    │
│                                                                           │
│  2. 数据分析工具                                                          │
│     ┌──────────────────────────────────────────────────────────────┐    │
│     │  脚本: lerobot-dataset-report                                │    │
│     │  功能: 分析HIL数据集                                          │    │
│     │  输出:                                                       │    │
│     │    - Success metrics (成功率)                                 │    │
│     │    - Intervention metrics (干预率)                            │    │
│     │    - Episode长度分布                                          │    │
│     └──────────────────────────────────────────────────────────────┘    │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## HIL实施要点

### 数据格式要求

```python
# HIL数据集必须包含的字段

# Episode级别元数据
{
    "episode_index": int,
    "length": int,
    "tasks": ["Fold the shirt"],
    "episode_success": "success" or "failure",  # 必需！
}

# Frame级别数据
{
    "observation.images.top_rgb": [...],
    "observation.state": [...],
    "action": [...],  # 实际执行的动作

    # HIL特定字段
    "complementary_info.policy_action": [...],     # 策略建议的动作
    "complementary_info.is_intervention": 0.0 or 1.0,  # 是否人工干预
    "complementary_info.state": 0.0 or 1.0 or 2.0,   # S0/S1/S2状态
    "complementary_info.collector_policy_id": "human" or "policy_id",

    # ACP字段 (价值推理后添加)
    "complementary_info.acp_indicator": 0.0 or 1.0,  # positive/negative
}
```

### LeHome特定的HIL实现

#### 选项1: 修改现有评估脚本（推荐开始）

```python
# 修改 scripts/utils/evaluation.py

# 当前: 只保存成功数据
if args.save_datasets:
    if success_flag:
        eval_dataset.save_episode()
    else:
        eval_dataset.clear_episode_buffer()  # ❌ 丢弃失败

# 修改后: 保存所有数据
if args.save_datasets:
    # 添加episode_success字段
    success_value = np.array([1.0 if success_flag else 0.0], dtype=np.float32)

    for frame_data in eval_dataset._episode_buffer:
        frame_data["episode_success"] = success_value

    eval_dataset.save_episode()
```

**优点**: 简单，立即可用
**缺点**: 没有人工干预信息

#### 选项2: 键盘控制HIL（需要实现）

```python
# 新建 scripts/hil_eval.py

# 实现键盘控制的HIL评估
# - 策略自动执行
# - 按'i'人工接管
# - 方向键控制机器人
# - 按's'/'f'标记成功/失败

# 优点: 完整的HIL数据
# 缺点: 需要较多实现工作
```

### 数据收集策略

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  Evo-RL数据收集策略                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Phase 1: 基础数据 (自动收集)                                             │
│  ─────────────────────────────                                         │
│  • 运行现有评估脚本                                                     │
│  • 保存成功 + 失败episode                                               │
│  • 目标: 每个garment类型 50 episodes                                   │
│  • 时间: 2-3小时                                                       │
│                                                                           │
│  Phase 2: HIL数据 (人工干预)                                             │
│  ─────────────────────────────                                         │
│  • 使用HIL评估脚本                                                     │
│  • 针对失败案例人工纠正                                                │
│  • 目标: 每个garment类型 20-30 HIL episodes                           │
│  • 时间: 4-6小时                                                       │
│                                                                           │
│  Phase 3: 迭代优化                                                      │
│  ────────────────────                                                  │
│  • 训练改进的Expert                                                    │
│  • 再次HIL收集                                                        │
│  • 持续改进                                                           │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 完整工作流程

### 使用Pistar06的Evo-RL流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│          LeHome + Pistar06 Evo-RL 完整流程                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ Phase 1: 数据收集                                                 │    │
│  ├─────────────────────────────────────────────────────────────────┤    │
│  │                                                                 │    │
│  │ # Step 1.1: 修改评估脚本保存失败数据                              │    │
│  │ # 修改文件: scripts/utils/evaluation.py:206-217                  │    │
│  │                                                                 │    │
│  │ # Step 1.2: 运行评估收集数据                                     │    │
│  │ python -m scripts.eval \                                         │    │
│  │   --policy_type moe_smolvla \                                    │    │
│  │   --policy_path outputs/moe_train/.../pant_short/checkpoints/last\│    │
│  │   --garment_type pant_short \                                    │    │
│  │   --num_episodes 50 \                                           │    │
│  │   --save_datasets \                                              │    │
│  │   --eval_dataset_path Datasets/eval_with_failures/pant_short    │    │
│  │                                                                 │    │
│  │ # Step 1.3: 验证数据                                            │    │
│  │ lerobot-dataset-report Datasets/eval_with_failures/pant_short   │    │
│  │                                                                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              ↓                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ Phase 2: 价值函数训练                                             │    │
│  ├─────────────────────────────────────────────────────────────────┤    │
│  │                                                                 │    │
│  │ # 使用Evo-RL的lerobot-value-train                               │    │
│  │ lerobot-value-train \                                          │    │
│  │   --dataset.repo_id=lehome_eval_pant_short \                    │    │
│  │   --dataset.root=Datasets/eval_with_failures/pant_short \       │    │
│  │   --value.type=pistar06 \                                       │    │
│  │   --value.vision_repo_id=google/siglip-so400m-patch14-384 \     │    │
│  │   --value.language_repo_id=google/gemma-3-270m \                │    │
│  │   --targets.success_field=episode_success \                      │    │
│  │   --targets.default_success=failure \                           │    │
│  │   --targets.c_fail_coef=1.0 \                                   │    │
│  │   --batch_size=64 \                                             │    │
│  │   --steps=10000 \                                               │    │
│  │   --output_dir=outputs/value_train/pistars06_pant_short         │    │
│  │                                                                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              ↓                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ Phase 3: ACP标签生成                                              │    │
│  ├─────────────────────────────────────────────────────────────────┤    │
│  │                                                                 │    │
│  │ # 使用Evo-RL的lerobot-value-infer                               │    │
│  │ lerobot-value-infer \                                          │    │
│  │   --dataset.repo_id=lehome_eval_pant_short \                    │    │
│  │   --dataset.root=Datasets/eval_with_failures/pant_short \       │    │
│  │   --inference.checkpoint_path=outputs/value_train/pistars06_pant_short/checkpoints/best \│    │
│  │   --acp.enable=true \                                           │    │
│  │   --acp.n_step=50 \                                             │    │
│  │   --acp.positive_ratio=0.3 \                                    │    │
│  │   --output_dir=outputs/acp_inference/pant_short                 │    │
│  │                                                                 │    │
│  │ # 注意: 没有is_intervention字段，所以force_intervention=false    │    │
│  │                                                                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              ↓                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ Phase 4: 用ACP训练Expert                                         │    │
│  ├─────────────────────────────────────────────────────────────────┤    │
│  │                                                                 │    │
│  │ # 使用lerobot-train with ACP                                    │    │
│  │ lerobot-train \                                                │    │
│  │   --dataset.repo_id=lehome_eval_pant_short_acp \                 │    │
│  │   --dataset.root=Datasets/eval_with_failures/pant_short \       │    │
│  │   --policy.type=smolvla \                                      │    │
│  │   --policy.pretrained_path=outputs/moe_train/.../pant_short/... \│    │
│  │   --acp.enable=true \                                           │    │
│  │   --acp.indicator_field=complementary_info.acp_indicator \       │    │
│  │   --acp.indicator_dropout_prob=0.3 \                            │    │
│  │   --batch_size=16 \                                             │    │
│  │   --steps=15000 \                                               │    │
│  │   --output_dir=outputs/moe_train_v2/acp_improved_pant_short      │    │
│  │                                                                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              ↓                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ Phase 5: 评估与迭代                                               │    │
│  ├─────────────────────────────────────────────────────────────────┤    │
│  │                                                                 │    │
│  │ # 评估改进后的Expert                                             │    │
│  │ python -m scripts.eval \                                        │    │
│  │   --policy_path outputs/moe_train_v2/acp_improved_pant_short/... \│    │
│  │   --garment_type pant_short \                                   │    │
│  │   --num_episodes 20                                            │    │
│  │                                                                 │    │
│  │ # 如果性能提升显著，回到Phase 1继续迭代                         │    │
│  │ # 如果性能不足，考虑HIL数据收集                                 │    │
│  │                                                                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 潜在问题与解决方案

### 问题1: 数据集格式不匹配

**问题**: LeRobot数据集可能缺少`episode_success`字段

**解决方案**:
```python
# 在评估脚本中添加
for frame_data in episode_buffer:
    if success_flag:
        frame_data["episode_success"] = np.array([1.0])
    else:
        frame_data["episode_success"] = np.array([0.0])
```

### 问题2: Pistar06输入格式

**问题**: Pistar06需要特定格式的输入

**要求**:
```python
# Pistar06需要的输入
{
    "observation.pistar06.images": [B, N, C, H, W],  # 多相机图像
    "observation.pistar06.image_attention_mask": [B, N],  # 相机mask
    "observation.language_tokens": [B, T],  # Tokenized task text
    "observation.language_attention_mask": [B, T],  # Text mask
    "observation.value_target": [B],  # 价值目标 (训练时)
}
```

**解决方案**: 使用Evo-RL的processor
```python
from lerobot.values.pistar06.processor_pistar06 import Pistar06PolicyProcessorPipeline

processor = Pistar06PolicyProcessorProcess(
    cfg=pistar06_config,
    dataset_meta=dataset_meta,
)
processed_batch = processor(raw_batch)
```

### 问题3: Pistar06与SmolVLA的特征不匹配

**问题**: Pistar06的SigLIP特征与SmolVLA的SmolVLM特征不同

**影响**:
- ✅ 如果只训练Pistar06，没关系
- ⚠️ 如果想共享特征，需要额外处理

**建议**:
- **短期**: 完全独立训练Pistar06
- **长期**: 考虑将Pistar06的vision encoder替换为SmolVLM vision

### 问题4: 计算资源

**问题**: Pistar06训练需要GPU

**要求**:
```
价值训练:
  - GPU: 推荐 (V100/A100)
  - 显存: ~8GB (batch_size=64)
  - 时间: ~2-3小时 (10K steps)

价值推理:
  - GPU: 可选 (CPU也可以，但慢)
  - 显存: ~4GB
  - 时间: ~30分钟 (1K episodes)

ACP训练:
  - GPU: 必需 (与原策略训练相同)
  - 显存: 与原策略相同
  - 时间: 与原策略训练相同
```

**CPU优化**:
```bash
# 如果只有CPU，可以：
# 1. 使用更小的batch_size
# 2. 使用更少的bins (num_bins=101)
# 3. 只在GPU上推理，CPU上训练
```

### 问题5: ACP与SmolVLA的集成

**问题**: SmolVLA可能不支持ACP

**检查**:
```python
# 查看SmolVLA是否支持task conditioning
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# 检查是否使用task text
policy = SmolVLAPolicy(...)
# SmolVLA应该支持task输入，因为它是VLA模型
```

**如果不支持**:
```python
# 需要修改SmolVLA的forward方法
# 在task文本中添加ACP标签

def modify_task_with_acp(task_text, acp_indicator):
    if acp_indicator == 1:
        return f"{task_text}\nAdvantage: positive"
    elif acp_indicator == 0:
        return f"{task_text}\nAdvantage: negative"
    else:
        return task_text  # dropout
```

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

## 总结

### 为什么选择Pistar06？

✅ **已验证**: Evo-RL团队已测试
✅ **成熟**: 完整的训练和推理流程
✅ **兼容**: 与Evo-RL生态系统完全兼容
✅ **文档**: 丰富的使用案例

### Evo-RL的核心价值

1. **Human-in-the-Loop**: 收集策略失败数据
2. **Value Learning**: 学习状态价值
3. **Advantage Calculation**: 计算advantage
4. **Conditional Policy**: 通过ACP条件化策略

### 与LeHome的集成路径

```
LeHome评估 → 保存失败数据 → Pistar06训练 → ACP生成 → Expert改进 → 重复
```

这就是使用Evo-RL + Pistar06在LeHome上实施价值函数引导策略改进的完整方案！

---

*创建于2026-03-19*
