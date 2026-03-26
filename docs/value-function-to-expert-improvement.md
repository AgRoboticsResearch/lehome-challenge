# 价值函数如何改进Expert：完整流程

> **创建日期**: 2026-03-19
> **主题**: Pistar06价值函数 → ACP标签 → SmolVLA Expert改进

---

## 目录

1. [核心思想](#核心思想)
2. [价值函数的作用](#价值函数的作用)
3. [ACP标签生成](#acp标签生成)
4. [Expert改进](#expert改进)
5. [完整示例](#完整示例)

---

## 核心思想

```
┌─────────────────────────────────────────────────────────────────────────┐
│              价值函数改进Expert的核心思想                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  传统Imitation Learning:                                                │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  Expert学习: 状态 → 最优动作                                   │        │
│  │  问题: 所有演示数据都被同等对待，不分好坏                       │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                                                                           │
│  Evo-RL (Value-Guided):                                                │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  1. 价值函数: 状态 → 价值 V(s)                                 │        │
│  │  2. Advantage: A = 实际收益 - 预测价值                           │        │
│  │  3. ACP标签: A > threshold → positive, 否则 → negative           │        │
│  │  4. Expert学习: 状态 + 标签 → 动作                             │        │
│  │     • positive样本: 学这些！                                    │        │
│  │     • negative样本: 别这样！                                     │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 价值函数的作用

### 1. 评估状态质量

```python
# Pistar06价值函数的输出
value_function(s) → V(s) ∈ [-1, 0]

# 价值含义
V(s) = 0.0   # 接近成功（好状态）
V(s) = -0.5  # 中等进展
V(s) = -1.0  # 离成功很远（差状态）
```

**问题**：价值函数本身不能直接改进Expert！

### 2. 但价值函数是关键的第一步

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    价值函数的作用                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  价值函数 V(s) 的直接用途:                                              │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  • 评估状态质量: "这个状态好不好？"                              │        │
│  │  • 比较不同状态: "状态A比状态B好吗？"                           │        │
│  │  • 选择动作: 在多个候选动作中，选择价值最高的                  │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                                                                           │
│  但对于改进Expert，我们需要知道:                                          │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  • 这个动作是好是坏？                                          │        │
│  │  • 这条轨迹是高效还是低效？                                    │        │
│  │  • 应该学哪些动作，避免哪些？                                  │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                                                                           │
│  这就需要Advantage！                                                    │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ACP标签生成

### 从价值到Advantage

```python
# Step 1: 计算实际收益（Return）
# 对于成功轨迹，Return ≈ -remaining_steps
# 对于失败轨迹，Return ≈ -(remaining_steps + c_fail * max_length)

# Step 2: 计算Advantage
A(s,a) = Return(s,a) - V(s)

# 例如：
状态s1 (episode开始):
  V(s1) = -0.8  # 价值函数预测
  Return = -0.5  # 实际走了50步，接近成功
  A(s1) = -0.5 - (-0.8) = +0.3  # 比预期好！

状态s2 (episode失败):
  V(s2) = -0.3  # 价值函数预测还不错
  Return = -0.9  # 但实际很快失败
  A(s2) = -0.9 - (-0.3) = -0.6  # 比预期差！
```

### Advantage → ACP标签

```python
# Step 3: 分位数二值化
# 按advantage排序，取top 30%为positive

advantages = [+0.3, +0.1, -0.2, -0.4, -0.6, ...]
sorted_advantages = sorted(advantages, reverse=True)
threshold = sorted_advantages[int(0.3 * len(advantages))]

acp_labels = [
    1 if adv > threshold else 0  # 1=positive, 0=negative
    for adv in advantages
]
```

### 完整流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  从价值到ACP标签的完整流程                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Episode数据:                                                            │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  Frame 0: V=-0.8, remaining=100, success=True                  │        │
│  │  Frame 1: V=-0.75, remaining=99, success=True                  │        │
│  │  ...                                                            │        │
│  │  Frame 50: V=-0.3, remaining=50, success=True                 │        │
│  │  Frame 51: V=-0.3, remaining=49, intervention开始              │        │
│  │  Frame 52: V=-0.2, remaining=48, 人工纠正                    │        │
│  │  ...                                                            │        │
│  │  Frame 100: V=0.0, remaining=0, success=True                   │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                                                                           │
│  计算Return (n-step):                                                    │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  Frame 0: Return = -(100) / 100 = -1.0                        │        │
│  │  Frame 50: Return = -(50) / 100 = -0.5                         │        │
│  │  Frame 51: Return = -(49) / 100 = -0.49                        │        │
│  │  Frame 52: Return = -(48) / 100 = -0.48 (人工纠正，更快)       │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                                                                           │
│  计算Advantage:                                                          │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  Frame 0: A = -1.0 - (-0.8) = -0.2 (比预期差)                 │        │
│  │  Frame 50: A = -0.5 - (-0.3) = -0.2 (比预期差)                │        │
│  │  Frame 52: A = -0.48 - (-0.2) = -0.28 (比预期差)              │        │
│  │  但人工纠正的帧会比较好...                                    │        │
│  │                                                               │        │
│  │  实际上，后半部分（更接近成功）的advantage会更高              │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                                                                           │
│  二值化 (top 30% = positive):                                            │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  Frame 0-40: negative (离成功远)                              │        │
│  │  Frame 41-70: negative (进展中)                               │        │
│  │  Frame 71-100: positive (接近成功！) ← 标记为positive          │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Expert改进

### ACP如何改进Expert

```python
# 原始Expert训练
def original_expert_loss(images, task, state, actions):
    # 所有样本同等对待
    pred_actions = expert(images, task, state)
    loss = flow_matching_loss(pred_actions, actions)
    return loss

# ACP训练
def acp_expert_loss(images, task, state, actions, acp_labels):
    # 根据ACP标签区分对待
    pred_actions = expert(images, task, state, acp_labels)

    # positive样本: 强调学习
    # negative样本: 学习避免
    loss = flow_matching_loss(pred_actions, actions)

    # 可以给positive样本更高权重
    weights = 1.0 + acp_labels  # positive=2.0, negative=1.0
    loss = (loss * weights).mean()

    return loss
```

### 具体改进机制

```
┌─────────────────────────────────────────────────────────────────────────┐
│              ACP如何改进SmolVLA Expert                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Positive样本 (标记为"好"):                                             │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  • 这些是接近成功的状态和动作                                   │        │
│  │  • 模型学习: "在这样的状态下，这样做能成功"                      │        │
│  │  • 作用: 强化正确的执行模式                                     │        │
│  │                                                               │        │
│  │  例如: Frame 80-100 (即将成功)                                 │        │
│  │    - 图像: 衣物已经部分折叠                                    │        │
│  │    - 动作: 精细的调整动作                                      │        │
│  │    - 标签: positive                                           │        │
│  │    - 学习: 重点学习这些关键动作                                │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                                                                           │
│  Negative样本 (标记为"差"):                                             │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  • 这些是离成功远或失败的状态和动作                             │        │
│  │  • 模型学习: "在这样的状态下，别这样做"                         │        │
│  │  • 作用: 避免低效的执行模式                                     │        │
│  │                                                               │        │
│  │  例如: Frame 0-40 (刚开始)                                     │        │
│  │    - 图像: 衣物完全展开                                        │        │
│  │    - 动作: 可能包含一些低效动作                                │        │
│  │    - 标签: negative                                           │        │
│  │    - 学习: 学习避免这些状态下的动作模式                         │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                                                                           │
│  条件学习:                                                               │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  Task文本修改:                                                 │        │
│  │  原始: "Fold the shirt"                                        │        │
│  │  Positive: "Fold the shirt\nAdvantage: positive"              │        │
│  │  Negative: "Fold the shirt\nAdvantage: negative"              │        │
│  │                                                               │        │
│  │  模型学习:                                                     │        │
│  │  • 相同状态 + positive标签 → 输出高效动作                      │        │
│  │  • 相同状态 + negative标签 → 输出不同动作（或避免）              │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 完整示例

### 数据示例

```python
# 假设我们有一个episode
episode = {
    "frames": [
        # Frame 0-30: 探索阶段（低效）
        {"image": img_0, "state": s_0, "action": a_0, "success": True},
        {"image": img_1, "state": s_1, "action": a_1, "success": True},
        ...
        {"image": img_30, "state": s_30, "action": a_30, "success": True},

        # Frame 31-60: 识别并接近目标
        {"image": img_31, "state": s_31, "action": a_31, "success": True},
        ...
        {"image": img_60, "state": s_60, "action": a_60, "success": True},

        # Frame 61-100: 精细调整，最终成功
        {"image": img_61, "state": s_61, "action": a_61, "success": True},
        ...
        {"image": img_100, "state": s_100, "action": a_100, "success": True},
    ],
    "episode_success": True,
    "length": 100
}
```

### Step 1: 价值函数预测

```python
# Pistar06预测每帧的价值
for frame in episode["frames"]:
    frame["value_pred"] = pistar06.predict_value(
        images=frame["image"],
        task="Fold the shirt",
        state=frame["state"]
    )

# 结果：
# Frame 0-30: value ≈ -0.8 ~ -0.6 (刚开始，离成功远)
# Frame 31-60: value ≈ -0.6 ~ -0.3 (进展中)
# Frame 61-100: value ≈ -0.3 ~ 0.0 (接近成功)
```

### Step 2: 计算Advantage

```python
# 计算每帧的advantage
for i, frame in enumerate(episode["frames"]):
    remaining = 100 - i - 1
    return_val = -remaining / 100  # 成功轨迹
    advantage = return_val - frame["value_pred"]
    frame["advantage"] = advantage

# 结果：
# Frame 0-30: advantage ≈ -0.2 ~ 0.0 (一般)
# Frame 31-60: advantage ≈ 0.0 ~ +0.2 (较好)
# Frame 61-100: advantage ≈ +0.2 ~ +0.4 (很好！)
```

### Step 3: 生成ACP标签

```python
# 取top 30%为positive
advantages = [f["advantage"] for f in episode["frames"]]
threshold = np.percentile(advantages, 70)  # top 30%

for frame in episode["frames"]:
    frame["acp_label"] = 1 if frame["advantage"] > threshold else 0

# 结果：
# Frame 0-70: acp_label = 0 (negative)
# Frame 71-100: acp_label = 1 (positive) ← 关键帧！
```

### Step 4: 训练改进的Expert

```python
# 训练时使用ACP标签
for frame in episode["frames"]:
    # 修改任务文本
    if frame["acp_label"] == 1:
        task_text = "Fold the shirt\nAdvantage: positive"
    else:
        task_text = "Fold the shirt\nAdvantage: negative"

    # 训练Expert
    pred_action = smolvla_expert(
        images=frame["image"],
        task=task_text,
        state=frame["state"]
    )

    # 计算损失
    loss = flow_matching_loss(pred_action, frame["action"])

    # 反向传播
    loss.backward()
```

### 效果对比

```
┌─────────────────────────────────────────────────────────────────────────┐
│              训练前后Expert行为对比                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  原始Expert (无ACP):                                                    │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  • 所有动作同等对待                                           │        │
│  │  • 学习平均行为                                               │        │
│  │  • 可能在关键步骤执行不到位                                   │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                                                                           │
│  改进Expert (有ACP):                                                    │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  • Positive样本: 学习关键帧的精准动作                          │        │
│  │  • Negative样本: 学习避免低效的动作                             │        │
│  │  • 条件化学习: 根据advantage调整输出                             │        │
│  │  • 结果: 更接近成功的高效执行                                   │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                                                                           │
│  具体例子:                                                               │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  Frame 80 (接近成功):                                          │        │
│  │    Original action: [0.1, 0.2, 0.3, ...] (可能不够精确)       │        │
│  │    ACP action: [0.15, 0.25, 0.35, ...] (更精确，更有效)       │        │
│  │                                                               │        │
│  │  Frame 20 (刚开始):                                              │        │
│  │    Original action: [0.5, 0.6, ...] (随意动作)               │        │
│  │    ACP action: [0.3, 0.4, ...] (学习更直接接近目标的动作)    │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 实际应用步骤

### 针对你的SmolVLA Expert

```bash
# Step 1: 收集数据（包含成功+失败）
python -m scripts.eval \
  --policy_type moe_smolvla \
  --policy_path outputs/moe_train/smolvla_moe_expert_pant_short/checkpoints/last \
  --garment_type pant_short \
  --num_episodes 50 \
  --save_datasets \
  --eval_dataset_path Datasets/eval_with_failures/pant_short

# Step 2: 训练Pistar06价值函数
lerobot-value-train \
  --dataset.root=Datasets/eval_with_failures/pant_short \
  --value.type=pistar06 \
  --targets.success_field=episode_success \
  --targets.default_success=failure \
  --output_dir=outputs/value_train/pistar06_pant_short \
  --steps=10000

# Step 3: 生成ACP标签
lerobot-value-infer \
  --dataset.root=Datasets/eval_with_failures/pant_short \
  --inference.checkpoint_path=outputs/value_train/pistar06_pant_short/checkpoints/best \
  --acp.enable=true \
  --acp.positive_ratio=0.3 \
  --acp.n_step=50 \
  --output_dir=outputs/acp_inference/pant_short

# Step 4: 用ACP标签训练Expert
lerobot-train \
  --dataset.root=Datasets/eval_with_failures/pant_short \
  --policy.type=smolvla \
  --policy.pretrained_path=outputs/moe_train/smolvla_moe_expert_pant_short/checkpoints/last \
  --acp.enable=true \
  --acp.indicator_field=complementary_info.acp_indicator \
  --acp.indicator_dropout_prob=0.3 \
  --output_dir=outputs/moe_train_acp/improved_pant_short \
  --steps=15000

# Step 5: 评估改进效果
python -m scripts.eval \
  --policy_path outputs/moe_train_acp/improved_pant_short/checkpoints/last \
  --garment_type pant_short \
  --num_episodes 20
```

---

## 总结

### 价值函数的间接作用

```
价值函数本身 → 不直接改进Expert
     ↓
Advantage计算 → 评估动作质量
     ↓
ACP标签生成 → 区分好/坏样本
     ↓
条件训练 → Expert学习区分高效/低效
     ↓
改进的Expert → 更好的性能
```

### 为什么这样有效？

1. **Positive样本**：提供"什么是好的执行"的明确信号
2. **Negative样本**：提供"什么是坏的执行"的明确信号
3. **条件学习**：Expert学习"在不同情况下采取不同策略"
4. **数据效率**：不需要更多数据，只是更好地利用现有数据

### 预期改进

| Expert | 原始成功率 | ACP训练后 | 提升 |
|--------|-----------|-----------|------|
| pant_short | 88% | 93%+ | +5% |
| pant_long | 48% | 63%+ | +15% |
| top_short | 42% | 57%+ | +15% |
| top_long | 73% | 83%+ | +10% |

---

*创建于2026-03-19*
