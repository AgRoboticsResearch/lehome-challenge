# MoE-SmolVLA 设计文档 v2

> **状态**: 实现阶段
> **更新**: 2026-03-17
> **版本说明**:
> - 整合 moe_smolvla_design.md + moe_implementation_plan.md + moe_teacher_forcing_plan.md
> - 修正了实现计划中的架构错误
> - 新增第五节：VLAFlowMatching 架构详解与组件独立性分析（state_proj 共享 vs 独立决策）
> - 新增 5.4.1 节：state_proj 共享假设的实证验证（基于 265K 帧数据分析）
> - 新增第十一节：MoE 完整实现指南（架构图、代码示例、注意事项、验证清单）
> - 新增第十二节：MoE vs 简单脚本调度对比分析（AdaMoE-VLA 论文参考）
> - 新增第十三节：比赛与真机部署考量
> - 新增第十四节：MoE Policy 实现常见坑点与最佳实践

---

## 一、问题背景与动机

### 1.1 比赛约束

- 只能提交**一个模型权重**
- **不能读取文件名**（无法直接知道 garment_type）
- 模型必须从**视觉输入**中自动识别服装类型

### 1.2 联合训练的灾难性失败

| 类型 | 单独训练 | 联合训练 | 下降 |
|------|----------|----------|------|
| pant_short | 88.3% | **1.6%** | -86.7% |
| pant_long | 48.3% | **1.5%** | -46.8% |
| top_long | 73.3% | **7.7%** | -65.6% |
| top_short | 41.7% | **7.4%** | -34.3% |

**根本原因**：
1. **任务长度差异**: pant_short (163帧) vs top_long (332帧) = 2倍
2. **动作模式冲突**: pant（向下折叠，双手协调 r≈0.73）vs top（向内折叠，双手独立 r≈0.31）
3. **梯度干扰**: 不同模式的梯度在共享参数中相互冲击

---

## 二、MoE 方案与核心设计

### 2.1 核心思路

用 **Mixture of Experts** 解决联合训练冲突：

```
输入图像 → Router (视觉分类) → 锁定对应 Expert → 执行该 Expert 的 Flow Matching
```

每个 Expert 只学一种服装类型，彻底消除梯度冲突。

### 2.2 ⚠️ 传统 MoE 在连续控制中的致命问题

**加权组合会导致动作抖动**：

```
Frame t:   Router probs [0.9, 0.1, 0.0, 0.0]
           output = 0.9 × E_0(x) + 0.1 × E_1(x)  → 向下移 5cm ✓

Frame t+1: 机械臂挡住镜头，Router 变不确定
           Router probs [0.5, 0.1, 0.0, 0.4]
           output = 0.5 × E_0(x) + 0.4 × E_3(x)  → 向下移 2.5cm + 被拉偏 ✗
```

**动作幅度被分类概率绑架 → 机械臂抽搐**

### 2.3 ✅ 解决方案：Hard Argmax Sticky Routing

**Episode 开始时一次路由，全程锁定，使用 argmax 而非加权**：

```
Frame 0:   Router → argmax → Expert 2 (top_long) → 🔒 锁定!
Frame 1:   强制使用 Expert 2（不再调用 Router）
Frame 2:   强制使用 Expert 2
...
Frame N:   强制使用 Expert 2
```

**关键**：Hard Argmax（Top-1），不是 Soft Top-K 加权。

---

## 三、Router 可行性实验（已完成）

| 实验 | 准确率 |
|------|--------|
| 冻结 base VLM + 2层MLP + 50 epoch | 82-84% |
| Action Training fine-tune VLM（无 router loss）| 58%（灾难性遗忘） |
| **冻结 base VLM + 4层MLP + 200 epoch** | **100%** ✅ |

**关键结论**：
- **不需要**对 VLM 做额外 fine-tune，冻结 frozen features 已足够
- **不需要** Teacher Forcing joint loss，只需单独训练 Router 分类头
- 关键是分类头足够深（4层）和训练充分（200 epoch）
- 单摄像头（top_rgb）已经足够，多摄像头反而因信息冗余降低准确率

---

## 四、整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MoE-SmolVLA                                  │
│                                                                     │
│  输入: [top_rgb] + [left_rgb] + [right_rgb] + State + Language      │
│           │                                                         │
│           ├─── top_rgb ──►  ┌─────────────────────┐               │
│           │                 │  Frozen SmolVLM      │               │
│           │                 │  (SigLIP + LM)       │               │
│           │                 └──────────┬──────────┘               │
│           │                            │ img_emb                   │
│           │                            │                           │
│           │                 ┌──────────▼──────────┐               │
│           │                 │  Router Head (4-MLP) │               │
│           │                 │  mean+std+max pool   │               │
│           │                 │  → argmax → Expert # │               │
│           │                 └──────────┬──────────┘               │
│           │                            │ 🔒 锁定                   │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  选中的 Action Expert    (例如: Expert 2 = top_long)         │   │
│  │                                                             │   │
│  │  SmolVLM KV-cache → lm_expert (Transformer) → 动作输出     │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.1 核心组件

| 组件 | 结构 | 是否冻结 | 说明 |
|------|------|----------|------|
| SmolVLM Backbone | SigLIP + LM | ✅ 冻结 | 共享视觉理解 |
| Router Head | 4层MLP+LayerNorm: 2880→512→256→128→4 | 🔄 独立训练 | 服装分类 |
| Expert 0-3 | SmolVLA lm_expert (各自独立) | 🔄 各自训练 | 动作专家 |
| action_out_proj | Linear | 🔄 各自训练 | 各 Expert 独立 |

---

## 五、VLAFlowMatching 架构详解与组件独立性分析

> [!IMPORTANT]
> 本节分析 MoE 架构中哪些组件需要独立，哪些可以共享

### 5.1 VLAFlowMatching 完整结构

```
VLAFlowMatching 结构层次:
│
├── vlm_with_expert
│   ├── vlm (SmolVLM)           ← 冻结，共享 (视觉语言理解)
│   │   ├── SigLIP (视觉编码器)
│   │   └── LM (语言模型层)
│   │
│   └── lm_expert (Transformer) ← ⚠️ 各 Expert 独立 (动作规划专精)
│
├── state_proj                  ← ✅ 共享 (本体感觉投影)
├── action_in_proj              ← ⚠️ 各 Expert 独立 (动作编码)
├── action_out_proj             ← ⚠️ 各 Expert 独立 (动作输出)
├── action_time_mlp_in          ← ✅ 共享 (时间编码)
└── action_time_mlp_out         ← ✅ 共享 (时间编码)
```

### 5.2 大脑比喻：理解各组件的作用

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    VLAFlowMatching = 完整大脑                            │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     VLM (SmolVLM)                                │   │
│  │                     视觉/语言皮层                                 │   │
│  │                                                                  │   │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │   │
│  │   │   视觉皮层    │  │   语言皮层    │  │   联合皮层    │         │   │
│  │   │  (SigLIP)    │  │  (Text Enc)  │  │  (LLM 层)    │         │   │
│  │   │              │  │              │  │              │         │   │
│  │   │ "看到衣服"    │  │ "叠衣服指令"  │  │ "理解任务"   │         │   │
│  │   └──────────────┘  └──────────────┘  └──────────────┘         │   │
│  │                                                                  │   │
│  │                      ↓ 输出: 视觉-语言特征 (KV Cache)            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              state_proj: 本体感觉神经 (✅ 共享)                   │   │
│  │                                                                  │   │
│  │   机器人关节状态 (12维) → 投影到 VLM 隐藏空间                    │   │
│  │   "我的手臂在什么位置？" → 转换成大脑能理解的信号                │   │
│  │                                                                  │   │
│  │   不管叠什么衣服，都需要知道手臂在哪里 → 可以共享                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              lm_expert: 运动皮层 (⚠️ 必须独立)                    │   │
│  │                                                                  │   │
│  │   这是需要针对不同任务专门训练的部分！                           │   │
│  │   - 叠短袖需要一种技能                                          │   │
│  │   - 叠长裤需要另一种技能                                        │   │
│  │   → 这就是为什么我们需要 4 个 Expert                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │         action_in_proj / action_out_proj: 神经-肌肉接口          │   │
│  │                        (⚠️ 必须独立)                              │   │
│  │                                                                  │   │
│  │   action_in_proj: 大脑的指令 → 转换成肌肉能执行的信号            │   │
│  │   action_out_proj: 运动皮层的规划 → 发送给肌肉的实际指令         │   │
│  │                                                                  │   │
│  │   叠短袖的肌肉协调模式 ≠ 叠长裤的肌肉协调模式                    │   │
│  │   → 这些也需要独立！                                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 组件独立性决策表

| 组件 | 独立性 | 原因 |
|------|--------|------|
| **VLM (SmolVLM)** | ✅ 共享 | 视觉理解能力通用，不需要针对衣物类型重新学习 |
| **state_proj** | ✅ 共享 | 本体感觉是通用的，只报告"手臂在哪里"，不决定"如何行动" |
| **lm_expert** | ⚠️ 独立 | 动作规划专精，不同衣物需要完全不同的策略 |
| **action_in_proj** | ⚠️ 独立 | 动作编码方式不同，不同任务的动作分布差异大 |
| **action_out_proj** | ⚠️ 独立 | 动作输出方式不同，肌肉协调模式各异 |
| **action_time_mlp** | ✅ 共享 | 时间编码是通用的 Flow Matching 机制 |

### 5.4 state_proj 为什么可以共享？详细分析

**背景**：所有数据都来自 SO101 机器人

```
┌─────────────────────────────────────────────────────────────────────────┐
│  SO101 硬件一致性：                                                      │
│                                                                         │
│  ├── 关节数量：12 (双臂各6)                                             │
│  ├── 关节范围：相同 (硬件限制)                                          │
│  ├── 动作空间：相同                                                     │
│  └── 运动学：相同                                                       │
│                                                                         │
│  这意味着 state_proj 输入的 12 维向量，语义完全一致！                    │
│  ├── dim 0 = 左臂肩部旋转角                                             │
│  ├── dim 1 = 左臂肩部抬升角                                             │
│  ├── ...                                                                │
│  └── dim 11 = 右臂夹爪开合度                                            │
└─────────────────────────────────────────────────────────────────────────┘
```

**state_proj vs action_proj 的本质区别**：

```
┌─────────────────────────────────────────────────────────────────────────┐
│  state_proj (状态 → VLM)：                                              │
│  ├── 输入：当前关节位置 (只读，被动)                                    │
│  ├── 语义：固定，由硬件决定                                             │
│  ├── 例子：不管叠什么衣服，"手臂在中间位置"的编码是一样的               │
│  └── 结论：可以共享 ✅                                                  │
│                                                                         │
│  action_in_proj (动作 → lm_expert)：                                    │
│  ├── 输入：目标动作速度 (主动，任务相关)                                │
│  ├── 语义：不同任务有不同的动作分布                                     │
│  ├── 例子：                                                            │
│  │   pant_short: [0.0, 0.5, ...] 肩部抬升，向下折叠                    │
│  │   top_long:   [0.5, 0.0, ...] 肩部旋转，向内折叠                    │
│  └── 结论：需要独立 ❌                                                  │
│                                                                         │
│  action_out_proj (lm_expert → 动作)：                                   │
│  ├── 输出：预测的动作速度                                               │
│  ├── 语义：不同任务有不同的输出模式                                     │
│  └── 结论：需要独立 ❌                                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.4.1 state_proj 共享假设的实证验证 (2026-03-17)

**验证方法**：分析 four_types_merged 数据集中不同服装类型的 state 分布

```bash
# 从 parquet 直接读取 observation.state，按 episode_index 分组统计
```

**结果汇总**：

| 服装类型 | 样本数 | 整体均值 | 整体标准差 | 左臂均值 | 右臂均值 |
|----------|--------|----------|------------|----------|----------|
| pant_short | 40,755 | 0.160 | 0.821 | 0.115 | 0.205 |
| pant_long | 65,909 | 0.190 | 0.826 | 0.120 | 0.260 |
| top_long | 83,068 | 0.180 | 0.843 | 0.157 | 0.202 |
| top_short | 76,066 | 0.151 | 0.847 | 0.139 | 0.163 |

**关键发现 1：整体分布相似**
- 均值范围：0.15 ~ 0.19（差异 < 0.04）
- 标准差范围：0.82 ~ 0.85（差异 < 0.03）
- 结论：SO101 硬件一致性假设成立

**关键发现 2：某些维度存在任务相关差异**

| 维度 | 关节 | pant_short | pant_long | top_long | top_short | 最大差异 |
|------|------|------------|-----------|----------|-----------|----------|
| 2 | L_elbow_flex | 0.531 | 0.941 | 0.752 | 0.736 | **0.410** |
| 7 | R_shoulder_lift | -0.683 | -0.285 | -0.762 | -0.784 | **0.499** |
| 8 | R_elbow_flex | 0.677 | 0.327 | 0.720 | 0.730 | **0.403** |
| 10 | R_wrist_roll | 0.349 | 0.589 | 0.321 | 0.212 | **0.378** |

差异最大的维度与折叠策略相关：
- `R_shoulder_lift`: pant_long 需要更高的右臂位置（0.50 差异）
- `L/R_elbow_flex`: 不同长度的衣物需要不同的肘部弯曲角度

**关键发现 3：类型间 L2 距离**

| 类型对 | L2 距离 | 解释 |
|--------|---------|------|
| top_long vs top_short | **0.16** | 最相似，都是上衣，策略相近 |
| pant_short vs top_long | 0.44 | 上衣 vs 裤子，中等差异 |
| pant_short vs top_short | 0.43 | 上衣 vs 裤子，中等差异 |
| pant_short vs pant_long | **0.82** | 裤子长度差异大，策略不同 |
| pant_long vs top_short | **0.83** | 最不同，长裤 vs 短上衣 |
| pant_long vs top_long | 0.78 | 长裤 vs 长上衣 |

**结论**：

1. **共享 state_proj 是合理的**：
   - 整体分布差异小（均值差异 < 0.04）
   - 12D 空间的 L2 距离（0.16-0.83）相对于标准差（~0.83）是可接受的
   - `state_proj` 是线性投影，可以学习保留所有维度的信息

2. **差异来源是任务策略，不是硬件**：
   - `pant_long` 的右臂位置明显不同（需要处理更长的裤腿）
   - `top_long` vs `top_short` 差异最小（都是向内折叠）
   - 这进一步证明 `action_*_proj` 需要独立，但 `state_proj` 可以共享

3. **建议保持共享设计**：
   - 参数量小（12×960 ≈ 11K），即使独立收益也不大
   - 共享可以防止过拟合（每个 Expert 数据量只有 1/4）
   - VLM 和 Router 的视觉特征已经提供了足够的任务区分信息

### 5.5 具体例子

**例子：相同状态，不同任务**

```python
# 假设某个时刻，机器人的关节状态是：
state = [0.5, 0.3, -0.2, 0.1, 0.0, 0.8,  # 左臂
         0.5, 0.3, -0.2, 0.1, 0.0, 0.8]  # 右臂

# 这个状态在两种任务中可能出现：
# - pant_short: 手臂在中间位置，准备向下抓取裤子
# - top_long:   手臂在中间位置，准备向内抓取上衣

# state_proj 的作用：
state_feature = state_proj(state)  # 投影到 VLM 空间
# 结果：告诉 VLM "手臂在中间位置"
# 这个投影对两种任务都是正确的！
```

**例子：不同动作，需要不同编码**

```python
# pant_short 的典型动作：
pant_action = [0.0, 0.5, 0.0, ...]  # 肩部抬升，向下折叠

# top_long 的典型动作：
top_action = [0.5, 0.0, 0.0, ...]   # 肩部旋转，向内折叠

# 如果共享 action_in_proj：
# 问题：这两个动作分布差异很大！共享的编码器可能无法很好地处理两种分布
```

### 5.6 最终 MoE 架构设计

```python
class MoEVLAFlowMatching:
    def __init__(self, base_vla, num_experts=4):
        # ========== 共享 (冻结) ==========
        self.vlm = base_vla.vlm                    # 视觉理解
        self.state_proj = base_vla.state_proj      # 本体感觉 ✅ 共享

        # ========== 时间编码 (共享) ==========
        self.action_time_mlp_in = base_vla.action_time_mlp_in
        self.action_time_mlp_out = base_vla.action_time_mlp_out

        # ========== 每个 Expert 独立 ==========
        self.experts = nn.ModuleList([
            nn.ModuleDict({
                'lm_expert': copy.deepcopy(base_vla.lm_expert),           # ❌ 独立
                'action_in_proj': copy.deepcopy(base_vla.action_in_proj), # ❌ 独立
                'action_out_proj': copy.deepcopy(base_vla.action_out_proj), # ❌ 独立
            })
            for _ in range(num_experts)
        ])
```

### 5.7 Checkpoint 加载确认

单独训练的 checkpoint 已经包含所有需要独立的权重：

```
smolvla_pant_short checkpoint 包含:
├── lm_expert.*          (专精 pant_short 的动作规划)
├── action_in_proj.*     (适配 pant_short 的动作编码)
├── action_out_proj.*    (适配 pant_short 的动作输出)
└── state_proj.*         (本体感觉，可以覆盖/忽略)

MoE 加载时：
├── Expert 0 ← pant_short checkpoint (lm_expert + action_*_proj)
├── Expert 1 ← pant_long checkpoint  (lm_expert + action_*_proj)
├── Expert 2 ← top_long checkpoint   (lm_expert + action_*_proj)
└── Expert 3 ← top_short checkpoint  (lm_expert + action_*_proj)

共享组件：
└── state_proj ← 从任意 checkpoint 加载一次即可
```

---

## 六、Expert 初始化策略（关键决策）

> [!IMPORTANT]
> 这是影响训练效率最大的设计决策，基于第五节的组件独立性分析

**方案 A（推荐）：从各类型单独训练的 checkpoint 热启动**

```
Expert 0 (pant_short) ← smolvla_pant_short/018000 的 lm_expert 权重
Expert 1 (pant_long)  ← smolvla_pant_long/xxx  的 lm_expert 权重
Expert 2 (top_long)   ← smolvla_top_long/xxx   的 lm_expert 权重
Expert 3 (top_short)  ← smolvla_top_short/xxx  的 lm_expert 权重
```

优点：每个 Expert 已经收敛到对应任务，MoE fine-tune 只需少量步数。

**方案 B：从 `smolvla_four_types` 统一初始化（4 Expert 共享初始值）**

```
所有 Expert ← smolvla_four_types/032000 的 lm_expert 权重（复制4份）
```

优点：简单，VLM 特征已适配服装任务。缺点：需要较长 fine-tune 分离。

**当前可用 checkpoints**：
- `outputs/train/smolvla_pant_short/checkpoints/018000` ✅
- `outputs/train/smolvla_four_types/checkpoints/032000` ✅

---

## 七、实现步骤（修正版）

### Step 1：Router Head 独立训练（已验证可行）

```python
# scripts/train_router.py
# 加载 smolvla_base 或 smolvla_four_types 的冻结 VLM
# 只训练 Router 分类头

router_head = nn.Sequential(
    nn.Linear(960 * 3, 512),   # 2880-dim: mean+std+max pooling
    nn.LayerNorm(512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 256),
    nn.LayerNorm(256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 4),          # 4 garment types
)

# 数据: GarmentTypeDataset（从 garment_info.json 读取 episode→type 映射）
# 优化: AdamW(lr=5e-4) + CosineAnnealingWarmRestarts + label_smoothing=0.1
# 目标: episode 首帧 → garment_type 分类
```

### Step 2：MoE Policy 模型文件

**新建文件**: `scripts/eval_policy/moe_smolvla_policy.py`

```python
class MoESmolVLAPolicy(BasePolicy):
    """
    MoE SmolVLA 推理策略
    - 共享 Frozen SmolVLM backbone (KV-cache 只算一次)
    - 4 套独立 Action Expert
    - Episode 开始时 Router 分类→锁定 Expert
    """

    def __init__(self, policy_path: str, router_path: str, device: str = "cuda"):
        # 加载 4 个 Expert checkpoints
        self.experts = [load_expert(f"{policy_path}/expert_{i}") for i in range(4)]
        # 加载 Router
        self.router = load_router(router_path)
        self._locked_expert_idx = None

    def reset(self):
        self._locked_expert_idx = None
        self._router_logits_buffer = []

    def select_action(self, observation_dict: dict) -> np.ndarray:
        # === Voting 阶段（前 3 帧）===
        if self._locked_expert_idx is None:
            logits = self._run_router(observation_dict)
            self._router_logits_buffer.append(logits)

            if len(self._router_logits_buffer) >= 3:
                agg = torch.stack(self._router_logits_buffer).sum(0)
                self._locked_expert_idx = agg.argmax(-1).item()
                logger.info(f"Router 锁定 Expert {self._locked_expert_idx}")

            # 空窗期 fallback: 用 Expert 0 作为默认（或上次结果）
            fallback_idx = 0
            return self.experts[fallback_idx].select_action(observation_dict)

        # === 锁定后 ===
        return self.experts[self._locked_expert_idx].select_action(observation_dict)
```

### Step 3：训练各 Expert（可并行）

```bash
# 4 个 Expert 分别 fine-tune（可以在不同 GPU 上并行）
python3 -m lerobot.scripts.train --config-path configs/train_smolvla_pant_short.yaml
python3 -m lerobot.scripts.train --config-path configs/train_smolvla_pant_long.yaml
python3 -m lerobot.scripts.train --config-path configs/train_smolvla_top_long.yaml
python3 -m lerobot.scripts.train --config-path configs/train_smolvla_top_short.yaml
```

> **这一步你们已经在做了！** pant_short 已经训练完成。

### Step 4：Router 训练

```bash
python3 scripts/train_router.py \
    --dataset Datasets/example/four_types_merged \
    --vlm_path outputs/train/smolvla_four_types/checkpoints/032000/pretrained_model \
    --output outputs/router/router_head.pt
```

### Step 5：Sticky Routing 推理集成

修改 `scripts/eval.py`，注册 `moe_smolvla` policy 类型。

---

## 八、已解决的关键设计问题

| 问题 | 早期文档的错误 | 正确方案 |
|------|----------------|----------|
| Expert 选择方式 | Top-2 加权组合 | **Hard argmax Top-1** |
| Router 结构 | `nn.Linear(hidden, 4)` 单层 | **4层 MLP + LayerNorm** |
| Expert 结构 | 自定义 TransformerDecoderLayer | **SmolVLA lm_expert 权重复用** |
| 是否需要 joint loss | 误以为需要 Teacher Forcing | **不需要，独立训练 Router 即可** |
| 摄像头选择 | 以为多摄像头更好 | **单摄 top_rgb 足够（100% probe）** |

---

## 九、风险与缓解

| 风险 | 概率 | 缓解方案 |
|------|------|----------|
| Router 在真实部署中准确率 < 100% | 中 | Voting N=3 过滤噪声，fallback 用 Expert 0 |
| Expert 过拟合（数据量只有 250 ep/类）| 中 | 使用预训练初始化 + 较小 lr |
| 推理时 KV-cache 显存 × 4 | 高 | 只加载 1 个 Expert 到 VRAM，其余 offload |
| 空窗期（前 3 帧）动作质量 | 低 | Fallback Expert 0 或 smolvla_base |

---

## 十、验证里程碑

```
✅ Phase 0: Router probe 验证 (100% 完成)
□  Phase 1: Expert 单独训练 (pant_short ✅, 其余 进行中)
□  Phase 2: Router 独立训练脚本 + 验证
□  Phase 3: MoESmolVLAPolicy 推理封装 + Isaac Sim 验证
□  Phase 4: 全类型成功率对比（MoE vs 单独训练 vs 联合训练）
```

---

## 十一、MoE 完整实现指南

> [!IMPORTANT]
> 本节提供 MoE-SmolVLA 的详细实现步骤和代码示例

### 11.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MoE-SmolVLA 架构                                 │
│                                                                         │
│  输入: images + state + language                                        │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    共享组件 (冻结或共享)                          │   │
│  │                                                                  │   │
│  │   ┌──────────────────────────────────────────────────────────┐ │   │
│  │   │  VLM (SmolVLM2-500M) ─────────────────────── 冻结, 共享   │ │   │
│  │   │  ├── SigLIP (视觉编码)                                   │ │   │
│  │   │  └── Text Model (16 layers)                              │ │   │
│  │   └──────────────────────────────────────────────────────────┘ │   │
│  │                              │                                  │   │
│  │   ┌──────────────────────────────────────────────────────────┐ │   │
│  │   │  state_proj ───────────────────────────────── 共享       │ │   │
│  │   │  Linear(12, 960)                                          │ │   │
│  │   └──────────────────────────────────────────────────────────┘ │   │
│  │                              │                                  │   │
│  │   ┌──────────────────────────────────────────────────────────┐ │   │
│  │   │  action_time_mlp_in/out ─────────────────────── 共享     │ │   │
│  │   │  时间编码，Flow Matching 通用                            │ │   │
│  │   └──────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│           ┌──────────────────┼──────────────────┐                      │
│           ▼                  ▼                  ▼                      │
│  ┌─────────────────┐  ┌─────────────┐  ┌────────────────────────────┐ │
│  │  Router Head    │  │  KV Cache   │  │   4 个独立 Expert           │ │
│  │  (独立训练)     │  │  (VLM 输出) │  │                            │ │
│  │                 │  │             │  │  ┌────────────────────────┐│ │
│  │  4层 MLP       │  │  共享,只算  │  │  │ Expert 0: pant_short   ││ │
│  │  2880→512→    │  │  一次       │  │  │ ├── lm_expert (~42M)   ││ │
│  │  256→128→4    │  │             │  │  │ ├── action_in_proj     ││ │
│  │                 │  │             │  │  │ └── action_out_proj   ││ │
│  │  Episode开始   │  │             │  │  ├────────────────────────┤│ │
│  │  → Voting 3帧  │  │             │  │  │ Expert 1: pant_long    ││ │
│  │  → argmax锁定  │  │             │  │  ├── ...                 ││ │
│  │                 │  │             │  │  ├────────────────────────┤│ │
│  │                 │  │             │  │  │ Expert 2: top_long     ││ │
│  │                 │  │             │  │  ├── ...                 ││ │
│  │                 │  │             │  │  ├────────────────────────┤│ │
│  │                 │  │             │  │  │ Expert 3: top_short    ││ │
│  │                 │  │             │  │  └── ...                 ││ │
│  │                 │  │             │  │  └────────────────────────┘│ │
│  └────────┬────────┘  └──────┬──────┘  └─────────────┬──────────────┘ │
│           │                  │                       │                  │
│           ▼                  ▼                       ▼                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      推理流程                                    │   │
│  │                                                                  │   │
│  │  Episode 开始:                                                   │   │
│  │    Frame 0-2: Router 投票 → 锁定 Expert ID                      │   │
│  │    Frame 3+: 使用锁定的 Expert 进行 Flow Matching                │   │
│  │                                                                  │   │
│  │  Flow Matching (每帧):                                          │   │
│  │    1. VLM 处理图像 → KV Cache (只算一次)                        │   │
│  │    2. state_proj 处理状态                                       │   │
│  │    3. 选中的 Expert 处理噪声动作 → 预测速度                     │   │
│  │    4. action_out_proj 输出最终动作                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 11.2 设计决策与原因

| 组件 | 决策 | 原因 |
|------|------|------|
| **VLM** | 冻结共享 | 视觉理解通用；节省显存；Router 实验证实冻结特征足够 (100%) |
| **state_proj** | 共享 | SO101 硬件一致；本体感觉通用；参数量小 (~11K) |
| **lm_expert** | 4 份独立 | 动作规划专精；消除梯度冲突；已有 checkpoint 可直接加载 |
| **action_in/out_proj** | 4 份独立 | 动作分布不同；神经-肌肉接口专精；checkpoint 已包含 |
| **action_time_mlp** | 共享 | 时间编码通用；Flow Matching 机制相同 |
| **Router** | Hard Argmax | 避免动作抖动；符合 SMP 论文最佳实践 |
| **Routing** | Sticky | 时间一致性；整个 Episode 锁定 |
| **Voting** | 3 帧 | 过滤噪声；提高准确性 |

### 11.3 SmolVLA 结构特点与 MoE 实现的关系

**关键洞察**：`lm_expert` 在 `vlm_with_expert` 内部，而 `action_*_proj` 在 `VLAFlowMatching` 层

```
VLAFlowMatching 结构层次:
│
├── vlm_with_expert (SmolVLMWithExpertModel)
│   ├── vlm (SmolVLM) ← 冻结
│   └── lm_expert ← 需要独立为 4 份
│
├── state_proj ← 共享
├── action_in_proj ← 需要独立为 4 份
├── action_out_proj ← 需要独立为 4 份
├── action_time_mlp_in ← 共享
└── action_time_mlp_out ← 共享
```

**Cross-Attention 机制（代码验证）**：

根据 `smolvlm_with_expert.py` 第 340-363 行的代码：

```python
# Expert 的 Query（用自己的 q_proj）
expert_query_state = expert_layer.self_attn.q_proj(expert_hidden_states)

# ⚠️ Expert 用自己的 k_proj 重新投影 VLM 的 hidden_states
expert_key_states = expert_layer.self_attn.k_proj(vlm_hidden_states)

# ⚠️ Expert 用自己的 v_proj 重新投影 VLM 的 hidden_states
expert_value_states = expert_layer.self_attn.v_proj(vlm_hidden_states)
```

**关键发现**：
- 每个 Expert 有自己独立的 `k_proj`, `v_proj`（维度：VLM_hidden → Expert_hidden）
- Cross-Attention 的流程是：
  1. VLM 输出 `vlm_hidden_states`
  2. **每个 Expert 用自己的 `k_proj`, `v_proj` 投影 `vlm_hidden_states`**
  3. Expert 用自己的 `q_proj` 投影 action tokens
  4. 计算 Attention(Q_expert, K_expert, V_expert)

**修正：KV Cache 不能直接共享**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Cross-Attention 数据流                                 │
│                                                                         │
│  VLM hidden_states [B, L, 960]  ← 可以共享存储                          │
│           │                                                             │
│           ├──────────────────┬──────────────────┐                       │
│           ▼                  ▼                  ▼                        │
│  Expert 0:                  Expert 1:          Expert 2:                │
│  k_proj_0(W_0)              k_proj_1(W_1)      k_proj_2(W_2)            │
│  v_proj_0(W_0)              v_proj_1(W_1)      v_proj_2(W_2)            │
│           │                  │                  │                        │
│           ▼                  ▼                  ▼                        │
│  K_0, V_0 [B,L,480]        K_1, V_1          K_2, V_2                  │
│                                                                         │
│  结论：每个 Expert 需要用自己的 k_proj, v_proj 重新计算 K, V           │
│        不能直接共享 KV Cache！                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

**正确的 MoE 实现**：
- 共享：VLM 的 `hidden_states` 输出（原始特征）
- 不共享：每个 Expert 的 `k_proj`, `v_proj` 投影结果
- Checkpoint 已包含每个 Expert 适配好的 `k_proj`, `v_proj`

### 11.4 四个 Expert 参数量分析

| 组件 | 单份参数量 | 份数 | 总参数量 |
|------|-----------|------|---------|
| VLM (冻结) | ~500M | 1 | ~500M |
| state_proj | ~11K | 1 | ~11K |
| action_time_mlp | ~691K | 1 | ~691K |
| **lm_expert** | **~42M** | **4** | **~168M** |
| **action_in_proj** | **~6K** | **4** | **~24K** |
| **action_out_proj** | **~6K** | **4** | **~24K** |
| Router Head | - | 1 | ~1.5M |

**MoE 总参数量**: ~670M
**激活参数量**: ~542M (VLM + 1个Expert + Router)
**额外存储**: ~170M (4个Expert - 1个共享)

**显存估算 (bfloat16)**:
- 全部加载: ~1.5 - 2.5 GB
- Expert offload: ~1.2 - 2.0 GB

### 11.5 实现代码示例

#### 11.5.1 MoE Policy 主类

```python
# 文件: scripts/eval_policy/moe_smolvla_policy.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np
from pathlib import Path
import copy

from scripts.eval_policy.base_policy import BasePolicy
from scripts.eval_policy.lerobot_policy import LeRobotPolicy


class MoESmolVLAPolicy(BasePolicy):
    """
    MoE SmolVLA 推理策略

    核心设计:
    - 共享: VLM, state_proj, action_time_mlp
    - 独立: 4 套 (lm_expert, action_in_proj, action_out_proj)
    - Sticky Routing: Episode 开始时锁定 Expert
    """

    GARMENT_TYPES = ["pant_short", "pant_long", "top_long", "top_short"]

    def __init__(
        self,
        expert_paths: Dict[str, str],  # {"pant_short": "path/to/ckpt", ...}
        router_path: str,
        base_vlm_path: Optional[str] = None,
        device: str = "cuda",
        num_voting_frames: int = 3,
    ):
        super().__init__()
        self.device = device
        self.num_voting_frames = num_voting_frames

        # 1. 加载共享的 VLM (从第一个 Expert 的 checkpoint)
        first_expert_path = list(expert_paths.values())[0]
        self._load_shared_components(first_expert_path, base_vlm_path)

        # 2. 加载 4 个 Expert
        self.experts = nn.ModuleDict()
        for garment_type, path in expert_paths.items():
            self.experts[garment_type] = self._load_expert(path)

        # 3. 加载 Router
        self.router = self._load_router(router_path)

        # 4. 初始化状态
        self._reset_episode_state()

    def _load_shared_components(self, expert_path: str, base_vlm_path: Optional[str]):
        """加载共享组件: VLM, state_proj, action_time_mlp"""
        base_policy = LeRobotPolicy.from_pretrained(expert_path)

        # 共享的 VLM (冻结)
        self.vlm_with_expert = base_policy.model.vlm_with_expert
        for param in self.vlm_with_expert.vlm.parameters():
            param.requires_grad = False

        # 共享的 state_proj
        self.state_proj = base_policy.model.state_proj

        # 共享的 action_time_mlp
        self.action_time_mlp_in = base_policy.model.action_time_mlp_in
        self.action_time_mlp_out = base_policy.model.action_time_mlp_out

        # 保存配置
        self.config = base_policy.config
        self.expert_hidden_size = self.vlm_with_expert.expert_hidden_size

    def _load_expert(self, checkpoint_path: str) -> nn.ModuleDict:
        """加载单个 Expert 的独立组件"""
        base_policy = LeRobotPolicy.from_pretrained(checkpoint_path)

        expert = nn.ModuleDict({
            'lm_expert': copy.deepcopy(base_policy.model.vlm_with_expert.lm_expert),
            'action_in_proj': copy.deepcopy(base_policy.model.action_in_proj),
            'action_out_proj': copy.deepcopy(base_policy.model.action_out_proj),
        })

        for param in expert.parameters():
            param.requires_grad = False

        return expert

    def _load_router(self, router_path: str) -> nn.Module:
        """加载 Router 分类头"""
        router = nn.Sequential(
            nn.Linear(960 * 3, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4),
        )

        router.load_state_dict(torch.load(router_path, map_location=self.device))
        router.eval()
        return router

    def _reset_episode_state(self):
        self._locked_expert_type: Optional[str] = None
        self._router_logits_buffer: list = []
        self._kv_cache = None

    def reset(self):
        self._reset_episode_state()

    def _run_router(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            img_emb = self.vlm_with_expert.embed_image(image)
            mean_f = img_emb.mean(dim=1)
            std_f = img_emb.std(dim=1)
            max_f = img_emb.max(dim=1).values
            feat = torch.cat([mean_f, std_f, max_f], dim=-1)
            feat = F.normalize(feat, p=2, dim=-1)
            logits = self.router(feat)
        return logits

    def select_action(self, observation_dict: dict) -> np.ndarray:
        image = self._get_top_rgb(observation_dict)

        # === Voting 阶段 ===
        if self._locked_expert_type is None:
            logits = self._run_router(image)
            self._router_logits_buffer.append(logits)

            if len(self._router_logits_buffer) >= self.num_voting_frames:
                agg_logits = torch.stack(self._router_logits_buffer).sum(0)
                expert_idx = agg_logits.argmax(dim=-1).item()
                self._locked_expert_type = self.GARMENT_TYPES[expert_idx]
                print(f"[MoE] Router 锁定 Expert: {self._locked_expert_type}")

            return self._fallback_action(observation_dict)

        # === 锁定后 ===
        return self._select_action_with_expert(observation_dict, self._locked_expert_type)
```

#### 11.5.2 Router 训练脚本

```python
# 文件: scripts/train_router.py

class RouterHead(nn.Module):
    """4层 MLP Router 分类头"""

    def __init__(self, input_dim: int = 960 * 3, num_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_router(dataset_path, vlm_path, output_path, epochs=200, lr=5e-4):
    """训练 Router 分类头"""

    # 1. 加载冻结的 VLM
    from lerobot.policies.smolvla import SmolVLAPolicy
    base_policy = SmolVLAPolicy.from_pretrained(vlm_path)
    vlm = base_policy.model.vlm_with_expert
    vlm.eval()
    for param in vlm.vlm.parameters():
        param.requires_grad = False

    # 2. 创建 Router 和数据集
    router = RouterHead()
    dataset = GarmentTypeDataset(dataset_path, f"{dataset_path}/meta/garment_info.json")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 3. 训练
    optimizer = torch.optim.AdamW(router.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(epochs):
        router.train()
        # ... 训练循环 ...

    torch.save(router.state_dict(), output_path)
```

### 11.6 实现注意事项

#### 11.6.1 Checkpoint Key 结构验证

**单独训练的 Checkpoint Key 结构**（代码验证）：

```
# 从 SmolVLA 代码分析，checkpoint 中的 key 结构：

model.vlm_with_expert.vlm.xxx                          # VLM (冻结，model.vlm_with_expert.lm_expert.layers.X.xxx           # lm_expert 层
model.vlm_with_expert.lm_expert.layers.X.self_attn.k_proj   # ⚠️ k_proj 在 lm_expert 内！
model.vlm_with_expert.lm_expert.layers.X.self_attn.v_proj   # ⚠️ v_proj 在 lm_expert 内！
model.vlm_with_expert.lm_expert.layers.X.mlp.xxx
model.vlm_with_expert.lm_expert.norm.weight
model.state_proj.weight
model.action_in_proj.weight
model.action_out_proj.weight
model.action_time_mlp_in.xxx
model.action_time_mlp_out.xxx
```

**⚠️ 修正：之前的 Key Mapping 代码是错误的！**

```python
# ❌ 错误的 mapping (文档之前建议的)
for key, value in state_dict.items():
    if 'lm_expert' in key:
        new_key = key.replace('vlm_with_expert.lm_expert.', '')
        expert_state[f'lm_expert.{new_key}'] = value  # 这会导致错误的 key！```

**✅ 正确的 Key Mapping**：

```python
def load_expert_weights(checkpoint_path: str, expert: nn.Module):
    """
    正确加载 Expert 权重

    Args:
        checkpoint_path: 单独训练的 checkpoint 路径
        expert: MoE Expert 模块 (包含 lm_expert, action_in_proj, action_out_proj)
    """
    from safetensors.torch import load_file

    # 加载 safetensors 或    state_dict = load_file(checkpoint_path)

    expert_state = {}
    for key, value in state_dict.items():
        # lm_expert 的 key
        if 'vlm_with_expert.lm_expert' in key:
            # model.vlm_with_expert.lm_expert.layers.0.xxx -> lm_expert.layers.0.xxx
            new_key = key.replace('model.vlm_with_expert.', '')
            expert_state[new_key] = value

        # action_in_proj 的 key
        elif 'action_in_proj' in key:
            # model.action_in_proj.weight -> action_in_proj.weight
            new_key = key.replace('model.', '')
            expert_state[new_key] = value

        # action_out_proj 的 key
        elif 'action_out_proj' in key:
            new_key = key.replace('model.', '')
            expert_state[new_key] = value

    # 加载到 expert
    missing, unexpected = expert.load_state_dict(expert_state, strict=False)

    return expert


# MoE 组装时加载 4 个 Expert
class MoESmolVLAPolicy:
    def _load_all_experts(self, expert_paths: Dict[str, str]):
        """加载 4 个 Expert"""
        self.experts = nn.ModuleDict()

        for garment_type, path in expert_paths.items():
            # 每个 Expert 是一个独立的模块
            expert = nn.ModuleDict({
                'lm_expert': copy.deepcopy(self.base_lm_expert),
                'action_in_proj': copy.deepcopy(self.base_action_in_proj),
                'action_out_proj': copy.deepcopy(self.base_action_out_proj),
            })

            # 加载权重
            load_expert_weights(path, expert)

            self.experts[garment_type] = expert
```

**关键点**：
1. `k_proj`, `v_proj` 是 `lm_expert` 的一部分
不是 VLM 的一部分
2. Checkpoint 已包含适配好的投影权重
3. MoE Expert 需要包含 `lm_expert`, `action_in_proj`, `action_out_proj` 三个组件
4. 需要去掉 `model.` 前缀，而不是去掉 `vlm_with_expert.lm_expert.`
```

#### 11.6.2 空窗期 Fallback 策略

| 方案 | 优点 | 缺点 |
|------|------|------|
| 零动作 | 安全 | 可能触发限位 |
| 默认 Expert | 简单 | 可能错误 |
| 原始 SmolVLA | 安全有效 | 需额外加载 |
| 投票领先者 | 最优 | 实现稍复杂 |

#### 11.6.3 显存优化 (Expert Offloading)

```python
class MoESmolVLAPolicy:
    def __init__(self, ...):
        # 只加载 Expert 0 到 GPU
        self.experts_on_gpu = {"pant_short": self._load_expert(paths["pant_short"])}
        self.expert_paths = paths
        self.experts_on_cpu = {}

    def _get_expert(self, garment_type: str) -> nn.Module:
        if garment_type in self.experts_on_gpu:
            return self.experts_on_gpu[garment_type]

        # 按需从 CPU 加载到 GPU
        if garment_type in self.experts_on_cpu:
            expert = self.experts_on_cpu[garment_type].to(self.device)
            self.experts_on_gpu[garment_type] = expert
            return expert

        # 从磁盘加载
        expert = self._load_expert(self.expert_paths[garment_type]).to(self.device)
        self.experts_on_gpu[garment_type] = expert
        return expert
```

### 11.7 验证清单

```
□ 1. Router 训练
   □ 准确率 > 95% (验证集)
   □ 保存 router_head.pt

□ 2. Expert Checkpoint 验证
   □ 4 个 Expert 都能正确加载
   □ 权重结构一致
   □ 单独评估成功率符合预期

□ 3. MoE 推理验证
   □ Router 正确分类 4 种衣物
   □ Sticky Routing 正确锁定 Expert
   □ Voting 机制正常工作

□ 4. Isaac Sim 评估
   □ 4 种衣物各跑 10+ episodes
   □ 成功率 vs 单独训练对比
   □ 成功率 vs 联合训练对比

□ 5. 性能验证
   □ 推理延迟 < 100ms
   □ 显存 < 4GB
   □ 无动作抖动
```

### 11.8 LoRA Expert 备选方案

如果显存紧张，可考虑 LoRA Expert：

| 方面 | 传统 Expert | LoRA Expert (r=8) |
|------|-------------|-------------------|
| 额外参数量 | ~168M | ~1M |
| 显存 (额外) | ~320 MB | ~2 MB |
| 实现复杂度 | 简单 | 中等 |
| 专精程度 | 最高 | 高 |

**建议**：优先使用传统 Expert，显存不足时再考虑 LoRA。

---

## 十二、MoE Router vs 简单脚本调度

> [!IMPORTANT]
> 本节分析 MoE Router 与简单脚本调度的本质区别，参考 AdaMoE-VLA 论文 (arXiv:2510.14300)

### 12.1 核心问题

**"MoE Router 和写一个脚本先用判断任务类别然后在推理有什么区别？"**

### 12.2 推理阶段：表面相似

```
┌─────────────────────────────────────────────────────────────────┐
│                    推理流程对比                                   │
├─────────────────────────────────────────────────────────────────┤
│  简单脚本：                                                       │
│    classify(image) → "pant_short" → load_model("pant_short") → action │
│                                                                   │
│  MoE Router：                                                     │
│    router(image) → expert_idx=0 → use_expert(0) → action          │
└─────────────────────────────────────────────────────────────────┘
```

**推理时行为确实等价！** 都是：分类 → 选择 → 执行。

### 12.3 训练阶段：本质区别

```
┌───────────────────────────────────────────────────────────────┐
│  简单脚本方案：4 个独立模型，各自训练                            │
│                                                               │
│  Expert 0 (pant_short) ← 只看 pant_short 数据                 │
│  Expert 1 (pant_long)  ← 只看 pant_long 数据                  │
│  Expert 2 (top_long)   ← 只看 top_long 数据                   │
│  Expert 3 (top_short)  ← 只看 top_short 数据                  │
│                                                               │
│  Router 分类器  ← 单独训练，和 Experts 完全解耦                 │
└───────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────┐
│  MoE 方案：端到端联合训练                                        │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                 Shared VLM Backbone                      │ │
│  │                    (冻结或微调)                           │ │
│  └───────────────────────┬─────────────────────────────────┘ │
│                          │                                    │
│          ┌───────────────┼───────────────┐                   │
│          ▼               ▼               ▼                   │
│    ┌──────────┐   ┌──────────┐   ┌──────────┐               │
│    │ Router   │   │ Expert 0 │   │ Expert 1 │ ...           │
│    │  Loss    │   │   Loss   │   │   Loss   │               │
│    └────┬─────┘   └────┬─────┘   └────┬─────┘               │
│         │              │              │                      │
│         └──────────────┴──────────────┘                      │
│                        │                                      │
│              Total Loss = Σ(w_i × Expert_Loss_i) + Router_Loss│
│                        │                                      │
│                        ▼                                      │
│              梯度回传到所有组件（包括共享的 VLM）              │
└───────────────────────────────────────────────────────────────┘
```

### 12.4 关键区别对比表

| 维度 | 简单脚本 | MoE 联合训练 |
|------|---------|-------------|
| **VLM 共享** | 4 个模型可能各自复制 VLM | VLM 权重物理共享 |
| **梯度流动** | 完全隔离 | Router 和 Experts 共享梯度 |
| **特征一致性** | Router 看到的特征 ≠ Experts 看到的特征 | 完全一致 |
| **参数效率** | 4× 全量参数 | 共享 VLM + 4× Expert 头 |

### 12.5 AdaMoE-VLA 论文洞察

论文标题：*"Expertise need not monopolize: Action-Specialized Mixture of Experts for Vision-Language-Action Learning"*

**核心创新**：解耦专家选择和权重

```
传统 MoE：  output = Σ(router_prob_i × expert_i(x))
                        ↑
                   权重 = 分类概率

AdaMoE：   output = Σ(scale_i × expert_i(x))
                        ↑
                   权重独立于选择
```

**对我们的启示**：
- 我们使用 **Hard Argmax (Top-1)**，避免了传统 MoE 的加权组合问题
- 服装类型是**互斥的**（pant_short 和 top_long 不会同时出现）
- **Sticky Routing** 保证时序一致性

### 12.6 结论

**推理效果**：MoE 和简单脚本调度在最终效果上**差异不大**

**MoE 的优势**：
1. **工程整洁**：一个模型文件，而不是 4 个
2. **参数共享**：VLM 只存一份，节省 ~500M 参数
3. **训练效率**：一次联合训练，而不是 4 次独立训练
4. **可扩展性**：新增服装类型只需加 Expert，不改 Router
5. **学术价值**：MoE 是热门方向

---

## 十三、比赛与真机部署考量

> [!IMPORTANT]
> 本节分析比赛约束和真机部署场景下的 MoE 优势

### 13.1 真机部署的关键约束

```
┌─────────────────────────────────────────────────────────────────┐
│                    真机部署 vs 离线实验                          │
├─────────────────────────────────────────────────────────────────┤
│  离线实验：                                                       │
│    - 模型加载时间：无所谓                                          │
│    - 内存占用：显存充足                                            │
│    - 推理延迟：没有硬性要求                                        │
│    - 容错：可以人工干预                                            │
│                                                                   │
│  真机部署 / 比赛：                                                 │
│    - 控制频率：需要 10-30 Hz (33-100ms/帧)                        │
│    - 内存限制：可能需要在单卡上运行                                 │
│    - 实时性：不能有卡顿或延迟尖峰                                   │
│    - 稳定性：Router 错误不能导致崩溃                               │
│    - 统一接口：比赛要求单一模型入口                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 13.2 场景分析：MoE vs 简单脚本

#### 场景 1：比赛模型提交

| 方案 | 优点 | 缺点 |
|------|------|------|
| **简单脚本** | 实现简单 | 可能不符合比赛要求的"单一模型"格式 |
| **MoE** | 单一模型文件，符合标准格式 | 实现复杂 |

```
比赛可能的提交要求：
- 一个 checkpoint 文件
- 一个统一的 inference 函数
- 支持 4 种服装类型的自动识别

→ MoE 更适合，因为它是"一个模型"
```

#### 场景 2：推理延迟

| 方案 | 延迟分析 |
|------|---------|
| **简单脚本** | 分类 (~5ms) + 模型推理 (~50ms) = **~55ms** |
| **MoE** | Router (~5ms) + Expert推理 (~50ms) = **~55ms** |

**延迟相近**，但有关键区别：

```
简单脚本的问题：
┌─────────────────────────────────────────────────────────────┐
│  Episode 1 (pant_short):                                    │
│    Frame 1: load_model(pant_short) ← 第一次加载，可能 1-2秒！  │
│    Frame 2-N: 推理 ~55ms                                     │
│                                                             │
│  Episode 2 (top_long):                                      │
│    Frame 1: load_model(top_long) ← 又要加载 1-2秒！          │
│    ...                                                      │
└─────────────────────────────────────────────────────────────┘

MoE 的优势：
┌─────────────────────────────────────────────────────────────┐
│  启动时：加载一次 MoE 模型（~3秒）                             │
│                                                             │
│  所有 Episode：                                              │
│    Frame 1-N: 推理 ~55ms，无切换延迟                         │
└─────────────────────────────────────────────────────────────┘
```

#### 场景 3：内存占用

```
简单脚本（预加载 4 个模型）：
┌─────────────────────────────────────────────────────────────┐
│  Model pant_short: ~670M 参数                                │
│  Model pant_long:  ~670M 参数                                │
│  Model top_long:    ~670M 参数                               │
│  Model top_short:   ~670M 参数                               │
│  ─────────────────────────────────────────                  │
│  总计: ~2.7GB 显存（如果预加载）                               │
│                                                             │
│  或者：动态加载，但每次切换有 1-2秒延迟                         │
└─────────────────────────────────────────────────────────────┘

MoE（共享 VLM）：
┌─────────────────────────────────────────────────────────────┐
│  Shared VLM:         ~500M 参数                              │
│  Router:             ~2M 参数                                │
│  4 × Expert Head:    ~170M 参数                              │
│  ─────────────────────────────────────────                  │
│  总计: ~670M 显存（单一模型）                                  │
└─────────────────────────────────────────────────────────────┘

→ MoE 节省 4× 显存！
```

### 13.3 MoE 内部的 Argmax 属于"单一模型"吗？

**答案：是的，绝对属于。**

```
┌─────────────────────────────────────────────────────────────────┐
│                    什么是"单一模型文件"？                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  从 PyTorch/LeRobot 的角度：                                      │
│    - 一个 checkpoint 目录 = 一个模型                              │
│    - 包含：config.json + model.safetensors                        │
│    - 加载方式：make_policy(policy_cfg, ds_meta=meta)             │
│                                                                   │
│  内部操作（如 argmax）是模型的实现细节：                           │
│    - nn.Linear 内部有矩阵乘法                                     │
│    - nn.Softmax 内部有指数运算                                    │
│    - MoE Router 内部有 argmax                                     │
│    → 这些都是模型内部的计算，不改变"单一模型"的本质                │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

**从评估系统的视角**：

```
评估系统看到的：
  ✅ 一个 checkpoint 目录
  ✅ 一个 policy_path 参数
  ✅ 一个 select_action() 方法
  ✅ 输入 observation → 输出 action

评估系统看不到的（也不关心）：
  ❌ 内部有 4 个 Expert
  ❌ 内部有 Router 分类器
  ❌ 内部有 argmax 操作
  ❌ 内部有 Voting 机制

这就像：
  - 你不知道 nn.Linear 内部有矩阵乘法
  - 你不知道 Transformer 内部有 Attention
  - 你不知道 MoE 内部有 Router + Argmax

→ 都是模型的"实现细节"，不影响"单一模型文件"的定义
```

### 13.4 比赛提交格式示例

```
submission/
├── pretrained_model/
│   ├── config.json          ← 模型配置
│   └── model.safetensors    ← 模型权重（包含 Router + 4 Expert）
└── policy.py                ← 推理代码（可选，或使用标准接口）

评估代码（比赛方）：
checkpoint_path = "submission/pretrained_model"
policy = make_policy(cfg, ds_meta)  # 加载你的 MoE 模型
policy.reset()                       # Episode 开始
action = policy.select_action(obs)   # 推理（内部有 Router + Argmax）

→ 完全符合"单一模型文件"的要求！
```

### 13.5 结论

| 问题 | 答案 |
|------|------|
| MoE 内部的 argmax 算"单一模型"吗？ | **是的**，是模型内部实现细节 |
| 评估系统能感知 Router 吗？ | **不能**，对 `select_action()` 透明 |
| 比赛会拒绝 MoE 吗？ | **不会**，格式完全兼容 |
| MoE 的优势是什么？ | 一个模型自动处理 4 种服装，无需手动指定 |

---

## 十四、MoE Policy 实现常见坑点与最佳实践

> [!IMPORTANT]
> 本节总结 MoE Policy 实现过程中的常见问题和解决方案

### 14.1 架构决策：继承 vs 组合

```
┌─────────────────────────────────────────────────────────────────┐
│                    方案对比                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  方案 A：继承 SmolVLAPolicy（推荐）                               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  class MoESmolVLAPolicy(SmolVLAPolicy):                     ││
│  │      def __init__(self, config):                            ││
│  │          super().__init__(config)  # 复用 VLM 加载等        ││
│  │          self.model = MoEVLAFlowMatching(config)  # 替换！  ││
│  │                                                             ││
│  │      def forward(self, batch):                              ││
│  │          # MoE 特殊逻辑                                     ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                   │
│  ✅ 优点：复用 prepare_images, prepare_state 等方法              │
│  ❌ 缺点：需要小心覆盖 self.model                                │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 14.2 常见坑点及解决方案

#### 坑点 1：Config 不兼容

```python
# ❌ 错误：直接用 SmolVLAConfig，MoE 参数无处放
config = SmolVLAConfig.from_pretrained(path)

# ✅ 正确：创建 MoESmolVLAConfig
@dataclass
class MoESmolVLAConfig(SmolVLAConfig):
    # MoE 特有参数
    num_experts: int = 4
    top_k: int = 1  # Hard Argmax
    voting_frames: int = 3
    router_hidden_dim: int = 256
    router_cls_weight: float = 0.1  # Router loss 权重

    # Expert 初始化
    expert_init_strategy: str = "clone"  # clone | random | pretrained
```

**坑点**：LeRobot 的 `make_policy` 会根据 `config.type` 选择 Policy 类。

```python
# third_party/lerobot/src/lerobot/policies/factory.py
POLICY_REGISTRY = {
    "smolvla": SmolVLAPolicy,
    "act": ACTPolicy,
    # 需要添加：
    "moe_smolvla": MoESmolVLAPolicy,
}
```

#### 坑点 2：权重加载/保存

```python
# ❌ 错误：直接 from_pretrained 会丢失 MoE 权重
policy = MoESmolVLAPolicy.from_pretrained("lerobot/smolvla_base")

# ✅ 正确：分步加载
class MoESmolVLAPolicy(PreTrainedPolicy):
    @classmethod
    def from_pretrained(cls, pretrained_path: str, **kwargs):
        config = MoESmolVLAConfig.from_pretrained(pretrained_path)
        policy = cls(config)

        # 1. 加载共享的 VLM 权重
        vlm_state_dict = load_state_dict(
            os.path.join(pretrained_path, "vlm_weights.safetensors")
        )
        policy.model.vlm_with_expert.load_state_dict(vlm_state_dict, strict=False)

        # 2. 加载 MoE Expert 权重（如果存在）
        moe_path = os.path.join(pretrained_path, "moe_experts.safetensors")
        if os.path.exists(moe_path):
            moe_state_dict = load_state_dict(moe_path)
            policy.model.load_experts(moe_state_dict)

        return policy
```

#### 坑点 3：Expert 初始化策略

```python
class MoEVLAFlowMatching(nn.Module):
    def __init__(self, config: MoESmolVLAConfig):
        super().__init__()
        self.config = config

        # 共享的 VLM
        self.vlm_with_expert = SmolVLMWithExpertModel(config)

        # MoE Experts
        self.experts = nn.ModuleList()

        for i in range(config.num_experts):
            if config.expert_init_strategy == "clone":
                # ✅ 克隆 lm_expert 结构
                expert = self._clone_expert(self.vlm_with_expert.lm_expert)
            elif config.expert_init_strategy == "pretrained":
                # 加载对应类型的预训练权重
                expert = self._load_pretrained_expert(i)
            else:
                # 随机初始化
                expert = self._create_expert()

            self.experts.append(expert)
```

#### 坑点 4：训练时的 Batch 组成

```python
# ❌ 错误：Batch 中混合多种服装类型，但梯度会冲突
batch = {
    "observation.images.top_rgb": [pant_short_img, top_long_img, ...],
    "garment_type": [0, 2, 1, 3, 0, ...],  # 混合
}

# ✅ 正确：按 Expert 分组计算 loss
def forward(self, batch):
    garment_types = batch["garment_type"]  # [B]

    # 1. 共享的 VLM 特征提取
    images, img_masks = self.prepare_images(batch)
    vlm_features = self.vlm_with_expert.embed_images(images)

    # 2. Router Loss（辅助损失）
    router_logits = self.router(vlm_features)
    router_loss = F.cross_entropy(router_logits, garment_types)

    # 3. 按 Expert 分组计算 Action Loss
    total_action_loss = 0
    for expert_idx in range(self.config.num_experts):
        mask = (garment_types == expert_idx)
        if mask.sum() == 0:
            continue

        expert_batch = {k: v[mask] for k, v in batch.items()}
        expert_features = vlm_features[mask]

        expert_loss = self._compute_expert_loss(
            expert_idx, expert_batch, expert_features
        )
        total_action_loss += expert_loss * mask.sum()

    total_action_loss /= len(garment_types)

    # 4. 组合 Loss
    total_loss = total_action_loss + self.config.router_cls_weight * router_loss

    return total_loss, {"action_loss": total_action_loss, "router_loss": router_loss}
```

#### 坑点 5：推理时的状态管理

```python
class MoESmolVLAPolicy(PreTrainedPolicy):
    def reset(self):
        """Episode 开始时调用"""
        super().reset()

        # ⚠️ 关键：重置 MoE 特有的状态
        self._locked_expert_idx = None
        self._router_logits_buffer = []
        self._voting_count = 0

        # 重置每个 Expert 的内部状态（如果有）
        for expert in self.model.experts:
            if hasattr(expert, 'reset'):
                expert.reset()

    def select_action(self, batch: dict, **kwargs) -> Tensor:
        # ⚠️ 关键：Voting 阶段的 fallback 策略
        if self._locked_expert_idx is None:
            # 获取 Router 预测
            router_logits = self._get_router_logits(batch)
            self._router_logits_buffer.append(router_logits)
            self._voting_count += 1

            if self._voting_count >= self.config.voting_frames:
                # 投票决定
                self._locked_expert_idx = self._vote_for_expert()

            # Fallback：使用默认 Expert
            return self._fallback_action(batch)

        # 锁定后：使用指定 Expert
        return self._expert_action(batch, self._locked_expert_idx)

    def _fallback_action(self, batch):
        """
        Voting 阶段的动作策略：
        方案 A：零动作（简单，但真机可能不安全）
        方案 B：使用默认 Expert (0)
        方案 C：使用当前 buffer 的加权平均
        """
        # 推荐：方案 B
        return self._expert_action(batch, expert_idx=0)
```

#### 坑点 6：CUDA 显存管理

```python
# ❌ 错误：4 个 Expert 都加载到 GPU，显存爆炸
# 每个 Expert ~42M 参数 * 4 = ~168M 额外参数

# ✅ 正确：方案 1 - 共享 VLM，只复制轻量级部分
# VLM ~500M 共享，Expert Head ~42M * 4 = ~168M
# 总计 ~670M，单卡可接受

# ✅ 正确：方案 2 - Expert Offloading
class MoESmolVLAPolicy:
    def __init__(self, ...):
        # 只加载 Expert 0 到 GPU
        self.experts_on_gpu = {"pant_short": self._load_expert(paths["pant_short"])}
        self.expert_paths = paths
        self.experts_on_cpu = {}

    def _get_expert(self, garment_type: str) -> nn.Module:
        if garment_type in self.experts_on_gpu:
            return self.experts_on_gpu[garment_type]

        # 按需从 CPU 加载到 GPU
        if garment_type in self.experts_on_cpu:
            expert = self.experts_on_cpu[garment_type].to(self.device)
            self.experts_on_gpu[garment_type] = expert
            return expert

        # 从磁盘加载
        expert = self._load_expert(self.expert_paths[garment_type]).to(self.device)
        self.experts_on_gpu[garment_type] = expert
        return expert
```

### 14.3 优雅的实现模式

#### 使用 Mixin 分离关注点

```python
class RouterMixin:
    """Router 相关逻辑"""

    def _build_router(self, config):
        self.router = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, config.num_experts),
        )

    def _route(self, images):
        feat = self._extract_router_features(images)
        return self.router(feat)


class ExpertMixin:
    """Expert 管理逻辑"""

    def _build_experts(self, config):
        self.experts = nn.ModuleList([
            self._create_expert(config) for _ in range(config.num_experts)
        ])
        self.action_in_projs = nn.ModuleList([...])
        self.action_out_projs = nn.ModuleList([...])


class MoEVLAFlowMatching(nn.Module, RouterMixin, ExpertMixin):
    """组合 Mixin"""

    def __init__(self, config):
        super().__init__()
        self._build_router(config)
        self._build_experts(config)
```

### 14.4 文件组织建议

```
source/lehome/lehome/policies/
├── __init__.py
├── moe_smolvla/
│   ├── __init__.py
│   ├── configuration.py       # MoESmolVLAConfig
│   ├── modeling.py            # MoESmolVLAPolicy, MoEVLAFlowMatching
│   ├── router.py              # Router 相关类
│   ├── experts.py             # Expert 管理类
│   └── mixins.py              # 可复用的 Mixin
└── ...

configs/
├── train_moe_smolvla.yaml     # 训练配置
└── eval_moe_smolvla.yaml      # 评估配置

tests/
├── test_moe_policy.py         # 单元测试
└── test_moe_integration.py    # 集成测试
```

### 14.5 实现检查清单

```
┌─────────────────────────────────────────────────────────────────┐
│                    MoE Policy 实现检查清单                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  配置层：                                                         │
│  □ MoESmolVLAConfig 继承 SmolVLAConfig                           │
│  □ 添加 num_experts, voting_frames 等参数                        │
│  □ 注册到 POLICY_REGISTRY                                        │
│                                                                   │
│  模型层：                                                         │
│  □ MoEVLAFlowMatching 替代 VLAFlowMatching                       │
│  □ Router 分类器正确初始化                                        │
│  □ 4 个 Expert 独立初始化                                         │
│  □ action_in_proj / action_out_proj 独立                         │
│  □ state_proj / action_time_mlp 共享                             │
│                                                                   │
│  训练层：                                                         │
│  □ forward() 正确计算混合 batch 的 loss                          │
│  □ Router 辅助损失正确加入                                        │
│  □ 梯度正确回传到所有 Expert                                      │
│                                                                   │
│  推理层：                                                         │
│  □ reset() 清除所有状态                                           │
│  □ Voting 机制正确实现                                            │
│  □ Fallback 策略合理                                              │
│  □ Sticky Routing 锁定 Expert                                     │
│                                                                   │
│  序列化：                                                         │
│  □ save_pretrained() 正确保存所有权重                            │
│  □ from_pretrained() 正确加载所有权重                            │
│  □ 与 LeRobot checkpoint 格式兼容                                │
│                                                                   │
│  测试：                                                           │
│  □ 单元测试：Router 输出形状                                      │
│  □ 单元测试：Voting 锁定逻辑                                      │
│  □ 单元测试：Expert 选择正确性                                    │
│  □ 集成测试：完整训练流程                                         │
│  □ 集成测试：完整推理流程                                         │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 14.6 坑点总结表

| 类别 | 关键坑点 | 解决方案 |
|------|---------|---------|
| **配置** | MoE 参数无处放 | 继承 SmolVLAConfig |
| **权重** | 加载/保存不兼容 | 分步加载，分别保存 |
| **初始化** | Expert 权重随机 | 克隆 base expert 或加载预训练 |
| **训练** | 混合 batch 梯度冲突 | 按 Expert 分组计算 |
| **推理** | Voting 阶段无动作 | Fallback 到默认 Expert |
| **显存** | 4 个 Expert 太大 | LoRA Expert 或 Expert Offloading |
| **状态** | reset() 不完整 | 清除所有 MoE 状态 |

---

## 十五、版本历史

| 版本 | 日期 | 更新内容 |
|------|------|---------|
| v1.0 | 2026-03-16 | 初始版本，整合 moe_smolvla_design.md + moe_implementation_plan.md |
| v1.1 | 2026-03-16 | 新增第五节：VLAFlowMatching 架构详解与组件独立性分析 |
| v1.2 | 2026-03-16 | 新增第十一节：MoE 完整实现指南 |
| v1.3 | 2026-03-17 | 新增第十二节：MoE Router vs 简单脚本调度分析 |
| v1.3 | 2026-03-17 | 新增第十三节：比赛与真机部署考量 |
| v1.4 | 2026-03-17 | 新增第十五节：Checkpoint Key mapping 与推理接口实现细节

    - 整合 reset() 调用时机分析、- LeRobot Policy 参考实现
    - VLAFlowMatching 结构与 Checkpoint Key mapping
    - KV Cache 共享技术细节
    - 比赛接口定义与实现建议

---

## 十五、Checkpoint Key Mapping 与推理接口实现细节

> [!IMPORTANT]
> 本节整合代码分析结果，提供精确的 Checkpoint 加载和推理实现指导

### 15.1 reset() 调用时机确认

**结论**：✅ 每个 Episode 开始时保证调用

在 `scripts/utils/evaluation.py` 第 104-108 行：

```python
for i in range(args.num_episodes):
    # 1. Reset Environment & Policy
    env.reset()
    policy.reset()  # ✅ 每个 Episode 开始时调用
    stabilize_garment_after_reset(env, args)

    # 2. Initial Observation
    observation_dict = env._get_observations()

    # 3. Episode loop
    for st in range(args.max_steps):
        action_np = policy.select_action(observation_dict)
        ...
```

**对 MoE 实现的意义**：
- MoE Policy 的 `reset()` 可以安全清除 `_locked_expert_idx` 和 `_router_logits_buffer`
- 不需要自己跟踪 episode 边界

### 15.2 比赛接口定义

在 `scripts/eval_policy/base_policy.py` 中定义了官方接口：

```python
class BasePolicy(abc.ABC):
    """
    Base Policy Class for LeHome Challenge.

    All participant submissions must inherit from this class and implement
    the `select_action` and `reset` methods.
    """

    def __init__(self, **kwargs):
        """加载模型权重和配置"""
        pass

    def reset(self):
        """
        Reset the policy state.
        Called at the beginning of each episode.
        """
        pass

    @abc.abstractmethod
    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Args:
            observation (Dict[str, np.ndarray]):
                - Images: (H, W, C), uint8, range [0, 255]
                - States: (N,), float32

        Returns:
            action (np.ndarray): Action command (float32).
        """
        raise NotImplementedError()
```

**注册方式**：

```python
from scripts.eval_policy import PolicyRegistry
from scripts.eval_policy.base_policy import BasePolicy

@PolicyRegistry.register("moe_smolvla")
class MoESmolVLAPolicy(BasePolicy):
    GARMENT_TYPES = ["pant_short", "pant_long", "top_long", "top_short"]

    def __init__(self, model_path: str, device: str = "cpu", **kwargs):
        super().__init__()
        # 加载模型...

    def reset(self):
        self._locked_expert_idx = None
        self._router_logits_buffer = []

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        # MoE 推理逻辑...
```

### 15.3 LeRobot Policy 参考实现

`scripts/eval_policy/lerobot_policy.py` 展示了关键流程：

```python
class LeRobotPolicy(BasePolicy):
    def __init__(self, policy_path, dataset_root, task_description, device="cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.task_description = task_description

        # 1. 加载 metadata (用于 normalization stats)
        meta = LeRobotDatasetMetadata(repo_id="lehome", root=dataset_root)

        # 2. 加载 config
        policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
        policy_cfg.pretrained_path = policy_path

        # 3. Filter metadata (只保留 policy 需要的 features)
        self.input_features = set(policy_cfg.input_features.keys())
        self._filter_metadata(meta, self.input_features)

        # 4. 创建 policy + preprocessor + postprocessor
        self.policy = make_policy(policy_cfg, ds_meta=meta)
        self.preprocessor, self.postprocessor = make_pre_post_processors(policy_cfg, meta)

        # 5. 推断 action_dim
        self.action_dim = self._infer_action_dim(meta, task_description)

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        # 1. Filter observations (只保留 policy 需要的)
        if self.input_features:
            observation = self._filter_observations(observation, self.input_features)

        # 2. Preprocess (Numpy -> Tensor Batch, Normalize)
        batch_obs = self._process_observation(observation)

        # 3. Inference
        with torch.inference_mode():
            batch_action = self.policy.select_action(batch_obs)

        # 4. Postprocess (Un-normalize)
        if self.postprocessor:
            batch_action = self.postprocessor(batch_action)

        # 5. Convert to Numpy (移除 batch 维度)
        return batch_action.squeeze(0).cpu().numpy()
```

**关键观察**：

1. **Numpy I/O**: 输入是 Numpy Dict，输出是 Numpy Array
2. **Batch 维度**: LeRobot 内部需要 batch 维度，但 `select_action` 输入输出都是单样本
3. **Normalization**: 需要 preprocessor 和 postprocessor 处理归一化

### 15.4 VLAFlowMatching 结构与 Checkpoint Key Mapping

#### 15.4.1 模型组件结构

```python
# third_party/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py

class VLAFlowMatching(nn.Module):
    def __init__(self, config: SmolVLAConfig):
        super().__init__()
        self.config = config

        # === VLM + Expert ===
        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=config.vlm_model_name,
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
            ...
        )

        # === 投影层 (各自独立) ===
        self.state_proj = nn.Linear(
            config.max_state_dim,
            self.vlm_with_expert.config.text_config.hidden_size
        )
        self.action_in_proj = nn.Linear(
            config.max_action_dim,
            self.vlm_with_expert.expert_hidden_size
        )
        self.action_out_proj = nn.Linear(
            self.vlm_with_expert.expert_hidden_size,
            config.max_action_dim
        )

        # === 时间编码 (共享) ===
        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2,
            self.vlm_with_expert.expert_hidden_size
        )
        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size,
            self.vlm_with_expert.expert_hidden_size
        )
```

#### 15.4.2 SmolVLMWithExpertModel 内部结构

```python
# third_party/lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py

class SmolVLMWithExpertModel(nn.Module):
    def __init__(self, ...):
        # VLM (冻结或微调)
        self.vlm = AutoModelForImageTextToText.from_pretrained(model_id, ...)

        # lm_expert (独立训练)
        lm_expert_config = copy.deepcopy(config.text_config)
        lm_expert_config.hidden_size = int(hidden_size * expert_width_multiplier)
        self.lm_expert = AutoModel.from_config(lm_expert_config)

        # ⚠️ 关键：lm_expert 有独立的 k_proj, v_proj
        for layer_idx in range(len(self.lm_expert.layers)):
            self.lm_expert.layers[layer_idx].self_attn.k_proj = nn.Linear(...)
            self.lm_expert.layers[layer_idx].self_attn.v_proj = nn.Linear(...)
```

#### 15.4.3 Checkpoint Key 实际结构

通过分析 `smolvla_pant_short/checkpoints/009000/pretrained_model/model.safetensors`：

```
=== lm_expert keys (共 145 个) ===
model.vlm_with_expert.lm_expert.layers.0.input_layernorm.weight
model.vlm_with_expert.lm_expert.layers.0.mlp.down_proj.weight
model.vlm_with_expert.lm_expert.layers.0.mlp.gate_proj.weight
model.vlm_with_expert.lm_expert.layers.0.mlp.up_proj.weight
model.vlm_with_expert.lm_expert.layers.0.post_attention_layernorm.weight
model.vlm_with_expert.lm_expert.layers.0.self_attn.k_proj.weight  # ⚠️ 独立的
model.vlm_with_expert.lm_expert.layers.0.self_attn.o_proj.weight
model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj.weight
model.vlm_with_expert.lm_expert.layers.0.self_attn.v_proj.weight  # ⚠️ 独立的
...

=== action_in_proj keys ===
model.action_in_proj.bias
model.action_in_proj.weight

=== action_out_proj keys ===
model.action_out_proj.bias
model.action_out_proj.weight

=== state_proj keys ===
model.state_proj.bias
model.state_proj.weight

=== action_time_mlp keys ===
model.action_time_mlp_in.bias
model.action_time_mlp_in.weight
model.action_time_mlp_out.bias
model.action_time_mlp_out.weight

总共 500 个 keys
```

### 15.5 MoE Checkpoint 加载的精确 Key Mapping

```python
def load_expert_weights(checkpoint_path: str, expert_idx: int) -> dict:
    """
    从单独训练的 checkpoint 加载 Expert 权重

    输入 checkpoint key:           model.vlm_with_expert.lm_expert.layers.0.xxx
    MoE 目标 key:              experts.{expert_idx}.lm_expert.layers.0.xxx
    """
    from safetensors import safe_open

    expert_state = {}

    with safe_open(checkpoint_path, framework='pt') as f:
        for key in f.keys():
            # === lm_expert ===
            if 'lm_expert' in key:
                # model.vlm_with_expert.lm_expert.xxx -> experts.{idx}.lm_expert.xxx
                new_key = key.replace(
                    'model.vlm_with_expert.lm_expert',
                    f'experts.{expert_idx}.lm_expert'
                )
                expert_state[new_key] = f.get_tensor(key)

            # === action_in_proj / action_out_proj ===
            elif 'action_in_proj' in key or 'action_out_proj' in key:
                # model.action_in_proj.xxx -> experts.{idx}.action_in_proj.xxx
                new_key = key.replace('model.', f'experts.{expert_idx}.')
                expert_state[new_key] = f.get_tensor(key)

    return expert_state


def load_shared_weights(checkpoint_path: str) -> dict:
    """
    加载共享组件: state_proj, action_time_mlp
    """
    shared_state = {}

    with safe_open(checkpoint_path, framework='pt') as f:
        for key in f.keys():
            if 'state_proj' in key or 'action_time_mlp' in key:
                shared_state[key] = f.get_tensor(key)

    return shared_state
```

### 15.6 KV Cache 共享的技术细节

**问题**：每个 Expert 有独立的 `k_proj`, `v_proj`，意味着 KV Cache **不能直接共享**

```python
# smolvlm_with_expert.py:113-122
for layer_idx in range(len(self.lm_expert.layers)):
    # 每个 Expert 有自己的 k_proj, v_proj
    self.lm_expert.layers[layer_idx].self_attn.k_proj = nn.Linear(...)
    self.lm_expert.layers[layer_idx].self_attn.v_proj = nn.Linear(...)
```

**解决方案**：存储 VLM 的原始输出，让每个 Expert 用自己的 k_proj/v_proj 投影

```python
class MoEVLAFlowMatching(nn.Module):
    def forward_with_expert(self, expert_idx: int, ...):
        # 1. VLM forward (共享，只算一次)
        vlm_output = self.vlm_with_expert.vlm(...)

        # 2. 每个 Expert 用自己的 k_proj/v_proj 投影 VLM 输出
        expert = self.experts[expert_idx]
        for layer in expert.lm_expert.layers:
            # 每个 Expert 独立投影
            k = layer.self_attn.k_proj(vlm_output)
            v = layer.self_attn.v_proj(vlm_output)
            ...
```

**实际实现建议**：

```python
class MoEVLAFlowMatching(nn.Module):
    def __init__(self, config):
        # 共享的 VLM (冻结)
        self.vlm = load_vlm(config.vlm_model_name)
        self.vlm.eval()
        for p in self.vlm.parameters():
            p.requires_grad = False

        # 共享的 state_proj, action_time_mlp
        self.state_proj = nn.Linear(...)
        self.action_time_mlp_in = nn.Linear(...)
        self.action_time_mlp_out = nn.Linear(...)

        # 4 个独立的 Expert (lm_expert + action_*_proj)
        self.experts = nn.ModuleList([
            self._create_expert(config) for _ in range(4)
        ])

    def _create_expert(self, config):
        return nn.ModuleDict({
            'lm_expert': self._clone_lm_expert(config),
            'action_in_proj': nn.Linear(...),
            'action_out_proj': nn.Linear(...),
        })
```

### 15.7 MoE Policy 完整实现建议

```python
# scripts/eval_policy/moe_smolvla_policy.py

from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from scripts.eval_policy import PolicyRegistry
from scripts.eval_policy.base_policy import BasePolicy


@PolicyRegistry.register("moe_smolvla")
class MoESmolVLAPolicy(BasePolicy):
    """
    MoE SmolVLA Policy for LeHome Challenge

    自动识别 4 种服装类型并使用对应的 Expert
    """

    GARMENT_TYPES = ["pant_short", "pant_long", "top_long", "top_short"]
    VOTING_FRAMES = 3

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        **kwargs
    ):
        super().__init__()
        self.device = torch.device(device)
        self.model_path = Path(model_path)

        # 加载模型
        self._load_model(model_path)

        # Episode 状态
        self._reset_episode_state()

    def _load_model(self, model_path: str):
        """加载 MoE 模型"""
        # 1. 加载共享组件 (从 Expert 0 的 checkpoint)
        shared_ckpt = self._get_checkpoint_path(model_path, expert_idx=0)
        self._load_shared_components(shared_ckpt)

        # 2. 加载 4 个 Expert
        self.experts = nn.ModuleList()
        for i in range(4):
            expert_ckpt = self._get_checkpoint_path(model_path, expert_idx=i)
            expert = self._load_expert(expert_ckpt, i)
            self.experts.append(expert)

        # 3. 加载 Router
        self.router = self._load_router(model_path)

    def _get_checkpoint_path(self, model_path: str, expert_idx: int) -> Path:
        """获取指定 Expert 的 checkpoint 路径"""
        garment_type = self.GARMENT_TYPES[expert_idx]
        return Path(model_path) / "experts" / garment_type / "pretrained_model"
 / "model.safetensors"

    def reset(self):
        """Episode 开始时调用"""
        self._locked_expert_idx = None
        self._router_logits_buffer = []
        self._voting_count = 0

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """核心推理逻辑"""
        # 1. 提取 top_rgb 图像
        image = self._get_top_rgb(observation)

        # 2. Voting 阶段
        if self._locked_expert_idx is None:
            logits = self._run_router(image)
            self._router_logits_buffer.append(logits)
            self._voting_count += 1

            if self._voting_count >= self.VOTING_FRAMES:
                # 投票决定
                self._locked_expert_idx = self._vote_for_expert()

            # Fallback: 使用 Expert 0 (或零动作)
            return self._fallback_action(observation)

        # 3. 锁定后：使用指定 Expert
        return self._expert_action(observation, self._locked_expert_idx)

    def _vote_for_expert(self) -> int:
        """投票决定 Expert"""
        agg_logits = torch.stack(self._router_logits_buffer).sum(dim=0)
        return agg_logits.argmax(dim=-1).item()

    def _fallback_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Voting 阶段的 Fallback 策略"""
        # 方案 A: 零动作 (安全但可能不 optimal)
        # return np.zeros(self.action_dim, dtype=np.float32)

        # 方案 B: 使用 Expert 0 (推荐)
        return self._expert_action(observation, expert_idx=0)

    def _expert_action(self, observation: Dict[str, np.ndarray], expert_idx: int) -> np.ndarray:
        """使用指定 Expert 进行推理"""
        with torch.inference_mode():
            # 预处理
            batch = self._preprocess(observation)

            # Expert 推理
            action = self.experts[expert_idx](batch)

            # 后处理
            return self._postprocess(action)
```

### 15.8 模型目录结构建议

```
submission/
├── pretrained_model/
│   ├── config.json                    # MoE 配置
│   ├── model.safetensors               # 共享组件 (VLM + state_proj + action_time_mlp)
│   ├── router.safetensors              # Router 权重
│   └── experts/
│       ├── pant_short/
│       │   └── model.safetensors       # Expert 0 权重
│       ├── pant_long/
│       │   └── model.safetensors       # Expert 1 权重
│       ├── top_long/
│       │   └── model.safetensors       # Expert 2 权重
│       └── top_short/
│           └── model.safetensors       # Expert 3 权重
└── policy.py                           # MoESmolVLAPolicy 实现 (可选)
```

### 15.9 实现检查清单

```
□ 1. Checkpoint 加载
   □ 确认 4 个 Expert 的 checkpoint 都存在
   □ 验证 key mapping 正确 (打印加载前后的 key 对比)
   □ 确认共享组件只加载一次

□ 2. Router 集成
   □ Router 准确率 > 95%
   □ Voting 机制正确实现
   □ Fallback 策略合理

□ 3. 推理验证
   □ reset() 清除所有状态
   □ Episode 内 Expert 锁定正确
   □ 输出动作形状正确 (12D for dual-arm)

□ 4. Isaac Sim 评估
   □ 4 种服装各跑 10+ episodes
   □ 成功率 vs 单独训练对比
   □ 无动作抖动
```

---

## 十六、比赛 Eval 框架下 MoE Policy 集成分析

> [!IMPORTANT]
> 本节详细分析 MoE Policy 在比赛评估框架下的可行性，基于实际代码分析 (`scripts/eval.py`, `scripts/utils/evaluation.py`, `scripts/eval_policy/base_policy.py`)

### 16.1 Eval 框架完整流程

```python
# scripts/utils/evaluation.py

┌─────────────────────────────────────────────────────────────────┐
│                    eval() 主函数 (行239)                         │
│                                                                 │
│  1. 创建环境配置 (env_cfg)                                        │
│  2. PolicyRegistry.create(policy_type, **kwargs) ← 创建 MoE     │
│  3. 加载评估列表 (garment 列表)                                  │
│  4. for each garment in eval_list:                              │
│     if garment_idx > 0:                                         │
│         env.switch_garment(garment_name, garment_stage)         │
│         policy.reset()  ← 切换 garment 时调用 ✅                │
│     run_evaluation_loop(env, policy, args)                      │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│              run_evaluation_loop() (行33)                       │
│                                                                 │
│  for i in range(num_episodes):                                  │
│    1. env.reset()                                               │
│    2. policy.reset()  ← 每个 episode 开始时调用 ✅              │
│    3. stabilize_garment_after_reset(env, args)                  │
│    4. observation_dict = env._get_observations()                │
│    5. for st in range(max_steps):                               │
│         action = policy.select_action(observation_dict)         │
│         env.step(action)                                        │
│         observation_dict = env._get_observations()              │
└─────────────────────────────────────────────────────────────────┘
```

### 16.2 reset() 调用时机与语义

#### 代码确认

```python
# scripts/utils/evaluation.py:104-108
for i in range(args.num_episodes):
    # 1. Reset Environment & Policy
    env.reset()
    policy.reset()  # ✅ 每个 Episode 开始时调用
    stabilize_garment_after_reset(env, args)
```

```python
# scripts/utils/evaluation.py:380-392
# Garment 切换时也会调用
if garment_idx > 0:
    if hasattr(env, "switch_garment"):
        env.switch_garment(garment_name, garment_stage)
        env.reset()
        policy.reset()  # ← 切换 garment 时调用
    else:
        env.close()
        env = gym.make(args.task, cfg=env_cfg).unwrapped
        policy.reset()  # ← 重新创建环境后调用
```

#### 对 MoE Policy 的意义

| 场景 | 调用时机 | MoE 需要清除的状态 |
|------|----------|-------------------|
| **Episode 开始** | 每个 episode 的第一帧前 | `_locked_expert_idx`, `_router_logits_buffer`, `_voting_count` |
| **Garment 切换** | 不同 garment 之间 | 同上，重新开始 Voting |

**结论**：✅ MoE Policy 的 `reset()` 可以安全清除所有状态，不需要自己跟踪 episode/garment 边界。

### 16.3 BasePolicy 接口规范

```python
# scripts/eval_policy/base_policy.py

class BasePolicy(abc.ABC):
    """
    Base Policy Class for LeHome Challenge.

    All participant submissions must inherit from this class and implement
    the `select_action` and `reset` methods.
    """

    def __init__(self, **kwargs):
        """
        Initialize the policy. Model weights and configurations should be
        loaded here.
        """
        pass

    def reset(self):
        """
        Reset the policy state.

        Called at the beginning of each episode (e.g., to clear RNN
        hidden states or action buffers).
        """
        pass

    @abc.abstractmethod
    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate an action based on the given observation.

        Args:
            observation (Dict[str, np.ndarray]): Environmental observation data (Numpy format).
                - Images: (H, W, C), uint8, range [0, 255]
                - States: (N,), float32

        Returns:
            action (np.ndarray): Action command (Numpy format, float32).
        """
        raise NotImplementedError("The select_action method must be implemented.")
```

#### 接口关键点

| 项目 | 规范 | 说明 |
|------|------|------|
| **输入图像格式** | `(H, W, C), uint8, [0, 255]` | 原始环境输出 |
| **输入状态格式** | `(N,), float32` | 关节位置等 |
| **输出动作格式** | `(N,), float32` | 关节动作 |
| **调用频率** | 每帧一次 (~30Hz) | 需要实时推理 |

### 16.4 LeRobotPolicy 参考实现分析

```python
# scripts/eval_policy/lerobot_policy.py:180-217

def _prepare_for_preprocessor(self, observation_dict):
    """Prepare observation dictionary for LeRobot preprocessor pipeline."""
    obs_for_preproc = {}
    for key, value in observation_dict.items():
        if not key.startswith("observation."):
            continue

        if isinstance(value, np.ndarray):
            value_tensor = torch.from_numpy(value).float()
            if value.ndim == 3 and value.shape[-1] == 3:  # Image: (H, W, C)
                # (H, W, C) -> (C, H, W), [0, 1] normalization
                value_tensor = value_tensor.permute(2, 0, 1).to(self.device) / 255.0
                obs_for_preproc[key] = value_tensor.unsqueeze(0)  # Add batch dim
            else:
                obs_for_preproc[key] = value_tensor.unsqueeze(0)

    # Create transition format with complementary_data for VLA models
    dummy_action = torch.zeros(1, self.action_dim, dtype=torch.float32, device=self.device)
    transition = {
        TransitionKey.OBSERVATION: obs_for_preproc,
        TransitionKey.ACTION: dummy_action,
        TransitionKey.COMPLEMENTARY_DATA: {"task": self.task_description},
    }
    return transition
```

**关键发现**：
1. **图像预处理**：`(H, W, C), uint8, [0, 255]` → `(C, H, W), float32, [0, 1]`
2. **Batch 维度**：所有数据都添加 batch 维度（即使是单样本推理）
3. **VLA 特殊处理**：需要 `COMPLEMENTARY_DATA` 包含 task description

### 16.5 MoE Policy 集成的关键问题点

#### 问题 1: 空窗期 的动作质量

```python
# MoE Policy 中的逻辑
def select_action(self, observation):
    if self._locked_expert_idx is None:
        # Voting 阶段（前 3 帧）
        logits = self._run_router(image)
        self._router_logits_buffer.append(logits)

        if len(self._router_logits_buffer) >= 3:
            self._locked_expert_idx = agg_logits.argmax(-1).item()

        return self._fallback_action(observation)  # ⚠️ 空窗期
```

**风险分析**：

| Fallback 方案 | 优点 | 缺点 | 风险等级 |
|--------------|------|------|----------|
| 零动作 | 安全，简单 | 可能触发限位，garment 不稳定 | 🔴 高 |
| Expert 0 (默认) | 简单 | 如果 Expert 0 错误，前 3 帧动作混乱 | 🟡 中 |
| 投票领先者 (动态) | 最平滑，利用所有信息 | 需要累加 logits | 🟢 低 |
| 原始 SmolVLA | 安全有效 | 需要额外加载模型，增加复杂度 | 🟢 低 |

**推荐方案**：使用当前投票领先者的 Expert

```python
def _fallback_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
    """Voting 阶段的智能 Fallback"""
    if len(self._router_logits_buffer) == 0:
        # 第一帧：使用 Expert 0
        return self._expert_action(observation, expert_idx=0)

    # 使用当前投票领先的 Expert
    agg_logits = torch.stack(self._router_logits_buffer).sum(0)
    leading_expert = agg_logits.argmax(-1).item()
    return self._expert_action(observation, expert_idx=leading_expert)
```

#### 问题 2: Observation 格式匹配

**环境输出 → Policy 输入**：
```
observation_dict = {
    "observation.images.top_rgb": (480, 640, 3), uint8, [0, 255]
    "observation.images.left_rgb": (480, 640, 3), uint8, [0, 255]
    "observation.images.right_rgb": (480, 640, 3), uint8, [0, 255]
    "observation.state": (12,), float32
    "observation.top_depth": (480, 640), float32  # 可选
}
```

**MoE Policy 需要确保**：
1. Router 的图像预处理与 LeRobot Policy 一致
2. 每个 Expert 的预处理与 LeRobot Policy 一致
3. 或者直接复用 LeRobotPolicy 的预处理逻辑

#### 问题 3: PolicyRegistry 创建时的参数传递

```python
# scripts/utils/evaluation.py:275-298
policy_kwargs = {"device": device}

if args.policy_type == "lerobot":
    policy_kwargs.update({
        "policy_path": args.policy_path,
        "dataset_root": args.dataset_root,
        "task_description": args.task_description,
    })
else:
    # For custom policies
    if args.policy_path:
        policy_kwargs["model_path"] = args.policy_path

policy = PolicyRegistry.create(args.policy_type, **policy_kwargs)
```

**MoE Policy 需要的参数**：
- `model_path`: MoE 模型路径
- `device`: 模型推理设备 ("cuda" 推荐，"cpu" 也可用)
  - 注意：Isaac Sim 的 garment 物理仿真必须在 CPU 上运行
  - 但模型推理和视频渲染可以在 GPU 上运行
- 可选：`dataset_root`, `task_description` (如果需要加载 metadata)

### 16.6 MoE Policy 完整实现建议

```python
# scripts/eval_policy/moe_smolvla_policy.py

from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from scripts.eval_policy import PolicyRegistry
from scripts.eval_policy.base_policy import BasePolicy
from scripts.eval_policy.lerobot_policy import LeRobotPolicy


@PolicyRegistry.register("moe_smolvla")
class MoESmolVLAPolicy(BasePolicy):
    """
    MoE SmolVLA Policy for LeHome Challenge

    关键设计：
    - Episode 开始时 Router 投票 3 帧，然后锁定 Expert
    - 切换 garment 时自动 reset，重新开始 Voting
    - 空窗期使用投票领先的 Expert
    """

    GARMENT_TYPES = ["pant_short", "pant_long", "top_long", "top_short"]
    VOTING_FRAMES = 3

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        dataset_root: Optional[str] = None,
        task_description: Optional[str] = "Fold the garment",
        **kwargs
    ):
        super().__init__()
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        self.task_description = task_description

        # 加载模型
        self._load_model(model_path, dataset_root)

        # Episode 状态
        self._reset_episode_state()

    def _load_model(self, model_path: str, dataset_root: Optional[str]):
        """加载 MoE 模型组件"""
        # 1. 加载共享组件 (从 Expert 0 的 checkpoint)
        expert_0_path = self._get_expert_path(model_path, expert_idx=0)
        self._load_shared_components(expert_0_path, dataset_root)

        # 2. 加载 4 个 Expert
        self.experts = nn.ModuleList()
        for i in range(4):
            expert_path = self._get_expert_path(model_path, expert_idx=i)
            expert = self._load_expert(expert_path, i)
            self.experts.append(expert)

        # 3. 加载 Router
        self.router = self._load_router(model_path)

    def _get_expert_path(self, model_path: str, expert_idx: int) -> Path:
        """获取指定 Expert 的 checkpoint 路径"""
        garment_type = self.GARMENT_TYPES[expert_idx]
        return Path(model_path) / "experts" / garment_type / "pretrained_model"

    def _reset_episode_state(self):
        """清除 Episode 状态"""
        self._locked_expert_idx: Optional[int] = None
        self._router_logits_buffer: list = []
        self._voting_count: int = 0

    def reset(self):
        """Episode 开始时调用"""
        self._reset_episode_state()
        # 如果每个 Expert 有内部状态，也需 reset
        # for expert in self.experts:
        #     if hasattr(expert, 'reset'):
        #         expert.reset()

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """核心推理逻辑"""
        # 1. 提取 top_rgb 图像
        image = observation.get("observation.images.top_rgb")
        if image is None:
            raise ValueError("Missing observation.images.top_rgb")

        # 2. Voting 阶段
        if self._locked_expert_idx is None:
            logits = self._run_router(image)
            self._router_logits_buffer.append(logits)
            self._voting_count += 1

            if self._voting_count >= self.VOTING_FRAMES:
                self._locked_expert_idx = self._vote_for_expert()

            # Fallback: 使用当前领先的 Expert
            return self._fallback_action(observation)

        # 3. 锁定后：使用指定 Expert
        return self._expert_action(observation, self._locked_expert_idx)

    def _vote_for_expert(self) -> int:
        """投票决定 Expert"""
        agg_logits = torch.stack(self._router_logits_buffer).sum(dim=0)
        expert_idx = agg_logits.argmax(dim=-1).item()
        garment_type = self.GARMENT_TYPES[expert_idx]
        print(f"[MoE] Locked Expert {expert_idx} ({garment_type})")
        return expert_idx

    def _fallback_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Voting 阶段的 Fallback"""
        if len(self._router_logits_buffer) == 0:
            # 第一帧：使用 Expert 0
            return self._expert_action(observation, expert_idx=0)

        # 使用当前投票领先的 Expert
        agg_logits = torch.stack(self._router_logits_buffer).sum(0)
        leading_expert = agg_logits.argmax(-1).item()
        return self._expert_action(observation, expert_idx=leading_expert)

    def _expert_action(self, observation: Dict[str, np.ndarray], expert_idx: int) -> np.ndarray:
        """使用指定 Expert 进行推理"""
        # 复用 LeRobotPolicy 的预处理逻辑
        # 这里需要调用对应 Expert 的 select_action
        # 具体实现取决于 Expert 的加载方式
        pass
```

### 16.7 潜在错误场景与缓解方案

| 场景 | 可能的错误 | 缓解方案 |
|------|-----------|----------|
| Router 分类错误 | 锁定错误的 Expert | Voting 机制 + 投票领先者 Fallback |
| 空窗期动作不稳定 | garment 位置偏移 | 使用投票领先的 Expert（而非固定 Expert 0） |
| 图像格式不匹配 | Router 报错 | 复用 LeRobotPolicy 预处理 |
| reset() 未正确实现 | 状态泄漏到下一个 episode | 确保 `_reset_episode_state()` 清除所有状态 |
| 切换 garment 后状态未清除 | 使用上一个 garment 的 Expert | Garment 切换时会自动调用 `reset()` |
| 显存不足 | OOM | Expert offloading 或 LoRA Expert |
| first frame 的 Router 输入 | buffer 为空 | 特殊处理：第一帧使用 Expert 0 |

### 16.8 评估命令示例

```bash
# 评估 MoE Policy (GPU 推理)
python -m scripts.eval \
    --policy_type moe_smolvla \
    --policy_path outputs/moe_smolvla/checkpoints/last/pretrained_model \
    --garment_type custom \
    --dataset_root Datasets/example/four_types_merged \
    --task_description "Fold the garment neatly" \
    --num_episodes 10 \
    --enable_cameras \
    --device cuda
```

**参数说明**：
- `--policy_type moe_smolvla`: 使用注册的 MoE Policy
- `--device cuda`: 模型在 GPU 上推理 (推荐，速度快)
  - 也可以用 `--device cpu` (较慢，但兼容性更好)
  - Isaac Sim 的 garment 物理仿真始终在 CPU 上运行
- `--policy_path`: MoE 模型路径
- `--garment_type custom`: 测试所有 4 种服装类型
- `--dataset_root`: 用于加载 metadata（如果需要）

### 16.9 实现验证清单

```
┌─────────────────────────────────────────────────────────────────┐
│                    MoE Policy 实现验证清单                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  接口兼容性:                                                     │
│  □ 继承 BasePolicy                                              │
│  □ @PolicyRegistry.register("moe_smolvla") 装饰器               │
│  □ __init__() 接受 model_path, device 等参数                     │
│  □ reset() 清除所有状态                                         │
│  □ select_action() 输入输出格式正确                              │
│                                                                   │
│  功能完整性:                                                     │
│  □ Router 可以正确运行（冻结 VLM + 4层 MLP）                     │
│  □ Voting 机制正确（3帧后锁定）                                  │
│  □ Expert 切换正确                                              │
│  □ Fallback 策略合理（投票领先者）                              │
│                                                                   │
│  状态管理:                                                       │
│  □ _reset_episode_state() 清除所有 MoE 状态                     │
│  □ reset() 在 episode 开始时被调用                              │
│  □ reset() 在 garment 切换时被调用                              │
│  □ Episode 之间状态隔离正确                                     │
│                                                                   │
│  边界情况:                                                       │
│  □ 第一帧的行为（buffer 为空）                                  │
│  □ Episode 之间的状态隔离                                       │
│  □ Garment 切换时的 reset                                        │
│  □ 图像格式匹配 (H, W, C), uint8, [0, 255]                      │
│                                                                   │
│  性能验证:                                                       │
│  □ 推理延迟 < 100ms/frame                                        │
│  □ 显存占用 < 4GB                                               │
│  □ 无动作抖动                                                   │
│                                                                   │
│  集成测试:                                                       │
│  □ 4 种服装各跑 10+ episodes                                    │
│  □ 成功率 vs 单独训练对比                                       │
│  □ Garment 切换测试                                             │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 16.10 与 LeRobotPolicy 的集成方案

**方案 A：完全独立实现**
- 优点：完全控制，无依赖
- 缺点：需要重写预处理逻辑

**方案 B：继承 LeRobotPolicy**
- 优点：复用预处理和后处理逻辑
- 缺点：需要仔细处理 `self.model` 的替换

```python
@PolicyRegistry.register("moe_smolvla")
class MoESmolVLAPolicy(LeRobotPolicy):
    def __init__(self, model_path: str, **kwargs):
        # 调用父类初始化（加载第一个 Expert 作为基础）
        expert_0_path = self._get_expert_path(model_path, 0)
        super().__init__(policy_path=expert_0_path, **kwargs)

        # 替换为 MoE 模型
        self.model = self._build_moe_model(model_path)

    def _build_moe_model(self, model_path: str):
        """构建 MoE 模型"""
        # 复用父类的 VLM、preprocessor、postprocessor
        # 添加 Router 和多个 Expert
        pass
```

**推荐**：优先使用方案 B，复用 LeRobotPolicy 的成熟实现。

---

## 十七、SmolVLA 模型组件与推理流程详解

> [!IMPORTANT]
> 本节详细讲解 SmolVLA 从输入到输出的完整数据流，以及各组件独立性决策的技术原理

### 17.1 完整数据流：从 RGB 图像到机械臂动作

以 `pant_short`（短裤）折叠任务为例，展示完整的推理过程。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SmolVLA 完整数据流                               │
│                     (pant_short 折叠任务示例)                              │
└─────────────────────────────────────────────────────────────────────────┘
```

### 17.2 输入数据准备

```python
# 原始输入（来自环境）
observation = {
    "observation.images.top_rgb": (480, 640, 3),  # uint8, [0, 255]
    "observation.images.left_rgb": (480, 640, 3),  # uint8, [0, 255]
    "observation.images.right_rgb": (480, 640, 3), # uint8, [0, 255]
    "observation.state": (12,),                   # float32 双臂关节位置
    "task": "Fold the garment neatly"              # 文本指令
}
```

### 17.3 图像预处理与 Token 化（共享组件）

```python
# 1. 图像归一化和重排
image = observation["observation.images.top_rgb"]  # (480, 640, 3), uint8
image_normalized = image.permute(2, 0, 1).float() / 255.0  # (3, 480, 640)
image_resized = F.interpolate(image_normalized, size=(512, 512))  # (3, 512, 512)

# 2. PixelShuffle 压缩视觉 Token（关键优化）
# 512×512×3 = 786,432 像素 → 64 个视觉 token
patches = rearrange(image_resized, 'c (h p1) (w p2) -> (h w) c p1 p2', p1=64, p2=64)
image_tokens = patches.mean(dim=(-2, -1))  # (64, 3)
```

**关键点**：
- ✅ **共享原因**：图像压缩方式对所有任务相同
- 输入：(3, 512, 512) → 输出：64 个视觉 token

### 17.4 VLM 视觉编码器（共享组件 - SigLIP）

```python
# SigLIP Vision Encoder (冻结)
vision_embeddings = vlm.vision_model(image_tokens)  # (64, 960)
```

**可视化过程**：

```
┌─────────────────────────────────────────────────────────────┐
│  SigLIP Vision Encoder (冻结，共享)                          │
│                                                              │
│  输入: 64 个图像 patch，每个 3×64×64                        │
│         ↓                                                    │
│  [Conv Layers → Attention → MLP] × N 层                      │
│         ↓                                                    │
│  输出: 64 个 embedding 向量，每个 960 维                      │
│                                                              │
│  示例:                                                       │
│    patch_0 → [0.2, -0.1, 0.5, ...]  # "衣物左上角"             │
│    patch_1 → [0.3, 0.1, -0.2, ...]  # "衣物中心"               │
└─────────────────────────────────────────────────────────────┘
```

**独立性决策**：
- ✅ **共享原因**：视觉理解能力通用
- 识别"这是衣物"、"这是桌子"的能力对所有任务相同
- SigLIP 在大规模数据上预训练，已学会通用视觉特征

### 17.5 状态投影（共享组件 - state_proj）

```python
state = observation["observation.state"]  # (12,) 双臂各 6 个关节
# 例如: [0.5, 0.3, -0.2, 0.1, 0.0, 0.8,  # 左臂
#        0.5, 0.3, -0.2, 0.1, 0.0, 0.8]  # 右臂

state_token = state_proj(state)  # (12,) -> (960,)
```

**独立性决策**：
- ✅ **共享原因**：本体感觉语义固定
  - `dim 0` = 左臂肩部旋转角（所有任务相同）
  - `dim 1` = 左臂肩部抬升角（所有任务相同）
  - state_proj 只是"翻译"，告诉 VLM "手臂在哪里"

**类比**：
```
state_proj 就像是一个"字典翻译器"：
- 输入："左肩在 0.5 弧度"
- 输出：VLM 能理解的 960 维向量，表示"左肩在 0.5 弧度"

这个翻译对所有任务都一样，所以可以共享！
```

### 17.6 VLM 语言模型处理（共享组件）

```python
# 拼接所有 tokens
all_tokens = torch.cat([
    vision_embeddings,    # (64, 960)  视觉
    state_token[None,:],  # (1, 960)   状态
    text_embeddings,      # (N, 960)   文本
], dim=0)  # (65+N, 960)

# 通过 VLM 的 Transformer 层
vlm_output = vlm.text_model(all_tokens)  # (65+N, 960)
```

**可视化过程**：

```
┌─────────────────────────────────────────────────────────────────┐
│           VLM Language Model (16层 Transformer)                 │
│                     (冻结，共享)                                │
│                                                                 │
│  输入序列:                                                      │
│  [Vision_0, ..., Vision_63, State, Text_0, Text_1, ...]         │
│         ↓                                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Layer 1-16: Cross-Attention + Self-Attention           │    │
│  │  Vision tokens ←attend→ Text tokens                      │    │
│  │  Vision tokens ←attend→ State token                      │    │
│  │  (模型学会理解："我看到的是短裤，要折叠它")                │    │
│  └─────────────────────────────────────────────────────────┘    │
│         ↓                                                        │
│  输出: (65+N, 960) - 融合了视觉、状态、文本的上下文向量           │
└─────────────────────────────────────────────────────────────────┘
```

**独立性决策**：
- ✅ **共享原因**："理解图像中是什么"、"理解指令含义"的能力通用
- VLM 输出的是"我看到一条短裤在桌子上，需要折叠"这样的**语义表示**

### 17.7 关键分歧点：lm_expert 的 Cross-Attention（独立组件）

```python
# ====== 这里开始，不同 Expert 有独立的路径 ======

# VLM 输出: (65, 960) - 所有 Expert 共享这个
vlm_hidden_states = vlm_output

# Expert 0 (pant_short) 的投影
K_0 = expert_0.layers[0].self_attn.k_proj(vlm_hidden_states)  # (65, 480)
V_0 = expert_0.layers[0].self_attn.v_proj(vlm_hidden_states)  # (65, 480)

# Expert 1 (pant_long) 的投影（不同权重！）
K_1 = expert_1.layers[0].self_attn.k_proj(vlm_hidden_states)  # (65, 480)
V_1 = expert_1.layers[0].self_attn.v_proj(vlm_hidden_states)  # (65, 480)
```

**可视化对比**：

```
┌─────────────────────────────────────────────────────────────────┐
│               Expert 0 (pant_short) 的 lm_expert                │
│                                                                 │
│  VLM 输出: (65, 960)                                           │
│         ↓                                                        │
│  Expert_0.k_proj: (960 → 480)                                  │
│  Expert_0.v_proj: (960 → 480)                                  │
│         ↓                                                        │
│  Cross-Attention: Action tokens attend to VLM features          │
│         ↓                                                        │
│  输出: 专门针对"向下折叠短裤"优化的特征                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│               Expert 1 (pant_long) 的 lm_expert                 │
│                                                                 │
│  VLM 输出: (65, 960) ← 相同！                                    │
│         ↓                                                        │
│  Expert_1.k_proj: (960 → 480) ← 不同权重！                      │
│  Expert_1.v_proj: (960 → 480) ← 不同权重！                      │
│         ↓                                                        │
│  输出: 专门针对"向下折叠长裤"优化的特征（需要处理更长的裤腿）   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│               Expert 2 (top_long) 的 lm_expert                  │
│                                                                 │
│  VLM 输出: (65, 960) ← 相同！                                    │
│         ↓                                                        │
│  Expert_2.k_proj: (960 → 480) ← 不同权重！                      │
│  Expert_2.v_proj: (960 → 480) ← 不同权重！                      │
│         ↓                                                        │
│  输出: 专门针对"向内折叠上衣"优化的特征                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**独立性决策**：
- ⚠️ **独立原因**：每个专家需要从 VLM 输出中提取**任务特定的信息**
- `pant_short` 专家关注："短裤在哪里？如何向下折叠？"
- `top_long` 专家关注："上衣在哪里？如何向内折叠？"
- 这些关注点通过 `k_proj`, `v_proj` 的权重来编码

**类比**：
```
想象三个不同的专家在看同一张照片：

- 短裤专家："我看到裤脚在这里，需要从两边向中间折"
- 长裤专家："我看到裤腿很长，需要先对折再向上折"
- 上衣专家："我看到袖子在这里，需要向内对折"

他们看到的是同一张照片（VLM 输出相同），但关注点和理解完全不同！
```

### 17.8 动作编码（独立组件 - action_in_proj）

```python
# 当前动作 (用于 Flow Matching 的条件)
current_action = torch.randn(12)  # 12维动作噪声

# Expert 0 的 action_in_proj
action_embedding_0 = action_in_proj_dict["pant_short"](current_action)
# (12,) -> (480,)

# Expert 1 的 action_in_proj（不同权重！）
action_embedding_1 = action_in_proj_dict["pant_long"](current_action)
# (12,) -> (480,)
```

**独立性决策**：
- ⚠️ **独立原因**：不同任务的**动作分布差异大**

**数据支持**（来自文档 5.4.1 节）：

```
不同服装类型的关节位置分布差异：

| 维度 | 关节 | pant_short | pant_long | top_long | top_short | 最大差异 |
|------|------|------------|-----------|----------|-----------|----------|
| 2 | L_elbow_flex | 0.531 | 0.941 | 0.752 | 0.736 | **0.410** |
| 7 | R_shoulder_lift | -0.683 | -0.285 | -0.762 | -0.784 | **0.499** |

→ 不同任务使用关节的方式完全不同！
→ 需要独立的 action_in_proj 来编码这些差异
```

### 17.9 时间编码（共享组件 - action_time_mlp）

```python
# Flow Matching 的时间步
sigma = torch.tensor([0.5])  # 噪声水平

# action_time_mlp_in: 编码时间信息
time_embedding = action_time_mlp_in(sigma)  # (1,) -> (480,)
```

**独立性决策**：
- ✅ **共享原因**：Flow Matching 的时间编码机制通用
- "如何从噪声逐步恢复到真实动作"的数学原理对所有任务相同

### 17.10 lm_expert 的 Self-Attention 和前馈层（独立组件）

```python
# 拼接 action embedding 和 time embedding
expert_input = torch.cat([action_embedding_0, time_embedding], dim=-1)  # (960,)

# 通过 lm_expert 的 Transformer 层
for layer in expert_0.layers:
    # Self-Attention: action tokens attend to each other
    attn_out = layer.self_attn(expert_input)

    # Feed-Forward Network
    ffn_out = layer.mlp(attn_out)

    expert_input = expert_input + attn_out + ffn_out

# 输出: (480,) - 任务特定的动作规划特征
expert_features_0 = expert_input
```

**独立性决策**：
- ⚠️ **独立原因**：学习任务特定的**动作序列模式**
- `pant_short`: "向下折叠 → 抓住 → 再向下折"
- `top_long`: "向内折叠 → 抓住袖子 → 再向内折"

### 17.11 动作解码（独立组件 - action_out_proj）

```python
# Expert 0 的 action_out_proj
predicted_action_0 = action_out_proj_dict["pant_short"](expert_features_0)
# (480,) -> (12,)

# Expert 1 的 action_out_proj（不同权重！）
predicted_action_1 = action_out_proj_dict["pant_long"](expert_features_1)
# (480,) -> (12,)
```

**独立性决策**：
- ⚠️ **独立原因**：将抽象特征转换为具体关节指令
- 不同任务的"肌肉协调模式"完全不同
- `pant_short`: 双手协调向下（相关性高 r≈0.73）
- `top_long`: 双手相对独立向内（相关性低 r≈0.31）

### 17.12 时间解码（共享组件 - action_time_mlp_out）

```python
# Flow Matching 的"去噪"步骤
final_action = action_time_mlp_out(predicted_action_0, sigma)
# (12,) -> (12,)
```

**独立性决策**：
- ✅ **共享原因**：Flow Matching 的数学原理通用

### 17.13 组件独立性决策总结

| 组件 | 独立性 | 原因 |
|------|--------|------|
| **VLM (SmolVLM)** | ✅ 共享 | 视觉理解能力通用，不需要针对衣物类型重新学习 |
| **state_proj** | ✅ 共享 | 本体感觉是通用的，只报告"手臂在哪里"，不决定"如何行动" |
| **lm_expert** | ⚠️ 独立 | 动作规划专精，不同衣物需要完全不同的策略 |
| **action_in_proj** | ⚠️ 独立 | 动作编码方式不同，不同任务的动作分布差异大 |
| **action_out_proj** | ⚠️ 独立 | 动作输出方式不同，肌肉协调模式各异 |
| **action_time_mlp** | ✅ 共享 | 时间编码是通用的 Flow Matching 机制 |

### 17.14 完整流程图总结

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SmolVLA 数据流全景图                             │
│                  (pant_short 折叠任务示例)                          │
└─────────────────────────────────────────────────────────────────────┘

输入层
├── Image (480×640×3, RGB) ──────────────┐
├── State (12D joint positions) ────────────┤
└── Text "Fold the garment" ────────────────┘
                                         ↓
                    【共享组件：VLM 特征提取】
                                         ↓
┌─────────────────────────────────────────────────────────────────────┐
│  VLM (SmolVLM2) - 冻结，共享                                      │
│  ├── SigLIP Vision Encoder: 图像 → 64×960 vision tokens           │
│  ├── state_proj: 12D state → 1×960 state token                   │
│  └── Text Model: 多模态融合 → (65+N)×960 VLM 输出                  │
└─────────────────────────────────────────────────────────────────────┘
                                         ↓
                    【VLM 输出：语义表示】
                  "我看到一条短裤在桌子上，需要折叠"
                                         ↓
                    【关键分歧点：4个独立 Expert】
              ┌────────────┬────────────┬────────────┬────────────┐
              ↓            ↓            ↓            ↓
         Expert 0    Expert 1    Expert 2    Expert 3
      (pant_short) (pant_long) (top_long) (top_short)
              ↓            ↓            ↓            ↓
    ┌──────────────────────────────────────────────────────────┐
    │  每个 Expert 的独立组件:                                    │
    │                                                           │
    │  1. k_proj, v_proj (重新投影 VLM 输出)                     │
    │     → 提取任务特定的关注点                                  │
    │                                                           │
    │  2. action_in_proj (编码当前动作)                          │
    │     → 适配不同任务的动作分布                                │
    │                                                           │
    │  3. lm_expert.layers (动作规划)                             │
    │     → 学习任务特定的动作序列模式                            │
    │                                                           │
    │  4. action_out_proj (输出解码)                             │
    │     → 转换为任务特定的关节指令                               │
    └──────────────────────────────────────────────────────────┘
              ↓            ↓            ↓            ↓
         Action 0      Action 1     Action 2     Action 3
      (12D, 向下折)  (12D, 向下折) (12D, 向内折) (12D, 向内折)
                                         ↓
                    【共享组件：Flow Matching 后处理】
                                         ↓
                            Final Action (12D)
```

### 17.15 为什么这种设计有效？

| 设计决策 | 效果 | 原因 |
|---------|------|------|
| **共享 VLM** | 节省显存，避免重复学习视觉理解 | 视觉理解通用 |
| **共享 state_proj** | 减少参数，防止过拟合 | 本体感觉语义固定 |
| **独立 lm_expert** | 消除梯度冲突 | 任务特定的动作规划 |
| **独立 action_*_proj** | 适配不同动作分布 | 动作模式差异大 |
| **共享 action_time_mlp** | 保持 Flow Matching 机制 | 数学原理通用 |

这种设计既保证了效率（共享组件），又保证了性能（独立组件），是 MoE 方案成功的关键！

---

## 十八、MoE Policy 代码架构设计

> [!IMPORTANT]
> 本节详细设计 MoESmolVLAPolicy 的完整代码架构，确保结构标准、优雅，并符合比赛 eval 框架要求

### 18.1 核心设计原则

```
┌─────────────────────────────────────────────────────────────────┐
│                    设计原则                                       │
├─────────────────────────────────────────────────────────────────┤
│  1. 符合 BasePolicy 接口规范                                      │
│  2. 复用 LeRobotPolicy 的预处理逻辑                               │
│  3. 模块化设计：Router、Expert、Manager 分离                      │
│  4. 状态管理清晰：Episode 级别隔离                                │
│  5. Checkpoint 格式标准：单一模型目录                             │
└─────────────────────────────────────────────────────────────────┘
```

### 18.2 完整架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MoESmolVLAPolicy                                 │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     Public Interface                            │    │
│  │  • __init__(model_path, device, **kwargs)                       │    │
│  │  • reset() - 清除 Episode 状态                                   │    │
│  │  • select_action(observation) -> action                         │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              ↓                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                  Component Manager                              │    │
│  │  • shared_vlm: SmolVLM2 (冻结)                                   │    │
│  │  • experts: Dict[garment_type, Expert]                          │    │
│  │  • router: GarmentRouter                                        │    │
│  │  • preprocessor: LeRobot preprocessor                           │    │
│  │  • postprocessor: LeRobot postprocessor                         │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              ↓                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                   State Manager                                  │    │
│  │  • _locked_expert_idx: Optional[int]                            │    │
│  │  • _router_logits_buffer: List[Tensor]                          │    │
│  │  • _voting_count: int                                           │    │
│  │  • reset_state() - 清除所有状态                                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              ↓                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Execution Flow                               │    │
│  │  1. preprocess observation                                      │    │
│  │  2. if not locked: router.classify() + voting                   │    │
│  │  3. expert_action = experts[locked_idx].select_action()         │    │
│  │  4. postprocess action                                          │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 18.3 核心类：MoESmolVLAPolicy

```python
@PolicyRegistry.register("moe_smolvla")
class MoESmolVLAPolicy(BasePolicy):
    """
    MoE SmolVLA Policy for LeHome Challenge.

    架构：
    - 共享 VLM (SmolVLM2-500M, 冻结)
    - 4 个独立 Expert (每种服装类型一个)
    - Router (基于视觉的分类器)
    - Episode 级别的 Expert 锁定机制

    使用示例：
    ```python
    policy = MoESmolVLAPolicy(
        model_path="outputs/moe_smolvla/checkpoints/last/pretrained_model",
        device="cpu"
    )
    action = policy.select_action(observation)
    ```
    """

    # 配置常量
    GARMENT_TYPES = ["pant_short", "pant_long", "top_long", "top_short"]
    VOTING_FRAMES = 3

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dataset_root: Optional[str] = None,
        task_description: Optional[str] = "Fold the garment",
        **kwargs
    ):
        """
        初始化 MoE Policy。

        Args:
            model_path: MoE 模型路径
                     期望结构: model_path/experts/{garment_type}/pretrained_model/
            device: 模型推理设备 ("cuda" 推荐, "cpu" 也可用)
                     注意：Isaac Sim 的 garment 物理仿真必须在 CPU 上运行，
                     但模型推理和视频渲染可以在 GPU 上运行
            dataset_root: 数据集路径 (用于加载 metadata)
            task_description: 任务描述
        """
        super().__init__()

        # 设备配置
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        self.task_description = task_description

        logger.info(f"Initializing MoESmolVLAPolicy from {model_path}")

        # 初始化组件
        self._load_shared_components()
        self._load_experts()
        self._load_router()
        self._load_processors(dataset_root)

        # 初始化状态
        self._reset_episode_state()

        logger.info(f"MoESmolVLAPolicy initialized on {self.device}")
        logger.info(f"Loaded {len(self.GARMENT_TYPES)} experts")

    def reset(self):
        """
        清除 Episode 状态。

        在以下情况调用：
        1. 每个 Episode 开始时
        2. 切换 garment 时
        """
        self._reset_episode_state()

        # 清除 Expert 的内部状态（如果有的话）
        for expert in self.experts.values():
            if hasattr(expert, 'reset'):
                expert.reset()

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """
        生成动作。

        执行流程：
        1. 预处理 observation
        2. 提取 VLM 特征
        3. Router 分类 (如果未锁定)
        4. Expert 推理
        5. 后处理 action

        Args:
            observation: 环境观测
                - observation.images.top_rgb: (480, 640, 3), uint8, [0, 255]
                - observation.state: (12,), float32

        Returns:
            action: (12,), float32
        """
        # 1. 预处理
        batch_obs = self._preprocess_observation(observation)

        # 2. 提取 VLM 特征 (共享)
        with torch.inference_mode():
            image = batch_obs["observation.images.top_rgb"]
            vision_embeddings = self.shared_vlm.embed_image(image)  # (B, 64, 960)

        # 3. Router 分类 (如果未锁定)
        if not self._is_locked():
            router_logits = self.router(vision_embeddings)  # (B, 4)
            self._add_vote(router_logits[0])

            if self._is_locked():
                garment_type = self.GARMENT_TYPES[self._locked_expert_idx]
                logger.info(f"[MoE] Locked Expert: {garment_type}")

        # 4. 选择 Expert
        if self._is_locked():
            expert_idx = self._locked_expert_idx
        else:
            expert_idx = self._get_fallback_expert_idx()

        # 5. Expert 推理
        with torch.inference_mode():
            expert = self._get_expert_by_idx(expert_idx)
            batch_action = self._run_expert(expert, batch_obs, vision_embeddings)

        # 6. 后处理
        if self.postprocessor:
            batch_action = self.postprocessor(batch_action)

        return batch_action.squeeze(0).cpu().numpy()
```

### 18.4 Router 组件：GarmentRouter

```python
class GarmentRouter(nn.Module):
    """
    基于 VLM 特征的服装分类器。

    设计：
    - 输入：VLM vision embeddings (B, 64, 960)
    - 输出：4 个类别的 logits (B, 4)
    - 架构：池化 + 3 层 MLP (960 → 512 → 256 → 4)

    实验结果：
    - 单摄像头 top_rgb 准确率：100%
    - 参数量：~1.5M
    """

    def __init__(
        self,
        vlm_hidden_size: int = 960,
        num_vision_tokens: int = 64,
        num_classes: int = 4,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.1
    ):
        super().__init__()

        # 池化层：压缩视觉 tokens
        self.token_pooler = nn.AdaptiveAvgPool1d(1)

        # MLP 分类器
        layers = []
        prev_dim = vlm_hidden_size
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, vision_embeddings: Tensor) -> Tensor:
        """
        Args:
            vision_embeddings: (B, 64, 960) VLM vision encoder 输出

        Returns:
            logits: (B, 4) 每个类别的 logit
        """
        # 池化视觉 tokens
        pooled = self.token_pooler(vision_embeddings.transpose(1, 2))
        pooled = pooled.squeeze(-1)  # (B, 960)

        # 分类
        logits = self.classifier(pooled)
        return logits
```

### 18.5 Expert Manager：ExpertManager

```python
class ExpertManager:
    """
    管理多个 SmolVLA Expert。

    功能：
    - 加载 4 个预训练的 Expert
    - 共享 VLM (冻结)
    - 独立的 lm_expert 和 action 投影层

    架构：
    ┌─────────────────────────────────────────────────────────┐
    │  shared_vlm: SmolVLM2 (冻结, ~500M 参数)                  │
    │     ↓                                                    │
    │  experts: ModuleDict                                      │
    │    ├── pant_short: lm_expert (~42M 参数)                   │
    │    ├── pant_long: lm_expert (~42M 参数)                    │
    │    ├── top_long: lm_expert (~42M 参数)                    │
    │    └── top_short: lm_expert (~42M 参数)                   │
    └─────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        model_path: Path,
        garment_types: List[str],
        device: torch.device,
        shared_vlm: Optional[nn.Module] = None
    ):
        self.model_path = model_path
        self.garment_types = garment_types
        self.device = device
        self.shared_vlm = shared_vlm

        # 加载 Experts
        self.experts = nn.ModuleDict()
        self._load_experts()

    def _load_experts(self):
        """加载所有 Expert"""
        for garment_type in self.garment_types:
            expert_path = self.model_path / "experts" / garment_type / "pretrained_model"
            logger.info(f"Loading Expert for {garment_type} from {expert_path}")

            # 加载 SmolVLA Policy
            policy_cfg = PreTrainedConfig.from_pretrained(expert_path)
            policy = make_policy(policy_cfg)
            policy.eval()
            policy.to(self.device)

            # 提取 lm_expert (不包含 VLM)
            self.experts[garment_type] = policy.model.lm_expert

            # 如果需要共享 VLM，提取第一个 Expert 的 VLM
            if self.shared_vlm is None and garment_type == self.garment_types[0]:
                self.shared_vlm = policy.model.vlm

    def get_expert(self, garment_type: str) -> nn.Module:
        """获取指定 Expert"""
        return self.experts[garment_type]

    def get_expert_by_idx(self, idx: int) -> nn.Module:
        """通过索引获取 Expert"""
        garment_type = self.garment_types[idx]
        return self.experts[garment_type]
```

### 18.6 状态管理：EpisodeStateManager

```python
class EpisodeStateManager:
    """
    管理 Episode 级别的状态。

    状态：
    - locked_expert_idx: 锁定的 Expert 索引
    - router_logits_buffer: Router 投票缓冲
    - voting_count: 当前投票计数

    关键设计：
    - Voting 期间使用当前领先的 Expert 作为 Fallback
    - Episode 切换时自动清除状态
    - 状态与推理逻辑解耦
    """

    def __init__(self, voting_frames: int = 3):
        self.voting_frames = voting_frames
        self.locked_expert_idx: Optional[int] = None
        self.router_logits_buffer: List[Tensor] = []
        self.voting_count: int = 0

    def reset(self):
        """清除所有状态"""
        self.locked_expert_idx = None
        self.router_logits_buffer = []
        self.voting_count = 0
        logger.debug("[StateManager] State reset")

    def is_locked(self) -> bool:
        """检查是否已锁定 Expert"""
        return self.locked_expert_idx is not None

    def add_vote(self, logits: Tensor) -> bool:
        """
        添加一次投票。

        Args:
            logits: (4,) Router 输出的 logits

        Returns:
            bool: 是否达到锁定条件
        """
        self.router_logits_buffer.append(logits.detach())
        self.voting_count += 1

        if self.voting_count >= self.voting_frames:
            self._lock_expert()
            return True
        return False

    def _lock_expert(self):
        """根据投票结果锁定 Expert"""
        agg_logits = torch.stack(self.router_logits_buffer).sum(dim=0)
        self.locked_expert_idx = agg_logits.argmax(dim=-1).item()
        logger.debug(f"[StateManager] Locked Expert {self.locked_expert_idx}")

    def get_fallback_expert_idx(self) -> int:
        """
        获取 Fallback Expert 索引。

        策略：使用当前投票领先的 Expert
        """
        if len(self.router_logits_buffer) == 0:
            return 0  # 第一帧：默认 Expert 0

        agg_logits = torch.stack(self.router_logits_buffer).sum(dim=0)
        return agg_logits.argmax(dim=-1).item()
```

### 18.7 完整的 select_action 流程

```python
def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
    """
    完整的推理流程。

    时间线：
    t=0: 预处理 → VLM特征 → Router → Expert0 → Action
    t=1: 预处理 → VLM特征 → Router → Expert领先者 → Action
    t=2: 预处理 → VLM特征 → Router → 投票完成 → 锁定Expert → Action
    t=3+: 预处理 → VLM特征 → 跳过Router → 锁定Expert → Action
    """
    # 1. 预处理
    batch_obs = self.preprocessor(observation)

    # 2. 提取 VLM 特征 (共享，只计算一次)
    with torch.inference_mode():
        image = batch_obs["observation.images.top_rgb"]
        vision_embeddings = self.shared_vlm.embed_image(image)

    # 3. Router 分类 (如果未锁定)
    if not self.state_manager.is_locked():
        router_logits = self.router(vision_embeddings)
        self.state_manager.add_vote(router_logits[0])

    # 4. 选择 Expert
    if self.state_manager.is_locked():
        expert_idx = self.state_manager.locked_expert_idx
    else:
        expert_idx = self.state_manager.get_fallback_expert_idx()

    # 5. Expert 推理
    with torch.inference_mode():
        expert = self.expert_manager.get_expert_by_idx(expert_idx)
        batch_action = self._run_expert(expert, batch_obs, vision_embeddings)

    # 6. 后处理
    if self.postprocessor:
        batch_action = self.postprocessor(batch_action)

    return batch_action.squeeze(0).cpu().numpy()
```

### 18.8 Checkpoint 格式设计

```
moe_smolvla/
├── config.json                    # MoE 配置
├── shared/
│   └── vlm/                       # 共享 VLM (可选引用)
├── router/
│   ├── model.safetensors          # Router 权重
│   └── config.json                # Router 配置
└── experts/
    ├── pant_short/
    │   └── pretrained_model/      # 完整的 SmolVLA checkpoint
    ├── pant_long/
    │   └── pretrained_model/
    ├── top_long/
    │   └── pretrained_model/
    └── top_short/
        └── pretrained_model/
```

### 18.9 关键设计决策

#### 决策 1：复用 LeRobotPolicy vs 独立实现

```python
# ✅ 推荐：部分复用
class MoESmolVLAPolicy(BasePolicy):
    def __init__(self, model_path, **kwargs):
        # 复用第一个 Expert 的 preprocessor/postprocessor
        expert_0_path = self._get_expert_path(model_path, 0)
        self._load_processors(expert_0_path)

        # 自己实现 MoE 逻辑
        self._build_moe_components(model_path)
```

| 方案 | 优点 | 缺点 | 推荐 |
|------|------|------|------|
| **完全复用 LeRobotPolicy** | 最少代码 | 需要替换 self.model，容易出错 | ❌ |
| **部分复用** | 平衡代码量和控制力 | 需要手动加载 processors | ✅ |
| **完全独立** | 最大控制力 | 重复代码多，易出错 | ❌ |

#### 决策 2：Router 输入来源

```python
# ✅ 推荐：只使用视觉特征
router_logits = self.router(vision_embeddings)

# ❌ 不推荐：加入 state 或 task description
# router_logits = self.router([vision_embeddings, state, task])
```

**原因**：
- 文档实验证明单摄像头准确率 100%
- 简化 Router 架构，减少参数
- 避免多模态融合的复杂性

#### 决策 3：Voting 缓冲区的 Fallback 策略

```python
# ✅ 推荐：使用当前投票领先的 Expert
def get_fallback_expert_idx(self) -> int:
    if len(self.router_logits_buffer) == 0:
        return 0  # 第一帧

    agg_logits = torch.stack(self.router_logits_buffer).sum(0)
    return agg_logits.argmax(-1).item()

# ❌ 不推荐：固定使用 Expert 0
# return 0
```

**原因**：
- 利用已有的 Router 信息
- 减少 Fallback 动作的错误率
- 平滑过渡到锁定状态

#### 决策 4：Checkpoint 结构

```python
# ✅ 推荐：标准目录结构
moe_model/
├── config.json           # MoE 配置
├── router/              # Router 权重
└── experts/             # 4 个 Expert

# ❌ 不推荐：单一大文件
# moe_model.safetensors  # 所有权重合并
```

**原因**：
- 每个 Expert 是完整的 SmolVLA checkpoint
- 可以单独使用每个 Expert
- 符合"单一模型文件"的提交要求（整体作为目录）

### 18.10 与 Eval 框架的兼容性检查

| 检查项 | 要求 | 实现 | 状态 |
|--------|------|------|------|
| **BasePolicy 继承** | 必须继承 | `class MoESmolVLAPolicy(BasePolicy)` | ✅ |
| **select_action 接口** | `select_action(obs) -> action` | 实现 | ✅ |
| **reset 接口** | `reset()` 清除状态 | `StateManager.reset()` | ✅ |
| **输入格式** | `(H,W,C), uint8, [0,255]` | 复用 LeRobot 预处理 | ✅ |
| **输出格式** | `(N,), float32` | 复用 LeRobot 后处理 | ✅ |
| **reset() 调用时机** | Episode 开始、garment 切换 | 框架自动调用 | ✅ |
| **状态隔离** | Episode 之间状态独立 | `reset()` 清除所有状态 | ✅ |

### 18.11 实现优先级

```
┌─────────────────────────────────────────────────────────────────┐
│                    实现优先级                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  P0 (核心功能):                                                  │
│  □ MoESmolVLAPolicy 基本框架                                     │
│  □ Expert 加载和管理                                             │
│  □ Router 分类                                                   │
│  □ Voting 和锁定机制                                             │
│  □ 与 eval 框架的集成                                            │
│                                                                   │
│  P1 (优化功能):                                                  │
│  □ 智能Fallback 策略                                             │
│  □ 性能优化 (KV Cache 共享)                                      │
│  □ 错误处理和日志                                               │
│                                                                   │
│  P2 (增强功能):                                                  │
│  □ Router 训练脚本                                               │
│  □ Expert 合并工具                                               │
│  □ 性能分析工具                                                 │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 18.12 文件结构

```
scripts/eval_policy/
├── __init__.py
├── base_policy.py
├── registry.py
├── lerobot_policy.py
└── moe_smolvla_policy.py          # 新增
    ├── MoESmolVLAPolicy          # 主类
    ├── GarmentRouter             # Router
    ├── ExpertManager             # Expert 管理
    └── EpisodeStateManager       # 状态管理
```

### 18.13 设计总结

这个架构确保了：

| 特性 | 实现 |
|------|------|
| **标准性** | 遵循 LeRobot 和 eval 框架的规范 |
| **优雅性** | 模块化、可维护、可扩展 |
| **可靠性** | 状态管理清晰，边界情况完善 |
| **效率性** | 共享 VLM，减少显存占用 |

---

## 十九、版本历史