# MoE-SmolVLA 设计文档

> **状态**: 设计阶段
> **更新**: 2026-03
> **关键词**: MoE, VLA, Sticky Routing, Flow Matching

## 0. 前沿洞察

### 0.1 为什么 MoE 是 VLA 领域的热点？

在 Vision-Language-Action (VLA) 领域引入 MoE 是 2025 下半年到 2026 年初机器人学习最前沿的热点之一。

**核心痛点**：当单一模型处理多种操作任务时，共享权重会遇到：
- **梯度冲突 (Gradient Interference)**
- **动作模式干涉**

**MoE 的价值**：让不同 Expert 专精不同任务，避免冲突。

### 0.2 相关工作

| 论文 | 核心贡献 |
|------|----------|
| **SMP** | Sticky Routing + Diffusion Policy |
| **HiMoE-VLA** | 解决不同机器人实体异构数据 |
| **AdaMoE-VLA** | 专门针对动作输出网络设计 |
| **MoDE-VLA** | 结合力觉/触觉的多模态 MoE |

### 0.3 两个致命工程痛点

在机器人控制中引入 MoE 时，有两个致命的工程痛点：

| 痛点 | 原因 | 后果 | 解决方案 |
|------|------|------|----------|
| **梯度冲突** | 不同任务的动作模式在共享参数中冲突 | 联合训练成功率暴跌到 1.6% | SMP: 正交技能基 |
| **时序抖动** | 每帧 Router 重新计算，概率波动 | 动作幅度被概率绑架，机械臂抽搐 | AdaMoE: 路由与加权解耦 |

### 0.4 传统 MoE 在连续动作上的灾难

```
┌─────────────────────────────────────────────────────────────────────┐
│              传统 MoE：分类概率直接乘以动作输出                       │
│                                                                      │
│  假设控制机械臂向下折叠 pant_short:                                 │
│                                                                      │
│  Frame t:   Router 概率 [0.9, 0.1, 0.0, 0.0]                        │
│            输出 = 0.9 × E_0(x) + 0.1 × E_1(x)                       │
│            → 机械臂向下移动 5cm ✓                                    │
│                                                                      │
│  Frame t+1: 机械臂挡住镜头，Router 变得不确定                        │
│            Router 概率 [0.5, 0.1, 0.0, 0.4]  ← 概率掉到 0.5         │
│            输出 = 0.5 × E_0(x) + 0.4 × E_3(x)                       │
│            → 机械臂只移动 2.5cm + 被干扰拉偏                         │
│                                                                      │
│  问题: 动作幅度被分类概率绑架！                                       │
│        Router 优化分类的梯度 ≠ Action Expert 优化动作的梯度         │
└─────────────────────────────────────────────────────────────────────┘
```

### 0.5 AdaMoE 解法：路由与加权解耦

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AdaMoE: Decoupling Routing & Weighting           │
│                                                                      │
│  传统 MoE:                                                           │
│    output = Σ (router_prob[i] × expert[i](x))                      │
│                 ↑ 概率直接乘以动作                                   │
│                                                                      │
│  AdaMoE:                                                            │
│    selected = argmax(router_prob)  ← Router 只做选择，变成 one-hot  │
│    weight = scale_adapter(state)   ← 独立网络预测权重               │
│    output = Σ (weight[i] × expert[i](x))                           │
│              ↑ 权重与分类概率解耦                                   │
│                                                                      │
│  优势:                                                               │
│    1. 分类 Loss 不干扰动作 Loss                                     │
│    2. 动作幅度稳定，不被概率波动绑架                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### 0.6 SMP 解法：Sticky Routing + 正交技能基

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SMP: Sticky Routing                              │
│                                                                      │
│  Episode 开始 (衣服平铺，视觉清晰):                                  │
│    Router → 选择 Expert 2 (top_long) → 锁定!                        │
│                                                                      │
│  Episode 执行中 (衣服被揉成一团，视觉混乱):                          │
│    不再 Router，强制使用 Expert 2                                   │
│    → 时间一致性，动作平滑                                            │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                    SMP: Orthogonal Skill Basis                      │
│                                                                      │
│  传统联合训练:                                                       │
│    同一参数空间拟合 "向下折叠" 和 "向内折叠"                         │
│    → 特征空间粘连，成功率暴跌                                       │
│                                                                      │
│  SMP MoE:                                                           │
│    E_0: 只学 pant_short (双手协调，向下折叠)                        │
│    E_1: 只学 pant_long  (更多 gripper，向下折叠)                    │
│    E_2: 只学 top_long   (双手独立，向内折叠)                        │
│    E_3: 只学 top_short  (视觉特征弱，向内折叠)                      │
│    → 正交的动作基，互不干扰                                         │
└─────────────────────────────────────────────────────────────────────┘
```

### 0.7 传统 MoE vs 机器人 MoE 对比

```
┌─────────────────────────────────────────────────────────────────────┐
│                传统 LLM MoE (Token-level Routing)                   │
│                                                                      │
│  每个时间步/Timestep，Router 都重新计算概率                          │
│                                                                      │
│  Frame t:   Router → Expert A (短袖)                                │
│  Frame t+1: Router → Expert B (长裤)  ← 因为机械臂遮挡/形变          │
│  Frame t+2: Router → Expert A (短袖)                                │
│                                                                      │
│  结果: 高频抖动 (Jittering) → 真机操作灾难！                         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                机器人 MoE (Sticky Routing / Task-level)             │
│                                                                      │
│  只在 Episode 初始阶段 Router 计算一次，然后锁定                     │
│                                                                      │
│  Frame 0:   Router → Expert A (短袖) → 锁定!                        │
│  Frame 1:   Expert A (锁定)                                         │
│  Frame 2:   Expert A (锁定)                                         │
│  ...                                                                │
│  Frame N:   Expert A (锁定)                                         │
│                                                                      │
│  结果: 时间一致性 (Temporal Consistency) → 动作平滑                  │
└─────────────────────────────────────────────────────────────────────┘
```

### 0.8 工程优势

| 指标 | 传统单模型 | MoE |
|------|------------|-----|
| **总参数量** | ~500M | ~2B (4 experts) |
| **激活参数量** | ~500M | ~500M (只激活 1-2 experts) |
| **推理速度** | 基准 | 相同 |
| **任务容量** | 有限 | 大幅提升 |

**关键洞察**：MoE 增加了总参数量（模型容量），但没有增加激活参数量（推理速度）。

---

## 1. 问题背景

### 1.1 当前挑战

在 LeHome Challenge 中，我们需要用**单一模型**处理 4 种服装类型的折叠任务：

| 类型 | 单独训练成功率 | 特点 |
|------|----------------|------|
| pant_short | 88.3% | 短任务，双手协调 |
| pant_long | 48.3% | 长任务，数据不一致 |
| top_long | 73.3% | 长任务，双手独立 |
| top_short | 41.7% | 视觉特征不明显 |

### 1.2 联合训练的问题

直接用 `four_types_merged` 训练的效果远不如单独训练：

| 类型 | 单独训练 | 联合训练 | 下降 |
|------|----------|----------|------|
| pant_short | 88.3% | 1.6% | -86.7% |
| pant_long | 48.3% | 1.5% | -46.8% |
| top_long | 73.3% | 7.7% | -65.6% |
| top_short | 41.7% | 7.4% | -34.3% |

### 1.3 问题根因

1. **任务长度差异**: pant_short (163帧) vs top_long (332帧) = 2倍
2. **Gripper 操作差异**: pant (29次/ep) vs top (64次/ep) = 2.2倍
3. **动作模式冲突**:
   - pant: 向下折叠，双手协调 (correlation ~0.73)
   - top: 向内折叠，双手独立 (correlation ~0.31)

### 1.4 比赛约束

- 不允许读取文件名 (无法直接知道 garment_type)
- 只能使用一个模型权重
- 模型必须从视觉输入中自动识别服装类型

---

## 2. MoE (Mixture of Experts) 原理

### 2.1 基本概念

```
传统网络:
  所有输入 → 同一套参数 → 输出

MoE 网络:
  输入 → Router → 选择 Top-K 专家 → 加权组合 → 输出
```

### 2.2 核心组件

1. **Router**: 决定使用哪些专家
2. **Experts**: 多个专门的网络
3. **Top-K 选择**: 只激活 K 个专家 (节省计算)

### 2.3 为什么 MoE 有效？

```
场景：学习 4 种不同的服装折叠策略

传统网络:
- 所有类型共享同一套参数
- 参数会"平均化"，无法专精
- pant_short 和 top_long 的策略冲突

MoE:
- Expert 0: 专门学 pant_short (向下折叠，双手协调)
- Expert 1: 专门学 pant_long (长裤，更多 gripper 操作)
- Expert 2: 专门学 top_long (向内折叠，双手独立)
- Expert 3: 专门学 top_short (短袖，视觉特征弱)
- Router: 从视觉特征自动选择合适的专家
```

### 2.4 参数量分析

| 配置 | 总参数量 | 激活参数量 | 说明 |
|------|----------|------------|------|
| 原始 SmolVLA | ~500M | ~500M | 全部激活 |
| MoE (4 experts, top-2) | ~800M | ~400M | 总参数增加，激活减少 |
| MoE (4 smaller experts) | ~500M | ~250M | 总参数不变，激活减半 |

---

## 3. SmolVLA 架构分析

### 3.1 整体架构

```
SmolVLA 数据流:

┌─────────────────────────────────────────────────────────────────┐
│                          PREFIX                                  │
│  Image ──► SigLIP ──┐                                          │
│  Lang ───► Embed ───┼──► SmolVLM (冻结) ──► KV Cache           │
│  State ──► proj ────┘                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          SUFFIX                                  │
│  Noisy Actions ──► action_in_proj ──┐                          │
│  Timestep ──────► Sinusoidal PE ────┼──► Action Expert ──► ... │
│                    MLP ─────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    action_out_proj ──► v_t (预测速度)
```

### 3.2 关键组件

| 组件 | 作用 | 是否可训练 |
|------|------|------------|
| SmolVLM | 视觉语言理解 | 通常冻结 |
| lm_expert (Action Expert) | 动作预测 | 可训练 |
| action_in_proj | 噪声动作投影 | 可训练 |
| action_out_proj | 最终动作投影 | 可训练 |

### 3.3 训练流程 (Flow Matching)

```python
# 训练时
x_t = t * noise + (1-t) * action  # 加噪
u_t = noise - action              # 目标速度

# VLM + Action Expert 处理
suffix_out = ActionExpert(x_t, t, VLM_KV_Cache)

# 预测速度
v_t = action_out_proj(suffix_out)

# Loss
loss = MSE(v_t, u_t)
```

---

## 4. MoE 与 SmolVLA 结合方案

### 4.1 核心机制：Sticky Routing

**关键设计原则**：Router 只在 Episode 初始阶段计算一次，然后锁定 Expert 选择。

```python
class StickyMoERouter:
    """
    粘性路由：只在 Episode 开始时计算路由，之后锁定
    """

    def __init__(self, num_experts: int = 4, top_k: int = 2):
        self.num_experts = num_experts
        self.top_k = top_k
        self.locked_weights = None  # 锁定的专家权重
        self.locked_indices = None  # 锁定的专家索引

    def route(self, visual_features: torch.Tensor, is_new_episode: bool = True):
        """
        Args:
            visual_features: 视觉特征 [batch, hidden_dim]
            is_new_episode: 是否是新 Episode 的开始
        Returns:
            weights: 专家权重
            indices: 专家索引
        """
        if is_new_episode or self.locked_weights is None:
            # 新 Episode：计算路由并锁定
            router_logits = self.router(visual_features)  # [batch, num_experts]
            router_probs = F.softmax(router_logits, dim=-1)

            # 选择 Top-K
            top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

            # 锁定!
            self.locked_weights = top_k_weights
            self.locked_indices = top_k_indices

            return top_k_weights, top_k_indices
        else:
            # 同一 Episode：使用锁定的路由
            return self.locked_weights, self.locked_indices

    def reset(self):
        """Episode 结束时重置"""
        self.locked_weights = None
        self.locked_indices = None
```

### 4.2 完整的 MoE Flow Matching 流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MoE SmolVLA 推理流程                              │
│                                                                      │
│  Step 1: Episode 开始                                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Image (初始帧) ──► SmolVLM ──► visual_features             │   │
│  │                                   │                          │   │
│  │                                   ▼                          │   │
│  │                            Router (计算一次)                │   │
│  │                                   │                          │   │
│  │                                   ▼                          │   │
│  │                    weights=[0.8, 0.2, 0, 0], indices=[0,1]  │   │
│  │                                   │                          │   │
│  │                                   ▼                          │   │
│  │                            🔒 LOCKED 🔒                      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  Step 2: Flow Matching 去噪循环 (N 步)                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  for step in range(num_steps):                              │   │
│  │      x_t = current_noisy_action                             │   │
│  │      t = current_timestep                                   │   │
│  │                                                             │   │
│  │      # 只使用锁定的 Expert                                   │   │
│  │      v_0 = Expert[0](x_t, t)  # weight=0.8                 │   │
│  │      v_1 = Expert[1](x_t, t)  # weight=0.2                 │   │
│  │                                                             │   │
│  │      # 加权组合                                              │   │
│  │      v_t = 0.8 * v_0 + 0.2 * v_1                           │   │
│  │                                                             │   │
│  │      # Flow Matching 更新                                   │   │
│  │      x_{t-dt} = x_t + dt * v_t                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  Step 3: 输出最终动作                                               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  action = x_0  # 去噪后的动作                               │   │
│  │  router.reset()  # Episode 结束，解锁                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.3 方案 A: 完整 MoE Action Expert (推荐)

**位置**: 替换整个 `lm_expert`

```
┌─────────────────────────────────────────────────────────────────┐
│                    SmolVLMWithExpertModel (MoE 版)              │
│                                                                  │
│  ┌─────────────┐                                                │
│  │   SmolVLM   │  ← 冻结，共享                                  │
│  │  (视觉理解)  │                                                │
│  └──────┬──────┘                                                │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────────────────────────┐                      │
│  │             MoE Layer                 │                      │
│  │                                       │                      │
│  │  Router: 从视觉特征推断服装类型       │                      │
│  │           │                           │                      │
│  │    ┌──────┼──────┬──────┐            │                      │
│  │    ▼      ▼      ▼      ▼            │                      │
│  │ ┌─────┐┌─────┐┌─────┐┌─────┐        │                      │
│  │ │Exp 0││Exp 1││Exp 2││Exp 3│        │                      │
│  │ │pant ││pant ││top  ││top  │        │                      │
│  │ │short││long ││long ││short│        │                      │
│  │ └──┬──┘└──┬──┘└──┬──┘└──┬──┘        │                      │
│  │    └──────┴──────┴──────┘            │                      │
│  │              │                        │                      │
│  │      Weighted Sum (Top-K)            │                      │
│  └──────────────────────────────────────┘                      │
│                    │                                             │
│                    ▼                                             │
│             suffix_out (MoE)                                    │
└─────────────────────────────────────────────────────────────────┘
```

**优点**:
- 每个专家完整专精一种类型
- 思考过程不互相干扰
- 效果最好

**缺点**:
- 实现复杂
- 参数量增加

### 4.2 方案 B: 投影层 MoE (简化版)

**位置**: 只在 `action_out_proj` 之前

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  ┌─────────────────────────────────────────────┐                │
│  │     Action Expert (共享 - 单一专家)          │                │
│  │     思考过程是通用的                         │                │
│  └─────────────────────────────────────────────┘                │
│                          │                                       │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────┐                │
│  │              MoE 投影层                      │                │
│  │                                              │                │
│  │  Router: 从 suffix_out 推断类型             │                │
│  │           │                                 │                │
│  │    ┌──────┼──────┬──────┐                  │                │
│  │    ▼      ▼      ▼      ▼                  │                │
│  │ ┌─────┐┌─────┐┌─────┐┌─────┐              │                │
│  │ │proj0││proj1││proj2││proj3│              │                │
│  │ └──┬──┘└──┬──┘└──┬──┘└──┬──┘              │                │
│  │    └──────┴──────┴──────┘                  │                │
│  │              │                              │                │
│  │      Weighted Sum                          │                │
│  └─────────────────────────────────────────────┘                │
│                          │                                       │
│                          ▼                                       │
│                    v_t (预测速度)                                │
└─────────────────────────────────────────────────────────────────┘
```

**优点**:
- 实现简单
- 参数量增加少

**缺点**:
- 思考过程共享，会冲突
- 效果不如方案 A

### 4.3 方案对比

| 方面 | 方案 A (完整 MoE) | 方案 B (投影层 MoE) |
|------|-------------------|---------------------|
| **修改位置** | 替换 lm_expert | 只在 action_out_proj 前 |
| **思考过程** | 每个专家独立 | 共享 (会冲突) |
| **专精程度** | 高 | 低 |
| **参数量** | 多 (4个完整Expert) | 少 (4个小投影层) |
| **实现难度** | 高 | 低 |
| **效果预期** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

### 4.4 大脑比喻

```
方案 A (完整 MoE):
┌────────────────────────────────────────────────────────────────┐
│  就像有 4 个不同的专业运动员                                   │
│                                                                │
│  运动员 A: 专精游泳                                            │
│  运动员 B: 专精跑步                                            │
│  运动员 C: 专精体操                                            │
│  运动员 D: 专精举重                                            │
│                                                                │
│  教练 (Router) 根据比赛类型选择让谁上场                        │
└────────────────────────────────────────────────────────────────┘

方案 B (投影层 MoE):
┌────────────────────────────────────────────────────────────────┐
│  就像一个全能运动员，但换不同的装备                            │
│                                                                │
│  同一个运动员 (共享思考过程)                                   │
│     ├── 游泳时换泳衣                                          │
│     ├── 跑步时换跑鞋                                          │
│     ├── 体操时换体操服                                        │
│     └── 举重时换举重带                                        │
│                                                                │
│  核心能力是共享的，无法像专业运动员那样专精                    │
└────────────────────────────────────────────────────────────────┘
```

---

## 5. 相关工作

### 5.1 SMP: Skill Mixture-of-Experts Policy (2026.01) ⭐ 最相关

**论文**: "Abstracting Robot Manipulation Skills via Mixture-of-Experts Diffusion Policies"
**链接**: https://arxiv.org/abs/2601.21251

**核心思想**:
- 基于 Diffusion Policy 的 MoE (和 SmolVLA 的 Flow Matching 类似)
- 学习正交的技能基 (orthogonal skill basis)
- Sticky Routing: 同一任务持续使用同一组 Expert

**效果**:
- 更高成功率 vs 大型 Diffusion baseline
- 更低推理成本 (不需要超大 backbone)
- 支持迁移学习

### 5.2 MoE-Loco (2025.03)

**论文**: "MoE-Loco: Mixture of Experts for Multitask Locomotion"
**链接**: https://arxiv.org/abs/2503.08564

**核心思想**:
- 针对腿式机器人的多任务运动
- 处理多种地形: bars, pits, stairs, slopes, baffles
- 支持四足和双足步态

**关键发现**:
- 不同 Expert 自然地专精于不同的运动行为
- 缓解了多任务强化学习中的梯度冲突
- 可用于任务迁移和技能组合

---

## 6. 实施建议

### 6.1 训练策略

```yaml
# MoE 训练配置
moe:
  num_experts: 4
  top_k: 2
  expert_names: [pant_short, pant_long, top_long, top_short]

# 损失权重
loss_weights:
  action_loss: 1.0              # 主损失
  classification_loss: 0.5      # 分类监督
  router_consistency_loss: 0.1  # Router 和分类器一致性
  load_balance_loss: 0.01       # 负载均衡

# 数据集需要包含 garment_type 标签
dataset:
  # 需要在数据中添加 garment_type 字段
  # 0: pant_short, 1: pant_long, 2: top_long, 3: top_short
```

### 6.2 实施路径

```
步骤 1: 验证 Router 可行性 (1-2天)
├── 训练一个简单的分类器
├── 输入: VLM 的视觉特征
├── 输出: 4 分类
└── 目标: 验证视觉特征能否区分服装类型

步骤 2: 实现方案 B (3-5天)
├── 修改 VLAFlowMatching
├── 添加 MoE 投影层
└── 快速验证 MoE 效果

步骤 3: 如果方案 B 有效，升级到方案 A (1-2周)
├── 修改 SmolVLMWithExpertModel
├── 完整的 MoE Transformer Expert
└── 更好的效果
```

### 6.3 步骤 1 代码示例

```python
# 验证 Router 可行性
class GarmentClassifier(nn.Module):
    """简单的分类器验证视觉特征能否区分服装类型"""

    def __init__(self, hidden_dim, num_classes=4):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, visual_features):
        """
        Args:
            visual_features: VLM 输出的视觉特征 [batch, hidden_dim]
        Returns:
            logits: 分类 logits [batch, 4]
        """
        return self.classifier(visual_features)

# 训练
def train_classifier(model, dataloader, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for batch in dataloader:
            visual_features = batch['visual_features']  # 从预训练 VLM 提取
            garment_type = batch['garment_type']        # 0-3 标签

            logits = model(visual_features)
            loss = criterion(logits, garment_type)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 评估准确率
    # 如果 > 90%，说明视觉特征可以区分服装类型
```

---

## 7. 预期效果

如果 MoE 方案成功：

| 类型 | 当前 (联合训练) | 预期 (MoE) | 目标 |
|------|-----------------|------------|------|
| pant_short | 1.6% | ~70% | 接近单独训练 (88.3%) |
| pant_long | 1.5% | ~40% | 接近单独训练 (48.3%) |
| top_long | 7.7% | ~60% | 接近单独训练 (73.3%) |
| top_short | 7.4% | ~35% | 接近单独训练 (41.7%) |

---

## 8. 参考

### 8.1 核心论文

1. **SMP (Skill Mixture-of-Experts Policy)** - https://arxiv.org/abs/2601.21251
   - Diffusion-based MoE，Sticky Routing
   - 正交技能基学习

2. **MoE-Loco** - https://arxiv.org/abs/2503.08564
   - 多任务运动控制
   - 梯度冲突缓解

### 8.2 VLA 领域 MoE 相关工作

| 论文 | 核心贡献 | 特点 |
|------|----------|------|
| **HiMoE-VLA** | 解决不同机器人实体异构数据 | 层次化 MoE |
| **AdaMoE-VLA** | 针对动作输出网络设计 | 自适应 MoE |
| **MoDE-VLA** | 结合力觉/触觉多模态 | 多模态 MoE |
| **SMP** | Sticky Routing + Diffusion | 时间一致性 |

### 8.3 其他参考

- SmolVLA: https://huggingface.co/papers/2506.01844
- Mixtral MoE: https://arxiv.org/abs/2401.04088
- Switch Transformer: https://arxiv.org/abs/2101.03961

---

## 9. 关键洞察：Teacher Forcing 训练策略

### 9.1 问题：LeRobot 是 Sample-level 训练

```
LeRobot 训练数据流:

原始数据: 250 episodes × ~163 frames = ~40,000 frames

┌─────────────────────────────────────────────────────────────────┐
│  EpisodeAwareSampler 的工作方式:                              │
│                                                              │
│  Episode 0: [F_0, F_1, F_2, ..., F_162]                            │
│  Episode 1: [F_0, F_1, F_2, ..., F_160]                            │
│  ...                                                          │
│  Episode 249: [F_0, F_1, F_2, ..., F_165]                          │
│                                                              │
│  展开成: indices = [0, 1, 2, ..., 40000]                        │
│                                                              │
│  然后 shuffle(indices) 随机打乱！                              │
└─────────────────────────────────────────────────────────────────┘

DataLoader 返回的 Batch:

Batch = [
    frame_42 (来自 Episode 5,  衬衫平铺),    ← 需要向内折叠
    frame_103 (来自 Episode 12, 裤子提起),
    frame_7 (来自 Episode 1,   衬衫对折),
    ...
]

每个 sample 来自不同的 Episode，顺序是随机的！
```

### 9.2 解决方案：Teacher Forcing 训练策略

**关键洞察**：数据集已经有 `garment_type` 标签 (0-3)，我们可以直接用标签选择 Expert，而不需要 Router 来决定！

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Teacher Forcing 训练流程                              │
│                                                                          │
│  Batch = [frame_42, frame_103, frame_7, frame_888, ...]                 │
│  标签 = [type_1,    type_0,    type_2,   type_1,   ...]                 │
│         (pant_long)(pant_short)(top_long)(pant_long)                   │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Expert 训练 (Teacher Forcing)                                   │   │
│  │                                                                  │   │
│  │  frame_42, label=1 (pant_long)                                   │   │
│  │      │                                                           │   │
│  │      ▼                                                           │   │
│  │  ┌────────────────────────────────────────────────────────────┐  │   │
│  │  │ 直接用 label=1 选择 Expert 1                                │  │   │
│  │  │ 不经过 Router！                                             │  │   │
│  │  │                                                              │  │   │
│  │  │ Expert 1.forward(frame_42) → v_t                            │  │   │
│  │  │ loss += MSE(v_t, target)                                    │  │   │
│  │  └────────────────────────────────────────────────────────────┘  │   │
│  │      │                                                           │   │
│  │  frame_103, label=0 (pant_short)                                 │   │
│  │      │                                                           │   │
│  │      ▼                                                           │   │
│  │  ┌────────────────────────────────────────────────────────────┐  │   │
│  │  │ 直接用 label=0 选择 Expert 0                                │  │   │
│  │  │ Expert 0.forward(frame_103) → v_t                           │  │   │
│  │  └────────────────────────────────────────────────────────────┘  │   │
│  │      │                                                           │   │
│  │  ... (其他帧同理)                                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Router 训练 (独立的分类任务)                                      │   │
│  │                                                                  │   │
│  │  可以单独训练，也可以作为辅助损失:                                  │   │
│  │                                                                  │   │
│  │  router_logits = Router(visual_features)                         │   │
│  │  router_loss = CrossEntropy(router_logits, garment_type)         │   │
│  │                                                                  │   │
│  │  注意: router_loss 不参与 Expert 的梯度！                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.3 优势

| 方面 | 原始设计 | Teacher Forcing 设计 |
|------|----------|------------------------|
| **Expert 选择** | Router 预测 | 标签直接指定 |
| **Router 角色** | 关键决策者 | 辅助分类器 |
| **训练稳定性** | Router 错误会影响 Expert | 完全解耦，极稳定 |
| **实现复杂度** | 需要处理 Router 错误传播 | 简单直接 |

### 9.4 推理时仍然需要 Sticky Routing

```
训练时: Teacher Forcing (用标签选 Expert)
推理时: Sticky Routing (用 Router 选一次，然后锁定)

┌─────────────────────────────────────────────────────────────────────────┐
│  训练阶段                                                             │
│                                                                      │
│  每帧独立处理:                                                        │
│  garment_type 标签 → 直接选择 Expert → 训练                            │
│                                                                      │
│  Router 作为辅助任务训练，但不影响 Expert                              │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  推理阶段                                                             │
│                                                                      │
│  Episode 开始:                                                        │
│  Frame 0 → Router → 选择 Expert → 锁定                                │
│                                                                      │
│  Episode 执行:                                                        │
│  Frame 1~N → 锁定的 Expert (不再 Router)                              │
│                                                                      │
│  Episode 结束:                                                        │
│  解锁，准备下一个 Episode                                             │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 10. 最终推荐架构

### 10.1 核心设计原则

| 原则 | 来源 | 具体实现 |
|------|------|----------|
| **Teacher Forcing** | 本设计 | 训练时用标签直接选 Expert，完全解耦 |
| **Sticky Routing** | SMP | 推理时只在 Episode 开始 Router 一次，之后锁定 |
| **路由与加权解耦** | AdaMoE | Router 只做选择，不做概率加权 |
| **损失隔离** | AdaMoE | Classification Loss 和 Action Loss 分开反传 |

### 10.2 训练策略：Teacher Forcing

```python
class MoESmolVLA(nn.Module):
    def __init__(self, ...):
        self.vlm = SmolVLM(...)           # 视觉编码器 (冻结)
        self.router = GarmentRouter(...)   # 辅助分类器
        self.experts = nn.ModuleList([     # 4 个专家
            ActionExpert(...) for _ in range(4)
        ])

    def forward(
        self,
        frames,
        garment_types=None,      # 训练时的标签
        inference_mode=False,   # 是否是推理模式
    ):
        # 提取视觉特征
        visual_features = self.vlm.encode(frames)

        if inference_mode:
            # ========== 推理模式: Sticky Routing ==========
            if self.locked_expert_idx is None:
                # 第一帧: Router 选择并锁定
                router_logits = self.router(visual_features)
                self.locked_expert_idx = router_logits.argmax(dim=-1)

            # 用锁定的 Expert
            action = self.experts[self.locked_expert_idx](visual_features)

        else:
            # ========== 训练模式: Teacher Forcing ==========
            # 直接用标签选择 Expert，不经过 Router！
            # garment_types: [batch] 每个样本的标签 0-3

            actions = []
            router_loss = 0.0

            for i in range(len(frames)):
                # Teacher Forcing: 直接用标签选择 Expert
                expert_idx = garment_types[i]
                action = self.experts[expert_idx](visual_features[i:i+1])
                actions.append(action)

                # 辅助: 计算 Router Loss (用于训练 Router 的分类能力)
                router_logits = self.router(visual_features[i:i+1])
                router_loss += F.cross_entropy(router_logits, garment_types[i:i+1])

            actions = torch.cat(actions, dim=0)

            return actions, router_loss / len(frames)
```

### 10.3 训练流程

```python
def train_step(model, batch, optimizer):
    """单个训练步骤 - Teacher Forcing 模式"""

    frames = batch["frames"]              # [batch, C, H, W]
    actions_target = batch["actions"]     # [batch, action_dim]
    garment_types = batch["garment_type"] # [batch] - 标签 0-3

    # 前向传播 (训练模式)
    actions_pred, router_loss = model(
        frames,
        garment_types=garment_types,
        inference_mode=False,  # 训练模式
    )

    # Expert Loss (只传给被 Teacher Forced 的 Expert)
    expert_loss = F.mse_loss(actions_pred, actions_target)

    # 总损失
    total_loss = expert_loss + 0.1 * router_loss

    # 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return expert_loss.item(), router_loss.item()
```

### 10.4 推理流程

```python
def inference_episode(model, frames, num_voting_frames=5):
    """推理一个完整 Episode - Sticky Routing 模式 (带多帧投票)"""

    model.reset()  # 解锁 Expert

    actions = []
    
    # 投票窗口缓存
    vote_logits = []

    for t, frame in enumerate(frames):
        if t < num_voting_frames:
            # 投票阶段：只运行 VLM 和 Router，不运行 Expert
            visual_features = model.vlm.encode(frame)
            logits = model.router(visual_features)
            vote_logits.append(logits)
            
            # 由于还没有选定 Expert，可以用一个基础动作兜底或保持原位
            # action = base_action_fallback(frame) 
            # 甚至对于早期帧直接使用 shared projector 进行兜底 (如果是方案 B)
            # 这里简化处理
            action = torch.zeros(action_dim) 
            
            if t == num_voting_frames - 1:
                # 投票结束，计算总票数并锁定
                avg_logits = torch.stack(vote_logits).mean(dim=0)
                model.locked_expert_idx = avg_logits.argmax(dim=-1)
                model.is_locked = True
                
        else:
            # 推理模式：使用锁定的 Expert 处理后续帧
            action = model(
                frame,
                inference_mode=True, 
            )
            
        actions.append(action)

    return torch.stack(actions)
```

### 9.2 完整架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MoE-SmolVLA: 融合 SMP + AdaMoE                       │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Episode 开始 (Frame 0~K，比如前5帧)             │   │
│  │                                                                  │   │
│  │  Image ──► SmolVLM ──► visual_features                          │   │
│  │                               │                                  │   │
│  │                               ▼                                  │   │
│  │                    ┌─────────────────────┐                       │   │
│  │                    │   Router (分类器)    │                       │   │
│  │                    │  计算概率 / Logits   │                       │   │
│  │                    └──────────┬──────────┘                       │   │
│  │                               │                                  │   │
│  │                               ▼                                  │   │
│  │                    ┌─────────────────────┐                       │   │
│  │                    │   多帧 Voting 锁定   │  ← 防止单帧误判       │   │
│  │                    │   selected = argmax(Σ logit)              │   │
│  │                    │   locked = True                             │   │
│  │                    └──────────────────────┘                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Episode 执行中 (Frame 1~N)                    │   │
│  │                                                                  │   │
│  │  ┌───────────┐  ┌───────────┐                                   │   │
│  │  │  Noisy    │  │ Timestep  │                                   │   │
│  │  │  Action   │  │    t      │                                   │   │
│  │  └─────┬─────┘  └─────┬─────┘                                   │   │
│  │        │              │                                          │   │
│  │        └──────┬───────┘                                          │   │
│  │               ▼                                                  │   │
│  │  ┌────────────────────────────────────────────────────────┐     │   │
│  │  │              只激活锁定的 Expert                         │     │   │
│  │  │                                                         │     │   │
│  │  │   Expert 0      Expert 1      Expert 2      Expert 3   │     │   │
│  │  │   (锁定)         (bypass)      (bypass)      (bypass)  │     │   │
│  │  │      │              ×             ×             ×       │     │   │
│  │  │      │                                                  │     │   │
│  │  │      ▼                                                  │     │   │
│  │  │   v_t = Expert_0(x_t, t, KV_Cache)                     │     │   │
│  │  │                                                        │     │   │
│  │  │   ← AdaMoE: 直接输出，无概率相乘                        │     │   │
│  └────────────────────────────────────────────────────────────────┘     │   │
│  │               │                                                  │   │
│  │               ▼                                                  │   │
│  │        ┌─────────────┐                                          │   │
│  │        │   v_t       │  → Flow Matching 更新 x_t               │   │
│  │        └─────────────┘                                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                         损失函数                                 │   │
│  │                                                                  │   │
│  │  ┌──────────────────────────────────────────────────────────┐   │   │
│  │  │  Classification Loss (只传给 Router)                      │   │   │
│  │  │  L_cls = CrossEntropy(router_logits, garment_type)       │   │   │
│  │  │  → 只在每个 Episode 的第一帧计算                          │   │   │
│  │  └──────────────────────────────────────────────────────────┘   │   │
│  │                                                                  │   │
│  │  ┌──────────────────────────────────────────────────────────┐   │   │
│  │  │  Action Loss (只传给激活的 Expert)                        │   │   │
│  │  │  L_action = MSE(v_t, noise - action)                     │   │   │
│  │  │  → 只反传给当前锁定的 Expert                             │   │   │
│  │  └──────────────────────────────────────────────────────────┘   │   │
│  │                                                                  │   │
│  │  Total Loss = L_action + α * L_cls + β * L_orthogonal         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.3 PyTorch 骨架代码

```python
class StickyMoEActionExpert(nn.Module):
    """
    融合 SMP + AdaMoE 思想的 MoE Action Expert
    """

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        num_experts: int = 4,
        num_layers: int = 4,
    ):
        super().__init__()
        self.num_experts = num_experts

        # Router: 只做分类，不做加权
        self.router = nn.Linear(hidden_dim, num_experts)

        # 多个 Expert (每个是一个小型 Transformer)
        self.experts = nn.ModuleList([
            TransformerExpert(hidden_dim, num_layers)
            for _ in range(num_experts)
        ])

        # 动作输出投影
        self.action_proj = nn.Linear(hidden_dim, action_dim)

        # 状态变量：锁定的 Expert 索引
        self.register_buffer("locked_expert_idx", torch.tensor(-1))
        self.is_locked = False

    def route_and_lock(self, visual_features: torch.Tensor) -> int:
        """
        在 Episode 开始时调用，选择并锁定 Expert
        """
        with torch.no_grad():
            logits = self.router(visual_features)  # [batch, num_experts]
            selected = logits.argmax(dim=-1)  # [batch] - 只做选择，不做加权
            self.locked_expert_idx = selected[0].item()  # 锁定
            self.is_locked = True
        return self.locked_expert_idx

    def forward(
        self,
        x: torch.Tensor,  # [batch, chunk, hidden]
        compute_router_loss: bool = False,
        garment_type: torch.Tensor = None,
    ) -> tuple:
        """
        前向传播：
        - 如果已锁定，只使用锁定的 Expert
        - 如果未锁定，使用 Router 选择
        """
        if self.is_locked:
            # SMP: 直接使用锁定的 Expert，无概率相乘
            expert_output = self.experts[self.locked_expert_idx](x)
            action = self.action_proj(expert_output)
            router_loss = None
        else:
            # Episode 开始时，需要 Router
            visual_features = x.mean(dim=1)  # [batch, hidden]
            logits = self.router(visual_features)

            # AdaMoE: 只用 argmax 选择，不用概率加权
            selected = logits.argmax(dim=-1)  # [batch]

            # 前向时仍需要处理 batch 中可能的不同选择
            # 但在 Sticky 场景下，通常 batch=1
            expert_output = self.experts[selected[0]](x)
            action = self.action_proj(expert_output)

            # 计算 Router Loss (分类损失)
            if compute_router_loss and garment_type is not None:
                router_loss = F.cross_entropy(logits, garment_type)
            else:
                router_loss = None

        return action, router_loss

    def reset(self):
        """Episode 结束时调用，解锁"""
        self.is_locked = False
        self.locked_expert_idx = -1


class TransformerExpert(nn.Module):
    """单个 Expert：小型 Transformer"""
    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
            ) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
```

### 9.4 训练流程

```python
def train_episode(model, batch, optimizer):
    """单个 Episode 的训练流程"""

    # 1. Episode 开始：Router 选择并锁定
    visual_features = extract_visual_features(batch["images"][0])  # 第一帧
    model.route_and_lock(visual_features)

    # 2. 计算 Router Loss (只在第一帧)
    garment_type = batch["garment_type"]
    router_loss = None

    # 3. 执行 Flow Matching 训练
    total_action_loss = 0
    for t in range(batch["actions"].shape[1]):  # 遍历时间步
        action, r_loss = model(
            x=batch["hidden_states"][t],
            compute_router_loss=(t == 0),  # 只在第一步计算 Router Loss
            garment_type=garment_type,
        )
        if r_loss is not None:
            router_loss = r_loss

        # Flow Matching Loss
        action_loss = F.mse_loss(action, batch["target_velocity"][t])
        total_action_loss += action_loss

    # 4. 反向传播 (损失隔离)
    optimizer.zero_grad()

    # Router Loss 只传给 Router
    if router_loss is not None:
        router_loss.backward(retain_graph=True)

    # Action Loss 只传给激活的 Expert
    total_action_loss.backward()

    optimizer.step()

    # 5. Episode 结束：解锁
    model.reset()
```

---

## 11. 总结

### 11.1 核心设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| **训练策略** | Teacher Forcing | 用标签直接选 Expert，完全解耦 Router 和 Expert |
| **推理策略** | Sticky Routing | 只在 Episode 开始 Router 一次，之后锁定 |
| **MoE 位置** | Action Expert (lm_expert) | 不干扰视觉理解，只专精动作预测 |
| **Expert 数量** | 4 | 对应 4 种服装类型 |
| **Router 角色** | 辅助分类器 | 训练时不影响 Expert，推理时用于初始选择 |

### 11.2 训练 vs 推理对比

| 阶段 | Expert 选择 | Router 作用 |
|------|-------------|-------------|
| **训练** | Teacher Forcing (标签) | 辅助分类损失 |
| **推理** | Sticky Routing (Router) | 初始选择 + 锁定 |

### 11.3 预期收益

1. **解决梯度冲突**：每个 Expert 只学一种类型，互不干扰
2. **训练稳定**：Teacher Forcing 保证 Expert 训练不受 Router 影响
3. **推理平滑**：Sticky Routing 保证时间一致性
4. **保持推理速度**：激活参数量不变

### 11.4 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| Router 分类错误 | 训练辅助分类损失，提升准确率 |
| 专家负载不均 | Teacher Forcing 天然均衡 |
| 过拟合特定 garment | 数据增强、正则化 |
| 泛化到 unseen | 增加训练数据多样性 |

### 11.5 下一步行动

1. **验证 Router 可行性** - 测试视觉特征能否区分 4 种类型
2. **实现 Teacher Forcing MoE** - 修改训练代码
3. **实现 Sticky Routing 推理** - 修改评估代码
4. **训练和评估** - 在 four_types 数据集上验证
