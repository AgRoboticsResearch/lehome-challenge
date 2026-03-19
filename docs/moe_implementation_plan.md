# MoE-SmolVLA 实现计划

> **状态**: 待实现
> **更新**: 2026-03-16
> **基于**: Router 可行性验证实验（100% 准确率）

## 一、实验结论回顾

Router 可行性验证通过：
- **单摄像头 (top_rgb) 足够区分 4 种服装类型**
- **分类器**: 4层 MLP (512→256→128→4)，200 epochs
- **准确率**: 100%
- **关键发现**: VLM backbone 可以保持冻结，只需要训练 Router 分类头

## 二、整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MoE-SmolVLA 架构                                │
│                                                                      │
│  输入: Image (top_rgb) + State + Language                           │
│         │                                                            │
│         ▼                                                            │
│  ┌─────────────────────┐                                            │
│  │   SmolVLM Backbone   │  ← 冻结，共享                            │
│  │   (视觉语言理解)     │                                            │
│  └──────────┬──────────┘                                            │
│             │                                                        │
│             ├──────────────────────┐                                │
│             │                      │                                │
│             ▼                      ▼                                │
│  ┌─────────────────┐    ┌─────────────────────────────────┐        │
│  │  Router 分类头    │    │      Action Experts (MoE)        │        │
│  │                 │    │                                 │        │
│  │  Linear(hidden,4) │    │  ┌─────┐┌─────┐┌─────┐┌─────┐  │        │
│  │  + Softmax      │    │  │Exp 0││Exp 1││Exp 2││Exp 3│  │        │
│  └────────┬────────┘    │  │pant ││pant ││top  ││top  │  │        │
│             │              │  │short││long ││long ││short│  │        │
│             │              │  └──┬──┘└──┬──┘└──┬──┘└──┬──┘  │        │
│             │                   │      │      │      │      │   │        │
│             │                   └──────┴──────┴──────┴──────┘   │        │
│             │                              │                        │        │
│             │                              ▼                        │        │
│             │                    ┌─────────────────┐             │        │
│             │                    │  Weighted Sum   │             │        │
│             │                    │  (Top-K 专家)   │             │        │
│             │                    └────────┬────────┘             │        │
│             │                             │                      │        │
│             ▼                             ▼                      │        │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                        action_out_proj                                │  │
│  │                              │                                       │  │
│  │                              ▼                                       │  │
│  │                          v_t (预测速度)                              │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.1 核心组件

| 组件 | 作用 | 是否可训练 |
|------|------|------------|
| SmolVLM | 视觉语言理解 | 冻结 |
| Router | 从视觉特征推断服装类型 | 可训练 |
| Action Experts | 4 个专家，每个专精一种类型 | 可训练 |
| action_out_proj | 最终动作投影 | 可训练 |

### 2.2 Sticky Routing 机制

```python
class StickyMoERouter:
    """
    粘性路由：只在 Episode 开始时计算路由，之后锁定
    """

    def __init__(self, hidden_dim: int, num_experts: int = 4, top_k: int = 2):
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(hidden_dim, num_experts)
        self.locked_weights = None
        self.locked_indices = None

    def route(self, visual_features: torch.Tensor, is_new_episode: bool = True):
        """
        Args:
            visual_features: 视觉特征 [batch, hidden_dim]
            is_new_episode: 是否是新 Episode 的开始
        Returns:
            weights: 专家权重 [batch, top_k]
            indices: 专家索引 [batch, top_k]
        """
        if is_new_episode or self.locked_weights is None:
            # 新 Episode：计算路由并锁定
            router_logits = self.router(visual_features)
            router_probs = F.softmax(router_logits, dim=-1)

            # 选择 Top-K
            top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

            # 锁定
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

### 2.3 MoE Action Expert

```python
class MoEActionExpert(nn.Module):
    """
    多个 Action Expert 的 MoE 层
    """

    def __init__(self, expert_hidden_dim: int, num_experts: int = 4):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=expert_hidden_dim, nhead=8)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor, expert_indices: torch.Tensor, expert_weights: torch.Tensor):
        """
        Args:
            x: 输入 [batch, seq_len, hidden_dim]
            expert_indices: 选择的专家索引 [batch, top_k]
            expert_weights: 专家权重 [batch, top_k]
        Returns:
            output: [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = x.shape

        # 计算每个选中专家的输出
        outputs = []
        for i, expert_idx in enumerate(expert_indices[0]):
            expert_out = self.experts[expert_idx](x)
            outputs.append(expert_out)

        # 加权组合
        output = sum(w * out for w, out in zip(expert_weights[0], outputs))
        return output
```

## 三、实现步骤

### 阶段 1: 修改模型架构

**文件**: `source/lehome/lehome/policies/moe_smolvla.py`

**任务**:
1. 创建 `MoESmolVLAPolicy` 类，继承 `SmolVLAPolicy`
2. 添加 `StickyMoERouter` 组件
3. 修改 `VLAFlowMatching` 为 `MoEVLAFlowMatching`
4. 添加多个 Action Expert

**代码结构**:
```python
# moe_smolvla.py

class StickyMoERouter(nn.Module):
    """Router 分类头 + 粘性路由逻辑"""
    pass

class MoEActionExperts(nn.Module):
    """4 个 Action Expert + Top-K 选择"""
    pass

class MoEVLAFlowMatching(nn.Module):
    """修改后的 Flow Matching，使用 MoE"""
    pass

class MoESmolVLAPolicy(SmolVLAPolicy):
    """MoE 版本的 SmolVLA Policy"""
    pass
```

### 阶段 2: 修改训练流程

**文件**: `source/lehome/lehome/policies/moe_smolvla.py`

**任务**:
1. 在 `forward()` 中实现 Sticky Routing
2. 修改 loss 计算，考虑专家权重的加权 loss
3. 添加 Router 辅助损失（可选，用于监督学习）

**关键代码**:
```python
def forward(self, batch, noise=None, time=None, reduction: str = "mean"):
    # 1. 提取视觉特征
    images, img_masks = self.prepare_images(batch)
    visual_features = self.model.vlm_with_expert.embed_image(images[0])

    # 2. Router 决策（只在 Episode 首帧）
    is_new_episode = batch.get("is_first_frame", torch.ones(batch_size, dtype=torch.bool))
    expert_weights, expert_indices = self.model.router.route(
        visual_features, is_new_episode
    )

    # 3. MoE Flow Matching
    state = self.prepare_state(batch)
    lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
    lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
    actions = self.prepare_action(batch)

    # 使用选中的专家进行 Flow Matching
    losses = self.model.moe_forward(
        images, img_masks, lang_tokens, lang_masks, state, actions,
        expert_weights, expert_indices,
        noise, time
    )

    # 4. 计算 loss
    # ... (同原始 SmolVLA)
```

### 阶段 3: 数据加载

**文件**: `lerobot/datasets/lerobot_dataset.py` (或新建 wrapper)

**任务**:
1. 修改 dataset 的 `__getitem__` 方法
2. 添加服装类型标签
3. 添加 Episode 边界信息

**代码**:
```python
class MoEDataset(LeRobotDataset):
    def __getitem__(self, idx):
        frame = super().__getitem__(idx)

        # 添加服装类型标签
        episode_idx = frame["episode_index"]
        frame["garment_type"] = self.get_garment_type(episode_idx)

        # 添加 Episode 边界信息
        frame["is_first_frame"] = (frame["frame_index"] == 0)

        return frame
```

### 阶段 4: 训练配置

**文件**: `configs/train_moe_smolvla.yaml`

```yaml
policy:
  type: moe_smolvla
  num_experts: 4
  top_k: 2
  router_hidden_dim: 256

  # 共享 VLM 配置
  vlm_model_name: HuggingFaceTB/SmolVLM2-500M-Video-Instruct
  freeze_vision_encoder: true
  train_expert_only: false  # MoE Experts 需要训练

dataset:
  repo_id: four_types_merged
  root: Datasets/example/four_types_merged

training:
  batch_size: 8
  steps: 100000
  lr: 1e-4
```

## 四、实现优先级

| 优先级 | 任务 | 预计工作量 | 依赖 |
|--------|------|------------|------|
| P0 | 创建 MoE 模型架构 | 1 天 | 无 |
| P0 | 实现 Sticky Routing | 0.5 天 | P0 |
| P1 | 修改训练流程 | 1 天 | P0 |
| P1 | 数据加载修改 | 0.5 天 | 无 |
| P2 | Router 辅助损失 | 0.5 天 | P0 |
| P2 | Expert 负载均衡 | 0.5 天 | P0 |

## 五、风险和挑战

1. **训练稳定性**: MoE 训练可能不稳定
   - 缓解: 使用较小的学习率，增加 warmup steps

2. **Expert 崩溃**: 某些 Expert 可能不被使用
   - 缓解: 添加负载均衡损失 (auxiliary loss)

3. **过拟合**: 每个 Expert 的数据量减少为 1/4
   - 缓解: 数据增强，正则化

## 六、验证指标

1. **Router 准确率**: 验证集上的分类准确率
2. **Expert 使用率**: 每个 Expert 被选中的频率
3. **任务成功率**: Isaac Sim 中的折叠成功率
4. **动作 MSE**: 预测动作与真实动作的 MSE

## 七、下一步行动

1. [ ] 创建 `moe_smolvla.py` 文件框架
2. [ ] 实现 `StickyMoERouter` 类
3. [ ] 实现 `MoEActionExperts` 类
4. [ ] 修改训练配置支持 MoE
5. [ ] 在小数据集上验证训练流程
