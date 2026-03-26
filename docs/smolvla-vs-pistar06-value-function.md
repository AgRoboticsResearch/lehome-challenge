# SmolVLA vs Pistar06 价值函数架构对比与改造方案

> **创建日期**: 2026-03-19
> **主题**: 基于SmolVLA实现价值函数的详细方案

---

## 目录

1. [架构对比](#架构对比)
2. [核心差异](#核心差异)
3. [改造方案](#改造方案)
4. [实现步骤](#实现步骤)
5. [代码框架](#代码框架)

---

## 架构对比

### Pistar06架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            Pistar06                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Input:                                                                  │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐          │
│  │ Images       │ Task Text    │ State        │ Episode Info │          │
│  │ [B,N,C,H,W]  │ [B] strings  │ [B, state_dim]│ success/len  │          │
│  └──────────────┴──────────────┴──────────────┴──────────────┘          │
│         │                │                │                            │
│         ▼                ▼                ▼                            │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │                    Pistar06Model                             │        │
│  │                                                              │        │
│  │  ┌────────────────┐         ┌────────────────┐              │        │
│  │  │Vision Encoder  │         │Language Model  │              │        │
│  │  │(SigLIP SO400M) │         │(Gemma 3 270M) │              │        │
│  │  │  1152 dim      │         │  2304 dim      │              │        │
│  │  └───────┬────────┘         └───────┬────────┘              │        │
│  │          │                          │                        │        │
│  │          ▼                          ▼                        │        │
│  │  ┌────────────────────────────────────────────┐            │        │
│  │  │            Image Projector  │ Language      │            │        │
│  │  │             1152→512      │ 2304→512      │            │        │
│  │  └───────────────────┬────────────────────────┘            │        │
│  │                      ▼                                     │        │
│  │           ┌───────────────────────┐                       │        │
│  │           │ Concat & LayerNorm    │                       │        │
│  │           │    512+512=1024        │                       │        │
│  │           └───────────┬───────────┘                       │        │
│  │                       ▼                                     │        │
│  │           ┌───────────────────────┐                       │        │
│  │           │      Value Head       │                       │        │
│  │           │  1024→512→201 bins    │                       │        │
│  │           └───────────┬───────────┘                       │        │
│  └───────────────────────┼───────────────────────────────────┘        │
│                          ▼                                             │
│                    Output: [B, 201] (Value logits)                      │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### SmolVLA架构（当前Expert）

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SmolVLA (Expert)                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Input:                                                                  │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐          │
│  │ Images       │ Task Text    │ State        │ Actions      │          │
│  │ [B,N,C,H,W]  │ [B] strings  │ [B, state_dim]│ [B, chunk, dim]│       │
│  └──────────────┴──────────────┴──────────────┴──────────────┘          │
│         │                │                │                │            │
│         ▼                ▼                ▼                ▼            │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │                    SmolVLMWithExpertModel                     │        │
│  │                                                              │        │
│  │  ┌────────────────┐         ┌────────────────┐              │        │
│  │  │Vision Encoder  │         │VLM Language    │              │        │
│  │  │(SmolVLM Vision)│         │(VLlama3)       │              │        │
│  │  │  768 dim       │         │  960 dim        │              │        │
│  │  └───────┬────────┘         └───────┬────────┘              │        │
│  │          │                          │                        │        │
│  │          ▼                          ▼                        │        │
│  │  ┌────────────────────────────────────────────┐            │        │
│  │  │          Embed Concatenation               │            │        │
│  │  │  [img + <IMG_START> + text + state]        │            │        │
│  │  └───────────────────┬────────────────────────┘            │        │
│  │                      ▼                                     │        │
│  │           ┌───────────────────────┐                       │        │
│  │           │   VLM Prefix Forward  │                       │        │
│  │           │   (frozen/finetune)   │                       │        │
│  │           └───────────┬───────────┘                       │        │
│  │                       ▼                                     │        │
│  │           ┌───────────────────────┐                       │        │
│  │           │    Expert LM          │                       │        │
│  │           │  (Gemma-based)        │                       │        │
│  │           │  + Action Decoder     │                       │        │
│  │           └───────────┬───────────┘                       │        │
│  └───────────────────────┼───────────────────────────────────┘        │
│                          ▼                                             │
│                    Output: [B, chunk_size, action_dim]                 │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 核心差异

### 1. Backbone差异

| 方面 | Pistar06 | SmolVLA | 影响 |
|------|----------|---------|------|
| **Vision Encoder** | SigLIP SO400M | SmolVLM Vision | 不同架构，特征维度不同 |
| **Vision Dim** | 1152 | 768 | 需要调整projector |
| **Language Model** | Gemma 3 270M | VLlama3 | 不同架构，特征维度不同 |
| **Language Dim** | 2304 | 960 | 需要调整projector |
| **跨模态融合** | 独立fusion层 | VLM内部融合 | 关键差异！ |

### 2. 融合策略差异

```
Pistar06融合策略:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Vision Features ────────────────────────┐                  │
│  Language Features ───────────────────┐   │                  │
│                                         │   │                  │
│                                         ▼   ▼                  │
│                                    Concat (1024)              │
│                                         │                      │
│                                         ▼                      │
│                                    LayerNorm                  │
│                                         │                      │
│                                         ▼                      │
│                                    Value Head                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘

SmolVLA融合策略:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  [IMG_START] + Vision Features + Text + State               │
│                        │                                     │
│                        ▼                                     │
│              VLM Transformer (多层自注意力)                   │
│                        │                                     │
│                        ▼                                     │
│              Expert LM (条件生成)                            │
│                        │                                     │
│                        ▼                                     │
│                    Action Head                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3. State处理差异

| 方面 | Pistar06 | SmolVLA |
|------|----------|---------|
| **处理方式** | 离散化后加入prompt | 直接作为额外token |
| **位置** | 在language model之前 | 在VLM transformer中 |
| **维度** | 256 bins离散化 | 直接连续值投影 |

---

## 改造方案

### 方案概述

将SmolVLA改造成价值函数需要：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SmolVLA → Value Function 改造                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  保留部分:                                                                │
│  ✅ SmolVLM Vision Encoder (冻结)                                       │
│  ✅ VLlama3 Language Model (冻结)                                       │
│  ✅ VLM Transformer (冻结或微调)                                        │
│                                                                           │
│  移除部分:                                                                │
│  ❌ Expert LM (不需要action生成)                                         │
│  ❌ Action Decoder (不需要预测action)                                    │
│                                                                           │
│  新增部分:                                                                │
│  ➕ Value Projection Layer (VLM特征→价值空间)                            │
│  ➕ Value Head (价值分布预测)                                            │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### 详细改造步骤

#### Step 1: 提取VLM Prefix特征

```python
def extract_vlm_prefix_features(
    self,
    images,          # [B, N, C, H, W]
    task_text,       # [B] (string list)
    state,           # [B, state_dim]
):
    """
    从SmolVLA的VLM部分提取prefix特征
    """
    # 1. 嵌入图像
    img_embs = []
    for img in images.transpose(0, 1):  # 遍历相机
        img_emb = self.vlm_with_expert.embed_image(img)  # [B, num_patches, 768]
        img_embs.append(img_emb)

    # 2. 嵌入文本
    tokens = self.tokenizer(task_text, ...)  # [B, T]
    lang_emb = self.vlm_with_expert.embed_language_tokens(tokens)  # [B, T, 960]

    # 3. 嵌入state
    state_emb = self.state_proj(state)  # [B, 1, 960]

    # 4. 拼接所有输入 (与SmolVLA相同的方式)
    prefix_embs = torch.cat([
        img_start_token,
        *img_embs,
        img_end_token,
        lang_emb,
        state_emb
    ], dim=1)  # [B, prefix_len, 960]

    # 5. 通过VLM transformer (冻结或微调)
    with torch.set_grad_enabled(not self.freeze_vlm):
        vlm_output = self.vlm_with_expert.vlm.model(
            inputs_embeds=prefix_embs,
            attention_mask=prefix_mask,
        )

    # 6. 提取最后一层的输出
    prefix_features = vlm_output.last_hidden_state  # [B, prefix_len, 960]

    return prefix_features
```

#### Step 2: 添加Value Projection

```python
class SmolVLAValueFunction(nn.Module):
    def __init__(self, smolvla_base_path, config):
        super().__init__()

        # 1. 加载SmolVLA base
        self.smolvla_base = load_smolvla_base(smolvla_base_path)
        self.vlm_with_expert = self.smolvla_base.vlm_with_expert

        # 2. 冻结VLM (可选)
        if config.freeze_vlm:
            for param in self.vlm_with_expert.vlm.parameters():
                param.requires_grad = False

        # 3. Value Projection Layer
        vlm_hidden_size = 960  # VLlama3 hidden size
        self.value_proj = nn.Sequential(
            nn.LayerNorm(vlm_hidden_size),
            nn.Linear(vlm_hidden_size, config.value_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.value_hidden_dim, config.value_hidden_dim),
        )

        # 4. Value Head
        self.value_head = nn.Linear(config.value_hidden_dim, config.num_bins)

        # 5. Bin centers (用于计算期望值)
        self.register_buffer(
            'bin_centers',
            torch.linspace(config.bin_min, config.bin_max, config.num_bins)
        )

    def forward(self, images, task_text, state):
        # 1. 提取VLM prefix特征
        prefix_features = self.extract_vlm_prefix_features(images, task_text, state)

        # 2. Mean pooling (取所有token的平均)
        pooled_features = prefix_features.mean(dim=1)  # [B, 960]

        # 3. Value projection
        value_features = self.value_proj(pooled_features)  # [B, value_hidden_dim]

        # 4. Value logits
        value_logits = self.value_head(value_features)  # [B, num_bins]

        return value_logits

    def predict_value(self, images, task_text, state):
        """推理时使用，返回期望值"""
        logits = self.forward(images, task_text, state)
        probs = F.softmax(logits, dim=-1)
        expected_value = (probs * self.bin_centers).sum(dim=-1)
        return expected_value
```

#### Step 3: 处理多相机输入

```python
def extract_vlm_prefix_features(self, images, task_text, state):
    """
    处理多个相机输入

    Args:
        images: [B, N, C, H, W] - N个相机
        task_text: List[str] - 任务描述
        state: [B, state_dim] - 机器人状态
    """
    bsize, num_cameras = images.shape[:2]

    # 处理每个相机的图像
    all_img_embs = []
    for cam_idx in range(num_cameras):
        cam_images = images[:, cam_idx]  # [B, C, H, W]
        cam_emb = self.vlm_with_expert.embed_image(cam_images)  # [B, num_patches, 768]
        all_img_embs.append(cam_emb)

    # 拼接所有相机特征
    # 方式1: 顺序拼接 (与SmolVLA相同)
    img_embs = torch.cat(all_img_embs, dim=1)  # [B, N*num_patches, 768]

    # 方式2: 平均池化 (简化)
    # img_embs = torch.stack(all_img_embs, dim=0).mean(dim=0)  # [B, num_patches, 768]

    return img_embs
```

---

## 实现步骤

### Phase 1: 基础实现（最小改动）

```python
# 文件: source/lehome/lehome/policies/smolvla_value.py

class SmolVLAValueFunction(nn.Module):
    """基于SmolVLA的价值函数"""

    def __init__(self, smolvla_policy, config):
        super().__init__()

        # 复用SmolVLA的VLM部分
        self.smolvla_policy = smolvla_policy
        self.vlm_with_expert = smolvla_policy.vlm_with_expert

        # 冻结VLM (与Expert相同)
        for param in self.vlm_with_expert.parameters():
            param.requires_grad = False

        # 只训练value head
        vlm_hidden_size = 960
        self.value_head = nn.Sequential(
            nn.Linear(vlm_hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 201),  # 201 bins for [-1, 0]
        )

    def forward(self, batch):
        # 使用SmolVLA的embedding方法
        with torch.no_grad():
            prefix_embs, pad_masks, att_masks = self.smolvla_policy.embed_prefix(
                images=batch["observation.images"],
                img_masks=batch["observation.image_attention_mask"],
                lang_tokens=batch["observation.language_tokens"],
                lang_masks=batch["observation.language_attention_mask"],
                state=batch["observation.state"],
            )

            # 通过VLM (冻结)
            vlm_output = self.vlm_with_expert.vlm.model(
                inputs_embeds=prefix_embs,
                attention_mask=att_masks,
            )
            features = vlm_output.last_hidden_state  # [B, T, 960]

        # Mean pooling
        pooled = features.mean(dim=1)  # [B, 960]

        # Value prediction
        value_logits = self.value_head(pooled)  # [B, 201]
        return value_logits
```

### Phase 2: 优化版本（更好的特征利用）

```python
class SmolVLAValueFunctionV2(nn.Module):
    """改进版：利用更多VLM特征"""

    def __init__(self, smolvla_policy, config):
        super().__init__()

        self.smolvla_policy = smolvla_policy
        self.vlm_with_expert = smolvla_policy.vlm_with_expert

        # 冻结VLM
        for param in self.vlm_with_expert.parameters():
            param.requires_grad = False

        vlm_hidden_size = 960

        # 使用attention pooling替代mean pooling
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=vlm_hidden_size,
            num_heads=8,
            batch_first=True
        )

        # 更大的value head
        self.value_head = nn.Sequential(
            nn.Linear(vlm_hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 201),
        )

        # 可学习的query token
        self.value_query = nn.Parameter(torch.randn(1, 1, vlm_hidden_size))

    def forward(self, batch):
        with torch.no_grad():
            prefix_embs, pad_masks, att_masks = self.smolvla_policy.embed_prefix(...)
            vlm_output = self.vlm_with_expert.vlm.model(
                inputs_embeds=prefix_embs,
                attention_mask=att_masks,
            )
            features = vlm_output.last_hidden_state  # [B, T, 960]

        # Attention pooling
        bsize = features.shape[0]
        query = self.value_query.expand(bsize, -1, -1)  # [B, 1, 960]
        pooled, _ = self.attention_pool(
            query,
            features,
            key_padding_mask=~pad_masks.bool()
        )  # [B, 1, 960]
        pooled = pooled.squeeze(1)  # [B, 960]

        # Value prediction
        value_logits = self.value_head(pooled)
        return value_logits
```

---

## 代码框架

### 完整实现框架

```python
# source/lehome/lehome/values/smolvla/configuration_smolvla_value.py

from dataclasses import dataclass
from lerobot.configs.policies import PreTrainedConfig

@PreTrainedConfig.register_subclass("smolvla_value")
@dataclass
class SmolVLAValueConfig(PreTrainedConfig):
    """SmolVLA价值函数配置"""

    # SmolVLA base model
    smolvla_model_path: str = ""

    # 冻结设置
    freeze_vlm: bool = True

    # Value head配置
    value_hidden_dim: int = 512
    dropout: float = 0.1

    # 价值分布配置
    num_bins: int = 201
    bin_min: float = -1.0
    bin_max: float = 0.0

    # Pooling方式
    pooling_type: str = "attention"  # "mean" or "attention"
```

```python
# source/lehome/lehome/values/smolvla/modeling_smolvla_value.py

class SmolVLAValueFunction(nn.Module):
    def __init__(self, cfg: SmolVLAValueConfig):
        super().__init__()

        # 加载SmolVLA base
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        smolvla_cfg = SmolVLAConfig.from_pretrained(cfg.smolvla_model_path)
        self.smolvla_base = SmolVLAPolicy(config=smolvla_cfg)

        # 冻结VLM
        if cfg.freeze_vlm:
            for param in self.smolvla_base.vlm_with_expert.parameters():
                param.requires_grad = False

        # Value components
        vlm_hidden_size = 960
        if cfg.pooling_type == "attention":
            self.pooling = nn.MultiheadAttention(vlm_hidden_size, 8, batch_first=True)
            self.query_token = nn.Parameter(torch.randn(1, 1, vlm_hidden_size))
        else:
            self.pooling = None

        self.value_head = self._build_value_head(cfg)

        # Bin centers
        self.register_buffer(
            'bin_centers',
            torch.linspace(cfg.bin_min, cfg.bin_max, cfg.num_bins)
        )

    def _build_value_head(self, cfg):
        layers = [
            nn.Linear(960, cfg.value_hidden_dim),
            nn.LayerNorm(cfg.value_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.value_hidden_dim, cfg.value_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.value_hidden_dim, cfg.num_bins),
        ]
        return nn.Sequential(*layers)

    def forward(self, batch):
        # Extract VLM features
        features = self._extract_features(batch)

        # Pooling
        pooled = self._pool_features(features, batch)

        # Value prediction
        value_logits = self.value_head(pooled)
        return value_logits

    def predict_value(self, batch):
        logits = self.forward(batch)
        probs = F.softmax(logits, dim=-1)
        return (probs * self.bin_centers).sum(dim=-1)
```

---

## 训练流程

### 训练命令

```bash
python -m scripts.train_value_function \
  --config.path=configs/value_train_smolvla.yaml \
  --dataset.repo_id=lehome/hil_pant_short \
  --value.type=smolvla_value \
  --value.smolvla_model_path=outputs/moe_train/smolvla_moe_base \
  --value.freeze_vlm=True \
  --value.pooling_type=attention \
  --output_dir=outputs/value_train/smolvla_pant_short \
  --batch_size=32 \
  --steps=10000
```

### 配置文件

```yaml
# configs/value_train_smolvla.yaml

value:
  type: smolvla_value
  smolvla_model_path: outputs/moe_train/smolvla_moe_base
  freeze_vlm: true
  value_hidden_dim: 512
  dropout: 0.1
  num_bins: 201
  bin_min: -1.0
  bin_max: 0.0
  pooling_type: attention

dataset:
  repo_id: lehome/hil_pant_short
  root: Datasets/hil/pant_short

training:
  batch_size: 32
  learning_rate: 1e-4
  steps: 10000
  val_every_n_steps: 500

targets:
  success_field: episode_success
  default_success: failure
  c_fail_coef: 1.0
```

---

## 总结

### 与Pistar06的主要区别

| 方面 | Pistar06 | SmolVLA Value Function |
|------|----------|------------------------|
| **Vision** | SigLIP SO400M (1152) | SmolVLM Vision (768) |
| **Language** | Gemma 3 270M (2304) | VLlama3 (960) |
| **Fusion** | 独立fusion层 | VLM内部融合 |
| **State** | 离散化到prompt | 作为token输入 |
| **输出** | 201 bins | 201 bins (相同) |

### 优势

✅ **特征一致性**: 与Expert使用相同的VLM backbone
✅ **训练成本低**: 只需训练value head
✅ **特征质量**: SmolVLA预训练在机器人数据上
✅ **实现简单**: 复用现有SmolVLA代码

### 需要注意

⚠️ **特征维度不同**: SmolVLA (960) vs Pistar06 (1024)
⚠️ **融合方式不同**: 需要适配VLM的输出
⚠️ **多相机处理**: 需要决定是否pooling或顺序处理

---

*创建于2026-03-19*
