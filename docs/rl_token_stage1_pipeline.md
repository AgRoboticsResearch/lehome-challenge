# RL Token Stage 1: Offline Training Pipeline

## Overview

RL Token 的核心思想：不直接让 RL 策略输出 action，而是让 RL 学习一个紧凑的 latent token，这个 token 能够重建 VLA 的内部表征。Stage 1 通过离线 demo 数据训练 Encoder/Decoder，产出可用于 Stage 2 在线 RL 的 `z_rl`。

参考文献: *RL Token: Bootstrapping Online RL with Vision-Language-Action Models* (Physical Intelligence, 2025)

---

## SmolVLA Architecture Dimensions (SmolVLM2-500M)

> **Verified by `scripts/utils/inspect_prefix_tokens.py` on 2026-04-02**

| 参数 | 值 | 来源 |
|------|-----|------|
| **vlm_hidden_size** | **960** | `text_config.hidden_size` |
| vlm_intermediate_size | 2560 | `text_config.intermediate_size` |
| vlm_num_layers (used) | 16 | `num_vlm_layers: 16` (total 32, use first 16) |
| vlm_num_attention_heads | 15 | `text_config.num_attention_heads` |
| vlm_head_dim | 64 | `text_config.head_dim` |
| vlm_num_key_value_heads | 5 | `text_config.num_key_value_heads` |
| **expert_hidden_size** | **720** | `int(960 * 0.75)` |
| vision_hidden_size | 768 | `vision_config.hidden_size` |
| vision_image_size | 512 | `vision_config.image_size` |
| vision_patch_size | 16 | `vision_config.patch_size` |
| pixel_shuffle scale_factor | 4 | `config.scale_factor` |
| image_seq_len per image | **64** | **实测: SigLIP→connector 输出 (B, 64, 960)** |
| language_seq_len | **3** | **实测: "fold the garment" → 3 tokens** |
| state_seq_len | **1** | pad 32D → state_proj → 1 token |
| **prefix_seq_len (M)** | **196** | **实测: 192 img + 3 lang + 1 state** |
| prefix_length (config) | -1 | 不做额外 padding |
| max_state_dim | 32 | `configuration_smolvla.py` |
| max_action_dim | 32 | `configuration_smolvla.py` |
| chunk_size H | 50 | `configuration_smolvla.py` |

### Token Composition (实测)

```
Camera 0: SigLIP→connector  → 64 tokens × 960D  (top_rgb)
Camera 1: SigLIP→connector  → 64 tokens × 960D  (left_rgb)
Camera 2: SigLIP→connector  → 64 tokens × 960D  (right_rgb)
Language: "fold the garment" →  3 tokens × 960D
State:    12D → pad 32D → proj →  1 token  × 960D
                                ─────────────────
                          Total: 196 tokens × 960D
```

### Projection Layers

| Layer | Shape | File Location |
|-------|-------|---------------|
| `state_proj` | `Linear(32 -> 960)` | `modeling_smolvla.py:571-573` |
| `action_in_proj` | `Linear(32 -> 720)` | `modeling_smolvla.py:574` |
| `action_out_proj` | `Linear(720 -> 32)` | `modeling_smolvla.py:575` |
| `action_time_mlp_in` | `Linear(1440 -> 720)` | `modeling_smolvla.py:577-579` |
| `action_time_mlp_out` | `Linear(720 -> 720)` | `modeling_smolvla.py:580-582` |

### Comparison with pi0.6 (Paper)

| | pi0.6 (Paper) | SmolVLA (This Project) |
|---|---|---|
| VLM backbone | Gemma-2B (2B params) | SmolVLM2-500M (500M params) |
| **vlm_hidden_size** | **2048** | **960** |
| **RL Token dim z_rl** | **2048** | **960** |
| image encoder | SigLIP (400M) | SigLIP (~93M, built-in) |
| action expert | 860M | ~375M (0.75x hidden) |
| **Prefix tokens M** | **~2300** | **196** |
| Images | 4 cameras | 3 cameras (top, left, right) |
| action dim | 14D | 12D |
| chunk_size H | 50 | 50 |
| RL chunk C | 10 | suggested 10-20 |
| control freq | 50Hz | 30Hz (demo) / 120Hz (sim) |

SmolVLA 的 196 tokens 比 π0 的 ~2300 tokens 小约 **12x**，这意味着 RL Token Encoder 的交叉注意力计算量小一个数量级。

---

## Stage 1 Data Flow

```
LeRobot Demo Data (30Hz):
  observation.images.{top,left,right}_rgb: (480, 640, 3)
  observation.state: (12,)
  task_description: "fold the garment"
        |
        v
  SmolVLA prepare_images() -> resize 512x512, normalize [-1,1]
  SmolVLA prepare_state() -> pad 32D
  Tokenize task_description -> lang_tokens (3 tokens)
        |
        v
  embed_prefix():
    3 x SigLIP(512x512) -> 3 x 64 = 192 image tokens (960D)
    + lang_tokens -> 3 language tokens (960D)
    + state_proj(32->960) -> 1 state token (960D)
    = prefix_embs: (B, 196, 960)
        |
        v
  VLM 16 layers pure self-attn (fill_kv_cache=True, Expert 全程 None)
        |
        v
  final RMSNorm
        |
        v
  z_{1:M} = (B, 196, 960)  <-- pure perception features
        |
        v
  Drop lang tokens [pos 192:195] (Paper Footnote 1: fixed instruction = zero info)
        |
        v
  z_target = (B, 193, 960)  [192 img + 1 state]
        |
        v
  append e_rl (learnable, 960D) -> (B, 194, 960)
        |
        v
  RL Token Encoder (small Transformer, 2-4 layers, dim=960)
        |
        v
  z_rl = output[:, -1, :] -> (B, 960)
        |
        |--> RL Token Decoder (autoregressive reconstruction of 193 tokens)
        |     Loss: sum_i ||d_phi([z_rl, z_bar_{1:i-1}])_i - z_bar_i||^2
        |     where z_bar = sg(z_target), NOT including lang tokens
        |
        +--> Save weights; Stage 2 uses as Actor/Critic state input
```

---

## Key Design Analysis

### Prefix Pass Layer Behavior (实测确认)

**关键发现**: `fill_kv_cache=True` 时，条件判断短路，所有 16 层都走 `forward_attn_layer`。

In `sample_actions()` (`modeling_smolvla.py:817-825`):

```python
_, past_key_values = self.vlm_with_expert.forward(
    inputs_embeds=[prefix_embs, None],  # suffix=None
    fill_kv_cache=True,                  # True -> 短路条件
)
```

In `forward()` (`smolvlm_with_expert.py:426-431`):

```python
for layer_idx in range(num_layers):  # 0..15
    if (
        fill_kv_cache          # True -> 短路，所有层都进入此分支
        or "cross" not in ...
        or layer_idx % 2 == 0
    ):
        forward_attn_layer(...)   # ALL 16 layers use self-attn
    else:
        forward_cross_attn_layer(...)  # prefix pass 中永远不会走到这里
```

In `forward_attn_layer()`, Expert 分支被跳过:

```python
for i, hidden_states in enumerate(inputs_embeds):
    if hidden_states is None or layer is None:
        continue  # i=1 (expert) -> hidden_states=None -> SKIP
```

**Layer behavior summary**:

| 场景 | `fill_kv_cache` | 层行为 |
|------|----------------|--------|
| **Prefix pass (推理)** | `True` | **全部 16 层** → `forward_attn_layer` (self-attn) |
| **Denoising pass (推理)** | `False` | 偶数层 → self-attn, 奇数层 → cross-attn |
| **训练 (`forward`)** | `False` | 偶数层 → self-attn, 奇数层 → cross-attn |

**Conclusion**: During prefix pass, Expert is completely no-op. The even/odd interleaving structure (`self_attn_every_n_layers=2`) is entirely bypassed. z_{1:M} is produced by 16 consecutive self-attention layers of a frozen VLM — a very clean feature extraction path.

### Implications

1. **z_{1:M} is purely perceptual** - encodes only "what was seen" (images, language, state), no "how to act" (action expert info)
2. **RL Token's job is to bridge this gap** - infer "behavioral intent" from "perception", then Actor refines actions on top
3. **SmolVLA's even/odd interleaving is completely bypassed in prefix pass** - not just "Expert skipped", but `forward_cross_attn_layer` is never even called
4. **Stage 2 Actor MUST condition on both z_rl and a_hat** (VLA's sampled action) - z_rl provides perception, a_hat provides behavioral mode

### Extraction Point

RL Token should be extracted from **VLM prefix output** (`outputs_embeds[0]`):

| Option | Source | Dim | Recommended |
|--------|--------|-----|-------------|
| **A. VLM prefix output** | `outputs_embeds[0]` after RMSNorm | **(B, 196, 960)** | **Yes** - matches paper |
| B. Pre-transformer embeddings | `embed_prefix()` direct output | (B, 196, 960) | No - lacks deep fusion |
| C. Expert suffix output | `outputs_embeds[1]` | (B, chunk_size, 720) | No - different dim, wrong role |

Current code discards prefix output (`modeling_smolvla.py:779`):

```python
(_, suffix_out), _ = self.vlm_with_expert.forward(...)
#  ^ prefix_out discarded
```

Needs to be changed to:

```python
(prefix_out, suffix_out), _ = self.vlm_with_expert.forward(...)
```

---

## Image Token Pipeline Detail

```
Input image: (B, 3, 512, 512)
        |
        v
SigLIP patch embedding: patch_size=16, num_patches=(512/16)^2 = 1024
        |
        v
SigLIP transformer encoder + LayerNorm: (B, 1024, 768)
        |
        v
Connector - pixel_shuffle(scale_factor=4):
  spatial: 1024 / (4^2) = 64 tokens
  embed_dim: 768 * (4^2) = 12288
  result: (B, 64, 12288)
        |
        v
Connector - Linear projection: 12288 -> 960
        |
        v
Output: (B, 64, 960)  -- 64 image tokens per image (实测确认)
```

No perceiver resampler is used in SmolVLM2. The `resampler_n_latents=64` in config is vestigial.

---

## Language Token Drop (Paper Footnote 1)

> **Paper Footnote 1**: "In our experiments each task has a fixed language instruction, so we drop language embeddings in this step; the construction applies to all VLA embeddings in general."

### Rationale

Our task description "fold the garment on the table" is **identical for all frames** across all episodes. Reconstructing language tokens forces z_rl to memorize a constant — zero information gain for RL.

### Token Layout and Masking

```
z_{1:M} = 196 tokens total

Position:  [0 ............. 191] [192 193 194] [195]
Content:   [img_0 ... img_191]  [lang_0..2]   [state]
Keep:      [✓ ............ ✓]  [✗  ✗  ✗]    [✓]

After drop: z_target = (B, 193, 960)  [192 img + 1 state]
```

### Implementation (3 lines)

```python
# In __init__:
self.keep_mask = torch.cat([
    torch.ones(192, dtype=torch.bool),   # image tokens: keep
    torch.zeros(3, dtype=torch.bool),     # language tokens: drop
    torch.ones(1, dtype=torch.bool),      # state token: keep
])

# Before reconstruction loss:
z_target = z_vlm[:, self.keep_mask, :]  # [B, 193, 960]
```

The same mask applies to both decoder input (autoregressive prefix) and reconstruction target.

### Reconstruction Strategy Comparison

| Strategy | Target Tokens | Autoreg Steps | Info Retained | Decision |
|----------|--------------|---------------|---------------|----------|
| Full (naive) | 196 | 196 | 100% | Wasteful — lang is constant |
| **Drop lang (paper)** | **193** | **193** | **~99%** | **Chosen** |
| Last 10 only | 10 | 10 | ~30% | Too aggressive |
| State only | 1 | 1 | ~5% | Insufficient |

SmolVLA's 193 tokens is already 12x smaller than pi0's ~2300 — no need for more aggressive pruning.

---

## RL Token Encoder/Decoder Design

### Encoder Architecture Analysis

| 决策 | 选择 | 理由 |
|------|------|------|
| **层数** | **2 层**（按需增到 4） | VLM 已做 16 层深度融合；Encoder 只需聚合（aggregation），不需变换（transformation）。2 层足以让 e_rl 通过 attention 加权汇总 193 个 token。先保守起步，避免 83K 帧数据上过拟合 |
| **注意力** | **双向**（TransformerEncoder） | Paper 公式 `z_rl = g_θ(z, e_rl)_{M+1}` 意味着 e_rl 可见所有 z token。双向让中间 token 也获得全局上下文（跨视角融合），e_rl 聚合时得到更好的表示 |
| **Heads** | **15**（head_dim=64） | 匹配 VLM 的 15 heads × 64 head_dim。64 是 Tensor Core 自然对齐单位。16 heads 导致 head_dim=60，反而不利于硬件效率 |
| **FFN 维度** | **1920**（0.75x VLM） | VLM 用 2560，但 Encoder 输入已是高度结构化的 VLM 输出，不需要完整 FFN 容量 |
| **z_rl 提取** | `output[:, -1, :]` | e_rl 拼在序列最后，取最后位置输出 |

### Encoder Specification

```python
# nn.TransformerEncoder specification
encoder_layer = nn.TransformerEncoderLayer(
    d_model=960,
    nhead=15,               # head_dim=64, match VLM
    dim_feedforward=1920,    # 0.75x VLM's 2560
    num_layers=2,            # conservative start; scale to 4 if needed
    activation='gelu',       # match Gemma2
    batch_first=True,
    norm_first=True,         # pre-norm for training stability
)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

# Learnable RL token
e_rl = nn.Parameter(torch.randn(1, 1, 960) * 0.02)  # small init

# Forward:
#   x = torch.cat([z_target, e_rl.expand(B, -1, -1)], dim=1)  # (B, 194, 960)
#   out = encoder(x)                                            # (B, 194, 960)
#   z_rl = out[:, -1, :]                                        # (B, 960)
```

**参数量**: ~12M (2 层) / ~24M (4 层)

### Decoder Architecture Analysis

| 决策 | 选择 | 理由 |
|------|------|------|
| **模块类型** | TransformerEncoder + causal mask | 是因果语言模型，不是 encoder-decoder architecture。z_rl 是序列的一部分，不是外部 memory。 PyTorch TransformerDecoder 需要 cross-attention， 是过度设计 |
| **训练方式** | Teacher forcing，一次 forward 并行 193 步 | Decoder 不做推理，只在 Stage 1 训练时使用。无需 KV-cache 或推理优化 |
| **层数** | **2 层**（同 Encoder） | Decoder 任务比 Encoder 更难（因果展开），先一致，再按需增加 |
| **维度** | 与 Encoder 完全一致 | d_model=960, nhead=15, dim_ff=1920 | 减少调参空间 |
| **Output projection** | Linear(960, 960) | 独立层，无 weight tying |
| **Causal mask** | 缓存复用（`generate_square_subsequent_mask`)） | 只创建一次 |

### Decoder Specification

```python
class RLTokenDecoder(nn.Module):
    """Autoregressive reconstruction of z_{1:M'} from z_rl.

    Training-only module. Discarded after Stage 1.
    Uses teacher forcing + causal mask for single-pass parallel prediction.
    """
    def __init__(self, d_model=960, nhead=15, dim_feedforward=1920, num_layers=2):
        super().__init__()
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, d_model)
        self._causal_mask = None  # lazy init, cache after first use

    def forward(self, z_rl, z_target):
        """
        Args:
            z_rl:     (B, 960)    - RL token from encoder
            z_target: (B, 193, 960) - target tokens (with stop-gradient applied outside)
        Returns:
            pred:   (B, 193, 960) - predictions for each position
        """
        B, M, D = z_target.shape

        # Causal mask (cache after first creation)
        if self._causal_mask is None or self._causal_mask.size(0) != M:
            self._causal_mask = nn.Transformer.generate_square_subsequent_mask(M)

        # Teacher forcing: [z_rl | z_0 ... z_{M-2}]
        decoder_in = torch.cat(
            [z_rl.unsqueeze(1),       # position 0: z_rl
            z_target[:, :-1, :],             # positions 1..M-1: z_0 .. z_{M-2}
        ], dim=1)  # (B, M, D)

        # Parallel prediction with causal mask
        output = self.decoder(decoder_in, mask=self._causal_mask)  # (B, M, D)
        pred = self.output_proj(output)                                # (B, M, D)
        return pred
```

**Loss**:
```python
pred = decoder(z_rl, z_target.detach())  # sg(z_target)
loss = F.mse_loss(pred, z_target.detach())
```

### Dimension Options for z_rl

| Dim | Pros | Cons | Recommendation |
|-----|------|------|----------------|
| **960** (no compression) | No info loss, simplest | Actor/Critic input larger, RL sample efficiency lower | **Start here** |
| 512 (moderate compression) | Balance | Needs projection layer | Experiment later |
| 256 (aggressive compression) | RL friendly, lightweight | Info bottleneck too tight for 193x960 input | High risk |

**Note**: Stage 1 uses z_rl at 960D (no projection). Stage 2 Actor may optionally project down to 256D — that's a separate design decision. Stage 1's job is faithful reconstruction; compression is Stage 2's concern.

---

## Caveats

1. **Training vs Inference discrepancy**: During SmolVLA training, `fill_kv_cache=False` and suffix is present, so VLM weights are optimized with suffix context. Prefix-only extraction is "missing suffix context" vs training. Paper (pi0) has same issue and ignores it successfully.

2. **Reconstruction cost**: Autoregressive decoding of 196 tokens x 960D is expensive (but much cheaper than pi0's ~2300 tokens). Possible optimizations:
   - Only reconstruct language + state tokens (4 tokens), let image tokens pass via attention
   - Pool image tokens before reconstruction
   - Use non-autoregressive (parallel) decoder

3. **960 vs 2048**: SmolVLA's 960D is smaller than pi0's 2048D. The information bottleneck is tighter. Monitor reconstruction quality carefully.

---

## Files to Create/Modify

1. `source/lehome/lehome/models/rl_token.py` - RLTokenEncoder + RLTokenDecoder
2. `source/lehome/lehome/models/vla_embedding_hook.py` - Extract prefix_out from frozen SmolVLA
3. `scripts/train_rl_token.py` - Stage 1 training script
4. `configs/train_rl_token.yaml` - Configuration

## Code Insertion Point

`modeling_smolvla.py:779` - Change `(_, suffix_out)` to `(prefix_out, suffix_out)` to capture prefix embeddings.

## Verification Script

Run `python -m scripts.utils.inspect_prefix_tokens` to verify all dimensions.
