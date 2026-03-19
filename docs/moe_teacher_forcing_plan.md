# MoE-SmolVLA Teacher Forcing 实现计划

基于 Router probe 实验结论（fine-tuned VLM + 4层MLP = 100%），实现真正的 Teacher Forcing 训练，让 Router 头和 VLM backbone 在联合 loss 下共同优化。

## 实验结论摘要

| 实验 | 准确率 |
|------|--------|
| 冻结 base VLM probe + 2层MLP + 50 epoch | 82-84% |
| Action Training fine-tune VLM probe（无 router loss）| 58%（变差！灾难性遗忘） |
| **冻结 base VLM (smolvla_base) + 4层MLP + 200 epoch** | **100%** |
| **结论** | 不需要 fine-tune VLM！Frozen features 已足够，关键是分类器深度和训练充分性 |

## 实现步骤

---

### ① Dataset — garment_type 标签注入

**新增文件**: `scripts/utils/garment_type_mapper.py`

从 `garment_info.json` 构建 `episode_index → garment_type_label` 映射，通过 DataLoader 的 collate_fn 注入到每个 batch。

```python
# garment_info.json 的 key 结构: "Top_Short_Seen_0", "Pant_Long_Seen_3", ...
TYPE_MAP = {"Top_Short": 3, "Top_Long": 2, "Pant_Short": 0, "Pant_Long": 1}

def build_episode_to_type(garment_info_path: str) -> dict[int, int]:
    """
    解析 garment_info.json，返回 {global_episode_idx: garment_type_label}
    注意：inner key (0~24) 是各服装内部的 episode 序号，需要配合顺序解析成全局 index
    """
    ...

class GarmentTypeCollator:
    """DataLoader collate_fn，从 batch["episode_index"] 查表注入 batch["garment_type"]"""
    def __call__(self, batch: list) -> dict:
        # batch["garment_type"] = Tensor[B], dtype=torch.long
        ...
```

**关键约束**：`garment_info.json` 的 inner key `0~24` 是每个 garment 内部 episode 的编号，需要按照 garment 顺序累计偏移量才能得到全局 episode index。

---

### ② Model — Router Head

**修改文件**: `third_party/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py`

在 `VLAFlowMatching.__init__` 中添加 Router head（结构与 probe 实验分类器一致）：

```python
hidden_size = self.vlm_with_expert.config.text_config.hidden_size

self.router_head = nn.Sequential(
    nn.Linear(hidden_size * 3, 512),   # 输入: mean+std+max pooling
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
    nn.Linear(128, num_garment_types),  # 输出: 4类 logits
)

def route(self, images: list[torch.Tensor]) -> torch.Tensor:
    """提取首个摄像头的视觉特征并返回 garment_type 分类 logits [B, 4]"""
    img = images[0]  # top_rgb [B, C, H, W]
    img_emb = self.vlm_with_expert.embed_image(img)  # [B, N_tokens, D]
    # mean+std+max pooling
    mean_f = img_emb.mean(dim=1)
    std_f  = img_emb.std(dim=1)
    max_f  = img_emb.max(dim=1).values
    feat = torch.cat([mean_f, std_f, max_f], dim=-1)  # [B, D*3]
    feat = F.normalize(feat, p=2, dim=-1)
    return self.router_head(feat)  # [B, 4]
```

**修改 `VLAFlowMatching.forward()`**，当 `garment_type` 不为 None 时加入 Router 分类 loss：

```python
def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions,
            noise=None, time=None, garment_type=None) -> tuple:
    # ... 原有 flow matching loss ...
    action_loss = F.mse_loss(u_t, v_t, reduction="none")  # [B, T, D]

    loss_dict = {"action_loss": action_loss.mean().item()}

    if garment_type is not None:
        router_logits = self.route(images)
        router_cls_loss = F.cross_entropy(router_logits, garment_type)
        loss_dict["router_cls_loss"] = router_cls_loss.item()
        # 联合 loss：action loss + λ * router loss
        combined = action_loss + self.config.router_cls_weight * router_cls_loss
    else:
        combined = action_loss

    return combined, loss_dict
```

**修改 `SmolVLAConfig`** 新增字段：
```python
router_cls_weight: float = 0.1   # Router loss 权重，建议范围 0.05~0.2
num_garment_types: int = 4
```

---

### ③ Training Config

**新增文件**: `configs/train_smolvla_moe.yaml`

```yaml
dataset:
  repo_id: repo_smolvla_four_types
  root: Datasets/example/four_types_merged
  garment_info_path: Datasets/example/four_types_merged/meta/garment_info.json

policy:
  type: smolvla
  device: cuda
  push_to_hub: false

  # 解冻 VLM backbone，让 Router 梯度能反传
  train_expert_only: false
  freeze_vision_encoder: false
  train_state_proj: true

  # MoE Router 配置
  router_cls_weight: 0.1
  num_garment_types: 4

  input_features:
    observation.state:
      type: STATE
      shape: [12]
    observation.images.top_rgb:
      type: VISUAL
      shape: [3, 480, 640]
    observation.images.left_rgb:
      type: VISUAL
      shape: [3, 480, 640]
    observation.images.right_rgb:
      type: VISUAL
      shape: [3, 480, 640]

  output_features:
    action:
      type: ACTION
      shape: [12]

output_dir: outputs/train/smolvla_moe
batch_size: 16
steps: 100000
save_freq: 2000
log_freq: 100

wandb:
  enable: true
  project: lehome-challenge
```

---

### ④ Inference — Sticky Routing

**修改文件**: `scripts/eval_policy/lerobot_policy.py`（或对应 wrapper）

在 `reset()` 和 `select_action()` 中加入 Voting 逻辑：

```python
def reset(self):
    super().reset()
    self._router_logits_buffer = []
    self._locked_expert_idx = None
    self._voting_frames = 5  # 前 N 帧做 Voting

def select_action(self, observation_dict: dict) -> np.ndarray:
    # === Voting 阶段：前 N 帧锁定 Expert ===
    if self._locked_expert_idx is None:
        images = self._prepare_images(observation_dict)
        with torch.no_grad():
            logits = self.policy.model.route(images)  # [1, 4]
        self._router_logits_buffer.append(logits.cpu())

        if len(self._router_logits_buffer) >= self._voting_frames:
            # 累计 logits 投票，argmax 锁定 Expert
            agg_logits = torch.stack(self._router_logits_buffer).sum(0)  # [1, 4]
            self._locked_expert_idx = agg_logits.argmax(-1).item()
            logger.info(f"Router 锁定 Expert: {self._locked_expert_idx}")

        # 空窗期 fallback：使用 Voting 阶段的混合推理（或输出零动作）
        # 方案A：零动作（简单，真机可能不安全）
        return np.zeros(self.action_dim)
        # 方案B：用原版 SmolVLA 推理（推荐，安全）
        # return super().select_action(observation_dict)

    # === 锁定后：强制路由到对应 Expert ===
    # 将 locked_expert_idx 传入 model，控制 action expert 选择
    return self._select_action_with_expert(observation_dict, self._locked_expert_idx)
```

> **空窗期方案选择**：
> - 方案 A（零动作）：简单，但真机可能触发安全限位
> - 方案 B（SmolVLA fallback）：前 N 帧照常执行动作，同时并行做 Voting。实现稍复杂但对真机友好

---

## Verification Plan

```bash
# 1. 验证 episode → type 映射
python3 -c "
from scripts.utils.garment_type_mapper import build_episode_to_type
mapper = build_episode_to_type('Datasets/example/four_types_merged/meta/garment_info.json')
print(f'Total episodes mapped: {len(mapper)}')  # 期望 1000
print(f'ep0 type: {mapper[0]}')  # 期望 3 (top_short)
"

# 2. 快速训练 2000 步验证 loss 下降
# 期望看到 wandb 里 router_cls_loss 从 ln(4)≈1.386 下降到 <0.5

# 3. 训练完成后跑 Router Probe 测试
python3 scripts/test_router_quick.py
# 期望: 整体准确率 > 95%（vs 当前 probe 100%）
```
