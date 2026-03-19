# Router Training Guide

用于 MoE-SmolVLA 系统的 Garment Router 训练文档。

## 概述

Router 的作用是根据 VLM 提取的视觉特征，将输入路由到正确的 Expert。它是 MoE 系统的关键组件。

## 关键修复 ⚠️

原始测试脚本 (`test_router_quick.py`) 存在 **严重的 label 映射错误**：

### 错误映射 (原始):
```python
EPISODE_TO_TYPE = {
    (0, 250): ("top_short", 3),    # ❌ Label 顺序错误
    (250, 500): ("top_long", 2),
    (500, 750): ("pant_short", 0),
    (750, 1000): ("pant_long", 1),
}
TYPE_NAMES = ["pant_short", "pant_long", "top_long", "top_short"]  # ❌ 顺序混乱
```

### 正确映射 (已修复):
```python
EPISODE_TO_TYPE = {
    (0, 250): "top_short",     # ✅ 与数据集顺序一致
    (250, 500): "top_long",
    (500, 750): "pant_short",
    (750, 1000): "pant_long",
}
TYPE_NAMES = ["top_short", "top_long", "pant_short", "pant_long"]  # ✅ 正确顺序
```

## 数据集结构

`four_types_merged` 数据集包含:
- **40 个 garment variants** (每种类型 10 个变体)
- **1000 个 episodes** (每个变体 25 个 episodes)
- **Episode 顺序**: top_short → top_long → pant_short → pant_long

| 类型 | Episode 范围 | 变体数量 | Episodes |
|------|-------------|---------|----------|
| top_short | 0-249 | 10 | Top_Short_Seen_0 到 Top_Short_Seen_9 |
| top_long | 250-499 | 10 | Top_Long_Seen_0 到 Top_Long_Seen_9 |
| pant_short | 500-749 | 10 | Pant_Short_Seen_0 到 Pant_Short_Seen_9 |
| pant_long | 750-999 | 10 | Pant_Long_Seen_0 到 Pant_Long_Seen_9 |

## 使用方法

### 1. 训练 Router

```bash
# 基础训练 (使用默认参数)
python -m scripts.train_router

# 自定义参数
python -m scripts.train_router \
    --dataset_root Datasets/example/four_types_merged \
    --output_dir outputs/train/router \
    --vlm_model lerobot/smolvla_base \
    --device cuda \
    --episode_stride 4 \
    --epochs 200 \
    --batch_size 32 \
    --lr 5e-4

# CPU 训练 (较慢)
python -m scripts.train_router --device cpu
```

### 2. 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset_root` | `Datasets/example/four_types_merged` | 数据集根目录 |
| `--output_dir` | `outputs/train/router` | 输出目录 |
| `--vlm_model` | `lerobot/smolvla_base` | VLM 模型路径 |
| `--device` | `cuda` | 训练设备 (cuda/cpu) |
| `--episode_stride` | `4` | Episode 采样步长 (每4个取1个) |
| `--epochs` | `200` | 训练轮数 |
| `--batch_size` | `32` | 批次大小 |
| `--lr` | `5e-4` | 学习率 |
| `--val_split` | `0.2` | 验证集比例 |
| `--cache_dir` | `{output_dir}/cache` | 特征缓存目录 |
| `--mode` | `train` | 运行模式 (train/eval) |
| `--checkpoint` | `None` | 评估模式下的检查点路径 |

### 3. 评估已训练的 Router

```bash
# 评估最佳模型
python -m scripts.train_router \
    --mode eval \
    --checkpoint outputs/train/router/checkpoints/best/router.pt \
    --device cuda
```

### 4. 输出文件

训练完成后，会生成以下文件:

```
outputs/train/router/
├── checkpoints/
│   ├── best/
│   │   └── router.pt          # 最佳验证准确率的模型
│   └── last/
│       └── router.pt          # 最后一个 epoch 的模型
└── cache/
    └── router_features.pt     # 缓存的 VLM 特征
```

## Router 架构

### 特征提取
- **输入**: RGB 图像 (top_rgb 相机)
- **预处理**: resize_with_pad(224, 224) + 归一化 [0,1] → [-1,1]
- **VLM**: SmolVLM2-500M (frozen)
- **特征**: mean + std + max pooling (960 × 3 = 2880 dims)
- **归一化**: L2 normalization

### 分类器
```
Input: [batch, 2880]
→ Linear(2880, 512) + LayerNorm + ReLU + Dropout(0.3)
→ Linear(512, 256) + LayerNorm + ReLU + Dropout(0.3)
→ Linear(256, 128) + ReLU + Dropout(0.2)
→ Linear(128, 4)
→ Output: [batch, 4] (logits)
```

### 训练配置
- **优化器**: AdamW (lr=5e-4, weight_decay=1e-3)
- **调度器**: CosineAnnealingWarmRestarts (T_0=50, T_mult=2)
- **损失函数**: CrossEntropyLoss (label_smoothing=0.1)
- **训练轮数**: 200 epochs

## 检查点格式

```python
{
    "router_state_dict": nn.Module.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "config": {
        "input_dim": 2880,
        "hidden_dims": [512, 256, 128],
        "num_classes": 4,
        "type_names": ["top_short", "top_long", "pant_short", "pant_long"],
        "episode_to_type": {...},
    },
    "metrics": {
        "val_acc": 1.0,
        "epoch": 180,
    },
    "timestamp": "2025-01-15T10:30:00",
}
```

## 集成到 MoE Policy

在 MoE Policy 中使用 Router:

```python
from scripts.train_router import GarmentRouter

# 加载 Router
checkpoint = torch.load("outputs/train/router/checkpoints/best/router.pt")
config = checkpoint["config"]

router = GarmentRouter(
    input_dim=config["input_dim"],
    hidden_dims=config["hidden_dims"],
    num_classes=config["num_classes"],
).to(device)
router.load_state_dict(checkpoint["router_state_dict"])
router.eval()

# 使用 Router
with torch.no_grad():
    result = router.predict(vlm_features)
    predicted_class = result["predicted_class"].item()
    class_name = config["type_names"][predicted_class]
    confidence = result["confidence"].item()
```

## 预期结果

基于原始测试脚本 (修复后)，预期结果:
- **训练准确率**: ~100%
- **验证准确率**: ~100%
- **混淆矩阵**: 对角线占主导

## 故障排查

### 问题 1: CUDA Out of Memory
```bash
# 解决方案: 减小 batch_size 或使用 CPU
python -m scripts.train_router --batch_size 16 --device cpu
```

### 问题 2: 特征提取慢
```bash
# 解决方案: 使用缓存 (默认启用)
python -m scripts.train_router --cache_dir outputs/train/router/cache
```

### 问题 3: 准确率低
- 检查 `--dataset_root` 是否正确
- 检查 `--vlm_model` 是否正确加载
- 检查数据集是否完整 (1000 episodes)

## 下一步

1. ✅ 训练 Router
2. ⏳ 实现 MoESmolVLAPolicy
3. ⏳ 集成 Router + Experts
4. ⏳ 测试端到端推理

## 相关文件

- 训练脚本: `scripts/train_router.py`
- 原始测试脚本: `scripts/test_router_quick.py` (⚠️ 有 bug)
- MoE 设计文档: `docs/moe_design_v2.md`
