# Model Arithmetic (MA) 在 LeHome 项目中的应用分析

## 概述

本文档分析了在 LeHome Challenge 项目中应用 Model Arithmetic (MA) 的可行性，特别是针对 SmolVLA 模型的方案。

---

## 1. 什么是 Model Arithmetic (MA)

Model Arithmetic 是一种**权重空间合并策略**，通过加权平均多个模型的参数来创建更强的组合模型。

### 核心公式
```
mixed_params = Σ(w_i * params_i)  where Σw_i = 1
```

### 六种混合方法

| 方法 | 描述 |
|------|------|
| `average` | 简单等权重 (1/N) |
| `inverse_loss` | 权重 ∝ 1/loss² (更好的模型权重更高) |
| `gradient_descent` | Adam 优化器寻找最优权重 |
| `adaptive_gradient_descent` | GD + loss 自适应步长 |
| `greedy` | 前向选择，逐步添加最佳 checkpoint |
| `manual` | 用户指定权重 |

---

## 2. kai0 的 MA 实现

### 代码位置
- `third_party/kai0/model_arithmetic/`
  - `arithmetic.py` - JAX/Flax 实现 (Orbax checkpoints)
  - `arithmetic_torch.py` - PyTorch 实现 (safetensors)
  - `common.py` - 共享工具函数

### 关键发现：kai0 混合所有参数

```python
# common.py - mix_params() 函数
for key in tqdm(params_list[0].keys(), desc="Mixing parameters"):
    stacked = np.stack([np.asarray(p[key], dtype=np.float64) for p in params_list], axis=0)
    mixed[key] = np.average(stacked, axis=0, weights=weights).astype(np.float32)
    # ↑ 混合所有参数，没有任何过滤！
```

**kai0 不区分 VLM backbone 和 action expert，全部混合。**

### kai0 的训练模式

| 模式 | 显存需求 | 说明 |
|------|---------|------|
| Full Fine-tuning (默认) | ~70 GB (A100/H100) | 所有参数可训练 |
| LoRA Fine-tuning | ~22.5 GB (RTX 4090) | 只有 LoRA adapter 可训练 |

**kai0 论文使用 Full Fine-tuning**，所以混合所有参数是合理的。

---

## 3. pi05 架构分析

### 架构组成

```
┌─────────────────────────────────────────────────────────┐
│  PaliGemma VLM Backbone (FROZEN during fine-tuning)     │
│  ├── vision_tower (SigLIP)                              │
│  ├── language_model (Gemma)                             │
│  └── projection layers                                  │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  Action Expert (TRAINABLE)                              │
│  ├── gemma_expert (flow-matching head)                  │
│  ├── action_in_proj                                     │
│  ├── action_out_proj                                    │
│  ├── time_mlp_in                                        │
│  └── time_mlp_out                                       │
└─────────────────────────────────────────────────────────┘
```

### 参数结构

```python
# Frozen (fine-tune 时相同)
model.paligemma_with_expert.paligemma.vision_tower.*
model.paligemma_with_expert.paligemma.language_model.*

# Trainable (fine-tune 时不同)
model.paligemma_with_expert.gemma_expert.*
model.action_in_proj.*
model.action_out_proj.*
model.time_mlp_in.*
model.time_mlp_out.*
```

### Frozen VLM 的问题

当 VLM backbone 被 freeze 时：
- 不同 checkpoint 的 VLM 权重**完全相同**
- MA 混合相同权重 = 无意义操作
- 浪费内存和计算资源

---

## 4. SmolVLA 做 MA 的优势

### 对比 pi05

| 对比项 | SmolVLA | pi05 |
|--------|---------|------|
| **模型大小** | ~875M (expert ~375M) | ~2B+ |
| **Full Fine-tune 显存** | ~24GB (RTX 3090) | ~70GB (A100) |
| **训练时间** | ~4小时 | ~8-12小时 |
| **Checkpoint 大小** | ~2-3GB | ~7GB |
| **MA 内存需求** | ~12GB | ~28GB |
| **实验迭代速度** | 快 | 慢 |

### SmolVLA 架构

```
SmolVLA (~875M parameters)
├── SmolVLM2-500M-Video-Instruct (VLM backbone)
├── Action Expert (~375M, width_multiplier=0.75)
├── state_proj
├── action_in_proj / action_out_proj
└── action_time_mlp
```

### 关键优势

1. **Full Fine-tune 可行** - 单卡 RTX 3090/4090 即可
2. **训练成本低** - 4小时 vs pi05 的 8-12小时
3. **MA 效率高** - 混合所有参数有意义（没有 frozen 部分）
4. **LeRobot 原生支持** - safetensors 格式兼容

---

## 5. 实施方案

### 5.1 训练多个模型

```bash
# 按不同 garment type 训练
lerobot-train --config_path=configs/train_smolvla_top_long.yaml
lerobot-train --config_path=configs/train_smolvla_pant_long.yaml
lerobot-train --config_path=configs/train_smolvla_pant_short.yaml
```

### 5.2 准备验证数据

```bash
python scripts/extract_validation_data.py \
    --dataset_root Datasets/example \
    --output validation_data.pkl \
    --num_batches 50
```

### 5.3 执行 MA

```bash
python scripts/model_arithmetic_smolvla.py \
    --checkpoints \
        outputs/train/smolvla_top_long/checkpoints/last \
        outputs/train/smolvla_pant_long/checkpoints/last \
        outputs/train/smolvla_pant_short/checkpoints/last \
    --validation_data validation_data.pkl \
    --method gradient_descent \
    --output outputs/ma_ensemble/smolvla_mixed
```

### 5.4 评估混合模型

```bash
python -m scripts.eval \
    --policy_type lerobot \
    --policy_path outputs/ma_ensemble/smolvla_mixed \
    --garment_type "top_long" \
    --num_episodes 10 \
    --enable_cameras \
    --device cpu
```

---

## 6. 需要实现的适配代码

### 6.1 Checkpoint 路径适配 (~20行)

```python
def resolve_lerobot_ckpt_path(path: str) -> str:
    """Convert LeRobot checkpoint structure to MA format."""
    p = Path(path).resolve()
    # LeRobot: .../pretrained_model/model.safetensors
    if (p / "pretrained_model" / "model.safetensors").exists():
        return str(p / "pretrained_model")
    raise FileNotFoundError(f"No model.safetensors found in {p}")
```

### 6.2 验证数据提取 (~100行)

```python
def extract_validation_data(dataset_root: Path, num_batches: int = 50) -> dict:
    """Extract validation batches from LeRobot dataset."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(root=dataset_root)
    val_data = {"observations": [], "actions": []}

    for i in range(min(num_batches, len(dataset))):
        sample = dataset[i]
        val_data["observations"].append(sample["observation"])
        val_data["actions"].append(sample["action"])

    return val_data
```

### 6.3 MA 主脚本 (~200行)

基于 kai0 的 `arithmetic_torch.py`，适配 LeRobot 的 checkpoint 格式。

---

## 7. 预期收益

| 指标 | 单模型 | MA 混合模型 |
|------|--------|-------------|
| 泛化能力 | 受限于训练数据 | 更好的跨 garment 泛化 |
| 成功率 | 基准 | 预期提升 10-30% |
| 鲁棒性 | 一般 | 更稳定 |

---

## 8. 研究背景

### 相关论文

1. **χ₀: Resource-Aware Robust Manipulation** (kai0 paper)
   - arXiv:2602.09021
   - 验证了 MA 在机器人操作中的有效性
   - 成功率提升 ~250%

2. **Model Soups** (Wortsman et al., 2022)
   - arXiv:2203.05482
   - 证明权重平均可以提升性能

### 关键洞察

> "Fine-tuned VLAs exhibit extreme parameter redundancy, akin to LLMs. Merging weights from subset-trained models surpasses training on the combined dataset."
> — kai0 paper

---

## 9. 结论

**SmolVLA + MA 是 LeHome 项目中务实且可行的选择：**

- ✅ Full fine-tune 在消费级 GPU 可行
- ✅ kai0 的 PyTorch MA 代码兼容
- ✅ 训练成本低，实验迭代快
- ✅ 没有 frozen 参数的尴尬
- ⚠️ 需要写适配脚本（~200行代码）

---

## 10. 下一步行动

1. [ ] 实现 `scripts/extract_validation_data.py`
2. [ ] 实现 `scripts/model_arithmetic_smolvla.py`
3. [ ] 训练多个 SmolVLA 模型
4. [ ] 执行 MA 并评估效果
5. [ ] 对比 MA 模型 vs 单模型性能

---

*文档生成时间: 2026-03-05*
*基于 kai0 和 LeRobot 代码库分析*
