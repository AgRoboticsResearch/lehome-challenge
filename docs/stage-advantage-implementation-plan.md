# Stage Advantage (SA) 在 LeHome + SmolVLA 中的实施计划

## 概述

本文档描述如何将 Kai0 的 Stage Advantage (SA) 技术应用到 LeHome 项目和 SmolVLA 模型中。

### Stage Advantage 核心思想

**Stage Advantage（阶段优势）** 是 Kai0 框架中的核心技术，用于解决长时程机器人操作任务中的训练不稳定性问题：

- 将长任务分解成多个**语义阶段**
- 在每个阶段内计算**优势值**（advantage），反映当前状态对任务完成的贡献度
- 用优势值作为训练信号，让策略模型学会区分"好状态"和"坏状态"

### Kai0 SA 实现的四阶段流水线

1. **Stage 0: GT 数据标注** - 计算 advantage 并离散化为 task_index
2. **Stage 1: 训练优势估计器** - 基于预测优势值
3. **Stage 2: 优势估计** - 用估计器标注新数据
4. **Stage 3: AWBC 训练** - 优势加权行为克隆

---

## 架构对比分析

| 方面 | Kai0 (π₀.₅) | LeHome (SmolVLA) |
|------|-------------|------------------|
| 语言通道 | PaliGemma tokenizer | SmolVLM2 tokenizer |
| 提示词长度 | 可配置 | `tokenizer_max_length: 48` |
| 架构 | VLA + head | VLM + Action Expert |
| 数据格式 | LeRobot | LeRobot ✅ (兼容!) |

**好消息**：两者都使用 **LeRobot 数据格式**，数据层可以直接复用！

---

## 实施步骤

### 阶段 1：数据增强层（添加进度/优势标注）

#### 1.1 添加 progress 计算

在 `scripts/dataset.py` 中添加新命令：

```python
# scripts/dataset.py 新增
def setup_progress_parser(subparsers):
    parser = subparsers.add_parser("progress", help="Add progress labels to dataset")
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--stage_nums", type=int, default=2)  # 分成几个阶段

# scripts/utils/dataset_processing.py 新增
def add_progress_labels(dataset_root, chunk_size=50, stage_nums=2):
    """
    为每一帧添加进度标签：
    - progress: 从起始帧到当前帧的任务进度 (0-1)
    - stage_progress_gt: 当前阶段内的进度 (0-1)
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset.load(dataset_root)

    for episode_index in range(len(ds)):
        # 计算基于 garment 状态的进度
        # 可以用粒子位置、衣物的平整度等指标
        progress = compute_garment_progress(ds, episode_index)

        # 将进度写入数据集
        # LeRobot 支持 delta_frame 写入
```

#### 进度计算方法

```python
# source/lehome/lehome/utils/progress.py
def compute_garment_progress(dataset, episode_index):
    """
    计算衣物操作的进度：
    1. 平整度：粒子 z 坐标方差（越小越平整）
    2. 覆盖度：粒子在目标区域的覆盖
    3. 形状匹配：与目标形状的相似度
    """
    frames = load_episode_frames(dataset, episode_index)

    progress_scores = []
    for frame in frames:
        # 从图像或状态计算进度
        garment_particles = extract_garment_particles(frame)
        flatness = 1.0 - np.std(garment_particles[:, 2])  # z轴方差
        progress_scores.append(flatness)

    # 归一化到 0-1
    progress = np.array(progress_scores)
    progress = (progress - progress.min()) / (progress.max() - progress.min() + 1e-6)

    return progress
```

#### 1.2 添加优势标注（复用 kai0 代码）

```bash
# 直接使用 kai0 的标注脚本
cp third_party/kai0/stage_advantage/annotation/gt_label.py scripts/stage_advantage/
cp third_party/kai0/stage_advantage/annotation/gt_labeling.sh scripts/stage_advantage/

# 使用方法
python -m scripts.stage_advantage.gt_label \
    Datasets/example/top_long_merged \
    --threshold 30 \
    --chunk-size 50 \
    --discretion-type binary \
    --stage-nums 2 \
    --advantage-source progress
```

**输出**：
- `meta/tasks.jsonl` - task_index 到提示词的映射
- 每帧添加 `task_index` 列

---

### 阶段 2：修改 SmolVLA 模型（支持优势条件）

#### 2.1 修改配置文件

```yaml
# configs/train_smolvla_sa_top_long.yaml
dataset:
  repo_id: repo_smolvla_sa_top_long
  root: Datasets/example/top_long_sa  # 带优势标注的数据

policy:
  type: smolvla
  device: cuda

  # 新增：优势相关配置
  use_advantage_prompt: true  # 启用优势提示
  advantage_mode: binary  # binary 或 continuous

  input_features:
    observation.state:
      type: STATE
      shape: [12]
    observation.images.top_rgb:
      type: VISUAL
      shape: [3, 480, 640]
    # ...
    observation.task_index:  # 新增：任务索引（优势等级）
      type: STATE
      shape: [1]
```

#### 2.2 修改 Processor（添加优势映射）

```python
# source/lehome/lehome/policies/smolvla_processor_sa.py
from lerobot.policies.smolvla.processor_smolvla import SmolVLAProcessor

class SmolVLAProcessorSA(SmolVLAProcessor):
    def __init__(self, *args, advantage_mode="binary", **kwargs):
        super().__init__(*args, **kwargs)
        self.advantage_mode = advantage_mode

        # 优势等级 → 提示词映射
        self.advantage_prompts = {
            0: "Pick up and manipulate the garment. Status: needs improvement.",
            1: "Pick up and manipulate the garment. Status: good progress.",
        }

    def prepare_language_from_task_index(self, task_index):
        """根据 task_index 生成优势提示词"""
        if isinstance(task_index, (list, np.ndarray)):
            task_index = task_index[0]
        return self.advantage_prompts.get(int(task_index), self.advantage_prompts[0])
```

#### 2.3 修改数据加载（注入优势提示）

```python
# third_party/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py
# 在 SmolVLAPolicy 类中修改

def __init__(self, cfg, dataset_stats):
    # ... 原有代码 ...

    # 新增：加载优势映射
    if cfg.use_advantage_prompt:
        self.use_advantage_prompt = True
        self.task_index_to_prompt = self._load_task_prompts(dataset_stats)

def _load_task_prompts(self, dataset_stats):
    """从 meta/tasks.jsonl 加载提示词映射"""
    import json
    tasks_file = Path(dataset_stats.repo_id) / "meta" / "tasks.jsonl"
    if tasks_file.exists():
        with open(tasks_file) as f:
            return {json.loads(line)["task_index"]: json.loads(line)["task"]
                    for line in f}
    return None

def select_action(self, batch):
    # ... 原有代码 ...

    # 新增：根据 task_index 选择提示词
    if self.use_advantage_prompt and "task_index" in batch:
        task_indices = batch["task_index"].cpu().numpy()
        prompts = [self.task_index_to_prompt.get(int(idx), "") for idx in task_indices]
        # 用优势提示词替换默认提示
        batch["observation.language_instruction"] = prompts
```

---

### 阶段 3：训练 AWBC 策略

#### 3.1 修改损失函数（可选）

如果希望加权高优势样本：

```python
# 在 training loop 中添加优势权重
def compute_loss_with_advantage(pred_actions, gt_actions, advantage_weights):
    """根据优势加权损失"""
    base_loss = F.mse_loss(pred_actions, gt_actions, reduction='none')

    # 高优势样本获得更高权重
    weights = torch.clamp(advantage_weights, 0.1, 2.0)
    weighted_loss = (base_loss * weights).mean()

    return weighted_loss
```

#### 3.2 启动训练

```bash
# 使用带优势标注的数据集训练
lerobot-train \
    --config configs/train_smolvla_sa_top_long.yaml \
    --dataset.root Datasets/example/top_long_sa \
    --policy.use_advantage_prompt true \
    --steps 100000
```

---

### 阶段 4：推理时使用高优势提示

```python
# 推理时使用正优势提示词
policy = SmolVLAPolicy.from_pretrained("outputs/train/smolvla_sa_top_long")

# 设置为高优势模式
task_prompt = "Pick up and manipulate the garment. Status: good progress."
observation = {
    "observation.state": state,
    "observation.images.top_rgb": image,
    "observation.language_instruction": task_prompt,  # 关键！
}

action = policy.select_action(observation)
```

---

## 实施优先级

```
高优先级（必须做）:
├── 1. 数据增强：添加 progress/stage_progress_gt 列
├── 2. 优势标注：使用 kai0 的 gt_label.py
└── 3. 提示词映射：加载 meta/tasks.jsonl 到策略

中优先级（推荐做）:
├── 4. 数据增强：时间/空间数据增强（来自 kai0 train_deploy_alignment）
└── 5. RTC 模式：SmolVLA 已支持，可直接启用

低优先级（可选）:
├── 6. 优势估计器：训练独立的进度预测网络
└── 7. Model Arithmetic：多模型融合
```

---

## 需要创建的新文件

```
lehome-challenge/
├── scripts/
│   ├── stage_advantage/
│   │   ├── gt_label.py          # 从 kai0 复制
│   │   └── gt_labeling.sh       # 从 kai0 复用
│   └── utils/
│       └── progress.py          # 新建：进度计算
├── source/lehome/lehome/
│   ├── policies/
│   │   └── smolvla_sa.py        # 新建：SA 版 SmolVLA
│   └── utils/
│       └── garment_progress.py  # 新建：衣物进度指标
└── configs/
    └── train_smolvla_sa_*.yaml  # 新建：SA 训练配置
```

---

## 参考资源

- Kai0 GitHub: https://github.com/OpenDriveLab/kai0
- Kai0 Paper: https://arxiv.org/abs/2602.09021
- Kai0 SA README: `third_party/kai0/stage_advantage/README.md`
- LeRobot SmolVLA: `third_party/lerobot/src/lerobot/policies/smolvla/`
