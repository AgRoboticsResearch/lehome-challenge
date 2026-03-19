# Evo-RL专家优化讨论记录

> **创建日期**: 2026-03-19
> **主题**: 将Evo-RL思想应用到MoE-SmolVLA专家优化
> **状态**: 设计阶段

---

## 目录

1. [背景与动机](#背景与动机)
2. [Evo-RL核心思想解析](#evo-rl核心思想解析)
3. [关键问题澄清](#关键问题澄清)
4. [HIL数据收集方案](#hil数据收集方案)
5. [失败数据收集方案](#失败数据收集方案)
6. [Expert优化方案](#expert优化方案)
7. [实施建议](#实施建议)

---

## 背景与动机

### 当前MoE系统状态

- ✅ Router训练完成，100%准确率
- ✅ 4个Expert独立训练完成
- ✅ Sticky Routing机制实现
- ⚠️ Expert性能未达到最优

### 优化目标

借鉴Evo-RL的思想，对每个Expert进行深度优化：
1. 提高Expert在特定服装类型上的成功率
2. 学习区分高效/低效执行模式
3. 通过迭代持续改进

---

## Evo-RL核心思想解析

### 核心组件对比

| 核心思想 | 作用 | 代码位置 |
|---------|------|----------|
| **HIL (Human-in-the-Loop)** | 收集策略失败数据+人工纠正 | `recording_hil.py`, `hil_processor.py` |
| **Value Function (pistar06)** | 学习状态价值，区分成功/失败 | `values/pistar06/` |
| **ACP (Advantage-Conditioned Policy)** | 条件策略训练（高效vs低效） | `rl/acp_hook.py`, `rl/acp_tags.py` |

### 完整闭环流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Evo-RL 完整闭环                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Phase 1: HIL数据收集                                                │
│  ─────────────────                                                  │
│  策略执行 → 人工观察 → 按'i'干预 → 纠正动作 → 标记成功/失败          │
│                                                                     │
│  Phase 2: 价值函数训练                                               │
│  ─────────────────────                                              │
│  pistar06学习区分成功/失败轨迹                                       │
│  • 成功: value = -remaining_steps / max_length                      │
│  • 失败: value = -(remaining_steps + c_fail) / max_length           │
│                                                                     │
│  Phase 3: ACP标签生成                                                │
│  ─────────────────────                                              │
│  advantage = 实际累积回报 - 价值预测                                  │
│  按top 30%分位数二值化 → positive/negative                          │
│  强制人工干预帧为positive (可选)                                     │
│                                                                     │
│  Phase 4: 策略训练                                                   │
│  ─────────────────                                                  │
│  使用ACP标签训练策略                                                 │
│  任务文本: "Fold the shirt\nAdvantage: positive/negative"          │
│                                                                     │
│  Phase 5: 部署与迭代                                                 │
│  ─────────────────────                                              │
│  新策略 → HIL收集 → 回到Phase 1                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Intervention State Machine (S0/S1/S2)

```
┌─────────────────────────────────────────────────────────────────┐
│              Intervention State Machine                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   [键盘按 'i']                                                 │
│   ┌─────────┐     ┌─────────┐                                  │
│   │   S0    │────▶│   S1    │  ◄── 人工接管控制               │
│   │ Policy  │     │ Active  │                                  │
│   │ Control │     │Intervention│                                │
│   │ (value=0)│     │ (value=1)│                                 │
│   └─────────┘     └─────────┘                                  │
│       ▲               │                                         │
│       │               │ [键盘释放 'i']                          │
│       │               ▼                                         │
│       │          ┌─────────┐                                    │
│       │          │   S2    │  ◄── 平滑过渡回策略                │
│       │          │ Release │                                    │
│       │          │ (value=2)│                                   │
│       └──────────│         │                                    │
│                  └─────────┘                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### HIL数据字段

```python
# 每一帧包含以下HIL特定字段：
{
    # 标准LeRobot字段
    "observation.state": [...],        # 机器人状态
    "observation.images.top_rgb": [...], # 相机图像
    "action": [...],                    # 实际执行的动作

    # HIL额外字段
    "complementary_info.policy_action": [...],     # 策略输出的动作
    "complementary_info.is_intervention": 0.0,      # 是否人工干预 (0/1)
    "complementary_info.state": 0.0,               # 状态机状态 (0=S0, 1=S1, 2=S2)
    "complementary_info.collector_policy_id": "..." # 动作来源ID
}

# Episode元数据
{
    "episode_index": 0,
    "episode_success": "success",  # 或 "failure"
    "length": 150,
}
```

### 快捷键控制

| 快捷键 | 功能 | 说明 |
|--------|------|------|
| **`i`** | Toggle Intervention | 在策略执行和人工控制之间切换 |
| **`s`** | Mark Success | 标记当前episode为成功并结束 |
| **`f`** | Mark Failure | 标记当前episode为失败并结束 |
| **→** | End Loop | 结束当前循环（episode） |
| **←** | Re-record | 重新录制当前episode |
| **`Esc`** | Stop | 停止录制会话 |

---

## 关键问题澄清

### 问题1：人工干预后的价值函数学习？

**问题**：如果人工干预后episode最终成功，价值函数如何区分人工干预前的"失败状态"？

**答案**：关键在于价值的"斜率"不同

```
成功轨迹的价值曲线：
Frame 0: value = -1.0  (离成功还很远)
Frame 50: value = -0.5 (开始接近)
Frame 100: value = 0.0 (成功！)

失败轨迹的价值曲线（即使后期有人工干预）：
Frame 0: value = -1.0  (离成功远)
Frame 50: value = -0.8 (策略执行导致偏离)
Frame 51: value = -0.3 (人工接管，开始恢复)
Frame 100: value = 0.0 (人工纠正后成功)
```

**核心洞察**：
- Frame 0-50: 价值函数学到"这些状态虽然最终被纠正，但策略执行导致了低效"
- Frame 51-100: 价值函数学到"人工接管后的恢复路径"

### 问题2：force_intervention_positive的作用

```python
# lerobot_value_infer.py:314-316
if force_intervention_positive:
    intervention_mask = interventions.astype(np.float32) > 0.5
    indicators[intervention_mask] = 1  # 强制人工干预帧为positive
```

| 模式 | force_intervention_positive | 学习目标 |
|------|---------------------------|----------|
| **学习策略改进** | False | 人工干预前的帧被标记为negative，策略学到"避免这些状态" |
| **学习人工技巧** | True | 人工干预帧被标记为positive，策略学到"模仿人工纠正" |

### 问题3：Data Curation vs Evo-RL ACP

| 方面 | Data Curation | Evo-RL ACP |
|------|---------------|-----------|
| **思想来源** | 通用机器学习实践 | Evo-RL创新 |
| **评估方法** | 启发式指标（长度、平滑度） | 学习的价值函数 |
| **标签类型** | 连续分数 | 二值标签（positive/negative） |
| **理论基础** | 无 | 基于强化学习的advantage |

**结论**：Data Curation不是Evo-RL的核心思想，只是一个通用的数据质量筛选方法。

---

## HIL数据收集方案

### 在LeHome仿真环境实现HIL

#### 方案概述

由于LeHome基于Isaac Sim仿真，我们可以通过键盘控制实现类似Evo-RL的HIL功能。

#### HIL评估脚本框架

```python
# scripts/hil_eval.py

"""
Human-in-the-Loop Evaluation for LeHome Simulation

快捷键：
- 'i': 切换干预模式 (policy <-> manual)
- 方向键: 控制机械臂移动
- '[' / ']': 控制gripper (开/关)
- 's': 标记成功并结束episode
- 'f': 标记失败并结束episode
- 'q': 退出
"""

class HILPolicy:
    """HIL Policy包装器：支持策略执行和人工干预切换"""

    def __init__(self, base_policy, device="cpu"):
        self.base_policy = base_policy
        self.device = device
        self.intervention_mode = False
        self.manual_action_offset = np.zeros(12, dtype=np.float32)
        self.action_scale = 0.02  # 人工控制步长

    def set_intervention(self, enabled: bool):
        """设置干预模式"""
        self.intervention_mode = enabled
        if not enabled:
            self.manual_action_offset = np.zeros(12, dtype=np.float32)

    def apply_manual_control(self, control_dict: dict):
        """应用人工控制输入"""
        if not self.intervention_mode:
            return

        # 左臂控制 (0-5)
        if control_dict.get('up'):
            self.manual_action_offset[1] += self.action_scale  # shoulder_lift
        if control_dict.get('down'):
            self.manual_action_offset[1] -= self.action_scale
        if control_dict.get('left'):
            self.manual_action_offset[0] += self.action_scale  # shoulder_pan
        if control_dict.get('right'):
            self.manual_action_offset[0] -= self.action_scale

        # 右臂控制 (6-11)
        if control_dict.get('page_up'):
            self.manual_action_offset[7] += self.action_scale
        if control_dict.get('page_down'):
            self.manual_action_offset[7] -= self.action_scale

        # Gripper控制
        if control_dict.get('gripper_open'):
            self.manual_action_offset[4] = self.action_scale
            self.manual_action_offset[10] = self.action_scale
        if control_dict.get('gripper_close'):
            self.manual_action_offset[4] = -self.action_scale
            self.manual_action_offset[10] = -self.action_scale

    def select_action(self, observation: dict) -> np.ndarray:
        """选择动作（策略或人工）"""
        if self.intervention_mode:
            # 人工干预模式
            current_state = observation.get("observation.state", np.zeros(12))
            action = current_state + self.manual_action_offset
            return action
        else:
            # 策略执行模式
            return self.base_policy.select_action(observation)
```

#### 使用方法

```bash
# 运行HIL评估（为pant_short专家收集数据）
python -m scripts.hil_eval \
  --task LeHome-BiSO101-Direct-Garment-v2 \
  --policy_type lerobot \
  --policy_path outputs/moe_train/smolvla_moe_expert_pant_short_no_st_proj/checkpoints/last/pretrained_model \
  --dataset_root Datasets/example/pant_short_merged \
  --garment_type pant_short \
  --device cpu \
  --num_episodes 20 \
  --max_steps 400 \
  --eval_dataset_path Datasets/hil \
  --hil_mode
```

#### HIL数据格式

```python
Episode {
    "episode_index": 5,
    "episode_success": "success",
    "length": 100,

    # Frames 0-49: 策略执行
    Frame 0-49: {
        "action": <策略动作>,
        "complementary_info.policy_action": <策略动作>,
        "complementary_info.is_intervention": 0.0,
        "complementary_info.state": 0.0,  # S0
        "complementary_info.collector_policy_id": "outputs/train/..."
    },

    # Frame 50: 人工开始干预
    Frame 50: {
        "action": <人工动作>,
        "complementary_info.policy_action": <策略动作>,  # 不同！
        "complementary_info.is_intervention": 1.0,
        "complementary_info.state": 1.0,  # S1
        "complementary_info.collector_policy_id": "human"
    }
}
```

---

## 失败数据收集方案

### 问题：当前评估只保存成功数据

**代码位置**: `scripts/utils/evaluation.py:206-217`

```python
# 旧代码：
if args.save_datasets:
    if success_flag:
        eval_dataset.save_episode()
        # ... save episode
    else:
        eval_dataset.clear_episode_buffer()  # ❌ 丢弃失败数据
```

### 解决方案：修改评估脚本

```python
# 新代码：
if args.save_datasets:
    # 添加episode_success标记
    success_value = np.array([1.0 if success_flag else 0.0], dtype=np.float32)

    # 为所有帧添加episode_success
    for frame_data in eval_dataset._episode_buffer:
        frame_data["episode_success"] = success_value

    # 保存成功和失败
    eval_dataset.save_episode()

    if success_flag:
        append_episode_initial_pose(...)
        episode_index += 1

    # 日志
    status = "SUCCESS" if success_flag else "FAILURE"
    logger.info(f"Episode saved with status: {status}")
```

### 使用方法

```bash
# 评估并保存成功+失败数据
python -m scripts.eval \
  --policy_type moe_smolvla \
  --policy_path outputs/moe_train/smolvla_moe_expert_pant_short_no_st_proj/checkpoints/last/pretrained_model \
  --garment_type pant_short \
  --dataset_root Datasets/example/pant_short_merged \
  --device cpu \
  --num_episodes 50 \
  --save_datasets \
  --eval_dataset_path Datasets/eval_with_failures/pant_short
```

### LeHome成功检测机制

**代码位置**: `source/lehome/lehome/utils/success_checker_chanllege.py`

```python
@step_interval(interval=50)
def success_checker_garment_fold(particle_object, garment_type: str):
    check_point_indices = particle_object.check_points
    success_distance = particle_object.success_distance
    p = get_object_particle_position(particle_object, check_point_indices)

    if garment_type == "top-long-sleeve" or garment_type == "top-short-sleeve":
        success, details = check_top_sleeve(p, success_distance)
    elif garment_type == "short-pant":
        success, details = check_pant_short(p, success_distance)
    elif garment_type == "long-pant":
        success, details = check_pant_long(p, success_distance)

    return result
```

---

## Expert优化方案

### 完整优化流程

```
┌─────────────────────────────────────────────────────────────────┐
│            LeHome HIL + Expert Optimization 流程                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 1: 收集HIL数据                                            │
│  ─────────────────────                                          │
│  方法A: HIL评估（人工干预）                                       │
│    python -m scripts.hil_eval --hil_mode                        │
│    ├─ 策略执行 → 观察                                         │
│    ├─ 按'i' → 人工干预                                        │
│    ├─ 方向键控制 → 纠正动作                                    │
│    └─ 按's'/'f' → 标记成功/失败                               │
│                                                                 │
│  方法B: 普通评估（自动收集失败）                                  │
│    python -m scripts.eval --save_datasets                      │
│    ├─ 策略自动执行                                            │
│    ├─ 仿真检测成功/失败                                        │
│    └─ 保存所有episode                                         │
│                                                                 │
│  输出：Datasets/hil/pant_short (包含HIL字段)                      │
│                                                                 │
│  Phase 2: 训练价值函数                                           │
│  ─────────────────────                                          │
│  lerobot-value-train                                           │
│    --dataset.root=Datasets/hil/pant_short                      │
│    --targets.success_field=episode_success                     │
│    --targets.default_success=failure                           │
│    ├─ 学习预测价值                                            │
│    ├─ 成功轨迹: value → 0                                    │
│    └─ 失败轨迹: value → -c_fail                                │
│                                                                 │
│  Phase 3: 生成ACP标签                                            │
│  ─────────────────────                                          │
│  lerobot-value-infer                                            │
│    --acp.intervention_field=complementary_info.is_intervention │
│    --acp.force_intervention_positive=false                     │
│    ├─ 计算advantage = 实际回报 - 预测价值                       │
│    ├─ 人工干预帧：标记为negative (学习避免)                     │
│    └─ 按top 30%分位数二值化                                     │
│                                                                 │
│  Phase 4: 训练改进的Expert                                       │
│  ───────────────────────────                                    │
│  lerobot-train --acp.enable=true                               │
│    --acp.indicator_field=complementary_info.acp_indicator       │
│    ├─ 任务文本注入ACP标签                                     │
│    ├─ "Fold the shirt\nAdvantage: positive/negative"           │
│    └─ Expert学习区分高效/低效状态                               │
│                                                                 │
│  Phase 5: 迭代优化                                              │
│  ───────────────────                                           │
│  重复Phase 1-4，持续改进Expert                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 详细命令

#### 步骤1：训练价值函数

```bash
lerobot-value-train \
  --dataset.repo_id=lehome_pant_short_hil \
  --dataset.root=Datasets/hil/pant_short \
  --value.type=pistar06 \
  --value.dtype=bfloat16 \
  --targets.success_field=episode_success \
  --targets.default_success=failure \
  --targets.c_fail_coef=1.0 \
  --batch_size=64 \
  --steps=10000 \
  --output_dir=outputs/value_train/hil_pant_short \
  --wandb.enable=true
```

#### 步骤2：生成ACP标签

```bash
lerobot-value-infer \
  --dataset.repo_id=lehome_pant_short_hil \
  --dataset.root=Datasets/hil/pant_short \
  --inference.checkpoint_path=outputs/value_train/hil_pant_short/checkpoints/best/pretrained_model \
  --acp.enable=true \
  --acp.n_step=50 \
  --acp.positive_ratio=0.3 \
  --acp.intervention_field=complementary_info.is_intervention \
  --acp.force_intervention_positive=false \
  --output_dir=outputs/acp_inference/hil_pant_short
```

#### 步骤3：用ACP标签训练Expert

```bash
lerobot-train \
  --dataset.repo_id=lehome_pant_short_hil_acp \
  --dataset.root=Datasets/hil/pant_short \
  --policy.type=smolvla \
  --policy.pretrained_path=outputs/moe_train/smolvla_moe_expert_pant_short_no_st_proj/checkpoints/last/pretrained_model \
  --policy.device=cpu \
  --batch_size=16 \
  --steps=15000 \
  --acp.enable=true \
  --acp.indicator_field=complementary_info.acp_indicator \
  --acp.indicator_dropout_prob=0.3 \
  --output_dir=outputs/moe_train_v2/hil_improved_pant_short \
  --wandb.enable=true
```

---

## 实施建议

### 渐进式实施路径

#### 阶段1：数据收集（最简单，立即开始）

1. **修改评估脚本**保存失败数据
   - 文件：`scripts/utils/evaluation.py`
   - 修改：第206-217行
   - 时间：5分钟

2. **运行评估收集数据**
   ```bash
   python -m scripts.eval \
     --policy_type moe_smolvla \
     --garment_type pant_short \
     --num_episodes 50 \
     --save_datasets \
     --eval_dataset_path Datasets/eval_with_failures/pant_short
   ```

3. **验证数据质量**
   ```bash
   lerobot-dataset-report --dataset Datasets/eval_with_failures/pant_short
   ```

**预期收益**：立即获得包含成功/失败的数据集

#### 阶段2：价值函数引导（中等复杂度）

1. **集成Evo-RL组件**
   - 复制`values/pistar06/`
   - 复制`rl/acp_*.py`
   - 修改配置文件

2. **训练价值函数**
   ```bash
   lerobot-value-train --config_path configs/value_train_pant_short.yaml
   ```

3. **生成ACP标签**
   ```bash
   lerobot-value-infer --config_path configs/value_infer_pant_short.yaml
   ```

**预期收益**：Expert学习区分高效/低效状态

#### 阶段3：完整闭环（完整实现）

1. **实现HIL评估脚本**
   - 创建`scripts/hil_eval.py`
   - 实现键盘控制
   - 支持人工干预

2. **迭代优化**
   - HIL收集 → 价值训练 → ACP → 重训练
   - 重复2-3轮

**预期收益**：持续改进Expert性能

### 数据需求估算

| 阶段 | 数据量 | 来源 | 时间 |
|------|--------|------|------|
| 阶段1 | 50 episodes/类型 | 自动评估 | 1-2小时 |
| 阶段2 | +30 HIL episodes/类型 | 人工干预 | 2-3小时 |
| 阶段3 | 迭代2-3轮 | 混合 | 1-2天 |

### 预期性能改进

| Expert | 当前成功率 | 阶段1后 | 阶段2后 | 阶段3后 |
|--------|-----------|---------|---------|---------|
| pant_short | ~88% | +5% | +10% | +15% |
| pant_long | ~48% | +8% | +15% | +20% |
| top_short | ~42% | +10% | +15% | +20% |
| top_long | ~73% | +5% | +10% | +15% |

---

## 参考资料

### Evo-RL代码位置

- HIL实现: `third_party/Evo-RL/src/lerobot/scripts/recording_hil.py`
- HIL处理器: `third_party/Evo-RL/src/lerobot/processor/hil_processor.py`
- 价值函数: `third_party/Evo-RL/src/lerobot/values/pistar06/`
- ACP标签: `third_party/Evo-RL/src/lerobot/rl/acp_tags.py`
- ACP Hook: `third_party/Evo-RL/src/lerobot/rl/acp_hook.py`
- 价值推理: `third_party/Evo-RL/src/lerobot/scripts/lerobot_value_infer.py`

### LeHome代码位置

- 评估脚本: `scripts/eval.py`
- 评估逻辑: `scripts/utils/evaluation.py`
- 成功检测: `source/lehome/lehome/utils/success_checker_chanllege.py`
- 任务环境: `source/lehome/lehome/tasks/bedroom/garment_bi_v2.py`
- MoE Policy: `scripts/eval_policy/moe_smolvla_policy.py`

### 相关文档

- Evo-RL README: `third_party/Evo-RL/README.md`
- MoE设计文档: `docs/moe_design_v2.md`
- Evo-RL集成计划: `docs/evo-rl-integration-plan.md`
- MoE专家优化设计: `docs/moe-expert-optimization-with-evo-rl.md`

---

## 下一步行动

### 立即可做

1. **修改评估脚本**保存失败数据
   - 最简单，立即见效
   - 时间：5分钟

2. **收集初步数据**
   - 运行评估收集成功+失败数据
   - 分析失败模式

3. **实现HIL评估框架**
   - 创建基础脚本
   - 测试键盘控制

### 需要确认

1. **优先级**：从哪个阶段开始？
2. **资源**：计算资源（GPU）是否充足？
3. **时间**：项目时间表如何？

---

*文档创建于2026-03-19*
*基于Evo-RL和LeHome项目分析*
