# LeHome Challenge 项目教程

> 本文档系统地讲解 LeHome Challenge 项目的代码框架、各模块作用和运行逻辑。
> 目标：从"能运行"到"能理解、能实现、能设计"。

---

## 目录

1. [代码能力层级](#一代码能力层级)
2. [整体架构](#二整体架构)
3. [Gym Environment 详解](#三gym-environment-详解)
4. [评估流程](#四评估流程)
5. [策略系统](#五策略系统)
6. [动手练习](#六动手练习)

---

## 一、代码能力层级

| 层级 | 名称 | 描述 | 检验标准 |
|------|------|------|----------|
| Level 1 | 能运行 | 按文档执行命令，遇到报错就卡住 | - |
| Level 2 | 能调试 | 能读 traceback，定位问题，独立解决 70% 错误 | 看到 `KeyError` 能判断是 config 还是 dataset 问题 |
| Level 3 | 能理解 | 不看文档也能解释执行流程 | 能说出 Isaac Sim / Isaac Lab / LeRobot 的职责边界 |
| Level 4 | 能实现 | 给需求能独立完成，代码符合项目风格 | 能独立添加新的 garment type |
| Level 5 | 能设计 | 能判断功能放在哪里，识别设计问题 | 能评估"支持三臂机器人"需要改哪些模块 |

**脱离 vibe coding 的标志：遇到问题，第一反应是"我去读代码/日志"而不是"我去问AI"。**

---

## 二、整体架构

### 2.1 架构层级图

```
┌─────────────────────────────────────────────────────────────────┐
│                        Isaac Sim                                 │
│                    (物理世界模拟器)                               │
│         模拟重力、碰撞、布料物理、光照、相机                        │
├─────────────────────────────────────────────────────────────────┤
│                        Isaac Lab                                 │
│                    (机器人实验室)                                 │
│         管理机器人关节、传感器、场景、动作执行                       │
├─────────────────────────────────────────────────────────────────┤
│                        LeHome                                    │
│                    (服装操作任务)                                 │
│         定义具体任务：叠衣服、挂衣服、双臂协调                       │
├─────────────────────────────────────────────────────────────────┤
│                        LeRobot                                   │
│                    (模仿学习教练)                                 │
│         从人类示范中学习策略：ACT、Diffusion Policy               │
└─────────────────────────────────────────────────────────────────┘
```

**每一层只关心自己的事：**
- Isaac Sim 不懂什么是"衣服"，只懂物理
- Isaac Lab 不懂什么是"叠衣服"，只懂机器人怎么动
- LeHome 不懂怎么"学习"，只懂任务定义和成功判断
- LeRobot 不懂物理和机器人，只懂"从数据中学策略"

### 2.2 代码目录与职责映射

```
lehome-challenge/
│
├── source/lehome/lehome/          # 任务层代码
│   ├── tasks/bedroom/             # 定义"叠衣服"任务
│   │   ├── __init__.py            # 注册环境
│   │   ├── garment_bi_v2.py       # 任务核心逻辑
│   │   └── garment_bi_cfg_v2.py   # 任务配置
│   │
│   ├── devices/                   # 输入设备
│   │   ├── keyboard/              # 键盘控制
│   │   └── lerobot/               # SO101 Leader Arm
│   │
│   └── utils/                     # 工具函数
│       ├── success_checker.py     # 判断任务成功
│       └── kinematics.py          # 运动学计算
│
├── scripts/                        # 入口脚本
│   ├── eval.py                    # 评估策略
│   ├── dataset.py                 # 数据集操作（不需要仿真）
│   └── dataset_sim.py             # 数据集操作（需要仿真）
│
└── configs/                        # 训练配置
    └── train_act.yaml             # 训练参数
```

### 2.3 数据流

```
【数据收集阶段】
人类操作 SO101 Leader Arm
        ↓
devices/lerobot/bi_so101_leader.py 读取摇杆位置
        ↓
scripts/dataset_sim.py record 记录数据
        ↓
保存到 Datasets/ （LeRobot 格式）

【训练阶段】
lerobot-train 读取 Datasets/
        ↓
LeRobot 框架训练 ACT/Diffusion Policy
        ↓
保存模型到 outputs/

【评估阶段】
scripts/eval.py 加载模型
        ↓
tasks/bedroom/garment_bi_v2.py 创建仿真环境
        ↓
模型输出动作 → 机器人执行 → 相机观察 → 循环
```

---

## 三、Gym Environment 详解

### 3.1 什么是 Gym Environment？

Gym（OpenAI Gym）是一个**标准化的环境接口**：

```python
class Environment:
    def reset() -> observation                    # 重置环境
    def step(action) -> (obs, reward, done, info) # 执行动作
```

**为什么需要这个接口？**
- 不管是训练 ACT、Diffusion Policy，还是手动控制
- 只要调用 `env.step(action)`，机器人就会动

### 3.2 环境注册 (`__init__.py`)

```python
gym.register(
    id="LeHome-BiSO101-Direct-Garment-v2",           # 环境名称（全局唯一ID）
    entry_point=f"{__name__}.garment_bi_v2:GarmentEnv",  # 环境类
    kwargs={
        "env_cfg_entry_point": f"{__name__}.garment_bi_cfg_v2:GarmentEnvCfg",
    },
)
```

**类比理解：**
- `gym.register()` 就像在餐厅菜单上登记一道菜
- `id` 是菜名
- `entry_point` 是厨师
- `env_cfg_entry_point` 是配方

**使用时：**
```python
import gymnasium as gym
import lehome.tasks.bedroom  # 触发 register

env = gym.make("LeHome-BiSO101-Direct-Garment-v2", cfg=env_cfg)
```

### 3.3 GarmentEnv 生命周期

```
┌─────────────────────────────────────────────────────────────┐
│                    GarmentEnv 生命周期                        │
├─────────────────────────────────────────────────────────────┤
│  1. __init__()         → 初始化配置、加载服装参数              │
│          ↓                                                  │
│  2. _setup_scene()     → 创建机器人、相机、场景、服装           │
│          ↓                                                  │
│  ┌───────────────────────────────────────────────┐         │
│  │  【每个 Episode 循环】                           │         │
│  │  3. _reset_idx()      → 重置机器人位置、服装状态  │         │
│  │          ↓                                    │         │
│  │  4. _get_observations() → 获取相机图像、关节位置  │         │
│  │          ↓                                    │         │
│  │  ┌─────────────────────────────────────┐     │         │
│  │  │  【每一步循环】                        │     │         │
│  │  │  5. _pre_physics_step(action)       │     │         │
│  │  │          ↓                          │     │         │
│  │  │  6. _apply_action() → 发送关节指令    │     │         │
│  │  │          ↓                          │     │         │
│  │  │  7. 物理引擎模拟（Isaac Sim）         │     │         │
│  │  │          ↓                          │     │         │
│  │  │  8. _get_observations()             │     │         │
│  │  │          ↓                          │     │         │
│  │  │  9. _get_rewards() → 计算奖励        │     │         │
│  │  │          ↓                          │     │         │
│  │  │  10. _get_dones() → 检查是否结束     │     │         │
│  │  └─────────────────────────────────────┘     │         │
│  └───────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### 3.4 关键方法详解

#### `__init__` —— 初始化

```python
def __init__(self, cfg: GarmentEnvCfg, ...):
    self.cfg = cfg
    self.action_scale = self.cfg.action_scale
    self.object = None  # 服装对象

    # 加载服装配置
    self.garment_loader = ChallengeGarmentLoader(cfg.garment_cfg_base_path)
    self.garment_config = self.garment_loader.load_garment_config(
        cfg.garment_name, cfg.garment_version
    )

    super().__init__(cfg, render_mode, **kwargs)  # 触发 _setup_scene
```

#### `_setup_scene` —— 创建场景

```python
def _setup_scene(self):
    # 创建两个机器人
    self.left_arm = Articulation(self.cfg.left_robot)
    self.right_arm = Articulation(self.cfg.right_robot)

    # 创建三个相机
    self.top_camera = TiledCamera(self.cfg.top_camera)
    self.left_camera = TiledCamera(self.cfg.left_wrist)
    self.right_camera = TiledCamera(self.cfg.right_wrist)

    # 创建服装
    self._create_garment_object()

    # 注册到场景
    self.scene.articulations["left_arm"] = self.left_arm
    self.scene.sensors["top_camera"] = self.top_camera
```

#### `_get_observations` —— 获取观测

```python
def _get_observations(self) -> dict:
    # 获取关节位置（12维：左臂6 + 右臂6）
    joint_pos = torch.cat([left_joint_pos, right_joint_pos], dim=1)

    # 获取相机图像
    top_rgb = self.top_camera.data.output["rgb"]
    left_rgb = self.left_camera.data.output["rgb"]
    right_rgb = self.right_camera.data.output["rgb"]

    return {
        "observation.state": joint_pos,           # (12,) float32
        "observation.images.top_rgb": top_rgb,    # (480, 640, 3) uint8
        "observation.images.left_rgb": left_rgb,
        "observation.images.right_rgb": right_rgb,
        "observation.top_depth": depth_mm,        # (480, 640) uint16
    }
```

**数据格式：**

| Key | Shape | 含义 |
|-----|-------|------|
| `observation.state` | (12,) | 12个关节角度（弧度） |
| `observation.images.*_rgb` | (480, 640, 3) | RGB图像，uint8，[0,255] |
| `observation.top_depth` | (480, 640) | 深度图，uint16，毫米 |

#### `_apply_action` —— 执行动作

```python
def _apply_action(self) -> None:
    # actions: (batch, 12)，前6个左臂，后6个右臂
    self.left_arm.set_joint_position_target(self.actions[:, :6])
    self.right_arm.set_joint_position_target(self.actions[:, 6:])
```

**使用位置控制**：告诉机器人"去哪里"，而不是"用多大力"。

#### `_get_rewards` —— 计算奖励

```python
def _get_rewards(self) -> torch.Tensor:
    result = success_checker_garment_fold(self.object, garment_type)

    if result["success"]:
        return 1.0  # 成功

    # Dense reward：越接近成功，奖励越高
    return computed_reward  # 0.0 ~ 0.9
```

### 3.5 配置类 `GarmentEnvCfg`

```python
@configclass
class GarmentEnvCfg(DirectRLEnvCfg):
    decimation = 1              # 动作频率 = 物理频率 / decimation
    episode_length_s = 60       # 每个episode最多60秒
    action_space = 12           # 动作维度
    observation_space = 12      # 状态维度

    sim: SimulationCfg = SimulationCfg(dt=1/90)  # 90Hz 物理仿真

    left_robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(
        prim_path="/World/Robot/Left_Robot",
        init_state=...,
    )

    top_camera: TiledCameraCfg = TiledCameraCfg(
        width=640,
        height=480,
        update_period=1/30.0,  # 30 FPS
    )
```

**配置分离的好处：**
- 改参数只需改配置，不动代码
- 同一份代码，不同配置 = 不同环境

---

## 四、评估流程

### 4.1 入口点 `eval.py`

```python
def main():
    # 1. 解析命令行参数
    parser = setup_eval_parser()
    args = parser.parse_args()

    # 2. 启动 Isaac Sim
    simulation_app = launch_app_from_args(args)

    try:
        # 3. 导入环境（触发 gym.register）
        import lehome.tasks.bedroom

        # 4. 运行评估
        eval(args, simulation_app)
    finally:
        close_app(simulation_app)
```

### 4.2 核心评估逻辑 `evaluation.py`

```python
def eval(args, simulation_app):
    # 1. 创建环境配置
    env_cfg = parse_env_cfg(args.task, device=args.device)

    # 2. 加载策略
    policy = PolicyRegistry.create(args.policy_type, **kwargs)

    # 3. 创建环境
    env = gym.make(args.task, cfg=env_cfg).unwrapped

    # 4. 对每个服装评估
    for garment_name in eval_list:
        env.switch_garment(garment_name)
        metrics = run_evaluation_loop(env, policy, args)
```

### 4.3 评估循环 `run_evaluation_loop`

```python
def run_evaluation_loop(env, policy, args):
    for episode in range(args.num_episodes):
        env.reset()
        policy.reset()
        obs = env._get_observations()

        for step in range(args.max_steps):
            # 策略推理：观测 → 动作
            action = policy.select_action(obs)

            # 转换为 tensor
            action_tensor = torch.from_numpy(action).unsqueeze(0)

            # 执行动作
            env.step(action_tensor)

            # 获取新观测
            obs = env._get_observations()

            if env._get_success():
                break
```

**核心循环：**
```
观测 ──→ 策略 ──→ 动作 ──→ 环境 ──→ 新观测
  ↑                                        │
  └────────────────────────────────────────┘
```

---

## 五、策略系统

### 5.1 策略基类 `BasePolicy`

```python
class BasePolicy(abc.ABC):
    def reset(self):
        """每个episode开始时调用"""
        pass

    @abc.abstractmethod
    def select_action(self, observation: Dict) -> np.ndarray:
        """核心方法：观测 → 动作"""
        raise NotImplementedError
```

### 5.2 策略注册表 `PolicyRegistry`

```python
# 注册策略
@PolicyRegistry.register("lerobot")
class LeRobotPolicy(BasePolicy):
    ...

# 使用策略
policy = PolicyRegistry.create("lerobot", policy_path="...")
```

### 5.3 LeRobot 策略适配器

```python
@PolicyRegistry.register("lerobot")
class LeRobotPolicy(BasePolicy):
    def __init__(self, policy_path, dataset_root, ...):
        # 加载模型
        self.policy = make_policy(policy_cfg, ds_meta=meta)

        # 创建预处理器
        self.preprocessor, self.postprocessor = make_pre_post_processors(...)

    def select_action(self, observation):
        # 1. 过滤观测
        obs = self._filter_observations(observation)

        # 2. 预处理（numpy → tensor，归一化）
        batch_obs = self._process_observation(obs)

        # 3. 推理
        action = self.policy.select_action(batch_obs)

        # 4. 后处理（反归一化）
        action = self.postprocessor(action)

        return action.cpu().numpy()
```

---

## 六、动手练习

### 练习 1：画流程图

在纸上画出从 `python -m scripts.eval --policy_type lerobot` 到机器人执行第一个动作的完整流程。

### 练习 2：理解观测

1. `observation.state` 的 shape 是什么？每个维度代表什么？
2. 相机图像是什么格式？数值范围是多少？
3. 如果只用顶部相机，需要改哪里？

### 练习 3：理解动作

1. 动作空间是多少维？为什么？
2. `_apply_action` 做了什么？
3. 如果动作超出 [-1, 1] 会怎样？

### 练习 4：追踪代码

找出以下定义的位置：
1. `success_checker_garment_fold`
2. `SO101_FOLLOWER_CFG`
3. `PolicyRegistry`

### 练习 5：理解关节顺序

关节顺序是 `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`。

问：
- 索引 2 是哪个关节？
- 如果要让 gripper 闭合，应该发送什么值？

---

## 附录：关键文件清单

| 文件 | 作用 |
|------|------|
| `source/lehome/lehome/tasks/bedroom/__init__.py` | 环境注册 |
| `source/lehome/lehome/tasks/bedroom/garment_bi_v2.py` | 环境核心逻辑 |
| `source/lehome/lehome/tasks/bedroom/garment_bi_cfg_v2.py` | 环境配置 |
| `scripts/eval.py` | 评估入口 |
| `scripts/utils/evaluation.py` | 评估循环 |
| `scripts/eval_policy/base_policy.py` | 策略基类 |
| `scripts/eval_policy/lerobot_policy.py` | LeRobot 适配器 |

---

*文档生成时间：2026-02-27*
