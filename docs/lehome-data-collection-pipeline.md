# LeHome Data Collection Pipeline

This document explains how LeHome implements SO101 teleoperation data collection, providing context for HIL evaluation integration.

## Overview

LeHome's data collection system enables recording teleoperation demonstrations using:
- **Bimanual SO101 leader arms** for human input
- **Isaac Lab simulation** for robot execution
- **LeRobot dataset format** (v3.0) for data storage

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      DATA COLLECTION SYSTEM                             │
│                                                                         │
│  ┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐  │
│  │   SO101 Leader   │         │   Recording       │         │   Isaac Lab      │  │
│  │    Arms          │────────►│   System         │────────►│   Environment    │  │
│  │  (Teleoperation)  │         │                  │         │   (Follower)     │  │
│  └──────────────────┘         └──────────────────┘         └──────────────────┘  │
│         ▲                                                   ▲               │
│         │                                                   │               │
│         │              Human controls robot via SO101           │               │
│         │                                                   │               │
│         └──────────────────────────┬───────────────────────────────┘           │
│                                      △                                      │
│                    ┌──────────────────────────────┐                  │
│                    │    LeRobot Dataset           │                  │
│                    │  - observation.state          │                  │
│                    │  - action                   │                  │
│                    │  - observation.images.*      │                  │
│                    │  - observation.top_depth     │                  │
│                    └──────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Entry Points

| File | Purpose |
|------|---------|
| `scripts/dataset_sim.py` | CLI entry point for dataset operations |
| `scripts/utils/dataset_record.py` | Main recording logic and episode management |

### 2. Device Layer

| File | Description |
|------|-------------|
| `source/lehome/lehome/devices/lerobot/bi_so101_leader.py` | Bimanual SO101 wrapper |
| `source/lehome/lehome/devices/lerobot/so101_leader.py` | Single-arm SO101 implementation |
| `source/lehome/lehome/lehome/devices/lerobot/common/motors/feetech.py` | Motor bus for hardware communication |

### 3. Action Processing

| File | Description |
|------|-------------|
| `source/lehome/lehome/devices/action_process.py` | Converts device actions to environment actions |

### 4. Environment

| File | Description |
|------|-------------|
| `source/lehome/lehome/tasks/bedroom/garment_bi_v2.py` | Garment manipulation task |

## Recording Control Keys

The recording system uses keyboard callbacks for control:

| Key | Function | Phase |
|-----|----------|-------|
| **B** | Start simulation (enable SO101 control) | Idle |
| **S** | Start recording | Idle |
| **N** | Save episode (mark as successful) | Recording |
| **D** | Discard episode and re-record | Recording |
| **ESC** | Abort recording and clear buffer | Any |

**Recording Flow**:
```
1. Press B → Enable SO101 control (_started = True)
2. Press S → Start recording (flags["start"] = True)
3. During recording:
   - Episode completes automatically → saved
   - Press N → Save episode immediately
   - Press D → Discard and re-record
4. Press ESC → Abort entire session
```

## Data Flow

### Complete Pipeline

```
SO101 Hardware Motor Positions
    ↓ (Serial Port)
FeetechMotorsBus.sync_read("Present_Position")
    ↓ (Motor positions in degrees)
SO101Leader.get_device_state()
    ↓ (Dict: {motor: position, ...})
BiSO101Leader.input2action()
    ↓ (Dict: {"started": bool, "joint_state": {...}, "motor_limits": {...}})
preprocess_device_action()
    ↓ (Convert: motor limits → joint limits → degrees → radians)
torch.Tensor (12,) in radians
    ↓
env.step(action)
    ↓ (Isaac Lab physics simulation)
env._get_observations()
    ↓ (Dict: observation.state, images, depth, etc.)
dataset.add_frame(frame)
    ↓ (LeRobot v3.0 format)
LeRobot Dataset (parquet files + images + videos)
```

### Motor to Joint Conversion

**Motor Limits** (normalized hardware range):
```python
SO101_FOLLOWER_MOTOR_LIMITS = {
    "shoulder_pan": (-100.0, 100.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 100.0),
    "wrist_flex": (-100.0, 100.0),
    "wrist_roll": (-100.0, 100.0),
    "gripper": (0.0, 100.0),
}
```

**Joint Limits** (simulation range in degrees):
```python
SO101_FOLLOWER_USD_JOINT_LIMLITS = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 90.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (-10, 100.0),
}
```

**Conversion Formula**:
```python
# Motor position → Joint position (degrees)
processed_degree = (joint_state[motor] - motor_limit[0]) /
                   (motor_limit[1] - motor_limit[0]) *
                   (joint_limit[1] - joint_limit[0]) + joint_limit[0]

# Degrees → Radians
processed_radian = processed_degree / 180.0 * π
```

## Device State Format

### Single Arm (SO101Leader)

```python
{
    "shoulder_pan.pos": float,  # degrees or normalized
    "shoulder_lift.pos": float,
    "elbow_flex.pos": float,
    "wrist_flex.pos": float,
    "wrist_roll.pos": float,
    "gripper.pos": float,      # 0-100 range
}
```

### Bimanual (BiSO101Leader.input2action())

```python
{
    "started": bool,              # True after B key pressed
    "reset": bool,                # Reset flag
    "joint_state": {
        "left_arm": {
            "shoulder_pan.pos": float,
            "shoulder_lift.pos": float,
            # ... all 6 joints
        },
        "right_arm": {
            "shoulder_pan.pos": float,
            "shoulder_lift.pos": float,
            # ... all 6 joints
        }
    },
    "motor_limits": {
        "left_arm": {
            "shoulder_pan": (min, max),
            # ... all 6 joints
        },
        "right_arm": {
            "shoulder_pan": (min, max),
            # ... all 6 joints
        }
    }
}
```

## Dataset Format (LeRobot v3.0)

### Directory Structure

```
Datasets/record/001/
├── data/
│   └── chunk-000/
│       ├── file-000.parquet         # Frame data
│       └── file-001.parquet
├── images/
│   ├── observation.images.top_rgb   # RGB images
│   ├── observation.images.left_rgb
│   └── observation.images.right_rgb
├── meta/
│   ├── episodes/
│   │   └── chunk-000/
│   │       └── file-000.parquet      # Episode metadata
│   ├── garment_info.json             # Custom garment metadata
│   ├── info.json                     # Dataset info
│   ├── stats.json                    # Dataset statistics
│   └── tasks.parquet                  # Task descriptions
├── videos/
│   ├── observation.images.top_rgb
│   ├── observation.images.left_rgb
│   └── observation.images.right_rgb
```

### Frame Data Structure

Each frame in `data/chunk-000/file-000.parquet`:

```python
{
    # Joint positions (12D for dual-arm)
    "observation.state": np.ndarray,  # (12,) in radians
    "action": np.ndarray,                # (12,) in radians

    # Camera images
    "observation.images.top_rgb": np.ndarray,  # (480, 640, 3)
    "observation.images.left_rgb": np.ndarray,
    "observation.images.right_rgb": np.ndarray,

    # Optional depth map
    "observation.top_depth": np.ndarray,  # (480, 640) uint16 millimeters

    # Task description
    "task": str,

    # Optional end-effector poses (16D for dual-arm)
    # "observation.ee_pose": np.ndarray,  # (16,) [left(8), right(8)]
    # "action.ee_pose": np.ndarray,
}
```

### Episode Metadata

Each row in `meta/episodes/chunk-000/file-000.parquet`:

```python
{
    "episode_index": int,          # Episode number
    "length": int,                   # Number of frames
    "task": str,                     # Task description

    # Custom metadata
    "episode_success": str,         # "success" or "failure"

    # Timestamps
    "timestamp_start": float,
    "timestamp_end": float,
}
```

## Recording Phases

### 1. Idle Phase

**Purpose**: Stabilize garment and wait for recording start

**Behavior**:
- Maintains current robot position
- Waits for **S** key press to start recording
- Can abort with **ESC** key

### 2. Recording Phase

**Purpose**: Record teleoperation demonstration

**Behavior**:
- Records frames at ~30 FPS
- Each frame contains:
  - Observations from cameras
  - Robot joint positions
  - Applied actions
- Continues until:
  - Episode truncated (environment condition)
  - **N** key pressed (mark successful)
  - **D** key pressed (discard and re-record)

### 3. Episode Saving

**Triggers**:
- Automatic: Episode truncated (max steps or task complete)
- Manual: **N** key pressed (mark successful)
- Discard: **D** key pressed (re-record)

**Data Saved**:
- All frames from current episode
- Episode metadata
- Task description

## Calibration

### Calibration Storage

Calibration data stored in:
```
source/lehome/lehome/devices/lerobot/.cache/
├── so101_leader.json      # Single arm calibration
├── left_arm_calibration.json
└── right_arm_calibration.json
```

### Calibration Data Structure

```python
{
    "shoulder_pan": {
        "id": 1,
        "drive_mode": 0,
        "homing_offset": int,
        "range_min": float,
        "range_max": float,
    },
    # ... other 5 motors
}
```

### Recalibration

Forced recalibration via `--recalibrate` flag:
1. Moves arm to middle of range
2. Records homing offsets
3. Moves through full range of motion
4. Records min/max positions
5. Saves to calibration file

## Recording Commands

### Basic Recording

```bash
python -m scripts.dataset_sim record \
    --teleop_device bi-so101leader \
    --garment_name Top_Long_Unseen_0 \
    --enable_record \
    --num_episode 10 \
    --log_success \
    --device "cpu" \
    --enable_cameras
```

### With Custom Ports

```bash
python -m scripts.dataset_sim record \
    --teleop_device bi-so101leader \
    --left_arm_port /dev/ttyACM0 \
    --right_arm_port /dev/ttyACM1 \
    # ... other parameters
```

### With Recalibration

```bash
python -m scripts.dataset_sim record \
    --teleop_device bi-so101leader \
    --recalibrate \
    # ... other parameters
```

## Key Differences: Data Collection vs HIL Evaluation

| Aspect | Data Collection | HIL Evaluation |
|--------|----------------|-----------------|
| **Purpose** | Record demonstrations | Evaluate policy with intervention |
| **Control** | Human-only | Policy + Human (switchable) |
| **Leader Arms** | Read-only | Read + Write (policy_sync) |
| **Torque** | Always disabled | Switches based on mode |
| **Data** | Single source (human) | Dual source (policy + human) |
| **Metadata** | Task description | + policy_action, is_intervention |

## Integration Points for HIL

### Reusable Components

1. **BiSO101Leader Class**: Already implemented, can be reused
2. **Torque Control**: `enable_torque()` / `disable_torque()` on `FeetechMotorsBus`
3. **State Format**: `{"joint_state": {"left_arm": {...}, "right_arm": {...}}}`
4. **Action Processing**: `preprocess_device_action()` handles conversion
5. **Calibration System**: Automatic with `.cache/*.json` storage

### HIL-Specific Needs

1. **Mode Switching**: Add torque mode toggle between POLICY/HUMAN
2. **Policy Sync**: Send policy actions to leader during POLICY mode
3. **Intervention Tracking**: Record when human takes over
4. **Dual Source Dataset**: Track both policy and human actions

## Hardware Requirements

### SO101 Leader Arms

- **Quantity**: 2 (left and right)
- **Connection**: USB serial (usually `/dev/ttyACM0`, `/dev/ttyACM1`)
- **Motors**: 6 STS3215 motors per arm
- **Power**: External power supply required

### Computer

- **OS**: Linux (Ubuntu recommended)
- **Python**: 3.11
- **Permissions**: Serial port access (`dialout` group)

### SO-ARM100 Reference

- Documentation: https://github.com/TheRobotStudio/SO-ARM100
- SDK: `scservo_sdk`
- Motor Control: Feetech protocol

## Troubleshooting

### Device Not Found

```bash
# Check serial devices
ls /dev/ttyACM*

# Check permissions
ls -la /dev/ttyACM*

# Add user to dialout group
sudo usermod -aG dialout $USER

# Test connection
python -c "from lehome.devices import BiSO101Leader; ..."
```

### Calibration Issues

```bash
# Recalibrate devices
python -m scripts.dataset_sim record \
    --teleop_device bi-so101leader \
    --recalibrate

# Clear calibration cache
rm source/lehome/lehome/devices/lerobot/.cache/*.json
```

### Action Not Applied

1. Check **B** key pressed first (enables control)
2. Check device connection
3. Check calibration is loaded
4. Check action limits

## References

- [Dataset Documentation](datasets.md) - Complete dataset guide
- [SO101 HIL Integration Plan](so101-hil-integration-plan.md) - HIL evaluation integration
- [LeRobot Dataset Format](https://github.com/huggingface/lerobot) - LeRobot v3.0 format
- [SO-ARM100 Documentation](https://github.com/TheRobotStudio/SO-ARM100) - Hardware reference
