# SO101 Bimanual Arms HIL Integration Plan

## Overview

This document outlines the integration of bimanual SO101 leader/follower arms into the Human-in-the-Loop (HIL) evaluation system for IsaacLab garment manipulation tasks.

**Goal**: Enable real teleoperation during HIL evaluation where human operator can control the robot using SO101 leader arms, with automatic switching between policy and human control.

### Critical Design Principle: policy_sync_to_teleop

**Important**: This integration implements the `policy_sync_to_teleop` feature from Evo-RL's `lerobot-human-inloop-record`. This means:

- **POLICY mode**: Policy actions are sent to **BOTH leader and follower arms** (synchronized execution)
- **HUMAN mode**: Human takes over leader arms, and leader actions are sent to follower arms
- **Intervention detection**: System detects deviation between leader position and policy output

This design allows seamless transition: the leader arms move with the follower during policy execution, so the human can naturally take over by grabbing the leader arms without sudden mode switches.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         HIL Evaluation System                          │
│                    (with policy_sync_to_teleop)                        │
│                                                                         │
│  ┌──────────────┐         ┌──────────────┐         ┌──────────────┐     │
│  │   IsaacLab    │         │  Keyboard     │         │   SO101      │     │
│  │  Environment  │◄──────►│  Handler     │◄──────►│   Leader      │     │
│  │              │         │              │         │   (Teleop)    │     │
│  │  Follower     │         │              │         │              │     │
│  │  Arms        │◄──────►│  Policy       │         │  Left/Right   │     │
│  │              │         │   └──────┐    │         │              │     │
│  └──────────────┘         └──────────┼────┘         └──────▲───────┘     │
│         ▲                           │                     │               │
│         │                           │                     │               │
│         │              POLICY mode: │              HUMAN mode:            │
│         │              policy → (leader + follower)    leader → follower  │
│         │                           │                     │               │
│         └──────────────────────┬─────────────────────────────┘           │
│                                △                                      │
│                    ┌──────────────────────────────┐                  │
│                    │    Evo-RL Dataset           │                  │
│                    │  - policy_action            │                  │
│                    │  - is_intervention          │                  │
│                    │  - state (POLICY/HUMAN)      │                  │
│                    │  - episode_success           │                  │
│                    └──────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Mode Switching Behavior

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MODE TRANSITION STATE MACHINE                       │
│                                                                         │
│  ┌────────────────┐          Press 'i'         ┌────────────────┐     │
│  │   POLICY MODE  │ ─────────────────────────►│  HUMAN MODE    │     │
│  │                │                            │                │     │
│  │  policy →      │  (toggle intervention)      │  leader →      │     │
│  │  leader+follower│                            │  follower      │     │
│  │                │◄───────────────────────────│                │     │
│  └────────────────┘      Press 'i' again       └────────────────┘     │
│                                                                         │
│  Key Insight: In POLICY mode, leader arms mirror follower movements.   │
│  Human can seamlessly take over by grabbing leader arms.               │
└─────────────────────────────────────────────────────────────────────────┘
```

## Hardware Requirements

### Devices Needed

| Device | Quantity | Purpose |
|--------|----------|---------|
| SO101 Leader Arms | 2 | Human teleoperation input |
| SO101 Follower Arms | 2 | Robot execution in IsaacLab |
| Intel RealSense (optional) | 1 | External view camera |

### Port Configuration

```bash
# Find connected SO101 devices
ls /dev/serial/by-id/ | grep -i shenzhen

# Example output:
# usb-TheRobotStudio_SO-ARM100_<ID>-if00 → /dev/ttyACM0 (Left Leader)
# usb-TheRobotStudio_SO-ARM100_<ID>-if00 → /dev/ttyACM1 (Right Leader)
# usb-TheRobotStudio_SO-ARM100_<ID>-if00 → /dev/ttyACM2 (Left Follower)
# usb-TheRobotStudio_SO-ARM100_<ID>-if00 → /dev/ttyACM3 (Right Follower)
```

## Implementation Steps

### Step 1: SO101 Device Manager Module

**File**: `scripts/devices/so101_manager.py`

**Purpose**: Abstraction layer for SO101 leader arms communication with policy_sync_to_teleop support

**IMPORTANT**: Uses **LeHome's existing SO101 device classes** (not Evo-RL's):
- `from lehome.devices import BiSO101Leader, SO101Leader`
- Located in: `source/lehome/lehome/devices/lerobot/`

```python
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

JOINT_ORDER = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper"
]

class SO101BimanualManager:
    """Manage bimanual SO101 leader arms for HIL evaluation with policy_sync_to_teleop.

    Uses LeHome's existing SO101 device implementation from:
    - source/lehome/lehome/devices/lerobot/bi_so101_leader.py
    - source/lehome/lehome/devices/lerobot/so101_leader.py

    Critical implementation notes:
    - Uses LeHome's BiSO101Leader class (not Evo-RL's)
    - Uses FeetechMotorsBus with scservo_sdk for motor control
    - Torque control: enable_torque() for POLICY mode, disable_torque() for HUMAN mode
    - SO101 uses degrees, IsaacLab uses radians
    - Calibration stored in .cache/*.json files
    """

    def __init__(self,
                 left_port: str,
                 right_port: str,
                 recalibrate: bool = False,
                 policy_sync_to_teleop: bool = True):
        """
        Args:
            left_port: Serial port for left SO101 leader arm (e.g., /dev/ttyACM0)
            right_port: Serial port for right SO101 leader arm (e.g., /dev/ttyACM1)
            recalibrate: Whether to force recalibration on connection
            policy_sync_to_teleop: If True, send policy actions to leader during POLICY mode
        """
        self.left_port = left_port
        self.right_port = right_port
        self.recalibrate = recalibrate
        self.policy_sync_to_teleop = policy_sync_to_teleop

        # BiSO101Leader instance (LeHome's implementation)
        self.teleop: Optional['BiSO101Leader'] = None

        # Track torque mode for proper mode switching
        self._torque_enabled = False  # True = POLICY mode, False = HUMAN mode

    def connect(self, env=None):
        """Connect to SO101 leader devices using LeHome's BiSO101Leader.

        Args:
            env: Optional environment instance (for device registration)
        """
        from lehome.devices import BiSO101Leader

        # Create bimanual SO101 leader device
        self.teleop = BiSO101Leader(
            env=env,
            left_port=self.left_port,
            right_port=self.right_port,
            recalibrate=self.recalibrate,
        )

        # Connect to devices
        self.teleop.connect()

        logger.info(f"SO101 bimanual leader arms connected: {self.left_port}, {self.right_port}")

        # Start in manual control mode (HUMAN mode - torque disabled)
        self._disable_torque()

    def get_leader_action(self) -> np.ndarray:
        """Read current action from SO101 leader arms.

        IMPORTANT: This requires torque disabled (manual control mode).
        Call _disable_torque() before this method if in POLICY mode.

        Returns:
            np.ndarray: (12,) joint positions in degrees [left_arm(6), right_arm(6)]
        """
        # Get device state from LeHome's BiSO101Leader
        device_state = self.teleop.get_device_state()

        # Extract joint positions from device state
        # device_state format: {"joint_state": {"left_arm": {...}, "right_arm": {...}}}
        joint_state = device_state.get("joint_state", {})

        left_arm = joint_state.get("left_arm", {})
        right_arm = joint_state.get("right_arm", {})

        # Convert to numpy array [left(6), right(6)] in degrees
        action = np.array([
            left_arm.get(joint, 0.0) for joint in JOINT_ORDER
        ] + [
            right_arm.get(joint, 0.0) for joint in JOINT_ORDER
        ])

        return action

    def send_to_leader(self, action: np.ndarray) -> bool:
        """Send action to SO101 leader arms (for policy_sync_to_teleop).

        CRITICAL: This enables torque mode. The leader arms will resist
        human movement when in this mode (allowing policy-driven movement).

        Args:
            action: (12,) array [left_arm(6), right_arm(6)] in degrees

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.policy_sync_to_teleop:
            return True  # Skip if disabled

        try:
            # Split into left and right actions (in degrees)
            left_action = action[:6]
            right_action = action[6:]

            # Send to each arm's motor bus
            # LeHome's SO101Leader exposes _bus for direct motor control
            self.teleop.left_arm._bus.enable_torque()
            self.teleop.right_arm._bus.enable_torque()

            # Write goal positions to motors
            for i, joint in enumerate(JOINT_ORDER):
                motor_pos = left_action[i]
                self.teleop.left_arm._bus.write("Goal_Position", joint, motor_pos)

            for i, joint in enumerate(JOINT_ORDER):
                motor_pos = right_action[i]
                self.teleop.right_arm._bus.write("Goal_Position", joint, motor_pos)

            self._torque_enabled = True
            return True

        except Exception as e:
            logger.error(f"Failed to send action to leader: {e}")
            return False

    def _disable_torque(self):
        """Disable torque for manual control mode (HUMAN mode).

        This allows the human to move leader arms freely.
        """
        if self.teleop:
            try:
                self.teleop.left_arm._bus.disable_torque()
                self.teleop.right_arm._bus.disable_torque()
                self._torque_enabled = False
            except Exception as e:
                logger.error(f"Failed to disable torque: {e}")

    def disconnect(self):
        """Disconnect from SO101 devices."""
        if self.teleop:
            self._disable_torque()  # Ensure torque is disabled before disconnect
            self.teleop.disconnect()

    def emergency_stop(self):
        """Emergency stop all SO101 devices.

        This disables torque to stop movement immediately.
        """
        self._disable_torque()
        logger.warning("SO101 emergency stop activated - torque disabled")
```

**Key Implementation Notes**:
1. Uses **LeHome's `BiSO101Leader`** from `lehome.devices` (not Evo-RL)
2. Uses `FeetechMotorsBus` with `scservo_sdk` (LeHome's motor control layer)
3. Torque control via `_bus.enable_torque()` / `_bus.disable_torque()`
4. SO101 uses **degrees**, IsaacLab uses **radians** (conversion handled in evaluation.py)
5. Calibration handled automatically by `BiSO101Leader` (stores in `.cache/*.json`)
6. Device state format: `{"joint_state": {"left_arm": {...}, "right_arm": {...}}}`

### Step 2: HIL Enhancement for SO101 Integration

**File**: `scripts/utils/evaluation.py`

**Modifications**:

#### 2.1 Add SO101 Manager Initialization

```python
# After HIL keyboard handler setup
if args.enable_hil and args.enable_so101:
    from scripts.devices.so101_manager import SO101BimanualManager

    so101_manager = SO101BimanualManager(
        leader_ports=(args.left_leader_port, args.right_leader_port),
        calibration_dir=args.so101_calibration_dir,
        policy_sync_to_teleop=args.policy_sync_to_teleop
    )
    so101_manager.connect(calibrate=args.so101_calibrate)

    logger.info("SO101 bimanual leader arms connected")
    if args.policy_sync_to_teleop:
        logger.info("Policy sync to teleop enabled: leader will mirror follower")
else:
    so101_manager = None
```

#### 2.2 Modify Action Selection Logic (with torque mode switching and unit conversion)

**CRITICAL**: This implementation handles:
1. Torque mode switching between POLICY and HUMAN modes
2. Degrees/radians conversion (SO101 uses degrees, IsaacLab uses radians)
3. Proper tracking of policy actions vs executed actions

```python
# 3. Policy Inference / SO101 Reading
if hil_handler and hil_handler.is_intervention_active() and so101_manager:
    # HUMAN MODE: Read from SO101 leader

    # CRITICAL: Enable manual control mode before reading
    # This disables torque so human can move leader arms freely
    so101_manager.enable_manual_control()

    # Read from leader (returns degrees)
    leader_action_degrees = so101_manager.get_leader_action()

    # Convert degrees → radians (IsaacLab expects radians)
    action_np = np.deg2rad(leader_action_degrees)

    # Validate action (safety limits in radians)
    action_np = validate_action(action_np)

    # Track what policy would have done (for Evo-RL dataset)
    policy_action_for_dataset = policy.select_action(observation_dict)
    last_policy_action = policy_action_for_dataset.copy()

    is_intervention = True

else:
    # POLICY MODE: Use policy inference with sync to leader

    # Get policy action (already in radians)
    policy_action_for_dataset = policy.select_action(observation_dict)
    action_np = policy_action_for_dataset.copy()
    last_policy_action = action_np.copy()
    is_intervention = False

    # CRITICAL: Send policy action to leader for seamless takeover
    # Convert radians → degrees (SO101 expects degrees)
    # This also enables torque mode automatically via send_feedback()
    if so101_manager and args.policy_sync_to_teleop:
        action_degrees = np.rad2deg(action_np)
        so101_manager.send_to_leader(action_degrees)

# 4. Execute action in IsaacLab (follower moves)
obs, reward, terminated, truncated, info = env.step(action_np)

# 5. Record for Evo-RL dataset
policy_actions.append(policy_action_for_dataset)  # What policy wanted
is_interventions.append(is_intervention)  # Human took over?
```

**Key Changes**:
1. Added `so101_manager.enable_manual_control()` call in HUMAN mode
2. Added degrees ↔ radians conversion for SO101 communication
3. Policy actions are tracked separately from executed actions
4. Torque mode is handled automatically by `send_feedback()`

### Step 3: Command Line Arguments

**File**: `scripts/utils/parser.py`

```python
# SO101 Device Configuration
parser.add_argument("--enable_so101", action="store_true",
                   help="Enable SO101 bimanual leader arms for HIL")
parser.add_argument("--left_leader_port", type=str, default="/dev/ttyACM0",
                   help="Port for left SO101 leader arm")
parser.add_argument("--right_leader_port", type=str, default="/dev/ttyACM1",
                   help="Port for right SO101 leader arm")
parser.add_argument("--so101_calibrate", action="store_true",
                   help="Force SO101 calibration on startup")
parser.add_argument("--so101_calibration_dir", type=str,
                   default=HF_LEROBOT_HOME / "so101_calibrations",
                   help="Directory for SO101 calibration files")

# Policy Sync to Teleop (Evo-RL feature)
parser.add_argument("--policy_sync_to_teleop", action="store_true", default=True,
                   help="Send policy actions to leader arms during POLICY mode (enables seamless human takeover)")
parser.add_argument("--no_policy_sync_to_teleop", dest="policy_sync_to_teleop",
                   action="store_false",
                   help="Disable policy sync to teleop (leader stays stationary during policy mode)")
```

### Step 4: Safety Considerations

#### 4.1 Action Limits (in radians)

**IMPORTANT**: IsaacLab uses radians. The validate_action function should clip actions to safe joint limits.

```python
import numpy as np

# Action limits in radians for IsaacLab
ACTION_LIMITS_RAD = {
    "shoulder_pan": (-3.14, 3.14),      # ±180°
    "shoulder_lift": (-1.57, 1.57),     # ±90°
    "elbow_flex": (-2.53, 2.53),        # ±145°
    "wrist_flex": (-1.57, 1.57),        # ±90°
    "wrist_roll": (-3.14, 3.14),        # ±180°
    "gripper": (0.0, 1.0),              # 0-1 (normalized)
}

def validate_action(action: np.ndarray) -> np.ndarray:
    """Validate and clip action to safe joint limits (radians).

    Args:
        action: (12,) array [left_arm(6), right_arm(6)] in radians

    Returns:
        np.ndarray: Clipped action within safe limits
    """
    # Create limits for both arms
    limits = [ACTION_LIMITS_RAD[j] for j in JOINT_ORDER for _ in range(2)]
    min_vals = np.array([l[0] for l in limits])
    max_vals = np.array([l[1] for l in limits])

    # Clip action to limits
    clipped = np.clip(action, min_vals, max_vals)

    # Check if any values were clipped
    if not np.allclose(action, clipped):
        logger.warning(f"Action clipped to safe limits: {action} → {clipped}")

    return clipped
```

#### 4.2 Emergency Stop

```python
# Emergency stop on critical errors
if error_detected:
    so101_manager.emergency_stop()
    hil_handler.stop()
    logger.error("Emergency stop activated!")
```

#### 4.3 Mode Transition Safety

**CRITICAL**: When switching from POLICY to HUMAN mode:
1. Always call `enable_manual_control()` before reading from leader
2. This prevents the leader arms from resisting human movement
3. The torque must be disabled for human to move leader freely

```python
# Example of safe mode transition
if switching_to_human_mode:
    so101_manager.enable_manual_control()  # Disable torque
    # Now safe to read from leader
    action = so101_manager.get_leader_action()
```

### Step 5: Testing Strategy

#### Phase 1: Device Connection Test

```bash
# Test SO101 device discovery
python scripts/test_so101_connection.py \
    --left_port /dev/ttyACM0 \
    --right_port /dev/ttyACM1
```

**Expected Output**:
```
✅ Found SO101 Leader on /dev/ttyACM0
✅ Found SO101 Leader on /dev/ttyACM1
✅ Both devices connected successfully
```

#### Phase 2: Action Reading Test

```bash
# Test reading actions from SO101
python scripts/test_so101_reading.py \
    --left_port /dev/ttyACM0 \
    --right_port /dev/ttyACM1
```

**Expected Output**:
```
Reading SO101 actions (press Ctrl+C to stop):
  Frame 0: [0.1, -0.5, 1.2, 0.0, 0.3, 0.8,  0.2, -0.3, 0.9, -0.1, 0.5, 0.7]
  Frame 1: [0.1, -0.5, 1.2, 0.0, 0.3, 0.8,  0.2, -0.3, 0.9, -0.1, 0.5, 0.7]
  ...
```

#### Phase 3: HIL Integration Test (Without IsaacLab)

```bash
# Test HIL + SO101 without simulation
python scripts/test_hil_so101.py \
    --left_port /dev/ttyACM0 \
    --right_port /dev/ttyACM1 \
    --max_frames 100
```

**Expected Output**:
```
HIL+SO101 Test
Mode: POLICY
Press 'i' to toggle intervention mode

[User presses 'i']
Mode switched to: HUMAN
Now reading from SO101 leader arms...
Action: [left_arm_6D, right_arm_6D]
```

#### Phase 4: Full Integration Test

```bash
# Full HIL evaluation with SO101
CUDA_VISIBLE_DEVICES=0 python -m scripts.eval \
    --policy_type lerobot \
    --policy_path outputs/moe_train/smolvla_moe_expert_pant_long_no_st_proj/checkpoints/004000/pretrained_model \
    --dataset_root Datasets/example/four_types_merged \
    --garment_type "pant_long" \
    --num_episodes 5 \
    --max_steps 300 \
    --enable_cameras \
    --device cpu \
    --headless \
    --enable_hil \
    --enable_so101 \
    --left_leader_port /dev/ttyACM0 \
    --right_leader_port /dev/ttyACM1 \
    --save_datasets \
    --save_mode all \
    --eval_dataset_path Datasets/hil_so101/pant_long
```

## Data Flow

### Action Flow (with policy_sync_to_teleop and torque mode switching)

```
┌────────────────────────────────────────────────────────────────────────┐
│                         EPISODE STEP                                 │
│                                                                         │
│  1. Poll Keyboard Handler                                              │
│     hil_handler.poll()                                                 │
│     │                                                                  │
│     ├─► Check: is_intervention_active()?                               │
│     │                                                                  │
│  2. Select Action Source & Execution Targets                          │
│     IF is_intervention AND so101_manager.connected:                   │
│         │                                                              │
│         │  HUMAN MODE                                                  │
│         │  ┌───────────────────────────────────────────────────┐     │
│         │  │ 1. CRITICAL: Enable manual control mode          │     │
│         │  │    so101_manager.enable_manual_control()         │     │
│         │  │    (Disables torque so human can move leader)    │     │
│         │  │                                                   │     │
│         │  │ 2. Read from SO101 leader arms                    │     │
│         │  │    leader_action_deg = get_leader_action()       │     │
│         │  │    (Returns degrees)                             │     │
│         │  │                                                   │     │
│         │  │ 3. Convert degrees → radians                      │     │
│         │  │    action_rad = deg2rad(leader_action_deg)       │     │
│         │  │                                                   │     │
│         │  │ 4. Validate leader action                         │     │
│         │  │    action_rad = validate_action(action_rad)      │     │
│         │  │                                                   │     │
│         │  │ 5. Send to follower ONLY (IsaacLab)               │     │
│         │  │    env.step(action_rad)  # Follower moves         │     │
│         │  │                                                   │     │
│         │  │ 6. Track for dataset                             │     │
│         │  │    executed_action = action_rad                  │     │
│         │  │    is_intervention = True                        │     │
│         │  └───────────────────────────────────────────────────┘     │
│         │                                                              │
│     ELSE:                                                             │
│         │                                                              │
│         │  POLICY MODE (policy_sync_to_teleop enabled)                │
│         │  ┌───────────────────────────────────────────────────┐     │
│         │  │ 1. Get policy action (radians)                   │     │
│         │  │    policy_action = policy.select_action(obs)      │     │
│         │  │                                                   │     │
│         │  │ 2. Validate policy action                         │     │
│         │  │    policy_action = validate_action(policy_action) │     │
│         │  │                                                   │     │
│         │  │ 3. Convert radians → degrees for SO101           │     │
│         │  │    action_deg = rad2deg(policy_action)           │     │
│         │  │                                                   │     │
│         │  │ 4. Send to BOTH leader AND follower              │     │
│         │  │    so101_manager.send_to_leader(action_deg)      │     │
│         │  │    (Enables torque automatically)               │     │
│         │  │    env.step(policy_action)  # Follower moves     │     │
│         │  │                                                   │     │
│         │  │ 5. Track for dataset                             │     │
│         │  │    executed_action = policy_action               │     │
│         │  │    is_intervention = False                       │     │
│         │  └───────────────────────────────────────────────────┘     │
│         │                                                              │
│  3. Track for Evo-RL Dataset                                         │
│     policy_actions.append(policy_action)  # What policy wanted          │
│     is_interventions.append(is_intervention)  # Human control?         │
│                                                                         │
│  4. Record Frame with Evo-RL Metadata                                │
│     frame = {                                                          │
│         "observation.state": ...,                                     │
│         "action": action_rad,  # Executed action (radians)            │
│         "complementary_info.policy_action": policy_action,            │
│         "complementary_info.is_intervention": is_intervention,         │
│         "complementary_info.state": 1 if is_intervention else 0,      │
│         "complementary_info.collector_policy_id": "human" if is_intervention else "policy" │
│     }                                                                 │
│     eval_dataset.add_frame(frame)                                    │
└────────────────────────────────────────────────────────────────────────┘
```

### Key Implementation Details

1. **Torque Mode Switching**:
   - HUMAN mode: `enable_manual_control()` disables torque
   - POLICY mode: `send_to_leader()` automatically enables torque via `send_feedback()`

2. **Unit Conversion**:
   - SO101 uses **degrees** (`use_degrees=True` in config)
   - IsaacLab uses **radians**
   - Conversion: `np.deg2rad()` and `np.rad2deg()`

3. **Mode Transition Safety**:
   - Always call `enable_manual_control()` before reading from leader
   - This prevents leader arms from resisting human movement

### Key Difference from Traditional Teleoperation

**Traditional approach** (NOT what we're implementing):
- Policy mode: policy → follower only
- Human mode: leader → follower
- Problem: Leader is stationary during policy, sudden jump when human takes over

**Evo-RL policy_sync_to_teleop approach** (what we're implementing):
- Policy mode: policy → (leader + follower in sync)
- Human mode: leader → follower
- Advantage: Leader mirrors follower during policy, seamless human takeover

This is why we need `send_to_leader()` method in the SO101 manager - to send policy actions to the leader arms during policy execution.

## Troubleshooting

### Common Issues

#### Issue 1: SO101 Device Not Found

```bash
# Check device permissions
ls -la /dev/ttyACM*

# Add user to dialout group (if needed)
sudo usermod -aG dialout $USER

# Check USB connection
lsusb | grep -i robot
```

#### Issue 2: Calibration Mismatch

```bash
# Recalibrate SO101 devices
python scripts/calibrate_so101.py \
    --left_port /dev/ttyACM0 \
    --right_port /dev/ttyACM1 \
    --save_dir ~/.cache/so101_calibrations/
```

#### Issue 3: Action Limits Exceeded

```python
# Check calibration range
# SO101 outputs in degrees, IsaacLab expects radians
# Conversion: radians = degrees * π / 180
```

#### Issue 4: Lag in Response

```python
# Optimize SO101 reading frequency
# Add caching if needed
# Use non-blocking reads
```

## Success Criteria

- [ ] SO101 devices connect and calibrate successfully
- [ ] Actions read from SO101 leader arms are accurate
- [ ] Keyboard toggles between POLICY and HUMAN modes smoothly
- [ ] HUMAN mode uses SO101 leader actions (validated)
- [ ] POLICY mode uses policy actions (baseline)
- [ ] **Policy sync to teleop works**: Leader arms move with follower during POLICY mode
- [ ] **Seamless takeover**: Human can grab leader arms during POLICY mode without sudden jumps
- [ ] Dataset records correct `is_intervention` flags
- [ ] Dataset records `collector_policy_id="human"` during intervention
- [ ] Dataset records `policy_action` field correctly (what policy wanted)
- [ ] Emergency stop works correctly
- [ ] No action limit violations

## Files to Create

1. `scripts/devices/__init__.py`
2. `scripts/devices/so101_manager.py` - SO101 device management
3. `scripts/test_so101_connection.py` - Connection test
4. `scripts/test_so101_reading.py` - Action reading test
5. `scripts/test_hil_so101.py` - Integration test

## Files to Modify

1. `scripts/utils/parser.py` - Add SO101 arguments
2. `scripts/utils/evaluation.py` - Integrate SO101 manager
3. `scripts/utils/hil_keyboard.py` - No changes needed

## Estimated Timeline

- **Step 1** (Device Manager): 2-3 hours
- **Step 2** (HIL Integration): 2-3 hours
- **Step 3** (Arguments): 30 minutes
- **Step 4** (Safety): 1-2 hours
- **Step 5** (Testing): 2-3 hours

**Total**: ~8-12 hours of development

## References

- Evo-RL `lerobot-human-inloop-record` command
- LeRobot SO101 teleoperator implementation
- IsaacLab device integration examples
- SO-ARM100 documentation: https://github.com/TheRobotStudio/SO-ARM100

---

## Plan Corrections Summary

This plan was corrected after reviewing the actual Evo-RL codebase. The following critical changes were made:

### Original Plan Issues (Fixed)

| Issue | Original | Corrected |
|-------|----------|-----------|
| **Import classes** | `RobotDevice`, `SO101Cfg` | `BiSOLeader`, `SOLeaderConfig` |
| **Send method** | `send_action()` | `send_feedback()` |
| **Calibration check** | `_check_calibration_files()` | Handled by `BiSOLeader.connect()` |
| **Torque control** | Not addressed | Added `enable_manual_control()` |
| **Units** | Ambiguous | Explicit degrees/radians conversion |

### Key Implementation Changes

1. **Use Evo-RL's BiSOLeader class** - This is the actual teleoperator class in Evo-RL
2. **Call `send_feedback()` not `send_action()`** - This is the actual method name
3. **Handle torque mode switching** - Critical for reading from leader arms
4. **Convert degrees ↔ radians** - SO101 uses degrees, IsaacLab uses radians
5. **Call `enable_manual_control()` before reading** - Required to disable torque

These corrections ensure the implementation will work with the actual Evo-RL codebase.

See `docs/so101-plan-review.md` for detailed analysis of the issues found.
