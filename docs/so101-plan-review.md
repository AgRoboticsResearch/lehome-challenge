# SO101 Integration Plan Review

## Summary

The plan is well-structured and correctly identifies the key components needed for SO101 integration. However, after reviewing the actual Evo-RL codebase, I found several **critical issues** that need to be addressed before implementation.

## Critical Issues Found

### 1. Wrong Import Paths and Classes ❌

**Plan uses**:
```python
from lerobot.common.robot_devices.robots.config import SO101Cfg
from lerobot.common.robot_devices.robots.utils import RobotDevice
```

**Actual implementation uses**:
```python
from lerobot.teleoperators.bi_so_leader import BiSOLeader
from lerobot.teleoperators.so_leader import SOLeaderTeleopConfig
```

**Impact**: The code won't work. These classes don't exist in Evo-RL.

---

### 2. Wrong Method Name for Sending Actions ❌

**Plan uses**:
```python
self.left_leader.send_action(left_dict)
self.right_leader.send_action(right_dict)
```

**Actual method is**:
```python
self.left_leader.send_feedback(left_dict)
self.right_leader.send_feedback(right_dict)
```

**Impact**: AttributeError at runtime.

---

### 3. Missing Torque Control Mode Understanding ⚠️

**Critical behavior not documented**:
- `get_action()` requires **manual control mode** (torque disabled)
- `send_feedback()` calls `set_manual_control(False)` which **enables torque**

**This means**:
- During POLICY mode: Leader arms are in torque-enabled mode (following policy)
- During HUMAN mode: Leader arms must be in manual control mode (reading human input)

**The plan doesn't address this mode switching!**

---

### 4. Missing Calibration Check Implementation ⚠️

**Plan references**:
```python
if calibrate or not self._check_calibration_files():
```

**Issue**: `_check_calibration_files()` method is not implemented.

**Actual behavior**: The `SOLeader.connect()` method handles calibration automatically:
```python
if not self.is_calibrated and calibrate:
    logger.info("Mismatch between calibration values...")
    self.calibrate()
```

---

### 5. Action Format is Correct ✅

The plan's action format matches the actual implementation:
```python
{
    "left_shoulder_pan.pos": float,
    "left_shoulder_lift.pos": float,
    # ... etc
}
```

---

## Recommended Implementation Changes

### Updated SO101 Device Manager

```python
import numpy as np
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

JOINT_ORDER = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper"
]

class SO101BimanualManager:
    """Manage bimanual SO101 leader arms for HIL evaluation with policy_sync_to_teleop."""

    def __init__(self,
                 leader_ports: tuple[str, str],
                 calibration_dir: Optional[Path] = None,
                 policy_sync_to_teleop: bool = True):
        """
        Args:
            leader_ports: (left_port, right_port) for SO101 leader arms
            calibration_dir: Directory for calibration files
            policy_sync_to_teleop: If True, send policy actions to leader during POLICY mode
        """
        self.leader_ports = leader_ports
        self.calibration_dir = calibration_dir
        self.policy_sync_to_teleop = policy_sync_to_teleop

        # Leader arms (using actual Evo-RL classes)
        self.teleop: Optional['BiSOLeader'] = None

        # Track mode for torque control
        self._torque_enabled = False  # True = POLICY mode, False = HUMAN mode

    def connect(self, calibrate: bool = False):
        """Connect to SO101 leader devices."""
        from lerobot.teleoperators.bi_so_leader import BiSOLeader
        from lerobot.teleoperators.so_leader import SOLeaderTeleopConfig, SOLeaderConfig

        # Create configuration for each arm
        left_config = SOLeaderConfig(
            port=self.leader_ports[0],
            use_degrees=True  # SO101 uses degrees by default
        )

        right_config = SOLeaderConfig(
            port=self.leader_ports[1],
            use_degrees=True
        )

        # Create bimanual configuration
        from lerobot.teleoperators.config import TeleoperatorConfig
        teleop_config = TeleoperatorConfig(
            type="bi_so_leader",
            id="hil_leader",
            calibration_dir=str(self.calibration_dir) if self.calibration_dir else None,
            left_arm_config=left_config,
            right_arm_config=right_config,
        )

        # Create teleoperator
        self.teleop = BiSOLeader(teleop_config)
        self.teleop.connect(calibrate=calibrate)

        logger.info("SO101 bimanual leader arms connected")

        # Start in manual control mode (HUMAN mode)
        self.teleop.set_manual_control(True)
        self._torque_enabled = False

    def get_leader_action(self) -> np.ndarray:
        """Read current action from SO101 leader arms.

        Note: This requires manual control mode (torque disabled).
        """
        # Read action dict from teleoperator
        action_dict = self.teleop.get_action()

        # Convert to numpy array [left(6), right(6)]
        action = np.array([
            action_dict[f"left_{joint}.pos"] for joint in JOINT_ORDER
        ] + [
            action_dict[f"right_{joint}.pos"] for joint in JOINT_ORDER
        ])

        return action

    def send_to_leader(self, action: np.ndarray) -> bool:
        """Send action to SO101 leader arms (for policy_sync_to_teleop).

        Note: This enables torque mode automatically via send_feedback().

        Args:
            action: (12,) array [left_arm(6), right_arm(6)] in degrees

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.policy_sync_to_teleop:
            return True  # Skip if disabled

        try:
            # Convert numpy array to action dict format
            action_dict = {}
            for i, joint in enumerate(JOINT_ORDER):
                action_dict[f"left_{joint}.pos"] = float(action[i])
            for i, joint in enumerate(JOINT_ORDER):
                action_dict[f"right_{joint}.pos"] = float(action[6 + i])

            # Send to leader (this enables torque automatically)
            self.teleop.send_feedback(action_dict)
            self._torque_enabled = True

            return True
        except Exception as e:
            logger.error(f"Failed to send action to leader: {e}")
            return False

    def enable_manual_control(self):
        """Enable manual control mode for reading human input."""
        if self.teleop and self._torque_enabled:
            self.teleop.set_manual_control(True)
            self._torque_enabled = False

    def disconnect(self):
        """Disconnect from SO101 devices."""
        if self.teleop:
            self.teleop.disconnect()

    def emergency_stop(self):
        """Emergency stop all SO101 devices."""
        if self.teleop:
            # Disable torque to stop movement
            self.teleop.set_manual_control(True)
            self._torque_enabled = False
```

### Updated Action Selection Logic

```python
# 3. Policy Inference / SO101 Reading
if hil_handler and hil_handler.is_intervention_active() and so101_manager:
    # HUMAN MODE: Read from SO101 leader
    # CRITICAL: Enable manual control mode before reading
    so101_manager.enable_manual_control()

    action_np = so101_manager.get_leader_action()

    # Convert degrees to radians (IsaacLab expects radians)
    action_np = np.deg2rad(action_np)

    # Validate action (safety limits)
    action_np = validate_action(action_np)

    # Track what policy would have done
    policy_action_for_dataset = policy.select_action(observation_dict)
    last_policy_action = policy_action_for_dataset.copy()

    is_intervention = True

else:
    # POLICY MODE: Use policy inference with sync to leader
    policy_action_for_dataset = policy.select_action(observation_dict)

    # IsaacLab uses radians, SO101 uses degrees
    action_degrees = np.rad2deg(policy_action_for_dataset)
    action_np = policy_action_for_dataset.copy()
    last_policy_action = action_np.copy()
    is_intervention = False

    # CRITICAL: Send policy action to leader (converts to degrees internally)
    # This enables torque automatically via send_feedback()
    if so101_manager and args.policy_sync_to_teleop:
        so101_manager.send_to_leader(action_degrees)

# 4. Execute action in IsaacLab (follower moves)
obs, reward, terminated, truncated, info = env.step(action_np)
```

---

## Additional Recommendations

### 1. Add Calibration Directory Argument

```python
parser.add_argument("--so101_calibration_dir", type=str, default=None,
                   help="Directory for SO101 calibration files (default: ~/.cache/lerobot/calibrations/)")
```

### 2. Add Degrees/Radians Conversion Helper

```python
def validate_action(action: np.ndarray) -> np.ndarray:
    """Validate and clip action to safe joint limits."""
    # IsaacLab action space is in radians
    ACTION_LIMITS_RAD = {
        "shoulder_pan": (-3.14, 3.14),
        "shoulder_lift": (-1.57, 1.57),
        "elbow_flex": (-2.53, 2.53),
        "wrist_flex": (-1.57, 1.57),
        "wrist_roll": (-3.14, 3.14),
        "gripper": (0.0, 1.0),
    }

    limits = [ACTION_LIMITS_RAD[j] for j in JOINT_ORDER for _ in range(2)]
    min_vals = [l[0] for l in limits]
    max_vals = [l[1] for l in limits]

    return np.clip(action, min_vals, max_vals)
```

### 3. Testing Priority

Given these issues, I recommend:

1. **Phase 0**: Create minimal SO101 connection test using actual Evo-RL classes
2. **Phase 1**: Test reading from leader arms in manual control mode
3. **Phase 2**: Test sending actions to leader arms (torque mode)
4. **Phase 3**: Test mode switching (manual ↔ torque)
5. **Phase 4**: Full integration with HIL evaluation

---

## Conclusion

The plan's architecture and data flow are correct, but the implementation details need significant updates to match the actual Evo-RL codebase. The main issues are:

1. Use `BiSOLeader` instead of `RobotDevice`
2. Use `send_feedback()` instead of `send_action()`
3. Handle torque control mode switching
4. Proper degrees/radians conversion

Would you like me to update the integration plan with these corrections before proceeding with implementation?
