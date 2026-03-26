# HIL Keyboard Intervention with Torque Control

## Context

Implement Human-in-the-Loop (HIL) intervention for the evaluation system. When the policy makes a mistake, a human operator can press 'i' key to take control, move the leader arms to correct the action, then press 'i' again to return control to the policy.

**Key requirement**: Proper torque control during mode transitions.
- **POLICY mode**: Torque ON (policy drives leader arms via `send_feedback`)
- **HUMAN mode**: Torque OFF (human can freely move leader arms)

---

## Current State

| Component | Status | Notes |
|-----------|--------|-------|
| `HILKeyboardHandler` | ✅ Done | Monitors 'i' key, provides `is_intervention_active()` |
| `SO101Leader.set_manual_control()` | ✅ Done | Toggles torque on/off |
| `BiSO101Leader.set_manual_control()` | ❌ Missing | Need to add |
| Evaluation loop mode switching | ❌ Missing | Currently only logs, doesn't switch control |
| Leader → Sim action during intervention | 📝 Documented | Uses `convert_action_from_so101_leader()` (READ path) |
| SmolVLA → Sim → Leader flow | 📝 Documented | See "POLICY MODE" section below |

---

## Implementation Plan

### Step 1: Add `set_manual_control()` to BiSO101Leader

**File**: `source/lehome/lehome/devices/lerobot/bi_so101_leader.py`

Add after line 82 (after `send_feedback` method):

```python
def set_manual_control(self, enabled: bool) -> None:
    """Toggle manual/policy control mode for both arms.

    Args:
        enabled: If True, disable torque (human moves leader).
                If False, enable torque (policy takes control).
    """
    self.left_so101_leader.set_manual_control(enabled)
    self.right_so101_leader.set_manual_control(enabled)
```

---

### Step 2: Create HIL Intervention Manager

**File**: `scripts/utils/hil_intervention.py` (NEW FILE)

A class that manages HIL state and coordinates between keyboard, leader device, and torque control.

Key responsibilities:
1. Track intervention mode state
2. Handle mode transitions (toggle torque)
3. Get action from appropriate source (policy vs leader)
4. Convert leader input to sim-compatible format
5. Send feedback to leader (only in policy mode)

---

### Step 3: Update Evaluation Loop

**File**: `scripts/utils/evaluation.py`

**Changes needed:**

1. **Import HIL manager** (around line 30):
   ```python
   from scripts.utils.hil_intervention import HILInterventionManager
   ```

2. **Create HIL manager** (after leader device setup, around line 58):
   ```python
   hil_manager = None
   if getattr(args, 'enable_hil', False) and leader_device is not None:
       hil_manager = HILInterventionManager(
           leader_device=leader_device,
           is_bimanual=('Bi' in args.task),
       )
       logger.info(f"HIL intervention manager created for {'bimanual' if 'Bi' in args.task else 'single arm'}")
   ```

3. **Handle mode transitions** (after line 242):
   ```python
   if hil_handler and hil_handler.is_intervention_toggled():
       is_intervention = hil_handler.is_intervention_active()
       if hil_manager:
           hil_manager.set_intervention_mode(is_intervention)
           mode = "HUMAN" if is_intervention else "POLICY"
           logger.info(f"[HIL] Mode switched to: {mode}")
       hil_handler.reset_toggle()
   ```

4. **Get action from appropriate source** (replace lines 244-256):
   ```python
   if hil_manager is not None:
       action_np = hil_manager.get_action(action_np, hil_handler)
   ```

5. **Update feedback logic** (replace lines 287-296):
   ```python
   if hil_manager is not None:
       hil_manager.send_feedback(action_np, verbose=verbose)
   elif leader_device is not None:
       # Legacy policy-sync mode (non-HIL)
       action_np = action.cpu().numpy().squeeze()
       try:
           leader_device.send_feedback(action_np, verbose=verbose)
       except Exception as e:
           logger.warning(f"Leader feedback failed: {e}")
           leader_device = None
   ```

---

### Step 4: Add CLI Arguments

**File**: `scripts/eval.py`

Add HIL-related arguments:
```python
parser.add_argument("--enable_hil", action="store_true",
                    help="Enable Human-in-the-Loop intervention mode")
parser.add_argument("--teleop_device", type=str, default=None,
                    choices=["so101leader", "bi-so101leader"],
                    help="Teleoperation device for HIL (required if --enable_hil)")
parser.add_argument("--left_arm_port", type=str, default="/dev/ttyACM0")
parser.add_argument("--right_arm_port", type=str, default="/dev/ttyACM1")
parser.add_argument("--recalibrate", action="store_true")
```

---

## Mode Transition Flow

```
POLICY MODE                          HUMAN MODE
───────────                          ───────────
Torque: ON                           Torque: OFF
Policy → Sim → Leader                Leader → Sim
send_feedback() active               send_feedback() skipped

         │ Press 'i'                    │ Press 'i'
         └──────────────────────────────┘
              set_manual_control(True/False)
```

---

## Action Flow: POLICY vs HUMAN Mode

### Complete Pipeline Comparison

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    ACTION FLOW: POLICY vs HUMAN MODE                            │
└─────────────────────────────────────────────────────────────────────────────────┘

  ┌─── POLICY MODE (Torque ON) ───────────────────────────────────────────────────┐
  │                                                                               │
  │  SmolVLA → Unnormalize → Rad→Deg → Joint→Motor → sync_write → Leader         │
  │     │                                                       │                │
  │     └─── action (radians) → env.step() → Sim moves ◄─────────┘                │
  │                                                                               │
  │  Uses: send_feedback() (WRITE path)                                          │
  └───────────────────────────────────────────────────────────────────────────────┘

  ┌─── HUMAN MODE (Torque OFF) ───────────────────────────────────────────────────┐
  │                                                                               │
  │  Human moves Leader → sync_read → Motor→Joint→Rad → env.step() → Sim moves   │
  │                              │                                                │
  │                              └── action (radians)                             │
  │                                                                               │
  │  Uses: convert_action_from_so101_leader() (READ path)                         │
  └───────────────────────────────────────────────────────────────────────────────┘
```

---

## POLICY MODE: SmolVLA → Sim Arm Pipeline

This section details how SmolVLA policy actions are applied to the simulation arm.

### Full Pipeline (SmolVLA → Sim → Leader)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    POLICY MODE: SmolVLA → SIM → LEADER                          │
└─────────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────┐
                    │   SmolVLA Policy     │
                    │  (Neural Network)    │
                    └──────────┬───────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │   Raw Policy Output            │
              │   Shape: [action_horizon, 12]  │
              │   Range: ~[-1, 1] (normalized) │
              └────────────────┬───────────────┘
                               │
          ┌────────────────────┴────────────────────┐
          │                                         │
          ▼                                         ▼
  ┌─────────────────────┐              ┌─────────────────────┐
  │   TO SIMULATION     │              │   TO LEADER ARM     │
  │   (Direct)          │              │   (Feedback)        │
  └──────────┬──────────┘              └──────────┬──────────┘
             │                                    │
             ▼                                    ▼
  ┌─────────────────────┐              ┌─────────────────────┐
  │  Unnormalize        │              │  Unnormalize        │
  │  (q01, q99)         │              │  (q01, q99)         │
  │                     │              │                     │
  │  norm → radians     │              │  norm → radians     │
  └──────────┬──────────┘              └──────────┬──────────┘
             │                                    │
             ▼                                    ▼
  ┌─────────────────────┐              ┌─────────────────────┐
  │  env.step(action)   │              │  Rad → Deg          │
  │                     │              │  × 180/π            │
  │  action in RADIANS  │              └──────────┬──────────┘
  │  goes directly!     │                         │
  └──────────┬──────────┘                         ▼
             │                         ┌─────────────────────┐
             ▼                         │  Joint→Motor        │
  ┌─────────────────────┐              │  Mapping            │
  │  Sim Arm Moves      │              │  USD limits →       │
  │  ✅                 │              │  Motor limits       │
  └─────────────────────┘              └──────────┬──────────┘
                                                  │
                                                  ▼
                                       ┌─────────────────────┐
                                       │  Clamp +            │
                                       │  sync_write()       │
                                       └──────────┬──────────┘
                                                  │
                                                  ▼
                                       ┌─────────────────────┐
                                       │  Leader Arm Moves   │
                                       │  ✅                 │
                                       └─────────────────────┘
```

### Key Insight: Sim Uses Radians Directly

The simulation accepts actions in **radians** directly - no conversion needed:

```python
# In evaluation.py
action = policy.select_action(observation)  # Returns normalized
# action is already in RADIANS after unnormalization by policy
env.step(action)  # Sim accepts radians directly!
```

The policy's `UnnormalizerProcessorStep` converts:
- `[-1, 1]` (normalized) → `[q01, q99]` (radians)

The sim's `JointPositionActionCfg` expects radians, so they match perfectly.

### Numerical Example: SmolVLA → Sim

```
SmolVLA output:        0.5      (normalized)
                         │
                         ▼
Unnormalize:       (0.5 + 1) × (1.0 - (-1.2)) / 2 + (-1.2)
                    = 0.45 rad
                         │
                         ▼
env.step(0.45)     →  Sim joint moves to 0.45 rad = 25.8°
```

---

## HUMAN MODE: Leader → Sim Pipeline

When human takes control, the READ path is used:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    HUMAN MODE: LEADER → SIM                                     │
└─────────────────────────────────────────────────────────────────────────────────┘

  Human moves Leader Arm
         │
         ▼
  ┌─────────────────────┐
  │  sync_read()        │
  │  "Present_Position" │
  │                     │
  │  Returns: motor     │
  │  values (-100,100)  │
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  convert_action_from_so101_leader()                         │
  │                                                             │
  │  Motor → Joint mapping:                                     │
  │  joint = (motor - M_min)/(M_max - M_min) × (J_max-J_min)    │
  │        + J_min                                              │
  │                                                             │
  │  Then: deg → rad                                            │
  └────────────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
  ┌─────────────────────┐
  │  action in RADIANS  │
  │  (same format as    │
  │   policy output)    │
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │  env.step(action)   │
  │  Sim arm follows    │
  │  human movement ✅  │
  └─────────────────────┘
```

### Numerical Example: Leader → Sim

```
Human moves leader to motor value: 50
                         │
                         ▼
Motor → Joint:     (50 - (-100)) / (100 - (-100)) × (110 - (-110)) + (-110)
                    = 55°
                         │
                         ▼
Deg → Rad:         55 × π/180 = 0.96 rad
                         │
                         ▼
env.step(0.96)     →  Sim joint moves to 0.96 rad = 55°
```

---

## Summary: READ vs WRITE Paths

| Aspect | READ Path (Human Mode) | WRITE Path (Policy Mode) |
|--------|------------------------|--------------------------|
| Function | `convert_action_from_so101_leader` | `send_feedback` |
| Direction | Leader → Sim | Sim → Leader |
| Input | Motor values | Radians (policy) |
| Output | Radians (for sim) | Motor values |
| Formula | `(m-M_min)/(M_max-M_min) × (J_max-J_min) + J_min` | `(j-J_min)/(J_max-J_min) × (M_max-M_min) + M_min` |
| Torque | OFF | ON |
| Used in | HUMAN mode | POLICY mode |

---

## Detailed Code Changes

### 1. `source/lehome/lehome/devices/lerobot/bi_so101_leader.py`
**Add**: `set_manual_control()` method (~5 lines)

```python
def set_manual_control(self, enabled: bool) -> None:
    """Toggle manual/policy control mode for both arms.

    Args:
        enabled: If True, disable torque (human moves leader).
                If False, enable torque (policy takes control).
    """
    self.left_so101_leader.set_manual_control(enabled)
    self.right_so101_leader.set_manual_control(enabled)
```

---

### 2. `scripts/utils/hil_intervention.py` (NEW FILE)
**Add**: HIL manager class (~80 lines)

```python
"""HIL Intervention Manager - coordinates keyboard, leader, and torque control."""

import numpy as np
from typing import Dict, Any, Optional
from lehome.devices.lerobot import SO101_FOLLOWER_MOTOR_LIMITS, SO101_FOLLOWER_USD_JOINT_LIMLITS


class HILInterventionManager:
    """Manages HIL intervention state and coordinates between keyboard, leader device, and torque control.

    Key responsibilities:
    1. Track intervention mode state
    2. Handle mode transitions (toggle torque)
    3. Get action from appropriate source (policy vs leader)
    4. Convert leader input into sim-compatible format
    5. Send feedback to leader (only in POLICY mode)
    """

    def __init__(self, leader_device, is_bimanual: bool):
        self.leader_device = leader_device
        self.is_bimanual = is_bimanual
        self._is_intervention = False
        self._first_transition = True

    def set_intervention_mode(self, is_intervention: bool) -> None:
        """Switch between POLICY and HUMAN mode.

        Handles torque control automatically.
        """
        if is_intervention != self._is_intervention:
            self._is_intervention = is_intervention
            self.leader_device.set_manual_control(is_intervention)

            if self._first_transition:
                self._first_transition = False

    def get_action(self, policy_action: np.ndarray, hil_handler) -> np.ndarray:
        """Get action from appropriate source.

        Args:
            policy_action: Action from policy (always computed)
            hil_handler: HIL keyboard handler

        Returns:
            Action in RADIANS (numpy array)
        """
        if self._is_intervention:
            return self._get_leader_action()
        return policy_action

    def _get_leader_action(self) -> np.ndarray:
        """Get action from leader device and convert to sim format.

        Returns:
            Action in RADIANS
        """
        if self.leader_device is None:
            return np.zeros(12 if self.is_bimanual else 6, dtype=np.float32)

        try:
            # Get raw action from leader
            leader_action = self.leader_device.input2action()

            # Check if leader is started
            if not leader_action.get('started', False):
                return np.zeros(12 if self.is_bimanual else 6, dtype=np.float32)

            # Check for reset
            if leader_action.get('reset', False):
                return np.zeros(12 if self.is_bimanual else 6, dtype=np.float32)

            # Convert to sim format
            return self._convert_leader_to_sim_action(leader_action)

        except Exception as e:
            return np.zeros(12 if self.is_bimanual else 6, dtype=np.float32)

    def _convert_leader_to_sim_action(self, leader_action: Dict) -> np.ndarray:
        """Convert leader action dictionary to sim action array.

        The leader device returns a dictionary with:
        - 'joint_state': Dict[joint_name, normalized value (-100 to 100)]
        - 'motor_limits': Dict[joint_name, (min, max)]

        The sim expects:
        - numpy array of joint positions in radians
        """
        if self.is_bimanual:
            # Dual-arm: expect bi_so101_leader format
            joint_state = leader_action.get('joint_state', {})
            motor_limits = leader_action.get('motor_limits', {})

            if not joint_state or not motor_limits:
                return np.zeros(12, dtype=np.float32)

            left_state = joint_state.get('left_arm', {})
            right_state = joint_state.get('right_arm', {})
            left_limits = motor_limits.get('left_arm', {})
            right_limits = motor_limits.get('right_arm', {})

            # Convert each arm
            left_action = self._convert_single_arm(left_state, left_limits)
            right_action = self._convert_single_arm(right_state, right_limits)

            return np.concatenate([left_action, right_action])
        else:
            # Single-arm: expect so101_leader format
            joint_state = leader_action.get('joint_state', {})
            motor_limits = leader_action.get('motor_limits', {})

            return self._convert_single_arm(joint_state, motor_limits)

    def _convert_single_arm(self, joint_state: Dict, motor_limits: Dict) -> np.ndarray:
        """Convert single arm joint state to radians.

        Uses the same conversion as convert_action_from_so101_leader.
        """
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                       "wrist_flex", "wrist_roll", "gripper"]

        action_rad = np.zeros(6, dtype=np.float32)

        for i, name in enumerate(joint_names):
            motor_val = joint_state.get(name, 0.0)
            motor_range = motor_limits.get(name, SO101_FOLLOWER_MOTOR_LIMITS[name])
            joint_range = SO101_FOLLOWER_USD_JOINT_LIMLITS[name]

            # Convert motor value → degrees
            degrees = (motor_val - motor_range[0]) / (motor_range[1] - motor_range[0]) * \
                         (joint_range[1] - joint_range[0]) + joint_range[0]

            # Convert degrees → radians
            action_rad[i] = degrees * np.pi / 180.0

        return action_rad

    def send_feedback(self, action: np.ndarray, verbose: bool = False) -> None:
        """Send feedback to leader (only in POLICY mode).

        Args:
            action: Action in RADIANS
            verbose: Print debug info
        """
        if not self._is_intervention:
            self.leader_device.send_feedback(action, verbose)
```

---

### 3. `scripts/utils/evaluation.py`
**Modify**: 3 locations (~30 lines total)

**Location 1 - Import** (around line 30):
```python
from scripts.utils.hil_intervention import HILInterventionManager
```

**Location 2 - Create HIL manager** (after leader device setup, around line 58):
```python
hil_manager = None
if getattr(args, 'enable_hil', False) and leader_device is not None:
    hil_manager = HILInterventionManager(
        leader_device=leader_device,
        is_bimanual=('Bi' in args.task),
    )
    logger.info(f"HIL intervention manager created for {'bimanual' if 'Bi' in args.task else 'single arm'}")
```

**Location 3 - Handle mode transitions** (after line 242):
```python
if hil_handler and hil_handler.is_intervention_toggled():
    is_intervention = hil_handler.is_intervention_active()
    if hil_manager:
        hil_manager.set_intervention_mode(is_intervention)
        mode = "HUMAN" if is_intervention else "POLICY"
        logger.info(f"[HIL] Mode switched to: {mode}")
    hil_handler.reset_toggle()
```

**Location 4 - Get action from appropriate source** (replace lines 244-256):
```python
if hil_manager is not None:
    action_np = hil_manager.get_action(action_np, hil_handler)
```

**Location 5 - Update feedback logic** (replace lines 287-296):
```python
if hil_manager is not None:
    hil_manager.send_feedback(action_np, verbose=verbose)
elif leader_device is not None:
    # Legacy policy-sync mode (non-HIL)
    action_np = action.cpu().numpy().squeeze()
    try:
        leader_device.send_feedback(action_np, verbose=verbose)
    except Exception as e:
        logger.warning(f"Leader feedback failed: {e}")
        leader_device = None
```

---

### 4. `scripts/eval.py`
**Add**: CLI arguments (~5 lines)

```python
parser.add_argument(
    "--enable_hil",
    action="store_true",
    help="Enable Human-in-the-Loop intervention mode"
)
```

---

## Summary

| File | Action | Lines |
|------|--------|-------|
| `bi_so101_leader.py` | Add method | ~5 |
| `hil_intervention.py` | NEW file | ~80 |
| `evaluation.py` | Modify 3 spots | ~30 |
| `eval.py` | Add CLI arg | ~5 |
| **Total** | | **~120 lines** |

---

## Verification

1. **Test with hardware**:
   ```bash
   python -m scripts.eval \
       --policy_type lerobot \
       --policy_path outputs/train/act_top_long/checkpoints/last/pretrained_model \
       --enable_hil \
       --teleop_device bi-so101leader \
       --left_arm_port /dev/ttyACM0 \
       --right_arm_port /dev/ttyACM1 \
       --num_episodes 1
   ```

2. **Expected behavior**:
   - Policy runs, leader arms follow policy (torque ON)
   - Press 'i': Torque turns OFF, human can move leaders
   - Human movements control sim arms
   - Press 'i': Torque turns ON, policy resumes control
