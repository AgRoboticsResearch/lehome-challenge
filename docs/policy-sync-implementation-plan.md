# Policy → Sim + Leader Sync: Implementation Plan

## Context

Make policy actions move BOTH sim arms and physical SO101 leader arms simultaneously.

**Key insight:** We mirror the existing read path (Leader → Sim) in reverse.

---

## Existing Read Path (Leader → Sim)

Reference: [`dataset_record.py:330`](dataset_record.py#L330), [`action_process.py:127`](action_process.py#L127)

```
get_device_state()
    ↓ sync_read("Present_Position")
    ↓ Returns: {"shoulder_pan": -50.0, ...}  (normalized motor values)
convert_action_from_so101_leader()
    ↓ Motor values (-100 to 100) → Degrees → Radians
    ↓ Returns: Tensor
env.step(action)
    ↓ Sim moves ✅
```

---

## New Write Path (Policy → Leader) - The Reverse

```
policy.select_action()
    ↓ Returns: [-1.5, 0.8, ...]  (RADIANS, numpy)
send_feedback(action)
    ↓ _convert_rad_to_motor()  [INVERSE of above function]
    ↓ RADIANS → Degrees → Motor values (-100 to 100)
    ↓ Returns: {"shoulder_pan": -50.0, ...}
sync_write("Goal_Position", motor_values)
    ↓ Leader moves ✅
```

---

## Files to Modify

### 1. `source/lehome/lehome/devices/lerobot/so101_leader.py`

**Add imports (at top with other imports):**
```python
from lehome.assets.robots.lerobot import (
    SO101_FOLLOWER_MOTOR_LIMITS,
    SO101_FOLLOWER_USD_JOINT_LIMLITS,  # ← ADD THIS
)
import numpy as np  # ← ADD THIS if not present
```

**Add method (after `configure()` method, around line 192):**

```python
def send_feedback(self, action: np.ndarray) -> None:
    """Send policy action to leader motors (inverse of reading from leader).

    This mirrors the read path:
    - READ:  sync_read("Present_Position") → convert_action_from_so101_leader() → sim
    - WRITE: policy → _convert_rad_to_motor() → sync_write("Goal_Position") → hardware

    Args:
        action: Joint positions in RADIANS (same format as policy outputs), shape (6,)
    """
    # Convert RADIANS → motor normalized values (INVERSE of convert_action_from_so101_leader)
    motor_values = {}
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                   "wrist_flex", "wrist_roll", "gripper"]

    for i, name in enumerate(joint_names):
        # Step 1: RADIANS → Degrees
        degrees = action[i] * 180.0 / np.pi

        # Step 2: Map from joint range (degrees) → motor range (normalized)
        # This is the INVERSE of the conversion in convert_action_from_so101_leader()
        joint_range = SO101_FOLLOWER_USD_JOINT_LIMLITS[name]   # e.g., (-110°, 110°)
        motor_range = self._motor_limits[name]                  # e.g., (-100, 100)

        # Formula: map degrees from joint range to motor range
        motor_val = (
            (degrees - joint_range[0]) / (joint_range[1] - joint_range[0])
            * (motor_range[1] - motor_range[0]) + motor_range[0]
        )

        # Step 3: Clip to valid motor range and store
        motor_values[name] = float(np.clip(motor_val, *motor_range))

    # Send to hardware (enable torque first to take control from human)
    self._bus.enable_torque()
    self._bus.sync_write("Goal_Position", motor_values)
```

**Lines added:** ~35 lines

---

### 2. `source/lehome/lehome/devices/lerobot/bi_so101_leader.py`

**Add method (at end of class):**

```python
def send_feedback(self, action: np.ndarray) -> None:
    """Send policy action to both leader arms.

    Args:
        action: Joint positions in RADIANS, shape (12,)
                 First 6 elements = left arm, last 6 = right arm
    """
    # Split action for left and right arms
    left_action = action[:6]
    right_action = action[6:]

    # Send to both leaders (each handles its own conversion)
    self.left_so101_leader.send_feedback(left_action)
    self.right_so101_leader.send_feedback(right_action)
```

**Lines added:** ~10 lines

---

### 3. `scripts/utils/evaluation.py`

**Step 1:** Add import (at top with other imports, around line 30):
```python
from scripts.utils.dataset_record import create_teleop_interface
```

**Step 2:** Add leader device setup (in `run_evaluation_loop()`, after HIL handler setup, around line 53):
```python
# Create leader device if policy sync is enabled
leader_device = None
if getattr(args, 'enable_policy_sync', False):
    # Check if teleop device type is compatible
    if hasattr(args, 'teleop_device') and args.teleop_device in ['bi-so101leader', 'so101leader']:
        try:
            leader_device = create_teleop_interface(env, args)
            logger.info(f"✅ Leader device created for policy sync: {args.teleop_device}")
        except Exception as e:
            logger.error(f"❌ Failed to create leader device: {e}")
            logger.warning("Continuing with sim-only execution (no leader feedback)")
            leader_device = None
    else:
        logger.warning(f"⚠️  Policy sync requires SO101 leader device")
        logger.warning(f"   Current device: {getattr(args, 'teleop_device', None)}")
        logger.warning("   Supported: bi-so101leader, so101leader")
```

**Step 3:** Add leader feedback call (after `env.step(action)` at line 266):
```python
# 6. Step Environment
env.step(action)

# 7. Send action to leader for tactile feedback (if enabled)
if leader_device is not None:
    action_np = action.cpu().numpy().squeeze()
    leader_device.send_feedback(action_np)
```

**Lines added:** ~20 lines

---

### 4. `scripts/utils/parser.py`

**Add argument (in `setup_eval_parser()` function with other eval args):**
```python
parser.add_argument(
    "--enable_policy_sync",
    action="store_true",
    help="Enable policy synchronization to physical leader arms (leader moves with policy for tactile feedback)"
)
```

**Lines added:** ~5 lines

---

## Summary

| File | Lines | Purpose |
|------|-------|---------|
| `so101_leader.py` | ~35 | Add `send_feedback()` method (RADIANS → motor values via `_convert_rad_to_motor()`) |
| `bi_so101_leader.py` | ~10 | Add `send_feedback()` for dual-arm |
| `evaluation.py` | ~20 | Create leader device and call send_feedback() |
| `parser.py` | ~5 | Add CLI flag |
| **Total** | **~70 lines** | |

---

## Testing

### Test with hardware:
```bash
python -m scripts.eval \
    --policy_type lerobot \
    --policy_path outputs/train/act_top_long/checkpoints/last/pretrained_model \
    --dataset_root Datasets/example/top_long_merged \
    --garment_type top_long \
    --enable_policy_sync \
    --teleop_device bi-so101leader \
    --left_arm_port /dev/ttyACM0 \
    --right_arm_port /dev/ttyACM1 \
    --num_episodes 1
```

**Expected result:**
- Sim arms move with policy ✅
- Leader arms move in sync with policy ✅
- Human feels policy actions through leader ✅

---

## Implementation Notes

1. **Mirrors existing pattern:** Uses inverse of `convert_action_from_so101_leader()` conversion
2. **Same API format:** `send_feedback()` accepts radians (same as policy output, same as sim input)
3. **Simple integration:** Just call `send_feedback()` after `env.step()` in eval loop
4. **Uses existing infrastructure:** `sync_write()`, `enable_torque()`, motor limits already exist

---

## References

- Read path: [`so101_leader.py:127-145`](source/lehome/lehome/devices/lerobot/so101_leader.py#L127-L145) - `get_device_state()` and `input2action()`
- Conversion: [`action_process.py:127-146`](source/lehome/lehome/devices/action_process.py#L127-L146) - `convert_action_from_so101_leader()`
- Motor API: [`motors_bus.py:1147`](source/lehome/lehome/devices/lerobot/common/motors/motors_bus.py#L1147) - `sync_write()` signature
- Device factory: [`dataset_record.py:58-89`](scripts/utils/dataset_record.py#L58-L89) - `create_teleop_interface()`
- Evo-RL reference: [`so_leader.py:137-171`](third_party/Evo-RL/src/lerobot/teleoperators/so_leader/so_leader.py#L137-L171) - `set_manual_control()` and `send_feedback()`

---

## Implementation Notes (Updated Post-Implementation)

### Torque Enable Pattern (Evo-RL Style)
The implementation uses Evo-RL's stateful toggle pattern instead of direct `enable_torque()` calls:

```python
# In __init__:
self._manual_control_enabled = True  # Initially True = human can move leader

# In configure():
self._bus.disable_torque()
self._manual_control_enabled = True  # Track state

def set_manual_control(self, enabled: bool) -> None:
    """Toggle manual/policy control mode."""
    if enabled:
        if not self._manual_control_enabled:
            self._bus.disable_torque()
            self._manual_control_enabled = True
    else:
        if self._manual_control_enabled:
            self._bus.enable_torque()
            self._manual_control_enabled = False

def send_feedback(self, action, verbose=False):
    self.set_manual_control(False)  # Only enables torque on first call!
    self._bus.sync_write("Goal_Position", motor_values)
```

**Benefits:**
- Only calls `enable_torque()` once (on first transition to policy control)
- No redundant bus operations every step
- State tracking prevents unnecessary errors

---

## Complete Action Pipeline: SmolVLA → SO101

This section documents the full pipeline from policy output to hardware command, including normalization, unit conversion, and range mapping.

### Pipeline Overview

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE ACTION PIPELINE (SmolVLA → SO101)                    │
└──────────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
  │  SmolVLA    │    │ Unnormalize │    │  Rad → Deg  │    │ Joint→Motor │
  │  Policy     │───▶│ (q01, q99)  │───▶│  ×180/π     │───▶│   Mapping   │
  │             │    │             │    │             │    │             │
  │ [-1, 1]     │    │ [q01, q99]  │    │ [deg range] │    │ [-100,100]  │
  │ normalized  │    │ radians     │    │ degrees     │    │ motor vals  │
  └─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘
                                                                   │
                              ┌────────────────────────────────────┘
                              │
                              ▼
  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
  │   Leader    │    │ sync_write  │    │   Motor     │    │   SO101     │
  │   Arm       │◀───│Goal_Position│◀───│   Ticks     │◀───│   Clamp     │
  │   Moves     │    │             │    │ ×4096/360   │    │             │
  │             │    │             │    │             │    │             │
  │  hardware   │    │  protocol   │    │  encoder    │    │  safety     │
  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### Stage Details

| Stage | Input | Output | Unit | Description |
|-------|-------|--------|------|-------------|
| 1. SmolVLA Policy | Observation | `[-1, 1]` | normalized | Neural network output |
| 2. Unnormalize | Normalized | `[q01, q99]` | radians | Dataset quantile denormalization |
| 3. Rad → Deg | Radians | Degrees | degrees | `deg = rad × 180/π` |
| 4. Joint→Motor | Joint deg | Motor val | motor norm | Linear mapping between ranges |
| 5. Clamp | Motor val | Clamped | motor norm | Enforce hardware limits |
| 6. Motor Ticks | Motor deg | Ticks | encoder | `ticks = deg × 4096/360` |

### Why Joint→Motor Mapping Exists

The simulation and hardware use different joint limits:

```
┌─────────────────────┐          ┌─────────────────────┐
│   SIMULATION        │          │     HARDWARE        │
│   (USD Limits)      │          │   (Motor Limits)    │
├─────────────────────┤          ├─────────────────────┤
│ shoulder_pan: ±110° │  ──map──▶│ shoulder_pan: ±100  │
│ shoulder_lift: ±100°│  ──map──▶│ shoulder_lift: ±100 │
│ elbow_flex: -100,90°│  ──map──▶│ elbow_flex:  ±100   │  ← DIFFERENT!
│ wrist_flex:  ±95°   │  ──map──▶│ wrist_flex:  ±100   │
│ wrist_roll:  ±160°  │  ──map──▶│ wrist_roll:  ±160   │
│ gripper: -10,100°   │  ──map──▶│ gripper:     0,100  │  ← DIFFERENT!
└─────────────────────┘          └─────────────────────┘
```

The mapping ensures:
- Sim position 0° → Motor position 0 (center preserved)
- Sim max (110°) → Motor max (100)
- Sim min (-110°) → Motor min (-100)

---

## Mathematical Verification: READ and WRITE are Inverses

### Code Comparison

**READ Path** (`convert_action_from_so101_leader` in `action_process.py:139-141`):
```python
# Motor → Joint → Radians (for sim)
processed_degree = (joint_state[joint_name] - motor_limit_range[0]) / (
    motor_limit_range[1] - motor_limit_range[0]
) * (joint_limit_range[1] - joint_limit_range[0]) + joint_limit_range[0]
```

**WRITE Path** (`send_feedback` in `so101_leader.py`):
```python
# Radians → Joint → Motor (for hardware)
motor_val = (
    (degrees - joint_range[0]) / (joint_range[1] - joint_range[0])
    * (motor_range[1] - motor_range[0]) + motor_range[0]
)
```

### Proof of Inverse

```
READ (Motor → Joint):
┌─────────────────────────────────────────────────────────────────────────────┐
│  joint = (motor - M_min) / (M_max - M_min) × (J_max - J_min) + J_min        │
└─────────────────────────────────────────────────────────────────────────────┘

WRITE (Joint → Motor):
┌─────────────────────────────────────────────────────────────────────────────┐
│  motor = (joint - J_min) / (J_max - J_min) × (M_max - M_min) + M_min        │
└─────────────────────────────────────────────────────────────────────────────┘

Prove: WRITE(READ(x)) = x

Step 1: Apply READ to motor value 'm'
        joint = (m - M_min) / (M_max - M_min) × (J_max - J_min) + J_min

Step 2: Apply WRITE to result
        new_motor = (joint - J_min) / (J_max - J_min) × (M_max - M_min) + M_min

Step 3: Substitute and simplify
        new_motor = [(m - M_min) / (M_max - M_min)] × (M_max - M_min) + M_min
        new_motor = (m - M_min) + M_min = m  ✓
```

---

## Numerical Examples

### Example 1: shoulder_pan (standard joint)

```
SmolVLA output:        0.5      (normalized, in [-1,1])
                         │
                         ▼
Dataset stats:        q01 = -1.2 rad,  q99 = 1.0 rad
                         │
                         ▼
Denormalize:     (0.5 + 1) × (1.0 - (-1.2)) / 2 + (-1.2)
                = 1.5 × 2.2 / 2 - 1.2
                = 0.45 rad         (in radians)
                         │
                         ▼
Rad → Deg:       0.45 × 180/π = 25.8°
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│  JOINT → MOTOR MAPPING                                         │
│                                                                │
│  joint_range = (-110°, 110°)  [USD limits]                     │
│  motor_range = (-100, 100)    [HW limits]                      │
│                                                                │
│  motor_val = (25.8 - (-110)) / (110 - (-110))                  │
│             × (100 - (-100)) + (-100)                          │
│  = 135.8 / 220 × 200 - 100                                     │
│  = 23.4          (motor normalized value)                      │
└────────────────────────────────────────────────────────────────┘
                         │
                         ▼
Clamp:           23.4 is within [-100, 100] ✓
                         │
                         ▼
Motor ticks:     23.4 × (4096/360) ≈ 266 ticks
                         │
                         ▼
SO101 Leader receives Goal_Position = 266 ✅
```

### Example 2: gripper (asymmetric range)

```
SmolVLA output:        -0.8     (normalized)
                         │
                         ▼
Dataset stats:        q01 = 0.0 rad,  q99 = 0.6 rad
                         │
                         ▼
Denormalize:     (-0.8 + 1) × (0.6 - 0.0) / 2 + 0.0
                = 0.06 rad = 3.4°
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│  JOINT → MOTOR MAPPING for gripper                             │
│                                                                │
│  joint_range = (-10°, 100°)  [USD limits]                      │
│  motor_range = (0, 100)      [HW limits]                       │
│                                                                │
│  motor_val = (3.4 - (-10)) / (100 - (-10)) × (100 - 0) + 0     │
│  = 13.4 / 110 × 100                                            │
│  = 12.2          (motor value for gripper)                     │
└────────────────────────────────────────────────────────────────┘
                         │
                         ▼
Gripper at 12.2 (slightly open position)
```

### Example 3: Round-trip Verification

```
START: Motor at position 50

READ (Motor → Sim):
  joint_deg = (50 - (-100)) / (100 - (-100)) × (110 - (-110)) + (-110)
            = 150/200 × 220 - 110 = 55°
  joint_rad = 55 × π/180 = 0.96 rad → sim receives this

WRITE (Sim → Motor):
  degrees = 0.96 × 180/π = 55°
  motor = (55 - (-110)) / (110 - (-110)) × (100 - (-100)) + (-100)
        = 165/220 × 200 - 100 = 50 ✓

RESULT: Round-trip preserves original value! (50 → 55° → 50)
```

---

## Summary Table

| Aspect | READ (`convert_action_from_so101_leader`) | WRITE (`send_feedback`) |
|--------|-------------------------------------------|-------------------------|
| Direction | Motor → Sim | Sim → Motor |
| Input | Motor normalized value | Radians (policy output) |
| Output | Radians (for sim) | Motor normalized value |
| Formula | `(m-M_min)/(M_max-M_min) × (J_max-J_min) + J_min` | `(j-J_min)/(J_max-J_min) × (M_max-M_min) + M_min` |
| Unit Conv | deg → rad at end | rad → deg at start |
| **Inverse** | ✓ Yes | ✓ Yes |
