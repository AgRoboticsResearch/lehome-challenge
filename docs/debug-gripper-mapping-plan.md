# Debug Script Plan: SO101 Leader Arm → Sim Motor Data Mapping

## Goal

Create a debug script that prints the complete data flow from SO101 leader arm to sim motor radians during teleoperation (no simulation required).

## Data Flow to Trace

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  DATA FLOW: SO101 Leader Arm → Sim                                                  │
└─────────────────────────────────────────────────────────────────────────────────────┘

  Hardware           FeetechMotorsBus        SO101Leader          convert_action
  (Motor)            (_normalize)           (input2action)       (motor → sim)
  ──────────         ─────────────          ────────────         ──────────────────
     │                    │                     │                    │
     │  Raw (0-4095)      │                     │                    │
     │                    ▼                     │                    │
     │              Normalized Value           │                    │
     │              ─────────────────          │                    │
     │              RANGE_M100_100:            │                    │
     │                -100 to +100              │                    │
     │              RANGE_0_100:               │                    │
     │                0 to 100 (gripper)        │                    │
     │                                         │                    │
     │                                         ▼                    │
     │                                   motor_limits              │
     │                                   ─────────────             │
     │                                   gripper: (0, 100)         │
     │                                   others: (-100, 100)       │
     │                                                          │
     │                                                          ▼
     │                                                    Degrees (°)
     │                                                          │
     │                                                          ▼
     │                                                    Radians (rad)
     │                                                          │
     └──────────────────────────────────────────────────────────────────────────────┘
```

## Key Data Points to Print

For each joint (especially gripper):

| Stage | Field | Source |
|-------|-------|--------|
| 1. Raw | `raw_value` (0-4095) | `sync_read("Present_Position", normalize=False)` |
| 2. Normalized | `norm_value` | `sync_read("Present_Position", normalize=True)` |
| 3. Motor Limits | `motor_range` | `SO101_FOLLOWER_MOTOR_LIMITS[name]` |
| 4. Joint Limits | `joint_range` | `SO101_FOLLOWER_USD_JOINT_LIMLITS[name]` |
| 5. Degrees | `degrees` | Formula: `(norm - motor_min)/(motor_max-motor_min) * (joint_max-joint_min) + joint_min` |
| 6. Radians | `radians` | `degrees * π / 180` |

## Implementation Plan

### File: `scripts/debug_motor_mapping.py`

```python
#!/usr/bin/env python
"""
Debug script for SO101 Leader Arm motor data mapping.

Prints the complete data flow from raw motor values to sim motor radians.
No simulation required - reads directly from hardware.

Usage:
    python scripts/debug_motor_mapping.py --port /dev/ttyACM0
"""

import argparse
import time
import numpy as np

from lehome.devices.lerobot import SO101Leader
from lehome.assets.robots.lerobot import (
    SO101_FOLLOWER_MOTOR_LIMITS,
    SO101_FOLLOWER_USD_JOINT_LIMLITS,
)


def print_header():
    """Print script header."""
    print("=" * 100)
    print("SO101 Leader Arm Motor Mapping Debug Tool")
    print("=" * 100)
    print("\nThis script traces the complete data flow:")
    print("  Raw (0-4095) → Normalized → Motor Limits → Degrees → Radians")
    print("\nPress Ctrl+C to exit\n")
    print("=" * 100)


def print_joint_data(joint_name, raw_val, norm_val, motor_range, joint_range, degrees, radians):
    """Print formatted data for a single joint."""
    # Highlight gripper with different color indicator
    marker = "🔵" if joint_name == "gripper" else "  "

    print(f"{marker} {joint_name:15} | {raw_val:6.0f} | {norm_val:8.2f} | "
          f"({motor_range[0]:6.1f}, {motor_range[1]:6.1f}) | "
          f"({joint_range[0]:6.1f}, {joint_range[1]:6.1f}) | "
          f"{degrees:8.2f} | {radians:8.4f}")


def main():
    parser = argparse.ArgumentParser(description="Debug SO101 motor mapping")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="Serial port")
    parser.add_argument("--interval", type=float, default=0.1, help="Print interval (seconds)")
    parser.add_argument("--recalibrate", action="store_true", help="Force recalibration")
    args = parser.parse_args()

    print_header()

    # Create dummy env for SO101Leader
    class DummyEnv:
        pass

    # Create leader device
    print(f"\nConnecting to SO101 Leader on {args.port}...")
    leader = SO101Leader(
        env=DummyEnv(),
        port=args.port,
        recalibrate=args.recalibrate,
    )

    print("✅ Connected!\n")

    # Joint names in order
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                   "wrist_flex", "wrist_roll", "gripper"]

    try:
        while True:
            # Read raw values (without normalization)
            raw_values = leader._bus.sync_read("Present_Position", normalize=False)

            # Read normalized values
            norm_values = leader._bus.sync_read("Present_Position", normalize=True)

            # Print table header
            print("\n" + "-" * 100)
            print(f"{'Joint':15} | {'Raw':>6} | {'Normalized':>8} | "
                  f"{'Motor Range':>15} | {'Joint Range (°)':>15} | "
                  f"{'Degrees':>8} | {'Radians':>8}")
            print("-" * 100)

            # Process each joint
            for name in joint_names:
                raw_val = raw_values.get(name, 0)
                norm_val = norm_values.get(name, 0)
                motor_range = SO101_FOLLOWER_MOTOR_LIMITS[name]
                joint_range = SO101_FOLLOWER_USD_JOINT_LIMLITS[name]

                # Convert: normalized → degrees → radians
                # Formula from convert_action_from_so101_leader()
                degrees = (norm_val - motor_range[0]) / (motor_range[1] - motor_range[0]) * \
                          (joint_range[1] - joint_range[0]) + joint_range[0]
                radians = degrees * np.pi / 180.0

                print_joint_data(name, raw_val, norm_val, motor_range, joint_range, degrees, radians)

            print("-" * 100)
            print("🔵 = Gripper (special handling: motor range 0-100, joint range -10° to 100°)")

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\nExiting...")
    finally:
        leader.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
```

## Verification

1. Run the script with SO101 leader arm connected:
   ```bash
   python scripts/debug_motor_mapping.py --port /dev/ttyACM0
   ```

2. Move the leader arm and observe:
   - Raw values change (0-4095)
   - Normalized values change (gripper: 0-100, others: -100 to 100)
   - Final radians should match expected sim values

3. Specifically test gripper:
   - Close gripper → should see radians near -0.175 rad (-10°)
   - Open gripper → should see radians near 1.745 rad (100°)

## Files to Create

| File | Description |
|------|-------------|
| `scripts/debug_motor_mapping.py` | New debug script |

## Summary

- **New file**: `scripts/debug_motor_mapping.py` (~80 lines)
- **Purpose**: Trace motor data from hardware to sim radians
- **No simulation required**: Reads directly from SO101 leader hardware
- **Output**: Real-time table showing all conversion stages
