#!/usr/bin/env python
"""
Debug script for SO101 Leader Arm motor data mapping.

Prints the complete data flow from raw motor values to sim motor radians.
No simulation required - reads directly from hardware.

Usage:
    # Single arm
    python scripts/debug_motor_mapping.py --port /dev/ttyACM0

    # Dual arms (left + right)
    python scripts/debug_motor_mapping.py --left-port /dev/ttyACM0 --right-port /dev/ttyACM1
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
    print("=" * 120)
    print("SO101 Leader Arm Motor Mapping Debug Tool")
    print("=" * 120)
    print("\nThis script traces the complete data flow:")
    print("  Raw (0-4095) → Normalized → Motor Limits → Degrees → Radians")
    print("\nPress Ctrl+C to exit\n")
    print("=" * 120)


def print_joint_data(joint_name, raw_val, norm_val, motor_range, joint_range, degrees, radians, is_gripper=False):
    """Print formatted data for a single joint."""
    marker = "[GRIPPER]" if is_gripper else "         "
    print(f"{marker} {joint_name:15} | {raw_val:6.0f} | {norm_val:8.2f} | "
          f"({motor_range[0]:6.1f}, {motor_range[1]:6.1f}) | "
          f"({joint_range[0]:6.1f}, {joint_range[1]:6.1f}) | "
          f"{degrees:8.2f} | {radians:8.4f}")


def process_arm(leader, arm_name, joint_names):
    """Read and process data for a single arm."""
    raw_values = leader._bus.sync_read("Present_Position", normalize=False)
    norm_values = leader._bus.sync_read("Present_Position", normalize=True)

    print(f"\n{'='*55} {arm_name} {'='*55}")
    print(f"{'Joint':15} | {'Raw':>6} | {'Normalized':>8} | "
          f"{'Motor Range':>15} | {'Joint Range (°)':>15} | "
          f"{'Degrees':>8} | {'Radians':>8}")
    print("-" * 120)

    for name in joint_names:
        raw_val = raw_values.get(name, 0)
        norm_val = norm_values.get(name, 0)
        motor_range = SO101_FOLLOWER_MOTOR_LIMITS[name]
        joint_range = SO101_FOLLOWER_USD_JOINT_LIMLITS[name]

        # Convert: normalized → degrees → radians
        degrees = (norm_val - motor_range[0]) / (motor_range[1] - motor_range[0]) * \
                  (joint_range[1] - joint_range[0]) + joint_range[0]
        radians = degrees * np.pi / 180.0

        print_joint_data(name, raw_val, norm_val, motor_range, joint_range, degrees, radians, is_gripper=(name == "gripper"))

    print("-" * 120)


class DummyEnv:
    pass


def main():
    parser = argparse.ArgumentParser(description="Debug SO101 motor mapping")
    parser.add_argument("--port", type=str, default=None, help="Single arm serial port")
    parser.add_argument("--left-port", type=str, default="/dev/ttyACM0", help="Left arm serial port")
    parser.add_argument("--right-port", type=str, default="/dev/ttyACM1", help="Right arm serial port")
    parser.add_argument("--interval", type=float, default=0.1, help="Print interval (seconds)")
    parser.add_argument("--recalibrate", action="store_true", help="Force recalibration")
    parser.add_argument("--dual", action="store_true", help="Use dual arm mode")
    args = parser.parse_args()

    print_header()

    # Determine mode
    if args.port:
        # Single arm mode
        dual_mode = False
        left_port = args.port
        right_port = None
    elif args.dual:
        # Explicit dual mode
        dual_mode = True
        left_port = args.left_port
        right_port = args.right_port
    else:
        # Default to dual mode
        dual_mode = True
        left_port = args.left_port
        right_port = args.right_port

    # Joint names in order
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                   "wrist_flex", "wrist_roll", "gripper"]

    # Connect to arm(s)
    leaders = {}
    try:
        print(f"\nConnecting to Left SO101 Leader on {left_port}...")
        leaders["left"] = SO101Leader(
            env=DummyEnv(),
            port=left_port,
            recalibrate=args.recalibrate,
            calibration_filename="left_so101_leader.json",
        )
        print("Left arm connected!")

        if dual_mode:
            print(f"\nConnecting to Right SO101 Leader on {right_port}...")
            leaders["right"] = SO101Leader(
                env=DummyEnv(),
                port=right_port,
                recalibrate=args.recalibrate,
                calibration_filename="right_so101_leader.json",
            )
            print("Right arm connected!")

        print("\nStarting motor mapping display...\n")

        while True:
            # Process left arm
            process_arm(leaders["left"], "LEFT ARM", joint_names)

            # Process right arm if dual mode
            if dual_mode and "right" in leaders:
                process_arm(leaders["right"], "RIGHT ARM", joint_names)

            print("\n[GRIPPER] = Gripper (motor range 0-100, joint range -10° to 100°)")
            print("Press Ctrl+C to exit")

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\nExiting...")
    finally:
        for name, leader in leaders.items():
            try:
                leader.disconnect()
                print(f"{name.capitalize()} arm disconnected.")
            except Exception as e:
                print(f"Error disconnecting {name} arm: {e}")


if __name__ == "__main__":
    main()
