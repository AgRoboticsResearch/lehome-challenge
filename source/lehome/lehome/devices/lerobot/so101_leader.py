import os
import json
from collections.abc import Callable
from typing import Dict, Tuple
from pynput.keyboard import Listener, Key

from .common.motors import (
    FeetechMotorsBus,
    Motor,
    MotorNormMode,
    MotorCalibration,
    OperatingMode,
)
from .common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from ..device_base import Device

from lehome.assets.robots.lerobot import (
    SO101_FOLLOWER_MOTOR_LIMITS,
    SO101_FOLLOWER_USD_JOINT_LIMLITS,
)
import numpy as np


class SO101Leader(Device):
    """A SO101 Leader device for SE(3) control."""

    def __init__(
        self,
        env,
        port: str = "/dev/ttyACM0",
        recalibrate: bool = False,
        calibration_file_name: str = "so101_leader.json",
    ):
        super().__init__(env)
        self.port = port

        # calibration
        self.calibration_path = os.path.join(
            os.path.dirname(__file__), ".cache", calibration_file_name
        )
        if not os.path.exists(self.calibration_path) or recalibrate:
            self.calibrate()
        calibration = self._load_calibration()

        self._bus = FeetechMotorsBus(
            port=self.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
                "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
                "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=calibration,
        )
        self._motor_limits = SO101_FOLLOWER_MOTOR_LIMITS

        # connect
        self.connect()

        # some flags and callbacks
        self._started = False
        self._reset_state = False
        self._additional_callbacks = {}

        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        self._display_controls()
        self.b_disable = False
        self.other_key_enable = False
        self._manual_control_enabled = True  # Initially True = human can move leader

    def __str__(self) -> str:
        """Returns: A string containing the information of so101 leader."""
        msg = "SO101-Leader device for SE(3) control.\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tMove SO101-Leader to control SO101-Follower\n"
        msg += "\tIf SO101-Follower can't synchronize with SO101-Leader, please add --recalibrate and rerun to recalibrate SO101-Leader.\n"
        return msg

    def _display_controls(self):
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (30 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("b", "start simulation")
        print_command("s", "start record")
        print_command("d", "delete the episode")
        print_command("n", "save the episode")
        print_command("ESC", "abort recording and clear buffer")
        print_command("move leader", "control follower in the simulation")
        print_command("Control+C", "quit")
        print("")

    def on_press(self, key):
        pass

    def on_release(self, key):
        """
        Key handler for key releases.
        Args:
            key (str): key that was pressed
        """
        try:
            if key.char == "b":
                if self.b_disable == False:
                    self._started = True
                    self._reset_state = False
                    self.other_key_enable = True
            elif key.char == "s":
                if self.other_key_enable == True:
                    self.b_disable = True
                    if "S" in self._additional_callbacks:
                        self._additional_callbacks["S"]()
            elif key.char == "n":
                if self.other_key_enable == True:
                    if "N" in self._additional_callbacks:
                        self._additional_callbacks["N"]()
            elif key.char == "d":
                if self.other_key_enable == True:
                    if "D" in self._additional_callbacks:
                        self._additional_callbacks["D"]()
        except AttributeError:
            # Handle special keys (like ESC)
            if key == Key.esc and "ESCAPE" in self._additional_callbacks:
                if self.other_key_enable == True:
                    self._additional_callbacks["ESCAPE"]()

    def get_device_state(self):
        return self._bus.sync_read("Present_Position")

    def input2action(self):
        state = {}
        reset = state["reset"] = self._reset_state
        state["started"] = self._started
        if reset:
            self._reset_state = False
            return state
        state["joint_state"] = self.get_device_state()
        ac_dict = {}
        ac_dict["reset"] = reset
        ac_dict["started"] = self._started
        ac_dict["so101_leader"] = True
        if reset:
            return ac_dict
        ac_dict["joint_state"] = state["joint_state"]
        ac_dict["motor_limits"] = self._motor_limits
        return ac_dict

    def reset(self):
        pass

    def add_callback(self, key: str, func: Callable):
        self._additional_callbacks[key] = func

    @property
    def started(self) -> bool:
        return self._started

    @property
    def reset_state(self) -> bool:
        return self._reset_state

    @reset_state.setter
    def reset_state(self, reset_state: bool):
        self._reset_state = reset_state

    @property
    def motor_limits(self) -> Dict[str, Tuple[float, float]]:
        return self._motor_limits

    @property
    def is_connected(self) -> bool:
        return self._bus.is_connected

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError("SO101-Leader is not connected.")
        self._bus.disconnect()
        print("SO101-Leader disconnected.")

    def connect(self):
        if self.is_connected:
            raise DeviceAlreadyConnectedError("SO101-Leader is already connected.")
        self._bus.connect()
        self.configure()
        print("SO101-Leader connected.")

    def configure(self) -> None:
        self._bus.disable_torque()
        self._manual_control_enabled = True  # Track that manual control is enabled
        self._bus.configure_motors()
        for motor in self._bus.motors:
            self._bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    def set_manual_control(self, enabled: bool) -> None:
        """Toggle manual/policy control mode.

        Args:
            enabled: If True, disable torque (human moves leader).
                    If False, enable torque (policy takes control).
        """
        if enabled:
            # Enable manual control (human moves leader)
            if not self._manual_control_enabled:
                try:
                    self._bus.disable_torque()
                    self._manual_control_enabled = True
                except Exception as e:
                    # Fallback: disable motors individually with retry
                    print(f"[WARN] Failed to disable torque bulk: {e}")
                    print("[INFO] Trying individual motor disable...")
                    success = True
                    for motor_name in self._bus.motors:
                        for retry in range(3):
                            try:
                                self._bus.write("Torque_Enable", motor_name, 0)
                                break
                            except Exception as e2:
                                if retry == 2:
                                    print(f"[WARN] Failed to disable {motor_name}: {e2}")
                                    success = False
                                else:
                                    import time
                                    time.sleep(0.1)
                    # Consider it disabled even if some motors failed
                    self._manual_control_enabled = True
        else:
            # Enable policy control (policy commands leader)
            if self._manual_control_enabled:
                try:
                    self._bus.enable_torque()
                    self._manual_control_enabled = False
                except Exception as e:
                    # Fallback: enable motors individually
                    print(f"[WARN] Failed to enable torque bulk: {e}")
                    print("[INFO] Trying individual motor enable...")
                    for motor_name in self._bus.motors:
                        try:
                            self._bus.write("Torque_Enable", motor_name, 1)
                        except Exception as e2:
                            print(f"[WARN] Failed to enable {motor_name}: {e2}")
                    # Consider it enabled even if some motors failed
                    self._manual_control_enabled = False

    def send_feedback(self, action: np.ndarray, verbose: bool = False) -> None:
        """Send policy action to leader motors (inverse of reading from leader).

        This mirrors the read path:
        - READ:  sync_read("Present_Position") → _normalize() → convert_action_from_so101_leader() → sim
        - WRITE: policy → send_feedback() → sync_write("Goal_Position") → hardware

        Uses calibration data to match the READ path, ensuring commands stay within
        the actual usable range of your specific hardware.

        Args:
            action: Joint positions in RADIANS (same format as policy outputs), shape (6,)
            verbose: If True, print conversion details for debugging
        """
        motor_values = {}
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                       "wrist_flex", "wrist_roll", "gripper"]

        for i, name in enumerate(joint_names):
            # Step 1: RADIANS → Degrees
            degrees = action[i] * 180.0 / np.pi

            # Step 2: Convert degrees to normalized motor value (using theoretical limits first)
            joint_range = SO101_FOLLOWER_USD_JOINT_LIMLITS[name]   # USD joint limits (degrees)
            motor_range = self._motor_limits[name]                  # Theoretical motor limits (normalized)

            # This is the INVERSE of convert_action_from_so101_leader()
            motor_val_theoretical = (
                (degrees - joint_range[0]) / (joint_range[1] - joint_range[0])
                * (motor_range[1] - motor_range[0]) + motor_range[0]
            )

            # Step 3: Clip to CALIBRATED range if available (matches READ behavior)
            if name in self._bus.calibration:
                cal = self._bus.calibration[name]
                motor = self._bus.motors[name]

                # Convert calibrated raw range to normalized (same as _normalize() does)
                if motor.norm_mode.name == "RANGE_0_100":
                    # Gripper: 0 to 100 scale
                    # Raw (0-4095) → Normalized (0-100)
                    cal_min_norm = (cal.range_min / 4095.0) * 100.0
                    cal_max_norm = (cal.range_max / 4095.0) * 100.0
                elif motor.norm_mode.name == "RANGE_M100_100":
                    # Other joints: -100 to 100 scale
                    # Raw (0-4095) → Normalized (-100 to 100)
                    # Center is 2048 (half of 4095)
                    cal_min_norm = ((cal.range_min - 2048.0) / 2048.0) * 100.0
                    cal_max_norm = ((cal.range_max - 2048.0) / 2048.0) * 100.0
                else:
                    # Fallback to theoretical if unknown norm mode
                    cal_min_norm = motor_range[0]
                    cal_max_norm = motor_range[1]

                # Clip to calibrated range
                motor_val_clipped = float(np.clip(motor_val_theoretical, cal_min_norm, cal_max_norm))
                motor_values[name] = motor_val_clipped

                if verbose:
                    clipped = motor_val_clipped != motor_val_theoretical
                    print(f"  {name}: {degrees:.1f}° → {motor_val_theoretical:.1f} → clipped to {motor_val_clipped:.1f} (range: {cal_min_norm:.1f} to {cal_max_norm:.1f}) {'✓' if not clipped else 'CLIPPED'}")
            else:
                # No calibration available, use theoretical limits
                motor_values[name] = float(np.clip(motor_val_theoretical, *motor_range))

                if verbose:
                    print(f"  {name}: {degrees:.1f}° → {motor_values[name]:.1f} (no calibration, range: {motor_range})")

        if verbose:
            print(f"Sending to leader: {motor_values}")

        # TEMPORARY: Skip motor 4 (wrist_flex) if it has communication issues
        # motor_values.pop("wrist_flex", None)

        # Switch to policy control mode (enables torque only once, on first call)
        self.set_manual_control(False)

        self._bus.sync_write("Goal_Position", motor_values)

    def calibrate(self):
        self._bus = FeetechMotorsBus(
            port=self.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
                "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
                "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
        )
        self.connect()

        print("\n Running calibration of SO101-Leader")
        self._bus.disable_torque()
        for motor in self._bus.motors:
            self._bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(
            "Move SO101-Leader to the middle of its range of motion and press ENTER..."
        )
        homing_offset = self._bus.set_half_turn_homings()
        print("Move all joints sequentially through their entire ranges of motion.")
        print("Recording positions. Press ENTER to stop...")
        range_mins, range_maxes = self._bus.record_ranges_of_motion()

        calibration = {}
        for motor, m in self._bus.motors.items():
            calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offset[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )
        self._bus.write_calibration(calibration)
        self._save_calibration(calibration)
        print(f"Calibration saved to {self.calibration_path}")

        self.disconnect()

    def _load_calibration(self) -> Dict[str, MotorCalibration]:
        with open(self.calibration_path, "r") as f:
            json_data = json.load(f)
        calibration = {}
        for motor_name, motor_data in json_data.items():
            calibration[motor_name] = MotorCalibration(
                id=int(motor_data["id"]),
                drive_mode=int(motor_data["drive_mode"]),
                homing_offset=int(motor_data["homing_offset"]),
                range_min=int(motor_data["range_min"]),
                range_max=int(motor_data["range_max"]),
            )
        return calibration

    def _save_calibration(self, calibration: Dict[str, MotorCalibration]):
        save_calibration = {
            k: {
                "id": v.id,
                "drive_mode": v.drive_mode,
                "homing_offset": v.homing_offset,
                "range_min": v.range_min,
                "range_max": v.range_max,
            }
            for k, v in calibration.items()
        }
        if not os.path.exists(os.path.dirname(self.calibration_path)):
            os.makedirs(os.path.dirname(self.calibration_path))
        with open(self.calibration_path, "w") as f:
            json.dump(save_calibration, f, indent=4)
