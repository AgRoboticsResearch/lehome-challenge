from collections.abc import Callable
import numpy as np

from .so101_leader import SO101Leader
from ..device_base import Device


class BiSO101Leader(Device):
    def __init__(self, env, left_port: str = '/dev/ttyACM0', right_port: str = '/dev/ttyACM1', recalibrate: bool = False):
        super().__init__(env)

        # use left so101 leader as the main device to store state
        print("Connecting to left_so101_leader...")
        self.left_so101_leader = SO101Leader(env, left_port, recalibrate, "left_so101_leader.json")
        print("Connecting to right_so101_leader...")
        self.right_so101_leader = SO101Leader(env, right_port, recalibrate, "right_so101_leader.json")

        self.right_so101_leader.listener.stop()

    def __str__(self) -> str:
        """Returns: A string containing the information of bi-so101 leader."""
        msg = "Bi-SO101-Leader device for SE(3) control.\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tMove Bi-SO101-Leader to control Bi-SO101-Follower\n"
        msg += "\tIf SO101-Follower can't synchronize with Bi-SO101-Leader, please add --recalibrate and rerun to recalibrate Bi-SO101-Leader.\n"
        return msg

    def add_callback(self, key: str, func: Callable):
        self.left_so101_leader.add_callback(key, func)
        self.right_so101_leader.add_callback(key, lambda: None)

    def reset(self):
        self.left_so101_leader.reset()
        self.right_so101_leader.reset()

    def get_device_state(self):
        return {
            "left_arm": self.left_so101_leader.get_device_state(),
            "right_arm": self.right_so101_leader.get_device_state()
        }

    def input2action(self):
        state = {}
        reset = state["reset"] = self.left_so101_leader.reset_state
        state['started'] = self.left_so101_leader.started
        if reset:
            self.left_so101_leader.reset_state = False
            return state
        state['joint_state'] = self.get_device_state()

        ac_dict = {}
        ac_dict["reset"] = reset
        ac_dict['started'] = self.left_so101_leader.started
        ac_dict['bi_so101_leader'] = True
        if reset:
            return ac_dict
        ac_dict['joint_state'] = state['joint_state']
        ac_dict['motor_limits'] = {
            'left_arm': self.left_so101_leader.motor_limits,
            'right_arm': self.right_so101_leader.motor_limits
        }
        return ac_dict

    def send_feedback(self, action: np.ndarray, verbose: bool = False) -> None:
        """Send policy action to both leader arms.

        Args:
            action: Joint positions in RADIANS, shape (12,)
                     First 6 elements = left arm, last 6 = right arm
            verbose: If True, print conversion details for debugging
        """
        # Split action for left and right arms
        left_action = action[:6]
        right_action = action[6:]

        # Send to both leaders (each handles its own conversion)
        if verbose:
            print("Left arm:")
        self.left_so101_leader.send_feedback(left_action, verbose=verbose)
        if verbose:
            print("Right arm:")
        self.right_so101_leader.send_feedback(right_action, verbose=verbose)

    def set_manual_control(self, enabled: bool) -> None:
        """Toggle manual/policy control mode for both arms.

        Args:
            enabled: If True, disable torque (human moves leader).
                    If False, enable torque (policy takes control).
        """
        self.left_so101_leader.set_manual_control(enabled)
        self.right_so101_leader.set_manual_control(enabled)
