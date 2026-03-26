"""HIL Intervention Manager - coordinates keyboard, leader device, and torque control for HIL intervention.

This uses the stateful toggle pattern.
"""

import numpy as np
from typing import Dict, Any, Optional

from lehome.assets.robots.lerobot import (
    SO101_FOLLOWER_MOTOR_LIMITS,
    SO101_FOLLOWER_USD_JOINT_LIMLITS,
)
from lehome.utils.logger import get_logger

logger = get_logger(__name__)


class HILInterventionManager:
    """Manages HIL intervention state and coordinates between keyboard, leader device, and torque control.

    Mode Transitions:
        - POLICY -> HUMAN: Torque OFF, human controls leader
        - HUMAN -> POLICY: Torque ON, policy controls leader

    Key Responsibilities:
        1. Track intervention mode state
        2. Handle mode transitions (toggle torque)
        3. Get action from appropriate source (policy vs leader)
        4. Convert leader input into sim-compatible format
        5. Send feedback to leader (only in POLICY mode)
    """

    def __init__(self, leader_device, is_bimanual: bool):
        """Initialize the HIL intervention manager.

        Args:
            leader_device: The leader device (SO101Leader or BiSO101Leader)
            is_bimanual: Whether using dual-arm setup
        """
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
            logger.warning(f"Failed to get leader action: {e}")
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

            # For gripper: motor values 0-100 map directly to joint_range (-10 to 100 degrees)
            if name == "gripper":
                # Linear mapping: 0 -> -10 degrees, 100 -> 100 degrees
                degrees = motor_val * (joint_range[1] - joint_range[0]) / 100.0 + joint_range[0]
            else:
                # For other joints, use the standard formula
                degrees = (motor_val - motor_range[0]) / (motor_range[1] - motor_range[0]) * \
                         (joint_range[1] - joint_range[0]) + joint_range[0]

            # Convert degrees -> radians
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
