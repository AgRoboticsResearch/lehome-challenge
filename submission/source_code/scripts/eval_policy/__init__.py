"""
LeHome Challenge Policy Module (Submission)

This module provides the base policy interface and MoE-SmolVLA implementation
for the LeHome Challenge evaluation framework.
"""

from .base_policy import BasePolicy
from .registry import PolicyRegistry

# Import policy implementations (this will auto-register them)
from .lerobot_policy import LeRobotPolicy
from .example_participant_policy import CustomPolicy
from .moe_smolvla_policy import MoESmolVLAPolicy

__all__ = [
    "BasePolicy",
    "PolicyRegistry",
    "LeRobotPolicy",
    "CustomPolicy",
    "MoESmolVLAPolicy",
]
