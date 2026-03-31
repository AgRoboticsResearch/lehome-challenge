"""Utility functions for LeHome scripts."""

from . import dataset_inspection
from . import parser

# Note: common, dataset_processing, evaluation, dataset_record and dataset_replay
# are not imported at module level to avoid importing Isaac Sim / heavy dependencies
# before they are needed. Import them lazily in the command handlers.

# Export commonly used functions for convenience
from .parser import (
    setup_record_parser,
    setup_replay_parser,
    setup_inspect_parser,
    setup_read_parser,
    setup_augment_parser,
    setup_merge_parser,
    setup_eval_parser,
)
from .dataset_inspection import inspect, read_states

# Note: common, dataset_processing, evaluation functions are not imported at module
# level to avoid pulling in Isaac Sim or other heavy dependencies.
# Import them lazily when needed:
#   from .common import launch_app, launch_app_from_args, close_app
#   from .dataset_processing import augment_ee_pose, merge_datasets, merge_garment_info
#   from .evaluation import <function>

__all__ = [
    "setup_record_parser",
    "setup_replay_parser",
    "setup_inspect_parser",
    "setup_read_parser",
    "setup_augment_parser",
    "setup_merge_parser",
    "setup_eval_parser",
    "inspect",
    "read_states",
]
