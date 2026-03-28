"""Dataset recording utility functions for teleoperation data collection."""

import argparse
import time
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, Union
import gymnasium as gym
import numpy as np
import torch

from isaacsim.simulation_app import SimulationApp
from isaaclab.envs import DirectRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lehome.devices import (
    Se3Keyboard,
    SO101Leader,
    BiSO101Leader,
    BiKeyboard,
)
from lehome.utils.env_utils import dynamic_reset_gripper_effort_limit_sim
from lehome.utils.record import (
    get_next_experiment_path_with_gap,
    append_episode_initial_pose,
)
from lehome.utils.logger import get_logger
from lehome.utils.success_checker_chanllege import success_checker_garment_fold

from .common import stabilize_garment_after_reset

logger = get_logger(__name__)


def log_success_check_details(result: dict) -> None:
    """Log detailed success check information similar to garment_bi_v2.py.

    Args:
        result: Result dict from success_checker_garment_fold containing:
            - success: bool
            - garment_type: str
            - thresholds: list
            - details: dict of condition info
    """
    if not isinstance(result, dict):
        return

    logger.info(
        f"[Success Check] Garment type: {result.get('garment_type', 'unknown')}, "
        f"Thresholds: {result.get('thresholds', [])}"
    )

    details = result.get("details", {})
    for key, condition_info in details.items():
        status = "✓" if condition_info.get("passed", False) else "✗"
        logger.info(
            f"  {condition_info.get('description', '')} -> {status}"
        )

    success = result.get("success", False)
    logger.info(
        f"[Success Check] Final result: {'Success ✓' if success else 'Failed ✗'}"
    )


def _log_detailed_success_check(env: DirectRLEnv, args: argparse.Namespace) -> None:
    """Perform and log detailed success check during recording.

    This bypasses the step_interval decorator to provide real-time feedback.

    Args:
        env: Environment instance.
        args: Command-line arguments.
    """
    try:
        # Check if environment has the required attributes
        if not hasattr(env, "object") or env.object is None:
            return
        if not hasattr(env, "garment_loader"):
            return
        if not hasattr(env, "cfg") or not hasattr(env.cfg, "garment_name"):
            return

        # Get garment type and run success checker
        garment_type = env.garment_loader.get_garment_type(env.cfg.garment_name)
        result = success_checker_garment_fold(env.object, garment_type)

        # Log detailed results
        log_success_check_details(result)

    except Exception as e:
        logger.debug(f"[Success Check] Error during detailed check: {e}")


def validate_task_and_device(args: argparse.Namespace) -> None:
    """Validate that task name matches the teleop device configuration.

    Args:
        args: Command-line arguments containing task and teleop_device.

    Raises:
        ValueError: If task is not specified.
        AssertionError: If task and device configuration mismatch.
    """
    if args.task is None:
        raise ValueError("Please specify --task.")
    if "Bi" in args.task:
        assert (
            args.teleop_device == "bi-so101leader"
            or args.teleop_device == "bi-keyboard"
        ), "Only support bi-so101leader or bi-keyboard for bi-arm task"
    else:
        assert (
            args.teleop_device == "so101leader" or args.teleop_device == "keyboard"
        ), "Only support so101leader or keyboard for single-arm task"


def create_teleop_interface(
    env: DirectRLEnv, args: argparse.Namespace
) -> Union[Se3Keyboard, SO101Leader, BiSO101Leader, BiKeyboard]:
    """Create teleoperation interface based on device type.

    Args:
        env: Environment instance.
        args: Command-line arguments containing teleop_device and related config.

    Returns:
        Teleoperation interface instance.

    Raises:
        ValueError: If teleop_device is invalid.
    """
    if args.teleop_device == "keyboard":
        return Se3Keyboard(env, sensitivity=0.25 * args.sensitivity)
    if args.teleop_device == "so101leader":
        return SO101Leader(env, port=args.port, recalibrate=args.recalibrate)
    if args.teleop_device == "bi-so101leader":
        return BiSO101Leader(
            env,
            left_port=args.left_arm_port,
            right_port=args.right_arm_port,
            recalibrate=args.recalibrate,
        )
    if args.teleop_device == "bi-keyboard":
        return BiKeyboard(env, sensitivity=0.25 * args.sensitivity)
    raise ValueError(
        f"Invalid device interface '{args.teleop_device}'. "
        f"Supported: 'keyboard', 'so101leader', 'bi-so101leader', 'bi-keyboard'."
    )


def register_teleop_callbacks(
    teleop_interface: Any, recording_enabled: bool = False
) -> Dict[str, bool]:
    """Register callback functions for teleoperation control keys.

    Key bindings:
        S: Start recording
        N: Mark current episode as successful (only active during recording)
        D: Discard current episode and re-record (only active during recording)
        ESC: Abort entire recording process and clear buffer

    Args:
        teleop_interface: Teleoperation interface instance.
        recording_enabled: Whether recording is enabled. If False, N/D keys are
            disabled in idle phase.

    Returns:
        Dictionary of status flags for recording control.
    """
    flags = {
        "start": False,  # S: Start recording
        "success": False,  # N: Success/early termination of current episode
        "remove": False,  # D: Discard current episode
        "abort": False,  # ESC: Abort entire recording process, clear buffer
    }

    def on_start():
        flags["start"] = True
        logger.info("[S] Recording started!")

    def on_success():
        if not recording_enabled or not flags["start"]:
            # Ignore N key in idle phase (before recording starts)
            logger.debug("[N] Ignored (recording not started yet)")
            return
        flags["success"] = True
        logger.info("[N] Mark the current episode as successful.")

    def on_remove():
        if not recording_enabled or not flags["start"]:
            # Ignore D key in idle phase (before recording starts)
            logger.debug("[D] Ignored (recording not started yet)")
            return
        flags["remove"] = True
        logger.info("[D] Discard the current episode and re-record.")

    def on_abort():
        flags["abort"] = True
        logger.warning("[ESC] Abort recording, clearing the current episode buffer...")

    teleop_interface.add_callback("S", on_start)
    teleop_interface.add_callback("N", on_success)
    teleop_interface.add_callback("D", on_remove)
    teleop_interface.add_callback("ESCAPE", on_abort)

    return flags


def create_dataset_if_needed(
    args: argparse.Namespace,
    dataset_index: int = 0,
) -> Tuple[Optional[LeRobotDataset], Optional[Path], Optional[Any], bool]:
    """Create LeRobotDataset if recording is enabled.

    Args:
        args: Command-line arguments containing recording configuration.
        dataset_index: Index of the current dataset (unused, kept for compatibility).

    Returns:
        Tuple of (dataset, json_path, solver, is_bi_arm):
            - dataset: LeRobotDataset instance or None if not recording
            - json_path: Path to object initial pose JSON file or None
            - solver: RobotKinematics solver instance or None
            - is_bi_arm: Boolean indicating if dual-arm configuration

    Raises:
        ValueError: If record_ee_pose is enabled but ee_urdf_path is not provided.
        FileNotFoundError: If URDF file is not found.
    """
    if not args.enable_record:
        return None, None, None, False

    action_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]

    is_bi_arm = ("Bi" in (args.task or "")) or (
        getattr(args, "teleop_device", "") or ""
    ).startswith("bi-")

    if is_bi_arm:
        left_names = [f"left_{n}" for n in action_names]
        right_names = [f"right_{n}" for n in action_names]
        joint_names = left_names + right_names
    else:
        joint_names = action_names

    dim = len(joint_names)
    features: Dict[str, Dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (dim,),
            "names": joint_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (dim,),
            "names": joint_names,
        },
    }

    if getattr(args, "enable_depth", False):
        features["observation.top_depth"] = {
            "dtype": "uint16",
            "shape": (480, 640),
            "names": ["height", "width"],
            "info": {
                "unit": "millimeters",
                "range_mm": [0, 65535],
                "range_m": [0.0, 65.535],
                "precision_mm": 1,
                "conversion": "depth_meters = uint16_value / 1000.0"
            }
        }

    if is_bi_arm:
        image_keys = ["top_rgb", "left_rgb", "right_rgb"]
    else:
        image_keys = ["top_rgb", "wrist_rgb"]

    for key in image_keys:
        features[f"observation.images.{key}"] = {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        }

    if getattr(args, "record_ee_pose", False):
        if is_bi_arm:
            ee_pose_dim = 16
            ee_pose_names = [
                "left_x",
                "left_y",
                "left_z",
                "left_qx",
                "left_qy",
                "left_qz",
                "left_qw",
                "left_gripper",
                "right_x",
                "right_y",
                "right_z",
                "right_qx",
                "right_qy",
                "right_qz",
                "right_qw",
                "right_gripper",
            ]
        else:
            ee_pose_dim = 8
            ee_pose_names = ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]

        features["observation.ee_pose"] = {
            "dtype": "float32",
            "shape": (ee_pose_dim,),
            "names": ee_pose_names,
        }
        features["action.ee_pose"] = {
            "dtype": "float32",
            "shape": (ee_pose_dim,),
            "names": ee_pose_names,
        }

    root_path = Path(getattr(args, "dataset_root", "Datasets/record"))

    # Extract garment type prefix from garment_name for folder naming
    # e.g., "Top_Long_Unseen_0" -> "top_long_20240328_143022"
    garment_name = getattr(args, "garment_name", None)
    if garment_name:
        parts = garment_name.split("_")
        if len(parts) >= 2:
            garment_prefix = f"{parts[0].lower()}_{parts[1].lower()}"  # "top_long"
        else:
            garment_prefix = parts[0].lower()
    elif hasattr(args, "garment_type"):
        garment_prefix = args.garment_type  # Fallback to garment_type if available
    else:
        garment_prefix = "dataset"

    # Create name with datetime: top_long_20240328_143022
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_prefix = f"{garment_prefix}_{timestamp}"

    dataset = LeRobotDataset.create(
        repo_id="abc",
        fps=30,
        root=get_next_experiment_path_with_gap(root_path, name_prefix=name_prefix),
        use_videos=True,
        image_writer_threads=8,
        image_writer_processes=0,
        features=features,
    )
    json_path = dataset.root / "meta" / "garment_info.json"

    solver = None
    if getattr(args, "record_ee_pose", False):
        if not args.ee_urdf_path:
            raise ValueError("--record_ee_pose requires --ee_urdf_path")

        urdf_path = Path(args.ee_urdf_path)
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")

        from lehome.utils import RobotKinematics

        if is_bi_arm:
            solver_joint_names = [n.replace("left_", "") for n in joint_names[:5]]
        else:
            solver_joint_names = joint_names[:5]

        solver = RobotKinematics(
            str(urdf_path),
            target_frame_name="gripper_frame_link",
            joint_names=solver_joint_names,
        )
        arm_type = "dual-arm" if is_bi_arm else "single-arm"
        logger.info(f"End-effector pose solver loaded ({arm_type})")

    return dataset, json_path, solver, is_bi_arm


def run_idle_phase(
    env: DirectRLEnv,
    teleop_interface: Any,
    args: argparse.Namespace,
    count_render: int,
) -> Tuple[Optional[Dict[str, Any]], int]:
    """Run idle phase before recording starts.

    Handles environment preparation, stabilization, and waits for user to press
    S key to start recording.

    Args:
        env: Environment instance.
        teleop_interface: Teleoperation interface.
        args: Command-line arguments.
        count_render: Current render count.

    Returns:
        Tuple of (object_initial_pose, updated_count_render).
    """
    dynamic_reset_gripper_effort_limit_sim(env, args.teleop_device)

    actions = teleop_interface.advance()
    object_initial_pose = None

    if count_render == 0:
        logger.info("[Idle Phase] Initializing observations...")
        env.initialize_obs()
        count_render += 1

        logger.info("[Idle Phase] Stabilizing garment after initialization...")
        stabilize_garment_after_reset(env, args)
        logger.info("[Idle Phase] Ready for recording")

    if actions is None:
        current_obs = env._get_observations()
        if "observation.state" in current_obs:
            current_state = current_obs["observation.state"]
            if isinstance(current_state, np.ndarray):
                maintain_action = (
                    torch.from_numpy(current_state).float().unsqueeze(0).to(env.device)
                )
            else:
                maintain_action = torch.zeros(
                    1, len(current_state), dtype=torch.float32, device=env.device
                )
        else:
            action_dim = 12 if "Bi" in args.task else 6
            maintain_action = torch.zeros(
                1, action_dim, dtype=torch.float32, device=env.device
            )
        env.step(maintain_action)
        env.render()
    else:
        env.step(actions)
        object_initial_pose = env.get_all_pose()

    if object_initial_pose is None:
        object_initial_pose = env.get_all_pose()

    return object_initial_pose, count_render


def run_recording_phase(
    env: DirectRLEnv,
    teleop_interface: Any,
    args: argparse.Namespace,
    flags: Dict[str, bool],
    dataset: LeRobotDataset,
    json_path: Path,
    initial_object_pose: Optional[Dict[str, Any]],
    ee_solver: Optional[Any] = None,
    is_bi_arm: bool = False,
) -> Dict[str, Any]:
    """Run recording phase after S key is pressed and recording is enabled.

    Records episodes until num_episode is reached. Each episode can be marked as
    successful (N key), discarded (D key), or aborted (ESC key).

    Args:
        env: Environment instance.
        teleop_interface: Teleoperation interface.
        args: Command-line arguments.
        flags: Status flags dictionary.
        dataset: LeRobotDataset instance.
        json_path: Path to object initial pose JSON file.
        initial_object_pose: Initial object pose dictionary.
        ee_solver: Optional kinematic solver for end-effector pose computation.
        is_bi_arm: Whether using dual-arm configuration.

    Returns:
        Final object initial pose dictionary.
    """
    episode_index = 0
    object_initial_pose = initial_object_pose
    step_counter = 0  # Track steps for periodic success check logging

    # Ensure we have a valid initial pose for the first episode
    if object_initial_pose is None:
        object_initial_pose = env.get_all_pose()

    while episode_index < args.num_episode:
        # Check if recording should be aborted
        if flags["abort"]:
            dataset.clear_episode_buffer()
            dataset.finalize()
            logger.warning(f"Recording aborted, completed {episode_index} episodes")
            return object_initial_pose

        flags["success"] = False
        flags["remove"] = False
        success_detected = False  # Track if env reports success during this episode
        step_counter = 0  # Track steps for periodic success check logging
        # Loop within a single episode
        while not flags["success"]:
            # Check if recording should be aborted
            if flags["abort"]:
                dataset.clear_episode_buffer()
                dataset.finalize()
                logger.warning(f"Recording aborted, completed {episode_index} episodes")
                return object_initial_pose

            step_counter += 1

            # Periodic detailed success check logging (every 30 steps)
            if step_counter % 30 == 0:
                _log_detailed_success_check(env, args)

            try:
                dynamic_reset_gripper_effort_limit_sim(env, args.teleop_device)
                actions = teleop_interface.advance()
            except Exception as e:
                logger.error(f"[Recording] Error in teleop interface: {e}")
                traceback.print_exc()
                actions = None

            if actions is None:
                env.render()
            else:
                env.step(actions)

            # Check success from environment and display to operator
            success = env._get_success()
            if success.item() and not success_detected:
                success_detected = True
                logger.info("✅ [Recording] SUCCESS detected by environment!")
                logger.info("   Press 'N' to confirm and save, or continue recording...")

            observations = env._get_observations()
            if (
                not getattr(args, "enable_depth", False)
                and "observation.top_depth" in observations
            ):
                observations.pop("observation.top_depth")

            if getattr(args, "enable_pointcloud", False):
                # Converting pointcloud online is time-consuming, please convert offline
                # pointcloud = env._get_workspace_pointcloud(
                #     num_points=4096, use_fps=True
                # )
                print("Converting pointcloud online is time-consuming, please convert offline")
            _, truncated = env._get_dones()
            frame = {**observations, "task": args.task_description}

            if (
                ee_solver is not None
                and "observation.state" in observations
                and "action" in observations
            ):
                from lehome.utils import compute_ee_pose_single_arm

                obs_state = np.array(
                    observations["observation.state"], dtype=np.float32
                )
                action_state = np.array(observations["action"], dtype=np.float32)

                if is_bi_arm:
                    obs_left = compute_ee_pose_single_arm(
                        ee_solver, obs_state[:6], args.ee_state_unit
                    )
                    obs_right = compute_ee_pose_single_arm(
                        ee_solver, obs_state[6:12], args.ee_state_unit
                    )
                    frame["observation.ee_pose"] = np.concatenate(
                        [obs_left, obs_right], axis=0
                    )

                    act_left = compute_ee_pose_single_arm(
                        ee_solver, action_state[:6], args.ee_state_unit
                    )
                    act_right = compute_ee_pose_single_arm(
                        ee_solver, action_state[6:12], args.ee_state_unit
                    )
                    frame["action.ee_pose"] = np.concatenate(
                        [act_left, act_right], axis=0
                    )
                else:
                    frame["observation.ee_pose"] = compute_ee_pose_single_arm(
                        ee_solver, obs_state, args.ee_state_unit
                    )
                    frame["action.ee_pose"] = compute_ee_pose_single_arm(
                        ee_solver, action_state, args.ee_state_unit
                    )

            dataset.add_frame(frame)

            if truncated or flags["remove"]:
                dataset.clear_episode_buffer()
                logger.info(f"Re-recording episode {episode_index}")
                try:
                    env.reset()
                    stabilize_garment_after_reset(env, args)
                    object_initial_pose = env.get_all_pose()
                except Exception as e:
                    logger.error(
                        f"[Recording] Failed to reset environment during re-recording: {e}"
                    )
                    traceback.print_exc()
                    try:
                        object_initial_pose = env.get_all_pose()
                    except Exception:
                        object_initial_pose = None
                flags["remove"] = False
                continue

        # Episode ended - show success status
        if success_detected:
            logger.info(f"[Recording] Episode {episode_index} ended with SUCCESS (env detected)")
        else:
            logger.info(f"[Recording] Episode {episode_index} ended (no env success detected)")

        save_start_time = time.time()
        logger.info(f"[Recording] Saving episode {episode_index}...")
        try:
            dataset.save_episode()
            save_duration = time.time() - save_start_time
            logger.info(
                f"[Recording] Episode {episode_index} saved (took {save_duration:.1f}s)"
            )
        except Exception as e:
            logger.error(f"[Recording] Failed to save episode {episode_index}: {e}")
            traceback.print_exc()

        garment_name = None
        if hasattr(env, "cfg") and hasattr(env.cfg, "garment_name"):
            garment_name = env.cfg.garment_name

        scale = None
        if hasattr(env, "object") and hasattr(env.object, "init_scale"):
            try:
                scale = env.object.init_scale
            except Exception:
                logger.warning("Failed to get scale from garment object")

        try:
            append_episode_initial_pose(
                json_path,
                episode_index,
                object_initial_pose,
                garment_name=garment_name,
                scale=scale,
            )
        except Exception as e:
            logger.error(
                f"[Recording] Failed to save episode metadata for episode {episode_index}: {e}"
            )
            traceback.print_exc()

        episode_index += 1
        logger.info(
            f"Episode {episode_index - 1} completed, progress: {episode_index}/{args.num_episode}"
        )

        try:
            env.reset()
            stabilize_garment_after_reset(env, args)
        except Exception as e:
            logger.error(f"[Recording] Failed to reset environment: {e}")
            traceback.print_exc()

        try:
            object_initial_pose = env.get_all_pose()
        except Exception as e:
            logger.error(f"[Recording] Failed to get initial pose: {e}")
            traceback.print_exc()
            object_initial_pose = None
    dataset.clear_episode_buffer()
    dataset.finalize()
    logger.info(f"All {args.num_episode} episodes recording completed!")
    return object_initial_pose


def run_live_control_without_record(
    env: DirectRLEnv,
    teleop_interface: Any,
    args: argparse.Namespace,
) -> None:
    """Run live teleoperation control without recording.

    Handles the case when S key is pressed but recording is not enabled.
    Performs simple teleoperation control without writing to dataset.

    Args:
        env: Environment instance.
        teleop_interface: Teleoperation interface.
        args: Command-line arguments.
    """
    dynamic_reset_gripper_effort_limit_sim(env, args.teleop_device)
    actions = teleop_interface.advance()

    if actions is None:
        current_obs = env._get_observations()
        if "observation.state" in current_obs:
            current_state = current_obs["observation.state"]
            if isinstance(current_state, np.ndarray):
                maintain_action = (
                    torch.from_numpy(current_state).float().unsqueeze(0).to(env.device)
                )
            else:
                maintain_action = torch.zeros(
                    1, len(current_state), dtype=torch.float32, device=env.device
                )
        else:
            action_dim = 12 if "Bi" in args.task else 6
            maintain_action = torch.zeros(
                1, action_dim, dtype=torch.float32, device=env.device
            )
        env.step(maintain_action)
        env.render()
    else:
        env.step(actions)

    if args.log_success:
        _ = env._get_success()


def record_dataset(args: argparse.Namespace, simulation_app: SimulationApp) -> None:
    """Record dataset.

    Supports recording multiple datasets with the same garment configuration.
    Each dataset will have its own folder with naming: {garment_prefix}_{YYYYMMDD_HHMMSS}
    (e.g., top_long_20240328_143022, top_long_20240328_150315)
    """
    # Get device configuration (default to "cpu" for compatibility)
    device = getattr(args, "device", "cpu")

    env_cfg = parse_env_cfg(
        args.task,
        device=device,
    )
    task_name = args.task

    env_cfg.garment_name = args.garment_name
    env_cfg.garment_version = args.garment_version
    env_cfg.garment_cfg_base_path = args.garment_cfg_base_path
    env_cfg.particle_cfg_path = args.particle_cfg_path

    if args.use_random_seed:
        env_cfg.use_random_seed = True
        logger.info("Using random seed (no fixed seed)")
    else:
        env_cfg.use_random_seed = False
        env_cfg.random_seed = args.seed
        logger.info(f"Using fixed random seed: {args.seed}")

    env: DirectRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped
    teleop_interface = create_teleop_interface(env, args)
    flags = register_teleop_callbacks(
        teleop_interface, recording_enabled=args.enable_record
    )
    teleop_interface.reset()

    count_render = 0
    printed_instructions = False
    idle_frame_counter = 0
    object_initial_pose: Optional[Dict[str, Any]] = None

    # Get number of datasets (default to 1 if not specified)
    num_datasets = getattr(args, "num_datasets", 1)

    try:
        # ========== IDLE PHASE (once before all datasets) ==========
        while simulation_app.is_running():
            with torch.inference_mode():
                if not flags["start"]:
                    pose, count_render = run_idle_phase(
                        env,
                        teleop_interface,
                        args,
                        count_render,
                    )
                    if pose is not None:
                        object_initial_pose = pose

                    if count_render > 0:
                        idle_frame_counter += 1
                        if idle_frame_counter == 100 and not printed_instructions:
                            logger.info("=" * 60)
                            logger.info("🎮 CONTROL INSTRUCTIONS 🎮")
                            logger.info("=" * 60)
                            logger.info(str(teleop_interface))
                            logger.info("=" * 60 + "\n\n")
                            printed_instructions = True

                elif args.enable_record:
                    # ========== DATASET LOOP ==========
                    for dataset_index in range(num_datasets):
                        logger.info("=" * 60)
                        logger.info(f"📦 DATASET {dataset_index + 1}/{num_datasets}")
                        logger.info("=" * 60)

                        # Create new dataset for this iteration
                        dataset, json_path, ee_solver, is_bi_arm = create_dataset_if_needed(
                            args, dataset_index=dataset_index
                        )

                        if dataset is None:
                            logger.error("Failed to create dataset, skipping...")
                            continue

                        logger.info(f"Dataset created at: {dataset.root}")

                        # Run recording phase for this dataset
                        object_initial_pose = run_recording_phase(
                            env,
                            teleop_interface,
                            args,
                            flags,
                            dataset,
                            json_path,
                            object_initial_pose,
                            ee_solver,
                            is_bi_arm,
                        )

                        # Check if recording was aborted
                        if flags["abort"]:
                            logger.warning("Recording aborted by user")
                            break

                        # Reset environment for next dataset (if not the last one)
                        if dataset_index < num_datasets - 1:
                            logger.info("Resetting environment for next dataset...")
                            try:
                                env.reset()
                                stabilize_garment_after_reset(env, args)
                                object_initial_pose = env.get_all_pose()
                            except Exception as e:
                                logger.error(f"Failed to reset environment: {e}")
                                traceback.print_exc()

                            # Reset flags for next dataset
                            flags["start"] = False
                            flags["success"] = False
                            flags["remove"] = False

                            logger.info(f"Ready for dataset {dataset_index + 2}/{num_datasets}")
                            logger.info("Press 'S' to start recording next dataset...")

                            # Wait for user to press S again for next dataset
                            while not flags["start"] and not flags["abort"]:
                                if not simulation_app.is_running():
                                    break
                                with torch.inference_mode():
                                    pose, _ = run_idle_phase(
                                        env,
                                        teleop_interface,
                                        args,
                                        count_render,
                                    )
                                    if pose is not None:
                                        object_initial_pose = pose

                            if flags["abort"]:
                                logger.warning("Recording aborted by user")
                                break

                    # All datasets completed
                    logger.info("=" * 60)
                    logger.info(f"✅ All {num_datasets} datasets completed!")
                    logger.info("=" * 60)
                    break

                else:
                    run_live_control_without_record(env, teleop_interface, args)

    except KeyboardInterrupt:
        logger.warning("\n[Ctrl+C] Interrupt signal detected")
        # Note: dataset is now created inside the loop, so we can't access it here
        # The cleanup is handled inside run_recording_phase
        logger.info("Recording interrupted")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        traceback.print_exc()

    finally:
        env.close()
