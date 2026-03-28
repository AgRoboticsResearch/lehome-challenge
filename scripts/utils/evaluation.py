import os
import argparse
import gymnasium as gym
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from isaaclab.envs import DirectRLEnv
from isaaclab_tasks.utils import parse_env_cfg

from scripts.eval_policy import PolicyRegistry
from scripts.eval_policy.base_policy import BasePolicy

from scripts.utils.eval_utils import (
    convert_ee_pose_to_joints,
    save_videos_from_observations,
    calculate_and_print_metrics,
)

from lehome.utils.record import (
    RateLimiter,
    get_next_experiment_path_with_gap,
    append_episode_initial_pose,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from .common import stabilize_garment_after_reset
from lehome.utils.logger import get_logger
from scripts.utils.dataset_record import create_teleop_interface
from scripts.utils.hil_intervention import HILInterventionManager

logger = get_logger(__name__)


def run_evaluation_loop(
    env: DirectRLEnv,
    policy: BasePolicy,
    args: argparse.Namespace,
    ee_solver: Optional[Any] = None,
    is_bimanual: bool = False,
    garment_name: Optional[str] = None,
    leader_device: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Core evaluation loop.
    Refactored to be agnostic of specific model implementations.
    """

    # --- HIL Keyboard Handler Setup (Optional) ---
    hil_handler = None
    if args.enable_hil:
        from scripts.utils.hil_keyboard import HILKeyboardHandler

        hil_handler = HILKeyboardHandler()
        hil_handler.start()
        logger.info("HIL mode enabled. Press 'i' to toggle intervention mode.")
        logger.info("  Mode starts in POLICY (automatic) mode")

    # --- HIL Intervention Manager Setup (Optional) ---
    # Create HIL manager when both --enable_hil and AND leader_device exists
    hil_manager = None
    if getattr(args, 'enable_hil', False) and leader_device is not None:
        logger.info("Creating HIL intervention manager...")
        hil_manager = HILInterventionManager(
            leader_device=leader_device,
            is_bimanual=is_bimanual,
        )
        logger.info(f"✅ HIL intervention manager created for {'bimanual' if is_bimanual else 'single arm'}")
    else:
        logger.debug(f"HIL manager not created: enable_hil={getattr(args, 'enable_hil', False)}, leader_device={leader_device is not None}")

    # --- Dataset Recording Setup (Optional) ---
    eval_dataset = None
    eval_dataset_success = None
    eval_dataset_failure = None
    json_path = None
    json_path_success = None
    json_path_failure = None
    episode_index = 0
    episode_index_success = 0
    episode_index_failure = 0

    if args.save_datasets:
        # Determine joint names and dimension (needed for complementary_info)
        action_names = [
            "shoulder_pan", "shoulder_lift", "elbow_flex",
            "wrist_flex", "wrist_roll", "gripper",
        ]
        if is_bimanual:
            left_names = [f"left_{n}" for n in action_names]
            right_names = [f"right_{n}" for n in action_names]
            joint_names = left_names + right_names
        else:
            joint_names = action_names
        dim = len(joint_names)

        features = None
        if args.dataset_root and Path(args.dataset_root).exists():
            source_dataset = LeRobotDataset(repo_id="collected_dataset", root=Path(args.dataset_root))
            features = dict(source_dataset.meta.features)
            fps = source_dataset.fps
        else:
            fps = 30  # Default FPS if no source dataset is provided
            features = {
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
            image_keys = ["top_rgb", "left_rgb", "right_rgb"] if is_bimanual else ["top_rgb", "wrist_rgb"]
            for key in image_keys:
                features[f"observation.images.{key}"] = {
                    "dtype": "video",
                    "shape": (480, 640, 3),
                    "names": ["height", "width", "channels"],
                }

        # Add Evo-RL complementary_info fields for HIL workflow
        # State codes: 0.0 = POLICY, 1.0 = ACTIVE (intervention), 2.0 = RELEASE
        features["complementary_info.policy_action"] = {
            "dtype": "float32",
            "shape": (dim,),
            "names": joint_names,
        }
        features["complementary_info.is_intervention"] = {
            "dtype": "float32",
            "shape": (1,),
        }
        features["complementary_info.state"] = {
            "dtype": "float32",
            "shape": (1,),
        }
        features["complementary_info.collector_policy_id"] = {
            "dtype": "string",
            "shape": (1,),
        }

        root_path = Path(args.eval_dataset_path)
        save_mode = args.save_mode

        if save_mode == "both":
            # Create two separate datasets for success and failure
            success_path = get_next_experiment_path_with_gap(root_path / "success", name_prefix=args.garment_type)
            failure_path = get_next_experiment_path_with_gap(root_path / "failure", name_prefix=args.garment_type)

            eval_dataset_success = LeRobotDataset.create(
                repo_id="lehome_eval_success",
                fps=fps,
                root=success_path,
                use_videos=True,
                image_writer_threads=8,
                image_writer_processes=0,
                features=features,
            )
            eval_dataset_failure = LeRobotDataset.create(
                repo_id="lehome_eval_failure",
                fps=fps,
                root=failure_path,
                use_videos=True,
                image_writer_threads=8,
                image_writer_processes=0,
                features=features,
            )
            json_path_success = eval_dataset_success.root / "meta" / "garment_info.json"
            json_path_failure = eval_dataset_failure.root / "meta" / "garment_info.json"
        else:
            # Create single dataset for all modes except "both"
            eval_dataset = LeRobotDataset.create(
                repo_id="lehome_eval",
                fps=fps,
                root=get_next_experiment_path_with_gap(root_path, name_prefix=args.garment_type),
                use_videos=True,
                image_writer_threads=8,
                image_writer_processes=0,
                features=features,
            )
            json_path = eval_dataset.root / "meta" / "garment_info.json"

    all_episode_metrics = []
    logger.info(f"Starting evaluation: {args.num_episodes} episodes")
    rate_limiter = RateLimiter(args.step_hz)

    for i in range(args.num_episodes):
        # 1. Reset Environment & Policy
        env.reset()
        policy.reset()
        stabilize_garment_after_reset(env, args)

        # 2. Initial Observation (Numpy)
        object_initial_pose = env.get_all_pose() if args.save_datasets else None
        observation_dict = env._get_observations()

        # --- IDLE PHASE (HIL mode only): Wait for 'b' to start episode ---
        if hil_handler is not None:
            logger.info(f"[HIL] Episode {i+1}/{args.num_episodes}: Waiting for 'b' to start...")
            hil_handler.reset_episode_start()  # Clear any previous start request

            idle_count = 0
            while True:
                if rate_limiter:
                    rate_limiter.sleep(env)

                # Poll for keyboard input
                hil_handler.poll()

                # Check for quit request
                if hil_handler.is_quit_requested():
                    logger.info("[HIL] Quit requested during idle phase. Exiting...")
                    # Cleanup and return
                    if hil_handler:
                        hil_handler.stop()
                    return all_episode_metrics, True  # quit_requested = True

                # Check for episode start
                if hil_handler.is_episode_start_requested():
                    logger.info(f"[HIL] Episode {i+1} started!")
                    hil_handler.reset_episode_start()
                    break

                # Maintain current position during idle (hold position)
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
                    action_dim = 12 if is_bimanual else 6
                    maintain_action = torch.zeros(
                        1, action_dim, dtype=torch.float32, device=env.device
                    )
                env.step(maintain_action)

                # Print idle message periodically
                idle_count += 1
                if idle_count % 120 == 0:  # Every ~1 second at 120Hz
                    logger.info(f"[HIL] Press 'b' to start episode {i+1}...")

        # Prepare for video recording
        episode_frames = (
            {k: [] for k in observation_dict.keys() if "images" in k}
            if args.save_video
            else {}
        )

        episode_return = 0.0
        episode_length = 0
        extra_steps = 0
        success_flag = False
        success = torch.tensor(False)
        episode_end_requested = False

        # Track policy actions and interventions for Evo-RL complementary_info
        policy_actions = []
        is_interventions = []
        last_policy_action = None

        # Track if at least one frame has been recorded (for quit handling)
        frames_recorded_this_episode = False

        for st in range(args.max_steps):
            if rate_limiter:
                rate_limiter.sleep(env)

            # HIL: Poll for keyboard input
            if hil_handler:
                hil_handler.poll()

                # Check for quit request
                if hil_handler.is_quit_requested():
                    logger.info("[HIL] Quit requested by user. Ending evaluation...")
                    break

                # Check for episode end request (s key)
                if hil_handler.is_episode_end_requested():
                    logger.info("[HIL] Episode end requested by user (s key)")
                    episode_end_requested = True
                    hil_handler.reset_episode_end()
                    break

                # Handle mode transitions with torque control
                if hil_handler.is_intervention_toggled():
                    is_intervention = hil_handler.is_intervention_active()
                    if hil_manager:
                        hil_manager.set_intervention_mode(is_intervention)
                        mode = "HUMAN" if is_intervention else "POLICY"
                        logger.info(f"[HIL] Mode switched to: {mode}")
                    hil_handler.reset_toggle()

            # 3. Get action from policy first
            # Policy always computes action (needed for complementary_info tracking)
            # Note: policy.select_action() expects numpy arrays, not tensors!
            action_np = policy.select_action(observation_dict)

            # Store policy action before any intervention override
            last_policy_action = action_np.copy()

            # 4. HIL Override: If in intervention mode, get action from leader device
            is_intervention = False
            if hil_handler:
                is_intervention = hil_handler.is_intervention_active()
                if is_intervention and hil_manager is not None:
                    # Override with leader action (HUMAN mode)
                    action_np = hil_manager.get_action(action_np, hil_handler)

            # Track for complementary_info
            policy_actions.append(last_policy_action)
            is_interventions.append(is_intervention)

            # 4. Prepare Action for Environment (Tensor)
            # Convert numpy action to tensor for Isaac Lab
            action = torch.from_numpy(action_np).float().to(args.device).unsqueeze(0)

            # 5. Inverse Kinematics (Optional Helper Logic)
            # If policy outputs EE pose but env needs joints
            if args.use_ee_pose and ee_solver is not None:
                current_joints = (
                    torch.from_numpy(observation_dict["observation.state"])
                    .float()
                    .to(args.device)
                )
                action = convert_ee_pose_to_joints(
                    ee_pose_action=action.squeeze(0),
                    current_joints=current_joints,
                    solver=ee_solver,
                    is_bimanual=is_bimanual,
                    state_unit="rad",
                    device=args.device,
                ).unsqueeze(0)

            # 6. Step Environment
            env.step(action)

            # 7. Send feedback to leader for tactile feedback
            if hil_manager is not None:
                # HIL mode: use HIL manager for feedback (handles POLICY/HUMAN mode)
                action_tensor = action.cpu().numpy().squeeze()
                verbose = getattr(args, 'policy_sync_verbose', False) and st < 10
                hil_manager.send_feedback(action_tensor, verbose=verbose)
            elif leader_device is not None:
                # Legacy policy-sync mode (non-HIL)
                action_np = action.cpu().numpy().squeeze()
                verbose = getattr(args, 'policy_sync_verbose', False) and st < 10
                try:
                    leader_device.send_feedback(action_np, verbose=verbose)
                except Exception as e:
                    logger.warning(f"⚠️  Leader feedback failed on step {st}: {e}")
                    logger.warning("Disabling leader feedback for remainder of episode")
                    leader_device = None  # Disable further attempts

            # Check success first
            if not success_flag:
                success = env._get_success()
                if success.item():
                    success_flag = True
                    extra_steps = 50  # Run a bit longer after success to settle

            # Get reward from environment (Isaac Lab stores rewards internally)
            reward_value = env._get_rewards()
            if isinstance(reward_value, torch.Tensor):
                reward = reward_value.item()
            else:
                reward = float(reward_value)

            # Accumulate reward for all steps (including post-success steps)
            episode_return += reward
            # Only count length before success (for consistency with episode termination)
            if not success_flag:
                episode_length += 1

            # Update Observation
            observation_dict = env._get_observations()

            # Recording
            if args.save_datasets:
                frame = {
                    k: v
                    for k, v in observation_dict.items()
                    if k != "observation.top_depth"
                }
                # Add task field (required by validate_frame, but not stored in data files)
                frame["task"] = args.task_description

                # Add Evo-RL complementary_info fields
                # State codes: 0.0 = POLICY, 1.0 = ACTIVE (intervention)
                # Use step index to get corresponding policy action
                step_idx = len(policy_actions) - 1 if policy_actions else 0
                if step_idx < len(policy_actions):
                    frame["complementary_info.policy_action"] = policy_actions[step_idx]
                    is_int_val = 1.0 if is_interventions[step_idx] else 0.0
                    frame["complementary_info.is_intervention"] = np.array([is_int_val], dtype=np.float32)
                    state_val = 1.0 if is_interventions[step_idx] else 0.0  # ACTIVE or POLICY
                    frame["complementary_info.state"] = np.array([state_val], dtype=np.float32)
                    # Determine collector policy ID for this episode
                    # Will be updated at end if any intervention occurred
                    frame["complementary_info.collector_policy_id"] = "policy"

                # Determine which dataset(s) to record to based on save_mode
                save_mode = args.save_mode
                if save_mode == "both":
                    # In "both" mode, we record to both datasets during the episode
                    # and decide which one to save at the end based on success/failure
                    # NOTE: LeRobot's add_frame() pops "task" from the frame, so we need to re-add it
                    try:
                        eval_dataset_success.add_frame(frame)
                        frame["task"] = args.task_description  # Re-add after pop
                        eval_dataset_failure.add_frame(frame)
                        frames_recorded_this_episode = True
                    except Exception as e:
                        logger.error(f"Error adding frame to dataset: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    # For other modes, record to the single dataset
                    # We'll filter episodes when saving
                    try:
                        eval_dataset.add_frame(frame)
                        frames_recorded_this_episode = True
                    except Exception as e:
                        logger.error(f"Error adding frame to dataset: {e}")
                        import traceback
                        traceback.print_exc()

            if args.save_video:
                for key, val in observation_dict.items():
                    if "images" in key:
                        episode_frames[key].append(val.copy())

            if success_flag:
                extra_steps -= 1
                if extra_steps <= 0:
                    break

        # Check if user requested quit during episode
        if hil_handler and hil_handler.is_quit_requested():
            # Cleanup and return
            if hil_handler:
                hil_handler.stop()
            return all_episode_metrics, True  # quit_requested = True

        # --- End of Episode Handling ---
        # Check if manual label was provided via keyboard (HIL mode)
        manual_label = None
        if hil_handler:
            manual_label = hil_handler.get_episode_label()

        # Determine episode success: manual label (HIL) > environment flag
        if manual_label:
            # HIL mode: user manually labeled the episode
            episode_success_value = manual_label
            is_success = (manual_label == "success")
        else:
            # Auto mode: use environment's success flag
            is_success = success.item() if success_flag else False
            episode_success_value = "success" if is_success else "failure"

        # Determine collector_policy_id based on whether any intervention occurred
        had_intervention = any(is_interventions) if is_interventions else False
        collector_policy_id = "human" if had_intervention else "policy"

        # Save Datasets based on save_mode
        if args.save_datasets:
            save_mode = args.save_mode

            # Check if we have any frames to save (can happen if quit before first frame)
            has_frames = len(policy_actions) > 0
            if not has_frames:
                logger.warning(f"[HIL] No frames recorded for episode {i+1}, skipping save. "
                             f"(Did you press 'q' during episode reset?)")
                continue

            # Debug: Log frame count before saving
            logger.info(f"[HIL] Episode {i+1}: Attempting to save with {len(policy_actions)} frames")

            if save_mode == "success":
                # Only save successful episodes
                if is_success:
                    extra_episode_metadata = {
                        "episode_success": episode_success_value,
                    }
                    try:
                        eval_dataset.save_episode(extra_episode_metadata=extra_episode_metadata)
                        logger.info(f"[HIL] Episode {i+1} saved successfully (success)")
                        append_episode_initial_pose(
                            json_path,
                            episode_index,
                            object_initial_pose,
                            garment_name=garment_name,
                        )
                        episode_index += 1
                    except Exception as e:
                        logger.error(f"[HIL] Error saving episode {i+1}: {e}")
                        import traceback
                        traceback.print_exc()
                        try:
                            eval_dataset.clear_episode_buffer()
                        except:
                            pass
                else:
                    # Clear the buffer for failed episodes
                    try:
                        eval_dataset.clear_episode_buffer()
                    except:
                        pass

            elif save_mode == "failure":
                # Only save failed episodes
                if not is_success:
                    extra_episode_metadata = {
                        "episode_success": episode_success_value,
                    }
                    try:
                        eval_dataset.save_episode(extra_episode_metadata=extra_episode_metadata)
                        logger.info(f"[HIL] Episode {i+1} saved successfully (failure)")
                        append_episode_initial_pose(
                            json_path,
                            episode_index,
                            object_initial_pose,
                            garment_name=garment_name,
                        )
                        episode_index += 1
                    except Exception as e:
                        logger.error(f"[HIL] Error saving episode {i+1}: {e}")
                        import traceback
                        traceback.print_exc()
                        try:
                            eval_dataset.clear_episode_buffer()
                        except:
                            pass
                else:
                    # Clear the buffer for successful episodes
                    try:
                        eval_dataset.clear_episode_buffer()
                    except:
                        pass

            elif save_mode == "both":
                # Save to appropriate dataset based on outcome
                extra_episode_metadata = {
                    "episode_success": episode_success_value,
                }
                if is_success:
                    try:
                        eval_dataset_success.save_episode(extra_episode_metadata=extra_episode_metadata)
                        logger.info(f"[HIL] Episode {i+1} saved successfully (success)")
                        append_episode_initial_pose(
                            json_path_success,
                            episode_index_success,
                            object_initial_pose,
                            garment_name=garment_name,
                        )
                        episode_index_success += 1
                        # Clear failure buffer
                        try:
                            eval_dataset_failure.clear_episode_buffer()
                        except:
                            pass
                    except Exception as e:
                        logger.error(f"[HIL] Error saving episode {i+1}: {e}")
                        import traceback
                        traceback.print_exc()
                        try:
                            eval_dataset_success.clear_episode_buffer()
                            eval_dataset_failure.clear_episode_buffer()
                        except:
                            pass
                else:
                    try:
                        eval_dataset_failure.save_episode(extra_episode_metadata=extra_episode_metadata)
                        logger.info(f"[HIL] Episode {i+1} saved successfully (failure)")
                        append_episode_initial_pose(
                            json_path_failure,
                            episode_index_failure,
                            object_initial_pose,
                            garment_name=garment_name,
                        )
                        episode_index_failure += 1
                        # Clear success buffer
                        try:
                            eval_dataset_success.clear_episode_buffer()
                        except:
                            pass
                    except Exception as e:
                        logger.error(f"[HIL] Error saving episode {i+1}: {e}")
                        import traceback
                        traceback.print_exc()
                        try:
                            eval_dataset_success.clear_episode_buffer()
                            eval_dataset_failure.clear_episode_buffer()
                        except:
                            pass

            elif save_mode == "all":
                # Save all episodes
                extra_episode_metadata = {
                    "episode_success": episode_success_value,
                }
                try:
                    eval_dataset.save_episode(extra_episode_metadata=extra_episode_metadata)
                    logger.info(f"[HIL] Episode {i+1} saved successfully")
                    append_episode_initial_pose(
                        json_path,
                        episode_index,
                        object_initial_pose,
                        garment_name=garment_name,
                    )
                    episode_index += 1
                except Exception as e:
                    logger.error(f"[HIL] Error saving episode {i+1}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Clear the buffer to prevent issues with next episode
                    try:
                        eval_dataset.clear_episode_buffer()
                    except:
                        pass

        # Save Videos (Using generic util)
        if args.save_video:
            save_videos_from_observations(
                episode_frames,
                success=success if success_flag else torch.tensor(False),
                save_dir=args.video_dir,
                episode_idx=i,
            )

        # Log Metrics
        all_episode_metrics.append(
            {"return": episode_return, "length": episode_length, "success": is_success}
        )

        # Add HIL mode indicator to log
        mode_str = ""
        if hil_handler:
            mode_str = f" [HIL: {'HUMAN' if hil_handler.is_intervention_active() else 'POLICY'}]"

        logger.info(
            f"Episode {i + 1}/{args.num_episodes}: Return={episode_return:.2f}, Length={episode_length}, Success={is_success}{mode_str}"
        )

    # Finalize dataset(s) to flush metadata buffers and close writers
    if args.save_datasets:
        save_mode = args.save_mode
        if save_mode == "both":
            if eval_dataset_success is not None:
                logger.info("Finalizing success dataset...")
                eval_dataset_success.finalize()
            if eval_dataset_failure is not None:
                logger.info("Finalizing failure dataset...")
                eval_dataset_failure.finalize()
        elif eval_dataset is not None:
            logger.info("Finalizing dataset...")
            eval_dataset.finalize()

    # Check if user requested quit (for HIL mode) - before stopping handler
    quit_requested = False
    if hil_handler:
        quit_requested = hil_handler.is_quit_requested()
        hil_handler.stop()

    return all_episode_metrics, quit_requested


def eval(args: argparse.Namespace, simulation_app: Any) -> None:
    """
    Main entry point for evaluation logic.
    """
    # 1. Environment Configuration
    env_cfg = parse_env_cfg(args.task, device=args.device)
    env_cfg.sim.use_fabric = False
    if args.use_random_seed:
        env_cfg.use_random_seed = True
    else:
        env_cfg.use_random_seed = False
        env_cfg.seed = args.seed
        # Propagate seed to sim config if structure exists
        if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "seed"):
            env_cfg.sim.seed = args.seed

    env_cfg.garment_cfg_base_path = args.garment_cfg_base_path
    env_cfg.particle_cfg_path = args.particle_cfg_path

    # 2. Initialize Policy (Using the Policy Registry)
    # This replaces create_il_policy, make_pre_post_processors, etc.
    logger.info(f"Initializing Policy Type: {args.policy_type}")

    # Check if policy is registered
    if not PolicyRegistry.is_registered(args.policy_type):
        available_policies = PolicyRegistry.list_policies()
        raise ValueError(
            f"Policy type '{args.policy_type}' not found in registry. "
            f"Available policies: {', '.join(available_policies)}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_bimanual = "Bi" in args.task or "bi" in args.task.lower()

    # Create policy instance from registry with appropriate arguments
    # Different policies may require different initialization arguments
    policy_kwargs = {
        "device": device,
    }

    if args.policy_type == "lerobot":
        # LeRobot policy requires policy_path and dataset_root
        if not args.policy_path:
            raise ValueError("--policy_path is required for lerobot policy type")
        if not args.dataset_root:
            raise ValueError("--dataset_root is required for lerobot policy type")
        policy_kwargs.update(
            {
                "policy_path": args.policy_path,
                "dataset_root": args.dataset_root,
                "task_description": args.task_description,
            }
        )
    else:
        # For custom policies, pass policy_path as model_path if provided
        if args.policy_path:
            policy_kwargs["model_path"] = args.policy_path

    # Create policy from registry
    policy = PolicyRegistry.create(args.policy_type, **policy_kwargs)
    logger.info(f"Policy '{args.policy_type}' loaded successfully")

    # 3. Initialize IK Solver (If needed)
    ee_solver = None
    if args.use_ee_pose:
        from lehome.utils import RobotKinematics

        urdf_path = args.ee_urdf_path  # Assuming path is handled or add check logic
        joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ]
        ee_solver = RobotKinematics(
            str(urdf_path),
            target_frame_name="gripper_frame_link",
            joint_names=joint_names,
        )
        logger.info(f"IK solver loaded.")

    # 4. Load Evaluation List
    # List of (name, stage)
    eval_list = []

    # Check if evaluating a single specific garment
    if args.garment_name:
        # Single garment mode: evaluate only the specified garment
        eval_list = [(args.garment_name, "Release")]
        logger.info(f"Single garment mode: evaluating '{args.garment_name}'")
    else:
        # Category mode: load all garments of the specified type
        if args.garment_type == "custom":
            # For 'custom' type, we load from the root Release_test_list.txt
            eval_list_path = os.path.join(
                args.garment_cfg_base_path, "Release", "Release_test_list.txt"
            )
        else:
            # Map argument to specific sub-category directory
            type_map = {
                "top_long": "Top_Long",
                "top_short": "Top_Short",
                "pant_long": "Pant_Long",
                "pant_short": "Pant_Short",
            }
            file_prefix = type_map.get(args.garment_type, "Top_Long")
            # Path: Assets/objects/Challenge_Garment/Release/Top_Long/Top_Long.txt
            eval_list_path = os.path.join(
                args.garment_cfg_base_path, "Release", file_prefix, f"{file_prefix}.txt"
            )

        logger.info(
            f"Loading evaluation list for category '{args.garment_type}' from: {eval_list_path}"
        )

        if not os.path.exists(eval_list_path):
            raise FileNotFoundError(f"Evaluation list not found: {eval_list_path}")

        with open(eval_list_path, "r") as f:
            names = [line.strip() for line in f.readlines() if line.strip()]
            for name in names:
                eval_list.append((name, "Release"))

        logger.info(f"Loaded {len(eval_list)} garments for category: {args.garment_type}")

    if not eval_list:
        raise ValueError(
            f"No garments found to evaluate for category '{args.garment_type}'."
        )

    # 5. Main Evaluation Loops
    all_garment_metrics = []

    # Init Env with first garment
    first_name, first_stage = eval_list[0]
    env_cfg.garment_name = first_name
    env_cfg.garment_version = first_stage
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.initialize_obs()

    # 6. Initialize Leader Device (for HIL or policy sync) - must be after env creation
    leader_device = None
    if getattr(args, 'enable_hil', False) or getattr(args, 'enable_policy_sync', False):
        if args.teleop_device in ['so101leader', 'bi-so101leader']:
            # Validate task and device match
            if is_bimanual and args.teleop_device != 'bi-so101leader':
                raise ValueError(
                    f"Bimanual task '{args.task}' requires 'bi-so101leader' device, "
                    f"but got '{args.teleop_device}'"
                )
            if not is_bimanual and args.teleop_device not in ['so101leader', 'keyboard']:
                raise ValueError(
                    f"Single-arm task '{args.task}' requires 'so101leader' device, "
                    f"but got '{args.teleop_device}'"
                )

            logger.info(f"Creating leader device: {args.teleop_device}")
            leader_device = create_teleop_interface(env=env, args=args)
            if leader_device:
                logger.info(f"✅ Leader device created: {args.teleop_device}")
                logger.info("  IMPORTANT: Press 'b' on the leader device to start it before using HIL intervention")
            else:
                logger.warning(f"⚠️ Failed to create leader device: {args.teleop_device}")
        else:
            logger.info(f"HIL/Policy sync enabled but teleop_device is '{args.teleop_device}' (not a leader arm)")

    try:
        for garment_idx, (garment_name, garment_stage) in enumerate(eval_list):
            logger.info(
                f"Evaluating: {garment_name} ({garment_stage}) ({garment_idx+1}/{len(eval_list)})"
            )

            # Switch Garment Logic
            if garment_idx > 0:
                if hasattr(env, "switch_garment"):
                    env.switch_garment(garment_name, garment_stage)
                    env.reset()
                    policy.reset()
                else:
                    env.close()
                    env_cfg.garment_name = garment_name
                    env_cfg.garment_version = garment_stage
                    env = gym.make(args.task, cfg=env_cfg).unwrapped
                    env.initialize_obs()
                    policy.reset()

            # Run Loop
            metrics, quit_requested = run_evaluation_loop(
                env=env,
                policy=policy,
                args=args,
                ee_solver=ee_solver,
                is_bimanual=is_bimanual,
                garment_name=garment_name,
                leader_device=leader_device,
            )

            all_garment_metrics.append(
                {"garment_name": garment_name, "metrics": metrics}
            )

            # Stop evaluation if user requested quit (HIL mode)
            if quit_requested:
                logger.info("[HIL] Quit requested by user. Stopping evaluation...")
                break

    finally:
        env.close()

    # Print summary across all garments
    logger.info("=" * 60)
    logger.info("Overall Summary")
    logger.info("=" * 60)

    if all_garment_metrics:
        # Aggregate all episode metrics
        all_episodes = []
        for garment_data in all_garment_metrics:
            for episode_metric in garment_data["metrics"]:
                episode_metric["garment_name"] = garment_data["garment_name"]
                all_episodes.append(episode_metric)

        # Print overall metrics
        calculate_and_print_metrics(all_episodes)

        # Print per-garment summary
        logger.info("=" * 60)
        logger.info("Per-Garment Summary")
        logger.info("=" * 60)
        for garment_data in all_garment_metrics:
            garment_name = garment_data["garment_name"]
            metrics = garment_data["metrics"]
            success_count = sum(1 for m in metrics if m["success"])
            success_rate = success_count / len(metrics) if metrics else 0.0
            avg_return = np.mean([m["return"] for m in metrics]) if metrics else 0.0
            logger.info(
                f"  {garment_name}: Success Rate = {success_rate:.2%}, Avg Return = {avg_return:.2f}"
            )
    else:
        logger.info("No metrics collected (all evaluations failed)")

    logger.info("=" * 60)
    logger.info("Evaluation completed successfully")
    logger.info("=" * 60)
