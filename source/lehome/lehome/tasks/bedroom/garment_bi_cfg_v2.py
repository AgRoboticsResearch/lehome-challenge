from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg

from lehome.assets.robots.lerobot import SO101_FOLLOWER_CFG


@configclass
class GarmentEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 1
    episode_length_s = 60
    action_scale = 1.0  # [N]
    action_space = 12
    observation_space = 12
    state_space = 0
    # simulation
    render_cfg = sim_utils.RenderCfg(rendering_mode="balanced", antialiasing_mode="FXAA")
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,  # Faster physics for better responsiveness (8.3ms per step)
        render_interval=decimation,  # Render every physics step for smooth visuals
        render=render_cfg,
        use_fabric=False,
    )
    # garment_name (str): Garment name in the format "Type_Length_Seen/Unseen_Index",
    # e.g., "Top_Long_Unseen_0", "Top_Short_Seen_1",
    garment_name: str = None
    garment_version: str = "Release"  # "Release" or "Holdout"
    garment_cfg_base_path: str = "Assets/objects/Challenge_Garment"
    particle_cfg_path: str = (
        "source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml"
    )
    # random seed
    use_random_seed: bool = True
    random_seed: int = 42

    # robot
    left_robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(
        prim_path="/World/Robot/Left_Robot",
        init_state=SO101_FOLLOWER_CFG.init_state.replace(
            pos=(-0.23, -0.25, 0.5),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "shoulder_pan": -1.1363,
                "shoulder_lift": 0.0,
                "elbow_flex": 0.0,
                "wrist_flex": 0.0,
                "wrist_roll": 0.0,
                "gripper": 0.0,
            },
        ),  # (pos=(2.7, -2.76, 0.21),
        # rot=(0.707, 0.0, 0.0, 0.707) )
    )
    right_robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(
        prim_path="/World/Robot/Right_Robot",
        init_state=SO101_FOLLOWER_CFG.init_state.replace(
            pos=(0.23, -0.25, 0.5),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "shoulder_pan": 1.1363,
                "shoulder_lift": 0.0,
                "elbow_flex": 0.0,
                "wrist_flex": 0.0,
                "wrist_roll": 0.0,
                "gripper": 0.0,
            },
        ),  # (pos=(2.7, -3.11, 0.21),
        # rot=(0.707, 0.0, 0.0, 0.707) )
    )
    left_wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/Left_Robot/gripper/left_wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.001, 0.1, -0.04),
            rot=(-0.404379, -0.912179, -0.0451242, 0.0486914),
            convention="ros",
        ),  # wxyz
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.5,
            focus_distance=400.0,
            horizontal_aperture=36.83,  # For a 75° FOV (assuming square image)
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,  # 30FPS for high-quality dataset recording
    )
    right_wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/Right_Robot/gripper/right_wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.001, 0.1, -0.04),
            rot=(-0.404379, -0.912179, -0.0451242, 0.0486914),
            convention="ros",
        ),  # wxyz
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.5,
            focus_distance=400.0,
            horizontal_aperture=36.83,  # For a 75° FOV (assuming square image)
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,  # 30FPS for high-quality dataset recording
    )
    top_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/Right_Robot/base/top_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.245, -0.44, 0.56),
            rot=(0.1650476, -0.9862856, 0.0, 0.0),
            convention="ros",
        ),  # wxyz
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.7,
            focus_distance=400.0,
            horizontal_aperture=38.11,  # For a 78° FOV (assuming square image)
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
    )
    left_side_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Cameras/Left_Side_View",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.32874, -0.17899, 0.61484),
            rot=(0.77174, 0.55257, -0.1628, -0.26941),  # wxyz - normalized quaternion
            convention="opengl",
        ),  # xwzy
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=36.0,  # For a ~75° FOV
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
    )
    right_side_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Cameras/Right_Side_View",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.2571, -0.13028, 0.5994),
            rot=(0.81141, 0.50518, 0.12366, 0.26667),  # wxyz - normalized quaternion
            convention="opengl",
        ),  # xwzy
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=36.0,  # For a ~75° FOV
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
    )
    left_bot_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Cameras/Left_Bot_View",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.19034, -0.21539, 0.63653),  # TODO: Tune position
            rot=(0.88838, 0.45893, -0.00943, 0.00876),  # TODO: Tune rotation (identity = no rotation)
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=36.0,  # For a ~75° FOV
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
    )
    right_bot_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Cameras/Right_Bot_View",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.09391, -0.24474, 0.62588),  # TODO: Tune position
            rot=(0.84521, 0.53443, 0.0, 0.0),  # TODO: Tune rotation (identity = no rotation)
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=36.0,  # For a ~75° FOV
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=4.0, replicate_physics=True
    )

    viewer = ViewerCfg(eye=(1.9, -4.7, 1.4), lookat=(1.3, 1.2, -1))
