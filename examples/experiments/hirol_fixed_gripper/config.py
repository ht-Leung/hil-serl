"""Configuration for HIROL Fixed Gripper task using HIROLRobotPlatform - with RGB+Depth support"""
import os
import sys
import jax
import numpy as np
import jax.numpy as jnp

# Add serl_hirol_infra to path
sys.path.insert(0, '/home/hanyu/code/hil-serl/serl_hirol_infra')

from hirol_env.envs.hirol_env import DefaultEnvConfig, HIROLEnv
from hirol_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    KeyboardRewardWrapper,
    GripperCloseEnv
)
from experiments.hirol_fixed_gripper.wrapper import GripperPenaltyWrapper, HIROLFixedGripperEnv, GripperInitWrapper
from hirol_env.envs.relative_env import RelativeFrame
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.rgbd_obs_wrapper import SERLRGBDObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig


class EnvConfig(DefaultEnvConfig):
    """Configuration for HIROL Fixed Gripper task"""

    # Robot configuration file path (HIROLRobotPlatform uses config files)
    ROBOT_CONFIG_PATH = None  # Will use default serl_fr3_config.yaml if None

    # Camera configuration with optional depth support
    REALSENSE_CAMERAS = {
        "wrist_1": {"serial_number": "332322073603", "depth": True},
        "front": {"serial_number": "244222075350", "depth": True},
        "side": {"serial_number": "243122071795", "depth": True},
    }
    IMAGE_CROP = {
        "wrist_1": lambda img: img[:,:],
        "front": lambda img: img[:, :],
        "side": lambda img: img[0:480, 100:640],
    }

    # Reset pose - slightly offset from target
    RESET_POSE = np.array([0.5, 0.1, 0.45, -np.pi, 0, 0])

    # Reward threshold - 2cm for position, 0.1 rad for orientation
    REWARD_THRESHOLD = np.array([0.02, 0.02, 0.02, 0.1, 0.1, 0.1])

    # Action scale: [position_scale, rotation_scale, gripper_scale]
    ACTION_SCALE = np.array([0.02, 0.06, 0])   # gripper fixed

    # Enable random reset for diversity
    RANDOM_RESET = False
    RANDOM_XY_RANGE = 0.05
    RANDOM_RZ_RANGE = 0.1

    # Workspace limits
    ABS_POSE_LIMIT_HIGH = np.array([0.7, 0.24, 0.55, np.pi, 0., +0.3*np.pi])
    ABS_POSE_LIMIT_LOW = np.array([0.4, 0, 0.34, np.pi, -0., -0.3*np.pi])

    # Display
    DISPLAY_IMAGE = True
    GRIPPER_SLEEP = 0.0
    MAX_EPISODE_LENGTH = 100

    # Gripper initial state for fixed gripper tasks
    GRIPPER_INIT_STATE = "close"  # "open", "close", or "none"


class TrainConfig(DefaultTrainingConfig):
    """Training configuration for HIROL Fixed Gripper task"""

    # Image and proprioception keys
    image_keys = ["side", "wrist_1", "front"]
    classifier_keys = ["side", "wrist_1", "front"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]

    # Training parameters
    checkpoint_period = 5000
    cta_ratio = 2
    random_steps = 0
    discount = 0.97
    buffer_period = 1000
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"

    # 深度处理配置
    use_depth = True  # 设为True启用深度处理
    depth_encoder_kwargs = {
        #         "input_points": 2048,
        # "num_stages": 3,
        # "embed_dim": 72,
        # "bottleneck_dim": 256,
        "input_points": 1024,  # 减少点数以节省内存
        "num_stages": 2,       # 减少阶段数
        "embed_dim": 36,       # 减少嵌入维度
        "bottleneck_dim": 128, # 减少bottleneck维度以节省内存
    }

    # 相机内参 (从camera_intrinsics.json标定得到的真实参数)
    camera_params = {
        "wrist_1": {"fx": 385.55, "fy": 385.55, "cx": 322.526, "cy": 242.489},
        "front": {"fx": 386.619, "fy": 386.619, "cx": 324.881, "cy": 234.11},
        "side": {"fx": 383.305, "fy": 383.305, "cx": 320.544, "cy": 243.725},
    }

    # Data paths - all relative to experiments/hirol_fixed_gripper/
    @property
    def data_path(self):
        """Base data path for this experiment"""
        from pathlib import Path
        return Path(__file__).parent.absolute()

    @property
    def demo_data_path(self):
        """Path for demonstration data"""
        return str(self.data_path / "demo_data")

    @property
    def classifier_data_path(self):
        """Path for classifier training data"""
        return str(self.data_path / "classifier_data")

    @property
    def classifier_ckpt_path(self):
        """Path for classifier checkpoints"""
        return str(self.data_path / "classifier_ckpt")

    def get_environment(self, fake_env=False, save_video=False, classifier=True):
        """Create and configure the HIROL Fixed Gripper environment"""

        # 根据深度配置更新相机设置
        env_config = EnvConfig()
        if self.use_depth:
            # 启用深度时，更新相机配置
            for cam_name in env_config.REALSENSE_CAMERAS:
                env_config.REALSENSE_CAMERAS[cam_name]["depth"] = True

        # Create base environment
        env = HIROLFixedGripperEnv(
            hz=10,
            fake_env=fake_env,
            save_video=save_video,
            config=env_config
        )
        env = GripperCloseEnv(env)  # Fixed gripper - always closed

        # Control initial gripper state
        env = GripperInitWrapper(env, init_mode=env_config.GRIPPER_INIT_STATE)

        # Add spacemouse intervention for human demonstrations
        if not fake_env:
            env = SpacemouseIntervention(env, gripper_enabled=False)
        env = RelativeFrame(env)

        # Add wrappers
        env = Quat2EulerWrapper(env)

        # Use RGBD wrapper if depth is enabled, otherwise use standard SERL wrapper
        if self.use_depth:
            env = SERLRGBDObsWrapper(
                env,
                proprio_keys=self.proprio_keys,
                image_keys=self.image_keys,
                use_depth=self.use_depth
            )
        else:
            env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)

        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        # Add keyboard reward if requested (press 's' to give reward)
        if classifier:
            env = KeyboardRewardWrapper(env)

        return env