import os
import jax
import numpy as np
import jax.numpy as jnp

from serl_hirol_infra.hirol_env.envs.hirol_env import DefaultEnvConfig
from serl_hirol_infra.hirol_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
)
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.hirol_reach.wrapper import HIROLReachEnv


class EnvConfig(DefaultEnvConfig):
    """Configuration for HIROL reach task"""
    REALSENSE_CAMERAS = {
        "wrist_1": {"serial_number": "332322073603"},
        "side": {"serial_number": "244222075350"},
    }
    IMAGE_CROP = {
        "wrist_1": lambda img: img[100:-100, 200:-200],
        "side": lambda img: img[200:-200, 300:-300],
    }
    # Target pose for reaching (x, y, z, roll, pitch, yaw)
    TARGET_POSE = np.array([0.2, 0.0, 0.2, -np.pi, 0, 0])
    # Reset pose - slightly offset from target
    RESET_POSE = np.array([0.3, 0.0, 0.48, -np.pi, 0, 0])
    # Reward threshold - 2cm for position, 0.1 rad for orientation
    REWARD_THRESHOLD = np.array([0.02, 0.02, 0.02, 0.1, 0.1, 0.1])
    # Action scale: [position_scale, rotation_scale, gripper_scale]
    ACTION_SCALE = np.array([0.02, 0.02, 1])
    # Enable random reset for diversity
    RANDOM_RESET = False
    DISPLAY_IMAGE = True
    # Random reset range
    RANDOM_XY_RANGE = 0.05
    RANDOM_RZ_RANGE = 0.1
    # Workspace limits
    ABS_POSE_LIMIT_HIGH = np.array([0.6, 0.2, 0.5, np.pi+0.5, 0.5, 0.5])
    ABS_POSE_LIMIT_LOW = np.array([0.3, -0.2, 0.1, np.pi-0.5, -0.5, -0.5])
    # Compliance parameters for smooth motion
    COMPLIANCE_PARAM = {
        "translational_stiffness": 1500,
        "translational_damping": 80,
        "rotational_stiffness": 100,
        "rotational_damping": 10,
        "translational_Ki": 0,
        "translational_clip_x": 0.01,
        "translational_clip_y": 0.01,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.01,
        "translational_clip_neg_y": 0.01,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.05,
        "rotational_clip_y": 0.05,
        "rotational_clip_z": 0.05,
        "rotational_clip_neg_x": 0.05,
        "rotational_clip_neg_y": 0.05,
        "rotational_clip_neg_z": 0.05,
        "rotational_Ki": 0,
    }
    # Precision parameters for reset
    PRECISION_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0.0,
        "translational_clip_x": 0.01,
        "translational_clip_y": 0.01,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.01,
        "translational_clip_neg_y": 0.01,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.03,
        "rotational_clip_y": 0.03,
        "rotational_clip_z": 0.03,
        "rotational_clip_neg_x": 0.03,
        "rotational_clip_neg_y": 0.03,
        "rotational_clip_neg_z": 0.03,
        "rotational_Ki": 0.0,
    }
    # MAX_EPISODE_LENGTH = 100
    # JOINT_RESET_PERIOD = 100  # Reset joints every 10 episodes


class TrainConfig(DefaultTrainingConfig):
    """Training configuration for HIROL reach task"""
    image_keys = ["side", "wrist_1"]
    classifier_keys = ["side"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    checkpoint_period = 2000
    cta_ratio = 2
    random_steps = 0
    discount = 0.98
    buffer_period = 1000
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        """Create and configure the HIROL reach environment"""
        env = HIROLReachEnv(
            fake_env=fake_env, save_video=save_video, config=EnvConfig()
        )
        if not fake_env:
            env = SpacemouseIntervention(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        
        if classifier:
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                """Binary reward classifier"""
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                logit = classifier(obs)
                # Handle both scalar and array outputs
                if hasattr(logit, 'shape') and logit.shape:
                    logit = logit.squeeze()
                    if logit.shape:  # Still has dimensions
                        logit = logit[0]
                return int(sigmoid(logit) > 0.5)

            from serl_hirol_infra.hirol_env.envs.wrappers import MultiCameraBinaryRewardClassifierWrapper
            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        
        return env