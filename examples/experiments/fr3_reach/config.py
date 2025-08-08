"""Configuration for FR3 reach task using FrankaInterface"""
import os
import sys
import jax
import numpy as np
import jax.numpy as jnp

# Add serl_hirol_infra to path
sys.path.insert(0, '/home/hanyu/code/hil-serl/serl_hirol_infra')

from hirol_env.envs.fr3_env import DefaultEnvConfig, FR3Env
from hirol_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
)
from hirol_env.envs.relative_env import RelativeFrame
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig


class EnvConfig(DefaultEnvConfig):
    """Configuration for FR3 reach task"""
    
    # Robot IP
    ROBOT_IP = "192.168.3.102"
    
    # Camera configuration  
    REALSENSE_CAMERAS = {
        "wrist_1": {"serial_number": "332322073603"},
        "side": {"serial_number": "244222075350"},
    }
    IMAGE_CROP = {
        # "wrist": lambda img: img[:, 250:],
        # "side": lambda img: img[100:500, 150:1100],
    }
    
    # Target pose for reaching (x, y, z, roll, pitch, yaw)
    # TARGET_POSE = np.array([0.1, 0.0, 0.3, -np.pi, 0, 0])
    
    # Reset pose - slightly offset from target
    RESET_POSE = np.array([0.5, 0.0, 0.3, -np.pi, 0, 0])
    
    # Reward threshold - 2cm for position, 0.1 rad for orientation
    REWARD_THRESHOLD = np.array([0.02, 0.02, 0.02, 0.1, 0.1, 0.1])
    
    # Action scale: [position_scale, rotation_scale, gripper_scale]
    # Optimized for SpaceMouse control (max output ~0.26 after scaling by 350)
    # Position: 0.26 * 0.02 = 5.2mm per frame, at 10Hz = 52mm/s max velocity
    ACTION_SCALE = np.array([0.01, 0.00, 1])  # Slightly increased for responsiveness
    
    # Critical Damped Tracker parameters (二阶临界阻尼跟踪器)
    # Natural frequency (rad/s) - controls tracking responsiveness
    # Lower values (15-20): More compliant, slower response
    # Medium values (20-30): Balanced (default 25)
    # Higher values (30-40): Faster response, stiffer
    # Formula: settling_time ≈ 4.6 / omega_n
    TRACKER_OMEGA_N = 25.0  # Default: 0.18s settling time
      
    # Enable random reset for diversity
    RANDOM_RESET = False
    RANDOM_XY_RANGE = 0.05
    RANDOM_RZ_RANGE = 0.1
    
    # Workspace limits
    ABS_POSE_LIMIT_HIGH = np.array([0.7, 0.3, 0.45, np.pi+0.5, 0.5, 0.5])
    ABS_POSE_LIMIT_LOW = np.array([0.3, -0.3, 0.2, np.pi-0.5, -0.5, -0.5])
    
    # Display
    DISPLAY_IMAGE = True
    GRIPPER_SLEEP = 0.1
    MAX_EPISODE_LENGTH = 200
    # JOINT_RESET_PERIOD = 100  # Reset joints every 20 episodes


class TrainConfig(DefaultTrainingConfig):
    """Training configuration for FR3 reach task"""
    
    # Image and proprioception keys
    image_keys = ["side", "wrist_1"]
    classifier_keys = ["side", "wrist_1"]  # Use both cameras for classifier
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    
    # Training parameters
    checkpoint_period = 2000
    cta_ratio = 2
    random_steps = 0
    discount = 0.98
    buffer_period = 1000
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"
    
    def get_environment(self, fake_env=False, save_video=False, classifier=True):
        """Create and configure the FR3 reach environment"""
        
        # Create base environment
        env = FR3Env(
            hz=10,
            fake_env=fake_env, 
            save_video=save_video, 
            config=EnvConfig()
        )
        
        # Add spacemouse intervention for human demonstrations
        if not fake_env:
            env = SpacemouseIntervention(env)
        env = RelativeFrame(env)
        
        # Add wrappers
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        
        # Add classifier-based reward if requested
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

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        
        return env