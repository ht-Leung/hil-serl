"""Configuration for HIROL Unified task using HIROLRobotPlatform"""
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
)
from experiments.hirol_unifined.wrapper import GripperPenaltyWrapper, HIROLUnifiedEnv
from hirol_env.envs.relative_env import RelativeFrame
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig


class EnvConfig(DefaultEnvConfig):
    """Configuration for HIROL Unified task"""
    
    # Robot configuration file path (HIROLRobotPlatform uses config files)
    ROBOT_CONFIG_PATH = None  # Will use default serl_fr3_config.yaml if None
    
    # Camera configuration  
    REALSENSE_CAMERAS = {
        "wrist_1": {"serial_number": "332322073603"},
        "front": {"serial_number": "244222075350"},
        "side": {"serial_number": "317422071787"},
    }
    IMAGE_CROP = {
        "wrist_1": lambda img: img[:,:],
        "front": lambda img: img[90:450, 140:520],
        "side": lambda img: img[130:450, 180:640],
        
    }
    
    # Target pose for reaching (x, y, z, roll, pitch, yaw)
    # TARGET_POSE = np.array([0.5, 0.1, 0.3, -np.pi, 0, 0])
    
    # Reset pose - slightly offset from target
    RESET_POSE = np.array([0.5, 0.1, 0.32, -np.pi, 0, 0])
    
    # Reward threshold - 2cm for position, 0.1 rad for orientation
    REWARD_THRESHOLD = np.array([0.02, 0.02, 0.02, 0.1, 0.1, 0.1])
    
    # Action scale: [position_scale, rotation_scale, gripper_scale]
    # Optimized for SpaceMouse control (max output ~0.26 after scaling by 350)
    # Position: 0.26 * 0.02 = 5.2mm per frame, at 10Hz = 52mm/s max velocity
    # Rotation: 0.26 * 0.25 = 0.065 rad = 3.7 degrees per frame
    ACTION_SCALE = np.array([0.02, 0.06, 1])  # Increased rotation scale for better responsiveness
    

      
    # Enable random reset for diversity
    RANDOM_RESET = False
    RANDOM_XY_RANGE = 0.05
    RANDOM_RZ_RANGE = 0.1
    
    # Workspace limits
    ABS_POSE_LIMIT_HIGH = np.array([0.7, 0.24, 0.45, np.pi, 0.5, +0.3*np.pi])
    ABS_POSE_LIMIT_LOW = np.array([0.30, 0, 0.22, np.pi, -0.5, -0.3*np.pi])
    
    # Display
    DISPLAY_IMAGE = True
    GRIPPER_SLEEP = 0.0
    MAX_EPISODE_LENGTH = 120
    # JOINT_RESET_PERIOD = 100  # Reset joints every 20 episodes


class TrainConfig(DefaultTrainingConfig):
    """Training configuration for HIROL Unified task"""
    
    # Image and proprioception keys
    image_keys = ["side", "wrist_1","front"]
    classifier_keys = [
                        # "side" ,
                       "wrist_1",
                       "front"
                       ] 
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    
    # Training parameters
    checkpoint_period = 2000
    cta_ratio = 2
    random_steps = 0
    discount = 0.97
    buffer_period = 1000
    encoder_type = "resnet-pretrained"
    
    setup_mode = "single-arm-learned-gripper"
    # setup_mode = "single-arm-fixed-gripper"
    
    # Data paths - all relative to experiments/fr3_reach/
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
        """Create and configure the HIROL Unified environment"""
        
        # Create base environment
        env = HIROLUnifiedEnv(
            hz=10,
            fake_env=fake_env, 
            save_video=save_video, 
            config=EnvConfig()
        )
        
        # Add spacemouse intervention for human demonstrations
        if not fake_env:
            env = SpacemouseIntervention(env, gripper_enabled=True)
        env = RelativeFrame(env)
        
        # Add wrappers
        env = Quat2EulerWrapper(env)
        env = GripperPenaltyWrapper(env, penalty=-0.02)  # Add gripper penalty tracking
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        
        # Add keyboard reward if requested (press 's' to give reward)
        if classifier:
            # env = KeyboardRewardWrapper(env)
            
            # 如果用视觉分类器作为奖励函数，可以使用以下代码
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=self.classifier_ckpt_path
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
                return int(sigmoid(logit) > 0.7) # 0.65/0.75



            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        
        return env