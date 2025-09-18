"""Configuration for HIROL Fixed Gripper task using HIROLRobotPlatform"""
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
from experiments.hirol_online_classifier_fixed_gripper.wrapper import GripperPenaltyWrapper, HIROLFixedGripperEnv, GripperInitWrapper
from experiments.hirol_online_classifier_fixed_gripper.human_feedback_classifier_wrapper import (
    HumanFeedbackClassifierWrapper,
    AdaptiveClassifierWrapper
)
from experiments.hirol_online_classifier_fixed_gripper.online_feedback_wrapper import (
    OnlineFeedbackWrapper,
    SimplifiedHumanRewardWrapper
)
from hirol_env.envs.relative_env import RelativeFrame
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig


class EnvConfig(DefaultEnvConfig):
    """Configuration for HIROL Fixed Gripper task"""
    
    # Robot configuration file path (HIROLRobotPlatform uses config files)
    ROBOT_CONFIG_PATH = None  # Will use default serl_fr3_config.yaml if None
    
    # Camera configuration  
    REALSENSE_CAMERAS = {
        "wrist_1": {"serial_number": "332322073603"},
        "front": {"serial_number": "244222075350"},
        "side": {"serial_number": "243122071795"},
    }
    IMAGE_CROP = {
        "wrist_1": lambda img: img[:,:],
    "front": lambda img: img[0:480, 110:550],
    "side": lambda img: img[0:480, 170:640],
        
    }
    
    # Target pose for reaching (x, y, z, roll, pitch, yaw)
    # TARGET_POSE = np.array([0.5, 0.1, 0.3, -np.pi, 0, 0])
    
    # Reset pose - slightly offset from target
    RESET_POSE = np.array([0.5, 0.1, 0.45, -np.pi, 0, 0])
    
    # Reward threshold - 2cm for position, 0.1 rad for orientation
    REWARD_THRESHOLD = np.array([0.02, 0.02, 0.02, 0.1, 0.1, 0.1])
    
    # Action scale: [position_scale, rotation_scale, gripper_scale]
    # Optimized for SpaceMouse control (max output ~0.26 after scaling by 350)
    # Position: 0.26 * 0.02 = 5.2mm per frame, at 10Hz = 52mm/s max velocity
    # Rotation: 0.26 * 0.25 = 0.065 rad = 3.7 degrees per frame
    ACTION_SCALE = np.array([0.02, 0.06, 0])   #  gripper 0 or 1 ;Increased rotation scale for better responsiveness
    

      
    # Enable random reset for diversity
    RANDOM_RESET = False
    RANDOM_XY_RANGE = 0.05
    RANDOM_RZ_RANGE = 0.1
    
    # Workspace limits
    ABS_POSE_LIMIT_HIGH = np.array([0.7, 0.24, 0.55, np.pi, 0.5, +0.3*np.pi])
    ABS_POSE_LIMIT_LOW = np.array([0.30, 0, 0.34, np.pi, -0.5, -0.3*np.pi])
    
    # Display
    DISPLAY_IMAGE = True
    GRIPPER_SLEEP = 0.0
    MAX_EPISODE_LENGTH = 100
    # JOINT_RESET_PERIOD = 100  # Reset joints every 20 episodes
    
    # Gripper initial state for fixed gripper tasks
    # Options: "open", "close", "none"
    GRIPPER_INIT_STATE = "close"  # Change this to control initial gripper state


class TrainConfig(DefaultTrainingConfig):
    """Training configuration for HIROL Fixed Gripper task"""
    
    # Image and proprioception keys
    image_keys = ["side", "wrist_1","front"]
    classifier_keys = [
                        "side" ,
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
    
    # setup_mode = "single-arm-learned-gripper"
    setup_mode = "single-arm-fixed-gripper"
    
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
    
    def get_environment(self, fake_env=False, save_video=False, classifier=True, feedback_mode="offline"):
        """Create and configure the HIROL Fixed Gripper environment"""
        
        # Create base environment
        env = HIROLFixedGripperEnv(
            hz=10,
            fake_env=fake_env, 
            save_video=save_video, 
            config=EnvConfig()
        )
        env = GripperCloseEnv(env)  # Fixed gripper - always closed
        
        # Control initial gripper state: "open", "close", or "none"
        # Configure this in EnvConfig.GRIPPER_INIT_STATE
        env = GripperInitWrapper(env, init_mode=EnvConfig.GRIPPER_INIT_STATE)
        
        # Add spacemouse intervention for human demonstrations
        if not fake_env:
            env = SpacemouseIntervention(env, gripper_enabled=False)
        env = RelativeFrame(env)
        
        # Add wrappers
        env = Quat2EulerWrapper(env)
        # env = GripperPenaltyWrapper(env, penalty=-0.02)  # Add gripper penalty tracking
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        
        # Add reward/feedback wrapper based on mode
        if classifier:
            # feedback_mode options: "offline", "online", "simple", "keyboard"
            
            if feedback_mode == "simple":
                # Simplest option - just keyboard reward
                env = SimplifiedHumanRewardWrapper(env)
                print("[Config] Using SimplifiedHumanRewardWrapper (press 's' for success)")
                
            elif feedback_mode == "keyboard":
                # Original keyboard-only reward
                env = KeyboardRewardWrapper(env)
                print("[Config] Using KeyboardRewardWrapper")
                
            elif feedback_mode == "online":
                # Online feedback for use with train_rlpd_online_classifier.py
                try:
                    classifier_model = load_classifier_func(
                        key=jax.random.PRNGKey(0),
                        sample=env.observation_space.sample(),
                        image_keys=self.classifier_keys,
                        checkpoint_path=self.classifier_ckpt_path
                    )
                    
                    def reward_func(obs):
                        """Binary reward classifier"""
                        logit = classifier_model(obs)
                        if hasattr(logit, 'shape') and logit.shape:
                            logit = logit.squeeze()
                            if logit.shape:
                                logit = logit[0]
                        return logit
                    
                    env = OnlineFeedbackWrapper(
                        env,
                        classifier_func=reward_func,
                        confidence_threshold=0.85,
                        query_threshold=0.65,
                        auto_query=True
                    )
                    print("[Config] Using OnlineFeedbackWrapper for online learning")
                except Exception as e:
                    print(f"[Config] Failed to load classifier: {e}")
                    print("[Config] Falling back to SimplifiedHumanRewardWrapper")
                    env = SimplifiedHumanRewardWrapper(env)
                    
            else:  # offline (default)
                # Original offline human feedback with file saving
                try:
                    classifier_model = load_classifier_func(
                        key=jax.random.PRNGKey(0),
                        sample=env.observation_space.sample(),
                        image_keys=self.classifier_keys,
                        checkpoint_path=self.classifier_ckpt_path
                    )
                    
                    def reward_func(obs):
                        """Binary reward classifier"""
                        logit = classifier_model(obs)
                        if hasattr(logit, 'shape') and logit.shape:
                            logit = logit.squeeze()
                            if logit.shape:
                                logit = logit[0]
                        return logit
                    
                    env = HumanFeedbackClassifierWrapper(
                        env,
                        classifier_func=reward_func,
                        confidence_threshold=0.85,
                        query_threshold=0.65,
                        feedback_buffer_size=1000,
                        auto_retrain_interval=100,
                        save_feedback_path=str(self.data_path / "feedback_data")
                    )
                    print("[Config] Using HumanFeedbackClassifierWrapper (offline mode)")
                except Exception as e:
                    print(f"[Config] Failed to load classifier: {e}")
                    print("[Config] Falling back to KeyboardRewardWrapper")
                    env = KeyboardRewardWrapper(env)
        
        return env