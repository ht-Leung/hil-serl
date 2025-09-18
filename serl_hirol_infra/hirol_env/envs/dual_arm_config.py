"""Configuration classes for HIROLDualArmEnv"""
import numpy as np
from typing import Dict, Any, Optional
from .hirol_env import DefaultEnvConfig
from factory.tasks.inferences_tasks.serl.serl_robot_interface import ComplianceParams


class LeftArmConfig(DefaultEnvConfig):
    """Configuration for left arm in dual-arm setup"""

    # Robot configuration - uses left arm HIROL config
    ROBOT_CONFIG_PATH: Optional[str] = None  # Will use default config if None

    # Left arm cameras
    REALSENSE_CAMERAS: Dict = {
        "left_wrist": {
            "serial_number": "332322073603",  # Example serial number
            "dim": (1280, 720),
            "exposure": 10500,
            "gain": 16,
            "fps": 30
        },
        "left_external": {
            "serial_number": "332322073604",  # Example serial number
            "dim": (1280, 720),
            "exposure": 10500,
            "gain": 16,
            "fps": 30
        }
    }

    # Image cropping functions for left arm
    IMAGE_CROP: dict[str, callable] = {
        "left_wrist": lambda x: x[100:580, 200:1080],  # Example crop
        "left_external": lambda x: x[50:670, 150:1130]  # Example crop
    }

    # Left arm task parameters
    TARGET_POSE: np.ndarray = np.array([0.5, 0.2, 0.3, 0, 0, 0])  # Left workspace
    GRASP_POSE: np.ndarray = np.array([0.5, 0.2, 0.2, 0, 0, 0])
    REWARD_THRESHOLD: np.ndarray = np.array([0.02, 0.02, 0.02, 0.1, 0.1, 0.1])
    ACTION_SCALE = np.array([0.05, 0.05, 0.05])  # Conservative scaling
    RESET_POSE = np.array([0.5, 0.2, 0.4, 0, 0, 0])  # Safe reset position

    # Left arm workspace limits
    ABS_POSE_LIMIT_HIGH = np.array([0.7, 0.4, 0.6, 0.5, 0.5, 3.14])
    ABS_POSE_LIMIT_LOW = np.array([0.3, 0.0, 0.1, -0.5, -0.5, -3.14])

    # Left arm compliance parameters
    COMPLIANCE_PARAM: ComplianceParams = ComplianceParams(
        translational_stiffness=1800.0,  # Slightly lower for left arm
        translational_damping=85.0,
        rotational_stiffness=120.0,
        rotational_damping=8.0,
    )


class RightArmConfig(DefaultEnvConfig):
    """Configuration for right arm in dual-arm setup"""

    # Robot configuration - uses right arm HIROL config
    ROBOT_CONFIG_PATH: Optional[str] = None  # Will use default config if None

    # Right arm cameras
    REALSENSE_CAMERAS: Dict = {
        "right_wrist": {
            "serial_number": "332322073605",  # Example serial number
            "dim": (1280, 720),
            "exposure": 10500,
            "gain": 16,
            "fps": 30
        },
        "right_external": {
            "serial_number": "332322073606",  # Example serial number
            "dim": (1280, 720),
            "exposure": 10500,
            "gain": 16,
            "fps": 30
        }
    }

    # Image cropping functions for right arm
    IMAGE_CROP: dict[str, callable] = {
        "right_wrist": lambda x: x[100:580, 200:1080],  # Example crop
        "right_external": lambda x: x[50:670, 150:1130]  # Example crop
    }

    # Right arm task parameters
    TARGET_POSE: np.ndarray = np.array([0.5, -0.2, 0.3, 0, 0, 0])  # Right workspace
    GRASP_POSE: np.ndarray = np.array([0.5, -0.2, 0.2, 0, 0, 0])
    REWARD_THRESHOLD: np.ndarray = np.array([0.02, 0.02, 0.02, 0.1, 0.1, 0.1])
    ACTION_SCALE = np.array([0.05, 0.05, 0.05])  # Conservative scaling
    RESET_POSE = np.array([0.5, -0.2, 0.4, 0, 0, 0])  # Safe reset position

    # Right arm workspace limits
    ABS_POSE_LIMIT_HIGH = np.array([0.7, 0.0, 0.6, 0.5, 0.5, 3.14])
    ABS_POSE_LIMIT_LOW = np.array([0.3, -0.4, 0.1, -0.5, -0.5, -3.14])

    # Right arm compliance parameters
    COMPLIANCE_PARAM: ComplianceParams = ComplianceParams(
        translational_stiffness=1800.0,  # Slightly lower for right arm
        translational_damping=85.0,
        rotational_stiffness=120.0,
        rotational_damping=8.0,
    )


class DualArmTaskConfig:
    """Task-specific configuration for dual-arm operations"""

    # Coordination parameters
    SYNC_RESET: bool = True
    SYNC_STEP: bool = True
    DISPLAY_DUAL_IMAGES: bool = True

    # Task types
    TASK_TYPE: str = "bimanual_manipulation"  # "bimanual_manipulation", "handover", "coordination"

    # Reward combination strategy
    REWARD_STRATEGY: str = "both_succeed"  # "both_succeed", "either_succeed", "weighted_sum"

    # Termination conditions
    TERMINATION_STRATEGY: str = "either_arm"  # "either_arm", "both_arms", "task_specific"

    # Dual-arm specific compliance (for coordination tasks)
    COORDINATION_COMPLIANCE: ComplianceParams = ComplianceParams(
        translational_stiffness=1500.0,  # Lower stiffness for coordination
        translational_damping=80.0,
        rotational_stiffness=100.0,
        rotational_damping=10.0,
    )


class DefaultDualArmConfig:
    """Default configuration for dual-arm HIROL environment"""

    def __init__(self):
        # Create left and right arm configurations
        self.left_config = LeftArmConfig()
        self.right_config = RightArmConfig()
        self.task_config = DualArmTaskConfig()

    def get_left_env_config(self) -> Dict[str, Any]:
        """Get configuration dict for left arm HIROLEnv"""
        return {
            "hz": 10,
            "fake_env": False,
            "save_video": False,
            "config": self.left_config,
            "set_load": False,
            "display_image": False  # Disable individual display, use dual display
        }

    def get_right_env_config(self) -> Dict[str, Any]:
        """Get configuration dict for right arm HIROLEnv"""
        return {
            "hz": 10,
            "fake_env": False,
            "save_video": False,
            "config": self.right_config,
            "set_load": False,
            "display_image": False  # Disable individual display, use dual display
        }

    def get_dual_arm_config(self) -> Dict[str, Any]:
        """Get configuration dict for HIROLDualArmEnv"""
        return {
            "left_env_config": self.get_left_env_config(),
            "right_env_config": self.get_right_env_config(),
            "display_images": self.task_config.DISPLAY_DUAL_IMAGES
        }


# Specific task configurations
class HandoverTaskConfig(DefaultDualArmConfig):
    """Configuration for object handover between arms"""

    def __init__(self):
        super().__init__()

        # Handover-specific parameters
        self.task_config.TASK_TYPE = "handover"
        self.task_config.REWARD_STRATEGY = "either_succeed"  # Either arm can succeed

        # Adjust arm positions for handover
        self.left_config.TARGET_POSE = np.array([0.45, 0.1, 0.3, 0, 0, 0])
        self.right_config.TARGET_POSE = np.array([0.55, -0.1, 0.3, 0, 0, 0])

        # Lower stiffness for smooth handover
        handover_compliance = ComplianceParams(
            translational_stiffness=1200.0,
            translational_damping=70.0,
            rotational_stiffness=80.0,
            rotational_damping=6.0,
        )
        self.left_config.COMPLIANCE_PARAM = handover_compliance
        self.right_config.COMPLIANCE_PARAM = handover_compliance


class BimanualManipulationConfig(DefaultDualArmConfig):
    """Configuration for bimanual manipulation tasks"""

    def __init__(self):
        super().__init__()

        # Bimanual-specific parameters
        self.task_config.TASK_TYPE = "bimanual_manipulation"
        self.task_config.REWARD_STRATEGY = "both_succeed"  # Both arms must succeed

        # Symmetric positions for bimanual manipulation
        self.left_config.TARGET_POSE = np.array([0.5, 0.15, 0.25, 0, 0, 0])
        self.right_config.TARGET_POSE = np.array([0.5, -0.15, 0.25, 0, 0, 0])

        # Higher stiffness for precise coordination
        bimanual_compliance = ComplianceParams(
            translational_stiffness=2200.0,
            translational_damping=95.0,
            rotational_stiffness=180.0,
            rotational_damping=12.0,
        )
        self.left_config.COMPLIANCE_PARAM = bimanual_compliance
        self.right_config.COMPLIANCE_PARAM = bimanual_compliance


# Simulation configurations
class DualArmSimConfig(DefaultDualArmConfig):
    """Configuration for dual-arm simulation environment"""

    def __init__(self):
        super().__init__()

        # Disable cameras for simulation
        self.left_config.REALSENSE_CAMERAS = {}
        self.right_config.REALSENSE_CAMERAS = {}

    def get_left_env_config(self) -> Dict[str, Any]:
        config = super().get_left_env_config()
        config["fake_env"] = True  # Use simulation
        return config

    def get_right_env_config(self) -> Dict[str, Any]:
        config = super().get_right_env_config()
        config["fake_env"] = True  # Use simulation
        return config