# HIROLDualArmEnv - Dual-Arm Environment for HIL-SERL

This document provides a quick guide to using the new HIROLDualArmEnv for dual-arm robotic tasks.

## Overview

HIROLDualArmEnv provides a unified interface for controlling two robot arms simultaneously, following the same design pattern as DualFrankaEnv. It combines two independent HIROLEnv instances to create a dual-arm environment compatible with existing HIL-SERL training infrastructure.

## Architecture

```
HIROLDualArmEnv
├── Left HIROLEnv → SerlRobotInterface → HIROLRobotPlatform (Left Arm)
└── Right HIROLEnv → SerlRobotInterface → HIROLRobotPlatform (Right Arm)
```

## Key Features

- **Parallel Control**: Left and right arms controlled simultaneously with thread-based execution
- **Compatible Format**: Observation and action spaces match DualFrankaEnv format
- **Modular Configuration**: Separate configs for left arm, right arm, and task coordination
- **Wrapper Support**: Compatible with all existing dual-arm wrappers
- **Multiple Task Types**: Built-in configs for handover, bimanual manipulation, etc.

## Quick Start

### Basic Usage

```python
from hirol_env.envs import HIROLDualArmEnv, DefaultDualArmConfig

# Create configuration
config = DefaultDualArmConfig()
dual_config = config.get_dual_arm_config()

# Initialize environment
env = HIROLDualArmEnv(**dual_config)

# Use like any Gym environment
obs, info = env.reset()
action = env.action_space.sample()  # 14D: [left_7dof, right_7dof]
obs, reward, done, truncated, info = env.step(action)

env.close()
```

### With Task-Specific Configuration

```python
from hirol_env.envs import HandoverTaskConfig, BimanualManipulationConfig

# For object handover tasks
handover_config = HandoverTaskConfig()
env = HIROLDualArmEnv(**handover_config.get_dual_arm_config())

# For bimanual manipulation tasks
bimanual_config = BimanualManipulationConfig()
env = HIROLDualArmEnv(**bimanual_config.get_dual_arm_config())
```

### With Wrappers

```python
from hirol_env.envs import DualQuat2EulerWrapper, DualSpacemouseIntervention

# Apply dual-arm wrappers
env = HIROLDualArmEnv(**dual_config)
env = DualQuat2EulerWrapper(env)  # Convert quaternions to euler angles
env = DualSpacemouseIntervention(env)  # Add spacemouse control
```

## Action and Observation Format

### Action Space
- **Format**: 14-dimensional vector
- **Structure**: `[left_arm_7dof, right_arm_7dof]`
- **Range**: [-1, 1] for each dimension

### Observation Space
```python
{
    "state": {
        "left/tcp_pose": np.ndarray[7],      # Left arm TCP pose
        "left/gripper_pos": np.ndarray[1],   # Left gripper position
        "right/tcp_pose": np.ndarray[7],     # Right arm TCP pose
        "right/gripper_pos": np.ndarray[1],  # Right gripper position
        # ... other left/ and right/ prefixed states
    },
    "images": {
        "left/camera_name": np.ndarray,      # Left arm cameras
        "right/camera_name": np.ndarray,     # Right arm cameras
        # ... other left/ and right/ prefixed images
    }
}
```

## Configuration Classes

### Available Configurations

1. **DefaultDualArmConfig**: Basic dual-arm setup
2. **HandoverTaskConfig**: Object handover between arms
3. **BimanualManipulationConfig**: Coordinated dual-arm manipulation
4. **DualArmSimConfig**: Simulation-only configuration

### Custom Configuration

```python
from hirol_env.envs.dual_arm_config import DefaultDualArmConfig

class MyTaskConfig(DefaultDualArmConfig):
    def __init__(self):
        super().__init__()

        # Customize left arm
        self.left_config.TARGET_POSE = np.array([0.6, 0.2, 0.3, 0, 0, 0])
        self.left_config.COMPLIANCE_PARAM.translational_stiffness = 1500.0

        # Customize right arm
        self.right_config.TARGET_POSE = np.array([0.6, -0.2, 0.3, 0, 0, 0])

        # Customize task behavior
        self.task_config.REWARD_STRATEGY = "both_succeed"
```

## Integration with Existing Code

### Replace DualFrankaEnv

HIROLDualArmEnv is designed as a drop-in replacement for DualFrankaEnv:

```python
# Before (DualFrankaEnv)
from serl_robot_infra.franka_env.envs.dual_franka_env import DualFrankaEnv
env = DualFrankaEnv(env_left, env_right)

# After (HIROLDualArmEnv)
from hirol_env.envs import HIROLDualArmEnv, DefaultDualArmConfig
config = DefaultDualArmConfig()
env = HIROLDualArmEnv(**config.get_dual_arm_config())
```

### Use with Existing Wrappers

All dual-arm wrappers from `serl_robot_infra` work with HIROLDualArmEnv:

```python
from hirol_env.envs.wrappers import (
    DualQuat2EulerWrapper,
    DualSpacemouseIntervention,
    DualGripperPenaltyWrapper
)

env = HIROLDualArmEnv(**dual_config)
env = DualQuat2EulerWrapper(env)
env = DualSpacemouseIntervention(env)
env = DualGripperPenaltyWrapper(env)
```

## Testing

```bash
# Run comprehensive tests
cd /home/hanyu/code/hil-serl/serl_hirol_infra
python test_hirol_dual_arm_env.py

# Run usage examples
python example_hirol_dual_arm_usage.py
```

## Robot Setup

### For Real Hardware

1. Configure robot IPs and parameters in your custom config class
2. Set up cameras in `REALSENSE_CAMERAS` dict
3. Adjust workspace limits and compliance parameters
4. Set `fake_env=False` in environment configs

### For Simulation

1. Use `DualArmSimConfig` or set `fake_env=True`
2. Disable cameras by setting `REALSENSE_CAMERAS = {}`
3. Test with the provided examples

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure HIROLRobotPlatform path is in `sys.path`
2. **Camera Issues**: Check serial numbers in `REALSENSE_CAMERAS` config
3. **Robot Connection**: Verify robot IPs and network connectivity
4. **Action Dimension**: Ensure actions are 14-dimensional arrays

### Debug Mode

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use simulation for testing
config = DualArmSimConfig()
env = HIROLDualArmEnv(**config.get_dual_arm_config())
```

## Next Steps

1. **Configure Hardware**: Set up your specific dual-arm robot configuration
2. **Train Policies**: Use with existing HIL-SERL training scripts
3. **Collect Data**: Integrate with data collection pipelines
4. **Deploy Models**: Use trained policies for real robot tasks

For more details, see the example files and test scripts in this directory.