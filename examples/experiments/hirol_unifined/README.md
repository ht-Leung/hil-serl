# HIROL Unified Task

This is a unified manipulation task framework using HIROLRobotPlatform and HIROLEnv.

## Overview

The HIROL Unified task demonstrates:
- Direct robot control using HIROLRobotPlatform's SerlRobotInterface
- Integration with HIL-SERL training pipeline
- Support for multiple robot platforms (FR3, Unitree G1, Monte01, AgiBot G1)
- Smooth trajectory control with 5th-order polynomial interpolation
- Flexible task configuration for various manipulation objectives

## Key Features

- **Unified Interface**: Uses SerlRobotInterface for consistent control across different robot platforms
- **Advanced Motion Control**: 
  - 5th-order polynomial interpolation for smooth trajectories
  - Critical damped tracking for compliant motion
  - Configurable compliance parameters
- **Multi-Robot Support**: Works with any robot supported by HIROLRobotPlatform
- **Force/Torque Sensing**: Integrated support for both internal (FR3) and external force sensors

## Configuration

Edit `config.py` to modify task parameters:

```python
# Robot configuration (uses HIROLRobotPlatform config files)
ROBOT_CONFIG_PATH = None  # Uses default serl_fr3_config.yaml

# Task definition
TARGET_POSE = np.array([0.5, 0.1, 0.3, -np.pi, 0, 0])  # Target position
RESET_POSE = np.array([0.5, 0.0, 0.4, -np.pi, 0, 0])   # Reset position
REWARD_THRESHOLD = np.array([0.02, 0.02, 0.02, 0.1, 0.1, 0.1])  # Success threshold

# Control parameters
ACTION_SCALE = np.array([0.02, 0.02, 1.0])  # Scale for position, rotation, gripper
```

## Robot Configuration

The robot is configured through HIROLRobotPlatform's YAML files. Default configuration is in:
`/home/hanyu/code/HIROLRobotPlatform/factory/tasks/inferences_tasks/serl/config/serl_fr3_config.yaml`

To use a custom configuration:
```python
# In config.py
ROBOT_CONFIG_PATH = "/path/to/your/custom_config.yaml"
```

## Running the Task

### 1. Set Environment Variables

```bash
# HIROLRobotPlatform environment
source /home/hanyu/code/HIROLRobotPlatform/dependencies/a2d_sdk/env.sh

# Python paths
export PYTHONPATH=$PYTHONPATH:/home/hanyu/code/hil-serl
export PYTHONPATH=$PYTHONPATH:/home/hanyu/code/HIROLRobotPlatform

# Activate conda environment
conda activate hilserl
```

### 2. Test the Environment

```bash
cd /home/hanyu/code/hil-serl/examples/experiments/hirol_unifined
python -c "from config import TrainConfig; env = TrainConfig().get_environment(); print('Environment created successfully')"
```

### 3. Start Training

In two separate terminals:

**Terminal 1 - Actor (Data Collection):**
```bash
cd /home/hanyu/code/hil-serl/examples/experiments/hirol_unifined
./run_actor.sh
```

**Terminal 2 - Learner (Policy Training):**
```bash
cd /home/hanyu/code/hil-serl/examples/experiments/hirol_unifined
./run_learner.sh
```

### 4. Human Demonstrations

#### Recording Demonstrations
```bash
cd /home/hanyu/code/hil-serl/examples
python record_demos.py --exp_name=hirol_unifined --successes_needed=20
```

#### Recording Success/Failure for Classifier
```bash
python record_success_fail.py --exp_name=hirol_unifined --successes_needed=200
python train_reward_classifier.py --exp_name=hirol_unifined
```

## Architecture

```
HIROLRobotPlatform
    ↓
SerlRobotInterface (Unified Interface)
    ↓
HIROLEnv (Gym Environment)
    ↓
Environment Wrappers
    ↓
HIL-SERL Training Pipeline
```

## Advanced Features

### 5th-Order Polynomial Trajectory

The environment uses 5th-order polynomial interpolation for smooth motion:
- Zero initial and final velocity/acceleration
- Smooth acceleration profile
- Configurable control frequency

### Force/Torque Integration

Supports multiple force sensing modes:
- FR3 internal force estimation (O_F_ext_hat_K)
- External ATI force sensors
- Automatic fallback between sensing modes

### Keyboard Controls

During operation:
- **ESC**: Emergency stop
- **G**: Recover gripper (press twice if needed)

### Compliance Modes

Three compliance parameter sets for different phases:
- **COMPLIANCE_PARAM**: Normal operation
- **PRECISION_PARAM**: Precise movements during reset
- **RESET_PARAM**: Transition movements

## Multi-Robot Support

To use with different robots, modify the YAML configuration:

### For Unitree G1:
```yaml
robot: "unitree_g1"
robot_config:
  unitree_g1:
    # Unitree G1 specific config
```

### For Monte01:
```yaml
robot: "monte01"
robot_config:
  monte01:
    # Monte01 specific config
```

## Troubleshooting

### Robot Connection Issues
```bash
# Check FR3 connection
ping 172.16.0.2  # or your robot's IP

# Verify ROS2 environment (for some robots)
source /home/hanyu/code/HIROLRobotPlatform/dependencies/a2d_sdk/env.sh
```

### Import Errors
Ensure all paths are set:
```bash
export PYTHONPATH=$PYTHONPATH:/home/hanyu/code/hil-serl
export PYTHONPATH=$PYTHONPATH:/home/hanyu/code/HIROLRobotPlatform
export PYTHONPATH=$PYTHONPATH:/home/hanyu/code/hil-serl/serl_hirol_infra
```

### Force/Torque Not Reading
- Check if using FR3 with panda-py (internal sensing)
- Verify external sensor configuration in YAML
- Ensure sensor IP is reachable

### Camera Issues
- Verify RealSense serial numbers in config
- Check USB3 connections
- Try `rs-enumerate-devices` to list cameras

## Performance Tips

1. **Control Frequency**: Adjust `hz` parameter based on your task (10-20 Hz typical)
2. **Smoother Settings**: Enable in YAML for smoother but slower motion
3. **Async Control**: Enable for high-frequency control loops
4. **Memory Management**: Use `XLA_PYTHON_CLIENT_PREALLOCATE=false`

## Next Steps

1. Configure robot-specific parameters
2. Collect demonstrations for your task
3. Train reward classifier if using vision-based rewards
4. Run HIL-SERL training
5. Evaluate and fine-tune with human corrections