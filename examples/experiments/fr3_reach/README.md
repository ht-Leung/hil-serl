# FR3 Reach Task

This is a simple reaching task for the Franka FR3 robot using the new FrankaInterface.

## Overview

The FR3 reach task demonstrates:
- Direct robot control using FrankaInterface (no Flask server needed)
- Integration with HIL-SERL training pipeline
- Smooth trajectory control with compliance
- Binary reward based on reaching target position

## Key Features

- **Direct Control**: Uses FrankaInterface to directly control the FR3 robot without intermediate servers
- **Two Motion Modes**: 
  - Servo mode for fast reactive control during training
  - Trajectory mode for smooth motions during reset
- **Compliance Control**: Configurable compliance parameters for safe interaction
- **Gripper Integration**: Full gripper control using FrankaHand

## Configuration

Edit `config.py` to modify task parameters:

```python
# Robot connection
ROBOT_IP = "192.168.1.206"  # FR3 robot IP

# Task definition
TARGET_POSE = np.array([0.5, 0.0, 0.3, -np.pi, 0, 0])  # Target position
RESET_POSE = np.array([0.5, 0.0, 0.4, -np.pi, 0, 0])   # Reset position
REWARD_THRESHOLD = np.array([0.02, 0.02, 0.02, 0.1, 0.1, 0.1])  # Success threshold

# Control parameters
ACTION_SCALE = np.array([0.05, 0.1, 1.0])  # Scale for position, rotation, gripper
```

## Running the Task

### 1. Test the Environment

First, test that the environment works correctly:

```bash
cd /home/hanyu/code/hil-serl/serl_hirol_infra/hirol_env/envs
python test_fr3_env.py --test all
```

### 2. Start Training

In two separate terminals:

**Terminal 1 - Actor (Data Collection):**
```bash
cd /home/hanyu/code/hil-serl/examples/experiments/fr3_reach
chmod +x run_actor.sh
./run_actor.sh
```

**Terminal 2 - Learner (Policy Training):**
```bash
cd /home/hanyu/code/hil-serl/examples/experiments/fr3_reach
chmod +x run_learner.sh
./run_learner.sh
```

### 3. Human Demonstrations

When the actor is running, you can provide demonstrations using:
- SpaceMouse for teleoperation
- Keyboard interventions during autonomous execution

#### Recording Demonstrations

**Option 1: Record continuous demonstrations**
```bash
# Set environment variables
export PYTHONPATH=$PYTHONPATH:/home/hanyu/code/hil-serl
export PYTHONPATH=$PYTHONPATH:/home/hanyu/code/HIROLRobotPlatform

# Navigate to examples directory
cd /home/hanyu/code/hil-serl/examples

# Record demonstrations with SpaceMouse (collects 20 successful demos by default)
python record_demos.py --exp_name=fr3_reach --successes_needed=20
```

**Option 2: Record success/failure episodes for classifier training（before demo recording）**
```bash
# Navigate to examples directory
cd /home/hanyu/code/hil-serl/examples

# Record and label episodes as success or failure (collects 200 transitions by default)
python record_success_fail.py --exp_name=fr3_reach --successes_needed=200

# Train the reward classifier using collected data
python train_reward_classifier.py --exp_name=fr3_reach  #--num_epochs=150 --batch_size=256
```

During recording:
- **record_demos.py**: 
  - Use SpaceMouse to control the robot
  - The script automatically detects successful task completion
  - Collects full trajectories until reaching the target number of successes
  
- **record_success_fail.py**:
  - Use SpaceMouse to control the robot
  - Press 'Space' to mark current state as success
  - States without space press are marked as failures
  - Used for training the success classifier

The demonstrations will be saved to:
- `./demos/` - Full trajectory demonstrations from record_demos.py
- `./classifier_data/` - Success/failure labeled transitions from record_success_fail.py
- `./classifier_ckpt/` - Trained classifier checkpoint after running train_reward_classifier.py

## Architecture

```
FR3 Robot Hardware
        ↓
  FrankaInterface (Direct Control)
        ↓
    FR3Env (Gym Interface)
        ↓
  Environment Wrappers
        ↓
  HIL-SERL Training Pipeline
```

## Advantages Over Server-Based Approach

1. **Lower Latency**: Direct control without HTTP overhead
2. **Better Reliability**: No network issues or server crashes
3. **Cleaner Architecture**: Single process for robot control
4. **Easier Debugging**: All code runs in one place
5. **Full Feature Access**: Direct access to all FR3Interface capabilities

## Compliance Parameters

The environment uses different compliance settings for different phases:

- **COMPLIANCE_PARAM**: Normal operation (softer, safer)
- **PRECISION_PARAM**: Reset movements (stiffer, more accurate)
- **RESET_PARAM**: Intermediate stiffness for transitions

## Safety Features

- Workspace limits enforced by `clip_safety_box()`
- Automatic error recovery with `clear_errors()`
- Gripper safety delays to prevent damage
- ESC key for emergency stop

## Troubleshooting

### Robot Not Connecting
- Check robot IP: `ping 192.168.3.102`
- Ensure robot is in execution mode
- Verify network configuration

### Import Errors
```bash
export PYTHONPATH=$PYTHONPATH:/home/hanyu/code/hil-serl
export PYTHONPATH=$PYTHONPATH:/home/hanyu/code/HIROLRobotPlatform
```

### Camera Issues
- Check RealSense serial numbers match your setup
- Verify USB3 connection for cameras
- Try unplugging and reconnecting cameras

## Next Steps

1. Collect human demonstrations
2. Train policy with HIL-SERL
3. Evaluate learned behavior
4. Fine-tune with human corrections
5. Deploy for autonomous execution