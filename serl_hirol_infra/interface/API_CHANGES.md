# FrankaInterface API Changes

## Position Control Methods

The interface now provides two distinct position control methods for different use cases:

### 1. `send_pos_command(pose)` - Servo Mode (Default)
- **Purpose**: Fast, direct control for reactive applications
- **Mode**: Servo (no trajectory planning)
- **Response**: Immediate
- **Use Cases**: 
  - Real-time control loops
  - Reactive behaviors
  - Fast point-to-point movements
  - Teleopertion

### 2. `send_pos_trajectory_command(pose, finish_time=2.0)` - Trajectory Mode
- **Purpose**: Smooth, planned motions
- **Mode**: Trajectory with planning
- **Response**: Smooth acceleration/deceleration
- **Use Cases**:
  - Predictable motions
  - Pick and place operations
  - Demonstration recording
  - When smooth motion is critical

## Example Usage

```python
from franka_interface import FrankaInterface
import numpy as np

robot = FrankaInterface()

# Get current pose
state = robot.get_state()
current_pose = state['pose']

# Define target
target_pose = current_pose.copy()
target_pose[0] += 0.1  # Move 10cm in X

# Method 1: Fast servo control (reactive)
robot.send_pos_command(target_pose)

# Method 2: Smooth trajectory (planned)
robot.send_pos_trajectory_command(target_pose, finish_time=3.0)
```

## Impedance Control

The `move_to_pose_impedance()` method also supports both modes:

```python
# Servo impedance (direct compliant control)
robot.move_to_pose_impedance(
    pose=target_pose,
    compliance_params=soft_params,
    mode="servo",
    duration=2.0  # Control duration
)

# Trajectory impedance (smooth compliant motion)
robot.move_to_pose_impedance(
    pose=target_pose,
    compliance_params=soft_params,
    mode="trajectory",
    duration=3.0  # Trajectory completion time
)
```

## Benefits

1. **Clear Intent**: Method names clearly indicate behavior
2. **Optimal Performance**: Choose speed vs smoothness based on task
3. **Backward Compatible**: Existing code using `send_pos_command` still works
4. **Flexible**: Easy to switch between modes as needed

## Migration Guide

If you were previously using:
```python
# Old: Always used one mode
robot.send_pos_command(pose)
```

Now you can choose:
```python
# Fast reactive control
robot.send_pos_command(pose)

# OR smooth planned motion
robot.send_pos_trajectory_command(pose, finish_time=2.0)
```