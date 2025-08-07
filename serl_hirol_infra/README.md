# SERL HIROL Infrastructure

This package provides the HIROL implementation of robot interfaces for SERL (Sample Efficient Robotic Learning).

## HIROLInterface

`HIROLInterface` is an implementation of the abstract `RobotInterface` that uses HIROLRobotPlatform's `RobotFactory` and `MotionFactory` for robot control.

### Features

- **Real-time robot state retrieval**: TCP pose, velocity, forces/torques, joint states, Jacobian
- **Position control**: Using IK controller with trajectory planning
- **Gripper control**: Binary and continuous modes
- **External force/torque sensing**: Access to FR3's `K_F_ext_hat_K` data
- **Motion planning**: Integrated with HIROLRobotPlatform's 5th-order polynomial trajectory planner

### Usage

```python
from serl_hirol_infra.interface.hirol_interface import HIROLInterface

# Create interface with configuration
interface = HIROLInterface(
    robot_config=robot_config,
    motion_config=motion_config,
    gripper_sleep=0.6,
    gripper_range=(0.0, 0.08)
)

# Get robot state
state = interface.get_state()
print(f"TCP pose: {state['pose']}")
print(f"TCP velocity: {state['vel']}")
print(f"External forces: {state['force']}")
print(f"Jacobian: {state['jacobian']}")

# Control robot
interface.send_pos_command(target_pose)  # 7D pose [x,y,z,qx,qy,qz,qw]
interface.open_gripper()
interface.send_gripper_command(-0.5, mode="binary")  # Close gripper

# Clean up
interface.close()
```

### Configuration

The interface requires two configuration dictionaries:

1. **robot_config**: Robot hardware/simulation settings
2. **motion_config**: Motion control settings (model, controller, trajectory planner)

See `config/hirol_interface_config_example.py` for detailed configuration examples.

### Testing

Run the test script to verify the interface:

```bash
cd serl_hirol_infra
python test_hirol_interface.py
```

### Integration with FrankaEnv

To use `HIROLInterface` with FrankaEnv, replace HTTP API calls with the interface:

```python
# Instead of using HTTP requests
# requests.post(self.url + "pose", json=data)

# Use HIROLInterface
self.robot_interface = HIROLInterface(robot_config, motion_config)
self.robot_interface.send_pos_command(pose)
```

### Notes

- Compliance control parameters are not yet implemented (position control only)
- External force/torque data requires FR3 hardware with force sensing
- The interface automatically handles coordinate transformations and data normalization