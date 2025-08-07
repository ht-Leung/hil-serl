#!/usr/bin/env python3
"""
Test script for HIROLInterface implementation
Uses real configuration from HIROLRobotPlatform
"""

import numpy as np
import time
import sys
import os
from pathlib import Path

# Add HIROLRobotPlatform to path
hirol_platform_path = Path(__file__).parent.parent.parent / "HIROLRobotPlatform"

# Set working directory to HIROLRobotPlatform to ensure relative imports work
original_cwd = os.getcwd()
os.chdir(str(hirol_platform_path))

# Add both paths to ensure imports work
sys.path.insert(0, str(hirol_platform_path))
sys.path.insert(0, str(hirol_platform_path.parent))

from interface.hirol_interface import HIROLInterface
from hardware.base.utils import dynamic_load_yaml

# Restore original working directory after imports
os.chdir(original_cwd)


def test_hirol_interface():
    """Test HIROLInterface functionality with real configuration"""
    
    # Load real configuration from HIROLRobotPlatform
    # Using franka_3d_mouse.yaml as base configuration
    config_path = hirol_platform_path / "teleop/config/franka_3d_mouse.yaml"
    
    # Change to teleop directory for loading config with relative paths
    original_cwd = os.getcwd()
    os.chdir(str(hirol_platform_path / "teleop"))
    config = dynamic_load_yaml(str(config_path))
    os.chdir(original_cwd)
    
    # Override some settings for testing
    config['use_hardware'] = True  # Start with simulation for safety
    config['use_simulation'] = False
    
    # Extract robot and motion configurations
    robot_config = {
        "robot": config['robot'],  
        "use_hardware": config['use_hardware'],
        "use_simulation": config['use_simulation'],
        "robot_config": config['robot_config'],
        "gripper": config['gripper'],  # Changed from tool_type to gripper
        "gripper_config": config['gripper_config'],  # Changed from tool_config to gripper_config
        "simulation": config['simulation'],  # Add simulation type
        "simulation_config": config['simulation_config'],  # Add simulation config
        "sensor_dicts": config.get('sensor_dicts', {}),
    }
    
    motion_config = {
        "model_type": config['model_type'],
        "controller_type": config['controller_type'],
        "use_trajectory_planner": config['use_trajectory_planner'],
        "buffer_type": config['buffer_type'],
        "plan_type": config['plan_type'],
        "trajectory_planner_type": config['trajectory_planner_type'],
        "traj_frequency": config['traj_frequency'],
        "control_frequency": config['control_frequency'],
        "model_config": config['model_config'],
        "controller_config": config['controller_config'],
        "trajectory_config": config['trajectory_config'],
    }
    
    print("=== Testing HIROLInterface with Real Configuration ===")
    print(f"Config loaded from: {config_path}")
    print(f"Robot: {config['robot']}")
    print(f"Controller: {config['controller_type']}")
    print(f"Use Hardware: {config['use_hardware']}")
    print(f"Use Simulation: {config['use_simulation']}")
    
    try:
        # Initialize interface
        print("\n1. Initializing HIROLInterface...")
        interface = HIROLInterface(
            robot_config=robot_config,
            motion_config=motion_config,
            gripper_sleep=0.6,
            gripper_range=(0.0, 0.08)
        )
        print("   ✓ Interface initialized successfully")
        
        # Test is_ready
        print("\n2. Testing is_ready()...")
        ready = interface.is_ready()
        print(f"   Robot ready: {ready}")
        
        # Test get_state
        print("\n3. Testing get_state()...")
        state = interface.get_state()
        
        print(f"   TCP Pose (7D): {state['pose']}")
        print(f"   - Position: [{state['pose'][0]:.3f}, {state['pose'][1]:.3f}, {state['pose'][2]:.3f}]")
        print(f"   - Quaternion: [{state['pose'][3]:.3f}, {state['pose'][4]:.3f}, {state['pose'][5]:.3f}, {state['pose'][6]:.3f}]")
        
        print(f"   TCP Velocity (6D): {state['vel']}")
        print(f"   External Force: {state['force']} N")
        print(f"   External Torque: {state['torque']} Nm")
        print(f"   Gripper Position (normalized): {state['gripper_pos']:.3f}")
        print(f"   Joint Positions (rad): {[f'{q:.3f}' for q in state['q']]}")
        print(f"   Joint Velocities (rad/s): {[f'{dq:.3f}' for dq in state['dq']]}")
        print(f"   Jacobian shape: {state['jacobian'].shape}")
        print(f"   Jacobian norm: {np.linalg.norm(state['jacobian']):.3f}")
        
        # Test helper methods
        print("\n4. Testing helper methods...")
        current_pose = interface.get_current_pose()
        print(f"   Current pose (helper): {current_pose}")
        
        joint_pos = interface.get_joint_positions()
        print(f"   Joint positions (helper): {[f'{q:.3f}' for q in joint_pos]}")
        
        joint_vel = interface.get_joint_velocities()
        print(f"   Joint velocities (helper): {[f'{dq:.3f}' for dq in joint_vel]}")
        
        gripper_width = interface.get_gripper_width()
        print(f"   Gripper width: {gripper_width:.4f} m")
        
        # Test gripper control
        print("\n5. Testing gripper control...")
        
        print("   Testing open_gripper()...")
        interface.open_gripper()
        time.sleep(1)
        print(f"   Gripper width after open: {interface.get_gripper_width():.4f} m")
        
        print("   Testing close_gripper()...")
        interface.close_gripper()
        time.sleep(1)
        print(f"   Gripper width after close: {interface.get_gripper_width():.4f} m")
        
        print("   Testing binary gripper command (open)...")
        interface.send_gripper_command(1.0, mode="binary")
        time.sleep(1)
        
        print("   Testing binary gripper command (close)...")
        interface.send_gripper_command(-1.0, mode="binary")
        time.sleep(1)
        
        print("   Testing continuous gripper command...")
        interface.send_gripper_command(0.5, mode="continuous")  # Half open
        time.sleep(2)
        print(f"   Gripper width at 0.5: {interface.get_gripper_width():.4f} m")
        
        # Test position control
        print("\n6. Testing position control...")
        current_pose = interface.get_current_pose()
        
        # Create a small movement in Z direction
        target_pose = current_pose.copy()
        
        target_pose[0] += 0.1 
        target_pose[1] += 0.1 
        target_pose[2] += 0.1  # Move 5cm up
        
        print(f"   Current Z: {current_pose[2]:.3f}")
        print(f"   Target Z: {target_pose[2]:.3f}")
        print("   Sending position command...")
        interface.send_pos_command(target_pose)
        
        # Wait for movement
        input("   Press Enter to check new position...")
        
        new_pose = interface.get_current_pose()
        print(f"   New Z position: {new_pose[2]:.3f}")
        print(f"   Movement error: {abs(new_pose[2] - target_pose[2]):.4f} m")
        
        # Test rotation control
        print("\n7. Testing rotation control...")
        current_pose = interface.get_current_pose()
        
        # Create a rotation around Z axis
        from scipy.spatial.transform import Rotation as R
        current_quat = current_pose[3:]
        current_rot = R.from_quat(current_quat)
        
        # Rotate 30 degrees around Z axis
        rotation_angle = np.pi / 6  # 30 degrees
        z_rotation = R.from_euler('z', rotation_angle)
        
        # Apply rotation in world frame
        new_rot = z_rotation * current_rot
        target_pose = current_pose.copy()
        target_pose[3:] = new_rot.as_quat()
        
        # Get initial and target Z angles
        current_euler = current_rot.as_euler('xyz', degrees=True)
        target_euler = new_rot.as_euler('xyz', degrees=True)
        initial_z = current_euler[2]
        target_z = target_euler[2]
        
        print(f"   Current orientation (euler): {current_euler}")
        print(f"   Target orientation (euler): {target_euler}")
        print(f"   Initial Z angle: {initial_z:.2f} degrees")
        print(f"   Target Z angle: {target_z:.2f} degrees")
        print(f"   Commanded rotation: {rotation_angle * 180 / np.pi:.1f} degrees")
        print("   Sending rotation command...")
        interface.send_pos_command(target_pose)
        
        # Wait for rotation
        input("   Press Enter to check new orientation...")
        
        new_pose = interface.get_current_pose()
        new_rot = R.from_quat(new_pose[3:])
        final_euler = new_rot.as_euler('xyz', degrees=True)
        final_z = final_euler[2]
        
        # Calculate actual Z rotation (handle angle wrap-around)
        z_diff = final_z - initial_z
        # Normalize to [-180, 180]
        if z_diff > 180:
            z_diff -= 360
        elif z_diff < -180:
            z_diff += 360
            
        print(f"   New orientation (euler): {final_euler}")
        print(f"   Final Z angle: {final_z:.2f} degrees")
        print(f"   Actual Z rotation: {z_diff:.2f} degrees")
        print(f"   Z rotation error: {abs(z_diff - 30.0):.2f} degrees")
        
        # Test compliance parameters (with warning expected)
        print("\n8. Testing compliance parameter update...")
        test_params = {
            "translational_stiffness": 2000,
            "translational_damping": 89,
            "rotational_stiffness": 150,
            "rotational_damping": 7,
        }
        interface.update_params(test_params)
        
        # Test joint reset
        print("\n9. Testing joint reset...")
        interface.joint_reset()
        print("   ✓ Joint reset completed")
        
        # Test move to home
        print("\n10. Testing move to home...")
        interface.move_to_home()
        print("   ✓ Move to home completed")
        
        # Final state check
        print("\n11. Final state check...")
        final_state = interface.get_state()
        print(f"   Final TCP position: [{final_state['pose'][0]:.3f}, {final_state['pose'][1]:.3f}, {final_state['pose'][2]:.3f}]")
        print(f"   Final gripper position: {final_state['gripper_pos']:.3f}")
        
        # Clean up
        print("\n12. Closing interface...")
        interface.close()
        print("   ✓ Interface closed successfully")
        
        print("\n=== All tests completed successfully! ===")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_hirol_interface()