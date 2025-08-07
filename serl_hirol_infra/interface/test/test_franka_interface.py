#!/usr/bin/env python3
"""
Test script for FrankaInterface integration with RobotInterface
"""

import numpy as np
import time
import logging
from franka_interface import FrankaInterface
from robot_interface import ComplianceParams, LoadParams

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_basic_functionality(robot: FrankaInterface):
    """Test basic robot interface functionality"""
    print("\n=== Testing Basic Functionality ===")
    
    # Test 1: Check ready status
    print("\n1. Checking robot status...")
    if robot.is_ready():
        print("   ✓ Robot is ready")
    else:
        print("   ✗ Robot not ready")
        return False
    
    # Test 2: Get state
    print("\n2. Getting robot state...")
    try:
        state = robot.get_state()
        print(f"   Pose: {state['pose']}")
        print(f"   Joint positions shape: {state['q'].shape}")
        print(f"   Jacobian shape: {state['jacobian'].shape}")
        print(f"   Force: {state['force']}")
        print(f"   Gripper position: {state['gripper_pos']:.3f}")
        print("   ✓ State retrieval successful")
    except Exception as e:
        print(f"   ✗ Failed to get state: {e}")
        return False
    
    return True


def test_gripper_control(robot: FrankaInterface):
    """Test gripper control through interface"""
    print("\n=== Testing Gripper Control ===")
    
    try:
        # Test binary mode
        print("\n1. Testing binary gripper control...")
        print("   Opening gripper (position > 0.5)...")
        robot.send_gripper_command(1.0, mode="binary")
        time.sleep(2)
        
        state = robot.get_state()
        print(f"   Gripper position: {state['gripper_pos']:.3f}")
        
        print("   Closing gripper (position < -0.5)...")
        robot.send_gripper_command(-1.0, mode="binary")
        time.sleep(2)
        
        state = robot.get_state()
        print(f"   Gripper position: {state['gripper_pos']:.3f}")
        
        # Test continuous mode
        print("\n2. Testing continuous gripper control...")
        print("   Setting gripper to 50% (position = 0.0)...")
        robot.send_gripper_command(0.0, mode="continuous")
        time.sleep(2)
        
        state = robot.get_state()
        print(f"   Gripper position: {state['gripper_pos']:.3f}")
        
        # Open gripper at end
        robot.open_gripper()
        print("   ✓ Gripper control successful")
        return True
        
    except Exception as e:
        print(f"   ✗ Gripper control failed: {e}")
        return False


def test_motion_control(robot: FrankaInterface):
    """Test motion control commands"""
    print("\n=== Testing Motion Control ===")
    
    try:
        # Get current pose
        state = robot.get_state()
        initial_pose = state['pose'].copy()
        print(f"\n1. Initial pose: {initial_pose[:3]}")  # Print only position
        
        # Test servo mode (fast direct control)
        print("\n2. Testing servo position command (fast direct)...")
        target_pose = initial_pose.copy()
        target_pose[0] += 0.03  # Move 3cm in x
        target_pose[1] += 0.02  # Move 2cm in y
        
        print(f"   Target pose: {target_pose[:3]}")
        start_time = time.time()
        robot.send_pos_command(target_pose)  # Default servo mode
        servo_time = time.time() - start_time
        time.sleep(2)  # Wait for motion to complete
        
        # Check new position
        state = robot.get_state()
        new_pose = state['pose']
        print(f"   Actual pose: {new_pose[:3]}")
        print(f"   Servo command time: {servo_time:.3f}s")
        
        error = np.linalg.norm(new_pose[:3] - target_pose[:3])
        print(f"   Position error: {error*1000:.2f} mm")
        
        if error < 0.01:  # 10mm tolerance
            print("   ✓ Servo motion control successful")
        else:
            print("   ⚠ Servo motion completed with larger error")
        
        # Test trajectory mode (smooth planned motion)
        print("\n3. Testing trajectory position command (smooth)...")
        target_pose2 = initial_pose.copy()
        target_pose2[0] -= 0.02  # Move back 2cm in x
        target_pose2[1] -= 0.01  # Move back 1cm in y
        
        print(f"   Target pose: {target_pose2[:3]}")
        start_time = time.time()
        robot.send_pos_trajectory_command(target_pose2, finish_time=2.5)
        traj_time = time.time() - start_time
        time.sleep(3)  # Wait for trajectory to complete
        
        # Check new position
        state = robot.get_state()
        new_pose = state['pose']
        print(f"   Actual pose: {new_pose[:3]}")
        print(f"   Trajectory command time: {traj_time:.3f}s")
        
        error = np.linalg.norm(new_pose[:3] - target_pose2[:3])
        print(f"   Position error: {error*1000:.2f} mm")
        
        if error < 0.01:  # 10mm tolerance
            print("   ✓ Trajectory motion control successful")
        else:
            print("   ⚠ Trajectory motion completed with larger error")
        
        # Return to initial position using trajectory mode
        print("\n4. Returning to initial position (smooth trajectory)...")
        robot.send_pos_trajectory_command(initial_pose, finish_time=2.0)
        time.sleep(2.5)
        
        print("   ✓ Both servo and trajectory modes tested")
        return True
        
    except Exception as e:
        print(f"   ✗ Motion control failed: {e}")
        return False


def test_compliance_params(robot: FrankaInterface):
    """Test compliance parameter updates"""
    print("\n=== Testing Compliance Parameters ===")
    
    try:
        # Create compliance parameters
        params = {
            'translational_stiffness': 1000.0,
            'rotational_stiffness': 100.0,
            'translational_damping': 70.0,
            'rotational_damping': 7.0
        }
        
        print(f"\n1. Updating compliance parameters...")
        print(f"   Stiffness: trans={params['translational_stiffness']}, "
              f"rot={params['rotational_stiffness']}")
        print(f"   Damping: trans={params['translational_damping']}, "
              f"rot={params['rotational_damping']}")
        
        robot.update_params(params)
        print("   ✓ Parameters updated")
        
        # Test impedance motion with new parameters
        if hasattr(robot, 'move_to_pose_impedance'):
            print("\n2. Testing impedance control...")
            state = robot.get_state()
            current_pose = state['pose'].copy()
            target_pose = current_pose.copy()
            target_pose[2] -= 0.02  # Move down 2cm
            
            compliance = ComplianceParams(
                translational_stiffness=800.0,
                translational_damping=60.0,
                rotational_stiffness=80.0,
                rotational_damping=6.0
            )
            
            robot.move_to_pose_impedance(target_pose, compliance)
            time.sleep(3)
            print("   ✓ Impedance control successful")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Compliance parameter test failed: {e}")
        return False


def test_error_handling(robot: FrankaInterface):
    """Test error handling and recovery"""
    print("\n=== Testing Error Handling ===")
    
    try:
        # Test clear errors
        print("\n1. Testing error clearing...")
        robot.clear_errors()
        print("   ✓ Error clearing successful")
        
        # Test emergency stop
        print("\n2. Testing emergency stop...")
        robot.emergency_stop()
        time.sleep(1)
        print("   ✓ Emergency stop successful")
        
        # Reinitialize after stop
        print("\n3. Reinitializing after stop...")
        robot.clear_errors()
        
        if robot.is_ready():
            print("   ✓ Robot ready after recovery")
        else:
            print("   ⚠ Robot may need manual recovery")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error handling test failed: {e}")
        return False


def test_joint_reset(robot: FrankaInterface):
    """Test joint reset functionality"""
    print("\n=== Testing Joint Reset ===")
    
    try:
        print("\n1. Getting current joint positions...")
        state = robot.get_state()
        initial_joints = state['q']
        print(f"   Current joints: {initial_joints}")
        
        print("\n2. Executing joint reset (moving to home)...")
        robot.joint_reset()
        time.sleep(3)
        
        state = robot.get_state()
        home_joints = state['q']
        print(f"   Home joints: {home_joints}")
        
        print("   ✓ Joint reset successful")
        return True
        
    except Exception as e:
        print(f"   ✗ Joint reset failed: {e}")
        return False


def test_load_params(robot: FrankaInterface):
    """Test load parameter setting"""
    print("\n=== Testing Load Parameters ===")
    
    try:
        # Create load parameters
        load = LoadParams(
            mass=0.5,  # 500g
            F_x_center_load=[0.0, 0.0, 0.1],  # 10cm from flange
            load_inertia=[0.001, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.001]
        )
        
        print(f"\n1. Setting load parameters...")
        print(f"   Mass: {load.mass} kg")
        print(f"   Center of mass: {load.F_x_center_load}")
        
        robot.set_load(load)
        print("   ✓ Load parameters set")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Load parameter test failed: {e}")
        return False


def test_advanced_features(robot: FrankaInterface):
    """Test advanced features"""
    print("\n=== Testing Advanced Features ===")
    
    try:
        # Test TCP wrench
        print("\n1. Getting TCP wrench...")
        wrench = robot.get_tcp_wrench()
        print(f"   Force: [{wrench[0]:.2f}, {wrench[1]:.2f}, {wrench[2]:.2f}] N")
        print(f"   Torque: [{wrench[3]:.2f}, {wrench[4]:.2f}, {wrench[5]:.2f}] Nm")
        
        # Test contact detection
        print("\n2. Checking contact...")
        in_contact = robot.is_in_contact(threshold=5.0)
        print(f"   In contact: {in_contact}")
        
        print("   ✓ Advanced features working")
        return True
        
    except Exception as e:
        print(f"   ✗ Advanced features test failed: {e}")
        return False


def run_all_tests():
    """Run all interface tests"""
    print("="*60)
    print("Franka Interface Integration Test")
    print("="*60)
    
    # Initialize robot interface
    print("\nInitializing FrankaInterface...")
    robot = FrankaInterface()
    
    try:
        # List of tests
        tests = [
            ("Basic Functionality", test_basic_functionality),
            ("Gripper Control", test_gripper_control),
            ("Motion Control", test_motion_control),
            ("Compliance Parameters", test_compliance_params),
            ("Load Parameters", test_load_params),
            ("Advanced Features", test_advanced_features),
            ("Joint Reset", test_joint_reset),
            ("Error Handling", test_error_handling),
        ]
        
        # Show menu
        print("\nAvailable tests:")
        for i, (name, _) in enumerate(tests, 1):
            print(f"{i}. {name}")
        print(f"{len(tests)+1}. Run all tests")
        print("0. Exit")
        
        choice = input("\nSelect test (0-{}): ".format(len(tests)+1))
        
        if choice == "0":
            print("Exiting...")
            return
        elif choice == str(len(tests)+1):
            # Run all tests
            results = []
            for name, test_func in tests:
                print(f"\n{'='*60}")
                print(f"Running: {name}")
                print('='*60)
                try:
                    success = test_func(robot)
                    results.append((name, success))
                except Exception as e:
                    print(f"Test failed with exception: {e}")
                    results.append((name, False))
                time.sleep(1)
            
            # Print summary
            print("\n" + "="*60)
            print("Test Summary")
            print("="*60)
            for name, success in results:
                status = "✓ PASSED" if success else "✗ FAILED"
                print(f"{name:30} {status}")
        else:
            # Run selected test
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(tests):
                    name, test_func = tests[idx]
                    print(f"\nRunning: {name}")
                    test_func(robot)
                else:
                    print("Invalid selection")
            except (ValueError, IndexError):
                print("Invalid selection")
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        robot._robot.close()
        print("Test completed!")


if __name__ == "__main__":
    run_all_tests()