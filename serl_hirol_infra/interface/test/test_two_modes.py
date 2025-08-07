#!/usr/bin/env python3
"""
Quick test demonstrating servo vs trajectory modes
"""

import numpy as np
import time
from franka_interface import FrankaInterface

def test_two_modes():
    """Compare servo (direct) vs trajectory (smooth) modes"""
    
    print("=" * 60)
    print("Servo vs Trajectory Mode Comparison")
    print("=" * 60)
    
    # Initialize robot
    robot = FrankaInterface()
    
    try:
        # Get initial state
        state = robot.get_state()
        initial_pose = state['pose'].copy()
        print(f"\nInitial position: {initial_pose[:3]}")
        
        # Define target position (5cm forward)
        target_pose = initial_pose.copy()
        target_pose[0] += 0.05
        
        # Test 1: Servo mode (fast, direct)
        print("\n1. SERVO MODE (Fast Direct Control)")
        print("   - Immediate response")
        print("   - No trajectory planning")
        print("   - Good for reactive control")
        start = time.time()
        robot.send_pos_command(target_pose)
        elapsed = time.time() - start
        print(f"   Command sent in: {elapsed*1000:.1f}ms")
        time.sleep(2)  # Wait for motion
        
        # Return to initial
        print("\n2. Returning to initial position...")
        robot.send_pos_command(initial_pose)
        time.sleep(2)
        
        # Test 2: Trajectory mode (smooth, planned)
        print("\n3. TRAJECTORY MODE (Smooth Planned Motion)")
        print("   - Smooth acceleration/deceleration")
        print("   - Trajectory planning")
        print("   - Good for predictable motions")
        start = time.time()
        robot.send_pos_trajectory_command(target_pose, finish_time=2.0)
        elapsed = time.time() - start
        print(f"   Command sent in: {elapsed*1000:.1f}ms")
        print("   Trajectory will complete in 2.0 seconds")
        time.sleep(2.5)
        
        # Return using trajectory
        print("\n4. Smooth return to initial position...")
        robot.send_pos_trajectory_command(initial_pose, finish_time=1.5)
        time.sleep(2)
        
        print("\n" + "=" * 60)
        print("Summary:")
        print("- Servo: Use for real-time control, fast response")
        print("- Trajectory: Use for smooth, predictable motions")
        print("=" * 60)
        
    finally:
        robot._robot.close()
        print("\nTest completed!")

if __name__ == "__main__":
    test_two_modes()