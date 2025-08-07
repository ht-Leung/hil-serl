#!/usr/bin/env python3
"""
Quick test to verify the ToolState fix works
"""

from franka_interface import FrankaInterface
import time

def test_gripper_width():
    """Test that gripper width can be read without errors"""
    print("Testing gripper width reading...")
    
    robot = FrankaInterface()
    
    try:
        # Test getting state (which includes gripper width)
        state = robot.get_state()
        print(f"✓ Got robot state successfully")
        print(f"  Gripper position: {state['gripper_pos']:.3f}")
        
        # Test gripper methods if available
        if robot._robot.has_gripper():
            width = robot._robot.get_gripper_width()
            print(f"✓ Direct gripper width: {width*1000:.1f} mm")
            
            # Test gripper control
            print("\nTesting gripper control...")
            robot.open_gripper()
            time.sleep(2)
            width = robot._robot.get_gripper_width()
            print(f"  After open: {width*1000:.1f} mm")
            
            robot.close_gripper()
            time.sleep(2)
            width = robot._robot.get_gripper_width()
            print(f"  After close: {width*1000:.1f} mm")
            
            print("\n✓ All gripper tests passed!")
        else:
            print("  No gripper available")
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        robot._robot.close()
        print("\nTest completed")

if __name__ == "__main__":
    test_gripper_width()