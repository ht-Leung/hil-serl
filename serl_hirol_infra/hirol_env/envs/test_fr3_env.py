#!/usr/bin/env python3
"""Test script for FR3Env using FrankaInterface"""

import numpy as np
import time
from fr3_env import FR3Env, DefaultEnvConfig


class TestEnvConfig(DefaultEnvConfig):
    """Test configuration for FR3Env"""
    
    # Robot IP
    ROBOT_IP = "192.168.3.102"
    
    # Simple reach task
    TARGET_POSE = np.array([0.5, 0.0, 0.3, -np.pi, 0, 0])
    RESET_POSE = np.array([0.5, 0.0, 0.4, -np.pi, 0, 0])
    
    # Reward threshold - 2cm for position, 0.1 rad for orientation
    REWARD_THRESHOLD = np.array([0.02, 0.02, 0.02, 0.1, 0.1, 0.1])
    
    # Action scale
    ACTION_SCALE = np.array([0.05, 0.1, 1.0])
    
    # Workspace limits
    ABS_POSE_LIMIT_HIGH = np.array([0.7, 0.3, 0.6, np.pi+0.5, 0.5, 0.5])
    ABS_POSE_LIMIT_LOW = np.array([0.3, -0.3, 0.1, np.pi-0.5, -0.5, -0.5])
    
    # Camera configuration - using correct serial numbers
    REALSENSE_CAMERAS = {
        "wrist_1": {"serial_number": "332322073603"},
        "side": {"serial_number": "244222075350"},
    }
    
    # Display
    DISPLAY_IMAGE = False
    MAX_EPISODE_LENGTH = 50
    

def test_basic_functionality():
    """Test basic environment functionality"""
    print("=" * 60)
    print("Testing FR3Env Basic Functionality")
    print("=" * 60)
    
    # Create environment
    config = TestEnvConfig()
    env = FR3Env(hz=10, fake_env=False, save_video=False, config=config)
    
    try:
        # Test reset
        print("\n1. Testing reset...")
        obs, info = env.reset()
        print(f"   Reset successful")
        print(f"   TCP pose: {obs['state']['tcp_pose'][:3]}")
        print(f"   Gripper: {obs['state']['gripper_pose']}")
        
        # Test step with zero action
        print("\n2. Testing zero action step...")
        action = np.zeros(7)
        obs, reward, done, truncated, info = env.step(action)
        print(f"   Step successful")
        print(f"   Reward: {reward}")
        print(f"   Done: {done}")
        
        # Test small movement
        print("\n3. Testing small movement...")
        action = np.array([0.1, 0, 0, 0, 0, 0, 0])  # Small x movement
        obs, reward, done, truncated, info = env.step(action)
        print(f"   New TCP pose: {obs['state']['tcp_pose'][:3]}")
        
        # Test gripper
        print("\n4. Testing gripper...")
        action = np.array([0, 0, 0, 0, 0, 0, 1])  # Open gripper
        obs, reward, done, truncated, info = env.step(action)
        time.sleep(1)
        print(f"   Gripper opened")
        
        action = np.array([0, 0, 0, 0, 0, 0, -1])  # Close gripper
        obs, reward, done, truncated, info = env.step(action)
        time.sleep(1)
        print(f"   Gripper closed")
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("\nEnvironment closed")


def test_episode():
    """Test a full episode"""
    print("=" * 60)
    print("Testing Full Episode")
    print("=" * 60)
    
    config = TestEnvConfig()
    env = FR3Env(hz=10, fake_env=False, save_video=False, config=config)
    t1 = time.time()
    try:
        obs, info = env.reset()
        print(f"Episode started at: {obs['state']['tcp_pose'][:3]}")
        print(f"Target position: {config.TARGET_POSE[:3]}")
        
        done = False
        step_count = 0
        
        while not done and step_count < config.MAX_EPISODE_LENGTH:
            # Simple proportional controller to reach target
            current_pos = obs['state']['tcp_pose'][:3]
            target_pos = config.TARGET_POSE[:3]
            
            # Compute position error
            error = target_pos - current_pos
            
            # Create action (scaled down)
            action = np.zeros(7)
            action[:3] = np.clip(error * 2, -1, 1)  # P-controller with gain=2
            
            # Step
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1
            
            if step_count % 10 == 0:
                print(f"Step {step_count}: pos={current_pos}, error={np.linalg.norm(error):.3f}")
        
        print(f"\nEpisode finished:")
        print(f"  Steps: {step_count}")
        print(f"  Success: {info.get('succeed', False)}")
        print(f"  Final position: {obs['state']['tcp_pose'][:3]}")
        t2 = time.time()
        print(f"  Time taken: {t2 - t1:.2f} seconds")
    finally:
        env.close()


def test_fake_env():
    """Test fake environment mode"""
    print("=" * 60)
    print("Testing Fake Environment Mode")
    print("=" * 60)
    
    config = TestEnvConfig()
    env = FR3Env(hz=10, fake_env=True, save_video=False, config=config)
    
    try:
        obs, info = env.reset()
        print("Fake environment reset successful")
        
        for i in range(5):
            action = np.random.uniform(-0.1, 0.1, 7)
            obs, reward, done, truncated, info = env.step(action)
            print(f"Step {i+1} completed")
        
        print("✓ Fake environment test passed")
        
    finally:
        env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test FR3Env")
    parser.add_argument("--test", type=str, default="basic",
                       choices=["basic", "episode", "fake", "all"],
                       help="Which test to run")
    args = parser.parse_args()
    
    if args.test == "basic" or args.test == "all":
        test_basic_functionality()
        print()
    
    if args.test == "episode" or args.test == "all":
        test_episode()
        print()
    
    if args.test == "fake" or args.test == "all":
        test_fake_env()
        print()
    
    print("All tests completed!")