#!/usr/bin/env python3
"""
Simple demonstration recording test for FR3
Tests basic SpaceMouse control without complex wrappers
"""

import sys
import numpy as np
import time

# Add paths
sys.path.append('/home/hanyu/code/hil-serl')
sys.path.append('/home/hanyu/code/HIROLRobotPlatform')
sys.path.insert(0, '/home/hanyu/code/hil-serl/serl_hirol_infra')

from hirol_env.envs.fr3_env import FR3Env, DefaultEnvConfig
from hirol_env.envs.wrappers import SpacemouseIntervention


class SimpleConfig(DefaultEnvConfig):
    """Simple configuration for testing"""
    ROBOT_IP = "192.168.3.102"
    REALSENSE_CAMERAS = {}  # No cameras for simple test
    TARGET_POSE = np.array([0.5, 0.0, 0.3, -np.pi, 0, 0])
    RESET_POSE = np.array([0.5, 0.0, 0.4, -np.pi, 0, 0])
    REWARD_THRESHOLD = np.array([0.02, 0.02, 0.02, 0.1, 0.1, 0.1])
    ACTION_SCALE = np.array([0.05, 0.1, 1.0])
    ABS_POSE_LIMIT_HIGH = np.array([0.7, 0.3, 0.6, np.pi+0.5, 0.5, 0.5])
    ABS_POSE_LIMIT_LOW = np.array([0.3, -0.3, 0.1, np.pi-0.5, -0.5, -0.5])
    MAX_EPISODE_LENGTH = 100
    DISPLAY_IMAGE = False


def test_spacemouse_control():
    """Test SpaceMouse control with FR3"""
    print("=" * 60)
    print("Testing SpaceMouse Control with FR3")
    print("=" * 60)
    
    # Create environment with SpaceMouse
    env = FR3Env(hz=10, fake_env=False, save_video=False, config=SimpleConfig())
    env = SpacemouseIntervention(env)
    
    print("\n✓ Environment created with SpaceMouse")
    print("\nInstructions:")
    print("- Use SpaceMouse to control the robot")
    print("- Press Ctrl+C to stop")
    print("=" * 60)
    
    try:
        # Reset environment
        obs, info = env.reset()
        print(f"\n✓ Reset complete")
        print(f"  TCP position: {obs['state']['tcp_pose'][:3]}")
        
        # Run control loop
        step_count = 0
        while True:
            # Get action (zeros, but SpaceMouse will override via intervention)
            action = np.zeros(7)
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            
            # Check if SpaceMouse intervened
            if "intervene_action" in info:
                action = info["intervene_action"]
                if np.any(action != 0):
                    print(f"Step {step_count}: SpaceMouse action detected")
            
            step_count += 1
            
            # Reset if episode done
            if done:
                print(f"\nEpisode done after {step_count} steps")
                obs, info = env.reset()
                step_count = 0
                print("Reset for new episode")
                
            time.sleep(0.01)  # Small delay
            
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        env.close()
        print("✓ Environment closed")


if __name__ == "__main__":
    test_spacemouse_control()