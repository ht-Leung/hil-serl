#!/usr/bin/env python3
"""
Test script for HIROLEnv compatibility with record_success_fail.py
"""

import numpy as np
import sys
from pathlib import Path

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent))

from hirol_env.envs.hirol_env import HIROLEnv, DefaultEnvConfig


class TestEnvConfig(DefaultEnvConfig):
    """Test configuration for HIROLEnv"""
    REALSENSE_CAMERAS = {
        "wrist_1": {
            "serial_number": "332322073603",
            "dim": (1280, 720),
            "exposure": 10500,
        },
        "side": {
            "serial_number": "244222075350", 
            "dim": (1280, 720),
            "exposure": 10500,
        },
    }
    IMAGE_CROP = {}
    TARGET_POSE = np.array([0.4, 0.0, 0.3, np.pi, 0, 0])
    RESET_POSE = np.array([0.4, 0.0, 0.4, np.pi, 0, 0])
    REWARD_THRESHOLD = np.array([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])
    ACTION_SCALE = np.array([0.05, 0.25, 1])
    RANDOM_RESET = False
    RANDOM_XY_RANGE = 0.05
    RANDOM_RZ_RANGE = 0.1
    ABS_POSE_LIMIT_HIGH = np.array([0.6, 0.3, 0.6, np.pi+0.5, 0.5, np.pi+0.5])
    ABS_POSE_LIMIT_LOW = np.array([0.2, -0.3, 0.1, np.pi-0.5, -0.5, -np.pi-0.5])
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
    }
    DISPLAY_IMAGE = True
    GRIPPER_SLEEP = 0.6
    MAX_EPISODE_LENGTH = 100
    JOINT_RESET_PERIOD = 10


def test_env_interface():
    """Test HIROLEnv interface compatibility"""
    
    print("=== Testing HIROLEnv Interface Compatibility ===")
    
    # Test initialization
    print("\n1. Testing environment initialization...")
    try:
        env = HIROLEnv(
            hz=10,
            fake_env=False,
            save_video=False,
            config=TestEnvConfig(),
            set_load=False
        )
        print("   ✓ Environment initialized successfully")
    except Exception as e:
        import traceback
        print(f"   ✗ Failed to initialize environment: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return
    
    input("Press Enter to continue with further tests...")
    
    # Test reset
    print("\n2. Testing reset()...")
    try:
        obs, info = env.reset()
        print(f"   ✓ Reset successful")
        print(f"   Observation keys: {list(obs.keys())}")
        print(f"   State keys: {list(obs['state'].keys())}")
        print(f"   Image keys: {list(obs['images'].keys())}")
        print(f"   Info: {info}")
    except Exception as e:
        print(f"   ✗ Reset failed: {e}")
        return
    
    input("Press Enter to continue with action space tests...")
    
    # Test action space
    print("\n3. Testing action space...")
    try:
        action = env.action_space.sample()
        print(f"   ✓ Action space shape: {action.shape}")
        print(f"   Action space low: {env.action_space.low}")
        print(f"   Action space high: {env.action_space.high}")
    except Exception as e:
        print(f"   ✗ Action space test failed: {e}")
        return
    input("Press Enter to continue with step() test...")
    
    # Test step
    print("\n4. Testing step()...")
    try:
        # Test with zero action
        action = np.zeros(env.action_space.shape)
        next_obs, rew, done, truncated, info = env.step(action)
        print(f"   ✓ Step successful")
        print(f"   Reward: {rew}")
        print(f"   Done: {done}")
        print(f"   Info: {info}")
    except Exception as e:
        print(f"   ✗ Step failed: {e}")
        return
    
    input("Press Enter to continue with observation space tests...")
    
    # Test observation space compatibility
    print("\n5. Testing observation space...")
    try:
        # Check state observation
        assert "tcp_pose" in obs["state"]
        assert obs["state"]["tcp_pose"].shape == (7,)  # xyz + quat
        
        assert "tcp_vel" in obs["state"]
        assert obs["state"]["tcp_vel"].shape == (6,)
        
        assert "gripper_pose" in obs["state"]
        
        assert "tcp_force" in obs["state"]
        assert obs["state"]["tcp_force"].shape == (3,)
        
        assert "tcp_torque" in obs["state"]
        assert obs["state"]["tcp_torque"].shape == (3,)
        
        print("   ✓ All state observations present and correct shape")
    except AssertionError as e:
        print(f"   ✗ Observation space mismatch: {e}")
        return
    
    input("Press Enter to continue with gripper control tests...")
    
    # Test gripper control
    print("\n6. Testing gripper control...")
    try:
        # Open gripper
        action = np.zeros(env.action_space.shape)
        action[6] = 1.0  # Open gripper
        env.step(action)
        print("   ✓ Open gripper command sent")
        
        # Close gripper
        action[6] = -1.0  # Close gripper
        env.step(action)
        print("   ✓ Close gripper command sent")
    except Exception as e:
        print(f"   ✗ Gripper control failed: {e}")
    
    input("Press Enter to continue with multiple steps test...")
    
    # Test multiple steps (simulate record_success_fail.py usage)
    print("\n7. Testing multiple steps (simulating record_success_fail.py)...")
    try:
        transitions = []
        for i in range(10):
            action = np.zeros(env.action_space.shape)
            # Small random movements
            action[:3] = np.random.uniform(-0.1, 0.1, 3)
            
            next_obs, rew, done, truncated, info = env.step(action)
            
            # Simulate transition recording
            transition = {
                "observations": obs,
                "actions": action,
                "next_observations": next_obs,
                "rewards": rew,
                "masks": 1.0 - done,
                "dones": done,
            }
            transitions.append(transition)
            
            obs = next_obs
            
            if done or truncated:
                obs, _ = env.reset()
                
        print(f"   ✓ Successfully collected {len(transitions)} transitions")
    except Exception as e:
        print(f"   ✗ Multiple steps failed: {e}")
    
    input("Press Enter to continue with environment cleanup...")
    
    # Clean up
    print("\n8. Testing cleanup...")
    try:
        env.close()
        print("   ✓ Environment closed successfully")
    except Exception as e:
        print(f"   ✗ Cleanup failed: {e}")
    
    print("\n=== All tests completed! ===")
    print("\nThe HIROLEnv interface is compatible with record_success_fail.py usage pattern.")


if __name__ == "__main__":
    test_env_interface()