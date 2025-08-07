#!/usr/bin/env python3
"""Test script to verify HIROLReachEnv initialization and basic functionality"""

import numpy as np
from config import TrainConfig, EnvConfig

def test_env_creation():
    """Test environment creation and basic operations"""
    print("=== Testing HIROLReachEnv ===")
    
    # Create config
    config = TrainConfig()
    
    # Create environment in fake mode for testing
    print("\n1. Creating environment in fake mode...")
    try:
        env = config.get_environment(fake_env=True, save_video=False, classifier=False)
        print("✓ Environment created successfully")
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        return False
    
    # Test reset
    print("\n2. Testing reset...")
    try:
        obs, info = env.reset()
        print("✓ Reset successful")
        print(f"  - Observation keys: {list(obs.keys())}")
        print(f"  - State shape: {obs['state'].shape}")
        print(f"  - Image shapes: {[(k, v.shape) for k, v in obs['images'].items()]}")
        print(f"  - Info keys: {list(info.keys())}")
    except Exception as e:
        print(f"✗ Reset failed: {e}")
        return False
    
    # Test step
    print("\n3. Testing step with zero action...")
    try:
        action = np.zeros(env.action_space.shape)
        next_obs, reward, done, truncated, info = env.step(action)
        print("✓ Step successful")
        print(f"  - Reward: {reward}")
        print(f"  - Done: {done}")
        print(f"  - Truncated: {truncated}")
        print(f"  - Success: {info.get('success', False)}")
    except Exception as e:
        print(f"✗ Step failed: {e}")
        return False
    
    # Test multiple steps
    print("\n4. Testing multiple steps...")
    try:
        for i in range(5):
            action = np.random.uniform(-1, 1, size=env.action_space.shape)
            next_obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                obs, info = env.reset()
                print(f"  - Episode ended at step {i+1}, reset successful")
        print("✓ Multiple steps successful")
    except Exception as e:
        print(f"✗ Multiple steps failed: {e}")
        return False
    
    # Test action and observation spaces
    print("\n5. Testing spaces...")
    print(f"  - Action space: {env.action_space}")
    print(f"  - Action space shape: {env.action_space.shape}")
    print(f"  - Observation space keys: {list(env.observation_space.spaces.keys())}")
    
    # Close environment
    print("\n6. Closing environment...")
    try:
        env.close()
        print("✓ Environment closed successfully")
    except Exception as e:
        print(f"✗ Failed to close environment: {e}")
        return False
    
    print("\n=== All tests passed! ===")
    return True


if __name__ == "__main__":
    test_env_creation()