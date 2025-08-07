#!/usr/bin/env python3
"""
Quick test to verify fr3_reach environment works with recording scripts
"""

import sys
sys.path.append('/home/hanyu/code/hil-serl')
sys.path.append('/home/hanyu/code/HIROLRobotPlatform')

from experiments.mappings import CONFIG_MAPPING

def test_fr3_reach_config():
    """Test that fr3_reach is properly registered and can be loaded"""
    print("=" * 60)
    print("Testing FR3 Reach Configuration")
    print("=" * 60)
    
    # Check if fr3_reach is registered
    if "fr3_reach" not in CONFIG_MAPPING:
        print("✗ fr3_reach not found in CONFIG_MAPPING")
        return False
    
    print("✓ fr3_reach found in CONFIG_MAPPING")
    
    # Try to load the configuration
    try:
        config = CONFIG_MAPPING["fr3_reach"]()
        print("✓ Configuration loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return False
    
    # Try to create environment
    try:
        env = config.get_environment(fake_env=False, save_video=False, classifier=False)
        print("✓ Environment created successfully")
        
        # Test basic functionality
        obs, info = env.reset()
        print(f"✓ Environment reset successful")
        
        # Handle different observation formats (after wrappers)
        try:
            # ChunkingWrapper stacks observations, so tcp_pose might be shape (horizon, 7)
            if isinstance(obs, dict) and 'state' in obs:
                tcp_pose = obs['state'].get('tcp_pose', None)
                if tcp_pose is not None:
                    # If it's stacked (from ChunkingWrapper), take the latest
                    if len(tcp_pose.shape) > 1:
                        tcp_pose = tcp_pose[-1]  # Get the latest observation
                    # Convert to numpy if needed
                    if hasattr(tcp_pose, 'numpy'):
                        tcp_pose = tcp_pose.numpy()
                    print(f"  Initial TCP position: {tcp_pose[:3]}")
            else:
                print(f"  Observation structure: {type(obs)}")
                print(f"  Keys: {obs.keys() if hasattr(obs, 'keys') else 'N/A'}")
        except Exception as e:
            print(f"  Note: Using wrapped observation format")
            # Just continue - the environment works even if display format is different
        
        # Clean up
        env.close()
        print("✓ Environment closed properly")
        
    except Exception as e:
        print(f"✗ Failed to create/use environment: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    print("\nTesting FR3 Reach Setup for HIL-SERL Recording Scripts")
    print("=" * 60)
    
    success = test_fr3_reach_config()
    
    if success:
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("FR3 Reach is ready for demonstration recording!")
        print("\nYou can now run:")
        print("  python record_demos.py --exp_name=fr3_reach")
        print("  python record_success_fail.py --exp_name=fr3_reach")
        print("=" * 60)
    else:
        print("\n✗ Some tests failed. Please check the errors above.")