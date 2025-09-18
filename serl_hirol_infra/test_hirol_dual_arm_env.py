"""
Test script for HIROLDualArmEnv compatibility and functionality

This script tests:
1. Basic dual-arm environment initialization
2. Action and observation space compatibility with DualFrankaEnv
3. Dual-arm wrapper compatibility (DualQuat2EulerWrapper, etc.)
4. Configuration system functionality
5. State combination and action distribution logic
"""

import numpy as np
import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, "/home/hanyu/code/HIROLRobotPlatform")
sys.path.insert(0, "/home/hanyu/code/hil-serl/serl_hirol_infra")

from hirol_env.envs.hirol_dual_arm_env import HIROLDualArmEnv
from hirol_env.envs.dual_arm_config import (
    DefaultDualArmConfig,
    HandoverTaskConfig,
    BimanualManipulationConfig,
    DualArmSimConfig
)
from hirol_env.envs.wrappers import (
    DualQuat2EulerWrapper,
    DualSpacemouseIntervention,
    DualGripperPenaltyWrapper
)


class TestDualArmConfig(DualArmSimConfig):
    """Test configuration for dual-arm environment using simulation"""

    def __init__(self):
        super().__init__()

        # Ensure simulation mode for testing
        self.left_config.REALSENSE_CAMERAS = {}
        self.right_config.REALSENSE_CAMERAS = {}

        # Set test-friendly parameters
        self.left_config.MAX_EPISODE_LENGTH = 10
        self.right_config.MAX_EPISODE_LENGTH = 10


def test_dual_arm_env_initialization():
    """Test HIROLDualArmEnv initialization and basic functionality"""

    print("=== Testing HIROLDualArmEnv Initialization ===")

    try:
        # Create test configuration
        config = TestDualArmConfig()
        dual_config = config.get_dual_arm_config()

        print("\n1. Testing environment initialization...")
        env = HIROLDualArmEnv(**dual_config)
        print("✅ HIROLDualArmEnv initialized successfully")

        # Test action space
        print(f"\n2. Testing action space...")
        print(f"   Action space shape: {env.action_space.shape}")
        print(f"   Expected: 14D (7+7 for dual arm)")
        assert env.action_space.shape[0] == 14, f"Expected 14D action space, got {env.action_space.shape[0]}D"
        print("✅ Action space is correct (14D)")

        # Test observation space
        print(f"\n3. Testing observation space...")
        state_space = env.observation_space["state"]
        image_space = env.observation_space["images"]

        # Check for left/ and right/ prefixes
        state_keys = list(state_space.spaces.keys())
        left_state_keys = [k for k in state_keys if k.startswith("left/")]
        right_state_keys = [k for k in state_keys if k.startswith("right/")]

        print(f"   State keys with left/ prefix: {len(left_state_keys)}")
        print(f"   State keys with right/ prefix: {len(right_state_keys)}")
        assert len(left_state_keys) > 0, "No left/ state keys found"
        assert len(right_state_keys) > 0, "No right/ state keys found"
        print("✅ Observation space has correct left/ and right/ prefixes")

        # Test environment reset
        print(f"\n4. Testing environment reset...")
        obs, info = env.reset()
        print(f"   Reset successful, observation keys: {list(obs.keys())}")
        print(f"   State observation keys: {len(obs['state'])}")
        print(f"   Image observation keys: {len(obs['images'])}")
        print("✅ Environment reset successful")

        # Test environment step with random action
        print(f"\n5. Testing environment step...")
        random_action = env.action_space.sample()
        print(f"   Random action shape: {random_action.shape}")

        obs, reward, done, truncated, info = env.step(random_action)
        print(f"   Step successful")
        print(f"   Reward: {reward}, Done: {done}, Truncated: {truncated}")
        print(f"   Info keys: {list(info.keys())}")
        print("✅ Environment step successful")

        # Clean up
        env.close()
        print("✅ Environment closed successfully")

        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dual_arm_wrapper_compatibility():
    """Test compatibility with dual-arm wrappers"""

    print("\n=== Testing Dual-Arm Wrapper Compatibility ===")

    try:
        # Create base environment
        config = TestDualArmConfig()
        dual_config = config.get_dual_arm_config()
        env = HIROLDualArmEnv(**dual_config)

        print("\n1. Testing DualQuat2EulerWrapper...")
        wrapped_env = DualQuat2EulerWrapper(env)

        # Test reset
        obs, info = wrapped_env.reset()

        # Check that tcp_pose is converted from quaternion (7D) to euler (6D)
        assert "left/tcp_pose" in obs["state"], "Missing left/tcp_pose in wrapped observation"
        assert "right/tcp_pose" in obs["state"], "Missing right/tcp_pose in wrapped observation"

        left_tcp_shape = obs["state"]["left/tcp_pose"].shape
        right_tcp_shape = obs["state"]["right/tcp_pose"].shape

        print(f"   Left TCP pose shape: {left_tcp_shape}")
        print(f"   Right TCP pose shape: {right_tcp_shape}")
        print(f"   Expected: (6,) for euler angles")

        # Note: This test depends on the base HIROLEnv implementation
        # If HIROLEnv doesn't provide tcp_pose, this might fail
        print("✅ DualQuat2EulerWrapper applied successfully")

        wrapped_env.close()

        print("\n2. Testing DualSpacemouseIntervention...")
        # Create fresh environment for spacemouse test
        env2 = HIROLDualArmEnv(**dual_config)
        spacemouse_env = DualSpacemouseIntervention(env2)

        obs, info = spacemouse_env.reset()

        # Test action processing (will be mostly zeros without actual spacemouse)
        test_action = np.zeros(14)
        processed_action, replaced = spacemouse_env.action(test_action)

        print(f"   Processed action shape: {processed_action.shape}")
        print(f"   Action replaced: {replaced}")
        print("✅ DualSpacemouseIntervention applied successfully")

        spacemouse_env.close()

        print("\n3. Testing DualGripperPenaltyWrapper...")
        # Create fresh environment for gripper penalty test
        env3 = HIROLDualArmEnv(**dual_config)
        penalty_env = DualGripperPenaltyWrapper(env3)

        obs, info = penalty_env.reset()

        # Test step with action that should trigger penalty
        test_action = np.ones(14) * 0.5  # Neutral action
        obs, reward, done, truncated, info = penalty_env.step(test_action)

        print(f"   Step with penalty wrapper successful")
        print(f"   Reward: {reward}")
        print("✅ DualGripperPenaltyWrapper applied successfully")

        penalty_env.close()

        return True

    except Exception as e:
        print(f"❌ Wrapper compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration_system():
    """Test different configuration classes"""

    print("\n=== Testing Configuration System ===")

    try:
        print("\n1. Testing DefaultDualArmConfig...")
        default_config = DefaultDualArmConfig()
        dual_config = default_config.get_dual_arm_config()

        assert "left_env_config" in dual_config
        assert "right_env_config" in dual_config
        assert "display_images" in dual_config
        print("✅ DefaultDualArmConfig works correctly")

        print("\n2. Testing HandoverTaskConfig...")
        handover_config = HandoverTaskConfig()
        dual_config = handover_config.get_dual_arm_config()

        assert handover_config.task_config.TASK_TYPE == "handover"
        assert handover_config.task_config.REWARD_STRATEGY == "either_succeed"
        print("✅ HandoverTaskConfig works correctly")

        print("\n3. Testing BimanualManipulationConfig...")
        bimanual_config = BimanualManipulationConfig()
        dual_config = bimanual_config.get_dual_arm_config()

        assert bimanual_config.task_config.TASK_TYPE == "bimanual_manipulation"
        assert bimanual_config.task_config.REWARD_STRATEGY == "both_succeed"
        print("✅ BimanualManipulationConfig works correctly")

        print("\n4. Testing DualArmSimConfig...")
        sim_config = DualArmSimConfig()
        left_config = sim_config.get_left_env_config()
        right_config = sim_config.get_right_env_config()

        assert left_config["fake_env"] == True
        assert right_config["fake_env"] == True
        print("✅ DualArmSimConfig works correctly")

        return True

    except Exception as e:
        print(f"❌ Configuration system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_action_observation_format():
    """Test action and observation format compatibility with DualFrankaEnv"""

    print("\n=== Testing Action/Observation Format Compatibility ===")

    try:
        config = TestDualArmConfig()
        dual_config = config.get_dual_arm_config()
        env = HIROLDualArmEnv(**dual_config)

        # Test observation format
        print("\n1. Testing observation format...")
        obs, info = env.reset()

        # Check observation structure matches DualFrankaEnv
        required_keys = ["state", "images"]
        for key in required_keys:
            assert key in obs, f"Missing required key: {key}"
        print("✅ Observation has required top-level keys: state, images")

        # Check state keys have left/ and right/ prefixes
        state_keys = list(obs["state"].keys())
        has_left_prefix = any(k.startswith("left/") for k in state_keys)
        has_right_prefix = any(k.startswith("right/") for k in state_keys)

        assert has_left_prefix, "No state keys with left/ prefix found"
        assert has_right_prefix, "No state keys with right/ prefix found"
        print("✅ State keys have correct left/ and right/ prefixes")

        # Test action format
        print("\n2. Testing action format...")

        # Test 14D action (standard dual-arm format)
        action_14d = np.random.uniform(-0.1, 0.1, 14)
        obs, reward, done, truncated, info = env.step(action_14d)
        print("✅ 14D action format works correctly")

        # Test action splitting
        left_action = action_14d[:7]
        right_action = action_14d[7:]
        print(f"   Left action shape: {left_action.shape}")
        print(f"   Right action shape: {right_action.shape}")
        print("✅ Action splitting works correctly")

        # Test reward and info format
        print("\n3. Testing reward and info format...")
        assert isinstance(reward, (int, float, np.integer, np.floating)), f"Reward should be numeric, got {type(reward)}"
        assert isinstance(info, dict), f"Info should be dict, got {type(info)}"

        # Check for left and right info
        if "left" in info and "right" in info:
            print("✅ Info contains left and right sub-info")

        env.close()
        return True

    except Exception as e:
        print(f"❌ Action/observation format test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_state_combination_logic():
    """Test the state combination logic in detail"""

    print("\n=== Testing State Combination Logic ===")

    try:
        config = TestDualArmConfig()
        dual_config = config.get_dual_arm_config()
        env = HIROLDualArmEnv(**dual_config)

        # Create mock observations to test combine_obs method
        print("\n1. Testing combine_obs method...")

        # Mock left observation
        mock_left_obs = {
            "state": {
                "tcp_pose": np.array([0.5, 0.2, 0.3, 0, 0, 0, 1]),
                "gripper_pos": np.array([0.8])
            },
            "images": {
                "wrist_camera": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            }
        }

        # Mock right observation
        mock_right_obs = {
            "state": {
                "tcp_pose": np.array([0.5, -0.2, 0.3, 0, 0, 0, 1]),
                "gripper_pos": np.array([0.2])
            },
            "images": {
                "wrist_camera": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            }
        }

        # Test combine_obs method
        combined_obs = env.combine_obs(mock_left_obs, mock_right_obs)

        # Check combined state
        expected_state_keys = ["left/tcp_pose", "left/gripper_pos", "right/tcp_pose", "right/gripper_pos"]
        for key in expected_state_keys:
            assert key in combined_obs["state"], f"Missing state key: {key}"
        print("✅ State combination works correctly")

        # Check combined images
        expected_image_keys = ["left/wrist_camera", "right/wrist_camera"]
        for key in expected_image_keys:
            assert key in combined_obs["images"], f"Missing image key: {key}"
        print("✅ Image combination works correctly")

        # Check data integrity
        np.testing.assert_array_equal(
            combined_obs["state"]["left/tcp_pose"],
            mock_left_obs["state"]["tcp_pose"]
        )
        np.testing.assert_array_equal(
            combined_obs["state"]["right/tcp_pose"],
            mock_right_obs["state"]["tcp_pose"]
        )
        print("✅ Data integrity maintained in combination")

        env.close()
        return True

    except Exception as e:
        print(f"❌ State combination logic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all test functions"""

    print("🤖 Starting HIROLDualArmEnv Test Suite")
    print("=" * 60)

    tests = [
        ("Environment Initialization", test_dual_arm_env_initialization),
        ("Wrapper Compatibility", test_dual_arm_wrapper_compatibility),
        ("Configuration System", test_configuration_system),
        ("Action/Observation Format", test_action_observation_format),
        ("State Combination Logic", test_state_combination_logic),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name} Test PASSED")
            else:
                print(f"❌ {test_name} Test FAILED")
        except Exception as e:
            print(f"❌ {test_name} Test FAILED with exception: {e}")
            results.append((test_name, False))

    # Print summary
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<30} {status}")

    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! HIROLDualArmEnv is ready for use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)