"""
Example usage of HIROLDualArmEnv with different configurations and wrappers

This script demonstrates:
1. Basic dual-arm environment usage
2. Different task configurations (handover, bimanual manipulation)
3. Using dual-arm wrappers
4. Integration with existing HIL-SERL training code patterns
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
    KeyboardRewardWrapper
)


def example_basic_usage():
    """Basic usage example with simulation"""

    print("=== Basic HIROLDualArmEnv Usage Example ===")

    # Create simulation configuration for testing
    config = DualArmSimConfig()
    dual_config = config.get_dual_arm_config()

    # Initialize environment
    env = HIROLDualArmEnv(**dual_config)

    print(f"Action space: {env.action_space}")
    print(f"Observation space keys: {list(env.observation_space.spaces.keys())}")

    # Reset environment
    obs, info = env.reset()
    print(f"Initial observation state keys: {list(obs['state'].keys())}")
    print(f"Initial observation image keys: {list(obs['images'].keys())}")

    # Run a few steps
    for step in range(5):
        # Generate random action: [left_arm_7dof, right_arm_7dof]
        action = env.action_space.sample() * 0.1  # Scale down for safety

        # Step environment
        obs, reward, done, truncated, info = env.step(action)

        print(f"Step {step + 1}: reward={reward}, done={done}")

        if done:
            print("Episode terminated, resetting...")
            obs, info = env.reset()

    env.close()
    print("✅ Basic usage example completed")


def example_handover_task():
    """Example using handover task configuration"""

    print("\n=== Handover Task Configuration Example ===")

    # Create handover task configuration
    config = HandoverTaskConfig()
    dual_config = config.get_dual_arm_config()

    # Modify for simulation testing
    dual_config["left_env_config"]["fake_env"] = True
    dual_config["right_env_config"]["fake_env"] = True

    env = HIROLDualArmEnv(**dual_config)

    print(f"Task type: {config.task_config.TASK_TYPE}")
    print(f"Reward strategy: {config.task_config.REWARD_STRATEGY}")
    print(f"Left arm target pose: {config.left_config.TARGET_POSE}")
    print(f"Right arm target pose: {config.right_config.TARGET_POSE}")

    # Reset and run simulation
    obs, info = env.reset()

    for step in range(3):
        # Simulate handover motion: left arm moves toward center, right arm receives
        left_action = np.array([0.05, -0.02, 0.0, 0.0, 0.0, 0.0, 0.0])  # Move left arm right
        right_action = np.array([-0.05, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0])  # Move right arm left

        action = np.concatenate([left_action, right_action])
        obs, reward, done, truncated, info = env.step(action)

        print(f"Handover step {step + 1}: reward={reward}")

    env.close()
    print("✅ Handover task example completed")


def example_bimanual_manipulation():
    """Example using bimanual manipulation configuration"""

    print("\n=== Bimanual Manipulation Configuration Example ===")

    # Create bimanual manipulation configuration
    config = BimanualManipulationConfig()
    dual_config = config.get_dual_arm_config()

    # Modify for simulation testing
    dual_config["left_env_config"]["fake_env"] = True
    dual_config["right_env_config"]["fake_env"] = True

    env = HIROLDualArmEnv(**dual_config)

    print(f"Task type: {config.task_config.TASK_TYPE}")
    print(f"Reward strategy: {config.task_config.REWARD_STRATEGY}")

    # Reset and run simulation
    obs, info = env.reset()

    for step in range(3):
        # Simulate coordinated bimanual motion
        # Both arms move in synchronized manner
        base_action = np.array([0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0])  # Move up
        left_action = base_action.copy()
        right_action = base_action.copy()

        action = np.concatenate([left_action, right_action])
        obs, reward, done, truncated, info = env.step(action)

        print(f"Bimanual step {step + 1}: reward={reward}")

    env.close()
    print("✅ Bimanual manipulation example completed")


def example_with_wrappers():
    """Example using dual-arm environment with wrappers"""

    print("\n=== Dual-Arm Environment with Wrappers Example ===")

    # Create base environment
    config = DualArmSimConfig()
    dual_config = config.get_dual_arm_config()
    dual_config["left_env_config"]["fake_env"] = True
    dual_config["right_env_config"]["fake_env"] = True

    # Create environment and apply wrappers
    env = HIROLDualArmEnv(**dual_config)

    # Apply dual quaternion to euler wrapper (if tcp_pose exists)
    try:
        env = DualQuat2EulerWrapper(env)
        print("✅ Applied DualQuat2EulerWrapper")
    except Exception as e:
        print(f"⚠️  DualQuat2EulerWrapper not applied: {e}")

    # Apply spacemouse intervention wrapper
    try:
        env = DualSpacemouseIntervention(env)
        print("✅ Applied DualSpacemouseIntervention")
    except Exception as e:
        print(f"⚠️  DualSpacemouseIntervention not applied: {e}")

    # Apply keyboard reward wrapper
    try:
        env = KeyboardRewardWrapper(env)
        print("✅ Applied KeyboardRewardWrapper")
    except Exception as e:
        print(f"⚠️  KeyboardRewardWrapper not applied: {e}")

    # Test wrapped environment
    obs, info = env.reset()
    print(f"Wrapped environment observation keys: {list(obs['state'].keys())}")

    # Run a few steps
    for step in range(2):
        action = np.random.uniform(-0.05, 0.05, 14)
        obs, reward, done, truncated, info = env.step(action)
        print(f"Wrapped step {step + 1}: reward={reward}")

    env.close()
    print("✅ Wrappers example completed")


def example_data_collection_pattern():
    """Example showing how to use HIROLDualArmEnv for data collection"""

    print("\n=== Data Collection Pattern Example ===")

    # Configuration for data collection
    config = DefaultDualArmConfig()
    dual_config = config.get_dual_arm_config()

    # Enable simulation for this example
    dual_config["left_env_config"]["fake_env"] = True
    dual_config["right_env_config"]["fake_env"] = True

    # Disable individual image display, use dual display
    dual_config["display_images"] = False  # Set to True to see images

    env = HIROLDualArmEnv(**dual_config)

    # Data collection simulation
    episode_data = []
    num_episodes = 2
    max_steps_per_episode = 5

    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")

        obs, info = env.reset()
        episode_trajectory = []

        for step in range(max_steps_per_episode):
            # Simulate expert demonstration or policy action
            action = np.random.uniform(-0.02, 0.02, 14)

            # Store trajectory data
            trajectory_point = {
                'observation': obs,
                'action': action,
                'step': step
            }

            # Step environment
            next_obs, reward, done, truncated, info = env.step(action)

            trajectory_point['reward'] = reward
            trajectory_point['done'] = done
            trajectory_point['next_observation'] = next_obs

            episode_trajectory.append(trajectory_point)
            obs = next_obs

            print(f"  Step {step + 1}: action_norm={np.linalg.norm(action):.3f}, reward={reward}")

            if done:
                break

        episode_data.append(episode_trajectory)
        print(f"  Episode {episode + 1} completed with {len(episode_trajectory)} steps")

    env.close()

    print(f"\n✅ Collected {len(episode_data)} episodes")
    print(f"Total trajectory points: {sum(len(ep) for ep in episode_data)}")
    print("✅ Data collection pattern example completed")


def main():
    """Run all examples"""

    print("🤖 HIROLDualArmEnv Usage Examples")
    print("=" * 60)

    examples = [
        ("Basic Usage", example_basic_usage),
        ("Handover Task", example_handover_task),
        ("Bimanual Manipulation", example_bimanual_manipulation),
        ("With Wrappers", example_with_wrappers),
        ("Data Collection Pattern", example_data_collection_pattern),
    ]

    for example_name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"❌ {example_name} example failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("🎉 All examples completed!")
    print("\nNext steps:")
    print("1. Run: python test_hirol_dual_arm_env.py (to test functionality)")
    print("2. Configure your specific robot setup in dual_arm_config.py")
    print("3. Integrate with your HIL-SERL training scripts")
    print("4. Use with existing dual-arm wrappers from serl_robot_infra")


if __name__ == "__main__":
    main()