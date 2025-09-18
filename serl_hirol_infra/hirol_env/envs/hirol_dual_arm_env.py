"""HIROLDualArmEnv - Dual-arm environment for HIROL using two HIROLEnv instances"""
import queue
import threading
import time
import numpy as np
import gymnasium as gym
import cv2
from typing import Dict, Any, Optional, Tuple
import copy

from .hirol_env import HIROLEnv, DefaultEnvConfig


class ImageDisplayer(threading.Thread):
    """Thread for displaying dual-arm images in a combined window"""

    def __init__(self, queue_obj):
        threading.Thread.__init__(self)
        self.queue = queue_obj
        self.daemon = True  # make this a daemon thread

    def run(self):
        window_created = False
        while True:
            try:
                # Use timeout to avoid blocking forever
                img_array = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if img_array is None:  # None is our signal to exit
                if window_created:
                    try:
                        cv2.destroyWindow('HIROLDualArmEnv')
                        cv2.waitKey(1)
                    except:
                        pass
                break

            # Drop old frames to show latest
            while not self.queue.empty():
                try:
                    img_array = self.queue.get_nowait()
                    if img_array is None:
                        self.queue.put(None)  # Put back the exit signal
                        break
                except queue.Empty:
                    break

            try:
                # Separate left and right camera images
                left_images = []
                right_images = []

                for key, img in img_array.items():
                    if "left/" in key and "full" not in key:
                        left_images.append(cv2.resize(img, (256, 256)))
                    elif "right/" in key and "full" not in key:
                        right_images.append(cv2.resize(img, (256, 256)))

                # Combine images
                if left_images and right_images:
                    left_frame = np.concatenate(left_images, axis=1)
                    right_frame = np.concatenate(right_images, axis=1)
                    frame = np.concatenate([left_frame, right_frame], axis=1)
                elif left_images:
                    frame = np.concatenate(left_images, axis=1)
                elif right_images:
                    frame = np.concatenate(right_images, axis=1)
                else:
                    continue

                cv2.imshow('HIROLDualArmEnv', frame[..., ::-1])
                cv2.waitKey(1)
                window_created = True

            except Exception as e:
                # Ignore display errors but continue running
                pass


class DualArmEnvConfig:
    """Configuration class for dual-arm HIROL environment"""

    def __init__(self, left_config: Optional[DefaultEnvConfig] = None,
                 right_config: Optional[DefaultEnvConfig] = None):
        self.left_config = left_config or DefaultEnvConfig()
        self.right_config = right_config or DefaultEnvConfig()

        # Dual-arm specific parameters
        self.display_dual_images = True
        self.sync_reset = True
        self.sync_step = True


class HIROLDualArmEnv(gym.Env):
    """
    Dual-arm HIROL Environment that combines two HIROLEnv instances.

    This environment provides a unified interface for controlling two robot arms,
    similar to DualFrankaEnv but using HIROLEnv as the base environment.

    The observation and action spaces follow the same convention as DualFrankaEnv:
    - Actions: [left_arm_7dof, right_arm_7dof] = 14-dimensional
    - States: {"left/tcp_pose", "right/tcp_pose", ...} format
    - Images: {"left/camera_name", "right/camera_name", ...} format
    """

    def __init__(
        self,
        left_env_config: Dict[str, Any],
        right_env_config: Dict[str, Any],
        display_images: bool = True,
    ):
        """
        Initialize HIROLDualArmEnv with two independent HIROLEnv instances.

        Args:
            left_env_config: Configuration dict for left arm HIROLEnv
            right_env_config: Configuration dict for right arm HIROLEnv
            display_images: Whether to display combined camera images
        """
        # Create left and right arm environments
        self.env_left = HIROLEnv(**left_env_config)
        self.env_right = HIROLEnv(**right_env_config)

        # Setup dual-arm action and observation spaces
        self._setup_dual_spaces()

        # Setup dual image display
        self.display_images = display_images
        if self.display_images:
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue)
            self.displayer.start()

        print("[HIROLDualArmEnv] Initialized with left and right arm environments")

    def _setup_dual_spaces(self):
        """Setup combined action and observation spaces for dual-arm control"""
        # Action space: combine left (7D) + right (7D) = 14D
        left_action_dim = self.env_left.action_space.shape[0]
        right_action_dim = self.env_right.action_space.shape[0]
        action_dim = left_action_dim + right_action_dim

        self.action_space = gym.spaces.Box(
            np.ones((action_dim,), dtype=np.float32) * -1,
            np.ones((action_dim,), dtype=np.float32),
        )

        # Observation space: combine with left/ and right/ prefixes
        # State space
        left_state_space = self.env_left.observation_space["state"]
        right_state_space = self.env_right.observation_space["state"]

        state_dict = {}
        # Add left arm states with "left/" prefix
        for key, space in left_state_space.spaces.items():
            state_dict[f"left/{key}"] = space
        # Add right arm states with "right/" prefix
        for key, space in right_state_space.spaces.items():
            state_dict[f"right/{key}"] = space

        # Image space
        left_image_space = self.env_left.observation_space["images"]
        right_image_space = self.env_right.observation_space["images"]

        image_dict = {}
        # Add left arm images with "left/" prefix
        for key, space in left_image_space.spaces.items():
            image_dict[f"left/{key}"] = space
        # Add right arm images with "right/" prefix
        for key, space in right_image_space.spaces.items():
            image_dict[f"right/{key}"] = space

        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Dict(state_dict),
            "images": gym.spaces.Dict(image_dict)
        })

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Step both arms in parallel with the given action.

        Args:
            action: 14-dimensional action [left_7dof, right_7dof]

        Returns:
            observation: Combined observation with left/ and right/ prefixes
            reward: Combined reward (logical AND of both arms)
            terminated: Combined termination (logical OR of both arms)
            truncated: Combined truncation (logical OR of both arms)
            info: Combined info dict
        """
        # Split action into left and right components
        left_action_dim = self.env_left.action_space.shape[0]
        action_left = action[:left_action_dim]
        action_right = action[left_action_dim:]

        # Execute actions in parallel using threads
        def step_env_left():
            global ob_left, reward_left, done_left, truncated_left, info_left
            ob_left, reward_left, done_left, truncated_left, info_left = self.env_left.step(action_left)

        def step_env_right():
            global ob_right, reward_right, done_right, truncated_right, info_right
            ob_right, reward_right, done_right, truncated_right, info_right = self.env_right.step(action_right)

        # Create and start threads
        thread_left = threading.Thread(target=step_env_left)
        thread_right = threading.Thread(target=step_env_right)
        thread_left.start()
        thread_right.start()

        # Wait for both threads to complete
        thread_left.join()
        thread_right.join()

        # Combine observations
        combined_obs = self.combine_obs(ob_left, ob_right)

        # Display combined images if enabled
        if self.display_images:
            self.img_queue.put(combined_obs['images'])

        # Combine rewards, termination, and info
        combined_reward = int(reward_left and reward_right)  # Both arms must succeed
        combined_done = done_left or done_right  # Either arm terminating ends episode
        combined_truncated = truncated_left or truncated_right

        combined_info = {
            "left": info_left,
            "right": info_right,
            "succeed": bool(reward_left and reward_right)
        }

        return combined_obs, combined_reward, combined_done, combined_truncated, combined_info

    def reset(self, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset both arm environments in parallel.

        Returns:
            observation: Combined observation with left/ and right/ prefixes
            info: Combined info dict
        """
        # Reset both environments in parallel
        def reset_env_left():
            global ob_left, info_left
            ob_left, info_left = self.env_left.reset(**kwargs)

        def reset_env_right():
            global ob_right, info_right
            ob_right, info_right = self.env_right.reset(**kwargs)

        thread_left = threading.Thread(target=reset_env_left)
        thread_right = threading.Thread(target=reset_env_right)
        thread_left.start()
        thread_right.start()
        thread_left.join()
        thread_right.join()

        # Combine observations
        combined_obs = self.combine_obs(ob_left, ob_right)

        # Combine info
        combined_info = {
            "left": info_left,
            "right": info_right
        }

        return combined_obs, combined_info

    def combine_obs(self, ob_left: Dict[str, Any], ob_right: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine left and right observations into dual-arm format.

        Args:
            ob_left: Left arm observation dict
            ob_right: Right arm observation dict

        Returns:
            Combined observation with left/ and right/ prefixes
        """
        # Combine state observations with prefixes
        left_state = {f"left/{key}": value for key, value in ob_left["state"].items()}
        right_state = {f"right/{key}": value for key, value in ob_right["state"].items()}

        # Combine image observations with prefixes
        left_images = {f"left/{key}": value for key, value in ob_left["images"].items()}
        right_images = {f"right/{key}": value for key, value in ob_right["images"].items()}

        combined_obs = {
            "state": {**left_state, **right_state},
            "images": {**left_images, **right_images}
        }

        return combined_obs

    def close(self):
        """Clean up resources for both environments"""
        try:
            # Stop image display
            if self.display_images and hasattr(self, 'img_queue'):
                self.img_queue.put(None)  # Signal displayer to stop
                if hasattr(self, 'displayer'):
                    self.displayer.join(timeout=1.0)

            # Close both environments
            self.env_left.close()
            self.env_right.close()

            print("[HIROLDualArmEnv] Successfully closed both arm environments")

        except Exception as e:
            print(f"[HIROLDualArmEnv] Error during cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.close()