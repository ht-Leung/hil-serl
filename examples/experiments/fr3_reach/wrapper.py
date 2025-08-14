'''
Author: Haotian Liang haotianliang10@gmail.com
Date: 2025-08-13 10:55:43
LastEditors: Haotian Liang haotianliang10@gmail.com
LastEditTime: 2025-08-13 10:59:09
FilePath: /code/hil-serl/examples/experiments/fr3_reach/wrapper.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
"""FR3 Reach Environment Wrapper"""

import sys
sys.path.insert(0, '/home/hanyu/code/hil-serl/serl_hirol_infra')

import numpy as np
import gymnasium as gym
from hirol_env.envs.fr3_env import FR3Env


class FR3ReachEnv(FR3Env):
    """
    FR3 Reach task environment.
    
    This is a simple reaching task where the robot needs to move its
    end-effector to a target position. The task is successful when the
    end-effector is within a threshold distance from the target.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def get_task_description(self):
        """Get a description of the task for logging"""
        return (
            f"FR3 Reach Task:\n"
            f"  Target: {self._TARGET_POSE[:3]}\n"
            f"  Threshold: {self._REWARD_THRESHOLD[:3]}\n"
            f"  Max steps: {self.max_episode_length}"
        )


class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        gripper_pose = obs["state"]["gripper_pose"]
        # Handle both array and scalar cases
        if isinstance(gripper_pose, np.ndarray):
            self.last_gripper_pos = gripper_pose[0] if gripper_pose.shape else gripper_pose.item()
        else:
            self.last_gripper_pos = gripper_pose
        return obs, info

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]

        if (action[-1] < -0.5 and self.last_gripper_pos > 0.9) or (
            action[-1] > 0.5 and self.last_gripper_pos < 0.9
        ):
            info["grasp_penalty"] = self.penalty
        else:
            info["grasp_penalty"] = 0.0

        gripper_pose = observation["state"]["gripper_pose"]
        # Handle both array and scalar cases
        if isinstance(gripper_pose, np.ndarray):
            self.last_gripper_pos = gripper_pose[0] if gripper_pose.shape else gripper_pose.item()
        else:
            self.last_gripper_pos = gripper_pose
        return observation, reward, terminated, truncated, info