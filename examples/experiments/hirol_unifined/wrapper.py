'''
Author: Haotian Liang haotianliang10@gmail.com
Date: 2025-08-13 10:55:43
LastEditors: Haotian Liang haotianliang10@gmail.com
LastEditTime: 2025-08-25 13:47:13
FilePath: /code/hil-serl/examples/experiments/fr3_reach/wrapper.py
'''
"""HIROL Unified Environment Wrapper"""

import sys
sys.path.insert(0, '/home/hanyu/code/hil-serl/serl_hirol_infra')

import numpy as np
import gymnasium as gym
from hirol_env.envs.hirol_env import HIROLEnv


class HIROLUnifiedEnv(HIROLEnv):
    """
    HIROL Unified task environment.
    
    This is a unified environment for various manipulation tasks using
    HIROLRobotPlatform. The task can be configured for reaching, grasping,
    placement, or other manipulation objectives.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def get_task_description(self):
        """Get a description of the task for logging"""
        return (
            f"HIROL Unified Task:\n"
            f"  Target: {self._TARGET_POSE[:3]}\n"
            f"  Threshold: {self._REWARD_THRESHOLD[:3]}\n"
            f"  Max steps: {self.max_episode_length}"
        )


class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.05, gripper_init_mode="open"):
        """
        Initialize gripper penalty wrapper
        
        Args:
            env: Base environment
            penalty: Penalty for gripper state changes
            gripper_init_mode: How to initialize gripper on reset ("open", "homing", or "none")
                              - "open": Call open_gripper() 
                              - "homing": Call gripper initialize for full homing
                              - "none": Don't reset gripper state
        """
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None
        self.gripper_init_mode = gripper_init_mode
        self._gripper_reset_enabled = True  # Track if gripper should be reset

    def _handle_gripper_reset(self):
        """Hook function called by base environment before waiting for Enter"""
        robot = getattr(self.env.unwrapped, 'robot', None)
        if self._gripper_reset_enabled and robot is not None:
            if self.gripper_init_mode == "open":
                print("[GripperPenaltyWrapper] Opening gripper")
                robot.open_gripper()
            elif self.gripper_init_mode == "homing":
                print("[GripperPenaltyWrapper] Homing gripper")
                # Call the gripper's initialize method for full homing
                if hasattr(robot, '_robot_system') and robot._robot_system:
                    robot_system = robot._robot_system
                    if hasattr(robot_system, '_tool') and robot_system._tool:
                        try:
                            robot_system._tool.initialize()
                            print("[GripperPenaltyWrapper] Gripper homing complete")
                        except Exception as e:
                            print(f"[GripperPenaltyWrapper] Gripper homing failed: {e}, falling back to open")
                            robot.open_gripper()
                    else:
                        # Fallback to open if tool not available
                        robot.open_gripper()
                else:
                    # Fallback to open if robot_system not available
                    robot.open_gripper()
            elif self.gripper_init_mode == "none":
                print("[GripperPenaltyWrapper] Maintaining gripper state")

    def reset(self, gripper_reset=True, **kwargs):
        """
        Reset the environment
        
        Args:
            gripper_reset: Whether to reset the gripper. Default True for backward compatibility.
                         If False, gripper state is maintained.
            **kwargs: Additional arguments to pass to the underlying environment
        """
        # Store gripper reset preference
        self._gripper_reset_enabled = gripper_reset
        
        # Set up the hook in the base environment
        self.env.unwrapped._pre_enter_hook = self._handle_gripper_reset
        
        # Call base environment reset (which will call our hook at the right time)
        obs, info = self.env.reset(**kwargs)
        
        # Update internal gripper state tracking
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