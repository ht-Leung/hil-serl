'''
Author: Haotian Liang haotianliang10@gmail.com
Date: 2025-08-05 15:48:57
LastEditors: Haotian Liang haotianliang10@gmail.com
LastEditTime: 2025-08-05 17:03:48
'''
from typing import OrderedDict, Dict, Tuple
import numpy as np
import time
from serl_hirol_infra.hirol_env.envs.hirol_env import HIROLEnv


class HIROLReachEnv(HIROLEnv):
    """HIROL environment for reach task"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_pose = self.config.TARGET_POSE.copy()
        
    def compute_reward(self, obs: Dict) -> bool:
        """
        Compute binary reward based on distance to target.
        Returns True if within threshold, False otherwise.
        """
        # Get current TCP pose from observation
        tcp_pose = obs["state"]["tcp_pose"]
        current_pos = tcp_pose[:3]
        current_quat = tcp_pose[3:7]
        
        # Convert quaternion to euler for easier comparison
        from serl_hirol_infra.hil_utils.rotations import quat_2_euler
        current_euler = quat_2_euler(current_quat)
        
        # Compute position and orientation errors
        pos_error = np.abs(current_pos - self.target_pose[:3])
        rot_error = np.abs(current_euler - self.target_pose[3:])
        
        # Wrap rotation error to [-pi, pi]
        rot_error = np.where(rot_error > np.pi, 2*np.pi - rot_error, rot_error)
        
        # Check if within threshold
        pos_within_threshold = np.all(pos_error < self.config.REWARD_THRESHOLD[:3])
        rot_within_threshold = np.all(rot_error < self.config.REWARD_THRESHOLD[3:])
        
        return bool(pos_within_threshold and rot_within_threshold)
    
    def get_reward(self) -> bool:
        """Legacy interface for compatibility"""
        obs = self._get_obs()
        return self.compute_reward(obs)
    
    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
        """Reset environment with optional target randomization"""
        # Optionally randomize target position
        if hasattr(self.config, 'RANDOMIZE_TARGET') and self.config.RANDOMIZE_TARGET:
            # Add small random offset to target position
            self.target_pose = self.config.TARGET_POSE.copy()
            self.target_pose[:2] += np.random.uniform(-0.03, 0.03, size=2)
            self.target_pose[2] += np.random.uniform(-0.02, 0.02)
        else:
            self.target_pose = self.config.TARGET_POSE.copy()
        
        # Call parent reset
        obs, info = super().reset(**kwargs)
        
        # Add target pose to info for visualization/debugging
        info["target_pose"] = self.target_pose.copy()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """Step the environment"""
        # Call parent step
        obs, reward, done, truncated, info = super().step(action)
        
        # Add target pose to info
        info["target_pose"] = self.target_pose.copy()
        
        # Check if task is successful
        if reward > 0:
            info["succeed"] = True
        else:
            info["succeed"] = False
            
        return obs, reward, done, truncated, info
    
    def go_to_reset(self, joint_reset: bool = True) -> None:
        """
        Custom reset procedure for reach task.
        Moves to reset position with optional randomization.
        """
        # Call parent go_to_reset which handles the actual movement
        super().go_to_reset(joint_reset=joint_reset)
        
        # Optional: Add any task-specific reset logic here
        # For reach task, the default behavior is sufficient