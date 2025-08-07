"""FR3 Reach Environment Wrapper"""

from serl_hirol_infra.hirol_env.envs.fr3_env import FR3Env


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