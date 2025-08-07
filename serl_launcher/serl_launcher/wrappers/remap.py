'''
Author: Haotian Liang haotianliang10@gmail.com
Date: 2025-08-04 09:58:54
LastEditors: Haotian Liang haotianliang10@gmail.com
LastEditTime: 2025-08-04 10:01:16
FilePath: /code/hil-serl/serl_launcher/serl_launcher/wrappers/remap.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from typing import Any

import gymnasium as gym
import gymnasium.spaces
import jax


class RemapWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, new_structure: Any):
        """
        Remap a dictionary observation space to some other flat structure specified by keys.

        Params:
            env: Environment to wrap.
            new_structure: A tuple/dictionary/singleton where leaves are keys in the original observation space.
        """
        super().__init__(env)
        self.new_structure = new_structure

        if isinstance(new_structure, tuple):
            self.observation_space = gym.spaces.Tuple(
                [env.observation_space[v] for v in new_structure]
            )
        elif isinstance(new_structure, dict):
            self.observation_space = gym.spaces.Dict(
                {k: env.observation_space[v] for k, v in new_structure.items()}
            )
        elif isinstance(new_structure, str):
            self.observation_space = env.observation_space[new_structure]
        else:
            raise TypeError(f"Unsupported type {type(new_structure)}")

    def observation(self, observation):
        return jax.tree.map(lambda x: observation[x], self.new_structure)
