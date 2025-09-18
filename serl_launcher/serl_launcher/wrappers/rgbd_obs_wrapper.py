"""
RGBD Observation Wrapper for HIL-SERL
Handles RGB+Depth observation space compatibility
"""
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np


class RGBDObsWrapper(gym.ObservationWrapper):
    """
    RGBD观测包装器，用于处理RGB+Depth数据的observation space兼容性

    当环境配置为输出RGBD数据（4通道）时，正确更新observation space
    以确保与replay buffer和多模态编码器的兼容性。

    Args:
        env: 环境实例
        image_keys: 图像键名列表（如 ['wrist_1', 'front', 'side']）
        use_depth: 是否启用深度处理，None表示自动检测
    """

    def __init__(self, env, image_keys=None, use_depth=None):
        super().__init__(env)
        self.image_keys = image_keys or []
        self.use_depth = use_depth

        # 检测并更新observation space
        self._update_observation_space()

    def _update_observation_space(self):
        """检测RGBD格式并更新observation space"""
        # 获取样本观测来检测数据格式
        try:
            sample_obs, _ = self.env.reset()
            rgbd_detected = False

            # 检测是否有RGBD数据
            if self.use_depth is None:
                # 自动检测模式
                for key in self.image_keys:
                    if key in sample_obs.get("images", {}):
                        img_shape = sample_obs["images"][key].shape
                        if len(img_shape) >= 3 and img_shape[-1] == 4:
                            rgbd_detected = True
                            print(f"检测到RGBD数据: {key} shape={img_shape}")
                            break
            else:
                rgbd_detected = self.use_depth

            # 如果检测到RGBD数据，更新observation space
            if rgbd_detected:
                self._create_rgbd_observation_space(sample_obs)
                print("✓ 已更新observation space支持RGBD格式")
            else:
                print("✓ 使用标准RGB observation space")

        except Exception as e:
            print(f"警告: RGBD检测失败，使用默认observation space: {e}")

    def _create_rgbd_observation_space(self, sample_obs):
        """创建支持RGBD的observation space"""
        if not hasattr(self.env, 'observation_space') or not hasattr(self.env.observation_space, 'spaces'):
            return

        # 创建新的observation space字典
        new_spaces = {}

        # 复制现有的非图像空间
        for key, space in self.env.observation_space.spaces.items():
            if key not in self.image_keys:
                new_spaces[key] = space
            else:
                # 更新图像空间为4通道RGBD
                if key in sample_obs.get("images", {}):
                    img_shape = sample_obs["images"][key].shape
                    if len(img_shape) >= 3 and img_shape[-1] == 4:
                        # 4通道RGBD格式
                        new_spaces[key] = Box(
                            low=0,
                            high=255,
                            shape=img_shape,
                            dtype=np.uint8
                        )
                        print(f"  更新 {key}: {img_shape} (RGBD)")
                    else:
                        # 保持原来的3通道RGB
                        new_spaces[key] = space
                        print(f"  保持 {key}: {space.shape} (RGB)")
                else:
                    # 键不存在，保持原空间
                    new_spaces[key] = space

        # 更新observation space
        self.observation_space = gym.spaces.Dict(new_spaces)

    def observation(self, obs):
        """观测转换（直接返回，不修改数据）"""
        return obs

    def reset(self, **kwargs):
        """重置环境"""
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


class SERLRGBDObsWrapper(gym.ObservationWrapper):
    """
    结合SERL观测处理和RGBD支持的包装器

    这个包装器结合了SERLObsWrapper的功能和RGBD数据的支持，
    确保observation space正确反映数据格式。
    """

    def __init__(self, env, proprio_keys=None, image_keys=None, use_depth=None):
        super().__init__(env)
        self.proprio_keys = proprio_keys
        self.image_keys = image_keys or []
        self.use_depth = use_depth

        # 初始化proprio keys
        if self.proprio_keys is None:
            self.proprio_keys = list(self.env.observation_space["state"].keys())

        # 创建proprio space
        self.proprio_space = gym.spaces.Dict(
            {key: self.env.observation_space["state"][key] for key in self.proprio_keys}
        )

        # 检测RGBD并创建observation space
        self._create_observation_space()

    def _create_observation_space(self):
        """创建支持RGBD的SERL observation space"""
        try:
            # 检查是否为fake_env模式，避免在未初始化的环境上调用reset
            is_fake_env = getattr(self.env, 'fake_env', False) or getattr(self.env, '_fake_env', False)

            rgbd_spaces = {}

            if not is_fake_env:
                # 真实环境：获取样本数据
                sample_obs, _ = self.env.reset()
                images = sample_obs.get("images", {})

                for key in self.image_keys:
                    if key in images:
                        img_shape = images[key].shape

                        # 根据实际数据形状创建空间
                        if len(img_shape) >= 3 and img_shape[-1] == 4 and self.use_depth:
                            # 4通道RGBD
                            rgbd_spaces[key] = Box(low=0, high=255, shape=img_shape, dtype=np.uint8)
                            print(f"SERL-RGBD: {key} -> {img_shape} (RGBD)")
                        elif len(img_shape) >= 3 and img_shape[-1] == 3:
                            # 3通道RGB
                            rgbd_spaces[key] = Box(low=0, high=255, shape=img_shape, dtype=np.uint8)
                            print(f"SERL-RGBD: {key} -> {img_shape} (RGB)")
                        else:
                            # 其他格式，从原始observation space获取
                            if hasattr(self.env, 'observation_space') and "images" in self.env.observation_space:
                                if key in self.env.observation_space["images"].spaces:
                                    rgbd_spaces[key] = self.env.observation_space["images"].spaces[key]
            else:
                # fake_env模式：基于配置和原始observation space推断
                print("检测到fake_env模式，基于配置推断RGBD observation space")

                # 尝试从环境的observation space获取图像空间
                if hasattr(self.env, 'observation_space') and "images" in self.env.observation_space.spaces:
                    base_image_spaces = self.env.observation_space.spaces["images"].spaces

                    for key in self.image_keys:
                        if key in base_image_spaces:
                            base_shape = base_image_spaces[key].shape

                            if self.use_depth:
                                # 假设RGBD格式：保持前面维度，最后一维改为4
                                rgbd_shape = base_shape[:-1] + (4,)
                                rgbd_spaces[key] = Box(low=0, high=255, shape=rgbd_shape, dtype=np.uint8)
                                print(f"SERL-RGBD (推断): {key} -> {rgbd_shape} (RGBD)")
                            else:
                                # RGB格式：保持原始形状
                                rgbd_spaces[key] = base_image_spaces[key]
                                print(f"SERL-RGBD (推断): {key} -> {base_shape} (RGB)")
                        else:
                            # 默认图像空间：128x128 RGB或RGBD
                            channels = 4 if self.use_depth else 3
                            default_shape = (1, 128, 128, channels)
                            rgbd_spaces[key] = Box(low=0, high=255, shape=default_shape, dtype=np.uint8)
                            print(f"SERL-RGBD (默认): {key} -> {default_shape} ({'RGBD' if self.use_depth else 'RGB'})")
                else:
                    # 完全默认情况：创建标准图像空间
                    for key in self.image_keys:
                        channels = 4 if self.use_depth else 3
                        default_shape = (1, 128, 128, channels)
                        rgbd_spaces[key] = Box(low=0, high=255, shape=default_shape, dtype=np.uint8)
                        print(f"SERL-RGBD (完全默认): {key} -> {default_shape} ({'RGBD' if self.use_depth else 'RGB'})")

            # 创建最终的observation space
            self.observation_space = gym.spaces.Dict({
                "state": gym.spaces.utils.flatten_space(self.proprio_space),
                **rgbd_spaces
            })

        except Exception as e:
            print(f"警告: SERL-RGBD空间创建失败，使用默认: {e}")
            # 回退到标准SERL格式
            try:
                if hasattr(self.env, 'observation_space') and "images" in self.env.observation_space.spaces:
                    self.observation_space = gym.spaces.Dict({
                        "state": gym.spaces.utils.flatten_space(self.proprio_space),
                        **(self.env.observation_space.spaces["images"].spaces),
                    })
                else:
                    # 最终回退：创建基本空间
                    basic_spaces = {}
                    for key in self.image_keys:
                        channels = 4 if self.use_depth else 3
                        basic_spaces[key] = Box(low=0, high=255, shape=(1, 128, 128, channels), dtype=np.uint8)

                    self.observation_space = gym.spaces.Dict({
                        "state": gym.spaces.utils.flatten_space(self.proprio_space),
                        **basic_spaces
                    })
            except Exception as fallback_e:
                print(f"严重错误: 无法创建observation space: {fallback_e}")
                raise

    def observation(self, obs):
        """SERL格式观测转换"""
        from gymnasium.spaces.utils import flatten

        return {
            "state": flatten(
                self.proprio_space,
                {key: obs["state"][key] for key in self.proprio_keys},
            ),
            **(obs["images"]),
        }

    def reset(self, **kwargs):
        """重置环境"""
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


# 便捷函数
def wrap_rgbd_env(env, image_keys, use_depth=True, proprio_keys=None):
    """
    便捷函数：为环境添加RGBD支持

    Args:
        env: 环境实例
        image_keys: 图像键列表
        use_depth: 是否启用深度
        proprio_keys: 本体感觉键列表

    Returns:
        包装后的环境
    """
    if use_depth:
        # 使用结合的SERL+RGBD包装器
        return SERLRGBDObsWrapper(
            env,
            proprio_keys=proprio_keys,
            image_keys=image_keys,
            use_depth=use_depth
        )
    else:
        # 使用标准SERL包装器
        from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
        return SERLObsWrapper(env, proprio_keys=proprio_keys)