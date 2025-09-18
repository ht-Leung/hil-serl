"""
多模态编码器：支持多相机RGB + 深度点云的并行处理
每个相机都有独立的RGB和深度编码器
"""
from typing import Dict, Iterable
import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange

from serl_launcher.vision.segnn_encoder import SegNNEncoder


class MultiModalEncodingWrapper(nn.Module):
    """
    多模态编码包装器：多相机RGB + 深度 + 状态

    Args:
        rgb_encoder: RGB编码器字典 {camera_name: encoder}
        depth_encoder: 深度编码器字典 {camera_name: segnn_encoder}
        use_proprio: 是否使用本体感觉
        proprio_latent_dim: 本体感觉潜在维度
        enable_stacking: 是否启用堆叠
        camera_keys: 相机名称列表 ['wrist_1', 'front', 'side']
    """
    rgb_encoder: Dict[str, nn.Module]     # {camera_name: rgb_encoder}
    depth_encoder: Dict[str, SegNNEncoder]  # {camera_name: depth_encoder}
    use_proprio: bool = False
    proprio_latent_dim: int = 64
    enable_stacking: bool = False
    camera_keys: Iterable[str] = ("wrist_1", "front", "side")

    @nn.compact
    def __call__(
        self,
        observations: Dict[str, jnp.ndarray],
        train: bool = False,
        stop_gradient: bool = False,
        is_encoded: bool = False,
    ) -> jnp.ndarray:
        """
        多模态编码

        期望的观测格式：
        - RGB: 'wrist_1', 'front', 'side' 等
        - 深度: 'depth_wrist_1', 'depth_front', 'depth_side' 等
        - 状态: 'state'
        """
        encoded_features = []

        # 遍历每个相机
        for camera_name in self.camera_keys:
            camera_features = []

            # 1. RGB编码（如果存在）
            if camera_name in observations and camera_name in self.rgb_encoder:
                rgb_image = observations[camera_name]

                # 检查是否是RGBD格式，如果是则拆分RGB
                if len(rgb_image.shape) >= 3 and rgb_image.shape[-1] == 4:
                    # 4通道数据，拆分RGB和深度
                    rgb_image = rgb_image[..., :3]  # 取前3个通道作为RGB
                    # print(f"检测到4通道数据，自动拆分RGB: {rgb_image.shape}")

                if not is_encoded and self.enable_stacking:
                    # 处理时序堆叠
                    if len(rgb_image.shape) == 4:
                        rgb_image = rearrange(rgb_image, "T H W C -> H W (T C)")
                    elif len(rgb_image.shape) == 5:
                        rgb_image = rearrange(rgb_image, "B T H W C -> B H W (T C)")

                # RGB编码
                rgb_features = self.rgb_encoder[camera_name](
                    rgb_image, train=train, encode=not is_encoded
                )

                if stop_gradient:
                    rgb_features = jax.lax.stop_gradient(rgb_features)

                camera_features.append(rgb_features)
                # print(f"RGB {camera_name}: {rgb_features.shape}")

            # 2. 深度编码
            depth_key = f"depth_{camera_name}"
            has_separate_depth = depth_key in observations
            has_rgbd_data = (camera_name in observations and
                           len(observations[camera_name].shape) >= 3 and
                           observations[camera_name].shape[-1] == 4)

            if ((has_separate_depth or has_rgbd_data) and
                camera_name in observations and
                camera_name in self.depth_encoder):

                if has_separate_depth:
                    # 独立深度数据
                    depth_image = observations[depth_key]
                    rgb_for_depth = observations[camera_name]
                    if len(rgb_for_depth.shape) >= 3 and rgb_for_depth.shape[-1] == 4:
                        rgb_for_depth = rgb_for_depth[..., :3]  # 提取RGB部分
                else:
                    # 从4通道RGBD数据中提取深度
                    rgbd_data = observations[camera_name]
                    depth_image = rgbd_data[..., 3]  # 第4个通道是深度
                    rgb_for_depth = rgbd_data[..., :3]  # 前3个通道是RGB
                    # print(f"从RGBD数据提取深度: depth={depth_image.shape}, rgb={rgb_for_depth.shape}")

                # 深度点云编码
                depth_features = self.depth_encoder[camera_name](
                    depth_image, rgb_for_depth, train=train
                )

                if stop_gradient:
                    depth_features = jax.lax.stop_gradient(depth_features)

                camera_features.append(depth_features)
                # print(f"Depth {camera_name}: {depth_features.shape}")

            # 3. 合并该相机的特征
            if camera_features:
                # print(f"Camera {camera_name} has {len(camera_features)} features")
                if len(camera_features) == 1:
                    camera_encoded = camera_features[0]
                else:
                    # print(f"Concatenating features: {[f.shape for f in camera_features]}")

                    # 确保所有特征具有相同的batch维度
                    normalized_features = []
                    target_ndim = max(len(f.shape) for f in camera_features)

                    for feat in camera_features:
                        if len(feat.shape) < target_ndim:
                            # 为缺少batch维度的特征添加batch维度
                            feat = jnp.expand_dims(feat, axis=0)
                        normalized_features.append(feat)

                    # print(f"Normalized features: {[f.shape for f in normalized_features]}")
                    camera_encoded = jnp.concatenate(normalized_features, axis=-1)
                    # print(f"Camera {camera_name} encoded: {camera_encoded.shape}")
                encoded_features.append(camera_encoded)

        # 4. 本体感觉编码（可选）
        if self.use_proprio and 'state' in observations:
            state = observations['state']

            if self.enable_stacking:
                if len(state.shape) == 2:
                    state = rearrange(state, "T C -> (T C)")
                    # 调整其他特征的形状以匹配
                    encoded_features = [f.reshape(-1) if len(f.shape) > 1 else f
                                      for f in encoded_features]
                elif len(state.shape) == 3:
                    state = rearrange(state, "B T C -> B (T C)")
            else:
                # 不启用stacking时，确保状态维度与相机特征匹配
                if len(state.shape) == 3:
                    # [B, T, C] → [B, T*C] (取最后一帧或flatten)
                    state = state[:, -1, :]  # 取最后一帧: [B, T, C] → [B, C]

            # 状态编码
            state_features = nn.Dense(
                self.proprio_latent_dim,
                kernel_init=nn.initializers.xavier_uniform()
            )(state)
            state_features = nn.LayerNorm()(state_features)
            state_features = nn.tanh(state_features)
            encoded_features.append(state_features)

        # 5. 特征融合
        if len(encoded_features) == 0:
            raise ValueError("No valid modalities found in observations")

        # 确保特征形状兼容
        if len(encoded_features) > 1:
            # 获取参考形状（除了最后一维）
            reference_shape = encoded_features[0].shape[:-1]
            adjusted_features = []

            for feat in encoded_features:
                if feat.shape[:-1] != reference_shape:
                    # 处理维度不匹配
                    if feat.ndim == 1 and len(reference_shape) > 0:
                        # 扩展1D特征
                        feat = jnp.expand_dims(feat, axis=0)
                        if feat.shape[:-1] != reference_shape:
                            feat = jnp.broadcast_to(feat, reference_shape + (feat.shape[-1],))
                adjusted_features.append(feat)

            encoded_features = adjusted_features

        # 最终拼接
        encoded = jnp.concatenate(encoded_features, axis=-1)
        return encoded


def create_multimodal_encoder(
    rgb_encoder_dict: Dict[str, nn.Module],
    depth_encoder_dict: Dict[str, SegNNEncoder],
    use_proprio: bool = False,
    camera_keys: Iterable[str] = ("wrist_1", "front", "side"),
) -> MultiModalEncodingWrapper:
    """
    创建多模态编码器工厂函数

    Args:
        rgb_encoder_dict: RGB编码器字典 {camera_name: encoder}
        depth_encoder_dict: 深度编码器字典 {camera_name: segnn_encoder}
        use_proprio: 是否使用状态信息
        camera_keys: 相机名称列表

    Returns:
        多模态编码器
    """
    return MultiModalEncodingWrapper(
        rgb_encoder=rgb_encoder_dict,
        depth_encoder=depth_encoder_dict,
        use_proprio=use_proprio,
        enable_stacking=True,
        camera_keys=camera_keys
    )


# 测试代码
if __name__ == "__main__":
    print("🧪 测试多相机多模态编码器")

    # 模拟编码器 - 确保输出形状一致
    class MockRGBEncoder(nn.Module):
        @nn.compact
        def __call__(self, x, train=False, encode=True):
            print(f"RGB encoder input shape: {x.shape}")
            if encode:
                # 模拟ResNet输出，保持batch维度
                batch_size = x.shape[0]
                print(f"RGB encoder batch size: {batch_size}")
                x_flat = x.reshape(batch_size, -1)
                print(f"RGB encoder flattened shape: {x_flat.shape}")
                features = nn.Dense(256)(x_flat)
                print(f"RGB encoder output shape: {features.shape}")
                return features
            return x

    class MockDepthEncoder(nn.Module):
        @nn.compact
        def __call__(self, depth, rgb, train=False):
            # 模拟SegNN输出，处理时序维度 [B,T,H,W] → [B,C]
            batch_size = depth.shape[0]
            print(f"Depth encoder input shapes: depth={depth.shape}, rgb={rgb.shape}")
            print(f"Depth encoder batch size: {batch_size}")
            # 如果有时序维度，取最后一帧 (模拟实际SegNN处理)
            if len(depth.shape) == 4:
                depth = depth[:, -1, :, :]  # [B,T,H,W] → [B,H,W]
            if len(rgb.shape) == 5:
                rgb = rgb[:, -1, :, :, :]  # [B,T,H,W,C] → [B,H,W,C]
            # 创建正确的输入形状
            dummy_input = jnp.ones((batch_size, 64))
            features = nn.Dense(128)(dummy_input)
            print(f"Depth encoder output shape: {features.shape}")
            return features

    # 创建测试数据
    key = jax.random.PRNGKey(42)
    batch_size = 2
    cameras = ["wrist_1", "front", "side"]

    # 模拟真实HIL-SERL数据流：ChunkingWrapper(obs_horizon=1) + 批处理
    observations = {
        # RGB图像：5维 [B, T=1, H, W, C] (ChunkingWrapper输出)
        "wrist_1": jax.random.normal(key, (batch_size, 1, 128, 128, 3)),
        "front": jax.random.normal(key, (batch_size, 1, 128, 128, 3)),
        "side": jax.random.normal(key, (batch_size, 1, 128, 128, 3)),
        # 深度图像：4维 [B, T=1, H, W]
        "depth_wrist_1": jax.random.normal(key, (batch_size, 1, 128, 128)),
        "depth_front": jax.random.normal(key, (batch_size, 1, 128, 128)),
        "depth_side": jax.random.normal(key, (batch_size, 1, 128, 128)),
        # 状态：3维 [B, T=1, C]
        "state": jax.random.normal(key, (batch_size, 1, 10))
    }

    # 创建编码器字典
    rgb_encoders = {cam: MockRGBEncoder() for cam in cameras}
    depth_encoders = {cam: MockDepthEncoder() for cam in cameras}

    # 创建编码器，使用正确的5维时序数据测试stacking
    encoder = MultiModalEncodingWrapper(
        rgb_encoder=rgb_encoders,
        depth_encoder=depth_encoders,
        use_proprio=True,
        enable_stacking=True,  # 现在使用5维数据，应该正确处理
        camera_keys=cameras
    )

    # 测试
    params = encoder.init(key, observations, train=True)
    features = encoder.apply(params, observations, train=True)

    print(f"✓ 相机数量: {len(cameras)}")
    print(f"✓ RGB输入形状: {[observations[cam].shape for cam in cameras]}")
    print(f"✓ 深度输入形状: {[observations[f'depth_{cam}'].shape for cam in cameras]}")
    print(f"✓ 输出特征形状: {features.shape}")
    print("🎯 多相机多模态编码器测试成功!")