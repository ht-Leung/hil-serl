"""
SegNN Encoder for HIL-SERL
将SegNN点云编码器集成到SERL框架中，作为第三并行分支
"""
import sys
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional

# 添加SegNN模块路径
sys.path.append('/home/hanyu/code/Seg-NN')
from models.encoder_jax_optimized import Encoder_Seg_Optimized

from serl_launcher.vision.depth_to_pointcloud import DepthToPointCloud


class CustomDepthConverter(DepthToPointCloud):
    """
    带相机参数的自定义深度转换器
    """
    camera_fx: float = 615.0
    camera_fy: float = 615.0
    camera_cx: float = 320.0
    camera_cy: float = 240.0

    def setup(self):
        # 在setup中设置相机参数
        self.fx = self.camera_fx
        self.fy = self.camera_fy
        self.cx = self.camera_cx
        self.cy = self.camera_cy


class SegNNEncoder(nn.Module):
    """
    SegNN点云编码器，用于处理深度+RGB图像

    Args:
        input_points: SegNN编码器的输入点数
        num_stages: SegNN编码器的阶段数
        embed_dim: SegNN编码器的嵌入维度
        k_neighbors: K近邻数量
        de_neighbors: 解码近邻数
        bottleneck_dim: 输出瓶颈维度
        depth_params: 深度转换器参数字典
    """
    input_points: int = 2048
    num_stages: int = 3
    embed_dim: int = 72
    k_neighbors: int = 32
    de_neighbors: int = 6
    alpha: float = 1000
    beta: float = 50
    bottleneck_dim: Optional[int] = 256

    # 深度转换器参数
    min_depth: float = 0.1
    max_depth: float = 2.0
    camera_fx: float = 615.0
    camera_fy: float = 615.0
    camera_cx: float = 320.0
    camera_cy: float = 240.0

    def setup(self):
        # 创建深度转换器（相机参数通过自定义类传递）
        self.depth_converter = CustomDepthConverter(
            max_points=self.input_points,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            camera_fx=self.camera_fx,
            camera_fy=self.camera_fy,
            camera_cx=self.camera_cx,
            camera_cy=self.camera_cy,
            add_noise=False
        )

        # 创建SegNN编码器
        self.segnn_encoder = Encoder_Seg_Optimized(
            input_points=self.input_points,
            num_stages=self.num_stages,
            embed_dim=self.embed_dim,
            k_neighbors=self.k_neighbors,
            de_neighbors=self.de_neighbors,
            alpha=self.alpha,
            beta=self.beta
        )

        # 添加瓶颈层
        if self.bottleneck_dim is not None:
            self.bottleneck = nn.Dense(self.bottleneck_dim)
            self.layer_norm = nn.LayerNorm()

    @nn.compact
    def __call__(self, depth_image: jnp.ndarray, rgb_image: jnp.ndarray,
                 train: bool = False) -> jnp.ndarray:
        """
        前向传播

        Args:
            depth_image: [H, W] 或 [B, H, W] 深度图
            rgb_image: [H, W, 3] 或 [B, H, W, 3] RGB图像
            train: 训练模式

        Returns:
            features: [bottleneck_dim] 或 [B, bottleneck_dim] 点云特征
        """
        # 生成随机种子用于点云采样
        if train:
            # 训练时使用随机种子
            key = self.make_rng('pointcloud_sampling')
        else:
            # 推理时使用固定种子保证一致性
            key = jax.random.PRNGKey(42)

        # 处理batch和时序维度
        if len(depth_image.shape) == 4:
            # [B, T, H, W] -> 取第一个样本和最后一帧
            batch_size = depth_image.shape[0]
            depth_image_single = depth_image[0, -1]  # [H, W]
            rgb_image_single = rgb_image[0, -1]      # [H, W, 3]
        elif len(depth_image.shape) == 3:
            # [B, H, W] -> 取第一个样本
            batch_size = depth_image.shape[0]
            depth_image_single = depth_image[0]  # [H, W]
            rgb_image_single = rgb_image[0]      # [H, W, 3]
        else:
            # [H, W] -> 直接处理
            batch_size = 1
            depth_image_single = depth_image     # [H, W]
            rgb_image_single = rgb_image         # [H, W, 3]

        # 1. 深度图转点云
        pointcloud = self.depth_converter(depth_image_single, rgb_image_single, key, train=train)

        # 2. 添加batch维度 [1, N, 9]
        pointcloud = jnp.expand_dims(pointcloud, axis=0)

        # 3. SegNN编码 (使用training模式)
        features = self.segnn_encoder(pointcloud, variant='training')

        # 4. 移除batch维度并展平 [N, C] -> [N*C]
        features = features.reshape(-1)

        # 5. 瓶颈层
        if self.bottleneck_dim is not None:
            features = self.bottleneck(features)
            features = self.layer_norm(features)
            features = nn.tanh(features)

        # 6. 恢复batch维度（如果需要）
        if batch_size > 1:
            # 扩展特征以匹配batch大小
            features = jnp.expand_dims(features, axis=0)
            features = jnp.repeat(features, batch_size, axis=0)

        return features


def create_segnn_encoder(
    input_points: int = 2048,
    num_stages: int = 3,
    embed_dim: int = 72,
    bottleneck_dim: int = 256,
    camera_params: Optional[dict] = None
) -> SegNNEncoder:
    """
    创建SegNN编码器的工厂函数

    Args:
        input_points: 输入点数
        num_stages: 编码阶段数
        embed_dim: 嵌入维度
        bottleneck_dim: 瓶颈维度
        camera_params: 相机参数字典

    Returns:
        SegNN编码器实例
    """
    encoder_kwargs = {
        'input_points': input_points,
        'num_stages': num_stages,
        'embed_dim': embed_dim,
        'bottleneck_dim': bottleneck_dim,
    }

    if camera_params:
        encoder_kwargs.update({
            'camera_fx': camera_params.get('fx', 615.0),
            'camera_fy': camera_params.get('fy', 615.0),
            'camera_cx': camera_params.get('cx', 320.0),
            'camera_cy': camera_params.get('cy', 240.0),
        })

    return SegNNEncoder(**encoder_kwargs)


# 测试代码
if __name__ == "__main__":
    print("🧪 测试SegNN编码器集成")

    # 创建测试数据
    key = jax.random.PRNGKey(42)
    H, W = 240, 320  # 较小的分辨率用于测试

    depth_key, rgb_key, test_key = jax.random.split(key, 3)
    depth = jax.random.uniform(depth_key, (H, W), minval=0.5, maxval=1.5)
    rgb = jax.random.uniform(rgb_key, (H, W, 3), minval=0, maxval=255)

    # 创建编码器
    encoder = create_segnn_encoder(
        input_points=512,  # 较少点数用于快速测试
        num_stages=2,
        embed_dim=36,
        bottleneck_dim=128
    )

    # 初始化参数
    init_key, apply_key = jax.random.split(test_key)
    params = encoder.init({'params': init_key, 'pointcloud_sampling': apply_key},
                         depth, rgb, train=True)

    # 执行编码
    features = encoder.apply(params, depth, rgb, train=True,
                           rngs={'pointcloud_sampling': apply_key})

    print(f"✓ 输入深度图形状: {depth.shape}")
    print(f"✓ 输入RGB图形状: {rgb.shape}")
    print(f"✓ 输出特征形状: {features.shape}")
    print(f"✓ 特征范围: [{jnp.min(features):.3f}, {jnp.max(features):.3f}]")

    print("\n🎯 SegNN编码器集成成功!")
    print("📋 准备集成到SAC agent中...")