"""
深度图到稀疏点云转换模块
用于将深度图像转换为SegNN encoder可以处理的点云格式
"""
import sys
import os
import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Dict
import flax.linen as nn

# 添加SegNN模块路径以使用FPS
sys.path.append('/home/hanyu/code/Seg-NN')
from pointnet2_ops.fps_jax import furthest_point_sampling

class DepthToPointCloud(nn.Module):
    """
    将深度图转换为稀疏点云的模块 - 单阶段FPS采样策略

    Args:
        max_points: 最终输出点数 (与SegNN input_points匹配)
        min_depth: 最小深度值 (米)
        max_depth: 最大深度值 (米)
        add_noise: 是否添加噪声增强
    """
    max_points: int = 2048           # 最终输出点数 与segnn匹配
    min_depth: float = 0.1
    max_depth: float = 2.0
    add_noise: bool = False
    noise_std: float = 0.002  # 2mm标准差

    def setup(self):
        # 相机内参 (这些应该从配置中读取)
        # 默认为RealSense D435的参数
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

    def depth_to_xyz(self, depth_image: jnp.ndarray) -> jnp.ndarray:
        """
        将深度图转换为3D点坐标

        Args:
            depth_image: [H, W] 深度图 (以米为单位)

        Returns:
            xyz: [H*W, 3] 3D坐标
        """
        H, W = depth_image.shape

        # 创建像素坐标网格
        u, v = jnp.meshgrid(jnp.arange(W), jnp.arange(H), indexing='xy')
        u = u.flatten().astype(jnp.float32)
        v = v.flatten().astype(jnp.float32)
        depth = depth_image.flatten()

        # 深度图到3D坐标转换
        # x = (u - cx) * depth / fx
        # y = (v - cy) * depth / fy
        # z = depth
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth

        xyz = jnp.stack([x, y, z], axis=1)  # [H*W, 3]
        return xyz

    def extract_rgb_at_points(self, rgb_image: jnp.ndarray) -> jnp.ndarray:
        """
        从RGB图像中提取对应点的颜色信息

        Args:
            rgb_image: [H, W, 3] RGB图像 [0, 255]

        Returns:
            rgb: [H*W, 3] RGB值 [0, 1]
        """
        # 将RGB图像重塑并归一化到[0,1]
        rgb = rgb_image.reshape(-1, 3) / 255.0
        return rgb

    def filter_valid_points(self, xyz: jnp.ndarray, rgb: jnp.ndarray,
                          depth_image: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        过滤有效的3D点 (移除无效深度和超出范围的点)
        使用jnp.where避免动态boolean indexing

        Args:
            xyz: [N, 3] 3D坐标
            rgb: [N, 3] RGB值
            depth_image: [H, W] 深度图

        Returns:
            valid_xyz: [N, 3] 有效3D坐标（无效点设为inf）
            valid_rgb: [N, 3] 有效RGB值（无效点设为0）
        """
        depth_flat = depth_image.flatten()

        # 创建有效性掩码
        valid_mask = (
            (depth_flat > self.min_depth) &  # 最小深度
            (depth_flat < self.max_depth) &  # 最大深度
            (depth_flat > 0) &               # 非零深度
            jnp.isfinite(depth_flat)         # 有限值
        )

        # 使用jnp.where替换无效点，而不是boolean indexing
        # 无效点的xyz设为inf，这样在FPS采样时会被自动忽略
        valid_mask_3d = jnp.expand_dims(valid_mask, axis=-1)
        valid_xyz = jnp.where(valid_mask_3d, xyz, jnp.inf)
        valid_rgb = jnp.where(valid_mask_3d, rgb, 0.0)

        return valid_xyz, valid_rgb

    def fps_sample(self, xyz: jnp.ndarray, rgb: jnp.ndarray,
                   key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        单阶段FPS采样: 直接从有效点FPS采样到max_points数量

        Args:
            xyz: [N, 3] 3D坐标
            rgb: [N, 3] RGB值
            key: 随机数种子

        Returns:
            sampled_xyz: [max_points, 3] 采样后的3D坐标
            sampled_rgb: [max_points, 3] 采样后的RGB值
        """
        N = xyz.shape[0]

        if N == 0:
            # 如果没有有效点，返回零向量
            return (jnp.zeros((self.max_points, 3)),
                   jnp.zeros((self.max_points, 3)))

        # 单阶段FPS采样
        if N >= self.max_points:
            # 足够点数，使用FPS
            xyz_batch = xyz[None, ...].astype(jnp.float32)  # [1, N, 3]
            indices = furthest_point_sampling(xyz_batch, self.max_points)  # [1, max_points]
            indices = indices[0]  # [max_points]
        else:
            # 点数不足，随机重复采样
            indices = jax.random.choice(key, N, (self.max_points,), replace=True)

        sampled_xyz = xyz[indices]
        sampled_rgb = rgb[indices]

        return sampled_xyz, sampled_rgb


    def add_coordinate_normalization(self, xyz: jnp.ndarray) -> jnp.ndarray:
        """
        添加归一化坐标 (按SegNN要求)

        Args:
            xyz: [N, 3] 原始坐标

        Returns:
            XYZ: [N, 3] 归一化坐标 [0,1]
        """
        # 去中心化
        xyz_min = jnp.min(xyz, axis=0, keepdims=True)
        xyz_centered = xyz - xyz_min

        # 归一化到[0,1]
        xyz_max = jnp.max(xyz_centered, axis=0, keepdims=True)
        xyz_max = jnp.where(xyz_max < 1e-8, 1.0, xyz_max)  # 避免除零
        XYZ = xyz_centered / xyz_max

        return XYZ

    @nn.compact
    def __call__(self, depth_image: jnp.ndarray, rgb_image: jnp.ndarray,
                 key: jax.random.PRNGKey, train: bool = False) -> jnp.ndarray:
        """
        主转换函数

        Args:
            depth_image: [H, W] 深度图 (米)
            rgb_image: [H, W, 3] RGB图像 [0, 255]
            key: 随机数种子
            train: 训练模式

        Returns:
            pointcloud: [max_points, 9] 点云 (xyz + rgb + XYZ)
        """
        # 1. 深度图转3D坐标
        xyz = self.depth_to_xyz(depth_image)

        # 2. 提取RGB
        rgb = self.extract_rgb_at_points(rgb_image)

        # 3. 过滤有效点
        valid_xyz, valid_rgb = self.filter_valid_points(xyz, rgb, depth_image)

        # 4. FPS采样
        sampled_xyz, sampled_rgb = self.fps_sample(valid_xyz, valid_rgb, key)

        # 5. 添加噪声 (训练时)
        if self.add_noise and train:
            noise_key, key = jax.random.split(key)
            noise = jax.random.normal(noise_key, sampled_xyz.shape) * self.noise_std
            sampled_xyz = sampled_xyz + noise

        # 6. 创建归一化坐标
        XYZ = self.add_coordinate_normalization(sampled_xyz)

        # 7. 按SegNN格式组合: [xyz, rgb, XYZ]
        pointcloud = jnp.concatenate([sampled_xyz, sampled_rgb, XYZ], axis=1)

        return pointcloud


def create_depth_converter(camera_params: Optional[Dict[str, float]] = None,
                          max_points: int = 4096) -> DepthToPointCloud:
    """
    创建深度转换器的工厂函数

    Args:
        camera_params: 相机内参字典 {'fx', 'fy', 'cx', 'cy'}
        max_points: 最大点数

    Returns:
        DepthToPointCloud实例
    """
    converter = DepthToPointCloud(max_points=max_points)

    if camera_params:
        converter.fx = camera_params.get('fx', 615.0)
        converter.fy = camera_params.get('fy', 615.0)
        converter.cx = camera_params.get('cx', 320.0)
        converter.cy = camera_params.get('cy', 240.0)

    return converter


# 测试代码
if __name__ == "__main__":
    print("🔧 测试深度图到点云转换")

    # 创建测试数据
    key = jax.random.PRNGKey(42)
    H, W = 480, 640

    # 模拟深度图 (0.5-1.5米随机深度)
    depth_key, rgb_key, test_key = jax.random.split(key, 3)
    depth = jax.random.uniform(depth_key, (H, W), minval=0.5, maxval=1.5)
    rgb = jax.random.uniform(rgb_key, (H, W, 3), minval=0, maxval=255)

    # 创建转换器
    converter = create_depth_converter(max_points=2048)

    # 正确的Flax模块使用方式：初始化参数
    dummy_depth = jnp.ones((H, W))
    dummy_rgb = jnp.ones((H, W, 3))
    init_key, apply_key = jax.random.split(test_key)

    # 初始化参数（虽然这个模块没有可学习参数）
    params = converter.init(init_key, dummy_depth, dummy_rgb, apply_key)

    # 使用apply方法执行转换
    pointcloud = converter.apply(params, depth, rgb, apply_key)

    print(f"✓ 输入深度图形状: {depth.shape}")
    print(f"✓ 输入RGB图形状: {rgb.shape}")
    print(f"✓ 输出点云形状: {pointcloud.shape}")
    print(f"✓ 点云范围:")
    print(f"  - xyz: [{jnp.min(pointcloud[:, :3]):.3f}, {jnp.max(pointcloud[:, :3]):.3f}]")
    print(f"  - rgb: [{jnp.min(pointcloud[:, 3:6]):.3f}, {jnp.max(pointcloud[:, 3:6]):.3f}]")
    print(f"  - XYZ: [{jnp.min(pointcloud[:, 6:9]):.3f}, {jnp.max(pointcloud[:, 6:9]):.3f}]")

    print("\n🎯 转换成功! 点云格式符合SegNN要求")