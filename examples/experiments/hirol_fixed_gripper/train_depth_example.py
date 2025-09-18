#!/usr/bin/env python3
"""
HIL-SERL RGB+Depth训练示例
演示如何启用深度处理的训练流程
"""

import sys
import os
sys.path.append('/home/hanyu/code/hil-serl/examples')

import jax
from serl_launcher.utils.launcher import make_sac_pixel_agent
from experiments.hirol_fixed_gripper.config import TrainConfig

def main():
    # 创建配置
    config = TrainConfig()

    # 启用深度处理 (默认是False)
    config.use_depth = True

    print("🚀 启动HIL-SERL RGB+Depth训练")
    print(f"✓ 使用深度: {config.use_depth}")
    print(f"✓ 相机列表: {config.image_keys}")
    print(f"✓ 深度编码器参数: {config.depth_encoder_kwargs}")
    print(f"✓ 相机内参: {list(config.camera_params.keys())}")

    # 创建环境
    env = config.get_environment(fake_env=False, save_video=False, classifier=True)

    # 获取观测样本
    sample_obs, _ = env.reset()
    sample_action = env.action_space.sample()

    print(f"✓ 观测键: {list(sample_obs.keys())}")
    print(f"✓ 图像形状: {[sample_obs[key].shape for key in config.image_keys]}")

    # 检查是否有深度数据
    has_depth = any(f"depth_{key}" in sample_obs for key in config.image_keys)
    print(f"✓ 检测到深度数据: {has_depth}")

    # 创建智能agent (自动检测RGB+Depth)
    agent = make_sac_pixel_agent(
        seed=42,
        sample_obs=sample_obs,
        sample_action=sample_action,
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
        # 深度相关参数
        use_depth=config.use_depth,
        depth_encoder_kwargs=config.depth_encoder_kwargs,
        camera_params=config.camera_params,
        # 训练参数
        discount=config.discount,
    )

    print("✓ Agent创建成功!")
    print(f"✓ Agent类型: {type(agent).__name__}")

    # 测试推理
    key = jax.random.PRNGKey(0)
    action = agent.sample_actions(sample_obs, seed=key, argmax=False)
    print(f"✓ 推理测试成功，动作形状: {action.shape}")

    print("\n🎯 RGB+Depth训练环境配置完成!")
    print("💡 提示:")
    print("   - 设置 config.use_depth = True 启用深度")
    print("   - 设置相机 depth=True 在EnvConfig.REALSENSE_CAMERAS中")
    print("   - 使用真实相机内参提高点云质量")
    print("   - SegNN编码器将自动处理点云特征")

if __name__ == "__main__":
    main()