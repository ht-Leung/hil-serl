"""分析微调ResNet vs 冻结ResNet的利弊"""
import jax
import jax.numpy as jnp
import numpy as np
from serl_launcher.networks.reward_classifier import create_classifier

print("=" * 70)
print("分析：微调 ResNet vs 冻结 ResNet")
print("=" * 70)

print("\n当前状况（冻结模式）:")
print("-" * 50)

# 创建分类器查看参数数量
key = jax.random.PRNGKey(42)
image_keys = ['wrist_1', 'front']
sample = {k: jnp.zeros((128, 128, 3)) for k in image_keys}
classifier = create_classifier(key, sample, image_keys)

# 统计参数
total_params = sum(x.size for x in jax.tree.leaves(classifier.params))
trainable_params = 0
frozen_params = 0

# 计算可训练参数（非ResNet部分）
for encoder_name in ['encoder_wrist_1', 'encoder_front']:
    if encoder_name in classifier.params["encoder_def"]:
        encoder_params = classifier.params["encoder_def"][encoder_name]
        if "pretrained_encoder" in encoder_params:
            # ResNet参数（冻结）
            resnet_params = sum(x.size for x in jax.tree.leaves(encoder_params["pretrained_encoder"]))
            frozen_params += resnet_params
        
        # 后处理层参数（可训练）
        for layer in ["SpatialLearnedEmbeddings_0", "Dense_0", "LayerNorm_0"]:
            if layer in encoder_params:
                layer_params = sum(x.size for x in jax.tree.leaves(encoder_params[layer]))
                trainable_params += layer_params

# 分类头参数
classifier_head_params = total_params - frozen_params - trainable_params

print(f"总参数: {total_params:,}")
print(f"冻结参数 (ResNet): {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
print(f"可训练参数: {trainable_params + classifier_head_params:,} ({(trainable_params + classifier_head_params)/total_params*100:.1f}%)")

print("\n" + "=" * 70)
print("对比分析：冻结 vs 微调")
print("=" * 70)

print("\n1. 数据需求:")
print("-" * 50)
print("""
冻结 ResNet（当前）:
- 需要数据量: 少 (几百到几千个样本)
- 原因: 只学习顶层分类器
- 适用: 数据有限的场景

微调 ResNet:
- 需要数据量: 多 (几千到几万个样本)
- 原因: 需要更新整个网络
- 风险: 数据不足时容易过拟合
""")

print("2. 训练速度:")
print("-" * 50)
print(f"""
冻结 ResNet（当前）:
- 可训练参数: {(trainable_params + classifier_head_params):,}
- 训练速度: 快
- 收敛: 通常 50-150 epochs

微调 ResNet:
- 可训练参数: {total_params:,} (增加 {frozen_params:,} 参数)
- 训练速度: 慢 ~{total_params/(trainable_params + classifier_head_params):.1f}x
- 收敛: 需要更多 epochs
""")

print("3. 特征质量:")
print("-" * 50)
print("""
冻结 ResNet（当前）:
- 使用 ImageNet 通用特征
- 优点: 鲁棒、泛化好
- 缺点: 可能不完全适合机器人任务

微调 ResNet:
- 学习任务特定特征
- 优点: 特征更贴合具体任务
- 缺点: 可能过拟合到训练数据
""")

print("4. 具体到机器人抓取任务:")
print("-" * 50)
print("""
考虑因素:
- ImageNet 包含大量日常物体，与抓取场景相关 ✓
- 主要判断：物体位置、抓取成功/失败
- 视觉特征：形状、边缘、纹理（ImageNet已覆盖）

当前数据量分析:
- 成功样本: ~200
- 失败样本: ~1000
- 总计: ~1200 样本

结论: 数据量较少，不足以支撑微调
""")

print("\n" + "=" * 70)
print("建议:")
print("-" * 50)
print("""
【保持当前冻结模式】的理由:

1. 数据量有限 (1200 样本)
   - 微调可能导致严重过拟合
   - 冻结模式更稳定

2. 任务相对简单
   - 二分类（成功/失败）
   - 不需要学习全新的视觉概念

3. 训练效率
   - 当前 2 epochs 已达 88% 准确率
   - 微调需要更长训练时间

【何时考虑微调】:

1. 数据量充足 (>5000 样本)
2. 任务与 ImageNet 差异大
   - 特殊光照条件
   - 非日常物体
   - 特殊视角

3. 当前性能不满足要求
   - 准确率 <85% 且无法提升
   - 特定失败模式无法解决

【折中方案】:

如果想要部分微调，可以：
1. 只微调最后几层 ResNet blocks
2. 使用更小的学习率 (1e-5)
3. 早停（early stopping）防止过拟合
""")

print("=" * 70)