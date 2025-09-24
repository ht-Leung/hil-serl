# WSRL-Style Offline Warmup for HIL-SERL

## 概述

本文档描述如何在HIL-SERL框架中实现类似WSRL (Efficient Online Reinforcement Learning Fine-Tuning Need Not Retain Offline Data)的离线预训练和在线微调机制。

## 背景

### HIL-SERL现有机制

1. **数据管理**
   - 使用`MemoryEfficientReplayBuffer`存储轨迹数据
   - 支持从`.pkl`文件加载演示数据
   - 实现了50/50的RLPD采样策略（50%演示，50%在线）

2. **训练流程**
   - Actor-Learner分布式架构
   - 通过`demo_path`加载离线演示数据
   - 在线训练时混合采样演示和在线数据

3. **Action表示**
   - 使用单步action（非action chunk）
   - `ChunkingWrapper`配置为`act_exec_horizon=None`

### WSRL核心思想

1. **离线预训练阶段**
   - 纯离线数据训练，快速获得初始策略
   - 使用加权更新，高质量数据权重更高
   - 逐渐降低离线数据比例，平滑过渡到在线

2. **在线微调阶段**
   - 不保留全部离线数据，只保留高质量子集
   - 使用优先级采样和加权更新
   - 保持计算和内存效率

## 实现方案

### 1. 离线预训练模块

#### 1.1 数据加载和预处理

```python
# 加载多个离线数据源
offline_sources = {
    "demos": {"path": demo_files, "weight": 2.0},
    "suboptimal": {"path": suboptimal_files, "weight": 1.0},
    "random": {"path": random_files, "weight": 0.5}
}

# 基于数据质量分配权重
def compute_data_quality(trajectory):
    # 考虑：成功率、奖励、轨迹长度等
    return quality_score
```

#### 1.2 加权策略更新

```python
# 修改SAC更新，支持加权损失
def weighted_update(batch, weights):
    # Actor loss with importance weights
    actor_loss = -jnp.mean(actor_objective * weights)

    # Critic loss with importance weights
    critic_loss = jnp.mean((q_pred - q_target)**2 * weights)

    return actor_loss, critic_loss
```

#### 1.3 渐进式过渡策略

```python
# 离线比例衰减
offline_ratio = initial_ratio * (decay_rate ** step)
offline_ratio = max(min_ratio, offline_ratio)

# 动态调整采样比例
demo_batch_size = int(batch_size * offline_ratio)
online_batch_size = batch_size - demo_batch_size
```

### 2. 在线微调优化

#### 2.1 选择性数据保留

```python
# 只保留高价值离线数据
def filter_offline_data(buffer, retention_ratio=0.2):
    # 基于TD误差、奖励等指标筛选
    priorities = compute_priorities(buffer)
    top_k = int(len(buffer) * retention_ratio)
    return buffer.select_top_k(priorities, top_k)
```

#### 2.2 优先级经验回放

```python
# 实现PER (Prioritized Experience Replay)
class PrioritizedReplayBuffer:
    def sample(self, batch_size, alpha=0.6, beta=0.4):
        # 基于优先级采样
        priorities = self.priorities ** alpha
        probs = priorities / priorities.sum()

        # 重要性采样权重
        weights = (N * probs[indices]) ** (-beta)
        weights /= weights.max()

        return batch, weights
```

### 3. 集成流程

#### 3.1 两阶段训练脚本

```bash
# 阶段1：离线预训练
python train_offline_warmup.py \
    --exp_name=hirol_fixed_gripper \
    --offline_data_path=demos.pkl \
    --warmup_steps=50000 \
    --checkpoint_path=warmup_ckpt

# 阶段2：在线微调
python train_rlpd.py \
    --exp_name=hirol_fixed_gripper \
    --checkpoint_path=warmup_ckpt \
    --demo_path=filtered_demos.pkl \
    --learner
```

#### 3.2 配置参数

```python
class WSRLConfig:
    # 离线预训练
    warmup_steps = 50000
    initial_offline_ratio = 1.0
    offline_decay_rate = 0.999
    min_offline_ratio = 0.5

    # 数据质量权重
    demo_weight = 2.0
    success_weight_bonus = 1.5

    # 在线微调
    retention_ratio = 0.2
    use_prioritized_replay = True
    priority_alpha = 0.6
    priority_beta_schedule = lambda t: 0.4 + 0.6 * t
```

## 实现步骤

### 第1步：创建离线预训练脚本
- [x] 分析现有数据加载机制
- [ ] 实现`train_offline_warmup.py`
- [ ] 添加加权更新支持到SAC agent

### 第2步：修改数据管理
- [ ] 实现`PrioritizedReplayBuffer`
- [ ] 添加数据质量评估函数
- [ ] 实现选择性数据保留

### 第3步：优化在线训练
- [ ] 修改`train_rlpd.py`支持warmup checkpoint
- [ ] 实现渐进式采样比例调整
- [ ] 添加优先级采样

### 第4步：实验验证
- [ ] 对比实验：with/without warmup
- [ ] 调参：offline_ratio, decay_rate, retention_ratio
- [ ] 性能指标：样本效率、最终性能、训练稳定性

## 预期效果

1. **样本效率提升**
   - 减少达到目标性能所需的在线交互次数
   - 从第一步就有合理的初始策略

2. **训练稳定性**
   - 避免早期探索的危险动作
   - 平滑的离线到在线过渡

3. **内存效率**
   - 不需要保留全部离线数据
   - 只保留高价值数据子集

## 潜在挑战和解决方案

### 挑战1：分布偏移
- **问题**：离线和在线数据分布差异
- **方案**：渐进式过渡 + 保守Q学习（CQL）正则化

### 挑战2：过拟合离线数据
- **问题**：策略过度拟合演示
- **方案**：早停 + 数据增强 + dropout

### 挑战3：计算开销
- **问题**：优先级采样增加计算
- **方案**：批量更新优先级 + 近似采样

## 参考资源

- WSRL论文: "Efficient Online Reinforcement Learning Fine-Tuning Need Not Retain Offline Data"
- HIL-SERL: https://github.com/rail-berkeley/serl
- RLPD: "Efficient Online Reinforcement Learning with Offline Data"
- PER: "Prioritized Experience Replay"

## 下一步行动

1. 实现基础版本的offline warmup脚本
2. 在简单任务上验证效果
3. 逐步添加高级特性（PER、CQL等）
4. 大规模实验和调参