# 固定夹爪插入力任务 - 超参数消融实验设计

## 1. 任务概述

**目标**: 实现固定（闭合）夹爪的机器人插入力任务，通过力反馈与阻抗控制将抓持的物体插入目标孔槽。

**主要挑战**:
- 高精度对准要求
- 力控制与位置控制的平衡
- 搜索策略优化
- HIL-SERL训练效率

## 2. 实验设计框架

### 2.1 控制参数（HIROLRobotPlatform层）

#### 阻抗控制参数
```python
# FR3默认参数
impedance_params = {
    "translational_stiffness": 2000,  # N/m, 测试范围: [1000, 2000, 3000]
    "rotational_stiffness": 150,      # Nm/rad, 测试范围: [100, 150, 200]
    "translational_damping": 89,      # 测试范围: [45, 89, 130]
    "rotational_damping": 7,          # 测试范围: [3.5, 7, 10.5]
}

# 精细控制参数
precision_params = {
    "translational_clip": [-0.01, 0.01],  # m, 测试: ±50%
    "rotational_clip": [-0.025, 0.025],   # rad, 测试: ±50%
    "delta_tau_max": 1.0,                 # Nm/s, 测试: [0.5, 1.0, 2.0]
    "filter_params": 0.005,               # 测试: [0.001, 0.005, 0.01]
}
```

### 2.2 HIL-SERL训练参数

#### 核心RL参数
```python
training_params = {
    # 优先级1 - 关键参数
    "batch_size": [128, 256, 512],
    "discount": [0.95, 0.97, 0.99],
    "cta_ratio": [1, 2, 4],  # CQL保守性

    # 优先级2 - 更新策略
    "steps_per_update": [25, 50, 100],
    "training_starts": [100, 500, 1000],

    # 优先级3 - 缓冲区配置
    "replay_buffer_capacity": [100000, 200000, 500000],
    "buffer_period": [500, 1000, 2000],
}
```

#### 演示数据参数
```python
demo_params = {
    "demo_data_amount": [0, 10, 50, 100],  # 轨迹数量
    "demo_sampling_ratio": [0.1, 0.3, 0.5],  # 演示vs在线数据比例
    "demo_augmentation": [True, False],

    # 人工介入
    "intervention_threshold": [0.3, 0.5, 0.7],
    "correction_weight": [1.0, 2.0, 5.0],
}
```

### 2.3 环境配置参数

#### 位置与工作空间
```python
workspace_params = {
    # Reset位置变化
    "RESET_POSE_variations": [
        [0.53, 0.1, 0.3],  # 更近
        [0.55, 0.1, 0.3],  # 标准
        [0.57, 0.1, 0.3],  # 更远
    ],

    # 随机重置范围
    "RANDOM_XY_RANGE": [0.01, 0.02, 0.03],
    "RANDOM_RZ_RANGE": [0.0, 0.05, 0.1],

    # 工作空间限制
    "workspace_size_factor": [0.8, 1.0, 1.2],
    "safety_margin": [0.01, 0.02, 0.03],  # m
}
```

#### 动作空间缩放
```python
action_params = {
    # ACTION_SCALE消融 [position, rotation, gripper]
    "action_scales": [
        [0.01, 0.02, 0],  # 保守
        [0.02, 0.04, 0],  # 当前
        [0.03, 0.06, 0],  # 激进
    ],

    # Chunking参数
    "obs_horizon": [1, 3, 5],
    "act_exec_horizon": [None, 5, 10],
}
```

### 2.4 感知系统参数

#### 相机配置
```python
camera_params = {
    # 相机组合实验
    "camera_combinations": [
        ["wrist_1"],                   # 单相机
        ["side", "wrist_1"],           # 双相机
        ["side", "wrist_1", "front"],  # 三相机（当前）
        ["front", "wrist_1"],          # 不同双相机组合
    ],

    # 分类器相机
    "classifier_cameras": [
        ["wrist_1"],                   # 仅腕部
        ["side", "front"],             # 仅外部
        ["side", "wrist_1", "front"],  # 全部
    ],

    # 图像分辨率
    "image_resolution": [(224, 224), (256, 256), (320, 320)],
}
```

#### 编码器与特征
```python
encoder_params = {
    # 编码器类型
    "encoder_types": [
        "resnet",            # 从头训练
        "resnet-pretrained", # 预训练（当前）
        "efficientnet",      # 轻量级
    ],

    # 特征维度
    "feature_dim": [256, 512, 1024],

    # 本体感知输入组合
    "proprio_combinations": [
        ["tcp_pose", "gripper_pose"],  # 基础
        ["tcp_pose", "tcp_vel", "gripper_pose"],  # +速度
        ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"],  # 完整
    ],
}
```

### 2.5 插入任务特定参数

#### 力控制策略
```python
force_control_params = {
    "insertion_force_target": [5, 10, 15, 20, 25, 30],  # N
    "force_ramp_time": [0.5, 1.0, 2.0, 3.0],  # s
    "max_insertion_force": 50,  # N
    "force_deadzone": [1, 2, 3],  # N
    "max_torque_during_insertion": [2, 2, 2],  # Nm
}
```

#### 搜索策略
```python
search_strategy_params = {
    # 螺旋搜索
    "spiral_amplitude": [0.5, 1.0, 2.0, 3.0],  # mm
    "spiral_frequency": [0.5, 1.0, 1.5, 2.0],  # Hz
    "spiral_decay_rate": [0.8, 0.9, 1.0],

    # 随机探索
    "exploration_noise_std": [0.001, 0.002, 0.003],  # m
    "exploration_decay_factor": [0.9, 0.95, 1.0],
}
```

#### 状态检测与切换
```python
state_detection_params = {
    "contact_force_threshold": [2, 3, 5],  # N
    "jamming_torque_threshold": [1.0, 1.5, 2.0],  # Nm
    "success_depth_threshold": [5, 10, 15],  # mm

    # 超时设置
    "approach_timeout": [3, 5, 8],  # s
    "search_timeout": [5, 10, 15],  # s
    "insertion_timeout": [10, 15, 20],  # s
}
```

## 3. 实验矩阵设计

### 3.1 优先级1：基线实验（第1周）

| 实验组 | 参数 | 值 | 重复次数 | 预期输出 |
|--------|------|-----|----------|----------|
| A1 | 默认配置 | - | 5 | 基线性能 |
| A2 | batch_size | 128, 256, 512 | 3 | 训练稳定性 |
| A3 | discount | 0.95, 0.97, 0.99 | 3 | 任务视野影响 |
| A4 | 相机组合 | 1, 2, 3相机 | 3 | 感知能力 |
| A5 | action_scale | ±50% | 3 | 控制精度 |

### 3.2 优先级2：筛选实验（第2周）

使用Plackett-Burman设计，测试12个参数，每个2水平：

| 参数 | 低水平 | 高水平 |
|------|--------|--------|
| cta_ratio | 1 | 4 |
| demo_amount | 10 | 100 |
| insertion_force | 10N | 20N |
| spiral_amplitude | 0.5mm | 2mm |
| random_xy_range | 0.01 | 0.03 |
| obs_horizon | 1 | 5 |
| proprio_keys | 基础 | 完整 |
| encoder_type | resnet | pretrained |
| intervention_threshold | 0.3 | 0.7 |
| force_ramp_time | 0.5s | 2s |
| workspace_factor | 0.8 | 1.2 |
| buffer_capacity | 100k | 500k |

### 3.3 优先级3：精细调优（第3-4周）

基于筛选结果，对top-5参数进行完全因子实验或贝叶斯优化。

## 4. 评估指标体系

### 4.1 性能指标
```python
performance_metrics = {
    "success_rate": "最终100次评估的成功率",
    "sample_efficiency": "达到80%成功率所需episodes",
    "convergence_speed": "Q-loss下降到稳定值的步数",
    "final_insertion_depth": "平均插入深度(mm)",
    "task_completion_time": "平均完成时间(s)",
}
```

### 4.2 鲁棒性指标
```python
robustness_metrics = {
    "position_tolerance": "不同初始位置的成功率方差",
    "angle_tolerance": "不同初始角度的成功率方差",
    "force_consistency": "插入力的标准差(N)",
    "recovery_ability": "从卡死状态恢复的成功率",
}
```

### 4.3 安全性指标
```python
safety_metrics = {
    "max_force_applied": "最大接触力(N)",
    "max_torque_applied": "最大扭矩(Nm)",
    "collision_count": "非预期碰撞次数",
    "force_overshoot": "力超调百分比",
}
```

### 4.4 计算效率指标
```python
efficiency_metrics = {
    "training_fps": "learner训练帧率",
    "inference_fps": "actor推理帧率",
    "gpu_memory": "显存占用(GB)",
    "cpu_utilization": "CPU使用率(%)",
}
```

## 5. 实验执行计划

### 5.1 环境准备
```bash
# 1. 设置实验目录结构
experiments/
├── configs/          # 参数配置文件
├── results/          # 实验结果
├── logs/            # 训练日志
├── checkpoints/     # 模型检查点
└── analysis/        # 分析脚本

# 2. 配置自动化脚本
python scripts/generate_configs.py --experiment baseline
python scripts/run_ablation.py --config configs/baseline.yaml
```

### 5.2 数据收集模板
```python
experiment_record = {
    "experiment_id": "exp_001",
    "timestamp": "2024-01-15_10:30:00",
    "config": {...},  # 完整配置
    "metrics": {
        "performance": {...},
        "robustness": {...},
        "safety": {...},
        "efficiency": {...},
    },
    "training_curve": [...],  # 训练曲线数据
    "notes": "观察到的特殊现象",
}
```

### 5.3 分析与可视化
```python
# 关键分析
- 参数敏感性分析（主效应图）
- 参数交互作用分析（交互图）
- 性能-效率帕累托前沿
- 收敛曲线对比
- 力/位轨迹可视化
```

## 6. 特别注意事项

### 6.1 实验控制
- **随机种子**: 每组实验至少3个种子，报告mean±std
- **早停机制**: 100 episodes无改善则停止
- **检查点保存**: 每1000步保存，保留best-5
- **资源管理**: GPU显存预分配设置

### 6.2 固定夹爪特殊考虑
- 夹爪始终保持闭合状态
- 无需学习夹爪控制
- 重点优化力控和搜索策略
- 考虑夹爪预紧力的影响

### 6.3 安全措施
- 设置力/力矩硬限制
- 实时监控异常值
- 紧急停止机制
- 定期检查硬件状态

## 7. 预期成果

### 7.1 主要产出
1. **最优参数组合**: 针对插入任务的最佳超参数配置
2. **参数重要性排序**: 识别关键影响因素
3. **性能基准**: 建立任务性能baseline
4. **设计指南**: 形成参数选择的最佳实践

### 7.2 成功标准
- 插入成功率 > 95%
- 平均完成时间 < 10s
- 最大接触力 < 30N
- 样本效率提升 > 50%

## 8. 时间线

| 周次 | 任务 | 交付物 |
|------|------|--------|
| 第1周 | 基线实验 | 基线性能报告 |
| 第2周 | 筛选实验 | 参数重要性分析 |
| 第3周 | 精细调优 | 最优参数组合 |
| 第4周 | 验证与总结 | 最终实验报告 |

## 附录A：自动化工具

```python
class AblationExperimentRunner:
    """自动化消融实验执行器"""

    def __init__(self, base_config_path):
        self.base_config = self.load_config(base_config_path)
        self.results = []

    def run_single_ablation(self, param_name, value, seeds=[0, 1, 2]):
        """运行单个参数的消融实验"""
        results = []
        for seed in seeds:
            config = self.modify_config(param_name, value)
            metrics = self.train_and_evaluate(config, seed)
            results.append(metrics)
        return self.aggregate_results(results)

    def run_grid_search(self, param_grid):
        """网格搜索多个参数"""
        from itertools import product

        keys = param_grid.keys()
        values = param_grid.values()

        for combination in product(*values):
            config_update = dict(zip(keys, combination))
            self.run_experiment(config_update)

    def run_bayesian_optimization(self, param_space, n_calls=50):
        """贝叶斯优化超参数"""
        from skopt import gp_minimize

        def objective(params):
            config = self.params_to_config(params)
            metrics = self.train_and_evaluate(config)
            return -metrics['success_rate']  # 最大化成功率

        result = gp_minimize(objective, param_space, n_calls=n_calls)
        return result
```

## 附录B：配置文件示例

```yaml
# config_ablation_example.yaml
experiment:
  name: "fixed_gripper_insertion_ablation"
  type: "grid_search"

base_config:
  # Training
  batch_size: 256
  discount: 0.97
  cta_ratio: 2

  # Environment
  reset_pose: [0.55, 0.1, 0.3, -3.14159, 0, 0]
  action_scale: [0.02, 0.04, 0]

  # Cameras
  image_keys: ["side", "wrist_1", "front"]

  # Force control
  insertion_force_target: 15

ablation_params:
  batch_size: [128, 256, 512]
  insertion_force_target: [10, 15, 20]
  action_scale:
    - [0.01, 0.02, 0]
    - [0.02, 0.04, 0]
    - [0.03, 0.06, 0]
```