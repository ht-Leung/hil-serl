# HIROLReach 任务实现文档

## 概述

HIROLReach 是一个基于 HIROL 机器人平台的到达任务（reach task）实现，用于测试和验证 HIROL-SERL 集成框架的功能。该任务要求机器人末端执行器到达指定的三维空间位置和姿态。

## 目录结构

```
examples/experiments/hirol_reach/
├── __init__.py          # Python 包初始化文件
├── config.py            # 环境和训练配置
├── wrapper.py           # HIROLReachEnv 环境实现
├── run_learner.sh       # 学习器启动脚本
├── run_actor.sh         # 执行器启动脚本
└── test_env.py          # 环境测试脚本
```

## 主要组件

### 1. 环境配置 (config.py)

#### EnvConfig 类
- 继承自 `DefaultEnvConfig`
- 定义任务相关参数：
  - `TARGET_POSE`: 目标位置和姿态 [x, y, z, roll, pitch, yaw]
  - `RESET_POSE`: 重置位置
  - `REWARD_THRESHOLD`: 成功判定阈值（位置 2cm，姿态 0.1 rad）
  - `ACTION_SCALE`: 动作缩放因子
  - 相机配置、工作空间限制、柔顺控制参数等

#### TrainConfig 类
- 继承自 `DefaultTrainingConfig`
- 定义训练相关参数：
  - 图像和本体感知键值
  - 训练超参数（折扣因子、缓冲区更新周期等）
  - `get_environment()` 方法创建并配置环境

### 2. 环境实现 (wrapper.py)

#### HIROLReachEnv 类
- 继承自 `HIROLEnv`
- 核心方法：
  - `compute_reward()`: 计算二元奖励（到达目标返回 1，否则返回 0）
  - `reset()`: 重置环境，支持目标位置随机化
  - `step()`: 执行动作并返回观察、奖励等信息
  - `go_to_reset()`: 自定义重置过程

### 3. 运行脚本

- `run_learner.sh`: 启动 RLPD 学习器
- `run_actor.sh`: 启动策略执行器

## 使用流程

### 1. 录制演示数据
```bash
# 设置PYTHONPATH
export PYTHONPATH=/home/hanyu/code/hil-serl:$PYTHONPATH

# 录制成功的演示轨迹
python examples/record_demos.py --exp_name hirol_reach --successes_needed 50
```

### 2. 录制成功/失败数据（用于奖励分类器）
```bash
python examples/record_success_fail.py --exp_name hirol_reach --successes_needed 2000
```

### 3. 训练奖励分类器
```bash
python examples/train_reward_classifier.py --exp_name hirol_reach
```

### 4. 启动 RLPD 训练

在两个终端中分别运行：

**终端 1 - 学习器：**
```bash
cd examples/experiments/hirol_reach
bash run_learner.sh
```

**终端 2 - 执行器：**
```bash
cd examples/experiments/hirol_reach
bash run_actor.sh
```

## 任务特性

- **动作空间**: 7 维连续动作（3D 位置增量 + 3D 旋转增量 + 夹爪动作）
- **观察空间**: 
  - 状态：TCP 位姿、速度、力/力矩、夹爪位置
  - 图像：手腕相机和侧面相机（128x128x3）
- **奖励函数**: 二元奖励，基于到目标的距离
- **成功条件**: 位置误差 < 2cm 且姿态误差 < 0.1 rad

## 扩展建议

1. **增加任务复杂度**：
   - 添加障碍物避让
   - 实现动态目标跟踪
   - 增加路径规划约束

2. **改进奖励函数**：
   - 使用连续奖励代替二元奖励
   - 添加速度平滑性奖励
   - 考虑能量效率

3. **增强鲁棒性**：
   - 添加干扰和噪声
   - 实现自适应控制参数
   - 增加安全约束

## 注意事项

1. **环境依赖**：
   - 需要安装 HIROL 机器人平台相关依赖
   - 当前存在 NumPy 版本兼容性问题，建议使用 `pip install numpy<2` 降级到 NumPy 1.x
   - 确保已安装 pinocchio 等机器人学相关库

2. **运行前准备**：
   - 设置正确的 PYTHONPATH：`export PYTHONPATH=/home/hanyu/code/hil-serl:$PYTHONPATH`
   - 确保 HIROL 机器人平台正确初始化
   - 检查相机序列号与实际硬件匹配

3. **参数说明**：
   - `record_demos.py` 使用 `--successes_needed` 参数，而非 `--num_demos`
   - 根据实际工作空间调整位置限制和目标位置
   - 在真实机器人上运行前先在仿真环境中测试（使用 `fake_env=True`）

4. **文件结构**：
   - Wrappers 位于 `/serl_hirol_infra/hirol_env/envs/wrappers.py`
   - 环境配置使用 `DefaultEnvConfig` 作为基类
   - 任务映射已添加到 `/examples/experiments/mappings.py`