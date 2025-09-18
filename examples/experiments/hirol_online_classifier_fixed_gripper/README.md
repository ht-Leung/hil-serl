# HIROL Online Classifier Fixed Gripper Task

带有人类反馈分类器系统的固定夹爪操作任务框架，基于HIROLRobotPlatform和HIROLEnv。

## 概述

本实验在`hirol_fixed_gripper`任务基础上增加了人类反馈机制，用于提升分类器准确率并减少训练过程中的假阳性问题。

### 主要特性
- 使用HIROLRobotPlatform的SerlRobotInterface进行直接机器人控制
- 集成HIL-SERL训练管道
- 支持多种机器人平台（FR3、Unitree G1、Monte01、AgiBot G1）
- 5阶多项式插值的平滑轨迹控制
- **人类反馈分类器系统用于减少假阳性**
- 基于置信度的智能查询策略
- 自动数据收集和重训练机制

## 人类反馈分类器系统

### 1. 实时人类反馈
训练过程中，您可以使用键盘快捷键纠正分类器预测：
- **`s`**：确认成功（真阳性）- 在每次step后执行
- **`f`**：标记假阳性（分类器错误，实际失败）
- **`n`**：标记假阴性（分类器漏判成功）
- **`c`**：跳过/继续（不提供反馈）
- **`r`**：强制触发分类器重训练（离线模式）
- **`p`**：暂停/恢复反馈收集（在线模式）

### 2. 基于置信度的查询策略
系统在分类器不确定时自动请求人类反馈：
- 高置信度（>85%）：接受预测
- 不确定（65-85%）：查询人类
- 低置信度（<65%）：可能错误

### 3. 自动数据收集
所有人类反馈自动保存到`feedback_data/`目录用于重训练。

## 快速开始

### 1. 环境设置

```bash
# HIROLRobotPlatform环境
source /home/hanyu/code/HIROLRobotPlatform/dependencies/a2d_sdk/env.sh

# Python路径
export PYTHONPATH=$PYTHONPATH:/home/hanyu/code/hil-serl
export PYTHONPATH=$PYTHONPATH:/home/hanyu/code/HIROLRobotPlatform
export PYTHONPATH=$PYTHONPATH:/home/hanyu/code/hil-serl/serl_hirol_infra

# 激活conda环境
conda activate hilserl
```

### 2. 初始数据准备

```bash
cd /home/hanyu/code/hil-serl/examples/experiments/hirol_online_classifier_fixed_gripper

# 复制现有分类器检查点（如果有）
cp -r ../hirol_fixed_gripper/classifier_ckpt ./classifier_ckpt

# 复制演示数据
cp -r ../hirol_fixed_gripper/demo_data ./demo_data

# 复制分类器训练数据
cp -r ../hirol_fixed_gripper/classifier_data ./classifier_data
```

### 3. 运行训练

#### 方式A：集成在线学习（推荐，无文件I/O冲突）
```bash
# 终端1 - Learner with online classifier training
python train_rlpd_online_classifier.py \
    --exp_name=hirol_online_classifier_fixed_gripper \
    --learner \
    --online_classifier \
    --classifier_retrain_interval=100

# 终端2 - Actor
python train_rlpd_online_classifier.py \
    --exp_name=hirol_online_classifier_fixed_gripper \
    --actor \
    --ip=localhost
```

#### 方式B：传统离线反馈（需手动重训练）
```bash
# 终端1 - Actor
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1
./run_actor.sh

# 终端2 - Learner
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3
./run_learner.sh
```

### 4. 提供人类反馈

观察终端输出并在提示时提供反馈：
```
[Classifier] Prediction: SUCCESS (confidence: 72%)
[Query] Uncertain prediction - please provide feedback!
```

根据机器人的实际表现按下相应按键。

### 5. 使用反馈重训练分类器

收集足够反馈后（通常100+样本）：

```bash
# 基础重训练
bash run_retrain.sh

# 自定义参数
bash run_retrain.sh --epochs 200 --weight 3.0 --incremental

# 或直接使用Python
python retrain_classifier.py \
    --exp_name=hirol_online_classifier_fixed_gripper \
    --num_epochs=150 \
    --feedback_weight=2.0
```

## 配置说明

### 选择反馈模式

编辑`config.py`中的`get_environment`方法调用：

```python
# 选择反馈模式
feedback_mode="online"   # 在线学习，无文件冲突（推荐）
feedback_mode="offline"  # 离线反馈，需手动重训练
feedback_mode="simple"   # 仅键盘's'键标记成功
feedback_mode="keyboard" # 原始键盘奖励模式
```

### 修改任务参数

编辑`config.py`修改其他参数：

```python
# 机器人配置（使用HIROLRobotPlatform配置文件）
ROBOT_CONFIG_PATH = None  # 使用默认serl_fr3_config.yaml

# 任务定义
RESET_POSE = np.array([0.5, 0.1, 0.45, -np.pi, 0, 0])  # 重置位置
REWARD_THRESHOLD = np.array([0.02, 0.02, 0.02, 0.1, 0.1, 0.1])  # 成功阈值

# 控制参数
ACTION_SCALE = np.array([0.02, 0.06, 0])  # 位置、旋转、夹爪缩放

# 人类反馈分类器参数
env = HumanFeedbackClassifierWrapper(
    env,
    classifier_func=reward_func,
    confidence_threshold=0.85,      # 高置信度阈值
    query_threshold=0.65,           # 查询阈值
    feedback_buffer_size=1000,      # 反馈缓冲区大小
    auto_retrain_interval=100,      # 自动重训练间隔
    save_feedback_path="./feedback_data"
)
```

## 系统架构

```
HIROLRobotPlatform
    ↓
SerlRobotInterface（统一接口）
    ↓
HIROLEnv（Gym环境）
    ↓
HumanFeedbackClassifierWrapper（人类反馈层）
    ↓
环境包装器（Quat2Euler, SERLObs, Chunking等）
    ↓
HIL-SERL训练管道
```

### 数据流程
```
机器人动作 → 观察 → 分类器预测 → 人类反馈
                      ↓              ↓
                 置信度检查      纠正数据
                      ↓              ↓
                不确定时询问    保存到缓冲区
                                     ↓
                                定期重训练
```

## 工作原理

### 渐进式学习策略
1. **实时纠正层**：训练时人类可立即纠正分类器判断
2. **数据积累层**：自动保存所有人类反馈用于后续训练

### 置信度驱动的查询
```python
# 只在不确定时询问人类
if 0.65 <= confidence <= 0.85:  # 不确定区间
    query_human_feedback()
```

### 加权重训练机制
- 人类反馈样本权重 = 2.0×原始数据权重
- 假阳性纠正样本会被重点学习
- 保留原始数据防止灾难性遗忘

### 自动化流程
```
收集反馈 → 达到阈值 → 保存数据 → 触发重训练 → 更新模型 → 继续收集
```

## 反馈数据格式

每个反馈样本包含：
```python
{
    'observation': {...},           # 完整观察字典
    'classifier_prediction': 0/1,   # 分类器预测
    'confidence': 0.0-1.0,          # 分类器置信度
    'true_label': 0/1,              # 人类提供的标签
    'timestamp': "2024-01-01T12:00:00"
}
```

## 性能指标

系统跟踪以下指标：
- **真阳性（TP）**：正确的成功预测
- **假阳性（FP）**：错误的成功预测
- **假阴性（FN）**：漏判的成功
- **真阴性（TN）**：正确的失败预测
- **查询率**：不确定预测的百分比
- **反馈准确率**：人类纠正率

## 高级功能

### 5阶多项式轨迹
环境使用5阶多项式插值实现平滑运动：
- 初始和最终速度/加速度为零
- 平滑的加速度曲线
- 可配置的控制频率

### 力/力矩集成
支持多种力感知模式：
- FR3内部力估计（O_F_ext_hat_K）
- 外部ATI力传感器
- 感知模式间的自动切换

### 顺应模式
三种顺应参数集用于不同阶段：
- **COMPLIANCE_PARAM**：正常操作
- **PRECISION_PARAM**：精确移动
- **RESET_PARAM**：过渡移动

## 多机器人支持

修改YAML配置以支持不同机器人：

### Unitree G1:
```yaml
robot: "unitree_g1"
robot_config:
  unitree_g1:
    # Unitree G1 specific config
```

### Monte01:
```yaml
robot: "monte01"
robot_config:
  monte01:
    # Monte01 specific config
```

## 有效反馈技巧

1. **关注边界情况**：特别注意接近成功/失败边界的情况
2. **保持一致性**：全程使用相同的判断标准
3. **快速响应**：收到查询时0.5秒内提供反馈
4. **监控统计**：查看终端显示的假阳性/假阴性率

## 高级配置

### 调整灵敏度
在`config.py`中修改阈值：
```python
confidence_threshold=0.85  # 更低=更宽松
query_threshold=0.65       # 更高=更频繁查询
```

### 加权学习
增加反馈重要性：
```bash
python retrain_classifier.py --feedback_weight=3.0  # 默认2.0
```

### 在线学习（实验性）
使用`AdaptiveClassifierWrapper`实现实时更新，无需完整重训练。

## 故障排除

### 机器人连接问题
```bash
# 检查FR3连接
ping 172.16.0.2  # 或您的机器人IP

# 验证ROS2环境（某些机器人需要）
source /home/hanyu/code/HIROLRobotPlatform/dependencies/a2d_sdk/env.sh
```

### 未找到分类器检查点
如果看到"Failed to load classifier"，先训练基础分类器：
```bash
cd ../hirol_fixed_gripper
python ../../train_reward_classifier.py --exp_name=hirol_fixed_gripper
cp -r classifier_ckpt ../hirol_online_classifier_fixed_gripper/
```

### 反馈未保存
检查`feedback_data/`目录是否存在且有写权限：
```bash
mkdir -p feedback_data
chmod 755 feedback_data
```

### 重训练失败
确保有原始训练数据和反馈数据：
```bash
ls classifier_data/  # 应有success/failure文件
ls feedback_data/    # 应有feedback_*.pkl文件
```

### 导入错误
确保所有路径已设置：
```bash
export PYTHONPATH=$PYTHONPATH:/home/hanyu/code/hil-serl
export PYTHONPATH=$PYTHONPATH:/home/hanyu/code/HIROLRobotPlatform
export PYTHONPATH=$PYTHONPATH:/home/hanyu/code/hil-serl/serl_hirol_infra
```

### 相机问题
- 验证配置中的RealSense序列号
- 检查USB3连接
- 尝试`rs-enumerate-devices`列出相机

## 性能提示

1. **控制频率**：根据任务调整`hz`参数（典型10-20 Hz）
2. **更平滑设置**：在YAML中启用以获得更平滑但更慢的运动
3. **异步控制**：为高频控制循环启用
4. **内存管理**：使用`XLA_PYTHON_CLIENT_PREALLOCATE=false`

## 核心优势

- **减少假阳性**：人类直接标记错误预测
- **主动学习**：优先查询不确定样本
- **持续改进**：训练过程中不断优化
- **数据效率**：重点学习纠正样本
- **线程安全**：在线学习模式避免文件I/O冲突

这种设计让分类器能在实际使用中不断进化，特别适合解决机器人任务中的边界情况。

## 重要说明

### 反馈时机
- 人类反馈在每次`step`之后进行，不是在`reset`时
- 系统会根据分类器置信度自动决定是否查询

### 文件安全性
- **在线模式**（推荐）：使用内存队列，无文件读写冲突
- **离线模式**：可能存在并发读写风险，建议在训练暂停时重训练

### 选择建议
- 实时实验：使用`feedback_mode="online"`配合`train_rlpd_online_classifier.py`
- 批量分析：使用`feedback_mode="offline"`配合手动重训练
- 简单测试：使用`feedback_mode="simple"`仅用键盘标记

## 下一步

1. 配置机器人特定参数
2. 为您的任务收集演示
3. 训练初始奖励分类器
4. 运行带人类反馈的HIL-SERL训练
5. 通过人类纠正评估和微调
6. 使用收集的反馈数据重训练分类器