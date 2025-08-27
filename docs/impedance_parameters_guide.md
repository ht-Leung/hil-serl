# HIL-SERL 阻抗参数配置指南

## 概述
HIL-SERL 系统中的阻抗参数控制机器人的柔顺性，影响机器人对外力的响应和位置控制的精度。本文档详细说明了阻抗参数的配置来源、优先级和使用方法。

## 阻抗参数来源

### 1. YAML 配置文件（初始默认值）
**位置**: `HIROLRobotPlatform/factory/tasks/inferences_tasks/serl/config/serl_fr3_config.yaml`

```yaml
controller_config:
  cartesian_impedance:
    translational_stiffness: 1500.0  # 平移刚度 (N/m)
    translational_damping: 80.0      # 平移阻尼 (Ns/m)
    rotational_stiffness: 100.0      # 旋转刚度 (Nm/rad)
    rotational_damping: 10.0         # 旋转阻尼 (Nms/rad)
```

这些参数在机器人系统初始化时被加载，作为控制器的初始默认值。

### 2. HIL-SERL 环境配置（运行时参数）
**位置**: `hil-serl/serl_hirol_infra/hirol_env/envs/hirol_env.py`

```python
class DefaultEnvConfig:
    # 常规控制模式 - 较柔软，适合交互任务
    COMPLIANCE_PARAM: ComplianceParams = ComplianceParams(
        translational_stiffness=1500,
        translational_damping=80,
        rotational_stiffness=100,
        rotational_damping=10,
    )
    
    # 精确模式 - 较硬，适合精确定位
    PRECISION_PARAM: ComplianceParams = ComplianceParams(
        translational_stiffness=2000,
        translational_damping=89,
        rotational_stiffness=150,
        rotational_damping=7,
    )
    
    # 复位模式 - 中等刚度
    RESET_PARAM: ComplianceParams = ComplianceParams(
        translational_stiffness=1800,
        translational_damping=85,
        rotational_stiffness=120,
        rotational_damping=8,
    )
```

### 3. 任务特定配置（可选）
**位置**: `hil-serl/examples/experiments/hirol_unifined/config.py`

```python
class EnvConfig(DefaultEnvConfig):
    # 继承默认配置，可根据需要覆盖
    # 当前 hirol_unifined 未覆盖，使用默认值
    pass
```

## 参数优先级

**优先级从高到低**：
1. **任务特定配置** (`hirol_unifined/config.py`) - 如果定义了覆盖值
2. **HIL-SERL 环境配置** (`hirol_env.py`) - 运行时动态参数
3. **YAML 配置文件** (`serl_fr3_config.yaml`) - 初始默认值

## 参数使用流程

### 初始化阶段
```python
# 1. 创建环境时
env = HIROLEnv(config=EnvConfig())

# 2. 内部初始化 SerlRobotInterface
self.robot = SerlRobotInterface(
    config_path=self.config.ROBOT_CONFIG_PATH,  # 如果为 None，使用默认 YAML
    auto_initialize=True
)
```

### 运行时切换
```python
# reset() 方法中
def reset(self):
    # 切换到常规柔顺模式
    self.robot.update_params(self.config.COMPLIANCE_PARAM)
    
# go_to_reset() 方法中
def go_to_reset(self):
    # 切换到精确模式进行准确复位
    self.robot.update_params(self.config.PRECISION_PARAM)
    # ... 执行复位动作 ...
    # 完成后切换回常规模式
    self.robot.update_params(self.config.COMPLIANCE_PARAM)
```

### 底层实现
```python
# SerlRobotInterface.update_params() 方法
def update_params(self, params: ComplianceParams):
    # 更新控制器参数
    controller.set_stiffness(
        translational_stiffness=params.translational_stiffness,
        rotational_stiffness=params.rotational_stiffness
    )
    controller.set_damping(
        translational_damping=params.translational_damping,
        rotational_damping=params.rotational_damping
    )
```

## 参数调节指南

### 刚度参数 (Stiffness)
- **低刚度 (500-1000 N/m)**: 机器人更柔软，适合人机协作
- **中刚度 (1000-2000 N/m)**: 平衡柔顺性和精度
- **高刚度 (2000-3000 N/m)**: 更精确的位置控制，但较硬

### 阻尼参数 (Damping)
- **低阻尼 (< 50 Ns/m)**: 响应快但可能振荡
- **中阻尼 (50-100 Ns/m)**: 平衡响应速度和稳定性
- **高阻尼 (> 100 Ns/m)**: 稳定但响应较慢

### 应用场景建议

| 场景 | 平移刚度 | 平移阻尼 | 旋转刚度 | 旋转阻尼 |
|-----|---------|---------|---------|---------|
| 人机协作 | 800-1200 | 60-80 | 60-80 | 8-10 |
| 精密装配 | 2000-2500 | 80-100 | 150-200 | 10-15 |
| 抓取操作 | 1500-1800 | 70-90 | 100-120 | 8-12 |
| 轨迹跟踪 | 1800-2200 | 85-95 | 120-150 | 9-11 |

## 自定义配置示例

### 为特定任务自定义参数
```python
# 在 experiments/your_task/config.py 中
from hirol_env.envs.hirol_env import DefaultEnvConfig
from factory.tasks.inferences_tasks.serl.serl_robot_interface import ComplianceParams

class EnvConfig(DefaultEnvConfig):
    # 自定义柔软参数用于细腻操作
    COMPLIANCE_PARAM = ComplianceParams(
        translational_stiffness=1000,  # 更柔软
        translational_damping=70,
        rotational_stiffness=80,
        rotational_damping=8,
    )
    
    # 自定义超精确模式
    PRECISION_PARAM = ComplianceParams(
        translational_stiffness=2500,  # 更硬
        translational_damping=100,
        rotational_stiffness=180,
        rotational_damping=12,
    )
```

### 动态调节参数
```python
# 在任务执行中动态调整
def perform_delicate_operation(self):
    # 临时使用更柔软的参数
    soft_params = ComplianceParams(
        translational_stiffness=800,
        translational_damping=60,
        rotational_stiffness=60,
        rotational_damping=6,
    )
    self.robot.update_params(soft_params)
    # ... 执行细腻操作 ...
    # 恢复常规参数
    self.robot.update_params(self.config.COMPLIANCE_PARAM)
```

## 注意事项

1. **参数范围限制**：确保参数在机器人硬件允许的范围内
2. **稳定性考虑**：过高的刚度配合过低的阻尼可能导致振荡
3. **任务适配**：根据具体任务需求调整参数
4. **安全第一**：在真实硬件上测试新参数时从保守值开始
5. **性能权衡**：更高的刚度提供更好的跟踪精度但降低柔顺性

## 调试建议

1. **观察机器人行为**：
   - 振荡：降低刚度或增加阻尼
   - 响应慢：增加刚度或降低阻尼
   - 位置误差大：增加刚度

2. **记录测试参数**：
   ```python
   log.info(f"Testing params: K_trans={params.translational_stiffness}, "
            f"D_trans={params.translational_damping}")
   ```

3. **渐进式调整**：每次调整参数变化不超过20-30%

## 相关文件路径

- 默认YAML配置: `HIROLRobotPlatform/factory/tasks/inferences_tasks/serl/config/serl_fr3_config.yaml`
- 环境基类: `hil-serl/serl_hirol_infra/hirol_env/envs/hirol_env.py`
- 机器人接口: `HIROLRobotPlatform/factory/tasks/inferences_tasks/serl/serl_robot_interface.py`
- 控制器实现: `HIROLRobotPlatform/controller/cartesian_impedance_controller.py`
- 任务配置示例: `hil-serl/examples/experiments/hirol_unifined/config.py`