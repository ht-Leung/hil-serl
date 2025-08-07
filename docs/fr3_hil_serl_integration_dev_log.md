# FR3 HIL-SERL Integration Development Log

## 项目概述
**日期**: 2025-08-06  
**目标**: 将HIROLRobotPlatform的FR3接口集成到HIL-SERL框架，替换原有的HTTP请求方式，实现直接的机器人控制

## 开发记录

### Phase 1: FR3环境实现 (已完成)

#### 1.1 代码结构分析
- **HIL-SERL架构**:
  - `serl_hirol_infra/`: 核心基础设施
  - `serl_robot_infra/`: 机器人接口（原HTTP方式）
  - `examples/experiments/`: 具体任务配置

#### 1.2 FR3接口封装
创建了 `serl_hirol_infra/interface/franka_interface.py`:
- 基于HIROLRobotPlatform的FR3Interface
- 实现RobotInterface抽象接口
- 支持位置控制、阻抗控制、夹爪控制

#### 1.3 FR3环境实现
创建了 `serl_hirol_infra/hirol_env/envs/fr3_env.py`:
- 继承gym.Env标准接口
- 集成相机捕获（RealSense）
- 支持SpaceMouse遥操
- 替换所有HTTP请求为直接接口调用

### Phase 2: 任务配置 (已完成)

#### 2.1 创建FR3 Reach任务
`examples/experiments/fr3_reach/config.py`:
```python
class EnvConfig(DefaultEnvConfig):
    # 机器人配置
    ROBOT_IP = "192.168.3.102"
    
    # 相机配置
    REALSENSE_CAMERAS = {
        "wrist_1": {"serial_number": "332322073603"},
        "side": {"serial_number": "244222075350"},
    }
    
    # 动作缩放参数
    ACTION_SCALE = np.array([0.02, 0.15, 1])
```

#### 2.2 注册任务映射
在 `examples/experiments/mappings.py` 添加:
```python
CONFIG_MAPPING = {
    "fr3_reach": FR3ReachTrainConfig,
}
```

### Phase 3: Bug修复记录

#### 3.1 模块导入问题
**错误**: `ModuleNotFoundError: No module named 'serl_hirol_infra'`
**原因**: Python路径未正确设置
**修复**: 添加sys.path配置
```python
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
```

#### 3.2 ToolState初始化问题
**错误**: `AttributeError: 'ToolState' object has no attribute '_position'`
**原因**: ToolState类缺少__init__方法
**修复**: 在`HIROLRobotPlatform/hardware/base/utils.py`添加初始化
```python
class ToolState:
    def __init__(self):
        self._position: np.float32 = 0.0
        self._force: np.float32 = 0.0
        self._is_grasped: bool = False
        self._tool_type: ToolType = ToolType.GRIPPER
```

#### 3.3 FrankaInterface参数问题
**错误**: `TypeError: FrankaInterface() got an unexpected keyword argument 'robot_ip'`
**原因**: FrankaInterface使用配置文件而非参数
**修复**: 移除robot_ip参数，使用默认配置

#### 3.4 ComplianceParams类型问题
**错误**: `TypeError: 'ComplianceParams' object is not iterable`
**原因**: update_params期望字典但收到对象
**修复**: 添加类型检查和转换
```python
if isinstance(params, ComplianceParams):
    params_dict = {
        'translational_stiffness': params.translational_stiffness,
        # ...
    }
    params = params_dict
```

#### 3.5 观察空间包装问题
**错误**: `IndexError: too many indices for array`
**原因**: ChunkingWrapper改变了观察结构
**修复**: 处理包装后的观察格式

### Phase 4: 遥操控制优化

#### 4.1 控制问题诊断
**现象**:
- 动作阻塞且夸张
- 高延迟（~1秒）
- 触发安全反射（cartesian_reflex, joint_velocity_violation）

#### 4.2 问题根源分析

##### 4.2.1 阻塞的Servo模式
**问题代码** (FR3Interface.execute_servo_motion):
```python
def execute_servo_motion(self, joint_target, duration=1.0):
    steps = int(duration * 100)  # 100Hz
    for _ in range(steps):
        self._fr3_arm.set_joint_command(mode, joint_target)
        time.sleep(0.01)
```
**问题**: 
- 每个命令阻塞1秒
- 发送相同目标100次
- 无法响应新命令

##### 4.2.2 绝对位置控制
- 每个命令是绝对目标位置
- 机器人试图在1秒内到达任意距离的目标
- 导致急速运动触发安全限制

##### 4.2.3 SpaceMouse缩放链
```
原始输入 → ÷350 (pyspacemouse) → ×ACTION_SCALE → 位置增量
最大值: 32767 ÷ 350 = 93.6
实际输出范围: [-93, 93]
```

#### 4.3 解决方案实施

##### 4.3.1 移除阻塞循环
**修改** `franka_interface.py`:
```python
def send_pos_command(self, pose: np.ndarray) -> None:
    # 直接IK计算 + 单次关节命令
    success, joint_target = self._robot._motion_controller.compute_ik(target_pose)
    if success:
        self._robot._fr3_arm.set_joint_command("position", joint_target)
```

##### 4.3.2 优化控制频率
- FR3Interface servo模式：从100Hz循环改为单次命令
- 轨迹模式finish_time：从2.0秒改为0.1秒（匹配10Hz）

##### 4.3.3 调整ACTION_SCALE
```python
# 原值：导致运动过小
ACTION_SCALE = np.array([0.015, 0.1, 1])

# 优化后：更好的响应性
ACTION_SCALE = np.array([0.02, 0.15, 1])
# 位置: 0.26 * 0.02 = 5.2mm/帧，10Hz下 = 52mm/s 最大速度
```

##### 4.3.4 移除_recover()频繁调用
```python
def _send_pos_command(self, pos: np.ndarray) -> None:
    if self.robot is not None:
        # self._recover()  # 禁用 - 只在真正需要时调用
        self.robot.send_pos_command(pos)
```

### Phase 5: 性能优化结果

#### 5.1 延迟改善
- **修复前**: 每个命令1秒延迟
- **修复后**: <10ms响应时间

#### 5.2 运动平滑性
- **修复前**: 阻塞、跳跃式运动
- **修复后**: 连续平滑运动

#### 5.3 安全性
- **修复前**: 频繁触发cartesian_reflex
- **修复后**: 在安全限制内平稳运行

## 测试验证

### 测试脚本
创建了多个测试脚本验证功能:
1. `test_fr3_direct_control.py` - 测试直接控制
2. `debug_spacemouse.py` - 监控SpaceMouse输出
3. `record_demo.py` / `record_success_fail.py` - 演示记录

### 测试结果
- ✅ 环境创建和重置
- ✅ SpaceMouse遥操控制
- ✅ 相机图像采集
- ✅ 演示数据记录
- ✅ 与HIL-SERL训练流程兼容

## 关键配置参数

### 控制参数
```python
# 控制频率
hz = 10  # 主控制循环

# 动作缩放
ACTION_SCALE = [0.02, 0.15, 1]  # [位置, 旋转, 夹爪]

# 阻抗参数
COMPLIANCE_PARAM = ComplianceParams(
    translational_stiffness=1500,
    translational_damping=80,
    rotational_stiffness=100,
    rotational_damping=10,
)
```

### 工作空间限制
```python
ABS_POSE_LIMIT_HIGH = [0.7, 0.3, 0.6, π+0.5, 0.5, 0.5]
ABS_POSE_LIMIT_LOW = [0.3, -0.3, 0.1, π-0.5, -0.5, -0.5]
```

## 未来改进建议

1. **增量位置控制**: 考虑实现真正的速度控制模式
2. **自适应缩放**: 根据当前速度动态调整ACTION_SCALE
3. **力反馈集成**: 利用力/扭矩传感器实现柔顺控制
4. **多线程优化**: 分离感知和控制线程提高响应性
5. **轨迹预测**: 使用SpaceMouse输入预测未来轨迹

## 经验总结

### 成功因素
1. **直接接口调用**: 消除HTTP开销
2. **单次命令执行**: 避免阻塞循环
3. **合理的参数调优**: 基于实际测量的缩放值
4. **错误恢复策略**: 只在需要时重新初始化

### 教训
1. **避免过度抽象**: 简单直接的控制流更可靠
2. **理解硬件限制**: SpaceMouse输出范围和机器人速度限制
3. **测试驱动**: 每个修改都需要实际硬件测试
4. **日志的重要性**: 详细日志帮助快速定位问题

## 代码提交记录

### 新增文件
- `serl_hirol_infra/interface/franka_interface.py`
- `serl_hirol_infra/hirol_env/envs/fr3_env.py` 
- `examples/experiments/fr3_reach/config.py`
- `examples/debug_spacemouse.py`

### 修改文件
- `HIROLRobotPlatform/hardware/base/utils.py` - 添加ToolState初始化
- `HIROLRobotPlatform/hardware/fr3/fr3interface.py` - 优化servo模式
- `examples/experiments/mappings.py` - 注册fr3_reach任务

## 参考资料

1. [HIL-SERL论文](https://hil-serl.github.io/)
2. [Franka Control Interface (FCI) 文档](https://frankaemika.github.io/docs/)
3. [SpaceMouse SDK 文档](https://www.3dconnexion.com/uk/support/developer-program/)
4. [OpenAI Gym 接口规范](https://gymnasium.farama.org/)

---

*文档更新于: 2025-08-06*  
*作者: Haotian Liang*