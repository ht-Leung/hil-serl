# 重力补偿问题分析

## 问题现象
- SpaceMouse无输入，机器人仍然下降
- 下降是持续的，不是一次性的

## 根本原因：控制频率与重力补偿

### 1. panda_py的JointPosition控制器

panda_py的`controllers.JointPosition()`内部实现了重力补偿：
- 需要高频率更新来维持稳定的重力补偿
- 在Franka的实时内核中运行（1kHz）
- 控制器期望持续接收位置命令

### 2. 控制频率的影响

**HIROLRobotPlatform原生（正常工作）**:
```
遥操循环: 200Hz (5ms)
控制频率: 800Hz
→ 控制器持续运行，重力补偿稳定
```

**HIL-SERL原实现（10Hz）**:
```
环境步进: 10Hz (100ms)
→ 控制器100ms才收到一次命令
→ 重力补偿在间隔期间可能不足
```

### 3. 累积误差机制

```python
# 每个step的流程
def step(self, action):
    # 1. 基于当前位置计算目标
    self.nextpos = self.currpos.copy()
    self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]
    
    # 2. 发送命令
    self._send_pos_command(self.nextpos)
    
    # 3. 更新当前位置（读取实际位置）
    self._update_currpos()
```

**问题链**：
1. 低频率导致重力补偿不足
2. 机器人在重力作用下微小下降（比如下降1mm）
3. `_update_currpos()`读取到下降后的位置
4. 下次step基于下降后的位置计算
5. 即使SpaceMouse无输入，目标位置也是下降后的位置
6. 形成正反馈循环，持续下降

## 解决方案

### 方案1：提高控制频率（已实施）
```python
# 从10Hz改为500Hz
env = FR3Env(hz=500, ...)
```

### 方案2：分离位置维持
创建高频控制线程，持续发送当前目标位置：
```python
class FR3Env:
    def __init__(self):
        self._control_thread = threading.Thread(target=self._high_freq_control)
        self._target_pose = None
        
    def _high_freq_control(self):
        while self._running:
            if self._target_pose is not None:
                self.robot.send_pos_command(self._target_pose)
            time.sleep(0.002)  # 500Hz
```

### 方案3：使用阻抗控制
切换到阻抗控制模式，设置高刚度：
- 阻抗控制内部处理重力补偿
- 更适合低频命令更新

### 方案4：锁定参考位置
不基于实际位置更新，而是维护理想位置：
```python
def step(self, action):
    # 基于理想位置而非实际位置
    self.ideal_pos[:3] += xyz_delta * self.action_scale[0]
    self._send_pos_command(self.ideal_pos)
```

## 验证方法

1. **测试不同频率**：
   - 10Hz, 50Hz, 100Hz, 500Hz
   - 观察哪个频率开始稳定

2. **监控位置漂移**：
   ```python
   initial_z = robot.get_pose()[2]
   # 等待10秒不给命令
   time.sleep(10)
   drift = robot.get_pose()[2] - initial_z
   print(f"Z drift: {drift}m")
   ```

3. **对比控制模式**：
   - 位置控制 vs 阻抗控制
   - 观察哪种模式更稳定