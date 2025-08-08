# Motion Smoothing Options for HIL-SERL FR3 Control

## 问题背景

HIL-SERL环境以10Hz频率运行，直接发送关节命令会导致机器人运动抖动。相比之下，teleoperation系统在800Hz运行，提供更平滑的控制。为了解决这个问题，我们实现了6种不同的运动平滑方案。

## Option 1: Blocking Servo Mode (阻塞式伺服模式)

### 原理
使用FR3Interface内置的servo模式，在指定时间内以1000Hz内部控制频率执行运动。

### 动机
- 利用底层已有的高频控制能力
- 最简单的实现方式

### 实现
```python
self._robot.move_to_pose(target_pose, mode="servo", duration=0.08)
```

### 优缺点
- ✅ 实现简单，运动非常平滑
- ✅ 利用底层1000Hz控制
- ❌ 阻塞主线程，可能影响数据收集
- ❌ 无法中断执行

### 适用场景
对实时性要求不高，可以接受阻塞的场景。

---

## Option 2: Direct IK Command (直接IK命令)

### 原理
计算逆运动学后直接发送关节位置命令，无插值。

### 动机
- 最快的响应速度
- 最小的计算开销

### 实现
```python
success, joint_target = self._robot._motion_controller.compute_ik(target_pose)
if success:
    self._robot._fr3_arm.set_joint_command("position", joint_target)
```

### 优缺点
- ✅ 响应极快，无延迟
- ✅ 计算开销最小
- ❌ 运动不平滑，存在抖动
- ❌ 关节加速度可能过大

### 适用场景
需要快速响应，可以容忍抖动的场景。

---

## Option 3: Quintic Polynomial Trajectory (五次多项式轨迹)

### 原理
使用五次多项式插值生成平滑轨迹，保证位置、速度、加速度连续。端点速度和加速度为零。

### 数学公式
```
s(t) = 10t³ - 15t⁴ + 6t⁵
其中 t ∈ [0, 1]
```

### 动机
- 数学上最优的点到点运动
- 保证所有阶导数连续

### 实现
```python
config = TrajectoryConfig(finish_time=0.1)
self._robot._motion_controller.execute_joint_trajectory(
    current_joints,
    joint_target,
    config
)
```

### 优缺点
- ✅ 数学最优，轨迹最平滑
- ✅ 加速度连续，减少机械冲击
- ❌ 强制端点速度为0，导致每个周期都有启停
- ❌ 不适合连续运动控制

### 适用场景
点到点精确定位任务，如装配、放置等。

---

## Option 4: S-Curve Interpolation (S曲线插值) 【当前启用】

### 原理
使用Smoothstep函数（3t² - 2t³）进行插值，提供平滑的S形速度曲线。

### 数学公式
```
smoothstep(t) = 3t² - 2t³
velocity(t) = 6t - 6t²
acceleration(t) = 6 - 12t
```

### 动机
- 在保持平滑的同时避免强制零速度
- 异步执行不阻塞主循环
- 速度曲线呈S形，自然平滑

### 实现
```python
def smooth_control():
    steps = 64  # 80ms at 800Hz
    dt = 0.00125  # 1/800 seconds
    
    for i in range(steps):
        t = (i + 1) / float(steps)
        alpha = t * t * (3.0 - 2.0 * t)  # Smoothstep
        interpolated_joints = current_joints + alpha * (joint_target - current_joints)
        self._robot._fr3_arm.set_joint_command("position", interpolated_joints)
        time.sleep(dt)

# 异步执行
control_thread = threading.Thread(target=smooth_control)
control_thread.daemon = True
control_thread.start()
```

### 优缺点
- ✅ 平滑的S形速度曲线
- ✅ 异步执行，不阻塞主循环
- ✅ 速度连续，适合连续控制
- ✅ 计算简单高效
- ❌ 加速度在端点有突变
- ❌ 线程管理增加复杂性

### 适用场景
**HIL-SERL的默认选择**，适合需要连续平滑控制的强化学习任务。

---

## Option 5: First-Order Low-Pass Filter (一阶低通滤波器)

### 原理
使用一阶IIR低通滤波器平滑目标位置。

### 数学公式
```
y[n] = α·x[n] + (1-α)·y[n-1]
其中 α ∈ (0, 1] 是滤波系数
```

### 动机
- 最简单的滤波实现
- 可动态调整平滑度
- 无需线程管理

### 实现
```python
if self._filtered_joint_target is None:
    self._filtered_joint_target = self._robot.get_joint_positions()

self._filtered_joint_target = (
    self._filter_alpha * joint_target + 
    (1 - self._filter_alpha) * self._filtered_joint_target
)

self._robot._fr3_arm.set_joint_command("position", self._filtered_joint_target)
```

### 参数选择
- α = 0.1-0.2: 非常平滑，响应慢
- α = 0.3-0.4: 平衡选择
- α = 0.5-0.7: 响应快，轻度平滑
- α = 0.8-1.0: 几乎无滤波

### 优缺点
- ✅ 实现最简单
- ✅ 计算开销极小
- ✅ 可动态调整滤波强度
- ❌ 存在稳态误差
- ❌ 响应延迟
- ❌ 对快速变化跟踪差

### 适用场景
简单任务，对精度要求不高但需要平滑运动的场景。

---

## Option 6: Trapezoidal Velocity Profile (梯形速度规划)

### 原理
将运动分为三个阶段：加速、匀速、减速，形成梯形速度曲线。

### 运动阶段
1. **加速阶段 (20%)**：二次曲线加速
2. **匀速阶段 (60%)**：恒定速度
3. **减速阶段 (20%)**：二次曲线减速

### 动机
- 工业标准的运动规划方法
- 速度连续，适合连续控制
- 可控制最大速度和加速度

### 实现
```python
def trapezoidal_control():
    accel_steps = int(0.2 * total_steps)
    decel_steps = int(0.2 * total_steps)
    const_steps = total_steps - accel_steps - decel_steps
    
    for i in range(total_steps):
        if i < accel_steps:
            # 加速阶段
            t = i / float(accel_steps)
            s = 0.5 * t * t
        elif i < accel_steps + const_steps:
            # 匀速阶段
            t = (i - accel_steps) / float(const_steps)
            s = s_accel_end + t * const_velocity
        else:
            # 减速阶段
            t = (i - accel_steps - const_steps) / float(decel_steps)
            s = 1.0 - 0.5 * (1.0 - t) * (1.0 - t)
        
        interpolated_joints = current_joints + s * delta
        self._robot._fr3_arm.set_joint_command("position", interpolated_joints)
```

### 优缺点
- ✅ 工业标准，可靠性高
- ✅ 可控制最大速度/加速度
- ✅ 速度连续，无启停
- ❌ 加速度有突变点
- ❌ 实现相对复杂

### 适用场景
工业应用，需要限制速度和加速度的场景。

---

## 性能对比

| 方案 | 平滑度 | 响应速度 | CPU开销 | 实现复杂度 | 精度 | 速度连续性 |
|------|--------|----------|---------|------------|------|-----------|
| Option 1 | ★★★★★ | ★★★ | ★★ | ★ | ★★★★★ | ★★ |
| Option 2 | ★ | ★★★★★ | ★ | ★ | ★★★ | ★ |
| Option 3 | ★★★★★ | ★★ | ★★★ | ★★★ | ★★★★★ | ★★ |
| Option 4 | ★★★★ | ★★★★ | ★★ | ★★ | ★★★★ | ★★★ |
| Option 5 | ★★★ | ★★★ | ★ | ★ | ★★★ | ★★★★ |
| Option 6 | ★★★★ | ★★★★ | ★★ | ★★★ | ★★★★ | ★★★★ |
| **Option 7** | ★★★★★ | ★★★★ | ★★ | ★★ | ★★★★ | ★★★★★ |

---

## Option 7: Critical Damped Second-Order Tracker (二阶临界阻尼跟踪器) 【当前启用】

### 原理
使用二阶临界阻尼系统实现平滑跟踪，保持速度连续性，完全避免启停问题。

### 数学模型
```
二阶系统动力学：
ẍ = ωn²(x_target - x) - 2ζωn·ẋ
其中：ζ = 1 (临界阻尼)

离散化（欧拉积分）：
v[n+1] = v[n] + a[n]·dt
x[n+1] = x[n] + v[n+1]·dt
```

### 动机
- **解决根本问题**：每段都从v=0开始导致的"停-走-停-走"现象
- **保持速度连续**：新目标到达时，从当前速度开始加速，而不是归零
- **常驻线程**：避免每次创建线程的开销
- **物理直观**：像弹簧-阻尼系统一样自然追踪目标

### 实现
```python
# 常驻800Hz控制线程
def _tracker_loop(self):
    dt = 1.0 / 800.0
    zeta = 1.0  # 临界阻尼
    
    while self._tracker_running:
        # 二阶系统动力学
        error = self._target_joints - self._current_joints
        acceleration = self._omega_n**2 * error - 2 * zeta * self._omega_n * self._joint_velocity
        
        # 欧拉积分
        self._joint_velocity += acceleration * dt
        self._current_joints += self._joint_velocity * dt
        
        # 发送命令
        self._robot._fr3_arm.set_joint_command("position", self._current_joints)
        time.sleep(dt)

# 10Hz主循环只更新目标
def send_pos_command(self, pose):
    joint_target = compute_ik(pose)
    self._target_joints = joint_target  # 线程自动跟踪
```

### 参数调节
- **ωn（自然频率）**：唯一可调参数
  - 15-20 rad/s：柔顺，响应慢（安定时间 0.23-0.31s）
  - 20-30 rad/s：平衡选择（安定时间 0.15-0.23s）
  - 30-40 rad/s：响应快，较硬（安定时间 0.12-0.15s）
  
- **安定时间公式**：t_s ≈ 4.6/ωn（达到95%目标）

### 优缺点
- ✅ **完全连续**：速度永不归零，真正的连续运动
- ✅ **最小延迟**：常驻线程，无需启动开销
- ✅ **物理合理**：模拟真实的弹簧-阻尼系统
- ✅ **单参数调节**：只需调整ωn即可改变响应特性
- ✅ **自适应**：自动处理大小移动，无需手动调整时长
- ❌ 无法精确控制到达时间
- ❌ 对快速往复运动可能有过冲

### 适用场景
**最适合HIL-SERL**：解决了10Hz控制的根本问题，实现真正的连续平滑控制。

---

## 推荐选择

### 当前选择：Option 7 (Critical Damped Tracker)
- **解决了根本问题**：速度连续，无启停
- **实现简洁高效**：二阶系统，物理直观
- **适合连续控制**：完美匹配HIL-SERL需求
- **易于调节**：单参数控制响应特性

### 其他场景推荐
- **快速原型验证**：Option 2 (Direct IK)
- **精确定位任务**：Option 3 (Quintic)
- **简单平滑需求**：Option 5 (Filter)
- **工业应用**：Option 6 (Trapezoidal)

---

## 配置说明

在 `config.py` 中可以配置相关参数：

```python
# Option 5 滤波器参数（已注释）
# FILTER_ALPHA = 0.3

# Option 6 梯形规划参数（已注释）
# MOTION_DURATION = 0.08
```

要切换不同的option，修改 `franka_interface.py` 中 `send_pos_command` 方法，取消注释相应的实现即可。