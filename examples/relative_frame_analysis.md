# RelativeFrame 环境包装器工作原理分析

## 概述
`RelativeFrame` 是一个Gym环境包装器，用于坐标系转换，将观察和动作从基座坐标系(base frame)转换到末端执行器坐标系(end-effector frame)。

## 核心功能

### 1. 坐标系定义
- **基座坐标系(Base/Spatial Frame)**: 机器人基座固定的世界坐标系
- **末端执行器坐标系(End-effector/Body Frame)**: 随末端执行器移动的局部坐标系
- **相对坐标系(Relative Frame)**: 以reset时的位姿为原点的相对坐标系

### 2. 主要组件

#### 2.1 Adjoint Matrix (伴随矩阵)
```python
self.adjoint_matrix = construct_adjoint_matrix(tcp_pose)
```
- 6x6矩阵，用于转换速度/力在不同坐标系间的表示
- 上半部分处理旋转，下半部分处理平移

#### 2.2 Homogeneous Transformation (齐次变换矩阵)
```python
self.T_r_o_inv = np.linalg.inv(construct_homogeneous_matrix(tcp_pose))
```
- 4x4矩阵，用于转换位置和姿态
- `T_r_o_inv`: 从基座坐标系到reset位姿坐标系的逆变换

## 工作流程

### Reset时
```python
def reset(self):
    obs, info = self.env.reset()
    
    # 1. 更新伴随矩阵（用于速度转换）
    self.adjoint_matrix = construct_adjoint_matrix(obs["state"]["tcp_pose"])
    
    # 2. 记录reset位姿作为相对参考系（如果启用）
    if self.include_relative_pose:
        self.T_r_o_inv = np.linalg.inv(
            construct_homogeneous_matrix(obs["state"]["tcp_pose"])
        )
    
    # 3. 转换观察到末端执行器坐标系
    return self.transform_observation(obs), info
```

### Step时
```python
def step(self, action):
    # 1. 动作转换：末端执行器坐标系 → 基座坐标系
    transformed_action = self.transform_action(action)
    
    # 2. 执行环境步进
    obs, reward, done, truncated, info = self.env.step(transformed_action)
    
    # 3. 更新伴随矩阵
    self.adjoint_matrix = construct_adjoint_matrix(obs["state"]["tcp_pose"])
    
    # 4. 观察转换：基座坐标系 → 末端执行器坐标系
    transformed_obs = self.transform_observation(obs)
    
    return transformed_obs, reward, done, truncated, info
```

## 转换详解

### 观察转换 (Base → End-effector)
```python
def transform_observation(self, obs):
    # 1. 速度转换：使用伴随矩阵的逆
    adjoint_inv = np.linalg.inv(self.adjoint_matrix)
    obs["state"]["tcp_vel"] = adjoint_inv @ obs["state"]["tcp_vel"]
    
    # 2. 位姿转换：相对于reset位姿（如果启用）
    if self.include_relative_pose:
        T_b_o = construct_homogeneous_matrix(obs["state"]["tcp_pose"])
        T_b_r = self.T_r_o_inv @ T_b_o  # 相对变换
        
        # 提取位置和姿态
        p_b_r = T_b_r[:3, 3]  # 相对位置
        theta_b_r = R.from_matrix(T_b_r[:3, :3]).as_quat()  # 相对姿态
        obs["state"]["tcp_pose"] = np.concatenate((p_b_r, theta_b_r))
    
    return obs
```

### 动作转换 (End-effector → Base)
```python
def transform_action(self, action):
    # 前6维（位置+旋转）从末端执行器坐标系转到基座坐标系
    action[:6] = self.adjoint_matrix @ action[:6]
    return action
```

## 使用场景

### 1. 相对控制
在末端执行器坐标系下，动作更直观：
- `[1, 0, 0, 0, 0, 0]` = 沿末端X轴前进
- `[0, 0, 1, 0, 0, 0]` = 沿末端Z轴（通常是接近方向）移动

### 2. 相对观察
当`include_relative_pose=True`时：
- 位姿是相对于reset时的位姿
- `[0, 0, 0, 0, 0, 0, 1]` = 仍在reset位置
- 便于学习相对运动策略

### 3. SpaceMouse集成
```python
if "intervene_action" in info:
    # SpaceMouse动作也需要逆转换
    info["intervene_action"] = self.transform_action_inv(info["intervene_action"])
```

## 与fr3_env的关系

在HIL-SERL中的使用流程：
```
SpaceMouse输入 → fr3_env.step() → RelativeFrame包装 → 
转换动作到基座系 → 执行 → 转换观察到末端系 → 返回给策略
```

### 关键差异
1. **fr3_env**: 在基座坐标系工作，处理绝对位姿
2. **RelativeFrame包装后**: 在末端执行器坐标系工作，处理相对运动

## 优势

1. **更自然的控制**: 动作在末端执行器局部坐标系定义
2. **泛化性更好**: 策略学习相对运动而非绝对位置
3. **简化学习**: 消除了基座坐标系的复杂性

## 潜在问题

1. **累积误差**: 长时间运行可能累积数值误差
2. **奇异性**: 某些姿态下矩阵求逆可能不稳定
3. **额外计算**: 每步需要矩阵运算开销

## 示例代码

```python
from franka_env.envs.relative_env import RelativeFrame

# 创建基础环境
base_env = FR3Env(...)

# 包装为相对坐标系
relative_env = RelativeFrame(
    base_env, 
    include_relative_pose=True  # 使用相对位姿
)

# 使用时
obs, _ = relative_env.reset()
# obs中的tcp_pose现在是[0,0,0,0,0,0,1]（相对原点）

action = np.array([0.01, 0, 0, 0, 0, 0, 0])  # 沿末端X轴移动
obs, reward, done, _, _ = relative_env.step(action)
# 动作自动转换到基座坐标系执行
```