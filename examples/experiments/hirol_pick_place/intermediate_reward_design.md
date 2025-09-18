# HIL-SERL Pick-Place 任务中间奖励设计方案

## 问题背景

当前 `hirol_pick_place` 任务仅在最终任务成功时提供奖励（通过视觉分类器），存在以下问题：
- **稀疏奖励问题**：只有完全成功才有奖励，训练初期探索效率低
- **缺乏引导**：智能体无法学习中间步骤的价值
- **收敛速度慢**：需要大量随机探索才能发现成功路径

## 现有奖励机制分析

### 当前实现
1. **基础奖励**：`HIROLEnv.step()` 中固定返回 `reward=0`
2. **夹爪惩罚**：`GripperPenaltyWrapper` 对频繁切换施加 -0.02 惩罚
3. **最终奖励**：`MultiCameraBinaryRewardClassifierWrapper` 使用视觉分类器判断成功

### 代码位置
- 基础环境：`/home/hanyu/code/hil-serl/serl_hirol_infra/hirol_env/envs/hirol_env.py:360`
- 夹爪惩罚：`/home/hanyu/code/hil-serl/examples/experiments/hirol_pick_place/wrapper.py:115-134`
- 视觉奖励：`/home/hanyu/code/hil-serl/examples/experiments/hirol_pick_place/config.py:159-172`

## 方案设计

### 方案 1：基于力反馈的抓取检测

#### 原理
监测夹爪状态和力传感器数据，当检测到夹爪闭合且有力反馈时，判定抓取成功。

#### 实现细节
```python
class GraspDetectionWrapper(gym.Wrapper):
    def __init__(self, env, grasp_reward=1.0, force_threshold=5.0):
        # grasp_reward: 抓取成功的奖励值
        # force_threshold: 力检测阈值（牛顿）
        
    def _detect_grasp(self, obs, action):
        gripper_pos = obs["state"]["gripper_pose"][0]
        tcp_force = obs["state"]["tcp_force"]
        force_magnitude = np.linalg.norm(tcp_force[:3])
        
        # 检测条件
        gripper_closed = gripper_pos > 0.5  # 夹爪闭合
        force_detected = force_magnitude > self.force_threshold
        
        return gripper_closed and force_detected
```

#### 优缺点
- ✅ **优点**：
  - 物理准确，避免空抓误判
  - 基于真实传感器反馈
  - 可靠性高
- ❌ **缺点**：
  - 依赖力传感器精度
  - 需要调试力阈值
  - 软物体可能力反馈较小

### 方案 2：基于夹爪状态的简单检测

#### 原理
仅根据夹爪位置变化和动作指令判断抓取。

#### 实现细节
```python
class SimpleGraspWrapper(gym.Wrapper):
    def __init__(self, env, grasp_reward=1.0, hold_steps=5):
        # hold_steps: 夹爪保持闭合的步数
        
    def _detect_grasp(self, obs, action):
        gripper_pos = obs["state"]["gripper_pose"][0]
        gripper_command = action[-1] < 0  # 闭合指令
        
        if gripper_pos > 0.7 and gripper_command:
            self.hold_counter += 1
        else:
            self.hold_counter = 0
            
        return self.hold_counter >= self.hold_steps
```

#### 优缺点
- ✅ **优点**：
  - 实现简单
  - 不依赖力传感器
  - 计算开销小
- ❌ **缺点**：
  - 可能空抓也得奖励
  - 无法区分抓取质量
  - 容易被利用（reward hacking）

### 方案 3：多阶段奖励系统

#### 原理
将 pick-place 任务分解为多个阶段，每个阶段完成给予递增奖励。

#### 阶段定义
1. **接近阶段** (Approach)：末端执行器接近目标物体
   - 检测：距离 < 5cm
   - 奖励：+0.1
   
2. **抓取阶段** (Grasp)：成功抓取物体
   - 检测：夹爪闭合 + 力反馈
   - 奖励：+1.0
   
3. **提升阶段** (Lift)：将物体提升到一定高度
   - 检测：Z轴提升 > 10cm
   - 奖励：+0.5
   
4. **放置阶段** (Place)：将物体放置到目标位置
   - 检测：视觉分类器
   - 奖励：+2.0

#### 实现框架
```python
class MultiStageRewardWrapper(gym.Wrapper):
    def __init__(self, env, stage_rewards={
        'approach': 0.1,
        'grasp': 1.0,
        'lift': 0.5,
        'place': 2.0
    }):
        self.stages_completed = {
            'approach': False,
            'grasp': False,
            'lift': False,
            'place': False
        }
        
    def _compute_stage_rewards(self, obs, action):
        total_reward = 0.0
        
        # 逐阶段检测
        if not self.stages_completed['approach']:
            if self._check_approach(obs):
                total_reward += self.stage_rewards['approach']
                self.stages_completed['approach'] = True
                
        # 只有前置阶段完成才检测后续阶段
        if self.stages_completed['approach'] and not self.stages_completed['grasp']:
            if self._check_grasp(obs, action):
                total_reward += self.stage_rewards['grasp']
                self.stages_completed['grasp'] = True
                
        # ... 继续其他阶段
        
        return total_reward
```

#### 优缺点
- ✅ **优点**：
  - 细粒度引导学习
  - 加速收敛
  - 可解释性强
  - 便于调试
- ❌ **缺点**：
  - 需要预定义物体位置
  - 实现复杂
  - 可能导致次优策略

### 方案 4：基于距离的连续奖励

#### 原理
根据末端执行器与目标的距离变化提供连续奖励信号。

#### 实现
```python
class DistanceRewardWrapper(gym.Wrapper):
    def __init__(self, env, distance_scale=0.1):
        self.last_distance = None
        
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        current_distance = self._compute_distance(obs)
        if self.last_distance is not None:
            distance_delta = self.last_distance - current_distance
            reward += distance_delta * self.distance_scale
            
        self.last_distance = current_distance
        return obs, reward, done, truncated, info
```

#### 优缺点
- ✅ **优点**：
  - 连续信号，梯度平滑
  - 自然引导接近行为
- ❌ **缺点**：
  - 可能导致局部最优
  - 需要知道目标位置

## 推荐实施方案

### 短期方案（快速实现）
采用**方案 1**（基于力反馈的抓取检测）：
- 实现简单，一个 Wrapper 即可
- 解决最关键的抓取奖励问题
- 与现有系统兼容性好

### 长期方案（最优效果）
采用**方案 3**（多阶段奖励系统）：
- 提供完整的任务引导
- 显著加速训练
- 便于分析各阶段性能

## 实施步骤

### 1. 创建 Wrapper 文件
在 `/home/hanyu/code/hil-serl/examples/experiments/hirol_pick_place/` 下创建 `grasp_reward_wrapper.py`

### 2. 修改配置文件
在 `config.py` 的 `get_environment()` 方法中添加：

```python
# 在第 143 行后添加
env = GripperPenaltyWrapper(env, penalty=-0.02)
env = GraspDetectionWrapper(env, grasp_reward=1.0)  # 新增中间奖励
env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
```

### 3. 参数调优建议

#### 力阈值调优
```python
# 根据物体类型调整
soft_objects: force_threshold = 2.0  # 软物体
rigid_objects: force_threshold = 5.0  # 刚性物体
heavy_objects: force_threshold = 10.0  # 重物体
```

#### 奖励权重平衡
```python
# 确保中间奖励不会压倒最终奖励
grasp_reward = 1.0      # 中间奖励
final_reward = 2.0      # 最终奖励（可能需要调整分类器）
penalty = -0.02         # 保持原有惩罚
```

### 4. 测试验证

#### 单元测试
```python
def test_grasp_detection():
    env = create_test_env()
    env = GraspDetectionWrapper(env)
    
    # 测试空抓不得奖励
    obs, reward, _, _, info = env.step(close_gripper_action)
    assert info["intermediate_reward"] == 0
    
    # 测试有力反馈时得奖励
    # ... 设置力反馈
    obs, reward, _, _, info = env.step(close_gripper_action)
    assert info["intermediate_reward"] == 1.0
```

#### 集成测试
1. 运行带中间奖励的 actor
2. 监控奖励分布
3. 对比训练曲线

## 监控指标

### 训练监控
- `grasp_success_rate`: 抓取成功率
- `intermediate_reward_frequency`: 中间奖励触发频率
- `episode_reward_breakdown`: 各类奖励占比

### 调试信息
```python
# 在 Wrapper 中添加详细日志
if grasp_detected:
    print(f"[Grasp] Detected! Force: {force:.1f}N, Gripper: {gripper_pos:.2f}")
```

## 潜在问题与解决

### 问题 1：奖励 Hacking
**现象**：智能体学会触发奖励但不完成任务
**解决**：
- 设置奖励上限
- 增加任务完成的权重
- 使用奖励衰减

### 问题 2：力传感器噪声
**现象**：误触发或漏检
**解决**：
- 使用滑动平均滤波
- 多帧确认机制
- 自适应阈值

### 问题 3：训练不稳定
**现象**：奖励信号导致策略震荡
**解决**：
- 渐进式引入中间奖励
- 使用奖励归一化
- 调整学习率

## 参考资源

- [Sparse Reward Reinforcement Learning](https://arxiv.org/abs/1707.01495)
- [Curriculum Learning in RL](https://arxiv.org/abs/2003.04960)
- [Reward Shaping Techniques](https://arxiv.org/abs/1906.04349)

## 更新记录

- 2025-08-29: 初始设计文档
- 待实施: 方案 1 - 基于力反馈的抓取检测
- 待测试: 多阶段奖励系统