<!--
 * @Author: Haotian Liang haotianliang10@gmail.com
 * @Date: 2025-08-07 10:59:07
 * @LastEditors: Haotian Liang haotianliang10@gmail.com
 * @LastEditTime: 2025-08-07 11:58:28
-->
# 遥操系统对比分析

## 关键差异对比

### 1. 控制频率
- **HIROLRobotPlatform**:
  - control_frequency: 800 Hz （控制器频率）
  - teleoperation_loop_time: 0.005s (200Hz 遥操循环)
  
- **HIL-SERL**:
  - hz: 10 Hz （环境步进频率）
  - 直接调用set_joint_command

### 2. 控制模式
- **HIROLRobotPlatform**:
  ```python
  # fr3_arm.py - 使用panda_py的控制器
  controllers.JointPosition()  # 带重力补偿的位置控制
  ```
  
- **HIL-SERL**:
  ```python
  # 直接IK + joint position命令
  self._robot._fr3_arm.set_joint_command("position", joint_target)
  ```





### 5. 增量控制 vs 绝对控制

- **HIROLRobotPlatform**:
  ```python
  # relative模式下
  self._init_pose[key][:3] += cur_ee_target[:3]  # 累积增量
  ```
  
- **HIL-SERL**:
  ```python
  self.nextpos = self.currpos.copy()
  self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]
  # 每次基于当前位置计算新位置
  ```

## 机器人一直下降的可能原因

1. **控制器重新初始化问题**：
   - 每次调用set_joint_command时都检查recover()
   - 如果有微小错误，控制器可能被重新初始化
   - 重新初始化后重力补偿可能需要时间稳定

2. **控制频率不匹配**：
   - HIL-SERL以10Hz发送命令
   - panda_py控制器期望更高频率
   - 低频率下重力补偿可能不够及时

3. **IK求解误差累积**：
   - 每次步进都计算IK
   - IK解可能有微小误差
   - 误差累积导致漂移

## 建议解决方案

1. **避免频繁的recover()调用**：
   - 只在真正需要时调用recover()
   - 或者修改recover()逻辑，减少不必要的控制器重新初始化

2. **提高控制频率**：
   - 考虑在FrankaInterface中启动一个高频控制线程
   - 维持稳定的重力补偿

3. **使用阻抗控制**：
   - 切换到阻抗控制模式
   - 更适合低频率命令更新

4. **锁定Z轴测试**：
   - 临时锁定Z轴，只允许XY运动
   - 隔离重力补偿问题