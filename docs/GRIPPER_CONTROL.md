# Fixed Gripper Task - Operation Guide

## 配置夹爪初始状态

在 `config.py` 中设置：

```python
class EnvConfig(DefaultEnvConfig):
    # ...
    GRIPPER_INIT_STATE = "close"  # 可选: "open", "close", "none"
```

## 操作流程

### 1. 正常操作流程
1. **启动程序** → 机器人移动到 reset 位置
2. **夹爪自动设置** → 根据 `GRIPPER_INIT_STATE` 配置自动打开或关闭
3. **等待按 Enter** → 显示 "press enter to start episode..."
4. **按 Enter** → 开始任务执行
5. **任务执行中** → 夹爪保持固定状态（由 `GripperCloseEnv` 控制）

### 2. 物体掉落恢复流程
如果任务执行过程中物体掉落：

1. **按 G 键** → 触发夹爪恢复（gripper recovery/homing）
   - 建议连续按两次 G 键确保恢复成功
   - 看到提示：`[Manual Recovery] Gripper recovery successful`

2. **手动放置物体** → 将物体重新放置到夹爪中

3. **继续任务** → 夹爪保持关闭状态继续执行

### 3. Episode 重置流程
当一个 episode 结束后：

1. **自动 reset** → 机器人回到初始位置
2. **夹爪自动设置** → 自动执行配置的初始状态
   - `GRIPPER_INIT_STATE = "close"` → 自动关闭夹爪
   - `GRIPPER_INIT_STATE = "open"` → 自动打开夹爪
3. **等待按 Enter** → 准备下一个 episode

## 键盘控制

| 按键 | 功能 |
|------|------|
| **G** | 夹爪恢复（gripper recovery/homing） |
| **ESC** | 紧急停止 |
| **Enter** | 开始 episode |

## 典型应用场景

### 场景 1：推动任务（Push Task）
```python
GRIPPER_INIT_STATE = "close"  # 夹爪始终关闭
```
- Reset 时自动关闭夹爪
- 用闭合的夹爪推动物体

### 场景 2：钩取任务（Hook Task）
```python
GRIPPER_INIT_STATE = "open"  # 夹爪始终张开
```
- Reset 时自动打开夹爪
- 用张开的夹爪钩取物体

## 注意事项

1. **GripperCloseEnv** 确保运行时夹爪动作始终为 0（关闭指令）
2. **GripperInitWrapper** 控制每次 reset 时的初始状态
3. 如果需要改变初始状态，只需修改 `GRIPPER_INIT_STATE` 配置
4. 按 G 键恢复夹爪不会改变配置的初始状态设定

## 调试提示

观察控制台输出确认夹爪状态：
- `[GripperInitWrapper] Closing gripper before episode start` - reset 时关闭
- `[GripperInitWrapper] Opening gripper before episode start` - reset 时打开
- `[Manual Recovery] Gripper recovery successful` - G 键恢复成功