# 相机资源管理解决方案

## 问题描述

在使用 HIL-SERL 系统时，经常出现相机设备繁忙（Device or resource busy）的错误，即使之前的进程已经被终止。这导致必须重启系统或手动释放相机资源才能继续使用。

### 根本原因

1. **进程异常退出**：当程序被 Ctrl+C 中断或异常崩溃时，相机资源没有被正确释放
2. **线程管理问题**：非守护线程在主进程退出后继续占用相机
3. **资源泄漏**：RealSense pipeline 没有正确关闭，导致设备句柄泄漏
4. **缺少析构函数**：Python 对象被垃圾回收时没有清理底层硬件资源

## 解决方案

### 1. 相机资源管理器

创建了全局的 `CameraResourceManager` 单例类，负责：
- 跟踪所有相机实例
- 在进程退出时自动清理所有相机资源
- 提供硬件重置功能

**文件**: `serl_hirol_infra/hirol_env/camera/camera_manager.py`

### 2. 改进的 RSCapture 类

增强了 `RSCapture` 类的资源管理：
- 添加 `__del__` 析构函数确保资源释放
- 实现重试机制处理设备繁忙
- 添加状态跟踪防止重复关闭
- 自动注册到全局管理器

**关键改进**：
```python
def __del__(self):
    """Destructor to ensure camera resources are released"""
    try:
        self.close()
    except:
        pass

def __init__(self, ...):
    # Register with camera manager for cleanup
    camera_manager.register_capture(self)
    
    # Try to start with retry logic for busy devices
    max_retries = 3
    for attempt in range(max_retries):
        try:
            self.profile = self.pipe.start(self.cfg)
            self._is_open = True
            break
        except RuntimeError as e:
            if "busy" in str(e).lower() and attempt < max_retries - 1:
                print(f"Camera {name} busy, retrying...")
                time.sleep(2)
```

### 3. 改进的 VideoCapture 类

优化了线程管理：
- 使用守护线程（daemon=True）
- 添加 `__del__` 析构函数
- 改进 close 方法，清空队列防止阻塞
- 添加状态标志防止重复关闭

### 4. 相机释放工具

创建了独立的相机释放脚本 `release_cameras.py`：
- 检查所有相机状态
- 强制释放占用的相机
- 支持自动模式和交互模式

**使用方法**：
```bash
# 交互式释放
conda run -n hilserl python serl_hirol_infra/hirol_env/camera/release_cameras.py

# 自动释放所有相机
conda run -n hilserl python serl_hirol_infra/hirol_env/camera/release_cameras.py --auto
```

### 5. HIROLEnv 清理逻辑改进

分离了机器人恢复和夹爪恢复功能：
- `_recover()`: 只负责清除机器人错误
- `_recover_gripper()`: 独立的夹爪恢复功能
- 改进信号处理器，防止重复调用

## 使用建议

### 预防措施

1. **正常退出程序**：尽量使用正常的退出方式而不是强制终止
2. **定期检查相机状态**：使用释放工具检查相机可用性
3. **使用 with 语句**：在可能的情况下使用上下文管理器

### 故障排除

当遇到相机占用问题时：

1. **检查僵尸进程**：
```bash
ps aux | grep -E "python.*hirol|python.*serl" | grep -v grep
```

2. **释放相机资源**：
```bash
conda run -n hilserl python serl_hirol_infra/hirol_env/camera/release_cameras.py --auto
```

3. **如果问题持续**：
```bash
# 查找并杀死所有相关进程
pkill -f "python.*hirol"
pkill -f "python.*serl"

# 等待几秒后重试
sleep 3
conda run -n hilserl python serl_hirol_infra/hirol_env/camera/release_cameras.py --auto
```

## 技术细节

### Linux 设备文件锁定

RealSense 相机使用 V4L2（Video4Linux2）接口，当进程异常退出时，可能导致：
- 文件描述符未关闭
- 内核驱动状态不一致
- 设备节点被锁定

### pyrealsense2 库行为

- pipeline.start() 获取设备独占访问权
- pipeline.stop() 释放设备
- 如果没有调用 stop()，设备保持占用状态
- 即使进程退出，USB 设备可能需要时间恢复

### Python 垃圾回收

- Python 的垃圾回收不保证调用 `__del__`
- 使用 atexit 模块确保清理函数被调用
- 弱引用（weakref）防止循环引用

## 未来改进

1. **实现上下文管理器**：为所有相机类实现 `__enter__` 和 `__exit__`
2. **添加健康检查**：定期检查相机连接状态
3. **实现连接池**：复用相机连接而不是每次重新创建
4. **添加监控**：记录相机使用情况和错误统计
5. **硬件看门狗**：在固定时间内无响应时自动重置

## 相关文件

- `/home/hanyu/code/hil-serl/serl_hirol_infra/hirol_env/camera/rs_capture.py` - RealSense 捕获类
- `/home/hanyu/code/hil-serl/serl_hirol_infra/hirol_env/camera/video_capture.py` - 视频捕获包装器
- `/home/hanyu/code/hil-serl/serl_hirol_infra/hirol_env/camera/camera_manager.py` - 相机资源管理器
- `/home/hanyu/code/hil-serl/serl_hirol_infra/hirol_env/camera/release_cameras.py` - 相机释放工具
- `/home/hanyu/code/hil-serl/serl_hirol_infra/hirol_env/envs/hirol_env.py` - 环境清理逻辑