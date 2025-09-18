# 在线学习系统说明

## 问题解答

### 1. 人类反馈时机
**问：反馈是在step还是reset时进行？**

答：反馈在每次`step`之后进行。具体流程：
1. 机器人执行动作
2. 获取新观察
3. 分类器预测奖励
4. 根据置信度决定是否查询人类
5. 如果查询，等待键盘输入（非阻塞，超时0.3秒）
6. 返回奖励和下一观察

### 2. 文件读写风险
**问：在线重训练是否有文件读写冲突？**

答：原版本存在风险，新的集成版本已解决：

**风险点：**
- 原版本：Actor写反馈文件 → 独立脚本读文件重训练 → 保存新检查点 → Actor加载检查点
- 问题：多进程同时读写可能导致文件损坏

**解决方案：**
- 使用内存队列代替文件传递反馈
- Learner进程中独立线程进行重训练
- 版本控制避免检查点冲突
- 线程锁保护关键资源

### 3. 集成到train_rlpd
**问：能否集成到训练循环？**

答：已实现`train_rlpd_online_classifier.py`，特点：
- 在Learner进程运行分类器重训练
- 不阻塞Actor数据收集
- 自动版本管理
- 线程安全

## 使用模式对比

### 模式1：离线反馈（原版本）
```python
# config.py
def get_environment(..., feedback_mode="offline"):
```
- 反馈保存到文件
- 手动运行重训练脚本
- 适合实验后批量分析

### 模式2：在线反馈（推荐）
```python
# config.py  
def get_environment(..., feedback_mode="online"):
```
- 反馈通过info dict传递
- 自动在后台重训练
- 无文件I/O冲突
- 实时改进分类器

### 模式3：简化键盘反馈
```python
# config.py
def get_environment(..., feedback_mode="simple"):
```
- 仅按's'标记成功
- 无分类器依赖
- 最轻量级选项

### 模式4：传统键盘奖励
```python
# config.py
def get_environment(..., feedback_mode="keyboard"):
```
- 原始KeyboardRewardWrapper
- 向后兼容

## 使用在线学习系统

### 方法1：使用集成训练脚本（推荐）

```bash
# 修改config.py设置反馈模式
# feedback_mode="online"

# 运行集成版训练
python train_rlpd_online_classifier.py \
    --exp_name=hirol_online_classifier_fixed_gripper \
    --learner \
    --online_classifier \
    --classifier_retrain_interval=100 \
    --feedback_weight=2.0

# 另一终端运行actor
python train_rlpd_online_classifier.py \
    --exp_name=hirol_online_classifier_fixed_gripper \
    --actor \
    --ip=localhost
```

### 方法2：使用原版train_rlpd + 在线wrapper

```bash
# 修改config.py
# feedback_mode="online"

# 使用标准训练脚本
python train_rlpd.py --exp_name=hirol_online_classifier_fixed_gripper --learner
python train_rlpd.py --exp_name=hirol_online_classifier_fixed_gripper --actor
```

反馈会通过info dict传递，但不会自动重训练。

## 键盘控制说明

### 在线反馈模式
- **`s`**：确认成功（真阳性）
- **`f`**：标记假阳性（实际失败）
- **`n`**：标记假阴性（漏判成功）
- **`c`**：跳过此次查询
- **`p`**：暂停/恢复反馈收集

### 简化模式
- **`s`**：标记当前episode为成功

## 系统架构

```
Actor进程                          Learner进程
    ↓                                  ↓
Environment                    OnlineClassifierTrainer
    ↓                                  ↓
OnlineFeedbackWrapper          分类器重训练线程
    ↓                                  ↓
获取分类器预测 ←─────────────→ 更新分类器版本
    ↓                                  ↓
查询人类反馈                    收集反馈数据
    ↓                                  ↓
info['human_feedback'] ─────→ 反馈队列
                                      ↓
                                定期重训练
```

## 性能优化建议

### 1. 减少查询频率
```python
# 只在极不确定时查询
query_threshold=0.70  # 提高下限
confidence_threshold=0.80  # 降低上限
```

### 2. 快速重训练
```python
# 减少重训练epochs
--classifier_epochs=5  # 默认10
--classifier_batch_size=128  # 减小批次
```

### 3. 异步反馈
```python
# 使用更短超时
timeout=0.1  # 100ms超时，不等待
```

## 常见问题

### Q: 为什么有时候不查询我？
A: 系统只在置信度65-85%之间查询，太确定或太不确定都不会询问。

### Q: 反馈后多久重训练？
A: 默认每100个反馈触发一次重训练，可通过`--classifier_retrain_interval`调整。

### Q: 可以暂停反馈吗？
A: 按'p'键可以暂停/恢复反馈收集。

### Q: 重训练会影响训练速度吗？
A: 重训练在独立线程进行，对主训练循环影响很小（<5%性能损失）。

## 实验建议

1. **初期密集反馈**：前1000步降低查询阈值，收集更多数据
2. **中期选择性反馈**：提高阈值，只纠正明显错误
3. **后期验证**：暂停反馈，观察分类器自主表现

## 数据分析

反馈数据和统计信息保存在：
```
checkpoints/
  online_classifier/
    v1/  # 各版本检查点
    v2/
    final_stats.pkl  # 最终统计
```

分析脚本：
```python
import pickle
stats = pickle.load(open("final_stats.pkl", "rb"))
print(f"假阳性率: {stats['false_positives']/stats['total_feedback']:.1%}")
```