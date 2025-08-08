'''
Author: Haotian Liang
Date: 2025-08-06
Description: Franka robot interface implementation using HIROLRobotPlatform
'''

import sys
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
import logging
import time
import threading

# Add HIROLRobotPlatform to path
platform_path = Path(__file__).parent.parent.parent.parent / "HIROLRobotPlatform"
sys.path.insert(0, str(platform_path))

from hardware.fr3.fr3interface import (
    FR3Interface as FR3,
    Pose,
    ImpedanceParams,
    TrajectoryConfig
)

from .robot_interface import RobotInterface, RobotState, ComplianceParams, LoadParams

# Configure logging
logger = logging.getLogger(__name__)


class FrankaInterface(RobotInterface):
    """Franka机器人接口实现"""
    
    def __init__(self, config_path: Optional[str] = None, auto_initialize: bool = True):
        """
        初始化Franka接口
        
        参数:
            config_path: FR3配置文件路径
            auto_initialize: 是否自动初始化机器人连接
        """
        self._robot = FR3(config_path=config_path, auto_initialize=auto_initialize)
        self._last_jacobian = np.zeros((6, 7))  # 缓存雅可比矩阵
        self._compliance_params = None
        self._load_params = None
        
        # 二阶临界阻尼跟踪器参数 (Critical Damped Tracker)
        self._tracker_thread = None
        self._tracker_running = False
        self._tracker_lock = threading.Lock()  # 线程同步锁
        self._pause_tracker = False  # 暂停标志，用于reset等场景
        self._target_joints = None  # 目标关节角度
        self._current_joints = None  # 当前插值的关节角度
        self._joint_velocity = None  # 关节速度
        
        # 二阶系统参数
        self._omega_n = 25.0  # 自然频率 (rad/s) - 可调参数，越大跟踪越紧但越硬
        # 经验值: 15-35 rad/s，对应安定时间 t_s = 4.6/omega_n
        # omega_n=25 -> t_s≈0.18s (95%到达)
        
        # 获取机器人模型用于计算雅可比
        self._robot_model = self._robot._robot_model
        
        # 启动常驻插补线程
        self._start_tracker_thread()
        
        logger.info("FrankaInterface initialized with critical damped tracker")
    
    def get_state(self) -> RobotState:
        """
        获取机器人当前状态
        
        返回:
            RobotState: 包含完整机器人状态的字典
        """
        # 获取位姿
        pose = self._robot.get_pose()
        pose_array = pose.to_array()  # [x, y, z, qx, qy, qz, qw]
        
        # 获取关节状态
        joint_positions = self._robot.get_joint_positions()
        joint_velocities = self._robot.get_joint_velocities()
        
        # 获取力/力矩
        force_torque = self._robot.get_tcp_force_torque(frame="base")
        force = force_torque['force']
        torque = force_torque['torque']
        
        # 获取夹爪状态
        if self._robot.has_gripper():
            gripper_width = self._robot.get_gripper_width()
            gripper_pos = gripper_width / 0.08  # 归一化到[0,1]
        else:
            gripper_pos = 0.0
        
        # 计算雅可比矩阵
        jacobian = self._compute_jacobian(joint_positions)
        
        # 计算TCP速度 (通过雅可比矩阵: v = J * dq)
        tcp_velocity = jacobian @ joint_velocities
        
        return RobotState(
            pose=pose_array,
            vel=tcp_velocity,
            force=force,
            torque=torque,
            gripper_pos=gripper_pos,
            q=joint_positions,
            dq=joint_velocities,
            jacobian=jacobian
        )
    
    def send_pos_command(self, pose: np.ndarray) -> None:
        """
        发送位置控制命令（使用高频插值平滑控制）
        
        参数:
            pose: np.ndarray(7,) - 目标TCP位姿 [x,y,z,qx,qy,qz,qw]
        """
        # 将numpy数组转换为Pose对象
        target_pose = Pose(
            x=float(pose[0]), y=float(pose[1]), z=float(pose[2]),
            qx=float(pose[3]), qy=float(pose[4]), 
            qz=float(pose[5]), qw=float(pose[6])
        )
        
        # Option 1: Use blocking servo mode with 100ms smooth trajectory (1000Hz internal)
        # Uncomment this for smoother motion at the cost of blocking
        # self._robot.move_to_pose(target_pose, mode="servo", duration=0.08)
        
        # Option 2: Direct IK command (current implementation - fast but may be jerky)
        # success, joint_target = self._robot._motion_controller.compute_ik(target_pose)
        # if success:
        #     # Send single joint command without blocking loop
        #     self._robot._fr3_arm.set_joint_command("position", joint_target)
        # else:
        #     logger.warning("IK failed for target pose")
        
        # Option 3: Use trajectory mode for smooth motion (五次多项式 速度是0 不太对)
        # 计算目标关节角度
        # success, joint_target = self._robot._motion_controller.compute_ik(target_pose)
        # if not success:
        #     logger.warning("IK failed for target pose")
        #     return

        # # 获取当前关节角度
        # current_joints = self._robot.get_joint_positions()

        # # 使用关节轨迹规划实现平滑运动（80ms内完成）
        # config = TrajectoryConfig(finish_time=0.1)
        # self._robot._motion_controller.execute_joint_trajectory(
        #     current_joints,
        #     joint_target,
        #     config
        # )
        
        
        # Option 7: Critical Damped Second-Order Tracker (二阶临界阻尼跟踪器) - 默认实现
        # 保持速度连续，避免启停，使用常驻线程实现平滑跟踪
        
        # Compute target joint angles
        success, joint_target = self._robot._motion_controller.compute_ik(target_pose)
        if not success:
            logger.warning("IK failed for target pose")
            return
        
        # 更新目标关节角度（线程会自动跟踪）
        self._target_joints = joint_target
        
    
    def pause_tracker(self) -> None:
        """暂停跟踪器线程（线程安全）"""
        with self._tracker_lock:
            self._pause_tracker = True
        time.sleep(0.002)  # 等待当前控制周期结束 (>1/800s)
        logger.debug("Tracker paused")
    
    def resume_tracker(self, sync_to_current: bool = True) -> None:
        """恢复跟踪器线程（线程安全）
        
        参数:
            sync_to_current: 是否同步跟踪器状态到当前关节位置
        """
        if sync_to_current:
            # 获取当前实际关节位置
            current_joints = self._robot.get_joint_positions()
            with self._tracker_lock:
                # 同步跟踪器状态，避免跳变
                self._current_joints = current_joints.copy()
                self._target_joints = current_joints.copy()
                self._joint_velocity = np.zeros(7)
                self._pause_tracker = False
        else:
            with self._tracker_lock:
                self._pause_tracker = False
        logger.debug("Tracker resumed")
    
    def send_pos_command_direct(self, pose: np.ndarray) -> None:
        """
        直接发送位置命令，绕过跟踪器（用于reset等特殊场景）
        
        参数:
            pose: np.ndarray(7,) - 目标TCP位姿 [x,y,z,qx,qy,qz,qw]
        """
        # 计算目标关节角度
        target_pose = Pose(
            x=float(pose[0]), y=float(pose[1]), z=float(pose[2]),
            qx=float(pose[3]), qy=float(pose[4]), 
            qz=float(pose[5]), qw=float(pose[6])
        )
        
        success, joint_target = self._robot._motion_controller.compute_ik(target_pose)
        if not success:
            logger.warning("IK failed for target pose")
            return
        
        # 暂停跟踪器
        self.pause_tracker()
        
        # 直接发送命令
        self._robot._fr3_arm.set_joint_command("position", joint_target)
        
        # 同步跟踪器状态到新位置
        with self._tracker_lock:
            self._current_joints = joint_target.copy()
            self._target_joints = joint_target.copy()
            self._joint_velocity = np.zeros(7)
        
        # 恢复跟踪器
        self.resume_tracker(sync_to_current=False)  # 已经手动同步了，不需要再同步
    
    def send_pos_trajectory_command(self, pose: np.ndarray, 
                                   finish_time: float = 2.0) -> None:
        """
        发送位置轨迹命令（trajectory模式，平滑运动）
        用于reset等需要精确到达的场景，绕过跟踪器
        
        参数:
            pose: np.ndarray(7,) - 目标TCP位姿 [x,y,z,qx,qy,qz,qw]
            finish_time: float - 轨迹完成时间（秒），默认2秒
        """
        # 将numpy数组转换为Pose对象
        target_pose = Pose(
            x=float(pose[0]), y=float(pose[1]), z=float(pose[2]),
            qx=float(pose[3]), qy=float(pose[4]), 
            qz=float(pose[5]), qw=float(pose[6])
        )
        
        # 暂停跟踪器，避免干扰
        self.pause_tracker()
        
        # 使用trajectory模式进行平滑运动
        config = TrajectoryConfig(finish_time=finish_time)
        self._robot.move_to_pose(target_pose, mode="trajectory", config=config)
        
        # 等待运动完成
        time.sleep(finish_time + 0.1)
        
        # 恢复跟踪器，自动同步到当前位置
        self.resume_tracker(sync_to_current=True)
    
    def send_gripper_command(self, position: float, mode: str = "binary") -> None:
        """
        控制夹爪
        
        参数:
            position: float - 目标位置
                binary模式: <-0.5 关闭, >0.5 打开
                continuous模式: [-1,1] 映射到夹爪开合范围
            mode: str - 控制模式 ("binary" 或 "continuous")
        """
        if not self._robot.has_gripper():
            logger.warning("Gripper not available")
            return
        
        if mode == "binary":
            if position < -0.5:
                self._robot.close_gripper()
            elif position > 0.5:
                self._robot.open_gripper()
        elif mode == "continuous":
            # 将[-1,1]映射到[0,0.08]米
            width = (position + 1.0) * 0.04  # [-1,1] -> [0,0.08]
            width = np.clip(width, 0.0, 0.08)
            self._robot.grasp(width=width)
        else:
            raise ValueError(f"Unknown gripper mode: {mode}")
        
        # 等待夹爪动作完成
        self._robot.wait_gripper_idle(timeout=2.0)
    
    def open_gripper(self) -> None:
        """完全打开夹爪"""
        if not self._robot.has_gripper():
            logger.warning("Gripper not available")
            return
        
        self._robot.open_gripper().wait_gripper_idle()
    
    def close_gripper(self) -> None:
        """完全关闭夹爪"""
        if not self._robot.has_gripper():
            logger.warning("Gripper not available")
            return
        
        self._robot.close_gripper().wait_gripper_idle()
    
    def joint_reset(self) -> None:
        """执行关节级别的重置，返回home位置"""
        # 暂停跟踪器，避免干扰
        self.pause_tracker()
        
        # 执行home运动
        self._robot.move_to_home()
        
        # 恢复跟踪器，同步到新位置
        self.resume_tracker(sync_to_current=True)
        logger.info("Robot reset to home position")
    
    def clear_errors(self) -> None:
        """清除机器人错误状态并恢复"""
        try:
            # 使用类似 Fr3Arm.recover() 的智能恢复机制
            # 首先检查是否真的有错误
            if hasattr(self._robot, '_fr3_robot'):
                try:
                    # 如果没有错误，raise_error() 会正常返回
                    self._robot._fr3_robot.raise_error()
                    # 没有错误，不需要恢复
                    return
                except:
                    # 有错误，进行恢复
                    logger.info("Robot error detected, recovering...")
                    self._robot._fr3_robot.recover()
                    # 重新初始化控制器
                    self._robot._reinitialize_controller()
                    logger.info("Robot errors cleared and controller reinitialized")
            else:
                # 如果没有 _fr3_robot 属性，降级到简单的重新初始化
                # 但应该避免频繁调用
                logger.warning("Using fallback error recovery")
                self._robot._reinitialize_controller()
        except Exception as e:
            logger.error(f"Failed to clear errors: {e}")
            raise
    
    def update_params(self, params: Any) -> None:
        """
        更新机器人控制参数
        
        参数:
            params: 控制参数字典或ComplianceParams对象
        """
        # 如果是ComplianceParams对象，转换为字典
        if isinstance(params, ComplianceParams):
            params_dict = {
                'translational_stiffness': params.translational_stiffness,
                'rotational_stiffness': params.rotational_stiffness,
                'translational_damping': params.translational_damping,
                'rotational_damping': params.rotational_damping
            }
            params = params_dict
        
        # 检查是否是柔顺控制参数
        if all(key in params for key in ['translational_stiffness', 'rotational_stiffness']):
            # 更新阻抗控制参数
            impedance_params = ImpedanceParams(
                translational_stiffness=params.get('translational_stiffness', 2000.0),
                rotational_stiffness=params.get('rotational_stiffness', 250.0),
                translational_damping=params.get('translational_damping', 89.0),
                rotational_damping=params.get('rotational_damping', 9.0)
            )
            self._robot.set_impedance_params(impedance_params)
            self._compliance_params = params
            logger.debug(f"Updated impedance parameters: {params}")
        
        # 其他参数更新
        for key, value in params.items():
            if key not in ['translational_stiffness', 'rotational_stiffness', 
                          'translational_damping', 'rotational_damping']:
                logger.debug(f"Parameter {key}={value} not directly supported, skipping")
    
    def set_load(self, load_params: LoadParams) -> None:
        """
        设置末端负载参数
        
        参数:
            load_params: LoadParams实例
        """
        # FR3接口暂时不直接支持负载设置，记录参数供将来使用
        self._load_params = load_params
        logger.info(f"Load parameters stored: mass={load_params.mass}kg, "
                   f"center={load_params.F_x_center_load}")
        
        # TODO: 如果FR3底层支持，可以在这里调用相应的API
        # self._robot._fr3_robot.set_load(...)
    
    def is_ready(self) -> bool:
        """检查机器人是否就绪"""
        try:
            # 检查是否能获取状态
            state = self._robot._fr3_arm._fr3_state
            if state is None:
                return False
            
            # 检查是否在错误状态
            # FR3的recover()方法会检查错误状态
            # 如果返回False说明没有错误
            if hasattr(self._robot._fr3_arm, 'recover'):
                in_error = self._robot._fr3_arm.recover()
                if in_error:
                    logger.warning("Robot in error state and recovered")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Robot not ready: {e}")
            return False
    
    def emergency_stop(self) -> None:
        """紧急停止"""
        try:
            # 停止所有运动
            self._robot.stop()
            
            # 如果可能，调用底层的停止方法
            if hasattr(self._robot._fr3_robot, 'stop_controller'):
                self._robot._fr3_robot.stop_controller()
            
            logger.warning("Emergency stop activated")
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
            raise
    
    def _compute_jacobian(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        计算雅可比矩阵
        
        参数:
            joint_positions: 关节角度
        
        返回:
            np.ndarray(6,7): 雅可比矩阵
        """
        try:
            # 使用robot model计算雅可比
            jacobian = self._robot_model.get_jacobian('fr3_hand_tcp', joint_positions)
            
            # 确保维度正确 (6x7)
            if jacobian.shape != (6, 7):
                # 如果形状不对，使用缓存值
                logger.warning(f"Jacobian shape mismatch: {jacobian.shape}, using cached")
                return self._last_jacobian
            
            self._last_jacobian = jacobian
            return jacobian
        except Exception as e:
            logger.warning(f"Failed to compute Jacobian: {e}, using cached")
            return self._last_jacobian
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        # 停止跟踪器线程
        self.stop_tracker()
        # 关闭机器人连接
        self._robot.close()
        return False
    
    # === 额外的便利方法 ===
    
    def move_to_pose_impedance(self, pose: np.ndarray, 
                               compliance_params: Optional[ComplianceParams] = None,
                               mode: str = "servo",
                               duration: float = 2.0) -> None:
        """
        使用阻抗控制移动到目标位姿
        
        参数:
            pose: 目标位姿 [x,y,z,qx,qy,qz,qw]
            compliance_params: 柔顺参数
            mode: 控制模式 "servo" (直接) 或 "trajectory" (平滑)
            duration: 持续时间(servo模式)或完成时间(trajectory模式)，默认2.0秒
        """
        # 转换位姿
        target_pose = Pose(
            x=float(pose[0]), y=float(pose[1]), z=float(pose[2]),
            qx=float(pose[3]), qy=float(pose[4]), 
            qz=float(pose[5]), qw=float(pose[6])
        )
        
        # 设置阻抗参数
        if compliance_params:
            impedance_params = ImpedanceParams(
                translational_stiffness=compliance_params.translational_stiffness,
                rotational_stiffness=compliance_params.rotational_stiffness,
                translational_damping=compliance_params.translational_damping,
                rotational_damping=compliance_params.rotational_damping
            )
        else:
            impedance_params = ImpedanceParams()  # 使用默认值
        
        # 执行阻抗控制
        if mode == "servo":
            self._robot.move_with_impedance(
                target_pose, 
                mode="servo",
                params=impedance_params,
                duration=duration
            )
        else:  # trajectory mode
            self._robot.move_with_impedance(
                target_pose, 
                mode="trajectory",
                params=impedance_params,
                config=TrajectoryConfig(finish_time=duration)
            )
    
    def get_tcp_wrench(self) -> np.ndarray:
        """
        获取TCP处的力/力矩
        
        返回:
            np.ndarray(6,): [fx, fy, fz, tx, ty, tz]
        """
        ft = self._robot.get_tcp_force_torque(frame="base")
        return np.concatenate([ft['force'], ft['torque']])
    
    def is_in_contact(self, threshold: float = 5.0) -> bool:
        """
        检查是否接触
        
        参数:
            threshold: 力阈值(N)
        
        返回:
            bool: 是否检测到接触
        """
        return self._robot.is_in_contact(force_threshold=threshold)
    
    def _start_tracker_thread(self) -> None:
        """启动常驻的二阶跟踪器线程"""
        # 初始化状态
        self._current_joints = self._robot.get_joint_positions()
        self._joint_velocity = np.zeros(7)
        self._target_joints = self._current_joints.copy()
        
        # 创建并启动线程
        self._tracker_running = True
        self._tracker_thread = threading.Thread(target=self._tracker_loop)
        self._tracker_thread.daemon = True
        self._tracker_thread.start()
        logger.info("Started critical damped tracker thread at 800Hz")
    
    def _tracker_loop(self) -> None:
        """常驻线程的主循环 - 二阶临界阻尼跟踪器"""
        control_freq = 800.0  # Hz
        dt = 1.0 / control_freq
        
        # 临界阻尼 ζ = 1
        zeta = 1.0
        
        while self._tracker_running:
            loop_start = time.perf_counter()
            
            # 线程安全地检查暂停状态
            with self._tracker_lock:
                is_paused = self._pause_tracker
                target_joints = self._target_joints
                current_joints = self._current_joints
                joint_velocity = self._joint_velocity
            
            # 如果暂停，跳过本次循环
            if is_paused:
                time.sleep(dt)
                continue
            
            if target_joints is not None and current_joints is not None:
                # 二阶系统动力学
                # ẍ = ωn²(x_target - x) - 2ζωn·ẋ
                error = target_joints - current_joints
                acceleration = self._omega_n**2 * error - 2 * zeta * self._omega_n * joint_velocity
                
                # 欧拉积分
                joint_velocity = joint_velocity + acceleration * dt
                current_joints = current_joints + joint_velocity * dt
                
                # 更新内部状态（线程安全）
                with self._tracker_lock:
                    self._joint_velocity = joint_velocity
                    self._current_joints = current_joints
                
                # 发送命令到机器人
                try:
                    self._robot._fr3_arm.set_joint_command("position", current_joints)
                except Exception as e:
                    logger.warning(f"Failed to send joint command: {e}")
            
            # 保持控制频率
            loop_time = time.perf_counter() - loop_start
            if loop_time < dt:
                time.sleep(dt - loop_time)
            elif loop_time > 2 * dt:
                logger.warning(f"Tracker loop running slow: {loop_time*1000:.1f}ms")
    
    def stop_tracker(self) -> None:
        """停止跟踪器线程"""
        self._tracker_running = False
        if self._tracker_thread and self._tracker_thread.is_alive():
            self._tracker_thread.join(timeout=1.0)
        logger.info("Stopped tracker thread")
    
    def set_tracker_frequency(self, omega_n: float) -> None:
        """
        设置跟踪器的自然频率
        
        参数:
            omega_n: 自然频率 (rad/s)
                    - 15-20: 柔顺，响应慢
                    - 20-30: 平衡（默认25）
                    - 30-40: 响应快，较硬
        """
        self._omega_n = np.clip(omega_n, 10.0, 50.0)
        t_settle = 4.6 / self._omega_n
        logger.info(f"Tracker frequency set to {omega_n:.1f} rad/s (95% settling time: {t_settle:.2f}s)")
    
    def __del__(self):
        """清理资源"""
        try:
            self.stop_tracker()
        except:
            pass  # 避免在析构时抛出异常


# === 测试代码 ===
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 创建接口
    with FrankaInterface() as robot:
        print("Testing FrankaInterface...")
        
        # 测试获取状态
        state = robot.get_state()
        print(f"Current pose: {state['pose']}")
        print(f"Joint positions: {state['q']}")
        print(f"Forces: {state['force']}")
        print(f"Gripper position: {state['gripper_pos']}")
        
        # 测试就绪状态
        if robot.is_ready():
            print("Robot is ready")
        
        # 测试夹爪
        print("\nTesting gripper...")
        robot.open_gripper()
        time.sleep(2)
        robot.close_gripper()
        time.sleep(2)
        
        # 测试运动
        print("\nTesting motion...")
        current_pose = state['pose'].copy()
        target_pose = current_pose.copy()
        target_pose[0] += 0.05  # 移动5cm
        
        robot.send_pos_command(target_pose)
        time.sleep(3)
        
        # 测试关节重置
        print("\nResetting to home...")
        robot.joint_reset()
        
        print("Test completed!")