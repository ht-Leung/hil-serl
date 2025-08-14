'''
Author: Haotian Liang haotianliang10@gmail.com
Date: 2025-08-04 10:06:50
LastEditors: Haotian Liang haotianliang10@gmail.com
LastEditTime: 2025-08-05 10:19:10
Description: interface for robot communication
'''
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, TypedDict
from dataclasses import dataclass


class RobotState(TypedDict):
    """机器人状态数据结构"""
    pose: np.ndarray        # [x, y, z, qx, qy, qz, qw] - TCP位姿(7,)
    vel: np.ndarray         # [vx, vy, vz, wx, wy, wz] - TCP速度(6,)
    force: np.ndarray       # [fx, fy, fz] - TCP力(3,)
    torque: np.ndarray      # [tx, ty, tz] - TCP力矩(3,)
    gripper_pos: float      # 夹爪开合度 [0-1]
    gripper_is_grasped: bool = False  # 夹爪是否夹住物体
    q: np.ndarray           # 关节角度(7,)
    dq: np.ndarray          # 关节速度(7,)
    jacobian: np.ndarray    # 雅可比矩阵(6,7)


@dataclass
class ComplianceParams:
    """柔顺控制参数"""
    translational_stiffness: float
    translational_damping: float
    rotational_stiffness: float
    rotational_damping: float
    translational_Ki: float = 0.0
    rotational_Ki: float = 0.0
    # 力/力矩限制
    translational_clip_x: float = 0.01
    translational_clip_y: float = 0.01
    translational_clip_z: float = 0.01
    translational_clip_neg_x: float = 0.01
    translational_clip_neg_y: float = 0.01
    translational_clip_neg_z: float = 0.01
    rotational_clip_x: float = 0.03
    rotational_clip_y: float = 0.03
    rotational_clip_z: float = 0.03
    rotational_clip_neg_x: float = 0.03
    rotational_clip_neg_y: float = 0.03
    rotational_clip_neg_z: float = 0.03


@dataclass
class LoadParams:
    """负载参数"""
    mass: float                      # 负载质量(kg)
    F_x_center_load: list[float]     # 负载质心位置[x,y,z]
    load_inertia: list[float]        # 负载惯性张量(9元素)


class RobotInterface(ABC):
    """机器人通信抽象接口"""
    
    @abstractmethod
    def get_state(self) -> RobotState:
        """
        获取机器人当前状态
        
        返回:
            RobotState: 包含以下字段的字典
                - pose: np.ndarray(7,) - TCP位姿 [x,y,z,qx,qy,qz,qw]
                - vel: np.ndarray(6,) - TCP速度 [vx,vy,vz,wx,wy,wz]
                - force: np.ndarray(3,) - TCP力 [fx,fy,fz]
                - torque: np.ndarray(3,) - TCP力矩 [tx,ty,tz]
                - gripper_pos: float - 夹爪开合度 [0-1]
                - q: np.ndarray(7,) - 关节角度
                - dq: np.ndarray(7,) - 关节速度
                - jacobian: np.ndarray(6,7) - 雅可比矩阵
        """
        pass
    
    @abstractmethod
    def send_pos_command(self, pose: np.ndarray) -> None:
        """
        发送位置控制命令
        
        参数:
            pose: np.ndarray(7,) - 目标TCP位姿 [x,y,z,qx,qy,qz,qw]
        """
        pass
    
    @abstractmethod
    def send_gripper_command(self, position: float, mode: str = "binary") -> None:
        """
        控制夹爪
        
        参数:
            position: float - 目标位置
                binary模式: <-0.5 关闭, >0.5 打开
                continuous模式: [-1,1] 映射到夹爪开合范围
            mode: str - 控制模式 ("binary" 或 "continuous")
        """
        pass
    
    @abstractmethod
    def open_gripper(self) -> None:
        """完全打开夹爪"""
        pass
    
    @abstractmethod
    def close_gripper(self) -> None:
        """完全关闭夹爪"""
        pass
    
    @abstractmethod
    def joint_reset(self) -> None:
        """执行关节级别的重置"""
        pass
    
    @abstractmethod
    def clear_errors(self) -> None:
        """清除机器人错误状态并恢复"""
        pass
    
    @abstractmethod
    def update_params(self, params: Dict[str, Any]) -> None:
        """
        更新机器人控制参数
        
        参数:
            params: 控制参数字典，可包含:
                - ComplianceParams的字段
                - 其他控制器特定参数
        """
        pass
    
    @abstractmethod
    def set_load(self, load_params: LoadParams) -> None:
        """
        设置末端负载参数
        
        参数:
            load_params: LoadParams实例
        """
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        """检查机器人是否就绪"""
        pass
    
    @abstractmethod
    def emergency_stop(self) -> None:
        """紧急停止"""
        pass