'''
Author: Haotian Liang haotianliang10@gmail.com
Date: 2025-08-05 09:31:53
LastEditors: Haotian Liang haotianliang10@gmail.com
LastEditTime: 2025-08-06 10:12:49
Description: HIROL interface implementation using RobotFactory and MotionFactory
'''

import numpy as np
import time
import warnings
from typing import Dict, Any, Optional
from scipy.spatial.transform import Rotation as R
from interface.robot_interface import RobotInterface, RobotState, LoadParams

# Import HIROL platform components
import sys
import os
from pathlib import Path

# HIROLRobotPlatform needs to be run from its directory for imports to work
hirol_platform_path = Path(__file__).resolve().parents[3] / "HIROLRobotPlatform"
original_cwd = os.getcwd()

# Temporarily change to HIROLRobotPlatform directory for imports
os.chdir(str(hirol_platform_path))
sys.path.insert(0, str(hirol_platform_path))

try:
    from factory.components.robot_factory import RobotFactory
    from factory.components.motion_factory import MotionFactory
    from hardware.base.utils import convert_homo_2_7D_pose
    import pinocchio as pin
finally:
    os.chdir(original_cwd)


class HIROLInterface(RobotInterface):
    """HIROL implementation of RobotInterface using RobotFactory and MotionFactory"""
    
    # Default gripper configuration
    GRIPPER_OPEN_POS = 0.08  # meters
    GRIPPER_CLOSE_POS = 0.0  # meters
    GRIPPER_OPEN_THRESHOLD = 0.75  # normalized threshold for considering gripper open
    
    def __init__(self, robot_config: Dict[str, Any], motion_config: Dict[str, Any], 
                 gripper_sleep: float = 0.6,
                 gripper_range: tuple = (0.0, 0.08)):
        """
        Initialize HIROL interface
        
        Args:
            robot_config: Configuration dict for RobotFactory
            motion_config: Configuration dict for MotionFactory
            gripper_sleep: Sleep time after gripper command (seconds)
            gripper_range: Tuple of (min, max) gripper positions in meters
        """
        self.robot_config = robot_config
        self.motion_config = motion_config
        
        # Initialize robot and motion factories
        self._robot_factory = RobotFactory(robot_config)
        self._motion_factory = MotionFactory(motion_config, self._robot_factory)
        
        # Create motion components (this also initializes robot system)
        self._motion_factory.create_motion_components()
        
        # Get end effector links
        self._ee_links = self._motion_factory.get_model_end_effector_link_list()
        self._robot_index = ['left', 'right'] if len(self._ee_links) == 2 else ['single']
        
        # Gripper configuration
        self._last_gripper_cmd_time = time.time()
        self._gripper_sleep = gripper_sleep
        self._gripper_range = gripper_range
        self.GRIPPER_CLOSE_POS = gripper_range[0]
        self.GRIPPER_OPEN_POS = gripper_range[1]
        
        # Initialize motion control
        self._motion_factory.update_execute_hardware(True)
        
    def get_state(self) -> RobotState:
        """
        Get current robot state
        
        Returns:
            RobotState dict containing:
                - pose: TCP pose [x,y,z,qx,qy,qz,qw]
                - vel: TCP velocity [vx,vy,vz,wx,wy,wz]
                - force: TCP force [fx,fy,fz]
                - torque: TCP torque [tx,ty,tz]
                - gripper_pos: Gripper position [0-1]
                - q: Joint positions
                - dq: Joint velocities
                - jacobian: Jacobian matrix (6x7)
        """
        # Get joint states
        joint_states = self._robot_factory.get_joint_states()
        
        # Get robot model and key
        key = self._robot_index[0]
        robot_model = self._motion_factory._robot_model
        
        # Get TCP pose
        tcp_pose = self._motion_factory.get_frame_pose(self._ee_links[0], key)
        
        # Get Jacobian matrix
        # Need to update robot model with current joint positions first
        jacobian = robot_model.get_jacobian(
            frame_name=self._ee_links[0],
            joint_position=joint_states._positions[:7],
            reference_frame=pin.LOCAL_WORLD_ALIGNED,
            model_type=key
        )
        
        # Calculate TCP velocity using Jacobian
        tcp_vel = jacobian @ joint_states._velocities[:7]
                
        # Get external force/torque
        # Try to access FR3 specific data if available
        forces = np.zeros(3)
        torques = np.zeros(3)
        
        # Check if we're in hardware mode and robot has FR3 specific state
        if self._robot_factory._use_hardware and hasattr(self._robot_factory, '_robot'):
            if hasattr(self._robot_factory._robot, '_fr3_state') and \
               self._robot_factory._robot._fr3_state is not None and \
               hasattr(self._robot_factory._robot._fr3_state, 'K_F_ext_hat_K'):
                # FR3 provides K_F_ext_hat_K as 6D wrench [fx, fy, fz, tx, ty, tz]
                k_f_ext = self._robot_factory._robot._fr3_state.K_F_ext_hat_K
                forces = np.array(k_f_ext[:3])
                torques = np.array(k_f_ext[3:])
            elif hasattr(self._robot_factory._robot, 'get_external_wrench'):
                # If there's a method to get external wrench
                wrench = self._robot_factory._robot.get_external_wrench()
                forces = wrench[:3]
                torques = wrench[3:]
        
        # Get tool state for gripper position
        tool_states = self._robot_factory._tool.get_tool_state()
        if not isinstance(tool_states, dict):
            tool_states = {key: tool_states}
        
        # Handle case where _position might not be initialized
        gripper_pos = 0.0
        if key in tool_states:
            tool_state = tool_states[key]
            if hasattr(tool_state, '_position'):
                gripper_pos = tool_state._position
            else:
                # Try to get width attribute for compatibility
                gripper_pos = getattr(tool_state, 'width', 0.0)
        
        # Normalize gripper position to [0, 1]
        gripper_range_size = self._gripper_range[1] - self._gripper_range[0]
        gripper_pos_normalized = np.clip(
            (gripper_pos - self._gripper_range[0]) / gripper_range_size, 0.0, 1.0
        )
        
        return RobotState(
            pose=tcp_pose,
            vel=tcp_vel,
            force=forces,
            torque=torques,
            gripper_pos=gripper_pos_normalized,
            q=joint_states._positions[:7],  # First 7 joints for single arm
            dq=joint_states._velocities[:7],
            jacobian=jacobian
        )
    
    def send_pos_command(self, pose: np.ndarray) -> None:
        """
        Send position control command
        
        Args:
            pose: Target TCP pose [x,y,z,qx,qy,qz,qw]
        """
        if len(pose) != 7:
            raise ValueError(f"Expected pose of length 7, got {len(pose)}")
        
        # Update high level command
        # For single arm, just send the 7D pose
        # For dual arm, would need to send 14D (2x7D)
        self._motion_factory.update_high_level_command(pose)
    
    def send_gripper_command(self, position: float, mode: str = "binary") -> None:
        """
        Control gripper
        
        Args:
            position: Target position
                binary mode: <-0.5 close, >0.5 open
                continuous mode: [-1,1] mapped to gripper range
            mode: Control mode ("binary" or "continuous")
        """
        current_time = time.time()
        key = self._robot_index[0]
        
        if mode == "binary":
            # Get current gripper state
            tool_states = self._robot_factory._tool.get_tool_state()
            if not isinstance(tool_states, dict):
                tool_states = {key: tool_states}
            
            # Handle case where _position might not be initialized
            current_gripper_pos = 0.0
            if key in tool_states:
                tool_state = tool_states[key]
                if hasattr(tool_state, '_position'):
                    current_gripper_pos = tool_state._position
                else:
                    current_gripper_pos = getattr(tool_state, 'width', 0.0)
            # Normalize current position for comparison
            gripper_range_size = self._gripper_range[1] - self._gripper_range[0]
            current_gripper_normalized = (current_gripper_pos - self._gripper_range[0]) / gripper_range_size
            
            # Check if enough time has passed since last command
            if current_time - self._last_gripper_cmd_time < self._gripper_sleep:
                return
            
            # Match FrankaEnv thresholds: >GRIPPER_OPEN_THRESHOLD is considered open
            # Close gripper: position <= -0.5 and gripper is currently open
            if position <= -0.5 and current_gripper_normalized > self.GRIPPER_OPEN_THRESHOLD:
                # Send normalized value for FrankaHand
                tool_command = {key: np.array([0.0])}  # 0.0 = fully closed
                self._robot_factory.set_tool_command(tool_command)
                self._last_gripper_cmd_time = current_time
                time.sleep(self._gripper_sleep)
            # Open gripper: position >= 0.5 and gripper is currently closed
            elif position >= 0.5 and current_gripper_normalized < self.GRIPPER_OPEN_THRESHOLD:
                # Send normalized value for FrankaHand
                tool_command = {key: np.array([1.0])}  # 1.0 = fully open
                self._robot_factory.set_tool_command(tool_command)
                self._last_gripper_cmd_time = current_time
                time.sleep(self._gripper_sleep)
                
        elif mode == "continuous":
            # Map [-1, 1] to [0, 1] for FrankaHand
            # FrankaHand expects normalized values [0,1] and will multiply by max_width
            normalized_pos = np.interp(position, [-1, 1], [0, 1])
            tool_command = {key: np.array([normalized_pos])}
            self._robot_factory.set_tool_command(tool_command)
        else:
            raise ValueError(f"Unknown gripper mode: {mode}")
    
    def open_gripper(self) -> None:
        """Fully open gripper"""
        key = self._robot_index[0]
        tool_command = {key: np.array([1.0])}  # 1.0 = fully open for FrankaHand
        self._robot_factory.set_tool_command(tool_command)
        time.sleep(self._gripper_sleep)
    
    def close_gripper(self) -> None:
        """Fully close gripper"""
        key = self._robot_index[0]
        tool_command = {key: np.array([0.0])}  # 0.0 = fully closed for FrankaHand
        self._robot_factory.set_tool_command(tool_command)
        time.sleep(self._gripper_sleep)
    
    def joint_reset(self) -> None:
        """Execute joint-level reset"""
        try:
            # Block motion during reset
            self._motion_factory.update_execute_hardware(False)


            # Move to start position
            self._motion_factory.move_to_start_blocking()
            time.sleep(1.5)

            # Clear trajectory buffer
            self._motion_factory.clear_traj_buffer()
        finally:
            # 确保无论如何都会重新启用
            self._motion_factory.update_execute_hardware(True)
            time.sleep(0.5)

    
    def clear_errors(self) -> None:
        """Clear robot error state and recover"""
        # HIROL platform doesn't have explicit error clearing
        # This is a placeholder for compatibility
        pass
    
    def update_params(self, params: Dict[str, Any]) -> None:
        """
        Update robot control parameters
        
        Args:
            params: Control parameters dict
        """
        # TODO: Implement compliance parameter updates when controller is ready
        # Currently using position control with ik_controller + trajectory
        warnings.warn("Compliance parameter updates not yet implemented - using position control")
        
        # Placeholder for future implementation:
        # if "translational_stiffness" in params:
        #     self._motion_factory._controller.set_stiffness(...)
        # if "translational_damping" in params:
        #     self._motion_factory._controller.set_damping(...)
    
    def set_load(self, load_params: LoadParams) -> None:
        """
        Set end effector load parameters
        
        Args:
            load_params: LoadParams instance
        """
        # TODO: Implement load parameter setting in HIROL platform
        # This would require updating the robot model with load parameters
        warnings.warn("Load parameter setting not yet implemented in HIROL platform")
    
    def is_ready(self) -> bool:
        """Check if robot is ready"""
        # In simulation mode, check if simulation is initialized
        if self._robot_factory._use_simulation:
            return hasattr(self._robot_factory, '_simulation') and \
                   self._robot_factory._simulation is not None
        # In hardware mode, check if robot is initialized
        elif self._robot_factory._use_hardware:
            return hasattr(self._robot_factory, '_robot') and \
                   self._robot_factory._robot is not None and \
                   hasattr(self._robot_factory, '_tool') and \
                   self._robot_factory._tool is not None
        return False
    
    def emergency_stop(self) -> None:
        """Emergency stop"""
        # Disable hardware execution immediately
        self._motion_factory.update_execute_hardware(False)
        
        # Stop all motion threads
        if hasattr(self._motion_factory, '_controller_thread_running'):
            self._motion_factory._controller_thread_running = False
        if hasattr(self._motion_factory, '_traj_thread_running'):
            self._motion_factory._traj_thread_running = False
        
        warnings.warn("Emergency stop activated - motion disabled")
    
    def close(self) -> None:
        """Clean up and close the interface"""
        try:
            self._motion_factory.close()
        except Exception as e:
            warnings.warn(f"Error closing motion factory: {e}")
        
        # Note: robot_factory.close() is already called by motion_factory.close()
        # so we don't need to call it again
    
    # Additional helper methods for FrankaEnv compatibility
    
    def get_current_pose(self) -> np.ndarray:
        """Get current TCP pose [x,y,z,qx,qy,qz,qw]"""
        key = self._robot_index[0]
        return self._motion_factory.get_frame_pose(self._ee_links[0], key)
    
    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions"""
        joint_states = self._robot_factory.get_joint_states()
        return joint_states._positions[:7]
    
    def get_joint_velocities(self) -> np.ndarray:
        """Get current joint velocities"""
        joint_states = self._robot_factory.get_joint_states()
        return joint_states._velocities[:7]
    
    def set_gripper_width(self, width: float) -> None:
        """
        Set gripper to specific width
        
        Args:
            width: Target width in meters
        """
        key = self._robot_index[0]
        width_clamped = np.clip(width, self._gripper_range[0], self._gripper_range[1])
        tool_command = {key: np.array([width_clamped])}
        self._robot_factory.set_tool_command(tool_command)
    
    def get_gripper_width(self) -> float:
        """Get current gripper width in meters"""
        key = self._robot_index[0]
        tool_states = self._robot_factory._tool.get_tool_state()
        if not isinstance(tool_states, dict):
            tool_states = {key: tool_states}
        
        if key in tool_states:
            tool_state = tool_states[key]
            if hasattr(tool_state, '_position'):
                return tool_state._position
            else:
                return getattr(tool_state, 'width', 0.0)
        return 0.0
    
    def move_to_home(self) -> None:
        """Move robot to neutral/home position using blocking motion"""
        self._motion_factory.update_execute_hardware(False)

        
        # Use MotionFactory's move_to_start_blocking
        self._motion_factory.move_to_start_blocking()
        time.sleep(1.0)
        
        # Resume normal operation

        self._motion_factory.update_execute_hardware(True)
    
    def move_to_joint_position(self, joint_positions: np.ndarray, blocking: bool = True) -> None:
        """
        Move to specific joint positions using panda_py's smooth motion
        
        Args:
            joint_positions: Target joint positions (7D array)
            blocking: Whether to wait for motion completion
        """
        if len(joint_positions) != 7:
            raise ValueError(f"Expected 7 joint positions, got {len(joint_positions)}")
        
        # Temporarily disable motion factory control
        self._motion_factory.update_execute_hardware(False)
   
        # Access the underlying FR3 robot if available
        if hasattr(self._robot_factory._robot, '_fr3_robot'):
            # Use panda_py's move_to_joint_position for smooth motion
            fr3_robot = self._robot_factory._robot._fr3_robot
            fr3_robot.move_to_joint_position(joint_positions)
            
            if blocking:
                # Wait for motion to complete
                time.sleep(0.1)
                while True:
                    current_state = self.get_state()
                    joint_error = np.abs(current_state['q'] - joint_positions)
                    if np.all(joint_error < 0.01):  # 0.01 rad tolerance
                        break
                    time.sleep(0.05)
        else:
            # Fallback to position control if not FR3
            warnings.warn("move_to_joint_position only supported for FR3, using position control")
            self._robot_factory._robot.set_joint_command('position', joint_positions)
            if blocking:
                time.sleep(2.0)  # Conservative wait time
        
        # Resume normal operation

        self._motion_factory.update_execute_hardware(True)
    
    def move_to_cartesian_pose(self, pose: np.ndarray, blocking: bool = True) -> None:
        """
        Move to specific cartesian pose with smooth motion
        
        Args:
            pose: Target TCP pose [x,y,z,qx,qy,qz,qw]
            blocking: Whether to wait for motion completion (not used as move_to_pose is blocking)
        """
        if len(pose) != 7:
            raise ValueError(f"Expected pose of length 7, got {len(pose)}")
        
        # Temporarily disable motion factory control
        self._motion_factory.update_execute_hardware(False)

        
        # Access the underlying FR3 robot if available
        if hasattr(self._robot_factory._robot, '_fr3_robot'):
            # Convert pose to 4x4 transformation matrix for panda_py
            # panda_py expects a 4x4 transformation matrix
            pose_matrix = np.eye(4)
            pose_matrix[:3, 3] = pose[:3]  # position
            # Convert quaternion to rotation matrix
            quat = pose[3:]  # qx, qy, qz, qw
            rot = R.from_quat(quat).as_matrix()
            pose_matrix[:3, :3] = rot
            
            # Use panda_py's move_to_pose for smooth motion
            fr3_robot = self._robot_factory._robot._fr3_robot
            try:
                # move_to_pose is a blocking operation by default
                fr3_robot.move_to_pose(pose_matrix)
            except Exception as e:
                warnings.warn(f"move_to_pose failed: {e}, falling back to IK + joint motion")
                # Fallback to IK + joint motion
                current_state = self.get_state()
                # Use the IK controller if available
                if hasattr(self._motion_factory, '_controller'):
                    target = [{self._ee_links[0]: pose}]
                    from hardware.base.utils import RobotJointState
                    joint_state = RobotJointState()
                    joint_state._positions = current_state['q']
                    joint_state._velocities = current_state['dq']
                    joint_state._accelerations = np.zeros(7)
                    joint_state._torques = np.zeros(7)
                    
                    success, joint_target, mode = self._motion_factory._controller.compute_controller(
                        target, joint_state
                    )
                    
                    if success and mode == 'position':
                        # move_to_joint_position is also blocking
                        fr3_robot.move_to_joint_position(joint_target[:7])
                    else:
                        warnings.warn("IK solution not found")
        else:
            warnings.warn("move_to_cartesian_pose only supported for FR3")
            
        # Resume normal operation
        self._motion_factory.update_execute_hardware(True)