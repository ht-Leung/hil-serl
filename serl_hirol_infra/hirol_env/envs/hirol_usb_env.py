"""Example of HIROLEnv subclass matching USBEnv interface"""
from collections import OrderedDict
import numpy as np
import copy
import time
from .hirol_env import HIROLEnv


class HIROLUSBEnv(HIROLEnv):
    """HIROL version of USBEnv demonstrating subclass compatibility"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def init_cameras(self, name_serial_dict=None):
        """Init cameras with special handling for side_classifier"""
        if self.cap is not None:  # close cameras if they are already open
            self.close_cameras()

        self.cap = OrderedDict()
        for cam_name, kwargs in name_serial_dict.items():
            if cam_name == "side_classifier":
                # Share the same camera instance as side_policy
                self.cap["side_classifier"] = self.cap["side_policy"]
            else:
                from franka_env.camera.video_capture import VideoCapture
                from franka_env.camera.rs_capture import RSCapture
                cap = VideoCapture(
                    RSCapture(name=cam_name, **kwargs)
                )
                self.cap[cam_name] = cap

    def reset(self, **kwargs):
        """Custom reset procedure for USB insertion task"""
        self._recover()
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.1)
        
        # Update to precision parameters
        if hasattr(self.config, 'PRECISION_PARAM') and self.config.PRECISION_PARAM:
            self.interface.update_params(self.config.PRECISION_PARAM)
        
        # Open gripper
        self._send_gripper_command(1.0)
        
        # Move above the target pose
        target = copy.deepcopy(self.currpos)
        target[2] = self.config.TARGET_POSE[2] + 0.05
        self.interpolate_move(target, timeout=0.5)
        time.sleep(0.5)
        
        # Move to target pose
        self.interpolate_move(self.config.TARGET_POSE, timeout=0.5)
        time.sleep(0.5)
        
        # Close gripper
        self._send_gripper_command(-1.0)

        self._update_currpos()
        reset_pose = copy.deepcopy(self.config.TARGET_POSE)
        reset_pose[1] += 0.04
        self.interpolate_move(reset_pose, timeout=0.5)
        
        # Change to compliance mode
        if hasattr(self.config, 'COMPLIANCE_PARAM') and self.config.COMPLIANCE_PARAM:
            self.interface.update_params(self.config.COMPLIANCE_PARAM)
        
        self.interpolate_move(self.config.RESET_POSE, timeout=1.0)
        
        self.curr_path_length = 0
        self._update_currpos()
        obs = self._get_obs()
        self.terminate = False
        return obs, {"succeed": False}