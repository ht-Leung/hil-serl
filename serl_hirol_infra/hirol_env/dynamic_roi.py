#!/usr/bin/env python3
"""
Dynamic ROI (Region of Interest) system for real-time adaptive cropping
during data collection and inference
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import deque
import threading
import time


@dataclass
class ROIConfig:
    """Configuration for dynamic ROI behavior"""
    # ROI tracking parameters
    smoothing_factor: float = 0.2  # Lower = smoother transitions
    min_roi_size: Tuple[int, int] = (200, 200)  # Minimum ROI size
    max_roi_size: Tuple[int, int] = (640, 480)  # Maximum ROI size
    padding: int = 50  # Padding around detected objects
    
    # Detection methods
    use_robot_state: bool = True  # Track robot end-effector
    use_motion_detection: bool = False  # Track motion areas
    use_object_detection: bool = False  # Use FastSAM or YOLO
    use_depth_filtering: bool = False  # Filter by depth range
    
    # Camera-specific settings
    camera_configs: Dict = None
    
    def __post_init__(self):
        if self.camera_configs is None:
            self.camera_configs = {
                "wrist_1": {
                    "fixed": False,  # Wrist camera usually keeps full view
                    "track_gripper": False,
                    "focus_distance": None
                },
                "front": {
                    "fixed": False,
                    "track_gripper": True,
                    "focus_distance": (0.3, 1.0)  # Focus 30cm-1m range
                },
                "side": {
                    "fixed": False, 
                    "track_gripper": True,
                    "focus_distance": (0.3, 1.0)
                }
            }


class DynamicROITracker:
    """
    Real-time dynamic ROI tracker that adapts cropping based on:
    - Robot end-effector position
    - Object detection
    - Motion detection
    - Task context
    """
    
    def __init__(self, config: ROIConfig = None):
        self.config = config or ROIConfig()
        
        # ROI state for each camera
        self.current_rois = {}
        self.roi_history = {}
        self.motion_history = {}
        
        # Robot state cache
        self.robot_ee_pos = None
        self.gripper_state = None
        
        # Optional: Object detector
        self.object_detector = None
        if self.config.use_object_detection:
            self._init_object_detector()
        
        # Thread safety
        self.lock = threading.Lock()
    
    def _init_object_detector(self):
        """Initialize object detection model (FastSAM/YOLO)"""
        try:
            # Try FastSAM first
            from ultralytics import FastSAM
            self.object_detector = FastSAM('FastSAM-s.pt')
            print("✓ FastSAM initialized for dynamic ROI")
        except ImportError:
            print("⚠ FastSAM not available. Install with: pip install ultralytics")
            self.config.use_object_detection = False
        except Exception as e:
            print(f"⚠ Failed to load object detector: {e}")
            self.config.use_object_detection = False
    
    def update_robot_state(self, ee_pos: np.ndarray, gripper_state: float):
        """Update cached robot state for ROI calculation"""
        with self.lock:
            self.robot_ee_pos = ee_pos.copy() if ee_pos is not None else None
            self.gripper_state = gripper_state
    
    def project_ee_to_image(self, ee_pos: np.ndarray, camera_name: str, 
                           camera_matrix: Optional[np.ndarray] = None) -> Tuple[int, int]:
        """
        Project robot end-effector position to image coordinates
        
        Args:
            ee_pos: 3D position of end-effector
            camera_name: Name of camera
            camera_matrix: Camera intrinsic matrix
        
        Returns:
            (x, y) pixel coordinates
        """
        if camera_matrix is None:
            # Default camera parameters for RealSense
            fx = fy = 600  # Approximate focal length
            cx, cy = 320, 240  # Image center
        else:
            fx = camera_matrix[0, 0]
            fy = camera_matrix[1, 1]
            cx = camera_matrix[0, 2]
            cy = camera_matrix[1, 2]
        
        # Simple projection (assuming camera at origin)
        # In practice, you'd use proper camera extrinsics
        x = int(fx * ee_pos[0] / ee_pos[2] + cx)
        y = int(fy * ee_pos[1] / ee_pos[2] + cy)
        
        # Clamp to image bounds
        x = np.clip(x, 0, 640)
        y = np.clip(y, 0, 480)
        
        return x, y
    
    def detect_workspace_from_robot(self, img: np.ndarray, camera_name: str) -> Tuple[int, int, int, int]:
        """
        Calculate ROI based on robot end-effector position
        
        Returns:
            (x1, y1, x2, y2): ROI bounding box
        """
        if self.robot_ee_pos is None:
            return (0, 0, img.shape[1], img.shape[0])
        
        # Project end-effector to image
        ee_x, ee_y = self.project_ee_to_image(self.robot_ee_pos, camera_name)
        
        # Calculate ROI around end-effector
        cam_config = self.config.camera_configs.get(camera_name, {})
        
        if cam_config.get("fixed", False):
            # Camera uses fixed full view
            return (0, 0, img.shape[1], img.shape[0])
        
        # Dynamic ROI size based on task phase
        if self.gripper_state < 0.5:  # Gripper closed - likely grasping
            roi_width = 250
            roi_height = 250
        else:  # Gripper open - searching/approaching
            roi_width = 350
            roi_height = 350
        
        # Calculate ROI bounds
        x1 = max(0, ee_x - roi_width // 2)
        y1 = max(0, ee_y - roi_height // 2)
        x2 = min(img.shape[1], ee_x + roi_width // 2)
        y2 = min(img.shape[0], ee_y + roi_height // 2)
        
        # Ensure minimum size
        if x2 - x1 < self.config.min_roi_size[0]:
            center_x = (x1 + x2) // 2
            x1 = max(0, center_x - self.config.min_roi_size[0] // 2)
            x2 = min(img.shape[1], center_x + self.config.min_roi_size[0] // 2)
        
        if y2 - y1 < self.config.min_roi_size[1]:
            center_y = (y1 + y2) // 2
            y1 = max(0, center_y - self.config.min_roi_size[1] // 2)
            y2 = min(img.shape[0], center_y + self.config.min_roi_size[1] // 2)
        
        return (x1, y1, x2, y2)
    
    def detect_objects_roi(self, img: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Use object detection to find relevant ROI
        
        Returns:
            (x1, y1, x2, y2): ROI bounding box
        """
        if self.object_detector is None:
            return (0, 0, img.shape[1], img.shape[0])
        
        try:
            # Run detection
            results = self.object_detector(img, imgsz=512, conf=0.4, iou=0.9)
            
            if len(results[0].boxes) > 0:
                # Get all detection boxes
                boxes = results[0].boxes.xyxy.cpu().numpy()
                
                # Calculate union of all boxes
                x1 = int(np.min(boxes[:, 0]))
                y1 = int(np.min(boxes[:, 1]))
                x2 = int(np.max(boxes[:, 2]))
                y2 = int(np.max(boxes[:, 3]))
                
                # Add padding
                x1 = max(0, x1 - self.config.padding)
                y1 = max(0, y1 - self.config.padding)
                x2 = min(img.shape[1], x2 + self.config.padding)
                y2 = min(img.shape[0], y2 + self.config.padding)
                
                return (x1, y1, x2, y2)
        except Exception as e:
            print(f"Object detection failed: {e}")
        
        return (0, 0, img.shape[1], img.shape[0])
    
    def detect_motion_roi(self, img: np.ndarray, camera_name: str) -> Tuple[int, int, int, int]:
        """
        Detect ROI based on motion between frames
        
        Returns:
            (x1, y1, x2, y2): ROI bounding box
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if camera_name not in self.motion_history:
            self.motion_history[camera_name] = deque(maxlen=3)
        
        self.motion_history[camera_name].append(gray)
        
        if len(self.motion_history[camera_name]) < 2:
            return (0, 0, img.shape[1], img.shape[0])
        
        # Calculate motion
        prev_gray = self.motion_history[camera_name][0]
        motion_mask = cv2.absdiff(gray, prev_gray)
        _, motion_mask = cv2.threshold(motion_mask, 25, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get bounding box of all motion
            x1 = img.shape[1]
            y1 = img.shape[0]
            x2 = 0
            y2 = 0
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w * h > 100:  # Filter small noise
                    x1 = min(x1, x)
                    y1 = min(y1, y)
                    x2 = max(x2, x + w)
                    y2 = max(y2, y + h)
            
            if x2 > x1 and y2 > y1:
                # Add padding
                x1 = max(0, x1 - self.config.padding)
                y1 = max(0, y1 - self.config.padding)
                x2 = min(img.shape[1], x2 + self.config.padding)
                y2 = min(img.shape[0], y2 + self.config.padding)
                return (x1, y1, x2, y2)
        
        return (0, 0, img.shape[1], img.shape[0])
    
    def smooth_roi(self, current_roi: Optional[Tuple], target_roi: Tuple, 
                  alpha: Optional[float] = None) -> Tuple:
        """
        Smooth ROI transitions to avoid jitter
        
        Args:
            current_roi: Current ROI
            target_roi: Target ROI
            alpha: Smoothing factor (override config)
        
        Returns:
            Smoothed ROI
        """
        if current_roi is None:
            return target_roi
        
        alpha = alpha or self.config.smoothing_factor
        
        smoothed = []
        for curr, targ in zip(current_roi, target_roi):
            smoothed.append(int(curr * (1 - alpha) + targ * alpha))
        
        return tuple(smoothed)
    
    def get_dynamic_roi(self, img: np.ndarray, camera_name: str) -> Tuple[int, int, int, int]:
        """
        Main method to get dynamic ROI for an image
        
        Args:
            img: Input image
            camera_name: Camera identifier
        
        Returns:
            (x1, y1, x2, y2): Dynamic ROI coordinates
        """
        rois = []
        
        # Collect ROIs from different methods
        if self.config.use_robot_state:
            robot_roi = self.detect_workspace_from_robot(img, camera_name)
            rois.append(robot_roi)
        
        if self.config.use_motion_detection:
            motion_roi = self.detect_motion_roi(img, camera_name)
            rois.append(motion_roi)
        
        if self.config.use_object_detection:
            object_roi = self.detect_objects_roi(img)
            rois.append(object_roi)
        
        if not rois:
            # Default to full image
            target_roi = (0, 0, img.shape[1], img.shape[0])
        else:
            # Combine ROIs (union)
            x1 = min(roi[0] for roi in rois)
            y1 = min(roi[1] for roi in rois)
            x2 = max(roi[2] for roi in rois)
            y2 = max(roi[3] for roi in rois)
            target_roi = (x1, y1, x2, y2)
        
        # Smooth transition
        current_roi = self.current_rois.get(camera_name)
        smoothed_roi = self.smooth_roi(current_roi, target_roi)
        
        # Update cache
        with self.lock:
            self.current_rois[camera_name] = smoothed_roi
        
        return smoothed_roi
    
    def apply_dynamic_crop(self, img: np.ndarray, camera_name: str) -> np.ndarray:
        """
        Apply dynamic cropping to image
        
        Args:
            img: Input image
            camera_name: Camera identifier
        
        Returns:
            Cropped image
        """
        x1, y1, x2, y2 = self.get_dynamic_roi(img, camera_name)
        return img[y1:y2, x1:x2]
    
    def get_crop_function(self, camera_name: str) -> Callable:
        """
        Get a crop function for a specific camera
        
        Returns:
            Function that applies dynamic crop
        """
        def dynamic_crop(img):
            return self.apply_dynamic_crop(img, camera_name)
        return dynamic_crop


class DynamicROIEnvironment:
    """
    Modified environment with dynamic ROI support
    """
    
    def __init__(self, base_env, roi_config: ROIConfig = None):
        self.env = base_env
        self.roi_tracker = DynamicROITracker(roi_config)
        
        # Override IMAGE_CROP with dynamic functions
        self._setup_dynamic_crop()
    
    def _setup_dynamic_crop(self):
        """Replace static crop with dynamic crop functions"""
        if hasattr(self.env.config, 'IMAGE_CROP'):
            self.env.config.IMAGE_CROP = {}
            for cam_name in self.env.config.REALSENSE_CAMERAS.keys():
                self.env.config.IMAGE_CROP[cam_name] = self.roi_tracker.get_crop_function(cam_name)
    
    def update_robot_state(self, ee_pos, gripper_state):
        """Update ROI tracker with current robot state"""
        self.roi_tracker.update_robot_state(ee_pos, gripper_state)
    
    def step(self, action):
        """Override step to update ROI tracker"""
        # Update robot state before getting observations
        if hasattr(self.env, 'currpos'):
            self.update_robot_state(
                self.env.currpos[:3],  # End-effector position
                self.env.gripper_state if hasattr(self.env, 'gripper_state') else 0.5
            )
        
        return self.env.step(action)
    
    def reset(self, **kwargs):
        """Override reset to update ROI tracker"""
        obs, info = self.env.reset(**kwargs)
        
        # Update robot state after reset
        if hasattr(self.env, 'currpos'):
            self.update_robot_state(
                self.env.currpos[:3],
                self.env.gripper_state if hasattr(self.env, 'gripper_state') else 0.5
            )
        
        return obs, info