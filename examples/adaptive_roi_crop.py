#!/usr/bin/env python3
"""Adaptive ROI detection for camera cropping using rules and Fast SAM"""

import cv2
import numpy as np
import pyrealsense2 as rs
from collections import OrderedDict
from typing import Dict, Tuple, Optional
import time

# Camera configuration
CAMERAS = {
    "wrist_1": {"serial_number": "332322073603"},
    "front": {"serial_number": "244222075350"},
    "side": {"serial_number": "317422071787"},
}

class AdaptiveROICropper:
    """Adaptive Region of Interest cropping with multiple strategies"""
    
    def __init__(self, strategy="rule_based", sam_model_path=None):
        """
        Initialize adaptive ROI cropper
        
        Args:
            strategy: "rule_based", "fast_sam", or "hybrid"
            sam_model_path: Path to Fast SAM model weights
        """
        self.strategy = strategy
        self.cameras = OrderedDict()
        self.roi_cache = {}
        self.setup_cameras()
        
        # Initialize Fast SAM if needed
        self.sam_predictor = None
        if strategy in ["fast_sam", "hybrid"] and sam_model_path:
            self.init_fast_sam(sam_model_path)
    
    def setup_cameras(self):
        """Initialize all cameras"""
        for cam_name, config in CAMERAS.items():
            try:
                pipe = rs.pipeline()
                cfg = rs.config()
                cfg.enable_device(config["serial_number"])
                cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
                pipe.start(cfg)
                self.cameras[cam_name] = pipe
                print(f"✓ Camera {cam_name} initialized")
            except Exception as e:
                print(f"✗ Failed to initialize {cam_name}: {e}")
    
    def init_fast_sam(self, model_path):
        """Initialize Fast SAM model for segmentation"""
        try:
            # Import Fast SAM dependencies
            import torch
            from fastsam import FastSAM, FastSAMPrompt
            
            self.sam_model = FastSAM(model_path)
            print("✓ Fast SAM initialized")
        except ImportError:
            print("⚠ Fast SAM not available. Install with: pip install fastsam")
            self.strategy = "rule_based"
        except Exception as e:
            print(f"⚠ Failed to load Fast SAM: {e}")
            self.strategy = "rule_based"
    
    def detect_robot_workspace_rules(self, img: np.ndarray, cam_name: str) -> Tuple[int, int, int, int]:
        """
        Rule-based detection of robot workspace
        
        Returns:
            (x1, y1, x2, y2): Bounding box of detected workspace
        """
        h, w = img.shape[:2]
        
        # Convert to HSV for color-based detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Strategy 1: Detect robot arm (typically metallic/white)
        # Adjust these thresholds based on your robot's appearance
        lower_robot = np.array([0, 0, 150])  # Light/metallic colors
        upper_robot = np.array([180, 30, 255])
        robot_mask = cv2.inRange(hsv, lower_robot, upper_robot)
        
        # Strategy 2: Detect workspace table (if distinct color)
        # Example for wooden/brown table
        lower_table = np.array([10, 30, 50])
        upper_table = np.array([25, 150, 200])
        table_mask = cv2.inRange(hsv, lower_table, upper_table)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(robot_mask, table_mask)
        
        # Apply morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get bounding box of all contours
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            
            for contour in contours:
                x, y, cont_w, cont_h = cv2.boundingRect(contour)
                # Filter small noise
                if cont_w * cont_h > 100:  # Minimum area threshold
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x + cont_w)
                    y_max = max(y_max, y + cont_h)
            
            # Add padding
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)
            
            # Camera-specific adjustments
            if cam_name == "wrist_1":
                # Wrist camera usually needs full view
                return (0, 0, w, h)
            elif cam_name == "front":
                # Front camera focus on center workspace
                center_bias = 0.1
                x_min = max(x_min, int(w * center_bias))
                x_max = min(x_max, int(w * (1 - center_bias)))
            elif cam_name == "side":
                # Side camera may need to exclude background
                y_min = max(y_min, int(h * 0.1))  # Exclude top 10%
            
            return (x_min, y_min, x_max, y_max)
        
        # Fallback to full image if detection fails
        return (0, 0, w, h)
    
    def detect_robot_workspace_sam(self, img: np.ndarray, cam_name: str) -> Tuple[int, int, int, int]:
        """
        Use Fast SAM for semantic segmentation of workspace
        
        Returns:
            (x1, y1, x2, y2): Bounding box of detected workspace
        """
        if self.sam_model is None:
            return self.detect_robot_workspace_rules(img, cam_name)
        
        try:
            # Run Fast SAM inference
            results = self.sam_model(
                img,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                retina_masks=True,
                imgsz=512,
                conf=0.4,
                iou=0.9
            )
            
            # Create prompt processor
            prompt_process = FastSAMPrompt(img, results, device='cuda' if torch.cuda.is_available() else 'cpu')
            
            # Use text prompt to identify robot and workspace
            # Adjust prompts based on your setup
            robot_mask = prompt_process.text_prompt(text='robot arm gripper')
            workspace_mask = prompt_process.text_prompt(text='table workspace surface')
            
            # Combine masks
            combined_mask = np.logical_or(robot_mask, workspace_mask).astype(np.uint8) * 255
            
            # Find bounding box
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Add padding
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(img.shape[1] - x, w + 2 * padding)
                h = min(img.shape[0] - y, h + 2 * padding)
                
                return (x, y, x + w, y + h)
            
        except Exception as e:
            print(f"SAM detection failed: {e}, falling back to rules")
        
        # Fallback to rule-based
        return self.detect_robot_workspace_rules(img, cam_name)
    
    def detect_motion_based_roi(self, img: np.ndarray, cam_name: str, 
                               prev_img: Optional[np.ndarray] = None) -> Tuple[int, int, int, int]:
        """
        Detect ROI based on motion (useful for dynamic scenes)
        
        Returns:
            (x1, y1, x2, y2): Bounding box of motion area
        """
        if prev_img is None:
            return (0, 0, img.shape[1], img.shape[0])
        
        # Calculate frame difference
        diff = cv2.absdiff(prev_img, img)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Threshold to get motion mask
        _, motion_mask = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours of motion
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get bounding box of all motion
            x_min, y_min = img.shape[1], img.shape[0]
            x_max, y_max = 0, 0
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w * h > 50:  # Filter small noise
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x + w)
                    y_max = max(y_max, y + h)
            
            # Add significant padding for motion areas
            padding = 50
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(img.shape[1], x_max + padding)
            y_max = min(img.shape[0], y_max + padding)
            
            return (x_min, y_min, x_max, y_max)
        
        return (0, 0, img.shape[1], img.shape[0])
    
    def get_adaptive_crop(self, img: np.ndarray, cam_name: str) -> Tuple[int, int, int, int]:
        """
        Get adaptive crop region based on selected strategy
        
        Returns:
            (x1, y1, x2, y2): Crop coordinates
        """
        if self.strategy == "rule_based":
            return self.detect_robot_workspace_rules(img, cam_name)
        elif self.strategy == "fast_sam":
            return self.detect_robot_workspace_sam(img, cam_name)
        elif self.strategy == "hybrid":
            # Combine multiple strategies
            rules_roi = self.detect_robot_workspace_rules(img, cam_name)
            sam_roi = self.detect_robot_workspace_sam(img, cam_name)
            
            # Take union of both ROIs
            x1 = min(rules_roi[0], sam_roi[0])
            y1 = min(rules_roi[1], sam_roi[1])
            x2 = max(rules_roi[2], sam_roi[2])
            y2 = max(rules_roi[3], sam_roi[3])
            
            return (x1, y1, x2, y2)
        else:
            # Default to full image
            return (0, 0, img.shape[1], img.shape[0])
    
    def smooth_roi_transition(self, current_roi: Tuple, target_roi: Tuple, 
                             alpha: float = 0.1) -> Tuple:
        """
        Smooth ROI transitions to avoid jitter
        
        Args:
            current_roi: Current ROI coordinates
            target_roi: Target ROI coordinates
            alpha: Smoothing factor (0-1, lower = smoother)
        
        Returns:
            Smoothed ROI coordinates
        """
        if current_roi is None:
            return target_roi
        
        smoothed = []
        for curr, targ in zip(current_roi, target_roi):
            smoothed.append(int(curr * (1 - alpha) + targ * alpha))
        
        return tuple(smoothed)
    
    def generate_config_code(self, roi_dict: Dict[str, Tuple]) -> str:
        """Generate Python config code for IMAGE_CROP"""
        code = "IMAGE_CROP = {\n"
        for cam_name, (x1, y1, x2, y2) in roi_dict.items():
            if (x1, y1, x2, y2) != (0, 0, 640, 480):  # Only if not full frame
                code += f'    "{cam_name}": lambda img: img[{y1}:{y2}, {x1}:{x2}],\n'
        code += "}"
        return code


class AdaptiveROIEnvironmentWrapper:
    """
    Environment wrapper that applies adaptive ROI cropping
    """
    
    def __init__(self, env, roi_cropper: AdaptiveROICropper):
        self.env = env
        self.roi_cropper = roi_cropper
        self.current_rois = {}
        
    def get_adaptive_crop_function(self, cam_name: str):
        """Create a dynamic crop function for a camera"""
        def adaptive_crop(img):
            # Get adaptive ROI
            target_roi = self.roi_cropper.get_adaptive_crop(img, cam_name)
            
            # Smooth transition
            current_roi = self.current_rois.get(cam_name)
            smoothed_roi = self.roi_cropper.smooth_roi_transition(current_roi, target_roi)
            self.current_rois[cam_name] = smoothed_roi
            
            x1, y1, x2, y2 = smoothed_roi
            return img[y1:y2, x1:x2]
        
        return adaptive_crop
    
    def update_env_crop_config(self):
        """Update environment's IMAGE_CROP with adaptive functions"""
        if hasattr(self.env.config, 'IMAGE_CROP'):
            for cam_name in self.env.config.REALSENSE_CAMERAS.keys():
                self.env.config.IMAGE_CROP[cam_name] = self.get_adaptive_crop_function(cam_name)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Adaptive ROI Detection for Robot Cameras")
    parser.add_argument("--strategy", choices=["rule_based", "fast_sam", "hybrid"], 
                       default="rule_based", help="ROI detection strategy")
    parser.add_argument("--sam_model", type=str, default=None, 
                       help="Path to Fast SAM model weights")
    args = parser.parse_args()
    
    # Initialize cropper
    cropper = AdaptiveROICropper(strategy=args.strategy, sam_model_path=args.sam_model)
    
    print(f"\n=== Adaptive ROI Detection ===")
    print(f"Strategy: {args.strategy}")
    print("Press 'r' for rule-based detection")
    print("Press 's' for SAM detection (if available)")
    print("Press 'm' for motion-based detection")
    print("Press 'p' to print configuration")
    print("Press ESC to exit\n")
    
    roi_dict = {}
    prev_frames = {}
    
    try:
        while True:
            for cam_name, pipe in cropper.cameras.items():
                # Get frame
                frames = pipe.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                    
                img = np.asanyarray(color_frame.get_data())
                
                # Get adaptive ROI
                x1, y1, x2, y2 = cropper.get_adaptive_crop(img, cam_name)
                roi_dict[cam_name] = (x1, y1, x2, y2)
                
                # Draw ROI
                display = img.copy()
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display, f"{cam_name} ROI: [{x1},{y1}] to [{x2},{y2}]", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show cropped
                cropped = img[y1:y2, x1:x2]
                if cropped.size > 0:
                    # Resize to 128x128 (final size used in training)
                    final = cv2.resize(cropped, (128, 128))
                    cv2.imshow(f"{cam_name} - Final (128x128)", final)
                
                cv2.imshow(f"{cam_name} - Original with ROI", display)
                prev_frames[cam_name] = img
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord('r'):
                cropper.strategy = "rule_based"
                print("Switched to rule-based detection")
            elif key == ord('s'):
                if cropper.sam_model is not None:
                    cropper.strategy = "fast_sam"
                    print("Switched to SAM detection")
                else:
                    print("SAM not available")
            elif key == ord('m'):
                print("Motion detection mode")
            elif key == ord('p'):
                print("\n" + "="*50)
                print(cropper.generate_config_code(roi_dict))
                print("="*50 + "\n")
    
    finally:
        cv2.destroyAllWindows()
        for pipe in cropper.cameras.values():
            pipe.stop()