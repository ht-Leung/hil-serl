#!/usr/bin/env python3
"""
Dynamic ROI wrapper for HIROLPickPlaceEnv
Enables real-time adaptive cropping during data collection and inference
"""

import numpy as np
from typing import Dict, Tuple
import sys
import os

# Add path for dynamic_roi module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from serl_hirol_infra.hirol_env.dynamic_roi import DynamicROITracker, ROIConfig
from experiments.hirol_pick_place.wrapper import HIROLPickPlaceEnv


class HIROLPickPlaceEnvWithDynamicROI(HIROLPickPlaceEnv):
    """
    Extended HIROLPickPlaceEnv with dynamic ROI support
    Real-time adaptive cropping based on robot state and scene content
    """
    
    def __init__(self, **kwargs):
        # Initialize base environment
        super().__init__(**kwargs)
        
        # Configure dynamic ROI
        roi_config = ROIConfig(
            smoothing_factor=0.15,  # Smooth transitions
            min_roi_size=(256, 256),  # Minimum ROI to maintain context
            max_roi_size=(640, 480),  # Full frame maximum
            padding=60,  # Padding around detected areas
            
            # Enable different detection methods
            use_robot_state=True,  # Primary: track robot position
            use_motion_detection=False,  # Optional: track motion
            use_object_detection=False,  # Optional: use FastSAM
            
            # Camera-specific settings
            camera_configs={
                "wrist_1": {
                    "fixed": False,  # Dynamic for wrist too
                    "track_gripper": False,  # Already at gripper
                    "focus_distance": None
                },
                "front": {
                    "fixed": False,
                    "track_gripper": True,
                    "focus_distance": (0.3, 0.8)  # Focus 30-80cm
                },
                "side": {
                    "fixed": False,
                    "track_gripper": True,
                    "focus_distance": (0.3, 0.8)
                }
            }
        )
        
        # Initialize ROI tracker
        self.roi_tracker = DynamicROITracker(roi_config)
        
        # Store original crop functions (if any)
        self.original_crop_functions = {}
        if hasattr(self.config, 'IMAGE_CROP'):
            self.original_crop_functions = self.config.IMAGE_CROP.copy()
        
        print("✓ Dynamic ROI initialized for HIROLPickPlaceEnv")
    
    def get_im(self) -> Dict[str, np.ndarray]:
        """
        Override get_im to use dynamic ROI cropping
        """
        # Update ROI tracker with current robot state
        if hasattr(self, 'currpos') and self.currpos is not None:
            # Get end-effector position (first 3 elements)
            ee_pos = self.currpos[:3]
            
            # Get gripper state
            gripper_state = self.gripper_state if hasattr(self, 'gripper_state') else 0.5
            
            # Update tracker
            self.roi_tracker.update_robot_state(ee_pos, gripper_state)
        
        # Process images with dynamic ROI
        images = {}
        display_images = {}
        full_res_images = {}
        
        for key, cap in self.cap.items():
            try:
                # Get raw image
                rgb = cap.read()
                
                # Apply dynamic ROI cropping
                x1, y1, x2, y2 = self.roi_tracker.get_dynamic_roi(rgb, key)
                cropped_rgb = rgb[y1:y2, x1:x2]
                
                # Fallback to full image if crop is too small
                if cropped_rgb.size == 0 or cropped_rgb.shape[0] < 50 or cropped_rgb.shape[1] < 50:
                    cropped_rgb = rgb
                    print(f"⚠ Dynamic ROI too small for {key}, using full frame")
                
                # Resize to observation space size
                resized = cv2.resize(
                    cropped_rgb, 
                    self.observation_space["images"][key].shape[:2][::-1]
                )
                
                # Store processed images
                images[key] = resized[..., ::-1]  # BGR to RGB
                display_images[key] = resized
                display_images[key + "_full"] = cropped_rgb
                full_res_images[key] = cropped_rgb.copy()
                
                # Optional: Visualize ROI boundaries on full image
                if self.display_image:
                    display_with_roi = rgb.copy()
                    cv2.rectangle(display_with_roi, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_with_roi, f"Dynamic ROI", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    display_images[key + "_roi_viz"] = display_with_roi
                    
            except Exception as e:
                print(f"Error processing {key}: {e}")
                # Fallback to parent implementation
                return super().get_im()
        
        # Store for video recording if enabled
        if self.save_video:
            self.recording_frames.append(full_res_images)
        
        # Update display queue
        if self.display_image:
            self.img_queue.put(display_images)
        
        return images
    
    def reset(self, **kwargs):
        """
        Override reset to initialize ROI tracker
        """
        obs, info = super().reset(**kwargs)
        
        # Reset ROI tracker state
        self.roi_tracker.current_rois = {}
        self.roi_tracker.motion_history = {}
        
        # Initialize with current robot state
        if hasattr(self, 'currpos') and self.currpos is not None:
            self.roi_tracker.update_robot_state(
                self.currpos[:3],
                self.gripper_state if hasattr(self, 'gripper_state') else 0.5
            )
        
        return obs, info
    
    def enable_static_crop(self):
        """Switch back to static cropping"""
        if self.original_crop_functions:
            self.config.IMAGE_CROP = self.original_crop_functions.copy()
            print("Switched to static cropping")
    
    def enable_dynamic_crop(self):
        """Switch to dynamic cropping"""
        self.config.IMAGE_CROP = {}
        print("Switched to dynamic cropping")
    
    def set_roi_method(self, use_robot_state=True, use_motion=False, use_objects=False):
        """
        Configure which ROI detection methods to use
        
        Args:
            use_robot_state: Track robot end-effector
            use_motion: Track motion areas
            use_objects: Use object detection
        """
        self.roi_tracker.config.use_robot_state = use_robot_state
        self.roi_tracker.config.use_motion_detection = use_motion
        self.roi_tracker.config.use_object_detection = use_objects
        
        print(f"ROI methods - Robot:{use_robot_state}, Motion:{use_motion}, Objects:{use_objects}")


def create_dynamic_roi_env(**kwargs):
    """
    Factory function to create environment with dynamic ROI
    """
    return HIROLPickPlaceEnvWithDynamicROI(**kwargs)


# Integration with existing config
def get_environment_with_dynamic_roi(fake_env=False, save_video=False, classifier=True):
    """
    Create environment with dynamic ROI support
    Compatible with existing training scripts
    """
    from experiments.hirol_pick_place.config import TrainConfig
    
    config = TrainConfig()
    
    # Create environment with dynamic ROI
    env = HIROLPickPlaceEnvWithDynamicROI(
        config=config.env_config,
        fake_env=fake_env,
        save_video=save_video,
        proprio_keys=config.proprio_keys,
    )
    
    # Add classifier wrapper if needed
    if classifier:
        from hil_env.wrappers import SpacemouseIntervention, RobotStateDiffWrapper, ReachClassifier
        
        env = RobotStateDiffWrapper(env)
        env = SpacemouseIntervention(env, gripper_enabled=True, action_sensitivity=3.0)
        env = ReachClassifier(
            env,
            model_path=config.classifier_ckpt_path,
            image_keys=config.classifier_keys,
        )
    
    return env


if __name__ == "__main__":
    import cv2
    
    # Test dynamic ROI environment
    print("Testing Dynamic ROI Environment...")
    
    # Create environment
    env = get_environment_with_dynamic_roi(fake_env=False, save_video=False, classifier=False)
    
    # Reset environment
    obs, info = env.reset()
    print(f"Environment reset. Observation keys: {obs.keys()}")
    
    # Test different ROI methods
    print("\nTesting different ROI methods:")
    
    # Test robot state tracking
    env.set_roi_method(use_robot_state=True, use_motion=False, use_objects=False)
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i}: ROI tracking robot position")
    
    # Test motion detection
    env.set_roi_method(use_robot_state=False, use_motion=True, use_objects=False)
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i}: ROI tracking motion")
    
    # Combined tracking
    env.set_roi_method(use_robot_state=True, use_motion=True, use_objects=False)
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i}: ROI tracking robot + motion")
    
    print("\n✓ Dynamic ROI test completed")
    
    # Cleanup
    env.close()