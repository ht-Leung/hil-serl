#!/usr/bin/env python3
"""Test script to visualize camera views and determine optimal crop parameters"""

import cv2
import numpy as np
import pyrealsense2 as rs
import time
from collections import OrderedDict

# Camera configuration
CAMERAS = {
    "wrist_1": {"serial_number": "332322073603"},
    "front": {"serial_number": "244222075350"},
    "side": {"serial_number": "243122071795"},
}

# Initialize crop parameters (y_start, y_end, x_start, x_end)
CROP_PARAMS = {
    "wrist_1": [0, 480, 0, 640],  # Full image initially
    "front": lambda img: img[0:480, 90:640],
    "side": lambda img: img[40:410, 250:640],
}

# Adjustment step size
STEP_SIZE = 10

class CameraCropTester:
    def __init__(self):
        self.cameras = OrderedDict()
        self.current_camera_idx = 0
        self.camera_names = list(CAMERAS.keys())
        self.setup_cameras()
        
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
                
    def get_frame(self, cam_name):
        """Get a frame from specified camera"""
        if cam_name not in self.cameras:
            return None
        
        pipe = self.cameras[cam_name]
        frames = pipe.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
            
        img = np.asanyarray(color_frame.get_data())
        return img
    
    def apply_crop(self, img, cam_name):
        """Apply crop to image"""
        y1, y2, x1, x2 = CROP_PARAMS[cam_name]
        return img[y1:y2, x1:x2]
    
    def draw_overlay(self, img, cam_name):
        """Draw crop boundaries and info on image"""
        overlay = img.copy()
        h, w = img.shape[:2]
        y1, y2, x1, x2 = CROP_PARAMS[cam_name]
        
        # Draw crop rectangle
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw grid
        cv2.line(overlay, (w//3, 0), (w//3, h), (128, 128, 128), 1)
        cv2.line(overlay, (2*w//3, 0), (2*w//3, h), (128, 128, 128), 1)
        cv2.line(overlay, (0, h//3), (w, h//3), (128, 128, 128), 1)
        cv2.line(overlay, (0, 2*h//3), (w, 2*h//3), (128, 128, 128), 1)
        
        # Add text info
        info = [
            f"Camera: {cam_name}",
            f"Original: 640x480",
            f"Crop: [{y1}:{y2}, {x1}:{x2}]",
            f"Size: {x2-x1}x{y2-y1}",
            "",
            "Controls:",
            "Tab: Switch camera",
            "W/S: Top edge",
            "X/Z: Bottom edge", 
            "A/D: Left edge",
            "Q/E: Right edge",
            "R: Reset crop",
            "P: Print config",
            "ESC: Exit"
        ]
        
        for i, text in enumerate(info):
            cv2.putText(overlay, text, (10, 20 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return overlay
    
    def run(self):
        """Main loop"""
        print("\n=== Camera Crop Configuration Tool ===")
        print("Press Tab to switch between cameras")
        print("Use keyboard to adjust crop boundaries")
        print("Press P to print final configuration")
        print("Press ESC to exit\n")
        
        while True:
            current_cam = self.camera_names[self.current_camera_idx]
            
            # Get frame
            frame = self.get_frame(current_cam)
            if frame is None:
                print(f"No frame from {current_cam}")
                time.sleep(0.1)
                continue
            
            # Show original with overlay
            original_with_overlay = self.draw_overlay(frame, current_cam)
            cv2.imshow(f"Original - {current_cam}", original_with_overlay)
            
            # Show cropped
            cropped = self.apply_crop(frame, current_cam)
            if cropped.size > 0:
                # Resize cropped to fixed size for display
                display_cropped = cv2.resize(cropped, (320, 240))
                cv2.imshow("Cropped View", display_cropped)
            
            # Handle keyboard input
            key = cv2.waitKey(30) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == 9:  # Tab - switch camera
                self.current_camera_idx = (self.current_camera_idx + 1) % len(self.camera_names)
                print(f"Switched to {self.camera_names[self.current_camera_idx]}")
            elif key == ord('w'):  # Move top edge up
                CROP_PARAMS[current_cam][0] = max(0, CROP_PARAMS[current_cam][0] - STEP_SIZE)
            elif key == ord('s'):  # Move top edge down
                CROP_PARAMS[current_cam][0] = min(CROP_PARAMS[current_cam][1] - 10, 
                                                  CROP_PARAMS[current_cam][0] + STEP_SIZE)
            elif key == ord('x'):  # Move bottom edge up
                CROP_PARAMS[current_cam][1] = max(CROP_PARAMS[current_cam][0] + 10,
                                                  CROP_PARAMS[current_cam][1] - STEP_SIZE)
            elif key == ord('z'):  # Move bottom edge down
                CROP_PARAMS[current_cam][1] = min(480, CROP_PARAMS[current_cam][1] + STEP_SIZE)
            elif key == ord('a'):  # Move left edge left
                CROP_PARAMS[current_cam][2] = max(0, CROP_PARAMS[current_cam][2] - STEP_SIZE)
            elif key == ord('d'):  # Move left edge right
                CROP_PARAMS[current_cam][2] = min(CROP_PARAMS[current_cam][3] - 10,
                                                  CROP_PARAMS[current_cam][2] + STEP_SIZE)
            elif key == ord('q'):  # Move right edge left
                CROP_PARAMS[current_cam][3] = max(CROP_PARAMS[current_cam][2] + 10,
                                                  CROP_PARAMS[current_cam][3] - STEP_SIZE)
            elif key == ord('e'):  # Move right edge right
                CROP_PARAMS[current_cam][3] = min(640, CROP_PARAMS[current_cam][3] + STEP_SIZE)
            elif key == ord('r'):  # Reset
                CROP_PARAMS[current_cam] = [0, 480, 0, 640]
                print(f"Reset {current_cam} to full frame")
            elif key == ord('p'):  # Print configuration
                self.print_config()
    
    def print_config(self):
        """Print the configuration in Python format"""
        print("\n" + "="*50)
        print("# Add this to your config.py:")
        print("IMAGE_CROP = {")
        for cam_name, params in CROP_PARAMS.items():
            y1, y2, x1, x2 = params
            if [y1, y2, x1, x2] != [0, 480, 0, 640]:  # Only print if cropped
                print(f'    "{cam_name}": lambda img: img[{y1}:{y2}, {x1}:{x2}],')
        print("}")
        print("="*50 + "\n")
    
    def cleanup(self):
        """Clean up resources"""
        cv2.destroyAllWindows()
        for pipe in self.cameras.values():
            pipe.stop()

if __name__ == "__main__":
    tester = CameraCropTester()
    try:
        tester.run()
    finally:
        tester.cleanup()
        tester.print_config()