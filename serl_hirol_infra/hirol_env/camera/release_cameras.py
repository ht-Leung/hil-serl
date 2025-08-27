#!/usr/bin/env python3
"""
Utility script to release all RealSense cameras
Run this when cameras are stuck in busy state
"""

import pyrealsense2 as rs
import time

def release_all_cameras():
    """Force release all RealSense cameras"""
    print("Releasing all RealSense cameras...")
    
    # Get context and devices
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("No RealSense devices found")
        return
    
    print(f"Found {len(devices)} RealSense device(s)")
    
    # Try to stop any active pipelines
    for i, dev in enumerate(devices):
        serial = dev.get_info(rs.camera_info.serial_number)
        print(f"  Device {i}: {serial}")
        
        # Create a pipeline for this device
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        
        try:
            # Try to start and immediately stop to reset the device
            print(f"    Attempting to reset device {serial}...")
            pipe.start(cfg)
            time.sleep(0.1)
            pipe.stop()
            print(f"    Successfully reset device {serial}")
        except Exception as e:
            print(f"    Device {serial} may be in use or already free: {e}")
        
        # Additional cleanup
        try:
            cfg.disable_all_streams()
        except:
            pass
    
    print("\nCamera release complete!")
    print("You may need to wait a few seconds before using the cameras again.")

def check_camera_status():
    """Check status of all RealSense cameras"""
    ctx = rs.context()
    devices = ctx.query_devices()
    
    print("\nCamera Status:")
    print("-" * 40)
    
    if len(devices) == 0:
        print("No RealSense devices found")
        return []
    
    serials = []
    for i, dev in enumerate(devices):
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        serials.append(serial)
        print(f"Device {i}: {name}")
        print(f"  Serial: {serial}")
        
        # Try to check if it's available
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        try:
            pipe.start(cfg)
            pipe.stop()
            print(f"  Status: Available ✓")
        except Exception as e:
            if "busy" in str(e).lower():
                print(f"  Status: Busy ✗")
            else:
                print(f"  Status: Error - {e}")
    
    print("-" * 40)
    return serials

if __name__ == "__main__":
    import sys
    
    print("RealSense Camera Release Tool")
    print("=" * 40)
    
    # Check current status
    serials = check_camera_status()
    
    if len(serials) > 0:
        # Check if running in auto mode
        auto_mode = '--auto' in sys.argv or '-a' in sys.argv
        
        if auto_mode:
            print("\nAuto mode: Releasing all cameras...")
            release_all_cameras()
        else:
            # Ask user if they want to release
            try:
                print("\nDo you want to force release all cameras? (y/n): ", end="")
                response = input().strip().lower()
                
                if response == 'y':
                    release_all_cameras()
                else:
                    print("Skipping camera release.")
            except (EOFError, KeyboardInterrupt):
                print("\nSkipping camera release.")
        
        # Check status again
        print("\nChecking camera status after operation...")
        time.sleep(2)
        check_camera_status()
    
    print("\nDone!")