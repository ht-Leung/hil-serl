#!/usr/bin/env python3
"""List all connected RealSense cameras and their serial numbers"""

import pyrealsense2 as rs

def list_realsense_cameras():
    """List all connected RealSense cameras"""
    context = rs.context()
    devices = context.devices
    
    print("=" * 60)
    print("Connected RealSense Cameras")
    print("=" * 60)
    
    if len(devices) == 0:
        print("No RealSense cameras detected!")
        return []
    
    camera_list = []
    for i, device in enumerate(devices):
        serial = device.get_info(rs.camera_info.serial_number)
        name = device.get_info(rs.camera_info.name)
        firmware = device.get_info(rs.camera_info.firmware_version)
        usb_type = device.get_info(rs.camera_info.usb_type_descriptor)
        
        print(f"\nCamera {i+1}:")
        print(f"  Serial Number: {serial}")
        print(f"  Model: {name}")
        print(f"  Firmware: {firmware}")
        print(f"  USB Type: {usb_type}")
        
        camera_list.append(serial)
    
    print("\n" + "=" * 60)
    print("Currently configured cameras in config.py:")
    print("  wrist_1: 332322073603")
    print("  front: 244222075350")
    print("  side: 243122071795")
    
    # Find unconfigured cameras
    configured = ["332322073603", "244222075350", "243122071795"]
    unconfigured = [s for s in camera_list if s not in configured]
    
    if unconfigured:
        print("\n" + "=" * 60)
        print("NEW/UNCONFIGURED cameras found:")
        for serial in unconfigured:
            print(f"  - {serial} (This could be your new camera with short USB cable)")
    
    return camera_list

if __name__ == "__main__":
    cameras = list_realsense_cameras()
    print(f"\nTotal cameras found: {len(cameras)}")
    
    if len(cameras) > 3:
        print("\nTo add the new camera to your test, update config.py:")
        print('REALSENSE_CAMERAS = {')
        print('    "wrist_1": {"serial_number": "332322073603"},')
        print('    "front": {"serial_number": "244222075350"},')
        print('    "side": {"serial_number": "243122071795"},')
        for i, serial in enumerate(cameras):
            if serial not in ["332322073603", "244222075350", "243122071795"]:
                print(f'    "new_cam": {{"serial_number": "{serial}"}},  # NEW CAMERA')
                break
        print('}')