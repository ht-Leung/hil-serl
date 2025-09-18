#!/usr/bin/env python3
"""
读取RealSense相机的实际内参并生成配置代码
"""

import pyrealsense2 as rs
import json
import sys

# Camera serial numbers from config
CAMERAS = {
    "wrist_1": "332322073603",
    "front": "244222075350",
    "side": "243122071795"
}

def read_camera_intrinsics():
    """读取所有相机的内参"""
    intrinsics_dict = {}

    for cam_name, serial_number in CAMERAS.items():
        print(f"\n🔍 Reading intrinsics for {cam_name} (Serial: {serial_number})...")

        try:
            # Initialize pipeline
            pipeline = rs.pipeline()
            config = rs.config()

            # Configure specific camera
            config.enable_device(serial_number)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

            # Start streaming
            profile = pipeline.start(config)

            # Get depth intrinsics
            depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
            depth_intrinsics = depth_profile.get_intrinsics()

            # Get depth scale
            device = profile.get_device()
            depth_sensor = device.first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()

            # Store intrinsics
            intrinsics_dict[cam_name] = {
                "fx": round(depth_intrinsics.fx, 3),
                "fy": round(depth_intrinsics.fy, 3),
                "cx": round(depth_intrinsics.ppx, 3),
                "cy": round(depth_intrinsics.ppy, 3),
                "width": depth_intrinsics.width,
                "height": depth_intrinsics.height,
                "depth_scale": round(depth_scale, 6),
                "model": str(depth_intrinsics.model)
            }

            print(f"✓ Successfully read {cam_name}:")
            print(f"  fx={depth_intrinsics.fx:.3f}, fy={depth_intrinsics.fy:.3f}")
            print(f"  cx={depth_intrinsics.ppx:.3f}, cy={depth_intrinsics.ppy:.3f}")
            print(f"  depth_scale={depth_scale:.6f}")
            print(f"  resolution={depth_intrinsics.width}x{depth_intrinsics.height}")

            # Stop pipeline
            pipeline.stop()

        except Exception as e:
            print(f"✗ Failed to read {cam_name}: {e}")
            # Use default values as fallback
            intrinsics_dict[cam_name] = {
                "fx": 600.0, "fy": 600.0,
                "cx": 320.0, "cy": 240.0,
                "width": 640, "height": 480,
                "depth_scale": 0.001,
                "model": "FALLBACK"
            }
            print(f"  Using fallback values for {cam_name}")

    return intrinsics_dict

def generate_config_code(intrinsics_dict):
    """生成Python配置代码"""
    print("\n" + "="*60)
    print("📝 Generated CAMERA_INTRINSICS configuration:")
    print("="*60)

    print("    CAMERA_INTRINSICS = {")
    for cam_name, intrinsics in intrinsics_dict.items():
        print(f'        "{cam_name}": {{')
        print(f'            "fx": {intrinsics["fx"]}, "fy": {intrinsics["fy"]},')
        print(f'            "cx": {intrinsics["cx"]}, "cy": {intrinsics["cy"]},')
        print(f'            "depth_scale": {intrinsics["depth_scale"]},')
        print(f'            "width": {intrinsics["width"]}, "height": {intrinsics["height"]}')
        print(f'        }},')
    print("    }")

    print("\n" + "="*60)
    return intrinsics_dict

def save_to_json(intrinsics_dict, filename="camera_intrinsics.json"):
    """保存到JSON文件作为备份"""
    with open(filename, 'w') as f:
        json.dump(intrinsics_dict, f, indent=4)
    print(f"💾 Saved intrinsics to {filename}")

def main():
    print("🎥 RealSense Camera Intrinsics Reader")
    print("=====================================")

    # Check if cameras are connected
    ctx = rs.context()
    devices = ctx.query_devices()
    print(f"🔌 Found {len(devices)} RealSense devices connected")

    if len(devices) == 0:
        print("❌ No RealSense cameras found! Please check connections.")
        sys.exit(1)

    # List connected devices
    for i, device in enumerate(devices):
        serial = device.get_info(rs.camera_info.serial_number)
        name = device.get_info(rs.camera_info.name)
        print(f"  Device {i}: {name} (Serial: {serial})")

    # Read intrinsics from all cameras
    intrinsics_dict = read_camera_intrinsics()

    # Generate configuration code
    config_dict = generate_config_code(intrinsics_dict)

    # Save backup
    save_to_json(config_dict)

    print("\n✅ Done! Copy the CAMERA_INTRINSICS configuration above to your config.py file.")
    print("💡 Tip: The intrinsics are also saved to camera_intrinsics.json for backup.")

if __name__ == "__main__":
    main()