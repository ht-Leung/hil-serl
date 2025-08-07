#!/usr/bin/env python3
"""
Test if clip_safety_box causes rotation drift
"""

import numpy as np
from scipy.spatial.transform import Rotation


def simulate_clip_safety_box(pose, rpy_bounding_box_low, rpy_bounding_box_high):
    """Simulate the clip_safety_box function from fr3_env.py"""
    # Convert quat to euler
    euler = Rotation.from_quat(pose[3:]).as_euler("xyz")
    
    # Clip first euler angle separately due to discontinuity
    sign = np.sign(euler[0])
    euler[0] = sign * (
        np.clip(
            np.abs(euler[0]),
            rpy_bounding_box_low[0],
            rpy_bounding_box_high[0],
        )
    )
    
    # Clip other angles
    euler[1:] = np.clip(
        euler[1:], rpy_bounding_box_low[1:], rpy_bounding_box_high[1:]
    )
    
    # Convert back to quat
    pose[3:] = Rotation.from_euler("xyz", euler).as_quat()
    
    return pose


def test_clip_safety_box_drift():
    """Test if repeated clip_safety_box causes drift"""
    print("=" * 60)
    print("Testing clip_safety_box Rotation Drift")
    print("=" * 60)
    
    # Initial pose (from RESET_POSE = [0.5, 0.0, 0.4, -np.pi, 0, 0])
    initial_euler = np.array([-np.pi, 0, 0])
    initial_quat = Rotation.from_euler("xyz", initial_euler).as_quat()
    initial_pose = np.array([0.5, 0.0, 0.4, initial_quat[0], initial_quat[1], initial_quat[2], initial_quat[3]])
    
    # Bounds from config
    rpy_bounding_box_low = np.array([np.pi-0.5, -0.5, -0.5])
    rpy_bounding_box_high = np.array([np.pi+0.5, 0.5, 0.5])
    
    print(f"Initial pose quaternion: {initial_pose[3:]}")
    print(f"Initial euler angles: {initial_euler}")
    print(f"RPY bounds: low={rpy_bounding_box_low}, high={rpy_bounding_box_high}")
    print()
    
    # Simulate multiple steps without any rotation change
    current_pose = initial_pose.copy()
    
    for i in range(100):
        # Simulate step() without rotation change
        # In fr3_env: self.nextpos[3:] = self.currpos[3:] when ACTION_SCALE[1]=0
        next_pose = current_pose.copy()  # No change
        
        # Apply clip_safety_box (this happens every step)
        next_pose = simulate_clip_safety_box(next_pose, rpy_bounding_box_low, rpy_bounding_box_high)
        
        # Check drift
        quat_drift = np.linalg.norm(next_pose[3:] - initial_pose[3:])
        
        if i % 20 == 0 or quat_drift > 1e-6:
            current_euler = Rotation.from_quat(next_pose[3:]).as_euler("xyz")
            euler_drift = current_euler - initial_euler
            print(f"Step {i:3d}:")
            print(f"  Quaternion drift: {quat_drift:.6e}")
            print(f"  Euler drift (deg): {np.degrees(euler_drift)}")
            if quat_drift > 1e-6:
                print(f"  ⚠️  SIGNIFICANT DRIFT DETECTED!")
        
        current_pose = next_pose
    
    # Final analysis
    final_quat_drift = np.linalg.norm(current_pose[3:] - initial_pose[3:])
    final_euler = Rotation.from_quat(current_pose[3:]).as_euler("xyz")
    final_euler_drift = final_euler - initial_euler
    
    print("\n" + "=" * 60)
    print("Final Results after 100 steps:")
    print(f"Quaternion drift: {final_quat_drift:.6e}")
    print(f"Euler angle drift (degrees): {np.degrees(final_euler_drift)}")
    print(f"Initial quaternion: {initial_pose[3:]}")
    print(f"Final quaternion:   {current_pose[3:]}")
    
    if final_quat_drift > 1e-10:
        print("\n⚠️  PROBLEM CONFIRMED: clip_safety_box causes rotation drift!")
        print("   Even without any rotation commands, the repeated")
        print("   quaternion↔euler conversions introduce cumulative errors.")
    else:
        print("\n✓ No significant drift detected")


if __name__ == "__main__":
    test_clip_safety_box_drift()
    
    print("\n" + "=" * 60)
    print("Solution:")
    print("=" * 60)
    print("""
To fix this issue, modify clip_safety_box to skip rotation clipping
when ACTION_SCALE[1] = 0, or only clip if outside bounds:

def clip_safety_box(self, pose):
    # Clip position
    pose[:3] = np.clip(pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high)
    
    # Only process rotation if needed
    if self.action_scale[1] > 0.001:  # Rotation is enabled
        euler = Rotation.from_quat(pose[3:]).as_euler("xyz")
        # ... clip euler angles ...
        pose[3:] = Rotation.from_euler("xyz", euler).as_quat()
    
    return pose
""")