#!/usr/bin/env python3
"""
Test rotation drift issue when ACTION_SCALE[1] = 0
"""

import numpy as np
from scipy.spatial.transform import Rotation

# Test if Rotation.from_euler with zeros creates identity
def test_zero_euler_rotation():
    """Test if zero euler angles produce identity rotation"""
    print("=" * 60)
    print("Testing Zero Euler Rotation")
    print("=" * 60)
    
    # Create identity quaternion [0, 0, 0, 1] (qx, qy, qz, qw)
    identity_quat = np.array([0, 0, 0, 1])
    
    # Test initial pose (like robot's reset pose)
    initial_pose_quat = Rotation.from_euler("xyz", [-np.pi, 0, 0]).as_quat()
    print(f"Initial pose quaternion: {initial_pose_quat}")
    
    # Simulate multiple steps with zero rotation
    current_quat = initial_pose_quat.copy()
    
    for i in range(100):
        # Simulate action[3:6] * 0.0 = [0, 0, 0]
        rotation_delta = np.array([0.0, 0.0, 0.0])
        
        # Apply the rotation update (same as in fr3_env.py)
        current_quat = (
            Rotation.from_euler("xyz", rotation_delta)
            * Rotation.from_quat(current_quat)
        ).as_quat()
        
        if i % 20 == 0:
            # Check drift
            drift = np.linalg.norm(current_quat - initial_pose_quat)
            print(f"Step {i:3d}: Quaternion = {current_quat}, Drift = {drift:.6e}")
    
    # Final drift check
    final_drift = np.linalg.norm(current_quat - initial_pose_quat)
    print(f"\nFinal drift after 100 steps: {final_drift:.6e}")
    
    if final_drift > 1e-10:
        print("⚠️  WARNING: Numerical drift detected even with zero rotation!")
    else:
        print("✓ No significant drift with zero rotation")
    
    print("\n" + "=" * 60)
    print("Testing with small non-zero values (noise)")
    print("=" * 60)
    
    # Test with small noise (like SpaceMouse noise)
    current_quat = initial_pose_quat.copy()
    
    for i in range(100):
        # Simulate small noise that might come from SpaceMouse
        # even when ACTION_SCALE[1] = 0, floating point errors might create tiny values
        noise = np.random.normal(0, 1e-15, 3)  # Very small noise
        
        # Apply the rotation update
        current_quat = (
            Rotation.from_euler("xyz", noise)
            * Rotation.from_quat(current_quat)
        ).as_quat()
        
        if i % 20 == 0:
            drift = np.linalg.norm(current_quat - initial_pose_quat)
            print(f"Step {i:3d}: Drift = {drift:.6e}")
    
    final_drift_noise = np.linalg.norm(current_quat - initial_pose_quat)
    print(f"\nFinal drift with tiny noise: {final_drift_noise:.6e}")
    
    # Convert to euler to see angular drift
    initial_euler = Rotation.from_quat(initial_pose_quat).as_euler("xyz")
    final_euler = Rotation.from_quat(current_quat).as_euler("xyz")
    euler_diff = final_euler - initial_euler
    print(f"Euler angle drift (degrees): {np.degrees(euler_diff)}")


def test_spacemouse_action_flow():
    """Test what happens to SpaceMouse action through the pipeline"""
    print("\n" + "=" * 60)
    print("Testing SpaceMouse Action Flow")
    print("=" * 60)
    
    # Simulate SpaceMouse output (after pyspacemouse scaling by 350)
    # Even when not touching, might have small values
    spacemouse_output = np.array([
        0.001,  # x translation
        0.002,  # y translation
        -0.001, # z translation
        0.003,  # roll
        -0.002, # pitch
        0.001   # yaw
    ])
    
    print(f"SpaceMouse output: {spacemouse_output}")
    
    # Add gripper action
    action = np.concatenate([spacemouse_output, [0.0]])  # 7D action
    
    # Apply ACTION_SCALE
    ACTION_SCALE = np.array([0.01, 0.00, 1])  # From config
    
    # Position scaling
    xyz_delta = action[:3] * ACTION_SCALE[0]
    print(f"Position delta after scaling: {xyz_delta}")
    
    # Rotation scaling (should be all zeros)
    rotation_delta = action[3:6] * ACTION_SCALE[1]
    print(f"Rotation delta after scaling: {rotation_delta}")
    print(f"Is rotation exactly zero? {np.all(rotation_delta == 0.0)}")
    
    # Check floating point representation
    for i, val in enumerate(rotation_delta):
        print(f"  rotation[{i}] = {val:.20e}")


if __name__ == "__main__":
    test_zero_euler_rotation()
    test_spacemouse_action_flow()
    
    print("\n" + "=" * 60)
    print("Conclusion:")
    print("=" * 60)
    print("""
If drift is still occurring with ACTION_SCALE[1] = 0, possible causes:
1. SpaceMouse values are being applied somewhere else in the pipeline
2. The SpacemouseIntervention wrapper might be modifying actions
3. Other wrappers (Quat2EulerWrapper) might introduce errors
4. The robot's impedance controller might have residual rotation stiffness
    """)