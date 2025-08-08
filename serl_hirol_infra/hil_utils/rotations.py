from scipy.spatial.transform import Rotation as R


def quat_2_euler(quat):
    """
    Convert quaternion to Euler angles using xyz convention.
    
    Args:
        quat: Quaternion as [qx, qy, qz, qw]
    
    Returns:
        [roll, pitch, yaw] angles in radians
        roll: rotation around x-axis
        pitch: rotation around y-axis
        yaw: rotation around z-axis
    """
    return R.from_quat(quat).as_euler("xyz")


def euler_2_quat(xyz):
    """
    Convert Euler angles to quaternion using xyz convention.
    
    Args:
        xyz: [roll, pitch, yaw] angles in radians
             roll: rotation around x-axis
             pitch: rotation around y-axis  
             yaw: rotation around z-axis
    
    Returns:
        Quaternion as [qx, qy, qz, qw]
    """
    # Use scipy's Rotation for consistent conversion
    # xyz convention matches quat_2_euler above
    return R.from_euler("xyz", xyz).as_quat()
