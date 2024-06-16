import numpy as np

from scipy.spatial.transform import Rotation as R


def _get_quaternion_from_euler(roll, pitch, yaw):
    """Convert an Euler angle to a quaternion.

    Args:
        roll (float): The roll (rotation around x-axis) angle in radians.
        pitch (float): The pitch (rotation around y-axis) angle in radians.
        yaw (float): The yaw (rotation around z-axis) angle in radians.

    Returns:
        list: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(
        pitch / 2
    ) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(
        pitch / 2
    ) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(
        pitch / 2
    ) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(
        pitch / 2
    ) * np.sin(yaw / 2)

    return [qx, qy, qz, qw]


for i in range(100000):
    r, p, y = np.random.rand(3) * 2 * np.pi
    q1 = np.array(_get_quaternion_from_euler(r, p, y))
    q2 = R.from_euler("xyz", [r, p, y]).as_quat()
    assert np.allclose(q1, q2)
