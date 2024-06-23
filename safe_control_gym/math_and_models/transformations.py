"""Transformations with NumPy

Based on github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/deep_mimic/mocap/transformation.py

For SDFormat specification, see http://sdformat.org/tutorials?tut=specify_pose&cat=specification&
"""

import casadi as cs
import numpy as np
import numpy.typing as npt


def csRotZ(psi: float) -> cs.MX:
    """Rotation matrix about Z axis following SDFormat.

    Args:
        psi: Scalar rotation.

    Returns:
        R: casadi Rotation matrix
    """
    return cs.blockcat([[cs.cos(psi), -cs.sin(psi), 0], [cs.sin(psi), cs.cos(psi), 0], [0, 0, 1]])


def csRotY(theta: float) -> cs.MX:
    """Rotation matrix about Y axis following SDFormat. .

    Args:
        theta: Scalar rotation

    Returns:
        R: casadi Rotation matrix
    """
    return cs.blockcat(
        [[cs.cos(theta), 0, cs.sin(theta)], [0, 1, 0], [-cs.sin(theta), 0, cs.cos(theta)]]
    )


def csRotX(phi: float) -> cs.MX:
    """Rotation matrix about X axis.

    Args:
        phi: Scalar rotation

    Returns:
        R: casadi Rotation matrix
    """
    return cs.blockcat([[1, 0, 0], [0, cs.cos(phi), -cs.sin(phi)], [0, cs.sin(phi), cs.cos(phi)]])


def csRotXYZ(phi: float, theta: float, psi: float) -> cs.MX:
    """Rotation matrix from euler angles.

    This represents the extrinsic X-Y-Z (or quivalently the intrinsic Z-Y-X (3-2-1)) euler angle
    rotation.

    Args:
        phi: roll (or rotation about X).
        theta: pitch (or rotation about Y).
        psi: yaw (or rotation about Z).

    Returns:
        R: casadi Rotation matrix
    """
    return csRotZ(psi) @ csRotY(theta) @ csRotX(phi)


def RotXYZ(phi: float, theta: float, psi: float) -> npt.NDArray[np.float_]:
    """Rotation matrix from euler angles as numpy array.

    This represents the extrinsic X-Y-Z (or quivalently the intrinsic Z-Y-X (3-2-1)) euler angle
    rotation.

    Args:
        phi: roll (or rotation about X).
        theta: pitch (or rotation about Y).
        psi: yaw (or rotation about Z).

    Returns:
        R: casadi Rotation matrix
    """
    return csRotXYZ(phi, theta, psi).toarray()
