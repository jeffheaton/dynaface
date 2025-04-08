from typing import Any

import numpy as np
from numpy.typing import NDArray


def get_inverse_transf(affine_transf: NDArray[Any]) -> NDArray[Any]:
    A = affine_transf[0:2, 0:2]
    b = affine_transf[:, 2]
    inv_A = np.linalg.inv(A)  # we assume A invertible!
    inv_affine = np.zeros((2, 3))
    inv_affine[0:2, 0:2] = inv_A
    inv_affine[:, 2] = -inv_A.dot(b)
    return inv_affine


def rotation_matrix_to_euler(rot_matrix: NDArray[Any]) -> NDArray[Any]:
    """Converts a rotation matrix to Euler angles."""
    a00, _, a02 = rot_matrix[0, 0], rot_matrix[0, 1], rot_matrix[0, 2]
    a10, a11, a12 = rot_matrix[1, 0], rot_matrix[1, 1], rot_matrix[1, 2]
    a20, _, a22 = rot_matrix[2, 0], rot_matrix[2, 1], rot_matrix[2, 2]
    if abs(1.0 - a10) <= np.finfo(float).eps:  # singularity at north pole
        yaw = np.arctan2(a02, a22)
        pitch = np.pi / 2.0
        roll = 0
    elif abs(-1.0 - a10) <= np.finfo(float).eps:  # singularity at south pole
        yaw = np.arctan2(a02, a22)
        pitch = -np.pi / 2.0
        roll = 0
    else:  # standard case
        yaw = np.arctan2(-a20, a00)
        pitch = np.arcsin(a10)
        roll = np.arctan2(-a12, a11)
    euler = np.array([yaw, pitch, roll]) * (180.0 / np.pi)
    euler = np.array([(-euler[0]) + 90, -euler[1], (-euler[2]) - 90])
    euler = np.where(euler > 180, euler - 360, euler)
    euler = np.where(euler < -180, euler + 360, euler)
    return euler
