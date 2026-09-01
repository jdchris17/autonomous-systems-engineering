"""module-3 -- rotations.py

Shared library: the three elementary rotation matrices and SO(3) checks.
Everything else in this module imports from here.

CONVENTION (the full statement is in README.md -- do not leave it implicit):

  * Angles are in radians.
  * rot_x(a), rot_y(a), rot_z(a) are ACTIVE, right-handed rotations: they
    rotate a *vector* by +a about a fixed coordinate axis (right-hand rule).
    Equivalently, their transpose rotates the coordinate frame.
  * A rotation R is valid (in SO(3)) when  R^T R = I  and  det(R) = +1.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Elementary rotations (active, right-handed)
# ---------------------------------------------------------------------------
def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0,   c,  -s],
                     [0.0,   s,   c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[  c, 0.0,   s],
                     [0.0, 1.0, 0.0],
                     [ -s, 0.0,   c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[  c,  -s, 0.0],
                     [  s,   c, 0.0],
                     [0.0, 0.0, 1.0]])


AXES = {"x": rot_x, "y": rot_y, "z": rot_z}


def axis_angle_to_R(axis, angle):
    """Rodrigues' formula: active rotation by `angle` (rad) about `axis`.

    This is the "construct it directly" rotation matrix that the quaternion
    route (axis-angle -> q -> R) is checked against.
    """
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    K = np.array([[0.0, -n[2], n[1]],
                  [n[2], 0.0, -n[0]],
                  [-n[1], n[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


# ---------------------------------------------------------------------------
# SO(3) validation
# ---------------------------------------------------------------------------
def orthonormality_error(R):
    """max |R^T R - I| entrywise -- zero for a true rotation."""
    return np.abs(R.T @ R - np.eye(3)).max()


def determinant_error(R):
    """|det(R) - 1| -- zero for a proper (non-reflecting) rotation."""
    return abs(np.linalg.det(R) - 1.0)


def is_rotation(R, tol=1e-9):
    return orthonormality_error(R) < tol and determinant_error(R) < tol


def assert_rotation(R, name="R", tol=1e-9):
    oe, de = orthonormality_error(R), determinant_error(R)
    if oe >= tol or de >= tol:
        raise AssertionError(
            f"{name} is not in SO(3):  |R^T R - I| = {oe:.2e}, "
            f"|det - 1| = {de:.2e}")
    return oe, de
