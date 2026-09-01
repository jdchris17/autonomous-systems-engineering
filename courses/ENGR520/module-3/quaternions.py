"""module-3 -- quaternions.py

Quaternion mathematics, implemented from scratch (no quaternion package).

CONVENTION (fixed here; see README.md):

  * Layout    : SCALAR-FIRST,  q = [w, x, y, z].
  * Product   : HAMILTON (not JPL).
  * Meaning   : a unit q encodes the SAME rotation as R_IB in this module --
                body -> inertial.  v_I = R(q) v_B  =  q (x) [0, v_B] (x) q*.
  * Unit q    : ||q|| = 1.  q and -q are the same rotation (double cover).

The five functions the exercise asks for are quaternion_multiply,
quaternion_conjugate, quaternion_normalize, axis_angle_to_quaternion,
quaternion_to_rotation_matrix. rotate_vector and IDENTITY are convenience.
"""

import numpy as np

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# core algebra
# ---------------------------------------------------------------------------
def quaternion_multiply(q1, q2):
    """Hamilton product q1 (x) q2, scalar-first. Not commutative."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quaternion_conjugate(q):
    """q* = [w, -x, -y, -z]. For a unit quaternion this is the inverse rotation."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quaternion_normalize(q):
    """Return q / ||q||; raises if q is (numerically) zero."""
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("cannot normalize a zero quaternion")
    return q / n


# ---------------------------------------------------------------------------
# conversions
# ---------------------------------------------------------------------------
def axis_angle_to_quaternion(axis, angle):
    """Unit quaternion for a rotation of `angle` (rad) about `axis`.

    q = [cos(angle/2), sin(angle/2) * axis_unit]
    """
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return IDENTITY.copy()
    axis = axis / n
    half = 0.5 * angle
    return np.concatenate([[np.cos(half)], np.sin(half) * axis])


def quaternion_to_rotation_matrix(q):
    """Rotation matrix R(q) with  v_I = R(q) @ v_B  (Hamilton, active)."""
    w, x, y, z = quaternion_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


# ---------------------------------------------------------------------------
# convenience
# ---------------------------------------------------------------------------
def rotate_vector(q, v):
    """Apply the rotation: returns q (x) [0, v] (x) q*  (vector part)."""
    v = np.asarray(v, dtype=float)
    pure = np.concatenate([[0.0], v])
    return quaternion_multiply(quaternion_multiply(q, pure),
                               quaternion_conjugate(q))[1:]


def quaternion_angle(q):
    """Rotation angle (rad) encoded by unit quaternion q, in [0, pi]."""
    w = abs(quaternion_normalize(q)[0])
    return 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
