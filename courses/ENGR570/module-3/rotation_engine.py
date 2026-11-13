"""
rotation_engine.py - Space Systems Engineering, Module 3, Experiment A

Quaternions, rotation matrices and rotation vectors, implemented in NumPy.

Conventions (used everywhere in this module)
--------------------------------------------
* Quaternions are scalar-first: q = [w, x, y, z].
  (SciPy's Rotation.as_quat() is scalar-last, [x, y, z, w], by default.)
* Hamilton product: i*j = k, j*k = i, k*i = j.
* Rotations are ACTIVE and right-handed: R(q) @ v turns the vector v within
  a fixed frame, counter-clockwise when seen looking down the axis.
* Composition: R(q (x) p) = R(q) R(p), so in q (x) p the rotation p is
  applied first, then q.
* q and -q are the same physical rotation.
"""

import numpy as np

# How far R^T R may be from I, or det R from 1, before a matrix is rejected.
ROTATION_TOLERANCE = 1e-9

# Below this angle [rad] the Rodrigues coefficients switch to Taylor series.
SMALL_ANGLE = 1e-6


# ---------------------------------------------------------------- utilities
def _quaternion(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,):
        raise ValueError(f"expected a quaternion [w, x, y, z], got shape {q.shape}")
    return q


def _vector3(v, name="vector"):
    v = np.asarray(v, dtype=float)
    if v.shape != (3,):
        raise ValueError(f"expected a 3-element {name}, got shape {v.shape}")
    return v


def quaternion_normalize(q):
    """Return q scaled to unit length."""
    q = _quaternion(q)
    norm = np.linalg.norm(q)
    if norm == 0.0:
        raise ValueError("cannot normalize the zero quaternion")
    return q / norm


def skew(v):
    """Cross-product matrix: skew(a) @ b == np.cross(a, b)."""
    x, y, z = _vector3(v)
    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])


def is_rotation_matrix(R, tolerance=ROTATION_TOLERANCE):
    """True if R is 3x3, orthogonal (R^T R = I) and proper (det R = +1)."""
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        return False
    orthogonal = np.max(np.abs(R.T @ R - np.eye(3))) <= tolerance
    proper = abs(np.linalg.det(R) - 1.0) <= tolerance
    return bool(orthogonal and proper)


def check_rotation_matrix(R, tolerance=ROTATION_TOLERANCE):
    """Return R as an array, or raise ValueError saying what is wrong."""
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"expected a 3x3 matrix, got shape {R.shape}")
    error = np.max(np.abs(R.T @ R - np.eye(3)))
    if error > tolerance:
        raise ValueError(f"matrix is not orthogonal (max |R^T R - I| = {error:.2e})")
    det = np.linalg.det(R)
    if abs(det - 1.0) > tolerance:
        raise ValueError(f"matrix is not a proper rotation (det = {det:.6f})")
    return R


def rotation_matrix_angle(R):
    """
    Rotation angle of R in radians, in [0, pi].

    Uses atan2 of the antisymmetric part and the trace, which stays accurate
    for tiny angles where acos((trace - 1) / 2) cannot resolve anything
    below ~1e-8 rad.
    """
    R = np.asarray(R, dtype=float)
    axis_times_2sin = np.array([R[2, 1] - R[1, 2],
                                R[0, 2] - R[2, 0],
                                R[1, 0] - R[0, 1]])
    return float(np.arctan2(np.linalg.norm(axis_times_2sin), np.trace(R) - 1.0))


# ------------------------------------------------------- required functions
def axis_angle_to_quaternion(axis, angle_rad):
    """Return scalar-first Hamilton unit quaternion.

    q = [cos(angle/2), sin(angle/2) * unit_axis]. The axis need not be
    normalized, but must not be zero.
    """
    axis = _vector3(axis, "axis")
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        raise ValueError("rotation axis must be non-zero")
    half = 0.5 * angle_rad
    return np.concatenate(([np.cos(half)], np.sin(half) * axis / norm))


def quaternion_multiply(q, p):
    """Hamilton product: q (x) p.

    With q = (qw, qv) and p = (pw, pv):
        scalar = qw pw - qv . pv
        vector = qw pv + pw qv + qv x pv
    As a rotation, q (x) p applies p first and then q.
    """
    q, p = _quaternion(q), _quaternion(p)
    qw, qv = q[0], q[1:]
    pw, pv = p[0], p[1:]
    return np.concatenate((
        [qw * pw - np.dot(qv, pv)],
        qw * pv + pw * qv + np.cross(qv, pv),
    ))


def quaternion_conjugate(q):
    """Return quaternion conjugate.

    [w, -x, -y, -z]. For a unit quaternion this is the inverse rotation.
    """
    q = _quaternion(q)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_to_matrix(q):
    """Return active rotation matrix corresponding to q.

    The input is normalized first, so a slightly non-unit quaternion still
    gives an exactly orthogonal matrix (to round-off).
    """
    w, x, y, z = quaternion_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quaternion(R):
    """Convert proper rotation matrix to scalar-first quaternion.

    Uses Shepperd's method. The textbook formula
        w = sqrt(1 + trace) / 2,   x = (R21 - R12) / (4 w), ...
    divides by w, which goes to zero as the rotation angle approaches
    180 deg, amplifying round-off without bound. Instead, the four
    quantities 4w^2, 4x^2, 4y^2, 4z^2 are all available from the diagonal;
    the largest one is square-rooted and the other three components are
    obtained by dividing by it. The divisor is then always >= 1/2.

    The result is returned with w >= 0 (q and -q are the same rotation).
    """
    R = check_rotation_matrix(R)
    trace = np.trace(R)
    # 4*component^2 - 1 for w, x, y, z respectively.
    choices = (trace, R[0, 0] - R[1, 1] - R[2, 2],
               R[1, 1] - R[0, 0] - R[2, 2], R[2, 2] - R[0, 0] - R[1, 1])
    largest = int(np.argmax(choices))

    if largest == 0:
        w = 0.5 * np.sqrt(1.0 + trace)
        s = 0.25 / w
        q = [w, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s,
             (R[1, 0] - R[0, 1]) * s]
    elif largest == 1:
        x = 0.5 * np.sqrt(1.0 + choices[1])
        s = 0.25 / x
        q = [(R[2, 1] - R[1, 2]) * s, x, (R[0, 1] + R[1, 0]) * s,
             (R[0, 2] + R[2, 0]) * s]
    elif largest == 2:
        y = 0.5 * np.sqrt(1.0 + choices[2])
        s = 0.25 / y
        q = [(R[0, 2] - R[2, 0]) * s, (R[0, 1] + R[1, 0]) * s, y,
             (R[1, 2] + R[2, 1]) * s]
    else:
        z = 0.5 * np.sqrt(1.0 + choices[3])
        s = 0.25 / z
        q = [(R[1, 0] - R[0, 1]) * s, (R[0, 2] + R[2, 0]) * s,
             (R[1, 2] + R[2, 1]) * s, z]

    q = quaternion_normalize(q)
    return -q if q[0] < 0.0 else q


def rotate_vector(q, vector):
    """Apply quaternion's active rotation to a 3D vector.

    Equivalent to the vector part of q (x) [0, v] (x) q*, expanded to
        v' = v + 2 w (u x v) + 2 u x (u x v),   q = [w, u]
    which avoids building two quaternion products.
    """
    q = quaternion_normalize(q)
    v = _vector3(vector)
    w, u = q[0], q[1:]
    uv = np.cross(u, v)
    return v + 2.0 * (w * uv + np.cross(u, uv))


def rotation_vector_to_matrix(rotvec):
    """Convert a rotation vector (radians) to an active matrix.

    The rotation vector's direction is the axis and its length the angle.
    Rodrigues' formula with K = skew(rotvec):
        R = I + (sin t / t) K + ((1 - cos t) / t^2) K^2,   t = |rotvec|
    (1 - cos t) is evaluated as 2 sin^2(t/2) to avoid cancellation, and
    both coefficients use their Taylor series for very small t, so a zero
    rotation vector returns the identity without dividing by zero.
    """
    rotvec = _vector3(rotvec, "rotation vector")
    angle = np.linalg.norm(rotvec)
    K = skew(rotvec)

    if angle < SMALL_ANGLE:
        a = 1.0 - angle**2 / 6.0
        b = 0.5 - angle**2 / 24.0
    else:
        a = np.sin(angle) / angle
        b = 2.0 * np.sin(0.5 * angle) ** 2 / angle**2

    return np.eye(3) + a * K + b * (K @ K)


def rotation_vector_to_quaternion(rotvec):
    """Unit quaternion for a rotation vector (radians)."""
    rotvec = _vector3(rotvec, "rotation vector")
    angle = np.linalg.norm(rotvec)
    if angle < SMALL_ANGLE:
        # sin(angle/2)/angle -> 1/2 - angle^2/48
        return quaternion_normalize(
            np.concatenate(([1.0], rotvec * (0.5 - angle**2 / 48.0))))
    return axis_angle_to_quaternion(rotvec, angle)
