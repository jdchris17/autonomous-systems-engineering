"""
imu_alignment.py - Space Systems Engineering, Module 5

Orientation of the IMU relative to the local vertical, the ideal specific
force it should sense, and the chain that carries a measured vertical from
IMU coordinates to celestial coordinates.

Frames (Module 3 conventions; C_AB converts B coordinates to A coordinates)
---------------------------------------------------------------------------
N  local navigation frame: x east, y north, z UP.
I  IMU sensing axes.
C  camera, G celestial: as in Modules 3 and 4.

Physics
-------
An accelerometer measures specific force, f = a - g: acceleration relative
to free fall. With gravity g_N = [0, 0, -g]:

    f_I = C_IN (a_N - g_N)

At rest (a = 0) this is +g along UP: the sensor feels the support force, so
the stationary reading points up, not down. f_N = [0, 0, 9.81] m/s^2.

Small-error conventions used by the error budget
------------------------------------------------
* accelerometer error d_a [m/s^2], IMU axes, added to the mean reading.
* extrinsic error d_theta_cal [rad], rotation vector in CAMERA axes:
      C_CI,used = R(d_theta_cal) C_CI,true
  (the same error as Module 3's C_IC,true = C_IC,nominal R(d_theta)).
* camera attitude error d_theta_cam [rad], rotation vector in CAMERA axes,
  Module 4's convention:  C_CG,estimated = R(d_theta_cam) C_CG,true.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "module-3"))
from rotation_engine import rotation_vector_to_matrix, skew  # noqa: E402
from sensor_frames import Transform  # noqa: E402

GRAVITY = 9.81                                   # m/s^2
UP_N = np.array([0.0, 0.0, 1.0])
EAST_N = np.array([1.0, 0.0, 0.0])
NORTH_N = np.array([0.0, 1.0, 0.0])


def _matrix(transform, to_frame, from_frame):
    """Accept a Transform (frames checked) or a bare 3x3 array."""
    if isinstance(transform, Transform):
        if (transform.to_frame, transform.from_frame) != (to_frame, from_frame):
            raise ValueError(f"expected C_{to_frame}{from_frame}, "
                             f"got {transform.name}")
        return transform.matrix
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"expected a 3x3 matrix, got shape {matrix.shape}")
    return matrix


# ------------------------------------------------------------- IMU orientation
def imu_orientation(tilt_deg=0.0, tilt_azimuth_deg=0.0, heading_deg=0.0):
    """
    C_IN for an IMU that is first turned by heading about UP and then
    tilted by tilt_deg about the horizontal axis at tilt_azimuth_deg
    (measured from east toward north).

    tilt_deg is exactly the angle between the IMU z axis and the vertical.
    tilt = heading = 0 gives IMU axes aligned with east, north, up.
    """
    azimuth = np.radians(tilt_azimuth_deg)
    tilt_axis = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
    imu_axes_in_nav = (rotation_vector_to_matrix(np.radians(tilt_deg) * tilt_axis)
                       @ rotation_vector_to_matrix(np.radians(heading_deg) * UP_N))
    return Transform("N", "I", imu_axes_in_nav).inverse()


def orientation_from_rotation_vector(rotvec_rad):
    """C_IN for IMU axes obtained by rotating the N axes by rotvec_rad."""
    return Transform("N", "I", rotation_vector_to_matrix(rotvec_rad)).inverse()


def orientation_with_up(up_imu, heading_deg=0.0):
    """
    A C_IN whose upward direction, in IMU coordinates, is exactly up_imu.
    The rotation about the vertical is set by heading_deg.
    """
    up = np.asarray(up_imu, dtype=float)
    up = up / np.linalg.norm(up)
    helper = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    east = np.cross(helper, up)
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    heading = np.radians(heading_deg)
    east_h = np.cos(heading) * east + np.sin(heading) * north
    north_h = np.cross(up, east_h)
    # Columns of C_IN are the N axes (east, north, up) in IMU coordinates.
    return Transform("I", "N", np.column_stack((east_h, north_h, up)))


def true_up_imu(C_IN):
    """The true upward unit vector expressed in IMU coordinates."""
    return _matrix(C_IN, "I", "N") @ UP_N


def specific_force(C_IN, acceleration_n=(0.0, 0.0, 0.0), g=GRAVITY):
    """
    Ideal accelerometer reading  f_I = C_IN (a_N - g_N)  in m/s^2.

    acceleration_n is the instrument's acceleration in N coordinates, one
    vector or a (T, 3) time series. Zero acceleration gives the stationary
    reading, g along the upward direction.
    """
    acceleration = np.asarray(acceleration_n, dtype=float)
    gravity_n = np.array([0.0, 0.0, -g])
    return (acceleration - gravity_n) @ _matrix(C_IN, "I", "N").T


# ------------------------------------------------------- camera / IMU alignment
def perturb_calibration(C_CI, delta_theta_rad):
    """
    C_CI with an extrinsic error:  R(delta_theta) C_CI.

    delta_theta_rad is a rotation vector in camera axes (not an Euler
    sequence). Use it to build the calibration the software believes from
    the true one, or the reverse with the sign flipped.
    """
    matrix = _matrix(C_CI, "C", "I")
    return Transform("C", "I",
                     rotation_vector_to_matrix(delta_theta_rad) @ matrix)


def vertical_in_celestial(up_imu, C_CI, C_GC):
    """
    Upward direction in celestial coordinates:  u_G = C_GC C_CI u_I.

    C_CI is the camera-from-IMU calibration and C_GC the celestial-from-
    camera attitude (the transpose of Module 4's C_CG).
    """
    up = np.asarray(up_imu, dtype=float)
    return _matrix(C_GC, "G", "C") @ (_matrix(C_CI, "C", "I") @ up)


def vertical_jacobians(up_imu, C_CI, C_GC, specific_force_magnitude=GRAVITY):
    """
    First-order sensitivity of u_G to each error source:

        d u_G = J_a d_a + J_cal d_theta_cal + J_cam d_theta_cam

    J_a   = C_GC C_CI (I - u_I u_I^T) / |f|      accelerometer error [m/s^2]
    J_cal = -C_GC [u_C x]                        extrinsic error [rad]
    J_cam = +C_GC [u_C x]                        camera attitude error [rad]

    with u_C = C_CI u_I. Both rotation errors act through the cross product
    with u_C: a rotation about the vertical itself (as seen in the camera)
    does not move the vertical at all.
    """
    up = np.asarray(up_imu, dtype=float)
    up = up / np.linalg.norm(up)
    c_ci = _matrix(C_CI, "C", "I")
    c_gc = _matrix(C_GC, "G", "C")
    up_camera = c_ci @ up
    cross = skew(up_camera)
    return {
        "accelerometer": c_gc @ c_ci @ (np.eye(3) - np.outer(up, up))
        / specific_force_magnitude,
        "calibration": -c_gc @ cross,
        "camera": c_gc @ cross,
    }


def vertical_covariance(jacobians, covariances):
    """
    Independent-error covariance of u_G:  sum_k J_k P_k J_k^T.

    Both arguments are dicts with matching keys. Returns
    (total 3x3, {key: 3x3 contribution}). Cross-covariances between sources
    are NOT included; correlated errors need extra terms.
    """
    parts = {key: jacobians[key] @ np.asarray(covariances[key], dtype=float)
             @ jacobians[key].T for key in covariances}
    return sum(parts.values()), parts


def angular_rms(covariance):
    """RMS angular error [rad] of a unit vector with this covariance."""
    return float(np.sqrt(np.trace(covariance)))
