"""
accelerometer_calibration.py - Space Systems Engineering, Module 5

Synthetic accelerometer error model and its correction.

Measurement model
-----------------
    raw = M f_true + b + noise

    f_true  ideal specific force in IMU axes [m/s^2]
    M       3x3 measurement matrix = I + diag(scale-factor errors)
            + cross-axis coupling on the off-diagonals
    b       constant bias [m/s^2]
    noise   independent zero-mean Gaussian on each axis and sample

Correction (see gravity_estimator.calibrate_accelerometer)
----------------------------------------------------------
    f_calibrated = K (raw - b),   K = M^-1  (the "correction matrix")

The default numbers are the illustrative values from the assignment. They
are not specifications of any real device.
"""

from dataclasses import dataclass

import numpy as np

from imu_alignment import GRAVITY, Transform, specific_force


@dataclass(frozen=True)
class AccelerometerErrorModel:
    bias: tuple = (0.01, -0.02, 0.005)        # m/s^2
    scale_factor_error: object = 0.001        # 0.1 % per axis (scalar or 3)
    cross_axis: float = 0.0005                # 0.05 % coupling between axes
    noise_sigma: float = 0.005                # m/s^2 per axis, per sample

    @property
    def bias_vector(self):
        return np.asarray(self.bias, dtype=float)

    @property
    def measurement_matrix(self):
        """M: maps true specific force to the (bias-free) sensor output."""
        scale = np.broadcast_to(np.asarray(self.scale_factor_error, float), (3,))
        coupling = self.cross_axis * (np.ones((3, 3)) - np.eye(3))
        return np.eye(3) + np.diag(scale) + coupling

    @property
    def correction_matrix(self):
        """K = M^-1: undoes scale-factor and cross-axis errors."""
        return np.linalg.inv(self.measurement_matrix)

    def measure(self, true_specific_force, rng):
        """Raw readings (N, 3) for ideal specific-force vectors (N, 3)."""
        f = np.atleast_2d(np.asarray(true_specific_force, dtype=float))
        raw = f @ self.measurement_matrix.T + self.bias_vector
        if self.noise_sigma > 0.0:
            raw = raw + rng.normal(0.0, self.noise_sigma, size=raw.shape)
        return raw


IDEAL_SENSOR = AccelerometerErrorModel(bias=(0.0, 0.0, 0.0),
                                       scale_factor_error=0.0, cross_axis=0.0,
                                       noise_sigma=0.0)


def simulate_stationary(C_IN, model, n_samples, rng, g=GRAVITY):
    """n_samples raw readings (n_samples, 3) from a motionless IMU."""
    ideal = np.tile(specific_force(C_IN, g=g), (n_samples, 1))
    return model.measure(ideal, rng)


def six_position_orientations():
    """
    The classic six static calibration poses: each IMU axis pointing up,
    then down. Returns a list of C_IN Transforms.
    """
    poses = []
    for axis in range(3):
        for sign in (1.0, -1.0):
            up_imu = np.zeros(3)
            up_imu[axis] = sign
            # Any right-handed completion will do.
            a = np.roll(up_imu, 1)
            east = a / np.linalg.norm(a)
            north = np.cross(up_imu, east)
            poses.append(Transform("I", "N",
                                   np.column_stack((east, north, up_imu))))
    return poses


def estimate_calibration(mean_raw_readings, true_specific_forces):
    """
    Least-squares bias and correction matrix from static poses.

    mean_raw_readings: (K, 3) averaged raw output in each pose.
    true_specific_forces: (K, 3) the ideal specific force in each pose
    (known orientation and known g).

    Solves raw = M f + b for the 12 unknowns. Needs at least four poses
    whose specific-force vectors are not all in one plane.
    Returns (bias, correction_matrix) ready for calibrate_accelerometer().
    """
    raw = np.asarray(mean_raw_readings, dtype=float)
    f = np.asarray(true_specific_forces, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 3 or raw.shape != f.shape:
        raise ValueError("both inputs must have the same shape (K, 3)")
    design = np.column_stack((f, np.ones(len(f))))
    if len(f) < 4 or np.linalg.matrix_rank(design) < 4:
        raise ValueError("need at least four poses that are not coplanar "
                         "to separate bias from the measurement matrix")
    solution, *_ = np.linalg.lstsq(design, raw, rcond=None)
    measurement_matrix = solution[:3].T
    bias = solution[3]
    return bias, np.linalg.inv(measurement_matrix)
