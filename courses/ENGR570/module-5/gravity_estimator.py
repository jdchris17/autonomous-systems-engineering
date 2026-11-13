"""
gravity_estimator.py - Space Systems Engineering, Module 5

Local vertical from stationary accelerometer samples, with uncertainty and
a measurement-quality indicator.

Scope: this is a stationary local-vertical sensor, not an inertial
navigator. It averages specific force and normalizes. Any real acceleration
of the instrument is indistinguishable, in the averaged vector, from a tilt.
"""

from dataclasses import dataclass, field

import numpy as np

from imu_alignment import GRAVITY, _matrix


# ------------------------------------------------------------------ data types
@dataclass(frozen=True)
class SampleStatistics:
    count: int
    mean: np.ndarray               # mean specific force [m/s^2], IMU axes
    covariance: np.ndarray         # 3x3 sample covariance of single samples
    mean_covariance: np.ndarray    # P_f = covariance / count
    magnitude: float               # |mean| [m/s^2]
    std_per_axis: np.ndarray       # per-axis sample standard deviation


@dataclass(frozen=True)
class UpUncertainty:
    """
    Uncertainty of the unit vector u. All angles in radians.

    covariance is P_u = J_u P_f J_u^T, a 3x3 matrix of rank two: a unit
    vector cannot move along itself. The meaningful part is the 2x2
    tangent_covariance, P_u expressed in the two tangent_basis directions
    perpendicular to u. Its values are angles squared.

    This is the random-noise uncertainty of the mean only. It assumes the
    samples are independent and contains nothing about bias, calibration
    error or real acceleration.
    """
    covariance: np.ndarray
    tangent_basis: np.ndarray      # (2, 3): two unit vectors perpendicular to u
    tangent_covariance: np.ndarray  # (2, 2) [rad^2]
    sigma_major: float             # largest 1-sigma angular uncertainty
    sigma_minor: float             # smallest
    angular_rms: float             # sqrt(trace) = RMS total angular error


# ------------------------------------------------------------------ validation
def _samples(samples, name):
    a = np.asarray(samples, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return a


def tangent_basis(unit_vector):
    """Two orthonormal vectors perpendicular to unit_vector, shape (2, 3)."""
    u = np.asarray(unit_vector, dtype=float)
    helper = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(u, helper)
    e1 /= np.linalg.norm(e1)
    return np.vstack((e1, np.cross(u, e1)))


# ---------------------------------------------------------- required interface
def calibrate_accelerometer(raw_samples, bias, correction_matrix):
    """
    Correct accelerometer measurements.

    Samples: (N, 3), units m/s^2.
    Return calibrated specific-force vectors.

    f = K (raw - bias), with bias (3,) in m/s^2 and correction_matrix K
    (3x3) the inverse of the sensor's measurement matrix. Bias is removed
    first, then scale-factor and cross-axis errors.
    """
    raw = _samples(raw_samples, "raw_samples")
    bias = np.asarray(bias, dtype=float)
    K = np.asarray(correction_matrix, dtype=float)
    if bias.shape != (3,):
        raise ValueError(f"bias must have shape (3,), got {bias.shape}")
    if K.shape != (3, 3):
        raise ValueError(f"correction_matrix must be 3x3, got {K.shape}")
    return (raw - bias) @ K.T


def estimate_up_direction(calibrated_samples):
    """
    Estimate the stationary upward unit vector.

    Return:
      - estimated unit vector in IMU coordinates
      - uncertainty representation
      - sample statistics

    The specific-force vectors are averaged first and the mean is then
    normalized (averaging unit vectors instead would bias the result
    toward the noisier directions). Uncertainty: the covariance of the
    mean, P_f = S / N, is propagated through the normalization with
    J_u = (I - u u^T) / |f|, giving P_u = J_u P_f J_u^T.

    Returns (u, UpUncertainty, SampleStatistics).
    """
    f = _samples(calibrated_samples, "calibrated_samples")
    n = len(f)
    if n < 2:
        raise ValueError("at least two samples are needed for an uncertainty")
    mean = f.mean(axis=0)
    magnitude = float(np.linalg.norm(mean))
    if magnitude == 0.0:
        raise ValueError("mean specific force is zero; no direction to estimate")
    u = mean / magnitude

    covariance = np.cov(f, rowvar=False)
    mean_covariance = covariance / n
    jacobian = (np.eye(3) - np.outer(u, u)) / magnitude
    P_u = jacobian @ mean_covariance @ jacobian.T

    basis = tangent_basis(u)
    tangent = basis @ P_u @ basis.T
    eigenvalues = np.clip(np.linalg.eigvalsh(tangent), 0.0, None)

    statistics = SampleStatistics(
        count=n, mean=mean, covariance=covariance,
        mean_covariance=mean_covariance, magnitude=magnitude,
        std_per_axis=np.sqrt(np.diag(covariance)))
    uncertainty = UpUncertainty(
        covariance=P_u, tangent_basis=basis, tangent_covariance=tangent,
        sigma_major=float(np.sqrt(eigenvalues[1])),
        sigma_minor=float(np.sqrt(eigenvalues[0])),
        angular_rms=float(np.sqrt(eigenvalues.sum())))
    return u, uncertainty, statistics


def gravity_to_camera(up_imu, C_CI):
    """
    Express the IMU-frame upward direction
    in camera coordinates.

    C_CI is the camera-from-IMU DCM (a Module 3 Transform or a 3x3 array).
    """
    up = np.asarray(up_imu, dtype=float)
    if up.shape != (3,):
        raise ValueError(f"up_imu must have shape (3,), got {up.shape}")
    return _matrix(C_CI, "C", "I") @ up


def vertical_angular_error(estimated_up, true_up):
    """
    Angular separation in radians.

    atan2(|a x b|, a . b): accurate for the very small angles involved, and
    independent of the vectors' lengths.
    """
    a = np.asarray(estimated_up, dtype=float)
    b = np.asarray(true_up, dtype=float)
    return float(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b)))


# ------------------------------------------------------------- sliding windows
def sliding_up_direction(samples, window):
    """
    estimate_up_direction() over every run of `window` consecutive samples.

    Returns (up, angular_rms, mean): arrays with T - window + 1 rows. Row k
    uses samples k .. k + window - 1, so its estimate becomes available at
    the time of sample k + window - 1 (a trailing average).
    """
    f = _samples(samples, "samples")
    if not 2 <= window <= len(f):
        raise ValueError("window must be between 2 and the number of samples")
    # Cumulative sums give every window mean and second moment in one pass.
    offset = f[0]
    d = f - offset                              # keeps the sums well scaled
    zero1, zero2 = np.zeros((1, 3)), np.zeros((1, 3, 3))
    c1 = np.vstack((zero1, np.cumsum(d, axis=0)))
    c2 = np.vstack((zero2, np.cumsum(d[:, :, None] * d[:, None, :], axis=0)))
    mean_d = (c1[window:] - c1[:-window]) / window
    second = (c2[window:] - c2[:-window]) / window
    covariance = (second - mean_d[:, :, None] * mean_d[:, None, :]) \
        * window / (window - 1)

    mean = mean_d + offset
    magnitude = np.linalg.norm(mean, axis=1)
    up = mean / magnitude[:, None]
    mean_covariance = covariance / window
    trace = np.trace(mean_covariance, axis1=1, axis2=2)
    radial = np.einsum("ni,nij,nj->n", up, mean_covariance, up)
    angular_rms = np.sqrt(np.clip(trace - radial, 0.0, None)) / magnitude
    return up, angular_rms, mean


# ---------------------------------------------------------- quality assessment
QUALITY_CAVEAT = (
    "IMU-only checks cannot prove the instrument is unaccelerated. A constant "
    "horizontal acceleration produces a steady, low-variance reading whose "
    "magnitude differs from g only at second order (a^2 / 2g), and it is "
    "indistinguishable from a tilt of atan(a / g). 'steady' means no "
    "disturbance was detected, not that the vertical is correct.")


@dataclass
class QualityReport:
    verdict: str                       # see assess_measurement_quality()
    detectable_disturbance: bool       # True when any check failed
    checks: dict = field(default_factory=dict)    # name -> True/False/None
    metrics: dict = field(default_factory=dict)
    caveat: str = QUALITY_CAVEAT


def assess_measurement_quality(samples, noise_sigma, expected_g=GRAVITY,
                               g_tolerance=0.02, previous_mean=None,
                               gyro_samples=None, gyro_noise_sigma=None,
                               gyro_rate_tolerance=2e-4, threshold_sigmas=4.0):
    """
    Basic quality indicator for one window of calibrated accelerometer data.

    Checks (True = passed, None = not evaluated):
      magnitude    | |mean f| - expected_g | within g_tolerance plus noise.
                   Sensitive to VERTICAL acceleration; almost blind to
                   horizontal acceleration, which changes the magnitude
                   only by a^2 / 2g.
      variance     per-axis sample variance consistent with noise_sigma.
                   Catches vibration, pulses and oscillation that the
                   window resolves. Blind to anything constant.
      consistency  mean agrees with previous_mean (the preceding window of
                   the same length) within noise. Catches slow change.
      rotation     mean gyro rate below gyro_rate_tolerance plus noise
                   (needs gyro_samples (N, 3) in rad/s).

    Verdicts:
      "disturbed"     magnitude or variance failed: a detectable failure.
      "rotating"      the gyro shows the instrument turning.
      "accelerating"  the mean is changing but the gyro shows no rotation,
                      so the change is translational acceleration.
      "changing"      the mean is changing and there is no gyro: tilt
                      change and slowly varying acceleration cannot be
                      told apart.
      "steady"        nothing detected. NOT a guarantee: see the caveat
                      attached to every report.
    """
    f = _samples(samples, "samples")
    n = len(f)
    if n < 2:
        raise ValueError("at least two samples are needed")
    k = threshold_sigmas
    mean = f.mean(axis=0)
    variance = f.var(axis=0, ddof=1)

    magnitude_error = abs(np.linalg.norm(mean) - expected_g)
    magnitude_limit = g_tolerance + k * noise_sigma / np.sqrt(n)
    # (n-1) s^2 / sigma^2 is chi-square; normal approximation for its tail.
    variance_ratio = float(np.max(variance) / noise_sigma**2)
    variance_limit = 1.0 + k * np.sqrt(2.0 / (n - 1))

    checks = {
        "magnitude": bool(magnitude_error <= magnitude_limit),
        "variance": bool(variance_ratio <= variance_limit),
        "consistency": None,
        "rotation": None,
    }
    metrics = {
        "magnitude_error": float(magnitude_error),
        "magnitude_limit": float(magnitude_limit),
        "variance_ratio": variance_ratio,
        "variance_limit": float(variance_limit),
    }
    if previous_mean is not None:
        step = float(np.linalg.norm(mean - np.asarray(previous_mean, float)))
        step_limit = k * noise_sigma * np.sqrt(2.0 / n)
        checks["consistency"] = bool(step <= step_limit)
        metrics.update(mean_step=step, mean_step_limit=float(step_limit))
    if gyro_samples is not None:
        gyro = _samples(gyro_samples, "gyro_samples")
        rate = float(np.linalg.norm(gyro.mean(axis=0)))
        sigma = 0.0 if gyro_noise_sigma is None else gyro_noise_sigma
        rate_limit = gyro_rate_tolerance + k * sigma / np.sqrt(len(gyro))
        checks["rotation"] = bool(rate <= rate_limit)
        metrics.update(gyro_rate=rate, gyro_rate_limit=float(rate_limit))

    if not (checks["magnitude"] and checks["variance"]):
        verdict = "disturbed"
    elif checks["rotation"] is False:
        verdict = "rotating"
    elif checks["consistency"] is False:
        verdict = "accelerating" if checks["rotation"] is True else "changing"
    else:
        verdict = "steady"
    return QualityReport(verdict=verdict,
                         detectable_disturbance=verdict != "steady",
                         checks=checks, metrics=metrics)
