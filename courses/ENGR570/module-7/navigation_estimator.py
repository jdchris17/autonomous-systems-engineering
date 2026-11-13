"""
navigation_estimator.py - Space Systems Engineering, Module 7

A statistical estimation layer on top of the Module 6 positioning engine.
It consumes the OUTPUTS of the earlier modules (star attitude, IMU vertical,
calibration, Earth orientation) and does not re-implement any of them.

Measurement
-----------
One measurement is a local-vertical direction in celestial coordinates,

    z_k = C_GC,k C_CI u_I,k          (unit vector, frame G)

with a random-error covariance R_k and the Earth orientation C_GE,k at its
assumed UTC timestamp. C_GE converts Earth-fixed coordinates to celestial
coordinates (it is the transpose of Module 6's C_EG).

Model
-----
    h_k(x) = C_GE,k u_E(lat, lon)          x = [lat_rad, lon_rad]

optionally extended with
  * a clock offset dt [s]: the true time is the assumed time + dt;
  * a camera-IMU calibration error d_theta [rad], a rotation vector in
    camera axes with C_CI,used = R(d_theta) C_CI,true (Module 5 convention).

Residuals are formed in the two-dimensional tangent plane of each measured
direction and whitened with the Cholesky factor of their covariance, so the
optimizer minimizes a dimensionless chi-square. Covariances are the inverse
of the whitened normal matrix; they are NOT rescaled by the residuals.

Units: states in radians and seconds; reported positions in degrees;
position covariance in metres squared, ordered (north, east).
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_distribution

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "module-6"))
import celestial_position as cp  # noqa: E402  (adds Modules 1-5 to sys.path)
from astropy.time import Time, TimeDelta  # noqa: E402
from astropy.utils import iers  # noqa: E402

from astronomical_time import EARTH_ROTATION_RATE_UT1, iers_status  # noqa: E402
from gravity_estimator import (  # noqa: E402
    assess_measurement_quality,
    calibrate_accelerometer,
    estimate_up_direction,
    tangent_basis,
)
from reference_frames import WGS84_A, WGS84_E2, rotation_z  # noqa: E402
from rotation_engine import rotation_vector_to_matrix, skew  # noqa: E402
from stellar_attitude import (  # noqa: E402
    attitude_covariance,
    attitude_information,
    wahba_svd,
)

ARCSEC = np.radians(1.0 / 3600.0)

# ---- observability thresholds --------------------------------------------------
# Singular values are taken from the whitened Jacobian after scaling every
# state to "one arcsecond of effect": 1 arcsec of latitude, 1 arcsec of
# great-circle longitude, the clock offset that turns Earth by 1 arcsec, and
# 1 arcsec of calibration rotation. 1/s is then a 1-sigma uncertainty in
# arcseconds along that direction of state space.
RANK_TOLERANCE = 1e-8                 # s_min / s_max below this: rank deficient
WEAK_SIGMA_LIMIT_ARCSEC = 3600.0      # weakest direction worse than 1 degree

# ---- acceptance thresholds -------------------------------------------------------
CHI2_TAIL_PROBABILITY = 1e-3          # reject fits this improbable
OUTLIER_PROBABILITY = 1e-4            # per-measurement residual test (2 dof)
REFERENCE_ERROR_FRACTION = 0.25       # EOP error vs reported sigma before DEGRADED

STANDING_CAVEAT = (
    "Conditional on declared assumptions: the instrument is unaccelerated, "
    "the accelerometer and camera-IMU calibrations are as stated, and local "
    "up is the ellipsoid normal. These measurements cannot verify them: a "
    "constant acceleration or a fixed calibration error shifts every "
    "observation consistently and leaves the residuals clean.")


# ------------------------------------------------------------------- geometry
def unit_up(latitude_rad, longitude_rad):
    """Ellipsoid normal in Earth-fixed coordinates."""
    cos_lat = np.cos(latitude_rad)
    return np.array([cos_lat * np.cos(longitude_rad),
                     cos_lat * np.sin(longitude_rad), np.sin(latitude_rad)])


def north_east_axes(latitude_rad, longitude_rad):
    """Local north and east unit vectors in Earth-fixed coordinates."""
    north = np.array([-np.sin(latitude_rad) * np.cos(longitude_rad),
                      -np.sin(latitude_rad) * np.sin(longitude_rad),
                      np.cos(latitude_rad)])
    east = np.array([-np.sin(longitude_rad), np.cos(longitude_rad), 0.0])
    return north, east


def radians_to_metres(latitude_rad):
    """
    diag(M, N cos lat): converts (d_lat, d_lon) in radians to (north, east)
    in metres on the WGS84 ellipsoid. M is the meridian radius of curvature
    and N the prime-vertical radius.
    """
    s2 = np.sin(latitude_rad) ** 2
    meridian = WGS84_A * (1.0 - WGS84_E2) / (1.0 - WGS84_E2 * s2) ** 1.5
    prime_vertical = WGS84_A / np.sqrt(1.0 - WGS84_E2 * s2)
    return np.diag([meridian, prime_vertical * np.cos(latitude_rad)])


# ----------------------------------------------------------------- measurement
@dataclass
class VerticalMeasurement:
    """One celestial-vertical observation with its random-error model."""
    up_celestial: np.ndarray            # (3,) unit vector, frame G
    covariance: np.ndarray              # (3, 3) random error [rad^2], rank 2
    C_GE: np.ndarray                    # (3, 3) Earth-fixed -> celestial
    utc: str                            # assumed timestamp of C_GE
    C_GC: np.ndarray = None             # celestial-from-camera attitude used
    up_camera: np.ndarray = None        # vertical in camera axes (as believed)
    jacobian_accel: np.ndarray = None   # d z / d accelerometer error [1/(m/s^2)]
    jacobian_calibration: np.ndarray = None   # d z / d calibration error [1/rad]
    quality: object = None              # Module 5 QualityReport, if run
    earth_orientation_source: str = None
    diagnostics: dict = field(default_factory=dict)


def measurement_from_sensors(reference_stars, measured_stars, star_sigma_rad,
                             accel_samples, C_CI, utc, *, earth=None,
                             accel_bias=None, accel_correction=None,
                             accel_noise_sigma=None):
    """
    Turn one set of sensor data into a VerticalMeasurement.

      1. camera attitude from matched stars (Module 4 SVD) and its
         covariance from the attitude information matrix;
      2. IMU-frame up and its covariance from accelerometer samples
         (Module 5);
      3. z = C_GC C_CI u_I;
      4. random-error covariance
             R = C_GI P_u C_GI^T + J_cam P_cam J_cam^T,   J_cam = C_GC [u_C x]

    Only random errors go into R. Fixed errors (accelerometer bias,
    camera-IMU misalignment) are NOT random between observations of one
    instrument; their sensitivities are stored separately so a batch
    estimator can treat them as shared.

    `earth` may be the (C_EG, info) pair from cp.earth_orientation(utc).
    """
    star_sigma = np.broadcast_to(np.asarray(star_sigma_rad, float),
                                 (len(measured_stars),))
    C_CG = wahba_svd(reference_stars, measured_stars, 1.0 / star_sigma**2)
    P_camera = attitude_covariance(attitude_information(measured_stars, star_sigma))

    samples = np.asarray(accel_samples, dtype=float)
    if accel_bias is not None or accel_correction is not None:
        samples = calibrate_accelerometer(
            samples, np.zeros(3) if accel_bias is None else accel_bias,
            np.eye(3) if accel_correction is None else accel_correction)
    up_imu, up_uncertainty, statistics = estimate_up_direction(samples)
    quality = (None if accel_noise_sigma is None
               else assess_measurement_quality(samples, accel_noise_sigma))

    c_ci = cp._dcm(C_CI, "C", "I")
    c_gc = C_CG.T
    up_camera = c_ci @ up_imu
    up_celestial = c_gc @ up_camera

    c_gi = c_gc @ c_ci
    j_camera = c_gc @ skew(up_camera)
    covariance = (c_gi @ up_uncertainty.covariance @ c_gi.T
                  + j_camera @ P_camera @ j_camera.T)

    if earth is None:
        earth = cp.earth_orientation(utc)
    C_EG, info = earth
    return VerticalMeasurement(
        up_celestial=up_celestial, covariance=covariance,
        C_GE=np.asarray(C_EG.matrix).T, utc=info["timestamp_utc"], C_GC=c_gc,
        up_camera=up_camera,
        jacobian_accel=c_gi @ (np.eye(3) - np.outer(up_imu, up_imu))
        / statistics.magnitude,
        jacobian_calibration=-j_camera, quality=quality,
        earth_orientation_source=info["earth_orientation_source"],
        diagnostics={"star_count": len(measured_stars),
                     "accel_sample_count": statistics.count,
                     "specific_force_magnitude": statistics.magnitude})


# ------------------------------------------------------ required: model + Jacobian
def predict_celestial_vertical(state, C_GE):
    """
    Predict celestial-frame local up from
    geodetic latitude and longitude.

    state = [latitude_rad, longitude_rad]
    C_GE: Earth-fixed -> celestial rotation at the observation time.
    """
    return np.asarray(C_GE) @ unit_up(state[0], state[1])


def celestial_vertical_jacobian(state, C_GE):
    """
    Analytic derivative of the predicted
    celestial vertical with respect to
    latitude and longitude.

    Returns a 3x2 matrix [d u_G / d lat, d u_G / d lon]:
        d u_E / d lat = north,     d u_E / d lon = cos(lat) * east
    The longitude column shrinks to zero at the poles, which is the
    coordinate singularity of Module 6.
    """
    north, east = north_east_axes(state[0], state[1])
    return np.asarray(C_GE) @ np.column_stack((north, np.cos(state[0]) * east))


# ------------------------------------------------------------------ estimation
@dataclass
class BatchResult:
    latitude_deg: float
    longitude_deg: float
    state: np.ndarray
    state_names: list
    covariance: object                  # full state covariance, or None
    position_covariance_ne: object      # 2x2 [m^2] (north, east), or None
    residuals: np.ndarray               # whitened, stacked
    chi2: float
    dof: int
    jacobian: np.ndarray                # whitened, at the solution
    singular_values: np.ndarray         # of the scaled whitened Jacobian
    observability: str
    converged: bool
    clock_offset_s: object = None
    calibration_rad: object = None
    diagnostics: dict = field(default_factory=dict)

    @property
    def lat_lon(self):
        return self.latitude_deg, self.longitude_deg

    @property
    def reduced_chi2(self):
        return self.chi2 / self.dof if self.dof > 0 else float("nan")


def state_scales(latitude_rad, state_names):
    """State change that produces about one arcsecond of measurement effect."""
    scale = {"latitude": ARCSEC,
             "longitude": ARCSEC / max(np.cos(latitude_rad), 1e-12),
             "clock": ARCSEC / EARTH_ROTATION_RATE_UT1}
    return np.array([scale.get(name, ARCSEC) for name in state_names])


def classify_observability(singular_values):
    """
    "well conditioned", "poorly conditioned" or "rank deficient", from the
    singular values of the scaled whitened Jacobian.
    """
    s = np.asarray(singular_values, dtype=float)
    if s.size == 0 or s.max() == 0.0 or s.min() <= RANK_TOLERANCE * s.max():
        return "rank deficient"
    if 1.0 / s.min() > WEAK_SIGMA_LIMIT_ARCSEC:
        return "poorly conditioned"
    return "well conditioned"


class _Problem:
    """Residuals and Jacobian for a batch of measurements."""

    def __init__(self, measurements, estimate_clock, estimate_calibration,
                 clock_model, shared_errors, calibration_prior_sigma_rad):
        if not measurements:
            raise ValueError("at least one measurement is required")
        if clock_model not in ("simple", "full"):
            raise ValueError("clock_model must be 'simple' or 'full'")
        self.m = list(measurements)
        self.n = len(self.m)
        self.estimate_clock = estimate_clock
        self.estimate_calibration = estimate_calibration
        self.clock_model = clock_model
        self.names = ["latitude", "longitude"]
        if estimate_clock:
            self.names.append("clock")
        if estimate_calibration:
            if any(k.C_GC is None for k in self.m):
                raise ValueError("calibration states need each measurement's C_GC")
            self.names += ["cal_x", "cal_y", "cal_z"]

        self.z = np.array([k.up_celestial / np.linalg.norm(k.up_celestial)
                           for k in self.m])
        self.basis = np.array([tangent_basis(z) for z in self.z])      # (N, 2, 3)

        # Full measurement covariance in stacked tangent coordinates.
        R = np.zeros((2 * self.n, 2 * self.n))
        for k, meas in enumerate(self.m):
            block = self.basis[k] @ np.asarray(meas.covariance) @ self.basis[k].T
            R[2 * k:2 * k + 2, 2 * k:2 * k + 2] = 0.5 * (block + block.T)
        for name, prior in (shared_errors or {}).items():
            attribute = {"accelerometer_bias": "jacobian_accel",
                         "calibration": "jacobian_calibration"}.get(name)
            if attribute is None:
                raise ValueError(f"unknown shared error {name!r}")
            if any(getattr(k, attribute) is None for k in self.m):
                raise ValueError(f"measurements lack the sensitivity for {name!r}")
            A = np.vstack([self.basis[k] @ getattr(meas, attribute)
                           for k, meas in enumerate(self.m)])          # (2N, 3)
            R += A @ np.asarray(prior, dtype=float) @ A.T
        try:
            self.L = cholesky(R, lower=True)
        except np.linalg.LinAlgError as error:
            raise ValueError("measurement covariance is not positive definite "
                             "in the tangent plane") from error
        self.prior_sigma = calibration_prior_sigma_rad

    # -- model pieces ------------------------------------------------------------
    def _c_ge(self, k, dt):
        meas = self.m[k]
        if not self.estimate_clock or dt == 0.0:
            return meas.C_GE
        if self.clock_model == "simple":
            # Earth-fixed axes turned further about the pole by omega * dt.
            return meas.C_GE @ rotation_z(EARTH_ROTATION_RATE_UT1 * dt)
        time = Time(meas.utc, scale="utc") + TimeDelta(dt, format="sec")
        return np.asarray(cp.earth_orientation(time)[0].matrix).T

    def _split(self, x):
        index = 2
        dt = 0.0
        if self.estimate_clock:
            dt = float(x[index])
            index += 1
        cal = np.asarray(x[index:index + 3]) if self.estimate_calibration else None
        return x[0], x[1], dt, cal

    def predict(self, x):
        lat, lon, dt, cal = self._split(x)
        up_e = unit_up(lat, lon)
        out = np.empty((self.n, 3))
        for k, meas in enumerate(self.m):
            h = self._c_ge(k, dt) @ up_e
            if cal is not None:
                h = meas.C_GC @ (rotation_vector_to_matrix(cal) @ (meas.C_GC.T @ h))
            out[k] = h
        return out

    def raw_jacobian(self, x):
        """Unwhitened tangent-plane Jacobian, shape (2N, n_states)."""
        lat, lon, dt, cal = self._split(x)
        up_e = unit_up(lat, lon)
        rows = []
        for k, meas in enumerate(self.m):
            c_ge = self._c_ge(k, dt)
            columns = [celestial_vertical_jacobian([lat, lon], c_ge)]
            if self.estimate_clock:
                if self.clock_model == "simple":
                    d_dt = EARTH_ROTATION_RATE_UT1 * (
                        c_ge @ np.cross([0.0, 0.0, 1.0], up_e))
                else:
                    step = 1.0
                    d_dt = (self._c_ge(k, dt + step) - self._c_ge(k, dt - step)) \
                        @ up_e / (2.0 * step)
                columns.append(d_dt[:, None])
            if cal is not None:
                rotation = meas.C_GC @ rotation_vector_to_matrix(cal) @ meas.C_GC.T
                columns = [rotation @ c for c in columns]
                in_camera = rotation_vector_to_matrix(cal) @ (meas.C_GC.T @ (c_ge @ up_e))
                columns.append(-meas.C_GC @ skew(in_camera))
            rows.append(self.basis[k] @ np.hstack(columns))
        return np.vstack(rows)

    def whiten(self, stacked):
        return solve_triangular(self.L, stacked, lower=True)

    def residuals(self, x):
        raw = np.einsum("kij,kj->ki", self.basis, self.predict(x) - self.z).ravel()
        r = self.whiten(raw)
        if self.estimate_calibration and self.prior_sigma is not None:
            r = np.concatenate((r, self._split(x)[3] / self.prior_sigma))
        return r

    def jacobian(self, x):
        J = self.whiten(self.raw_jacobian(x))
        if self.estimate_calibration and self.prior_sigma is not None:
            prior = np.zeros((3, len(self.names)))
            prior[:, -3:] = np.eye(3) / self.prior_sigma
            J = np.vstack((J, prior))
        return J


def batch_position_estimate(
    measurements,
    initial_position,
    *,
    estimate_clock=False,
    estimate_calibration=False,
    clock_model="simple",
    shared_errors=None,
    calibration_prior_sigma_rad=None,
):
    """
    Weighted nonlinear least-squares estimate.

    Return estimated position, covariance,
    residual statistics, and diagnostics.

    measurements: list of VerticalMeasurement.
    initial_position: (latitude_deg, longitude_deg) starting guess.
    estimate_clock: add a clock offset state [s]. clock_model "simple" turns
        the Earth-fixed frame about its z axis at the constant sidereal
        rate; "full" re-evaluates Module 2's complete Earth orientation.
    estimate_calibration: add a camera-IMU rotation-vector error (3 states),
        optionally with a Gaussian prior of calibration_prior_sigma_rad.
    shared_errors: {"accelerometer_bias": P_b, "calibration": P_cal}, 3x3
        prior covariances of errors that are FIXED across the batch. Their
        cross-observation covariance A P A^T is added to the measurement
        covariance before whitening, so the reported uncertainty does not
        shrink with the number of observations as if they were independent.

    The covariance is returned only when the scaled Jacobian is well
    conditioned. If the state is not determined by the measurements the
    result says so (observability "rank deficient" or "poorly conditioned")
    and covariance is None; a nearly singular normal matrix is not inverted.
    """
    problem = _Problem(measurements, estimate_clock, estimate_calibration,
                       clock_model, shared_errors, calibration_prior_sigma_rad)
    x0 = np.zeros(len(problem.names))
    x0[0], x0[1] = np.radians(initial_position[0]), np.radians(initial_position[1])
    scales = state_scales(x0[0], problem.names)

    fit = least_squares(problem.residuals, x0, jac=problem.jacobian,
                        x_scale=scales, method="trf", xtol=1e-15, ftol=1e-15,
                        gtol=1e-15, max_nfev=200)
    x = fit.x.copy()
    x[1] = np.radians(float(cp.wrap_longitude(np.degrees(x[1]))))
    residuals = problem.residuals(x)
    J = problem.jacobian(x)
    scales = state_scales(x[0], problem.names)
    singular_values = np.linalg.svd(J * scales, compute_uv=False)
    observability = classify_observability(singular_values)

    covariance = position_ne = None
    if observability == "well conditioned":
        covariance = np.linalg.inv(J.T @ J)
        T = radians_to_metres(x[0])
        position_ne = T @ covariance[:2, :2] @ T.T

    n_rows = len(residuals)
    per_measurement = np.sum(residuals[:2 * problem.n].reshape(-1, 2) ** 2, axis=1)
    information = np.sum(J[:2 * problem.n, :2].reshape(problem.n, 2, 2) ** 2,
                         axis=(1, 2))
    _, _, dt, cal = problem._split(x)
    step = fit.x - x0
    return BatchResult(
        latitude_deg=float(np.degrees(x[0])), longitude_deg=float(np.degrees(x[1])),
        state=x, state_names=list(problem.names), covariance=covariance,
        position_covariance_ne=position_ne, residuals=residuals,
        chi2=float(residuals @ residuals), dof=int(n_rows - len(x)), jacobian=J,
        singular_values=singular_values, observability=observability,
        converged=bool(fit.success and fit.status > 0),
        clock_offset_s=dt if estimate_clock else None,
        calibration_rad=cal if estimate_calibration else None,
        diagnostics={
            "function_evaluations": int(fit.nfev),
            "optimizer_message": fit.message,
            "measurement_chi2": per_measurement,
            "information_fraction": information / information.sum(),
            "state_scales": scales,
            "measurement_count": problem.n,
            "shared_errors": sorted(shared_errors) if shared_errors else [],
            "initial_step_arcsec": float(np.linalg.norm(step[:2]) / ARCSEC),
        })


# ------------------------------------------------------------------ statistics
def nees(north_east_error_m, position_covariance_ne):
    """Normalized estimation error squared, dp^T P^-1 dp (2 dof)."""
    dp = np.asarray(north_east_error_m, dtype=float)
    return float(dp @ np.linalg.solve(position_covariance_ne, dp))


def inside_confidence_ellipse(north_east_error_m, position_covariance_ne,
                              probability=0.95):
    return nees(north_east_error_m, position_covariance_ne) <= \
        chi2_distribution.ppf(probability, 2)


# ------------------------------------------------------------ reference status
def reference_status(utc):
    """
    Earth-orientation data status for a timestamp, with a rough UT1
    uncertainty.

    For predicted values the uncertainty follows the growth law quoted in
    IERS Bulletin A, about 0.00025 * D^0.75 seconds for D days past the last
    measured value. That is an approximation for budgeting, not a guarantee.
    """
    time = utc if isinstance(utc, Time) else Time(utc, scale="utc")
    status = iers_status(time)
    table = iers.earth_orientation_table.get()
    flags = np.array(table["UT1Flag"])
    mjd = np.array(table["MJD"].value)
    last_measured = float(mjd[flags != "P"][-1])
    days_predicted = max(0.0, status["mjd"] - last_measured)
    if not status["available"]:
        ut1_sigma = 0.9                 # only |UT1-UTC| < 0.9 s is guaranteed
    elif status["is_prediction"]:
        ut1_sigma = 0.00025 * days_predicted ** 0.75
    else:
        ut1_sigma = 2e-5
    return {"available": status["available"], "source": status["source"],
            "is_prediction": status["is_prediction"],
            "days_predicted": days_predicted, "ut1_sigma_s": ut1_sigma,
            "utc": time.utc.isot}


# ---------------------------------------------------------- acceptance policy
@dataclass
class NavigationAssessment:
    status: str                         # VALID / DEGRADED / UNOBSERVABLE / REJECTED
    reasons: list
    metrics: dict
    caveat: str = STANDING_CAVEAT


def assess_navigation_solution(
    estimate,
    covariance,
    jacobian,
    residuals,
    measurement_quality,
    reference_status,
    *,
    converged=True,
    state_names=("latitude", "longitude"),
    measurement_count=None,
):
    """
    Assess numerical convergence, observability,
    statistical consistency, and known limitations.

    Return status, reasons, and diagnostic metrics.

    estimate: state vector [lat_rad, lon_rad, ...].
    covariance: state covariance, or None if it was withheld.
    jacobian, residuals: WHITENED, at the solution.
    measurement_quality: one entry per measurement, each None, a Module 5
        QualityReport, or anything with a `detectable_disturbance` flag.
    reference_status: dict from reference_status(), describing the
        Earth-orientation data.

    Statuses
      REJECTED      a measurement or model validity check failed.
      UNOBSERVABLE  the measurements do not uniquely constrain the state.
      DEGRADED      usable, but uncertainty or reference data limit confidence.
      VALID         observable, converged and consistent with the declared
                    measurement assumptions. This is conditional on those
                    assumptions (see `caveat`); it is not a statement that
                    the position is correct.
    """
    estimate = np.asarray(estimate, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    J = np.asarray(jacobian, dtype=float)
    names = list(state_names)
    n_measurements = (len(measurement_quality) if measurement_count is None
                      else measurement_count)
    rejected, unobservable, degraded, notes = [], [], [], []
    metrics = {}

    # -- numerical validity --------------------------------------------------------
    if not converged:
        rejected.append("optimizer did not converge")
    if not (np.all(np.isfinite(estimate)) and np.all(np.isfinite(residuals))
            and np.all(np.isfinite(J))):
        rejected.append("non-finite values in the estimate, residuals or Jacobian")

    # -- observability ---------------------------------------------------------------
    scales = state_scales(estimate[0], names)
    singular_values = np.linalg.svd(J * scales, compute_uv=False)
    observability = classify_observability(singular_values)
    metrics["singular_values"] = singular_values
    metrics["observability"] = observability
    metrics["condition_number"] = float(
        singular_values.max() / max(singular_values.min(), 1e-300))
    if observability != "well conditioned":
        _, _, vt = np.linalg.svd(J * scales)
        weakest = vt[-1]
        involved = [names[i] for i in np.flatnonzero(np.abs(weakest) > 0.2)]
        metrics["weakest_direction"] = {
            name: round(float(value), 3) + 0.0
            for name, value in zip(names, weakest)}
        unobservable.append(
            f"state is {observability}: the measurements do not separate "
            f"{' / '.join(involved)}")
    if covariance is None:
        notes.append("no covariance is reported")

    # -- measurement validity ----------------------------------------------------------
    for index, quality in enumerate(measurement_quality or []):
        if quality is not None and getattr(quality, "detectable_disturbance", False):
            rejected.append(f"measurement {index}: accelerometer quality check "
                            f"failed ({getattr(quality, 'verdict', 'flagged')})")

    # -- statistical consistency --------------------------------------------------------
    chi2 = float(residuals @ residuals)
    dof = len(residuals) - len(estimate)
    metrics.update(chi2=chi2, dof=dof)
    pairs = residuals[:2 * n_measurements].reshape(-1, 2)
    per_measurement = np.sum(pairs**2, axis=1)
    metrics["measurement_chi2"] = per_measurement
    if dof > 0:
        metrics["reduced_chi2"] = chi2 / dof
        metrics["chi2_upper_tail_probability"] = float(
            chi2_distribution.sf(chi2, dof))
        if chi2 > chi2_distribution.ppf(1.0 - CHI2_TAIL_PROBABILITY, dof):
            rejected.append(
                f"residuals inconsistent with the noise model: chi2 = "
                f"{chi2:.1f} for {dof} degrees of freedom")
        elif chi2 < chi2_distribution.ppf(CHI2_TAIL_PROBABILITY, dof):
            degraded.append(
                f"residuals smaller than the noise model predicts (chi2 = "
                f"{chi2:.1f} for {dof} dof): stated uncertainties may be wrong")
        outlier_limit = chi2_distribution.ppf(1.0 - OUTLIER_PROBABILITY, 2)
        outliers = np.flatnonzero(per_measurement > outlier_limit)
        if outliers.size:
            rejected.append("inconsistent measurement(s) "
                            f"{outliers.tolist()}: normalized residual above "
                            f"{outlier_limit:.1f}")
    else:
        degraded.append("no redundant measurements: residuals cannot test "
                        "the fit (zero degrees of freedom)")

    # -- reference data -------------------------------------------------------------------
    reference = reference_status or {}
    metrics["reference"] = dict(reference)
    if reference and not reference.get("available", True):
        degraded.append("no Earth-orientation data for this time: UT1 is only "
                        "known to within 0.9 s")
    ut1_sigma = reference.get("ut1_sigma_s")
    if ut1_sigma is not None:
        east_error = (EARTH_ROTATION_RATE_UT1 * ut1_sigma
                      * radians_to_metres(estimate[0])[1, 1])
        metrics["reference_east_error_m"] = float(east_error)
        if covariance is not None and observability == "well conditioned":
            T = radians_to_metres(estimate[0])
            position = T @ np.asarray(covariance)[:2, :2] @ T.T
            east_sigma = float(np.sqrt(position[1, 1]))
            metrics["east_sigma_m"] = east_sigma
            metrics["north_sigma_m"] = float(np.sqrt(position[0, 0]))
            if east_error > REFERENCE_ERROR_FRACTION * east_sigma:
                degraded.append(
                    f"Earth-orientation uncertainty ({ut1_sigma:.3g} s of UT1, "
                    f"{east_error:.0f} m east) is not negligible beside the "
                    f"reported {east_sigma:.0f} m east sigma and is not in the "
                    f"covariance ({reference.get('source', 'unknown source')})")
    if np.cos(estimate[0]) < 0.02:
        degraded.append("within about 1 degree of a pole: longitude is poorly "
                        "defined; use north/east metres")

    if rejected:
        status, reasons = "REJECTED", rejected + unobservable + degraded
    elif unobservable:
        status, reasons = "UNOBSERVABLE", unobservable + degraded
    elif degraded:
        status, reasons = "DEGRADED", degraded
    else:
        status = "VALID"
        reasons = ["converged, observable, and residuals consistent with the "
                   "declared noise model"]
    reasons = reasons + notes
    if status in ("VALID", "DEGRADED"):
        reasons.append("stationarity and calibration are assumed, not verified")
    return NavigationAssessment(status=status, reasons=reasons, metrics=metrics)


def assess_batch_result(result, measurements, reference=None):
    """assess_navigation_solution() applied to a BatchResult."""
    if reference is None:
        reference = reference_status(measurements[0].utc)
    return assess_navigation_solution(
        result.state, result.covariance, result.jacobian, result.residuals,
        [m.quality for m in measurements], reference,
        converged=result.converged, state_names=result.state_names,
        measurement_count=len(measurements))
