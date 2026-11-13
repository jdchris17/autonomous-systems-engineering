"""
pipeline.py - the complete terrestrial celestial fix behind one function.

    solve_navigation_fix(observation_batch, config, references)

The function coordinates the subsystems built in Modules 1-8. It contains
no celestial mechanics and no estimation mathematics of its own:

  Stage 1  ingest and validate the batch              (this file)
  Stage 2  celestial reference directions             references (ERFA, Module 8)
  Stage 3  camera attitude from matched stars         Module 4
  Stage 4  local vertical and its quality             Module 5
  Stage 5  terrestrial position                       Modules 6 and 7
  Stage 6  navigation integrity                       Module 7 + integrity.py

Stages 2-5 are repeated because the reference directions depend on where
the observer is (refraction acts along the local vertical; diurnal
aberration depends on the site). This is Module 8's iteration: start with
no position-dependent correction, solve, recompute the references at the
solution, and repeat until the position stops moving.

The function never raises for bad data. Anything it cannot use comes back
as a NavigationResult with status REJECTED or UNOBSERVABLE and a reason.
"""

import traceback
from dataclasses import dataclass, field, replace

import numpy as np

from . import integrity
from .astrometry import STANDARD_ATMOSPHERE
from .attitude import (
    InsufficientGeometryError,
    attitude_covariance,
    attitude_information,
    centroid_sigma_to_angle,
    ransac_attitude,
    robust_wahba,
    wahba_svd,
)
from .contracts import (
    REJECTED,
    UNOBSERVABLE,
    NavigationConfig,
    NavigationInputError,
    NavigationResult,
)
from .estimation import (
    ARCSEC,
    _Problem,
    assess_batch_result,
    batch_position_estimate,
    classify_observability,
    measurement_from_sensors,
    north_east_axes,
    radians_to_metres,
    state_scales,
    unit_up,
)
from .frames import (
    WGS84_A,
    WGS84_E2,
    Transform,
    check_rotation_matrix,
    rotation_vector_to_matrix,
)
from .geodesy import earth_fixed_up_to_geodetic, horizontal_position_error
from .gravity import (
    assess_measurement_quality,
    calibrate_accelerometer,
    normal_gravity,
)
from .timekeeping import EARTH_ROTATION_RATE_UT1

NONPHYSICAL_FORCE_BAND = 1.0          # |f| further than this from g: bad data
POLAR_COS_LATITUDE = 0.02             # within ~1.1 deg of a pole
INITIAL_POSITION_TRUST_M = 100e3      # hint must agree with the data this well
STAR_CHI2_TAIL = 1e-3
# Rotation used near the poles: Earth-fixed axes turned 90 deg about y, so
# the pole lies on the "equator" of the working frame and Module 7's
# latitude/longitude state is nowhere near its singularity.
_POLAR_FRAME = rotation_vector_to_matrix([0.0, np.pi / 2.0, 0.0])


# ------------------------------------------------------------------ working data
@dataclass
class _Exposure:
    index: int
    eop: object
    C_EG: np.ndarray
    star_ids: list
    star_index: np.ndarray               # rows of the catalog
    rays: np.ndarray                     # (N, 3) camera-frame unit vectors
    star_sigma: np.ndarray               # (N,) rad
    accel: np.ndarray                    # calibrated samples
    gyro: object
    notes: list = field(default_factory=list)
    # filled by the stages
    reference: np.ndarray = None
    used: np.ndarray = None              # boolean (N,): stars in the final fit
    rejected_stars: list = field(default_factory=list)
    below_altitude: int = 0
    attitude_sigma_arcsec: float = None
    star_chi2: float = None
    star_dof: int = 0
    star_residuals: np.ndarray = None
    quality: object = None
    measurement: object = None
    mean_zenith_rad: float = None
    boresight_enu: np.ndarray = None


class _Dropped(Exception):
    """One exposure cannot be used; the batch continues without it."""


# ============================================================ public interface
def solve_navigation_fix(observation_batch, config, references):
    """
    Determine terrestrial position from a batch of
    celestial and inertial observations.

    Inputs:
      observation_batch: camera/star and IMU measurements
      config: instrument calibration and estimator settings
      references: locally available astronomical models

    Returns:
      navigation result, uncertainty, integrity status,
      and diagnostic provenance.
    """
    context = {"excluded": [], "iterations": [], "notes": []}
    try:
        return _solve(observation_batch, config, references, context)
    except NavigationInputError as error:
        return _no_fix(REJECTED, [f"invalid input: {error}"], observation_batch,
                       references, context)
    except Exception as error:      # fail safe: never hand back a traceback
        result = _no_fix(
            REJECTED,
            [f"internal failure ({type(error).__name__}: {error}); no position "
             "is reported"], observation_batch, references, context)
        result.diagnostics["traceback"] = traceback.format_exc()
        return result


# ================================================================== the stages
def _solve(batch, config, references, context):
    _check_config(config)
    exposures = _ingest(batch, config, references, context)            # stage 1
    if not exposures:
        return _nothing_usable(batch, references, context)

    # ---- first position: no position-dependent correction, no rejection ------
    # Always derived from the measurements. A supplied initial position is a
    # hint: it is used only if the measurements agree with it, because a wrong
    # one would put the reference stars below the horizon and the expected
    # gravity at the wrong latitude.
    exposures = _run(exposures, context, lambda e: _attitude_and_vertical(
        e, config, references, None, robust=False))
    if not exposures:
        return _nothing_usable(batch, references, context)
    position = _mean_single_fix(exposures)
    context["iterations"].append({"stage": "initial, uncorrected references",
                                  "position": position})
    if config.initial_position is not None:
        hint = tuple(float(v) for v in config.initial_position)
        gap = _distance(hint, position)
        if gap <= INITIAL_POSITION_TRUST_M:
            position = hint
        else:
            context["notes"].append(
                f"supplied initial position is {gap / 1000.0:,.0f} km from the "
                "position the measurements indicate and was not used")

    # ---- stage 4b: accelerometer quality, before the vertical is trusted -----
    for exposure in exposures:
        exposure.quality = assess_measurement_quality(
            exposure.accel, config.accel_noise_sigma,
            expected_g=normal_gravity(position[0], config.height_m),
            g_tolerance=config.gravity_tolerance, gyro_samples=exposure.gyro,
            gyro_noise_sigma=config.gyro_noise_sigma)
    disturbed = [e for e in exposures if e.quality.detectable_disturbance]
    for exposure in disturbed:
        context["excluded"].append({
            "observation": exposure.index, "category": "motion",
            "reason": f"accelerometer quality check failed "
                      f"({exposure.quality.verdict})"})
    exposures = [e for e in exposures if not e.quality.detectable_disturbance]
    if not exposures:
        return _no_fix(REJECTED, ["every accelerometer window shows detectable "
                                  "motion: no gravity-based position"], batch,
                       references, context)

    # ---- stages 2-5, iterated on the candidate position -----------------------
    solution, converged = None, False
    for _ in range(config.max_iterations):
        exposures = _run(exposures, context, lambda e: _attitude_and_vertical(
            e, config, references, position, robust=True))
        if not exposures:
            return _nothing_usable(batch, references, context)
        solution = _position(exposures, position, config)              # stage 5
        move = _distance(solution.geodetic, position)
        context["iterations"].append({"stage": "corrected references",
                                      "position": solution.geodetic,
                                      "update_m": move})
        position = solution.geodetic
        if move < config.convergence_m:
            converged = True
            break

    return integrity.build_result(batch, config, references, exposures, solution,
                                  converged, context)               # stage 6


# ------------------------------------------------------------- stage 1: ingest
def _check_config(config):
    if not isinstance(config, NavigationConfig):
        raise NavigationInputError("config is not a NavigationConfig")
    if config.C_CI is None:
        raise NavigationInputError("config.C_CI (camera-IMU alignment) is missing")
    try:
        check_rotation_matrix(_matrix(config.C_CI))
    except Exception as error:
        raise NavigationInputError(f"config.C_CI is not a rotation: {error}")
    for name in ("centroid_sigma_px", "accel_noise_sigma"):
        value = getattr(config, name)
        if not (np.isfinite(value) and value > 0.0):
            raise NavigationInputError(f"config.{name} must be positive")
    for name in ("accel_bias_sigma", "misalignment_sigma_arcsec",
                 "principal_point_sigma_px", "clock_sigma_s",
                 "deflection_sigma_arcsec"):
        value = getattr(config, name)
        if not (np.isfinite(value) and value >= 0.0):
            raise NavigationInputError(f"config.{name} must be zero or positive")
    if config.initial_position is not None:
        lat, lon = config.initial_position
        if not (np.isfinite(lat) and np.isfinite(lon) and abs(lat) <= 90.0):
            raise NavigationInputError("config.initial_position is not a "
                                       "latitude and longitude")


def _matrix(transform):
    return np.asarray(transform.matrix if isinstance(transform, Transform)
                      else transform, dtype=float)


def _ingest(batch, config, references, context):
    observations = getattr(batch, "observations", None)
    if observations is None or len(observations) == 0:
        raise NavigationInputError("the batch contains no observations")
    camera = config.camera
    bias = np.asarray(config.accel_bias, dtype=float)
    correction = (np.eye(3) if config.accel_correction is None
                  else np.asarray(config.accel_correction, dtype=float))
    exposures = []
    for index, observation in enumerate(observations):
        try:
            exposures.append(_ingest_one(index, observation, config, references,
                                         camera, bias, correction))
        except _Dropped as dropped:
            context["excluded"].append({"observation": index,
                                        "category": dropped.args[1],
                                        "reason": dropped.args[0]})
    return exposures


def _ingest_one(index, observation, config, references, camera, bias, correction):
    invalid = "invalid input"
    try:
        eop = references.eop(observation.time_utc)
    except Exception as error:
        raise _Dropped(f"timestamp {observation.time_utc!r} is not a UTC time "
                       f"({error})", invalid)

    try:
        pixels = np.asarray(observation.centroids_px, dtype=float)
        accel = np.asarray(observation.accel_samples, dtype=float)
    except (TypeError, ValueError) as error:
        raise _Dropped(f"measurements are not numeric ({error})", invalid)
    ids = list(observation.star_ids)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise _Dropped(f"centroids must have shape (N, 2), got {pixels.shape}",
                       invalid)
    if len(ids) != len(pixels):
        raise _Dropped(f"{len(pixels)} centroids but {len(ids)} star identities",
                       invalid)
    if len(set(ids)) != len(ids):
        raise _Dropped("the same catalog star is matched to two centroids", invalid)
    if not np.all(np.isfinite(pixels)):
        raise _Dropped("centroids contain NaN or infinite values", invalid)
    on_sensor = ((pixels[:, 0] >= 0) & (pixels[:, 0] <= camera.width)
                 & (pixels[:, 1] >= 0) & (pixels[:, 1] <= camera.height))
    if not np.all(on_sensor):
        raise _Dropped(f"{int((~on_sensor).sum())} centroid(s) lie outside the "
                       f"{camera.width} x {camera.height} sensor", invalid)
    if accel.ndim != 2 or accel.shape[1] != 3 or len(accel) < 2:
        raise _Dropped("accelerometer samples must have shape (M >= 2, 3), got "
                       f"{accel.shape}", invalid)
    if not np.all(np.isfinite(accel)):
        raise _Dropped("accelerometer samples contain NaN or infinite values",
                       invalid)
    gyro = observation.gyro_samples
    if gyro is not None:
        gyro = np.asarray(gyro, dtype=float)
        if gyro.ndim != 2 or gyro.shape[1] != 3 or not np.all(np.isfinite(gyro)):
            raise _Dropped("gyro samples are malformed", invalid)

    calibrated = calibrate_accelerometer(accel, bias, correction)
    magnitude = float(np.linalg.norm(calibrated.mean(axis=0)))
    if abs(magnitude - 9.81) > NONPHYSICAL_FORCE_BAND:
        raise _Dropped(f"mean specific force is {magnitude:.2f} m/s^2, not a "
                       "stationary instrument on Earth", invalid)

    star_index, known = references.indices(ids)
    notes = []
    if not np.all(known):
        missing = [i for i, k in zip(ids, known) if not k]
        notes.append(f"{len(missing)} star(s) not in the catalog were ignored: "
                     f"{missing[:5]}")
    if int(known.sum()) < config.min_stars:
        raise _Dropped(f"{int(known.sum())} matched catalog stars, fewer than the "
                       f"{config.min_stars} required: attitude unobservable",
                       "insufficient stars")
    rays = camera.unproject(pixels[known])
    return _Exposure(
        index=index, eop=eop, C_EG=eop.earth_from_celestial(),
        star_ids=[i for i, k in zip(ids, known) if k], star_index=star_index,
        rays=rays,
        star_sigma=np.asarray(centroid_sigma_to_angle(camera, rays,
                                                      config.centroid_sigma_px)),
        accel=calibrated, gyro=gyro, notes=notes)


def _run(exposures, context, stage):
    """Apply a stage to every exposure, setting aside the ones it drops."""
    kept = []
    for exposure in exposures:
        try:
            stage(exposure)
            kept.append(exposure)
        except _Dropped as dropped:
            context["excluded"].append({"observation": exposure.index,
                                        "category": dropped.args[1],
                                        "reason": dropped.args[0]})
    return kept


# ------------------------------------------- stages 2-4 for one exposure
def effective_atmosphere(config):
    """(atmosphere used for refraction or None, fractional sigma, description)."""
    if not config.apply_refraction:
        return None, 0.0, "refraction not applied (configured off)"
    if config.atmosphere is None:
        return (STANDARD_ATMOSPHERE, config.assumed_atmosphere_fraction_sigma,
                "no weather data: standard atmosphere assumed")
    a = config.atmosphere
    return (a, config.refraction_fraction_sigma,
            f"refraction for {a.pressure_hpa:.0f} hPa, {a.temperature_c:.0f} C")


def _attitude_and_vertical(e, config, references, position, *, robust):
    # ---- stage 2: reference directions ------------------------------------------
    usable = np.ones(len(e.rays), dtype=bool)
    if position is None:
        reference = references.propagated(e.star_index, e.eop)
    else:
        atmosphere, _, _ = effective_atmosphere(config)
        reference, enu = references.observed(
            e.star_index, e.eop, position[0], position[1], config.height_m,
            atmosphere)
        altitude = np.degrees(np.arcsin(np.clip(enu[:, 2], -1.0, 1.0)))
        usable = altitude >= config.min_star_altitude_deg
        e.below_altitude = int((~usable).sum())
        if int(usable.sum()) < config.min_stars:
            raise _Dropped(
                f"only {int(usable.sum())} matched stars above "
                f"{config.min_star_altitude_deg:g} deg altitude: attitude "
                "unobservable", "insufficient stars")
        e.mean_zenith_rad = float(np.mean(np.arccos(np.clip(enu[usable, 2], -1, 1))))
        mean_enu = enu[usable].mean(axis=0)
        e.boresight_enu = mean_enu / np.linalg.norm(mean_enu)
    e.reference = reference

    # ---- stage 3: camera attitude -------------------------------------------------
    r, b, sigma = reference[usable], e.rays[usable], e.star_sigma[usable]
    try:
        if not robust:
            wahba_svd(r, b, 1.0 / sigma**2)
            inliers = np.ones(len(r), dtype=bool)
            C = None
        else:
            fit = robust_wahba(r, b, sigma, config.outlier_false_alarm,
                               config.min_stars)
            rejected_fraction = 1.0 - fit.inliers.mean()
            if (not fit.converged
                    or rejected_fraction > config.max_rejected_star_fraction):
                # Many rejections: one-at-a-time rejection may have followed
                # the wrong consensus. RANSAC starts from star pairs instead.
                alternative = ransac_attitude(
                    r, b, sigma, rng=np.random.default_rng(e.index),
                    false_alarm_probability=config.outlier_false_alarm,
                    min_inliers=config.min_stars)
                if alternative.converged and (
                        not fit.converged
                        or alternative.inliers.sum() > fit.inliers.sum()):
                    fit = alternative
            if not fit.converged:
                raise _Dropped("star matches are mutually inconsistent and no "
                               f"consistent subset was found ({fit.status})",
                               "catalog mismatch")
            inliers, C = fit.inliers, fit.C
        if int(inliers.sum()) < config.min_stars:
            raise InsufficientGeometryError("too few consistent stars")
        covariance = attitude_covariance(attitude_information(b[inliers],
                                                              sigma[inliers]))
    except InsufficientGeometryError as error:
        raise _Dropped(f"attitude unobservable: {error}", "insufficient stars")
    except ValueError as error:
        raise _Dropped(f"attitude solution failed: {error}", "invalid input")

    used = np.zeros(len(e.rays), dtype=bool)
    used[np.flatnonzero(usable)[inliers]] = True
    e.used = used
    e.rejected_stars = [e.star_ids[i] for i in np.flatnonzero(usable)[~inliers]]
    e.attitude_sigma_arcsec = float(
        np.sqrt(np.linalg.eigvalsh(covariance).max()) / ARCSEC)

    # ---- stage 4: vertical, carried into celestial axes (Module 7) -----------------
    info = {"timestamp_utc": e.eop.utc, "earth_orientation_source": e.eop.source}
    try:
        e.measurement = measurement_from_sensors(
            r[inliers], b[inliers], sigma[inliers], e.accel, config.C_CI,
            e.eop.utc, earth=(Transform("E", "G", e.C_EG), info))
    except ValueError as error:
        raise _Dropped(f"vertical could not be formed: {error}", "invalid input")
    e.measurement.quality = e.quality
    if C is not None:
        predicted = r[inliers] @ np.asarray(C).T
        angles = np.arctan2(np.linalg.norm(np.cross(b[inliers], predicted), axis=1),
                            np.einsum("ij,ij->i", b[inliers], predicted))
        e.star_residuals = angles
        e.star_chi2 = float(np.sum((angles / sigma[inliers]) ** 2))
        e.star_dof = 2 * int(inliers.sum()) - 3


def _mean_single_fix(exposures):
    """Module 6 direct inversion, averaged: a starting point only."""
    total = np.zeros(3)
    for e in exposures:
        total += e.C_EG @ e.measurement.up_celestial
    lat, lon = earth_fixed_up_to_geodetic(total / np.linalg.norm(total))
    return lat, (0.0 if np.isnan(lon) else lon)


def _distance(a, b):
    return float(horizontal_position_error(a, b))


# ---------------------------------------------------------- stage 5: position
@dataclass
class _Solution:
    geodetic: tuple                     # (lat, lon) used for the next iteration
    astronomical: tuple                 # before the deflection model
    longitude_defined: bool
    batch: object                       # Module 7 BatchResult
    measurements: list                  # as given to the estimator
    covariance_ne: object               # estimator covariance [m^2], or None
    precision_ne: object                # random errors only [m^2], or None
    polar_frame: bool
    calibration: dict
    clock: dict
    up_earth_fixed: np.ndarray


def _to_working(lat_lon):
    u = _POLAR_FRAME @ unit_up(np.radians(lat_lon[0]), np.radians(lat_lon[1]))
    return (float(np.degrees(np.arcsin(np.clip(u[2], -1, 1)))),
            float(np.degrees(np.arctan2(u[1], u[0]))))


def _position(exposures, start, config):
    measurements = [e.measurement for e in exposures]
    polar = np.cos(np.radians(start[0])) < POLAR_COS_LATITUDE
    if polar:
        # u_G = C_GE u_E = (C_GE Q^T)(Q u_E): solve for Q u_E instead.
        measurements = [replace(m, C_GE=m.C_GE @ _POLAR_FRAME.T)
                        for m in measurements]
        start = _to_working(start)

    misalignment = config.misalignment_sigma_arcsec * ARCSEC
    principal_point = config.principal_point_sigma_px / config.camera.fx
    calibration_prior = (misalignment**2 * np.eye(3)
                         + principal_point**2 * np.diag([1.0, 1.0, 0.0]))
    shared = {}
    if config.accel_bias_sigma > 0.0:
        shared["accelerometer_bias"] = config.accel_bias_sigma**2 * np.eye(3)

    # ---- calibration states: only if the DATA determine them ------------------
    calibration = {"requested": config.estimate_calibration, "estimated": False,
                   "status": "FIXED: configured calibration used as given"}
    estimate_calibration = False
    prior_sigma = None
    if config.estimate_calibration:
        prior_sigma = float(np.sqrt(calibration_prior[0, 0]))
        observability, singular = _data_observability(
            measurements, start, shared, estimate_clock=False,
            estimate_calibration=True)
        calibration["data_observability"] = observability
        calibration["singular_values"] = singular
        if prior_sigma <= 0.0:
            calibration["status"] = ("WITHHELD: no calibration uncertainty is "
                                     "declared, so there is nothing to estimate")
        elif observability != "well conditioned":
            calibration["status"] = (
                f"WITHHELD: calibration states are {observability} for this "
                "set of attitudes; configured calibration used with its "
                "declared uncertainty")
        else:
            estimate_calibration = True
    if not estimate_calibration and calibration_prior[0, 0] > 0.0:
        shared["calibration"] = calibration_prior

    # ---- clock state: same rule ---------------------------------------------------
    clock = {"requested": config.estimate_clock, "estimated": False}
    estimate_clock = False
    if config.estimate_clock:
        if polar:
            clock["status"] = "WITHHELD: clock offset is not observable near a pole"
        else:
            observability, _ = _data_observability(
                measurements, start, shared, estimate_clock=True,
                estimate_calibration=estimate_calibration)
            clock["data_observability"] = observability
            if observability == "well conditioned":
                estimate_clock = True
            else:
                clock["status"] = (f"WITHHELD: clock offset is {observability} "
                                   "over this observation span")

    kwargs = dict(estimate_clock=estimate_clock,
                  estimate_calibration=estimate_calibration,
                  shared_errors=shared or None,
                  calibration_prior_sigma_rad=prior_sigma
                  if estimate_calibration else None)
    batch = batch_position_estimate(measurements, start, **kwargs)

    covariance = precision = None
    if batch.covariance is not None:
        random_only = _Problem(measurements, estimate_clock, estimate_calibration,
                               "simple", None, kwargs["calibration_prior_sigma_rad"])
        J = random_only.jacobian(batch.state)
        precision = np.linalg.inv(J.T @ J)[:2, :2]
        covariance = batch.covariance[:2, :2]

    lat_w, lon_w = np.radians(batch.latitude_deg), np.radians(batch.longitude_deg)
    up = unit_up(lat_w, lon_w)
    if polar:
        up = _POLAR_FRAME.T @ up
    lat, lon = earth_fixed_up_to_geodetic(up)
    longitude_defined = not np.isnan(lon)
    axes_lon = lon if longitude_defined else 0.0
    north, east = north_east_axes(np.radians(lat), np.radians(axes_lon))

    # State covariance [rad^2 of lat, lon] -> north/east metres at the solution.
    north_w, east_w = north_east_axes(lat_w, lon_w)
    if polar:
        north_w, east_w = _POLAR_FRAME.T @ north_w, _POLAR_FRAME.T @ east_w
    turn = np.array([[north @ north_w, north @ east_w],
                     [east @ north_w, east @ east_w]])
    # (d_lat, d_lon) -> angles along working north/east -> true north/east
    # -> metres. East metres per radian of east angle is the prime-vertical
    # radius N, which stays finite at the pole (N cos(lat) d_lon does not).
    meridian = radians_to_metres(np.radians(lat))[0, 0]
    prime_vertical = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(np.radians(lat)) ** 2)
    to_metres = (np.diag([meridian, prime_vertical]) @ turn
                 @ np.diag([1.0, np.cos(lat_w)]))
    if covariance is not None:
        covariance = to_metres @ covariance @ to_metres.T
        precision = to_metres @ precision @ to_metres.T

    if estimate_calibration:
        estimate = np.asarray(batch.calibration_rad)
        posterior = batch.covariance[-3:, -3:] if batch.covariance is not None \
            else None
        calibration.update(
            estimated=True, status="ESTIMATED from the batch",
            rotation_vector_arcsec=(estimate / ARCSEC).tolist(),
            sigma_arcsec=None if posterior is None else
            (np.sqrt(np.diag(posterior)) / ARCSEC).tolist(),
            prior_sigma_arcsec=prior_sigma / ARCSEC,
            corrected_C_CI=rotation_vector_to_matrix(estimate).T
            @ _matrix(config.C_CI))
    if estimate_clock:
        index = batch.state_names.index("clock")
        clock.update(estimated=True, status="ESTIMATED from the batch",
                     offset_s=float(batch.clock_offset_s),
                     sigma_s=None if batch.covariance is None else
                     float(np.sqrt(batch.covariance[index, index])))

    # ---- deflection model: astronomical -> geodetic ------------------------------
    geodetic_up = up
    if config.deflection_model_arcsec is not None:
        xi, eta = np.asarray(config.deflection_model_arcsec, dtype=float) * ARCSEC
        # The plumb line is n + xi*north + eta*east, normalized, with the
        # axes taken at the ellipsoid normal n. Invert that exactly (a first
        # order subtraction is wrong by xi^2: millimetres for arcseconds).
        scale = np.sqrt(1.0 + xi * xi + eta * eta)
        n_axis, e_axis = north, east
        for _ in range(4):
            geodetic_up = scale * up - xi * n_axis - eta * e_axis
            geodetic_up /= np.linalg.norm(geodetic_up)
            g = earth_fixed_up_to_geodetic(geodetic_up)
            n_axis, e_axis = north_east_axes(
                np.radians(g[0]), np.radians(0.0 if np.isnan(g[1]) else g[1]))
    g_lat, g_lon = earth_fixed_up_to_geodetic(geodetic_up)
    longitude_defined = not np.isnan(g_lon)
    return _Solution(
        geodetic=(g_lat, g_lon if longitude_defined else 0.0),
        astronomical=(lat, axes_lon), longitude_defined=longitude_defined,
        batch=batch, measurements=measurements, covariance_ne=covariance,
        precision_ne=precision, polar_frame=bool(polar), calibration=calibration,
        clock=clock, up_earth_fixed=geodetic_up)


def _data_observability(measurements, start, shared, *, estimate_clock,
                        estimate_calibration):
    """Observability of the requested states from the data alone (no prior)."""
    problem = _Problem(measurements, estimate_clock, estimate_calibration,
                       "simple", shared or None, None)
    x = np.zeros(len(problem.names))
    x[0], x[1] = np.radians(start[0]), np.radians(start[1])
    J = problem.jacobian(x)
    singular = np.linalg.svd(J * state_scales(x[0], problem.names),
                             compute_uv=False)
    if len(singular) < len(problem.names):
        return "rank deficient", singular
    return classify_observability(singular), singular


# ------------------------------------------------------------------ no position
def _nothing_usable(batch, references, context):
    """Every exposure was set aside: say why, and report no position."""
    categories = {item["category"] for item in context["excluded"]}
    if categories <= {"insufficient stars"}:
        return _no_fix(UNOBSERVABLE, ["camera attitude is not determined: no "
                                      "exposure has enough usable matched stars"],
                       batch, references, context)
    return _no_fix(REJECTED, ["no usable observation in the batch"], batch,
                   references, context)


def _no_fix(status, reasons, batch, references, context):
    return integrity.empty_result(status, reasons, batch, references, context)


def time_error_east_m(latitude_deg, seconds):
    """Eastward ground displacement of a time error at a latitude."""
    radii = radians_to_metres(np.radians(latitude_deg))
    return float(EARTH_ROTATION_RATE_UT1 * seconds * radii[1, 1])
