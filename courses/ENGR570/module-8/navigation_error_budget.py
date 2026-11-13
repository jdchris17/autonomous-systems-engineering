"""
navigation_error_budget.py - Space Systems Engineering, Module 8

The integration point: the complete terrestrial navigation model with
declared physical and sensor uncertainties, built on the existing stack
(Module 3 camera, Module 4 attitude, Module 5 gravity, Module 6 position,
Module 7 batch estimator and covariance tools).

Each effect enters at the level where its physics acts
------------------------------------------------------
  image-to-ray model     focal length, principal point, radial distortion
                         (CameraModel)
  astronomical reference proper motion, aberration, light deflection
                         (astrometric_corrections), refraction
                         (atmospheric_refraction, in the local frame)
  sensor-frame calibration  camera-IMU misalignment, image/IMU timing
  inertial sensor        accelerometer noise and bias
  time and Earth orientation  clock error, UT1 and polar-motion error
  terrestrial gravity model   deflection of the vertical
None of them is a generic "position error adjustment".

Error classes (Module 7 distinctions)
-------------------------------------
  random         drawn independently for every observation
  fixed          drawn once per simulated instrument, constant in its batch
  environmental  set by the atmosphere and sky; varies with pointing
  geographic     fixed for every observation from one place

A trial is ONE instrument at one place making n_observations. Statistics
across trials are population accuracy. The scatter of one instrument's own
observations is reported separately as repeatability.

All default numbers are illustrative assumptions, not sensor specifications.
"""

import sys
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import erfa
import numpy as np
from astropy.time import Time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "module-7"))
import navigation_estimator as ne  # noqa: E402  (adds Modules 1-6 to sys.path)
import navigation_scenarios as ns  # noqa: E402
import position_simulator as ps  # noqa: E402
from astronomical_time import EARTH_ROTATION_RATE_UT1, time_scales  # noqa: E402
from compare_reference_frames import polar_motion  # noqa: E402
from imu_alignment import perturb_calibration  # noqa: E402
from rotation_engine import rotation_vector_to_matrix  # noqa: E402
from sensor_frames import PinholeCamera, Transform  # noqa: E402
from star_field_simulation import centroid_sigma_to_angle, unit_square_positions  # noqa: E402

from atmospheric_refraction import (  # noqa: E402
    STANDARD_ATMOSPHERE,
    Atmosphere,
    unrefract_enu,
)

cp = ne.cp
ARCSEC = ne.ARCSEC
MAS = ARCSEC / 1000.0
GRAVITY = 9.81
CATALOG_AGE_YEARS = 10.86          # epoch 2016.0 to the observation date
CENTROID_FLOOR_PX = 1e-3           # keeps covariances positive when noise is off
ACCEL_NOISE_FLOOR = 1e-6


# ------------------------------------------------------------ image-to-ray model
@dataclass(frozen=True)
class CameraModel(PinholeCamera):
    """
    Module 3's pinhole camera extended with a focal-length scale and one
    radial distortion term.

    Convention: with normalized image coordinates x = X/Z, y = Y/Z
    (dimensionless; r = sqrt(x^2 + y^2) is the tangent of the off-axis
    angle) the distorted coordinates are

        x_d = x (1 + k1 r^2),    y_d = y (1 + k1 r^2)

    and the pixel is (cx + fx x_d, cy + fy y_d). k1 is dimensionless. For
    the 20 degree camera r reaches 0.176 at the edge of the long axis, so
    k1 = 0.02 moves an edge star by 0.02 * 0.176^3 * fx = 0.6 px.
    """
    focal_scale: float = 1.0
    k1: float = 0.0

    @property
    def fx(self):
        return (self.width / (2.0 * np.tan(np.radians(self.horizontal_fov_deg) / 2.0))
                * self.focal_scale)

    def project(self, camera_vectors):
        v = np.atleast_2d(np.asarray(camera_vectors, dtype=float))
        in_front = v[:, 2] > 0.0
        pixels = np.full((len(v), 2), np.nan)
        x = v[in_front, 0] / v[in_front, 2]
        y = v[in_front, 1] / v[in_front, 2]
        factor = 1.0 + self.k1 * (x * x + y * y)
        pixels[in_front, 0] = self.cx + self.fx * x * factor
        pixels[in_front, 1] = self.cy + self.fy * y * factor
        visible = np.zeros(len(v), dtype=bool)
        visible[in_front] = ((pixels[in_front, 0] >= 0) & (pixels[in_front, 0] < self.width)
                             & (pixels[in_front, 1] >= 0) & (pixels[in_front, 1] < self.height))
        return pixels, visible

    def unproject(self, pixels):
        p = np.atleast_2d(np.asarray(pixels, dtype=float))
        xd = (p[:, 0] - self.cx) / self.fx
        yd = (p[:, 1] - self.cy) / self.fy
        x, y = xd.copy(), yd.copy()
        for _ in range(12):                    # fixed point; k1 r^2 << 1
            factor = 1.0 + self.k1 * (x * x + y * y)
            x, y = xd / factor, yd / factor
        rays = np.column_stack((x, y, np.ones(len(p))))
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)


# -------------------------------------------------------------- Earth orientation
class EarthOrientation:
    """
    Earth-orientation data for a short observing session, with its
    provenance, and a fast GCRS -> ITRS matrix that accepts perturbations.

    Carries the source, how far into the predicted range the session lies
    and the associated UT1 and polar-motion uncertainties, rather than a
    yes/no "data available" flag. The prediction-uncertainty growth laws
    (0.00025 D^0.75 s for UT1-UTC, 0.00068 D^0.80 arcsec for polar motion,
    D days past the last measured value) are those I recall from IERS
    Bulletin A and should be confirmed against the bulletin before being
    relied on.
    """

    def __init__(self, start_utc):
        time = Time(start_utc, scale="utc")
        scales = time_scales(time)
        self.start_utc = time.utc.isot
        self._tt = scales.tt_jd_parts
        self._ut1 = scales.ut1_jd_parts
        self.utc_jd = (float(time.utc.jd1), float(time.utc.jd2))
        self.dut1_s = scales.dut1_seconds
        self.xp, self.yp = polar_motion(time)
        status = ne.reference_status(time)
        self.source = status["source"]
        self.is_prediction = status["is_prediction"]
        self.days_predicted = status["days_predicted"]
        self.ut1_sigma_s = status["ut1_sigma_s"]
        self.polar_motion_sigma_arcsec = prediction_uncertainty(
            self.days_predicted)[1] if self.is_prediction else 5e-5

    def matrix(self, offset_s=0.0, *, dut1_error_s=0.0, xp_error_rad=0.0,
               yp_error_rad=0.0, dx_rad=0.0, dy_rad=0.0):
        """
        C_EG at start + offset_s. Optional perturbations: an error in
        UT1-UTC, in the polar-motion coordinates, and celestial-pole
        offsets dX, dY added to the IAU 2006/2000A pole.
        """
        days = offset_s / 86400.0
        tt1, tt2 = self._tt[0], self._tt[1] + days
        ut1, ut2 = self._ut1[0], self._ut1[1] + days + dut1_error_s / 86400.0
        x, y, s = erfa.xys06a(tt1, tt2)
        celestial_to_intermediate = erfa.c2ixys(x + dx_rad, y + dy_rad, s)
        polar = erfa.pom00(self.xp + xp_error_rad, self.yp + yp_error_rad,
                           erfa.sp00(tt1, tt2))
        return erfa.c2tcio(celestial_to_intermediate, erfa.era00(ut1, ut2), polar)

    def record(self):
        return {"source": self.source, "is_prediction": self.is_prediction,
                "days_predicted": self.days_predicted,
                "ut1_minus_utc_s": self.dut1_s, "ut1_sigma_s": self.ut1_sigma_s,
                "polar_motion_arcsec": (self.xp / ARCSEC, self.yp / ARCSEC),
                "polar_motion_sigma_arcsec": self.polar_motion_sigma_arcsec}


def prediction_uncertainty(days_past_last_measurement):
    """(UT1 sigma [s], polar-motion sigma [arcsec]) for predicted EOP."""
    d = max(float(days_past_last_measurement), 0.0)
    return 0.00025 * d**0.75, 0.00068 * d**0.80


# ---------------------------------------------------------------------- scenario
@dataclass(frozen=True)
class Scenario:
    """Everything that defines one error-budget run. Illustrative defaults."""
    name: str = "baseline"
    # observer and session
    latitude_deg: float = 45.0
    longitude_deg: float = -75.0
    start_utc: str = ps.OBSERVATION_UTC
    n_observations: int = 10
    spacing_s: float = 1.0
    zenith_angle_deg: float = 30.0
    diverse_attitudes: bool = False
    # camera
    hfov_deg: float = 20.0
    n_stars: int = 20
    # random, per observation
    centroid_noise_px: float = 0.2
    accel_noise: float = 0.005
    accel_samples: int = 200
    # fixed per instrument
    accel_bias_sigma: float = 0.005                 # m/s^2 per axis
    misalignment_sigma_arcsec: float = 180.0        # 0.05 deg per axis
    principal_point_sigma_px: float = 0.5
    focal_length_sigma_fraction: float = 5e-4
    lens_k1: float = 0.02                           # true distortion of the lens
    clock_sigma_s: float = 0.1
    image_imu_offset_s: float = 0.0
    platform_rate_deg_s: float = 0.0
    # environment (the real world)
    atmosphere: object = STANDARD_ATMOSPHERE        # None = no atmosphere
    stellar_apparent_place: bool = True             # aberration, light deflection
    proper_motion_mas_yr: float = 100.0             # per axis, 1 sigma
    # geographic and reference data
    deflection_sigma_arcsec: float = 5.0            # per component
    eop_ut1_sigma_s: float = 0.0018
    eop_pm_sigma_arcsec: float = 0.005
    # what the estimator models
    model_refraction: bool = False
    model_apparent_place: bool = False
    model_proper_motion: bool = False
    model_distortion: bool = False
    deflection_model_residual_arcsec: object = None  # None = not corrected
    estimate_calibration: bool = False
    declare_uncertainties: bool = True

    def with_changes(self, **changes):
        return replace(self, **changes)


BASELINE = Scenario()

# Module 6/7 world: no physical perturbations beyond those two modules.
MODULE7_WORLD = dict(
    principal_point_sigma_px=0.0, focal_length_sigma_fraction=0.0, lens_k1=0.0,
    atmosphere=None, stellar_apparent_place=False, proper_motion_mas_yr=0.0,
    deflection_sigma_arcsec=0.0, eop_ut1_sigma_s=0.0, eop_pm_sigma_arcsec=0.0)

_PERFECT = dict(
    centroid_noise_px=0.0, accel_noise=0.0, accel_bias_sigma=0.0,
    misalignment_sigma_arcsec=0.0, clock_sigma_s=0.0, image_imu_offset_s=0.0,
    platform_rate_deg_s=0.0,
    # A deflection model with a residual error is itself an error source.
    deflection_model_residual_arcsec=None, **MODULE7_WORLD)

# name -> (error class, scenario fields that constitute the source)
SOURCES = {
    "Star centroid noise": ("random", ["centroid_noise_px"]),
    "Accelerometer noise": ("random", ["accel_noise"]),
    "Accelerometer bias": ("fixed", ["accel_bias_sigma"]),
    "Camera-IMU misalignment": ("fixed", ["misalignment_sigma_arcsec"]),
    "Camera intrinsics": ("fixed", ["principal_point_sigma_px",
                                    "focal_length_sigma_fraction", "lens_k1"]),
    "Clock": ("fixed", ["clock_sigma_s"]),
    "Atmospheric refraction": ("environmental", ["atmosphere"]),
    "Stellar apparent place": ("environmental", ["stellar_apparent_place",
                                                 "proper_motion_mas_yr"]),
    "Deflection of the vertical": ("geographic", [
        "deflection_sigma_arcsec", "deflection_model_residual_arcsec"]),
    "Earth orientation data": ("geographic", ["eop_ut1_sigma_s",
                                              "eop_pm_sigma_arcsec"]),
}


def perfect(scenario):
    """The scenario with every error source switched off."""
    return scenario.with_changes(**_PERFECT)


def isolate(scenario, source):
    """Only `source` active, at the level it has in `scenario`."""
    _, names = SOURCES[source]
    # Isolation runs measure the error a source causes, not a covariance, so
    # the estimator is run unweighted by shared-error priors: with the random
    # noise switched off, a correlated-error weighting would be ill-posed.
    return perfect(scenario).with_changes(
        name=f"only {source}", declare_uncertainties=False,
        **{n: getattr(scenario, n) for n in names})


# ------------------------------------------------------------------- simulation
def _instrument_errors(s, rng, fixed):
    """Fixed and geographic errors for one instrument at one place."""
    z = rng.standard_normal(16)            # always 16, so streams stay aligned
    errors = {
        "accel_bias": z[0:3] * s.accel_bias_sigma,
        "misalignment_rad": z[3:6] * s.misalignment_sigma_arcsec * ARCSEC,
        "principal_point_px": z[6:8] * s.principal_point_sigma_px,
        "focal_fraction": z[8] * s.focal_length_sigma_fraction,
        "clock_s": z[9] * s.clock_sigma_s,
        "deflection_arcsec": z[10:12] * s.deflection_sigma_arcsec,   # north, east
        "dut1_s": z[12] * s.eop_ut1_sigma_s,
        "polar_motion_rad": z[13:15] * s.eop_pm_sigma_arcsec * ARCSEC,
        "azimuth_deg": 360.0 * rng.random(),
        "roll_deg": 360.0 * rng.random(),
        "deflection_model_noise": rng.standard_normal(2),
    }
    if fixed:
        errors.update({k: np.asarray(v, dtype=float) if np.ndim(v) else float(v)
                       for k, v in fixed.items()})
    return errors


def simulate_instrument(scenario, rng, earth=None, fixed=None):
    """
    One instrument, one place, n_observations.

    Returns (measurements, truth): a list of Module 7 VerticalMeasurement
    built exactly as the estimator would build them (wrong calibration,
    wrong clock and missing corrections included), and the truth dictionary
    for validation. `fixed` overrides drawn errors, e.g.
    {"principal_point_px": [1.0, 0.0]}.
    """
    s = scenario
    earth = EarthOrientation(s.start_utc) if earth is None else earth
    e = _instrument_errors(s, rng, fixed)

    nominal = CameraModel(horizontal_fov_deg=s.hfov_deg)
    true_camera = CameraModel(
        horizontal_fov_deg=s.hfov_deg,
        cx=nominal.cx + e["principal_point_px"][0],
        cy=nominal.cy + e["principal_point_px"][1],
        focal_scale=1.0 + e["focal_fraction"], k1=s.lens_k1)
    believed_camera = CameraModel(horizontal_fov_deg=s.hfov_deg,
                                  k1=s.lens_k1 if s.model_distortion else 0.0)

    c_ci_true = ps.DEFAULT_C_CI
    c_ci_believed = perturb_calibration(c_ci_true, e["misalignment_rad"])
    c_en = ps.local_axes_ecef(s.latitude_deg, s.longitude_deg)
    xi, eta = e["deflection_arcsec"] * ARCSEC
    plumb = np.array([eta, xi, 1.0])                  # east, north, up
    plumb /= np.linalg.norm(plumb)

    if s.diverse_attitudes:
        attitudes = ns.diverse_attitudes(s.n_observations, max_zenith=55.0)
    else:
        attitudes = [(s.zenith_angle_deg, e["azimuth_deg"], e["roll_deg"])] \
            * s.n_observations

    centroid = max(s.centroid_noise_px, CENTROID_FLOOR_PX)
    accel_sigma = max(s.accel_noise, ACCEL_NOISE_FLOOR)
    lon, lat = np.radians(s.longitude_deg), np.radians(s.latitude_deg)
    measurements = []
    for k, (zenith, azimuth, roll) in enumerate(attitudes):
        offset = k * s.spacing_s
        c_eg_true = earth.matrix(offset)
        c_eg_believed = earth.matrix(
            offset + e["clock_s"], dut1_error_s=e["dut1_s"],
            xp_error_rad=e["polar_motion_rad"][0],
            yp_error_rad=e["polar_motion_rad"][1])
        to_celestial = c_eg_true.T @ c_en               # ENU -> G, truth

        c_nc = ps.camera_axes_local(zenith, azimuth, roll)     # camera in ENU
        # Stars as the camera really sees them (observed directions).
        camera_dirs = nominal.unproject(
            unit_square_positions(s.n_stars, rng) * [nominal.width, nominal.height])
        observed_enu = camera_dirs @ c_nc.T

        # ---- the estimator's reference model for those same stars -----------
        vacuum_enu = (unrefract_enu(observed_enu, s.atmosphere)
                      if s.atmosphere is not None else observed_enu)
        model_enu = observed_enu if s.model_refraction else vacuum_enu
        reference = model_enu @ to_celestial.T
        if s.stellar_apparent_place and not s.model_apparent_place:
            # Replace the apparent place by the ICRS astrometric direction.
            az = np.arctan2(vacuum_enu[:, 0], vacuum_enu[:, 1])
            zd = np.arctan2(np.hypot(vacuum_enu[:, 0], vacuum_enu[:, 1]),
                            vacuum_enu[:, 2])
            ra, dec = erfa.atoc13(
                "A", az, zd, earth.utc_jd[0], earth.utc_jd[1] + offset / 86400.0,
                earth.dut1_s, lon, lat, 0.0, earth.xp, earth.yp, 0.0, 0.0, 0.0, 0.55)
            icrs = np.column_stack((np.cos(dec) * np.cos(ra),
                                    np.cos(dec) * np.sin(ra), np.sin(dec)))
            reference = reference + (icrs - vacuum_enu @ to_celestial.T)
        pm_draw = rng.standard_normal((s.n_stars, 3))
        if not (s.model_proper_motion or s.model_apparent_place):
            # Catalog positions at the catalog epoch, not the date.
            reference = reference - pm_draw * (s.proper_motion_mas_yr * MAS
                                               * CATALOG_AGE_YEARS)
        reference /= np.linalg.norm(reference, axis=1, keepdims=True)

        # ---- image formation and processing -----------------------------------
        if s.image_imu_offset_s and s.platform_rate_deg_s:
            # The image is taken a little after the IMU sample while the
            # platform turns about a horizontal (east) axis.
            turn = np.radians(s.platform_rate_deg_s) * s.image_imu_offset_s
            image_dirs = (observed_enu @ rotation_vector_to_matrix(
                [turn, 0.0, 0.0])) @ c_nc
        else:
            image_dirs = camera_dirs
        pixels, _ = true_camera.project(image_dirs)
        pixels = pixels + rng.standard_normal(pixels.shape) * s.centroid_noise_px
        measured = believed_camera.unproject(pixels)
        star_sigma = centroid_sigma_to_angle(nominal, camera_dirs, centroid)

        # ---- accelerometer -------------------------------------------------------
        c_in = np.asarray(c_ci_true.matrix).T @ c_nc.T          # ENU -> IMU
        accel = (GRAVITY * (c_in @ plumb) + e["accel_bias"]
                 + rng.standard_normal((s.accel_samples, 3)) * accel_sigma)

        info = {"timestamp_utc": earth.start_utc,
                "earth_orientation_source": earth.source}
        measurements.append(ne.measurement_from_sensors(
            reference, measured, star_sigma, accel, c_ci_believed, earth.start_utc,
            earth=(Transform("E", "G", c_eg_believed), info)))

    truth = {"position": (s.latitude_deg, s.longitude_deg), "errors": e,
             "focal_px": nominal.fx}
    return measurements, truth


def solve_instrument(scenario, measurements, truth):
    """
    Batch position estimate for one instrument, with the declared
    uncertainties and the deflection correction if the scenario has one.
    Returns (lat_lon, position covariance in NE metres or None, BatchResult).
    """
    s = scenario
    start = ns.single_fix(measurements[0])
    equivalent_pp = s.principal_point_sigma_px / truth["focal_px"]      # rad
    shared = {}
    kwargs = {}
    if s.declare_uncertainties:
        if s.accel_bias_sigma > 0:
            shared["accelerometer_bias"] = s.accel_bias_sigma**2 * np.eye(3)
        calibration = ((s.misalignment_sigma_arcsec * ARCSEC) ** 2 * np.eye(3)
                       + equivalent_pp**2 * np.diag([1.0, 1.0, 0.0]))
        if s.estimate_calibration:
            kwargs["calibration_prior_sigma_rad"] = max(
                float(np.sqrt(calibration[0, 0])), ARCSEC)
        elif calibration[0, 0] > 0:
            shared["calibration"] = calibration
    result = ne.batch_position_estimate(
        measurements, start, estimate_calibration=s.estimate_calibration,
        shared_errors=shared or None, **kwargs)

    lat, lon = result.latitude_deg, result.longitude_deg
    covariance = result.position_covariance_ne
    deflection_sigma = s.deflection_sigma_arcsec
    if s.deflection_model_residual_arcsec is not None:
        # Geodetic = astronomical minus the modelled deflection.
        modelled = (truth["errors"]["deflection_arcsec"]
                    + truth["errors"]["deflection_model_noise"]
                    * s.deflection_model_residual_arcsec)
        lat -= modelled[0] / 3600.0
        lon -= modelled[1] / 3600.0 / np.cos(np.radians(lat))
        deflection_sigma = s.deflection_model_residual_arcsec
    if covariance is not None and s.declare_uncertainties:
        T = ne.radians_to_metres(np.radians(lat))
        time_sigma = np.hypot(s.clock_sigma_s, s.eop_ut1_sigma_s)
        covariance = covariance + np.diag([
            (deflection_sigma * ARCSEC * T[0, 0]) ** 2,
            (deflection_sigma * ARCSEC * T[1, 1] / np.cos(np.radians(lat))) ** 2
            + (EARTH_ROTATION_RATE_UT1 * time_sigma * T[1, 1]) ** 2])
    return (lat, lon), covariance, result


# ------------------------------------------------------------------ Monte Carlo
@dataclass
class BudgetResult:
    scenario: Scenario
    trials: int
    seed: int
    errors_ne: np.ndarray                # (trials, 2) metres north, east
    horizontal: np.ndarray               # (trials,) metres
    rms: float
    median: float
    p95: float
    repeatability_rms: float             # scatter within one instrument's batch
    reported_rms: float                  # from the estimator's covariance
    mean_nees: float
    coverage95: float                    # percent inside the 95 % ellipse
    contributions: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)


def _run(scenario, trials, seed):
    rng = np.random.default_rng(seed)
    earth = EarthOrientation(scenario.start_utc)
    truth_position = (scenario.latitude_deg, scenario.longitude_deg)
    errors, covariances, scatter, withheld = [], [], [], 0
    for _ in range(trials):
        measurements, truth = simulate_instrument(scenario, rng, earth)
        position, covariance, _ = solve_instrument(scenario, measurements, truth)
        errors.append(cp.north_east_error(position, truth_position))
        fixes = np.array([cp.north_east_error(ns.single_fix(m), truth_position)
                          for m in measurements])
        scatter.append(np.sum(np.var(fixes, axis=0)))
        if covariance is None:
            withheld += 1
            covariances.append(np.full((2, 2), np.nan))
        else:
            covariances.append(covariance)
    return (np.array(errors), np.array(covariances), np.array(scatter), withheld,
            earth)


def summarize(errors_ne):
    horizontal = np.hypot(errors_ne[:, 0], errors_ne[:, 1])
    return {"rms": float(np.sqrt(np.mean(horizontal**2))),
            "median": float(np.median(horizontal)),
            "p95": float(np.percentile(horizontal, 95))}


def run_error_budget(
    scenario,
    *,
    trials=1000,
    seed=5708,
    contributions=True,
    contribution_trials=None,
):
    """
    Run the complete terrestrial navigation model
    with declared physical and sensor uncertainties.

    Return position errors, diagnostics, and
    per-source contribution statistics.

    Each trial is a different instrument at the scenario's location making
    scenario.n_observations observations, solved with Module 7's batch
    estimator. Per-source contributions are separate runs with only that
    source active (contribution_trials each, default trials // 4, at least
    200); because the sources are nearly independent their squares add up
    to approximately the combined mean-square error.
    """
    errors, covariances, scatter, withheld, earth = _run(scenario, trials, seed)
    stats = summarize(errors)
    valid = ~np.isnan(covariances[:, 0, 0])
    nees = np.einsum("bi,bij,bj->b", errors[valid],
                     np.linalg.inv(covariances[valid]), errors[valid])
    result = BudgetResult(
        scenario=scenario, trials=trials, seed=seed, errors_ne=errors,
        horizontal=np.hypot(errors[:, 0], errors[:, 1]), rms=stats["rms"],
        median=stats["median"], p95=stats["p95"],
        repeatability_rms=float(np.sqrt(np.mean(scatter))),
        reported_rms=float(np.sqrt(np.mean(
            covariances[valid][:, 0, 0] + covariances[valid][:, 1, 1]))),
        mean_nees=float(np.mean(nees)),
        coverage95=float(100.0 * np.mean(nees <= 5.991)),
        diagnostics={"covariance_withheld": withheld,
                     "earth_orientation": earth.record(),
                     "mean_error_ne": errors.mean(axis=0),
                     "rms_north": float(np.sqrt(np.mean(errors[:, 0] ** 2))),
                     "rms_east": float(np.sqrt(np.mean(errors[:, 1] ** 2)))})
    if contributions:
        n = contribution_trials or max(200, trials // 4)
        for source, (category, _) in SOURCES.items():
            part, *_ = _run(isolate(scenario, source), n, seed)
            result.contributions[source] = dict(category=category,
                                                **summarize(part))
    return result


def sensitivity_analysis(
    baseline_scenario,
    parameter_sweeps,
    *,
    trials=300,
    seed=5708,
    metric="rms",
):
    """
    Quantify sensitivity of horizontal position
    error to each model or calibration parameter.

    parameter_sweeps: {scenario_field: [multiplicative factors]}, for
    positive uncertainty parameters. For each field the scenario is run at
    every factor (same random seed throughout) and the dimensionless
    sensitivity S = d ln E / d ln p is the least-squares slope of ln(error)
    against ln(factor). S = 1 means the error is proportional to the
    parameter; S near 0 means the parameter is not limiting.
    """
    out = {}
    for name, factors in parameter_sweeps.items():
        base = getattr(baseline_scenario, name)
        values = []
        for factor in factors:
            errors, *_ = _run(baseline_scenario.with_changes(**{name: base * factor}),
                              trials, seed)
            values.append(summarize(errors)[metric])
        slope = np.polyfit(np.log(factors), np.log(values), 1)[0]
        out[name] = {"baseline_value": base, "factors": list(factors),
                     "errors": values, "sensitivity": float(slope)}
    return out


def compare_improvements(
    baseline_scenario,
    candidate_changes,
    *,
    trials=400,
    seed=5708,
):
    """
    Evaluate prospective corrections or
    hardware/calibration improvements.

    Return change in RMS and 95th-percentile
    horizontal error.

    candidate_changes: {name: {scenario_field: new_value, ...}}. Every
    candidate is applied to the baseline on its own, with the same random
    seed, so the differences are due to the change and not to sampling.
    Returns a list of rows sorted by RMS reduction, largest first; the
    first row is the baseline itself.
    """
    errors, *_ = _run(baseline_scenario, trials, seed)
    base = summarize(errors)
    rows = []
    for name, changes in candidate_changes.items():
        errors, *_ = _run(baseline_scenario.with_changes(**changes), trials, seed)
        stats = summarize(errors)
        rows.append({"name": name, **stats,
                     "rms_reduction": base["rms"] - stats["rms"],
                     "p95_reduction": base["p95"] - stats["p95"],
                     "rms_ratio": stats["rms"] / base["rms"]})
    rows.sort(key=lambda row: -row["rms_reduction"])
    return [{"name": "baseline", **base, "rms_reduction": 0.0,
             "p95_reduction": 0.0, "rms_ratio": 1.0}] + rows


def scenario_fields():
    return [f.name for f in fields(Scenario)]
