"""
qualification.py - representative operational scenarios, run through the
complete engine.

A Scenario describes a population: how good the hardware is, what the
environment does, how the instrument is used, and what the navigator
declares about all of it. One trial draws ONE instrument at ONE place,
simulates its observation batch and calls solve_navigation_fix(). Every
number reported comes out of that public function.

Each fixed error is drawn from its own distribution once per trial and
enters the simulated world, never the answer. When `declare` is True the
config declares the same sigmas the errors were drawn from, which is the
condition under which the reported covariance should be honest.

All magnitudes are illustrative engineering assumptions. "Phase 1" and
"Phase 2" name accuracy objectives (6-10 NM; 100-200 m), not the measured
capability of any phone or dedicated instrument.
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy.stats import chi2 as chi2_distribution

from .astrometry import Atmosphere, prediction_uncertainty
from .contracts import FIX_STATUSES, STATUSES
from .estimation import ARCSEC, diverse_attitudes, nees
from .frames import CameraModel
from .geodesy import DEFAULT_C_CI, horizontal_position_error, north_east_error
from .gravity import perturb_calibration
from .pipeline import solve_navigation_fix
from .references import (
    MEASURED_POLAR_SIGMA_ARCSEC,
    MEASURED_UT1_SIGMA_S,
    ReferenceData,
)
from .simulation import TrueInstrument, World, matched_config, simulate_batch

NAUTICAL_MILE = 1852.0
CHI2_95 = float(chi2_distribution.ppf(0.95, 2))
MISLEADING_NEES = float(chi2_distribution.ppf(0.9999, 2))    # far outside the ellipse


@dataclass(frozen=True)
class Scenario:
    name: str = "scenario"
    description: str = ""
    # ---- where and how -------------------------------------------------------
    position: object = (45.0, -75.0)            # or "global"
    n_observations: int = 10
    spacing_s: float = 1.0
    diverse: bool = False                       # deliberately varied attitudes
    zenith_deg: object = 30.0                   # number, or (low, high) per trial
    max_zenith_deg: float = 55.0                # for diverse sequences
    n_stars: object = 20                        # number, or (low, high) per trial
    star_layout: str = "uniform"
    start_utc: str = "2026-11-11T12:00:00"
    # ---- random noise (true) ---------------------------------------------------
    centroid_noise_px: float = 0.0
    accel_noise: float = 0.0
    accel_samples: int = 200
    # ---- fixed instrument errors (true 1-sigma, drawn per trial) ------------
    accel_bias_sigma: float = 0.0
    misalignment_sigma_arcsec: float = 0.0
    principal_point_sigma_px: float = 0.0
    focal_sigma_fraction: float = 0.0           # not declarable: see report
    clock_sigma_s: float = 0.0
    # ---- environment ---------------------------------------------------------------
    atmosphere: bool = True
    pressure_hpa: tuple = (1013.0, 0.0)         # mean, sigma of the true pressure
    temperature_c: tuple = (15.0, 15.0)         # low, high of the true temperature
    weather_measured: bool = True               # False: no weather data in config
    weather_error: tuple = (0.0, 0.0)           # sigma of measured hPa, deg C
    deflection_sigma_arcsec: float = 0.0        # true deflection, per component
    deflection_model_sigma_arcsec: object = None   # residual of a model, or None
    eop_age_days: object = 0.0                  # None: Earth orientation exact
    # ---- what the navigator does and declares -------------------------------------
    estimate_calibration: bool = False
    declare: bool = True
    declared_overrides: tuple = ()              # ((config field, value), ...)
    convergence_m: float = 0.5                  # then remaining error < 2 mm
    # ---- injected fault (scenario F) ------------------------------------------------
    fault: str = "none"

    def with_changes(self, **changes):
        return replace(self, **changes)


# ------------------------------------------------------------------ the scenarios
_PHONE = dict(centroid_noise_px=0.2, accel_noise=0.005, accel_bias_sigma=0.005,
              misalignment_sigma_arcsec=180.0, principal_point_sigma_px=0.5,
              focal_sigma_fraction=2e-4, clock_sigma_s=0.1,
              pressure_hpa=(1013.0, 12.0), temperature_c=(0.0, 30.0),
              weather_error=(2.0, 3.0), deflection_sigma_arcsec=5.0,
              eop_age_days=30.0)
_PRECISION = dict(centroid_noise_px=0.05, accel_noise=0.001, accel_bias_sigma=5e-4,
                  misalignment_sigma_arcsec=20.0, principal_point_sigma_px=0.05,
                  focal_sigma_fraction=2e-5, clock_sigma_s=1e-3,
                  pressure_hpa=(1013.0, 12.0), temperature_c=(0.0, 30.0),
                  weather_error=(1.0, 0.5), deflection_sigma_arcsec=5.0,
                  deflection_model_sigma_arcsec=1.0, eop_age_days=7.0)

SCENARIOS = {
    "A": Scenario(
        name="A - Ideal reference",
        description="No noise or model error; random place and attitudes.",
        position="global", n_observations=4, diverse=True, eop_age_days=None,
        temperature_c=(15.0, 15.0), convergence_m=0.01),
    "B": Scenario(
        name="B - Phase 1 baseline",
        description="Illustrative phone-class errors, stationary, one attitude, "
                    "ten exposures.", **_PHONE),
    "C": Scenario(
        name="C - Calibrated stationary",
        description="Same hardware as B, twenty exposures at deliberately "
                    "varied attitudes, calibration states estimated.",
        n_observations=20, diverse=True, estimate_calibration=True, **_PHONE),
    "D": Scenario(
        name="D - Precision hardware",
        description="Reduced noise and bias, measured weather, deflection "
                    "model to 1 arcsec, varied attitudes.",
        n_observations=20, diverse=True, estimate_calibration=True, n_stars=30,
        **_PRECISION),
    "E": Scenario(
        name="E - Environmental stress",
        description="Precision hardware in a harsh, poorly known environment: "
                    "-30 to +45 C, no weather data, pointing down to 65 deg "
                    "zenith angle, thermal bias 5x, deflection model 3 arcsec, "
                    "year-old Earth-orientation data.",
        n_observations=20, diverse=True, estimate_calibration=True, n_stars=30,
        max_zenith_deg=65.0,
        **{**_PRECISION, "accel_bias_sigma": 2.5e-3,
           "pressure_hpa": (1000.0, 30.0), "temperature_c": (-30.0, 45.0),
           "deflection_sigma_arcsec": 15.0, "deflection_model_sigma_arcsec": 3.0,
           "eop_age_days": 365.0},
        weather_measured=False),
}

# Scenario F: phone-class hardware with one injected fault per trial.
FAULTS = {
    "none": "healthy batch (control)",
    "vibration_all": "shaking during every exposure",
    "vibration_some": "shaking during 3 of 10 exposures",
    "lift": "0.3 m/s^2 vertical acceleration",
    "constant_acceleration": "steady 0.1 m/s^2 horizontal acceleration",
    "mismatch_15": "15 % of stars matched to the wrong catalog entry",
    "mismatch_40": "40 % of stars matched to the wrong catalog entry",
    "few_stars": "two matched stars per exposure",
    "clustered_stars": "all stars within 0.08 deg",
    "no_eop": "no Earth-orientation data on the device",
    "corrupt_some": "NaN centroids in 2 of 10 exposures",
    "corrupt_all": "NaN accelerometer data in every exposure",
}


def fault_scenario(fault):
    return SCENARIOS["B"].with_changes(name=f"F - {fault}", fault=fault,
                                       description=FAULTS[fault])


# ----------------------------------------------------------------------- one trial
def _draw(value, rng):
    """A number, or a uniform draw from a (low, high) pair."""
    if np.ndim(value) == 0:
        return value
    return rng.uniform(value[0], value[1])


def draw_trial(s, rng):
    """Draw one instrument, place and environment. Returns a dict of pieces."""
    if isinstance(s.position, str):
        latitude = float(np.degrees(np.arcsin(rng.uniform(-np.sin(np.radians(85.0)),
                                                          np.sin(np.radians(85.0))))))
        longitude = float(rng.uniform(-180.0, 180.0))
    else:
        latitude, longitude = s.position

    z = rng.standard_normal(16)
    bias = z[0:3] * s.accel_bias_sigma
    misalignment = z[3:6] * s.misalignment_sigma_arcsec * ARCSEC
    principal_point = z[6:8] * s.principal_point_sigma_px
    focal = z[8] * s.focal_sigma_fraction
    clock = z[9] * s.clock_sigma_s
    deflection = z[10:12] * s.deflection_sigma_arcsec
    if s.eop_age_days is None:                  # reference data taken as exact
        ut1_sigma = polar_sigma = 0.0
    else:
        ut1_sigma, polar_sigma = prediction_uncertainty(s.eop_age_days)
        ut1_sigma = max(ut1_sigma, MEASURED_UT1_SIGMA_S)
        polar_sigma = max(polar_sigma, MEASURED_POLAR_SIGMA_ARCSEC)
    eop_error = (z[12] * ut1_sigma, z[13] * polar_sigma * ARCSEC,
                 z[14] * polar_sigma * ARCSEC)

    true_atmosphere = measured_atmosphere = None
    if s.atmosphere:
        true_atmosphere = Atmosphere(
            pressure_hpa=float(s.pressure_hpa[0] + z[15] * s.pressure_hpa[1]),
            temperature_c=float(rng.uniform(*s.temperature_c)))
        if s.weather_measured:
            e = rng.standard_normal(2)
            measured_atmosphere = Atmosphere(
                pressure_hpa=true_atmosphere.pressure_hpa + e[0] * s.weather_error[0],
                temperature_c=true_atmosphere.temperature_c
                + e[1] * s.weather_error[1])

    n = s.n_observations
    if s.diverse:
        turn = rng.uniform(0.0, 360.0)
        attitudes = [(zenith, (azimuth + turn) % 360.0, roll) for
                     zenith, azimuth, roll in
                     diverse_attitudes(n, max_zenith=s.max_zenith_deg)]
    else:
        attitudes = [(float(_draw(s.zenith_deg, rng)), rng.uniform(0.0, 360.0),
                      rng.uniform(0.0, 360.0))] * n
    n_stars = s.n_stars if np.ndim(s.n_stars) == 0 else int(
        rng.integers(s.n_stars[0], s.n_stars[1] + 1))

    nominal = CameraModel()
    instrument = TrueInstrument(
        camera=CameraModel(cx=nominal.cx + principal_point[0],
                           cy=nominal.cy + principal_point[1],
                           focal_scale=1.0 + focal),
        accel_bias=tuple(bias), centroid_noise_px=s.centroid_noise_px,
        accel_noise=s.accel_noise, accel_samples=s.accel_samples)
    world = World(latitude_deg=latitude, longitude_deg=longitude,
                  atmosphere=true_atmosphere, deflection_arcsec=tuple(deflection),
                  clock_error_s=float(clock), eop_error=eop_error)

    # ---- what the navigator believes and declares -------------------------------
    model = sigma_deflection = None
    if s.deflection_model_sigma_arcsec is not None:
        model = tuple(deflection + rng.standard_normal(2)
                      * s.deflection_model_sigma_arcsec)
        sigma_deflection = s.deflection_model_sigma_arcsec
    else:
        sigma_deflection = s.deflection_sigma_arcsec
    declared = dict(
        accel_bias_sigma=s.accel_bias_sigma,
        misalignment_sigma_arcsec=s.misalignment_sigma_arcsec,
        principal_point_sigma_px=s.principal_point_sigma_px,
        clock_sigma_s=s.clock_sigma_s,
        deflection_sigma_arcsec=sigma_deflection) if s.declare else {}
    weather_fraction = 0.0
    if s.atmosphere and s.weather_measured:
        # Refraction scales as P / T: propagate the weather-sensor errors.
        weather_fraction = float(np.hypot(s.weather_error[0] / 1013.0,
                                          s.weather_error[1] / 288.0))
    config = matched_config(
        world, instrument,
        camera=nominal,
        C_CI=np.asarray(perturb_calibration(DEFAULT_C_CI, misalignment).matrix),
        accel_bias=(0.0, 0.0, 0.0),
        apply_refraction=s.atmosphere, atmosphere=measured_atmosphere,
        refraction_fraction_sigma=weather_fraction if s.declare else 0.0,
        deflection_model_arcsec=model,
        estimate_calibration=s.estimate_calibration,
        convergence_m=s.convergence_m, **declared)
    if s.declared_overrides:
        config = replace(config, **dict(s.declared_overrides))
    return dict(world=world, instrument=instrument, config=config,
                attitudes=attitudes, n_stars=n_stars,
                reference_options=dict(eop_age_days=s.eop_age_days))


def _apply_fault(fault, pieces, rng):
    """Change the drawn world or simulation options to contain one fault."""
    simulation = {}
    world = pieces["world"]
    n = len(pieces["attitudes"])
    if fault == "vibration_all":
        world = replace(world, vibration_amplitude=0.3)
    elif fault == "vibration_some":
        world = replace(world, vibration_amplitude=0.3,
                        disturbed_observations=tuple(
                            int(i) for i in rng.choice(n, size=3, replace=False)))
    elif fault == "lift":
        world = replace(world, acceleration_enu=(0.0, 0.0, 0.3))
    elif fault == "constant_acceleration":
        angle = rng.uniform(0.0, 2.0 * np.pi)
        world = replace(world, acceleration_enu=(0.1 * np.cos(angle),
                                                 0.1 * np.sin(angle), 0.0))
    elif fault == "mismatch_15":
        simulation["mismatch_fraction"] = 0.15
    elif fault == "mismatch_40":
        simulation["mismatch_fraction"] = 0.40
    elif fault == "few_stars":
        pieces["n_stars"] = 2
    elif fault == "clustered_stars":
        simulation.update(star_layout="clustered", cluster_radius_deg=0.08)
    elif fault == "no_eop":
        pieces["reference_options"] = dict(eop_available=False)
    pieces["world"] = world
    return simulation


def _corrupt(fault, batch, rng):
    if fault == "corrupt_some":
        for index in rng.choice(len(batch), size=2, replace=False):
            batch.observations[int(index)].centroids_px[0, 0] = np.nan
    elif fault == "corrupt_all":
        for observation in batch.observations:
            observation.accel_samples[5, 1] = np.nan


def run_trial(scenario, rng):
    """One instrument, one place, one call of the engine. Returns a record."""
    pieces = draw_trial(scenario, rng)
    simulation = _apply_fault(scenario.fault, pieces, rng)
    batch, catalog, truth = simulate_batch(
        pieces["world"], pieces["instrument"], pieces["attitudes"], rng,
        start_utc=scenario.start_utc, spacing_s=scenario.spacing_s,
        n_stars=pieces["n_stars"], star_layout=simulation.pop(
            "star_layout", scenario.star_layout), **simulation)
    _corrupt(scenario.fault, batch, rng)
    references = ReferenceData(catalog, **pieces["reference_options"])
    result = solve_navigation_fix(batch, pieces["config"], references)

    position = truth["position"]
    record = {"status": result.status, "latitude": position[0],
              "longitude": position[1], "n_stars": pieces["n_stars"],
              "zenith": float(np.mean([a[0] for a in pieces["attitudes"]])),
              "observations_used": result.observations_used,
              "error_m": np.nan, "north_m": np.nan, "east_m": np.nan,
              "sigma_m": np.nan, "nees": np.nan, "withheld_error_m": np.nan,
              "calibration_estimated":
                  result.calibration_status.startswith("ESTIMATED")}
    if result.has_fix:
        longitude = result.longitude_deg if result.longitude_deg is not None else 0.0
        fix = (result.latitude_deg, longitude)
        north, east = north_east_error(fix, position)
        record.update(error_m=float(horizontal_position_error(fix, position)),
                      north_m=float(north), east_m=float(east),
                      sigma_m=result.horizontal_sigma_m,
                      nees=nees([north, east], result.covariance_ne_m2))
    else:
        withheld = result.diagnostics.get("withheld_position")
        if withheld and withheld["latitude_deg"] is not None:
            record["withheld_error_m"] = float(horizontal_position_error(
                (withheld["latitude_deg"], withheld["longitude_deg"] or 0.0),
                position))
    return record


# --------------------------------------------------------------------- statistics
def summarize(records):
    """
    Population statistics. Accuracy numbers are over the trials that
    produced a fix; availability is reported beside them, never folded in.
    """
    statuses = [r["status"] for r in records]
    counts = {status: statuses.count(status) for status in STATUSES}
    fixes = [r for r in records if r["status"] in FIX_STATUSES
             and np.isfinite(r["error_m"])]
    summary = {"trials": len(records), "status_counts": counts,
               "fixes": len(fixes),
               "fix_probability": len(fixes) / max(len(records), 1)}
    if not fixes:
        return summary
    error = np.array([r["error_m"] for r in fixes])
    nees_values = np.array([r["nees"] for r in fixes])
    summary.update(
        rms_m=float(np.sqrt(np.mean(error**2))), median_m=float(np.median(error)),
        p95_m=float(np.percentile(error, 95)), max_m=float(error.max()),
        reported_rms_m=float(np.sqrt(np.mean(
            np.array([r["sigma_m"] for r in fixes]) ** 2))),
        mean_nees=float(np.mean(nees_values)),
        coverage95=float(np.mean(nees_values <= CHI2_95)),
        misleading=int(np.sum(nees_values > MISLEADING_NEES)),
        within_100m=float(np.mean(error <= 100.0)),
        within_200m=float(np.mean(error <= 200.0)),
        within_6nm=float(np.mean(error <= 6 * NAUTICAL_MILE)),
        within_10nm=float(np.mean(error <= 10 * NAUTICAL_MILE)))
    return summary


def run_scenario(scenario, trials, seed):
    """(summary dict, list of trial records). Reproducible from the seed."""
    rng = np.random.default_rng(seed)
    records = [run_trial(scenario, rng) for _ in range(trials)]
    summary = summarize(records)
    summary.update(name=scenario.name, seed=seed)
    return summary, records


def coverage_tolerance(trials, sigmas):
    """Allowed deviation of observed 95 % coverage for a number of trials."""
    return sigmas * np.sqrt(0.95 * 0.05 / trials)


def nees_tolerance(trials, sigmas):
    """Allowed deviation of mean NEES (2 dof, variance 4 per trial) from 2."""
    return sigmas * 2.0 / np.sqrt(trials)
