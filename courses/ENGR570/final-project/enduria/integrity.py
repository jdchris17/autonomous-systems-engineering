"""
integrity.py - stage 6: turn a solution into a NavigationResult.

Decides the status, composes the reported uncertainty and writes down what
was assumed. Module 7's acceptance policy (convergence, observability,
residual chi-square, per-measurement outliers) is applied unchanged to the
batch solution; this file adds the checks that only exist at system level
(excluded observations, rejected star matches, star geometry, reference
data, polar longitude) and the uncertainty terms that are not measurement
noise.

Statuses
  REJECTED      a validity check failed; no position is reported.
  UNOBSERVABLE  the measurements do not determine the position.
  DEGRADED      a position is reported, with a stated reason to trust it less.
  VALID         every available check passed. Conditional on `assumptions`.
"""

import platform

import numpy as np
from scipy.stats import chi2 as chi2_distribution

from .astrometry import refraction_coefficients
from .contracts import DEGRADED, REJECTED, UNOBSERVABLE, VALID, NavigationResult
from .estimation import ARCSEC, assess_batch_result, radians_to_metres
from .frames import WGS84_A, WGS84_E2
from .timekeeping import EARTH_ROTATION_RATE_UT1

STAR_CHI2_TAIL = 1e-3
POLE_SIGMAS = 3.0                 # longitude withheld if the pole is this close
EARTH_RADIUS_M = 6371008.8

UNVERIFIABLE = (
    "The instrument was not accelerating. A constant acceleration is "
    "indistinguishable from a tilt and leaves every residual clean: 1 mm/s^2 "
    "of undetected horizontal acceleration moves the fix by about 650 m.",
    "The accelerometer calibration and camera-IMU alignment are as "
    "configured, within their declared uncertainties.",
    "The catalog identity of every star that passed residual screening is "
    "correct.",
    "The timestamps are correct within the declared clock uncertainty.",
)


def _versions():
    import astropy
    import erfa
    import scipy
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "astropy": astropy.__version__,
            "erfa": erfa.__version__}


def _reference_text(references, eop):
    if eop is None:
        return "not evaluated"
    if not eop.available:
        return f"DEGRADED: {eop.source}"
    age = (f", {eop.days_predicted:.0f} days past last measured value"
           if eop.is_prediction else "")
    return f"OK: {eop.source}{age}; catalog '{references.catalog_name}'"


def _first_time(batch):
    try:
        return str(batch.observations[0].time_utc)
    except Exception:
        return "unknown"


def empty_result(status, reasons, batch, references, context):
    """A result that reports no position."""
    try:
        described = references.describe()
    except Exception:
        described = {}
    return NavigationResult(
        latitude_deg=None, longitude_deg=None, covariance_ne_m2=None,
        status=status, observation_time_utc=_first_time(batch),
        observations_used=0, stars_used=0, residual_rms_arcsec=None,
        calibration_status="not evaluated", reference_status="not evaluated",
        diagnostics={"excluded_observations": context["excluded"],
                     "iterations": context["iterations"],
                     "references": described, "versions": _versions()},
        assumptions=[], reasons=list(reasons) + _exclusion_reasons(context, None),
        integrity={"checks": {}, "observations_excluded": len(context["excluded"])},
        observability={"classification": "not evaluated"})


def _exclusion_reasons(context, total):
    by_category = {}
    for item in context["excluded"]:
        by_category.setdefault(item["category"], []).append(item)
    reasons = []
    for category, items in by_category.items():
        indices = [i["observation"] for i in items]
        of = "" if total is None else f" of {total}"
        reasons.append(f"{len(items)}{of} observation(s) excluded ({category}): "
                       f"{items[0]['reason']} [observations {indices[:8]}]")
    return reasons


# ---------------------------------------------------------------- uncertainty
def systematic_terms(config, exposures, solution, latitude_deg):
    """
    Declared systematic and reference-model uncertainty, as north/east
    covariances in metres squared. None of these average down with more
    observations from the same place and instrument.
    """
    from .pipeline import effective_atmosphere

    lat = np.radians(latitude_deg)
    meridian = radians_to_metres(lat)[0, 0]
    prime_vertical = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    terms = {}

    deflection = config.deflection_sigma_arcsec * ARCSEC
    terms["deflection_of_vertical"] = np.diag([(deflection * meridian) ** 2,
                                               (deflection * prime_vertical) ** 2])

    eop = exposures[0].eop
    time_sigma = 0.0 if solution.clock["estimated"] else float(
        np.hypot(config.clock_sigma_s, eop.ut1_sigma_s))
    east = EARTH_ROTATION_RATE_UT1 * time_sigma * prime_vertical * np.cos(lat)
    terms["time_and_ut1"] = np.diag([0.0, east**2])

    polar = eop.polar_motion_sigma_arcsec * ARCSEC * EARTH_RADIUS_M
    terms["polar_motion"] = np.diag([polar**2, polar**2])

    atmosphere, fraction, _ = effective_atmosphere(config)
    effect = np.zeros(2)
    if atmosphere is not None and fraction > 0.0:
        a, b = refraction_coefficients(atmosphere)
        for e in exposures:
            # A fractional error in the refraction tilts this exposure's
            # attitude by that fraction of the refraction at the boresight,
            # along the boresight azimuth.
            east_c, north_c, up_c = e.boresight_enu
            horizontal = np.hypot(east_c, north_c)
            if horizontal < 1e-9:
                continue
            tan_z = horizontal / max(up_c, 1e-6)
            refraction = a * tan_z + b * tan_z**3
            effect += refraction * np.array([north_c * meridian,
                                             east_c * prime_vertical]) / horizontal
        effect /= len(exposures)
    terms["refraction_model"] = fraction**2 * np.outer(effect, effect)
    return terms


# -------------------------------------------------------------------- the result
def build_result(batch, config, references, exposures, solution, converged,
                 context):
    from .pipeline import effective_atmosphere

    total = len(batch.observations)
    rejected, unobservable, degraded, notes = [], [], [], []
    checks = {}

    # ---- Module 7 acceptance policy on the batch solution --------------------------
    m7 = assess_batch_result(solution.batch, solution.measurements, reference={})
    core = [r for r in m7.reasons
            if not r.startswith(("stationarity and calibration", "converged,",
                                 "no covariance"))]
    {REJECTED: rejected, UNOBSERVABLE: unobservable, DEGRADED: degraded,
     VALID: notes}[m7.status].extend(core if m7.status != VALID else [])
    checks["estimator_convergence"] = bool(solution.batch.converged)
    checks["observability"] = m7.metrics.get("observability") == "well conditioned"
    checks["residual_consistency"] = not any("chi2" in r or "inconsistent" in r
                                             for r in core)
    if not converged:
        rejected.append(
            f"position did not settle within {config.max_iterations} iterations "
            f"(last update {context['iterations'][-1].get('update_m', float('nan')):.3g} m)")
    checks["reference_iteration_converged"] = bool(converged)

    # ---- observations set aside -----------------------------------------------------
    degraded += _exclusion_reasons(context, total)
    checks["all_observations_usable"] = not context["excluded"]

    # ---- star matches -----------------------------------------------------------------
    stars_used = int(sum(e.used.sum() for e in exposures))
    stars_rejected = int(sum(len(e.rejected_stars) for e in exposures))
    worst_fraction = max(len(e.rejected_stars) /
                         max(len(e.rejected_stars) + int(e.used.sum()), 1)
                         for e in exposures)
    checks["star_matches_consistent"] = \
        worst_fraction <= config.max_rejected_star_fraction
    if worst_fraction > config.max_rejected_star_fraction:
        degraded.append(
            f"unresolved integrity risk: up to {100 * worst_fraction:.0f}% of the "
            "star matches in one exposure were rejected as inconsistent; the "
            "remaining matches may not be the correct set")
    elif stars_rejected:
        notes.append(f"{stars_rejected} star match(es) rejected by residual "
                     "screening and not used")
    chi2 = sum(e.star_chi2 for e in exposures if e.star_chi2 is not None)
    dof = sum(e.star_dof for e in exposures if e.star_chi2 is not None)
    star_tail = float(chi2_distribution.sf(chi2, dof)) if dof > 0 else None
    checks["star_residuals_match_noise_model"] = \
        None if star_tail is None else star_tail >= STAR_CHI2_TAIL
    if star_tail is not None and star_tail < STAR_CHI2_TAIL:
        degraded.append(
            f"star residuals exceed the centroid noise model (chi2 {chi2:.0f} "
            f"for {dof} dof): possible wrong matches, lens model error or a "
            "missing correction")
    below = sum(e.below_altitude for e in exposures)
    if below:
        notes.append(f"{below} star(s) below {config.min_star_altitude_deg:g} deg "
                     "altitude were not used (refraction model unreliable)")
    for e in exposures:
        notes.extend(e.notes)
    notes.extend(context.get("notes", []))

    # ---- star geometry ------------------------------------------------------------------
    worst_attitude = max(e.attitude_sigma_arcsec for e in exposures)
    checks["star_geometry"] = worst_attitude <= config.attitude_sigma_limit_arcsec
    if worst_attitude > config.attitude_sigma_limit_arcsec:
        degraded.append(
            f"poor star geometry: weakest attitude axis has 1-sigma "
            f"{worst_attitude:,.0f} arcsec (limit "
            f"{config.attitude_sigma_limit_arcsec:g}); this is included in the "
            "covariance")

    # ---- reference data ------------------------------------------------------------------
    eop = exposures[0].eop
    checks["earth_orientation_data"] = bool(eop.available)
    if not eop.available:
        degraded.append(f"{eop.source}; longitude uncertainty enlarged to cover "
                        f"{eop.ut1_sigma_s:g} s of UT1")
    atmosphere, _, atmosphere_text = effective_atmosphere(config)
    checks["atmospheric_data"] = (None if not config.apply_refraction
                                  else config.atmosphere is not None)
    if config.apply_refraction and config.atmosphere is None:
        degraded.append("no atmospheric data supplied: refraction computed for "
                        "a standard atmosphere, with enlarged uncertainty")

    # ---- calibration and clock states ----------------------------------------------------
    for state in (solution.calibration, solution.clock):
        if state["requested"] and not state["estimated"]:
            notes.append(state["status"])

    # ---- uncertainty ----------------------------------------------------------------------
    lat, lon = solution.geodetic
    covariance = precision = None
    limitations = {}
    if solution.covariance_ne is not None:
        terms = systematic_terms(config, exposures, solution, lat)
        instrument = solution.covariance_ne - solution.precision_ne
        covariance = solution.covariance_ne + sum(terms.values())
        covariance = 0.5 * (covariance + covariance.T)
        precision = solution.precision_ne
        limitations = {
            "fixed_instrument_errors_sigma_m": float(
                np.sqrt(max(np.trace(instrument), 0.0))),
            **{f"{name}_sigma_m": float(np.sqrt(np.trace(value)))
               for name, value in terms.items()},
            "undetected_acceleration_m_per_mm_s2": 1e-3 / 9.81 * EARTH_RADIUS_M,
            "not_modelled": [
                "acceleration of the instrument",
                "error in a calibration beyond its declared sigma",
                "refraction anomalies (inversions, horizontal gradients)",
            ],
        }

    # ---- longitude near a pole --------------------------------------------------------------
    longitude = lon if solution.longitude_defined else None
    pole_distance = (90.0 - abs(lat)) * np.pi / 180.0 * EARTH_RADIUS_M
    if covariance is not None:
        major = float(np.sqrt(np.linalg.eigvalsh(covariance).max()))
        if pole_distance < POLE_SIGMAS * major:
            longitude = None
    checks["longitude_defined"] = longitude is not None
    if longitude is None:
        degraded.append(
            f"the pole is {pole_distance:,.1f} m away, inside the position "
            "uncertainty: longitude is not defined and is not reported; use "
            "latitude with the north/east covariance")

    # ---- status ----------------------------------------------------------------------------------
    if rejected:
        status, reasons = REJECTED, rejected + unobservable + degraded
    elif unobservable or covariance is None:
        status, reasons = UNOBSERVABLE, unobservable + degraded
        if not unobservable:
            reasons.insert(0, "the estimator returned no covariance")
    elif degraded:
        status, reasons = DEGRADED, degraded
    else:
        status = VALID
        reasons = ["all available checks passed; conditional on the listed "
                   "assumptions"]
    reasons = reasons + notes

    residuals = np.concatenate([e.star_residuals for e in exposures
                                if e.star_residuals is not None])
    residual_rms = float(np.sqrt(np.mean(residuals**2)) / ARCSEC) \
        if residuals.size else None
    accepted = status in (VALID, DEGRADED)

    assumptions = list(UNVERIFIABLE) + [
        "Local up is the WGS84 ellipsoid normal"
        + (" corrected by the configured deflection model"
           if config.deflection_model_arcsec is not None else "")
        + f" (declared residual {config.deflection_sigma_arcsec:g} arcsec per "
          "component).",
        f"Atmosphere: {atmosphere_text}.",
        f"Earth orientation: {eop.source}.",
        f"Observer height {config.height_m:g} m is given, not estimated.",
    ]
    position_record = {"latitude_deg": lat,
                       "longitude_deg": lon if solution.longitude_defined else None,
                       "covariance_ne_m2": covariance}
    diagnostics = {
        "iterations": context["iterations"],
        "excluded_observations": context["excluded"],
        "per_observation": [{
            "observation": e.index, "utc": e.eop.utc,
            "stars_used": int(e.used.sum()), "stars_rejected": e.rejected_stars,
            "attitude_sigma_arcsec": e.attitude_sigma_arcsec,
            "accelerometer_verdict": e.quality.verdict,
            "specific_force": e.measurement.diagnostics["specific_force_magnitude"],
        } for e in exposures],
        "batch": {"chi2": solution.batch.chi2, "dof": solution.batch.dof,
                  "state_names": solution.batch.state_names,
                  "function_evaluations":
                      solution.batch.diagnostics["function_evaluations"],
                  "polar_working_frame": solution.polar_frame},
        "astronomical_position": solution.astronomical,
        "earth_fixed_up": solution.up_earth_fixed,
        "calibration": solution.calibration,
        "clock": solution.clock,
        "earth_orientation": eop.record(),
        "references": references.describe(),
        "versions": _versions(),
        "observation_span": (exposures[0].eop.utc, exposures[-1].eop.utc),
    }
    if not accepted:
        diagnostics["withheld_position"] = position_record

    return NavigationResult(
        latitude_deg=lat if accepted else None,
        longitude_deg=longitude if accepted else None,
        covariance_ne_m2=covariance if accepted else None,
        status=status,
        observation_time_utc=exposures[0].eop.utc,
        observations_used=len(exposures),
        stars_used=stars_used,
        residual_rms_arcsec=residual_rms,
        calibration_status=solution.calibration["status"],
        reference_status=_reference_text(references, eop),
        diagnostics=diagnostics,
        assumptions=assumptions,
        reasons=reasons,
        precision={} if precision is None else {
            "covariance_ne_m2": precision,
            "sigma_north_m": float(np.sqrt(precision[0, 0])),
            "sigma_east_m": float(np.sqrt(precision[1, 1])),
            "meaning": "measurement noise only, under the declared noise model"},
        accuracy_limitations=limitations,
        integrity={"checks": checks, "stars_rejected": stars_rejected,
                   "observations_excluded": len(context["excluded"]),
                   "observations_supplied": total,
                   "unverified_assumptions": list(UNVERIFIABLE)},
        observability={
            "classification": m7.metrics.get("observability"),
            "state_names": solution.batch.state_names,
            "singular_values": m7.metrics.get("singular_values"),
            "condition_number": m7.metrics.get("condition_number"),
            "weakest_direction": m7.metrics.get("weakest_direction")})
