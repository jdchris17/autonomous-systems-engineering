"""
Test suite 3 - calibration and orientation invariance.

Hold the geographic position fixed and vary the instrument's attitude.
With perfect calibration the answer must not move. With a fixed camera-IMU
misalignment it must move in an attitude-dependent way, and Module 7's
diverse-attitude procedure must then recover both position and calibration.

Acceptance criterion: the estimator detects rank-deficient calibration
configurations and does not report unsupported covariance.
"""

import numpy as np
import pytest

from enduria import ReferenceData, solve_navigation_fix
from enduria.astrometry import STANDARD_ATMOSPHERE
from enduria.estimation import ARCSEC, diverse_attitudes
from enduria.frames import rotation_vector_to_matrix
from enduria.gravity import perturb_calibration
from enduria.simulation import TrueInstrument, World, matched_config, simulate_batch
from helpers import ideal_case, ne_error, noisy_instrument, position_error_m
from tolerances import SEED, tolerance

POSITION = (45.0, -75.0)
WORLD = World(latitude_deg=POSITION[0], longitude_deg=POSITION[1],
              atmosphere=STANDARD_ATMOSPHERE)
MISALIGNMENT_ARCSEC = np.array([120.0, -80.0, 150.0])     # true, fixed
DECLARED_SIGMA_ARCSEC = 180.0

ORIENTATIONS = [(zenith, azimuth, roll)
                for zenith in (5.0, 30.0, 55.0)
                for azimuth, roll in ((0.0, 0.0), (90.0, 135.0), (180.0, 270.0),
                                      (270.0, 45.0))]


def _misaligned_config(instrument, **overrides):
    """Config whose C_CI is wrong by the fixed misalignment (Module 5 convention)."""
    believed = perturb_calibration(instrument.c_ci, MISALIGNMENT_ARCSEC * ARCSEC)
    return matched_config(WORLD, instrument,
                          C_CI=np.asarray(getattr(believed, "matrix", believed)),
                          misalignment_sigma_arcsec=DECLARED_SIGMA_ARCSEC, **overrides)


def _solve(instrument, attitudes, seed, config):
    batch, catalog, truth = simulate_batch(WORLD, instrument, list(attitudes),
                                           np.random.default_rng(seed))
    return solve_navigation_fix(batch, config, ReferenceData(catalog)), truth


# ----------------------------------------------------------- perfect calibration
@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_position_is_invariant_to_attitude_with_perfect_calibration(orientation):
    result, truth = ideal_case(*POSITION, [orientation], SEED + 30)
    assert position_error_m(result, truth["position"]) < tolerance("ideal_position_m")


def test_all_orientations_agree_with_each_other():
    fixes = [ideal_case(*POSITION, [o], SEED + 31)[0] for o in ORIENTATIONS]
    latitudes = [f.latitude_deg for f in fixes]
    longitudes = [f.longitude_deg for f in fixes]
    assert np.ptp(latitudes) * 111e3 < 1e-3 and np.ptp(longitudes) * 79e3 < 1e-3


# ------------------------------------------------------------ fixed misalignment
def test_misalignment_moves_the_fix_with_attitude():
    instrument = TrueInstrument()
    errors = []
    for orientation in [(30.0, az, 0.0) for az in (0.0, 90.0, 180.0, 270.0)]:
        result, truth = _solve(instrument, [orientation], SEED + 32,
                               _misaligned_config(instrument))
        errors.append(ne_error(result, truth["position"]))
    errors = np.array(errors)
    size = np.linalg.norm(errors, axis=1)
    # Hundreds of arcseconds of tilt are kilometres on the ground ...
    assert size.min() > 1000.0
    # ... and turning the instrument half way round reverses the error,
    # because the misalignment is fixed in the instrument, not in the Earth.
    assert np.linalg.norm(errors[0] + errors[2]) < 0.05 * size[0]
    assert np.linalg.norm(errors[1] + errors[3]) < 0.05 * size[1]


def test_single_fix_reports_the_declared_misalignment_in_its_uncertainty():
    instrument = TrueInstrument()
    result, truth = _solve(instrument, [(30.0, 40.0, 15.0)], SEED + 33,
                           _misaligned_config(instrument))
    error = position_error_m(result, truth["position"])
    assert error > 1000.0
    # The error is real, but the reported uncertainty covers it.
    assert error < 3.0 * result.horizontal_sigma_m
    assert result.accuracy_limitations["fixed_instrument_errors_sigma_m"] > 5000.0


def test_diverse_attitudes_recover_position_and_calibration_noise_free():
    instrument = TrueInstrument()
    config = _misaligned_config(instrument, estimate_calibration=True)
    result, truth = _solve(instrument, diverse_attitudes(20, max_zenith=55.0),
                           SEED + 34, config)
    assert result.calibration_status.startswith("ESTIMATED")
    assert position_error_m(result, truth["position"]) < 0.5
    estimate = np.array(result.diagnostics["calibration"]["rotation_vector_arcsec"])
    assert np.abs(estimate - MISALIGNMENT_ARCSEC).max() < tolerance("calibration_arcsec")
    # The corrected matrix is the true one.
    corrected = result.diagnostics["calibration"]["corrected_C_CI"]
    residual = rotation_vector_to_matrix([0, 0, 0]) - corrected @ instrument.c_ci.T
    assert np.abs(residual).max() < 5e-6


def test_diverse_attitudes_recover_position_and_calibration_with_noise():
    instrument = noisy_instrument()
    config = _misaligned_config(instrument, estimate_calibration=True)
    result, truth = _solve(instrument, diverse_attitudes(20, max_zenith=55.0),
                           SEED + 35, config)
    assert result.has_fix and result.calibration_status.startswith("ESTIMATED")
    calibration = result.diagnostics["calibration"]
    estimate = np.array(calibration["rotation_vector_arcsec"])
    sigma = np.array(calibration["sigma_arcsec"])
    assert np.all(np.abs(estimate - MISALIGNMENT_ARCSEC) < 4.0 * sigma)
    assert np.all(sigma < 0.5 * DECLARED_SIGMA_ARCSEC)       # the data taught something
    error = position_error_m(result, truth["position"])
    assert error < 4.0 * result.horizontal_sigma_m
    # Far better than the same instrument held at one attitude.
    held = _solve(instrument, [(30.0, 40.0, 15.0)] * 20, SEED + 35,
                  _misaligned_config(instrument))[0]
    assert result.horizontal_sigma_m < 0.05 * held.horizontal_sigma_m
    # Consistency of two formulations: treating the misalignment as a shared
    # error with its declared covariance (no calibration states) is the same
    # estimator as estimating it with that prior and marginalizing.
    marginal, _ = _solve(instrument, diverse_attitudes(20, max_zenith=55.0),
                         SEED + 35, _misaligned_config(instrument))
    assert marginal.horizontal_sigma_m == pytest.approx(result.horizontal_sigma_m,
                                                        rel=1e-3)
    assert abs(position_error_m(marginal, truth["position"]) - error) < 1.0


# ------------------------------------------------- rank-deficient configurations
@pytest.mark.parametrize("attitudes, label", [
    ([(30.0, 40.0, 15.0)], "one exposure"),
    ([(30.0, 40.0, 15.0)] * 10, "ten exposures at one attitude"),
    ([(30.0, 40.0, 15.0), (50.0, 200.0, 100.0)], "two attitudes"),
])
def test_rank_deficient_calibration_is_detected_and_withheld(attitudes, label):
    instrument = noisy_instrument()
    asked = _misaligned_config(instrument, estimate_calibration=True)
    result, truth = _solve(instrument, attitudes, SEED + 36, asked)
    assert result.calibration_status.startswith("WITHHELD"), label
    calibration = result.diagnostics["calibration"]
    assert calibration["data_observability"] != "well conditioned"
    assert "rotation_vector_arcsec" not in calibration        # no estimate offered
    assert any("WITHHELD" in reason for reason in result.reasons)

    # No unsupported covariance: the uncertainty is exactly what the same
    # batch gets when calibration estimation is not requested at all, and it
    # still carries the full declared misalignment.
    plain, _ = _solve(instrument, attitudes, SEED + 36, _misaligned_config(instrument))
    assert np.allclose(result.covariance_ne_m2, plain.covariance_ne_m2, rtol=1e-9)
    assert result.horizontal_sigma_m > 0.5 * DECLARED_SIGMA_ARCSEC * 30.9
    assert position_error_m(result, truth["position"]) < 4.0 * result.horizontal_sigma_m


def test_more_exposures_at_one_attitude_do_not_shrink_a_fixed_error():
    instrument = noisy_instrument()
    config = _misaligned_config(instrument)
    one, _ = _solve(instrument, [(30.0, 40.0, 15.0)], SEED + 37, config)
    many, _ = _solve(instrument, [(30.0, 40.0, 15.0)] * 20, SEED + 37, config)
    assert many.precision["sigma_north_m"] < 0.4 * one.precision["sigma_north_m"]
    assert many.horizontal_sigma_m > 0.95 * one.horizontal_sigma_m


def test_calibration_request_without_declared_uncertainty_is_withheld():
    instrument = noisy_instrument()
    config = matched_config(WORLD, instrument, estimate_calibration=True)
    result, _ = _solve(instrument, diverse_attitudes(12), SEED + 38, config)
    assert result.has_fix and result.calibration_status.startswith("WITHHELD")
