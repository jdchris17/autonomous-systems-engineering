"""
Test suite 5 - statistical calibration.

Known synthetic truth is used to ask whether the engine's reported
uncertainty is statistically honest. For each population the test reports
horizontal RMS, median, 95th percentile, mean NEES and the fraction of
trials inside the reported 95 % ellipse.

Acceptance criterion: with a correctly specified noise model, coverage and
NEES are statistically compatible with theory. The tolerance is a number
of standard errors, so it tightens automatically with more trials:

    coverage:  95 % +/- k * sqrt(0.95 * 0.05 / N)
    mean NEES:  2   +/- k * 2 / sqrt(N)          (2 degrees of freedom)

Trials per population come from the ENDURIA_TRIALS environment variable
(default 500). The numbers are written to verification_statistics.json for
the verification report.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from enduria.qualification import (
    SCENARIOS,
    Scenario,
    coverage_tolerance,
    nees_tolerance,
    run_scenario,
)
from tolerances import MONTE_CARLO_TRIALS, SEED, tolerance

pytestmark = pytest.mark.statistical
OUTPUT = Path(__file__).resolve().parents[1] / "verification_statistics.json"

RANDOM_ONLY = Scenario(
    name="random errors only",
    description="Centroid and accelerometer noise; every fixed error zero.",
    centroid_noise_px=0.2, accel_noise=0.005, eop_age_days=None,
    temperature_c=(15.0, 15.0), position="global", zenith_deg=(0.0, 55.0))
SHARED = SCENARIOS["B"].with_changes(
    name="shared systematic errors, declared",
    description="Scenario B at random places: bias, misalignment, principal "
                "point, clock, deflection, weather and Earth-orientation errors "
                "drawn once per instrument and declared at their true sigmas.",
    position="global", zenith_deg=(0.0, 55.0), focal_sigma_fraction=0.0)
CALIBRATED = SCENARIOS["C"].with_changes(
    name="calibration states estimated", position="global",
    focal_sigma_fraction=0.0,
    description="Scenario C at random places, calibration estimated in the batch.")
PRECISION = SCENARIOS["D"].with_changes(
    name="precision hardware", position="global", focal_sigma_fraction=0.0,
    description="Scenario D at random places.")
UNDECLARED = SHARED.with_changes(
    name="shared systematic errors, NOT declared", declare=False,
    description="The same errors with nothing declared: a deliberately "
                "mis-specified model, to show the test can fail.")

POPULATIONS = {
    "random": (RANDOM_ONLY, MONTE_CARLO_TRIALS, SEED + 80),
    "shared": (SHARED, MONTE_CARLO_TRIALS, SEED + 81),
    "calibrated": (CALIBRATED, max(MONTE_CARLO_TRIALS // 2, 100), SEED + 82),
    "precision": (PRECISION, max(MONTE_CARLO_TRIALS // 2, 100), SEED + 83),
    "undeclared": (UNDECLARED, max(MONTE_CARLO_TRIALS // 4, 100), SEED + 84),
}
_cache = {}


def population(key):
    if key not in _cache:
        scenario, trials, seed = POPULATIONS[key]
        summary, _ = run_scenario(scenario, trials, seed)
        summary["description"] = scenario.description
        summary["coverage_tolerance"] = coverage_tolerance(
            summary["fixes"], tolerance("coverage_sigmas"))
        summary["nees_tolerance"] = nees_tolerance(
            summary["fixes"], tolerance("nees_sigmas"))
        _cache[key] = summary
        OUTPUT.write_text(json.dumps(_cache, indent=2), encoding="utf-8")
    return _cache[key]


def assert_consistent(summary):
    assert summary["fix_probability"] > 0.97, summary["status_counts"]
    assert abs(summary["coverage95"] - 0.95) <= summary["coverage_tolerance"], summary
    assert abs(summary["mean_nees"] - 2.0) <= summary["nees_tolerance"], summary
    # The reported sigma predicts the actual RMS error.
    assert summary["rms_m"] == pytest.approx(summary["reported_rms_m"], rel=0.12)
    assert summary["median_m"] < summary["rms_m"] < summary["p95_m"]


def test_properly_modelled_random_errors_are_covered():
    assert_consistent(population("random"))


def test_declared_shared_systematic_errors_are_covered():
    summary = population("shared")
    assert_consistent(summary)
    # Fixed errors dominate: far larger than the random-only population.
    assert summary["rms_m"] > 20 * population("random")["rms_m"]


def test_estimated_calibration_is_covered():
    summary = population("calibrated")
    assert_consistent(summary)
    assert summary["rms_m"] < 0.1 * population("shared")["rms_m"]


def test_precision_hardware_is_covered():
    assert_consistent(population("precision"))


def test_undeclared_systematic_errors_are_exposed_by_the_same_statistics():
    """Power of the test: a wrong uncertainty model must fail it, badly."""
    summary = population("undeclared")
    assert summary["fixes"] > 0
    assert summary["coverage95"] < 0.20
    assert summary["mean_nees"] > 50.0
    assert summary["rms_m"] > 10 * summary["reported_rms_m"]


@pytest.mark.parametrize("trials, expected_points", [(100, 7.6), (500, 3.4),
                                                     (1000, 2.4), (10000, 0.76)])
def test_tolerance_depends_on_the_number_of_trials(trials, expected_points):
    points = 100.0 * coverage_tolerance(trials, tolerance("coverage_sigmas"))
    assert points == pytest.approx(expected_points, rel=0.03)
    assert nees_tolerance(trials, 1.0) == pytest.approx(2.0 / np.sqrt(trials))


def test_monte_carlo_is_reproducible_from_its_seed():
    first, records = run_scenario(RANDOM_ONLY, 12, SEED + 85)
    again, _ = run_scenario(RANDOM_ONLY, 12, SEED + 85)
    other, _ = run_scenario(RANDOM_ONLY, 12, SEED + 86)
    assert first["rms_m"] == again["rms_m"] and first["mean_nees"] == again["mean_nees"]
    assert first["rms_m"] != other["rms_m"]
    assert len(records) == 12
