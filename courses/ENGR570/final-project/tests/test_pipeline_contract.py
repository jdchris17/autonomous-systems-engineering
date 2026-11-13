"""
Work Package B - the public operation and the navigation-result contract.

One function runs a synthetic batch from start to finish and returns one
standard object. These tests pin down that object: its fields, what is
None when, and the four things it must report separately (precision,
accuracy limitations, integrity, observability).
"""

import copy
import dataclasses

import numpy as np
import pytest

from enduria import (
    FIX_STATUSES,
    STATUSES,
    VALID,
    NavigationResult,
    ReferenceData,
    solve_navigation_fix,
)
from enduria.astrometry import STANDARD_ATMOSPHERE
from enduria.simulation import World, matched_config, simulate_batch
from helpers import DEFAULT_ATTITUDE, noisy_instrument, position_error_m
from tolerances import SEED

CONTRACT_FIELDS = {
    "latitude_deg", "longitude_deg", "covariance_ne_m2", "status",
    "observation_time_utc", "observations_used", "stars_used",
    "residual_rms_arcsec", "calibration_status", "reference_status",
    "diagnostics", "assumptions"}


@pytest.fixture(scope="module")
def case():
    world = World(atmosphere=STANDARD_ATMOSPHERE, deflection_arcsec=(4.0, -3.0))
    instrument = noisy_instrument(accel_bias=(0.004, -0.002, 0.003))
    batch, catalog, truth = simulate_batch(world, instrument, [DEFAULT_ATTITUDE] * 10,
                                           np.random.default_rng(SEED + 70))
    config = matched_config(world, instrument, accel_bias=(0.0, 0.0, 0.0),
                            accel_bias_sigma=0.005, misalignment_sigma_arcsec=60.0,
                            clock_sigma_s=0.1, deflection_sigma_arcsec=1.0)
    references = ReferenceData(catalog)
    return batch, config, references, truth, solve_navigation_fix(batch, config,
                                                                  references)


def test_result_has_the_contract_fields_with_the_contract_types(case):
    result = case[4]
    assert isinstance(result, NavigationResult)
    assert CONTRACT_FIELDS <= {f.name for f in dataclasses.fields(result)}
    assert isinstance(result.latitude_deg, float)
    assert isinstance(result.longitude_deg, float)
    assert result.covariance_ne_m2.shape == (2, 2)
    assert result.status in STATUSES
    assert isinstance(result.observation_time_utc, str)
    assert result.observations_used == 10 and result.stars_used == 200
    assert 1.0 < result.residual_rms_arcsec < 30.0       # 0.2 px is about 7.6 arcsec
    assert isinstance(result.calibration_status, str)
    assert isinstance(result.reference_status, str)
    assert isinstance(result.diagnostics, dict)
    assert isinstance(result.assumptions, list) and result.assumptions


def test_covariance_is_a_covariance(case):
    covariance = case[4].covariance_ne_m2
    assert np.allclose(covariance, covariance.T)
    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)


def test_precision_and_accuracy_limitations_are_separate(case):
    result = case[4]
    precision = result.precision["covariance_ne_m2"]
    # Random uncertainty alone is smaller than the reported total ...
    assert np.all(np.linalg.eigvalsh(result.covariance_ne_m2 - precision) > 0.0)
    assert np.trace(precision) < 0.2 * np.trace(result.covariance_ne_m2)
    # ... and each systematic term is itemized in metres.
    limits = result.accuracy_limitations
    for term in ("fixed_instrument_errors_sigma_m", "deflection_of_vertical_sigma_m",
                 "time_and_ut1_sigma_m", "polar_motion_sigma_m",
                 "refraction_model_sigma_m"):
        assert limits[term] >= 0.0
    assert limits["deflection_of_vertical_sigma_m"] == pytest.approx(
        np.hypot(30.87, 30.96), rel=0.02)            # 1 arcsec north and east
    assert limits["time_and_ut1_sigma_m"] == pytest.approx(0.1 * 329.4, rel=0.02)
    # The itemized terms add up to the reported covariance.
    total = sum(limits[t] ** 2 for t in limits if t.endswith("_sigma_m")) \
        + np.trace(precision)
    assert total == pytest.approx(np.trace(result.covariance_ne_m2), rel=1e-6)


def test_integrity_and_observability_are_reported(case):
    result = case[4]
    checks = result.integrity["checks"]
    for name in ("estimator_convergence", "observability", "residual_consistency",
                 "star_matches_consistent", "star_geometry",
                 "earth_orientation_data", "longitude_defined"):
        assert name in checks
    assert result.integrity["unverified_assumptions"]
    assert result.observability["classification"] == "well conditioned"
    assert result.observability["state_names"] == ["latitude", "longitude"]
    assert len(result.observability["singular_values"]) == 2


def test_valid_is_conditional_and_says_on_what(case):
    result = case[4]
    assert result.status == VALID
    assert "conditional" in result.reasons[0]
    text = " ".join(result.assumptions)
    for phrase in ("not accelerating", "calibration", "catalog identity",
                   "timestamps", "ellipsoid normal", "Atmosphere",
                   "Earth orientation"):
        assert phrase in text


def test_diagnostics_carry_provenance(case):
    diagnostics = case[4].diagnostics
    assert {"python", "numpy", "scipy", "astropy", "erfa"} <= set(
        diagnostics["versions"])
    assert diagnostics["references"]["downloads"] == "disabled"
    assert diagnostics["earth_orientation"]["source"]
    assert len(diagnostics["per_observation"]) == 10
    stages = [step["stage"] for step in diagnostics["iterations"]]
    assert stages[0].startswith("initial") and stages[-1] == "corrected references"
    assert diagnostics["iterations"][-1]["update_m"] < 0.01


def test_the_answer_is_right_within_its_own_uncertainty(case):
    truth, result = case[3], case[4]
    assert position_error_m(result, truth["position"]) < 3.5 * result.horizontal_sigma_m


def test_same_input_gives_the_same_output_and_inputs_are_not_modified(case):
    batch, config, references, _, first = case
    before = copy.deepcopy(batch)
    again = solve_navigation_fix(batch, config, references)
    assert again.latitude_deg == first.latitude_deg
    assert again.longitude_deg == first.longitude_deg
    assert np.array_equal(again.covariance_ne_m2, first.covariance_ne_m2)
    for original, now in zip(before.observations, batch.observations):
        assert np.array_equal(original.centroids_px, now.centroids_px)
        assert np.array_equal(original.accel_samples, now.accel_samples)
        assert original.star_ids == now.star_ids


def test_no_position_without_validity_information(case):
    """Whatever the status, a position never appears without its covariance."""
    batch, config, references, _, _ = case
    damaged = copy.deepcopy(batch)
    for observation in damaged.observations:
        observation.accel_samples = observation.accel_samples * 0.0
    refused = solve_navigation_fix(damaged, config, references)
    assert refused.status not in FIX_STATUSES
    assert refused.latitude_deg is None and refused.longitude_deg is None
    assert refused.covariance_ne_m2 is None and not refused.has_fix
    assert refused.reasons and "no position reported" in refused.summary()


def test_summary_is_readable(case):
    text = case[4].summary()
    assert text.startswith("VALID: lat +4") and "1-sigma" in text
