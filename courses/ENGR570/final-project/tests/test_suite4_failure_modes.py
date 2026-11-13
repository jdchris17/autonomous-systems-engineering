"""
Test suite 4 - navigation failure modes.

One group of tests per row of the failure-condition table. Each asserts the
EXPECTED SYSTEM BEHAVIOUR, which is sometimes "refuse", sometimes "report a
larger uncertainty", and in one case "admit that this cannot be detected".

Tests marked `known_limitation` pass when the software correctly declines
to claim something it cannot know. They document limits, not capabilities.
"""

import copy

import numpy as np
import pytest

from enduria import (
    DEGRADED,
    REJECTED,
    UNOBSERVABLE,
    VALID,
    NavigationConfig,
    ObservationBatch,
    ReferenceData,
    solve_navigation_fix,
)
from enduria.astrometry import STANDARD_ATMOSPHERE
from enduria.estimation import diverse_attitudes
from enduria.simulation import World, matched_config, simulate_batch
from helpers import (
    DEFAULT_ATTITUDE,
    noisy_instrument,
    position_error_m,
    run,
)
from tolerances import SEED

POSITION = (45.0, -75.0)
ATTITUDES = [DEFAULT_ATTITUDE] * 10
known_limitation = pytest.mark.known_limitation


def world(**changes):
    settings = dict(latitude_deg=POSITION[0], longitude_deg=POSITION[1],
                    atmosphere=STANDARD_ATMOSPHERE)
    settings.update(changes)
    return World(**settings)


def baseline(seed, **simulation):
    """A healthy noisy batch and everything needed to solve it."""
    w, instrument = world(), noisy_instrument()
    batch, catalog, truth = simulate_batch(w, instrument, ATTITUDES,
                                           np.random.default_rng(seed), **simulation)
    return batch, catalog, truth, matched_config(w, instrument)


def test_healthy_batch_is_valid():
    batch, catalog, truth, config = baseline(SEED + 40)
    result = solve_navigation_fix(batch, config, ReferenceData(catalog))
    assert result.status == VALID, result.summary()
    assert position_error_m(result, truth["position"]) < 4 * result.horizontal_sigma_m
    assert all(v is not False for v in result.integrity["checks"].values())


# --------------------------------------------------- insufficient matched stars
@pytest.mark.parametrize("n_stars", [0, 1, 2])
def test_too_few_stars_everywhere_reports_unobservable_attitude(n_stars):
    batch, catalog, _, config = baseline(SEED + 41, n_stars=max(n_stars, 1))
    if n_stars == 0:
        for observation in batch.observations:
            observation.centroids_px = np.zeros((0, 2))
            observation.star_ids = []
    result = solve_navigation_fix(batch, config, ReferenceData(catalog))
    assert result.status == UNOBSERVABLE
    assert result.latitude_deg is None and result.covariance_ne_m2 is None
    assert "attitude" in " ".join(result.reasons)


def test_exposures_with_too_few_stars_are_set_aside():
    counts = [20, 20, 2, 20, 1, 20, 20, 20, 20, 20]
    batch, catalog, truth, config = baseline(SEED + 42, n_stars=counts)
    result = solve_navigation_fix(batch, config, ReferenceData(catalog))
    assert result.status == DEGRADED and result.observations_used == 8
    excluded = result.diagnostics["excluded_observations"]
    assert sorted(x["observation"] for x in excluded) == [2, 4]
    assert all(x["category"] == "insufficient stars" for x in excluded)
    assert position_error_m(result, truth["position"]) < 4 * result.horizontal_sigma_m


# -------------------------------------------- incorrect catalog correspondences
def test_gross_mismatches_are_detected_and_removed():
    clean = baseline(SEED + 43)
    batch, catalog, truth, config = baseline(SEED + 43, mismatch_fraction=0.15)
    result = solve_navigation_fix(batch, config, ReferenceData(catalog))
    wrong_ids = {wrong for o in truth["observations"] for _, wrong in o["mismatched"]}
    rejected = {s for o in result.diagnostics["per_observation"]
                for s in o["stars_rejected"]}
    assert len(wrong_ids) == 30 and wrong_ids <= rejected
    assert len(rejected - wrong_ids) <= 2            # false rejections are rare
    assert result.has_fix
    reference = solve_navigation_fix(clean[0], clean[3], ReferenceData(clean[1]))
    assert position_error_m(result, truth["position"]) < \
        2.0 * position_error_m(reference, POSITION) + 3 * result.horizontal_sigma_m


def test_mismatches_that_escape_rejection_are_caught_by_the_residual_test():
    """
    With rejection effectively switched off, wrong matches a few hundred
    arcseconds out stay in the fit. The pooled star-residual test still
    sees that the centroids do not fit the declared noise.
    """
    batch, catalog, _, config = baseline(SEED + 43, mismatch_fraction=0.15,
                                         mismatch_offset_deg=(0.02, 0.06))
    blind = solve_navigation_fix(
        batch, NavigationConfig(**{**config.__dict__, "outlier_false_alarm": 1e-300}),
        ReferenceData(catalog))
    assert blind.integrity["stars_rejected"] == 0
    assert blind.status != VALID
    assert blind.integrity["checks"]["star_residuals_match_noise_model"] is False
    screened = solve_navigation_fix(batch, config, ReferenceData(catalog))
    assert screened.integrity["stars_rejected"] >= 30
    assert screened.integrity["checks"]["star_residuals_match_noise_model"] is True


def test_many_mismatches_leave_an_unresolved_integrity_risk():
    batch, catalog, truth, config = baseline(SEED + 44, mismatch_fraction=0.4)
    result = solve_navigation_fix(batch, config, ReferenceData(catalog))
    assert result.status in (DEGRADED, REJECTED, UNOBSERVABLE)
    text = " ".join(result.reasons)
    assert "integrity risk" in text or "inconsistent" in text
    if result.has_fix:
        assert position_error_m(result, truth["position"]) < 5 * result.horizontal_sigma_m


@known_limitation
def test_a_mismatch_inside_the_noise_cannot_be_detected():
    """A wrong star 10 arcsec from the right one fits a 7.6 arcsec noise model."""
    batch, catalog, truth, config = baseline(
        SEED + 45, mismatch_fraction=0.1,
        mismatch_offset_deg=(10 / 3600, 12 / 3600))
    result = solve_navigation_fix(batch, config, ReferenceData(catalog))
    assert result.integrity["stars_rejected"] <= 2          # not caught ...
    assert result.has_fix
    # ... and the software does not claim otherwise.
    assert any("catalog identity" in a for a in result.assumptions)


# ------------------------------------------------------------ poor star geometry
def test_clustered_stars_degrade_with_larger_reported_uncertainty():
    spread = baseline(SEED + 46)
    good = solve_navigation_fix(spread[0], spread[3], ReferenceData(spread[1]))
    batch, catalog, truth, config = baseline(SEED + 46, star_layout="clustered",
                                             cluster_radius_deg=0.08)
    result = solve_navigation_fix(batch, config, ReferenceData(catalog))
    assert result.status == DEGRADED
    assert any("poor star geometry" in r for r in result.reasons)
    assert result.integrity["checks"]["star_geometry"] is False
    assert result.horizontal_sigma_m > 3.0 * good.horizontal_sigma_m
    assert position_error_m(result, truth["position"]) < 4 * result.horizontal_sigma_m


def test_coincident_stars_give_no_attitude():
    batch, catalog, _, config = baseline(SEED + 47, n_stars=5)
    for observation in batch.observations:
        first = observation.star_ids[0]
        observation.centroids_px = np.tile(observation.centroids_px[0], (3, 1)) \
            + np.arange(3)[:, None] * 1e-9
        duplicates = {f"{first}-copy{i}": catalog[first] for i in range(3)}
        catalog.update(duplicates)
        observation.star_ids = list(duplicates)
    result = solve_navigation_fix(batch, config, ReferenceData(catalog))
    assert not result.has_fix and result.latitude_deg is None


# -------------------------------------------------- significant detected acceleration
@pytest.mark.parametrize("disturbance", [
    dict(vibration_amplitude=0.3),                      # shaking
    dict(acceleration_enu=(0.0, 0.0, 0.3)),             # lift: changes |f|
])
def test_detected_motion_everywhere_is_rejected(disturbance):
    w, instrument = world(**disturbance), noisy_instrument()
    result, truth, *_ = run(w, instrument, ATTITUDES, SEED + 48,
                            matched_config(world(), instrument))
    assert result.status == REJECTED
    assert result.latitude_deg is None
    assert "motion" in " ".join(result.reasons)


def test_disturbed_exposures_are_excluded_and_the_fix_is_degraded():
    w = world(vibration_amplitude=0.3, disturbed_observations=(1, 5, 6))
    instrument = noisy_instrument()
    result, truth, *_ = run(w, instrument, ATTITUDES, SEED + 49,
                            matched_config(world(), instrument))
    assert result.status == DEGRADED and result.observations_used == 7
    excluded = result.diagnostics["excluded_observations"]
    assert sorted(x["observation"] for x in excluded) == [1, 5, 6]
    assert position_error_m(result, truth["position"]) < 4 * result.horizontal_sigma_m


def test_rotation_seen_by_the_gyro_is_rejected():
    w, instrument = world(), noisy_instrument(gyro_noise=1e-4)
    batch, catalog, _ = simulate_batch(w, instrument, ATTITUDES,
                                       np.random.default_rng(SEED + 50))
    for observation in batch.observations:
        observation.gyro_samples = observation.gyro_samples + [0.01, 0.0, 0.0]
    result = solve_navigation_fix(batch, matched_config(w, instrument),
                                  ReferenceData(catalog))
    assert result.status == REJECTED and "rotating" in " ".join(result.reasons)


# -------------------------------------------------- constant undetected acceleration
@known_limitation
@pytest.mark.parametrize("acceleration", [0.005, 0.02])
def test_constant_horizontal_acceleration_is_not_detected_and_not_denied(acceleration):
    """
    The case the sensors cannot see. A steady horizontal acceleration looks
    exactly like a tilt. The software must (1) not pretend to detect it and
    (2) carry the limitation in every result, quantified.
    """
    w = world(acceleration_enu=(acceleration, 0.0, 0.0))
    instrument = noisy_instrument()
    result, truth, *_ = run(w, instrument, ATTITUDES, SEED + 51,
                            matched_config(world(), instrument))
    error = position_error_m(result, truth["position"])
    # (1) every check passes and the answer is wrong by far more than sigma.
    assert result.status == VALID
    assert error > 10 * result.horizontal_sigma_m
    assert all(o["accelerometer_verdict"] == "steady"
               for o in result.diagnostics["per_observation"])
    # (2) the limitation is stated, with its size.
    assert any("not accelerating" in a for a in result.assumptions)
    assert any("not accelerating" in a
               for a in result.integrity["unverified_assumptions"])
    assert "acceleration of the instrument" in \
        result.accuracy_limitations["not_modelled"]
    per_mm_s2 = result.accuracy_limitations["undetected_acceleration_m_per_mm_s2"]
    assert error == pytest.approx(per_mm_s2 * acceleration * 1000.0, rel=0.05)


# ------------------------------------------------------- missing astronomical data
def test_no_earth_orientation_data_degrades_and_widens_longitude():
    batch, catalog, truth, config = baseline(SEED + 52)
    with_data = solve_navigation_fix(batch, config, ReferenceData(catalog))
    without = solve_navigation_fix(batch, config,
                                   ReferenceData(catalog, eop_available=False))
    assert without.status == DEGRADED
    assert without.reference_status.startswith("DEGRADED")
    assert without.integrity["checks"]["earth_orientation_data"] is False
    east = lambda r: np.sqrt(r.covariance_ne_m2[1, 1])
    added = np.sqrt(east(without) ** 2 - east(with_data) ** 2)
    assert 250.0 < added < 350.0                 # 0.9 s of UT1 is ~297 m here
    assert without.covariance_ne_m2[0, 0] == pytest.approx(
        with_data.covariance_ne_m2[0, 0], rel=0.05)
    assert position_error_m(without, truth["position"]) < 4 * without.horizontal_sigma_m


def test_date_beyond_the_earth_orientation_table_does_not_raise():
    w = world(eop_error=(0.35, 0.0, 0.0))             # the real UT1-UTC that day
    instrument = noisy_instrument()
    result, truth, *_ = run(w, instrument, ATTITUDES, SEED + 53,
                            matched_config(world(), instrument),
                            start_utc="2035-06-01T04:00:00")
    assert result.status == DEGRADED
    assert "no Earth-orientation data" in result.reference_status
    assert position_error_m(result, truth["position"]) < 4 * result.horizontal_sigma_m


def test_stale_earth_orientation_data_is_propagated_not_ignored():
    batch, catalog, _, config = baseline(SEED + 54)
    fresh = solve_navigation_fix(batch, config, ReferenceData(catalog, eop_age_days=0))
    stale = solve_navigation_fix(batch, config,
                                 ReferenceData(catalog, eop_age_days=730))
    assert stale.accuracy_limitations["time_and_ut1_sigma_m"] > \
        10 * fresh.accuracy_limitations["time_and_ut1_sigma_m"]
    assert "730 days" in stale.reference_status


def test_stars_missing_from_the_catalog_are_ignored_and_reported():
    batch, catalog, truth, config = baseline(SEED + 55)
    trimmed = dict(catalog)
    removed = batch.observations[0].star_ids[:4]
    for identity in removed:
        del trimmed[identity]
    result = solve_navigation_fix(batch, config, ReferenceData(trimmed))
    assert result.has_fix and result.stars_used == 196
    assert any("not in the catalog" in r for r in result.reasons)
    nothing = solve_navigation_fix(batch, config, ReferenceData({}))
    assert nothing.status == UNOBSERVABLE and nothing.latitude_deg is None


def test_missing_weather_data_uses_a_standard_atmosphere_and_says_so():
    w, instrument = world(), noisy_instrument()
    attitudes = [(55.0, 40.0, 15.0)] * 10
    told = matched_config(w, instrument)
    untold = matched_config(w, instrument, atmosphere=None, apply_refraction=True)
    result, truth, batch, catalog = run(w, instrument, attitudes, SEED + 56, untold)
    informed = solve_navigation_fix(batch, told, ReferenceData(catalog))
    assert result.status == DEGRADED
    assert any("standard atmosphere" in r for r in result.reasons)
    assert result.accuracy_limitations["refraction_model_sigma_m"] > 200.0
    assert informed.accuracy_limitations["refraction_model_sigma_m"] == 0.0
    assert any("standard atmosphere assumed" in a for a in result.assumptions)


def test_refraction_is_refused_when_configured_off():
    w, instrument = world(), noisy_instrument()
    config = matched_config(w, instrument, apply_refraction=False)
    result, *_ = run(w, instrument, [(55.0, 40.0, 15.0)] * 10, SEED + 57, config)
    assert any("refraction not applied" in a for a in result.assumptions)
    assert result.integrity["checks"]["atmospheric_data"] is None


# ----------------------------------------------------------- uncertain absolute time
def test_clock_uncertainty_becomes_longitude_uncertainty():
    w, instrument = world(clock_error_s=2.0), noisy_instrument()
    sure = matched_config(world(), instrument)
    unsure = matched_config(world(), instrument, clock_sigma_s=2.0)
    result, truth, batch, catalog = run(w, instrument, ATTITUDES, SEED + 58, unsure)
    overconfident = solve_navigation_fix(batch, sure, ReferenceData(catalog))
    north, east = np.sqrt(np.diag(result.covariance_ne_m2))
    assert east > 2.0 * 300.0                 # 2 s at 45 deg is about 658 m
    assert north == pytest.approx(np.sqrt(overconfident.covariance_ne_m2[0, 0]),
                                  rel=0.01)
    error = position_error_m(result, truth["position"])
    assert 400.0 < error < 4 * result.horizontal_sigma_m
    # Without the declaration the same error is far outside the ellipse.
    assert error > 4 * overconfident.horizontal_sigma_m


def test_clock_offset_is_withheld_when_the_span_cannot_separate_it():
    w, instrument = world(), noisy_instrument()
    config = matched_config(w, instrument, clock_sigma_s=1.0, estimate_clock=True)
    result, *_ = run(w, instrument, ATTITUDES, SEED + 59, config)
    assert result.has_fix
    assert result.diagnostics["clock"]["estimated"] is False
    assert any("clock offset" in r for r in result.reasons)


# ------------------------------------------------------ poor calibration observability
def test_calibration_is_withheld_at_one_attitude_and_estimated_with_many():
    w, instrument = world(), noisy_instrument()
    config = matched_config(w, instrument, misalignment_sigma_arcsec=180.0,
                            estimate_calibration=True)
    held, *_ = run(w, instrument, ATTITUDES, SEED + 60, config)
    moved, *_ = run(w, instrument, diverse_attitudes(15, max_zenith=55.0),
                    SEED + 60, config)
    assert held.calibration_status.startswith("WITHHELD")
    assert moved.calibration_status.startswith("ESTIMATED")


# ------------------------------------------------------------ extreme polar geometry
def test_next_to_the_pole_longitude_is_withheld_not_invented():
    w = world(latitude_deg=89.9999, longitude_deg=40.0)       # 11 m from the pole
    instrument = noisy_instrument()
    result, truth, *_ = run(w, instrument, ATTITUDES, SEED + 61,
                            matched_config(w, instrument))
    assert result.status == DEGRADED
    assert result.latitude_deg is not None and result.longitude_deg is None
    assert result.integrity["checks"]["longitude_defined"] is False
    assert result.covariance_ne_m2 is not None
    assert position_error_m(result, (90.0, 0.0)) < 4 * result.horizontal_sigma_m


def test_near_the_pole_position_is_good_in_metres_and_longitude_sigma_is_honest():
    w = world(latitude_deg=89.9, longitude_deg=40.0)          # 11 km from the pole
    instrument = noisy_instrument()
    result, truth, *_ = run(w, instrument, ATTITUDES, SEED + 62,
                            matched_config(w, instrument))
    assert result.has_fix and result.longitude_deg is not None
    assert position_error_m(result, truth["position"]) < 4 * result.horizontal_sigma_m
    # The same east uncertainty is a far larger longitude uncertainty here.
    east_sigma = np.sqrt(result.covariance_ne_m2[1, 1])
    longitude_sigma_deg = np.degrees(east_sigma / (6.39e6 * np.cos(np.radians(89.9))))
    assert longitude_sigma_deg > 100 * np.degrees(east_sigma / 6.39e6 / np.cos(0.785))
    assert result.diagnostics["batch"]["polar_working_frame"] is True


# -------------------------------------------------------- invalid or nonphysical input
def _corrupt(batch, index, change):
    damaged = copy.deepcopy(batch)
    change(damaged.observations[index])
    return damaged


def _set(attribute, value):
    return lambda observation: setattr(observation, attribute, value)


def _poke(attribute, index, value):
    def change(observation):
        array = np.array(getattr(observation, attribute), dtype=float)
        array[index] = value
        setattr(observation, attribute, array)
    return change


CORRUPTIONS = {
    "NaN centroid": _poke("centroids_px", (3, 0), np.nan),
    "infinite centroid": _poke("centroids_px", (0, 1), np.inf),
    "centroid off the sensor": _poke("centroids_px", (2, 0), 5000.0),
    "negative pixel": _poke("centroids_px", (2, 1), -40.0),
    "centroid array wrong shape": _set("centroids_px", np.zeros((20, 3))),
    "identity count mismatch": lambda o: o.star_ids.pop(),
    "duplicate identity": lambda o: o.star_ids.__setitem__(1, o.star_ids[0]),
    "NaN accelerometer sample": _poke("accel_samples", (7, 2), np.nan),
    "accelerometer wrong shape": _set("accel_samples", np.zeros((200, 2))),
    "single accelerometer sample": _set("accel_samples", np.array([[0, 0, 9.8]])),
    "zero specific force": _set("accel_samples", np.zeros((200, 3))),
    "free fall then 3 g": _set("accel_samples", np.full((200, 3), 17.0)),
    "text instead of numbers": _set("accel_samples", "not data"),
    "unparseable time": _set("time_utc", "the day after tomorrow"),
    "missing time": _set("time_utc", None),
    "malformed gyro": _set("gyro_samples", np.full((200, 3), np.nan)),
}


@pytest.mark.parametrize("name", list(CORRUPTIONS))
def test_one_corrupt_exposure_is_set_aside_with_a_diagnostic(name):
    batch, catalog, truth, config = baseline(SEED + 63)
    damaged = _corrupt(batch, 4, CORRUPTIONS[name])
    result = solve_navigation_fix(damaged, config, ReferenceData(catalog))
    assert result.status == DEGRADED, f"{name}: {result.summary()}"
    excluded = result.diagnostics["excluded_observations"]
    assert [x["observation"] for x in excluded] == [4]
    assert excluded[0]["category"] == "invalid input" and excluded[0]["reason"]
    assert result.observations_used == 9
    assert position_error_m(result, truth["position"]) < 4 * result.horizontal_sigma_m


@pytest.mark.parametrize("name", list(CORRUPTIONS))
def test_a_batch_that_is_all_corrupt_is_rejected_without_raising(name):
    batch, catalog, _, config = baseline(SEED + 64)
    damaged = copy.deepcopy(batch)
    for observation in damaged.observations:
        CORRUPTIONS[name](observation)
    result = solve_navigation_fix(damaged, config, ReferenceData(catalog))
    assert result.status == REJECTED, name
    assert result.latitude_deg is None and result.longitude_deg is None
    assert result.covariance_ne_m2 is None and result.reasons


@pytest.mark.parametrize("bad_batch", [None, ObservationBatch([]), "batch", 42])
def test_unusable_batch_objects_are_rejected_without_raising(bad_batch):
    _, catalog, _, config = baseline(SEED + 65)
    result = solve_navigation_fix(bad_batch, config, ReferenceData(catalog))
    assert result.status == REJECTED and result.latitude_deg is None


@pytest.mark.parametrize("change", [
    dict(C_CI=None), dict(C_CI=np.ones((3, 3))), dict(C_CI=2.0 * np.eye(3)),
    dict(centroid_sigma_px=0.0), dict(accel_noise_sigma=-1.0),
    dict(clock_sigma_s=float("nan")), dict(initial_position=(120.0, 0.0)),
])
def test_invalid_configuration_is_rejected_without_raising(change):
    batch, catalog, _, config = baseline(SEED + 66)
    broken = NavigationConfig(**{**config.__dict__, **change})
    result = solve_navigation_fix(batch, broken, ReferenceData(catalog))
    assert result.status == REJECTED and "invalid input" in result.reasons[0]


def test_wrong_object_types_are_rejected_without_raising():
    batch, catalog, _, config = baseline(SEED + 67)
    assert solve_navigation_fix(batch, {"camera": None},
                                ReferenceData(catalog)).status == REJECTED
    assert solve_navigation_fix(batch, config, None).status == REJECTED


def test_a_wrong_initial_position_is_noticed_and_not_used():
    batch, catalog, truth, config = baseline(SEED + 68)
    hinted = NavigationConfig(**{**config.__dict__, "initial_position": (-10.0, 100.0)})
    result = solve_navigation_fix(batch, hinted, ReferenceData(catalog))
    plain = solve_navigation_fix(batch, config, ReferenceData(catalog))
    assert any("initial position" in r for r in result.reasons)
    assert position_error_m(result, truth["position"]) == pytest.approx(
        position_error_m(plain, truth["position"]), abs=0.01)
