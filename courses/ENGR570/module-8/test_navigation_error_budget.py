"""Tests for navigation_error_budget.py. Run with:  python -m pytest -v"""

import numpy as np
import pytest

import astrometric_corrections as ac
import navigation_error_budget as nb
from atmospheric_refraction import STANDARD_ATMOSPHERE, refraction_arcsec
from compare_reference_frames import reference_matrix

cp = nb.cp
EARTH = nb.EarthOrientation(nb.BASELINE.start_utc)
PERFECT = nb.perfect(nb.BASELINE)
TRUTH = (45.0, -75.0)
ARCSEC_M = np.radians(1 / 3600) * 6.37e6           # ~30.9 m per arcsecond


def one(scenario, fixed=None, seed=0):
    """Position error [m] and details for one instrument."""
    measurements, truth = nb.simulate_instrument(
        scenario, np.random.default_rng(seed), EARTH, fixed)
    position, covariance, result = nb.solve_instrument(scenario, measurements, truth)
    north, east = cp.north_east_error(position, TRUTH)
    return float(np.hypot(north, east)), north, east, covariance, result


# ------------------------------------------------------------------ camera model
def test_camera_model_reduces_to_the_pinhole_camera():
    plain = nb.PinholeCamera()
    model = nb.CameraModel()
    rays = np.array([[0.0, 0, 1], [0.1, -0.05, 1], [-0.15, 0.08, 1]])
    np.testing.assert_allclose(model.project(rays)[0], plain.project(rays)[0])
    assert model.fx == pytest.approx(plain.fx)


def test_distortion_convention_and_inverse():
    camera = nb.CameraModel(k1=0.02)
    edge = np.array([[np.tan(np.radians(10.0)), 0.0, 1.0]])
    shift = camera.project(edge)[0][0, 0] - nb.CameraModel().project(edge)[0][0, 0]
    assert shift == pytest.approx(0.02 * np.tan(np.radians(10)) ** 3 * camera.fx)
    assert shift == pytest.approx(0.6, abs=0.05)                # pixels
    rng = np.random.default_rng(0)
    rays = np.column_stack((rng.uniform(-0.17, 0.17, 50),
                            rng.uniform(-0.09, 0.09, 50), np.ones(50)))
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    for k1 in (-0.2, 0.02, 0.3):
        cam = nb.CameraModel(k1=k1, cx=963.0, cy=538.0, focal_scale=1.01)
        np.testing.assert_allclose(cam.unproject(cam.project(rays)[0]), rays,
                                   atol=1e-13)


def test_focal_scale_and_principal_point_move_pixels_as_expected():
    ray = np.array([[0.1, 0.05, 1.0]])
    base = nb.CameraModel().project(ray)[0][0]
    scaled = nb.CameraModel(focal_scale=1.01).project(ray)[0][0]
    np.testing.assert_allclose(scaled - [960, 540], 1.01 * (base - [960, 540]))
    moved = nb.CameraModel(cx=962.0, cy=539.0).project(ray)[0][0]
    np.testing.assert_allclose(moved - base, [2.0, -1.0])


# ------------------------------------------------------------ Earth orientation
def test_fast_earth_matrix_matches_the_module_2_reference():
    np.testing.assert_allclose(EARTH.matrix(), reference_matrix(EARTH.start_utc),
                               atol=1e-14)
    np.testing.assert_allclose(EARTH.matrix(10.0),
                               reference_matrix("2026-11-11T12:00:10"), atol=1e-12)


def test_earth_orientation_record_carries_source_and_uncertainty():
    record = EARTH.record()
    assert "IERS" in record["source"] and record["is_prediction"]
    assert record["days_predicted"] == pytest.approx(13.5, abs=0.6)
    assert 0 < record["ut1_sigma_s"] < 0.01
    assert 0 < record["polar_motion_sigma_arcsec"] < 0.05
    ut1, pm = nb.prediction_uncertainty(365.0)
    assert ut1 == pytest.approx(0.00025 * 365**0.75)
    assert pm == pytest.approx(0.00068 * 365**0.80)
    assert nb.prediction_uncertainty(-3.0) == (0.0, 0.0)


def test_earth_orientation_perturbations_have_the_expected_size():
    up = EARTH.matrix().T @ np.array([1.0, 0.0, 0.0])        # a celestial vector

    def shift(**kwargs):
        moved = EARTH.matrix(**kwargs) @ up
        return np.degrees(np.arccos(np.clip(moved @ np.array([1.0, 0, 0]), -1, 1))) * 3600

    assert shift(dut1_error_s=1.0) == pytest.approx(15.041, abs=0.01)
    assert shift(xp_error_rad=np.radians(1 / 3600)) == pytest.approx(1.0, abs=0.01)
    assert shift(dx_rad=np.radians(1 / 3600)) < 1.01
    assert shift() == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------------- single instruments
def test_perfect_scenario_recovers_the_position():
    error, *_ = one(PERFECT)
    assert error < 0.1


def test_isolate_and_perfect_only_change_error_fields():
    lone = nb.isolate(nb.BASELINE, "Clock")
    assert lone.clock_sigma_s == nb.BASELINE.clock_sigma_s
    assert lone.accel_bias_sigma == 0 and lone.atmosphere is None
    assert lone.n_stars == nb.BASELINE.n_stars
    assert set(nb.SOURCES) >= {"Accelerometer bias", "Atmospheric refraction"}
    assert {c for c, _ in nb.SOURCES.values()} == {
        "random", "fixed", "environmental", "geographic"}


@pytest.mark.parametrize("deflection", [(5.0, 0.0), (0.0, -10.0), (3.0, 4.0)])
def test_uncorrected_deflection_moves_the_position_by_its_arc(deflection):
    scenario = PERFECT.with_changes(deflection_sigma_arcsec=1.0)
    error, north, east, *_ = one(scenario, {"deflection_arcsec": deflection})
    assert error == pytest.approx(np.hypot(*deflection) * ARCSEC_M, rel=0.01)
    assert np.sign(north) == np.sign(deflection[0]) or deflection[0] == 0
    assert np.sign(east) == np.sign(deflection[1]) or deflection[1] == 0


def test_deflection_correction_removes_it():
    scenario = PERFECT.with_changes(deflection_sigma_arcsec=1.0,
                                    deflection_model_residual_arcsec=0.0)
    error, *_ = one(scenario, {"deflection_arcsec": (8.0, -6.0)})
    assert error < 0.5


def test_clock_error_gives_the_earth_rotation_displacement():
    scenario = PERFECT.with_changes(clock_sigma_s=1.0)
    error, north, east, *_ = one(scenario, {"clock_s": 1.0})
    assert error == pytest.approx(329.4, abs=1.5)
    assert east < 0 and abs(north) < 1.0             # a fast clock puts you west


def test_principal_point_error_is_a_pointing_error():
    scenario = PERFECT.with_changes(principal_point_sigma_px=1.0)
    error, *_ = one(scenario, {"principal_point_px": (1.0, 0.0)})
    focal = nb.CameraModel().fx
    assert error == pytest.approx(np.degrees(1.0 / focal) * 3600 * ARCSEC_M, rel=0.2)


def test_misalignment_error_and_its_removal_by_calibration_states():
    fixed = {"misalignment_rad": np.radians([0.05, -0.03, 0.04])}
    plain = PERFECT.with_changes(misalignment_sigma_arcsec=180.0,
                                 centroid_noise_px=0.2, accel_noise=0.005)
    error_plain, *_ = one(plain, fixed)
    solved = plain.with_changes(diverse_attitudes=True, n_observations=20,
                                estimate_calibration=True)
    error_solved, *_ = one(solved, fixed)
    assert error_plain > 2000.0
    assert error_solved < 400.0


def test_unmodelled_refraction_biases_the_position_by_about_the_refraction():
    scenario = PERFECT.with_changes(atmosphere=STANDARD_ATMOSPHERE)
    error, *_ = one(scenario)
    expected = refraction_arcsec(60.0) * ARCSEC_M          # boresight at 60 deg altitude
    assert error == pytest.approx(expected, rel=0.1)
    corrected, *_ = one(scenario.with_changes(model_refraction=True))
    assert corrected < 0.1


def test_refraction_matters_little_for_a_zenith_pointing_camera():
    low = one(PERFECT.with_changes(atmosphere=STANDARD_ATMOSPHERE,
                                   zenith_angle_deg=60.0))[0]
    high = one(PERFECT.with_changes(atmosphere=STANDARD_ATMOSPHERE,
                                    zenith_angle_deg=0.0))[0]
    assert low > 2500.0 and high < 0.05 * low


def test_unmodelled_apparent_place_is_bounded_by_aberration():
    scenario = PERFECT.with_changes(stellar_apparent_place=True)
    error, *_ = one(scenario)
    assert 30.0 < error < 20.6 * ARCSEC_M
    corrected, *_ = one(scenario.with_changes(model_apparent_place=True))
    assert corrected < 0.1


def test_proper_motion_only_matters_when_the_catalog_is_not_propagated():
    world = PERFECT.with_changes(stellar_apparent_place=True,
                                 proper_motion_mas_yr=100.0)
    aberration_only = world.with_changes(model_proper_motion=True)
    assert one(world)[0] != pytest.approx(one(aberration_only)[0], abs=1.0)
    assert one(world.with_changes(model_apparent_place=True))[0] < 0.1


def test_fast_apparent_place_path_matches_the_rigorous_chain():
    # The budget obtains the ICRS direction of each observed star by
    # inverting the ERFA chain. Forward-modelling those stars rigorously
    # must reproduce the observed directions.
    rng = np.random.default_rng(3)
    location = ac.ObserverLocation(45.0, -75.0)
    camera_axes = nb.ps.camera_axes_local(30.0, 100.0, 20.0)
    rays = nb.CameraModel().unproject(rng.uniform(100, 900, (15, 2)))
    enu = rays @ camera_axes.T
    icrs = ac.icrs_from_observed_enu(enu, nb.BASELINE.start_utc, location,
                                     STANDARD_ATMOSPHERE)
    stars = [ac.CatalogStar(ra_deg=float(np.degrees(np.arctan2(v[1], v[0]))),
                            dec_deg=float(np.degrees(np.arcsin(v[2]))),
                            epoch_jyear=2000.0) for v in icrs]
    _, forward = ac.observed_directions(stars, nb.BASELINE.start_utc, location,
                                        STANDARD_ATMOSPHERE)
    np.testing.assert_allclose(forward, enu, atol=1e-9)


def test_image_to_imu_timing_error_needs_platform_motion():
    still = PERFECT.with_changes(image_imu_offset_s=0.1)
    moving = still.with_changes(platform_rate_deg_s=1.0)
    assert one(still)[0] < 0.1
    # 1 deg/s for 0.1 s is 0.1 deg = 360 arcsec about a horizontal axis.
    assert one(moving)[0] == pytest.approx(360 * ARCSEC_M, rel=0.15)


def test_declared_uncertainty_covers_the_fixed_errors():
    scenario = nb.BASELINE.with_changes(**nb.MODULE7_WORLD)
    _, _, _, covariance, _ = one(scenario)
    naive = one(scenario.with_changes(declare_uncertainties=False))[3]
    assert np.sqrt(np.trace(covariance)) > 5000.0
    assert np.sqrt(np.trace(naive)) < 300.0


# ------------------------------------------------------------------ Monte Carlo
def test_run_error_budget_is_reproducible_and_structured():
    a = nb.run_error_budget(nb.BASELINE, trials=12, seed=5, contributions=False)
    b = nb.run_error_budget(nb.BASELINE, trials=12, seed=5, contributions=False)
    np.testing.assert_array_equal(a.errors_ne, b.errors_ne)
    assert a.errors_ne.shape == (12, 2) and a.horizontal.shape == (12,)
    assert a.rms >= a.median > 0 and a.p95 >= a.median
    assert a.repeatability_rms < a.rms               # fixed errors dominate
    assert a.diagnostics["earth_orientation"]["source"].startswith("IERS")
    c = nb.run_error_budget(nb.BASELINE, trials=12, seed=6, contributions=False)
    assert not np.array_equal(a.errors_ne, c.errors_ne)


def test_contributions_add_up_to_roughly_the_total():
    result = nb.run_error_budget(nb.BASELINE, trials=120, seed=11,
                                 contribution_trials=120)
    assert set(result.contributions) == set(nb.SOURCES)
    root_sum_square = np.sqrt(sum(c["rms"] ** 2
                                  for c in result.contributions.values()))
    assert root_sum_square == pytest.approx(result.rms, rel=0.25)
    largest = max(result.contributions, key=lambda k: result.contributions[k]["rms"])
    assert largest == "Camera-IMU misalignment"
    assert result.contributions["Clock"]["category"] == "fixed"


def test_baseline_covariance_is_roughly_consistent():
    result = nb.run_error_budget(nb.BASELINE, trials=150, seed=12,
                                 contributions=False)
    assert 85.0 < result.coverage95 <= 100.0
    assert result.reported_rms == pytest.approx(result.rms, rel=0.25)


def test_sensitivity_is_one_for_the_dominant_parameter_and_zero_for_a_minor_one():
    scenario = PERFECT.with_changes(misalignment_sigma_arcsec=180.0,
                                    clock_sigma_s=0.01)
    out = nb.sensitivity_analysis(
        scenario, {"misalignment_sigma_arcsec": [0.5, 2.0],
                   "clock_sigma_s": [0.5, 2.0]}, trials=40, seed=1)
    assert out["misalignment_sigma_arcsec"]["sensitivity"] == pytest.approx(1.0, abs=0.05)
    assert abs(out["clock_sigma_s"]["sensitivity"]) < 0.05
    assert out["clock_sigma_s"]["baseline_value"] == 0.01


def test_compare_improvements_ranks_by_rms_reduction():
    rows = nb.compare_improvements(
        nb.BASELINE,
        {"better alignment": {"misalignment_sigma_arcsec": 18.0},
         "better clock": {"clock_sigma_s": 0.01}}, trials=60, seed=2)
    assert rows[0]["name"] == "baseline" and rows[0]["rms_ratio"] == 1.0
    assert rows[1]["name"] == "better alignment"
    assert rows[1]["rms_reduction"] > 1000.0 > abs(rows[2]["rms_reduction"])
    assert rows[1]["rms"] == pytest.approx(rows[0]["rms"] - rows[1]["rms_reduction"])
    assert rows[1]["p95_reduction"] > 0
