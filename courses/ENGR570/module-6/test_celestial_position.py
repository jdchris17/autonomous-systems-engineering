"""Tests for celestial_position.py and position_simulator.py.
Run with:  python -m pytest -v"""

import numpy as np
import pytest
from astropy.time import Time, TimeDelta

import celestial_position as cp
import position_simulator as ps
from accelerometer_calibration import AccelerometerErrorModel
from astronomical_time import EARTH_ROTATION_RATE_UT1
from imu_alignment import perturb_calibration
from reference_frames import geodetic_to_ecef
from rotation_engine import rotation_vector_to_matrix
from sensor_frames import Transform

UTC = ps.OBSERVATION_UTC
C_EG, EOP_INFO = cp.earth_orientation(UTC)
ARCSEC = np.radians(1 / 3600)


def simulate(lat, lon, seed=0, **kwargs):
    return ps.simulate_observation(lat, lon, UTC, C_EG=C_EG,
                                   rng=np.random.default_rng(seed), **kwargs)


# ------------------------------------------------- earth_fixed_up_to_geodetic
@pytest.mark.parametrize("lat", [-89.9, -60.0, -30.0, 0.0, 30.0, 45.0, 80.0, 89.9])
@pytest.mark.parametrize("lon", [-180.0, -90.0, -75.0, 0.0, 90.0, 179.0])
def test_normal_vector_round_trip(lat, lon):
    out_lat, out_lon = cp.earth_fixed_up_to_geodetic(
        ps.true_vertical_ecef(lat, lon))
    assert out_lat == pytest.approx(lat, abs=1e-12)
    assert cp.longitude_difference(out_lon, lon) == pytest.approx(0, abs=1e-9)
    assert -180.0 <= out_lon < 180.0


def test_latitude_is_geodetic_not_geocentric():
    # The ellipsoid normal at geodetic 45 N gives 45; the POSITION vector of
    # that point is at geocentric latitude 44.81, and would be wrong.
    lat, lon = cp.earth_fixed_up_to_geodetic(ps.true_vertical_ecef(45.0, 10.0))
    assert lat == pytest.approx(45.0, abs=1e-12)
    position = geodetic_to_ecef(45.0, 10.0, 0.0)
    geocentric, _ = cp.earth_fixed_up_to_geodetic(position)
    assert geocentric == pytest.approx(44.8076, abs=1e-3)


def test_normal_is_the_gradient_of_the_ellipsoid_position():
    # Moving up by h shifts the WGS84 position exactly along the normal.
    for lat, lon in [(45.0, -75.0), (-70.0, 130.0), (5.0, 0.0)]:
        step = geodetic_to_ecef(lat, lon, 1000.0) - geodetic_to_ecef(lat, lon, 0.0)
        np.testing.assert_allclose(step / 1000.0, ps.true_vertical_ecef(lat, lon),
                                   atol=1e-9)


def test_axes_and_unnormalized_input():
    assert cp.earth_fixed_up_to_geodetic([1, 0, 0]) == (0.0, 0.0)
    assert cp.earth_fixed_up_to_geodetic([0, 5, 0]) == (0.0, 90.0)
    assert cp.earth_fixed_up_to_geodetic([0, -1, 0]) == (0.0, -90.0)
    assert cp.earth_fixed_up_to_geodetic([-2, 0, 0]) == (0.0, -180.0)


def test_southern_hemisphere_sign():
    lat, lon = cp.earth_fixed_up_to_geodetic(ps.true_vertical_ecef(-33.9, 151.2))
    assert lat == pytest.approx(-33.9) and lon == pytest.approx(151.2)
    assert ps.true_vertical_ecef(-33.9, 151.2)[2] < 0


@pytest.mark.parametrize("up, expected_lat", [([0, 0, 1], 90.0), ([0, 0, -1], -90.0),
                                              ([1e-14, -1e-14, 1], 90.0)])
def test_poles_return_nan_longitude(up, expected_lat):
    lat, lon = cp.earth_fixed_up_to_geodetic(up)
    assert lat == pytest.approx(expected_lat)
    assert np.isnan(lon)


def test_just_off_the_pole_longitude_is_reported():
    lat, lon = cp.earth_fixed_up_to_geodetic(ps.true_vertical_ecef(89.999999, 33.0))
    assert lon == pytest.approx(33.0, abs=1e-6)


def test_longitude_wrapping():
    assert cp.wrap_longitude(179.999) == pytest.approx(179.999)
    assert cp.wrap_longitude(-180.001) == pytest.approx(179.999)
    assert cp.wrap_longitude(180.0) == -180.0
    assert cp.wrap_longitude(540.0) == -180.0
    assert cp.longitude_difference(179.999, -180.001) == pytest.approx(0, abs=1e-9)
    assert cp.longitude_difference(-179.9, 179.9) == pytest.approx(0.2)
    assert cp.longitude_difference(179.9, -179.9) == pytest.approx(-0.2)


@pytest.mark.parametrize("bad", [[0, 0, 0], [np.nan, 0, 1], [1, 0], [np.inf, 0, 0]])
def test_invalid_vectors_are_rejected(bad):
    with pytest.raises(ValueError):
        cp.earth_fixed_up_to_geodetic(bad)


# ------------------------------------------------------- geodesic error metric
def test_geodesic_distances_match_wgs84_values():
    assert cp.horizontal_position_error((1, 0), (0, 0)) == pytest.approx(
        110574.389, abs=1e-3)                    # one degree of latitude
    assert cp.horizontal_position_error((0, 1), (0, 0)) == pytest.approx(
        111319.491, abs=1e-3)                    # one degree of longitude
    assert cp.horizontal_position_error((90, 0), (0, 0)) == pytest.approx(
        10001965.729, abs=1e-3)                  # quarter meridian
    assert cp.horizontal_position_error((45, -75), (45, -75)) == 0.0


def test_geodesic_is_not_the_spherical_approximation():
    spherical = np.radians(1.0) * 6371008.8
    assert abs(cp.horizontal_position_error((1, 0), (0, 0)) - spherical) > 500


def test_geodesic_across_the_date_line_and_symmetry():
    across = cp.horizontal_position_error((0, 179.999), (0, -179.999))
    assert across == pytest.approx(0.002 * 111319.491, rel=1e-6)
    a, b = (45.0, -75.0), (45.01, -74.98)
    assert cp.horizontal_position_error(a, b) == pytest.approx(
        cp.horizontal_position_error(b, a))


def test_north_east_components():
    north, east = cp.north_east_error((45.001, -75.0), (45.0, -75.0))
    assert north == pytest.approx(111.13, abs=0.02) and abs(east) < 1e-6
    north, east = cp.north_east_error((45.0, -74.999), (45.0, -75.0))
    assert east == pytest.approx(78.85, abs=0.02) and abs(north) < 0.01
    north, east = cp.north_east_error((44.999, -75.001), (45.0, -75.0))
    assert north < 0 and east < 0


def test_pole_points_accept_nan_longitude_but_others_do_not():
    assert cp.horizontal_position_error((90.0, np.nan), (90.0, 12.0)) == 0.0
    assert cp.horizontal_position_error((90.0, np.nan), (89.0, 12.0)) == pytest.approx(
        111693.9, abs=1.0)
    with pytest.raises(ValueError, match="NaN"):
        cp.horizontal_position_error((45.0, np.nan), (45.0, 0.0))
    with pytest.raises(ValueError):
        cp.horizontal_position_error((91.0, 0.0), (45.0, 0.0))


# ---------------------------------------------------------------- the solver
def test_celestial_vertical_to_position():
    up_g = C_EG.inverse().apply(ps.true_vertical_ecef(45.0, -75.0))
    lat, lon = cp.celestial_vertical_to_position(up_g, C_EG)
    assert lat == pytest.approx(45.0, abs=1e-12)
    assert lon == pytest.approx(-75.0, abs=1e-12)
    assert cp.celestial_vertical_to_position(up_g, C_EG.matrix) == (lat, lon)


def test_solver_chain_and_result_structure():
    obs = simulate(45.0, -75.0)
    truth = obs.truth
    solution = cp.solve_terrestrial_position(
        truth["up_imu"], obs.C_CI, truth["C_CG"], C_EG, timestamp_utc=UTC,
        earth_orientation_source=EOP_INFO["earth_orientation_source"])
    assert solution.latitude_deg == pytest.approx(45.0, abs=1e-11)
    assert solution.longitude_deg == pytest.approx(-75.0, abs=1e-11)
    np.testing.assert_allclose(solution.up_ecef, truth["up_ecef"], atol=1e-14)
    np.testing.assert_allclose(solution.up_celestial, truth["up_celestial"],
                               atol=1e-14)
    assert solution.valid and solution.status == "ok"
    assert solution.longitude_observable
    assert solution.height_m is None and solution.height_estimated is False
    assert solution.timestamp_utc == UTC
    assert "IERS" in solution.earth_orientation_source
    assert any("height" in a for a in solution.assumptions)
    assert any("ellipsoid normal" in a for a in solution.assumptions)
    assert solution.position_covariance is None        # reserved for Module 7
    assert solution.diagnostics["longitude_dilution"] == pytest.approx(
        1 / np.cos(np.radians(45.0)))
    assert solution.lat_lon == (solution.latitude_deg, solution.longitude_deg)


def test_solver_checks_frames_and_matrices():
    obs = simulate(45.0, -75.0)
    up, c_cg = obs.truth["up_imu"], obs.truth["C_CG"]
    with pytest.raises(ValueError, match="expected C_CI"):
        cp.solve_terrestrial_position(up, obs.C_CI.inverse(), c_cg, C_EG)
    with pytest.raises(ValueError, match="expected C_EG"):
        cp.solve_terrestrial_position(up, obs.C_CI, c_cg, C_EG.inverse())
    with pytest.raises(ValueError, match="expected C_CG"):
        cp.solve_terrestrial_position(up, obs.C_CI,
                                      Transform("G", "C", c_cg.T), C_EG)
    with pytest.raises(ValueError):
        cp.solve_terrestrial_position(up, obs.C_CI, 2 * c_cg, C_EG)
    with pytest.raises(ValueError):
        cp.solve_terrestrial_position([0, 0, 0], obs.C_CI, c_cg, C_EG)


def test_passing_c_gc_instead_of_c_cg_gives_a_wrong_position():
    # Guards the convention: the solver transposes C_CG itself.
    obs = simulate(45.0, -75.0)
    wrong = cp.solve_terrestrial_position(obs.truth["up_imu"], obs.C_CI,
                                          obs.truth["C_CG"].T, C_EG)
    assert cp.horizontal_position_error(wrong.lat_lon, (45.0, -75.0)) > 1e5


def test_pole_solution_is_flagged_not_invented():
    for lat in (90.0, -90.0):
        obs = simulate(lat, 25.0)
        solution = ps.solve_observation(obs)
        assert solution.latitude_deg == pytest.approx(lat, abs=1e-9)
        assert np.isnan(solution.longitude_deg)
        assert not solution.longitude_observable
        assert "pole" in solution.status
        assert solution.valid                     # latitude is still good
        assert cp.horizontal_position_error(solution.lat_lon, (lat, 25.0)) < 1e-6


# ------------------------------------------------------------ the simulator
def test_simulator_geometry_is_self_consistent():
    obs = simulate(-20.0, 140.0, zenith_angle_deg=50.0)
    t = obs.truth
    np.testing.assert_allclose(t["up_celestial"], t["C_EG"].T @ t["up_ecef"],
                               atol=1e-15)
    np.testing.assert_allclose(t["up_camera"], t["C_CG"] @ t["up_celestial"],
                               atol=1e-15)
    np.testing.assert_allclose(obs.C_CI.matrix @ t["up_imu"], t["up_camera"],
                               atol=1e-15)
    # Stationary specific force is g along the true up, in IMU axes.
    np.testing.assert_allclose(t["specific_force_imu"], 9.81 * t["up_imu"],
                               atol=1e-13)
    np.testing.assert_allclose(obs.accel_samples.mean(axis=0),
                               t["specific_force_imu"], atol=1e-13)
    # Stars: measured = C_CG reference, and all inside the field of view.
    np.testing.assert_allclose(obs.measured_stars, obs.reference_stars @ t["C_CG"].T,
                               atol=1e-14)
    assert np.all(obs.measured_stars[:, 2] > 0.9)


@pytest.mark.parametrize("zenith", [0.0, 2.0, 30.0, 60.0, 85.0])
def test_boresight_zenith_angle_is_as_requested(zenith):
    obs = simulate(45.0, -75.0, zenith_angle_deg=zenith, azimuth_deg=70.0)
    assert np.degrees(np.arccos(np.clip(obs.truth["up_camera"][2], -1, 1))) == \
        pytest.approx(zenith, abs=1e-6)
    assert np.linalg.det(obs.truth["C_CG"]) == pytest.approx(1.0)


def test_local_axes_are_right_handed_east_north_up():
    axes = ps.local_axes_ecef(45.0, -75.0)
    np.testing.assert_allclose(axes.T @ axes, np.eye(3), atol=1e-15)
    assert np.linalg.det(axes) == pytest.approx(1.0)
    assert axes[2, 1] > 0                         # north has a +z component
    np.testing.assert_allclose(ps.local_axes_ecef(0.0, 0.0),
                               [[0, 0, 1], [1, 0, 0], [0, 1, 0]], atol=1e-15)


# -------------------------------------------------------- end-to-end recovery
@pytest.mark.parametrize("lat, lon", [(45.0, -75.0), (0.0, 0.0), (-33.9, 151.2),
                                      (80.0, 179.0), (-60.0, -180.0),
                                      (89.9, 90.0), (30.0, 179.999)])
def test_noise_free_position_recovery(lat, lon):
    solution = ps.solve_observation(simulate(lat, lon, seed=3))
    assert cp.horizontal_position_error(solution.lat_lon, (lat, lon)) < 1e-6
    assert solution.latitude_deg == pytest.approx(lat, abs=1e-10)
    assert solution.status == "ok"


def test_estimator_never_receives_the_true_position():
    # Two observers with different truth dictionaries but identical
    # measurements must get identical answers.
    obs = simulate(45.0, -75.0, seed=4)
    first = ps.solve_observation(obs)
    obs.truth.clear()
    second = ps.solve_observation(obs)
    assert first.lat_lon == second.lat_lon


@pytest.mark.parametrize("zenith, azimuth, roll", [(2.0, 0.0, 0.0), (30.0, 200.0, 77.0),
                                                   (85.0, 310.0, -40.0)])
def test_instrument_orientation_does_not_change_the_position(zenith, azimuth, roll):
    solution = ps.solve_observation(simulate(
        45.0, -75.0, seed=5, zenith_angle_deg=zenith, azimuth_deg=azimuth,
        roll_deg=roll))
    assert cp.horizontal_position_error(solution.lat_lon, (45.0, -75.0)) < 1e-6


def test_height_does_not_change_the_solution():
    low = ps.solve_observation(simulate(45.0, -75.0, seed=6, height_m=0.0))
    high = ps.solve_observation(simulate(45.0, -75.0, seed=6, height_m=5000.0))
    assert cp.horizontal_position_error(low.lat_lon, high.lat_lon) < 1e-6
    assert low.height_m is None and high.height_m is None
    # The accelerometer magnitude does differ; the solver does not use it.
    assert (low.diagnostics["specific_force_magnitude"]
            - high.diagnostics["specific_force_magnitude"]) == pytest.approx(
        0.0154, abs=1e-4)


def test_known_accelerometer_calibration_is_applied():
    model = AccelerometerErrorModel(noise_sigma=0.0)
    obs = simulate(45.0, -75.0, seed=7, accel_model=model)
    raw = ps.solve_observation(obs)
    fixed = ps.solve_observation(obs, accel_bias=model.bias_vector,
                                 accel_correction=model.correction_matrix)
    assert cp.horizontal_position_error(raw.lat_lon, (45.0, -75.0)) > 1000
    assert cp.horizontal_position_error(fixed.lat_lon, (45.0, -75.0)) < 1e-6


# ------------------------------------------------------------ error propagation
def test_clock_error_moves_longitude_at_earth_rate():
    obs = simulate(45.0, -75.0, seed=8)
    late = Time(UTC, scale="utc") + TimeDelta(10.0, format="sec")
    solution = ps.solve_observation(obs, assumed_utc=late)
    shift = cp.longitude_difference(solution.longitude_deg, -75.0)
    # A clock that runs fast places the observer WEST of the truth.
    assert shift == pytest.approx(-np.degrees(EARTH_ROTATION_RATE_UT1 * 10.0),
                                  rel=1e-5)
    assert solution.latitude_deg == pytest.approx(45.0, abs=1e-7)
    assert cp.horizontal_position_error(solution.lat_lon, (45.0, -75.0)) == \
        pytest.approx(3294.3, abs=1.0)


def test_extrinsic_misalignment_moves_the_position_by_its_arc_length():
    obs = simulate(45.0, -75.0, seed=9)
    up_c = obs.truth["up_camera"]
    axis = np.cross(up_c, [0, 0, 1.0])
    axis /= np.linalg.norm(axis)
    wrong = perturb_calibration(obs.C_CI, 60 * ARCSEC * axis)   # one arcminute
    solution = ps.solve_observation(obs, C_CI=wrong)
    error = cp.horizontal_position_error(solution.lat_lon, (45.0, -75.0))
    assert error == pytest.approx(1852.0, rel=0.01)             # one nautical mile
    about_vertical = perturb_calibration(obs.C_CI, 60 * ARCSEC * up_c)
    unaffected = ps.solve_observation(obs, C_CI=about_vertical)
    assert cp.horizontal_position_error(unaffected.lat_lon, (45.0, -75.0)) < 1e-5


def test_horizontal_acceleration_gives_a_plausible_wrong_position():
    obs = simulate(45.0, -75.0, seed=10, acceleration_enu=(0.05, 0.0, 0.0),
                   accel_model=AccelerometerErrorModel(
                       bias=(0, 0, 0), scale_factor_error=0.0, cross_axis=0.0,
                       noise_sigma=0.005), n_accel_samples=1000)
    solution = ps.solve_observation(obs, accel_noise_sigma=0.005)
    error = cp.horizontal_position_error(solution.lat_lon, (45.0, -75.0))
    assert error == pytest.approx(np.arctan(0.05 / 9.81) * 6.37e6, rel=0.02)
    assert solution.valid and solution.status == "ok"   # nothing looks wrong
    assert solution.quality.verdict == "steady"
    assert "cannot prove" in solution.quality.caveat


def test_vibration_invalidates_the_solution():
    obs = simulate(45.0, -75.0, seed=11, n_accel_samples=200)
    shaken = obs.accel_samples.copy()
    shaken[:, 0] += 0.3 * np.sin(np.arange(200))
    obs.accel_samples = shaken + np.random.default_rng(0).normal(0, 0.005, shaken.shape)
    solution = ps.solve_observation(obs, accel_noise_sigma=0.005)
    assert not solution.valid
    assert "quality check failed" in solution.status


def test_earth_orientation_reports_its_source():
    transform, info = cp.earth_orientation(UTC)
    assert transform.name == "C_EG"
    assert info["timestamp_utc"].startswith("2026-11-11T12:00:00")
    assert "IERS" in info["earth_orientation_source"]
    assert abs(info["ut1_minus_utc_s"]) < 0.9
    np.testing.assert_allclose(transform.matrix.T @ transform.matrix, np.eye(3),
                               atol=1e-14)
    # A rotation matrix rotated back by one sidereal turn of ERA is close to
    # the same matrix: check it really is time dependent.
    later, _ = cp.earth_orientation("2026-11-11T13:00:00")
    assert np.degrees(np.arccos((np.trace(later.matrix @ transform.matrix.T) - 1)
                                / 2)) == pytest.approx(15.041, abs=0.001)
