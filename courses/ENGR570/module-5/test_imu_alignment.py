"""Tests for imu_alignment.py and accelerometer_calibration.py.
Run with:  python -m pytest -v"""

import numpy as np
import pytest

from accelerometer_calibration import (
    IDEAL_SENSOR,
    AccelerometerErrorModel,
    estimate_calibration,
    simulate_stationary,
    six_position_orientations,
)
from imu_alignment import (
    GRAVITY,
    Transform,
    angular_rms,
    imu_orientation,
    orientation_from_rotation_vector,
    orientation_with_up,
    perturb_calibration,
    rotation_vector_to_matrix,
    specific_force,
    true_up_imu,
    vertical_covariance,
    vertical_in_celestial,
    vertical_jacobians,
)

TOL = 1e-13
C_CI = Transform("C", "I", rotation_vector_to_matrix([0.3, -0.2, 0.9]))
C_GC = Transform("G", "C", rotation_vector_to_matrix([-1.1, 0.4, 0.2]))


def angle(a, b):
    return np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))


# ------------------------------------------------------------- IMU orientation
def test_level_imu_has_axes_east_north_up():
    c_in = imu_orientation()
    assert (c_in.to_frame, c_in.from_frame) == ("I", "N")
    np.testing.assert_allclose(c_in.matrix, np.eye(3), atol=TOL)
    np.testing.assert_allclose(true_up_imu(c_in), [0, 0, 1], atol=TOL)


@pytest.mark.parametrize("tilt", [0.0, 1.0, 10.0, 45.0, 90.0, 135.0, 180.0])
@pytest.mark.parametrize("azimuth", [0.0, 70.0, 200.0])
def test_tilt_is_the_angle_between_imu_z_and_vertical(tilt, azimuth):
    up = true_up_imu(imu_orientation(tilt, azimuth, heading_deg=33.0))
    assert np.degrees(angle(up, [0, 0, 1])) == pytest.approx(tilt, abs=1e-10)
    assert np.linalg.norm(up) == pytest.approx(1.0)


def test_heading_alone_does_not_move_the_vertical():
    np.testing.assert_allclose(true_up_imu(imu_orientation(0.0, 0.0, 123.0)),
                               [0, 0, 1], atol=TOL)


def test_tilt_about_east_axis_tips_z_toward_south():
    # Right-handed +10 deg about east carries the IMU z axis from up toward
    # -north, so UP has a positive component along the IMU +y axis.
    up = true_up_imu(imu_orientation(10.0, tilt_azimuth_deg=0.0))
    np.testing.assert_allclose(
        up, [0, np.sin(np.radians(10)), np.cos(np.radians(10))], atol=TOL)


def test_orientation_with_up_reproduces_the_requested_vector():
    rng = np.random.default_rng(0)
    for _ in range(50):
        up = rng.normal(size=3)
        c_in = orientation_with_up(up, heading_deg=rng.uniform(0, 360))
        np.testing.assert_allclose(true_up_imu(c_in), up / np.linalg.norm(up),
                                   atol=TOL)
        assert np.linalg.det(c_in.matrix) == pytest.approx(1.0)


def test_orientation_from_rotation_vector():
    c_in = orientation_from_rotation_vector([0, 0, np.pi / 2])
    # IMU x axis now points north, so north reads as +x in IMU coordinates.
    np.testing.assert_allclose(c_in.apply([0, 1, 0]), [1, 0, 0], atol=TOL)


# -------------------------------------------------------------- specific force
def test_stationary_specific_force_points_up_with_magnitude_g():
    np.testing.assert_allclose(specific_force(imu_orientation()),
                               [0, 0, 9.81], atol=TOL)
    c_in = imu_orientation(25.0, 40.0, 10.0)
    f = specific_force(c_in)
    assert np.linalg.norm(f) == pytest.approx(GRAVITY)
    np.testing.assert_allclose(f / GRAVITY, true_up_imu(c_in), atol=TOL)


def test_horizontal_acceleration_tilts_the_apparent_vertical():
    f = specific_force(imu_orientation(), acceleration_n=[0.05, 0, 0])
    np.testing.assert_allclose(f, [0.05, 0, 9.81], atol=TOL)
    apparent_tilt = angle(f, [0, 0, 1])
    assert apparent_tilt == pytest.approx(np.arctan(0.05 / 9.81))
    # Magnitude changes only at second order.
    assert np.linalg.norm(f) - 9.81 == pytest.approx(0.05**2 / (2 * 9.81),
                                                     rel=1e-3)


def test_free_fall_reads_zero_and_upward_acceleration_adds():
    np.testing.assert_allclose(
        specific_force(imu_orientation(), acceleration_n=[0, 0, -9.81]),
        [0, 0, 0], atol=TOL)
    np.testing.assert_allclose(
        specific_force(imu_orientation(), acceleration_n=[0, 0, 1.0]),
        [0, 0, 10.81], atol=TOL)


def test_specific_force_time_series():
    accelerations = np.array([[0, 0, 0], [0.1, 0, 0], [0, -0.2, 0]])
    c_in = imu_orientation(15.0, 30.0)
    f = specific_force(c_in, accelerations)
    assert f.shape == (3, 3)
    for row, a in zip(f, accelerations):
        np.testing.assert_allclose(row, specific_force(c_in, a), atol=TOL)


def test_wrong_frame_transform_is_refused():
    with pytest.raises(ValueError, match="expected C_IN"):
        specific_force(C_CI)
    with pytest.raises(ValueError):
        true_up_imu(np.eye(2))


# ----------------------------------------------------- celestial vertical chain
def test_vertical_chain_matches_transform_composition_and_inverts():
    up_i = true_up_imu(imu_orientation(20.0, 50.0))
    up_g = vertical_in_celestial(up_i, C_CI, C_GC)
    np.testing.assert_allclose(up_g, (C_GC @ C_CI).apply(up_i), atol=TOL)
    np.testing.assert_allclose((C_GC @ C_CI).inverse().apply(up_g), up_i,
                               atol=TOL)
    assert np.linalg.norm(up_g) == pytest.approx(1.0)
    # Plain arrays are accepted too.
    np.testing.assert_allclose(
        vertical_in_celestial(up_i, C_CI.matrix, C_GC.matrix), up_g, atol=TOL)


def test_perturb_calibration_is_a_camera_frame_rotation_vector():
    delta = np.radians([0.1, -0.2, 0.05])
    used = perturb_calibration(C_CI, delta)
    assert used.name == "C_CI"
    np.testing.assert_allclose(used.matrix,
                               rotation_vector_to_matrix(delta) @ C_CI.matrix)
    back = perturb_calibration(used, -delta)
    np.testing.assert_allclose(back.matrix, C_CI.matrix, atol=TOL)
    # Module 3 convention: C_IC,true = C_IC,used R(delta).
    np.testing.assert_allclose(C_CI.matrix.T,
                               used.matrix.T @ rotation_vector_to_matrix(delta),
                               atol=TOL)


def test_misalignment_about_the_vertical_is_invisible():
    up_i = true_up_imu(imu_orientation(20.0, 50.0))
    up_c = C_CI.apply(up_i)
    truth = vertical_in_celestial(up_i, C_CI, C_GC)
    about = perturb_calibration(C_CI, np.radians(0.5) * up_c)
    assert angle(vertical_in_celestial(up_i, about, C_GC), truth) < 1e-15
    perpendicular = np.cross(up_c, [1.0, 0, 0])
    perpendicular /= np.linalg.norm(perpendicular)
    across = perturb_calibration(C_CI, np.radians(0.5) * perpendicular)
    assert np.degrees(angle(vertical_in_celestial(up_i, across, C_GC),
                            truth)) == pytest.approx(0.5, abs=1e-9)


# --------------------------------------------------------------------- Jacobians
def test_jacobians_match_finite_differences():
    c_in = imu_orientation(20.0, 50.0)
    up_i = true_up_imu(c_in)
    f = GRAVITY * up_i
    truth = vertical_in_celestial(up_i, C_CI, C_GC)
    J = vertical_jacobians(up_i, C_CI, C_GC)
    eps = 1e-7
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = eps
        # accelerometer error
        perturbed = (f + step) / np.linalg.norm(f + step)
        numeric = (vertical_in_celestial(perturbed, C_CI, C_GC) - truth) / eps
        np.testing.assert_allclose(J["accelerometer"][:, axis], numeric,
                                   atol=1e-7)
        # extrinsic calibration error: C_CI,used = R(d) C_CI,true
        numeric = (vertical_in_celestial(
            up_i, perturb_calibration(C_CI, step), C_GC) - truth) / eps
        np.testing.assert_allclose(J["calibration"][:, axis], numeric,
                                   atol=1e-6)
        # camera attitude error: C_CG,est = R(d) C_CG,true
        c_cg_est = rotation_vector_to_matrix(step) @ C_GC.matrix.T
        numeric = (vertical_in_celestial(up_i, C_CI, c_cg_est.T) - truth) / eps
        np.testing.assert_allclose(J["camera"][:, axis], numeric, atol=1e-6)


def test_jacobians_are_blind_along_the_vertical():
    up_i = true_up_imu(imu_orientation(20.0, 50.0))
    J = vertical_jacobians(up_i, C_CI, C_GC)
    up_c = C_CI.apply(up_i)
    np.testing.assert_allclose(J["accelerometer"] @ up_i, 0, atol=TOL)
    np.testing.assert_allclose(J["calibration"] @ up_c, 0, atol=TOL)
    np.testing.assert_allclose(J["camera"] @ up_c, 0, atol=TOL)


def test_covariance_sum_and_angular_rms():
    up_i = true_up_imu(imu_orientation(20.0, 50.0))
    J = vertical_jacobians(up_i, C_CI, C_GC)
    sigma_a, sigma_cal = 0.002, np.radians(0.05)
    total, parts = vertical_covariance(
        J, {"accelerometer": sigma_a**2 * np.eye(3),
            "calibration": sigma_cal**2 * np.eye(3)})
    # Isotropic errors: two tangent directions each get sigma^2.
    assert angular_rms(parts["accelerometer"]) == pytest.approx(
        np.sqrt(2) * sigma_a / GRAVITY)
    assert angular_rms(parts["calibration"]) == pytest.approx(
        np.sqrt(2) * sigma_cal)
    np.testing.assert_allclose(total, sum(parts.values()))
    assert angular_rms(total) == pytest.approx(
        np.hypot(angular_rms(parts["accelerometer"]),
                 angular_rms(parts["calibration"])))


# -------------------------------------------------------- accelerometer model
def test_default_model_matches_the_assignment_table():
    model = AccelerometerErrorModel()
    np.testing.assert_allclose(model.bias_vector, [0.01, -0.02, 0.005])
    np.testing.assert_allclose(np.diag(model.measurement_matrix), 1.001)
    off_diagonal = model.measurement_matrix[~np.eye(3, dtype=bool)]
    np.testing.assert_allclose(off_diagonal, 0.0005)
    assert model.noise_sigma == 0.005
    np.testing.assert_allclose(model.correction_matrix @ model.measurement_matrix,
                               np.eye(3), atol=1e-15)


def test_ideal_sensor_returns_the_true_specific_force():
    c_in = imu_orientation(30.0, 10.0)
    raw = simulate_stationary(c_in, IDEAL_SENSOR, 5, np.random.default_rng(0))
    assert raw.shape == (5, 3)
    np.testing.assert_allclose(raw, np.tile(specific_force(c_in), (5, 1)))


def test_simulated_measurements_have_the_modelled_mean_and_noise():
    model = AccelerometerErrorModel()
    c_in = imu_orientation(30.0, 10.0)
    raw = simulate_stationary(c_in, model, 200_000, np.random.default_rng(1))
    expected = model.measurement_matrix @ specific_force(c_in) + model.bias_vector
    np.testing.assert_allclose(raw.mean(axis=0), expected, atol=5e-5)
    np.testing.assert_allclose(raw.std(axis=0), 0.005, rtol=0.01)


def test_per_axis_scale_factors():
    model = AccelerometerErrorModel(bias=(0, 0, 0), cross_axis=0.0,
                                    scale_factor_error=(0.01, 0.0, -0.02),
                                    noise_sigma=0.0)
    np.testing.assert_allclose(model.measure([1.0, 1.0, 1.0], None),
                               [[1.01, 1.0, 0.98]])


def test_six_positions_cover_each_axis_up_and_down():
    poses = six_position_orientations()
    ups = np.array([true_up_imu(p) for p in poses])
    expected = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                         [0, 0, 1], [0, 0, -1]], dtype=float)
    np.testing.assert_allclose(ups, expected, atol=TOL)
    for pose in poses:
        assert np.linalg.det(pose.matrix) == pytest.approx(1.0)


def test_six_position_calibration_recovers_the_model_exactly_without_noise():
    model = AccelerometerErrorModel(noise_sigma=0.0)
    poses = six_position_orientations()
    truth = np.array([specific_force(p) for p in poses])
    raw = model.measure(truth, None)
    bias, correction = estimate_calibration(raw, truth)
    np.testing.assert_allclose(bias, model.bias_vector, atol=1e-13)
    np.testing.assert_allclose(correction, model.correction_matrix, atol=1e-13)


def test_calibration_needs_enough_non_coplanar_poses():
    poses = six_position_orientations()
    truth = np.array([specific_force(p) for p in poses])
    with pytest.raises(ValueError, match="at least four"):
        estimate_calibration(truth[:3], truth[:3])
    coplanar = truth[[0, 1, 2, 3]]            # x and y only: all in one plane
    with pytest.raises(ValueError, match="coplanar"):
        estimate_calibration(coplanar, coplanar)
    with pytest.raises(ValueError):
        estimate_calibration(truth, truth[:4])
