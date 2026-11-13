"""Tests for gravity_estimator.py. Run with:  python -m pytest -v"""

import numpy as np
import pytest

from accelerometer_calibration import AccelerometerErrorModel, simulate_stationary
from gravity_estimator import (
    QUALITY_CAVEAT,
    assess_measurement_quality,
    calibrate_accelerometer,
    estimate_up_direction,
    gravity_to_camera,
    sliding_up_direction,
    tangent_basis,
    vertical_angular_error,
)
from imu_alignment import (
    GRAVITY,
    Transform,
    imu_orientation,
    rotation_vector_to_matrix,
    specific_force,
    true_up_imu,
)

ARCSEC = np.radians(1 / 3600)
C_IN = imu_orientation(20.0, 50.0, 15.0)
UP = true_up_imu(C_IN)
SIGMA = 0.005


def noisy_stationary(n, seed, sigma=SIGMA, extra=(0.0, 0.0, 0.0)):
    rng = np.random.default_rng(seed)
    return specific_force(C_IN) + np.asarray(extra) + rng.normal(0, sigma, (n, 3))


# ------------------------------------------------------------------ calibration
def test_calibration_inverts_the_error_model():
    model = AccelerometerErrorModel(noise_sigma=0.0)
    raw = simulate_stationary(C_IN, model, 4, np.random.default_rng(0))
    calibrated = calibrate_accelerometer(raw, model.bias_vector,
                                         model.correction_matrix)
    np.testing.assert_allclose(calibrated, np.tile(specific_force(C_IN), (4, 1)),
                               atol=1e-13)


def test_calibration_removes_bias_before_applying_the_matrix():
    out = calibrate_accelerometer([[3.0, 5.0, 7.0]], [1.0, 1.0, 1.0],
                                  np.diag([2.0, 1.0, 0.5]))
    np.testing.assert_allclose(out, [[4.0, 4.0, 3.0]])


def test_identity_calibration_changes_nothing():
    raw = noisy_stationary(10, 1)
    np.testing.assert_array_equal(
        calibrate_accelerometer(raw, np.zeros(3), np.eye(3)), raw)


@pytest.mark.parametrize("raw, bias, matrix", [
    (np.zeros((4, 2)), np.zeros(3), np.eye(3)),
    (np.zeros(3), np.zeros(3), np.eye(3)),
    (np.zeros((4, 3)), np.zeros(2), np.eye(3)),
    (np.zeros((4, 3)), np.zeros(3), np.eye(2)),
    (np.full((4, 3), np.nan), np.zeros(3), np.eye(3)),
])
def test_calibration_input_validation(raw, bias, matrix):
    with pytest.raises(ValueError):
        calibrate_accelerometer(raw, bias, matrix)


# ------------------------------------------------------------------ up direction
def test_noise_free_samples_give_the_exact_vertical():
    samples = np.tile(specific_force(C_IN), (10, 1))
    up, uncertainty, stats = estimate_up_direction(samples)
    np.testing.assert_allclose(up, UP, atol=1e-15)
    assert uncertainty.angular_rms == pytest.approx(0.0, abs=1e-15)
    assert stats.count == 10
    assert stats.magnitude == pytest.approx(GRAVITY)


def test_estimate_is_the_normalized_mean_not_the_mean_of_unit_vectors():
    samples = np.array([[0.0, 0.0, 1.0], [10.0, 0.0, 0.0]])
    up, _, stats = estimate_up_direction(samples)
    np.testing.assert_allclose(stats.mean, [5.0, 0.0, 0.5])
    np.testing.assert_allclose(up, np.array([5.0, 0.0, 0.5]) / np.hypot(5, 0.5))


def test_statistics_match_numpy():
    samples = noisy_stationary(500, 2)
    _, _, stats = estimate_up_direction(samples)
    np.testing.assert_allclose(stats.mean, samples.mean(axis=0))
    np.testing.assert_allclose(stats.covariance, np.cov(samples.T))
    np.testing.assert_allclose(stats.mean_covariance, np.cov(samples.T) / 500)
    np.testing.assert_allclose(stats.std_per_axis, samples.std(axis=0, ddof=1))


def test_uncertainty_is_rank_two_with_no_radial_component():
    up, uncertainty, _ = estimate_up_direction(noisy_stationary(1000, 3))
    P = uncertainty.covariance
    np.testing.assert_allclose(P @ up, 0, atol=1e-18)           # radial null
    assert np.linalg.matrix_rank(P, tol=1e-20) == 2
    np.testing.assert_allclose(P, P.T, atol=1e-22)
    basis = uncertainty.tangent_basis
    np.testing.assert_allclose(basis @ up, 0, atol=1e-14)
    np.testing.assert_allclose(basis @ basis.T, np.eye(2), atol=1e-14)
    assert np.trace(uncertainty.tangent_covariance) == pytest.approx(np.trace(P))
    assert uncertainty.sigma_major >= uncertainty.sigma_minor > 0
    assert uncertainty.angular_rms == pytest.approx(
        np.hypot(uncertainty.sigma_major, uncertainty.sigma_minor))


def test_predicted_uncertainty_has_the_expected_size():
    # Isotropic noise: each tangent direction gets sigma / (g sqrt(N)).
    _, uncertainty, _ = estimate_up_direction(noisy_stationary(20000, 4))
    expected = SIGMA / (GRAVITY * np.sqrt(20000))
    assert uncertainty.sigma_major == pytest.approx(expected, rel=0.03)
    assert uncertainty.sigma_minor == pytest.approx(expected, rel=0.03)
    assert uncertainty.angular_rms == pytest.approx(np.sqrt(2) * expected,
                                                    rel=0.03)


def test_predicted_uncertainty_matches_monte_carlo_scatter():
    errors, predicted = [], []
    for seed in range(1500):
        up, uncertainty, _ = estimate_up_direction(noisy_stationary(200, seed))
        errors.append(vertical_angular_error(up, UP))
        predicted.append(uncertainty.angular_rms)
    rms = np.sqrt(np.mean(np.square(errors)))
    assert rms == pytest.approx(np.sqrt(np.mean(np.square(predicted))), rel=0.06)


def test_uncertainty_shrinks_as_one_over_root_n():
    _, small, _ = estimate_up_direction(noisy_stationary(100, 5))
    _, large, _ = estimate_up_direction(noisy_stationary(10000, 5))
    assert small.angular_rms / large.angular_rms == pytest.approx(10.0, rel=0.2)


def test_bias_shifts_the_estimate_but_not_the_reported_uncertainty():
    horizontal = 0.01 * C_IN.apply([1.0, 0.0, 0.0])           # 0.01 m/s^2 east
    clean_up, clean_unc, _ = estimate_up_direction(noisy_stationary(5000, 6))
    bias_up, bias_unc, _ = estimate_up_direction(
        noisy_stationary(5000, 6, extra=horizontal))
    assert vertical_angular_error(bias_up, UP) == pytest.approx(
        np.arctan(0.01 / GRAVITY), rel=0.02)
    assert vertical_angular_error(bias_up, UP) > 20 * bias_unc.angular_rms
    assert bias_unc.angular_rms == pytest.approx(clean_unc.angular_rms, rel=1e-3)
    assert vertical_angular_error(clean_up, UP) < 5 * clean_unc.angular_rms


def test_up_direction_input_validation():
    with pytest.raises(ValueError, match="two samples"):
        estimate_up_direction([[0, 0, 9.81]])
    with pytest.raises(ValueError, match="zero"):
        estimate_up_direction([[1.0, 0, 0], [-1.0, 0, 0]])
    with pytest.raises(ValueError):
        estimate_up_direction(np.zeros((5, 2)))


def test_tangent_basis_for_axis_aligned_vectors():
    for u in np.vstack((np.eye(3), -np.eye(3))):
        basis = tangent_basis(u)
        np.testing.assert_allclose(basis @ u, 0, atol=1e-15)
        np.testing.assert_allclose(basis @ basis.T, np.eye(2), atol=1e-15)


# ------------------------------------------------------- camera frame and errors
def test_gravity_to_camera_applies_c_ci():
    c_ci = Transform("C", "I", rotation_vector_to_matrix([0, 0, np.pi / 2]))
    # Columns of C_CI are the IMU axes in camera coordinates: the IMU frame
    # is the camera frame turned +90 deg about z, so IMU x lies along
    # camera y and IMU y along camera -x.
    np.testing.assert_allclose(gravity_to_camera([1.0, 0, 0], c_ci), [0, 1, 0],
                               atol=1e-15)
    np.testing.assert_allclose(gravity_to_camera([0, 1.0, 0], c_ci), [-1, 0, 0],
                               atol=1e-15)
    np.testing.assert_allclose(gravity_to_camera(UP, c_ci.matrix),
                               c_ci.apply(UP))
    assert np.linalg.norm(gravity_to_camera(UP, c_ci)) == pytest.approx(1.0)


def test_gravity_to_camera_checks_frames_and_shapes():
    with pytest.raises(ValueError, match="expected C_CI"):
        gravity_to_camera(UP, Transform("I", "C", np.eye(3)))
    with pytest.raises(ValueError):
        gravity_to_camera([1.0, 0.0], np.eye(3))


def test_vertical_angular_error():
    assert vertical_angular_error([0, 0, 1], [0, 0, 1]) == 0.0
    assert vertical_angular_error([0, 0, 2], [5, 0, 0]) == pytest.approx(np.pi / 2)
    assert vertical_angular_error([0, 0, 1], [0, 0, -1]) == pytest.approx(np.pi)
    tiny = 1e-10
    assert vertical_angular_error([np.sin(tiny), 0, np.cos(tiny)],
                                  [0, 0, 1]) == pytest.approx(tiny, rel=1e-9)
    a, b = np.array([0.1, 0.2, 0.9]), np.array([0.3, -0.1, 0.8])
    assert vertical_angular_error(a, b) == pytest.approx(
        vertical_angular_error(b, a))


# ------------------------------------------------------------- sliding windows
def test_sliding_windows_match_the_single_window_estimator():
    samples = noisy_stationary(400, 7)
    samples[150:200] += [0.3, -0.1, 0.05]            # a disturbance mid-record
    window = 60
    up, rms, mean = sliding_up_direction(samples, window)
    assert up.shape == (400 - window + 1, 3)
    for k in (0, 37, 140, 190, 340):
        ref_up, ref_unc, ref_stats = estimate_up_direction(samples[k:k + window])
        np.testing.assert_allclose(up[k], ref_up, atol=1e-13)
        np.testing.assert_allclose(mean[k], ref_stats.mean, atol=1e-12)
        assert rms[k] == pytest.approx(ref_unc.angular_rms, rel=1e-6)


def test_sliding_window_validation():
    samples = noisy_stationary(20, 8)
    for bad in (1, 21):
        with pytest.raises(ValueError):
            sliding_up_direction(samples, bad)
    up, _, _ = sliding_up_direction(samples, 20)
    assert up.shape == (1, 3)


# ---------------------------------------------------------- quality assessment
def test_clean_stationary_data_is_steady_and_carries_the_caveat():
    verdicts = []
    for seed in range(200):
        previous = noisy_stationary(100, 1000 + seed).mean(axis=0)
        report = assess_measurement_quality(noisy_stationary(100, seed), SIGMA,
                                            previous_mean=previous)
        verdicts.append(report.verdict)
        assert report.caveat == QUALITY_CAVEAT
    assert verdicts.count("steady") >= 197            # false alarms are rare


def test_vibration_fails_the_variance_check():
    samples = noisy_stationary(100, 9)
    samples[:, 0] += 0.2 * np.sin(2 * np.pi * np.arange(100) / 100)
    report = assess_measurement_quality(samples, SIGMA)
    assert report.verdict == "disturbed" and report.detectable_disturbance
    assert report.checks["variance"] is False


def test_vertical_acceleration_fails_the_magnitude_check():
    lift = specific_force(C_IN, acceleration_n=[0, 0, 0.3]) - specific_force(C_IN)
    report = assess_measurement_quality(noisy_stationary(100, 10, extra=lift),
                                        SIGMA)
    assert report.checks["magnitude"] is False
    assert report.verdict == "disturbed"


def test_constant_horizontal_acceleration_is_not_detected():
    # The fundamental ambiguity: the vertical is wrong by 0.29 deg and every
    # IMU-only check still passes.
    push = specific_force(C_IN, acceleration_n=[0.05, 0, 0]) - specific_force(C_IN)
    samples = noisy_stationary(1000, 11, extra=push)
    previous = noisy_stationary(1000, 12, extra=push).mean(axis=0)
    report = assess_measurement_quality(samples, SIGMA, previous_mean=previous,
                                        gyro_samples=np.zeros((1000, 3)),
                                        gyro_noise_sigma=5e-4)
    assert report.verdict == "steady"
    assert all(report.checks.values())
    up, uncertainty, _ = estimate_up_direction(samples)
    error = vertical_angular_error(up, UP)
    assert np.degrees(error) == pytest.approx(0.292, abs=0.005)
    assert error > 100 * uncertainty.angular_rms
    assert "cannot prove" in report.caveat


def test_changing_mean_is_ambiguous_without_a_gyro_and_resolved_with_one():
    drift = specific_force(C_IN, acceleration_n=[0.01, 0, 0]) - specific_force(C_IN)
    samples = noisy_stationary(100, 13, extra=drift)
    previous = noisy_stationary(100, 14).mean(axis=0)
    quiet_gyro = np.random.default_rng(0).normal(0, 5e-4, (100, 3))
    turning_gyro = quiet_gyro + [0.002, 0, 0]

    no_gyro = assess_measurement_quality(samples, SIGMA, previous_mean=previous)
    assert no_gyro.verdict == "changing"
    assert no_gyro.checks["rotation"] is None

    still = assess_measurement_quality(samples, SIGMA, previous_mean=previous,
                                       gyro_samples=quiet_gyro,
                                       gyro_noise_sigma=5e-4)
    assert still.verdict == "accelerating"

    turning = assess_measurement_quality(samples, SIGMA, previous_mean=previous,
                                         gyro_samples=turning_gyro,
                                         gyro_noise_sigma=5e-4)
    assert turning.verdict == "rotating"


def test_quality_needs_at_least_two_samples():
    with pytest.raises(ValueError):
        assess_measurement_quality([[0, 0, 9.81]], SIGMA)
