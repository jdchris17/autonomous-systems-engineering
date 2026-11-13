"""Tests for stellar_attitude.py. Run with:  python -m pytest -v"""

import numpy as np
import pytest

from star_field_simulation import (
    PinholeCamera,
    centroid_sigma_to_angle,
    observe_with_angular_noise,
    observe_with_centroid_noise,
    stars_in_field,
    to_celestial,
    true_attitude_matrix,
)
from stellar_attitude import (
    InsufficientGeometryError,
    angular_residuals,
    attitude_covariance,
    attitude_error,
    attitude_information,
    ransac_attitude,
    residual_threshold_factor,
    robust_wahba,
    rotation_matrix_to_vector,
    triad,
    wahba_svd,
)

# star_field_simulation has already put module-3 on sys.path.
from rotation_engine import rotation_vector_to_matrix  # noqa: E402

ARCSEC = np.radians(1.0 / 3600.0)
C_TRUE = true_attitude_matrix()
CAMERA = PinholeCamera()


def star_field(n, seed=0):
    """(reference (N,3) celestial, measured (N,3) camera), noise-free."""
    cam = stars_in_field(CAMERA, n, np.random.default_rng(seed))
    return to_celestial(cam), cam


def is_proper_rotation(C):
    return (np.allclose(C.T @ C, np.eye(3), atol=1e-12)
            and np.linalg.det(C) == pytest.approx(1.0, abs=1e-12))


# ----------------------------------------------------------------------- TRIAD
def test_triad_recovers_known_attitude():
    ref, cam = star_field(2, seed=1)
    C = triad(ref, cam)
    assert is_proper_rotation(C)
    assert attitude_error(C, C_TRUE)[1] < 1e-13


def test_triad_accepts_unnormalized_vectors():
    ref, cam = star_field(2, seed=2)
    np.testing.assert_allclose(triad(3.0 * ref, 0.2 * cam), triad(ref, cam),
                               atol=1e-14)


def test_triad_reproduces_first_star_exactly_under_noise():
    ref, cam = star_field(2, seed=3)
    noisy = observe_with_centroid_noise(CAMERA, cam, 1.0,
                                        np.random.default_rng(0))
    residuals = angular_residuals(triad(ref, noisy), ref, noisy)
    assert residuals[0] < 1e-14          # anchor star: exact
    assert residuals[1] > 1e-8           # second star absorbs the mismatch


def test_triad_needs_exactly_two_nonparallel_stars():
    ref, cam = star_field(3, seed=4)
    with pytest.raises(ValueError, match="exactly 2"):
        triad(ref, cam)
    with pytest.raises(InsufficientGeometryError):
        triad([ref[0], ref[0]], [cam[0], cam[0]])
    with pytest.raises(InsufficientGeometryError):
        triad([ref[0], -ref[0]], [cam[0], -cam[0]])


# ------------------------------------------------------------------ Wahba / SVD
@pytest.mark.parametrize("n", [2, 3, 5, 20, 40])
def test_svd_recovers_known_attitude(n):
    ref, cam = star_field(n, seed=n)
    C = wahba_svd(ref, cam)
    assert is_proper_rotation(C)
    assert attitude_error(C, C_TRUE)[1] < 1e-12


def test_svd_works_with_two_stars_although_profile_matrix_is_rank_two():
    ref, cam = star_field(2, seed=7)
    profile = cam.T @ ref
    assert np.linalg.matrix_rank(profile) == 2
    assert attitude_error(wahba_svd(ref, cam), C_TRUE)[1] < 1e-12


def test_svd_and_triad_agree_for_two_exact_stars():
    ref, cam = star_field(2, seed=8)
    np.testing.assert_allclose(wahba_svd(ref, cam), triad(ref, cam), atol=1e-12)


def test_svd_recovers_arbitrary_attitudes_all_sky():
    rng = np.random.default_rng(5)
    for _ in range(50):
        C = rotation_vector_to_matrix(rng.normal(size=3) * 1.5)
        ref = rng.normal(size=(6, 3))
        ref /= np.linalg.norm(ref, axis=1, keepdims=True)
        assert attitude_error(wahba_svd(ref, ref @ C.T), C)[1] < 1e-12


def test_svd_returns_proper_rotation_even_for_wild_noise():
    rng = np.random.default_rng(6)
    for _ in range(200):
        ref = rng.normal(size=(3, 3))
        cam = rng.normal(size=(3, 3))          # unrelated: reflection-prone
        try:
            C = wahba_svd(ref, cam)
        except InsufficientGeometryError:
            continue
        assert is_proper_rotation(C)


def test_svd_minimizes_wahba_loss():
    ref, cam = star_field(10, seed=9)
    noisy = observe_with_centroid_noise(CAMERA, cam, 0.5,
                                        np.random.default_rng(1))
    weights = np.linspace(0.5, 2.0, 10)

    def loss(C):
        return 0.5 * np.sum(weights * np.sum((noisy - ref @ C.T) ** 2, axis=1))

    best = wahba_svd(ref, noisy, weights)
    rng = np.random.default_rng(2)
    for _ in range(200):
        nudge = rotation_vector_to_matrix(rng.normal(size=3) * 1e-4)
        assert loss(best) <= loss(nudge @ best)


def test_weights_pull_the_solution_toward_trusted_stars():
    ref, cam = star_field(6, seed=10)
    bad = cam.copy()
    bad[0] = observe_with_angular_noise(cam[:1], 200 * ARCSEC,
                                        np.random.default_rng(3))[0]
    equal = attitude_error(wahba_svd(ref, bad), C_TRUE)[1]
    weights = np.ones(6)
    weights[0] = 1e-6
    down = attitude_error(wahba_svd(ref, bad, weights), C_TRUE)[1]
    assert down < 0.01 * equal


def test_uniform_weight_scale_does_not_matter():
    ref, cam = star_field(5, seed=11)
    noisy = observe_with_centroid_noise(CAMERA, cam, 0.3,
                                        np.random.default_rng(4))
    np.testing.assert_allclose(wahba_svd(ref, noisy, np.full(5, 7.0)),
                               wahba_svd(ref, noisy), atol=1e-13)


def test_svd_rejects_insufficient_directional_information():
    ref, cam = star_field(3, seed=12)
    with pytest.raises(InsufficientGeometryError):
        wahba_svd(ref[:1], cam[:1])                     # one star
    with pytest.raises(InsufficientGeometryError):
        wahba_svd(np.tile(ref[0], (4, 1)), np.tile(cam[0], (4, 1)))   # same star
    with pytest.raises(InsufficientGeometryError):
        wahba_svd([ref[0], -ref[0]], [cam[0], -cam[0]])  # antiparallel


# ------------------------------------------------------------------ validation
@pytest.mark.parametrize("solver", [wahba_svd, lambda r, b: triad(r[:2], b[:2])])
def test_inputs_must_be_finite_and_nonzero(solver):
    ref, cam = star_field(3, seed=13)
    for bad_value in (np.nan, np.inf):
        broken = cam.copy()
        broken[1, 2] = bad_value
        with pytest.raises(ValueError, match="NaN or infinite"):
            solver(ref, broken)
    zero = ref.copy()
    zero[0] = 0.0
    with pytest.raises(ValueError, match="zero-length"):
        solver(zero, cam)


def test_shape_checks():
    ref, cam = star_field(3, seed=14)
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        wahba_svd(ref[:, :2], cam)
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        wahba_svd(ref[0], cam[0])
    with pytest.raises(ValueError, match="must match"):
        wahba_svd(ref, cam[:2])


@pytest.mark.parametrize("weights", [[1, 0, 1], [1, -2, 1], [1, np.nan, 1],
                                     [1, 1]])
def test_weights_must_be_positive_and_match(weights):
    ref, cam = star_field(3, seed=15)
    with pytest.raises(ValueError):
        wahba_svd(ref, cam, weights)


# --------------------------------------------------------------- attitude error
def test_attitude_error_returns_the_applied_rotation_vector():
    phi = np.array([3.0, -7.0, 40.0]) * ARCSEC
    C_est = rotation_vector_to_matrix(phi) @ C_TRUE
    vector, angle = attitude_error(C_est, C_TRUE)
    np.testing.assert_allclose(vector, phi, atol=1e-15)
    assert angle == pytest.approx(np.linalg.norm(phi))


def test_attitude_error_axes_are_camera_axes():
    # A pure roll about the boresight leaves the boresight star in place.
    roll = rotation_vector_to_matrix([0, 0, 1e-3]) @ C_TRUE
    vector, _ = attitude_error(roll, C_TRUE)
    np.testing.assert_allclose(vector, [0, 0, 1e-3], atol=1e-15)
    boresight_celestial = C_TRUE.T @ [0, 0, 1]
    np.testing.assert_allclose(roll @ boresight_celestial, [0, 0, 1], atol=1e-15)


def test_attitude_error_zero_large_and_antisymmetric():
    assert attitude_error(C_TRUE, C_TRUE)[1] == 0.0
    big = np.array([0.4, -1.0, 2.0])
    vector, angle = attitude_error(rotation_vector_to_matrix(big) @ C_TRUE, C_TRUE)
    np.testing.assert_allclose(vector, big, atol=1e-13)
    back, _ = attitude_error(C_TRUE, rotation_vector_to_matrix(big) @ C_TRUE)
    np.testing.assert_allclose(back, -big, atol=1e-13)


def test_attitude_error_rejects_non_rotations():
    with pytest.raises(ValueError):
        attitude_error(np.eye(3) * 2, C_TRUE)


def test_rotation_matrix_to_vector_resolves_tiny_angles():
    phi = np.array([1e-12, -2e-12, 5e-13])
    np.testing.assert_allclose(
        rotation_matrix_to_vector(rotation_vector_to_matrix(phi)), phi,
        atol=1e-17)


# ---------------------------------------------------------- information matrix
def test_information_of_a_single_star():
    J = attitude_information([[0, 0, 2.0]], 0.5)
    np.testing.assert_allclose(J, np.diag([4.0, 4.0, 0.0]))   # no roll info
    with pytest.raises(InsufficientGeometryError):
        attitude_covariance(J)


def test_information_matrix_is_symmetric_and_adds_up():
    _, cam = star_field(10, seed=16)
    sigma = np.linspace(1, 3, 10) * ARCSEC
    J = attitude_information(cam, sigma)
    np.testing.assert_allclose(J, J.T)
    assert np.all(np.linalg.eigvalsh(J) > 0)
    parts = sum(attitude_information(cam[i:i + 1], sigma[i]) for i in range(10))
    np.testing.assert_allclose(J, parts, rtol=1e-12)


def test_information_scales_with_noise_and_star_count():
    _, cam = star_field(8, seed=17)
    J = attitude_information(cam, 2 * ARCSEC)
    np.testing.assert_allclose(attitude_information(cam, 4 * ARCSEC), J / 4)
    np.testing.assert_allclose(
        attitude_information(np.vstack((cam, cam)), 2 * ARCSEC), 2 * J)


def test_roll_is_the_weakest_axis_for_a_narrow_field():
    _, cam = star_field(20, seed=18)
    sigma = np.sqrt(np.diag(attitude_covariance(
        attitude_information(cam, 5 * ARCSEC))))
    assert sigma[2] > 5 * sigma[0] and sigma[2] > 5 * sigma[1]


def test_information_validates_sigma():
    _, cam = star_field(3, seed=19)
    for bad in (0.0, -1.0, np.nan, [1.0, 2.0]):
        with pytest.raises(ValueError):
            attitude_information(cam, bad)


def test_predicted_covariance_matches_monte_carlo():
    ref, cam = star_field(12, seed=20)
    sigma = 5 * ARCSEC
    predicted = attitude_covariance(attitude_information(cam, sigma))
    rng = np.random.default_rng(21)
    errors = np.array([
        attitude_error(wahba_svd(ref, observe_with_angular_noise(cam, sigma, rng)),
                       C_TRUE)[0] for _ in range(4000)])
    assert np.all(np.abs(errors.mean(axis=0)) < 4 * np.sqrt(np.diag(predicted) / 4000))
    np.testing.assert_allclose(np.cov(errors.T), predicted,
                               rtol=0.12, atol=0.05 * predicted.max())


def test_centroid_noise_to_angle_conversion():
    sigma = centroid_sigma_to_angle(CAMERA, [[0, 0, 1.0]], 0.2)
    assert sigma[0] == pytest.approx(0.2 / CAMERA.fx)
    off_axis = centroid_sigma_to_angle(CAMERA, [[0.3, 0, 1.0]], 0.2)
    assert off_axis[0] < sigma[0]


# -------------------------------------------------------------------- residuals
def test_residuals_are_zero_for_the_true_attitude_and_equal_applied_offsets():
    ref, cam = star_field(5, seed=22)
    np.testing.assert_allclose(angular_residuals(C_TRUE, ref, cam), 0,
                               atol=1e-15)
    wrong = ref.copy()
    wrong[3] = ref[4]
    residuals = angular_residuals(C_TRUE, wrong, cam)
    expected = np.arccos(np.clip(cam[3] @ cam[4], -1, 1))
    assert residuals[3] == pytest.approx(expected, rel=1e-9)
    assert np.all(np.delete(residuals, 3) < 1e-15)


def test_threshold_factor_from_rayleigh_model():
    assert residual_threshold_factor(1e-3) == pytest.approx(3.7169, abs=1e-4)
    assert np.exp(-residual_threshold_factor(0.05) ** 2 / 2) == pytest.approx(0.05)
    with pytest.raises(ValueError):
        residual_threshold_factor(0.0)


# ------------------------------------------------------------ robust estimators
def noisy_field_with_mismatches(n, wrong, seed):
    rng = np.random.default_rng(seed)
    cam = stars_in_field(CAMERA, n, rng)
    ref = to_celestial(cam)
    measured = observe_with_centroid_noise(CAMERA, cam, 0.2, rng)
    sigma = centroid_sigma_to_angle(CAMERA, cam, 0.2)
    decoys = to_celestial(stars_in_field(CAMERA, 40, rng))
    bad = rng.choice(n, size=wrong, replace=False)
    corrupted = ref.copy()
    corrupted[bad] = decoys[:wrong]
    return corrupted, measured, sigma, np.sort(bad), ref


def test_iterative_rejection_removes_a_single_mismatch():
    ref, meas, sigma, bad, clean_ref = noisy_field_with_mismatches(20, 1, 30)
    plain = attitude_error(wahba_svd(ref, meas, 1 / sigma**2), C_TRUE)[1]
    result = robust_wahba(ref, meas, sigma)
    assert result.converged
    assert result.rejected == list(bad)
    assert not result.inliers[bad[0]] and result.inliers.sum() == 19
    keep = result.inliers
    oracle = wahba_svd(clean_ref[keep], meas[keep], 1 / sigma[keep] ** 2)
    np.testing.assert_allclose(result.C, oracle, atol=1e-13)
    assert attitude_error(result.C, C_TRUE)[1] < 0.01 * plain


def test_iterative_rejection_keeps_every_star_when_matches_are_correct():
    kept = []
    for seed in range(40):
        ref, meas, sigma, _, _ = noisy_field_with_mismatches(20, 0, 100 + seed)
        result = robust_wahba(ref, meas, sigma)
        assert result.converged
        kept.append(result.inliers.sum())
    assert np.mean(kept) > 19.8          # false alarms are rare by design


def test_iterative_rejection_reports_when_it_runs_out_of_stars():
    rng = np.random.default_rng(31)
    cam = stars_in_field(CAMERA, 5, rng)
    ref = to_celestial(stars_in_field(CAMERA, 5, rng))    # all matches wrong
    result = robust_wahba(ref, cam, 1 * ARCSEC)
    assert not result.converged
    assert "stars left" in result.status or "degenerate" in result.status


def test_ransac_recovers_attitude_with_half_the_matches_wrong():
    ref, meas, sigma, bad, _ = noisy_field_with_mismatches(20, 10, 32)
    result = ransac_attitude(ref, meas, sigma)
    assert result.converged
    assert not result.inliers[bad].any()
    assert result.inliers.sum() >= 9
    assert attitude_error(result.C, C_TRUE)[1] < 120 * ARCSEC
    # Deterministic when every pair is tried.
    again = ransac_attitude(ref, meas, sigma)
    np.testing.assert_array_equal(result.C, again.C)


def test_ransac_matches_plain_svd_when_all_matches_are_correct():
    ref, meas, sigma, _, _ = noisy_field_with_mismatches(20, 0, 33)
    result = ransac_attitude(ref, meas, sigma)
    if result.inliers.all():
        np.testing.assert_allclose(result.C, wahba_svd(ref, meas, 1 / sigma**2),
                                   atol=1e-13)
    assert result.inliers.sum() >= 19


def test_ransac_reports_failure_when_nothing_is_consistent():
    rng = np.random.default_rng(34)
    cam = stars_in_field(CAMERA, 12, rng)
    ref = to_celestial(stars_in_field(CAMERA, 12, rng))
    result = ransac_attitude(ref, cam, 1 * ARCSEC)
    assert not result.converged


def test_ransac_sampling_path_with_many_stars():
    ref, meas, sigma, bad, _ = noisy_field_with_mismatches(40, 8, 35)
    result = ransac_attitude(ref, meas, sigma, rng=np.random.default_rng(0),
                             max_hypotheses=200)
    assert result.converged and not result.inliers[bad].any()
    assert attitude_error(result.C, C_TRUE)[1] < 120 * ARCSEC
