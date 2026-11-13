"""Tests for rotation_engine.py. Run with:  python -m pytest -v"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rotation_engine import (
    axis_angle_to_quaternion,
    check_rotation_matrix,
    is_rotation_matrix,
    matrix_to_quaternion,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_normalize,
    quaternion_to_matrix,
    rotate_vector,
    rotation_matrix_angle,
    rotation_vector_to_matrix,
    rotation_vector_to_quaternion,
    skew,
)

X, Y, Z = np.eye(3)
TOL = 1e-12


def random_quaternions(n, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def scipy_matrix(q):
    """SciPy reference. SciPy quaternions are scalar-LAST: [x, y, z, w]."""
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


# ------------------------------------------------------------- construction
def test_axis_angle_quaternion_components():
    q = axis_angle_to_quaternion(Z, np.pi / 2)
    np.testing.assert_allclose(q, [np.sqrt(0.5), 0, 0, np.sqrt(0.5)], atol=TOL)


def test_axis_angle_normalizes_axis_and_is_unit():
    a = axis_angle_to_quaternion([0, 0, 10.0], 0.7)
    b = axis_angle_to_quaternion([0, 0, 1.0], 0.7)
    np.testing.assert_allclose(a, b, atol=TOL)
    assert np.linalg.norm(a) == pytest.approx(1.0)


def test_zero_angle_is_identity_quaternion():
    np.testing.assert_allclose(axis_angle_to_quaternion([1, 2, 3], 0.0),
                               [1, 0, 0, 0])


def test_zero_axis_and_bad_shapes_are_rejected():
    with pytest.raises(ValueError):
        axis_angle_to_quaternion([0, 0, 0], 1.0)
    with pytest.raises(ValueError):
        quaternion_multiply([1, 0, 0], [1, 0, 0, 0])
    with pytest.raises(ValueError):
        rotate_vector([1, 0, 0, 0], [1, 0])
    with pytest.raises(ValueError):
        quaternion_normalize([0, 0, 0, 0])


# ---------------------------------------------------------------- algebra
def test_hamilton_basis_products():
    i, j, k = [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
    np.testing.assert_allclose(quaternion_multiply(i, j), k)       # ij = k
    np.testing.assert_allclose(quaternion_multiply(j, k), i)       # jk = i
    np.testing.assert_allclose(quaternion_multiply(k, i), j)       # ki = j
    np.testing.assert_allclose(quaternion_multiply(j, i), [0, 0, 0, -1])
    np.testing.assert_allclose(quaternion_multiply(i, i), [-1, 0, 0, 0])


def test_identity_and_conjugate_properties():
    identity = np.array([1.0, 0, 0, 0])
    for q in random_quaternions(20):
        np.testing.assert_allclose(quaternion_multiply(q, identity), q, atol=TOL)
        np.testing.assert_allclose(quaternion_multiply(identity, q), q, atol=TOL)
        np.testing.assert_allclose(
            quaternion_multiply(q, quaternion_conjugate(q)), identity, atol=TOL)


def test_product_is_associative_and_keeps_unit_norm():
    q, p, r = random_quaternions(3, seed=5)
    left = quaternion_multiply(quaternion_multiply(q, p), r)
    right = quaternion_multiply(q, quaternion_multiply(p, r))
    np.testing.assert_allclose(left, right, atol=TOL)
    assert np.linalg.norm(left) == pytest.approx(1.0)


def test_conjugate_of_product_reverses_order():
    q, p = random_quaternions(2, seed=6)
    np.testing.assert_allclose(
        quaternion_conjugate(quaternion_multiply(q, p)),
        quaternion_multiply(quaternion_conjugate(p), quaternion_conjugate(q)),
        atol=TOL)


# ------------------------------------------------- required validation tests
def test_matrix_is_orthogonal():                       # R(q)^T R(q) = I
    for q in random_quaternions(200, seed=1):
        R = quaternion_to_matrix(q)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-14)


def test_matrix_has_determinant_one():                 # det R(q) = 1
    for q in random_quaternions(200, seed=2):
        assert np.linalg.det(quaternion_to_matrix(q)) == pytest.approx(1.0,
                                                                       abs=1e-14)


def test_product_maps_to_matrix_product():             # R(q (x) p) = R(q) R(p)
    qs, ps = random_quaternions(100, seed=3), random_quaternions(100, seed=4)
    for q, p in zip(qs, ps):
        np.testing.assert_allclose(
            quaternion_to_matrix(quaternion_multiply(q, p)),
            quaternion_to_matrix(q) @ quaternion_to_matrix(p), atol=1e-14)


def test_q_and_minus_q_are_the_same_rotation():        # R(q) = R(-q)
    for q in random_quaternions(50, seed=7):
        np.testing.assert_array_equal(quaternion_to_matrix(q),
                                      quaternion_to_matrix(-q))
        np.testing.assert_allclose(rotate_vector(q, [1, 2, 3]),
                                   rotate_vector(-q, [1, 2, 3]), atol=TOL)


@pytest.mark.parametrize(
    "axis, start, expected",
    [
        (X, Y, Z), (X, Z, -Y), (X, X, X),      # about x: y -> z -> -y
        (Y, Z, X), (Y, X, -Z), (Y, Y, Y),      # about y: z -> x -> -z
        (Z, X, Y), (Z, Y, -X), (Z, Z, Z),      # about z: x -> y -> -x
    ],
)
def test_basis_vectors_rotated_90_degrees(axis, start, expected):
    q = axis_angle_to_quaternion(axis, np.pi / 2)
    np.testing.assert_allclose(rotate_vector(q, start), expected, atol=TOL)
    np.testing.assert_allclose(quaternion_to_matrix(q) @ start, expected,
                               atol=TOL)
    np.testing.assert_allclose(rotation_vector_to_matrix(axis * np.pi / 2)
                               @ start, expected, atol=TOL)


def test_round_trip_quaternion_matrix_quaternion():
    for q in random_quaternions(500, seed=8):
        R = quaternion_to_matrix(q)
        q_back = matrix_to_quaternion(R)
        # Compare the physical orientation, not components (q ~ -q).
        np.testing.assert_allclose(quaternion_to_matrix(q_back), R, atol=1e-14)
        assert abs(np.dot(q, q_back)) == pytest.approx(1.0, abs=1e-14)
        assert q_back[0] >= 0.0


@pytest.mark.parametrize("axis", [X, Y, Z, [1, 1, 0], [1, -2, 3], [-1, 1e-3, 0],
                                  [1e-9, 1, 1e-9]])
@pytest.mark.parametrize("delta", [0.0, 1e-15, 1e-12, 1e-9, 1e-6, 1e-3, -1e-9])
def test_round_trip_near_180_degrees(axis, delta):
    q = axis_angle_to_quaternion(axis, np.pi - delta)
    R = quaternion_to_matrix(q)
    q_back = matrix_to_quaternion(R)
    assert np.all(np.isfinite(q_back))
    assert np.linalg.norm(q_back) == pytest.approx(1.0, abs=1e-14)
    np.testing.assert_allclose(quaternion_to_matrix(q_back), R, atol=1e-14)
    # The recovered axis must match too, not just reproduce the matrix.
    assert abs(np.dot(q, q_back)) == pytest.approx(1.0, abs=1e-12)


def test_naive_formula_fails_where_shepperd_does_not():
    # Documents why matrix_to_quaternion is not the textbook one-liner.
    q = axis_angle_to_quaternion([1, -2, 3], np.pi - 1e-9)
    R = quaternion_to_matrix(q)
    w = 0.5 * np.sqrt(max(0.0, 1.0 + np.trace(R)))
    with np.errstate(divide="ignore", invalid="ignore"):
        naive = np.array([w, (R[2, 1] - R[1, 2]) / (4 * w),
                          (R[0, 2] - R[2, 0]) / (4 * w),
                          (R[1, 0] - R[0, 1]) / (4 * w)])
    naive_bad = (not np.all(np.isfinite(naive))
                 or abs(abs(np.dot(naive, q)) - 1.0) > 1e-6)
    assert naive_bad
    assert abs(np.dot(matrix_to_quaternion(R), q)) == pytest.approx(1.0,
                                                                   abs=1e-12)


def test_matrix_to_quaternion_every_branch():
    cases = {
        "w": axis_angle_to_quaternion([1, 2, 3], 0.2),
        "x": axis_angle_to_quaternion([1, 0.1, 0.1], 3.0),
        "y": axis_angle_to_quaternion([0.1, 1, 0.1], 3.0),
        "z": axis_angle_to_quaternion([0.1, 0.1, 1], 3.0),
    }
    for q in cases.values():
        np.testing.assert_allclose(
            quaternion_to_matrix(matrix_to_quaternion(quaternion_to_matrix(q))),
            quaternion_to_matrix(q), atol=1e-14)
    np.testing.assert_allclose(matrix_to_quaternion(np.eye(3)), [1, 0, 0, 0])


def test_matrix_to_quaternion_rejects_non_rotations():
    reflection = np.diag([1.0, 1.0, -1.0])
    assert not is_rotation_matrix(reflection)
    with pytest.raises(ValueError, match="proper"):
        matrix_to_quaternion(reflection)
    with pytest.raises(ValueError, match="orthogonal"):
        matrix_to_quaternion(np.eye(3) * 1.01)
    with pytest.raises(ValueError):
        check_rotation_matrix(np.eye(2))


# ----------------------------------------------------- vectors and rotvecs
def test_rotate_vector_matches_matrix_and_sandwich_product():
    rng = np.random.default_rng(9)
    for q in random_quaternions(50, seed=10):
        v = rng.normal(size=3)
        sandwich = quaternion_multiply(
            quaternion_multiply(q, np.concatenate(([0.0], v))),
            quaternion_conjugate(q))
        assert sandwich[0] == pytest.approx(0.0, abs=TOL)
        np.testing.assert_allclose(rotate_vector(q, v), sandwich[1:], atol=TOL)
        np.testing.assert_allclose(rotate_vector(q, v),
                                   quaternion_to_matrix(q) @ v, atol=TOL)


def test_rotation_preserves_length_and_axis():
    q = axis_angle_to_quaternion([1, 2, 3], 1.1)
    v = np.array([0.3, -4.0, 2.0])
    assert np.linalg.norm(rotate_vector(q, v)) == pytest.approx(np.linalg.norm(v))
    np.testing.assert_allclose(rotate_vector(q, [1, 2, 3]), [1, 2, 3], atol=TOL)


def test_conjugate_undoes_rotation():
    q = axis_angle_to_quaternion([1, 2, 3], 1.1)
    v = np.array([0.3, -4.0, 2.0])
    np.testing.assert_allclose(
        rotate_vector(quaternion_conjugate(q), rotate_vector(q, v)), v, atol=TOL)


def test_rotation_vector_matches_axis_angle_quaternion():
    rng = np.random.default_rng(11)
    for _ in range(100):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(0, np.pi)
        expected = quaternion_to_matrix(axis_angle_to_quaternion(axis, angle))
        np.testing.assert_allclose(rotation_vector_to_matrix(axis * angle),
                                   expected, atol=1e-14)
        np.testing.assert_allclose(
            quaternion_to_matrix(rotation_vector_to_quaternion(axis * angle)),
            expected, atol=1e-14)


@pytest.mark.parametrize("angle", [0.0, 1e-12, 1e-8, 9.9e-7, 1.1e-6, 1e-4])
def test_rotation_vector_small_angles(angle):
    rotvec = angle * np.array([1.0, -2.0, 2.0]) / 3.0
    R = rotation_vector_to_matrix(rotvec)
    assert is_rotation_matrix(R, tolerance=1e-15)
    # First order: R = I + skew(rotvec)
    np.testing.assert_allclose(R, np.eye(3) + skew(rotvec), atol=angle**2 + 1e-18)
    assert rotation_matrix_angle(R) == pytest.approx(angle, rel=1e-9, abs=1e-18)
    np.testing.assert_allclose(
        quaternion_to_matrix(rotation_vector_to_quaternion(rotvec)), R,
        atol=1e-15)


def test_rotation_matrix_angle():
    for angle in (0.0, 1e-10, 0.3, np.pi / 2, 3.0, np.pi):
        R = rotation_vector_to_matrix(angle * np.array([0.0, 0.6, 0.8]))
        assert rotation_matrix_angle(R) == pytest.approx(angle, abs=1e-12)


def test_skew_is_the_cross_product():
    a, b = np.array([1.0, -2.0, 0.5]), np.array([0.3, 4.0, -1.0])
    np.testing.assert_allclose(skew(a) @ b, np.cross(a, b))


# ------------------------------------------------- independent reference
def test_agrees_with_scipy_reference():
    rng = np.random.default_rng(12)
    qs, ps = random_quaternions(100, seed=13), random_quaternions(100, seed=14)
    for q, p in zip(qs, ps):
        np.testing.assert_allclose(quaternion_to_matrix(q), scipy_matrix(q),
                                   atol=1e-14)
        # SciPy composes the same way: (r_q * r_p) applies p first.
        composed = (Rotation.from_quat([q[1], q[2], q[3], q[0]])
                    * Rotation.from_quat([p[1], p[2], p[3], p[0]])).as_matrix()
        np.testing.assert_allclose(
            quaternion_to_matrix(quaternion_multiply(q, p)), composed,
            atol=1e-14)
        rotvec = rng.normal(size=3)
        np.testing.assert_allclose(rotation_vector_to_matrix(rotvec),
                                   Rotation.from_rotvec(rotvec).as_matrix(),
                                   atol=1e-14)
        x, y, z, w = Rotation.from_matrix(scipy_matrix(q)).as_quat()
        assert abs(np.dot(matrix_to_quaternion(scipy_matrix(q)),
                          [w, x, y, z])) == pytest.approx(1.0, abs=1e-14)
