import math

import pytest

from matrix import Matrix
from least_squares import dot, gram_schmidt
from eigen import (
    diagonalize_symmetric,
    is_symmetric,
    matrices_close,
    orthogonal_columns,
    power_iteration,
    random_symmetric,
    verify_diagonalization,
)


def build_symmetric_with_known_eigenvalues(eigenvalues, seed=0):
    """Build A = P D P^T for a known diagonal D and a random orthonormal P
    (via Gram-Schmidt), giving a symmetric matrix whose exact eigenvalues
    are known in advance.
    """
    n = len(eigenvalues)
    import random

    rng = random.Random(seed)
    raw_vectors = [[rng.uniform(-3, 3) for _ in range(n)] for _ in range(n)]
    orthonormal_columns = gram_schmidt(raw_vectors)  # each is a row here
    p = Matrix([list(row) for row in zip(*orthonormal_columns)])  # columns = eigenvectors
    d = Matrix([[eigenvalues[i] if i == j else 0.0 for j in range(n)] for i in range(n)])
    a = p * d * p.transpose()  # P^-1 == P^T since P is orthonormal
    return a, p, d


# ============================================================
# Diagonal matrices
# ============================================================

class TestDiagonalMatrices:
    def test_diagonal_matrix_is_symmetric(self):
        a = Matrix([[3, 0, 0], [0, 5, 0], [0, 0, -1]])
        assert is_symmetric(a)

    def test_power_iteration_finds_largest_magnitude_diagonal_entry(self):
        a = Matrix([[2, 0, 0], [0, -7, 0], [0, 0, 3]])
        result = power_iteration(a, seed=0)
        assert math.isclose(result.eigenvalue, -7, abs_tol=1e-6)
        assert result.converged

    def test_diagonalize_recovers_the_diagonal_entries(self):
        a = Matrix([[2, 0, 0], [0, -7, 0], [0, 0, 3]])
        p, d = diagonalize_symmetric(a, seed=0)
        found = sorted(d.rows[i][i] for i in range(3))
        assert found == pytest.approx(sorted([2, -7, 3]), abs=1e-5)
        assert verify_diagonalization(a, p, d)


# ============================================================
# Symmetric matrices
# ============================================================

class TestSymmetricMatrices:
    def test_random_symmetric_is_actually_symmetric(self):
        for seed in range(5):
            a = random_symmetric(4, seed=seed)
            assert is_symmetric(a)

    def test_eigenvectors_of_symmetric_matrix_are_orthogonal(self):
        a = random_symmetric(4, seed=1)
        p, d = diagonalize_symmetric(a, seed=2)
        assert orthogonal_columns(p, tol=1e-4)

    def test_diagonalization_reconstructs_original_matrix(self):
        a = random_symmetric(4, seed=3)
        p, d = diagonalize_symmetric(a, seed=4)
        assert verify_diagonalization(a, p, d, tol=1e-4)

    def test_diagonalize_rejects_non_symmetric_matrix(self):
        a = Matrix([[1, 2], [3, 4]])
        with pytest.raises(ValueError):
            diagonalize_symmetric(a)


# ============================================================
# Identity matrix
# ============================================================

class TestIdentityMatrix:
    def test_power_iteration_eigenvalue_is_one(self):
        # every nonzero vector is an eigenvector of I, with eigenvalue 1
        result = power_iteration(Matrix.identity(4), seed=0)
        assert math.isclose(result.eigenvalue, 1.0, abs_tol=1e-9)
        assert result.converged

    def test_diagonalization_of_identity_is_identity(self):
        i4 = Matrix.identity(4)
        p, d = diagonalize_symmetric(i4, seed=0)
        assert matrices_close(d, i4, tol=1e-9)
        assert verify_diagonalization(i4, p, d)


# ============================================================
# Rotation matrices (observe behavior)
# ============================================================

class TestRotationMatrices:
    def test_genuine_rotation_does_not_converge(self):
        # eigenvalues of a 90-degree rotation are +-i: no real dominant
        # eigenvector exists, so power iteration should NOT converge
        rot90 = Matrix.rotation_2d(90, degrees=True)
        result = power_iteration(rot90, num_iterations=500, seed=0)
        assert result.converged is False

    def test_180_degree_rotation_is_negative_identity_and_converges(self):
        # a 180-degree rotation is -I, which *is* symmetric with a real
        # (repeated) eigenvalue -1, so this should converge immediately
        rot180 = Matrix.rotation_2d(180, degrees=True)
        result = power_iteration(rot180, seed=0)
        assert result.converged
        assert math.isclose(result.eigenvalue, -1.0, abs_tol=1e-6)

    def test_zero_degree_rotation_is_identity(self):
        rot0 = Matrix.rotation_2d(0, degrees=True)
        result = power_iteration(rot0, seed=0)
        assert math.isclose(result.eigenvalue, 1.0, abs_tol=1e-9)


# ============================================================
# Random matrices with known dominant eigenvalues
# ============================================================

class TestKnownDominantEigenvalues:
    def test_power_iteration_finds_the_known_dominant_eigenvalue(self):
        a, p, _ = build_symmetric_with_known_eigenvalues([5, 3, 1], seed=7)
        result = power_iteration(a, seed=0)
        assert math.isclose(result.eigenvalue, 5.0, abs_tol=1e-6)

        expected_vector = [row[0] for row in p.rows]  # first column of P
        same_sign = all(
            math.isclose(a1, a2, abs_tol=1e-4) for a1, a2 in zip(result.eigenvector, expected_vector)
        )
        opposite_sign = all(
            math.isclose(a1, -a2, abs_tol=1e-4) for a1, a2 in zip(result.eigenvector, expected_vector)
        )
        assert same_sign or opposite_sign

    def test_diagonalize_recovers_all_known_eigenvalues(self):
        a, _, _ = build_symmetric_with_known_eigenvalues([8, -2, 0.5], seed=8)
        p, d = diagonalize_symmetric(a, seed=1)
        found = sorted(d.rows[i][i] for i in range(3))
        assert found == pytest.approx(sorted([8, -2, 0.5]), abs=1e-4)

    def test_convergence_is_slower_when_top_two_eigenvalues_are_close(self):
        # eigenvalue gap of 5 vs 4.9 converges far more slowly than 5 vs 1
        close_gap, _, _ = build_symmetric_with_known_eigenvalues([5, 4.9, 1], seed=9)
        wide_gap, _, _ = build_symmetric_with_known_eigenvalues([5, 1, 0.5], seed=10)

        close_result = power_iteration(close_gap, num_iterations=5000, tol=1e-10, seed=0)
        wide_result = power_iteration(wide_gap, num_iterations=5000, tol=1e-10, seed=0)

        assert close_result.iterations > wide_result.iterations


# ============================================================
# Comparison against NumPy
# ============================================================

class TestComparisonAgainstNumpy:
    def test_dominant_eigenvalue_matches_numpy(self):
        np = pytest.importorskip("numpy")
        a = random_symmetric(5, seed=11)
        result = power_iteration(a, seed=0)

        np_vals, _ = np.linalg.eig(np.array(a.rows, dtype=float))
        expected = max(np_vals.real, key=abs)
        assert math.isclose(result.eigenvalue, expected, abs_tol=1e-5)

    def test_dominant_eigenvector_matches_numpy_up_to_sign(self):
        np = pytest.importorskip("numpy")
        a = random_symmetric(5, seed=12)
        result = power_iteration(a, seed=0)

        np_vals, np_vecs = np.linalg.eig(np.array(a.rows, dtype=float))
        idx = int(np.argmax(np.abs(np_vals.real)))
        expected_vector = np_vecs[:, idx].real
        if np.dot(expected_vector, result.eigenvector) < 0:
            expected_vector = -expected_vector

        for ours, theirs in zip(result.eigenvector, expected_vector.tolist()):
            assert math.isclose(ours, theirs, abs_tol=1e-4)

    def test_full_eigenvalue_set_matches_numpy(self):
        np = pytest.importorskip("numpy")
        a = random_symmetric(4, seed=13)
        _, d = diagonalize_symmetric(a, seed=1)
        ours = sorted(d.rows[i][i] for i in range(4))

        np_vals, _ = np.linalg.eig(np.array(a.rows, dtype=float))
        theirs = sorted(v.real for v in np_vals)

        for ours_val, theirs_val in zip(ours, theirs):
            assert math.isclose(ours_val, theirs_val, abs_tol=1e-4)
