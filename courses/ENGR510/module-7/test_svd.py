import math
import random

import pytest

from matrix import Matrix
from svd import (
    compress_matrix,
    compression_ratio,
    effective_rank,
    frobenius_norm,
    reconstruct,
    singular_values_of,
    svd,
)


def random_matrix(rows, cols, seed):
    rng = random.Random(seed)
    return Matrix([[rng.uniform(-5, 5) for _ in range(cols)] for _ in range(rows)])


# ============================================================
# Reconstruction accuracy: A ~= U Sigma V^T
# ============================================================

class TestReconstructionAccuracy:
    def test_square_matrix_reconstructs(self):
        a = Matrix([[4, 0], [3, -5]])
        u, sigma, vt = svd(a)
        error = frobenius_norm(a - reconstruct(u, sigma, vt))
        assert error < 1e-6

    def test_tall_matrix_reconstructs(self):
        a = Matrix([[3, 1, 1], [1, 3, 1], [1, 1, 3], [2, 0, 1]])  # 4x3
        u, sigma, vt = svd(a)
        error = frobenius_norm(a - reconstruct(u, sigma, vt))
        assert error < 1e-6

    def test_wide_matrix_reconstructs(self):
        a = Matrix([[3, 1, 1, 2], [1, 3, 1, 0]])  # 2x4
        u, sigma, vt = svd(a)
        error = frobenius_norm(a - reconstruct(u, sigma, vt))
        assert error < 1e-6

    def test_random_matrices_reconstruct(self):
        for seed in range(5):
            a = random_matrix(5, 4, seed)
            u, sigma, vt = svd(a)
            assert frobenius_norm(a - reconstruct(u, sigma, vt)) < 1e-5

    def test_singular_values_are_non_negative_and_descending(self):
        a = random_matrix(5, 5, seed=7)
        _, sigma, _ = svd(a)
        values = singular_values_of(sigma)
        assert all(v >= 0 for v in values)
        assert values == sorted(values, reverse=True)


# ============================================================
# Rank approximation
# ============================================================

class TestRankApproximation:
    def test_full_rank_compression_reconstructs_exactly(self):
        a = Matrix([[3, 1, 1], [1, 3, 1], [1, 1, 3], [2, 0, 1]])
        full = compress_matrix(a, min(a.shape))
        assert frobenius_norm(a - full) < 1e-6

    def test_error_decreases_monotonically_with_k(self):
        a = random_matrix(6, 5, seed=3)
        errors = [frobenius_norm(a - compress_matrix(a, k)) for k in range(1, 6)]
        assert all(errors[i] >= errors[i + 1] - 1e-9 for i in range(len(errors) - 1))

    def test_rank_1_approximation_matches_top_singular_triplet(self):
        a = random_matrix(4, 4, seed=5)
        u, sigma, vt = svd(a)
        u1, s1, v1t = u.rows[0][0], sigma.rows[0][0], vt.rows[0]
        rank1 = compress_matrix(a, 1)
        # rank-1 approx should equal sigma1 * u1 (outer) v1^T
        expected_row0 = [s1 * u.rows[0][0] * v for v in v1t]
        assert all(math.isclose(a1, a2, abs_tol=1e-6) for a1, a2 in zip(rank1.rows[0], expected_row0))

    def test_out_of_range_k_raises(self):
        a = random_matrix(4, 3, seed=1)
        with pytest.raises(ValueError):
            compress_matrix(a, 0)
        with pytest.raises(ValueError):
            compress_matrix(a, 4)  # only 3 singular values exist for a 4x3 matrix

    def test_effective_rank_of_low_rank_matrix(self):
        # exactly rank 2 by construction (sum of 2 outer products)
        u1, u2 = [1, 2, 1, 0], [0, 1, -1, 2]
        v1, v2 = [2, 1, 0], [1, -1, 2]
        a = Matrix([[u1[i] * v1[j] + u2[i] * v2[j] for j in range(3)] for i in range(4)])
        _, sigma, _ = svd(a)
        assert effective_rank(sigma) == 2


# ============================================================
# Compression ratios
# ============================================================

class TestCompressionRatios:
    def test_ratio_formula(self):
        # 100x100 matrix, k=10 -> original 10000, compressed 10*(100+100+1)=2010
        ratio = compression_ratio(100, 100, 10)
        assert math.isclose(ratio, 10000 / 2010, rel_tol=1e-9)

    def test_ratio_decreases_as_k_increases(self):
        ratios = [compression_ratio(200, 150, k) for k in (1, 5, 10, 20, 50)]
        assert all(ratios[i] > ratios[i + 1] for i in range(len(ratios) - 1))

    def test_full_rank_ratio_can_be_below_one(self):
        # storing every singular triplet can cost MORE than the original
        # matrix once k approaches min(m, n) -- compression isn't free
        m, n = 20, 20
        full_k = min(m, n)
        ratio = compression_ratio(m, n, full_k)
        assert ratio < 1.0

    def test_small_k_gives_large_ratio(self):
        assert compression_ratio(500, 500, 1) > 100


# ============================================================
# Comparison with NumPy
# ============================================================

class TestComparisonWithNumpy:
    def test_singular_values_match_numpy(self):
        np = pytest.importorskip("numpy")
        a = random_matrix(5, 4, seed=11)
        _, sigma, _ = svd(a)
        ours = singular_values_of(sigma)

        theirs = np.linalg.svd(np.array(a.rows, dtype=float), compute_uv=False).tolist()
        for o, t in zip(ours, theirs):
            assert math.isclose(o, t, abs_tol=1e-5)

    def test_reconstruction_matches_numpy_reconstruction(self):
        np = pytest.importorskip("numpy")
        a = random_matrix(4, 4, seed=12)
        u, sigma, vt = svd(a)
        ours = reconstruct(u, sigma, vt)

        a_np = np.array(a.rows, dtype=float)
        np_u, np_s, np_vt = np.linalg.svd(a_np, full_matrices=False)
        theirs = (np_u * np_s) @ np_vt

        for r1, r2 in zip(ours.rows, theirs.tolist()):
            for v1, v2 in zip(r1, r2):
                assert math.isclose(v1, v2, abs_tol=1e-5)

    def test_low_rank_approximation_error_matches_numpy_eckart_young(self):
        # Eckart-Young theorem: the best rank-k approximation error in
        # Frobenius norm equals sqrt(sum of squares of the DROPPED
        # singular values) -- verify our compress_matrix hits that bound.
        np = pytest.importorskip("numpy")
        a = random_matrix(6, 5, seed=13)
        k = 2
        approx = compress_matrix(a, k)
        our_error = frobenius_norm(a - approx)

        theirs = np.linalg.svd(np.array(a.rows, dtype=float), compute_uv=False)
        expected_error = math.sqrt(sum(v**2 for v in theirs[k:]))
        assert math.isclose(our_error, expected_error, abs_tol=1e-4)

    def test_rectangular_matrix_matches_numpy(self):
        np = pytest.importorskip("numpy")
        a = random_matrix(3, 6, seed=14)  # wide
        _, sigma, _ = svd(a)
        ours = singular_values_of(sigma)
        theirs = np.linalg.svd(np.array(a.rows, dtype=float), compute_uv=False).tolist()
        for o, t in zip(ours, theirs):
            assert math.isclose(o, t, abs_tol=1e-5)
