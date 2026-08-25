import math
import random

import pytest

from matrix import Matrix
from least_squares import norm, vec_sub
from numerical import (
    condition_number,
    forward_substitution,
    hilbert_matrix,
    lu_decompose,
    solve_via_inverse,
    solve_via_lu,
    solve_via_qr,
    solve_via_svd,
)


def random_square_matrix(n, seed, low=-5, high=5):
    rng = random.Random(seed)
    return Matrix([[rng.uniform(low, high) for _ in range(n)] for _ in range(n)])


# ============================================================
# Nearly singular matrices
# ============================================================

class TestNearlySingularMatrices:
    def test_condition_number_is_huge(self):
        a = Matrix([[1, 1], [1, 1 + 1e-10]])
        assert condition_number(a) > 1e8

    def test_numpy_agrees_it_is_ill_conditioned(self):
        np = pytest.importorskip("numpy")
        a = Matrix([[1, 1], [1, 1 + 1e-10]])
        assert np.linalg.cond(np.array(a.rows)) > 1e8

    def test_small_perturbation_causes_large_solution_change(self):
        # classic near-singular example: tiny change in b, big change in x
        a = Matrix([[1, 1], [1, 1.0001]])
        b1 = [2, 2]
        b2 = [2, 2.0001]
        x1 = solve_via_lu(a, b1)
        x2 = solve_via_lu(a, b2)
        assert norm(vec_sub(x1, x2)) > 0.1  # disproportionate to the ~1e-4 nudge in b

    def test_lu_raises_on_an_exactly_singular_matrix(self):
        a = Matrix([[1, 2], [2, 4]])  # second row is a multiple of the first
        with pytest.raises(ValueError):
            lu_decompose(a)


# ============================================================
# Large matrices
# ============================================================

class TestLargeMatrices:
    def test_numpy_solve_is_accurate_at_size_200(self):
        np = pytest.importorskip("numpy")
        rng = np.random.default_rng(0)
        a = rng.uniform(-1, 1, size=(200, 200))
        x_true = rng.uniform(-1, 1, size=200)
        b = a @ x_true
        x = np.linalg.solve(a, b)
        assert np.linalg.norm(a @ x - b) < 1e-8

    def test_our_lu_solve_is_accurate_at_a_moderate_size(self):
        # our own pure-Python LU is O(n^3) but not vectorized, so this stays
        # modest; still large enough to exercise pivoting nontrivially
        n = 40
        a = random_square_matrix(n, seed=1)
        rng = random.Random(2)
        x_true = [rng.uniform(-3, 3) for _ in range(n)]
        b = a.multiply_vector(x_true)
        x = solve_via_lu(a, b)
        assert norm(vec_sub(x, x_true)) < 1e-6


# ============================================================
# Floating-point precision limits
# ============================================================

class TestFloatingPointPrecisionLimits:
    def test_naive_sum_is_not_associative(self):
        x = (0.1 + 0.2) + 0.3
        y = 0.1 + (0.2 + 0.3)
        assert x != y

    def test_zero_one_plus_zero_two_is_not_exactly_zero_three(self):
        assert 0.1 + 0.2 != 0.3
        assert math.isclose(0.1 + 0.2, 0.3)

    def test_adding_below_machine_epsilon_has_no_effect(self):
        import sys

        eps = sys.float_info.epsilon
        assert 1.0 + eps / 2 == 1.0
        assert 1.0 + eps != 1.0

    def test_catastrophic_cancellation_in_naive_quadratic_formula(self):
        a_coef, b_coef, c_coef = 1.0, -1e8, 1.0
        disc = b_coef**2 - 4 * a_coef * c_coef
        sqrt_disc = math.sqrt(disc)
        naive_small_root = (-b_coef - sqrt_disc) / (2 * a_coef)
        stable_small_root = c_coef / (a_coef * ((-b_coef + sqrt_disc) / (2 * a_coef)))
        true_small_root = 1e-8
        naive_error = abs(naive_small_root - true_small_root) / true_small_root
        stable_error = abs(stable_small_root - true_small_root) / true_small_root
        assert naive_error > 0.01  # significant precision loss
        assert stable_error < 1e-8  # essentially exact


# ============================================================
# Repeated matrix multiplication
# ============================================================

class TestRepeatedMatrixMultiplication:
    def test_naive_power_matches_matrix_pow_operator(self):
        a = Matrix([[1, 1], [0, 1]])
        naive = Matrix.identity(2)
        for _ in range(5):
            naive = naive * a
        assert naive == a**5

    def test_error_grows_with_iteration_count(self):
        # eigenvalues must NOT be < 1: if everything decays toward the zero
        # matrix, both the naive result and numpy's ground truth underflow
        # together and their absolute difference shrinks regardless of
        # accumulated rounding -- normalize so the dominant eigenvalue is
        # exactly 1 (same trick the numerical.py demo uses) to keep
        # magnitudes non-trivial across all k
        np = pytest.importorskip("numpy")
        from eigen import diagonalize_symmetric

        raw = Matrix([[4, 1, 0], [1, 3, 1], [0, 1, 2]])
        _, d = diagonalize_symmetric(raw, seed=0)
        a = raw * (1 / d.rows[0][0])

        errors = []
        for k in (10, 100, 1000, 10000):
            naive = Matrix.identity(3)
            for _ in range(k):
                naive = naive * a
            truth = np.linalg.matrix_power(np.array(a.rows), k)
            errors.append(float(np.linalg.norm(np.array(naive.rows) - truth)))
        # error should trend upward as more multiplications accumulate
        # rounding, not shrink by an order of magnitude
        assert errors[-1] > errors[0]

    def test_closed_form_power_stays_accurate_at_large_k(self):
        np = pytest.importorskip("numpy")
        from eigen import diagonalize_symmetric

        a = Matrix([[0.9, 0.05], [0.05, 0.9]])
        p, d = diagonalize_symmetric(a, seed=0)
        k = 5000
        d_k = Matrix([[d.rows[i][i] ** k if i == j else 0.0 for j in range(2)] for i in range(2)])
        closed_form = p * d_k * p.inverse()
        truth = np.linalg.matrix_power(np.array(a.rows), k)
        assert float(np.linalg.norm(np.array(closed_form.rows) - truth)) < 1e-6


# ============================================================
# Long iterative computations
# ============================================================

class TestLongIterativeComputations:
    def test_power_iteration_converges_within_a_large_iteration_budget(self):
        from eigen import power_iteration

        a = Matrix([[4, 1, 0], [1, 3, 1], [0, 1, 2]])
        result = power_iteration(a, num_iterations=100000, tol=1e-14, seed=3)
        assert result.converged

    def test_more_iterations_never_makes_the_estimate_worse(self):
        from eigen import power_iteration

        from numerical import symmetric_with_known_eigenvalues

        a, _, _ = symmetric_with_known_eigenvalues([10.0, 9.9, 1.0], seed=0)
        true_eigenvalue = 10.0
        errors = []
        for max_iter in (10, 100, 1000, 10000):
            result = power_iteration(a, num_iterations=max_iter, tol=0.0, seed=1)
            errors.append(abs(result.eigenvalue - true_eigenvalue))
        assert all(errors[i] >= errors[i + 1] - 1e-12 for i in range(len(errors) - 1))


# ============================================================
# General solver machinery: LU / forward-backward substitution / condition number
# ============================================================

class TestSolverMachinery:
    def test_lu_reconstructs_pa_equals_lu(self):
        a = random_square_matrix(4, seed=5)
        p, l, u = lu_decompose(a)
        reconstructed = p * a
        product = l * u
        for r1, r2 in zip(reconstructed.rows, product.rows):
            for v1, v2 in zip(r1, r2):
                assert math.isclose(v1, v2, abs_tol=1e-9)

    def test_all_four_solvers_agree_on_a_well_conditioned_system(self):
        a = Matrix([[4, 1, 0], [1, 3, 1], [0, 1, 2]])
        b = [1, 2, 3]
        results = [solve_via_inverse(a, b), solve_via_lu(a, b), solve_via_qr(a, b), solve_via_svd(a, b)]
        base = results[0]
        for other in results[1:]:
            assert norm(vec_sub(base, other)) < 1e-6

    def test_condition_number_of_identity_is_one(self):
        assert math.isclose(condition_number(Matrix.identity(4)), 1.0, abs_tol=1e-9)

    def test_hilbert_matrix_condition_number_grows_with_size(self):
        kappas = [condition_number(hilbert_matrix(n)) for n in (2, 3, 4)]
        assert kappas[0] < kappas[1] < kappas[2]

    def test_forward_substitution_solves_lower_triangular_system(self):
        l = Matrix([[2, 0], [1, 3]])
        x = forward_substitution(l, [4, 5])
        assert x == pytest.approx([2.0, 1.0])
