import math

import pytest

from matrix import Matrix
from least_squares import least_squares as closed_form_least_squares, norm, vec_sub
from numerical import condition_number, hilbert_matrix
from optimization import (
    gradient_descent,
    make_least_squares_objective,
    make_quadratic,
    numerical_gradient,
    optimal_step_size,
)


# ============================================================
# Convergence speed
# ============================================================

class TestConvergenceSpeed:
    def test_well_conditioned_converges_quickly(self):
        a = Matrix([[3, 0], [0, 1]])  # kappa = 3
        b = [1, 1]
        f, grad = make_quadratic(a, b)
        lr = optimal_step_size(a)
        _, history = gradient_descent(f, [5.0, -4.0], lr, grad_f=grad, num_iterations=1000)
        assert history["converged"]
        assert len(history["f"]) - 1 < 100

    def test_iterations_grow_with_condition_number(self):
        iteration_counts = []
        for eigenvalues in ([1, 1], [1, 10], [1, 100]):
            a = Matrix([[eigenvalues[0], 0], [0, eigenvalues[1]]])
            b = [1, 1]
            f, grad = make_quadratic(a, b)
            lr = optimal_step_size(a)
            _, history = gradient_descent(f, [5.0, -4.0], lr, grad_f=grad, num_iterations=50000, tol=1e-8)
            assert history["converged"]
            iteration_counts.append(len(history["f"]) - 1)
        assert iteration_counts[0] < iteration_counts[1] < iteration_counts[2]

    def test_loss_decreases_monotonically_at_a_safe_step_size(self):
        a = Matrix([[4, 1], [1, 3]])
        b = [1, 2]
        f, grad = make_quadratic(a, b)
        lr = optimal_step_size(a) * 0.5  # conservative, well under the optimal step
        _, history = gradient_descent(f, [10.0, -10.0], lr, grad_f=grad, num_iterations=500)
        losses = history["f"]
        assert all(losses[i] >= losses[i + 1] - 1e-9 for i in range(len(losses) - 1))


# ============================================================
# Different starting points
# ============================================================

class TestDifferentStartingPoints:
    def test_all_starts_reach_the_same_minimum(self):
        a = Matrix([[3, 0], [0, 1]])
        b = [1, 1]
        f, grad = make_quadratic(a, b)
        lr = optimal_step_size(a)
        x_star = a.inverse().multiply_vector(b)

        for x0 in ([0, 0], [10, 10], [-8, 6], [100, -50]):
            x_final, history = gradient_descent(f, x0, lr, grad_f=grad, num_iterations=5000)
            assert history["converged"]
            assert norm(vec_sub(x_final, x_star)) < 1e-4

    def test_farther_start_needs_at_least_as_many_iterations(self):
        a = Matrix([[3, 0], [0, 1]])
        b = [1, 1]
        f, grad = make_quadratic(a, b)
        lr = optimal_step_size(a)
        x_star = a.inverse().multiply_vector(b)

        _, near = gradient_descent(f, [x_star[0] + 0.1, x_star[1] + 0.1], lr, grad_f=grad, num_iterations=5000)
        _, far = gradient_descent(f, [x_star[0] + 100, x_star[1] + 100], lr, grad_f=grad, num_iterations=5000)
        assert len(far["f"]) >= len(near["f"])


# ============================================================
# Different step sizes
# ============================================================

class TestDifferentStepSizes:
    def test_too_small_step_still_converges_but_slowly(self):
        a = Matrix([[3, 0], [0, 1]])
        b = [1, 1]
        f, grad = make_quadratic(a, b)
        lr_optimal = optimal_step_size(a)
        _, small = gradient_descent(f, [5.0, -4.0], lr_optimal * 0.05, grad_f=grad, num_iterations=20000)
        _, optimal = gradient_descent(f, [5.0, -4.0], lr_optimal, grad_f=grad, num_iterations=20000)
        assert small["converged"] and optimal["converged"]
        assert len(small["f"]) > len(optimal["f"])

    def test_step_past_divergence_threshold_diverges(self):
        a = Matrix([[3, 0], [0, 1]])  # lambda_max = 3, threshold = 2/3
        b = [1, 1]
        f, grad = make_quadratic(a, b)
        _, history = gradient_descent(f, [5.0, -4.0], (2 / 3) * 1.1, grad_f=grad, num_iterations=10000)
        assert history["diverged"]

    def test_step_just_under_threshold_still_converges(self):
        a = Matrix([[3, 0], [0, 1]])
        b = [1, 1]
        f, grad = make_quadratic(a, b)
        _, history = gradient_descent(f, [5.0, -4.0], (2 / 3) * 0.99, grad_f=grad, num_iterations=20000, tol=1e-8)
        assert history["converged"]
        assert not history["diverged"]


# ============================================================
# Poorly conditioned problems
# ============================================================

class TestPoorlyConditionedProblems:
    def test_hilbert_quadratic_needs_far_more_iterations_than_identity_like_case(self):
        well = Matrix([[3, 0], [0, 1]])
        b_well = [1, 1]
        f_well, grad_well = make_quadratic(well, b_well)
        lr_well = optimal_step_size(well)
        _, hist_well = gradient_descent(f_well, [5.0, -4.0], lr_well, grad_f=grad_well, num_iterations=50000, tol=1e-6)

        poor = hilbert_matrix(5)
        b_poor = [1.0] * 5
        f_poor, grad_poor = make_quadratic(poor, b_poor)
        lr_poor = optimal_step_size(poor)
        _, hist_poor = gradient_descent(f_poor, [0.0] * 5, lr_poor, grad_f=grad_poor, num_iterations=50000, tol=1e-6)

        assert hist_well["converged"]
        # the Hilbert system either needs vastly more iterations or fails
        # to converge at all within the same generous budget
        assert (not hist_poor["converged"]) or len(hist_poor["f"]) > 10 * len(hist_well["f"])

    def test_condition_number_of_hilbert_grows_with_size(self):
        kappas = [condition_number(hilbert_matrix(n)) for n in (2, 3, 4)]
        assert kappas[0] < kappas[1] < kappas[2]


# ============================================================
# Supporting pieces: numerical gradient, objective correctness, scipy check
# ============================================================

class TestNumericalGradient:
    def test_matches_analytic_gradient_for_quadratic(self):
        a = Matrix([[4, 1], [1, 3]])
        b = [1, 2]
        f, grad_exact = make_quadratic(a, b)
        x = [2.5, -1.0]
        numeric = numerical_gradient(f, x)
        exact = grad_exact(x)
        for n, e in zip(numeric, exact):
            assert math.isclose(n, e, abs_tol=1e-6)

    def test_matches_analytic_gradient_for_least_squares(self):
        a = Matrix([[2, 0, 1], [1, 3, 0], [0, 1, 2]])
        b = [1, -1, 2]
        f, grad_exact = make_least_squares_objective(a, b)
        x = [0.5, 0.2, -0.3]
        numeric = numerical_gradient(f, x)
        exact = grad_exact(x)
        for n, e in zip(numeric, exact):
            assert math.isclose(n, e, abs_tol=1e-4)


class TestGradientDescentCorrectness:
    def test_quadratic_minimum_matches_direct_solve(self):
        a = Matrix([[4, 1], [1, 3]])
        b = [1, 2]
        f, grad = make_quadratic(a, b)
        lr = optimal_step_size(a)
        x_final, history = gradient_descent(f, [0, 0], lr, grad_f=grad, num_iterations=5000)
        x_star = a.inverse().multiply_vector(b)
        assert history["converged"]
        assert norm(vec_sub(x_final, x_star)) < 1e-4

    def test_least_squares_gradient_descent_matches_closed_form(self):
        xs = [0, 1, 2, 3, 4, 5]
        ys = [1.2, 2.9, 5.1, 6.8, 9.2, 10.9]
        a = Matrix([[x, 1] for x in xs])
        f, grad = make_least_squares_objective(a, ys)
        ata = a.transpose() * a
        lr = optimal_step_size(ata) * 0.5
        x_gd, history = gradient_descent(f, [0.0, 0.0], lr, grad_f=grad, num_iterations=5000)
        x_closed = closed_form_least_squares(a, ys)
        assert history["converged"]
        assert norm(vec_sub(x_gd, x_closed)) < 1e-3


class TestScipyComparison:
    def test_gradient_descent_matches_scipy_bfgs(self):
        np = pytest.importorskip("numpy")
        from scipy.optimize import minimize as scipy_minimize

        a = Matrix([[3, 0], [0, 1]])
        b = [1, 1]
        f, grad = make_quadratic(a, b)
        lr = optimal_step_size(a)
        x_gd, history = gradient_descent(f, [5.0, -4.0], lr, grad_f=grad)

        result = scipy_minimize(
            lambda x: f(x.tolist()), np.array([5.0, -4.0]), jac=lambda x: np.array(grad(x.tolist())), method="BFGS"
        )
        assert history["converged"]
        for ours, theirs in zip(x_gd, result.x.tolist()):
            assert math.isclose(ours, theirs, abs_tol=1e-5)
