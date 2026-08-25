"""Gradient descent: numerical gradients, minimizing quadratics and
least-squares problems, learning rate behavior, and a scipy comparison.
"""

from __future__ import annotations

import math
import pathlib
import random

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize as scipy_minimize

from matrix import Matrix
from least_squares import dot, least_squares as closed_form_least_squares, norm, vec_scale, vec_sub
from eigen import diagonalize_symmetric
from numerical import condition_number, hilbert_matrix

OUTPUT_DIR = pathlib.Path(__file__).parent


# ============================================================
# Numerical gradient
# ============================================================

def numerical_gradient(f, x, h: float = 1e-5):
    """Central-difference approximation: df/dx_i ~= (f(x+h e_i) - f(x-h e_i)) / 2h."""
    grad = []
    for i in range(len(x)):
        x_plus = list(x)
        x_plus[i] += h
        x_minus = list(x)
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad


# ============================================================
# Objective functions: quadratic bowl and least-squares
# ============================================================

def make_quadratic(a: Matrix, b: list, c: float = 0.0):
    """f(x) = 0.5 x^T A x - b^T x + c, gradient = A x - b. A should be
    symmetric positive definite for a genuine (convex) bowl."""

    def f(x):
        ax = a.multiply_vector(x)
        return 0.5 * dot(x, ax) - dot(b, x) + c

    def grad(x):
        return vec_sub(a.multiply_vector(x), b)

    return f, grad


def make_least_squares_objective(a: Matrix, b: list):
    """f(x) = ||Ax - b||^2, gradient = 2 A^T (Ax - b)."""

    def f(x):
        residual = vec_sub(a.multiply_vector(x), b)
        return dot(residual, residual)

    def grad(x):
        residual = vec_sub(a.multiply_vector(x), b)
        return vec_scale(a.transpose().multiply_vector(residual), 2.0)

    return f, grad


# ============================================================
# Gradient descent
# ============================================================

def gradient_descent(f, x0, learning_rate: float, grad_f=None, num_iterations: int = 1000, tol: float = 1e-10):
    """Minimize f via fixed-step gradient descent. Uses grad_f if given,
    otherwise falls back to numerical_gradient(f, x). Returns (x_final,
    history) where history has 'x' and 'f' lists, one entry per step
    (including the starting point), plus 'diverged' if a non-finite value
    was hit (e.g. from too large a learning rate).
    """
    grad = grad_f or (lambda x: numerical_gradient(f, x))
    x = list(x0)
    history = {"x": [list(x)], "f": [f(x)], "diverged": False, "converged": False}

    for _ in range(num_iterations):
        g = grad(x)
        x = [xi - learning_rate * gi for xi, gi in zip(x, g)]
        fx = f(x)
        if not math.isfinite(fx):
            history["diverged"] = True
            break
        history["x"].append(list(x))
        history["f"].append(fx)
        if norm(g) < tol:
            history["converged"] = True
            break

    return x, history


def optimal_step_size(a: Matrix) -> float:
    """For a symmetric positive definite A, the standard optimal fixed
    step size for gradient descent on 0.5 x^T A x - b^T x is
    2 / (lambda_max + lambda_min)."""
    _, d = diagonalize_symmetric(a, seed=0)
    eigenvalues = [d.rows[i][i] for i in range(d.num_rows)]
    return 2 / (max(eigenvalues) + min(eigenvalues))


# ============================================================
# Plotting
# ============================================================

def plot_convergence(series: dict, title: str, out_path: pathlib.Path, logy: bool = True, xlabel="iteration", ylabel="f(x)"):
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, values in series.items():
        ax.plot(range(len(values)), values, label=label)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ============================================================
# Demos
# ============================================================

def demo_numerical_gradient():
    print("=" * 78)
    print("1. Numerical gradient (finite differences) vs. analytic gradient")
    print("=" * 78)
    a = Matrix([[4, 1], [1, 3]])
    b = [1, 2]
    f, grad_exact = make_quadratic(a, b)

    x = [2.5, -1.0]
    numeric = numerical_gradient(f, x)
    exact = grad_exact(x)
    print(f"  quadratic bowl, x = {x}")
    print(f"  analytic gradient  = {[round(v, 6) for v in exact]}")
    print(f"  numerical gradient = {[round(v, 6) for v in numeric]}")
    print(f"  max component diff = {max(abs(a1 - a2) for a1, a2 in zip(exact, numeric)):.2e}")

    a2 = Matrix([[2, 0, 1], [1, 3, 0]])  # 2x3, least-squares objective
    b2 = [1, -1]
    f2, grad2_exact = make_least_squares_objective(a2, b2)
    x2 = [0.5, 0.2, -0.3]
    numeric2 = numerical_gradient(f2, x2)
    exact2 = grad2_exact(x2)
    print(f"\n  least-squares objective ||Ax-b||^2, x = {x2}")
    print(f"  analytic gradient  = {[round(v, 6) for v in exact2]}")
    print(f"  numerical gradient = {[round(v, 6) for v in numeric2]}")
    print(f"  max component diff = {max(abs(a1 - a2) for a1, a2 in zip(exact2, numeric2)):.2e}")
    print()


def demo_gradient_descent_quadratic():
    print("=" * 78)
    print("2. Gradient descent on a quadratic bowl")
    print("=" * 78)
    a = Matrix([[3, 0], [0, 1]])  # condition number 3, mild
    b = [1, 1]
    f, grad = make_quadratic(a, b)
    x_star = a.inverse().multiply_vector(b)  # exact minimum: Ax=b
    lr = optimal_step_size(a)

    x0 = [5.0, -4.0]
    x_analytic, hist_analytic = gradient_descent(f, x0, lr, grad_f=grad)
    x_numeric, hist_numeric = gradient_descent(f, x0, lr, grad_f=None)  # finite-difference gradient

    print(f"  A = [[3,0],[0,1]], b = {b}, exact minimum x* = {[round(v, 6) for v in x_star]}")
    print(f"  learning rate = 2/(lambda_max+lambda_min) = {lr:.4f}")
    print(f"  GD (analytic gradient): {[round(v, 6) for v in x_analytic]}, "
          f"{len(hist_analytic['f']) - 1} iterations")
    print(f"  GD (numerical gradient): {[round(v, 6) for v in x_numeric]}, "
          f"{len(hist_numeric['f']) - 1} iterations")

    plot_convergence(
        {"analytic gradient": hist_analytic["f"], "numerical gradient": hist_numeric["f"]},
        "Gradient descent convergence on a quadratic bowl",
        OUTPUT_DIR / "opt_2_quadratic_convergence.png",
    )
    print(f"  saved plot -> opt_2_quadratic_convergence.png")
    print()


def demo_gradient_descent_least_squares():
    print("=" * 78)
    print("3. Gradient descent on a least-squares problem")
    print("=" * 78)
    xs = [0, 1, 2, 3, 4, 5]
    ys = [1.2, 2.9, 5.1, 6.8, 9.2, 10.9]  # ~ y = 2x + 1
    a = Matrix([[x, 1] for x in xs])
    f, grad = make_least_squares_objective(a, ys)

    closed_form = closed_form_least_squares(a, ys)  # from least_squares.py

    ata = a.transpose() * a
    lr = optimal_step_size(ata) * 0.5  # ||Ax-b||^2 has Hessian 2 A^T A; halve the A^T A-based step size
    x0 = [0.0, 0.0]
    x_gd, history = gradient_descent(f, x0, lr, grad_f=grad, num_iterations=5000)

    print(f"  fitting y = m x + c to noisy data")
    print(f"  closed-form (normal equations): m={closed_form[0]:.4f}, c={closed_form[1]:.4f}")
    print(f"  gradient descent:               m={x_gd[0]:.4f}, c={x_gd[1]:.4f}  ({len(history['f'])-1} iterations)")
    print(f"  agreement: max diff = {max(abs(a1 - a2) for a1, a2 in zip(closed_form, x_gd)):.2e}")

    plot_convergence(
        {"gradient descent loss": history["f"]},
        "Gradient descent on ||Ax-b||^2 (line-fit)",
        OUTPUT_DIR / "opt_3_least_squares_convergence.png",
    )
    print(f"  saved plot -> opt_3_least_squares_convergence.png")
    print()


def demo_learning_rate_investigation():
    print("=" * 78)
    print("4. Learning rate investigation")
    print("=" * 78)
    a = Matrix([[3, 0], [0, 1]])
    b = [1, 1]
    f, grad = make_quadratic(a, b)
    lr_optimal = optimal_step_size(a)
    x0 = [5.0, -4.0]

    rates = {
        "too small (0.05x optimal)": lr_optimal * 0.05,
        "well-tuned (optimal)":      lr_optimal,
        "too large (1.05x threshold)": (2 / 3) * 1.05,  # divergence threshold is 2/lambda_max = 2/3
    }
    series = {}
    for label, lr in rates.items():
        x_final, history = gradient_descent(f, x0, lr, grad_f=grad, num_iterations=5000)
        if history["diverged"]:
            status = "DIVERGED (loss overflowed to inf)"
        elif history["converged"]:
            status = f"converged in {len(history['f'])-1} iterations"
        else:
            status = f"did NOT converge (hit the {5000}-iteration cap; loss still {history['f'][-1]:.3e})"
        print(f"  {label:<30} lr={lr:.4f}  -> {status}")
        series[f"{label} (lr={lr:.3f})"] = history["f"]

    plot_convergence(
        series,
        "Learning rate comparison on the same quadratic bowl",
        OUTPUT_DIR / "opt_4_learning_rates.png",
    )
    print(f"  saved plot -> opt_4_learning_rates.png")
    print(
        "  -> too small crawls toward the minimum, wasting iterations on tiny\n"
        "     steps. Well-tuned (2/(lambda_max+lambda_min)) converges fast and\n"
        "     smoothly. Too large overshoots the bowl on every step and the loss\n"
        "     grows without bound instead of shrinking -- gradient descent doesn't\n"
        "     just converge slowly past a threshold, it actively diverges."
    )
    print()


def demo_scipy_comparison():
    print("=" * 78)
    print("5. Comparison against scipy.optimize.minimize")
    print("=" * 78)
    a = Matrix([[3, 0], [0, 1]])
    b = [1, 1]
    f, grad = make_quadratic(a, b)
    x0 = [5.0, -4.0]
    lr = optimal_step_size(a)

    x_gd, history = gradient_descent(f, x0, lr, grad_f=grad)

    def f_np(x):
        return f(x.tolist())

    def grad_np(x):
        return np.array(grad(x.tolist()))

    result = scipy_minimize(f_np, np.array(x0), jac=grad_np, method="BFGS")
    print(f"  our gradient descent: x = {[round(v, 6) for v in x_gd]}, f(x) = {f(x_gd):.3e}, "
          f"{len(history['f'])-1} iterations")
    print(f"  scipy BFGS:           x = {[round(v, 6) for v in result.x.tolist()]}, f(x) = {result.fun:.3e}, "
          f"{result.nit} iterations")
    print(f"  agreement: max diff = {max(abs(a1 - a2) for a1, a2 in zip(x_gd, result.x.tolist())):.2e}")
    print(
        "  -> scipy's BFGS builds up curvature (an approximate inverse Hessian)\n"
        "     as it goes, so it reaches the same minimum in far fewer iterations\n"
        "     than our fixed-step gradient descent. The point of writing our own\n"
        "     is to understand what a single gradient step actually does, not to\n"
        "     out-perform a tuned library optimizer."
    )
    print()


# ============================================================
# Suggested testing
# ============================================================

def demo_convergence_speed_vs_conditioning():
    print("=" * 78)
    print("6a. Convergence speed vs. condition number")
    print("=" * 78)
    print(f"  {'kappa':>10}  {'iterations to converge':>24}")
    for eigenvalues in ([1, 1], [1, 5], [1, 20], [1, 100]):
        a = Matrix([[eigenvalues[0], 0], [0, eigenvalues[1]]])
        b = [1, 1]
        f, grad = make_quadratic(a, b)
        lr = optimal_step_size(a)
        _, history = gradient_descent(f, [5.0, -4.0], lr, grad_f=grad, num_iterations=20000, tol=1e-8)
        kappa = max(eigenvalues) / min(eigenvalues)
        print(f"  {kappa:>10.1f}  {len(history['f'])-1:>24}")
    print(
        "  -> even with the theoretically optimal step size, iterations needed\n"
        "     grows with kappa(A): a poorly-conditioned bowl is steep in one\n"
        "     direction and shallow in another, so no single fixed step size can\n"
        "     be 'right' for both at once -- exactly the same condition-number\n"
        "     story from numerical.py, now showing up as optimizer iteration count\n"
        "     instead of solver sensitivity."
    )
    print()


def demo_different_starting_points():
    print("=" * 78)
    print("6b. Different starting points")
    print("=" * 78)
    a = Matrix([[3, 0], [0, 1]])
    b = [1, 1]
    f, grad = make_quadratic(a, b)
    lr = optimal_step_size(a)
    x_star = a.inverse().multiply_vector(b)

    print(f"  {'start':>18}  {'final x':>22}  {'iterations':>12}")
    for x0 in ([0, 0], [10, 10], [-8, 6], [100, -50]):
        x_final, history = gradient_descent(f, x0, lr, grad_f=grad, num_iterations=5000)
        print(f"  {str(x0):>18}  {str([round(v, 4) for v in x_final]):>22}  {len(history['f'])-1:>12}")
    print(
        f"  -> every start converges to the SAME minimum x* = {[round(v,4) for v in x_star]}: this\n"
        "     objective is convex (a single bowl, positive definite A), so there's\n"
        "     no local minimum to get trapped in. Iteration count varies with\n"
        "     starting distance, but the destination never does."
    )
    print()


def demo_step_size_sweep():
    print("=" * 78)
    print("6c. Step size sweep")
    print("=" * 78)
    a = Matrix([[3, 0], [0, 1]])
    b = [1, 1]
    f, grad = make_quadratic(a, b)
    divergence_threshold = 2 / 3  # 2 / lambda_max
    x0 = [5.0, -4.0]

    fractions = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.05]
    iterations_needed = []
    labels = []
    for frac in fractions:
        lr = divergence_threshold * frac
        _, history = gradient_descent(f, x0, lr, grad_f=grad, num_iterations=10000, tol=1e-8)
        if history["diverged"]:
            status = "diverged"
        elif history["converged"]:
            status = f"{len(history['f']) - 1} iterations"
        else:
            status = f"did NOT converge within 10000 iterations (loss={history['f'][-1]:.3e})"
        iterations_needed.append(len(history["f"]) - 1 if history["converged"] else None)
        labels.append(f"{frac:.2f}x threshold")
        print(f"  lr = {frac:.2f} x divergence threshold ({lr:.4f}) -> {status}")
    print(
        "  -> iterations-to-converge falls as the step size approaches (but stays\n"
        "     below) the divergence threshold 2/lambda_max, then the system flips\n"
        "     to diverging entirely just past it -- there's a real cliff edge, not\n"
        "     a gentle tradeoff."
    )
    print()


def demo_poorly_conditioned_problem():
    print("=" * 78)
    print("6d. Poorly conditioned problem: Hilbert matrix")
    print("=" * 78)
    n = 5
    a = hilbert_matrix(n)
    b = [1.0] * n
    f, grad = make_quadratic(a, b)
    kappa = condition_number(a)
    lr = optimal_step_size(a)

    x0 = [0.0] * n
    x_final, history = gradient_descent(f, x0, lr, grad_f=grad, num_iterations=50000, tol=1e-8)
    x_exact = a.inverse().multiply_vector(b)

    print(f"  {n}x{n} Hilbert matrix, kappa = {kappa:.3e}")
    print(f"  gradient descent ran {len(history['f'])-1} iterations "
          f"({'converged' if len(history['f'])-1 < 50000 else 'hit the iteration cap, did NOT converge'})")
    print(f"  ||x_gd - x_exact|| = {norm(vec_sub(x_final, x_exact)):.3e}")
    print(
        "  -> compare this to the well-conditioned quadratic in section 2, which\n"
        "     converged in a handful of iterations: the same algorithm, same\n"
        "     tolerance, same optimal-step-size formula, needs orders of magnitude\n"
        "     more iterations here (or doesn't finish at all within a generous\n"
        "     budget) purely because of conditioning. This is exactly why\n"
        "     least_squares.py and numerical.py use CLOSED-FORM solves (normal\n"
        "     equations, QR, SVD) instead of gradient descent where possible --\n"
        "     they don't degrade with kappa(A) the way a fixed-step iterative\n"
        "     method does."
    )
    print()


if __name__ == "__main__":
    demo_numerical_gradient()
    demo_gradient_descent_quadratic()
    demo_gradient_descent_least_squares()
    demo_learning_rate_investigation()
    demo_scipy_comparison()
    demo_convergence_speed_vs_conditioning()
    demo_different_starting_points()
    demo_step_size_sweep()
    demo_poorly_conditioned_problem()
