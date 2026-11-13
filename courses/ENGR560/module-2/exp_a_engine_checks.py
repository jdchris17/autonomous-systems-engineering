"""
Experiment A: exercise the gradient-descent engine against each stated requirement.

  1. Works with more than two parameters  -> fit a 4-parameter cubic and a 6-feature random model
  2. Vectorized NumPy                      -> see gd_core.mse_loss / mse_gradient (no Python loops)
  3. Records the complete trajectory       -> history lengths = iterations + 1
  4. Detects nonfinite values              -> run with a learning rate far above the stable limit
  5. Leaves theta_initial unchanged        -> compare before/after
  6. Clear stopping criterion              -> converged / max_iterations / nonfinite statuses
"""

import numpy as np

from gd_core import (CONVERGED, MAX_ITER, NONFINITE, design_matrix, gradient_descent, mse_hessian)


def check(label, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return ok


def main():
    print("Experiment A: gradient-descent engine checks")
    rng = np.random.default_rng(2)
    results = []

    # Cubic truth, 4 parameters, x on [-1, 1] so the problem is reasonably conditioned.
    x = rng.uniform(-1, 1, 200)
    y = 1.0 - 2.0 * x + 0.5 * x**2 + 3.0 * x**3 + rng.normal(0, 0.1, x.size)
    X = design_matrix(x, degree=3)
    theta0 = np.zeros(4)
    theta0_before = theta0.copy()
    eta = 1.0 / np.linalg.eigvalsh(mse_hessian(X)).max()
    theta, loss_h, theta_h, info = gradient_descent(X, y, theta0, eta, max_iterations=200_000,
                                                    tolerance=1e-8)
    ref = np.linalg.solve(X.T @ X, X.T @ y)   # normal equations, used only to check the answer
    print(f"  cubic, 4 parameters: status={info['status']}, iterations={info['iterations']},"
          f" theta={np.round(theta, 4)}, |theta - theta_ref| = {np.linalg.norm(theta - ref):.2e}")
    results.append(check("more than two parameters (4-parameter cubic converges to normal-equation"
                         " solution)", info["status"] == CONVERGED
                         and np.linalg.norm(theta - ref) < 1e-6))
    results.append(check("trajectory recorded: len(theta_history) == len(loss_history) =="
                         " iterations + 1",
                         len(theta_h) == len(loss_h) == info["iterations"] + 1))
    results.append(check("loss history is non-increasing for a stable learning rate",
                         bool(np.all(np.diff(loss_h) <= 1e-12))))
    results.append(check("initial parameter array left unchanged",
                         np.array_equal(theta0, theta0_before)))

    # 6 random features
    Xr = np.column_stack([np.ones(300), rng.normal(size=(300, 5))])
    yr = Xr @ np.array([0.5, -1, 2, 0, 1.5, -0.7]) + rng.normal(0, 0.2, 300)
    th, _, _, inf6 = gradient_descent(Xr, yr, np.zeros(6), 0.2, max_iterations=10_000)
    ref6 = np.linalg.solve(Xr.T @ Xr, Xr.T @ yr)
    results.append(check(f"6-parameter model converges ({inf6['iterations']} iterations,"
                         f" |error| = {np.linalg.norm(th - ref6):.1e})",
                         inf6["status"] == CONVERGED and np.linalg.norm(th - ref6) < 1e-6))

    # Divergence detection
    _, lh, _, bad = gradient_descent(X, y, theta0, 50 * eta, max_iterations=100_000)
    results.append(check(f"divergence detected as '{bad['status']}' after {bad['iterations']}"
                         f" updates (last finite loss {lh[np.isfinite(lh)][-1]:.2e})",
                         bad["status"] == NONFINITE))

    # Budget exhaustion
    _, _, _, short = gradient_descent(X, y, theta0, eta, max_iterations=10)
    results.append(check(f"iteration limit reported as '{short['status']}'"
                         f" (gradient norm still {short['grad_norm']:.2e})",
                         short["status"] == MAX_ITER and short["iterations"] == 10))

    print(f"  {sum(results)}/{len(results)} checks passed")


if __name__ == "__main__":
    main()
