"""
Experiment B: teach the computer the relationship.

Model:  y_hat = theta0 + theta1 x  =  X theta,  with X = [1  x_i].
theta is estimated with numpy.linalg.lstsq from the 30 training pairs only
(learning_core.fit_linear never sees TRUE_INTERCEPT or TRUE_SLOPE).

Also reports the classical standard errors of the estimates, using the residual
variance from the fit (again: nothing from the true model), so the size of the
estimation error can be judged against what the data itself says is plausible.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from learning_core import (TRUE_INTERCEPT, TRUE_SLOPE, design_matrix, fit_linear, predict,
                           starter_datasets, true_function)

OUT = Path(__file__).parent / "outputs"


def standard_errors(x, y, theta):
    """OLS standard errors: sqrt(diag(s^2 (X^T X)^-1)), s^2 = RSS / (N - p)."""
    X = design_matrix(x)
    resid = y - X @ theta
    n, p = X.shape
    s2 = resid @ resid / (n - p)
    cov = s2 * np.linalg.inv(X.T @ X)
    return np.sqrt(np.diag(cov)), np.sqrt(s2)


def main():
    OUT.mkdir(exist_ok=True)
    x_train, y_train, _, _ = starter_datasets()

    theta = fit_linear(x_train, y_train)
    se, s = standard_errors(x_train, y_train, theta)

    print("Experiment B: least-squares fit on N = 30 training observations")
    print(f"  {'parameter':<14}{'learned':>10}{'true':>10}{'error':>10}{'std err':>10}{'error/SE':>10}")
    for name, est, true, e in (("intercept th0", theta[0], TRUE_INTERCEPT, se[0]),
                               ("slope th1", theta[1], TRUE_SLOPE, se[1])):
        print(f"  {name:<14}{est:>10.4f}{true:>10.4f}{est - true:>+10.4f}{e:>10.4f}"
              f"{(est - true) / e:>+10.2f}")
    print(f"  residual std estimated from the fit: {s:.3f}  (true noise std 1.5)")

    xs = np.linspace(-5, 5, 200)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(x_train, y_train, s=28, color="tab:blue", edgecolor="k", linewidth=0.4,
               label="training observations", zorder=3)
    ax.plot(xs, true_function(xs), color="k", lw=2,
            label=f"true relationship: y = {TRUE_INTERCEPT:g} + {TRUE_SLOPE:g}x")
    ax.plot(xs, predict(theta, xs), color="tab:orange", lw=2, ls="--",
            label=f"learned model: y = {theta[0]:.3f} + {theta[1]:.3f}x")
    ax.set_xlabel("input x")
    ax.set_ylabel("sensor output y")
    ax.set_title("Experiment B: least-squares fit vs the true relationship")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "exp_b_fit.png", dpi=150)
    plt.close(fig)
    print(f"  saved {OUT / 'exp_b_fit.png'}")
    return theta


if __name__ == "__main__":
    main()
