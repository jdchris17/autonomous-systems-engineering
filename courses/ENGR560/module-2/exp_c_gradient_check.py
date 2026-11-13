"""
Experiment C: numerical gradient checking with central differences.

    g_num_j = (J(theta + h e_j) - J(theta - h e_j)) / (2h)
    E = ||g_a - g_n|| / max(1, ||g_a||, ||g_n||)

Error budget for central differences:
    truncation  ~ (h^2 / 6) * |J'''|          (shrinks as h gets smaller)
    roundoff    ~ eps_machine * |J| / h       (GROWS as h gets smaller; eps ~ 2.2e-16)
Total is smallest near h ~ (3 eps |J| / |J'''|)^(1/3), i.e. around 1e-5 in double precision.

Three test problems:
  1. Module 1 MSE, 2 parameters. J is an exact quadratic, so J''' = 0: there is NO truncation
     error at all and only roundoff remains - accuracy just gets worse as h shrinks.
  2. MSE with a 4-parameter cubic design (shows the checker is not tied to p = 2).
  3. A non-quadratic loss, mean log-cosh of the residual, where truncation is real and the
     classic "V-shaped" error curve appears.
Then three deliberately broken gradients (factor 2 dropped, residual sign flipped, only one
example used) show what a failed check looks like.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gd_core import (design_matrix, fit_linear, mse_gradient, mse_loss, numerical_gradient,
                     relative_discrepancy, starter_datasets)

OUT = Path(__file__).parent / "outputs"
H_MAIN = [1e-2, 1e-4, 1e-6, 1e-8]
H_SWEEP = np.logspace(-1, -12, 45)


def logcosh_loss(X, y, theta):
    r = X @ theta - y
    # log(cosh(r)) computed stably: |r| + log1p(exp(-2|r|)) - log 2
    a = np.abs(r)
    return float(np.mean(a + np.log1p(np.exp(-2 * a)) - np.log(2.0)))


def logcosh_gradient(X, y, theta):
    return X.T @ np.tanh(X @ theta - y) / y.size


def main():
    OUT.mkdir(exist_ok=True)
    x, y, _, _ = starter_datasets()
    X = design_matrix(x)
    X3 = design_matrix(x, degree=3)
    rng = np.random.default_rng(7)

    problems = {
        "MSE, line (p=2)": (X, mse_loss, mse_gradient, [
            ("zeros", np.zeros(2)),
            ("least-squares optimum", fit_linear(x, y)),
            ("random A", rng.normal(0, 3, 2)),
            ("random B (large)", rng.normal(0, 50, 2)),
        ]),
        "MSE, cubic (p=4)": (X3, mse_loss, mse_gradient, [
            ("zeros", np.zeros(4)),
            ("random", rng.normal(0, 1, 4)),
        ]),
        "log-cosh, line (p=2)": (X, logcosh_loss, logcosh_gradient, [
            ("zeros", np.zeros(2)),
            ("random", rng.normal(0, 1, 2)),
        ]),
    }

    print("Experiment C: relative discrepancy E between analytical and central-difference gradients")
    hdr = f"  {'problem':<22}{'theta':<24}" + "".join(f"{f'h={h:g}':>11}" for h in H_MAIN)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    sweep = {}
    for pname, (Xp, loss, grad, thetas) in problems.items():
        for tname, th in thetas:
            J = lambda t, Xp=Xp, loss=loss: loss(Xp, y, t)  # noqa: E731
            g_a = grad(Xp, y, th)
            errs = [relative_discrepancy(g_a, numerical_gradient(J, th, h)) for h in H_MAIN]
            print(f"  {pname:<22}{tname:<24}" + "".join(f"{e:>11.2e}" for e in errs))
            if tname in ("random A", "random") and pname != "MSE, cubic (p=4)":
                sweep[pname] = [relative_discrepancy(g_a, numerical_gradient(J, th, h))
                                for h in H_SWEEP]

    # Deliberately buggy gradients, checked at a random point with h = 1e-5
    th = np.array([1.0, 0.5])
    J = lambda t: mse_loss(X, y, t)  # noqa: E731
    g_n = numerical_gradient(J, th, 1e-5)
    bugs = {
        "correct (2/N) X^T r": mse_gradient(X, y, th),
        "bug: dropped factor 2": (1.0 / y.size) * X.T @ (X @ th - y),
        "bug: sign of residual flipped": (2.0 / y.size) * X.T @ (y - X @ th),
        "bug: only last example used": 2.0 * X[-1] * (X[-1] @ th - y[-1]),
    }
    print("\n  Bug detection at theta = [1, 0.5], h = 1e-5:")
    for k, g in bugs.items():
        e = relative_discrepancy(g, g_n)
        print(f"    {k:<32} E = {e:.2e}   {'OK' if e < 1e-6 else 'FAILS CHECK'}")

    fig, ax = plt.subplots(figsize=(8, 5))
    for pname, errs in sweep.items():
        ax.loglog(H_SWEEP, np.maximum(errs, 1e-18), "o-", ms=3, label=pname)
    eps = np.finfo(float).eps
    ax.loglog(H_SWEEP, eps / H_SWEEP, "k:", lw=1, label="roundoff ~ eps / h")
    ax.loglog(H_SWEEP, 0.05 * H_SWEEP**2, "k--", lw=1, label="truncation ~ h^2")
    for h in H_MAIN:
        ax.axvline(h, color="gray", lw=0.6, alpha=0.5)
    ax.set_ylim(1e-17, 1e-1)
    ax.invert_xaxis()
    ax.set_xlabel("finite-difference step h (smaller to the right)")
    ax.set_ylabel("relative discrepancy E")
    ax.set_title("Gradient check: smaller h is not always better")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "exp_c_gradient_check.png", dpi=150)
    plt.close(fig)
    print(f"\n  saved {OUT / 'exp_c_gradient_check.png'}")


if __name__ == "__main__":
    main()
