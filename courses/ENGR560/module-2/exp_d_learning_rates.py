"""
Experiment D: learning rates, Hessian eigenvalues, and feature scaling.

Same data as Experiment B (Module 1 starter, 30 points). Start [0, 0], tolerance ||grad|| <= 1e-8.

Part 1  eta in {0.001, 0.01, 0.05, 0.1, 0.5}, 5000-update budget.
Part 2  H = (2/N) X^T X. For the quadratic J, the error e_k = theta_k - theta* obeys
            e_{k+1} = (I - eta H) e_k
        so along eigenvector i the error is multiplied by (1 - eta lambda_i) every step.
        Stable  <=>  |1 - eta lambda_i| < 1 for all i  <=>  0 < eta < 2 / lambda_max.
        Speed is set by the SLOWEST mode, rho(eta) = max_i |1 - eta lambda_i|, which is
        minimised at eta* = 2 / (lambda_min + lambda_max), giving rho* = (kappa - 1)/(kappa + 1).
Part 3  x -> 100 x (targets unchanged), then standardize with training mean/std.
        Compare condition numbers, convergence speed, stable eta range, test-set accuracy.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gd_core import (CONVERGED, NONFINITE, design_matrix, gradient_descent, mse, mse_gradient,
                     mse_hessian, mse_loss, starter_datasets)

OUT = Path(__file__).parent / "outputs"
ETAS = [0.001, 0.01, 0.05, 0.1, 0.5]
BUDGET = 5000
TOL = 1e-8
COLORS = dict(zip(ETAS, ["tab:purple", "tab:blue", "tab:green", "tab:orange", "tab:red"]))


def ls_solution(X, y):
    """Reference via normal equations (independent of the optimizer)."""
    return np.linalg.solve(X.T @ X, X.T @ y)


def spectrum(X):
    lam = np.linalg.eigvalsh(mse_hessian(X))
    return lam, lam.max() / lam.min(), 2.0 / lam.max()


def rho(eta, lam):
    return np.max(np.abs(1.0 - eta * lam))


def predicted_iterations(eta, lam, g0_norm, tol=TOL):
    """Updates until ||grad|| ~ tol, if the slowest mode dominates: rho^k ||g0|| = tol."""
    r = rho(eta, lam)
    if r >= 1:
        return np.inf
    if r == 0:
        return 1
    return np.log(tol / g0_norm) / np.log(r)


def fmt_status(info):
    return {"converged": "converged", "max_iterations": "hit limit", "nonfinite": "DIVERGED"}[
        info["status"]]


# --------------------------------------------------------------------------- Part 1 + 2
def part1_and_2(x, y):
    X = design_matrix(x)
    th_ls = ls_solution(X, y)
    j_ls = mse_loss(X, y, th_ls)
    lam, kappa, eta_max = spectrum(X)
    g0 = np.linalg.norm(mse_gradient(X, y, np.zeros(2)))

    print("D Part 1: learning-rate comparison (original x, budget 5000 updates)")
    print(f"  {'eta':>6} {'status':>10} {'updates':>8} {'final loss':>14} {'|theta - LS|':>13}"
          f" {'rho(eta)':>9} {'predicted':>10}")
    runs = {}
    for eta in ETAS:
        th, lh, thh, info = gradient_descent(X, y, np.zeros(2), eta, BUDGET, TOL)
        runs[eta] = (lh, info)
        d = np.linalg.norm(th - th_ls)
        pred = predicted_iterations(eta, lam, g0)
        print(f"  {eta:>6g} {fmt_status(info):>10} {info['iterations']:>8} {lh[-1]:>14.6g}"
              f" {d:>13.3g} {rho(eta, lam):>9.4f} {pred:>10.0f}")
    print(f"  least-squares loss J* = {j_ls:.10f}")

    print("\nD Part 2: Hessian eigenvalue analysis")
    print(f"  H = (2/N) X^T X = {np.array2string(mse_hessian(X), precision=4)}")
    print(f"  eigenvalues: lambda_min = {lam.min():.4f}, lambda_max = {lam.max():.4f},"
          f"  condition number kappa = {kappa:.3f}")
    print(f"  largest stable constant learning rate: eta < 2/lambda_max = {eta_max:.5f}")
    eta_opt = 2.0 / (lam.min() + lam.max())
    print(f"  fastest constant learning rate eta* = 2/(lambda_min+lambda_max) = {eta_opt:.5f},"
          f"  rho* = {rho(eta_opt, lam):.4f} = (kappa-1)/(kappa+1)")
    for f in (0.98, 1.02):
        eta = f * eta_max
        _, lh, _, info = gradient_descent(X, y, np.zeros(2), eta, 20_000, TOL)
        print(f"  check eta = {f:.2f} x limit = {eta:.5f}: {fmt_status(info)} after"
              f" {info['iterations']} updates")

    # Sweep eta finely across the whole stable range and a little beyond
    sweep = np.linspace(0.002, 1.08 * eta_max, 140)
    its = []
    for eta in sweep:
        _, _, _, info = gradient_descent(X, y, np.zeros(2), eta, 20_000, TOL)
        its.append(info["iterations"] if info["status"] == CONVERGED else np.nan)
    its = np.array(its)
    best = sweep[np.nanargmin(its)]
    print(f"  fine sweep: fewest updates = {np.nanmin(its):.0f} at eta = {best:.4f}"
          f" (theory eta* = {eta_opt:.4f})")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    ax = axes[0]
    for eta, (lh, info) in runs.items():
        gap = np.where(np.isfinite(lh) & (lh < 1e13), lh - j_ls, np.nan)
        ax.semilogy(np.arange(lh.size), np.maximum(gap, 1e-17), color=COLORS[eta],
                    label=f"eta={eta} ({fmt_status(info)}, {info['iterations']} it)")
    ax.set_ylim(1e-17, 1e12)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlim(left=0)
    ax.set_xlabel("iteration")
    ax.set_ylabel("J(theta_k) - J*")
    ax.set_title("Part 1: loss histories (gap to least-squares loss)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.plot(sweep, its, "o", ms=3, color="tab:blue", label="observed updates to converge")
    pred = [predicted_iterations(e, lam, g0) for e in sweep]
    ax.plot(sweep, pred, "k-", lw=1, label="predicted log(tol/|g0|) / log rho(eta)")
    ax.axvline(eta_max, color="tab:red", ls="--", label=f"stability limit 2/lambda_max = {eta_max:.4f}")
    ax.axvline(eta_opt, color="tab:green", ls=":", label=f"eta* = {eta_opt:.4f}")
    for eta in ETAS:
        if eta < 1.08 * eta_max:
            ax.axvline(eta, color=COLORS[eta], lw=0.6, alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("learning rate eta")
    ax.set_ylabel("updates until ||grad|| <= 1e-8")
    ax.set_title("Part 2: stability limit vs speed of convergence")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "exp_d_learning_rates.png", dpi=150)
    plt.close(fig)
    print(f"  saved {OUT / 'exp_d_learning_rates.png'}")


# --------------------------------------------------------------------------- Part 3
def to_original_units(rep, theta, mu=None, sd=None):
    """Map parameters back to y = a + b x so all representations are directly comparable."""
    if rep == "original x":
        return theta
    if rep == "scaled 100x":
        return np.array([theta[0], 100.0 * theta[1]])
    # standardized: y = a + b (100x - mu)/sd
    return np.array([theta[0] - theta[1] * mu / sd, theta[1] * 100.0 / sd])


def part3(x, y, x_test, y_test):
    xs = 100.0 * x
    mu, sd = xs.mean(), xs.std()
    reps = {
        "original x": (x, x_test),
        "scaled 100x": (xs, 100.0 * x_test),
        "standardized": ((xs - mu) / sd, (100.0 * x_test - mu) / sd),
    }
    print("\nD Part 3: feature scaling (targets unchanged; standardization uses TRAINING mean/std)")
    print(f"  training stats of 100x: mean = {mu:.3f}, std = {sd:.3f}")
    print(f"  {'representation':<15}{'lambda_min':>12}{'lambda_max':>13}{'kappa':>12}"
          f"{'eta limit':>12}{'eta*':>12}{'rho*':>10}")
    spec = {}
    for name, (xr, _) in reps.items():
        lam, kappa, eta_max = spectrum(design_matrix(xr))
        spec[name] = (lam, kappa, eta_max)
        eta_opt = 2 / lam.sum()
        print(f"  {name:<15}{lam.min():>12.4g}{lam.max():>13.4g}{kappa:>12.4g}{eta_max:>12.4g}"
              f"{eta_opt:>12.4g}{rho(eta_opt, lam):>10.6f}")

    print(f"\n  {'representation':<15}{'eta':>9}{'status':>11}{'updates':>9}{'|theta-LS|':>12}"
          f"{'intercept':>11}{'slope (x)':>11}{'test MSE':>11}")
    histories = {}
    for name, (xr, xt) in reps.items():
        X, Xt = design_matrix(xr), design_matrix(xt)
        th_ls = ls_solution(X, y)
        lam, kappa, eta_max = spec[name]
        extra = [0.9 * eta_max] if name == "scaled 100x" else []
        for eta in ETAS + extra:
            th, lh, thh, info = gradient_descent(X, y, np.zeros(2), eta, BUDGET, TOL)
            histories[(name, eta)] = (lh, thh, info)
            if info["status"] == NONFINITE:
                print(f"  {name:<15}{eta:>9.3g}{fmt_status(info):>11}{info['iterations']:>9}"
                      f"{'-':>12}{'-':>11}{'-':>11}{'-':>11}")
                continue
            a, b = to_original_units(name, th, mu, sd)
            test = mse(y_test, Xt @ th)
            print(f"  {name:<15}{eta:>9.3g}{fmt_status(info):>11}{info['iterations']:>9}"
                  f"{np.linalg.norm(th - th_ls):>12.3g}{a:>11.5f}{b:>11.5f}{test:>11.5f}")
        th_ls_test = mse(y_test, Xt @ th_ls)
        a, b = to_original_units(name, th_ls, mu, sd)
        print(f"  {name:<15}{'LS ref':>9}{'':>11}{'':>9}{'':>12}{a:>11.5f}{b:>11.5f}{th_ls_test:>11.5f}")

    # Loss histories per representation
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    for ax, name in zip(axes, reps):
        X = design_matrix(reps[name][0])
        j_ls = mse_loss(X, y, ls_solution(X, y))
        etas = ETAS + ([0.9 * spec[name][2]] if name == "scaled 100x" else [])
        for eta in etas:
            lh, _, info = histories[(name, eta)]
            gap = np.where(np.isfinite(lh) & (lh < 1e13), lh - j_ls, np.nan)
            ax.semilogy(np.arange(lh.size), np.maximum(gap, 1e-17), color=COLORS.get(eta, "k"),
                        ls="-" if eta in COLORS else "--",
                        label=f"eta={eta:.3g}: {fmt_status(info)} ({info['iterations']})")
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xlim(left=0)
        ax.set_ylim(1e-17, 1e12)
        ax.set_title(f"{name}: kappa = {spec[name][1]:.3g}")
        ax.set_xlabel("iteration")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("J(theta_k) - J*")
    fig.suptitle("Part 3: same model, three parameterizations, same learning rates")
    fig.tight_layout()
    fig.savefig(OUT / "exp_d_feature_scaling.png", dpi=150)
    plt.close(fig)

    # Paths on each representation's own loss surface
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    pick = {"original x": 0.05, "scaled 100x": 0.9 * spec["scaled 100x"][2], "standardized": 0.1}
    for ax, name in zip(axes, reps):
        xr = reps[name][0]
        X = design_matrix(xr)
        th_ls = ls_solution(X, y)
        lh, thh, info = histories[(name, pick[name])]
        pad0 = max(1.0, abs(th_ls[0])) * 1.3
        pad1 = max(abs(th_ls[1]) * 1.3, 1e-3)
        t0 = np.linspace(-0.3 * pad0, pad0 * 1.2, 250)
        t1 = np.linspace(-0.3 * pad1, pad1 * 1.2, 250)
        T0, T1 = np.meshgrid(t0, t1)
        J = np.mean((T0[..., None] + T1[..., None] * xr - y) ** 2, axis=-1)
        ax.contour(T0, T1, J, levels=np.geomspace(J.min() * 1.001, J.max(), 20), cmap="viridis",
                   linewidths=0.7)
        ax.plot(thh[:, 0], thh[:, 1], ".-", color="tab:red", ms=2, lw=0.8)
        ax.plot(*th_ls, "*", color="gold", mec="k", ms=14)
        ax.set_title(f"{name}, eta={pick[name]:.3g}: {fmt_status(info)} ({info['iterations']})",
                     fontsize=10)
        ax.set_xlabel("theta0")
        ax.set_ylabel("theta1")
        if name == "scaled 100x":
            ax.text(0.02, 0.02, "axes NOT to equal scale: on equal axes the\n"
                    "valley is ~300x longer than it is wide",
                    transform=ax.transAxes, fontsize=8, va="bottom")
        else:
            ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle("Part 3: optimizer paths (star = least-squares solution in that representation)")
    fig.tight_layout()
    fig.savefig(OUT / "exp_d_scaling_paths.png", dpi=150)
    plt.close(fig)
    print(f"  saved {OUT / 'exp_d_feature_scaling.png'}")
    print(f"  saved {OUT / 'exp_d_scaling_paths.png'}")


def main():
    OUT.mkdir(exist_ok=True)
    x, y, x_test, y_test = starter_datasets()
    part1_and_2(x, y)
    part3(x, y, x_test, y_test)


if __name__ == "__main__":
    main()
