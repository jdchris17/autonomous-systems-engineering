"""
Experiment A: regression methods and polynomial modelling.

Truth: y = 1.5 - 0.8 x + 0.35 x^2 + eps,  x ~ U(-5, 5),  eps ~ N(0, 1).
Four fits on ONE shared training set per (N, seed):
    linear     NumPy lstsq on [1, x]
    quadratic  NumPy lstsq on [1, x, x^2]
    quadratic  Module 2 gradient descent on [1, z1, z2], z = standardized (x, x^2)
    quadratic  scikit-learn LinearRegression on [x, x^2]
All parameters are reported as raw coefficients [c0, c1, c2] (the common coordinate system).

Additional challenge: degree up to 15. Conditioning of the raw Vandermonde matrix vs the
standardized one, three solvers (normal equations, lstsq, gradient descent), and generalization.
"""

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from reg_core import (OUT, QUAD_NOISE, QUAD_TRUE, Standardizer, generate_quadratic,
                      gradient_descent, mse, mse_hessian, ols, poly, with_intercept)

SIZES = (20, 50, 200, 1000)
SEEDS = range(50)
N_TEST = 2000
GD_ETA = 0.25          # H = (2/N) Z^T Z with standardized columns has lambda_max ~ 2(1 + |corr|) < 4
GD_TOL = 1e-10
METHODS = ["linear lstsq", "quad lstsq", "quad grad. descent", "quad sklearn"]


def fit_all(x, y):
    """Return raw coefficients [c0, c1, c2] for the four methods, plus GD info."""
    out = {}
    lin = ols(with_intercept(poly(x, 1)), y)
    out["linear lstsq"] = np.array([lin[0], lin[1], 0.0])
    out["quad lstsq"] = ols(with_intercept(poly(x, 2)), y)

    st = Standardizer(poly(x, 2))
    Z = with_intercept(st.transform(poly(x, 2)))
    th, _, _, info = gradient_descent(Z, y, np.zeros(3), GD_ETA, max_iterations=100_000,
                                      tolerance=GD_TOL)
    out["quad grad. descent"] = st.to_raw(th[0], th[1:])

    sk = LinearRegression().fit(poly(x, 2), y)
    out["quad sklearn"] = np.concatenate([[sk.intercept_], sk.coef_])
    return out, info


def predict_raw(c, x):
    return with_intercept(poly(x, len(c) - 1)) @ c


def part_methods():
    test_rng = np.random.default_rng(3000)
    x_te, y_te = generate_quadratic(N_TEST, test_rng)
    print("Experiment A: four estimators on one shared training set (raw coefficients c0, c1, c2)")
    print(f"  truth: c = {QUAD_TRUE},  noise variance = {QUAD_NOISE**2}\n")

    # One representative fit, printed in full
    rng = np.random.default_rng([0, 50])
    x, y = generate_quadratic(50, rng)
    fits, info = fit_all(x, y)
    print("  N = 50, seed 0:")
    print(f"    {'method':<20}{'c0':>10}{'c1':>10}{'c2':>10}{'train MSE':>11}{'test MSE':>10}")
    for m in METHODS:
        c = fits[m]
        print(f"    {m:<20}{c[0]:>10.5f}{c[1]:>10.5f}{c[2]:>10.5f}"
              f"{mse(y, predict_raw(c, x)):>11.4f}{mse(y_te, predict_raw(c, x_te)):>10.4f}")
    print(f"    gradient descent: {info['status']} after {info['iterations']} updates")
    ref = fits["quad lstsq"]
    for m in ("quad grad. descent", "quad sklearn"):
        print(f"    |{m} - quad lstsq| = {np.linalg.norm(fits[m] - ref):.2e}")
    plot_fits(x, y, fits)

    # Repeated over sizes and seeds
    print(f"\n  Repeated: {len(SEEDS)} seeds per N, independent test set of {N_TEST}"
          " (mean values)")
    print(f"    {'N':>5} {'method':<20}{'train MSE':>10}{'test MSE':>10}{'|c - c_true|':>14}"
          f"{'|c - c_lstsq|':>15}")
    summary = {}
    for n in SIZES:
        acc = {m: {"train": [], "test": [], "err": [], "diff": []} for m in METHODS}
        for s in SEEDS:
            x, y = generate_quadratic(n, np.random.default_rng([s, n]))
            fits, info = fit_all(x, y)
            assert info["status"] == "converged", info
            for m in METHODS:
                c = fits[m]
                acc[m]["train"].append(mse(y, predict_raw(c, x)))
                acc[m]["test"].append(mse(y_te, predict_raw(c, x_te)))
                acc[m]["err"].append(np.linalg.norm(c - QUAD_TRUE))
                acc[m]["diff"].append(np.linalg.norm(c - fits["quad lstsq"]))
        summary[n] = acc
        for m in METHODS:
            a = {k: np.mean(v) for k, v in acc[m].items()}
            diff = "-" if m == "linear lstsq" else f"{a['diff']:.1e}"
            print(f"    {n:>5} {m:<20}{a['train']:>10.4f}{a['test']:>10.4f}{a['err']:>14.4f}{diff:>15}")
        print()
    return summary


def plot_fits(x, y, fits):
    xs = np.linspace(-5, 5, 300)
    truth = predict_raw(QUAD_TRUE, xs)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    ax.scatter(x, y, s=18, color="tab:gray", edgecolor="k", lw=0.3, label="training data (N=50)")
    ax.plot(xs, truth, "k", lw=2, label="truth")
    styles = {"linear lstsq": ("tab:red", "-"), "quad lstsq": ("tab:blue", "-"),
              "quad grad. descent": ("tab:orange", "--"), "quad sklearn": ("tab:green", ":")}
    for m, (col, ls) in styles.items():
        ax.plot(xs, predict_raw(fits[m], xs), color=col, ls=ls, lw=2, label=m)
    ax.set_title("Fitted curves (three quadratic fits coincide)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    for ax, m, col in ((axes[1], "linear lstsq", "tab:red"), (axes[2], "quad lstsq", "tab:blue")):
        res = y - predict_raw(fits[m], x)
        ax.scatter(x, res, s=18, color=col, alpha=0.8)
        # binned mean of residuals to expose structure
        bins = np.linspace(-5, 5, 8)
        idx = np.digitize(x, bins) - 1
        centers = 0.5 * (bins[1:] + bins[:-1])
        means = [res[idx == k].mean() if np.any(idx == k) else np.nan for k in range(len(centers))]
        ax.plot(centers, means, "k-o", ms=4, label="binned mean residual")
        if m == "linear lstsq":
            # the part of the truth a line cannot represent
            c = fits[m]
            ax.plot(xs, truth - predict_raw(c, xs), "k:", label="truth - fitted line")
        ax.axhline(0, color="gray", lw=1)
        ax.set_title(f"Residuals: {m}")
        ax.set_xlabel("x")
        ax.set_ylabel("y - y_hat")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "exp_a_fits_residuals.png", dpi=150)
    plt.close(fig)
    print(f"    saved {OUT / 'exp_a_fits_residuals.png'}")


def plot_sizes(summary):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cols = {"linear lstsq": "tab:red", "quad lstsq": "tab:blue"}
    for m, col in cols.items():
        ax.plot(SIZES, [np.mean(summary[n][m]["test"]) for n in SIZES], "o-", color=col,
                label=f"{m}: test")
        ax.plot(SIZES, [np.mean(summary[n][m]["train"]) for n in SIZES], "s--", color=col,
                alpha=0.6, label=f"{m}: train")
    ax.axhline(QUAD_NOISE**2, color="k", ls=":", label="noise variance 1.0")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training size N")
    ax.set_ylabel("mean MSE (50 seeds)")
    ax.set_title("Linear model plateaus at its bias; quadratic reaches the noise floor")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "exp_a_sizes.png", dpi=150)
    plt.close(fig)
    print(f"  saved {OUT / 'exp_a_sizes.png'}")


# --------------------------------------------------------------------------- additional challenge
DEGREES = (1, 2, 3, 5, 8, 10, 12, 15)
N_HI = 50
GD_BUDGET = 20_000
GD_SEEDS = 10          # gradient descent is slow at high degree; run it on the first 10 seeds


def part_high_degree():
    test_rng = np.random.default_rng(3001)
    x_te, y_te = generate_quadratic(N_TEST, test_rng)
    print("Additional challenge: high-degree polynomials (N = 50, 50 seeds; medians;"
          f" gradient descent on {GD_SEEDS} seeds, budget {GD_BUDGET:,}, eta = 1/lambda_max)")
    print("  NE = normal equations solve(X^T X, X^T y) in the raw basis, compared with lstsq in the"
          " standardized basis: max prediction difference and relative coefficient difference")
    print(f"  {'deg':>4}{'cond raw X':>12}{'cond std X':>12}{'cond H(std)':>13}"
          f"{'train MSE':>11}{'test MSE':>11}{'NE pred diff':>13}{'NE coef rel':>12}{'GD status':>16}"
          f"{'GD train gap':>14}")
    rows = []
    for d in DEGREES:
        conds_raw, conds_std, conds_h, tr, te, ne_diff, ne_coef, gd_stat, gd_gap = ([] for _ in range(9))
        for s in SEEDS:
            x, y = generate_quadratic(N_HI, np.random.default_rng([s, 7]))
            Xr = with_intercept(poly(x, d))
            st = Standardizer(poly(x, d))
            Z = with_intercept(st.transform(poly(x, d)))
            conds_raw.append(np.linalg.cond(Xr))
            conds_std.append(np.linalg.cond(Z))
            conds_h.append(np.linalg.cond(mse_hessian(Z)))

            c_ls = ols(Xr, y)                                  # SVD-based lstsq, raw basis
            th_std = ols(Z, y)                                 # lstsq, standardized basis
            c_std = st.to_raw(th_std[0], th_std[1:])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    c_ne = np.linalg.solve(Xr.T @ Xr, Xr.T @ y)  # normal equations, raw basis
                except np.linalg.LinAlgError:
                    c_ne = np.full(d + 1, np.nan)
            pred_std = Z @ th_std
            tr.append(mse(y, pred_std))
            te.append(mse(y_te, with_intercept(st.transform(poly(x_te, d))) @ th_std))
            # disagreement between solvers, measured on predictions over the training range
            ne_diff.append(np.max(np.abs(Xr @ c_ne - pred_std)))
            ne_coef.append(np.linalg.norm(c_ne - c_std) / np.linalg.norm(c_std))

            if s >= GD_SEEDS:
                continue

            H = mse_hessian(Z)
            eta = 1.0 / np.linalg.eigvalsh(H).max()
            th_gd, lh, _, info = gradient_descent(Z, y, np.zeros(d + 1), eta, GD_BUDGET, 1e-8)
            gd_stat.append(info["status"])
            gd_gap.append(lh[-1] - mse(y, pred_std))
        conv = np.mean([g == "converged" for g in gd_stat])
        rows.append((d, np.median(tr), np.median(te), np.median(conds_h)))
        print(f"  {d:>4}{np.median(conds_raw):>12.2e}{np.median(conds_std):>12.2e}"
              f"{np.median(conds_h):>13.2e}{np.median(tr):>11.4f}{np.median(te):>11.4g}"
              f"{np.median(ne_diff):>13.2e}{np.median(ne_coef):>12.2e}{f'{conv:.0%} converged':>16}"
              f"{np.median(gd_gap):>14.2e}")

    # Visual for one seed
    x, y = generate_quadratic(N_HI, np.random.default_rng([0, 7]))
    xs = np.linspace(-5, 5, 400)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    ax = axes[0]
    ax.scatter(x, y, s=16, color="tab:gray", edgecolor="k", lw=0.3, zorder=3)
    ax.plot(xs, predict_raw(QUAD_TRUE, xs), "k", lw=2, label="truth")
    for d, col in ((2, "tab:blue"), (5, "tab:green"), (10, "tab:orange"), (15, "tab:red")):
        st = Standardizer(poly(x, d))
        th = ols(with_intercept(st.transform(poly(x, d))), y)
        ax.plot(xs, with_intercept(st.transform(poly(xs, d))) @ th, color=col, lw=1.5,
                label=f"degree {d}")
    ax.set_ylim(-5, 20)
    ax.set_title("High-degree fits chase the noise (N = 50, seed 0)")
    ax.set_xlabel("x")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ds = [r[0] for r in rows]
    ax.semilogy(ds, [r[1] for r in rows], "s--", label="train MSE (median)")
    ax.semilogy(ds, [r[2] for r in rows], "o-", label="test MSE (median)")
    ax.axhline(QUAD_NOISE**2, color="k", ls=":", label="noise variance")
    ax2 = ax.twinx()
    ax2.semilogy(ds, [r[3] for r in rows], "^-", color="tab:purple", alpha=0.6,
                 label="cond(H), standardized")
    ax2.set_ylabel("condition number of H", color="tab:purple")
    ax.set_xlabel("polynomial degree")
    ax.set_ylabel("MSE")
    ax.set_title("Generalization and conditioning vs degree")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "exp_a_high_degree.png", dpi=150)
    plt.close(fig)
    print(f"  saved {OUT / 'exp_a_high_degree.png'}")


def main():
    OUT.mkdir(exist_ok=True)
    summary = part_methods()
    plot_sizes(summary)
    print()
    part_high_degree()


if __name__ == "__main__":
    main()
