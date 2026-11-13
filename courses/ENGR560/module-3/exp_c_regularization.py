"""
Experiment C: ridge and lasso regularization.

Truth: y = 1 + 0.8 x - 0.5 x^2 + 0.12 x^3 + eps,  x ~ U(-3, 3),  eps ~ N(0, 1)
Splits per seed: 30 training, 30 validation, and one shared 2000-point test set.
Candidate degrees d in {1, 2, 3, 5, 10, 15}. Features [x, ..., x^d] standardized with TRAINING
mean/std; the intercept is never penalized.

  OLS    alpha = 0
  Ridge  (1/n)||y - b - Z beta||^2 + alpha ||beta||^2          implemented in reg_core.ridge
  Lasso  (1/(2n))||y - b - Z beta||^2 + alpha ||beta||_1        scikit-learn Lasso

Model selection: degree and alpha are chosen on the VALIDATION set; the test set is used only
once, for the final comparison of the selected models.
"""

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso

from reg_core import OUT, Standardizer, mse, poly, ridge

TRUE_C = np.array([1.0, 0.8, -0.5, 0.12])
NOISE = 1.0
N_TRAIN, N_VAL, N_TEST = 30, 30, 2000
DEGREES = (1, 2, 3, 5, 10, 15)
RIDGE_ALPHAS = np.logspace(-8, 2, 21)
LASSO_ALPHAS = np.logspace(-4, 0.5, 19)
SEEDS = range(100)
SHOW_DEG = 15
WARN_WHERE = []


def gen(n, rng):
    x = rng.uniform(-3, 3, n)
    y = TRUE_C[0] + TRUE_C[1] * x + TRUE_C[2] * x**2 + TRUE_C[3] * x**3 + rng.normal(0, NOISE, n)
    return x, y


def truth(x):
    return TRUE_C[0] + TRUE_C[1] * x + TRUE_C[2] * x**2 + TRUE_C[3] * x**3


class Split:
    """Standardized features for one degree, fitted on training data only."""

    def __init__(self, d, xtr, xva, xte):
        self.st = Standardizer(poly(xtr, d))
        self.Ztr = self.st.transform(poly(xtr, d))
        self.Zva = self.st.transform(poly(xva, d))
        self.Zte = self.st.transform(poly(xte, d))


def fit_ridge(sp, y, alpha):
    b, beta = ridge(sp.Ztr, y, alpha)
    return b, beta


def fit_lasso(sp, y, alpha, quiet=False):
    with warnings.catch_warnings():
        if quiet:
            warnings.simplefilter("ignore", ConvergenceWarning)
        m = Lasso(alpha=alpha, max_iter=100_000, tol=1e-7).fit(sp.Ztr, y)
    return m.intercept_, m.coef_


def errors(sp, b, beta, ytr, yva, yte):
    return (mse(ytr, b + sp.Ztr @ beta), mse(yva, b + sp.Zva @ beta), mse(yte, b + sp.Zte @ beta))


def run_seed(seed, xte, yte):
    rng = np.random.default_rng([seed, 21])
    xtr, ytr = gen(N_TRAIN, rng)
    xva, yva = gen(N_VAL, rng)
    res = {"ols": {}, "ridge": {}, "lasso": {}}
    n_warn = 0
    for d in DEGREES:
        sp = Split(d, xtr, xva, xte)
        res["ols"][d] = errors(sp, *fit_ridge(sp, ytr, 0.0), ytr, yva, yte)
        for a in RIDGE_ALPHAS:
            res["ridge"][(d, a)] = errors(sp, *fit_ridge(sp, ytr, a), ytr, yva, yte)
        for a in LASSO_ALPHAS:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)
                b, beta = fit_lasso(sp, ytr, a)
                k = sum(issubclass(x.category, ConvergenceWarning) for x in w)
                n_warn += k
                if k:
                    WARN_WHERE.append((d, a))
            res["lasso"][(d, a)] = errors(sp, b, beta, ytr, yva, yte)
    return res, n_warn, (xtr, ytr, xva, yva)


def select(res):
    """Best key by validation MSE for each family; returns (key, test MSE)."""
    out = {}
    for fam in ("ols", "ridge", "lasso"):
        key = min(res[fam], key=lambda k: res[fam][k][1])
        out[fam] = (key, res[fam][key][2])
    return out


def main():
    OUT.mkdir(exist_ok=True)
    xte, yte = gen(N_TEST, np.random.default_rng(4242))
    floor = mse(yte, truth(xte))
    all_res, sel, warn_total = [], [], 0
    for s in SEEDS:
        res, nw, _ = run_seed(s, xte, yte)
        all_res.append(res)
        sel.append(select(res))
        warn_total += nw

    print(f"Experiment C: ridge and lasso. N_train = {N_TRAIN}, N_val = {N_VAL}, N_test = {N_TEST},"
          f" {len(SEEDS)} seeds")
    print(f"  true model is cubic; test MSE of the TRUE function = {floor:.4f} (noise floor)")
    print(f"  lasso convergence warnings: {warn_total} of"
          f" {len(SEEDS) * len(DEGREES) * len(LASSO_ALPHAS)} fits"
          + (f"; all at degree >= {min(d for d, _ in WARN_WHERE)} with alpha <="
             f" {max(a for _, a in WARN_WHERE):.1e}" if WARN_WHERE else ""))

    print("\n  Unregularized OLS by degree (median over seeds):")
    print(f"    {'degree':>6}{'train MSE':>11}{'val MSE':>11}{'test MSE':>11}")
    for d in DEGREES:
        e = np.array([r["ols"][d] for r in all_res])
        print(f"    {d:>6}{np.median(e[:, 0]):>11.4f}{np.median(e[:, 1]):>11.4f}"
              f"{np.median(e[:, 2]):>11.4f}")

    print(f"\n  Degree {SHOW_DEG}: test MSE vs regularization strength (median over seeds)")
    for fam, grid in (("ridge", RIDGE_ALPHAS), ("lasso", LASSO_ALPHAS)):
        line = ", ".join(f"{a:.0e}: {np.median([r[fam][(SHOW_DEG, a)][2] for r in all_res]):.3f}"
                         for a in grid[::3])
        print(f"    {fam}: {line}")

    print("\n  Final comparison: degree (and alpha) chosen on VALIDATION, scored once on TEST")
    print(f"    {'model':<34}{'mean test':>10}{'median':>9}{'90th pct':>10}   selected degrees")
    rows = [("OLS, degree 15 (no selection)", [r["ols"][15][2] for r in all_res], None),
            ("OLS, degree 3 (oracle: true form)", [r["ols"][3][2] for r in all_res], None)]
    for fam, label in (("ols", "OLS, degree chosen on val"), ("ridge", "Ridge, degree+alpha on val"),
                       ("lasso", "Lasso, degree+alpha on val")):
        degs = [s[fam][0] if fam == "ols" else s[fam][0][0] for s in sel]
        rows.append((label, [s[fam][1] for s in sel], degs))
    for label, v, degs in rows:
        v = np.array(v)
        dtxt = ""
        if degs is not None:
            u, c = np.unique(degs, return_counts=True)
            dtxt = ", ".join(f"d{a}:{b}" for a, b in zip(u, c))
        print(f"    {label:<34}{v.mean():>10.4g}{np.median(v):>9.4g}{np.percentile(v, 90):>10.4g}   {dtxt}")
    ridge_beats = np.mean([s["ridge"][1] < s["ols"][1] for s in sel])
    print(f"    validated ridge beats validated OLS on test in {ridge_beats:.0%} of seeds")

    plot_three_panel(all_res, xte, yte)


def plot_three_panel(all_res, xte, yte):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    floor = mse(yte, truth(xte))

    ax = axes[0]
    e = np.array([[r["ols"][d] for d in DEGREES] for r in all_res])       # seeds x deg x 3
    for k, (lab, st, col) in enumerate((("train", "s--", "tab:blue"), ("validation", "^:", "tab:green"),
                                        ("test", "o-", "tab:red"))):
        med = np.median(e[:, :, k], axis=0)
        q1, q3 = np.percentile(e[:, :, k], [25, 75], axis=0)
        ax.plot(DEGREES, med, st, color=col, label=f"{lab} (median)")
        ax.fill_between(DEGREES, q1, q3, color=col, alpha=0.15)
    ax.axhline(floor, color="k", ls=":", label="noise floor")
    ax.set_yscale("log")
    ax.set_xlabel("polynomial degree")
    ax.set_ylabel("MSE")
    ax.set_title("(1) OLS: training vs test MSE across degree")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    for fam, grid, col in (("ridge", RIDGE_ALPHAS, "tab:blue"), ("lasso", LASSO_ALPHAS, "tab:orange")):
        for d, ls in ((SHOW_DEG, "-"), (5, "--")):
            v = np.array([[r[fam][(d, a)][2] for a in grid] for r in all_res])
            ax.plot(grid, np.median(v, axis=0), ls, color=col, marker="o", ms=3,
                    label=f"{fam}, degree {d}")
    ax.axhline(np.median([r["ols"][SHOW_DEG][2] for r in all_res]), color="gray", ls="-.",
               label=f"OLS degree {SHOW_DEG}")
    ax.axhline(np.median([r["ols"][3][2] for r in all_res]), color="tab:green", ls=":",
               label="OLS degree 3 (true form)")
    ax.axhline(floor, color="k", ls=":", label="noise floor")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("regularization strength alpha (ridge and lasso scales differ)")
    ax.set_ylabel("median test MSE")
    ax.set_title("(2) Regularization strength vs test error")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    # (3) coefficient magnitudes along the ridge path, degree 15, seed 0
    _, _, (xtr, ytr, xva, yva) = run_seed(0, xte[:10], yte[:10])
    sp = Split(SHOW_DEG, xtr, xva, xte[:10])
    B = np.array([fit_ridge(sp, ytr, a)[1] for a in RIDGE_ALPHAS])
    L = np.array([fit_lasso(sp, ytr, a, quiet=True)[1] for a in LASSO_ALPHAS])
    ax = axes[2]
    cmap = plt.get_cmap("viridis")
    for j in range(SHOW_DEG):
        ax.plot(RIDGE_ALPHAS, np.abs(B[:, j]) + 1e-12, color=cmap(j / SHOW_DEG), lw=1)
    ax.plot(RIDGE_ALPHAS, np.linalg.norm(B, axis=1), "k", lw=2.5, label="ridge ||beta||_2")
    ax.plot(LASSO_ALPHAS, np.abs(L).sum(axis=1) + 1e-12, "--", color="tab:orange", lw=2.5,
            label="lasso ||beta||_1")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, None)
    ax.set_xlabel("alpha")
    ax.set_ylabel("|standardized coefficient|")
    ax.set_title(f"(3) Coefficient magnitudes, degree {SHOW_DEG} (seed 0;\n thin = ridge |beta_j|,"
                 " colour = power j)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "exp_c_regularization.png", dpi=150)
    plt.close(fig)

    # extra: lasso sparsity path and fitted curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    ax = axes[0]
    for j in range(SHOW_DEG):
        ax.plot(LASSO_ALPHAS, L[:, j], color=cmap(j / SHOW_DEG), lw=1.2)
    ax2 = ax.twinx()
    ax2.step(LASSO_ALPHAS, (np.abs(L) > 1e-10).sum(axis=1), "k--", where="mid", lw=1)
    ax2.set_ylabel("non-zero coefficients (dashed)")
    ax.set_xscale("log")
    ax.set_xlabel("lasso alpha")
    ax.set_ylabel("standardized coefficient")
    ax.set_title(f"Lasso path, degree {SHOW_DEG}: coefficients hit exactly zero")
    ax.grid(alpha=0.3)

    ax = axes[1]
    xs = np.linspace(-3, 3, 300)
    sps = Split(SHOW_DEG, xtr, xva, xs)
    ax.scatter(xtr, ytr, s=18, color="tab:gray", edgecolor="k", lw=0.3, label="training (30)", zorder=3)
    ax.plot(xs, truth(xs), "k", lw=2.5, label="truth (cubic)")
    b, beta = fit_ridge(sps, ytr, 0.0)
    ax.plot(xs, b + sps.Zte @ beta, color="tab:red", lw=1.2, label=f"OLS degree {SHOW_DEG}")
    for a, col in ((1e-3, "tab:blue"), (1e-1, "tab:cyan")):
        b, beta = fit_ridge(sps, ytr, a)
        ax.plot(xs, b + sps.Zte @ beta, color=col, lw=1.8, label=f"ridge alpha={a:g}")
    b, beta = fit_lasso(sps, ytr, 0.05, quiet=True)
    ax.plot(xs, b + sps.Zte @ beta, "--", color="tab:orange", lw=1.8, label="lasso alpha=0.05")
    ax.set_ylim(-12, 8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Degree-{SHOW_DEG} fits, seed 0")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "exp_c_lasso_path_and_fits.png", dpi=150)
    plt.close(fig)
    print(f"\n  saved {OUT / 'exp_c_regularization.png'}")
    print(f"  saved {OUT / 'exp_c_lasso_path_and_fits.png'}")


if __name__ == "__main__":
    main()
