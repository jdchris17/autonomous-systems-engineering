"""
Experiment B: weighted least squares for sensor calibration.

Truth:   delta_u(r) = 0.15 + 0.4 r^2 px,  r ~ U(0, 1)
Noise:   eps ~ N(0, sigma(I)^2),  sigma(I) = 0.05 + 0.8 / sqrt(I),  I ~ U(1, 100)
Model:   delta_u_hat = a + b r^2        (the correct features 1 and r^2)

Estimator 1  OLS: argmin sum (y_i - X_i theta)^2
Estimator 2  WLS: argmin sum w_i (y_i - X_i theta)^2,  w_i = 1 / sigma_i^2

Theory (correct sigma): Cov(theta_WLS) = (X^T W X)^-1, the Gauss-Markov minimum. OLS is still
unbiased, with sandwich covariance (X^T X)^-1 X^T S X (X^T X)^-1, S = diag(sigma_i^2).

Engineering extension: the reported sigma is wrong. The data are always generated with the TRUE
sigma; only the weights (and the formal uncertainty WLS reports) use the reported values.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reg_core import (CAL_TRUE, OUT, calibration_truth, generate_calibration_data, mse, ols,
                      rmse, wls)

N_TRAIN = 200
N_TEST = 5000
N_SEEDS = 2000


def design(r):
    return np.column_stack([np.ones_like(r), r**2])


def fit_both(d, sigma_used=None):
    X = design(d["r"])
    y = d["observed_bias"]
    s = d["sigma"] if sigma_used is None else sigma_used
    return ols(X, y), wls(X, y, 1.0 / s**2)


def scores(theta, test):
    X = design(test["r"])
    pred = X @ theta
    return {"mse_noisy": mse(test["observed_bias"], pred),
            "rmse_truth": rmse(test["true_bias"], pred),
            "param_err": float(np.linalg.norm(theta - CAL_TRUE))}


def theory_cov(X, sigma_true, sigma_rep):
    """Actual covariance of OLS and WLS (weights from sigma_rep), and WLS's formal covariance."""
    S = np.diag(sigma_true**2)
    A = np.linalg.inv(X.T @ X)
    cov_ols = A @ X.T @ S @ X @ A
    W = np.diag(1.0 / sigma_rep**2)
    B = np.linalg.inv(X.T @ W @ X)
    cov_wls_actual = B @ X.T @ W @ S @ W @ X @ B
    return cov_ols, cov_wls_actual, B


def main_comparison(test):
    print(f"Experiment B: OLS vs WLS, N_train = {N_TRAIN}, test N = {N_TEST}, {N_SEEDS} seeds")
    res = {"OLS": [], "WLS": []}
    thetas = {"OLS": [], "WLS": []}
    for s in range(N_SEEDS):
        d = generate_calibration_data(N_TRAIN, np.random.default_rng([s, 11]))
        th_o, th_w = fit_both(d)
        thetas["OLS"].append(th_o)
        thetas["WLS"].append(th_w)
        res["OLS"].append(scores(th_o, test))
        res["WLS"].append(scores(th_w, test))
    noise_floor = np.mean(test["sigma"] ** 2)
    realized = mse(test["observed_bias"], test["true_bias"])
    print(f"  noisy-test MSE of the PERFECT calibration: expected {noise_floor:.5f},"
          f" realized on this test set {realized:.5f}")
    print(f"  {'':<5}{'test MSE (noisy)':>18}{'RMSE vs truth [px]':>20}{'|theta - true|':>16}"
          f"{'mean a':>9}{'sd a':>8}{'mean b':>9}{'sd b':>8}")
    for m in ("OLS", "WLS"):
        a = {k: np.array([r[k] for r in res[m]]) for k in res[m][0]}
        th = np.array(thetas[m])
        print(f"  {m:<5}{a['mse_noisy'].mean():>18.5f}{a['rmse_truth'].mean():>20.5f}"
              f"{a['param_err'].mean():>16.5f}{th[:, 0].mean():>9.4f}{th[:, 0].std():>8.4f}"
              f"{th[:, 1].mean():>9.4f}{th[:, 1].std():>8.4f}")
    win = np.mean([w["rmse_truth"] < o["rmse_truth"] for o, w in zip(res["OLS"], res["WLS"])])
    ratio = (np.mean([o["rmse_truth"] ** 2 for o in res["OLS"]])
             / np.mean([w["rmse_truth"] ** 2 for w in res["WLS"]]))
    print(f"  WLS has the smaller calibration error in {win:.1%} of trials"
          f" (so OLS wins {1 - win:.1%} of the time)")
    print(f"  mean squared calibration error ratio OLS / WLS = {ratio:.2f}"
          " (relative efficiency of WLS)")

    # Theory check on one representative design
    d = generate_calibration_data(N_TRAIN, np.random.default_rng([0, 11]))
    c_o, c_w, _ = theory_cov(design(d["r"]), d["sigma"], d["sigma"])
    print(f"  theory for seed-0 design: sd(b) OLS = {np.sqrt(c_o[1, 1]):.4f},"
          f" WLS = {np.sqrt(c_w[1, 1]):.4f}  (empirical sds above average over designs)")
    return res, thetas


def plot_main(test, res):
    d = generate_calibration_data(N_TRAIN, np.random.default_rng([0, 11]))
    th_o, th_w = fit_both(d)
    rs = np.linspace(0, 1, 200)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    ax = axes[0]
    sc = ax.scatter(d["r"], d["observed_bias"], c=d["sigma"], cmap="viridis_r", s=14, zorder=3)
    ax.errorbar(d["r"], d["observed_bias"], yerr=d["sigma"], fmt="none", ecolor="gray",
                alpha=0.3, lw=0.6)
    plt.colorbar(sc, ax=ax, label="reported sigma [px] (dark = bright source)")
    ax.plot(rs, calibration_truth(rs), "k", lw=2.5, label="true bias 0.15 + 0.4 r^2")
    ax.plot(rs, design(rs) @ th_o, "tab:red", lw=2, ls="--",
            label=f"OLS: {th_o[0]:.3f} + {th_o[1]:.3f} r^2")
    ax.plot(rs, design(rs) @ th_w, "tab:blue", lw=2, label=f"WLS: {th_w[0]:.3f} + {th_w[1]:.3f} r^2")
    ax.set_xlabel("normalized radial position r")
    ax.set_ylabel("calibration residual delta_u [px]")
    ax.set_title(f"One training set (N = {N_TRAIN}, seed 0)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    o = np.array([r["rmse_truth"] for r in res["OLS"]])
    w = np.array([r["rmse_truth"] for r in res["WLS"]])
    ax = axes[1]
    bins = np.linspace(0, max(o.max(), w.max()), 60)
    ax.hist(o, bins=bins, alpha=0.55, color="tab:red", label=f"OLS (mean {o.mean():.4f})")
    ax.hist(w, bins=bins, alpha=0.55, color="tab:blue", label=f"WLS (mean {w.mean():.4f})")
    ax.set_xlabel("RMSE vs noise-free calibration [px]")
    ax.set_ylabel("trials")
    ax.set_title(f"Calibration error over {N_SEEDS} seeds")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2]
    lim = max(o.max(), w.max())
    ax.scatter(o, w, s=4, alpha=0.35)
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlabel("OLS calibration RMSE [px]")
    ax.set_ylabel("WLS calibration RMSE [px]")
    ax.set_title(f"Per trial: WLS better below the line ({np.mean(w < o):.0%})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "exp_b_ols_vs_wls.png", dpi=150)
    plt.close(fig)
    print(f"  saved {OUT / 'exp_b_ols_vs_wls.png'}")


# --------------------------------------------------------------------------- engineering extension
FAINT = 20.0
SCENARIOS = [
    ("correct sigma", lambda s, I: s),
    ("all sigma x0.33 (uniformly overconfident)", lambda s, I: s / 3.0),
    ("faint (I<20) sigma / 2", lambda s, I: np.where(I < FAINT, s / 2.0, s)),
    ("faint (I<20) sigma / 5", lambda s, I: np.where(I < FAINT, s / 5.0, s)),
    ("faint (I<20) sigma / 20", lambda s, I: np.where(I < FAINT, s / 20.0, s)),
    ("constant sigma (= OLS)", lambda s, I: np.full_like(s, s.mean())),
]


def extension(test):
    print("\nEngineering extension: WLS with wrong reported uncertainties"
          f" ({N_SEEDS // 2} seeds; data always use the TRUE sigma)")
    print(f"  {'weights from':<42}{'RMSE vs truth':>14}{'actual sd(b)':>13}{'reported sd(b)':>15}"
          f"{'chi2/dof':>10}{'95% CI cover b':>16}")
    out = []
    for name, rep_fn in SCENARIOS:
        rm, bs, sd_rep, chi, cover = [], [], [], [], []
        for s in range(N_SEEDS // 2):
            d = generate_calibration_data(N_TRAIN, np.random.default_rng([s, 11]))
            X, y = design(d["r"]), d["observed_bias"]
            s_rep = rep_fn(d["sigma"], d["intensity"])
            th = wls(X, y, 1.0 / s_rep**2)
            formal = np.linalg.inv(X.T @ (X / s_rep[:, None] ** 2))
            rm.append(scores(th, test)["rmse_truth"])
            bs.append(th[1])
            sd_rep.append(np.sqrt(formal[1, 1]))
            z = (y - X @ th) / s_rep
            chi.append(z @ z / (N_TRAIN - 2))
            cover.append(abs(th[1] - CAL_TRUE[1]) <= 1.96 * np.sqrt(formal[1, 1]))
        out.append((name, np.mean(rm), np.std(bs), np.mean(sd_rep), np.mean(chi), np.mean(cover)))
        print(f"  {name:<42}{np.mean(rm):>14.5f}{np.std(bs):>13.4f}{np.mean(sd_rep):>15.4f}"
              f"{np.mean(chi):>10.2f}{np.mean(cover):>16.1%}")

    fig, ax = plt.subplots(figsize=(9, 4.6))
    names = [o[0] for o in out]
    ypos = np.arange(len(names))
    ax.barh(ypos - 0.2, [o[2] for o in out], 0.4, color="tab:red", label="actual sd of slope b")
    ax.barh(ypos + 0.2, [o[3] for o in out], 0.4, color="tab:blue", label="sd WLS claims")
    ax.set_yticks(ypos, names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("standard deviation of fitted slope b [px]")
    ax.set_title("Overconfident uncertainties: worse estimates, and a fit that claims to be better")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(OUT / "exp_b_wrong_sigma.png", dpi=150)
    plt.close(fig)
    print(f"  saved {OUT / 'exp_b_wrong_sigma.png'}")


def main():
    OUT.mkdir(exist_ok=True)
    test = generate_calibration_data(N_TEST, np.random.default_rng(999))
    res, _ = main_comparison(test)
    plot_main(test, res)
    extension(test)


if __name__ == "__main__":
    main()
