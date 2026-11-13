"""
Experiment D: robust regression and outlier rejection.

Calibration data from Experiment B (model a + b r^2, heteroscedastic noise with KNOWN sigma_i).
A fraction of TRAINING observations gets an extra offset ~ N(1.0, 2.0^2) px (reg_core.corrupt).
Their reported sigma is unchanged - the corruption is unrecognized.

Estimators (all fit a + b r^2):
  OLS     minimize sum r_i^2
  WLS     minimize sum (r_i / sigma_i)^2
  Huber   scikit-learn HuberRegressor on the WHITENED problem (rows divided by sigma_i), so it is a
          robust version of WLS: quadratic for |r_i/sigma_i| <= 1.35 s, linear beyond (s is the
          scale HuberRegressor estimates jointly)
  RANSAC  scikit-learn RANSACRegressor on the whitened problem: random 2-point models, consensus
          set = points with |r_i| < 3 sigma_i, final WLS fit on the largest consensus set

Evaluation uses a clean, independent set: RMSE against the noise-free calibration function.
"""

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import HuberRegressor, LinearRegression, RANSACRegressor

from reg_core import (CAL_TRUE, OUT, calibration_truth, corrupt, generate_calibration_data, mse,
                      ols, rmse, wls)

N_TRAIN = 200
FRACTIONS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40)
SEEDS = range(200)
METHODS = ("OLS", "WLS", "Huber", "RANSAC")
COLORS = {"OLS": "tab:red", "WLS": "tab:purple", "Huber": "tab:green", "RANSAC": "tab:blue"}
RANSAC_THRESHOLD = 3.0     # in units of the reported sigma


def design(r):
    return np.column_stack([np.ones_like(r), r**2])


def fit_all(d, seed):
    X, y, s = design(d["r"]), d["observed_bias"], d["sigma"]
    Xw, yw = X / s[:, None], y / s
    out = {"OLS": ols(X, y), "WLS": wls(X, y, 1 / s**2)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hub = HuberRegressor(epsilon=1.35, alpha=0.0, fit_intercept=False, max_iter=2000).fit(Xw, yw)
    out["Huber"] = hub.coef_
    ran = RANSACRegressor(estimator=LinearRegression(fit_intercept=False), min_samples=2,
                          residual_threshold=RANSAC_THRESHOLD, max_trials=500,
                          random_state=seed).fit(Xw, yw)
    out["RANSAC"] = ran.estimator_.coef_
    return out, ran.inlier_mask_


def detection(inlier_mask, outlier):
    """Treat 'flagged as outlier by RANSAC' as the positive class."""
    flagged = ~inlier_mask
    tp = np.sum(flagged & outlier)
    fp = np.sum(flagged & ~outlier)
    fn = np.sum(~flagged & outlier)
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    false_reject = fp / max(np.sum(~outlier), 1)
    return precision, recall, false_reject


def main():
    OUT.mkdir(exist_ok=True)
    r_eval = np.random.default_rng(77).uniform(0, 1, 5000)
    truth_eval = calibration_truth(r_eval)
    Xe = design(r_eval)

    stats = {f: {m: {"rmse": [], "perr": [], "train": []} for m in METHODS} for f in FRACTIONS}
    det = {f: [] for f in FRACTIONS}
    hidden = {f: [] for f in FRACTIONS}
    for f in FRACTIONS:
        for s in SEEDS:
            rng = np.random.default_rng([s, 31])
            d = corrupt(generate_calibration_data(N_TRAIN, rng), f, rng)
            fits, inl = fit_all(d, s)
            for m in METHODS:
                th = fits[m]
                stats[f][m]["rmse"].append(rmse(truth_eval, Xe @ th))
                stats[f][m]["perr"].append(np.linalg.norm(th - CAL_TRUE))
                stats[f][m]["train"].append(mse(d["observed_bias"], design(d["r"]) @ th))
            if f > 0:
                det[f].append(detection(inl, d["outlier"]))
                gross = d["observed_bias"] - d["true_bias"]
                hidden[f].append(np.mean(np.abs(gross[d["outlier"]]) < RANSAC_THRESHOLD
                                         * d["sigma"][d["outlier"]]))

    print(f"Experiment D: robust regression, N_train = {N_TRAIN}, {len(SEEDS)} seeds per fraction")
    print(f"  corruption: extra offset ~ N({1.0}, {2.0}^2) px on a random fraction of training points")
    print("  RMSE vs noise-free calibration on a clean evaluation set [px] (median, [25th-75th pct])")
    hdr = f"  {'fraction':>8}" + "".join(f"{m:>22}" for m in METHODS)
    print(hdr)
    for f in FRACTIONS:
        cells = []
        for m in METHODS:
            v = np.array(stats[f][m]["rmse"])
            q1, q2, q3 = np.percentile(v, [25, 50, 75])
            cells.append(f"{q2:.4f} [{q1:.3f}-{q3:.3f}]")
        print(f"  {f:>8.0%}" + "".join(f"{c:>22}" for c in cells))

    print("\n  Parameter error |theta - theta_true| (median)    and    training MSE on the corrupted"
          " training data (median)")
    print(f"  {'fraction':>8}" + "".join(f"{m:>10}" for m in METHODS) + "   |"
          + "".join(f"{m:>10}" for m in METHODS))
    for f in FRACTIONS:
        pe = "".join(f"{np.median(stats[f][m]['perr']):>10.4f}" for m in METHODS)
        tr = "".join(f"{np.median(stats[f][m]['train']):>10.4f}" for m in METHODS)
        print(f"  {f:>8.0%}{pe}   |{tr}")

    print("\n  RANSAC outlier detection (positive = flagged as outlier; mean over seeds)")
    print(f"  {'fraction':>8}{'precision':>11}{'recall':>9}{'false-reject rate':>19}"
          f"{'corrupted but within 3 sigma':>30}")
    for f in FRACTIONS[1:]:
        a = np.array(det[f], dtype=float)
        print(f"  {f:>8.0%}{np.nanmean(a[:, 0]):>11.3f}{np.nanmean(a[:, 1]):>9.3f}"
              f"{np.nanmean(a[:, 2]):>19.3f}{np.mean(hidden[f]):>30.1%}")

    plot(stats, det)


def plot(stats, det):
    # Fitted curves for one corrupted dataset
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    rs = np.linspace(0, 1, 200)
    for ax, f in zip(axes[:2], (0.10, 0.30)):
        rng = np.random.default_rng([3, 31])
        d = corrupt(generate_calibration_data(N_TRAIN, rng), f, rng)
        fits, inl = fit_all(d, 3)
        ok, bad = ~d["outlier"], d["outlier"]
        ax.scatter(d["r"][ok], d["observed_bias"][ok], s=10, color="tab:gray", alpha=0.7,
                   label="clean measurements")
        ax.scatter(d["r"][bad], d["observed_bias"][bad], s=26, marker="x", color="k",
                   label="corrupted (label known only to us)")
        ax.scatter(d["r"][~inl], d["observed_bias"][~inl], s=70, facecolor="none",
                   edgecolor="tab:blue", lw=0.8, label="rejected by RANSAC")
        ax.plot(rs, calibration_truth(rs), "k", lw=3, label="true calibration")
        for m in METHODS:
            ax.plot(rs, design(rs) @ fits[m], color=COLORS[m], lw=1.8,
                    ls="--" if m in ("OLS", "WLS") else "-", label=m)
        ax.set_ylim(-1.5, 3.5)
        ax.set_xlabel("r")
        ax.set_ylabel("delta_u [px]")
        ax.set_title(f"{f:.0%} of training points corrupted (seed 3)")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)

    ax = axes[2]
    for m in METHODS:
        v = np.array([stats[f][m]["rmse"] for f in FRACTIONS])
        q1, q2, q3 = np.percentile(v, [25, 50, 75], axis=1)
        ax.plot(FRACTIONS, q2, "o-", color=COLORS[m], label=m)
        ax.fill_between(FRACTIONS, q1, q3, color=COLORS[m], alpha=0.15)
    ax.set_yscale("log")
    ax.set_xlabel("fraction of corrupted training samples")
    ax.set_ylabel("RMSE vs true calibration [px]")
    ax.set_title("Calibration error vs contamination (median, IQR)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "exp_d_robust.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.3))
    fr = FRACTIONS[1:]
    a = np.array([np.nanmean(np.array(det[f], dtype=float), axis=0) for f in fr])
    ax.plot(fr, a[:, 0], "o-", label="precision (flagged points that were corrupted)")
    ax.plot(fr, a[:, 1], "s-", label="recall (corrupted points that were flagged)")
    ax.plot(fr, a[:, 2], "^-", label="false-reject rate (clean points flagged)")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("fraction of corrupted training samples")
    ax.set_title("RANSAC inlier/outlier detection vs known labels")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "exp_d_ransac_detection.png", dpi=150)
    plt.close(fig)
    print(f"\n  saved {OUT / 'exp_d_robust.png'}")
    print(f"  saved {OUT / 'exp_d_ransac_detection.png'}")


if __name__ == "__main__":
    main()
