"""
Reusable calibration estimators and evaluation (Experiment E).

CalibrationModel(features, method)
    features  "none"    baseline: no correction, predicted bias = 0
              "r"       [1, r]                 (deliberately wrong form)
              "r2"      [1, r^2]               (the physical form used in Experiments B and D)
              "quad"    [1, r, r^2]
              "quartic" [1, r, r^2, r^3, r^4]
    method    "ols"     least squares, classical homoscedastic uncertainty s^2 (X^T X)^-1
              "wls"     weights 1/sigma_i^2, covariance (X^T W X)^-1 from the reported sigma
              "huber"   HuberRegressor on the whitened problem
              "ransac"  RANSACRegressor on the whitened problem, threshold 3 sigma
    For huber/ransac the covariance is (X^T W X)^-1 over the points the method keeps
    (|r_i| < 3 sigma_i for Huber, the consensus set for RANSAC): an approximation that ignores
    the selection step.

Each fitted model provides
    predict(r)            mean calibration delta_u_hat(r)
    mean_var(r)           variance of the ESTIMATED MEAN FUNCTION at r: phi(r)^T Cov phi(r)
    noise_var(sigma)      variance of ONE new measurement around the mean: sigma_i^2 for the
                          sigma-aware methods, the pooled s^2 for OLS (it assumes one noise level)

evaluate(model, splits, truth_fn=None, theta_true=None)
    splits = {"train": d, "val": d, "test": d}; each d is a calibration dict (reg_core format),
    optionally with an "outlier" mask. Returns {split: {metric: value}}.
"""

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import HuberRegressor, LinearRegression, RANSACRegressor

from reg_core import ols, wls

Z95 = 1.959964


def features(name, r):
    r = np.asarray(r, dtype=float)
    one = np.ones_like(r)
    return {"none": np.zeros((r.size, 0)),
            "r": np.column_stack([one, r]),
            "r2": np.column_stack([one, r**2]),
            "quad": np.column_stack([one, r, r**2]),
            "quartic": np.column_stack([one, r, r**2, r**3, r**4])}[name]


class CalibrationModel:
    def __init__(self, feats="r2", method="wls", seed=0):
        self.feats, self.method, self.seed = feats, method, seed
        self.name = "no correction" if feats == "none" else f"{method.upper()} [{feats}]"

    def fit(self, d):
        r, y, s = d["r"], d["observed_bias"], d["sigma"]
        X = features(self.feats, r)
        p = X.shape[1]
        self.keep = np.ones(r.size, dtype=bool)
        if p == 0:
            self.theta, self.cov, self.s2 = np.zeros(0), np.zeros((0, 0)), np.nan
            return self
        Xw, yw = X / s[:, None], y / s
        if self.method == "ols":
            self.theta = ols(X, y)
            res = y - X @ self.theta
            self.s2 = res @ res / (r.size - p)
            self.cov = self.s2 * np.linalg.inv(X.T @ X)
            return self
        if self.method == "wls":
            self.theta = wls(X, y, 1 / s**2)
        elif self.method == "huber":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                h = HuberRegressor(epsilon=1.35, alpha=0.0, fit_intercept=False,
                                   max_iter=2000).fit(Xw, yw)
            self.theta = h.coef_
            self.keep = np.abs(yw - Xw @ self.theta) < 3.0
        elif self.method == "ransac":
            m = RANSACRegressor(estimator=LinearRegression(fit_intercept=False), min_samples=p,
                                residual_threshold=3.0, max_trials=500,
                                random_state=self.seed).fit(Xw, yw)
            self.theta = m.estimator_.coef_
            self.keep = m.inlier_mask_
        else:
            raise ValueError(self.method)
        Xk = Xw[self.keep]
        self.cov = np.linalg.inv(Xk.T @ Xk)
        self.s2 = np.nan
        return self

    def predict(self, r):
        X = features(self.feats, r)
        return X @ self.theta if X.shape[1] else np.zeros(np.size(r))

    def mean_var(self, r):
        X = features(self.feats, r)
        if X.shape[1] == 0:
            return np.zeros(np.size(r))
        return np.einsum("ij,jk,ik->i", X, self.cov, X)

    def noise_var(self, sigma):
        if self.method == "ols" and self.feats != "none":
            return np.full_like(sigma, self.s2)
        return sigma**2


def evaluate(model, splits, truth_fn=None, theta_true=None):
    """Metrics per split. Coverage metrics use only uncorrupted points when a mask is given."""
    out = {}
    for name, d in splits.items():
        r, y, s = d["r"], d["observed_bias"], d["sigma"]
        pred = model.predict(r)
        res = y - pred
        vm = model.mean_var(r)
        vn = model.noise_var(s)
        clean = ~d.get("outlier", np.zeros(r.size, dtype=bool))
        m = {"MSE": float(np.mean(res**2)),
             "RMSE": float(np.sqrt(np.mean(res**2))),
             "MAE": float(np.mean(np.abs(res))),
             "MedAE": float(np.median(np.abs(res)))}
        z = res / np.sqrt(vm + vn)
        m["z_mean"] = float(np.mean(z[clean]))
        m["z_std"] = float(np.std(z[clean]))
        m["frac_|z|>3"] = float(np.mean(np.abs(z) > 3))
        m["median|z|"] = float(np.median(np.abs(z)))        # robust, needs no clean labels
        m["PI95_cover"] = float(np.mean(np.abs(res[clean]) <= Z95 * np.sqrt(vm + vn)[clean]))
        faint = clean & (d["intensity"] < 20)
        bright = clean & (d["intensity"] >= 50)
        m["PI95_faint"] = float(np.mean(np.abs(res[faint]) <= Z95 * np.sqrt(vm + vn)[faint]))
        m["PI95_bright"] = float(np.mean(np.abs(res[bright]) <= Z95 * np.sqrt(vm + vn)[bright]))
        if truth_fn is not None:
            err = truth_fn(r) - pred
            m["RMSE_truth"] = float(np.sqrt(np.mean(err**2)))
            m["CI95_mean_cover"] = (float(np.mean(np.abs(err) <= Z95 * np.sqrt(vm)))
                                    if model.feats != "none" else 0.0)
        if theta_true is not None and model.feats == "r2":
            m["param_err"] = float(np.linalg.norm(model.theta - theta_true))
        out[name] = m
    return out


def residual_plots(models, d, path, truth_fn=None, title=""):
    """Residuals vs r, vs intensity, and standardized-residual histograms for several models."""
    k = len(models)
    fig, axes = plt.subplots(3, k, figsize=(4.6 * k, 10.5), squeeze=False)
    clean = ~d.get("outlier", np.zeros(d["r"].size, dtype=bool))
    rs = np.linspace(0, 1, 200)
    for j, mdl in enumerate(models):
        pred = mdl.predict(d["r"])
        res = d["observed_bias"] - pred
        z = res / np.sqrt(mdl.mean_var(d["r"]) + mdl.noise_var(d["sigma"]))
        ax = axes[0, j]
        ax.scatter(d["r"][clean], res[clean], s=4, alpha=0.4, color="tab:gray")
        ax.scatter(d["r"][~clean], res[~clean], s=10, marker="x", color="tab:red", alpha=0.6)
        if truth_fn is not None:
            ax.plot(rs, truth_fn(rs) - mdl.predict(rs), "k", lw=2, label="truth - model")
            ax.legend(fontsize=7, loc="upper left")
        bins = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(d["r"], bins) - 1, 0, 9)
        med = [np.median(res[clean & (idx == b)]) for b in range(10)]
        ax.plot(0.5 * (bins[1:] + bins[:-1]), med, "o-", color="tab:blue", ms=3)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylim(-2, 2)
        ax.set_title(mdl.name, fontsize=10)
        ax.set_xlabel("r")
        if j == 0:
            ax.set_ylabel("residual [px]\n(blue = binned median)")
        ax.grid(alpha=0.3)

        ax = axes[1, j]
        ax.scatter(d["intensity"][clean], res[clean], s=4, alpha=0.4, color="tab:gray")
        ax.scatter(d["intensity"][~clean], res[~clean], s=10, marker="x", color="tab:red", alpha=0.6)
        I = np.linspace(1, 100, 200)
        sig = 0.05 + 0.8 / np.sqrt(I)
        ax.plot(I, 2 * sig, "k--", lw=1)
        ax.plot(I, -2 * sig, "k--", lw=1, label="+-2 sigma(I)")
        ax.set_ylim(-2, 2)
        ax.set_xlabel("intensity I")
        if j == 0:
            ax.set_ylabel("residual [px]")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        ax = axes[2, j]
        zc = z[clean]
        ax.hist(np.clip(zc, -6, 6), bins=np.linspace(-6, 6, 49), density=True, alpha=0.6,
                color="tab:blue", label=f"clean: std {np.std(zc):.2f}")
        g = np.linspace(-6, 6, 200)
        ax.plot(g, np.exp(-g**2 / 2) / np.sqrt(2 * np.pi), "k", lw=1.5, label="N(0,1)")
        ax.set_xlabel("standardized residual")
        if j == 0:
            ax.set_ylabel("density")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
