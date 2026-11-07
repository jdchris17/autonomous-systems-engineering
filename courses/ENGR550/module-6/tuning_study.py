"""
Module 6: filter tuning study with consistency diagnostics.

Truth always uses Q_true = 0.3^2 G G^T and R_true = 0.5^2. Five filters assume different
Q and R and run on the same 300 trials (same truth, same measurements):

    Matched                       Q_true        R_true     calibration reference
    Smaller covariances           0.1 Q_true    0.1 R_true overconfidence
    Larger covariances            10 Q_true     10 R_true  underconfidence
    Too little process noise      0.01 Q_true   R_true     overtrusts the motion prediction
    Too little measurement noise  Q_true        0.1 R_true overtrusts the sensor

Diagnostics: RMSE, normalized innovation r = nu / sqrt(S), NIS = r^2, +-2 sigma coverage,
innovation autocorrelation at lags 1-10, and a chi-squared check of final-time NIS.

Interpretation (seed 2, 300 trials):
    Bounds too narrow (overconfident): Smaller covariances (NIS ~10, coverage ~49%),
        Too little measurement noise (NIS ~9.4, coverage 52% / 70%), and Too little process
        noise, whose coverage keeps falling (to ~55% / ~22% at 10 s) because P keeps shrinking
        while the real error does not.
    Bounds too wide (underconfident): Larger covariances (NIS ~0.1, coverage ~100%).
    Accuracy: scaling Q and R together leaves the steady-state gain exactly unchanged, so
        after the transient "Smaller covariances" gives the SAME estimates as matched; only the
        reported P is wrong. The pooled RMSE differences for both scaled cases come from the
        first ~2 s, because P_0 is fixed and its weight relative to Q and R changes.
        Changing the Q/R balance changes the gain and hurts accuracy throughout:
        too little process noise -> gain too small, sluggish, persistent POSITIVE innovation
        autocorrelation; too little measurement noise -> gain too large, chases noise,
        NEGATIVE autocorrelation.
    Caution: Too little process noise passes the mean-NIS check (1.07 pooled, 1.12 at the
        final time, inside the 95% interval) because S = H P^- H^T + R is dominated by the
        correct R. NIS alone would miss it; autocorrelation and coverage catch it.

Implementation note: in a linear Kalman filter P, S and K depend only on F, G, H, Q, R and
P_0, never on the measurements. So each case computes its covariance/gain sequence once
and applies it to all trials at once. A check confirms this matches the ordinary per-trial
filter (Module 5 style) exactly.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from kalman import correct, predict

# ---------------------------------------------------------------- settings
DURATION = 10.0    # s
H_STEP = 0.1       # s
X0_PRIOR = np.array([1.0, -0.5])
P0 = np.diag([1.0, 1.0])
SIGMA_A = 0.3      # m/s^2, true acceleration-error SD
SIGMA_P = 0.5      # m, true measurement SD
N_TRIALS = 300
SEED = 2           # fixed, recorded seed
MAX_LAG = 10
COVERAGE_TARGET = 0.9545
SETTLE_TIME = 2.0  # s, start of the supplementary "after the transient" window
OUT_DIR = Path(__file__).resolve().parent

H = np.array([[1.0, 0.0]])

# name, Q multiplier, R multiplier, plot color
CASES = [
    ("Matched", 1.0, 1.0, "k"),
    ("Smaller covariances", 0.1, 0.1, "tab:red"),
    ("Larger covariances", 10.0, 10.0, "tab:blue"),
    ("Too little process noise", 0.01, 1.0, "tab:purple"),
    ("Too little measurement noise", 1.0, 0.1, "tab:green"),
]
BALANCE_CASES = ("Matched", "Too little process noise", "Too little measurement noise")


def F(h):
    return np.array([[1.0, h],
                     [0.0, 1.0]])


def G(h):
    return np.array([h**2 / 2, h])


def accel_command(t):
    """Module 1 acceleration schedule."""
    t = t + 1e-9
    if t < 3.0:
        return 1.0
    if t < 6.0:
        return 0.0
    return -1.0


Q_TRUE = SIGMA_A**2 * np.outer(G(H_STEP), G(H_STEP))
R_TRUE = np.array([[SIGMA_P**2]])


# ---------------------------------------------------------------- truth
def generate_trials(n_trials=N_TRIALS, seed=SEED):
    """Per trial: x_0 ~ N(x0_prior, P0), fresh acceleration errors and measurement noise."""
    n = int(round(DURATION / H_STEP)) + 1
    t = np.arange(n) * H_STEP
    u = np.array([accel_command(tk) for tk in t])
    Fh, Gh = F(H_STEP), G(H_STEP)
    rng = np.random.default_rng(seed)
    x = np.zeros((n_trials, n, 2))
    z = np.zeros((n_trials, n))
    for j in range(n_trials):
        x[j, 0] = rng.multivariate_normal(X0_PRIOR, P0)
        a = rng.normal(0.0, SIGMA_A, n - 1)
        noise = rng.normal(0.0, SIGMA_P, n)
        for k in range(n - 1):
            x[j, k + 1] = Fh @ x[j, k] + Gh * (u[k] + a[k])
        z[j] = x[j, :, 0] + noise
    return t, u, x, z


# ---------------------------------------------------------------- filter
def covariance_pass(Q, R, n):
    """Data-independent part of the filter: P^-, P^+, S, K at every step."""
    Fh, Gh = F(H_STEP), G(H_STEP)
    P_corr = np.zeros((n, 2, 2))
    S = np.zeros(n)
    K = np.zeros((n, 2))
    P_pred = P0.copy()
    for k in range(n):
        if k > 0:
            _, P_pred = predict(np.zeros(2), P_corr[k - 1], Fh, Gh, 0.0, Q)
        _, P_corr[k], _, S_k, K_k = correct(np.zeros(2), P_pred, np.zeros(1), H, R)
        S[k], K[k] = S_k[0, 0], K_k[:, 0]
    return P_corr, S, K


def run_case(u, z, Q, R):
    """Run one tuning case on all trials. Returns corrected states, innovations and covariances."""
    n_trials, n = z.shape
    Fh, Gh = F(H_STEP), G(H_STEP)
    P_corr, S, K = covariance_pass(Q, R, n)
    x_corr = np.zeros((n_trials, n, 2))
    nu = np.zeros((n_trials, n))
    x_pred = np.tile(X0_PRIOR, (n_trials, 1))
    for k in range(n):
        if k > 0:
            x_pred = x_corr[:, k - 1] @ Fh.T + Gh * u[k - 1]
        nu[:, k] = z[:, k] - x_pred[:, 0]
        x_corr[:, k] = x_pred + nu[:, k, None] * K[k]
    return {"x_corr": x_corr, "nu": nu, "S": S, "K": K, "P_corr": P_corr}


def run_single_trial(u, z_j, Q, R):
    """Ordinary per-trial filter, used only to check the vectorized version."""
    Fh, Gh = F(H_STEP), G(H_STEP)
    x, P = X0_PRIOR.copy(), P0.copy()
    out = np.zeros((len(z_j), 2))
    for k in range(len(z_j)):
        if k > 0:
            x, P = predict(x, P, Fh, Gh, u[k - 1], Q)
        x, P, _, _, _ = correct(x, P, z_j[k], H, R)
        out[k] = x
    return out


# ---------------------------------------------------------------- diagnostics
def autocorrelation(r, max_lag=MAX_LAG):
    """Pooled estimate: lagged products within each trial only, averaged across trials."""
    d = r - r.mean()
    denom = np.mean(d**2)
    return np.array([np.mean(d[:, :-lag] * d[:, lag:]) / denom for lag in range(1, max_lag + 1)])


def diagnostics(x, res, settled):
    err = res["x_corr"] - x
    r = res["nu"] / np.sqrt(res["S"])
    nis = r**2
    sd = np.sqrt(np.diagonal(res["P_corr"], axis1=1, axis2=2))      # (n, 2), same for every trial
    inside = np.abs(err) <= 2 * sd
    return {
        "rmse_p": np.sqrt(np.mean(err[..., 0] ** 2)),
        "rmse_v": np.sqrt(np.mean(err[..., 1] ** 2)),
        "mean_r": r.mean(),
        "mean_nis": nis.mean(),
        "cov_p": inside[..., 0].mean(),
        "cov_v": inside[..., 1].mean(),
        "nis_t": nis.mean(axis=0),
        "cov_p_t": inside[..., 0].mean(axis=0),
        "cov_v_t": inside[..., 1].mean(axis=0),
        "nis_final": nis[:, -1].mean(),
        "acf": autocorrelation(r),
        "K_final": res["K"][-1],
        "sd_final": sd[-1],
        "settled_rmse_p": np.sqrt(np.mean(err[:, settled, 0] ** 2)),
        "settled_rmse_v": np.sqrt(np.mean(err[:, settled, 1] ** 2)),
        "settled_nis": nis[:, settled].mean(),
        "settled_acf1": autocorrelation(r[:, settled], max_lag=1)[0],
    }


def print_predictions():
    print("Prediction (before running): scaling Q and R together leaves the steady-state gain")
    print("  unchanged (K depends on the ratio Q/R once P_0 is forgotten), so the trajectory")
    print("  estimates and RMSE should be nearly the same as matched after the first second.")
    print("  What changes is the reported P, by about the same factor:")
    print("    0.1x both -> bounds too narrow: NIS ~ 10, coverage well below 95%")
    print("    10x both  -> bounds too wide:   NIS ~ 0.1, coverage near 100%")
    print("  Changing only one of Q or R changes the gain, so it should change accuracy too.\n")


def print_table(results, chi_lo, chi_hi):
    print(f"Results: {N_TRIALS} trials x {len(results['Matched']['nis_t'])} samples, all samples pooled "
          f"(seed {SEED})")
    header = (f"{'Case':<29} {'pos RMSE':>8} {'vel RMSE':>8} {'mean NIS':>8} {'pos cov':>8} "
              f"{'vel cov':>8} {'mean r':>7} | {'final NIS':>9}")
    print(header)
    print("-" * len(header))
    for name, *_ in CASES:
        d = results[name]
        print(f"{name:<29} {d['rmse_p']:7.3f}m {d['rmse_v']:5.3f}m/s {d['mean_nis']:8.3f} "
              f"{d['cov_p']:8.1%} {d['cov_v']:8.1%} {d['mean_r']:+7.3f} | {d['nis_final']:9.3f}")
    print(f"Matched-model expectations: mean NIS ~ 1, coverage ~ {COVERAGE_TARGET:.2%}, mean r ~ 0.")
    m = results["Matched"]["nis_final"]
    verdict = "inside" if chi_lo <= m <= chi_hi else "OUTSIDE"
    print(f"\nFinal-time check (matched case): mean NIS over {N_TRIALS} trials = {m:.3f}, "
          f"95% interval [{chi_lo:.3f}, {chi_hi:.3f}] -> {verdict}")
    print("  (final-time NIS for the other cases is shown descriptively; the interval only")
    print("   applies when the filter's model is correct.)\n")


def print_settled(results):
    print(f"Supplement: samples with t >= {SETTLE_TIME:.0f} s only (separates the transient from steady state)")
    print(f"{'Case':<29} {'pos RMSE':>8} {'vel RMSE':>9} {'mean NIS':>8} {'acf lag 1':>9}")
    for name, *_ in CASES:
        d = results[name]
        print(f"{name:<29} {d['settled_rmse_p']:7.3f}m {d['settled_rmse_v']:5.3f}m/s "
              f"{d['settled_nis']:8.3f} {d['settled_acf1']:+9.3f}")
    print()


def print_autocorrelation(results, n_pairs_lag1):
    band = 2 / np.sqrt(n_pairs_lag1)
    print(f"Normalized-innovation autocorrelation (white-noise band about +-{band:.3f})")
    print(f"{'Case':<29} " + " ".join(f"{lag:>6}" for lag in range(1, MAX_LAG + 1)))
    for name, *_ in CASES:
        print(f"{name:<29} " + " ".join(f"{v:+6.3f}" for v in results[name]["acf"]))
    print()


def print_steady_state(results):
    print("Steady state at t = 10 s (gain and reported SDs)")
    print(f"{'Case':<29} {'K_p':>7} {'K_v [1/s]':>10} {'SD_p [m]':>9} {'SD_v [m/s]':>11}")
    for name, *_ in CASES:
        d = results[name]
        print(f"{name:<29} {d['K_final'][0]:7.4f} {d['K_final'][1]:10.4f} "
              f"{d['sd_final'][0]:9.3f} {d['sd_final'][1]:11.3f}")
    print()


def plot(t, results, chi_lo, chi_hi, n_pairs_lag1):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_nis, ax_lag = axes[0]
    ax_cp, ax_cv = axes[1]
    cov_band = 2 * np.sqrt(COVERAGE_TARGET * (1 - COVERAGE_TARGET) / N_TRIALS)

    ax_nis.axhspan(chi_lo, chi_hi, color="gray", alpha=0.2, label="matched 95% interval (per time)")
    for name, _, _, color in CASES:
        ax_nis.plot(t, results[name]["nis_t"], lw=1.3, color=color, label=name)
    ax_nis.axhline(1, color="k", ls="--", lw=1)
    ax_nis.set_yscale("log")
    ax_nis.set_xlabel("time [s]")
    ax_nis.set_ylabel(f"ensemble mean NIS ({N_TRIALS} trials)")
    ax_nis.set_title("(a) Mean NIS vs time (reference 1; gray = matched 95% interval)")

    for ax, key, label in ((ax_cp, "cov_p_t", "position"), (ax_cv, "cov_v_t", "velocity")):
        ax.axhspan(COVERAGE_TARGET - cov_band, COVERAGE_TARGET + cov_band, color="gray", alpha=0.2,
                   label="matched $\\pm 2$ binomial SE")
        for name, _, _, color in CASES:
            ax.plot(t, results[name][key], lw=1.3, color=color, label=name)
        ax.axhline(COVERAGE_TARGET, color="k", ls="--", lw=1)
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.set_xlabel("time [s]")
        ax.set_ylabel(f"fraction with $|e| \\leq 2\\sqrt{{P^+}}$")
        ax.set_title(f"({'c' if label == 'position' else 'd'}) {label.capitalize()} coverage vs time "
                     f"(reference 95.45%)")
    ax_cp.legend(fontsize=8, loc="lower left")         # one shared legend: colors are the same in every panel

    lags = np.arange(1, MAX_LAG + 1)
    band = 2 / np.sqrt(n_pairs_lag1)
    ax_lag.axhspan(-band, band, color="gray", alpha=0.2, label="approx. white-noise band")
    offsets = dict(zip(BALANCE_CASES, (-0.2, 0.0, 0.2)))
    for name, _, _, color in CASES:
        if name in BALANCE_CASES:
            ax_lag.bar(lags + offsets[name], results[name]["acf"], width=0.2, color=color, label=name)
    ax_lag.axhline(0, color="k", lw=0.8)
    ax_lag.set_xticks(lags)
    ax_lag.set_xlabel("lag [samples of 0.1 s]")
    ax_lag.set_ylabel("$\\hat\\rho(\\ell)$ of normalized innovations")
    ax_lag.set_title("(b) Innovation autocorrelation, lags 1-10")
    ax_lag.legend(fontsize=8)

    for ax in axes.ravel():
        ax.grid(alpha=0.3)
    fig.suptitle(f"Kalman filter tuning study: {N_TRIALS} trials, true $\\sigma_a$ = {SIGMA_A} m/s$^2$, "
                 f"$\\sigma_p$ = {SIGMA_P} m, seed {SEED}")
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true", help="save the figure without opening a window")
    args = parser.parse_args()

    print_predictions()
    t, u, x, z = generate_trials()

    results, outputs = {}, {}
    for name, q_mult, r_mult, _ in CASES:
        outputs[name] = run_case(u, z, q_mult * Q_TRUE, r_mult * R_TRUE)
        results[name] = diagnostics(x, outputs[name], t >= SETTLE_TIME)

    worst = 0.0
    for name, q_mult, r_mult, _ in CASES:
        for j in (0, N_TRIALS - 1):
            ref = run_single_trial(u, z[j], q_mult * Q_TRUE, r_mult * R_TRUE)
            worst = max(worst, np.max(np.abs(ref - outputs[name]["x_corr"][j])))
    print(f"Check: vectorized filter matches the per-trial filter (max difference {worst:.1e})\n")

    chi_lo, chi_hi = chi2.ppf([0.025, 0.975], N_TRIALS) / N_TRIALS
    n_pairs_lag1 = N_TRIALS * (len(t) - 1)
    print_table(results, chi_lo, chi_hi)
    print_steady_state(results)
    print_settled(results)
    print_autocorrelation(results, n_pairs_lag1)

    fig = plot(t, results, chi_lo, chi_hi, n_pairs_lag1)
    fig.savefig(OUT_DIR / "tuning_study.png", dpi=130)
    print("Saved tuning_study.png")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
