"""
Module 5, Parts 3-4: position-velocity Kalman filter on the Module 1 trajectory.

Truth:  x_{k+1} = F x_k + G (u_k + a_k),   a_k ~ N(0, sigma_a^2)   (a_k unknown to the filter)
Sensor: z_k = p_k + n_k,                   n_k ~ N(0, sigma_p^2)
Filter: Q = sigma_a^2 G G^T,  R = sigma_p^2,  receives only u_k and z_k.

Timing:
    1. at t_0, correct the initial prior with z_0
    2. predict t_k -> t_{k+1} with u_k
    3. correct that prediction with z_{k+1}
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from kalman import correct, predict
from scalar_update import hand_check

# ---------------------------------------------------------------- settings
DURATION = 10.0    # s
H_STEP = 0.1       # s
X0_TRUE = np.array([0.0, 0.0])
X0_PRIOR = np.array([1.0, -0.5])          # initial estimate, before the first measurement
P0_PRIOR = np.diag([1.0**2, 1.0**2])      # m^2, (m/s)^2
SIGMA_A = 0.3      # m/s^2
SIGMA_P = 0.5      # m
SEED = 2           # fixed, recorded seed
INSPECT_STEP = 30  # update printed in detail (t = 3.0 s, when the command switches)
TOL = 1e-12
OUT_DIR = Path(__file__).resolve().parent

H = np.array([[1.0, 0.0]])
R = np.array([[SIGMA_P**2]])


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


def generate_truth_and_measurements(seed=SEED):
    """Simulate the true motion (with acceleration errors) and the position readings."""
    n = int(round(DURATION / H_STEP)) + 1
    t = np.arange(n) * H_STEP
    u = np.array([accel_command(tk) for tk in t])
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, SIGMA_A, n - 1)       # acceleration errors, one per interval
    noise = rng.normal(0.0, SIGMA_P, n)       # measurement noise, one per sample

    x = np.zeros((n, 2))
    x[0] = X0_TRUE
    for k in range(n - 1):
        x[k + 1] = F(H_STEP) @ x[k] + G(H_STEP) * (u[k] + a[k])
    z = x[:, 0] + noise
    return t, u, x, z


def run_filter(u, z):
    """Kalman filter with full logging. Index k holds the prediction for t_k and its correction."""
    n = len(z)
    Fh, Gh = F(H_STEP), G(H_STEP)
    Q = SIGMA_A**2 * np.outer(Gh, Gh)
    log = {
        "x_pred": np.zeros((n, 2)), "P_pred": np.zeros((n, 2, 2)),
        "x_corr": np.zeros((n, 2)), "P_corr": np.zeros((n, 2, 2)),
        "nu": np.zeros(n), "S": np.zeros(n), "K": np.zeros((n, 2)),
        "prior": (X0_PRIOR.copy(), P0_PRIOR.copy()),
    }
    x_pred, P_pred = X0_PRIOR.copy(), P0_PRIOR.copy()   # the "prediction" at t_0 is the prior
    for k in range(n):
        if k > 0:
            x_pred, P_pred = predict(log["x_corr"][k - 1], log["P_corr"][k - 1], Fh, Gh, u[k - 1], Q)
        x_corr, P_corr, nu, S, K = correct(x_pred, P_pred, z[k], H, R)
        log["x_pred"][k], log["P_pred"][k] = x_pred, P_pred
        log["x_corr"][k], log["P_corr"][k] = x_corr, P_corr
        log["nu"][k], log["S"][k], log["K"][k] = nu[0], S[0, 0], K[:, 0]
    return log


def rmse(err):
    return np.sqrt(np.mean(err**2))


def run_checks(log):
    print("Implementation checks")
    results = []

    ok = hand_check(verbose=False)
    results.append(ok)
    print(f"  [{'pass' if ok else 'FAIL'}] scalar hand calculation (nu 2, S 5, K 0.8, p+ 11.6, P+ 0.8)")

    worst_asym, worst_eig = 0.0, np.inf
    for P in np.concatenate([log["P_pred"], log["P_corr"]]):
        worst_asym = max(worst_asym, np.max(np.abs(P - P.T)))
        worst_eig = min(worst_eig, np.linalg.eigvalsh(P).min())
    ok = worst_asym < TOL and worst_eig > -TOL
    results.append(ok)
    print(f"  [{'pass' if ok else 'FAIL'}] all {2 * len(log['P_pred'])} logged covariances symmetric and PSD "
          f"(max |P - P^T| = {worst_asym:.1e}, min eigenvalue = {worst_eig:.2e})")

    shrink = np.diagonal(log["P_corr"], axis1=1, axis2=2) - np.diagonal(log["P_pred"], axis1=1, axis2=2)
    ok = np.all(shrink <= TOL)
    results.append(ok)
    print(f"  [{'pass' if ok else 'FAIL'}] correction never increases a marginal variance "
          f"(largest change: P_pp {shrink[:, 0].max():+.2e}, P_vv {shrink[:, 1].max():+.2e})")
    return all(results)


def inspect_update(log, t, k=INSPECT_STEP):
    nu, K = log["nu"][k], log["K"][k]
    dx = log["x_corr"][k] - log["x_pred"][k]
    print(f"\nInspecting the update at t = {t[k]:.1f} s (k = {k})")
    print(f"  predicted  p = {log['x_pred'][k, 0]:.4f} m,  v = {log['x_pred'][k, 1]:.4f} m/s")
    print(f"  innovation nu = {nu:+.4f} m,  S = {log['S'][k]:.4f} m^2")
    print(f"  gain       K_p = {K[0]:.4f} (dimensionless),  K_v = {K[1]:.4f} 1/s")
    print(f"  correction dp = {dx[0]:+.6f} m    K_p * nu = {K[0] * nu:+.6f} m    match: {np.isclose(dx[0], K[0] * nu)}")
    print(f"             dv = {dx[1]:+.6f} m/s  K_v * nu = {K[1] * nu:+.6f} m/s  match: {np.isclose(dx[1], K[1] * nu)}")
    P = log["P_pred"][k]
    print(f"  closed form K = [P_pp, P_vp]/(P_pp + R) = "
          f"[{P[0, 0] / (P[0, 0] + R[0, 0]):.4f}, {P[1, 0] / (P[0, 0] + R[0, 0]):.4f}]")


def plot(t, x, z, log):
    xc, Pc = log["x_corr"], log["P_corr"]
    sd = np.sqrt(np.diagonal(Pc, axis1=1, axis2=2))
    err = xc - x

    fig = plt.figure(figsize=(13, 9))
    outer = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    ax = fig.add_subplot(outer[0, 0])
    ax.plot(t, z, ".", ms=4, color="tab:gray", alpha=0.7, label="measured $z_k$")
    ax.plot(t, x[:, 0], lw=2, color="k", label="true $p$")
    ax.plot(t, xc[:, 0], lw=1.5, color="tab:blue", label="filter $\\hat p^+$")
    ax.set_ylabel("position [m]")
    ax.set_title("(a) Position")
    ax.legend(loc="upper left", fontsize=9)

    ax = fig.add_subplot(outer[0, 1])
    ax.plot(t, x[:, 1], lw=2, color="k", label="true $v$")
    ax.plot(t, xc[:, 1], lw=1.5, color="tab:orange", label="filter $\\hat v^+$")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylabel("velocity [m/s]")
    ax.set_title("(b) Velocity")
    ax.legend(loc="lower left", fontsize=9)

    inner = outer[1, 0].subgridspec(2, 1, hspace=0.1)
    ax_ep = fig.add_subplot(inner[0])
    ax_ev = fig.add_subplot(inner[1], sharex=ax_ep)
    for axe, i, unit, color, name in ((ax_ep, 0, "m", "tab:blue", "position"),
                                     (ax_ev, 1, "m/s", "tab:orange", "velocity")):
        axe.fill_between(t, -2 * sd[:, i], 2 * sd[:, i], color=color, alpha=0.2,
                         label=f"$\\pm 2\\sqrt{{P^+_{{{'pp' if i == 0 else 'vv'}}}}}$")
        axe.plot(t, err[:, i], lw=1.2, color=color, label=f"{name} error")
        axe.axhline(0, color="k", lw=0.5)
        axe.set_ylabel(f"error [{unit}]")
        axe.legend(loc="upper right", fontsize=8, ncol=2)
        axe.grid(alpha=0.3)
    ax_ep.set_title("(c) Estimation errors with $\\pm 2\\sigma$ bounds")
    ax_ep.tick_params(labelbottom=False)
    ax_ev.set_xlabel("time [s]")

    ax = fig.add_subplot(outer[1, 1])
    ax.plot(t, log["K"][:, 0], lw=2, color="tab:blue")
    ax.set_ylabel("$K_p$ [dimensionless, m/m]", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(t, log["K"][:, 1], lw=2, color="tab:orange")
    ax2.set_ylabel("$K_v$ [1/s, (m/s)/m]", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax.set_title("(d) Kalman gain components")
    ax.set_xlabel("time [s]")

    for a in fig.axes:
        if a not in (ax_ep, ax_ev, ax2):
            a.grid(alpha=0.3)
    for a in (fig.axes[0], fig.axes[1]):
        a.set_xlabel("time [s]")
    fig.suptitle(f"Position-velocity Kalman filter ($\\sigma_a$ = {SIGMA_A} m/s$^2$, "
                 f"$\\sigma_p$ = {SIGMA_P} m, seed {SEED})")
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true", help="save the figure without opening a window")
    args = parser.parse_args()

    t, u, x, z = generate_truth_and_measurements()
    log = run_filter(u, z)

    x_prior, P_prior = log["prior"]
    print(f"Initial prior (kept for comparison): x = {x_prior}, P = diag{np.diag(P_prior)}")
    print(f"After correcting with z_0 = {z[0]:.4f} m: x = {np.round(log['x_corr'][0], 4)}, "
          f"P diag = {np.round(np.diag(log['P_corr'][0]), 4)}")
    print(f"  (velocity is unchanged at t_0: the prior has P_vp = 0, so K_v = {log['K'][0, 1]:.1f})\n")

    run_checks(log)
    inspect_update(log, t)

    err = log["x_corr"] - x
    settled = t >= 1.0
    print(f"\nRMSE for this realization (seed {SEED})       all samples   t >= 1 s")
    print(f"  raw position measurements                {rmse(z - x[:, 0]):8.3f} m  {rmse(z[settled] - x[settled, 0]):8.3f} m")
    print(f"  filtered position                        {rmse(err[:, 0]):8.3f} m  {rmse(err[settled, 0]):8.3f} m")
    print(f"  filtered velocity                        {rmse(err[:, 1]):8.3f} m/s{rmse(err[settled, 1]):8.3f} m/s")
    print(f"  (final predicted SDs: position {np.sqrt(log['P_corr'][-1, 0, 0]):.3f} m, "
          f"velocity {np.sqrt(log['P_corr'][-1, 1, 1]):.3f} m/s)")
    sd = np.sqrt(np.diagonal(log["P_corr"], axis1=1, axis2=2))
    inside = np.mean(np.abs(err) <= 2 * sd, axis=0)
    print(f"  fraction of errors inside +-2 sigma: position {inside[0]:.0%}, velocity {inside[1]:.0%} "
          f"(~95% if the filter is consistent)")

    fig = plot(t, x, z, log)
    fig.savefig(OUT_DIR / "kalman_filter.png", dpi=130, bbox_inches="tight")
    print("\nSaved kalman_filter.png")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
