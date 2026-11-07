"""
Module 8, Parts 2-5: Rauch-Tung-Striebel fixed-interval smoother.

Backward pass (k = N-1, ..., 0), starting from x_s[N] = x_corr[N], P_s[N] = P_corr[N]:
    C_k    = P_corr[k] F_k^T (P_pred[k+1])^{-1}        (computed with a linear solve)
    x_s[k] = x_corr[k] + C_k (x_s[k+1] - x_pred[k+1])
    P_s[k] = P_corr[k] + C_k (P_s[k+1] - P_pred[k+1]) C_k^T

Index note: x_pred[k+1] is the forward prediction FOR t_{k+1}, made from x_corr[k] and the known
acceleration. x_s[k+1] is the best estimate of that same instant using all the data. Their
difference is what the later data revealed about t_{k+1} that the forward pass did not yet know.

The backward correction in words:
    The filter at t_k only knows the past. Later readings may show that the state at t_{k+1}
    was not where the filter predicted. Because x_{k+1} = F x_k + G u + w, a surprise at k+1
    is partly a surprise about k. C_k = Cov(x_k, x_{k+1} | data to k) Cov(x_{k+1} | data to k)^{-1}
    is the regression of x_k on x_{k+1}: it says how much of a correction at k+1 to carry back
    to k, and into which states. Position and velocity are linked through p_{k+1} = p_k + h v_k,
    so if the end of the outage shows the position at 12 s was 0.5 m further on than predicted,
    the smoother infers that the earlier velocity was slightly higher and the positions during
    the outage were further along, and it spreads the correction back through the gap. The
    process noise sets how far back it reaches: a large Q makes x_k and x_{k+1} less tied, so C
    shrinks and the correction fades faster.

Part 4 health checks: finite values, symmetry (measured before symmetrizing), PSD of P and of
P_corr - P_s (smoothing must not add uncertainty in any direction), and x_s[N] == x_corr[N].
Part 5: filter a prefix ending at 10 s, check it matches the full run through 10 s, smooth both,
and compare at 9 s.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from forward_pass import forward_pass, make_events
from sensor_timeline import H_STEP, OUTAGE_TICKS

OUT_DIR = Path(__file__).resolve().parent
SYM_TOL = 1e-12        # relative asymmetry tolerance
PSD_TOL = 1e-12        # eigenvalue tolerance, relative to the largest eigenvalue
PREFIX_TICK = 500      # t = 10 s
COMPARE_TICK = 450     # t = 9 s


def rts_smooth(hist):
    """Backward RTS pass over a forward History. Returns x_s, P_s, C."""
    n = len(hist.t)
    x_s = np.zeros_like(hist.x_corr)
    P_s = np.zeros_like(hist.P_corr)
    C = np.zeros_like(hist.F_steps)
    x_s[-1], P_s[-1] = hist.x_corr[-1], hist.P_corr[-1]
    for k in range(n - 2, -1, -1):
        cross_cov = hist.P_corr[k] @ hist.F_steps[k].T              # Cov(x_k, x_{k+1} | data to k)
        C[k] = np.linalg.solve(hist.P_pred[k + 1], cross_cov.T).T   # P_pred symmetric
        x_s[k] = hist.x_corr[k] + C[k] @ (x_s[k + 1] - hist.x_pred[k + 1])
        P_s[k] = hist.P_corr[k] + C[k] @ (P_s[k + 1] - hist.P_pred[k + 1]) @ C[k].T
    return x_s, P_s, C


# ---------- Part 4: numerical health ----------

def asymmetry(Ps):
    """Max relative asymmetry |P - P^T| / max|P| over a stack, measured before symmetrizing."""
    return max(np.max(np.abs(P - P.T)) / np.max(np.abs(P)) for P in Ps)


def min_eig_ratio(Ps):
    """Smallest eigenvalue of the symmetric part, relative to the largest, over a stack."""
    worst = np.inf
    for P in Ps:
        eig = np.linalg.eigvalsh(0.5 * (P + P.T))
        worst = min(worst, eig.min() / max(eig.max(), 1e-300))
    return worst


def health_checks(hist, x_s, P_s):
    print("Part 4: numerical health")
    arrays = {"x_pred": hist.x_pred, "P_pred": hist.P_pred, "x_corr": hist.x_corr,
              "P_corr": hist.P_corr, "x_s": x_s, "P_s": P_s}
    finite = all(np.all(np.isfinite(a)) for a in arrays.values())
    print(f"  all logged values finite: {finite}")
    for name in ("P_pred", "P_corr", "P_s"):
        asym, mineig = asymmetry(arrays[name]), min_eig_ratio(arrays[name])
        print(f"  {name:<7} max rel. asymmetry {asym:.1e} ({'ok' if asym < SYM_TOL else 'FAIL'})   "
              f"min eig / max eig {mineig:.2e} ({'ok' if mineig > -PSD_TOL else 'FAIL'})")
    end_ok = np.array_equal(x_s[-1], hist.x_corr[-1]) and np.array_equal(P_s[-1], hist.P_corr[-1])
    print(f"  final smoothed state and covariance equal final filtered values: {end_ok}")

    # P_corr - P_s: the difference can be ~0 near the end, so compare to the size of P_corr.
    worst, worst_k = np.inf, None
    for k in range(len(hist.t)):
        D = hist.P_corr[k] - P_s[k]
        eig = np.linalg.eigvalsh(0.5 * (D + D.T)).min() / np.max(np.abs(hist.P_corr[k]))
        if eig < worst:
            worst, worst_k = eig, k
    print(f"  P_corr - P_s PSD: min eig / max|P_corr| = {worst:.1e} at t = {hist.t[worst_k]:.2f} s "
          f"({'ok' if worst > -PSD_TOL else 'FAIL'})\n")


# ---------- Part 3: compare ----------

def rmse(err):
    return np.sqrt(np.mean(err**2))


def compare(hist, x_s, P_s, x_true):
    t = hist.t
    err_f = x_true - hist.x_corr
    err_s = x_true - x_s
    gap = (t >= 8.0 - 1e-9) & (t < 12.0 - 1e-9)
    k10 = int(round(10.0 / H_STEP))
    rows = [
        ("Position RMSE, full recording (m)", rmse(err_f[:, 0]), rmse(err_s[:, 0])),
        ("Velocity RMSE, full recording (m/s)", rmse(err_f[:, 1]), rmse(err_s[:, 1])),
        ("Position RMSE, 8 <= t < 12 s (m)", rmse(err_f[gap, 0]), rmse(err_s[gap, 0])),
        ("Position SD at t = 10 s (m)", np.sqrt(hist.P_corr[k10, 0, 0]), np.sqrt(P_s[k10, 0, 0])),
        ("Velocity SD at t = 10 s (m/s)", np.sqrt(hist.P_corr[k10, 1, 1]), np.sqrt(P_s[k10, 1, 1])),
    ]
    print("Part 3: filter vs smoother (same data)")
    print(f"  {'Metric':<38}{'Filter':>9}{'Smoother':>10}")
    for name, f, s in rows:
        print(f"  {name:<38}{f:9.3f}{s:10.3f}")
    ratio = np.sqrt(P_s[:, 0, 0] / hist.P_corr[:, 0, 0])
    print(f"  smoothed/filtered position SD: min {ratio.min():.2f} (t = {t[ratio.argmin()]:.2f} s), "
          f"max {ratio.max():.3f}\n")
    return err_f, err_s


def plot(hist, x_s, P_s, err_f, err_s):
    t = hist.t
    outage = (OUTAGE_TICKS[0] * H_STEP, OUTAGE_TICKS[1] * H_STEP)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 8))
    labels = [("position", "m"), ("velocity", "m/s")]
    for i, ax in enumerate(axes):
        sd_f = np.sqrt(hist.P_corr[:, i, i])
        sd_s = np.sqrt(P_s[:, i, i])
        ax.axvspan(*outage, color="0.85", label="position outage")
        ax.fill_between(t, -2 * sd_f, 2 * sd_f, color="tab:blue", alpha=0.15, lw=0,
                        label="filter ±2σ")
        ax.plot(t, 2 * sd_s, color="tab:orange", lw=1.2, ls="--", label="smoother ±2σ")
        ax.plot(t, -2 * sd_s, color="tab:orange", lw=1.2, ls="--")
        ax.plot(t, err_f[:, i], color="tab:blue", lw=1.0, label="filter error")
        ax.plot(t, err_s[:, i], color="tab:orange", lw=1.6, label="smoother error")
        ax.axhline(0, color="k", lw=0.5)
        name, unit = labels[i]
        ax.set_ylabel(f"{name} error ({unit})")
        ax.set_title(f"({'ab'[i]}) {name} error: truth − estimate")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", ncol=3, fontsize=9)
    axes[0].set_ylim(-1.0, 1.1)
    axes[1].set_ylim(-0.42, 0.42)
    axes[1].set_xlabel("time (s)")
    fig.suptitle("Module 8: Kalman filter vs RTS smoother (seed 2, no bias, position outage 8–12 s)")
    fig.tight_layout()
    path = OUT_DIR / "filter_vs_smoother.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------- Part 5: information timing ----------

def information_timing(hist_full, x_s_full, P_s_full, x_true):
    print("Part 5: information timing (prefix recording ending at t = 10 s)")
    _, events_prefix = make_events(last_tick=PREFIX_TICK)
    hist_pre = forward_pass(events_prefix)
    x_s_pre, P_s_pre, _ = rts_smooth(hist_pre)
    n = PREFIX_TICK + 1
    dx = max(np.max(np.abs(hist_pre.x_corr - hist_full.x_corr[:n])),
             np.max(np.abs(hist_pre.x_pred - hist_full.x_pred[:n])))
    dP = max(np.max(np.abs(hist_pre.P_corr - hist_full.P_corr[:n])),
             np.max(np.abs(hist_pre.P_pred - hist_full.P_pred[:n])))
    print(f"  1. filtered estimates through 10 s, prefix vs full: max |dx| = {dx:.1e}, "
          f"max |dP| = {dP:.1e}")

    k = COMPARE_TICK
    print(f"  2-3. estimates of the state at t = {k * H_STEP:.0f} s:")
    print(f"     {'':<22}{'p (m)':>9}{'SD_p':>8}{'p err':>8}{'v (m/s)':>10}{'SD_v':>8}")
    rows = [("truth", x_true[k], None),
            ("filter (data to 9 s)", hist_full.x_corr[k], hist_full.P_corr[k]),
            ("smoother, prefix", x_s_pre[k], P_s_pre[k]),
            ("smoother, full", x_s_full[k], P_s_full[k])]
    for name, x, P in rows:
        if P is None:
            print(f"     {name:<22}{x[0]:9.3f}{'':>16}{x[1]:10.3f}")
        else:
            print(f"     {name:<22}{x[0]:9.3f}{np.sqrt(P[0, 0]):8.3f}{x_true[k, 0] - x[0]:+8.3f}"
                  f"{x[1]:10.3f}{np.sqrt(P[1, 1]):8.3f}")
    print(f"  The prefix recording ends inside the outage: its last position reading is at 7 s, "
          f"so its\n  smoother has only velocity readings from 9 to 10 s to add. The full smoother "
          f"also has the\n  position readings from 12 s on.\n")


def main():
    x_true, events = make_events()
    hist = forward_pass(events)
    x_s, P_s, _ = rts_smooth(hist)

    health_checks(hist, x_s, P_s)
    err_f, err_s = compare(hist, x_s, P_s, x_true)
    path = plot(hist, x_s, P_s, err_f, err_s)
    print(f"Figure saved: {path.name}\n")
    information_timing(hist, x_s, P_s, x_true)


if __name__ == "__main__":
    main()
