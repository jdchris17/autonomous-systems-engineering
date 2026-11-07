"""
Module 14, Part 4 and deliverable: initialization cases, the unwrapped-innovation failure, figure.

Same truth and noise draws for every run.
    A   offset [1 m, -1 m, 0.3 m/s, 10 deg],  SDs [2 m, 2 m, 0.5 m/s, 20 deg]
    B   heading offset 120 deg,               heading SD 90 deg
    C   same estimate as B,                   heading SD 1 deg (badly overconfident)
    A, unwrapped: case A with the bearing innovation NOT wrapped.

Verdicts come from an ensemble of 60 seeds (one 10 s window of one run is too noisy: seed 2's
case-A NEES over 30-40 s is 8.5, while the ensemble mean is 4.2 for an expected 4):
    accuracy recovered   heading RMS over 30-40 s < 5 deg in a seed
    consistent           ensemble mean NEES (t >= 10 s) within ~1.5x of 4, heading 2-sigma coverage ~0.95
    overconfident        accuracy recovered but NEES well above 4 / coverage well below 0.95
    remains inaccurate   accuracy not recovered in a meaningful fraction of seeds
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ekf import run_ekf
from planar_model import (H_STEP, LANDMARK, N_STEPS, draws, measurements, simulate_truth, wrap)

OUT_DIR = Path(__file__).resolve().parent
OFFSET_POS_V = np.array([1.0, -1.0, 0.3])
CASES = {
    "A": (10.0, 20.0, True),
    "B": (120.0, 90.0, True),
    "C": (120.0, 1.0, True),
    "A, unwrapped bearing": (10.0, 20.0, False),
}
COLORS = {"A": "tab:blue", "B": "tab:orange", "C": "tab:red", "A, unwrapped bearing": "tab:purple"}


def initial(x_true0, head_off_deg, head_sd_deg):
    x0 = x_true0 + np.r_[OFFSET_POS_V, np.radians(head_off_deg)]
    x0[3] = wrap(x0[3])
    P0 = np.diag([2.0**2, 2.0**2, 0.5**2, np.radians(head_sd_deg)**2])
    return x0, P0


def errors(x_true, xs):
    e = x_true - xs
    e[:, 3] = wrap(e[:, 3])
    return e


def nees(e, Ps):
    return np.array([ei @ np.linalg.solve(Pi, ei) for ei, Pi in zip(e, Ps)])


def summarize(name, x_true, xs, Ps):
    e = errors(x_true, xs)
    t = np.arange(N_STEPS + 1) * H_STEP
    pos = np.hypot(e[:, 0], e[:, 1])
    late = t >= 30.0 - 1e-9
    ne = nees(e, Ps)
    s = {
        "pos_rmse": np.sqrt(np.mean(pos**2)),
        "head_rmse": np.degrees(np.sqrt(np.mean(e[:, 3]**2))),
        "final_pos": pos[-1], "final_head": np.degrees(e[-1, 3]), "final_v": e[-1, 2],
        "late_pos": np.sqrt(np.mean(pos[late]**2)), "late_head": np.degrees(np.sqrt(np.mean(e[late, 3]**2))),
        "late_nees": np.mean(ne[late]), "max_nees": np.max(ne),
        "head_sd_final": np.degrees(np.sqrt(Ps[-1, 3, 3])),
    }
    big = np.abs(np.degrees(e[:, 3])) > 10
    s["head_settle"] = t[np.nonzero(big)[0][-1] + 1] if big.any() and not big[-1] else (0.0 if not big.any() else None)
    return s, e, pos, ne


def ensemble(name, n_seeds=60):
    off, sd, wrap_on = CASES[name]
    t = np.arange(N_STEPS + 1) * H_STEP
    rec, nees10, nees30, cov = 0, [], [], []
    for seed in range(n_seeds):
        nz = draws(seed)
        xt, u = simulate_truth(nz)
        x0, P0 = initial(xt[0], off, sd)
        xs, Ps, _ = run_ekf(x0, P0, u, measurements(xt, nz), wrap_bearing_innovation=wrap_on)
        e = errors(xt, xs)
        ne = nees(e, Ps)
        m10, m30 = t >= 10 - 1e-9, t >= 30 - 1e-9
        rec += np.degrees(np.sqrt(np.mean(e[m30, 3]**2))) < 5
        nees10.append(np.mean(ne[m10])); nees30.append(np.mean(ne[m30]))
        cov.append(np.mean(np.abs(e[m10, 3]) <= 2 * np.sqrt(Ps[m10, 3, 3])))
    frac = rec / n_seeds
    n10 = np.mean(nees10)
    if frac < 0.9:
        verdict = "remains inaccurate / diverges"
    elif n10 > 6 or np.mean(cov) < 0.9:
        verdict = "recovers accuracy, stays overconfident"
    else:
        verdict = "recovers (accurate and consistent)"
    return {"frac": frac, "nees10": n10, "nees10_med": np.median(nees10), "nees30": np.mean(nees30),
            "cov": np.mean(cov), "worst": np.max(nees10), "verdict": verdict}


def boundary_report(logs_w, logs_u, x_true):
    """First bearing update whose raw innovation crosses the -pi/pi cut."""
    print("\nBearing boundary crossing (case A)")
    first = next((i for i, b in enumerate(logs_u) if abs(b["raw"]) > np.pi), None)
    if first is None:
        print("  no crossing found")
        return None
    bu, bw = logs_u[first], logs_w[first]
    assert bu["k"] == bw["k"]
    k = bu["k"]
    true_brg = np.degrees(wrap(np.arctan2(LANDMARK[1] - x_true[k, 1], LANDMARK[0] - x_true[k, 0]) - x_true[k, 3]))
    print(f"  t = {k * H_STEP:.2f} s: measured z = {np.degrees(bu['z']):+.2f} deg, predicted g(x^-) = "
          f"{np.degrees(bu['pred']):+.2f} deg (true bearing {true_brg:+.2f} deg)")
    for label, b in (("wrapped", bw), ("unwrapped", bu)):
        dx = b["dx"]
        print(f"  {label:<10} innovation {np.degrees(b['nu']):+8.2f} deg, NIS {b['nis']:10.1f}  ->  correction "
              f"dp = ({dx[0]:+7.2f}, {dx[1]:+7.2f}) m, dv = {dx[2]:+6.3f} m/s, dpsi = {np.degrees(dx[3]):+8.2f} deg")
    return k


def plot(x_true, runs, summaries, logs_w, logs_u, k_cross, path):
    t = np.arange(N_STEPS + 1) * H_STEP
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 1, 1])
    ax_tr = fig.add_subplot(gs[0, 0])
    ax_h = fig.add_subplot(gs[0, 1])
    ax_p = fig.add_subplot(gs[1, :])
    ax_bi = fig.add_subplot(gs[2, 0])
    ax_bh = fig.add_subplot(gs[2, 1])

    ax_tr.plot(x_true[:, 0], x_true[:, 1], color="k", lw=2.2, label="truth")
    for name, (xs, Ps, e) in runs.items():
        ls = ":" if "unwrapped" in name else "-"
        ax_tr.plot(xs[:, 0], xs[:, 1], color=COLORS[name], lw=1.2, ls=ls, label=name)
    ax_tr.plot(*x_true[0, :2], "ko", ms=5)
    ax_tr.plot(*LANDMARK, "k*", ms=14, label="landmark")
    ax_tr.set_aspect("equal", adjustable="datalim")
    ax_tr.set_xlabel("x (m)"); ax_tr.set_ylabel("y (m)")
    ax_tr.set_title("(a) planar trajectories (equal axes; dot = start)")
    ax_tr.legend(fontsize=8, loc="upper left")

    for name in ("A", "B", "C"):
        xs, Ps, e = runs[name]
        ax_h.plot(t, np.degrees(e[:, 3]), color=COLORS[name], lw=1.3, label=f"{name} error")
        ax_h.plot(t, 2 * np.degrees(np.sqrt(Ps[:, 3, 3])), color=COLORS[name], lw=0.8, ls="--")
        ax_h.plot(t, -2 * np.degrees(np.sqrt(Ps[:, 3, 3])), color=COLORS[name], lw=0.8, ls="--")
    ax_h.set_ylim(-130, 25)
    ax_h.set_xlabel("time (s)"); ax_h.set_ylabel("deg")
    ax_h.set_title("(b) wrapped heading error, dashed = filter ±2σ")
    ax_h.legend(fontsize=8)

    for name, (xs, Ps, e) in runs.items():
        ls = ":" if "unwrapped" in name else "-"
        pos = np.hypot(e[:, 0], e[:, 1])
        sd = np.sqrt(Ps[:, 0, 0] + Ps[:, 1, 1])
        ax_p.semilogy(t, pos, color=COLORS[name], lw=1.3, ls=ls, label=f"{name}: |position error|")
        ax_p.semilogy(t, 2 * sd, color=COLORS[name], lw=0.8, ls="--")
    ax_p.set_xlabel("time (s)"); ax_p.set_ylabel("m")
    ax_p.set_title("(c) position-error magnitude (solid) vs filter's 2·sqrt(P_xx + P_yy) (dashed)")
    ax_p.legend(fontsize=8, ncol=4, loc="upper right")
    ax_p.set_ylim(1e-2, 1e2)

    tw = np.array([b["k"] for b in logs_w]) * H_STEP
    tu = np.array([b["k"] for b in logs_u]) * H_STEP
    ax_bi.plot(tw, np.degrees([b["nu"] for b in logs_w]), "o-", ms=3, color="tab:blue", lw=0.8, label="wrapped innovation")
    ax_bi.plot(tu, np.degrees([b["nu"] for b in logs_u]), "s-", ms=3, color="tab:purple", lw=0.8, label="unwrapped innovation")
    if k_cross is not None:
        tc = k_cross * H_STEP
        ax_bi.axvline(tc, color="k", lw=0.6, ls="--")
        ax_bi.set_xlim(tc - 3, tc + 6)
    for b in logs_u:
        if abs(b["raw"]) > np.pi:
            ax_bh.axvline(b["k"] * H_STEP, color="tab:purple", lw=0.8, ls="--", alpha=0.7)
    ax_bi.set_ylabel("bearing innovation (deg)"); ax_bi.set_xlabel("time (s)")
    ax_bi.set_title("(d) case A bearing innovations around the ±180° crossing")
    ax_bi.legend(fontsize=8)
    for name in ("A", "A, unwrapped bearing"):
        ax_bh.plot(t, np.degrees(runs[name][2][:, 3]), color=COLORS[name], lw=1.4,
                   ls=":" if "unwrapped" in name else "-", label=name)
    ax_bh.set_ylabel("heading error (deg)"); ax_bh.set_xlabel("time (s)")
    ax_bh.set_title("(e) heading error, wrapped vs unwrapped (dashed: |ν| > 180°)")
    ax_bh.legend(fontsize=8)
    for ax in (ax_tr, ax_h, ax_p, ax_bi, ax_bh):
        ax.grid(alpha=0.3, which="both")
    fig.suptitle("Module 14: EKF with position (1 Hz) and landmark bearing (5 Hz)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    noise = draws()
    x_true, u = simulate_truth(noise)
    meas = measurements(x_true, noise)
    runs, summaries, blogs = {}, {}, {}
    for name, (off, sd, wrap_on) in CASES.items():
        x0, P0 = initial(x_true[0], off, sd)
        xs, Ps, blog = run_ekf(x0, P0, u, meas, wrap_bearing_innovation=wrap_on)
        s, e, _, _ = summarize(name, x_true, xs, Ps)
        runs[name], summaries[name], blogs[name] = (xs, Ps, e), s, blog

    print(f"{'case':<22}{'pos RMSE':>9}{'head RMSE':>10}{'final |dp|':>11}{'final dpsi':>11}{'final dv':>9}"
          f"{'30-40s pos':>11}{'30-40s head':>12}{'NEES 30-40':>11}{'max NEES':>11}{'head <10deg':>12}")
    for name, s in summaries.items():
        settle = "never" if s["head_settle"] is None else f"{s['head_settle']:.2f} s"
        print(f"{name:<22}{s['pos_rmse']:8.3f}m{s['head_rmse']:7.2f}deg{s['final_pos']:10.3f}m{s['final_head']:+8.2f}deg"
              f"{s['final_v']:+9.3f}{s['late_pos']:10.3f}m{s['late_head']:9.2f}deg{s['late_nees']:11.2f}"
              f"{s['max_nees']:11.1f}{settle:>12}")
    print("(seed 2; head <10deg: time after which |heading error| stays below 10 deg; expected NEES 4)")

    print("\nEnsemble over 60 seeds (same seeds for every case)")
    print(f"  {'case':<22}{'accuracy recovered':>19}{'NEES t>=10 mean/median':>24}{'NEES 30-40':>11}"
          f"{'heading 2sig cov.':>18}{'worst seed NEES':>16}   verdict")
    for name in CASES:
        en = ensemble(name)
        print(f"  {name:<22}{en['frac']:18.0%} {en['nees10']:12.1f} / {en['nees10_med']:<9.1f}{en['nees30']:11.2f}"
              f"{en['cov']:18.3f}{en['worst']:16.1f}   {en['verdict']}")

    print("\nEarly behaviour (first 5 s): largest NEES and when")
    for name in ("A", "B", "C"):
        xs, Ps, e = runs[name]
        ne = nees(e, Ps)[:101]
        k = int(np.argmax(ne))
        print(f"  {name}: max NEES {ne[k]:.1f} at t = {k * H_STEP:.2f} s; heading error then {np.degrees(e[k, 3]):+.1f} deg, "
              f"claimed heading SD {np.degrees(np.sqrt(Ps[k, 3, 3])):.1f} deg")
        bl = blogs[name][:15]
        print(f"     bearing NIS of first 15 updates: {', '.join(f'{b['nis']:.0f}' for b in bl)}")

    k_cross = boundary_report(blogs["A"], blogs["A, unwrapped bearing"], x_true)
    events = [b for b in blogs["A, unwrapped bearing"] if abs(b["raw"]) > np.pi]
    print(f"  unwrapped run: {len(events)} updates with |z - g| > 180 deg, at t = "
          + ", ".join(f"{b['k'] * H_STEP:.2f}" for b in events) + " s; their heading corrections: "
          + ", ".join(f"{np.degrees(b['dx'][3]):+.1f}" for b in events) + " deg")
    path = OUT_DIR / "ekf_cases.png"
    plot(x_true, runs, summaries, blogs["A"], blogs["A, unwrapped bearing"], k_cross, path)
    print(f"\nFigure saved: {path.name}")


if __name__ == "__main__":
    main()
