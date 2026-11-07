"""
Module 15, Parts 3-5: outage and degraded fixes, sensor ablation, and the error budget.

Every configuration sees the same truth, the same inertial readings and the same aiding readings;
only what the filter uses (and whether it models the biases) changes.
    full fusion        inertial + position + wheel, biases estimated
    no wheel speed     inertial + position,         biases estimated
    no position fixes  inertial + wheel,            biases estimated
    biases fixed at 0  inertial + position + wheel, 4-state model (no bias compensation)
Position fixes: none for 18 <= t < 28 s; SD 0.5 m/axis before 30 s, 2 m/axis from 30 s (R follows).
Each run's aiding updates (time, sensor, innovation, S, NIS) are written to logs/<config>.csv, and
estimates with covariance diagonals to logs/<config>_states.csv.

Part 5 error budget, unaided straight-line propagation for T = 10 s at v = 2 m/s:
    initial heading error 10 deg      cross-track  v T dpsi           = 2 x 10 x 0.1745 = 3.49 m
    accel-bias error 0.05 m/s^2       along-track  1/2 db_a T^2        = 0.5 x 0.05 x 100 = 2.50 m
    gyro-bias error 0.005 rad/s       heading      db_g T              = 0.05 rad = 2.9 deg
                                      cross-track  1/2 v db_g T^2      = 0.5 x 2 x 0.005 x 100 = 0.50 m
The run compares these with the residual errors the filter actually carries into the outage.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nav_ekf import run
from nav_model import (B_A_TRUE, B_G_TRUE, H_STEP, N_STEPS, OUTAGE, T_POS_LATE, draws, simulate, wrap)

OUT_DIR = Path(__file__).resolve().parent
LOG_DIR = OUT_DIR / "logs"
CONFIGS = {
    "full fusion":       dict(use_position=True, use_wheel=True, estimate_biases=True),
    "no wheel speed":    dict(use_position=True, use_wheel=False, estimate_biases=True),
    "no position fixes": dict(use_position=False, use_wheel=True, estimate_biases=True),
    "biases fixed at 0": dict(use_position=True, use_wheel=True, estimate_biases=False),
}
COLORS = {"full fusion": "tab:blue", "no wheel speed": "tab:orange", "no position fixes": "tab:red",
          "biases fixed at 0": "tab:purple"}
K_OUT0, K_OUT1 = int(round(OUTAGE[0] / H_STEP)), int(round(OUTAGE[1] / H_STEP))


def errors(x_true, xs):
    e = x_true - xs[:, :4]
    e[:, 3] = wrap(e[:, 3])
    return e


def along_cross(e_pos, psi):
    return (e_pos[0] * np.cos(psi) + e_pos[1] * np.sin(psi), -e_pos[0] * np.sin(psi) + e_pos[1] * np.cos(psi))


def metrics(x_true, xs, Ps, log):
    e = errors(x_true, xs)
    pos = np.hypot(e[:, 0], e[:, 1])
    t = np.arange(N_STEPS + 1) * H_STEP
    out = (t >= OUTAGE[0] - 1e-9) & (t < OUTAGE[1] - 1e-9)
    resume = next((r for r in log if r["sensor"] == "position" and r["k"] == K_OUT1), None)
    prior = resume["prior"] if resume else xs[K_OUT1]
    e_res = x_true[K_OUT1, :2] - prior[:2]
    m = {"pos_rmse": np.sqrt(np.mean(pos**2)), "v_rmse": np.sqrt(np.mean(e[:, 2]**2)),
         "head_rmse": np.degrees(np.sqrt(np.mean(e[:, 3]**2))), "out_max": np.max(pos[out]),
         "pre_resume": np.hypot(*e_res), "final_pos": pos[-1], "final_head": np.degrees(e[-1, 3])}
    m["pre_resume_ac"] = along_cross(e_res, x_true[K_OUT1, 3])
    if xs.shape[1] == 6:
        m["ba"], m["ba_sd"] = xs[-1, 4], np.sqrt(Ps[-1, 4, 4])
        m["bg"], m["bg_sd"] = xs[-1, 5], np.sqrt(Ps[-1, 5, 5])
    nis = {}
    for sensor, segs in (("position", ((0, 18), (28, 30), (30, 40.01))), ("wheel", ((0, 18), (18, 28), (28, 40.01)))):
        for a, b in segs:
            vals = [r["nis"] for r in log if r["sensor"] == sensor and a - 1e-9 <= r["t"] < b - 1e-9]
            nis[(sensor, a, b)] = (np.mean(vals) if vals else np.nan, len(vals))
    m["nis"] = nis
    return m, e, pos


def write_logs(name, xs, Ps, log):
    LOG_DIR.mkdir(exist_ok=True)
    slug = name.replace(" ", "_")
    with open(LOG_DIR / f"{slug}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "sensor", "innov_1", "innov_2", "S_11", "S_22", "NIS"])
        for r in log:
            nu, S = r["nu"], r["S"]
            w.writerow([r["t"], r["sensor"], f"{nu[0]:.6f}", f"{nu[1]:.6f}" if len(nu) > 1 else "",
                        f"{S[0, 0]:.6f}", f"{S[1, 1]:.6f}" if len(nu) > 1 else "", f"{r['nis']:.4f}"])
    names = ["p_x", "p_y", "v", "psi", "b_a", "b_g"][:xs.shape[1]]
    with open(LOG_DIR / f"{slug}_states.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t"] + names + [f"sd_{s}" for s in names])
        for k in range(len(xs)):
            w.writerow([round(k * H_STEP, 4)] + [f"{v:.6f}" for v in xs[k]] +
                       [f"{np.sqrt(Ps[k, i, i]):.6f}" for i in range(xs.shape[1])])


def report(results):
    print("Part 4: sensor ablation (seed 2)")
    print(f"  {'configuration':<19}{'pos RMSE':>9}{'v RMSE':>8}{'psi RMSE':>9}{'outage max':>11}{'pre-resume':>11}"
          f"{' (along, cross)':>16}{'final |dp|':>11}{'final dpsi':>11}")
    for name, (m, _, _) in results.items():
        a, c = m["pre_resume_ac"]
        print(f"  {name:<19}{m['pos_rmse']:8.2f}m{m['v_rmse']:8.3f}{m['head_rmse']:7.2f}dg{m['out_max']:10.2f}m"
              f"{m['pre_resume']:10.2f}m   ({a:+6.2f}, {c:+6.2f}){m['final_pos']:10.2f}m{m['final_head']:+9.2f}dg")
    print(f"\n  final bias estimates (true b_a = {B_A_TRUE}, b_g = {B_G_TRUE})")
    for name, (m, _, _) in results.items():
        if "ba" in m:
            print(f"  {name:<19} b_a = {m['ba']:+.4f} +- {m['ba_sd']:.4f} m/s^2 ({(m['ba'] - B_A_TRUE) / m['ba_sd']:+.1f} sd)"
                  f"   b_g = {m['bg']:+.5f} +- {m['bg_sd']:.5f} rad/s ({(m['bg'] - B_G_TRUE) / m['bg_sd']:+.1f} sd)")
    print("\n  mean NIS by segment [count] (expected: position 2, wheel 1)")
    for name, (m, _, _) in results.items():
        cells = []
        for (s, a, b), (v, n) in m["nis"].items():
            if n:
                cells.append(f"{s[:3]} {a:g}-{min(b, 40):g}s {v:6.2f}[{n}]")
        print(f"  {name:<19} " + "   ".join(cells))


def budget(x_true, results):
    T, v = 10.0, 2.0
    print("\nPart 5: error budget, unaided straight-line propagation, T = 10 s, v = 2 m/s")
    rows = [("initial heading error 10 deg", "cross-track v T dpsi", v * T * np.radians(10)),
            ("accel-bias error 0.05 m/s^2", "along-track 1/2 db_a T^2", 0.5 * 0.05 * T**2),
            ("gyro-bias error 0.005 rad/s", "heading db_g T", np.degrees(0.005 * T)),
            ("gyro-bias error 0.005 rad/s", "cross-track 1/2 v db_g T^2", 0.5 * v * 0.005 * T**2)]
    for src, eff, val in rows:
        unit = "deg" if "heading db_g" in eff else "m"
        print(f"  {src:<30} {eff:<28} {val:6.2f} {unit}")
    print("\n  Signed (error = truth - estimate): a bias the filter under-estimates (db > 0) makes the estimate run"
          " ahead,\n  so along = dv T - 1/2 db_a T^2 and cross = v T dpsi - 1/2 v db_g T^2 (straight line, unaided).")
    print("\n  Same formulas with the RESIDUAL errors each filter carries into the outage (t = 18 s):")
    print(f"  {'configuration':<19}{'dpsi':>8}{'db_a':>9}{'db_g':>10}{'  -> v T dpsi':>14}{'1/2 db_a T^2':>14}"
          f"{'1/2 v db_g T^2':>16}{'dv T':>8}{'pred. (along, cross)':>23}{'observed growth (along, cross)':>33}")
    k = K_OUT0
    for name, (m, e, _) in results.items():
        xs = m["_xs"]
        dpsi = e[k, 3]
        dba = B_A_TRUE - (xs[k, 4] if xs.shape[1] == 6 else 0.0)
        dbg = B_G_TRUE - (xs[k, 5] if xs.shape[1] == 6 else 0.0)
        vel = x_true[k, 2]
        log = m["_log"]
        resume = next((r for r in log if r["sensor"] == "position" and r["k"] == K_OUT1), None)
        prior = resume["prior"] if resume else xs[K_OUT1]
        growth = (x_true[K_OUT1, :2] - prior[:2]) - (x_true[k, :2] - xs[k, :2])
        a, c = along_cross(growth, x_true[k, 3])
        dv = e[k, 2]
        cross_h, along_a, cross_g = vel * T * dpsi, -0.5 * dba * T**2, -0.5 * vel * dbg * T**2
        print(f"  {name:<19}{np.degrees(dpsi):+7.2f}d{dba:+9.4f}{dbg:+10.5f}{cross_h:+14.2f}"
              f"{along_a:+14.2f}{cross_g:+16.2f}{dv * T:+8.2f}     ({along_a + dv * T:+6.2f}, {cross_h + cross_g:+6.2f})"
              f"              ({a:+6.2f}, {c:+6.2f}) m")
    print("  (error = truth - estimate; growth = position error just before the 28 s fix minus the error at 18 s,"
          " resolved along/across the truth's heading at 18 s.\n   The vehicle turns from 25 s and wheel/bias"
          " updates continue, so these are scale checks, not predictions.)")


def plot(x_true, results, path):
    t = np.arange(N_STEPS + 1) * H_STEP
    fig = plt.figure(figsize=(14, 15))
    gs = fig.add_gridspec(4, 2)
    ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]),
          fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]),
          fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1])]
    out = (t >= OUTAGE[0] - 1e-9) & (t < OUTAGE[1] - 1e-9)
    ax[0].plot(x_true[:, 0], x_true[:, 1], color="k", lw=2.5, label="truth")
    ax[0].plot(x_true[out, 0], x_true[out, 1], color="gold", lw=5, alpha=0.6, label="truth during outage")
    for name, (m, e, pos) in results.items():
        xs = m["_xs"]
        ax[0].plot(xs[:, 0], xs[:, 1], color=COLORS[name], lw=1.1, label=name)
    ax[0].set_aspect("equal", adjustable="datalim")
    ax[0].set_title("(a) trajectories (equal axes)")
    ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("y (m)")
    ax[0].legend(fontsize=8)
    for name, (m, e, pos) in results.items():
        Ps, c = m["_Ps"], COLORS[name]
        ax[1].semilogy(t, pos, color=c, lw=1.2, label=name)
        ax[1].semilogy(t, 2 * np.sqrt(Ps[:, 0, 0] + Ps[:, 1, 1]), color=c, lw=0.8, ls="--")
        ax[2].plot(t, np.degrees(e[:, 3]), color=c, lw=1.2, label=name)
        ax[2].plot(t, 2 * np.degrees(np.sqrt(Ps[:, 3, 3])), color=c, lw=0.7, ls="--")
        ax[2].plot(t, -2 * np.degrees(np.sqrt(Ps[:, 3, 3])), color=c, lw=0.7, ls="--")
        ax[3].plot(t, e[:, 2], color=c, lw=1.0, label=name)
        if Ps.shape[1] == 6:
            xs = m["_xs"]
            for i, axb in ((4, ax[4]), (5, ax[5])):
                axb.plot(t, xs[:, i], color=c, lw=1.4, label=name)
                axb.fill_between(t, xs[:, i] - 2 * np.sqrt(Ps[:, i, i]), xs[:, i] + 2 * np.sqrt(Ps[:, i, i]),
                                 color=c, alpha=0.12, lw=0)
        log = m["_log"]
        for sensor, axn in (("position", ax[6]), ("wheel", ax[7])):
            pts = [(r["t"], r["nis"]) for r in log if r["sensor"] == sensor]
            if pts:
                tt, nn = zip(*pts)
                axn.semilogy(tt, nn, ".", color=c, ms=4, label=name)
    ax[4].axhline(B_A_TRUE, color="k", ls=":", lw=1, label="true b_a")
    ax[5].axhline(B_G_TRUE, color="k", ls=":", lw=1, label="true b_g")
    ax[6].axhline(2, color="k", lw=0.8); ax[7].axhline(1, color="k", lw=0.8)
    ax[1].set_title("(b) |position error| (solid) and 2·sqrt(P_xx+P_yy) (dashed)"); ax[1].set_ylabel("m")
    ax[1].set_ylim(1e-2, 2e2)
    ax[2].set_title("(c) wrapped heading error with ±2σ"); ax[2].set_ylabel("deg"); ax[2].set_ylim(-25, 25)
    ax[3].set_title("(d) speed error"); ax[3].set_ylabel("m/s"); ax[3].set_ylim(-1.5, 1.5)
    ax[4].set_title("(e) accelerometer-bias estimate ±2σ"); ax[4].set_ylabel("m/s²")
    ax[5].set_title("(f) gyro-bias estimate ±2σ"); ax[5].set_ylabel("rad/s")
    ax[6].set_title("(g) position-fix NIS (line: expected 2)"); ax[7].set_title("(h) wheel-speed NIS (line: expected 1)")
    for a in ax[1:]:
        a.axvspan(*OUTAGE, color="gold", alpha=0.15, lw=0)
        a.axvspan(T_POS_LATE, 40, color="0.5", alpha=0.08, lw=0)
        a.grid(alpha=0.3, which="both")
        a.set_xlabel("time (s)")
    ax[0].grid(alpha=0.3)
    ax[1].legend(fontsize=8, loc="upper left")
    ax[4].legend(fontsize=8); ax[5].legend(fontsize=8)
    fig.suptitle("Module 15: inertial EKF with bias states (gold: position outage 18–28 s; grey: 2 m fixes from 30 s)")
    fig.tight_layout()
    fig.savefig(path, dpi=105)
    plt.close(fig)


def ensemble_check(n_seeds=30):
    """Seed 2 is one realization: check NEES (per state, t >= 10 s) and final bias z-scores over seeds."""
    print(f"\nConsistency over {n_seeds} seeds (NEES per state expected ~1; bias z-scores expected mean 0, SD 1)")
    for name, cfg in CONFIGS.items():
        ne, zb, zg = [], [], []
        for seed in range(n_seeds):
            xt, imu, aid = simulate(draws(seed))
            xs, Ps, _ = run(xt[0], imu, aid, **cfg)
            e = errors(xt, xs)
            if xs.shape[1] == 6:
                e = np.c_[e, B_A_TRUE - xs[:, 4], B_G_TRUE - xs[:, 5]]
                zb.append(e[-1, 4] / np.sqrt(Ps[-1, 4, 4]))
                zg.append(e[-1, 5] / np.sqrt(Ps[-1, 5, 5]))
            k0 = int(round(10 / H_STEP))
            ne.append(np.mean([ei @ np.linalg.solve(P, ei) for ei, P in zip(e[k0:], Ps[k0:])]) / xs.shape[1])
        extra = (f"   b_a z {np.mean(zb):+.2f} (SD {np.std(zb):.2f}),  b_g z {np.mean(zg):+.2f} (SD {np.std(zg):.2f})"
                 if zb else "")
        print(f"  {name:<19} NEES/state {np.mean(ne):6.2f}{extra}")


def main():
    noise = draws()
    x_true, imu, aiding = simulate(noise)
    results = {}
    for name, cfg in CONFIGS.items():
        xs, Ps, log = run(x_true[0], imu, aiding, **cfg)
        m, e, pos = metrics(x_true, xs, Ps, log)
        m["_xs"], m["_Ps"], m["_log"] = xs, Ps, log
        results[name] = (m, e, pos)
        write_logs(name, xs, Ps, log)
    report(results)
    budget(x_true, results)
    ensemble_check()
    path = OUT_DIR / "ablation.png"
    plot(x_true, results, path)
    print(f"\nFigure saved: {path.name}; logs in {LOG_DIR.name}/")


if __name__ == "__main__":
    main()
