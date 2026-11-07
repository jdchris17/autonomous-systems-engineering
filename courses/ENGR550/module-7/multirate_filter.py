"""
Module 7, Parts 2-4 and deliverable: event-driven multi-rate Kalman filter.

At each base tick k:
    1. predict from tick k-1 using that interval's acceleration reading (dt from timestamps)
    2. process any velocity / position readings stamped at tick k, one after another
    3. log the estimate and covariance
At t_0 the readings are processed directly against the prior.

Two-state model (x = [p, v]):  x^-_{k+1} = F x^+_k + G a_m,k,   Q = sigma_eta^2 G G^T
Bias model (x = [p, v, b_a]):  true accel = a_m - b_a - eta, so
    F_b = [[1, h, -h^2/2], [0, 1, -h], [0, 0, 1]],   G_b = [h^2/2, h, 0]^T,   Q_b = sigma_eta^2 G_b G_b^T

Position gate: reject a position correction when NIS_p = nu_p^2 / S_p > 6.63 (~99th pct, 1 dof).
A rejected reading leaves the state as it was (predicted, or already velocity-corrected).

Why velocity readings reveal an acceleration bias (and Module 4's position bias was hidden):
    b_a enters the DYNAMICS. Integrating a biased acceleration makes the predicted velocity
    drift by -b_a per second, a ramp that no initial condition can imitate, since the initial
    velocity is a constant offset, not a ramp. Repeated velocity readings measure that ramp, so b_a is
    observable: the columns for v and b_a in the observability matrix differ (-k h for b_a).
    Module 4's position-sensor bias entered only the MEASUREMENT, as p + b. Its column was
    identical to position's at every time, so no sequence of readings could separate them.
    The difference is whether the unknown leaves its own signature in how the state evolves.

Runs:  A  b_a = 0     gate off   [p, v]
       B  b_a = 0     gate on    [p, v]
       C  b_a = 0.08  gate on    [p, v]
       D  b_a = 0.08  gate on    [p, v, b_a]
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from kalman import correct
from sensor_timeline import (F, G, H_STEP, N_INTERVALS, OUTAGE_TICKS, OUTLIER_TICKS, SEED,
                             SIGMA_ETA, SIGMA_P, SIGMA_V, build_events, draw_noise, simulate_truth)

BIAS = 0.08        # m/s^2
GATE = 6.63        # NIS threshold for position readings
OUT_DIR = Path(__file__).resolve().parent
LOG_DIR = OUT_DIR / "event_logs"
R = {"velocity": np.array([[SIGMA_V**2]]), "position": np.array([[SIGMA_P**2]])}


def F_b(h):
    return np.array([[1.0, h, -h**2 / 2],
                     [0.0, 1.0, -h],
                     [0.0, 0.0, 1.0]])


def G_b(h):
    return np.array([h**2 / 2, h, 0.0])


TWO_STATE = {
    "F": F, "G": G,
    "H": {"position": np.array([[1.0, 0.0]]), "velocity": np.array([[0.0, 1.0]])},
    "x0": np.zeros(2), "P0": np.diag([1.0, 1.0]),
}
BIAS_STATE = {
    "F": F_b, "G": G_b,
    "H": {"position": np.array([[1.0, 0.0, 0.0]]), "velocity": np.array([[0.0, 1.0, 0.0]])},
    "x0": np.zeros(3), "P0": np.diag([1.0**2, 1.0**2, 0.1**2]),
}
RUNS = {
    "A": {"bias": 0.0, "gate": False, "model": TWO_STATE, "color": "tab:gray"},
    "B": {"bias": 0.0, "gate": True, "model": TWO_STATE, "color": "tab:blue"},
    "C": {"bias": BIAS, "gate": True, "model": TWO_STATE, "color": "tab:red"},
    "D": {"bias": BIAS, "gate": True, "model": BIAS_STATE, "color": "tab:green"},
}


def run_filter(model, events, gate=True, order=("velocity", "position")):
    """Event-driven filter. Returns per-tick estimates, covariances and the measurement event log."""
    accel = {e.k: e for e in events if e.sensor == "accel"}
    readings = {}
    for e in events:
        if e.sensor != "accel":
            readings.setdefault(e.k, []).append(e)

    n_ticks = N_INTERVALS + 1
    n = len(model["x0"])
    xs = np.zeros((n_ticks, n))
    Ps = np.zeros((n_ticks, n, n))
    log = []
    x, P = model["x0"].copy(), model["P0"].copy()
    for k in range(n_ticks):
        t_k = k * H_STEP
        if k > 0:
            a = accel[k - 1]
            dt = t_k - a.t                         # elapsed time from timestamps
            Fh, Gh = model["F"](dt), model["G"](dt)
            x = Fh @ x + Gh * a.value
            P = Fh @ P @ Fh.T + SIGMA_ETA**2 * np.outer(Gh, Gh)
        for e in sorted(readings.get(k, []), key=lambda e: order.index(e.sensor)):
            x_new, P_new, nu, S, _ = correct(x, P, e.value, model["H"][e.sensor], R[e.sensor])
            nis = nu[0] ** 2 / S[0, 0]
            accepted = not (gate and e.sensor == "position" and nis > GATE)
            if accepted:
                x, P = x_new, P_new
            log.append({"t": round(e.t, 10), "sensor": e.sensor, "value": e.value, "innovation": nu[0],
                        "S": S[0, 0], "NIS": nis, "decision": "accept" if accepted else "reject"})
        xs[k], Ps[k] = x, P
    return xs, Ps, log


def observability_rank(F_mat, H_mat):
    n = F_mat.shape[0]
    blocks, HFk = [], np.atleast_2d(H_mat)
    for _ in range(n):
        blocks.append(HFk)
        HFk = HFk @ F_mat
    return np.linalg.matrix_rank(np.vstack(blocks))


def order_check(x_true, u, noise):
    """Sequential corrections at shared timestamps: velocity-first vs position-first."""
    print("Check: correction order at shared timestamps (velocity first vs position first)")
    for name in ("A", "D"):
        cfg = RUNS[name]
        events = build_events(x_true, u, noise, bias=cfg["bias"])
        xs1, Ps1, log1 = run_filter(cfg["model"], events, cfg["gate"], ("velocity", "position"))
        xs2, Ps2, log2 = run_filter(cfg["model"], events, cfg["gate"], ("position", "velocity"))
        dec1 = sorted((r["t"], r["sensor"], r["decision"]) for r in log1)
        dec2 = sorted((r["t"], r["sensor"], r["decision"]) for r in log2)
        print(f"  run {name}: max |dx| = {np.max(np.abs(xs1 - xs2)):.1e}, max |dP| = "
              f"{np.max(np.abs(Ps1 - Ps2)):.1e}, gate decisions identical: {dec1 == dec2}")
    print("  (with gating on, the position NIS itself depends on the order, because a prior velocity\n"
          "   correction changes S_p; equivalence holds as long as no decision flips.)\n")


def write_log(name, log):
    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / f"run_{name}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log[0].keys()))
        writer.writeheader()
        writer.writerows(log)
    return path


def plot(t, x_true, results):
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 11),
                             gridspec_kw={"height_ratios": [1.2, 1, 0.8]})
    outage = (OUTAGE_TICKS[0] * H_STEP, OUTAGE_TICKS[1] * H_STEP)
    outlier_t = [k * H_STEP for k in OUTLIER_TICKS]

    ax = axes[0]
    for name, cfg in RUNS.items():
        err = results[name]["x"][:, 0] - x_true[:, 0]
        ax.plot(t, err, lw=1.3, color=cfg["color"], label=f"run {name}")
    ax.set_ylabel("position error [m]")
    ax.set_title("(a) Position error, all runs")
    ax.set_ylim(-1.0, 2.3)

    ax = axes[1]
    for name in ("B", "D"):
        err = results[name]["x"][:, 0] - x_true[:, 0]
        sd = np.sqrt(results[name]["P"][:, 0, 0])
        color = RUNS[name]["color"]
        ax.fill_between(t, -2 * sd, 2 * sd, color=color, alpha=0.15, label=f"run {name} $\\pm 2\\sigma_p$")
        ax.plot(t, err, lw=1.2, color=color, label=f"run {name} error")
    ax.set_ylabel("position error [m]")
    ax.set_title("(b) Position error and $\\pm 2\\sigma$ bounds, runs B and D")

    ax = axes[2]
    b_hat = results["D"]["x"][:, 2]
    b_sd = np.sqrt(results["D"]["P"][:, 2, 2])
    ax.fill_between(t, b_hat - 2 * b_sd, b_hat + 2 * b_sd, color="tab:green", alpha=0.2,
                    label="$\\pm 2\\sigma$")
    ax.plot(t, b_hat, lw=1.5, color="tab:green", label="estimate $\\hat b_a$ (run D)")
    ax.axhline(BIAS, color="k", ls="--", lw=1, label=f"true $b_a$ = {BIAS} m/s$^2$")
    ax.set_ylabel("acceleration bias [m/s$^2$]")
    ax.set_xlabel("time [s]")
    ax.set_title("(c) Estimated acceleration bias, run D")

    for i, ax in enumerate(axes):
        ax.axvspan(*outage, color="orange", alpha=0.12, label="position outage" if i < 2 else None)
        for j, to in enumerate(outlier_t):
            ax.axvline(to, color="purple", ls=":", lw=1.2,
                       label="+8 m outlier" if (j == 0 and i < 2) else None)
        ax.axhline(0, color="k", lw=0.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right", ncol=2)
    fig.suptitle(f"Multi-rate fusion: 50 Hz accel, 5 Hz velocity, 1 Hz position (seed {SEED})")
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true", help="save the figure without opening a window")
    args = parser.parse_args()

    x_true, u = simulate_truth()
    noise = draw_noise()
    t = np.arange(N_INTERVALS + 1) * H_STEP

    print("Observability of the bias model (h = 0.02 s)")
    for label, Hm in (("velocity only", [[0, 1, 0]]), ("position only", [[1, 0, 0]]),
                      ("position + velocity", [[1, 0, 0], [0, 1, 0]])):
        print(f"  {label:<20} rank {observability_rank(F_b(H_STEP), np.array(Hm, float))} of 3")
    print("  (velocity alone: v and b_a observable, absolute p is not -- as in Module 4, Part 1)\n")

    order_check(x_true, u, noise)

    results = {}
    outage = slice(*OUTAGE_TICKS)
    print(f"{'Run':<4} {'b_a':>5} {'gate':>5} {'states':>7} | {'pos RMSE':>8} {'vel RMSE':>9} "
          f"{'outage pos RMSE':>15} {'rejected':>8}  rejected at")
    for name, cfg in RUNS.items():
        events = build_events(x_true, u, noise, bias=cfg["bias"])
        xs, Ps, log = run_filter(cfg["model"], events, cfg["gate"])
        results[name] = {"x": xs, "P": Ps, "log": log}
        err = xs[:, :2] - x_true
        rej = [r["t"] for r in log if r["decision"] == "reject"]
        print(f"{name:<4} {cfg['bias']:5.2f} {'on' if cfg['gate'] else 'off':>5} {len(cfg['model']['x0']):>7} | "
              f"{np.sqrt(np.mean(err[:, 0]**2)):7.3f}m {np.sqrt(np.mean(err[:, 1]**2)):6.3f}m/s "
              f"{np.sqrt(np.mean(err[outage, 0]**2)):14.3f}m {len(rej):>8}  "
              f"{', '.join(f'{v:g} s' for v in rej) or '-'}")
        write_log(name, log)

    xD, PD = results["D"]["x"], results["D"]["P"]
    print(f"\nRun D final bias estimate: {xD[-1, 2]:+.4f} m/s^2, SD {np.sqrt(PD[-1, 2, 2]):.4f} m/s^2 "
          f"(true {BIAS:+.2f})")

    print("\nPosition-reading decisions (NIS, threshold 6.63):")
    pos_rows = {name: [r for r in results[name]["log"] if r["sensor"] == "position"] for name in RUNS}
    print(f"{'t [s]':>6} " + " ".join(f"{'run ' + n:>14}" for n in RUNS))
    for i, row in enumerate(pos_rows["A"]):
        cells = []
        for name in RUNS:
            r = pos_rows[name][i]
            cells.append(f"{r['NIS']:8.2f} {'acc' if r['decision'] == 'accept' else 'REJ':>5}")
        print(f"{row['t']:6.1f} " + " ".join(cells))

    fig = plot(t, x_true, results)
    fig.savefig(OUT_DIR / "multirate_fusion.png", dpi=130)
    print(f"\nEvent logs written to {LOG_DIR.name}/run_A..D.csv")
    print("Saved multirate_fusion.png")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
