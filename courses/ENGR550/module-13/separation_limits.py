"""
Module 13, Part 4: limits of the separation principle (10 Hz Kalman estimate, gains held fixed).

Separation says: design the estimator and the state-feedback controller independently, connect
them, and the closed-loop poles are the union of both sets. That holds for the MODEL the two were
designed on. Here the truth departs from it:
    true drag c_true = 0.1 1/s (estimator and controller keep c = 0.2: a, b, feedforward c r)
    unknown sustained load d = -0.3 m/s^2 from t = 12 s (no state for it in either design).

Runs (same noise draws, 10 Hz sensor, SD 0.1 m/s):
    matched            c_true = 0.2, no load     (Part 3 nominal estimated-state case)
    load only          c_true = 0.2, load from 12 s
    drag + load        c_true = 0.1, load from 12 s   (the Part 4 case)
    drag only          c_true = 0.1, no load          (added to separate the two effects)

What the estimator's model gets wrong: true v' = -c_true v + u + d, model v' = -c v + u, so the
model is missing a force f = (c - c_true) v + d. At v = 3 m/s: drag error +0.3 m/s^2, load -0.3,
so before 12 s f = +0.3 and after 12 s f = 0 in the drag + load case (they cancel at 3 m/s).
A persistent model force makes the Kalman prediction lag the truth between readings; the readings
pull it back but with a small steady gain (0.11), so the estimate settles with a BIAS.
The controller integrates r - v_hat, so it drives the ESTIMATE to r: an estimate bias becomes a
tracking error the integrator can't see or remove.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from output_feedback import SENSORS, U_LIM, draws, simulate

OUT_DIR = Path(__file__).resolve().parent
SENSOR = SENSORS["estimated, 10 Hz"]
RUNS = {
    "matched":     {"c_true": 0.2, "load": 0.0},
    "load only":   {"c_true": 0.2, "load": -0.3},
    "drag only":   {"c_true": 0.1, "load": 0.0},
    "drag + load": {"c_true": 0.1, "load": -0.3},
}
COLORS = {"matched": "k", "load only": "tab:orange", "drag only": "tab:purple", "drag + load": "tab:red"}
WINDOWS = {"6-12 s": (6, 12), "20-30 s": (20, 30)}


def win(t, t0, t1):
    return (t >= t0 - 1e-9) & (t < t1 - 1e-9)


def predicted_bias(force, gain=0.109, every=5):
    """Steady estimate bias for a constant model force f (noise-free, periodic steady state).

    Between readings the truth gains ~ every*h*f on the model; each reading removes a fraction 'gain'
    of the accumulated gap. Just after a reading the gap g satisfies g = (1 - gain)(g + every h f),
    so g = (1 - gain) every h f / gain (drag and feedback dynamics neglected over 0.1 s).
    """
    return (1 - gain) * every * 0.02 * force / gain


def main():
    eta, z = draws()
    logs = {name: simulate(SENSOR, eta, z, **cfg) for name, cfg in RUNS.items()}
    print("Part 4: model mismatch with fixed estimator and controller gains (10 Hz sensor)")
    print(f"  {'run':<13}{'window':>9}{'mean r - v':>12}{'mean v - v_hat':>16}{'RMS v - v_hat':>15}"
          f"{'filter SD':>11}{'integral':>10}{'model force':>13}{'pred. bias':>12}")
    for name, log in logs.items():
        cfg = RUNS[name]
        t = log["t"]
        for wname, (t0, t1) in WINDOWS.items():
            m = win(t, t0, t1)
            ee = log["v"] - log["v_hat"]
            force = (0.2 - cfg["c_true"]) * 3.0 + (cfg["load"] if t0 >= 12 else 0.0)
            print(f"  {name:<13}{wname:>9}{np.mean((log['r'] - log['v'])[m]):12.4f}{np.mean(ee[m]):16.4f}"
                  f"{np.sqrt(np.mean(ee[m]**2)):15.4f}{np.mean(np.sqrt(log['P'][m])):11.4f}"
                  f"{np.mean(log['integ'][m]):10.3f}{force:13.2f}{predicted_bias(force):12.4f}")
        sat = 0.02 * np.sum(np.abs(log["u_raw"]) > U_LIM)
        k = log["t"] >= 12
        print(f"  {'':<13}  saturated {sat:.2f} s, overshoot {np.max(log['v'][t < 12]) - 3:+.3f} m/s, "
              f"largest dip after 12 s {3 - np.min(log['v'][k]):.3f} m/s, "
              f"NEES (t >= 2 s) {np.mean(((log['v'] - log['v_hat'])**2 / log['P'])[t >= 2]):.2f}")
    print("  integral = mean -k_xi xi in the window; the command needed at 3 m/s is c_true*3 - d, "
          "so the integral must supply (c_true - 0.2)*3 - d on top of the 0.6 feedforward.")

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 10))
    t = logs["matched"]["t"]
    for name, log in logs.items():
        c = COLORS[name]
        ee = log["v"] - log["v_hat"]
        sm = np.convolve(ee, np.ones(50) / 50, mode="same")
        axes[0].plot(t, log["r"] - log["v"], color=c, lw=0.9, label=name)
        axes[1].plot(t, sm, color=c, lw=1.6, label=name)
        axes[2].plot(t, log["integ"], color=c, lw=1.6, label=name)
    sd = np.sqrt(logs["matched"]["P"])
    axes[1].fill_between(t, -2 * sd, 2 * sd, color="0.5", alpha=0.15, lw=0, label="filter ±2σ")
    for ax in axes:
        ax.axvline(12, color="tab:red", lw=0.7, ls="--")
        ax.axhline(0, color="k", lw=0.5)
        ax.grid(alpha=0.3)
    axes[0].set_ylim(-0.7, 0.45)
    axes[0].set_title("(a) tracking error r − v (true speed)")
    axes[1].set_ylim(-0.35, 0.35)
    axes[1].set_title("(b) estimation error v − v̂ (1 s moving average) with the filter's claimed ±2σ")
    axes[2].set_title("(c) integral contribution −k_ξ ξ (what the integrator learned)")
    axes[0].set_ylabel("m/s"); axes[1].set_ylabel("m/s"); axes[2].set_ylabel("m/s²")
    axes[2].set_xlabel("time (s)")
    axes[0].legend(loc="lower right", fontsize=9, ncol=4)
    axes[1].legend(loc="upper right", fontsize=8, ncol=5)
    fig.suptitle("Module 13: separation under model mismatch (true drag 0.1 vs assumed 0.2; load −0.3 m/s² at 12 s)")
    fig.tight_layout()
    path = OUT_DIR / "separation_limits.png"
    fig.savefig(path, dpi=115)
    plt.close(fig)
    print(f"\nFigure saved: {path.name}")


if __name__ == "__main__":
    main()
