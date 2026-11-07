"""
Module 13, Part 3: true-state benchmark vs Kalman-estimate feedback vs degraded sensing.

All three runs share one process-noise sequence and one standard-normal measurement sequence (scaled
by each sensor's SD and sampled at its own ticks); each run's readings come from its OWN vehicle
trajectory. During the 2 Hz sensor's outage (12 <= t < 16 s) the filter keeps predicting with the
applied command and the controller keeps acting on that prediction.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from output_feedback import SENSORS, U_LIM, draws, simulate

OUT_DIR = Path(__file__).resolve().parent
COLORS = {"true-state benchmark": "k", "estimated, 10 Hz": "tab:blue", "degraded, 2 Hz + outage": "tab:red"}
WINDOWS = {"5-12 s": (5, 12), "12-16 s (outage)": (12, 16), "16-30 s": (16, 30)}


def win(t, t0, t1):
    return (t >= t0 - 1e-9) & (t < t1 - 1e-9)


def rms(x):
    return np.sqrt(np.mean(x**2))


def summarize(logs):
    print("Tracking error r - v (true speed), RMS by window; IAE over the whole run")
    print(f"  {'case':<26}" + "".join(f"{w:>18}" for w in WINDOWS) + f"{'IAE':>8}{'overshoot':>11}"
          f"{'saturated':>11}{'u SD 20-30':>12}")
    for name, log in logs.items():
        t, e = log["t"], log["r"] - log["v"]
        cells = "".join(f"{rms(e[win(t, *w)]):18.4f}" for w in WINDOWS.values())
        sat = 0.02 * np.sum(np.abs(log["u_raw"]) > U_LIM)
        print(f"  {name:<26}{cells}{0.02 * np.sum(np.abs(e)):8.3f}{np.max(log['v'][t < 12]) - 3:11.3f}"
              f"{sat:10.2f}s{np.std(log['u'][win(t, 20, 30)]):12.4f}")
    print("\nEstimation error v - v_hat: actual RMS vs filter's claimed SD (mean sqrt P), by window")
    print(f"  {'case':<26}" + "".join(f"{w:>24}" for w in WINDOWS) + f"{'NEES t>=2 s':>13}")
    for name, log in logs.items():
        if name == "true-state benchmark":
            continue
        t, ee, sd = log["t"], log["v"] - log["v_hat"], np.sqrt(log["P"])
        cells = "".join(f"{rms(ee[win(t, *w)]):12.4f} / {np.mean(sd[win(t, *w)]):.4f}" for w in WINDOWS.values())
        nees = np.mean((ee**2 / log["P"])[t >= 2])
        print(f"  {name:<26}{cells}{nees:13.2f}")
    deg = logs["degraded, 2 Hz + outage"]
    t = deg["t"]
    k0, k1 = int(round(12 / 0.02)), int(round(16 / 0.02))
    print(f"\n  degraded sensor: filter SD {np.sqrt(deg['P'][k0 - 1]):.3f} m/s entering the outage -> "
          f"{np.sqrt(deg['P'][k1 - 1]):.3f} m/s at its end; |v - v_hat| at the end {abs(deg['v'][k1 - 1] - deg['v_hat'][k1 - 1]):.3f} m/s;"
          f" first reading after: correction {deg['v_hat'][k1] - (deg['v_hat'][k1 - 1]):+.3f} m/s (incl. one prediction step)")


def consistency(n_seeds=100):
    """Seed 2 is one realization: check the filter's NEES and 2-sigma coverage over many seeds."""
    print(f"\nFilter consistency over {n_seeds} seeds (t >= 2 s; expected NEES 1, coverage ~0.95)")
    for name, s in SENSORS.items():
        if s is None:
            continue
        nees, cov = [], []
        for seed in range(n_seeds):
            eta, z = draws(seed)
            log = simulate(s, eta, z)
            m = log["t"] >= 2
            e2 = (log["v"] - log["v_hat"])[m]**2
            nees.append(np.mean(e2 / log["P"][m]))
            cov.append(np.mean(e2 <= 4 * log["P"][m]))
        print(f"  {name:<26} NEES {np.mean(nees):.3f}   coverage {np.mean(cov):.3f}")


def plot(logs, path):
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 12))
    for ax in axes:
        ax.axvspan(12, 16, color="tab:red", alpha=0.08, lw=0)
    t = next(iter(logs.values()))["t"]
    axes[0].plot(t, next(iter(logs.values()))["r"], color="0.5", ls=":", lw=1, label="reference")
    for name, log in logs.items():
        c = COLORS[name]
        axes[0].plot(t, log["v"], color=c, lw=1.2, label=name)
        axes[2].plot(t, log["u"], color=c, lw=0.9, label=name)
        axes[3].plot(t, log["integ"], color=c, lw=1.4, label=name)
        if name != "true-state benchmark":
            sd = np.sqrt(log["P"])
            axes[1].fill_between(t, -2 * sd, 2 * sd, color=c, alpha=0.15, lw=0)
            axes[1].plot(t, log["v"] - log["v_hat"], color=c, lw=1.0, label=f"{name}: v − v̂ and ±2σ")
    axes[0].set_ylim(2.4, 3.6)
    axes[0].set_title("(a) true speed (zoomed on the 3 m/s target; shaded: 2 Hz sensor outage)")
    axes[1].set_ylim(-0.4, 0.4)
    axes[1].set_title("(b) estimation error with the filter's ±2σ")
    axes[2].set_ylim(-0.5, 2.2)
    axes[2].set_title(f"(c) applied command (limit ±{U_LIM:g})")
    axes[3].set_title("(d) integral contribution −k_ξ ξ")
    for ax, lab in zip(axes, ("m/s", "m/s", "m/s²", "m/s²")):
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=8, ncol=2)
    axes[1].legend(loc="lower right", fontsize=8)
    axes[3].set_xlabel("time (s)")
    fig.suptitle("Module 13: LQI with true speed vs Kalman estimate (shared process-noise and measurement-noise draws)")
    fig.tight_layout()
    fig.savefig(path, dpi=115)
    plt.close(fig)


def main():
    eta, z = draws()
    logs = {name: simulate(s, eta, z) for name, s in SENSORS.items()}
    for name, log in logs.items():
        print(f"{name}: {int(log['reading'].sum())} readings")
    print()
    summarize(logs)
    consistency()
    path = OUT_DIR / "sensing_cases.png"
    plot(logs, path)
    print(f"\nFigure saved: {path.name}")


if __name__ == "__main__":
    main()
