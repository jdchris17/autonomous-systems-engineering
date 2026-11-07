"""
Module 10, Part 2: tracking and disturbance rejection (exact measurements, no actuator limit).

    r = 0 (t < 1), 3 m/s (t >= 1);   d = 0 (t < 12), -0.3 m/s^2 (t >= 12);   30 s.

P-only equilibrium error: at rest v' = 0, so 0 = -c v + Kp (r - v) + d. With e = r - v, v = r - e:
    0 = -c (r - e) + Kp e + d   ->   e_ss = (c r - d) / (c + Kp).
    Kp = 1: before the disturbance (0.6 - 0) / 1.2 = 0.50 m/s; after, (0.6 + 0.3) / 1.2 = 0.75 m/s.
    Holding 3 m/s needs a steady push of c r - d = 0.6 (then 0.9) m/s^2 against drag and the load.
    A P controller can only push when there is error (u = Kp e), so some error must remain to
    produce that push. Raising Kp shrinks e_ss but never to zero.

Integral action: I_k accumulates Ki h e_k. At an equilibrium, I must stop changing, which needs
e = 0. So if the loop settles at all, it settles with zero error, and I itself holds the whole
sustained command (I = c r - d). The integrator is a memory of the push that was needed; it keeps
supplying it without needing error to be present. When the load changes, a transient error
re-charges I to the new required value (0.6 -> 0.9 m/s^2 here), then the error decays to zero.

Closed-loop poles (continuous approximation, h << time constants):
    P:   s = -(c + Kp)                               = -1.2
    PI:  s^2 + (c + Kp) s + Ki = 0                   -> -0.6 +- 0.37j   (damping ratio 0.85)
    PID (derivative on measurement): (1 + Kd) s^2 + (c + Kp) s + Ki = 0  -> -0.55 +- 0.39j
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pid_core import GAINS, H_STEP, p_steady_error, simulate, ticks, window

OUT_DIR = Path(__file__).resolve().parent
T_END = 30.0
COLORS = {"P": "tab:gray", "PI": "tab:blue", "PID": "tab:green"}


def reference():
    n = ticks(T_END)
    k = np.arange(n)
    r = np.where(k >= ticks(1.0), 3.0, 0.0)
    d = np.where(k >= ticks(12.0), -0.3, 0.0)
    return r, d


def run_all(gains=GAINS):
    r, d = reference()
    return {name: simulate(g, r, d) for name, g in gains.items()}


def metrics(log):
    err = log["r"] - log["v"]              # true tracking error
    t = log["t"]
    return {"e_7_11": np.mean(err[window(t, 7, 11)]), "e_25_30": np.mean(err[window(t, 25, 30)]),
            "e_end": err[-1], "u_peak": np.max(np.abs(log["u"])), "I_end": log["I"][-1]}


def plot(logs, path):
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(11, 11))
    t = next(iter(logs.values()))["t"]
    axes[0].plot(t, next(iter(logs.values()))["r"], color="k", lw=1, ls=":", label="reference")
    for name, log in logs.items():
        c = COLORS[name]
        axes[0].plot(t, log["v"], color=c, lw=1.5, label=name)
        axes[1].plot(t, log["r"] - log["v"], color=c, lw=1.5, label=name)
        axes[2].plot(t, log["u"], color=c, lw=1.5, label=name)
        axes[3].plot(t, log["I"], color=c, lw=1.5, label=name)
    for ax in axes:
        ax.axvline(12, color="tab:red", lw=0.8, ls="--")
        ax.axhline(0, color="k", lw=0.5)
        ax.grid(alpha=0.3)
    axes[1].axhline(0.5, color="tab:gray", lw=0.8, ls=":")
    axes[1].axhline(0.75, color="tab:gray", lw=0.8, ls=":")
    axes[1].text(29.8, 0.52, "P prediction 0.50", ha="right", fontsize=8, color="tab:gray")
    axes[1].text(29.8, 0.77, "P prediction 0.75", ha="right", fontsize=8, color="tab:gray")
    axes[0].set_ylabel("speed (m/s)"); axes[0].set_title("(a) speed vs reference")
    axes[1].set_ylabel("error r − v (m/s)"); axes[1].set_title("(b) tracking error")
    axes[1].set_ylim(-0.6, 3.1)
    axes[2].set_ylabel("command u (m/s²)"); axes[2].set_title("(c) command")
    axes[3].set_ylabel("I (m/s²)"); axes[3].set_title("(d) integral contribution")
    axes[3].set_xlabel("time (s)")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].text(12.1, 0.3, "disturbance −0.3 m/s² on", color="tab:red", fontsize=8)
    fig.suptitle("Module 10: tracking and disturbance rejection (no noise, no actuator limit)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    logs = run_all()
    print(f"P-only predicted e_ss: before d {p_steady_error(3, 0, 1):.3f} m/s, "
          f"after d {p_steady_error(3, -0.3, 1):.3f} m/s")
    print(f"{'ctrl':<5}{'mean e 7-11 s':>15}{'mean e 25-30 s':>16}{'e at 30 s':>11}{'peak |u|':>10}"
          f"{'I at 30 s':>11}")
    for name, log in logs.items():
        m = metrics(log)
        print(f"{name:<5}{m['e_7_11']:15.4f}{m['e_25_30']:16.4f}{m['e_end']:11.4f}{m['u_peak']:10.3f}"
              f"{m['I_end']:11.4f}")
    print("  (PI/PID integral at 30 s should equal c r - d = 0.9 m/s^2)")
    path = OUT_DIR / "tracking.png"
    plot(logs, path)
    print(f"Figure saved: {path.name}")


if __name__ == "__main__":
    main()
