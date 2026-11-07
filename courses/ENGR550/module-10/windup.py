"""
Module 10, Part 3: integrator windup and conditional-integration anti-windup.

PI (Kp = 1, Ki = 0.5), no disturbance, |u| <= 1 m/s^2, reference
    r = 0 (t < 1), 8 m/s (1 <= t < 10), 1 m/s (t >= 10).
With u at its +1 limit the plant can only reach u_max / c = 5 m/s, so the 8 m/s request is
unreachable: the error stays >= 3 m/s, and ordinary integration keeps adding Ki h e every step.

Why that hurts at 10 s: the stored I is a command the controller "owes". When r drops to 1 m/s the
error becomes negative (about -4 m/s), so Kp e = -4, but I has grown to well over +4, so u_raw
stays POSITIVE and the vehicle keeps accelerating at full throttle for a while, even though it is
already 4 m/s too fast. Only after I has been integrated back down (slowly, at Ki e per second)
does the command reverse and braking start. The longer and harder the saturation, the longer the
delay: that is windup.

Conditional integration stops adding to I while u_raw is beyond u_max and e > 0, since that extra
integral has no effect on the applied command (it is clipped) and only builds up debt. I stays
near the value that was in use when the limit was hit, so when r drops, Kp e immediately pulls
u_raw negative and braking starts at once.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pid_core import C_DRAG, GAINS, simulate, ticks, window

OUT_DIR = Path(__file__).resolve().parent
T_END = 30.0
U_LIM = 1.0
BAND = 0.05
RUNS = {"ordinary integration": False, "conditional integration": True}
COLORS = {"ordinary integration": "tab:red", "conditional integration": "tab:blue"}


def reference():
    k = np.arange(ticks(T_END))
    r = np.where(k >= ticks(10.0), 1.0, np.where(k >= ticks(1.0), 8.0, 0.0))
    return r, np.zeros_like(r)


def run_all():
    r, d = reference()
    return {name: simulate(GAINS["PI"], r, d, u_lim=U_LIM, anti_windup=aw) for name, aw in RUNS.items()}


def settle_time(log, t_drop=10.0, target=1.0, band=BAND):
    """Time after t_drop to enter and REMAIN within target +- band (true speed). None if not reached."""
    t, v = log["t"], log["v"]
    after = t >= t_drop - 1e-9
    outside = after & (np.abs(v - target) > band)
    if outside[-1]:
        return None
    last_out = np.nonzero(outside)[0]
    k = last_out[-1] + 1 if len(last_out) else np.nonzero(after)[0][0]
    return t[k] - t_drop


def metrics(log):
    t = log["t"]
    k10 = ticks(10.0)
    pos_cmd = np.nonzero((t >= 10.0 - 1e-9) & (log["u"] < 0))[0]
    return {"I_at_10": log["I"][k10], "v_at_10": log["v"][k10], "v_peak_after": np.max(log["v"][t >= 10]),
            "brake_delay": t[pos_cmd[0]] - 10.0 if len(pos_cmd) else None,
            "min_after": np.min(log["v"][t >= 10]), "settle": settle_time(log)}


def plot(logs, path):
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 10))
    t = next(iter(logs.values()))["t"]
    axes[0].plot(t, next(iter(logs.values()))["r"], color="k", lw=1, ls=":", label="reference")
    axes[0].axhline(U_LIM / C_DRAG, color="tab:gray", lw=0.8, ls="--")
    axes[0].text(0.3, U_LIM / C_DRAG + 0.15, "max reachable speed u_max / c = 5 m/s", fontsize=8,
                 color="tab:gray")
    axes[0].fill_between(t, 1 - BAND, 1 + BAND, where=t >= 10, color="tab:green", alpha=0.3, lw=0,
                         label="1 ± 0.05 m/s band")
    for name, log in logs.items():
        c = COLORS[name]
        axes[0].plot(t, log["v"], color=c, lw=1.6, label=name)
        axes[1].plot(t, log["u_raw"], color=c, lw=1.0, ls="--", label=f"raw, {name}")
        axes[1].plot(t, log["u"], color=c, lw=1.8, label=f"applied, {name}")
        axes[2].plot(t, log["I"], color=c, lw=1.6, label=name)
    axes[1].axhspan(-U_LIM, U_LIM, color="0.9", zorder=0)
    axes[1].set_ylim(-5, 9)
    for ax in axes:
        ax.axvline(10, color="k", lw=0.6)
        ax.axhline(0, color="k", lw=0.5)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("speed (m/s)"); axes[0].set_title("(a) speed: the 8 m/s request is unreachable")
    axes[1].set_ylabel("command (m/s²)"); axes[1].set_title("(b) raw (dashed) and applied (solid) "
                                                             "command; grey band = actuator range ±1")
    axes[2].set_ylabel("I (m/s²)"); axes[2].set_title("(c) integral contribution")
    axes[2].set_xlabel("time (s)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].legend(loc="upper right", fontsize=8, ncol=2)
    fig.suptitle("Module 10: integrator windup, PI with |u| ≤ 1 m/s²")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    logs = run_all()
    print(f"{'run':<25}{'I at 10 s':>10}{'v at 10 s':>10}{'peak v >10s':>12}{'brake starts':>14}"
          f"{'min v >10s':>11}{'settle in 1+-0.05':>19}")
    for name, log in logs.items():
        m = metrics(log)
        settle = "not reached" if m["settle"] is None else f"{m['settle']:.2f} s"
        brake = "never" if m["brake_delay"] is None else f"+{m['brake_delay']:.2f} s"
        print(f"{name:<25}{m['I_at_10']:10.2f}{m['v_at_10']:10.3f}{m['v_peak_after']:12.3f}{brake:>14}"
              f"{m['min_after']:11.3f}{settle:>19}")
    path = OUT_DIR / "windup.png"
    plot(logs, path)
    print(f"Figure saved: {path.name}")


if __name__ == "__main__":
    main()
