"""
Module 12, Parts 3-4: LQI vs PI on the speed plant, with and without actuator limits.

Experiment (30 s, h = 0.02 s, exact speed measurement, v0 = 0, integral states start at zero):
    r = 0 -> 3 m/s at t = 1 s;   d = 0 -> -0.3 m/s^2 at t = 12 s.
Controllers, both with feedforward c r:
    PI:   u = c r + 1.5 (r - v) + I,           I_{k+1}  = I_k + 0.5 h (r - v)
    LQI:  u = c r - K [v - r, xi],             xi_{k+1} = xi_k + h (r - v)
Command is computed from the current integral state, then the state is updated.
Limited runs: u = clip(u_raw, -2, 2), with conditional integration: freeze the integral state when
u_raw is beyond a limit AND the proposed change in the integral contribution (PI: 0.5 h (r - v);
LQI: -k_xi h (r - v)) would push it further beyond that limit.

Prediction for increasing R_cost (made before running):
    Gains: command is more expensive, so both |k_e| and |k_xi| fall (roughly k_xi ~ sqrt(q_xi / R)).
    Recovery: smaller gains -> slower poles -> slower load recovery, larger IAE.
    Effort: feedback effort h sum (u - c r)^2 falls; the total h sum u^2 changes little, since it is
    dominated by the feedforward c r = 0.6 and the 0.3 load compensation that every design must
    eventually supply. Overshoot: with cheap command (R = 0.25) the poles are real and well damped;
    at R = 4 they become complex (damping ratio ~0.89), so a little more overshoot is expected.
    Saturation: higher gains push the raw command further past 2 m/s^2 at the step, so R = 0.25
    should saturate longest; R = 4 may never saturate.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lqi_design import A_PLANT, B_PLANT, C_DRAG, H_STEP, PI_GAINS, designs

OUT_DIR = Path(__file__).resolve().parent
T_END = 30.0
U_LIM = 2.0
BAND = 0.05
N = int(round(T_END / H_STEP))
K_STEP, K_DIST = int(round(1.0 / H_STEP)), int(round(12.0 / H_STEP))
COLORS = {"PI": "k", "LQI R=0.25": "tab:red", "LQI R=1": "tab:blue", "LQI R=4": "tab:green"}


def signals():
    k = np.arange(N)
    return np.where(k >= K_STEP, 3.0, 0.0), np.where(k >= K_DIST, -0.3, 0.0)


def freeze(u_raw, delta, u_lim):
    return u_lim is not None and ((u_raw > u_lim and delta > 0) or (u_raw < -u_lim and delta < 0))


def run_pi(Kp, Ki, u_lim=None):
    r, d = signals()
    v, I = 0.0, 0.0
    log = {key: np.zeros(N) for key in ("v", "u_raw", "u", "integ")}
    for k in range(N):
        u_raw = C_DRAG * r[k] + Kp * (r[k] - v) + I
        u = u_raw if u_lim is None else float(np.clip(u_raw, -u_lim, u_lim))
        log["v"][k], log["u_raw"][k], log["u"][k], log["integ"][k] = v, u_raw, u, I
        dI = Ki * H_STEP * (r[k] - v)
        if not freeze(u_raw, dI, u_lim):
            I += dI
        v = A_PLANT * v + B_PLANT * (u + d[k])
    return finish(log, r)


def run_lqi(K, u_lim=None):
    r, d = signals()
    k_e, k_xi = K
    v, xi = 0.0, 0.0
    log = {key: np.zeros(N) for key in ("v", "u_raw", "u", "integ")}
    for k in range(N):
        u_raw = C_DRAG * r[k] - (k_e * (v - r[k]) + k_xi * xi)
        u = u_raw if u_lim is None else float(np.clip(u_raw, -u_lim, u_lim))
        log["v"][k], log["u_raw"][k], log["u"][k], log["integ"][k] = v, u_raw, u, -k_xi * xi
        dxi = H_STEP * (r[k] - v)
        if not freeze(u_raw, -k_xi * dxi, u_lim):     # direction judged on the contribution -k_xi xi
            xi += dxi
        v = A_PLANT * v + B_PLANT * (u + d[k])
    return finish(log, r)


def finish(log, r):
    log["t"] = np.arange(N) * H_STEP
    log["r"] = r
    return log


def metrics(log, u_lim=None):
    t, v, r, u = log["t"], log["v"], log["r"], log["u"]
    after = t >= 12.0 - 1e-9
    outside = after & (np.abs(v - 3.0) > BAND)
    if outside[-1]:
        recovery = None
    else:
        idx = np.nonzero(outside)[0]
        recovery = t[idx[-1] + 1] - 12.0 if len(idx) else 0.0
    return {
        "overshoot": max(0.0, np.max(v[t < 12.0 - 1e-9]) - 3.0),
        "recovery": recovery,
        "iae": H_STEP * np.sum(np.abs(r - v)),
        "effort": H_STEP * np.sum(u**2),
        "fb_effort": H_STEP * np.sum((u - C_DRAG * r)**2),
        "sat": H_STEP * np.sum(np.abs(log["u_raw"]) > u_lim) if u_lim else 0.0,
        "dip": 3.0 - np.min(v[after]),
    }


def run_all(u_lim=None):
    logs = {"PI": run_pi(*PI_GAINS, u_lim=u_lim)}
    for name, K in designs().items():
        logs[name] = run_lqi(K, u_lim=u_lim)
    return logs


def equivalence_check():
    """LQI with K = [k_e, k_xi] must equal PI with Kp = k_e, Ki = -k_xi (with and without limits)."""
    worst = 0.0
    for K in designs().values():
        for lim in (None, U_LIM):
            a, b = run_lqi(K, lim), run_pi(K[0], -K[1], lim)
            worst = max(worst, np.max(np.abs(a["v"] - b["v"])), np.max(np.abs(a["u"] - b["u"])))
    print(f"Check: LQI(K) == PI(Kp = k_e, Ki = -k_xi), max difference in v and u: {worst:.1e}\n")


def table(logs, u_lim, title):
    print(title)
    print(f"  {'controller':<12}{'Kp':>7}{'Ki':>7}{'overshoot':>11}{'load dip':>10}{'recovery':>10}{'IAE':>8}"
          f"{'effort':>9}{'fb effort':>11}{'saturated':>11}")
    gains = {"PI": PI_GAINS, **{n: (K[0], -K[1]) for n, K in designs().items()}}
    for name, log in logs.items():
        m = metrics(log, u_lim)
        rec = "not reached" if m["recovery"] is None else f"{m['recovery']:.2f} s"
        print(f"  {name:<12}{gains[name][0]:7.3f}{gains[name][1]:7.3f}{m['overshoot']:11.3f}{m['dip']:10.3f}"
              f"{rec:>10}{m['iae']:8.3f}{m['effort']:9.2f}{m['fb_effort']:11.3f}{m['sat']:10.2f}s")
    print()


def plot(logs_free, logs_lim, path):
    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(13, 10))
    for col, (logs, title) in enumerate(((logs_free, "no actuator limit"),
                                         (logs_lim, f"|u| ≤ {U_LIM:g} m/s², conditional integration"))):
        t = logs["PI"]["t"]
        axes[0, col].plot(t, logs["PI"]["r"], color="0.5", ls=":", lw=1, label="reference")
        axes[0, col].fill_between(t, 3 - BAND, 3 + BAND, where=t >= 12, color="tab:olive", alpha=0.25, lw=0)
        for name, log in logs.items():
            ls = "--" if name == "PI" else "-"
            axes[0, col].plot(t, log["v"], color=COLORS[name], ls=ls, lw=1.5, label=name)
            axes[1, col].plot(t, log["u"], color=COLORS[name], ls=ls, lw=1.5)
            axes[2, col].plot(t, log["integ"], color=COLORS[name], ls=ls, lw=1.5)
        if col == 1:
            axes[1, col].axhline(U_LIM, color="0.4", lw=0.8, ls=":")
        axes[0, col].set_title(f"speed — {title}", fontsize=10)
        axes[1, col].set_title("applied command u", fontsize=10)
        axes[2, col].set_title("integral contribution (PI: I;  LQI: −k_ξ ξ)", fontsize=10)
        axes[0, col].set_ylim(-0.2, 3.6)
        axes[1, col].set_ylim(-0.5, 7.5)
        for ax in axes[:, col]:
            ax.axvline(12, color="tab:red", lw=0.6, ls="--")
            ax.grid(alpha=0.3)
        axes[2, col].set_xlabel("time (s)")
    axes[0, 0].legend(loc="lower right", fontsize=9)
    axes[0, 0].set_ylabel("m/s"); axes[1, 0].set_ylabel("m/s²"); axes[2, 0].set_ylabel("m/s²")
    fig.suptitle("Module 12: LQI (three input penalties) vs PI (Kp = 1.5, Ki = 0.5); "
                 "step to 3 m/s at 1 s, load −0.3 m/s² at 12 s")
    fig.tight_layout()
    fig.savefig(path, dpi=115)
    plt.close(fig)


def main():
    equivalence_check()
    free, lim = run_all(None), run_all(U_LIM)
    table(free, None, "Part 4 metrics, no actuator limit")
    table(lim, U_LIM, f"Part 4 metrics, |u| <= {U_LIM:g} m/s^2 with conditional integration")
    path = OUT_DIR / "lqi_vs_pi.png"
    plot(free, lim, path)
    print(f"Figure saved: {path.name}")


if __name__ == "__main__":
    main()
