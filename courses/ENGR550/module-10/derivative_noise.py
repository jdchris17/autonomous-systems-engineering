"""
Module 10, Part 4: derivative action with noisy speed measurements.

Same tracking experiment as Part 2 (r: 0 -> 3 m/s at 1 s, d = -0.3 m/s^2 from 12 s), now with
    y_k = v_k + n_k,  n_k ~ N(0, 0.05^2) m/s  (one fixed noise sequence shared by all runs),
    |u| <= 2 m/s^2, conditional integration on in every run.
Runs: PI; PID with the raw (unfiltered) derivative; PID with the filtered derivative (tau_d = 0.1 s).

Expected derivative noise: D_raw = (n_k - n_{k-1}) / h has SD sqrt(2) sigma_n / h = 3.54 m/s^2, so
Kd D_raw adds ~0.354 m/s^2 of command noise. The first-order filter with alpha = tau/(tau + h) = 0.833
cuts this. Its response to differenced white noise has impulse response (1-alpha)/h * [1, alpha-1,
alpha(alpha-1), ...], whose squared sum gives
    SD(D) = (1 - alpha) sqrt(2 / (1 + alpha)) sigma_n / h = 0.174 * 2.5 = 0.435 m/s^2,
an 8x reduction, so Kd D contributes ~0.044 m/s^2. The P term adds Kp sigma_n = 0.05 m/s^2.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pid_core import GAINS, H_STEP, simulate, ticks, window
from tracking import reference

OUT_DIR = Path(__file__).resolve().parent
SIGMA_N = 0.05
U_LIM = 2.0
SEED = 2
RUNS = {
    "PI": (GAINS["PI"], True),
    "PID, raw derivative": (GAINS["PID"], False),
    "PID, filtered derivative": (GAINS["PID"], True),
}
COLORS = {"PI": "tab:blue", "PID, raw derivative": "tab:red", "PID, filtered derivative": "tab:green"}


def run_all(runs=RUNS):
    r, d = reference()
    noise = np.random.default_rng(SEED).normal(0.0, SIGMA_N, len(r))
    return {name: simulate(g, r, d, u_lim=U_LIM, anti_windup=True, noise=noise, filtered=filt)
            for name, (g, filt) in runs.items()}, noise


def metrics(log):
    t = log["t"]
    w = window(t, 25, 30)
    err = log["r"] - log["v"]
    return {"rmse": np.sqrt(np.mean(err[w]**2)), "u_sd": np.std(log["u"][w]),
            "kd_sd": np.std(GAINS["PID"].Kd * log["D"][w]), "sat": np.mean(np.abs(log["u"][w]) >= U_LIM),
            "rmse_full": np.sqrt(np.mean(err[t >= 1]**2))}


def plot(logs, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw={"width_ratios": [2, 1]})
    t = next(iter(logs.values()))["t"]
    zoom = window(t, 25, 30)
    for name, log in logs.items():
        c = COLORS[name]
        lw = 0.8 if "raw" in name else 1.4
        z = 3 if name == "PI" else 2
        axes[0, 0].plot(t, log["r"] - log["v"], color=c, lw=lw, label=name)
        axes[1, 0].plot(t, log["u"], color=c, lw=lw, label=name, zorder=z)
        axes[0, 1].plot(t[zoom], (log["r"] - log["v"])[zoom], color=c, lw=lw)
        axes[1, 1].plot(t[zoom], log["u"][zoom], color=c, lw=lw, zorder=z)
    axes[0, 0].set_ylim(-0.4, 0.6)
    axes[0, 0].set_title("(a) true tracking error r − v"); axes[0, 0].set_ylabel("m/s")
    axes[1, 0].set_title(f"(b) applied command (limits ±{U_LIM:g})"); axes[1, 0].set_ylabel("m/s²")
    axes[0, 1].set_title("(c) error, 25–30 s"); axes[1, 1].set_title("(d) command, 25–30 s")
    axes[0, 0].legend(loc="upper right", fontsize=9)
    for ax in axes.ravel():
        ax.axhline(0, color="k", lw=0.5)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("time (s)")
    fig.suptitle(f"Module 10: derivative action with speed noise σ = {SIGMA_N} m/s (same noise in all runs)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    logs, noise = run_all()
    raw = np.diff(noise) / H_STEP
    print(f"Raw differentiated noise SD: {np.std(raw):.3f} m/s^2 (theory {np.sqrt(2) * SIGMA_N / H_STEP:.3f}),"
          f" x Kd = {0.1 * np.std(raw):.3f} m/s^2")
    print(f"{'run':<26}{'true RMSE 25-30':>16}{'u SD 25-30':>12}{'Kd*D SD':>9}{'time at limit':>15}"
          f"{'RMSE t>=1 s':>13}")
    for name, log in logs.items():
        m = metrics(log)
        kd = f"{m['kd_sd']:.3f}" if "PID" in name else "-"
        print(f"{name:<26}{m['rmse']:16.4f}{m['u_sd']:12.4f}{kd:>9}{m['sat']:15.1%}{m['rmse_full']:13.4f}")
    path = OUT_DIR / "derivative_noise.png"
    plot(logs, path)
    print(f"Figure saved: {path.name}")


if __name__ == "__main__":
    main()
