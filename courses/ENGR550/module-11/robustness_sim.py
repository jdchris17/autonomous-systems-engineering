"""
Module 11, Parts 3 and 4: simulate altered plants and command delay; closed-loop poles; noise.

Simulator (Module 10's PI loop, extended):
    v_{k+1} = a v_k + (1 - a)/c (beta u_{k-m} + d_k),   a = e^{-c h}
    u_k = Kp e_k + I_k,  I_{k+1} = I_k + Ki h e_k,  e_k = r_k - y_k,  y_k = v_k + n_k
    A FIFO buffer of length m holds commands not yet applied; it starts full of zeros, so the
    plant receives u_{k-m} (and 0 for k < m).
No noise, disturbance or actuator limit in Part 3; reference 1 m/s from t = 1 s; 30 s.

Metrics: overshoot = max v - 1 (m/s); IAE = h sum |r_k - v_k| over the run; settling time = time
after the step at which v enters and stays inside 1 +- 0.02 m/s ("not settled" if it is outside
at the end). Stability: largest root magnitude of z^m (z-a)(z-1) + b[Kp(z-1) + Ki h].
"""

from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from loop_analysis import margins, predicted_command_sd
from loop_models import CONTROLLERS, H_STEP, char_poly, delay_samples, max_pole, plant_coeffs

OUT_DIR = Path(__file__).resolve().parent
T_END, T_STEP = 30.0, 1.0
BAND = 0.02
SIGMA_N, SEED = 0.05, 2
PLANTS = {
    "nominal":            (0.2, 1.0, 0.0),
    "delayed":            (0.2, 1.0, 0.4),
    "more delayed":       (0.2, 1.0, 0.6),
    "altered plant":      (0.1, 1.5, 0.0),
    "altered + delayed":  (0.1, 1.5, 0.4),
}
COLORS = {"baseline": "tab:blue", "faster": "tab:red"}


def simulate(ctrl, c, beta, m, noise=None):
    n = int(round(T_END / H_STEP))
    k_step = int(round(T_STEP / H_STEP))
    a, b = plant_coeffs(c, beta)
    r = np.where(np.arange(n) >= k_step, 1.0, 0.0)
    noise = np.zeros(n) if noise is None else noise
    buf = deque([0.0] * m)
    v, I = 0.0, 0.0
    log = {key: np.zeros(n) for key in ("v", "u", "u_applied")}
    for k in range(n):
        e = r[k] - (v + noise[k])
        u = ctrl.Kp * e + I
        I += ctrl.Ki * H_STEP * e
        buf.append(u)
        u_applied = buf.popleft()          # u_{k-m}
        log["v"][k], log["u"][k], log["u_applied"][k] = v, u, u_applied
        v = a * v + b * u_applied          # b = beta (1-a)/c, d = 0
    log["t"] = np.arange(n) * H_STEP
    log["r"] = r
    return log


def metrics(log):
    t, v, r = log["t"], log["v"], log["r"]
    iae = H_STEP * np.sum(np.abs(r - v))
    after = t >= T_STEP - 1e-9
    outside = after & (np.abs(v - 1.0) > BAND)
    if outside[-1] or not np.all(np.isfinite(v)):
        settle = None
    else:
        idx = np.nonzero(outside)[0]
        settle = t[idx[-1] + 1] - T_STEP if len(idx) else 0.0
    return {"overshoot": np.max(v) - 1.0, "iae": iae, "settle": settle}


def part3():
    print("Part 3: step response (1 m/s at 1 s), closed-loop poles, and margins on each plant")
    print(f"  {'plant':<19}{'c':>5}{'beta':>5}{'delay':>6}  {'ctrl':<9}{'overshoot':>10}{'IAE':>9}"
          f"{'settle':>12}{'max|pole|':>11}{'PM':>8}{'GM':>8}   verdict")
    logs = {}
    for pname, (c, beta, tau) in PLANTS.items():
        m = delay_samples(tau)
        for cname, ctrl in CONTROLLERS.items():
            log = simulate(ctrl, c, beta, m)
            logs[(pname, cname)] = log
            mt = metrics(log)
            rho = max_pole(ctrl, m, c, beta)
            mg = margins_for(ctrl, m, c, beta)
            settle = "not settled" if mt["settle"] is None else f"{mt['settle']:.2f} s"
            pm = "-" if mg["pm"] is None else f"{mg['pm']:.1f}deg"
            gm = "none" if mg["gm"] is None else f"{mg['gm']:.1f}dB"
            verdict = "stable" if rho < 1 else "UNSTABLE"
            ov = f"{mt['overshoot']:.3f}" if abs(mt["overshoot"]) < 1e3 else f"{mt['overshoot']:.1e}"
            iae = f"{mt['iae']:.3f}" if mt["iae"] < 1e4 else f"{mt['iae']:.1e}"
            print(f"  {pname:<19}{c:5.1f}{beta:5.1f}{tau:6.1f}  {cname:<9}{ov:>10}{iae:>9}{settle:>12}"
                  f"{rho:11.4f}{pm:>8}{gm:>8}   {verdict}")
    print()
    return logs


def margins_for(ctrl, m, c, beta):
    """Margins on an altered plant: reuse loop_analysis.margins by temporarily changing defaults."""
    import loop_analysis
    import loop_models
    orig = loop_analysis.L_z
    loop_analysis.L_z = lambda z, ct, mm=0: loop_models.L_z(z, ct, mm, c, beta)
    try:
        return margins(ctrl, m)
    finally:
        loop_analysis.L_z = orig


def poly_check():
    """Roots of the characteristic polynomial satisfy 1 + L(z) = 0."""
    from loop_models import L_z
    worst = 0.0
    for (c, beta, tau) in PLANTS.values():
        m = delay_samples(tau)
        for ctrl in CONTROLLERS.values():
            for z in np.roots(char_poly(ctrl, m, c, beta)):
                if abs(z - 1) > 1e-6 and abs(z) > 1e-6:
                    worst = max(worst, abs(1 + L_z(z, ctrl, m, c, beta)) / max(1, abs(L_z(z, ctrl, m, c, beta))))
    print(f"Check: roots of the characteristic polynomial satisfy 1 + L(z) = 0 (max rel. residual {worst:.1e})\n")


def plot_steps(logs, path):
    fig, axes = plt.subplots(len(PLANTS), 1, sharex=True, figsize=(11, 13))
    for ax, (pname, (c, beta, tau)) in zip(axes, PLANTS.items()):
        ax.fill_between([T_STEP, T_END], 1 - BAND, 1 + BAND, color="tab:green", alpha=0.25, lw=0)
        ax.plot(logs[(pname, "baseline")]["t"], logs[(pname, "baseline")]["r"], color="k", ls=":", lw=1)
        for cname in CONTROLLERS:
            log = logs[(pname, cname)]
            ax.plot(log["t"], log["v"], color=COLORS[cname], lw=1.5,
                    label=f"{cname} (Kp={CONTROLLERS[cname].Kp:g}, Ki={CONTROLLERS[cname].Ki:g})")
        ax.set_ylim(-0.6, 2.2)
        ax.set_ylabel("speed (m/s)")
        unstable = [cn for cn, ct in CONTROLLERS.items() if max_pole(ct, delay_samples(tau), c, beta) > 1]
        note = f"   — {', '.join(unstable)}: UNSTABLE (growing, off scale)" if unstable else ""
        ax.set_title(f"{pname}: c = {c}, β = {beta}, delay {tau} s{note}", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=9)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Module 11: 1 m/s step on nominal, delayed and altered plants (green: 1 ± 0.02 band)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def part4_noise():
    print("Part 4: nominal plant, no delay, measurement noise SD 0.05 m/s (same sequence for both)")
    n = int(round(T_END / H_STEP))
    noise = np.random.default_rng(SEED).normal(0.0, SIGMA_N, n)
    t = np.arange(n) * H_STEP
    w = t >= 25.0 - 1e-9
    for cname, ctrl in CONTROLLERS.items():
        log = simulate(ctrl, 0.2, 1.0, 0, noise)
        err = log["r"] - log["v"]
        print(f"  {cname:<9} command SD 25-30 s = {np.std(log['u'][w]):.4f} m/s^2"
              f"  (predicted from |CS| by Parseval: {predicted_command_sd(ctrl, SIGMA_N):.4f};"
              f" Kp x sigma = {ctrl.Kp * SIGMA_N:.3f})   true-speed RMSE 25-30 s = {np.sqrt(np.mean(err[w]**2)):.4f} m/s")


def main():
    poly_check()
    logs = part3()
    plot_steps(logs, OUT_DIR / "step_responses.png")
    part4_noise()
    print("\nFigure saved: step_responses.png")


if __name__ == "__main__":
    main()
