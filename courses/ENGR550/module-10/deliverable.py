"""
Module 10 deliverable: runs all three experiments, regenerates the figures, the gain-adjustment
check, and the compact results table.

Gain adjustment (one change, justified by the comparison below):
    Starting PI (Kp = 1, Ki = 0.5) overshoots 3 m/s by 0.29 m/s. The derivative term does not help:
    with D taken from the measurement on this FIRST-ORDER plant, -Kd v' just adds to the effective
    inertia ((1 + Kd) v' = ...), which lowers the damping ratio slightly (overshoot 0.29 -> 0.33)
    and adds command noise. Raising Kp to 1.5 adds real damping (the s coefficient c + Kp goes
    1.2 -> 1.7), halving the overshoot (0.12 m/s), cutting the load-step error peak (0.18 -> 0.13
    m/s) and improving noisy tracking, at the cost of 1.5x the command noise (still far below the
    raw-derivative PID). Larger gains (Kp = 2-3) reject the load faster but double or triple the
    command noise and saturate more often; not worth it for these requirements.
    Final gains: Kp = 1.5 1/s, Ki = 0.5 1/s^2, Kd = 0.
"""

import numpy as np

import derivative_noise
import tracking
import windup
from pid_core import GAINS, Gains, simulate, window

FINAL = Gains(1.5, 0.5, 0.0)
CANDIDATES = {
    "PI start (1, 0.5, 0)": GAINS["PI"],
    "PID start (1, 0.5, 0.1)": GAINS["PID"],
    "PI Kp=1.5 (final)": FINAL,
    "PI Kp=2, Ki=1": Gains(2.0, 1.0, 0.0),
    "PI Kp=3, Ki=2": Gains(3.0, 2.0, 0.0),
}


def gain_check():
    r, d = tracking.reference()
    noise = np.random.default_rng(derivative_noise.SEED).normal(0.0, derivative_noise.SIGMA_N, len(r))
    print("Gain-adjustment check (tracking experiment; noise run uses |u| <= 2, conditional integration)")
    print(f"  {'gains (Kp, Ki, Kd)':<25}{'overshoot':>10}{'load-step peak e':>17}{'noisy RMSE 25-30':>18}"
          f"{'u SD 25-30':>12}")
    for name, g in CANDIDATES.items():
        log = simulate(g, r, d)
        t, e = log["t"], log["r"] - log["v"]
        nl = simulate(g, r, d, u_lim=derivative_noise.U_LIM, anti_windup=True, noise=noise)
        m = derivative_noise.metrics(nl)
        print(f"  {name:<25}{log['v'][t < 12].max() - 3:10.3f}{e[t >= 12].max():17.3f}"
              f"{m['rmse']:18.4f}{m['u_sd']:12.4f}")
    print()


def main():
    # figures for the three experiments (starting gains, as specified in the problem set)
    tr = tracking.run_all()
    tracking.plot(tr, tracking.OUT_DIR / "tracking.png")
    wu = windup.run_all()
    windup.plot(wu, windup.OUT_DIR / "windup.png")
    dn, _ = derivative_noise.run_all()
    derivative_noise.plot(dn, derivative_noise.OUT_DIR / "derivative_noise.png")

    gain_check()

    # final-gain versions of each experiment
    r, d = tracking.reference()
    tr["PI final (1.5, 0.5)"] = simulate(FINAL, r, d)
    rw, dw = windup.reference()
    for name, aw in windup.RUNS.items():
        wu[f"{name}, final gains"] = simulate(FINAL, rw, dw, u_lim=windup.U_LIM, anti_windup=aw)
    dn.update(derivative_noise.run_all({"PI final (1.5, 0.5)": (FINAL, True)})[0])

    print("Deliverable table")
    print("  Tracking experiment: mean tracking error r - v (m/s)")
    for name, log in tr.items():
        m = tracking.metrics(log)
        print(f"    {name:<28} 7-11 s: {m['e_7_11']:+.4f}    25-30 s: {m['e_25_30']:+.4f}")
    print("  Windup experiment: time after the 10 s drop to enter and stay within 1 +- 0.05 m/s")
    for name, log in wu.items():
        s = windup.settle_time(log)
        print(f"    {name:<44} {'not reached' if s is None else f'{s:.2f} s'}")
    print("  Noise experiment (25-30 s)")
    for name, log in dn.items():
        m = derivative_noise.metrics(log)
        print(f"    {name:<28} true-speed RMSE {m['rmse']:.4f} m/s    applied-command SD {m['u_sd']:.4f} m/s^2")
    print(f"\nFinal gains: Kp = {FINAL.Kp} 1/s, Ki = {FINAL.Ki} 1/s^2, Kd = {FINAL.Kd}")


if __name__ == "__main__":
    main()
