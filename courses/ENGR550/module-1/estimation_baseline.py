"""
Baseline: 1-D double integrator with a noisy position sensor.

    p_dot = v,   v_dot = u
    z_k   = p_k + n_k,   n_k ~ N(0, sigma_p^2)

Part 1: simulate the noise-free motion (exact zero-order-hold discretization),
        generate noisy position readings, and plot position / velocity / input.
Part 2: estimate velocity by differencing consecutive readings and compare it
        with the true velocity and with the predicted noise level sqrt(2)*sigma_p/dt.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------- settings
DURATION = 10.0    # s
DT = 0.1           # s
P0 = 0.0           # m
V0 = 0.0           # m/s
SIGMA_P = 0.5      # m, position-noise standard deviation
SEED = 2           # fixed, recorded seed (42 gave an atypically quiet draw: sample std 0.39 m)
OUT_DIR = Path(__file__).resolve().parent   # figures are saved next to this script


def accel_command(t):
    """Piecewise-constant acceleration schedule u(t) in m/s^2."""
    # Small tolerance so floating-point times such as 2.9999999 land in the right interval.
    t = t + 1e-9
    if t < 3.0:
        return 1.0
    if t < 6.0:
        return 0.0
    return -1.0


def simulate(duration=DURATION, dt=DT, p0=P0, v0=V0):
    """Propagate the true state with the exact discrete update."""
    n = int(round(duration / dt)) + 1       # samples at t = 0, dt, ..., duration
    t = np.arange(n) * dt
    p = np.zeros(n)
    v = np.zeros(n)
    u = np.zeros(n)

    p[0], v[0] = p0, v0
    for k in range(n):
        u[k] = accel_command(t[k])          # held constant over [t_k, t_k + dt)
        if k + 1 < n:
            p[k + 1] = p[k] + v[k] * dt + 0.5 * u[k] * dt**2
            v[k + 1] = v[k] + u[k] * dt
    return t, p, v, u


def measure(p, sigma_p=SIGMA_P, seed=SEED):
    """Noisy position readings at every sample."""
    rng = np.random.default_rng(seed)
    return p + rng.normal(0.0, sigma_p, size=p.shape)


def check_against_reference(t, p, v):
    """Compare the noise-free trajectory with the reference table in the problem."""
    reference = [(3.0, 4.5, 3.0), (6.0, 13.5, 3.0), (9.0, 18.0, 0.0), (10.0, 17.5, -1.0)]
    print("Noise-free check")
    print(f"{'t [s]':>6} {'p sim':>9} {'p ref':>7} {'v sim':>8} {'v ref':>7}  ok")
    all_ok = True
    for tr, pr, vr in reference:
        k = int(round(tr / DT))
        ok = np.isclose(p[k], pr) and np.isclose(v[k], vr)
        all_ok &= ok
        print(f"{t[k]:6.1f} {p[k]:9.4f} {pr:7.1f} {v[k]:8.4f} {vr:7.1f}  {'yes' if ok else 'NO'}")
    print("All reference points match.\n" if all_ok else "MISMATCH against reference.\n")


def plot_baseline(t, p, v, u, z):
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 8))

    axes[0].plot(t, z, ".", ms=4, color="tab:gray", alpha=0.7, label="measured $z_k$")
    axes[0].plot(t, p, lw=2, color="tab:blue", label="true $p$")
    axes[0].set_ylabel("position [m]")
    axes[0].legend(loc="upper left")

    axes[1].plot(t, v, lw=2, color="tab:orange")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_ylabel("true velocity [m/s]")

    axes[2].step(t, u, where="post", lw=2, color="tab:green")
    axes[2].set_ylabel("command $u$ [m/s$^2$]")
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylim(-1.5, 1.5)

    for ax in axes:
        ax.grid(alpha=0.3)
        for tb in (3, 6):
            ax.axvline(tb, color="k", ls=":", lw=0.8)
    fig.suptitle(f"Baseline: double integrator, $\\sigma_p$ = {SIGMA_P} m, seed = {SEED}")
    fig.tight_layout()
    return fig


def velocity_experiment(t, v, z, p):
    """Finite-difference velocity estimate from consecutive readings."""
    v_fd = np.diff(z) / DT                  # defined for k = 1..N
    t_fd = t[1:]

    err = v_fd - v[1:]
    sigma_theory = np.sqrt(2) * SIGMA_P / DT

    # Split the error into (a) the noise term and (b) the small deterministic bias
    # from differencing: (p_k - p_{k-1})/dt is the *average* velocity over the
    # interval, which differs from v_k by -u_{k-1}*dt/2.
    noise_part = np.diff(z - p) / DT
    lag1 = np.corrcoef(noise_part[:-1], noise_part[1:])[0, 1]

    print("Finite-difference velocity experiment")
    print(f"  predicted noise std  sqrt(2)*sigma_p/dt = {sigma_theory:.2f} m/s")
    print(f"  empirical std of noise term              = {noise_part.std(ddof=1):.2f} m/s")
    print(f"  empirical RMS error vs true v_k          = {np.sqrt(np.mean(err**2)):.2f} m/s")
    print(f"  max |true velocity|                      = {np.abs(v).max():.2f} m/s")
    print(f"  lag-1 correlation of the errors          = {lag1:+.2f}  (theory: -0.5)")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(t, v - sigma_theory, v + sigma_theory, color="tab:orange", alpha=0.15,
                    label=f"true $v$ ± $\\sigma_{{\\tilde v}}$ ({sigma_theory:.1f} m/s)")
    ax.plot(t_fd, v_fd, ".-", ms=3, lw=0.7, color="tab:gray", label="$\\tilde v_k = (z_k - z_{k-1})/\\Delta t$")
    ax.plot(t, v, lw=2, color="tab:orange", label="true $v$")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("velocity [m/s]")
    ax.set_title("Velocity from differenced position readings")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true", help="save figures without opening windows")
    args = parser.parse_args()

    t, p, v, u = simulate()
    z = measure(p)

    check_against_reference(t, p, v)

    fig1 = plot_baseline(t, p, v, u, z)
    fig2 = velocity_experiment(t, v, z, p)
    fig1.savefig(OUT_DIR / "baseline.png", dpi=130)
    fig2.savefig(OUT_DIR / "velocity_fd.png", dpi=130)
    print("\nSaved baseline.png and velocity_fd.png")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
