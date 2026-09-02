"""module-4 -- damped_oscillator.py  (Computational Block II: Damping)

Generate four systems by damping ratio, not by guessing c:

    zeta = c / (2 sqrt(k m))     ->     c = 2 zeta sqrt(k m)

    zeta = 0.1  0.5  1.0  2.0

Same m, k, and initial displacement as Block I. Classify each response from
the eigenvalues of A:

    zeta < 1   underdamped     complex pair, decaying oscillation
    zeta = 1   critically damped   repeated real pole at -omega_0
    zeta > 1   overdamped      two real poles, no oscillation

Question: why isn't "more damping" the same as "faster settling"?
Answer falls out of where the poles go -- printed at the end.
"""

import numpy as np
import matplotlib.pyplot as plt

from state_space import oscillator_AB, linear_dynamics, eigenvalues, simulate

M, K = 1.0, 4.0
X0 = np.array([1.0, 0.0])
ZETAS = [0.1, 0.5, 1.0, 2.0]
DT, T_END = 1e-3, 30.0
SETTLE_TOL = 0.02                    # +/- 2% of the initial displacement


def classify(zeta):
    if zeta < 1 - 1e-9:
        return "underdamped"
    if zeta > 1 + 1e-9:
        return "overdamped"
    return "critically damped"


def settling_time(t, x, tol):
    """Last time |x| leaves the +/- tol band, i.e. when it has settled for good."""
    outside = np.where(np.abs(x) > tol * abs(X0[0]))[0]
    return t[outside[-1] + 1] if len(outside) and outside[-1] + 1 < len(t) else t[-1]


def main():
    w0 = np.sqrt(K / M)
    sqrt_km = np.sqrt(K * M)

    print("=" * 78)
    print("DAMPED OSCILLATOR   c = 2 zeta sqrt(k m),  same x0 for all")
    print("=" * 78)
    print(f"m = {M}, k = {K}  ->  omega_0 = {w0:.4f} rad/s,  2 sqrt(km) = "
          f"{2*sqrt_km:.4f}")
    print()
    header = (f"{'zeta':>5} {'c':>7} {'eigenvalues of A':>28} {'type':>18} "
             f"{'max reverse x':>14} {'t_settle(2%)':>13}")
    print(header)
    print("-" * len(header))

    runs = []
    for zeta in ZETAS:
        c = 2 * zeta * sqrt_km
        A, B = oscillator_AB(M, K, c)
        lam = eigenvalues(A)
        t, X = simulate(linear_dynamics(A, B), X0, T_END, DT)
        x = X[:, 0]
        ts = settling_time(t, x, SETTLE_TOL)
        reverse = x.min()               # most negative excursion (overshoot past 0)

        lam_str = (f"{lam[0].real:+.3f}{lam[0].imag:+.3f}j, "
                   f"{lam[1].real:+.3f}{lam[1].imag:+.3f}j")
        print(f"{zeta:>5.1f} {c:>7.3f} {lam_str:>28} {classify(zeta):>18} "
              f"{reverse:>14.4f} {ts:>13.3f}")
        runs.append((zeta, c, lam, t, X, ts))

    # settling time swept in zeta, to show it is not monotonic
    # (coarser step / bigger dt here -- we only need the trend, not precision)
    print("\nsweeping settling time vs zeta ...", flush=True)
    zsweep = np.concatenate([np.linspace(0.05, 0.99, 40),
                             np.linspace(1.0, 5.0, 45)])
    ts_sweep = np.empty_like(zsweep)
    for i, z in enumerate(zsweep):
        A, B = oscillator_AB(M, K, 2 * z * sqrt_km)
        tt, XX = simulate(linear_dynamics(A, B), X0, 25.0, 3e-3)
        ts_sweep[i] = settling_time(tt, XX[:, 0], SETTLE_TOL)
    z_best = zsweep[np.argmin(ts_sweep)]

    print()
    print(f"settling time is minimised near zeta = {z_best:.2f} "
          f"(t_settle = {ts_sweep.min():.2f} s),")
    print(f"and is LARGER at zeta = 2.0 (t_settle = {runs[-1][5]:.2f} s) than at "
          f"zeta = 1.0 (t_settle = {runs[2][5]:.2f} s).")
    print()
    print("WHY 'more damping' != 'faster settling':")
    print("  For zeta > 1 the poles are real: lambda = -omega_0 (zeta -/+ sqrt(zeta^2 - 1)).")
    print("  One pole races off to -infinity, but the OTHER one,")
    print("      lambda_slow = -omega_0 (zeta - sqrt(zeta^2 - 1))  ~  -omega_0 / (2 zeta),")
    print("  drifts back toward the origin as zeta grows. The response is the sum")
    print("  of two decaying exponentials and the slow one dominates, so its time")
    print("  constant 1/|lambda_slow| GROWS with zeta. Physically: a heavily damped")
    print("  mass is so resisted that it creeps back to equilibrium.")
    print("  Fastest return is at (near) critical damping, where both poles sit as")
    print("  far into the left half-plane as they can jointly reach, -omega_0.")
    print("=" * 78)

    _plot(runs, w0, zsweep, ts_sweep, z_best)


def _plot(runs, w0, zsweep, ts_sweep, z_best):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    for zeta, c, lam, t, X, ts in runs:
        ax.plot(t, X[:, 0], label=f"zeta = {zeta}")
    ax.axhspan(-0.02, 0.02, color="grey", alpha=0.2)
    ax.set_xlim(0, 20)
    ax.set_xlabel("t (s)"); ax.set_ylabel("x(t)")
    ax.set_title("Free response from the same x0 (grey band = +/-2%)")
    ax.legend(); ax.grid(True)

    ax = axes[0, 1]
    for zeta, c, lam, t, X, ts in runs:
        ax.plot(X[:, 0], X[:, 1], label=f"zeta = {zeta}")
    ax.plot(1, 0, "ko", ms=4)
    ax.set_xlabel("x"); ax.set_ylabel("v")
    ax.set_title("Phase portraits: spiral (zeta<1) vs direct decay (zeta>=1)")
    ax.legend(fontsize=8); ax.grid(True)

    ax = axes[1, 0]
    for zeta, c, lam, t, X, ts in runs:
        ax.plot(lam.real, lam.imag, "o", ms=9, label=f"zeta = {zeta}")
    th = np.linspace(np.pi/2, 3*np.pi/2, 100)
    ax.plot(w0*np.cos(th), w0*np.sin(th), "k--", lw=0.8)   # |lambda| = omega_0 arc
    ax.axvline(0, color="0.6", lw=0.8); ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("Re(lambda)  (decay rate)")
    ax.set_ylabel("Im(lambda)  (frequency)")
    ax.set_title("Pole migration: as zeta grows, one pole -> origin")
    ax.legend(fontsize=8); ax.grid(True)

    ax = axes[1, 1]
    ax.plot(zsweep, ts_sweep)
    ax.axvline(1.0, color="grey", ls=":", lw=1)
    ax.plot(z_best, ts_sweep.min(), "r*", ms=13, label=f"min near zeta={z_best:.2f}")
    for zeta, c, lam, t, X, ts in runs:
        ax.plot(zeta, ts, "ko", ms=5)
    ax.set_xlabel("damping ratio zeta")
    ax.set_ylabel("2% settling time (s)")
    ax.set_title("Settling time vs zeta -- not monotonic")
    ax.legend(); ax.grid(True)

    fig.tight_layout()
    fig.savefig("damped_oscillator.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
