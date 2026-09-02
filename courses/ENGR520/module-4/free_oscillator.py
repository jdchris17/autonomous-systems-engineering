"""module-4 -- free_oscillator.py  (Computational Block I: Free Oscillator)

    m = 1, k = 4, c = 0     ->   omega_0 = sqrt(k/m) = 2 rad/s
    x(0) = 1, v(0) = 0
    T = 2 pi / omega_0 = pi s

Propagate x(t), v(t) with RK4, measure the period, and check energy
conservation:

    K(t) = 1/2 m v^2      U(t) = 1/2 k x^2      E(t) = K + U

With c = 0 the eigenvalues of A are purely imaginary (lambda = +/- i omega_0):
zero real part means zero decay -- undamped oscillation, E constant.
"""

import numpy as np
import matplotlib.pyplot as plt

from state_space import (oscillator_AB, linear_dynamics, eigenvalues,
                         simulate, zero_crossing_period)

M, K, C = 1.0, 4.0, 0.0
X0 = np.array([1.0, 0.0])          # [position, velocity]
N_PERIODS = 6
DT = 1e-3


def main():
    A, B = oscillator_AB(M, K, C)
    w0 = np.sqrt(K / M)
    T_analytic = 2 * np.pi / w0
    lam = eigenvalues(A)

    print("=" * 70)
    print("FREE OSCILLATOR   m x'' + c x' + k x = 0")
    print("=" * 70)
    print(f"m = {M}, k = {K}, c = {C}")
    print(f"A =\n{A}")
    print(f"eigenvalues of A : {lam[0]:.4f}, {lam[1]:.4f}")
    print(f"   -> real part {lam[0].real:+.3e} (decay rate),  "
          f"imag part {abs(lam[0].imag):.4f} rad/s (frequency)")
    print(f"omega_0 = sqrt(k/m) = {w0:.6f} rad/s")
    print(f"analytic period T = 2*pi/omega_0 = {T_analytic:.6f} s")
    print()

    f = linear_dynamics(A, B, u=None)
    t, X = simulate(f, X0, N_PERIODS * T_analytic, DT)
    x, v = X[:, 0], X[:, 1]

    # analytic solution for this IC: x = cos(w0 t), v = -w0 sin(w0 t)
    x_exact = np.cos(w0 * t)
    v_exact = -w0 * np.sin(w0 * t)
    print(f"max |x_num - cos(w0 t)|      : {np.abs(x - x_exact).max():.2e}")
    print(f"max |v_num - (-w0 sin w0 t)| : {np.abs(v - v_exact).max():.2e}")

    T_measured = zero_crossing_period(t, x)
    print(f"measured period (zero crossings) : {T_measured:.6f} s   "
          f"(error {T_measured - T_analytic:+.2e} s)")
    print()

    Kx = 0.5 * M * v**2
    U = 0.5 * K * x**2
    E = Kx + U
    print(f"energy: E(0) = 1/2 k x0^2 = {E[0]:.6f} J")
    print(f"        E drift over {N_PERIODS} periods: "
          f"{np.ptp(E):.2e} J  ({np.ptp(E)/E[0]:.2e} fractional)")
    print("        K and U each swing between 0 and E; their sum is flat.")
    print("=" * 70)

    _plot(t, x, v, Kx, U, E, T_analytic)


def _plot(t, x, v, Kx, U, E, T):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    ax = axes[0, 0]
    ax.plot(t, x, label="x(t)")
    ax.plot(t, v, label="v(t)")
    for n in range(1, int(t[-1] / T) + 1):
        ax.axvline(n * T, color="grey", ls=":", lw=0.7)
    ax.set_xlabel("t (s)"); ax.set_ylabel("state")
    ax.set_title("Free response (grey lines: analytic period)")
    ax.legend(); ax.grid(True)

    ax = axes[0, 1]
    ax.plot(x, v)
    ax.set_xlabel("x"); ax.set_ylabel("v")
    ax.set_title("Phase portrait: a closed ellipse (energy conserved)")
    ax.set_aspect("equal", "box"); ax.grid(True)

    ax = axes[1, 0]
    ax.plot(t, Kx, label="K(t)")
    ax.plot(t, U, label="U(t)")
    ax.plot(t, E, "k", lw=2, label="E = K + U")
    ax.set_xlabel("t (s)"); ax.set_ylabel("energy (J)")
    ax.set_title("Kinetic / potential exchange, total constant")
    ax.legend(); ax.grid(True)

    ax = axes[1, 1]
    ax.plot(t, E - E[0])
    ax.set_xlabel("t (s)"); ax.set_ylabel("E(t) - E(0)  (J)")
    ax.set_title("Energy drift (RK4 truncation error only)")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig("free_oscillator.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
