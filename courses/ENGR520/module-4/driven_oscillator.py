"""module-4 -- driven_oscillator.py  (Computational Block III: Driven Oscillator)

Harmonic forcing:

    m x'' + c x' + k x = F0 cos(omega t)

Drive at  omega = 0.25, 0.75, 1.0, 1.5, 3.0  times omega_0, from rest.

Every response splits into two pieces:

  * TRANSIENT   -- the homogeneous solution. Oscillates at the damped natural
                   frequency and decays as exp(-zeta omega_0 t). It depends on
                   the initial conditions and is gone after a few 1/(zeta omega_0).
  * STEADY STATE-- the particular solution. Oscillates forever at the DRIVE
                   frequency omega, with amplitude
                       X(omega) = (F0/m) / sqrt((omega_0^2 - omega^2)^2
                                                 + (2 zeta omega_0 omega)^2)
                   and phase lag  phi = atan2(2 zeta omega_0 omega,
                                              omega_0^2 - omega^2).

The exercise's point: the LARGEST steady-state amplitude is near omega = omega_0
(resonance), not at the highest drive frequency. A fast-wiggling trajectory is
not a large one.
"""

import numpy as np
import matplotlib.pyplot as plt

from state_space import oscillator_AB, linear_dynamics, simulate

M, K, ZETA = 1.0, 4.0, 0.1
F0 = 1.0
W0 = np.sqrt(K / M)
C = 2 * ZETA * np.sqrt(K * M)
RATIOS = [0.25, 0.75, 1.0, 1.5, 3.0]
DT, T_END = 2e-3, 90.0


def steady_state(omega):
    """Analytic steady-state amplitude X and phase lag phi (rad)."""
    X = (F0 / M) / np.hypot(W0**2 - omega**2, 2 * ZETA * W0 * omega)
    phi = np.arctan2(2 * ZETA * W0 * omega, W0**2 - omega**2)
    return X, phi


def main():
    static = F0 / K                                   # zero-frequency deflection
    tau_transient = 1.0 / (ZETA * W0)                  # transient time constant
    wr = W0 * np.sqrt(max(0.0, 1 - 2 * ZETA**2))       # amplitude-resonance freq

    print("=" * 78)
    print("DRIVEN OSCILLATOR   m x'' + c x' + k x = F0 cos(omega t),  from rest")
    print("=" * 78)
    print(f"m={M}, k={K}, zeta={ZETA}  ->  omega_0={W0:.3f} rad/s, c={C:.3f}, F0={F0}")
    print(f"static deflection F0/k          = {static:.4f}")
    print(f"transient time constant 1/(z*w0) = {tau_transient:.2f} s "
          f"(transient ~gone by {4*tau_transient:.0f} s)")
    print(f"amplitude resonance at omega_r  = {wr:.4f} rad/s "
          f"(= {wr/W0:.3f} omega_0)")
    print()
    print(f"{'omega/w0':>9} {'omega':>7} {'X_analytic':>11} {'X_measured':>11} "
          f"{'X/static':>9} {'phase lag':>10}")
    print("-" * 72)

    runs = []
    for ratio in RATIOS:
        omega = ratio * W0
        Xa, phi = steady_state(omega)
        f = linear_dynamics(*oscillator_AB(M, K, C), u=lambda t, w=omega: F0 * np.cos(w * t))
        t, Xstate = simulate(f, np.array([0.0, 0.0]), T_END, DT)
        x = Xstate[:, 0]

        period = 2 * np.pi / omega
        t_start = max(5 * tau_transient, T_END - 6 * period)  # after transient
        tail = t >= t_start
        Xm = 0.5 * (x[tail].max() - x[tail].min())
        print(f"{ratio:>9.2f} {omega:>7.3f} {Xa:>11.4f} {Xm:>11.4f} "
              f"{Xa/static:>9.2f} {np.rad2deg(phi):>9.1f} deg")
        runs.append((ratio, omega, t, x, Xa, phi))

    print()
    print("Note omega = 3 omega_0 wiggles fastest but its amplitude "
          f"({runs[-1][4]:.3f}) is")
    print(f"~{runs[2][4]/runs[-1][4]:.0f}x SMALLER than the resonant case "
          f"({runs[2][4]:.3f}). Near omega_0 the")
    print("push is in phase with the velocity, so it feeds energy in every cycle")
    print("and the amplitude builds until damping losses balance the input. Far")
    print("above omega_0 the mass's inertia wins: it barely responds before the")
    print("force reverses.")
    print("=" * 78)

    _plot_responses(runs, tau_transient)
    _plot_frequency_response(runs, static, wr)


def _plot_responses(runs, tau):
    fig, axes = plt.subplots(len(runs), 1, figsize=(12, 13.5))
    for i, (ax, (ratio, omega, t, x, Xa, phi)) in enumerate(zip(axes, runs)):
        period = 2 * np.pi / omega
        xmax = min(t[-1], max(5 * tau, 12 * period))     # transient + ~12 cycles
        ax.axvspan(0, 4 * tau, color="orange", alpha=0.10,
                   label="transient alive (~4 tau)" if i == 0 else None)
        ax.plot(t, x, lw=0.9, color="tab:blue", label="numerical x(t)")
        ax.plot(t, Xa * np.cos(omega * t - phi), "r--", lw=1,
                label="analytic steady state")
        ax.axhline(Xa, color="0.6", lw=0.7)
        ax.axhline(-Xa, color="0.6", lw=0.7)
        ax.set_xlim(0, xmax)
        ax.set_ylabel("x")
        ax.legend(loc="lower right", fontsize=8, ncol=3)
        ax.grid(True, alpha=0.3)
        tag = "  <-- RESONANCE" if abs(ratio - 1.0) < 1e-9 else ""
        ax.set_title(
            f"omega = {ratio:g} omega_0    steady amplitude X = {Xa:.3f}  "
            f"(= {Xa/(F0/K):.2f} x static F0/k)    phase lag {np.rad2deg(phi):.0f} deg"
            f"{tag}", fontsize=10, loc="left")
    axes[-1].set_xlabel("t (s)")
    fig.suptitle("Driven oscillator: transient vs steady state", fontsize=13)
    fig.text(0.5, 0.945,
             "Blue = full numerical solution from rest.  Red dashed = analytic "
             "steady state  X cos(omega t - phi).  Orange band = first 4 tau, "
             "tau = 1/(zeta*omega_0) = %.0f s,\nwhere the decaying homogeneous "
             "(transient) part is still visible.  After it the blue curve lies "
             "on the red one -- the system has forgotten its initial conditions "
             "and just tracks the drive.\n"
             "Each panel has its own time axis (faster drive -> shorter window). "
             "The omega = omega_0 panel shows the resonant amplitude building up "
             "over ~%.0f cycles before damping caps it." % (tau, 1 / (2 * ZETA)),
             ha="center", va="top", fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 1, 0.925])
    fig.savefig("driven_oscillator_responses.png", dpi=120)


def _plot_frequency_response(runs, static, wr):
    w = np.linspace(0.02, 3.2 * W0, 800)
    X = np.array([steady_state(wi)[0] for wi in w])
    phase = np.array([np.rad2deg(steady_state(wi)[1]) for wi in w])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    ax1.plot(w / W0, X, color="tab:blue")
    ax1.axhline(static, color="0.6", ls=":", lw=1)
    ax1.axvline(wr / W0, color="grey", ls="--", lw=0.8)
    for ratio, omega, t, x, Xa, phi in runs:
        ax1.plot(ratio, Xa, "o", ms=8)
        ax1.annotate(f"{ratio:g} w0", (ratio, Xa), textcoords="offset points",
                     xytext=(6, 4), fontsize=8)
    ax1.set_ylabel("steady-state amplitude  X(omega)")
    ax1.set_title(f"Frequency response  (zeta = {ZETA}):  peak amplification "
                  f"X_max / (F0/k) ~ 1/(2 zeta) = {1/(2*ZETA):.0f}")
    ax1.grid(True, alpha=0.3)
    ax1.annotate("dotted line = static deflection F0/k", (2.6, static),
                 textcoords="offset points", xytext=(0, 6), fontsize=8, color="0.4")
    ax1.text(0.06, X.max() * 0.30, "omega -> 0:\nquasi-static,\nX -> F0/k\n"
             "(spring wins,\nmass keeps up)", fontsize=8)
    ax1.text(1.12, X.max() * 0.80,
             "omega ~ omega_0: RESONANCE\n"
             "force ends up in phase with VELOCITY,\n"
             "so it does positive work every cycle;\n"
             "amplitude grows until damping loss = input",
             fontsize=8)
    ax1.text(2.1, X.max() * 0.28, "omega >> omega_0:\ninertia-limited,\n"
             "X ~ F0/(m omega^2) -> 0\n(force reverses before\nthe mass can respond)",
             fontsize=8)

    ax2.plot(w / W0, phase, color="tab:purple")
    ax2.axvline(wr / W0, color="grey", ls="--", lw=0.8)
    for ratio, omega, t, x, Xa, phi in runs:
        ax2.plot(ratio, np.rad2deg(phi), "o", ms=8)
    ax2.set_xlabel("omega / omega_0")
    ax2.set_ylabel("phase lag (deg)")
    ax2.set_yticks([0, 45, 90, 135, 180])
    ax2.grid(True, alpha=0.3)
    ax2.text(0.05, 20, "in phase (0 deg): displacement follows the force", fontsize=8)
    ax2.text(1.05, 95, "90 deg lag exactly at omega_0:\nforce leads displacement by "
             "a quarter cycle,\nso it aligns with VELOCITY -> max power in", fontsize=8)
    ax2.text(2.0, 150, "180 deg: mass moves opposite the force", fontsize=8)

    fig.tight_layout()
    fig.savefig("driven_oscillator_frequency_response.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
