"""Frequency response of a first-order LTI system: numerically transform
h(t) = (1/tau) exp(-t/tau) u(t) and compare with H(w) = 1 / (1 + j w tau)."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "module-8", "signal_toolkit"))
from transforms import fourier_transform

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
TAUS = [0.01, 0.1, 1.0]
COLORS = ["firebrick", "darkorange", "steelblue"]
OMEGA = np.logspace(-1, 3, 161)                 # rad/s
TAIL = 20                                       # integrate h out to 20 tau (tail ~ 2e-9)


def h_of_t(t, tau):
    return np.exp(-t / tau) / tau


def H_exact(omega, tau):
    return 1.0 / (1.0 + 1j * omega * tau)


def H_numeric(omega, tau):
    dt = min(tau / 200, 1e-4)
    t = np.arange(0.0, TAIL * tau + dt / 2, dt)
    return fourier_transform(t, h_of_t(t, tau), omega)


def crossing(omega, mag, level):
    """Frequency where mag falls through `level` (log-frequency interpolation)."""
    i = np.argmax(mag < level)
    frac = (mag[i - 1] - level) / (mag[i - 1] - mag[i])
    return np.exp(np.log(omega[i - 1]) + frac * (np.log(omega[i]) - np.log(omega[i - 1])))


if __name__ == "__main__":
    print("Numerical H(w) = integral_0^(20 tau) h(t) exp(-j w t) dt  vs.  1 / (1 + j w tau)\n")
    print(f"{'tau (s)':>8}{'max |H_num - H_exact|':>23}{'-3 dB w (rad/s)':>18}{'expected 1/tau':>16}{'-3 dB f (Hz)':>14}{'phase at 1/tau':>16}{'63% rise time (s)':>19}{'bandwidth x tau':>17}")
    num = {}
    for tau in TAUS:
        Hn = H_numeric(OMEGA, tau)
        num[tau] = Hn
        err = np.max(np.abs(Hn - H_exact(OMEGA, tau)))
        wc = crossing(OMEGA, np.abs(Hn), 1 / np.sqrt(2))
        w_ref = 1 / tau
        ph = np.degrees(np.angle(H_numeric(np.array([w_ref]), tau)[0]))
        print(f"{tau:>8g}{err:>23.2e}{wc:>18.3f}{w_ref:>16.3f}{wc / (2 * np.pi):>14.3f}{ph:>16.2f}{tau:>19g}{wc * tau:>17.3f}")

    print(f"\nRoll-off: at 10x the cutoff |H| = {abs(H_exact(10.0, 1.0)):.4f} (-20 dB), at 100x it is {abs(H_exact(100.0, 1.0)):.4f} (-40 dB)")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for tau, c in zip(TAUS, COLORS):
        Hn = num[tau]
        He = H_exact(OMEGA, tau)
        axes[0, 0].semilogx(OMEGA, np.abs(He), color=c, linewidth=2, label=f"exact, tau = {tau:g} s")
        axes[0, 0].semilogx(OMEGA[::6], np.abs(Hn[::6]), "o", color=c, markersize=5, markerfacecolor="none")
        axes[0, 0].plot(1 / tau, 1 / np.sqrt(2), "k^", markersize=7)
        axes[0, 1].semilogx(OMEGA, np.degrees(np.angle(He)), color=c, linewidth=2, label=f"exact, tau = {tau:g} s")
        axes[0, 1].semilogx(OMEGA[::6], np.degrees(np.angle(Hn[::6])), "o", color=c, markersize=5, markerfacecolor="none")
        axes[1, 0].loglog(OMEGA, np.abs(Hn - He) + 1e-18, color=c, linewidth=1.5, label=f"tau = {tau:g} s")
        tt = np.linspace(0, 5, 500)
        axes[1, 1].plot(tt, 1 - np.exp(-tt / tau), color=c, linewidth=2, label=f"tau = {tau:g} s")
    axes[0, 0].plot([], [], "o", color="gray", markerfacecolor="none", label="numerical (every 6th point)")
    axes[0, 0].plot([], [], "k^", label="-3 dB at w = 1/tau")
    axes[0, 0].set_title("Magnitude |H(w)|: bigger tau -> narrower bandwidth")
    axes[0, 0].set_ylabel("|H|")
    axes[0, 1].set_title("Phase angle H(w): 0 to -90 deg, -45 deg at w = 1/tau")
    axes[0, 1].set_ylabel("degrees")
    axes[1, 0].set_title("Numerical error |H_num - H_exact|")
    axes[1, 1].set_title("Time response to a step: bigger tau -> slower")
    axes[1, 1].set_xlabel("t (s)")
    axes[1, 1].set_ylabel("step response")
    for ax in axes[0]:
        ax.set_xlabel("w (rad/s)")
    axes[1, 0].set_xlabel("w (rad/s)")
    for ax in axes.flatten():
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle("First-order LTI system: h(t) = (1/tau) e^{-t/tau} u(t)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "frequency_response.png"), dpi=150)
    print("Saved frequency_response.png")

    print(
        """
What this means
---------------
Integrating the impulse response against e^{-jwt} gives the system's frequency
response, and it matches 1/(1 + j w tau) to numerical precision. That one
formula says how the system treats every input frequency:
  - Magnitude: about 1 at low frequency (slow inputs pass unchanged), falling to
    0.707 (-3 dB) at w = 1/tau, then dropping 20 dB per decade. Fast inputs are
    attenuated: this is a low-pass filter.
  - Phase: 0 at low frequency, -45 deg at w = 1/tau, approaching -90 deg. Faster
    inputs are also delayed more.
  - The time-frequency trade-off: tau is both the time constant and the inverse
    of the bandwidth. Larger tau gives a slower step response (63% at t = tau)
    AND a narrower passband (-3 dB at 1/tau). Bandwidth x tau = 1 in every row.
    A sensor cannot be both fast and heavily smoothing: the same tau that
    filters out noise also delays and rounds off genuine changes."""
    )
