"""Numerical Fourier transform of a rectangular pulse, compared with the
expected sinc shape, then how width and delay change |X| and the phase."""

import os

import matplotlib.pyplot as plt
import numpy as np

from transforms import fourier_transform

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DT = 2e-4
T = np.arange(-2.0, 2.0 + DT / 2, DT)          # 4 s window, 20001 samples
OMEGA = np.linspace(-40.0, 40.0, 2001)         # rad/s
MASK_LEVEL = 0.02                              # phase is meaningless where |X| is this small


def rect_pulse(t, width, center=0.0):
    """1 inside the pulse, 0 outside, 0.5 exactly on the edges (midpoint convention)."""
    x = (np.abs(t - center) < width / 2).astype(float)
    x[np.isclose(np.abs(t - center), width / 2, atol=1e-9)] = 0.5
    return x


def rect_transform(omega, width, center=0.0):
    """Exact X(w) = width * sinc(w width / 2 pi) * exp(-j w center)."""
    return width * np.sinc(omega * width / (2 * np.pi)) * np.exp(-1j * omega * center)


def masked_phase(X, level=MASK_LEVEL, cut=-np.pi):
    """Phase of X wrapped into [cut, cut + 2 pi), hidden where |X| < level.

    For a real X the phase is exactly 0 or pi, and rounding noise in the sign of
    a ~1e-13 imaginary part would flip pi to -pi. Moving the cut to -pi/2 keeps
    both values on one side of it.
    """
    phase = np.mod(np.angle(X) - cut, 2 * np.pi) + cut
    phase[np.abs(X) < level] = np.nan
    return phase


if __name__ == "__main__":
    tau = 1.0
    X = fourier_transform(T, rect_pulse(T, tau), OMEGA)
    X_exact = rect_transform(OMEGA, tau)

    print(f"Rectangular pulse, width {tau:g} s, amplitude 1, sampled every {DT:g} s ({len(T)} samples)")
    print(f"Expected: X(w) = tau * sinc(w tau / 2 pi) = 2 sin(w tau / 2) / w\n")
    print(f"  X(0)              numeric = {X[np.argmin(abs(OMEGA))].real:.6f}   exact = {tau:.6f}  (the area under the pulse)")
    print(f"  max |X - X_exact| over the grid = {np.max(np.abs(X - X_exact)):.2e}")
    print(f"  max |imag X| (exact X is real)  = {np.max(np.abs(X.imag)):.2e}")

    zeros = 2 * np.pi * np.arange(1, 7) / tau
    idx = [np.argmin(np.abs(OMEGA - z)) for z in zeros]
    print("\n  |X| at the frequency-grid point nearest each expected zero, w = 2 pi n / tau")
    print("  (not exactly zero because the grid step is 0.04 rad/s):")
    print("      n   w = 2 pi n/tau     |X| numeric")
    for n, i in enumerate(idx, start=1):
        print(f"    {n:>3}   {OMEGA[i]:>13.3f}   {abs(X[i]):>13.2e}")
    side = np.abs(X)[(OMEGA > 2 * np.pi) & (OMEGA < 4 * np.pi)].max()
    print(f"\n  First sidelobe: {side:.4f}  ({20 * np.log10(side / tau):.1f} dB below the peak)")

    mag_err = np.max(np.abs(np.abs(X) - np.abs(X_exact)))
    ph_num = masked_phase(X, cut=-np.pi / 2)
    ph_ref = masked_phase(X_exact, cut=-np.pi / 2)
    ph_diff = np.nanmax(np.abs(np.exp(1j * ph_num) - np.exp(1j * ph_ref)))
    print(f"  max ||X| - |X_exact||           = {mag_err:.2e}")
    print(f"  phase mismatch where |X| >= {MASK_LEVEL}    = {ph_diff:.2e} (max |e^(j phase) difference|)")

    # Figure 1: |X| and phase for the centered pulse.
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    axes[0].plot(OMEGA, np.abs(X), color="steelblue", linewidth=2, label="numerical |X(w)|")
    axes[0].plot(OMEGA, np.abs(X_exact), "r--", linewidth=1, label="exact  tau |sinc(w tau / 2 pi)|")
    axes[0].plot(zeros, np.zeros_like(zeros), "kv", markersize=6, label="w = 2 pi n / tau")
    axes[0].set_ylabel("|X(w)|")
    axes[0].set_title("Magnitude: the sinc shape")
    axes[0].legend(fontsize=8)
    axes[1].plot(OMEGA, ph_num, ".", color="steelblue", markersize=3, label="numerical")
    axes[1].plot(OMEGA, ph_ref, "r-", linewidth=0.8, alpha=0.6, label="exact")
    axes[1].set_yticks([-np.pi, 0, np.pi])
    axes[1].set_yticklabels(["-$\\pi$", "0", "$\\pi$"])
    axes[1].set_ylabel("angle X(w)")
    axes[1].set_title(f"Phase (hidden where |X| < {MASK_LEVEL}): 0 or $\\pi$, since X is real and its sign alternates")
    axes[1].legend(fontsize=8)
    axes[2].semilogy(OMEGA, np.abs(X - X_exact) + 1e-18, color="darkorange")
    axes[2].set_ylabel("|X - X_exact|")
    axes[2].set_xlabel("w (rad/s)")
    axes[2].set_title("Numerical error (trapezoid rule, 5000 samples across the pulse)")
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "numerical_transform.png"), dpi=150)

    # Figure 2: width and delay.
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for width, c in [(0.25, "firebrick"), (0.5, "darkorange"), (1.0, "steelblue"), (2.0, "darkgreen")]:
        Xw = fourier_transform(T, rect_pulse(T, width), OMEGA)
        axes[0].plot(OMEGA, np.abs(Xw), color=c, linewidth=1.6, label=f"width {width:g} s")
    axes[0].set_title("Narrower pulse -> wider, lower spectrum")
    axes[0].set_xlabel("w (rad/s)")
    axes[0].set_ylabel("|X(w)|")
    axes[0].legend()

    t0 = 0.75
    Xd = fourier_transform(T, rect_pulse(T, tau, center=t0), OMEGA)
    axes[1].plot(OMEGA, np.abs(X), color="lightgray", linewidth=4, label="centered pulse")
    axes[1].plot(OMEGA, np.abs(Xd), color="steelblue", linewidth=1.4, label=f"delayed by {t0:g} s")
    axes[1].set_title("Delay does not change |X|")
    axes[1].set_xlabel("w (rad/s)")
    axes[1].legend()

    axes[2].plot(OMEGA, masked_phase(Xd), ".", color="steelblue", markersize=3, label="numerical")
    axes[2].plot(OMEGA, masked_phase(rect_transform(OMEGA, tau, t0)), "r-", linewidth=0.8, alpha=0.6, label="exact")
    axes[2].set_title(f"Delay adds phase -w t0 (t0 = {t0:g} s), wrapped to (-$\\pi$, $\\pi$]")
    axes[2].set_xlabel("w (rad/s)")
    axes[2].set_ylabel("angle X(w)")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "numerical_transform_width_delay.png"), dpi=150)

    d_ph = np.nanmax(np.abs(np.exp(1j * masked_phase(Xd)) - np.exp(1j * masked_phase(rect_transform(OMEGA, tau, t0)))))
    print(f"\nDelayed pulse (t0 = {t0:g} s): max ||X_delayed| - |X||     = {np.max(np.abs(np.abs(Xd) - np.abs(X))):.2e}")
    print(f"                                 phase mismatch vs exact = {d_ph:.2e}")
    print("Saved numerical_transform.png and numerical_transform_width_delay.png")

    print(
        """
What this means
---------------
Integrating x(t) e^{-jwt} for every w in a frequency array turns the pulse
into a function of frequency, X(w), and it lands on the expected sinc:
  - X(0) is the pulse's area (1 x 1 = 1): zero frequency = the average/DC content.
  - |X| falls to zero at w = 2 pi n / tau, and the lobes between the zeros
    shrink slowly. The first sidelobe is only about 13 dB below the peak, because
    the pulse has sharp edges and sharp edges need high frequencies.
  - The phase is 0 or pi only: the pulse is symmetric about t = 0, so X is real,
    and the sign flip in each sinc lobe shows up as a pi jump. The phase is
    undefined where |X| is ~0, so it is hidden there.
  - Width trades against bandwidth: a pulse 4x narrower has a spectrum 4x wider
    (and lower). Short in time means spread in frequency, and vice versa.
  - Delaying the pulse leaves |X| unchanged and adds a straight-line phase
    -w t0: magnitude says WHICH frequencies are present, phase says WHEN.
The Fourier series handled periodic signals with a list of discrete a_k; this is
the same integral for a single finite pulse, where the 'list' becomes a
continuous function of frequency."""
    )
