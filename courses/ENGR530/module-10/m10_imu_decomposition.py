"""Engineering challenge: decompose the IMU signal into Fourier components,
then filter by scaling each component with a frequency response H."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "module-8", "signal_toolkit"))
from fourier import fourier_coefficients, fourier_synthesis

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
T0 = 2.0                # compatible period: 0.5 Hz has period 2 s, 8 Hz fits exactly 16 times
F0 = 1 / T0             # fundamental 0.5 Hz, so 0.5 Hz is k = 1 and 8 Hz is k = 16
N = 2000                # 1 kHz sampling over one period
FC = 2.0                # filter cutoff (Hz), between the motion (0.5) and the vibration (8)
ORDER = 4


def butterworth_lowpass(f, fc=FC, order=ORDER):
    """Complex frequency response H(j 2 pi f) of an analog Butterworth low-pass (DC gain 1)."""
    s = 1j * 2 * np.pi * np.asarray(f, dtype=float)
    wc = 2 * np.pi * fc
    k = np.arange(1, order + 1)
    poles = wc * np.exp(1j * np.pi * (2 * k + order - 1) / (2 * order))
    H = np.ones_like(s, dtype=complex)
    for p in poles:
        H = H * (-p) / (s - p)
    return H


def filter_coefficients(coeffs, H_of_f):
    """Frequency-domain filtering: b_k = H(j k w0) * a_k."""
    return {k: H_of_f(k * F0) * a for k, a in coeffs.items()}


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


if __name__ == "__main__":
    t = np.linspace(0.0, T0, N + 1)
    motion = 20 * np.sin(2 * np.pi * 0.5 * t)
    vibration = 3 * np.sin(2 * np.pi * 8 * t)
    omega = motion + vibration

    # 1. Identify the Fourier components.
    a = fourier_coefficients(t, omega, T0, 40)
    print("FOURIER COMPONENTS of w(t) = 20 sin(2 pi 0.5 t) + 3 sin(2 pi 8 t), analyzed over T0 = 2 s")
    print(f"{'k':>4}{'freq (Hz)':>11}{'a_k':>22}{'2|a_k|':>9}{'angle (deg)':>13}   meaning")
    for k, meaning in [(1, "desired platform motion"), (16, "vibration")]:
        print(f"{k:>4}{k * F0:>11g}{a[k]:>22.4f}{2 * abs(a[k]):>9.3f}{np.degrees(np.angle(a[k])):>13.1f}   {meaning}")
    others = max(abs(a[k]) for k in a if abs(k) not in (1, 16))
    print(f"  every other |a_k| <= {others:.1e}")
    print("  (a sine has a_k = amplitude/(2j): magnitude A/2 at angle -90 deg)\n")

    # 2. A system that strongly attenuates 8 Hz and largely preserves 0.5 Hz.
    H = butterworth_lowpass
    print(f"SYSTEM: {ORDER}th-order Butterworth low-pass, cutoff {FC:g} Hz")
    for f in (0.5, 2.0, 8.0):
        print(f"  |H({f:g} Hz)| = {abs(H(f)):.4f}   angle = {np.degrees(np.angle(H(f))):7.2f} deg")

    b = filter_coefficients(a, H)
    y = fourier_synthesis(b, F0, t)

    y_theory = (20 * abs(H(0.5)) * np.sin(2 * np.pi * 0.5 * t + np.angle(H(0.5)))
                + 3 * abs(H(8.0)) * np.sin(2 * np.pi * 8 * t + np.angle(H(8.0))))
    print(f"\nOUTPUT COEFFICIENTS: 2|b_1| = {2 * abs(b[1]):.4f} (was 20),  2|b_16| = {2 * abs(b[16]):.4f} (was 3)")
    print(f"  0.5 Hz component kept: {abs(b[1]) / abs(a[1]) * 100:.2f}%   8 Hz component kept: {abs(b[16]) / abs(a[16]) * 100:.3f}%")
    print(f"  max |reconstruction - closed-form 20|H1| sin(..) + 3|H16| sin(..)| = {np.max(np.abs(y - y_theory)):.1e}")
    # H is complex: |H| scales each component, angle(H) delays it. A real (causal)
    # filter delays the 0.5 Hz motion; using |H| alone is the zero-phase (offline) version.
    delay = -np.angle(H(0.5)) / (2 * np.pi * 0.5)
    motion_delayed = 20 * np.sin(2 * np.pi * 0.5 * (t - delay))
    H_zero_phase = lambda f: abs(H(f))
    y_zero = fourier_synthesis(filter_coefficients(a, H_zero_phase), F0, t)
    print(f"\n  the phase of H delays the 0.5 Hz motion by {delay:.3f} s")
    print(f"  RMSE(input,            true motion)          = {rmse(omega, motion):.4f}   (the vibration)")
    print(f"  RMSE(causal output,    true motion)          = {rmse(y, motion):.4f}   (dominated by the 0.21 s delay)")
    print(f"  RMSE(causal output,    motion delayed 0.21 s) = {rmse(y, motion_delayed):.4f}")
    print(f"  RMSE(zero-phase output, true motion)          = {rmse(y_zero, motion):.4f}")

    # 3. Put the noise back: a real record, filtered the same way.
    rng = np.random.default_rng(0)
    noisy = omega + rng.normal(0.0, 2.0, size=len(t))
    a_noisy = fourier_coefficients(t, noisy, T0, 400)
    y_noisy = fourier_synthesis(filter_coefficients(a_noisy, H), F0, t)
    y_noisy_zero = fourier_synthesis(filter_coefficients(a_noisy, H_zero_phase), F0, t)
    print("\nWITH NOISE (std 2 deg/s) added back, same filter applied to its coefficients:")
    print(f"  RMSE(noisy record,      true motion)          = {rmse(noisy, motion):.3f}")
    print(f"  RMSE(causal filtered,   motion delayed 0.21 s) = {rmse(y_noisy, motion_delayed):.3f}")
    print(f"  RMSE(zero-phase filtered, true motion)         = {rmse(y_noisy_zero, motion):.3f}")

    # Figure.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    f_axis = np.logspace(-1, 1.5, 400)
    ax = axes[0, 0]
    ax.semilogx(f_axis, abs(H(f_axis)), color="steelblue", linewidth=2)
    for f, c in [(0.5, "darkgreen"), (8.0, "firebrick")]:
        ax.plot(f, abs(H(f)), "o", color=c, markersize=8, label=f"{f:g} Hz: |H| = {abs(H(f)):.3g}")
    ax.set_title(f"Frequency response |H|: {ORDER}th-order Butterworth, cutoff {FC:g} Hz")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("|H|")
    ax.legend()

    ax = axes[0, 1]
    ks = np.arange(0, 21)
    ax.stem(ks * F0 - 0.08, [2 * abs(a[k]) for k in ks], linefmt="C0-", markerfmt="C0o", basefmt=" ", label="input 2|a_k|")
    ax.stem(ks * F0 + 0.08, [2 * abs(b[k]) for k in ks], linefmt="C3-", markerfmt="C3s", basefmt=" ", label="output 2|b_k|")
    ax.set_title("Component amplitudes before and after: the 8 Hz spike is removed")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("amplitude (deg/s)")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(t, omega, color="lightgray", linewidth=2, label="input w(t)")
    ax.plot(t, y, color="steelblue", linewidth=2, label=f"causal Butterworth output (delayed {delay:.2f} s)")
    ax.plot(t, y_zero, color="darkorange", linewidth=1.5, label="zero-phase output (|H| only)")
    ax.plot(t, motion, "k--", linewidth=1, label="true motion 20 sin(2$\\pi$0.5t)")
    ax.set_title("Clean signal: the vibration ripple disappears, the motion stays")
    ax.set_xlabel("t (s)")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1, 1]
    ax.plot(t, noisy, color="lightgray", linewidth=0.8, label="noisy record")
    ax.plot(t, y_noisy, color="steelblue", linewidth=2, label="causal filtered")
    ax.plot(t, y_noisy_zero, color="darkorange", linewidth=1.5, label="zero-phase filtered")
    ax.plot(t, motion, "k--", linewidth=1, label="true motion")
    ax.set_title("With noise: filtering by frequency removes vibration and most noise")
    ax.set_xlabel("t (s)")
    ax.legend(loc="upper right", fontsize=8)
    for ax in axes.flatten():
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "imu_decomposition.png"), dpi=150)
    print("\nSaved imu_decomposition.png")

    print(
        """
What this means
---------------
Decomposition turned a signal that looked like one wiggly curve into a short
list: a 0.5 Hz component of amplitude 20 (the platform motion) and an 8 Hz
component of amplitude 3 (the vibration). Once the signal is a list of
components, filtering is just multiplication: each coefficient a_k gets scaled
by the system's gain at its frequency, b_k = H(j k w0) a_k.
Here H is nearly 1 at 0.5 Hz and about 0.004 at 8 Hz, so the motion passes and
the vibration is essentially gone. The output was rebuilt by synthesis and
matches the closed-form prediction. H also has a phase: this causal filter
delays the 0.5 Hz motion by about 0.21 s, which is why its raw RMSE against the
true motion looks bad even though the amplitude is preserved. Using |H| alone
(zero phase) removes the delay, the same idea as the noncausal filter of
Module 9: possible offline, impossible in real time.
This is the payoff of the module: frequency decomposition, then
frequency-selective filtering. It also works on the noisy record, where the
same filter that removes the 8 Hz vibration removes most of the noise, because
noise is spread over all frequencies while the filter keeps only a narrow band.
In the time domain (Module 8) those three ingredients were hard to separate by
eye; by frequency they are easy."""
    )
