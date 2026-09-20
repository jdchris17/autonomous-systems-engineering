"""Engineering challenge: the IMU signal in the frequency domain.
    w(t) = 20 sin(2 pi 0.5 t) + 3 sin(2 pi 8 t) + n(t)
"""

import os

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FS = 200                # Hz
DURATION = 10.0         # s: 5 periods of the 0.5 Hz motion, 80 of the 8 Hz vibration, bins 0.1 Hz apart
NOISE_STD = 2.0


def make_signal(seed=0):
    t = np.arange(int(FS * DURATION)) / FS
    motion = 20 * np.sin(2 * np.pi * 0.5 * t)
    vibration = 3 * np.sin(2 * np.pi * 8 * t)
    noise = np.random.default_rng(seed).normal(0.0, NOISE_STD, size=len(t))
    return t, motion, vibration, noise


def amplitude_spectrum(x):
    """One-sided amplitude spectrum: a sinusoid of amplitude A shows up as a peak of height A."""
    N = len(x)
    X = np.fft.rfft(x)
    amp = 2 * np.abs(X) / N
    amp[0] /= 2
    return np.fft.rfftfreq(N, 1 / FS), amp


if __name__ == "__main__":
    t, motion, vibration, noise = make_signal()
    w = motion + vibration + noise
    N = len(w)
    f, amp = amplitude_spectrum(w)

    def near(f0):
        return int(np.argmin(np.abs(f - f0)))

    print(f"fs = {FS} Hz, {DURATION:g} s record, N = {N}, FFT bins {f[1]:.1f} Hz apart\n")
    print("Strongest spectral lines:")
    order = np.argsort(amp)[::-1][:3]
    for i in order[:2]:
        print(f"  {f[i]:>6.2f} Hz   amplitude {amp[i]:.3f}")
    print("  expected: 0.5 Hz -> 20, 8 Hz -> 3 (small differences are the noise)")
    print(f"  next largest bin: {f[order[2]]:.2f} Hz at {amp[order[2]]:.3f}, just the biggest random noise bin\n")

    print("Noise floor (median amplitude of the bins away from the two lines):")
    for lo, hi in [(10, 20), (30, 50), (60, 80), (80, 99)]:
        band = (f >= lo) & (f < hi)
        print(f"  {lo:>3d}-{hi:<3d} Hz: {np.median(amp[band]):.3f}")
    print("  (flat across frequency: that is what broadband / white noise looks like)")
    floor = np.median(amp[(f > 20) & (f < 99)])

    # Parseval: where does the signal's power (mean square) live?
    power = np.abs(np.fft.rfft(w)) ** 2 / N**2 * 2
    power[0] /= 2
    p_motion = power[(f >= 0.4) & (f <= 0.6)].sum()
    p_vib = power[(f >= 7.9) & (f <= 8.1)].sum()
    print(f"\nPower (mean square):  time domain {np.mean(w ** 2):.2f}   sum over the spectrum {power.sum():.2f}   (Parseval)")
    print(f"  at 0.5 Hz: {p_motion:6.2f}   (expected 20^2/2 = 200)")
    print(f"  at 8 Hz  : {p_vib:6.2f}   (expected 3^2/2 = 4.5)")
    print(f"  the rest : {power.sum() - p_motion - p_vib:6.2f}   (expected noise power {NOISE_STD ** 2:.1f}, spread thinly over all {len(f)} bins)")

    fig, axes = plt.subplots(3, 1, figsize=(13, 13))
    axes[0].plot(t, w, color="gray", linewidth=0.8)
    axes[0].set_title("Time domain: motion, vibration and noise mixed together (Module 8's question)")
    axes[0].set_xlabel("t (s)")
    axes[0].set_ylabel("deg/s")

    axes[1].stem(f[f <= 15], amp[f <= 15], basefmt=" ")
    axes[1].annotate("0.5 Hz: platform motion (20)", xy=(0.5, amp[near(0.5)]), xytext=(2, 17), arrowprops=dict(arrowstyle="->"))
    axes[1].annotate("8 Hz: vibration (3)", xy=(8, amp[near(8)]), xytext=(9.5, 8), arrowprops=dict(arrowstyle="->"))
    axes[1].annotate("noise floor", xy=(12, floor * 1.5), xytext=(12.2, 3), arrowprops=dict(arrowstyle="->"))
    axes[1].set_title("Frequency domain (0-15 Hz): the components are separate and obvious")
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel("amplitude")

    axes[2].plot(f, 20 * np.log10(amp + 1e-12), color="steelblue", linewidth=0.9)
    axes[2].axhline(20 * np.log10(floor), color="firebrick", linestyle="--", label=f"noise floor about {floor:.2f}")
    axes[2].set_ylim(-45, 30)
    axes[2].set_title("Whole spectrum in dB: two lines above a flat, broadband noise floor")
    axes[2].set_xlabel("frequency (Hz)")
    axes[2].set_ylabel("dB")
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "imu_spectrum.png"), dpi=150)
    print("\nSaved imu_spectrum.png")

    print(
        """
What this means
---------------
Module 8 asked whether motion, vibration and noise can be told apart in the time
domain, and the answer was 'not really': the vibration and the noise are about
the same size and blur into one ripple. In the spectrum each ingredient
occupies a different place:
  - a tall line at 0.5 Hz (height 20): the deliberate platform motion,
  - a smaller line at 8 Hz (height 3): the vibration, cleanly separated from the
    motion by a factor of 16 in frequency,
  - a low floor spread evenly across every frequency: the noise. Its total power
    is comparable to the vibration's, but spread over 1000 bins each bin holds
    almost nothing, so the two lines stand well above it.
This is why engineers use spectral analysis: signals that overlap in time are
often well separated in frequency, and once you can see them separately you can
decide what to keep and what to remove."""
    )
