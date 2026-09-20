"""Engineering challenge: a synthetic IMU angular-rate signal
    w(t) = 20 sin(2 pi 0.5 t) + 3 sin(2 pi 8 t) + n(t)
Plot the components and the measured total. No filtering yet."""

import os

import matplotlib.pyplot as plt
import numpy as np

from signals import sinusoid

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FS = 200
DURATION = 10.0
NOISE_STD = 2.0
ZOOM = (2.0, 3.0)


def sine(amplitude, frequency):
    """amplitude * sin(2*pi*frequency*t) via the toolkit's cosine generator."""
    return sinusoid(amplitude, frequency, -np.pi / 2, DURATION, FS)


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    t, motion = sine(20.0, 0.5)
    _, vibration = sine(3.0, 8.0)
    noise = rng.normal(0.0, NOISE_STD, size=len(t))
    total = motion + vibration + noise

    rows = [
        ("platform motion: 20 sin(2$\\pi$0.5t)", motion, "steelblue"),
        ("vibration: 3 sin(2$\\pi$8t)", vibration, "darkorange"),
        (f"noise n(t): Gaussian, std {NOISE_STD:g}", noise, "gray"),
        ("measured total $\\omega$(t)", total, "black"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(14, 10), gridspec_kw={"width_ratios": [3, 1.3]})
    for i, (title, sig, color) in enumerate(rows):
        axes[i, 0].plot(t, sig, color=color, linewidth=1)
        axes[i, 0].set_title(title, fontsize=10)
        axes[i, 0].set_ylabel("deg/s")
        axes[i, 0].axvspan(*ZOOM, color="gold", alpha=0.25)

        zoom = (t >= ZOOM[0]) & (t <= ZOOM[1])
        axes[i, 1].plot(t[zoom], sig[zoom], color=color, linewidth=1.2)
        axes[i, 1].set_title(f"zoom: {ZOOM[0]:g}-{ZOOM[1]:g} s", fontsize=10)

        for ax in axes[i]:
            ax.set_ylim(-28, 28)
            ax.grid(alpha=0.3)
    axes[3, 0].set_xlabel("t (s)")
    axes[3, 1].set_xlabel("t (s)")
    fig.suptitle("Synthetic IMU angular rate: components and measured signal (time domain only)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "imu_signal.png"), dpi=150)

    rms = lambda s: np.sqrt(np.mean(s**2))
    print(f"fs = {FS} Hz, duration = {DURATION:g} s, units deg/s\n")
    print(f"{'component':<18}{'amplitude / std':>17}{'RMS':>9}")
    print(f"{'platform motion':<18}{20.0:>17.2f}{rms(motion):>9.2f}")
    print(f"{'vibration':<18}{3.0:>17.2f}{rms(vibration):>9.2f}")
    print(f"{'noise':<18}{NOISE_STD:>17.2f}{rms(noise):>9.2f}")
    print(f"{'measured total':<18}{'':>17}{rms(total):>9.2f}")
    print("\nSaved imu_signal.png")

    print(
        """
What this means
---------------
Looking only in the time domain, the three ingredients are NOT equally easy
to tell apart:
  - Physical motion: easy. At 20 deg/s (RMS ~14) it dwarfs everything, and the
    0.5 Hz swing is plainly visible in the total.
  - Vibration vs noise: hard. The vibration (RMS ~2.1) and the noise (RMS ~2.0)
    are almost the same size, and both show up in the total as one fuzzy
    ripple riding on the big swing. In the zoomed total you cannot tell by eye
    how much of the wiggle is a clean 8 Hz tone and how much is random.
Each component is obvious in its own row above, but that is because we built
them and can see them separately. A real IMU gives you only the bottom row.
Separating them by their frequency content is exactly what Fourier analysis
will do."""
    )
