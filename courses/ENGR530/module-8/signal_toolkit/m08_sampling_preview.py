"""Sampling preview: sample x(t) = sin(2*pi*5*t) at several rates and see
which sample sets look like the underlying waveform."""

import os

import matplotlib.pyplot as plt
import numpy as np

from signals import sinusoid

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

FREQUENCY = 5.0
DURATION = 1.0
DENSE_RATE = 5000
SAMPLE_RATES = [100, 30, 12, 8]


def sine(sampling_rate):
    """x(t) = sin(2*pi*5*t) = cos(2*pi*5*t - pi/2), from the toolkit."""
    return sinusoid(1.0, FREQUENCY, -np.pi / 2, DURATION, sampling_rate)


if __name__ == "__main__":
    t_dense, x_dense = sine(DENSE_RATE)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)

    print(f"x(t) = sin(2 pi 5 t), {FREQUENCY:g} Hz, shown for {DURATION:g} s ({int(FREQUENCY * DURATION)} cycles)\n")
    print(f"{'fs (Hz)':>8}{'samples in 1 s':>16}{'samples per cycle':>19}")
    for ax, fs in zip(axes.flatten(), SAMPLE_RATES):
        t_s, x_s = sine(fs)

        ax.plot(t_dense, x_dense, color="lightgray", linewidth=2, label="continuous x(t)")
        ax.plot(t_s, x_s, "--", color="steelblue", linewidth=0.8, alpha=0.7, label="connect-the-dots")
        ax.plot(t_s, x_s, "o", color="firebrick", markersize=5, label="samples")
        ax.set_title(f"fs = {fs} Hz  ({fs / FREQUENCY:g} samples per cycle)")
        ax.grid(alpha=0.3)

        print(f"{fs:>8}{len(t_s):>16}{fs / FREQUENCY:>19.1f}")

    for ax in axes[1]:
        ax.set_xlabel("t (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("x(t)")
    fig.suptitle("Sampling x(t) = sin(2$\\pi$5t) at four rates")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(OUT_DIR, "sampling_preview.png"), dpi=150)
    print("\nSaved sampling_preview.png")

    print(
        """
What this means
---------------
Same signal, four sets of snapshots, and they are not equally trustworthy:
  - 100 Hz (20 per cycle): the dots trace the sine so closely you could
    rebuild the wave by eye.
  - 30 Hz (6 per cycle): still clearly a 5 Hz sine, just coarser.
  - 12 Hz (2.4 per cycle): the dots barely touch each cycle, and joining them
    gives a shape that no longer looks like the original.
  - 8 Hz (1.6 per cycle): fewer than two samples per cycle. The dots trace
    a slower, different-looking wave than the 5 Hz signal that produced them.
Takeaway: how well a sample set represents a signal depends on how many
samples land in each cycle. Once that gets too small, the samples can no longer
tell you what the signal was doing between them. That threshold is what the
sampling theorem makes precise; here we only observe that it exists."""
    )
