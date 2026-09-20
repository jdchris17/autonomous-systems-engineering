"""Demo of the signal toolkit: build x(t) = 3 cos(2*pi*5*t + pi/4) and verify
its amplitude, period, and phase behavior; then show the other generators."""

import os

import matplotlib.pyplot as plt
import numpy as np

from signals import discrete_impulse, exponential, sinusoid, unit_step

AMPLITUDE = 3.0
FREQUENCY = 5.0
PHASE = np.pi / 4
DURATION = 1.0
SAMPLING_RATE = 1000

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def local_maxima_times(t, x):
    """Times of interior samples that exceed both neighbors."""
    idx = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1
    return t[idx]


if __name__ == "__main__":
    t, x = sinusoid(AMPLITUDE, FREQUENCY, PHASE, DURATION, SAMPLING_RATE)
    _, x_nophase = sinusoid(AMPLITUDE, FREQUENCY, 0.0, DURATION, SAMPLING_RATE)

    # Amplitude: the peak value.
    measured_amp = np.max(np.abs(x))

    # Period: average spacing between successive peaks.
    peak_times = local_maxima_times(t, x)
    measured_period = np.mean(np.diff(peak_times))
    expected_period = 1 / FREQUENCY

    # Phase: x(0) = A cos(phase), and the first peak sits at t = -phase/(2 pi f)
    # mod one period (positive phase shifts the waveform left / earlier).
    expected_x0 = AMPLITUDE * np.cos(PHASE)
    expected_first_peak = (-PHASE / (2 * np.pi * FREQUENCY)) % expected_period
    measured_first_peak = peak_times[0]
    shift_seconds = PHASE / (2 * np.pi * FREQUENCY)

    print("x(t) = 3 cos(2 pi 5 t + pi/4),  fs = 1000 Hz,  1 s\n")
    print(f"{'check':<28}{'measured':>12}{'expected':>12}")
    print(f"{'amplitude (peak)':<28}{measured_amp:>12.4f}{AMPLITUDE:>12.4f}")
    print(f"{'period (s)':<28}{measured_period:>12.4f}{expected_period:>12.4f}")
    print(f"{'x(0)':<28}{x[0]:>12.4f}{expected_x0:>12.4f}")
    print(f"{'first peak time (s)':<28}{measured_first_peak:>12.4f}{expected_first_peak:>12.4f}")
    print(f"\nPhase pi/4 = {shift_seconds * 1000:.1f} ms of lead: the waveform is the no-phase cosine shifted "
          f"{shift_seconds * 1000:.1f} ms earlier.")

    # Figure 1: the sinusoid, with the zero-phase cosine for comparison.
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, x, color="steelblue", linewidth=2, label="3 cos(2$\\pi$5t + $\\pi$/4)")
    ax.plot(t, x_nophase, "--", color="gray", linewidth=1, label="3 cos(2$\\pi$5t)  (zero phase)")
    ax.axhline(AMPLITUDE, color="firebrick", linestyle=":", linewidth=1)
    ax.axhline(-AMPLITUDE, color="firebrick", linestyle=":", linewidth=1, label="$\\pm$ amplitude")
    ax.plot(0, x[0], "o", color="darkorange", label=f"x(0) = {x[0]:.3f}")
    ax.annotate("", xy=(peak_times[0], AMPLITUDE + 0.3), xytext=(peak_times[1], AMPLITUDE + 0.3),
                arrowprops=dict(arrowstyle="<->", color="black"))
    ax.text((peak_times[0] + peak_times[1]) / 2, AMPLITUDE + 0.45, f"period = {measured_period:.2f} s",
            ha="center")
    ax.set_xlim(0, 0.6)
    ax.set_ylim(-4, 4.4)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("x(t)")
    ax.set_title("Sinusoid: amplitude, period, phase")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "sinusoid_check.png"), dpi=150)

    # Figure 2: the other generators.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    te, xe = exponential(2.0, -3.0, 2.0, SAMPLING_RATE)
    axes[0].plot(te, xe, color="steelblue")
    axes[0].set_title("exponential: 2 e$^{-3t}$")
    axes[0].set_xlabel("t (s)")
    ts, xs = unit_step(1.0, SAMPLING_RATE, t0=0.3)
    axes[1].plot(ts, xs, color="steelblue")
    axes[1].set_title("unit_step (t0 = 0.3 s)")
    axes[1].set_xlabel("t (s)")
    n, d = discrete_impulse(20, n0=5)
    axes[2].stem(n, d)
    axes[2].set_title("discrete_impulse ($n_0$ = 5)")
    axes[2].set_xlabel("n")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "other_signals.png"), dpi=150)
    print("Saved sinusoid_check.png and other_signals.png")

    print(
        """
What this means
---------------
A sinusoid is fully described by three numbers, and each one is directly
readable off the plot:
  - Amplitude (3): the peak height. It sets the signal's size, nothing else.
  - Frequency (5 Hz) <-> period (0.2 s): they are the same fact, T = 1/f.
    Peaks are 0.2 s apart in the data, matching the formula.
  - Phase (pi/4): where in its cycle the signal is at t = 0. It does not
    change the shape, only slides it in time: a positive phase makes the
    signal LEAD, so it starts already partway up (x(0) = 2.12, not 3) and
    peaks 25 ms before t = 0. Same wave, different starting point.
The other generators are the building blocks for what follows: exponentials
model decay/growth, the step models something switching on, and the impulse
is the single 'ping' used to probe systems."""
    )
