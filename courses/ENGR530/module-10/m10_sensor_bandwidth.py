"""Engineering challenge: sensor bandwidth. Given |H| at four frequencies,
compute the output amplitudes |b_k| = |H(j k w0)| |a_k| and see which
motions the sensor reproduces faithfully."""

import os

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FREQS = np.array([1.0, 5.0, 20.0, 50.0])       # Hz (harmonics k = 1, 5, 20, 50 of a 1 Hz fundamental)
H_MAG = np.array([1.0, 0.98, 0.60, 0.10])
IN_AMP = np.ones(4)                            # input amplitudes: the challenge gives none, so use 1 each
F0 = 1.0


if __name__ == "__main__":
    a_mag = IN_AMP / 2            # |a_k| for a sinusoid of amplitude A is A/2
    b_mag = H_MAG * a_mag         # |b_k| = |H| |a_k|
    out_amp = 2 * b_mag
    loss_pct = (1 - H_MAG) * 100
    db = 20 * np.log10(H_MAG)

    print("|b_k| = |H(j k w0)| |a_k|   (input: unit-amplitude sinusoid at each frequency)\n")
    print(f"{'f (Hz)':>7}{'|H|':>7}{'|a_k|':>8}{'|b_k|':>8}{'in amp':>8}{'out amp':>9}{'gain (dB)':>11}{'amplitude lost':>16}")
    for f, h, a, b, ia, oa, d, l in zip(FREQS, H_MAG, a_mag, b_mag, IN_AMP, out_amp, db, loss_pct):
        print(f"{f:>7g}{h:>7.2f}{a:>8.3f}{b:>8.3f}{ia:>8.2f}{oa:>9.3f}{d:>11.2f}{l:>15.0f}%")

    # -3 dB point by interpolating |H| against log-frequency (a rough estimate from 4 points).
    target = 1 / np.sqrt(2)
    lo, hi = 1, 2                                  # between 5 Hz and 20 Hz
    frac = (H_MAG[lo] - target) / (H_MAG[lo] - H_MAG[hi])
    f3db = FREQS[lo] * (FREQS[hi] / FREQS[lo]) ** frac
    print(f"\n-3 dB point (|H| = 0.707), interpolated between 5 and 20 Hz: about {f3db:.0f} Hz (rough)")

    # Time-domain view: the four components summed. Assumes zero phase shift (only |H| is given).
    t = np.linspace(0.0, 1.0, 4001)
    x_in = sum(A * np.sin(2 * np.pi * f * t) for A, f in zip(IN_AMP, FREQS))
    y_out = sum(A * h * np.sin(2 * np.pi * f * t) for A, h, f in zip(IN_AMP, H_MAG, FREQS))
    print(f"RMSE(input, output) with equal-amplitude inputs = {np.sqrt(np.mean((x_in - y_out) ** 2)):.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    x_pos = np.arange(4)
    axes[0].bar(x_pos - 0.18, IN_AMP, 0.36, label="input amplitude", color="lightgray")
    axes[0].bar(x_pos + 0.18, out_amp, 0.36, label="output amplitude", color="steelblue")
    for i, (oa, l) in enumerate(zip(out_amp, loss_pct)):
        axes[0].text(i + 0.18, oa + 0.02, f"{oa:.2f}", ha="center", fontsize=9)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([f"{f:g} Hz" for f in FREQS])
    axes[0].set_title("Amplitude in vs. out, per harmonic")
    axes[0].legend()

    axes[1].semilogx(FREQS, H_MAG, "o-", color="steelblue", markersize=8)
    axes[1].axhline(target, color="firebrick", linestyle="--", label="-3 dB (0.707)")
    axes[1].axvline(f3db, color="firebrick", linestyle=":", label=f"about {f3db:.0f} Hz")
    axes[1].axvspan(0.8, f3db, color="green", alpha=0.08)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title("Sensor |H| at the four frequencies")
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel("|H|")
    axes[1].legend()

    axes[2].plot(t, x_in, color="lightgray", linewidth=2, label="input (sum of four)")
    axes[2].plot(t, y_out, color="steelblue", linewidth=1.5, label="sensor output")
    axes[2].set_xlim(0, 0.5)
    axes[2].set_title("Time domain, assuming no phase shift (only |H| is given)")
    axes[2].set_xlabel("t (s)")
    axes[2].legend(loc="upper right", fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "sensor_bandwidth.png"), dpi=150)
    print("Saved sensor_bandwidth.png")

    print(
        """
What this means
---------------
Each harmonic is scaled independently: output amplitude = |H| x input amplitude.
  - 1 Hz and 5 Hz (|H| = 1.00, 0.98): reproduced faithfully. At most 2% of the
    amplitude is lost, so slow motions such as a platform swaying at a few Hz
    come through essentially as they are.
  - 20 Hz (|H| = 0.60): significantly distorted. The sensor reports only 60%
    of the true amplitude, a 40% under-reading (-4.4 dB), and it is already
    past the roughly 14 Hz half-power point (a rough interpolation).
  - 50 Hz (|H| = 0.10): effectively invisible. 90% of the amplitude is lost, so
    a 50 Hz vibration is reported at one tenth of its real size.
So the sensor's usable band is roughly up to 5 Hz, with the edge somewhere
between 5 and 20 Hz. Motions and vibrations below that are measured
faithfully; a fast vibration or a sharp transient (whose energy sits at high
frequency) is under-reported or smoothed. Two cautions: this only used |H|
(phase is unknown, so timing distortion is not captured), and the input
amplitudes were assumed equal because none were given. Frequency response is
the sensor's capability, written as a function of frequency."""
    )
