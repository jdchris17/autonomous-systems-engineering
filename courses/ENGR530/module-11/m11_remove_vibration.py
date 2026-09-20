"""Engineering challenge: remove the 8 Hz vibration by an idealized frequency-domain
operation. Y(f) = H(f) X(f), then inverse transform."""

import os

import matplotlib.pyplot as plt
import numpy as np

from m11_imu_spectrum import DURATION, FS, amplitude_spectrum, make_signal

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
NOTCH = (7.0, 9.0)      # Hz: H = 0 inside this band (both signs of frequency)
LOWPASS = 3.0           # Hz: comparison filter, H = 1 below this and 0 above


def band_stop(f, lo, hi):
    """Ideal notch: 0 for lo <= |f| <= hi, 1 elsewhere."""
    return np.where((np.abs(f) >= lo) & (np.abs(f) <= hi), 0.0, 1.0)


def low_pass(f, cutoff):
    return np.where(np.abs(f) <= cutoff, 1.0, 0.0)


def apply_filter(x, H_of_f):
    """Y(f) = H(f) X(f) on the full (two-sided) FFT, then inverse transform."""
    f = np.fft.fftfreq(len(x), 1 / FS)
    Y = H_of_f(f) * np.fft.fft(x)
    return np.fft.ifft(Y).real, f, Y


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


if __name__ == "__main__":
    t, motion, vibration, noise = make_signal()
    w = motion + vibration + noise
    N = len(w)

    y_notch, f_full, Y_notch = apply_filter(w, lambda f: band_stop(f, *NOTCH))
    y_low, _, Y_low = apply_filter(w, lambda f: low_pass(f, LOWPASS))

    f, amp_raw = amplitude_spectrum(w)
    _, amp_notch = amplitude_spectrum(y_notch)
    _, amp_low = amplitude_spectrum(y_low)

    def at(amp, f0):
        return amp[int(np.argmin(np.abs(f - f0)))]

    print(f"H(f): ideal notch, 0 for {NOTCH[0]:g} <= |f| <= {NOTCH[1]:g} Hz and 1 elsewhere\n")
    print(f"{'':<26}{'0.5 Hz amp':>12}{'8 Hz amp':>11}{'RMSE vs true motion':>22}")
    print(f"{'raw signal':<26}{at(amp_raw, 0.5):>12.3f}{at(amp_raw, 8):>11.3f}{rmse(w, motion):>22.3f}")
    print(f"{'after notch (7-9 Hz)':<26}{at(amp_notch, 0.5):>12.3f}{at(amp_notch, 8):>11.3f}{rmse(y_notch, motion):>22.3f}")
    print(f"{'after low-pass (3 Hz)':<26}{at(amp_low, 0.5):>12.3f}{at(amp_low, 8):>11.3f}{rmse(y_low, motion):>22.3f}")

    print(f"\nThe notch removes the vibration but only {(NOTCH[1] - NOTCH[0]) / (FS / 2) * 100:.0f}% of the noise band, so the residual is mostly noise:")
    print(f"  noise power left after notch:    {np.var(y_notch - motion):.2f}   (noise alone: {noise.var():.2f})")
    print(f"  noise power left after low-pass: {np.var(y_low - motion):.2f}")

    # Figure 1: raw signal, raw spectrum, filtered spectrum, filtered signal.
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].plot(t, w, color="gray", linewidth=0.8)
    axes[0, 0].set_title("1. Raw time signal")
    axes[0, 0].set_xlabel("t (s)")
    axes[0, 0].set_ylabel("deg/s")

    m = f <= 20
    axes[0, 1].stem(f[m], amp_raw[m], basefmt=" ")
    ax2 = axes[0, 1].twinx()
    ax2.plot(f[m], band_stop(f[m], *NOTCH), color="firebrick", linewidth=1.5, label="H(f)")
    ax2.set_ylim(-0.05, 1.5)
    ax2.set_ylabel("H(f)", color="firebrick")
    axes[0, 1].set_title("2. Raw spectrum |X(f)| with the idealized H(f)")
    axes[0, 1].set_xlabel("frequency (Hz)")
    axes[0, 1].set_ylabel("amplitude")

    axes[1, 0].stem(f[m], amp_notch[m], basefmt=" ")
    axes[1, 0].set_ylim(axes[0, 1].get_ylim())
    axes[1, 0].set_title("3. Filtered spectrum |Y(f)| = |H(f) X(f)|: the 8 Hz line is gone")
    axes[1, 0].set_xlabel("frequency (Hz)")
    axes[1, 0].set_ylabel("amplitude")

    axes[1, 1].plot(t, w, color="lightgray", linewidth=1, label="raw")
    axes[1, 1].plot(t, y_notch, color="steelblue", linewidth=1.3, label="filtered (inverse FFT)")
    axes[1, 1].plot(t, motion, "k--", linewidth=1, label="true motion")
    axes[1, 1].set_title("4. Filtered time signal")
    axes[1, 1].set_xlabel("t (s)")
    axes[1, 1].legend(fontsize=8, loc="upper right")
    for ax in axes.flatten():
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "remove_vibration.png"), dpi=150)

    # Figure 2: notch vs low-pass.
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    ff = np.linspace(0, 20, 2000)
    axes[0].plot(ff, band_stop(ff, *NOTCH), color="steelblue", linewidth=2, label=f"notch {NOTCH[0]:g}-{NOTCH[1]:g} Hz")
    axes[0].plot(ff, low_pass(ff, LOWPASS) * 0.96, color="darkorange", linewidth=2, label=f"low-pass {LOWPASS:g} Hz (drawn slightly lower)")
    axes[0].set_ylim(-0.05, 1.15)
    axes[0].set_title("Two idealized H(f)")
    axes[0].set_xlabel("frequency (Hz)")
    axes[0].legend()
    sel = t <= 4
    axes[1].plot(t[sel], w[sel], color="lightgray", linewidth=1, label="raw")
    axes[1].plot(t[sel], y_notch[sel], color="steelblue", linewidth=1.3, label="notch: vibration gone, noise stays")
    axes[1].plot(t[sel], y_low[sel], color="darkorange", linewidth=1.8, label="low-pass: vibration and noise gone")
    axes[1].plot(t[sel], motion[sel], "k--", linewidth=1, label="true motion")
    axes[1].set_title("Result (first 4 s)")
    axes[1].set_xlabel("t (s)")
    axes[1].legend(fontsize=8, loc="upper right")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "remove_vibration_compare.png"), dpi=150)
    print("Saved remove_vibration.png and remove_vibration_compare.png")

    print(
        f"""
What this means
---------------
Filtering here is three steps: transform, multiply, inverse transform. No
convolution loop, no delay: Y(f) = H(f) X(f) just sets each frequency's gain, and
the inverse FFT turns the result back into a time signal.
  - The notch zeroes the 7-9 Hz band. The 8 Hz line disappears from the spectrum
    ({at(amp_raw, 8):.1f} -> {at(amp_notch, 8):.1f}) while the 0.5 Hz line is untouched, and in time the ripple riding
    on the swing is gone.
  - The error against the true motion falls from {rmse(w, motion):.2f} to {rmse(y_notch, motion):.2f}, but not to zero:
    the notch only removes 2 Hz of a 100 Hz band, so nearly all the noise stays.
    The low-pass removes the vibration and most of the noise ({rmse(y_low, motion):.2f}) because the
    wanted signal occupies only a narrow band near 0.5 Hz.
  - Idealized means brick-wall edges, which real filters cannot have, and this
    only works cleanly offline on a whole record. Both caveats are why Module 8/9
    filters (moving averages, first-order sensors) are the practical versions of
    this same idea: a chosen |H(f)| applied to the spectrum.
Decompose, choose which frequencies to keep, rebuild: a complete
frequency-domain filtering operation."""
    )
