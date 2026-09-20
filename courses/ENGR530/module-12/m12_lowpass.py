"""Low-pass filtering: an idealized frequency-domain filter, then a realizable
digital FIR filter, scored by RMSE against the known clean 2 Hz signal.
    x(t) = 5 sin(2 pi 2 t) + 2 sin(2 pi 20 t) + n(t)
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "module-8", "signal_toolkit"))
from transforms import dtft

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FS = 200                  # Hz
DURATION = 10.0           # s: whole numbers of periods for both tones
NOISE_STD = 1.0
CUTOFF = 6.0              # Hz, between the 2 Hz signal and the 20 Hz interference
TAPS = 121                # FIR length (odd) -> delay of (TAPS - 1)/2 = 60 samples = 0.3 s


def make_signal(seed=0):
    t = np.arange(int(FS * DURATION)) / FS
    clean = 5 * np.sin(2 * np.pi * 2 * t)
    vib = 2 * np.sin(2 * np.pi * 20 * t)
    noise = np.random.default_rng(seed).normal(0.0, NOISE_STD, size=len(t))
    return t, clean, vib, noise


def ideal_lowpass(x, cutoff):
    """Idealized filter: Y(f) = H(f) X(f) with H = 1 for |f| <= cutoff and 0 above (brick wall)."""
    f = np.fft.fftfreq(len(x), 1 / FS)
    return np.fft.ifft(np.where(np.abs(f) <= cutoff, 1.0, 0.0) * np.fft.fft(x)).real


def design_fir_lowpass(cutoff, taps=TAPS):
    """Windowed-sinc FIR: the ideal low-pass's impulse response (a sinc), truncated and Hamming-windowed.

    The ideal filter's h[n] is non-causal and infinitely long. Truncating to `taps`
    samples and delaying by (taps - 1)/2 makes it causal and realizable.
    """
    n = np.arange(taps) - (taps - 1) / 2
    h = 2 * cutoff / FS * np.sinc(2 * cutoff / FS * n) * np.hamming(taps)
    return h / h.sum()                     # unity gain at DC


def fir_filter(x, h):
    """Causal FIR filtering, run to steady state.

    x is one period of a record; filtering two back-to-back copies and keeping the
    second removes the start-up transient (what a long-running filter would output).
    Returns the output as the filter produces it, i.e. delayed by (len(h) - 1)/2.
    """
    n = len(x)
    return np.convolve(np.tile(x, 2), h)[n:2 * n]


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def amplitude_spectrum(x):
    N = len(x)
    amp = 2 * np.abs(np.fft.rfft(x)) / N
    amp[0] /= 2
    return np.fft.rfftfreq(N, 1 / FS), amp


if __name__ == "__main__":
    t, clean, vib, noise = make_signal()
    x = clean + vib + noise
    N = len(x)
    D = (TAPS - 1) // 2

    h = design_fir_lowpass(CUTOFF)
    y_ideal = ideal_lowpass(x, CUTOFF)
    y_fir_raw = fir_filter(x, h)              # as produced: delayed by D samples
    y_fir = np.roll(y_fir_raw, -D)            # delay removed (offline alignment)

    f, a_raw = amplitude_spectrum(x)
    _, a_ideal = amplitude_spectrum(y_ideal)
    _, a_fir = amplitude_spectrum(y_fir)

    def at(a, f0):
        return a[int(np.argmin(np.abs(f - f0)))]

    omega = 2 * np.pi * np.array([2.0, 20.0]) / FS
    H_fir = np.abs(dtft(h, np.arange(TAPS), omega))

    print(f"x(t) = 5 sin(2 pi 2 t) + 2 sin(2 pi 20 t) + n(t),  n ~ Normal(0, {NOISE_STD}),  fs = {FS} Hz, {DURATION:g} s")
    print(f"Low-pass cutoff {CUTOFF:g} Hz (between 2 and 20).  FIR: {TAPS}-tap windowed sinc (Hamming), delay {D} samples = {D / FS:.2f} s\n")
    print(f"Gain of the FIR at 2 Hz = {H_fir[0]:.4f}, at 20 Hz = {H_fir[1]:.2e}  (ideal filter: 1 and 0)\n")
    print(f"{'':<34}{'2 Hz amp':>9}{'20 Hz amp':>11}{'RMSE vs clean 2 Hz':>21}")
    print(f"{'raw':<34}{at(a_raw, 2):>9.3f}{at(a_raw, 20):>11.3f}{rmse(x, clean):>21.3f}")
    print(f"{'ideal frequency-domain filter':<34}{at(a_ideal, 2):>9.3f}{at(a_ideal, 20):>11.3f}{rmse(y_ideal, clean):>21.3f}")
    print(f"{'FIR (delay removed)':<34}{at(a_fir, 2):>9.3f}{at(a_fir, 20):>11.4f}{rmse(y_fir, clean):>21.3f}")
    print(f"{'FIR as produced (0.30 s late)':<34}{'':>9}{'':>11}{rmse(y_fir_raw, clean):>21.3f}")

    noise_left = np.var(y_ideal - clean)
    print(f"\nNoise variance: {noise.var():.3f} raw -> {noise_left:.3f} after the ideal filter "
          f"(passband is {CUTOFF:g} of {FS / 2:g} Hz = {CUTOFF / (FS / 2) * 100:.0f}% of the noise band)")

    # Figure 1: raw time, raw spectrum, filtered spectrum, filtered time.
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    show = t <= 2.0
    axes[0, 0].plot(t[show], x[show], color="gray", linewidth=1, label="raw x(t)")
    axes[0, 0].plot(t[show], clean[show], "k--", linewidth=1.3, label="clean 2 Hz signal")
    axes[0, 0].set_title(f"1. Raw time signal   (RMSE vs clean = {rmse(x, clean):.2f})")
    axes[0, 0].set_xlabel("t (s)")
    axes[0, 0].legend(fontsize=8, loc="upper right")

    m = f <= 40
    axes[0, 1].stem(f[m], a_raw[m], basefmt=" ")
    axes[0, 1].axvline(CUTOFF, color="firebrick", linestyle="--", label=f"cutoff {CUTOFF:g} Hz")
    axes[0, 1].set_title("2. Raw spectrum: 2 Hz signal, 20 Hz interference, noise floor")
    axes[0, 1].set_xlabel("frequency (Hz)")
    axes[0, 1].set_ylabel("amplitude")
    axes[0, 1].legend()

    axes[1, 0].plot(f[m], a_ideal[m], "-o", color="steelblue", markersize=3, linewidth=1, label="ideal")
    axes[1, 0].plot(f[m], a_fir[m], "-s", color="darkorange", markersize=3, linewidth=1, label="FIR")
    axes[1, 0].axvline(CUTOFF, color="firebrick", linestyle="--")
    axes[1, 0].set_ylim(axes[0, 1].get_ylim())
    axes[1, 0].set_title("3. Filtered spectra: the 20 Hz line and the noise above the cutoff are gone")
    axes[1, 0].set_xlabel("frequency (Hz)")
    axes[1, 0].set_ylabel("amplitude")
    axes[1, 0].legend()

    axes[1, 1].plot(t[show], x[show], color="lightgray", linewidth=1, label="raw")
    axes[1, 1].plot(t[show], y_ideal[show], color="steelblue", linewidth=2, label=f"ideal (RMSE {rmse(y_ideal, clean):.2f})")
    axes[1, 1].plot(t[show], y_fir[show], color="darkorange", linewidth=1.5, label=f"FIR, delay removed (RMSE {rmse(y_fir, clean):.2f})")
    axes[1, 1].plot(t[show], y_fir_raw[show], color="darkgreen", linewidth=1.2, linestyle=":", label=f"FIR as produced (RMSE {rmse(y_fir_raw, clean):.2f})")
    axes[1, 1].plot(t[show], clean[show], "k--", linewidth=1, label="clean 2 Hz")
    axes[1, 1].set_title("4. Filtered time signals")
    axes[1, 1].set_xlabel("t (s)")
    axes[1, 1].legend(fontsize=8, loc="upper right")
    for ax in axes.flatten():
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "lowpass.png"), dpi=150)

    # Figure 2: the FIR design and how it differs from the ideal filter.
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    axes[0].stem(np.arange(TAPS) - D, h, basefmt=" ")
    axes[0].set_title(f"FIR impulse response: a sinc x Hamming window ({TAPS} taps, centered at n = {D} once delayed)")
    axes[0].set_xlabel("n - center")
    ff = np.linspace(0, 50, 2001)
    Hf = np.abs(dtft(h, np.arange(TAPS), 2 * np.pi * ff / FS))
    axes[1].plot(ff, 20 * np.log10(Hf + 1e-12), color="darkorange", linewidth=2, label="FIR |H| (dB)")
    axes[1].plot(ff, np.where(ff <= CUTOFF, 0.0, -100), color="steelblue", linewidth=1.5, linestyle="--", label="ideal brick wall")
    axes[1].axvline(2, color="gray", linestyle=":")
    axes[1].axvline(20, color="gray", linestyle=":")
    axes[1].text(2.3, -95, "2 Hz signal")
    axes[1].text(20.3, -95, "20 Hz interference")
    axes[1].set_ylim(-100, 5)
    axes[1].set_title("Frequency responses: the realizable filter has a gradual transition, not a wall")
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel("dB")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "lowpass_fir_design.png"), dpi=150)
    print("Saved lowpass.png and lowpass_fir_design.png")

    print(
        f"""
What this means
---------------
Before/after RMSE against the known clean 2 Hz signal turns 'does the filter
help' into a number: {rmse(x, clean):.2f} raw -> {rmse(y_ideal, clean):.2f} with the ideal filter and {rmse(y_fir, clean):.2f} with a realizable FIR.
  - Both remove the 20 Hz interference completely and everything above the
    6 Hz cutoff. What remains is the noise inside 0-6 Hz, which the filter
    cannot separate from the signal, so the RMSE stops at roughly
    the noise that fits in the passband, not at zero.
  - The ideal filter has a brick-wall response, which needs an infinitely long,
    non-causal impulse response. The FIR is that response truncated, windowed
    and delayed: a gradual transition instead of a wall, and a cost of {D / FS:.2f} s of
    delay. Left as produced, the delayed output is badly wrong against the
    undelayed truth ({rmse(y_fir_raw, clean):.2f}), so the delay must be accounted for. Offline you can
    shift it back; in real time you cannot, which is the causal/noncausal
    trade-off from Module 9.
  - Here the two frequencies are far apart (2 vs 20 Hz), so a cutoff anywhere in
    between works and the gradual FIR transition costs almost nothing. That
    would not be true if the interference sat close to the signal."""
    )
