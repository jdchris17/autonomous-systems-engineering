"""Aliasing demonstration: one physical 70 Hz sinusoid sampled at six rates,
and the frequency it appears to have."""

import os

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
F_TRUE = 70.0
PHASE = np.pi / 4           # a generic phase; with exactly zero phase the fs = 140 samples would all be 0
SAMPLE_RATES = [500, 200, 140, 120, 100, 80]
WINDOW = 0.1                # s of waveform shown (7 cycles of 70 Hz)
DENSE_RATE = 20000          # "continuous-like" grid


def signed_alias(f, fs):
    """The frequency f - k fs, in (-fs/2, fs/2], that the samples cannot tell apart from f."""
    k = np.floor(f / fs + 0.5)
    return f - k * fs


def amplitude_spectrum(x):
    """One-sided amplitude spectrum for a record of exactly 1 s (so bins are 1 Hz apart)."""
    N = len(x)
    amp = 2 * np.abs(np.fft.rfft(x)) / N
    amp[0] /= 2
    if N % 2 == 0:
        amp[-1] /= 2                    # the Nyquist bin has no mirror image to add
    return np.fft.rfftfreq(N, 1 / N), amp


if __name__ == "__main__":
    t_dense = np.arange(int(DENSE_RATE * WINDOW)) / DENSE_RATE
    true_wave = np.sin(2 * np.pi * F_TRUE * t_dense + PHASE)

    print(f"x(t) = sin(2 pi {F_TRUE:g} t + pi/4), sampled for 1 s at each rate\n")
    print(f"{'fs (Hz)':>8}{'Nyquist':>9}{'samples/cycle':>15}{'predicted apparent f':>22}{'FFT peak (Hz)':>15}{'FFT amplitude':>15}   note")
    fig, axes = plt.subplots(len(SAMPLE_RATES), 2, figsize=(15, 21), gridspec_kw={"width_ratios": [1.5, 1]})
    rows = []
    for row, fs in enumerate(SAMPLE_RATES):
        n = np.arange(fs)                                   # 1 s of samples
        xs = np.sin(2 * np.pi * F_TRUE * n / fs + PHASE)
        f_axis, amp = amplitude_spectrum(xs)
        peak = f_axis[np.argmax(amp)]
        f_sig = signed_alias(F_TRUE, fs)
        apparent = abs(f_sig)

        if fs / 2 > F_TRUE:
            note = "below Nyquist: 70 Hz is reported correctly"
        elif fs / 2 == F_TRUE:
            note = "exactly at Nyquist: samples alternate sign, amplitude depends on phase"
        else:
            note = "ABOVE Nyquist: aliased" + (", phase inverted" if f_sig < 0 else "")
        rows.append((fs, apparent, peak))
        print(f"{fs:>8}{fs / 2:>9g}{fs / F_TRUE:>15.2f}{apparent:>22g}{peak:>15g}{amp.max():>15.3f}   {note}")

        ax = axes[row, 0]
        ax.plot(t_dense, true_wave, color="lightgray", linewidth=4, label="true 70 Hz waveform")
        m = n / fs <= WINDOW
        ax.plot(n[m] / fs, xs[m], "o", color="firebrick", markersize=6, label=f"samples at {fs} Hz", zorder=3)
        if fs / 2 <= F_TRUE:
            ax.plot(t_dense, np.sin(2 * np.pi * f_sig * t_dense + PHASE), "--", color="steelblue", linewidth=1.8,
                    label=f"a {apparent:g} Hz sinusoid these samples also fit" + (" (phase-inverted)" if f_sig < 0 else ""))
        ax.set_xlim(0, WINDOW)
        ax.set_ylim(-1.3, 1.3)
        ax.set_title(f"fs = {fs} Hz   (Nyquist {fs / 2:g} Hz)  ->  apparent frequency {apparent:g} Hz", fontsize=11)
        ax.legend(fontsize=8, loc="upper right", ncol=1)
        ax.grid(alpha=0.3)

        ax = axes[row, 1]
        ax.stem(f_axis, amp, basefmt=" ")
        ax.set_xlim(0, fs / 2)
        ax.set_ylim(0, 1.15)
        if F_TRUE <= fs / 2:
            ax.axvline(F_TRUE, color="darkgreen", linestyle=":", label="true 70 Hz")
        else:
            ax.text(0.98, 0.86, "true 70 Hz is off this axis", transform=ax.transAxes, ha="right", color="darkgreen")
        ax.annotate(f"peak at {peak:g} Hz", xy=(peak, amp.max()), xytext=(peak + fs * 0.05, amp.max() * 0.7),
                    arrowprops=dict(arrowstyle="->"))
        ax.set_title("sampled spectrum", fontsize=11)
        ax.set_ylabel("amplitude")
        ax.grid(alpha=0.3)
        if F_TRUE <= fs / 2:
            ax.legend(fontsize=8, loc="upper left")
    axes[-1, 0].set_xlabel("t (s)")
    axes[-1, 1].set_xlabel("frequency (Hz)")
    fig.suptitle("A 70 Hz sinusoid sampled at six rates", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(os.path.join(OUT_DIR, "aliasing.png"), dpi=150)

    # Figure 2: the folding map.
    f_scan = np.linspace(0, 300, 3001)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for fs, c in zip(SAMPLE_RATES, plt.cm.viridis(np.linspace(0, 0.9, len(SAMPLE_RATES)))):
        ax.plot(f_scan, np.abs(signed_alias(f_scan, fs)), color=c, linewidth=1.4, label=f"fs = {fs} Hz")
        ax.plot(F_TRUE, abs(signed_alias(F_TRUE, fs)), "o", color=c, markersize=8)
    ax.axvline(F_TRUE, color="gray", linestyle=":")
    ax.set_xlabel("true frequency (Hz)")
    ax.set_ylabel("apparent frequency (Hz)")
    ax.set_title("Folding map: apparent = |f - k fs|. Dots mark the 70 Hz signal at each sampling rate")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "aliasing_fold_map.png"), dpi=150)
    print("\nSaved aliasing.png and aliasing_fold_map.png")

    print(
        """
What this means
---------------
The same physical 70 Hz sinusoid gives six different-looking sample sets:
  - fs = 500 and 200 Hz: at least 2.86 samples per cycle, well inside the Nyquist
    limit (fs/2 > 70). The spectrum peak is at 70 Hz: the samples describe the
    signal correctly.
  - fs = 140 Hz: 70 Hz sits exactly at the Nyquist frequency. Two samples per
    cycle land wherever the phase puts them (here at +/-0.707), so the amplitude
    depends on the phase: with a zero-phase sine every sample would be 0.
  - fs = 120, 100, 80 Hz: below the required rate. The blue dashed curve is a
    slower sinusoid passing through the very same points, and the spectrum
    shows only that one: 50, 30 and 10 Hz. Nothing in the samples says the
    signal was ever 70 Hz. The rule is apparent f = |70 - k fs| (k a whole
    number chosen to land in 0..fs/2), and a negative result means the alias is
    phase-inverted.
Once sampled, the information distinguishing 70 Hz from its alias is gone. It
cannot be repaired afterwards, so the signal must be limited to below fs/2
BEFORE it is sampled: that is the anti-alias filter."""
    )
