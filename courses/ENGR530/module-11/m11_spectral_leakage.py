"""Windowing and spectral leakage: an FFT of a sinusoid whose record holds a
whole number of periods vs. one that ends partway through a cycle, then the
effect of rectangular, Hann, and Hamming windows."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "module-8", "signal_toolkit"))
from signals import time_axis
from transforms import dtft

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FS = 128                     # Hz
N = 128                      # samples -> a 1 s record, so FFT bins are 1 Hz apart
F_INT = 10.0                 # 10 whole periods in the record
F_FRAC = 10.5                # 10.5 periods: the record ends half way through a cycle
FLOOR_DB = -120.0
WINDOWS = [("rectangular (none)", np.ones(N)), ("Hann", np.hanning(N)), ("Hamming", np.hamming(N))]
COLORS = ["firebrick", "steelblue", "darkgreen"]


def amplitude_spectrum(x, w):
    """One-sided amplitude of the windowed FFT, scaled so a unit sine peaks near 1."""
    X = np.fft.fft(x * w)
    return 2 * np.abs(X[:N // 2 + 1]) / np.sum(w)


def dense_amplitude(x, w, freqs):
    """Same quantity on a fine frequency grid: the underlying DTFT that the FFT bins sample."""
    omega = 2 * np.pi * freqs / FS
    return 2 * np.abs(dtft(x * w, np.arange(N), omega)) / np.sum(w)


def to_db(a, ref):
    return np.maximum(20 * np.log10(np.maximum(a, 1e-30) / ref), FLOOR_DB)


def window_response(w, deltas):
    """|W| in dB relative to its peak, at offsets `deltas` (in bins) from the sinusoid."""
    omega = 2 * np.pi * deltas / FS
    W = np.abs(dtft(w, np.arange(N), omega))
    return to_db(W, W.max()), W


if __name__ == "__main__":
    t = time_axis(N / FS, FS)
    records = {"integer periods (10 Hz)": np.sin(2 * np.pi * F_INT * t),
               "partial cycle (10.5 Hz)": np.sin(2 * np.pi * F_FRAC * t)}
    bins = np.arange(N // 2 + 1)
    f_dense = np.linspace(0, FS / 2, 4001)

    print(f"x(t) = sin(2 pi f0 t), fs = {FS} Hz, N = {N} samples (1 s), FFT bins 1 Hz apart\n")
    print(f"{'record':<26}{'window':<20}{'peak bin':>9}{'peak amp':>10}{'bins 1 from peak (dB)':>23}{'worst |k-k0|>=8 (dB)':>22}{'energy beyond 2 bins':>22}")
    spectra = {}
    for (rname, x) in records.items():
        for (wname, w) in WINDOWS:
            a = amplitude_spectrum(x, w)
            spectra[(rname, wname)] = a
            k0 = int(np.argmax(a))
            near = max(a[k0 - 1], a[k0 + 1]) / a[k0]
            far = a[np.abs(bins - k0) >= 8].max() / a[k0]
            energy = a**2
            beyond = energy[np.abs(bins - k0) > 2].sum() / energy.sum()
            print(f"{rname:<26}{wname:<20}{k0:>9d}{a[k0]:>10.4f}{20 * np.log10(near + 1e-30):>23.1f}{20 * np.log10(far + 1e-30):>22.1f}{beyond:>22.2e}")

    a_int = spectra[("integer periods (10 Hz)", "rectangular (none)")]
    a_frac = spectra[("partial cycle (10.5 Hz)", "rectangular (none)")]
    print(f"\nInteger periods, no window: all other bins are at rounding level ({a_int[np.arange(N // 2 + 1) != 10].max():.1e}).")
    print(f"Partial cycle, no window: the peak drops to {a_frac.max():.4f} ({20 * np.log10(a_frac.max()):.1f} dB) and every bin is nonzero;")
    print(f"  bins 10 and 11 hold {a_frac[10]:.4f} and {a_frac[11]:.4f}, and bin 30 (20 bins away) still holds {a_frac[30]:.4f} ({20 * np.log10(a_frac[30] / a_frac.max()):.1f} dB).")

    # Window frequency responses: the shape that gets convolved with the line spectrum.
    deltas = np.linspace(-8, 8, 3201)
    print("\nWhat each window does to a pure tone (its own spectrum, in bins from the tone):")
    print(f"{'window':<20}{'first null (bins)':>19}{'main lobe width (bins)':>24}{'highest sidelobe (dB)':>23}")
    win_db = {}
    for (wname, w) in WINDOWS:
        db, W = window_response(w, deltas)
        win_db[wname] = db
        pos = deltas >= 0
        d, m = deltas[pos], W[pos]
        i = np.argmax((m[1:-1] < m[:-2]) & (m[1:-1] <= m[2:])) + 1        # first local minimum
        null = d[i]
        side = 20 * np.log10(m[i:].max() / m.max())
        print(f"{wname:<20}{null:>19.2f}{2 * null:>24.2f}{side:>23.1f}")

    # Figure 1: spectra in dB, 3 windows x 2 records, FFT bins over the underlying DTFT.
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True, sharey=True)
    for col, (rname, x) in enumerate(records.items()):
        for row, ((wname, w), c) in enumerate(zip(WINDOWS, COLORS)):
            ax = axes[row, col]
            a = spectra[(rname, wname)]
            ref = a.max()
            dense = to_db(dense_amplitude(x, w, f_dense), ref)
            ax.plot(f_dense, dense, color="lightgray", linewidth=2.5, label="underlying spectrum (fine grid)")
            ax.vlines(bins, FLOOR_DB, to_db(a, ref), color=c, linewidth=1.3)
            ax.plot(bins, to_db(a, ref), "o", color=c, markersize=3, label="FFT bins")
            ax.set_xlim(0, 40)
            ax.set_ylim(FLOOR_DB, 5)
            ax.set_title(f"{rname}  |  {wname}", fontsize=10)
            ax.grid(alpha=0.3)
            if col == 0:
                ax.set_ylabel("dB re peak")
            if row == 2:
                ax.set_xlabel("frequency (Hz)  = FFT bin")
            if row == 0:
                ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Spectral leakage: energy from one sinusoid spreading into neighbouring bins")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "spectral_leakage.png"), dpi=150)

    # Figure 2: why. Records, windows, and the window spectra that get convolved in.
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    t2 = np.concatenate([t, t + N / FS])
    for ax, (rname, x) in zip(axes[0], records.items()):
        ax.plot(t2, np.concatenate([x, x]), color="steelblue", linewidth=1.6)
        ax.axvline(N / FS, color="firebrick", linestyle="--", label="end of record, then repeats")
        ax.set_title(f"{rname}: the record, shown repeated")
        ax.set_xlabel("t (s)")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
    for (wname, w), c in zip(WINDOWS, COLORS):
        axes[1, 0].plot(t, w, color=c, linewidth=2, label=wname)
        axes[1, 1].plot(deltas, win_db[wname], color=c, linewidth=1.6, label=wname)
    axes[1, 0].set_title("The windows (multiplied into the record)")
    axes[1, 0].set_xlabel("t (s)")
    axes[1, 1].set_title("Their spectra, centered on the tone (convolved with its line)")
    axes[1, 1].set_xlabel("offset from the tone (bins)")
    axes[1, 1].set_ylabel("dB re peak")
    axes[1, 1].set_ylim(-100, 3)
    for ax in axes[1]:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "spectral_leakage_windows.png"), dpi=150)
    print("\nSaved spectral_leakage.png and spectral_leakage_windows.png")

    print(
        """
What this means
---------------
finite observation -> time multiplication -> frequency convolution -> leakage.
  - We never see the infinite sinusoid, only N samples of it. That is the sine
    multiplied by a rectangular window (1 inside the record, 0 outside).
  - Multiplying in time convolves in frequency. The sine's spectrum is two
    sharp lines; convolving them with the window's spectrum (a sinc-like shape
    with sidelobes) replaces each line by that shape.
  - The FFT only samples that shape at the bin frequencies. With a whole number
    of periods every bin except the peak lands exactly on a zero of the shape,
    so the spectrum looks like one clean line. That is luck of alignment, not
    absence of leakage.
  - With 10.5 periods the sinusoid sits between bins. The bins now fall on the
    shape's skirt and sidelobes: the peak is lower (about -3.7 dB) and energy
    smears into every bin, decaying slowly, still visible 20 bins away. The
    record ends mid-cycle, so the repeated record has a jump at the boundary,
    and a jump needs many frequencies to build.
  - Tapering the record's ends (Hann, Hamming) removes that jump. Their spectra
    have far lower sidelobes, so far-away leakage collapses (roughly -31 dB
    first sidelobe for Hann, -43 dB for Hamming, and much faster decay for Hann),
    but the main lobe is twice as wide (4 bins instead of 2), so the peak is
    blunter and nearby frequencies are harder to tell apart. Even for the
    integer-period record, a window makes the peak spread over 3 bins.
No window is best; each trades sidelobe level against main-lobe width. Choosing
among them comes later."""
    )
