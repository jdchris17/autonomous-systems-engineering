"""Stage 6: analyze the sampled spectrum.

FFT of z[n], automatic detection of the dominant peaks, and a comparison of every
detected peak against the known truth. Also runs the same analysis on a stream
sampled WITHOUT the anti-alias filter, to show the tool flagging what should not be there.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import project_signal as P
import stage4_antialias as S4
import stage5_sample as S5

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Truth table: (name, frequency in Hz, true amplitude). The 20 Hz row is where the 80 Hz
# interference WOULD appear if it aliased; its "truth" for a filtered stream is ~0.
TRUTH = [("bias (DC)", 0.0, P.B0), ("drift", P.F_B, P.B_DRIFT), ("desired: primary motion", P.F1, P.A1),
         ("desired: maneuvering", P.F2, P.A2), ("vibration", P.F_V, P.A_V)]
ALIAS_LINE = ("alias of 80 Hz interference", abs(P.F_H - S5.FS * round(P.F_H / S5.FS)))


def refine_frequency(f, amp, i):
    """Sub-bin peak frequency from a parabola through the peak bin and its two neighbors."""
    if i <= 0 or i >= len(amp) - 1:
        return f[i]
    a, b, c = amp[i - 1], amp[i], amp[i + 1]
    denom = a - 2 * b + c
    return f[i] if denom == 0 else f[i] + 0.5 * (a - c) / denom * (f[1] - f[0])


def local_floor(amp, half_window=30):
    """Running median of the spectrum: the noise floor AT each frequency.

    The sampled noise is not flat (the anti-alias filter shapes it), so a single global
    floor would put the threshold far too low where the noise is highest. A median is
    robust to the few spectral lines inside each window.
    """
    n = len(amp)
    return np.array([np.median(amp[max(0, i - half_window):min(n, i + half_window + 1)]) for i in range(n)])


def detect_peaks(z, fs, k=8.0):
    """FFT amplitude spectrum and its dominant peaks: (frequency, amplitude, bin index)."""
    f, amp = P.amplitude_spectrum(z, fs)
    floor_curve = local_floor(amp)
    threshold = k * floor_curve
    idx = [i for i in range(1, len(amp) - 1) if amp[i] > threshold[i] and amp[i] >= amp[i - 1] and amp[i] >= amp[i + 1]]
    if amp[0] > threshold[0]:
        idx.insert(0, 0)
    peaks = [(refine_frequency(f, amp, i) if i > 0 else 0.0, float(amp[i]), i) for i in idx]
    return f, amp, floor_curve, peaks


def identify(freq, tol=0.04):
    """Nearest truth line within tol Hz (under half an FFT bin), or UNEXPECTED."""
    candidates = [(name, f0) for name, f0, _ in TRUTH] + [ALIAS_LINE]
    name, f0 = min(candidates, key=lambda c: abs(freq - c[1]))
    return name if abs(freq - f0) <= tol else "UNEXPECTED"


if __name__ == "__main__":
    a = S5.acquire()
    z, fs = a["z_n"], a["fs"]
    N = len(z)
    f, amp, floor, peaks = detect_peaks(z, fs)
    df = f[1] - f[0]

    print(f"FFT of z[n]: {N} samples at {fs:g} Hz -> {len(f)} bins from 0 to {fs / 2:g} Hz, resolution {df:g} Hz (= 1 / {N * S5.TS:g} s)")
    print(f"noise floor (local running median): {floor.min():.4f} to {floor.max():.4f} across the band; a peak must stand 8x above the local floor\n")

    print("Dominant spectral peaks:")
    for fr, am, _ in sorted(peaks, key=lambda p: -p[1]):
        print(f"  {fr:5.2f} Hz   amplitude {am:6.3f}")

    hf_gain = lambda f0: float(S4.gain(f0))
    print("\nCOMPARISON AGAINST THE KNOWN TRUTH")
    print(f"  {'peak (Hz)':>10}{'measured':>10}{'true':>8}{'true x |H|':>12}{'error vs true x |H|':>21}   identified as")
    matched, unexpected = set(), []
    for fr, am, _ in sorted(peaks, key=lambda p: p[0]):
        name = identify(fr)
        if name == "UNEXPECTED":
            unexpected.append(fr)
            print(f"  {fr:>10.2f}{am:>10.3f}{'-':>8}{'-':>12}{'-':>21}   UNEXPECTED")
            continue
        matched.add(name)
        if name.startswith("alias"):
            print(f"  {fr:>10.2f}{am:>10.3f}{P.A_H:>8.3f}{P.A_H * hf_gain(P.F_H):>12.4f}{'':>21}   {name}")
            continue
        true_amp = next(t for n_, _, t in TRUTH if n_ == name)
        f0 = next(f0 for n_, f0, _ in TRUTH if n_ == name)
        exp = true_amp * (hf_gain(f0) if f0 > 0 else 1.0)
        print(f"  {fr:>10.2f}{am:>10.3f}{true_amp:>8.3f}{exp:>12.3f}{100 * (am - exp) / exp:>19.2f} %   {name}")
    missing = [n_ for n_, _, _ in TRUTH if n_ not in matched]
    print(f"\n  truth lines found: {len(matched & {t[0] for t in TRUTH})} of {len(TRUTH)}; missing: {missing if missing else 'none'}; unexpected peaks: {unexpected if unexpected else 'none'}")
    max_ferr = max(abs(fr - next(f0 for n_, f0, _ in TRUTH if n_ == identify(fr))) for fr, _, _ in peaks if identify(fr) in {t[0] for t in TRUTH})
    print(f"  largest frequency error among matched lines: {max_ferr:.4f} Hz (bin width {df:g} Hz)")
    a20 = float(amp[int(round(ALIAS_LINE[1] / df))])
    print(f"  {ALIAS_LINE[1]:g} Hz bin (where the 80 Hz interference would alias): {a20:.4f}, {'not detected' if identify(ALIAS_LINE[1]) not in [identify(p[0]) for p in peaks] else 'DETECTED'}"
          f" (unfiltered it would be {P.A_H:g}; the filter leaves {P.A_H * hf_gain(P.F_H):.4f}, so this bin is just noise)")

    print("\nSAME ANALYSIS, STREAM SAMPLED WITHOUT THE ANTI-ALIAS FILTER")
    f2, amp2, floor2, peaks2 = detect_peaks(a["z_raw_n"], fs)
    for fr, am, _ in sorted(peaks2, key=lambda p: p[0]):
        name = identify(fr)
        print(f"  {fr:>6.2f} Hz   amplitude {am:>6.3f}   {name}" + ("   <-- the aliased 80 Hz interference" if name.startswith("alias") else ""))
    print(f"  median noise floor {np.median(floor2):.4f} vs {np.median(floor):.4f} with the filter")

    fig, axes = plt.subplots(3, 1, figsize=(13, 14))
    ax = axes[0]
    ax.semilogy(f[1:], amp[1:] + 1e-9, color="steelblue", linewidth=0.9, label="|FFT| of z[n]")
    for fr, am, _ in peaks:
        name = identify(fr)
        ax.plot(fr, am, "v", color="firebrick", markersize=8)
        ax.annotate(f"{fr:.2f} Hz", xy=(fr, am), xytext=(fr + 0.6, am * 1.5), fontsize=8)
    for name, f0, t in TRUTH:
        if f0 > 0:
            ax.plot(f0, t, "kx", markersize=9)
    ax.plot([], [], "kx", label="known truth line")
    ax.plot([], [], "v", color="firebrick", label="detected peak")
    ax.semilogy(f, 8 * floor, color="gray", linestyle=":", label="detection threshold (8x local floor)")
    ax.set_ylim(1e-3, 40)
    ax.set_xlim(0, fs / 2)
    ax.set_title("Sampled spectrum with detected peaks and the known truth")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("amplitude")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    names = [t[0].replace("desired: ", "") for t in TRUTH]
    meas = [float(amp[int(round(t[1] / df))]) for t in TRUTH]
    exp = [t[2] * (hf_gain(t[1]) if t[1] > 0 else 1.0) for t in TRUTH]
    x = np.arange(len(TRUTH))
    ax.bar(x - 0.19, [t[2] for t in TRUTH], 0.38, color="lightgray", label="true amplitude")
    ax.bar(x + 0.19, meas, 0.38, color="steelblue", label="measured from the FFT")
    ax.plot(x + 0.19, exp, "k_", markersize=25, markeredgewidth=2, label="true x |H| of the anti-alias filter")
    for i, m_ in enumerate(meas):
        ax.text(i + 0.19, m_ + 0.2, f"{m_:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title("Measured vs. true amplitude, line by line")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.semilogy(f2[1:], amp2[1:] + 1e-9, color="firebrick", linewidth=0.9, label="no anti-alias filter")
    ax.semilogy(f[1:], amp[1:] + 1e-9, color="steelblue", linewidth=0.9, alpha=0.8, label="with the anti-alias filter")
    ax.annotate("aliased 80 Hz -> 20 Hz", xy=(ALIAS_LINE[1], 2.0), xytext=(27, 6), arrowprops=dict(arrowstyle="->"), color="firebrick")
    ax.set_ylim(1e-3, 40)
    ax.set_xlim(0, fs / 2)
    ax.set_title("The same analysis flags the alias when the anti-alias filter is left out")
    ax.set_xlabel("frequency (Hz)")
    ax.legend(fontsize=8, loc="upper right")
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "stage6_analyze.png"), dpi=150)
    print("\nSaved stage6_analyze.png")

    print(
        """
What this means
---------------
The analysis code finds the peaks without being told where to look: a threshold set
from the measured noise floor, local maxima above it, and a parabola through each
peak for a sub-bin frequency. Then it becomes an analysis TOOL rather than a plot,
because every result is checked against the known truth:
  - all five truth lines are found at their true frequencies (0.5, 2 and 15 Hz for
    the motion and vibration, 0.05 Hz for the drift, DC for the bias), to within
    a small fraction of the 0.05 Hz bin width,
  - measured amplitudes agree with the truth once the (tiny) gain of the analog
    filter is included, and the 15 Hz vibration shows the only visible effect of
    the filter (a few percent low),
  - nothing unexpected is reported: no line at 20 Hz, so the 80 Hz interference
    was stopped before it could alias.
The last check is what makes the tool trustworthy. Leave the anti-alias filter out
and the same code reports a peak at 20 Hz, amplitude 2, that looks exactly like a
real 20 Hz signal: the only reason we know it is an alias is that we know the truth.
On real data that knowledge has to come from the acquisition design, which is why
the assumptions printed in Stage 5 matter."""
    )
