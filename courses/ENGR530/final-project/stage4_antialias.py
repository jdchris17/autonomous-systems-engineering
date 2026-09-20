"""Stage 4: design the anti-alias strategy.

Useful content is below 5 Hz; choose fs = 100 Hz (Nyquist 50 Hz). Anything above 50 Hz
can alias, and the simulated measurement contains exactly that: 80 Hz interference and
broadband noise out to 1000 Hz. Compare sampling with and without an analog-like
low-pass filter placed BEFORE the ADC.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import project_signal as P

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "module-12"))
from m12_sampling_rate_selection import butter_response  # noqa: E402  (analog Butterworth, from module 12)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
F_USE = 5.0                    # useful band (Hz)
FS = 100                       # chosen ADC rate (Hz)
NYQ = FS / 2
FACTOR = P.FS_ANALOG // FS
ORDER = 4
FC = float(np.sqrt(F_USE * P.F_H))       # 20 Hz: geometric mean of the useful-band edge and the interference


def gain(f, fc=FC, order=ORDER):
    return np.abs(butter_response(f, fc, order))


def analog_filter(x):
    """Analog-like anti-alias filter: multiply the fine-grid spectrum by the complex H(f)."""
    f = np.fft.fftfreq(len(x), 1 / P.FS_ANALOG)
    return np.fft.ifft(butter_response(f, FC, ORDER) * np.fft.fft(x)).real


def fold(f, fs=FS):
    return abs(f - fs * np.round(f / fs))


def band_limit(x, fs, cutoff):
    """Ideal digital low-pass, used only to measure how much noise lands in the useful band."""
    f = np.fft.fftfreq(len(x), 1 / fs)
    return np.fft.ifft(np.where(np.abs(f) <= cutoff, 1.0, 0.0) * np.fft.fft(x)).real


def db(g):
    return 20 * np.log10(g)


if __name__ == "__main__":
    c = P.build_analog()
    z = c["z_analog"]

    print("DECISIONS")
    print(f"  useful content   : below {F_USE:g} Hz (motion tops out at {P.F2:g} Hz; bias and drift are below 0.1 Hz)")
    print(f"  ADC rate         : {FS} Hz -> Nyquist {NYQ:g} Hz ({NYQ / F_USE:.0f}x the useful band)")
    print(f"  anything above {NYQ:g} Hz can alias. In this measurement that is the {P.F_H:g} Hz interference and noise from {NYQ:g} to {P.FS_ANALOG // 2} Hz.")
    print(f"  The {P.F_V:g} Hz vibration is BELOW Nyquist, so sampling keeps it as a true 15 Hz line and a later digital filter can remove it.\n")

    print(f"WHERE THINGS LAND AFTER SAMPLING AT {FS} Hz (apparent frequency = |f - k x {FS}|)")
    print(f"  {P.F_H:g} Hz interference -> {fold(P.F_H):g} Hz   (outside 0-{F_USE:g} Hz, but it would sit in the spectrum looking like a real 20 Hz signal)")
    print(f"  noise 50-150 Hz    -> folds onto 50-0 Hz, and so does every 100 Hz-wide slice above it: {P.FS_ANALOG // 2 // FS - 0} slices stack up on top of 0-{NYQ:g} Hz")
    print(f"  danger zone for the useful band: content at {FS - F_USE:g}-{FS + F_USE:g} Hz, {2 * FS - F_USE:g}-{2 * FS + F_USE:g} Hz, ... folds directly onto 0-{F_USE:g} Hz")

    print(f"\nTHE FILTER: {ORDER}th-order Butterworth low-pass, corner {FC:g} Hz (geometric mean of {F_USE:g} and {P.F_H:g} Hz)")
    rows = [("useful band edge", F_USE), ("15 Hz vibration", P.F_V), ("corner", FC), ("Nyquist", NYQ),
            ("80 Hz interference", P.F_H), (f"danger zone edge ({FS - F_USE:g} Hz)", FS - F_USE), ("500 Hz", 500.0)]
    print(f"  {'frequency':<32}{'Hz':>7}{'gain':>10}{'dB':>8}")
    for name, f0 in rows:
        g = float(gain(f0))
        print(f"  {name:<32}{f0:>7g}{g:>10.5f}{db(g):>8.1f}")
    g5 = float(gain(F_USE))
    delay_ms = 1e3 / (2 * np.pi * FC) / np.sin(np.pi / (2 * ORDER))
    print(f"  passband: {abs(db(g5)):.5f} dB of loss at {F_USE:g} Hz; low-frequency group delay about {delay_ms:.1f} ms")
    print(f"  the 15 Hz vibration passes almost untouched ({db(float(gain(P.F_V))):.1f} dB): removing it is the digital filter's job, not this one's")

    # Pipelines.
    y_a = z[::FACTOR]                                   # A: sample the raw analog signal
    y_b = analog_filter(z)[::FACTOR]                    # B: analog low-pass, then sample
    noise_a = c["noise"][::FACTOR]
    noise_b = analog_filter(c["noise"])[::FACTOR]
    hf_b = analog_filter(c["hf"])[::FACTOR]

    f_lo, a_a = P.amplitude_spectrum(y_a, FS)
    _, a_b = P.amplitude_spectrum(y_b, FS)

    def at(a, f0):
        return float(a[int(np.argmin(np.abs(f_lo - f0)))])

    floor_a = float(np.median(a_a[(f_lo > 25) & (f_lo < 49)]))
    floor_b = float(np.median(a_b[(f_lo > 25) & (f_lo < 49)]))
    n_a = float(np.sqrt(np.mean(band_limit(noise_a, FS, F_USE) ** 2)))
    n_b = float(np.sqrt(np.mean(band_limit(noise_b, FS, F_USE) ** 2)))
    n_pred_a = P.SIGMA_N * np.sqrt(F_USE / NYQ)
    n_pred_b = P.SIGMA_N * np.sqrt(F_USE / (P.FS_ANALOG / 2))

    print(f"\nRESULTS after sampling at {FS} Hz ({len(y_a)} samples over {P.DURATION:g} s)")
    print(f"  {'':<40}{'A: sample only':>16}{'B: filter, then sample':>24}")
    print(f"  {'amplitude at 0.5 Hz (true 10)':<40}{at(a_a, 0.5):>16.3f}{at(a_b, 0.5):>24.3f}")
    print(f"  {'amplitude at 2 Hz (true 4)':<40}{at(a_a, 2.0):>16.3f}{at(a_b, 2.0):>24.3f}")
    print(f"  {'amplitude at 15 Hz (vibration, true 3)':<40}{at(a_a, 15.0):>16.3f}{at(a_b, 15.0):>24.3f}")
    print(f"  {'amplitude at 20 Hz (alias of the 80 Hz)':<40}{at(a_a, 20.0):>16.3f}{at(a_b, 20.0):>24.4f}")
    print(f"  {'noise floor, median bin 25-49 Hz':<40}{floor_a:>16.4f}{floor_b:>24.4f}   (ratio {floor_a / floor_b:.1f}x)")
    print(f"  {'noise RMS inside 0-5 Hz':<40}{n_a:>16.3f}{n_b:>24.3f}   (predicted {n_pred_a:.3f} and {n_pred_b:.3f}: ratio {n_a / n_b:.1f}x)")
    print(f"  the 80 Hz line leaves B with amplitude {float(np.std(hf_b) * np.sqrt(2)):.4f} (was {P.A_H:g}: attenuated by {P.A_H / (np.std(hf_b) * np.sqrt(2)):.0f}x)")
    print(f"  analog filter cost on the wanted lines: 0.5 Hz amplitude {at(a_b, 0.5) / at(a_a, 0.5) * 100:.2f}% of the unfiltered, 2 Hz {at(a_b, 2.0) / at(a_a, 2.0) * 100:.2f}%")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    f_z, a_z = P.amplitude_spectrum(z, P.FS_ANALOG)
    ax = axes[0, 0]
    ax.semilogy(f_z[1:], a_z[1:] + 1e-9, color="lightgray", linewidth=0.7, label="z_analog spectrum")
    ff = np.logspace(-1, np.log10(1000), 800)
    ax.semilogy(ff, gain(ff), color="firebrick", linewidth=2.2, label=f"anti-alias |H| ({ORDER}th-order, {FC:g} Hz)")
    ax.axvline(NYQ, color="black", linestyle="--", label=f"Nyquist {NYQ:g} Hz")
    ax.axvspan(0, F_USE, color="steelblue", alpha=0.15, label=f"useful band 0-{F_USE:g} Hz")
    ax.axvspan(FS - F_USE, FS + F_USE, color="darkorange", alpha=0.2, label="folds onto useful band")
    ax.set_xscale("log")
    ax.set_xlim(0.1, 1000)
    ax.set_ylim(1e-5, 30)
    ax.set_title("1. Before the ADC: the analog filter sits between the signal and Nyquist")
    ax.set_xlabel("frequency (Hz)")
    ax.legend(fontsize=7, loc="lower left")

    for ax, a, title, col in [(axes[0, 1], a_a, f"2. A: sample only ({FS} Hz)", "firebrick"), (axes[1, 0], a_b, f"3. B: analog filter, then sample ({FS} Hz)", "seagreen")]:
        ax.semilogy(f_lo[1:], a[1:] + 1e-9, color=col, linewidth=0.9)
        ax.set_xlim(0, NYQ)
        ax.set_ylim(1e-4, 30)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("amplitude")
        ax.set_title(title)
        ax.axvspan(0, F_USE, color="steelblue", alpha=0.12)
    axes[0, 1].annotate("alias of the 80 Hz\ninterference", xy=(20, at(a_a, 20.0)), xytext=(28, 5), arrowprops=dict(arrowstyle="->"), color="firebrick")
    axes[0, 1].axhline(floor_a, color="gray", linestyle=":", label=f"noise floor {floor_a:.4f}")
    axes[1, 0].axhline(floor_b, color="gray", linestyle=":", label=f"noise floor {floor_b:.4f}")
    axes[1, 0].annotate("15 Hz vibration (kept for now)", xy=(15, at(a_b, 15.0)), xytext=(22, 6), arrowprops=dict(arrowstyle="->"))
    axes[1, 0].annotate("no alias at 20 Hz: only noise", xy=(20, at(a_b, 20.0)), xytext=(24, 0.3), arrowprops=dict(arrowstyle="->"), color="seagreen")
    for ax in (axes[0, 1], axes[1, 0]):
        ax.legend(fontsize=8, loc="lower right")

    ax = axes[1, 1]
    labels = ["amplitude at 20 Hz\n(alias of the 80 Hz)", "noise RMS inside\nthe 0-5 Hz band"]
    vals_a = [at(a_a, 20.0), n_a]
    vals_b = [at(a_b, 20.0), n_b]
    x = np.arange(2)
    ax.bar(x - 0.19, vals_a, 0.38, color="firebrick", label="A: sample only")
    ax.bar(x + 0.19, vals_b, 0.38, color="seagreen", label="B: filter first")
    for i in range(2):
        ax.text(i - 0.19, vals_a[i] + 0.03, f"{vals_a[i]:.3g}", ha="center", fontsize=8)
        ax.text(i + 0.19, vals_b[i] + 0.03, f"{vals_b[i]:.3g}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("4. What the filter buys")
    ax.legend()
    for ax in axes.flatten():
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "stage4_antialias.png"), dpi=150)
    print("\nSaved stage4_antialias.png")

    print(
        f"""
What this means
---------------
The spectrum in Stage 3 said the useful content is tiny (below 5 Hz) and the
measurement is not: an 80 Hz interference line and noise out to 1000 Hz sit far above
the {NYQ:g} Hz Nyquist limit of a {FS} Hz ADC. Sampling does not ignore them, it folds them:
  - The 80 Hz line reappears at 20 Hz with full amplitude ({at(a_a, 20.0):.2f}) as a fake
    signal. Here it lands outside the 0-5 Hz band, so it can still be removed
    digitally, but only because of where it happens to fall. A line at 98 Hz would
    land at 2 Hz, on top of the maneuvering component, and no later filter could
    separate them.
  - The noise does the quiet damage. All noise from 50 to 1000 Hz folds onto 0-50 Hz,
    the noise inside the useful band is {n_a / n_b:.1f}x larger ({n_a:.2f} vs {n_b:.2f} RMS), close to the
    square root of the 20x bandwidth that folds in (predicted {n_pred_a / n_pred_b:.1f}x). This cannot be
    filtered off afterwards, because after sampling it is indistinguishable from
    real content in that band. (The 25-49 Hz floor is lower still in B, by {floor_a / floor_b:.0f}x,
    but that includes the filter's own roll-off there, so it overstates the effect.)
A {ORDER}th-order {FC:g} Hz analog filter fixes both before the ADC: it cuts 80 Hz by {abs(db(float(gain(P.F_H)))):.0f} dB
and the {FS - F_USE:g} Hz danger-zone edge by {abs(db(float(gain(FS - F_USE)))):.0f} dB, while leaving 0-5 Hz untouched. The corner is
the geometric mean of the useful band edge and the interference, so it is far from
both. It costs about {delay_ms:.0f} ms of delay and deliberately does NOT remove the 15 Hz
vibration: that is below Nyquist, remains a true 15 Hz line after sampling, and is a
job for the digital filter in the next stage. This is why the pipeline is analog
protection first, then sampling, then digital filtering."""
    )
