"""Anti-alias filter simulation: why signal conditioning must happen BEFORE digitization.
    5 Hz desired motion + 90 Hz vibration, ADC at 100 Hz (90 Hz aliases to 10 Hz).
"""

import os

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FS_HIGH = 1000            # "continuous-like" rate
DURATION = 10.0
FS_ADC = 100              # final sample rate
FACTOR = FS_HIGH // FS_ADC
CUTOFF = 20.0             # Hz, anti-alias corner: well above 5, and 90 is far into the stopband
ORDER = 4
NOISE_STD = 0.2


def make_signal(seed=0):
    t = np.arange(int(FS_HIGH * DURATION)) / FS_HIGH
    motion = 1.0 * np.sin(2 * np.pi * 5 * t)
    vibration = 0.5 * np.sin(2 * np.pi * 90 * t)
    noise = np.random.default_rng(seed).normal(0.0, NOISE_STD, size=len(t))
    return t, motion, vibration, noise


def butterworth_lowpass(f, fc=CUTOFF, order=ORDER):
    """Complex frequency response of an analog Butterworth low-pass, DC gain 1."""
    s = 1j * 2 * np.pi * np.asarray(f, dtype=float)
    wc = 2 * np.pi * fc
    k = np.arange(1, order + 1)
    poles = wc * np.exp(1j * np.pi * (2 * k + order - 1) / (2 * order))
    H = np.ones_like(s, dtype=complex)
    for p in poles:
        H = H * (-p) / (s - p)
    return H


def analog_like_filter(x, fs):
    """Apply the analog Butterworth to a record by multiplying its spectrum by H(f)."""
    f = np.fft.fftfreq(len(x), 1 / fs)
    return np.fft.ifft(butterworth_lowpass(f) * np.fft.fft(x)).real


def amplitude_spectrum(x, fs):
    N = len(x)
    amp = 2 * np.abs(np.fft.rfft(x)) / N
    amp[0] /= 2
    return np.fft.rfftfreq(N, 1 / fs), amp


def contamination(y, fs):
    """RMS of what is left after removing the best-fit 5 Hz sinusoid from the samples."""
    n = np.arange(len(y)) / fs
    A = np.column_stack([np.sin(2 * np.pi * 5 * n), np.cos(2 * np.pi * 5 * n)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(np.sqrt(np.mean((y - A @ coef) ** 2)))


if __name__ == "__main__":
    t, motion, vibration, noise = make_signal()
    x = motion + vibration + noise

    # Pipeline A: signal -> downsample.  Pipeline B: signal -> analog-like low-pass -> downsample.
    y_a = x[::FACTOR]
    y_b = analog_like_filter(x, FS_HIGH)[::FACTOR]
    # Pipeline C (for contrast): downsample first, then the same low-pass in the digital domain.
    y_c = analog_like_filter(y_a, FS_ADC)
    t_adc = t[::FACTOR]

    f_hi, a_hi = amplitude_spectrum(x, FS_HIGH)
    f_lo, a_a = amplitude_spectrum(y_a, FS_ADC)
    _, a_b = amplitude_spectrum(y_b, FS_ADC)
    _, a_c = amplitude_spectrum(y_c, FS_ADC)

    def at(a, f, f0):
        return a[int(np.argmin(np.abs(f - f0)))]

    H = butterworth_lowpass
    print(f"High-rate data at {FS_HIGH} Hz: 5 Hz motion (1.0) + 90 Hz vibration (0.5) + white noise (std {NOISE_STD})")
    print(f"ADC rate {FS_ADC} Hz (Nyquist {FS_ADC / 2:g} Hz): 90 Hz aliases to |90 - 100| = 10 Hz")
    print(f"Anti-alias filter: {ORDER}th-order Butterworth, corner {CUTOFF:g} Hz; |H(5 Hz)| = {abs(H(5.0)):.5f}, |H(90 Hz)| = {abs(H(90.0)):.5f}\n")
    print(f"{'pipeline':<44}{'5 Hz amp':>10}{'10 Hz amp':>11}{'residual RMS':>14}{'noise floor':>13}")
    floor = lambda a, f: np.median(a[(f > 20) & (f < 49)])
    print(f"{'high-rate data (before the ADC)':<44}{at(a_hi, f_hi, 5):>10.3f}{'-':>11}{'-':>14}{floor(a_hi, f_hi):>13.4f}   (90 Hz line: {at(a_hi, f_hi, 90):.3f})")
    print(f"{'A: downsample only':<44}{at(a_a, f_lo, 5):>10.3f}{at(a_a, f_lo, 10):>11.3f}{contamination(y_a, FS_ADC):>14.3f}{floor(a_a, f_lo):>13.4f}")
    print(f"{'B: low-pass, then downsample':<44}{at(a_b, f_lo, 5):>10.3f}{at(a_b, f_lo, 10):>11.4f}{contamination(y_b, FS_ADC):>14.3f}{floor(a_b, f_lo):>13.4f}")
    print(f"{'C: downsample, then the same low-pass':<44}{at(a_c, f_lo, 5):>10.3f}{at(a_c, f_lo, 10):>11.3f}{contamination(y_c, FS_ADC):>14.3f}{floor(a_c, f_lo):>13.4f}")
    print("\n  residual RMS = what remains after removing the best-fit 5 Hz sinusoid (aliased vibration + noise)")
    print(f"  noise floor = median amplitude of the 20-49 Hz bins: A is {floor(a_a, f_lo) / floor(a_hi, f_hi):.1f}x the high-rate floor "
          f"(50-500 Hz noise folded in), B is {floor(a_b, f_lo) / floor(a_hi, f_hi):.2f}x (the filter removed it first)")

    delay_phase = np.angle(H(5.0))
    ref5 = np.sin(2 * np.pi * 5 * t_adc + delay_phase)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax = axes[0, 0]
    m = f_hi <= 200
    ax.plot(f_hi[m], a_hi[m], color="steelblue", linewidth=1)
    ax.set_ylim(0, 1.15)
    ax2 = ax.twinx()
    ff = np.linspace(0, 200, 2000)
    ax2.plot(ff, np.abs(H(ff)), color="firebrick", linewidth=1.6, label=f"anti-alias filter |H| (corner {CUTOFF:g} Hz)")
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("|H|", color="firebrick")
    ax.axvline(FS_ADC / 2, color="black", linestyle="--")
    ax.text(FS_ADC / 2 + 2, 0.55, "ADC Nyquist\n50 Hz", fontsize=9)
    ax.annotate("5 Hz motion", xy=(5, 1.0), xytext=(14, 0.95), arrowprops=dict(arrowstyle="->"))
    ax.annotate("90 Hz vibration", xy=(90, 0.5), xytext=(105, 0.75), arrowprops=dict(arrowstyle="->"))
    ax.set_title("Before the ADC: the vibration sits beyond the 50 Hz Nyquist limit")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("amplitude")
    ax2.legend(loc="center right", fontsize=8)

    for ax, a, title in [(axes[0, 1], a_a, "Pipeline A: downsample only"), (axes[1, 0], a_b, "Pipeline B: low-pass, then downsample")]:
        ax.stem(f_lo, a, basefmt=" ")
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 1.15)
        ax.set_title(title)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("amplitude")
    axes[0, 1].annotate("ALIAS of the 90 Hz vibration\n(amplitude 0.5)", xy=(10, at(a_a, f_lo, 10)), xytext=(18, 0.7), arrowprops=dict(arrowstyle="->"), color="firebrick")
    axes[1, 0].annotate("nothing at 10 Hz", xy=(10, 0.03), xytext=(18, 0.35), arrowprops=dict(arrowstyle="->"), color="darkgreen")

    ax = axes[1, 1]
    sel = t_adc <= 1.0
    ax.plot(t_adc[sel], ref5[sel], "k--", linewidth=1.2, label="5 Hz motion (delayed by the filter's phase)")
    ax.plot(t_adc[sel], y_a[sel], "o-", color="firebrick", markersize=4, linewidth=1, label=f"A (residual RMS {contamination(y_a, FS_ADC):.2f})")
    ax.plot(t_adc[sel], y_b[sel], "s-", color="steelblue", markersize=4, linewidth=1.2, label=f"B (residual RMS {contamination(y_b, FS_ADC):.2f})")
    ax.set_title("What the 100 Hz samples look like (first second)")
    ax.set_xlabel("t (s)")
    ax.legend(fontsize=8, loc="upper right")
    for ax in axes.flatten():
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "antialias.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.stem(f_lo - 0.25, a_a, linefmt="C3-", markerfmt="C3o", basefmt=" ", label="A: downsample only")
    ax.stem(f_lo + 0.25, a_c, linefmt="C2-", markerfmt="C2s", basefmt=" ", label="C: downsample, THEN low-pass")
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 1.15)
    ax.annotate("the alias is now a 'real' 10 Hz signal:\nfiltering after the ADC cannot remove it", xy=(10, at(a_c, f_lo, 10)),
                xytext=(13, 0.75), arrowprops=dict(arrowstyle="->"))
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("amplitude")
    ax.set_title("Filtering after digitization is too late")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "antialias_too_late.png"), dpi=150)
    print("Saved antialias.png and antialias_too_late.png")

    print(
        """
What this means
---------------
Pipeline A keeps every 10th sample of the raw data. The 90 Hz vibration is
beyond the 50 Hz Nyquist limit, so it lands at |90 - 100| = 10 Hz with its full
amplitude (0.5), looking like a genuine 10 Hz component next to the 5 Hz motion.
The noise fares no better: everything from 50 to 500 Hz folds into 0-50 Hz too,
so A's noise floor is about 3x higher (the square root of the 10x rate drop).
Pipeline B low-passes first. The filter has almost no effect at 5 Hz but cuts 90
Hz by a factor of about 400, so almost nothing is left to alias, and the noise
above the corner is removed as well. The 10 Hz line disappears and the residual
error drops from about 0.4 to about 0.04.
Pipeline C shows why the order matters: applying the same filter AFTER the ADC
leaves the alias untouched. Once sampled, an alias is indistinguishable from a
real 10 Hz signal, and a 10 Hz signal is exactly what the low-pass is supposed
to keep. The information that separated 90 Hz from 10 Hz was lost at the
sampling instant, so the protection has to be analog hardware ahead of the ADC.
It also forces a design choice: the corner must be low enough to crush
everything above fs/2 yet high enough not to disturb the 5 Hz you want."""
    )
