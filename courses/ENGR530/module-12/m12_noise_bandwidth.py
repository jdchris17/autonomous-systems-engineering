"""Noise bandwidth and SNR: a signal with useful content below 5 Hz, broadband
noise, and low-pass filters with cutoffs of 5, 10, 20, 50, 100 Hz."""

import os

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FS = 1000
DURATION = 20.0
NOISE_STD = 0.5                          # white noise, spread evenly over 0-500 Hz
COMPONENTS = [(1.0, 0.5), (1.0, 1.5), (1.0, 3.0), (1.0, 4.5)]      # (amplitude, Hz): all below 5 Hz
CUTOFFS = [5, 10, 20, 50, 100]
ORDER = 2                                 # gentle roll-off, so a cutoff at 5 Hz starts to bite


def make_signal(seed=0):
    t = np.arange(int(FS * DURATION)) / FS
    s = sum(a * np.sin(2 * np.pi * f * t) for a, f in COMPONENTS)
    n = np.random.default_rng(seed).normal(0.0, NOISE_STD, size=len(t))
    return t, s, n


def lowpass_gain(f, cutoff, order=ORDER):
    """Butterworth magnitude response. Applied as a zero-phase filter, so the numbers
    isolate bandwidth effects and are not muddied by a phase shift."""
    return 1.0 / np.sqrt(1.0 + (np.abs(f) / cutoff) ** (2 * order))


def apply_lowpass(x, cutoff):
    f = np.fft.fftfreq(len(x), 1 / FS)
    return np.fft.ifft(lowpass_gain(f, cutoff) * np.fft.fft(x)).real


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def evaluate(s, n, cutoff):
    """Return signal distortion, output noise RMS, SNR ignoring distortion, and total SNR."""
    ys = apply_lowpass(s, cutoff)
    yn = apply_lowpass(n, cutoff)
    y = apply_lowpass(s + n, cutoff)
    sig_err = rms(ys - s)
    noise_rms = rms(yn)
    snr_noise = 20 * np.log10(rms(ys) / noise_rms)
    snr_total = 20 * np.log10(rms(s) / rms(y - s))
    return sig_err, noise_rms, snr_noise, snr_total


if __name__ == "__main__":
    t, s, n = make_signal()
    x = s + n
    nbw_factor = np.pi / (2 * ORDER * np.sin(np.pi / (2 * ORDER)))        # noise bandwidth / cutoff for a Butterworth

    print(f"s(t): {', '.join(f'{a:g} sin(2 pi {f:g} t)' for a, f in COMPONENTS)}   (RMS {rms(s):.3f}, all below 5 Hz)")
    print(f"noise: white, std {NOISE_STD}, spread over 0-{FS // 2} Hz.  Filter: {ORDER}nd-order Butterworth low-pass (zero phase)\n")
    print(f"{'cutoff (Hz)':>12}{'signal error':>14}{'noise RMS':>11}{'predicted':>11}{'SNR (noise only)':>18}{'SNR (incl. distortion)':>24}")
    table = {}
    for fc in CUTOFFS:
        sig_err, noise_rms, snr_n, snr_t = evaluate(s, n, fc)
        predicted = NOISE_STD * np.sqrt(nbw_factor * fc / (FS / 2))
        table[fc] = (sig_err, noise_rms, snr_n, snr_t)
        print(f"{fc:>12g}{sig_err:>14.4f}{noise_rms:>11.4f}{predicted:>11.4f}{snr_n:>15.1f} dB{snr_t:>21.1f} dB")
    print(f"  (predicted noise RMS = sigma sqrt(noise bandwidth / 500 Hz), noise bandwidth = {nbw_factor:.3f} x cutoff)")
    print(f"\nUnfiltered: noise RMS {rms(n):.3f}, SNR {20 * np.log10(rms(s) / rms(n)):.1f} dB")

    # Continuous sweep of the cutoff.
    sweep = np.logspace(np.log10(3), np.log10(300), 60)
    res = np.array([evaluate(s, n, fc) for fc in sweep])
    best = int(np.argmax(res[:, 3]))
    best_listed = max(CUTOFFS, key=lambda fc: table[fc][3])
    print(f"\nBest cutoff on a fine sweep: {sweep[best]:.1f} Hz (total SNR {res[best, 3]:.1f} dB); best of the five listed: {best_listed} Hz ({table[best_listed][3]:.1f} dB)")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    f_full = np.fft.rfftfreq(len(x), 1 / FS)
    amp = 2 * np.abs(np.fft.rfft(x)) / len(x)
    ax = axes[0, 0]
    ax.loglog(f_full[1:], amp[1:], color="lightgray", linewidth=0.8, label="spectrum of s + n")
    ax.axvspan(0.3, 5, color="steelblue", alpha=0.12, label="useful band (< 5 Hz)")
    ff = np.logspace(-0.5, 2.7, 500)
    for fc, c in zip(CUTOFFS, plt.cm.viridis(np.linspace(0, 0.9, len(CUTOFFS)))):
        ax.loglog(ff, lowpass_gain(ff, fc), color=c, linewidth=1.5, label=f"|H|, cutoff {fc} Hz")
    ax.set_ylim(1e-3, 3)
    ax.set_title("The signal occupies a narrow band; the noise is everywhere")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("amplitude / gain")
    ax.legend(fontsize=7, ncol=2, loc="lower left")

    ax = axes[0, 1]
    ax.loglog(sweep, res[:, 1], color="firebrick", linewidth=2, label="output noise RMS")
    ax.loglog(sweep, res[:, 0] + 1e-6, color="steelblue", linewidth=2, label="signal error (distortion)")
    for fc in CUTOFFS:
        ax.plot(fc, table[fc][1], "o", color="firebrick")
        ax.plot(fc, max(table[fc][0], 1e-6), "s", color="steelblue")
    ax.set_ylim(1e-4, 1)
    ax.axvline(5, color="gray", linestyle=":")
    ax.set_title("Noise grows as sqrt(cutoff); distortion collapses once the cutoff clears 5 Hz")
    ax.set_xlabel("low-pass cutoff (Hz)")
    ax.set_ylabel("RMS")
    ax.legend()

    ax = axes[1, 0]
    ax.semilogx(sweep, res[:, 2], "--", color="firebrick", linewidth=1.8, label="SNR counting noise only")
    ax.semilogx(sweep, res[:, 3], color="black", linewidth=2.2, label="SNR counting noise + distortion")
    for fc in CUTOFFS:
        ax.plot(fc, table[fc][3], "o", color="black", markersize=8)
        ax.annotate(f"{fc} Hz\n{table[fc][3]:.1f} dB", xy=(fc, table[fc][3]), xytext=(0, -32), textcoords="offset points", ha="center", fontsize=8)
    ax.plot(sweep[best], res[best, 3], "*", color="gold", markeredgecolor="black", markersize=18, label=f"best: {sweep[best]:.1f} Hz")
    ax.set_title("SNR vs. cutoff: noise pushes the cutoff down, distortion pushes it up")
    ax.set_xlabel("low-pass cutoff (Hz)")
    ax.set_ylabel("SNR (dB)")
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1, 1]
    show = t <= 2.0
    ax.plot(t[show], s[show], "k--", linewidth=1.5, label="true s(t)")
    for fc, c in [(5, "firebrick"), (10, "seagreen"), (100, "darkorange")]:
        ax.plot(t[show], apply_lowpass(x, fc)[show], color=c, linewidth=1.2 if fc == 100 else 1.8,
                label=f"cutoff {fc} Hz (SNR {table[fc][3]:.1f} dB)", alpha=0.85)
    ax.set_title("Filtered outputs: 5 Hz rounds off the signal, 100 Hz lets the noise in")
    ax.set_xlabel("t (s)")
    ax.legend(fontsize=8, loc="upper right")
    for ax in axes.flatten():
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "noise_bandwidth.png"), dpi=150)
    print("\nSaved noise_bandwidth.png")

    ratio = table[100][1] / table[5][1]
    print(
        f"""
What this means
---------------
Filtering is a tradeoff, and the numbers show both sides:
  - Unnecessary bandwidth admits unnecessary noise. The white noise is spread
    evenly over 0-500 Hz, so the noise a low-pass lets through is proportional to
    its bandwidth and the RMS grows as sqrt(cutoff). Going from a 5 Hz to a 100 Hz
    cutoff (20x the bandwidth) multiplies the output noise by {ratio:.1f}. The signal does not
    gain anything from the extra bandwidth (it ends at 4.5 Hz), so the SNR that
    ignores distortion falls steadily from {table[5][2]:.1f} dB to {table[100][2]:.1f} dB.
  - Cutting too hard damages the signal. At a 5 Hz cutoff this filter is already
    down to about 0.77 at the 4.5 Hz component, so the signal error is {table[5][0]:.2f}, larger
    than the noise it saved. The total SNR is {table[5][3]:.1f} dB, worse than at 10 Hz.
  - The best cutoff is therefore just above the signal's edge: {table[10][3]:.1f} dB at 10 Hz on the
    listed cutoffs, {res[best, 3]:.1f} dB at {sweep[best]:.1f} Hz on the fine sweep. Everything past that is
    paying for noise and buying nothing.
More filtering is not better and less filtering is not better: the right
bandwidth is set by how wide the signal really is."""
    )
