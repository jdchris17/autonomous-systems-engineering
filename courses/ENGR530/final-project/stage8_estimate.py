"""Stage 8: estimate.

Two complementary things to recover from z[n]:
  1. the desired WAVEFORM x_hat[n]           (Stage 7's filtered output, plus a parametric reconstruction)
  2. the PARAMETERS A1, A2, f1, f2 (and the phases) inside it

    "Did I recover the waveform?"  and  "Did I identify the physical parameters?"
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import project_signal as P
import stage5_sample as S5
import stage7_filter as S7

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FS = S5.FS
EDGE = 2.0                       # seconds of x_hat ignored at each end when estimating (filter start-up region)
TRUTH = {"A1": P.A1, "A2": P.A2, "f1": P.F1, "f2": P.F2}


def peak_fft(y, fs, band):
    """FFT-peak estimate of (frequency, amplitude) inside `band` (Hz), with parabolic interpolation."""
    f, amp = P.amplitude_spectrum(y, fs)
    idx = np.where((f >= band[0]) & (f <= band[1]))[0]
    i = int(idx[np.argmax(amp[idx])])
    a, b, c = amp[i - 1], amp[i], amp[i + 1]
    denom = a - 2 * b + c
    delta = 0.0 if denom == 0 else 0.5 * (a - c) / denom
    return f[i] + delta * (f[1] - f[0]), float(b - 0.25 * (a - c) * delta)


def _design(t, f1, f2):
    w1, w2 = 2 * np.pi * f1 * t, 2 * np.pi * f2 * t
    return np.column_stack([np.sin(w1), np.cos(w1), np.sin(w2), np.cos(w2), np.ones_like(t)])


def _sse(t, y, f1, f2):
    A = _design(t, f1, f2)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ coef
    return float(r @ r), coef


def _golden(fun, lo, hi, iters=40):
    g = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc_, fd_ = fun(c), fun(d)
    for _ in range(iters):
        if fc_ < fd_:
            b, d, fd_ = d, c, fc_
            c = b - g * (b - a)
            fc_ = fun(c)
        else:
            a, c, fc_ = c, d, fd_
            d = a + g * (b - a)
            fd_ = fun(d)
    return 0.5 * (a + b)


def fit_two_sines(t, y, f1_0, f2_0, span=0.15, passes=4):
    """Least-squares fit of y ~ A1 sin(2 pi f1 t + p1) + A2 sin(2 pi f2 t + p2) + c.

    Amplitudes and phases are linear given the frequencies, so only f1 and f2 are searched
    (alternating golden-section searches in +/- span Hz around the FFT-peak start).
    """
    f1, f2 = f1_0, f2_0
    for _ in range(passes):
        f1 = _golden(lambda f: _sse(t, y, f, f2)[0], f1 - span, f1 + span)
        f2 = _golden(lambda f: _sse(t, y, f1, f)[0], f2 - span, f2 + span)
    _, (s1, c1, s2, c2, c0) = _sse(t, y, f1, f2)
    return {"A1": float(np.hypot(s1, c1)), "f1": f1, "p1": float(np.arctan2(c1, s1)),
            "A2": float(np.hypot(s2, c2)), "f2": f2, "p2": float(np.arctan2(c2, s2)), "c": float(c0)}


def estimate_parameters(x_hat, fs=FS, edge=EDGE, method="ls"):
    """Estimate A1, A2, f1, f2 (and phases) from the waveform estimate, ignoring `edge` s at each end."""
    n0 = int(edge * fs)
    seg = np.asarray(x_hat)[n0:len(x_hat) - n0] if n0 else np.asarray(x_hat)
    t = np.arange(len(seg)) / fs + (n0 / fs)
    f1, a1 = peak_fft(seg, fs, (0.2, 1.0))
    f2, a2 = peak_fft(seg, fs, (1.0, 4.0))
    if method == "fft":
        return {"A1": a1, "f1": f1, "A2": a2, "f2": f2}
    return fit_two_sines(t, seg, f1, f2)


def parametric_waveform(params, t):
    """x_p(t) = A1 sin(2 pi f1 t + p1) + A2 sin(2 pi f2 t + p2): the waveform rebuilt from the estimates."""
    return (params["A1"] * np.sin(2 * np.pi * params["f1"] * t + params["p1"])
            + params["A2"] * np.sin(2 * np.pi * params["f2"] * t + params["p2"]))


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def crlb_freq_std(amplitude, n_samples, fs=FS, sigma2=0.05):
    """Cramer-Rao bound on the std of a sinusoid's frequency (Hz) in white noise of per-sample variance sigma2.

    var(f_cycles_per_sample) >= 24 sigma^2 / ((2 pi)^2 A^2 N (N^2 - 1)).
    sigma2 = 0.05 is the white-equivalent variance of this project's noise at the ADC (S/2 x fs/2 with S = sigma_n^2 / 1000).
    """
    var = 24 * sigma2 / ((2 * np.pi) ** 2 * amplitude**2 * n_samples * (n_samples**2 - 1))
    return float(np.sqrt(var) * fs)


if __name__ == "__main__":
    a = S5.acquire()
    z, x_true, t_n = a["z_n"], a["x_true_n"], a["t_n"]
    x_hat = S7.estimate_waveform(z)
    n0 = int(EDGE * FS)

    print("1. DID I RECOVER THE WAVEFORM?")
    par_fft = estimate_parameters(x_hat, method="fft")
    par_ls = estimate_parameters(x_hat, method="ls")
    # x_true's own phases: phase 0 for the first line, PHI for the second
    x_par = parametric_waveform(par_ls, t_n)
    print(f"  {'estimate':<46}{'RMSE vs x_true':>16}{'excl. 2 s at each end':>24}")
    print(f"  {'raw z[n]':<46}{rmse(z, x_true):>16.3f}{rmse(z[n0:-n0], x_true[n0:-n0]):>24.3f}")
    print(f"  {'filtered x_hat[n] (Stage 7, model-free)':<46}{rmse(x_hat, x_true):>16.3f}{rmse(x_hat[n0:-n0], x_true[n0:-n0]):>24.3f}")
    print(f"  {'parametric x_p[n] from the fitted A, f, phase':<46}{rmse(x_par, x_true):>16.3f}{rmse(x_par[n0:-n0], x_true[n0:-n0]):>24.3f}")

    print("\n2. DID I IDENTIFY THE PHYSICAL PARAMETERS?  (whole record, edges excluded)")
    print(f"  {'':<10}{'true':>9}{'FFT peak':>12}{'err':>10}{'LS fit':>12}{'err':>11}")
    for key, unit in [("A1", ""), ("A2", ""), ("f1", " Hz"), ("f2", " Hz")]:
        tv = TRUTH[key]
        print(f"  {key + unit:<10}{tv:>9.4f}{par_fft[key]:>12.5f}{par_fft[key] - tv:>+10.5f}{par_ls[key]:>12.5f}{par_ls[key] - tv:>+11.5f}")
    ph_true = {"p1": 0.0, "p2": P.PHI}
    print(f"  phase p1  {ph_true['p1']:>9.4f}{'':>12}{'':>10}{par_ls['p1']:>12.5f}{par_ls['p1'] - ph_true['p1']:>+11.5f}   rad")
    print(f"  phase p2  {ph_true['p2']:>9.4f}{'':>12}{'':>10}{par_ls['p2']:>12.5f}{par_ls['p2'] - ph_true['p2']:>+11.5f}   rad  (true phi = pi/3)")

    # Record-length study: shorter analysis windows, many noise seeds.
    print("\n3. HOW DOES THE ESTIMATE DEPEND ON RECORD LENGTH?  (20 noise realizations; windows centered in x_hat)")
    windows = [3, 5, 7, 11, 16]
    seeds = range(20)
    errs = {(m, k, T): [] for m in ("fft", "ls") for k in TRUTH for T in windows}
    for sd in seeds:
        aa = S5.acquire(seed=sd)
        xh = S7.estimate_waveform(aa["z_n"])
        for T in windows:
            i0 = int((P.DURATION / 2 - T / 2) * FS)
            seg = xh[i0:i0 + int(T * FS)]
            for m in ("fft", "ls"):
                p = estimate_parameters(seg, edge=0.0, method=m)
                for k in TRUTH:
                    errs[(m, k, T)].append(p[k] - TRUTH[k])
    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
    print(f"  {'window (s)':>10}{'cycles of 0.5 Hz':>18}{'f1 err FFT':>13}{'f1 err LS':>12}{'CRLB f1':>10}{'f2 err FFT':>13}{'f2 err LS':>12}{'A1 err FFT':>12}{'A1 err LS':>11}")
    for T in windows:
        print(f"  {T:>10}{T * P.F1:>18.1f}{rms(errs[('fft', 'f1', T)]):>13.5f}{rms(errs[('ls', 'f1', T)]):>12.5f}{crlb_freq_std(P.A1, T * FS):>10.5f}"
              f"{rms(errs[('fft', 'f2', T)]):>13.5f}{rms(errs[('ls', 'f2', T)]):>12.5f}{rms(errs[('fft', 'A1', T)]):>12.4f}{rms(errs[('ls', 'A1', T)]):>11.4f}")
    print("  (RMS error over 20 noise seeds, in Hz for f and in signal units for A)")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax = axes[0, 0]
    sh = (t_n >= 8) & (t_n <= 14)
    ax.plot(t_n[sh], x_hat[sh], color="steelblue", linewidth=1.6, label="filtered x_hat[n]")
    ax.plot(t_n[sh], x_par[sh], color="firebrick", linewidth=1.2, linestyle="--", label="parametric x_p[n]")
    ax.plot(t_n[sh], x_true[sh], "k", linewidth=0.8, alpha=0.7, label="x_true[n]")
    ax.set_title("Recovered waveform (6 s from the middle of the record)")
    ax.set_xlabel("t (s)")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[0, 1]
    keys = ["A1", "A2", "f1", "f2"]
    rel_fft = [abs(par_fft[k] - TRUTH[k]) / TRUTH[k] * 100 for k in keys]
    rel_ls = [abs(par_ls[k] - TRUTH[k]) / TRUTH[k] * 100 for k in keys]
    xx = np.arange(4)
    ax.bar(xx - 0.19, [max(v, 1e-7) for v in rel_fft], 0.38, color="steelblue", label="FFT peak")
    ax.bar(xx + 0.19, [max(v, 1e-7) for v in rel_ls], 0.38, color="firebrick", label="least-squares fit")
    ax.set_yscale("log")
    ax.set_xticks(xx)
    ax.set_xticklabels(["A1", "A2", "f1", "f2"])
    ax.set_ylabel("relative error (%)")
    ax.set_title("Parameter estimates, whole record (20 s)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.loglog(windows, [rms(errs[("fft", "f1", T)]) for T in windows], "o-", color="steelblue", label="f1, FFT peak")
    ax.loglog(windows, [rms(errs[("ls", "f1", T)]) for T in windows], "s-", color="firebrick", label="f1, LS fit")
    ax.loglog(windows, [rms(errs[("fft", "f2", T)]) for T in windows], "o--", color="steelblue", alpha=0.6, label="f2, FFT peak")
    ax.loglog(windows, [rms(errs[("ls", "f2", T)]) for T in windows], "s--", color="firebrick", alpha=0.6, label="f2, LS fit")
    ax.loglog(windows, [crlb_freq_std(P.A1, T * FS) for T in windows], "k:", linewidth=2, label="Cramer-Rao bound, f1")
    ax.set_xlabel("analysis window (s)")
    ax.set_ylabel("RMS frequency error (Hz)")
    ax.set_title("Frequency error vs. record length (20 noise seeds)")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.loglog(windows, [rms(errs[("fft", "A1", T)]) for T in windows], "o-", color="steelblue", label="A1, FFT peak")
    ax.loglog(windows, [rms(errs[("ls", "A1", T)]) for T in windows], "s-", color="firebrick", label="A1, LS fit")
    ax.loglog(windows, [rms(errs[("fft", "A2", T)]) for T in windows], "o--", color="steelblue", alpha=0.6, label="A2, FFT peak")
    ax.loglog(windows, [rms(errs[("ls", "A2", T)]) for T in windows], "s--", color="firebrick", alpha=0.6, label="A2, LS fit")
    ax.set_xlabel("analysis window (s)")
    ax.set_ylabel("RMS amplitude error")
    ax.set_title("Amplitude error vs. record length")
    ax.legend(fontsize=8)
    for ax in axes.flatten():
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "stage8_estimate.png"), dpi=150)
    print("\nSaved stage8_estimate.png")

    print(
        """
What this means
---------------
The two questions have different answers and both matter.
  - Waveform: the filtered x_hat[n] recovers the motion without assuming anything
    about its form. The parametric x_p[n] assumes two sinusoids, which the spectrum
    justified, and is far more accurate because it keeps only the four numbers that
    describe the signal and throws away the noise inside the passband too.
  - Parameters: on the full 20 s record both methods identify the amplitudes and
    frequencies to a small fraction of a percent. The record-length study shows where
    they part ways. When the record holds a non-integer number of cycles, the FFT
    peak is limited by its bin spacing (a tone between bins reads up to 36% low in
    amplitude, "scalloping loss": the 3-2.9 amplitude errors above) while the least-squares fit
    keeps improving with record length and approaches the Cramer-Rao bound, the
    smallest error any unbiased estimator could achieve at this noise level.
  - Success is therefore judged two ways: how close the WAVEFORM is (RMSE, next
    stage) and how close the PARAMETERS are. A model-free estimate can match the
    waveform well and still say nothing about A and f; a fitted model gives the
    physical numbers directly."""
    )
