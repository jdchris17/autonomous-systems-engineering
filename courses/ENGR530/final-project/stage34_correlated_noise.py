"""Stage 34: the final correlated-noise case.

    n[k] = rho n[k-1] + eps[k]        rho = 0 (white)   vs   rho = 0.95 (strongly correlated)

Both noises are scaled to the SAME raw standard deviation, then sent through the same averaging /
low-pass filters and the same estimator. Total variance alone does not tell the story; the
spectral structure does.
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np

import project_signal as P
import stage7_filter as S7
import stage8_estimate as S8

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FS = S7.FS
SIGMA = 1.0                      # raw standard deviation of BOTH noises
RHOS = [0.0, 0.95]
N = int(P.DURATION * FS)         # 2000 samples per record
T_N = np.arange(N) / FS
X_TRUE = P.A1 * np.sin(2 * np.pi * P.F1 * T_N) + P.A2 * np.sin(2 * np.pi * P.F2 * T_N + P.PHI)
EDGE = int(S8.EDGE * FS)
DEFAULT_TRIALS = 500
LP_ONLY = S7.band_sections(use_hp=False)


def ar1_noise(rho, n, rng, sigma=SIGMA):
    """AR(1) noise n[k] = rho n[k-1] + eps[k] with stationary std = sigma.

    eps has std sigma sqrt(1 - rho^2) so that the STATIONARY variance is sigma^2 for every rho.
    The first sample is drawn from the stationary distribution (no start-up transient).
    """
    eps = rng.normal(0.0, sigma * np.sqrt(1 - rho**2), size=n)
    out = np.empty(n)
    out[0] = rng.normal(0.0, sigma)
    for k in range(1, n):
        out[k] = rho * out[k - 1] + eps[k]
    return out


def ar1_psd(rho, w, sigma=SIGMA):
    """Theoretical two-sided PSD of AR(1) noise at digital frequency w (rad/sample): sigma^2 (1-rho^2) / |1 - rho e^{-jw}|^2."""
    return sigma**2 * (1 - rho**2) / np.abs(1 - rho * np.exp(-1j * w)) ** 2


def ma_variance_theory(rho, M, sigma=SIGMA):
    """Exact variance of an M-point moving average of AR(1) noise: (sigma^2/M) [1 + 2 sum_{k=1}^{M-1} (1 - k/M) rho^k]."""
    k = np.arange(1, M)
    return sigma**2 / M * (1 + 2 * np.sum((1 - k / M) * rho**k))


def moving_average(x, M):
    return np.convolve(x, np.ones(M) / M, mode="valid")


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def trial(args):
    """One record of each noise type through the estimator. Returns per-rho numbers."""
    seed, rho = args
    rng = np.random.default_rng(1000 + seed)
    n = ar1_noise(rho, N, rng)
    z = X_TRUE + n                                            # no bias, drift or vibration: isolate the noise
    x_hat = S7.estimate_waveform(z, equalize=False)           # there is no analog front end in this synthetic case
    par = S8.estimate_parameters(x_hat, method="ls")
    lp = S7.zero_phase(n, LP_ONLY)                            # the 5 Hz low-pass alone, applied to the noise
    return {"rho": rho, "raw_std": float(n.std()), "lp_std": float(lp.std()), "ma10_std": float(moving_average(n, 10).std()),
            "rmse_hat": rmse(x_hat[EDGE:-EDGE], X_TRUE[EDGE:-EDGE]), "f1": par["f1"], "f2": par["f2"], "A1": par["A1"], "A2": par["A2"]}


if __name__ == "__main__":
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TRIALS
    workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"Correlated-noise experiment: {n_trials} independent records per noise type, {N} samples at {FS} Hz, raw std {SIGMA:g} for both")
    t0 = time.time()
    jobs = [(s, r) for r in RHOS for s in range(n_trials)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        res = list(pool.map(trial, jobs, chunksize=max(1, len(jobs) // (workers * 4))))
    print(f"finished in {time.time() - t0:.0f} s\n")
    by = {r: [x for x in res if x["rho"] == r] for r in RHOS}
    m = lambda r, k: np.array([x[k] for x in by[r]])

    print("1. SAME RAW STANDARD DEVIATION")
    for r in RHOS:
        print(f"   rho = {r:<5}  measured std {m(r, 'raw_std').mean():.4f}   (target {SIGMA:g})")

    print("\n2. THE SAME FILTERS, VERY DIFFERENT RESULTS  (output std of the noise alone)")
    print(f"   {'filter':<34}{'rho = 0':>10}{'rho = 0.95':>13}{'ratio':>8}")
    for name, k in [("10-point moving average", "ma10_std"), ("5 Hz zero-phase low-pass", "lp_std")]:
        a, b = m(0.0, k).mean(), m(0.95, k).mean()
        print(f"   {name:<34}{a:>10.4f}{b:>13.4f}{b / a:>8.1f}x")
    print("   moving-average output std vs. theory (white: sigma/sqrt(M);  AR(1): the exact formula), measured over the records")
    print(f"   {'M':>4}{'white: theory':>15}{'meas.':>9}{'rho=0.95: theory':>19}{'meas.':>9}{'ratio':>8}")
    rng = np.random.default_rng(7)
    long_w, long_c = ar1_noise(0.0, 400000, rng), ar1_noise(0.95, 400000, rng)
    for M in (2, 5, 10, 20, 50, 100):
        tw, tc = np.sqrt(ma_variance_theory(0.0, M)), np.sqrt(ma_variance_theory(0.95, M))
        print(f"   {M:>4}{tw:>15.4f}{moving_average(long_w, M).std():>9.4f}{tc:>19.4f}{moving_average(long_c, M).std():>9.4f}{tc / tw:>8.1f}x")
    print("   averaging M white samples cuts the std by sqrt(M); averaging correlated samples barely helps: they are nearly the same number")

    print("\n3. WHERE THE NOISE POWER SITS (fraction of the variance inside a frequency band)")
    w_grid = np.linspace(1e-6, np.pi, 200001)
    for lo, hi in [(0.0, 0.15), (0.15, 5.0), (0.0, 5.0), (5.0, 50.0)]:
        sel = (w_grid >= 2 * np.pi * lo / FS) & (w_grid <= 2 * np.pi * hi / FS)
        fr = {r: np.trapezoid(ar1_psd(r, w_grid)[sel], w_grid[sel]) / np.trapezoid(ar1_psd(r, w_grid), w_grid) for r in RHOS}
        print(f"   {lo:>5g}-{hi:<4g} Hz   white {100 * fr[0.0]:>5.1f}%   rho=0.95 {100 * fr[0.95]:>5.1f}%")
    print(f"   PSD at DC relative to white: {ar1_psd(0.95, 0.0):.1f}x  (sigma^2 (1+rho)/(1-rho));  at 0.5 Hz: {ar1_psd(0.95, 2 * np.pi * 0.5 / FS):.1f}x;  at 2 Hz: {ar1_psd(0.95, 2 * np.pi * 2 / FS):.2f}x")

    print(f"\n4. THROUGH THE WHOLE ESTIMATOR (Stage 7 band-pass + Stage 8 fit), {n_trials} records each, edges excluded")
    print(f"   {'':<26}{'rho = 0':>12}{'rho = 0.95':>14}{'ratio':>8}")
    for name, k in [("E[RMSE] of x_hat", "rmse_hat")]:
        a, b = m(0.0, k), m(0.95, k)
        print(f"   {name:<26}{a.mean():>12.4f}{b.mean():>14.4f}{b.mean() / a.mean():>7.1f}x   (std over trials {a.std(ddof=1):.4f} vs {b.std(ddof=1):.4f})")
    for key, tv, unit, fmt in [("f1", P.F1, "Hz", ".6f"), ("f2", P.F2, "Hz", ".6f"), ("A1", P.A1, "", ".4f"), ("A2", P.A2, "", ".4f")]:
        a, b = m(0.0, key) - tv, m(0.95, key) - tv
        sa, sb = a.std(ddof=1), b.std(ddof=1)
        print(f"   {'std of ' + key + ' error':<26}{sa:>12{fmt}}{sb:>14{fmt}}{sb / sa:>7.1f}x")
    crlb_white = S8.crlb_freq_std(P.A1, N - 2 * EDGE, sigma2=SIGMA**2)
    print(f"   white-noise Cramer-Rao bound for f1 at sigma = {SIGMA:g}: {crlb_white:.6f} Hz: rho = 0 is at {crlb_white / (m(0.0, 'f1') - P.F1).std(ddof=1) * 100:.0f}% of it; "
          f"rho = 0.95 is {(m(0.95, 'f1') - P.F1).std(ddof=1) / crlb_white:.1f}x above it")

    # Averaged periodograms for the picture.
    rng = np.random.default_rng(11)
    ps = {}
    for r in RHOS:
        acc = np.zeros(N // 2 + 1)
        for _ in range(200):
            acc += np.abs(np.fft.rfft(ar1_noise(r, N, rng))) ** 2 / (N * FS)
        ps[r] = 2 * acc / 200
    f = np.fft.rfftfreq(N, 1 / FS)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    ax = axes[0, 0]
    rng = np.random.default_rng(3)
    for r, c in zip(RHOS, ["gray", "firebrick"]):
        ax.plot(T_N[:800], ar1_noise(r, 800, rng), color=c, linewidth=0.9, label=f"rho = {r}")
    ax.set_title("Same std (1.0), very different texture (8 s)")
    ax.set_xlabel("t (s)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for r, c in zip(RHOS, ["gray", "firebrick"]):
        ax.semilogy(f[1:], ps[r][1:], color=c, linewidth=0.8, alpha=0.7, label=f"rho = {r}: averaged periodogram")
        ax.semilogy(f[1:], 2 * ar1_psd(r, 2 * np.pi * f[1:] / FS) / FS, "k--", linewidth=1.2, label="theory" if r == 0.95 else None)
    ax.axvspan(0.15, 5, color="steelblue", alpha=0.12, label="filter passband 0.15-5 Hz")
    ax.set_xscale("log")
    ax.set_title("Power spectral density: correlation moves the noise to low frequency")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD (per Hz)")
    ax.legend(fontsize=7)

    ax = axes[0, 2]
    Ms = np.array([1, 2, 5, 10, 20, 50, 100])
    for r, c in zip(RHOS, ["gray", "firebrick"]):
        ax.loglog(Ms, [np.sqrt(ma_variance_theory(r, M)) for M in Ms], "o-", color=c, label=f"rho = {r}")
    ax.loglog(Ms, 1 / np.sqrt(Ms), "k:", label="1/sqrt(M)")
    ax.set_title("Moving-average output std vs. window length")
    ax.set_xlabel("M (samples averaged)")
    ax.set_ylabel("output std")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    labels = ["10-pt moving\naverage", "5 Hz low-pass", "band-pass\nestimator RMSE"]
    a_v = [m(0.0, "ma10_std").mean(), m(0.0, "lp_std").mean(), m(0.0, "rmse_hat").mean()]
    b_v = [m(0.95, "ma10_std").mean(), m(0.95, "lp_std").mean(), m(0.95, "rmse_hat").mean()]
    x = np.arange(3)
    ax.bar(x - 0.19, a_v, 0.38, color="gray", label="rho = 0")
    ax.bar(x + 0.19, b_v, 0.38, color="firebrick", label="rho = 0.95")
    for i in range(3):
        ax.text(i - 0.19, a_v[i] + 0.01, f"{a_v[i]:.3f}", ha="center", fontsize=8)
        ax.text(i + 0.19, b_v[i] + 0.01, f"{b_v[i]:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Same input std, same filters: output error")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1, 1]
    for r, c in zip(RHOS, ["gray", "firebrick"]):
        ax.hist((m(r, "f1") - P.F1) * 1e3, bins=30, density=True, color=c, alpha=0.6, label=f"rho = {r}")
    ax.set_title("Frequency estimate f1: error over trials")
    ax.set_xlabel("f1 error (mHz)")
    ax.legend(fontsize=8)

    ax = axes[1, 1 + 1]
    for r, c in zip(RHOS, ["gray", "firebrick"]):
        ax.hist(m(r, "rmse_hat"), bins=30, density=True, color=c, alpha=0.6, label=f"rho = {r}")
    ax.set_title("RMSE of x_hat over trials")
    ax.set_xlabel("RMSE (edges excluded)")
    ax.legend(fontsize=8)
    for ax in axes.flatten():
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "stage34_correlated_noise.png"), dpi=150)
    print("\nSaved stage34_correlated_noise.png")

    r_ma = m(0.95, "ma10_std").mean() / m(0.0, "ma10_std").mean()
    r_rm = m(0.95, "rmse_hat").mean() / m(0.0, "rmse_hat").mean()
    r_f1 = (m(0.95, "f1") - P.F1).std(ddof=1) / (m(0.0, "f1") - P.F1).std(ddof=1)
    fr_in = {r: np.trapezoid(ar1_psd(r, w_grid)[(w_grid >= 2 * np.pi * 0.15 / FS) & (w_grid <= 2 * np.pi * 5 / FS)], w_grid[(w_grid >= 2 * np.pi * 0.15 / FS) & (w_grid <= 2 * np.pi * 5 / FS)]) / np.trapezoid(ar1_psd(r, w_grid), w_grid) for r in RHOS}
    print(
        f"""
What this means
---------------
Two noises with the SAME standard deviation ({SIGMA:g}) behave completely differently:
  - A 10-point moving average leaves {m(0.0, 'ma10_std').mean():.2f} of the white noise but {m(0.95, 'ma10_std').mean():.2f} of the rho = 0.95
    noise, {r_ma:.1f}x more. Averaging works because independent errors cancel; neighbouring
    samples of correlated noise share the same error, so there is little to cancel. The
    variance formula for the average shows it exactly: the correlation terms rho^k
    add to it instead of leaving 1/M.
  - The reason is in the spectrum. White noise spreads its power evenly, so a filter
    that keeps {100 * fr_in[0.0]:.0f}% of the bandwidth keeps about {100 * fr_in[0.0]:.0f}% of the noise. The rho = 0.95 noise
    puts {100 * fr_in[0.95]:.0f}% of its power in the 0.15-5 Hz band the estimator has to keep, because the
    band where the signal lives is exactly where correlated noise is concentrated
    (PSD {ar1_psd(0.95, 0.0):.0f}x white at DC).
  - Through the whole estimator, the RMSE of x_hat is {r_rm:.1f}x larger and the scatter of the
    frequency estimate {r_f1:.1f}x larger for the correlated noise, at identical raw std. The
    white-noise Cramer-Rao bound no longer applies: it assumes independent samples.
Total variance told us nothing about any of this. Probability (the correlation
structure of the noise) and frequency (where the signal and the filter sit in the
spectrum) are one subject: a noise's PSD is what a filter sees, and the same filter
can be nearly perfect or nearly useless depending on where that PSD puts the power."""
    )
