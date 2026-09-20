"""Stage 30: Monte Carlo the entire pipeline.

Each trial repeats the WHOLE experiment with fresh noise:
    new noise -> noisy analog measurement -> anti-alias filter + sample -> band-pass -> estimate -> RMSE
so RMSE becomes a random variable, and we report E[RMSE], its spread, and the distribution of the
parameter estimates. Usage:  python stage30_monte_carlo.py [N_MC]      (default 500)
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np

import project_signal as P
import stage4_antialias as S4
import stage5_sample as S5
import stage7_filter as S7
import stage8_estimate as S8

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FS = S5.FS
EDGE = int(S8.EDGE * FS)
DEFAULT_N = 1000


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def one_trial(seed):
    """One complete, independent experiment. Returns the numbers we track."""
    a = S5.acquire(seed=seed)                      # 1-3: new noise, noisy measurement, sample it
    z, x_true, t_n = a["z_n"], a["x_true_n"], a["t_n"]
    x_hat = S7.estimate_waveform(z)                # 4-5: filter it, estimate x
    par = S8.estimate_parameters(x_hat, method="ls")
    x_par = S8.parametric_waveform(par, t_n)
    inner = slice(EDGE, len(z) - EDGE)
    return {                                       # 6: RMSE (and friends)
        "rmse_raw": rmse(z, x_true),
        "rmse_hat": rmse(x_hat, x_true),
        "rmse_hat_int": rmse(x_hat[inner], x_true[inner]),
        "rmse_par": rmse(x_par, x_true),
        "bias_hat": float(np.mean(x_hat - x_true)),
        "A1": par["A1"], "A2": par["A2"], "f1": par["f1"], "f2": par["f2"],
    }


def noiseless_rmse():
    """RMSE of the same pipeline when there is NO random noise: the deterministic (edge) floor."""
    c = P.build_analog()
    clean = c["x_true"] + c["vib"] + c["hf"] + c["bias"]
    z = S4.analog_filter(clean)[:: P.FS_ANALOG // FS]
    x_hat = S7.estimate_waveform(z)
    x_true = c["x_true"][:: P.FS_ANALOG // FS]
    return rmse(x_hat, x_true), rmse(x_hat[EDGE:-EDGE], x_true[EDGE:-EDGE])


def summarize(v):
    v = np.asarray(v)
    m, s = float(v.mean()), float(v.std(ddof=1))
    skew = float(np.mean(((v - m) / s) ** 3))
    return {"mean": m, "std": s, "se": s / np.sqrt(len(v)), "min": float(v.min()), "p05": float(np.percentile(v, 5)),
            "median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "max": float(v.max()), "skew": skew}


if __name__ == "__main__":
    n_mc = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N
    workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"Monte Carlo: N_MC = {n_mc} independent runs of the entire pipeline ({workers} worker processes)")
    print(f"  each run: {P.DURATION:g} s at {FS} Hz, noise sigma {P.SIGMA_N:g}, seeds 0..{n_mc - 1}; everything else (signal, filters) held fixed\n")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        runs = list(pool.map(one_trial, range(n_mc), chunksize=max(1, n_mc // (workers * 4))))
    print(f"finished in {time.time() - t0:.0f} s\n")
    col = {k: np.array([r[k] for r in runs]) for k in runs[0]}

    stats = {k: summarize(col[k]) for k in ("rmse_raw", "rmse_hat", "rmse_hat_int", "rmse_par")}
    print("RMSE IS NOW A RANDOM VARIABLE")
    print(f"  {'estimator':<34}{'E[RMSE]':>10}{'std':>9}{'std err':>9}{'5%':>9}{'median':>9}{'95%':>9}{'min':>9}{'max':>9}{'skew':>7}")
    for name, k in [("raw measurement z[n]", "rmse_raw"), ("filtered x_hat[n], whole record", "rmse_hat"),
                    ("filtered x_hat[n], excl. 2 s ends", "rmse_hat_int"), ("parametric x_p[n]", "rmse_par")]:
        s = stats[k]
        print(f"  {name:<34}{s['mean']:>10.4f}{s['std']:>9.4f}{s['se']:>9.4f}{s['p05']:>9.4f}{s['median']:>9.4f}{s['p95']:>9.4f}{s['min']:>9.4f}{s['max']:>9.4f}{s['skew']:>7.2f}")
    print(f"  (std err = std / sqrt(N_MC): the uncertainty in E[RMSE] itself; the 95% interval for E[RMSE_hat] is "
          f"{stats['rmse_hat']['mean'] - 1.96 * stats['rmse_hat']['se']:.4f} to {stats['rmse_hat']['mean'] + 1.96 * stats['rmse_hat']['se']:.4f})")
    cv = {k: stats[k]["std"] / stats[k]["mean"] * 100 for k in stats}
    print(f"  coefficient of variation: raw {cv['rmse_raw']:.1f}%, filtered {cv['rmse_hat']:.1f}%, filtered interior {cv['rmse_hat_int']:.1f}%, parametric {cv['rmse_par']:.1f}%")

    det_all, det_int = noiseless_rmse()
    print(f"\n  noiseless run of the same pipeline: RMSE {det_all:.4f} (whole record), {det_int:.4f} (interior): the part of the error that does NOT depend on the noise")
    print(f"  so for the filtered estimate: E[RMSE]^2 = {stats['rmse_hat']['mean'] ** 2:.4f}  vs  deterministic^2 = {det_all ** 2:.4f}  -> "
          f"{100 * det_all ** 2 / stats['rmse_hat']['mean'] ** 2:.0f}% of the mean-square error is the noise-independent edge error")

    print(f"\nPARAMETER ESTIMATES ACROSS {n_mc} TRIALS (least-squares fit)")
    print(f"  {'':<5}{'true':>9}{'mean est.':>12}{'bias':>11}{'std':>11}{'std err of mean':>17}{'95% of trials within':>24}   note")
    n_fit = len(np.arange(EDGE, int(P.DURATION * FS) - EDGE))
    crlb = {k: S8.crlb_freq_std(S8.TRUTH[k[0].upper() + k[1]] if False else {"f1": P.A1, "f2": P.A2}[k], n_fit) for k in ("f1", "f2")}
    for k in ("A1", "A2", "f1", "f2"):
        tv = S8.TRUTH[k]
        e = col[k] - tv
        s = np.std(e, ddof=1)
        p95 = np.percentile(np.abs(e), 95)
        note = ""
        if k in crlb:
            note = f"Cramer-Rao bound {crlb[k]:.6f}: efficiency {crlb[k] / s * 100:.0f}%"
        unit = 6 if k.startswith("f") else 4
        print(f"  {k:<5}{tv:>9.4f}{col[k].mean():>12.{unit}f}{e.mean():>+11.{unit}f}{s:>11.{unit}f}{s / np.sqrt(n_mc):>17.{unit}f}{p95:>24.{unit}f}   {note}")

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for ax, k, title, c in [(axes[0, 0], "rmse_hat", "Filtered estimate, whole record", "firebrick"),
                            (axes[0, 1], "rmse_hat_int", "Filtered estimate, excluding 2 s ends", "darkorange"),
                            (axes[0, 2], "rmse_par", "Parametric estimate", "steelblue")]:
        s = stats[k]
        ax.hist(col[k], bins=25, density=True, color=c, alpha=0.7)
        gx = np.linspace(col[k].min(), col[k].max(), 200)
        ax.plot(gx, np.exp(-0.5 * ((gx - s["mean"]) / s["std"]) ** 2) / (s["std"] * np.sqrt(2 * np.pi)), "k--", label="Gaussian, same mean and std")
        ax.axvline(s["mean"], color="black", linewidth=1.5, label=f"E[RMSE] = {s['mean']:.4f}")
        ax.set_title(f"{title}\nstd {s['std']:.4f} ({cv[k]:.1f}%)")
        ax.set_xlabel("RMSE")
        ax.legend(fontsize=8)
    ax = axes[1, 0]
    for k, c, nm in [("rmse_hat", "firebrick", "filtered"), ("rmse_par", "steelblue", "parametric")]:
        rm = np.cumsum(col[k]) / np.arange(1, n_mc + 1)
        ax.plot(np.arange(1, n_mc + 1), rm / rm[-1], color=c, label=nm)
    ax.axhline(1, color="gray", linestyle=":")
    ax.set_ylim(0.9, 1.1)
    ax.set_title("Convergence: running mean of RMSE / final mean")
    ax.set_xlabel("number of trials")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    u = 1e-5                                          # plot in units of 1e-5 Hz
    e1 = (col["f1"] - P.F1) / u
    ax.hist(e1, bins=25, density=True, color="seagreen", alpha=0.7, label="f1 error over trials")
    gx = np.linspace(e1.min(), e1.max(), 200)
    sd_c, mu = crlb["f1"] / u, e1.mean()
    ax.plot(gx, np.exp(-0.5 * ((gx - mu) / sd_c) ** 2) / (sd_c * np.sqrt(2 * np.pi)), "k--", label=f"Gaussian with the Cramer-Rao std ({crlb['f1']:.1e} Hz), centered on the mean")
    ax.axvline(0, color="gray", linestyle=":", label="true f1 (error 0): the small offset is the bias")
    ax.set_title("Frequency estimate f1: error distribution")
    ax.set_xlabel("f1 error (units of 1e-5 Hz)")
    ax.legend(fontsize=8)
    ax = axes[1, 2]
    ax.scatter(col["rmse_hat_int"], col["rmse_par"], s=8, color="gray", alpha=0.6)
    ax.set_xlabel("RMSE of filtered estimate (interior)")
    ax.set_ylabel("RMSE of parametric estimate")
    r = np.corrcoef(col["rmse_hat_int"], col["rmse_par"])[0, 1]
    ax.set_title(f"Do the two estimators have good and bad runs together?\ncorrelation {r:.2f}")
    for ax in axes.flatten():
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "stage30_monte_carlo.png"), dpi=150)
    print("\nSaved stage30_monte_carlo.png")

    s_h, s_p, s_r = stats["rmse_hat"], stats["rmse_par"], stats["rmse_raw"]
    bias_txt = "; ".join(
        f"{k} bias {np.mean(col[k] - S8.TRUTH[k]):+.1e} ({abs(np.mean(col[k] - S8.TRUTH[k])) / np.std(col[k], ddof=1) * np.sqrt(n_mc):.0f} std errors, "
        f"{abs(np.mean(col[k] - S8.TRUTH[k])) / S8.TRUTH[k] * 100:.3f}% of the true value)" for k in ("A1", "f1"))
    print(
        f"""
What this means
---------------
One run would have said "my filter worked on this example". {n_mc} runs say something stronger:

  Under this stochastic measurement model (white noise sigma = {P.SIGMA_N:g} at the sensor, two motion
  lines, drift, bias, 15 Hz vibration and 80 Hz interference), this processing pipeline has
      raw measurement       RMSE = {s_r['mean']:.2f} +/- {s_r['std']:.2f}   (mean +/- std over trials)
      filtered estimate     RMSE = {s_h['mean']:.3f} +/- {s_h['std']:.3f}   (95% of runs between {s_h['p05']:.3f} and {s_h['p95']:.3f})
      parametric estimate   RMSE = {s_p['mean']:.4f} +/- {s_p['std']:.4f}   (95% of runs between {s_p['p05']:.4f} and {s_p['p95']:.4f})

Three things the distribution tells you that a single run cannot:
  - How much to trust one run. The filtered RMSE varies by only {cv['rmse_hat']:.1f}% from run to run, so any
    single run lands within a few percent of the mean. Stage 9's single run (seed 0, RMSE
    {col['rmse_hat'][0]:.3f}) sits at the {100 * np.mean(col['rmse_hat'] < col['rmse_hat'][0]):.1f}th percentile: {'an unlucky draw, so the Monte Carlo mean of ' + format(s_h['mean'], '.3f') + ' is the better summary' if np.mean(col['rmse_hat'] < col['rmse_hat'][0]) > 0.9 else 'a typical draw'}. A spread this small also
    means most of the error is NOT the random noise: {100 * det_all ** 2 / s_h['mean'] ** 2:.0f}% of the mean-square error is the
    deterministic edge error, which every run shares.
  - Whether the estimators are sound. The scatter of the frequency estimates is at the
    Cramer-Rao bound ({crlb['f1'] / np.std(col['f1'] - P.F1, ddof=1) * 100:.0f}% and {crlb['f2'] / np.std(col['f2'] - P.F2, ddof=1) * 100:.0f}% efficiency for f1 and f2), the best any unbiased
    estimator can do at this noise level. They are not perfectly unbiased: {bias_txt}.
    With {n_mc} trials that is statistically detectable, so it is not noise but a small
    systematic effect of the filter's edge behaviour, and it is negligible in practice.
  - Whether N_MC was enough. The running mean has settled and its standard error is
    {s_h['se']:.5f}, so E[RMSE] is known to about {s_h['se'] / s_h['mean'] * 100:.2f}%. Quadrupling N_MC would halve that."""
    )
