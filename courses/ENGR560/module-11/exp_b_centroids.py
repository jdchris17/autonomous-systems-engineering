"""
Experiment B: centroid accuracy of the intensity-weighted centroid vs Gaussian PSF fitting.

20 independent images x 50 stars per condition. The processing architecture is FIXED at the nominal
design (matched filter for FWHM 3 px, centroid aperture radius 4 px, 11 x 11 fitting cutout, initial
fit sigma from FWHM 3) while the true PSF is defocused:
    nominal focus     FWHM 3.0 px
    moderate defocus  FWHM 4.5 px
    strong defocus    FWHM 6.5 px
Detections are matched one-to-one to the truth (Hungarian, max 2 px). Errors e = sqrt(dx^2 + dy^2) are
reported overall and by true-SNR bin, together with recall and false positives (missed stars are
reported, not hidden). Reference: the Cramer-Rao bound for a Gaussian PSF, per-axis
sigma_x >= sigma_PSF / SNR, i.e. radial RMS >= sqrt(2) sigma_PSF / SNR (no pixelation).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from starfield import FWHM_TO_SIGMA, detect, gaussian_fit, match, simulate, summarize, weighted_centroid

OUT = Path(__file__).parent / "outputs"
CONDITIONS = {"nominal focus": 3.0, "moderate defocus": 4.5, "strong defocus": 6.5}
N_IMAGES = 20
SNR_BINS = (0, 10, 20, 50, 100, np.inf)
NOMINAL_SIGMA = 3.0 * FWHM_TO_SIGMA


def process_image(seed, fwhm):
    img, truth, hot, meta = simulate(seed=seed, fwhm=fwhm)
    det, diag = detect(img, fwhm_assumed=3.0, return_diagnostics=True)
    tr = np.column_stack([truth["x"], truth["y"]])
    pairs, _ = match(det, tr, max_dist=2.0)
    rows = []
    for di, ti in pairs:
        cx, cy = weighted_centroid(diag["subtracted"], det[di, 0], det[di, 1], radius=4.0)
        f = gaussian_fit(img, diag["background"], det[di, 0], det[di, 1], NOMINAL_SIGMA, meta["read_noise"], half=5)
        rows.append((truth["snr"][ti], np.hypot(cx - tr[ti, 0], cy - tr[ti, 1]),
                     np.hypot(f["x"] - tr[ti, 0], f["y"] - tr[ti, 1]), f["sigma"]))
    matched = set(pairs[:, 1]) if len(pairs) else set()
    missed_snr = [truth["snr"][k] for k in range(len(tr)) if k not in matched]
    return rows, missed_snr, len(det) - len(pairs), meta["sigma"]


def main():
    OUT.mkdir(exist_ok=True)
    print(f"Experiment B: centroid accuracy ({N_IMAGES} images x 50 stars per condition; fixed nominal architecture)")
    res = {}
    for name, fwhm in CONDITIONS.items():
        out = Parallel(n_jobs=-1)(delayed(process_image)(100 + s, fwhm) for s in range(N_IMAGES))
        rows = np.array([r for o in out for r in o[0]])
        missed = np.array([m for o in out for m in o[1]])
        fp = sum(o[2] for o in out)
        res[name] = (rows, missed, fp, out[0][3])

    print(f"\n    {'condition':<18}{'method':<22}{'recall':>8}{'FP':>5}{'median':>9}{'p90':>8}{'RMSE':>8}{'fitted sigma':>22}")
    for name, (rows, missed, fp, sig) in res.items():
        n = len(rows) + len(missed)
        for col, meth in ((1, "weighted centroid"), (2, "Gaussian PSF fit")):
            s = summarize(rows[:, col])
            fs = f"{np.median(rows[:, 3]):.2f} (true {sig:.2f})" if col == 2 else ""
            print(f"    {name:<18}{meth:<22}{len(rows) / n:>8.3f}{fp:>5}{s['median']:>9.3f}{s['p90']:>8.3f}{s['rmse']:>8.3f}{fs:>22}")

    print("\n  By true SNR (median / 90th percentile centroid error in px; recall per bin):")
    hdr = "".join(f"{f'SNR {lo:g}-{hi:g}':>26}" for lo, hi in zip(SNR_BINS[:-1], SNR_BINS[1:]))
    print(f"    {'condition / method':<36}{hdr}")
    for name, (rows, missed, fp, sig) in res.items():
        allsnr = np.r_[rows[:, 0], missed]
        rec = []
        for lo, hi in zip(SNR_BINS[:-1], SNR_BINS[1:]):
            n_all = np.sum((allsnr >= lo) & (allsnr < hi))
            n_det = np.sum((rows[:, 0] >= lo) & (rows[:, 0] < hi))
            rec.append(f"recall {n_det}/{n_all}" if n_all else "-")
        print(f"    {name + ': recall':<36}" + "".join(f"{r:>26}" for r in rec))
        for col, meth in ((1, "weighted centroid"), (2, "Gaussian PSF fit")):
            cells = []
            for lo, hi in zip(SNR_BINS[:-1], SNR_BINS[1:]):
                m = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
                cells.append(f"{np.median(rows[m, col]):.3f} / {np.percentile(rows[m, col], 90):.3f}" if m.sum() >= 3 else "-")
            print(f"    {'  ' + meth:<36}" + "".join(f"{c:>26}" for c in cells))

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2), sharey=True)
    for ax, (name, (rows, missed, fp, sig)) in zip(axes, res.items()):
        for col, meth, c in ((1, "weighted centroid", "tab:orange"), (2, "Gaussian PSF fit", "tab:blue")):
            ax.scatter(rows[:, 0], rows[:, col], s=5, alpha=0.3, color=c)
            edges = np.geomspace(5, 400, 12)
            mid = np.sqrt(edges[1:] * edges[:-1])
            med = [np.median(rows[(rows[:, 0] >= a) & (rows[:, 0] < b), col]) if np.sum((rows[:, 0] >= a) & (rows[:, 0] < b)) >= 5
                   else np.nan for a, b in zip(edges[:-1], edges[1:])]
            ax.plot(mid, med, "o-", color=c, lw=2, label=f"{meth} (binned median)")
        snr = np.geomspace(5, 400, 50)
        ax.plot(snr, np.sqrt(2) * sig / snr, "k--", label="Cramer-Rao radial RMS, sqrt(2) sigma / SNR")
        if len(missed):
            ax.scatter(missed, np.full(len(missed), 2.5), marker="v", color="red", s=20, label=f"missed stars ({len(missed)})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(5e-3, 3)
        ax.set_xlabel("true star SNR")
        ax.set_title(f"{name} (true FWHM {CONDITIONS[name]} px)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="lower left")
    axes[0].set_ylabel("centroid error |e| [px]")
    fig.tight_layout()
    fig.savefig(OUT / "exp_b_centroid_vs_snr.png", dpi=130)
    plt.close(fig)
    print("\n  saved", OUT / "exp_b_centroid_vs_snr.png")


if __name__ == "__main__":
    main()
