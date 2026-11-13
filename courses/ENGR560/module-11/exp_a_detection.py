"""
Experiment A: simulate a 512 x 512 image and detect stars with the six-step classical detector.

Nominal image: 50 stars, FWHM 3 px (pixel-integrated Gaussian PSF), uniform background 100 e-,
read noise 5 e-, 10-20 hot pixels, star flux log-uniform 300 - 30000 e-.
Detector: detect(image) -> N x 2 array of (x, y); threshold 5 sigma of the filtered background,
5 x 5 local-maximum window, minimum separation 4 px, filter matched to the nominal PSF.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from starfield import detect, match, simulate

OUT = Path(__file__).parent / "outputs"


def main():
    OUT.mkdir(exist_ok=True)
    img, truth, hot, meta = simulate(seed=0)
    det, diag = detect(img, return_diagnostics=True)
    tr = np.column_stack([truth["x"], truth["y"]])
    pairs, dist = match(det, tr, max_dist=2.0)
    matched_det, matched_true = set(pairs[:, 0]), set(pairs[:, 1])
    false = [i for i in range(len(det)) if i not in matched_det]
    missed = [k for k in range(len(tr)) if k not in matched_true]
    hp, _ = match(det[false], np.column_stack([hot["x"], hot["y"]]).astype(float), 1.0) if false else (np.zeros((0, 2)), None)
    print("Experiment A: simulate an image and detect stars")
    print(f"  image {meta['size']} x {meta['size']}, 50 stars, FWHM {meta['fwhm']} px, background {meta['background']} e-,"
          f" read noise {meta['read_noise']} e-, {len(hot['x'])} hot pixels; true star SNR {truth['snr'].min():.1f}-{truth['snr'].max():.0f}")
    print("  detector steps:")
    print(f"    1. background (sigma-clipped median) {diag['background']:.2f} e-, RMS {diag['rms']:.2f} e-"
          f" (expected sqrt(B + RN^2) = {np.sqrt(meta['background'] + meta['read_noise'] ** 2):.2f})")
    print("    2. background subtracted")
    print(f"    3. Gaussian filter (sigma {meta['fwhm'] / 2.3548:.2f} px): background RMS drops to {diag['filtered_rms']:.2f} e-"
          f" ({diag['rms'] / diag['filtered_rms']:.1f}x less noise per pixel)")
    print(f"    4. local maxima: {diag['n_peaks_before_threshold']} (almost all are noise bumps)")
    print(f"    5. threshold 5 x filtered RMS: {diag['n_after_threshold']} peaks")
    print(f"    6. minimum separation 4 px: {len(det)} detections returned (N x 2 array, shape {det.shape})")
    print(f"  matched to truth (one-to-one, <= 2 px): {len(pairs)} true matches, {len(missed)} missed stars,"
          f" {len(false)} false detections ({len(hp)} of them are hot pixels)")
    print(f"  recall {len(pairs) / 50:.2f}, precision {len(pairs) / len(det):.2f}; missed stars' true SNR:"
          f" {np.round(truth['snr'][missed], 1).tolist()}")
    print(f"  peak-pixel position error of matched detections: median {np.median(dist):.2f} px (integer pixels - centroiding"
          " comes in Experiment B)")

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0.02, 0.05, 0.55, 0.9])
    lo, hi = np.percentile(img, [5, 99.7])
    ax.imshow(np.clip(img, lo, hi), cmap="gray", origin="lower")
    ax.scatter(truth["x"][list(matched_true)], truth["y"][list(matched_true)], s=140, facecolor="none",
               edgecolor="lime", lw=1.3, label=f"true star, detected ({len(pairs)})")
    ax.scatter(truth["x"][missed], truth["y"][missed], s=140, facecolor="none", edgecolor="red", lw=2,
               label=f"true star, MISSED ({len(missed)})")
    ax.scatter(det[false, 0], det[false, 1], s=90, marker="x", c="orange", lw=2, label=f"false detection ({len(false)})")
    ax.scatter(hot["x"], hot["y"], s=30, marker="s", facecolor="none", edgecolor="cyan", lw=0.8,
               label=f"hot pixel (truth, {len(hot['x'])})")
    ax.set_title("Simulated image: truth vs detections (one-to-one match, 2 px)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax.set_xticks([])
    ax.set_yticks([])
    # step-by-step zoom around a faint star
    k = int(np.argsort(truth["snr"])[3])
    cx, cy = int(truth["x"][k]), int(truth["y"][k])
    sl = (slice(max(cy - 20, 0), cy + 21), slice(max(cx - 20, 0), cx + 21))
    panels = (("raw image", img[sl]), ("background subtracted", diag["subtracted"][sl]),
              ("Gaussian filtered", diag["filtered"][sl]),
              ("filtered > 5 sigma", np.where(diag["filtered"][sl] > 5 * diag["filtered_rms"], diag["filtered"][sl], np.nan)))
    for n, (title, a) in enumerate(panels):
        axp = fig.add_axes([0.6 + 0.2 * (n % 2), 0.52 - 0.45 * (n // 2), 0.18, 0.4])
        axp.imshow(a, cmap="viridis", origin="lower")
        axp.plot(truth["x"][k] - sl[1].start, truth["y"][k] - sl[0].start, "r+", ms=14)
        axp.set_title(f"{title}\n(faint star, SNR {truth['snr'][k]:.1f})", fontsize=9)
        axp.set_xticks([])
        axp.set_yticks([])
    fig.savefig(OUT / "exp_a_detections.png", dpi=110)
    plt.close(fig)
    print("  saved", OUT / "exp_a_detections.png")


if __name__ == "__main__":
    main()
