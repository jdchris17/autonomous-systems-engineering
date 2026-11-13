"""
Module 12 data: detector-proposed 32 x 32 candidate patches with labels.

Image simulation = Module 11's model (pixel-integrated Gaussian PSF, Poisson + read noise in
electrons, hot pixels) plus ONE extra artifact family, elongated cosmic-ray streaks, so that the
detector also proposes elongated false candidates (needed for Experiment B). Streak: a line of length
6-20 px, any angle, Gaussian cross-section sigma 0.6 px, 500-5000 e- deposited.

Imaging conditions
  nominal       flux 300-30000 e-, background 100 e-, read noise 5 e-, FWHM 3 px     (Module 11)
  low SNR       flux 100-3000 e-,  background 300 e-, read noise 10 e-, FWHM 3 px    (Module 11)
  strong defocus  nominal but FWHM 6.5 px                                           (Module 11)
  training mix  per image: background U(80, 320), read noise U(4, 10), flux floor U(100, 300),
                flux ceiling U(3000, 30000), FWHM U(2.7, 3.3) - spans nominal..low-SNR, never defocused

Candidates come from the UNCHANGED Module 11 detector (matched filter for FWHM 3, 5 sigma). Each is
labelled by one-to-one matching (Hungarian, <= 2 px) to the true stars:
  star      -> subtype "bright star" (true SNR >= 20) or "faint star"
  otherwise -> "hot pixel" (within 1.5 px of a hot pixel), "elongated" (within 2 px of a streak's
               core), else "noise / other" (noise peaks, star wings, blends)
Splits are by IMAGE (independent seeds): train / validation / test never share an image.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "module-11"))

from starfield import (FWHM_TO_SIGMA, background_stats, detect, gaussian_fit, match,  # noqa: E402,F401
                       pixel_integrated_gaussian)

OUT = HERE / "outputs"
PATCH = 32
PAD = 2                        # extract 36 x 36 so training can apply +-2 px translations by cropping
SUBTYPES = ("bright star", "faint star", "hot pixel", "elongated", "noise / other")
CONDITIONS = {
    "nominal": dict(flux_range=(300.0, 30000.0), background=100.0, read_noise=5.0, fwhm=3.0),
    "low SNR": dict(flux_range=(100.0, 3000.0), background=300.0, read_noise=10.0, fwhm=3.0),
    "strong defocus": dict(flux_range=(300.0, 30000.0), background=100.0, read_noise=5.0, fwhm=6.5),
}


def simulate(seed, size=512, n_stars=50, fwhm=3.0, background=100.0, read_noise=5.0, n_hot=(10, 20),
             flux_range=(300.0, 30000.0), n_streaks=(2, 6), margin=10):
    rng = np.random.default_rng(seed)
    sigma = fwhm * FWHM_TO_SIGMA
    x = rng.uniform(margin, size - 1 - margin, n_stars)
    y = rng.uniform(margin, size - 1 - margin, n_stars)
    flux = np.exp(rng.uniform(np.log(flux_range[0]), np.log(flux_range[1]), n_stars))
    expected = np.full((size, size), background, float)
    half = int(np.ceil(6 * sigma))
    for xk, yk, fk in zip(x, y, flux):
        i0, j0 = int(round(xk)) - half, int(round(yk)) - half
        stamp = pixel_integrated_gaussian((2 * half + 1, 2 * half + 1), xk, yk, sigma, fk, i0, j0)
        ys, xs = slice(max(j0, 0), min(j0 + 2 * half + 1, size)), slice(max(i0, 0), min(i0 + 2 * half + 1, size))
        expected[ys, xs] += stamp[ys.start - j0:ys.stop - j0, xs.start - i0:xs.stop - i0]
    nh = int(rng.integers(n_hot[0], n_hot[1] + 1))
    hx, hy = rng.integers(0, size, nh), rng.integers(0, size, nh)
    expected[hy, hx] += rng.uniform(500, 5000, nh)
    ns = int(rng.integers(n_streaks[0], n_streaks[1] + 1))
    streaks = []
    YY, XX = np.mgrid[0:size, 0:size]
    for _ in range(ns):
        cx, cy = rng.uniform(20, size - 20, 2)
        L, th, F = rng.uniform(6, 20), rng.uniform(0, np.pi), rng.uniform(500, 5000)
        j0, j1, i0, i1 = int(cy - 15), int(cy + 16), int(cx - 15), int(cx + 16)
        dx, dy = XX[j0:j1, i0:i1] - cx, YY[j0:j1, i0:i1] - cy
        along = dx * np.cos(th) + dy * np.sin(th)
        across = -dx * np.sin(th) + dy * np.cos(th)
        prof = np.exp(-0.5 * (across / 0.6) ** 2) * (np.abs(along) <= L / 2)
        if prof.sum() > 0:
            expected[j0:j1, i0:i1] += F * prof / prof.sum()
        t = np.linspace(-L / 2, L / 2, 15)
        streaks.append(np.column_stack([cx + t * np.cos(th), cy + t * np.sin(th)]))
    image = (rng.poisson(expected) + rng.normal(0, read_noise, (size, size))).astype(np.float32)
    snr = flux / np.sqrt(flux + 4 * np.pi * sigma**2 * (background + read_noise**2))
    truth = {"x": x, "y": y, "flux": flux, "snr": snr}
    art = {"hot": np.column_stack([hx, hy]).astype(float),
           "streak_core": np.vstack(streaks) if streaks else np.zeros((0, 2))}
    return image, truth, art, {"fwhm": fwhm, "sigma": sigma, "background": background, "read_noise": read_noise}


def training_condition(rng):
    return dict(background=rng.uniform(80, 320), read_noise=rng.uniform(4, 10),
                flux_range=(rng.uniform(100, 300), rng.uniform(3000, 30000)), fwhm=rng.uniform(2.7, 3.3))


def extract(image, xy, bkg, rms, size=PATCH + 2 * PAD):
    """Patches centred on the candidate pixel, background-subtracted and divided by the image's measured
    background RMS (so pixel values are in units of the detector's own noise - no learned parameters;
    relative brightness / SNR information is preserved). Out-of-image pixels are filled with 0 (= bkg)."""
    h = size // 2
    P = np.zeros((len(xy), size, size), np.float32)
    ny, nx = image.shape
    for n, (x, y) in enumerate(xy):
        i, j = int(round(x)), int(round(y))
        y0, y1, x0, x1 = j - h, j - h + size, i - h, i - h + size
        sy0, sy1, sx0, sx1 = max(y0, 0), min(y1, ny), max(x0, 0), min(x1, nx)
        P[n, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = (image[sy0:sy1, sx0:sx1] - bkg) / rms
    return P


def label_candidates(det, truth, art):
    tr = np.column_stack([truth["x"], truth["y"]])
    pairs, _ = match(det, tr, 2.0)
    y = np.zeros(len(det), np.int64)
    sub = np.array(["noise / other"] * len(det), dtype=object)
    snr = np.full(len(det), np.nan)
    true_idx = np.full(len(det), -1)
    for di, ti in pairs:
        y[di] = 1
        sub[di] = "bright star" if truth["snr"][ti] >= 20 else "faint star"
        snr[di] = truth["snr"][ti]
        true_idx[di] = ti
    for di in np.nonzero(y == 0)[0]:
        if len(art["hot"]) and np.min(np.hypot(*(art["hot"] - det[di]).T)) <= 1.5:
            sub[di] = "hot pixel"
        elif len(art["streak_core"]) and np.min(np.hypot(*(art["streak_core"] - det[di]).T)) <= 2.0:
            sub[di] = "elongated"
    return y, sub, snr, true_idx


def build(seeds, condition=None, with_images=False):
    """Detector-proposed patches for a list of image seeds. condition=None -> training mix."""
    Ps, ys, subs, snrs, grp, info = [], [], [], [], [], []
    for s in seeds:
        rng = np.random.default_rng([s, 12])
        cond = training_condition(rng) if condition is None else CONDITIONS[condition]
        image, truth, art, meta = simulate(seed=s, **cond)
        det, diag = detect(image, fwhm_assumed=3.0, return_diagnostics=True)
        y, sub, snr, _ = label_candidates(det, truth, art)
        Ps.append(extract(image, det, diag["background"], diag["rms"]))
        ys.append(y)
        subs.append(sub)
        snrs.append(snr)
        grp.append(np.full(len(det), s))
        info.append((len(det), len(truth["x"])))
    return {"X": np.concatenate(Ps), "y": np.concatenate(ys), "subtype": np.concatenate(subs),
            "snr": np.concatenate(snrs), "group": np.concatenate(grp), "per_image": info}


SPLITS = {"train": range(10000, 10100), "val": range(20000, 20030), "test": range(30000, 30040)}
