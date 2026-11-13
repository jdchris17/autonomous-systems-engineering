"""
Final deliverable: hybrid detector = Module 11 catalogue + CNN score, with classification BEFORE fitting.

    detect (matched filter, 5 sigma)  ->  extract 32x32 patches  ->  CNN on the whole batch
      ->  reject candidates with P(star) < threshold  ->  Gaussian PSF fit ONLY for the retained ones
      ->  Module 11 quality flags (kept as information; optional rule filtering via `reject_flags`)

hybrid_detect_sources(image, ...) returns the same dict of arrays as Module 11's detect_sources()
(x, y, flux, snr, sigma, sx, sy, centroid_uncertainty, quality_flags, background, background_rms) plus
    cnn_score  P(star) from the CNN
    timing     seconds spent in detect / patches / cnn / fit / total
The threshold defaults to the one frozen on the validation images (stored in the checkpoint).
"""

import time

import numpy as np
import torch

from cnn import center_crop, load, transform
from data12 import FWHM_TO_SIGMA, OUT, detect, extract, gaussian_fit

FLAG_SHARP, FLAG_EDGE, FLAG_CROWDED, FLAG_LOW_SNR, FLAG_BAD_FIT = 1, 2, 4, 8, 16


class HybridDetector:
    def __init__(self, checkpoint=OUT / "cnn_best.pt", threshold=None):
        self.model, self.pre, ck = load(checkpoint)
        self.threshold_logit = ck["threshold_logit"] if threshold is None else float(np.log(threshold / (1 - threshold)))
        self.threshold = 1 / (1 + np.exp(-self.threshold_logit))

    def score(self, image, det, bkg, rms):
        P = center_crop(extract(image, det, bkg, rms))
        with torch.inference_mode():
            logit = self.model(torch.from_numpy(transform(P, self.pre["mean"], self.pre["std"])[:, None])).numpy()
        return logit

    def __call__(self, image, fwhm=3.0, k=5.0, read_noise=5.0, uncertainty_scale=1.118, reject_flags=0):
        t0 = time.perf_counter()
        det, diag = detect(image, fwhm_assumed=fwhm, k=k, return_diagnostics=True)
        bkg, rms = diag["background"], diag["rms"]
        t1 = time.perf_counter()
        patches = center_crop(extract(image, det, bkg, rms))
        x = torch.from_numpy(transform(patches, self.pre["mean"], self.pre["std"])[:, None])
        t2 = time.perf_counter()
        with torch.inference_mode():
            logit = self.model(x).numpy() if len(det) else np.zeros(0, np.float32)
        t3 = time.perf_counter()
        keep = logit >= self.threshold_logit
        cand, score = det[keep], 1 / (1 + np.exp(-logit[keep].astype(float)))
        s0 = fwhm * FWHM_TO_SIGMA
        n = len(cand)
        out = {k_: np.zeros(n) for k_ in ("x", "y", "flux", "snr", "sigma", "sx", "sy")}
        flags = np.zeros(n, int)
        ny, nx = image.shape
        for i, (xc, yc) in enumerate(cand):
            f = gaussian_fit(image, bkg, xc, yc, s0, read_noise, half=5)
            ok = f["ok"] and np.isfinite(f["x"]) and np.isfinite(f["sigma"])
            out["x"][i], out["y"][i] = (f["x"], f["y"]) if ok else (xc, yc)      # fall back to the peak pixel
            out["flux"][i], out["sigma"][i], out["sx"][i], out["sy"][i] = f["flux"], f["sigma"], f["sx"], f["sy"]
            sig = f["sigma"] if np.isfinite(f["sigma"]) else s0
            F = max(f["flux"], 0) if np.isfinite(f["flux"]) else 0.0
            out["snr"][i] = F / np.sqrt(F + 4 * np.pi * sig**2 * rms**2) if F > 0 else 0.0
            if not ok or f["sigma"] > 1.6 * s0:
                flags[i] |= FLAG_BAD_FIT
            if np.isfinite(f["sigma"]) and f["sigma"] < 0.6:
                flags[i] |= FLAG_SHARP
            if xc < 6 or yc < 6 or xc > nx - 7 or yc > ny - 7:
                flags[i] |= FLAG_EDGE
        t4 = time.perf_counter()
        if n > 1:
            D = np.hypot(cand[:, None, 0] - cand[None, :, 0], cand[:, None, 1] - cand[None, :, 1])
            np.fill_diagonal(D, np.inf)
            flags[D.min(axis=1) < 2 * fwhm] |= FLAG_CROWDED
        flags[out["snr"] < 10] |= FLAG_LOW_SNR
        out["centroid_uncertainty"] = uncertainty_scale * np.hypot(out["sx"], out["sy"])
        out["quality_flags"], out["cnn_score"] = flags, score
        if reject_flags:
            good = (flags & reject_flags) == 0
            out = {k_: v[good] for k_, v in out.items()}
        out["background"], out["background_rms"] = bkg, rms
        out["n_candidates"], out["n_fitted"] = len(det), n
        out["timing"] = {"detect": t1 - t0, "patches": t2 - t1, "cnn": t3 - t2, "fit": t4 - t3,
                         "total": time.perf_counter() - t0}
        return out


def hybrid_detect_sources(image, detector=None, **kw):
    return (detector or HybridDetector())(image, **kw)
