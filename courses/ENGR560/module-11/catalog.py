"""
Integrated source catalog (Module 11 final output).

detect_sources(image, uncertainty_scale) -> dict of NumPy arrays:
    x, y                  Gaussian-PSF-fit centroid (pixels; x = column)
    flux                  fitted total flux (electrons)
    snr                   flux / sqrt(flux + 4 pi sigma_fit^2 * rms^2)    (PSF-photometry noise model)
    centroid_uncertainty  radial 1-sigma uncertainty sqrt(sx^2 + sy^2) from the fit covariance,
                          multiplied by an EMPIRICAL scale factor calibrated on independent images
                          (calibrate_uncertainty); the per-axis model value is also returned
    quality_flags         bit mask:
        1  SHARP     fitted sigma < 0.6 px (narrower than any optical PSF: hot pixel / cosmic ray)
        2  EDGE      within 6 px of the image border (cutout truncated)
        4  CROWDED   another detection within 2 x FWHM (centroids / fluxes may be biased by a neighbour)
        8  LOW_SNR   snr < 10
        16 BAD_FIT   least-squares failed, or fitted sigma > 1.6 x the nominal PSF sigma (blend,
                     extended source or strong defocus)
"""

import numpy as np

from starfield import FWHM_TO_SIGMA, background_stats, detect, gaussian_fit

FLAG_NAMES = {1: "SHARP", 2: "EDGE", 4: "CROWDED", 8: "LOW_SNR", 16: "BAD_FIT"}


def detect_sources(image, fwhm=3.0, k=5.0, read_noise=5.0, uncertainty_scale=1.0):
    det, diag = detect(image, fwhm_assumed=fwhm, k=k, return_diagnostics=True)
    bkg, rms = diag["background"], diag["rms"]
    s0 = fwhm * FWHM_TO_SIGMA
    n = len(det)
    out = {k_: np.zeros(n) for k_ in ("x", "y", "flux", "snr", "sigma", "sx", "sy", "centroid_uncertainty")}
    flags = np.zeros(n, int)
    ny, nx = image.shape
    for i, (x, y) in enumerate(det):
        f = gaussian_fit(image, bkg, x, y, s0, read_noise, half=5)
        out["x"][i], out["y"][i], out["flux"][i], out["sigma"][i] = f["x"], f["y"], f["flux"], f["sigma"]
        out["sx"][i], out["sy"][i] = f["sx"], f["sy"]
        sig = f["sigma"] if np.isfinite(f["sigma"]) else s0
        F = max(f["flux"], 0) if np.isfinite(f["flux"]) else 0.0
        out["snr"][i] = F / np.sqrt(F + 4 * np.pi * sig**2 * rms**2) if F > 0 else 0.0
        if not f["ok"] or not np.isfinite(f["sigma"]) or f["sigma"] > 1.6 * s0:
            flags[i] |= 16
        if np.isfinite(f["sigma"]) and f["sigma"] < 0.6:
            flags[i] |= 1
        if x < 6 or y < 6 or x > nx - 7 or y > ny - 7:
            flags[i] |= 2
    if n > 1:
        D = np.hypot(det[:, None, 0] - det[None, :, 0], det[:, None, 1] - det[None, :, 1])
        np.fill_diagonal(D, np.inf)
        flags[D.min(axis=1) < 2 * fwhm] |= 4
    flags[out["snr"] < 10] |= 8
    out["centroid_uncertainty"] = uncertainty_scale * np.hypot(out["sx"], out["sy"])
    out["quality_flags"] = flags
    out["background"], out["background_rms"] = bkg, rms
    return out


def flag_string(f):
    return "|".join(name for bit, name in FLAG_NAMES.items() if f & bit) or "OK"
