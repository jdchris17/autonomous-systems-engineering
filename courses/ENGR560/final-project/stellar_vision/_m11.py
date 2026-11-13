"""
VERBATIM copy of stable Module 11 routines (source: module-11/starfield.py, sha256 5e09ca3bfedb39cb...).
Copied: FWHM_TO_SIGMA, pixel_integrated_gaussian, background_stats, detect, gaussian_fit, match.
Do not edit: tests/test_regression.py checks these against the original module.
"""

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.special import ndtr

FWHM_TO_SIGMA = 1 / (2 * np.sqrt(2 * np.log(2)))


def pixel_integrated_gaussian(shape, x0, y0, sigma, flux=1.0, x_offset=0, y_offset=0):
    ny, nx = shape
    xi = np.arange(nx) + x_offset
    yj = np.arange(ny) + y_offset
    px = ndtr((xi + 0.5 - x0) / sigma) - ndtr((xi - 0.5 - x0) / sigma)
    py = ndtr((yj + 0.5 - y0) / sigma) - ndtr((yj - 0.5 - y0) / sigma)
    return flux * np.outer(py, px)


def background_stats(image, nsigma=3.0, iters=5):
    """Sigma-clipped median and MAD-based RMS (robust to stars and hot pixels)."""
    v = image.ravel()
    for _ in range(iters):
        med = np.median(v)
        rms = 1.4826 * np.median(np.abs(v - med))
        v = v[np.abs(v - med) < nsigma * rms]
    return float(np.median(v)), float(1.4826 * np.median(np.abs(v - np.median(v))))


def detect(image, fwhm_assumed=3.0, k=5.0, min_sep=4.0, window=5, return_diagnostics=False):
    bkg, rms = background_stats(image)                                    # 1
    sub = image - bkg                                                     # 2
    filt = ndimage.gaussian_filter(sub, fwhm_assumed * FWHM_TO_SIGMA)     # 3
    _, frms = background_stats(filt)                                      # noise of the FILTERED image
    peaks = (filt == ndimage.maximum_filter(filt, size=window))           # 4
    peaks &= filt > k * frms                                              # 5
    yy, xx = np.nonzero(peaks)
    order = np.argsort(-filt[yy, xx])
    kept = []
    for idx in order:                                                     # 6 (brightest first)
        if all((xx[idx] - xx[j]) ** 2 + (yy[idx] - yy[j]) ** 2 >= min_sep**2 for j in kept):
            kept.append(idx)
    coords = np.column_stack([xx[kept], yy[kept]]).astype(float) if kept else np.zeros((0, 2))
    if not return_diagnostics:
        return coords
    return coords, {"background": bkg, "rms": rms, "filtered_rms": frms, "subtracted": sub, "filtered": filt,
                    "n_peaks_before_threshold": int((filt == ndimage.maximum_filter(filt, size=window)).sum()),
                    "n_after_threshold": int(len(xx))}


def gaussian_fit(image, bkg, x, y, sigma_guess, read_noise, half=5):
    """Weighted least-squares fit of a pixel-integrated circular Gaussian + constant to a cutout.

    Returns dict(x, y, flux, sigma, local_bkg, sx, sy, ok). sx, sy are standard errors from the
    covariance (J^T W J)^-1 of the fit, scaled by the reduced chi-square."""
    ny, nx = image.shape
    i0, j0 = int(round(x)) - half, int(round(y)) - half
    i0, j0 = min(max(i0, 0), nx - 2 * half - 1), min(max(j0, 0), ny - 2 * half - 1)
    cut = image[j0:j0 + 2 * half + 1, i0:i0 + 2 * half + 1]
    var = np.maximum(cut, 0) + read_noise**2                       # Poisson + read noise (electrons)
    w = 1.0 / np.sqrt(var)
    f0 = max(np.sum(cut - bkg), 10.0)

    def resid(p):
        F, xc, yc, s, b = p
        model = b + pixel_integrated_gaussian(cut.shape, xc, yc, s, F, i0, j0)
        return ((cut - model) * w).ravel()
    p0 = [f0, x, y, sigma_guess, bkg]
    lo = [0, x - 3, y - 3, 0.3, bkg - 50 * read_noise]
    hi = [np.inf, x + 3, y + 3, 4 * sigma_guess + 3, bkg + 50 * read_noise]
    try:
        r = least_squares(resid, p0, bounds=(lo, hi), x_scale="jac")
        Jm = r.jac
        dof = max(r.fun.size - 5, 1)
        cov = np.linalg.pinv(Jm.T @ Jm) * max(np.sum(r.fun**2) / dof, 1e-12)
        F, xc, yc, s, b = r.x
        return {"x": xc, "y": yc, "flux": F, "sigma": s, "local_bkg": b, "sx": float(np.sqrt(cov[1, 1])),
                "sy": float(np.sqrt(cov[2, 2])), "ok": bool(r.success), "chi2_red": float(np.sum(r.fun**2) / dof)}
    except (ValueError, np.linalg.LinAlgError):
        return {"x": x, "y": y, "flux": np.nan, "sigma": np.nan, "local_bkg": bkg, "sx": np.nan, "sy": np.nan,
                "ok": False, "chi2_red": np.nan}


def match(det_xy, true_xy, max_dist=2.0):
    """One-to-one matching (Hungarian algorithm) with a maximum allowed distance.

    Returns (pairs as (det_index, true_index) array, distances)."""
    if len(det_xy) == 0 or len(true_xy) == 0:
        return np.zeros((0, 2), int), np.zeros(0)
    D = np.hypot(det_xy[:, None, 0] - true_xy[None, :, 0], det_xy[:, None, 1] - true_xy[None, :, 1])
    cost = np.where(D <= max_dist, D, 1e6)
    r, c = linear_sum_assignment(cost)
    ok = D[r, c] <= max_dist
    return np.column_stack([r[ok], c[ok]]), D[r[ok], c[ok]]
