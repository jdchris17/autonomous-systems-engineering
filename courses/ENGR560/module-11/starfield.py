"""
Module 11 core: star-field simulation, a classical detector, two centroid estimators, and matching.

Coordinates: x = column, y = row; the centre of pixel (row j, column i) is at (x, y) = (i, j).

SIMULATION (electron-count model)
  expected(i, j) = B + sum_k F_k * P_k(i, j) + hot pixels
  P_k(i, j) is the circular Gaussian PSF INTEGRATED over the pixel area. A Gaussian is separable, so
  the integral is exact with error functions:
      P(i, j) = [Phi((i + 0.5 - x0)/s) - Phi((i - 0.5 - x0)/s)] * [Phi((j + 0.5 - y0)/s) - Phi((j - 0.5 - y0)/s)]
  with s = FWHM / 2.3548 and Phi the standard normal CDF (no pixel-centre approximation needed).
  measured = Poisson(expected) + N(0, RN^2)       (photon counting + read noise, all in electrons)
  Stars: positions uniform with subpixel coordinates (10 px from the edges); flux log-uniform.
  Hot pixels: single pixels with 500-5000 extra electrons.
  True SNR of a star (optimal PSF-weighted photometry, background + read + source noise):
      SNR = F / sqrt(F + 4 pi s^2 (B + RN^2))

DETECTOR (detect)
  1 background: sigma-clipped median and MAD-based RMS of the whole image (uniform background)
  2 subtraction
  3 filtering with a Gaussian of the ASSUMED PSF width (matched filter for a point source)
  4 local maxima (maximum filter, 5 x 5 window)
  5 threshold: filtered peak > k * RMS of the filtered background (measured, not assumed)
  6 minimum separation: peaks closer than d_min to a brighter accepted peak are dropped
  returns an N x 2 array of (x, y) peak pixel coordinates (plus diagnostics)

CENTROIDS
  weighted_centroid  intensity-weighted mean position in a circular aperture of background-subtracted
                     pixels, iterated (re-centred) 3 times
  gaussian_fit       scipy.optimize.least_squares fit of amplitude, x0, y0, sigma and a local
                     background to a (2r+1)^2 cutout, using the pixel-integrated Gaussian, residuals
                     weighted by the Poisson + read-noise variance; returns the covariance-based
                     centroid uncertainty as well
"""

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.special import ndtr

FWHM_TO_SIGMA = 1 / (2 * np.sqrt(2 * np.log(2)))


# --------------------------------------------------------------------------- simulation
def pixel_integrated_gaussian(shape, x0, y0, sigma, flux=1.0, x_offset=0, y_offset=0):
    ny, nx = shape
    xi = np.arange(nx) + x_offset
    yj = np.arange(ny) + y_offset
    px = ndtr((xi + 0.5 - x0) / sigma) - ndtr((xi - 0.5 - x0) / sigma)
    py = ndtr((yj + 0.5 - y0) / sigma) - ndtr((yj - 0.5 - y0) / sigma)
    return flux * np.outer(py, px)


def simulate(seed=0, size=512, n_stars=50, fwhm=3.0, background=100.0, read_noise=5.0, n_hot=(10, 20),
             flux_range=(300.0, 30000.0), margin=10):
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
    hot_val = rng.uniform(500, 5000, nh)
    expected[hy, hx] += hot_val
    image = rng.poisson(expected).astype(float) + rng.normal(0, read_noise, (size, size))
    snr = flux / np.sqrt(flux + 4 * np.pi * sigma**2 * (background + read_noise**2))
    truth = {"x": x, "y": y, "flux": flux, "snr": snr}
    hot = {"x": hx, "y": hy, "electrons": hot_val}
    meta = {"fwhm": fwhm, "sigma": sigma, "background": background, "read_noise": read_noise, "size": size}
    return image, truth, hot, meta


# --------------------------------------------------------------------------- detection
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


# --------------------------------------------------------------------------- centroids
def weighted_centroid(sub, x, y, radius=4.0, iters=3):
    ny, nx = sub.shape
    cx, cy = float(x), float(y)
    for _ in range(iters):
        i0, i1 = max(int(cx - radius - 1), 0), min(int(cx + radius + 2), nx)
        j0, j1 = max(int(cy - radius - 1), 0), min(int(cy + radius + 2), ny)
        J, I = np.mgrid[j0:j1, i0:i1]
        m = (I - cx) ** 2 + (J - cy) ** 2 <= radius**2
        w = sub[j0:j1, i0:i1][m]
        if w.sum() <= 0:
            break
        cx, cy = float(np.sum(w * I[m]) / w.sum()), float(np.sum(w * J[m]) / w.sum())
    return cx, cy


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


# --------------------------------------------------------------------------- matching
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


def summarize(errors):
    e = np.asarray(errors)
    if e.size == 0:
        return {"median": np.nan, "p90": np.nan, "rmse": np.nan}
    return {"median": float(np.median(e)), "p90": float(np.percentile(e, 90)), "rmse": float(np.sqrt(np.mean(e**2)))}
