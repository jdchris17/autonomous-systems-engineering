"""
Synthetic 32 x 32 astronomical image patches (Module 6, Experiment A).

Image formation model (expected counts in photo-electrons per pixel, evaluated at pixel centres)
    I(u, v) = B + A exp(-q(u - u0, v - v0) / 2) + eps(u, v)
    q = rotated elliptical quadratic form with axes sigma_major, sigma_minor and position angle
    theta (a circular PSF is sigma_major = sigma_minor). Noise: measured = Poisson(expected) +
    N(0, read_noise^2), the course's starter model. No saturation, flat field, PSF undersampling
    integral, cosmic-ray morphology or crosstalk - those are refined in later modules.

Image (capture-session) level conditions, shared by every patch of image j  [Module 5 discipline]
    background   B_j     = 50 exp(N(0, 0.5))  e-/pixel
    read noise   RN_j    ~ U(3, 8) e-
    focus        sigma_j = 1.1 exp(N(0, 0.15)) px (stellar PSF width)
    PSF shape    ellipticity e_j ~ U(0, 0.25), position angle ~ U(0, pi)   (optical asymmetry)
    star frac.   p_j     ~ U(0.35, 0.65)   (balanced on average; `star_fraction` overrides)

Sources (centroid offset from the patch centre ~ U(-1.5, 1.5) px in each axis)
  star (y=1)          image PSF (x small per-star jitter); flux from N(>F) ~ F^-0.8 above 120 e-;
                      only sources with true matched-filter SNR >= 4 are kept (a detection cut)
                      subtype "bright star" if true SNR >= 25 else "faint star"
  hot pixel           a single pixel (5% bleed into the 4 neighbours), amplitude log-U(4, 60) x noise
  diffuse             sigma ~ logN(ln 3.5, 0.3), ellipticity U(0, 0.5), SNR U(4, 20)
  elongated           cosmic-ray / trail fragment: sigma_minor U(0.4, 0.8), sigma_major U(2, 5),
                      any angle, SNR U(5, 60)
  noise peak          weak compact bump standing in for a noise-triggered detection:
                      sigma U(0.7, 1.6), ellipticity U(0, 0.4), SNR U(4, 7)
True matched-filter SNR of a Gaussian of flux F: F / (sqrt(B + RN^2) * sqrt(4 pi sigma_x sigma_y)).
Overlap is deliberate: faint stars vs noise peaks (same SNR, similar width), undersampled stars vs
hot pixels, defocused stars vs diffuse sources.
"""

import numpy as np

SIZE = 32
CENTRE = (SIZE - 1) / 2.0
SUBTYPES = ("bright star", "faint star", "hot pixel", "diffuse", "elongated", "noise peak")
ARTIFACT_MIX = {"hot pixel": 0.25, "diffuse": 0.25, "elongated": 0.25, "noise peak": 0.25}
YY, XX = np.mgrid[0:SIZE, 0:SIZE].astype(float)


def gaussian_source(size, amplitude, sigma, x0, y0, background=0.0):
    """Course starter: circular Gaussian evaluated at pixel centres."""
    yy, xx = np.mgrid[0:size, 0:size]
    radius2 = (xx - x0) ** 2 + (yy - y0) ** 2
    return background + amplitude * np.exp(-radius2 / (2.0 * sigma**2))


def elliptical_source(amplitude, s_major, s_minor, theta, x0, y0, background=0.0):
    """Elliptical Gaussian (reduces to gaussian_source when s_major == s_minor)."""
    dx, dy = XX - x0, YY - y0
    c, s = np.cos(theta), np.sin(theta)
    a = (c * dx + s * dy) / s_major
    b = (-s * dx + c * dy) / s_minor
    return background + amplitude * np.exp(-0.5 * (a**2 + b**2))


def noisy_patch(expected, read_noise, rng):
    """Course starter noise model: Poisson photon counts + Gaussian read noise."""
    photons = rng.poisson(np.maximum(expected, 0.0))
    return (photons + rng.normal(0.0, read_noise, expected.shape)).astype(np.float64)


def image_conditions(rng, star_fraction=None):
    return {"B": 50.0 * np.exp(rng.normal(0, 0.5)), "RN": rng.uniform(3, 8),
            "sigma": 1.1 * np.exp(rng.normal(0, 0.15)), "ell": rng.uniform(0, 0.25),
            "pa": rng.uniform(0, np.pi),
            "pstar": rng.uniform(0.35, 0.65) if star_fraction is None else star_fraction}


def _source(kind, g, rng):
    """Return (expected source image without background, parameter dict)."""
    noise = np.sqrt(g["B"] + g["RN"] ** 2)
    x0 = CENTRE + rng.uniform(-1.5, 1.5)
    y0 = CENTRE + rng.uniform(-1.5, 1.5)
    if kind == "star":
        s = g["sigma"] * np.exp(rng.normal(0, 0.03))
        smaj, smin, th = s, s * (1 - g["ell"]), g["pa"]
        while True:
            flux = 120.0 * (1 - rng.random()) ** (-1 / 0.8)
            snr = flux / (noise * np.sqrt(4 * np.pi * smaj * smin))
            if snr >= 4 and flux < 4e5:
                break
        amp = flux / (2 * np.pi * smaj * smin)
        img = elliptical_source(amp, smaj, smin, th, x0, y0)
    elif kind == "hot pixel":
        amp = noise * 10 ** rng.uniform(np.log10(4), np.log10(60))
        img = np.zeros((SIZE, SIZE))
        ix, iy = int(round(x0)), int(round(y0))
        x0, y0 = float(ix), float(iy)
        img[iy, ix] = amp
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            img[iy + dy, ix + dx] = 0.05 * amp
        smaj = smin = 0.45
        th = 0.0
        flux = img.sum()
        snr = flux / noise / np.sqrt(1.2)
    else:
        if kind == "diffuse":
            smaj = rng.lognormal(np.log(3.5), 0.3)
            smin = smaj * (1 - rng.uniform(0, 0.5))
            snr = rng.uniform(4, 20)
        elif kind == "elongated":
            smaj, smin = rng.uniform(2, 5), rng.uniform(0.4, 0.8)
            snr = rng.uniform(5, 60)
        elif kind == "noise peak":
            smaj = rng.uniform(0.7, 1.6)
            smin = smaj * (1 - rng.uniform(0, 0.4))
            snr = rng.uniform(4, 7)
        th = rng.uniform(0, np.pi)
        flux = snr * noise * np.sqrt(4 * np.pi * smaj * smin)
        amp = flux / (2 * np.pi * smaj * smin)
        img = elliptical_source(amp, smaj, smin, th, x0, y0)
    return img, {"flux": flux, "amp": amp, "s_major": smaj, "s_minor": smin, "theta": th,
                 "x0": x0, "y0": y0, "snr_true": snr}


def generate_dataset(n_images, per_image, rng, star_fraction=None, first_id=0):
    """Grouped patch dataset.

    Returns a dict with X (N, 32, 32) measured patches, expected (noise-free, incl. background),
    source (noise-free, background-free), y, subtype, group, and per-patch parameter arrays.
    """
    X, E, S, y, sub, grp = [], [], [], [], [], []
    par = {k: [] for k in ("flux", "amp", "s_major", "s_minor", "theta", "x0", "y0", "snr_true",
                           "B", "RN", "img_sigma", "img_ell")}
    kinds = list(ARTIFACT_MIX)
    probs = np.array([ARTIFACT_MIX[k] for k in kinds])
    for j in range(n_images):
        g = image_conditions(rng, star_fraction)
        n = int(rng.integers(per_image[0], per_image[1] + 1))
        for _ in range(n):
            is_star = rng.random() < g["pstar"]
            kind = "star" if is_star else kinds[rng.choice(len(kinds), p=probs)]
            src, p = _source(kind, g, rng)
            expected = g["B"] + src
            X.append(noisy_patch(expected, g["RN"], rng))
            E.append(expected)
            S.append(src)
            y.append(int(is_star))
            sub.append(("bright star" if p["snr_true"] >= 25 else "faint star") if is_star else kind)
            grp.append(first_id + j)
            for k in ("flux", "amp", "s_major", "s_minor", "theta", "x0", "y0", "snr_true"):
                par[k].append(p[k])
            par["B"].append(g["B"])
            par["RN"].append(g["RN"])
            par["img_sigma"].append(g["sigma"])
            par["img_ell"].append(g["ell"])
    out = {"X": np.array(X), "expected": np.array(E), "source": np.array(S), "y": np.array(y),
           "subtype": np.array(sub, dtype=object), "group": np.array(grp)}
    out.update({k: np.array(v) for k, v in par.items()})
    return out


# --------------------------------------------------------------------------- measurements
def border_background(P):
    """Per-patch background and noise from the 3-pixel border (no learned parameters)."""
    m = np.ones((SIZE, SIZE), bool)
    m[3:-3, 3:-3] = False
    b = P[:, m]
    med = np.median(b, axis=1)
    mad = 1.4826 * np.median(np.abs(b - med[:, None]), axis=1)
    return med, mad


def measure(P, bg=None, noise=None, s_w=2.0, iters=6):
    """Adaptive Gaussian-weighted moments on background-subtracted patches (vectorised).

    Returns dict of arrays: x, y (centroid), width (geometric-mean sigma, de-weighted), roundness
    (sqrt(lambda_min / lambda_max) of the de-weighted second-moment matrix), peak (max of the
    background-subtracted patch within 3 px of the centroid), snr (Gaussian-weighted matched-filter
    SNR), flux (sum within 4 px).
    """
    P = np.asarray(P, float)
    if bg is None or noise is None:
        b0, n0 = border_background(P)
        bg = b0 if bg is None else bg
        noise = n0 if noise is None else noise
    I = P - np.asarray(bg)[:, None, None]
    # start at the brightest pixel of a 3x3-smoothed central region
    k = np.zeros_like(I)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            k += np.roll(np.roll(I, dy, axis=1), dx, axis=2)
    c = k[:, 8:24, 8:24].reshape(len(I), -1).argmax(axis=1)
    yc, xc = 8.0 + c // 16, 8.0 + c % 16
    for _ in range(iters):
        W = np.exp(-((XX - xc[:, None, None]) ** 2 + (YY - yc[:, None, None]) ** 2) / (2 * s_w**2))
        WI = W * I
        M0 = WI.sum(axis=(1, 2))
        ok = M0 > 0
        nx = np.where(ok, (WI * XX).sum(axis=(1, 2)) / np.where(ok, M0, 1), xc)
        ny = np.where(ok, (WI * YY).sum(axis=(1, 2)) / np.where(ok, M0, 1), yc)
        xc, yc = np.clip(nx, 2, SIZE - 3), np.clip(ny, 2, SIZE - 3)
    W = np.exp(-((XX - xc[:, None, None]) ** 2 + (YY - yc[:, None, None]) ** 2) / (2 * s_w**2))
    WI = W * I
    M0 = WI.sum(axis=(1, 2))
    safe = np.where(M0 > 0, M0, 1.0)
    dx, dy = XX - xc[:, None, None], YY - yc[:, None, None]
    mxx = (WI * dx * dx).sum(axis=(1, 2)) / safe
    myy = (WI * dy * dy).sum(axis=(1, 2)) / safe
    mxy = (WI * dx * dy).sum(axis=(1, 2)) / safe
    tr, det = mxx + myy, mxx * myy - mxy**2
    disc = np.sqrt(np.maximum(tr**2 / 4 - det, 0))
    l1, l2 = tr / 2 + disc, np.maximum(tr / 2 - disc, 1e-3)
    cap = 0.95 * s_w**2
    deweight = lambda l: np.clip(l, 1e-3, cap) * s_w**2 / (s_w**2 - np.clip(l, 1e-3, cap))  # noqa: E731
    L1, L2 = deweight(l1), deweight(l2)
    width = np.where(M0 > 0, (L1 * L2) ** 0.25, np.nan)
    round_ = np.where(M0 > 0, np.sqrt(L2 / L1), np.nan)
    r2 = dx**2 + dy**2
    peak = np.where(r2 <= 9, I, -np.inf).max(axis=(1, 2))
    flux = np.where(r2 <= 16, I, 0).sum(axis=(1, 2))
    snr = M0 / (noise * np.sqrt((W**2).sum(axis=(1, 2))))
    return {"x": xc, "y": yc, "width": width, "roundness": round_, "peak": peak, "snr": snr,
            "flux": flux}


def engineered_features(P):
    """Module 4/5 invariant feature set measured from pixels: log width, roundness, log SNR."""
    m = measure(P)
    w = np.nan_to_num(m["width"], nan=2.0)
    r = np.nan_to_num(m["roundness"], nan=0.0)
    return np.column_stack([np.log(np.clip(w, 0.2, 20)), r, np.log(np.clip(m["snr"], 0.1, None))])


# Fixed, reproducible datasets
DEV_SEED, TEST_SEED = 601, 602
N_DEV_IMAGES, N_TEST_IMAGES, PER_IMAGE = 60, 30, (40, 60)


def development_set(**kw):
    return generate_dataset(N_DEV_IMAGES, PER_IMAGE, np.random.default_rng(DEV_SEED), **kw)


def untouched_test_set(**kw):
    return generate_dataset(N_TEST_IMAGES, PER_IMAGE, np.random.default_rng(TEST_SEED), first_id=1000, **kw)
