"""
Synthetic star / artifact candidate generator (Module 4, Experiment A).

Simplified image-feature model
------------------------------
Each detected source has latent physical parameters
    A      peak signal above background [counts]
    w      Gaussian width (geometric mean of the two axes) [pixels]
    rho    roundness = minor/major axis ratio, in (0, 1]
    B      local sky background [counts / pixel]
and the per-pixel noise is background dominated:
    sigma_pix = noise_scale * sqrt(B + READ^2)
For a Gaussian profile measured with a matched filter the detection signal-to-noise is
    SNR = flux / (sigma_pix * sqrt(4 pi) w),   flux = 2 pi w^2 A   =>   SNR = sqrt(pi) w A / sigma_pix
so peak intensity, width, background and SNR are physically linked, not independent.

Measured features (what a source extractor would report) add measurement noise that grows as
the source gets fainter:
    background   B_hat   = B + N(0, (sigma_pix / 8)^2)                 (sky annulus estimate)
    peak         P_hat   = B + A + N(0, sigma_pix^2)                   (brightest pixel)
    width        w_hat   = w * exp(N(0, (0.06 + 0.9 / SNR)^2))         (moments are noisy when faint)
    roundness    rho_hat = clip(rho + N(0, (0.03 + 0.6 / sqrt(SNR))^2), 0.05, 1)
    snr          SNR_hat = sqrt(pi) w_hat (P_hat - B_hat) / sigma_hat,  sigma_hat from B_hat
Only candidates with SNR_hat >= DETECT_SNR are kept (a detection threshold), so every population
is truncated at the same place - this is what makes faint stars and noise peaks overlap.

Populations (subtypes)
  stars (y = 1)      PSF width ~ N(1.25, 0.12) px x defocus, roundness ~ 1 - |N(0, 0.06)|,
                     flux from a power law N(>F) ~ F^-0.8 (many faint, a tail of bright)
                     -> subtype "bright star" if true SNR >= 20 else "faint star"
  artifacts (y = 0)
    hot pixel        w ~ N(0.60, 0.10): narrower than any PSF; round; amplitude log-uniform
    diffuse          w ~ logN(ln 3.2, 0.35): scattered light / extended glow; any roundness
    noise peak       random fluctuation just above threshold: w ~ logN(ln 1.15, 0.35),
                     roundness ~ U(0.25, 1), amplitude ~ (2.5 - 6) sigma_pix
    compact irregular cosmic-ray hits / trail fragments: w ~ logN(ln 0.95, 0.25),
                     elongated roundness ~ U(0.10, 0.60), wide amplitude range
    ghost (UNSEEN)   internal reflection: slightly broad (w ~ N(1.9, 0.25)), round, faint.
                     Never used in training; only in the distribution-shift experiment.
"""

import numpy as np

FEATURE_NAMES = ("peak_intensity", "width", "roundness", "background", "snr")
STAR_TYPES = ("bright star", "faint star")
ARTIFACT_TYPES = ("hot pixel", "diffuse", "noise peak", "compact irregular", "ghost")
READ = 5.0
DETECT_SNR = 3.0

# Default (training) conditions
DEFAULT = {
    "star_fraction": 0.4,
    "noise_scale": 1.0,       # multiplies sigma_pix (sensor noise)
    "defocus": 1.0,           # multiplies stellar PSF width (and ghosts)
    "background_scale": 1.0,  # multiplies the sky level
    "artifact_mix": {"hot pixel": 0.20, "diffuse": 0.20, "noise peak": 0.40,
                     "compact irregular": 0.20, "ghost": 0.0},
}


def conditions(**changes):
    c = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT.items()}
    for k, v in changes.items():
        if k == "artifact_mix":
            c[k] = {**{t: 0.0 for t in ARTIFACT_TYPES}, **v}
        else:
            c[k] = v
    s = sum(c["artifact_mix"].values())
    c["artifact_mix"] = {k: v / s for k, v in c["artifact_mix"].items()}
    return c


def _latent(kind, m, rng, cond):
    """Draw m latent sources (A, w, rho, B) of one subtype family."""
    B = 100.0 * cond["background_scale"] * rng.lognormal(0.0, 0.35, m)
    sig = cond["noise_scale"] * np.sqrt(B + READ**2)
    if kind == "star":
        w = rng.normal(1.25, 0.12, m) * cond["defocus"]
        rho = 1.0 - np.abs(rng.normal(0.0, 0.06, m))
        # power-law flux: N(>F) ~ F^-0.8 above F_min (many faint stars, a tail of bright ones)
        flux = 60.0 * (1.0 - rng.random(m)) ** (-1.0 / 0.8)
        A = flux / (2 * np.pi * w**2)
    elif kind == "hot pixel":
        w = rng.normal(0.60, 0.10, m)
        rho = 1.0 - np.abs(rng.normal(0.0, 0.10, m))
        A = sig * 10 ** rng.uniform(0.6, 2.3, m)
    elif kind == "diffuse":
        w = rng.lognormal(np.log(3.2), 0.35, m)
        rho = rng.uniform(0.35, 1.0, m)
        A = sig * 10 ** rng.uniform(-0.3, 0.6, m)
    elif kind == "noise peak":
        w = rng.lognormal(np.log(1.15), 0.35, m)
        rho = rng.uniform(0.25, 1.0, m)
        A = sig * rng.uniform(2.5, 6.0, m) / (np.sqrt(np.pi) * w)
        A *= rng.uniform(0.8, 1.6, m)
    elif kind == "compact irregular":
        w = rng.lognormal(np.log(0.95), 0.25, m)
        rho = rng.uniform(0.10, 0.60, m)
        A = sig * 10 ** rng.uniform(0.3, 1.8, m)
    elif kind == "ghost":
        w = rng.normal(1.9, 0.25, m) * cond["defocus"]
        rho = 1.0 - np.abs(rng.normal(0.0, 0.08, m))
        A = sig * 10 ** rng.uniform(-0.1, 0.5, m)
    else:
        raise ValueError(kind)
    w = np.clip(w, 0.3, None)
    return A, w, rho, B, sig


def _measure(A, w, rho, B, sig, rng, cond):
    snr_true = np.sqrt(np.pi) * w * A / sig
    m = A.size
    B_hat = B + rng.normal(0, sig / 8.0, m)
    P_hat = B + A + rng.normal(0, sig, m)
    w_hat = w * np.exp(rng.normal(0, 0.06 + 0.9 / np.maximum(snr_true, 1.0), m))
    rho_hat = np.clip(rho + rng.normal(0, 0.03 + 0.6 / np.sqrt(np.maximum(snr_true, 1.0)), m),
                      0.05, 1.0)
    sig_hat = cond["noise_scale"] * np.sqrt(np.maximum(B_hat, 1.0) + READ**2)
    snr_hat = np.sqrt(np.pi) * w_hat * (P_hat - B_hat) / sig_hat
    X = np.column_stack([P_hat, w_hat, rho_hat, B_hat, snr_hat])
    return X, snr_true


def _draw(kind, m, rng, cond):
    """m DETECTED sources of one family (rejection sampling at the detection threshold)."""
    feats, snrs = [], []
    have = 0
    while have < m:
        need = max(2 * (m - have), 16)
        A, w, rho, B, sig = _latent(kind, need, rng, cond)
        X, snr_true = _measure(A, w, rho, B, sig, rng, cond)
        keep = X[:, 4] >= DETECT_SNR
        feats.append(X[keep])
        snrs.append(snr_true[keep])
        have += keep.sum()
    return np.vstack(feats)[:m], np.concatenate(snrs)[:m]


def generate_candidates(n, rng, star_fraction=None, cond=None, return_subtypes=False):
    """Synthetic labelled candidates.

    Returns X (n, 5) in FEATURE_NAMES order, y (1 = star, 0 = artifact) and, optionally, the
    subtype string of every source.
    """
    cond = DEFAULT if cond is None else cond
    frac = cond["star_fraction"] if star_fraction is None else star_fraction
    y = (rng.random(n) < frac).astype(int)
    X = np.zeros((n, len(FEATURE_NAMES)))
    sub = np.empty(n, dtype=object)

    idx = np.nonzero(y == 1)[0]
    if idx.size:
        Xs, snr = _draw("star", idx.size, rng, cond)
        X[idx] = Xs
        sub[idx] = np.where(snr >= 20, "bright star", "faint star")

    idx = np.nonzero(y == 0)[0]
    if idx.size:
        types = list(cond["artifact_mix"])
        probs = np.array([cond["artifact_mix"][t] for t in types])
        kinds = rng.choice(len(types), size=idx.size, p=probs)
        for k, t in enumerate(types):
            sel = idx[kinds == k]
            if sel.size:
                X[sel], _ = _draw(t, sel.size, rng, cond)
                sub[sel] = t
    if return_subtypes:
        return X, y, sub
    return X, y


def engineered(X):
    """Classifier inputs: logs for the positive, scale-free quantities.

    [log peak, log width, roundness, log background, log SNR]
    """
    return np.column_stack([np.log(np.maximum(X[:, 0], 1.0)), np.log(X[:, 1]), X[:, 2],
                            np.log(np.maximum(X[:, 3], 1.0)), np.log(np.maximum(X[:, 4], 1e-3))])


ENG_NAMES = ("log peak", "log width", "roundness", "log background", "log SNR")

# Fixed, reproducible splits used by every experiment
SPLIT_SEEDS = {"train": 401, "val": 402, "test": 403}
SPLIT_SIZES = {"train": 4000, "val": 2000, "test": 4000}


def standard_splits(cond=None, subtypes=True):
    out = {}
    for name in ("train", "val", "test"):
        out[name] = generate_candidates(SPLIT_SIZES[name], np.random.default_rng(SPLIT_SEEDS[name]),
                                        cond=cond, return_subtypes=subtypes)
    return out
