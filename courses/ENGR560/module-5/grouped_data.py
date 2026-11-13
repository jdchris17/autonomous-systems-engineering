"""
Grouped (per-image) star / artifact candidates for Module 5.

Extends the Module 4 feature model (module-4/star_data.py) so that every candidate belongs to an
image j with shared conditions g_j, and x_ij = f(source_ij, g_j, eps_ij).

Image-level variables g_j  (REALISTIC physical / instrumental assumptions)
  focus offset      defocus_j = exp(N(0, 0.10))       multiplies the stellar PSF width
  background        sky_j     = exp(N(0, 0.35))       image sky level; candidates in one image only
                                                      scatter around it by exp(N(0, 0.10))
  noise scale       noise_j   = exp(N(0, 0.10))       read-out / gain conditions of that exposure
  PSF shape         ell_j     ~ U(0, 0.12)            optical ellipticity: lowers every star's roundness
  measurement bias  wcal_j    = exp(N(0, 0.06))       plate-scale / centroiding calibration error:
                                                      multiplies every MEASURED width in the image
                    rbias_j   ~ N(0, 0.03)            shared roundness-estimator bias
  composition       pstar_j   ~ U(0.15, 0.65)         star fraction varies with the field (galactic
                                                      latitude, crowding)
                    mix_j     ~ Dirichlet(8 * base)   artifact mix varies (detector temperature ->
                                                      hot pixels, moonlight -> diffuse glow, ...)
Together these give each image a measurable "nuisance signature": candidates from one image share
background, width calibration and PSF shape, and the label proportions differ by image. A
classifier can partly recognise the image and use its star fraction - that is the leakage channel.

ARTIFICIAL stress-test mechanism (off by default, `stress=True`)
  detector batch bias: every background measurement in image j is multiplied by
  exp(1.5 (pstar_j - 0.4)), i.e. star-rich images read systematically brighter. This has no physical
  basis; it plants an image-level shortcut that is correlated with the label proportions.

Control (`group_effects=False`): every image uses the nominal conditions, pstar = 0.4, the nominal
artifact mix, no measurement bias, and the Module 4 per-source background spread - i.e. images are
just arbitrary batches of independent candidates.

Scenario multipliers (for Experiment E) are applied on top of the per-image draws. A sensor-noise
scenario scales both the pixel noise (noise_mult: lowers SNR, so faint stars drop below the detection
threshold) and the scatter of the width / roundness estimates at a given SNR (feature_noise_mult:
e.g. pointing jitter or poorer centroiding), which is what actually erodes class separation.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "module-4"))

from star_data import (DETECT_SNR, ENG_NAMES, FEATURE_NAMES, READ, engineered)  # noqa: E402,F401

OUT = HERE / "outputs"
STAR_TYPES = ("bright star", "faint star")
ARTIFACT_TYPES = ("hot pixel", "diffuse", "noise peak", "compact irregular", "ghost")
BASE_MIX = {"hot pixel": 0.20, "diffuse": 0.20, "noise peak": 0.40, "compact irregular": 0.20,
            "ghost": 0.0}
INVARIANT = [1, 2, 4]          # log width, roundness, log SNR (Module 4's selected feature set)

NOMINAL = {"noise_mult": 1.0, "defocus_mult": 1.0, "background_mult": 1.0,
           "star_fraction": None, "ghost_fraction": 0.0, "feature_noise_mult": 1.0}


def scenario(**kw):
    s = dict(NOMINAL)
    s.update(kw)
    return s


def image_conditions(rng, group_effects=True, sc=NOMINAL):
    if group_effects:
        g = {"defocus": float(np.exp(rng.normal(0, 0.20))),
             "sky": float(np.exp(rng.normal(0, 0.35))),
             "noise": float(np.exp(rng.normal(0, 0.15))),
             "ell": float(rng.uniform(0, 0.20)),
             "wcal": float(np.exp(rng.normal(0, 0.08))),
             "rbias": float(rng.normal(0, 0.05)),
             "pstar": float(rng.uniform(0.10, 0.70)),
             "batch": float(np.exp(rng.normal(0, 0.6)))}
        keys = [k for k in BASE_MIX if k != "ghost"]
        w = rng.dirichlet([8 * BASE_MIX[k] for k in keys])
        mix = dict(zip(keys, w))
    else:
        g = {"defocus": 1.0, "sky": 1.0, "noise": 1.0, "ell": 0.0, "wcal": 1.0, "rbias": 0.0,
             "pstar": 0.4, "batch": float(np.exp(rng.normal(0, 0.6)))}
        mix = {k: v for k, v in BASE_MIX.items() if k != "ghost"}
    g["defocus"] *= sc["defocus_mult"]
    g["sky"] *= sc["background_mult"]
    g["noise"] *= sc["noise_mult"]
    g["fnoise"] = sc["feature_noise_mult"]
    if sc["star_fraction"] is not None:
        g["pstar"] = sc["star_fraction"]
    gf = sc["ghost_fraction"]
    mix = {k: v * (1 - gf) for k, v in mix.items()}
    mix["ghost"] = gf
    g["mix"] = mix
    return g


def _latent(kind, m, rng, g, group_effects):
    if group_effects:
        B = 100.0 * g["sky"] * rng.lognormal(0.0, 0.05, m)
    else:
        B = 100.0 * g["sky"] * rng.lognormal(0.0, 0.35, m)
    sig = g["noise"] * np.sqrt(B + READ**2)
    if kind == "star":
        w = rng.normal(1.25, 0.12, m) * g["defocus"]
        rho = 1.0 - np.abs(rng.normal(0.0, 0.05, m)) - g["ell"]
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
        A = sig * rng.uniform(2.5, 6.0, m) / (np.sqrt(np.pi) * w) * rng.uniform(0.8, 1.6, m)
    elif kind == "compact irregular":
        w = rng.lognormal(np.log(0.95), 0.25, m)
        rho = rng.uniform(0.10, 0.60, m)
        A = sig * 10 ** rng.uniform(0.3, 1.8, m)
    elif kind == "ghost":
        w = rng.normal(1.9, 0.25, m) * g["defocus"]
        rho = 1.0 - np.abs(rng.normal(0.0, 0.08, m)) - 0.5 * g["ell"]
        A = sig * 10 ** rng.uniform(-0.1, 0.5, m)
    else:
        raise ValueError(kind)
    return A, np.clip(w, 0.3, None), rho, B, sig


def _measure(A, w, rho, B, sig, rng, g, stress):
    """Module 4 measurement model + the image's measurement biases."""
    snr_true = np.sqrt(np.pi) * w * A / sig
    m = A.size
    B_hat = B + rng.normal(0, sig / 8.0, m)
    P_hat = B + A + rng.normal(0, sig, m)
    fn = g.get("fnoise", 1.0)
    w_hat = g["wcal"] * w * np.exp(rng.normal(0, fn * (0.06 + 0.9 / np.maximum(snr_true, 1.0)), m))
    rho_hat = np.clip(rho + g["rbias"] + rng.normal(0, fn * (0.03 + 0.6 / np.sqrt(np.maximum(snr_true, 1.0))), m),
                      0.05, 1.0)
    sig_hat = g["noise"] * np.sqrt(np.maximum(B_hat, 1.0) + READ**2)
    snr_hat = np.sqrt(np.pi) * w_hat * (P_hat - B_hat) / sig_hat
    B_rep = B_hat * g["batch"] if stress else B_hat   # reported background (batch code if stressed)
    return np.column_stack([P_hat, w_hat, rho_hat, B_rep, snr_hat]), snr_true


def _draw(kind, m, rng, g, group_effects, stress):
    feats, snrs, have = [], [], 0
    while have < m:
        need = max(2 * (m - have), 16)
        X, s = _measure(*_latent(kind, need, rng, g, group_effects), rng, g, stress)
        keep = X[:, 4] >= DETECT_SNR
        feats.append(X[keep])
        snrs.append(s[keep])
        have += keep.sum()
    return np.vstack(feats)[:m], np.concatenate(snrs)[:m]


def generate_grouped_candidates(n_images, candidates_per_image, rng, group_effects=True,
                                stress=False, sc=NOMINAL, first_id=0):
    """Returns X (n, 5), y, groups (image id per row), subtype per row, and the image table."""
    X_all, y_all, groups, subs, table = [], [], [], [], []
    lo, hi = candidates_per_image
    for j in range(n_images):
        image_id = first_id + j
        g = image_conditions(rng, group_effects, sc)          # shared by every candidate here
        n = int(rng.integers(lo, hi + 1))
        y = (rng.random(n) < g["pstar"]).astype(int)
        X = np.zeros((n, 5))
        sub = np.empty(n, dtype=object)
        idx = np.nonzero(y == 1)[0]
        if idx.size:
            Xs, snr = _draw("star", idx.size, rng, g, group_effects, stress)
            X[idx] = Xs
            sub[idx] = np.where(snr >= 20, "bright star", "faint star")
        idx = np.nonzero(y == 0)[0]
        if idx.size:
            types = list(g["mix"])
            p = np.array([g["mix"][t] for t in types])
            kinds = rng.choice(len(types), size=idx.size, p=p / p.sum())
            for k, t in enumerate(types):
                sel = idx[kinds == k]
                if sel.size:
                    X[sel], _ = _draw(t, sel.size, rng, g, group_effects, stress)
                    sub[sel] = t
        X_all.append(X)
        y_all.append(y)
        groups.append(np.full(n, image_id))
        subs.append(sub)
        table.append({"image": image_id, "n": n, **{k: v for k, v in g.items() if k != "mix"}})
    return (np.vstack(X_all), np.concatenate(y_all), np.concatenate(groups),
            np.concatenate(subs), table)


# Fixed, reproducible datasets used across the module
DEV_SEED, TEST_SEED = 501, 502
N_DEV_IMAGES, N_TEST_IMAGES = 50, 30
PER_IMAGE = (100, 200)


def development_set(**kw):
    return generate_grouped_candidates(N_DEV_IMAGES, PER_IMAGE, np.random.default_rng(DEV_SEED), **kw)


def untouched_test_set(**kw):
    """Independent images, never used for any choice. Image ids start at 1000."""
    return generate_grouped_candidates(N_TEST_IMAGES, PER_IMAGE, np.random.default_rng(TEST_SEED),
                                       first_id=1000, **kw)
