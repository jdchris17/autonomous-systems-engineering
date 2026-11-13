"""
Module 9 data: the Module 6 grouped patch datasets (via Module 7's cache) plus a shifted-conditions
generator for the distribution-shift tests of Experiment E.

Representations
  eng      log width, roundness, log SNR (measured from pixels, Module 6)
  hybrid   eng + the first 8 flux-normalised PCA components (PCA-C, fit on development images only)
Splits (all by image)
  development  60 images (2982 patches): every choice - hyperparameters, calibration, thresholds
  test         30 untouched images (1499 patches): final evaluation only
  shifted      30-image scenario sets generated here, each with its own seed
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "module-7"))
sys.path.insert(0, str(HERE.parent / "module-6"))

import data7  # noqa: E402
from exp_c_reconstruction import preprocess  # noqa: E402
from pca_lda import PCAFromScratch  # noqa: E402
from patches import (ARTIFACT_MIX, CENTRE, PER_IMAGE, SIZE, SUBTYPES, _source,  # noqa: E402,F401
                     elliptical_source, engineered_features, image_conditions, noisy_patch)

OUT = HERE / "outputs"
N_PCA = 8
SUBTYPES_SHIFT = SUBTYPES + ("ghost",)


def load():
    """Development / test arrays plus the hybrid representation and a fitted PCA-C transform."""
    z = data7.load()
    cache = OUT / "pcaC_model.npz"
    if not cache.exists():
        OUT.mkdir(exist_ok=True)
        F, _ = preprocess(z["dev_X"].astype(float), "C")
        p = PCAFromScratch(N_PCA).fit(F)
        np.savez(cache, mean=p.mean_, comps=p.components_)
    pm = np.load(cache)
    d = {}
    for tag in ("dev", "test"):
        d[tag] = {"eng": z[f"{tag}_eng"], "y": z[f"{tag}_y"], "group": z[f"{tag}_group"],
                  "subtype": z[f"{tag}_subtype"], "snr": z[f"{tag}_snr_true"],
                  "hybrid": np.hstack([z[f"{tag}_eng"], z[f"{tag}_pcaC"][:, :N_PCA]])}
    d["pca_mean"], d["pca_comps"] = pm["mean"], pm["comps"]
    return d


def features_from_patches(P, pca_mean, pca_comps):
    """Feature extraction for new patches: engineered features + PCA-C scores."""
    eng = engineered_features(P)
    F, _ = preprocess(P, "C")
    return eng, np.hstack([eng, (F - pca_mean) @ pca_comps.T])


# --------------------------------------------------------------------------- shifted scenarios
SCENARIOS = {
    "nominal": {},
    "measurement noise (RN x3)": {"rn": 3.0},
    "optical defocus (PSF x1.6)": {"defocus": 1.6},
    "bright background (B x6)": {"background": 6.0},
    "rare stars (5%)": {"pstar": 0.05},
    "unseen ghosts (35% of artifacts)": {"ghost": 0.35},
}


def _ghost(g, rng):
    """Internal-reflection ghost: a faint, round, out-of-focus pupil image (ring) never seen in training."""
    noise = np.sqrt(g["B"] + g["RN"] ** 2)
    x0, y0 = CENTRE + rng.uniform(-1.5, 1.5, 2)
    s_out = g["sigma"] * rng.uniform(2.0, 3.0)
    ring = (elliptical_source(1.0, s_out, s_out, 0, x0, y0) - 0.7 * elliptical_source(1.0, 0.6 * s_out, 0.6 * s_out, 0, x0, y0))
    ring = np.clip(ring, 0, None)
    snr = rng.uniform(5, 15)
    flux = snr * noise * np.sqrt(4 * np.pi) * s_out
    return ring / ring.sum() * flux


def generate_shifted(scenario, n_images=30, seed=0, first_id=5000):
    """Module 6 image formation with one condition changed (same per-image variation otherwise)."""
    sc = SCENARIOS[scenario]
    rng = np.random.default_rng([900, seed, list(SCENARIOS).index(scenario)])
    kinds = list(ARTIFACT_MIX)
    probs = np.array([ARTIFACT_MIX[k] for k in kinds])
    X, y, sub, grp = [], [], [], []
    for j in range(n_images):
        g = image_conditions(rng, sc.get("pstar"))
        g["RN"] *= sc.get("rn", 1.0)
        g["sigma"] *= sc.get("defocus", 1.0)
        g["B"] *= sc.get("background", 1.0)
        n = int(rng.integers(PER_IMAGE[0], PER_IMAGE[1] + 1))
        for _ in range(n):
            if rng.random() < g["pstar"]:
                src, p = _source("star", g, rng)
                y.append(1)
                sub.append("bright star" if p["snr_true"] >= 25 else "faint star")
            elif rng.random() < sc.get("ghost", 0.0):
                src = _ghost(g, rng)
                y.append(0)
                sub.append("ghost")
            else:
                kind = kinds[rng.choice(len(kinds), p=probs)]
                src, _ = _source(kind, g, rng)
                y.append(0)
                sub.append(kind)
            X.append(noisy_patch(g["B"] + src, g["RN"], rng))
            grp.append(first_id + j)
    return np.array(X), np.array(y), np.array(sub), np.array(grp)
