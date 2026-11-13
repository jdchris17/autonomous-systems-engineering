"""
Shared evaluation conditions and helpers for the capstone experiments (synthetic data only).

Condition matrix (Experiment C; Experiment B uses the negative controls in scenarios.py):
  nominal                  Module 13 Experiment C images (seeds 200-205, identical renders)
  low SNR                  background 300 e-, read noise 10 e-, stars 4x fainter (m = 8 -> 250 e-)
  strong defocus           PSF FWHM 6.5 px (detector / CNN still assume 3 px; no retraining)
  hot pixels & streaks     60-90 hot pixels, 12-20 cosmic-ray streaks per image
  missing & uncatalogued   solver catalogue missing 20% of its stars; 120 uncatalogued field stars
  incorrect sky / catalog  3 fields from a different catalogue + 3 fields pointed 5.5 deg from the
                           search centre (outside the 2.5 deg search region) - must NOT solve
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stellar_vision import StarCatalog, StellarVisionPipeline, VisionConfig  # noqa: E402
from stellar_vision.geometry import attitude_from_pointing, vec_to_radec  # noqa: E402
from stellar_vision.simulation import CENTER, make_catalog, random_pointing, render  # noqa: E402

CAT = StarCatalog.load(ROOT / "stellar_vision" / "data" / "catalog_m13.npz")
OTHER = make_catalog(seed=999, name="other-catalog-seed999")

CONDITIONS = {
    "nominal": {"seeds": range(200, 206), "render": {}, "read_noise": 5.0},
    "low SNR": {"seeds": range(410, 416), "render": {"background": 300.0, "read_noise": 10.0, "zp_flux": 250.0},
                "read_noise": 10.0},
    "strong defocus": {"seeds": range(420, 426), "render": {"fwhm": 6.5}, "read_noise": 5.0},
    "hot pixels & streaks": {"seeds": range(430, 436), "render": {"n_hot": (60, 90), "n_streaks": (12, 20)},
                             "read_noise": 5.0},
    "missing & uncatalogued": {"seeds": range(440, 446), "render": {"n_field": 120}, "read_noise": 5.0,
                               "catalog_missing": 0.2},
    "incorrect sky / catalog": {"seeds": range(450, 456), "render": {}, "read_noise": 5.0, "incorrect": True},
}
CALIBRATION_SEEDS = range(300, 312)        # threshold setting only - never used for evaluation


def pointing_outside(rng, offset=5.5):
    R0 = attitude_from_pointing(CENTER[0], CENTER[1], 0.0)
    phi = rng.uniform(0, 2 * np.pi)
    rho = np.radians(offset)
    local = np.array([np.sin(rho) * np.cos(phi), np.sin(rho) * np.sin(phi), np.cos(rho)])
    ra, dec = vec_to_radec(R0.T @ local)
    return attitude_from_pointing(float(ra), float(dec), rng.uniform(0, 360))


def make_field(cname, seed):
    """Returns (image, truth, R_true, solver catalogue, expect_solution)."""
    c = CONDITIONS[cname]
    rng = np.random.default_rng(seed)
    if c.get("incorrect"):
        if seed % 2 == 0:                                       # different catalogue, same sky region
            R, _ = random_pointing(rng)
            image, truth = render(OTHER, R, seed, **c["render"])
            truth["cat_idx"][:] = -1
        else:                                                   # right catalogue, outside the search region
            R = pointing_outside(rng)
            image, truth = render(CAT, R, seed, **c["render"])
        return image, truth, R, CAT, False
    R, _ = random_pointing(rng)
    image, truth = render(CAT, R, seed, **c["render"])
    cat = CAT
    if c.get("catalog_missing"):
        keep = np.random.default_rng(seed + 7).random(len(CAT)) >= c["catalog_missing"]
        cat = CAT.subset(keep, f"catalog-minus-20pct-seed{seed}")     # unique name: pipelines are cached by it
        removed = set(np.nonzero(~keep)[0].tolist())
        truth["cat_idx"] = np.array([-1 if (i < 0 or i in removed) else int(np.searchsorted(np.nonzero(keep)[0], i))
                                     for i in truth["cat_idx"]])
    return image, truth, R, cat, True


def config_for(mode, cname=None, **acceptance):
    cfg = VisionConfig(detection_mode=mode)
    if cname is not None:
        cfg.calibration = replace(cfg.calibration, read_noise_e=CONDITIONS[cname]["read_noise"])
    if acceptance:
        cfg.acceptance = replace(cfg.acceptance, **acceptance)
    return cfg


def truth_labels(result, truth, max_dist=2.0):
    """Label every source-table row: >= 0 catalogue index (of the catalogue used), -1 field star / artifact."""
    from stellar_vision._m11 import match
    src = result.sources
    t = np.full(len(src["x"]), -2)
    if len(src["x"]) and len(truth["x"]):
        pairs, d = match(np.column_stack([src["x"], src["y"]]), np.column_stack([truth["x"], truth["y"]]), max_dist)
        for di, ti in pairs:
            t[di] = truth["cat_idx"][ti]
    return t
