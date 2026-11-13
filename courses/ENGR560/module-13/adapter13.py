"""
Adapter: Module 12 hybrid-detector catalogue -> astrometric source list.

Rules: coordinates are NEVER modified (copied bit-for-bit; zero-based pixel-centre convention kept),
measurements and uncertainties are preserved, rows are only re-ordered (brightest first) and every row
keeps the index of the detector row it came from. Conversion to the 1-based FITS convention happens only
when a file is written for Astrometry.net (write_xyls), and is the single documented +1.0.
"""

import numpy as np

FIELDS = ("x", "y", "flux", "snr", "centroid_uncertainty", "quality_flags", "cnn_score")


def to_source_list(cat, reject_flags=0):
    n = len(cat["x"])
    keep = np.ones(n, bool) if not reject_flags else (np.asarray(cat["quality_flags"]) & reject_flags) == 0
    flux = np.where(np.isfinite(cat["flux"]), cat["flux"], -np.inf)
    order = np.nonzero(keep)[0][np.argsort(-flux[keep], kind="stable")]
    out = {k: np.asarray(cat[k])[order].copy() for k in FIELDS if k in cat}
    out["source_row"] = order
    # per-axis 1-sigma pixel uncertainty for weighting (the catalogue reports the radial value)
    cu = np.asarray(cat["centroid_uncertainty"], float)[order] if "centroid_uncertainty" in cat else np.full(len(order), np.nan)
    out["sigma_px"] = np.where(np.isfinite(cu) & (cu > 0), cu / np.sqrt(2), 0.3)
    return out


def check_unmodified(cat, src):
    r = src["source_row"]
    return bool(np.array_equal(np.asarray(cat["x"])[r], src["x"]) and np.array_equal(np.asarray(cat["y"])[r], src["y"]))


def write_xyls(src, path, one_based=True):
    from astropy.io import fits
    off = 1.0 if one_based else 0.0
    cols = [fits.Column(name="X", format="D", array=np.asarray(src["x"], np.float64) + off),
            fits.Column(name="Y", format="D", array=np.asarray(src["y"], np.float64) + off),
            fits.Column(name="FLUX", format="D", array=np.asarray(src["flux"], np.float64))]
    fits.BinTableHDU.from_columns(cols).writeto(path, overwrite=True)
