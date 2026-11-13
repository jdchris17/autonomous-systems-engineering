"""
Pipeline stages 2-4. Each stage is a plain function / class with a typed input and output so that it can
be tested on its own.

  2 calibrate()            image (ADU) -> CalibratedImage (electrons, background, noise)
  3 detect_candidates()    CalibratedImage -> Candidates            (Module 11 matched filter, unchanged)
  3' screens               Candidates -> Screen                      classical: keep all (shape rules act
                                                                     after the fit); hybrid: Module 12 CNN
                                                                     rejects BEFORE fitting
  4 fit_centroids()        CalibratedImage + Candidates + Screen -> SourceTable (Module 11 Gaussian PSF
                                                                     fit, Module 12 flag logic, quality rules)
"""

from dataclasses import dataclass

import numpy as np

from . import _m11
from .config import resolve

FLAGS = {"SHARP": 1, "EDGE": 2, "CROWDED": 4, "LOW_SNR": 8, "BAD_FIT": 16, "FIT_FAILED": 32}


def flag_string(f):
    return "|".join(n for n, b in FLAGS.items() if f & b) or "OK"


# --------------------------------------------------------------------------------------- stage 2
@dataclass
class CalibratedImage:
    data: np.ndarray            # electrons (2-D)
    background: float           # sigma-clipped median, e-
    noise_rms: float            # MAD-based background RMS, e-
    version: str
    corrections: tuple          # which corrections were applied


def _load_frame(path):
    if str(path).lower().endswith(".npy"):
        return np.load(resolve(path))
    from astropy.io import fits
    return fits.getdata(resolve(path))


def calibrate(image, calib):
    img = np.asarray(image)
    if img.ndim != 2 or img.size == 0 or not np.all(np.isfinite(img)):
        raise ValueError("image must be a finite, non-empty 2-D array")
    applied = []
    data = img
    if calib.bias_adu or calib.dark_frame or calib.flat_field or calib.gain_e_per_adu != 1.0:
        data = img.astype(np.float64) - calib.bias_adu
        applied.append("bias")
        if calib.dark_frame:
            data = data - _load_frame(calib.dark_frame)
            applied.append("dark")
        if calib.flat_field:
            data = data / _load_frame(calib.flat_field)
            applied.append("flat")
        data = data * calib.gain_e_per_adu
        applied.append("gain")
    bkg, rms = _m11.background_stats(data)
    return CalibratedImage(data, bkg, rms, calib.version, tuple(applied))


# --------------------------------------------------------------------------------------- stage 3
@dataclass
class Candidates:
    xy: np.ndarray              # N x 2 peak pixel coordinates (x = column, y = row), zero-based
    background: float           # detector's own background estimate (e-)
    rms: float
    filtered_rms: float
    fwhm_assumed: float


def detect_candidates(cal, det_cfg, camera):
    fwhm = det_cfg.fwhm_assumed_px or camera.psf_fwhm_px
    xy, diag = _m11.detect(cal.data, fwhm_assumed=fwhm, k=det_cfg.k_sigma, min_sep=det_cfg.min_sep_px,
                           window=det_cfg.window_px, return_diagnostics=True)
    return Candidates(xy, diag["background"], diag["rms"], diag["filtered_rms"], fwhm)


@dataclass
class Screen:
    keep: np.ndarray            # bool per candidate: passed to the PSF fit
    score: np.ndarray           # P(star) from the CNN (NaN for the classical screen)
    name: str


class ClassicalScreen:
    """No pre-fit rejection: the classical physical-shape rules (SHARP etc.) need the PSF fit and are
    applied after it (fit_centroids). Does not import torch."""
    name = "classical"

    def __call__(self, cal, cand):
        n = len(cand.xy)
        return Screen(np.ones(n, bool), np.full(n, np.nan), self.name)


class CNNScreen:
    """Module 12 CNN on 32 x 32 patches, batched, BEFORE fitting (identical to module-12 HybridDetector)."""
    name = "hybrid"

    def __init__(self, checkpoint, threshold=None, deterministic=True):
        import torch
        from ._m12_cnn import StarCNN
        if deterministic:
            torch.set_num_threads(1)
        ck = torch.load(resolve(checkpoint), weights_only=True)
        self.model = StarCNN()
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.eval()
        self.pre = ck["preprocessing"]
        self.threshold_logit = ck["threshold_logit"] if threshold is None else float(np.log(threshold / (1 - threshold)))
        self.threshold = 1 / (1 + np.exp(-self.threshold_logit))
        self._torch = torch

    def __call__(self, cal, cand):
        from ._m12_cnn import center_crop, extract, transform
        torch = self._torch
        patches = center_crop(extract(cal.data, cand.xy, cand.background, cand.rms))
        x = torch.from_numpy(transform(patches, self.pre["mean"], self.pre["std"])[:, None])
        with torch.inference_mode():
            logit = self.model(x).numpy() if len(cand.xy) else np.zeros(0, np.float32)
        keep = logit >= self.threshold_logit
        return Screen(keep, 1 / (1 + np.exp(-logit.astype(float))), self.name)


# --------------------------------------------------------------------------------------- stage 4
@dataclass
class SourceTable:
    """One row per FITTED candidate. `accepted` = passed the quality rules (these go to the solver).
    Coordinates are the detector's measurements and are never modified downstream."""
    x: np.ndarray
    y: np.ndarray
    flux: np.ndarray
    snr: np.ndarray
    sigma: np.ndarray                   # fitted PSF sigma (px)
    sx: np.ndarray
    sy: np.ndarray
    centroid_uncertainty: np.ndarray    # radial 1-sigma (px), empirically calibrated
    quality_flags: np.ndarray
    cnn_score: np.ndarray
    candidate_index: np.ndarray         # row in Candidates
    accepted: np.ndarray
    background: float
    background_rms: float

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items()}

    def __len__(self):
        return len(self.x)


def fit_centroids(cal, cand, screen, det_cfg, calib, camera):
    fwhm = cand.fwhm_assumed
    s0 = fwhm * _m11.FWHM_TO_SIGMA
    keep_idx = np.nonzero(screen.keep)[0]
    xy = cand.xy[keep_idx]
    n = len(xy)
    out = {k: np.zeros(n) for k in ("x", "y", "flux", "snr", "sigma", "sx", "sy")}
    flags = np.zeros(n, int)
    ny, nx = cal.data.shape
    bkg, rms = cand.background, cand.rms
    for i, (xc, yc) in enumerate(xy):
        f = _m11.gaussian_fit(cal.data, bkg, xc, yc, s0, calib.read_noise_e, half=det_cfg.fit_half_width_px)
        ok = f["ok"] and np.isfinite(f["x"]) and np.isfinite(f["sigma"])
        out["x"][i], out["y"][i] = (f["x"], f["y"]) if ok else (xc, yc)
        out["flux"][i], out["sigma"][i], out["sx"][i], out["sy"][i] = f["flux"], f["sigma"], f["sx"], f["sy"]
        sig = f["sigma"] if np.isfinite(f["sigma"]) else s0
        F = max(f["flux"], 0) if np.isfinite(f["flux"]) else 0.0
        out["snr"][i] = F / np.sqrt(F + 4 * np.pi * sig**2 * rms**2) if F > 0 else 0.0
        if not ok or f["sigma"] > 1.6 * s0:
            flags[i] |= FLAGS["BAD_FIT"]
        if not ok:
            flags[i] |= FLAGS["FIT_FAILED"]
        if np.isfinite(f["sigma"]) and f["sigma"] < 0.6:
            flags[i] |= FLAGS["SHARP"]
        if xc < 6 or yc < 6 or xc > nx - 7 or yc > ny - 7:
            flags[i] |= FLAGS["EDGE"]
    if n > 1:
        D = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
        np.fill_diagonal(D, np.inf)
        flags[D.min(axis=1) < 2 * fwhm] |= FLAGS["CROWDED"]
    flags[out["snr"] < 10] |= FLAGS["LOW_SNR"]
    reject = sum(FLAGS[r] for r in det_cfg.post_fit_reject)
    return SourceTable(out["x"], out["y"], out["flux"], out["snr"], out["sigma"], out["sx"], out["sy"],
                       calib.uncertainty_scale * np.hypot(out["sx"], out["sy"]), flags, screen.score[keep_idx],
                       keep_idx, (flags & reject) == 0, bkg, rms)


# ---------------------------------------------------------------------------- adapter (stage 4 -> 5)
def to_source_list(table):
    """Module 13 adapter: accepted rows, brightest first; coordinates copied untouched; per-axis sigma =
    radial uncertainty / sqrt(2) (0.3 px if unavailable). `source_row` maps back to the table."""
    acc = np.nonzero(table.accepted)[0]
    flux = np.where(np.isfinite(table.flux), table.flux, -np.inf)
    order = acc[np.argsort(-flux[acc], kind="stable")]
    cu = table.centroid_uncertainty[order]
    return {"x": table.x[order].copy(), "y": table.y[order].copy(), "flux": table.flux[order].copy(),
            "snr": table.snr[order].copy(), "centroid_uncertainty": cu.copy(),
            "quality_flags": table.quality_flags[order].copy(), "cnn_score": table.cnn_score[order].copy(),
            "sigma_px": np.where(np.isfinite(cu) & (cu > 0), cu / np.sqrt(2), 0.3), "source_row": order}


def write_xyls(src, path, one_based=True):
    """Astrometry.net source list (FITS binary table). The ONLY coordinate change: +1.0 to 1-based FITS."""
    from astropy.io import fits
    off = 1.0 if one_based else 0.0
    cols = [fits.Column(name="X", format="D", array=np.asarray(src["x"], np.float64) + off),
            fits.Column(name="Y", format="D", array=np.asarray(src["y"], np.float64) + off),
            fits.Column(name="FLUX", format="D", array=np.asarray(src["flux"], np.float64))]
    fits.BinTableHDU.from_columns(cols).writeto(path, overwrite=True)
