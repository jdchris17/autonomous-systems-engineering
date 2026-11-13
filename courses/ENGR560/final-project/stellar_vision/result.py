"""
Output contract: AstrometricResult (verified solution OR explicit, coded failure).

Units and frames (see geometry.py): pixels zero-based; attitude R_CI (camera <- inertial); angles in
degrees except uncertainties (arcsec) and the covariance (rad^2). The attitude covariance is that of the
small-angle error vector phi in the CAMERA frame, R_est = exp([phi]x) R_true: components (x, y) are
cross-boresight, z is roll about the boresight.
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class FailureCode(str, Enum):
    INVALID_INPUT = "INVALID_INPUT"                    # image / camera / config unusable
    NO_SOURCES = "NO_SOURCES"                          # detector found nothing
    INSUFFICIENT_SOURCES = "INSUFFICIENT_SOURCES"      # fewer usable sources than min_sources
    NO_HYPOTHESIS = "NO_HYPOTHESIS"                    # no geometrically compatible catalogue triangle
    INSUFFICIENT_MATCHES = "INSUFFICIENT_MATCHES"      # best hypothesis lacks independent support
    LOW_CATALOG_FRACTION = "LOW_CATALOG_FRACTION"      # predicted catalogue stars not seen
    CAMERA_MODEL_INVALID = "CAMERA_MODEL_INVALID"      # needs a focal length outside the permitted range
    RESIDUALS_INCONSISTENT = "RESIDUALS_INCONSISTENT"  # residuals implausible given the uncertainties
    WEAK_GEOMETRY = "WEAK_GEOMETRY"                    # attitude poorly observable (star configuration)
    AMBIGUOUS = "AMBIGUOUS"                            # a distinct competing hypothesis is nearly as good


@dataclass
class AstrometricResult:
    success: bool                                  # True only for a VERIFIED solution (or verify_solution=False)
    verified: bool
    failure_code: FailureCode | None = None
    failure_reason: str | None = None

    attitude: np.ndarray | None = None             # R_CI 3 x 3
    quaternion: np.ndarray | None = None           # (w, x, y, z) of R_CI, scalar first
    boresight_radec_deg: tuple | None = None
    roll_deg: float | None = None                  # camera +x from local east towards north
    attitude_covariance: np.ndarray | None = None  # 3 x 3 rad^2, camera frame (cross-x, cross-y, roll)
    attitude_sigma_arcsec: dict | None = None      # {"cross_x", "cross_y", "roll"}
    focal_length_px: float | None = None           # fitted

    matched_star_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    matched_source_index: np.ndarray = field(default_factory=lambda: np.zeros(0, int))  # rows of `sources`
    measured_centroids: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))      # px, zero-based
    residuals_px: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))           # measured - predicted
    normalized_residuals_sq: np.ndarray = field(default_factory=lambda: np.zeros(0))     # d_i^2

    n_detected: int = 0                            # detector candidates
    n_sources: int = 0                             # sources passed to the solver
    n_matched: int = 0
    n_verify: int = 0                              # matches beyond the hypothesis triple
    reprojection_rms_px: float | None = None
    catalog_match_fraction: float | None = None
    wcs: object | None = None                      # astropy.wcs.WCS (TAN), exact pinhole equivalent
    sources: dict | None = None                    # full source table (all fitted candidates + accepted flag)
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self):
        """JSON-serialisable view (arrays -> lists, WCS -> FITS header cards)."""
        out = {}
        for k, v in self.__dict__.items():
            if k == "wcs":
                out[k] = dict(v.to_header()) if v is not None else None
            elif k == "sources" and v is not None:
                out[k] = {kk: np.asarray(vv).tolist() for kk, vv in v.items()}
            else:
                out[k] = _plain(v)
        return out


def _plain(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, Enum):
        return v.value
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v
