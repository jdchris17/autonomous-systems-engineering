"""
Explicit, serialisable configuration. Everything the pipeline depends on is declared here: camera
intrinsics, calibration version, star catalogue, detector / classifier settings (incl. CNN checkpoint),
solver settings and acceptance criteria. Relative paths resolve against the package data directory.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

from .geometry import CameraModel

DATA_DIR = Path(__file__).resolve().parent / "data"


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else (DATA_DIR / p)


@dataclass
class CalibrationConfig:
    version: str = "m11-m12-synthetic-2026.1"
    read_noise_e: float = 5.0               # e- rms, used in the PSF-fit pixel weights
    gain_e_per_adu: float = 1.0             # image (ADU) -> electrons
    bias_adu: float = 0.0
    dark_frame: str | None = None           # optional .npy / FITS (ADU), subtracted
    flat_field: str | None = None           # optional .npy / FITS (normalised to 1), divided
    uncertainty_scale: float = 1.118        # empirical centroid-uncertainty calibration (Module 11)


@dataclass
class DetectionConfig:
    k_sigma: float = 5.0                    # matched-filter threshold (x filtered-background RMS)
    min_sep_px: float = 4.0
    window_px: int = 5
    fwhm_assumed_px: float | None = None    # None -> camera.psf_fwhm_px
    fit_half_width_px: int = 5
    post_fit_reject: tuple = ("SHARP", "FIT_FAILED")   # quality rules applied in BOTH modes
    cnn_checkpoint: str = "cnn_best.pt"     # hybrid mode only
    cnn_threshold: float | None = None      # P(star); None -> threshold frozen in the checkpoint


@dataclass
class SolverConfig:
    catalog: str = "catalog_m13.npz"
    search_center_radec_deg: tuple = (150.0, 20.0)
    search_radius_deg: float = 2.5          # bound on the boresight position (not a blind solve)
    tolerance_sigma_px: float = 0.2         # centroid sigma used for triangle tolerances / match radius
    k_det: int = 14                         # brightest sources used to form triangles
    max_hypotheses: int = 20000
    precheck_min_matches: int = 6
    polish_iterations: int = 50             # 0 = Module 13 behaviour (4 alternations only)
    mag_limit: float | None = None


@dataclass
class AcceptanceConfig:
    """Reference thresholds; they worked on the synthetic experiments and are NOT universal statistical
    guarantees. Residual / geometry thresholds = ~3x the largest value seen for CORRECT solutions on the
    calibration fields (seeds 300-311: nominal, low SNR, defocus, artifacts; both modes) - never on the
    evaluation fields (experiments/calibrate_thresholds.py): reduced chi2 max 1.61, RMS 0.53 px,
    roll sigma 63", cross-boresight sigma 1.95"."""
    min_sources: int = 5                    # fewer usable sources -> INSUFFICIENT_SOURCES
    min_verify_matches: int = 6             # matches beyond the 3 hypothesis stars
    min_catalog_fraction: float = 0.5       # matched / catalogue stars predicted in the frame
    max_reduced_chi2: float = 5.0          # sum d_i^2 / (2N - 4), d_i^2 = r_i^T Sigma_i^-1 r_i
    max_rms_px: float = 1.5                 # absolute sanity bound on the reprojection RMS
    max_roll_sigma_arcsec: float = 200.0    # attitude observability (Fisher covariance)
    max_cross_sigma_arcsec: float = 6.0
    ambiguity_ratio: float = 0.8            # runner-up with >= this fraction of the best's matches -> AMBIGUOUS
    distinct_hypothesis_deg: float = 0.25   # rotations further apart than this are different solutions
    focal_bound_margin: float = 0.002       # fitted f within this of the permitted bound -> CAMERA_MODEL_INVALID


@dataclass
class VisionConfig:
    detection_mode: str = "hybrid"          # "classical" | "hybrid"
    verify_solution: bool = True
    offline: bool = True                    # never touches the network; external solvers are not used
    deterministic: bool = True              # single-threaded torch inference
    camera: CameraModel | str = "camera_m13.json"
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    acceptance: AcceptanceConfig = field(default_factory=AcceptanceConfig)

    def __post_init__(self):
        if self.detection_mode not in ("classical", "hybrid"):
            raise ValueError(f"detection_mode must be 'classical' or 'hybrid', not {self.detection_mode!r}")
        if isinstance(self.camera, (str, Path)):
            self.camera = CameraModel.from_json(resolve(self.camera))
        elif isinstance(self.camera, dict):
            self.camera = CameraModel.from_json(self.camera)

    # --- serialisation --------------------------------------------------------------------------
    def to_dict(self):
        d = {}
        for f in fields(self):
            v = getattr(self, f.name)
            d[f.name] = v.to_dict() if hasattr(v, "to_dict") else asdict(v) if is_dataclass(v) else v
        return d

    def to_json(self, path=None):
        s = json.dumps(self.to_dict(), indent=2, default=list)
        if path:
            Path(path).write_text(s)
        return s

    @classmethod
    def from_dict(cls, d):
        sub = {"calibration": CalibrationConfig, "detection": DetectionConfig, "solver": SolverConfig,
               "acceptance": AcceptanceConfig}
        kw = {}
        for k, v in d.items():
            if k in sub:
                v = dict(v)
                for t in ("post_fit_reject", "search_center_radec_deg"):
                    if t in v and isinstance(v[t], list):
                        v[t] = tuple(v[t])
                kw[k] = sub[k](**v)
            else:
                kw[k] = v
        return cls(**kw)

    @classmethod
    def from_json(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def fingerprint(self):
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]
