"""
stellar_vision - astronomical perception pipeline (ENGR 560 capstone), built from Modules 11-13.

    from stellar_vision import StellarVisionPipeline, VisionConfig
    pipeline = StellarVisionPipeline.from_config(VisionConfig(detection_mode="hybrid", verify_solution=True))
    result = pipeline.solve(image="night_sky.fits", camera="camera_config.json")

Stages: 1 image -> 2 calibration -> 3 classical detection -> 3' screening (classical rules | CNN)
        -> 4 centroids + uncertainty -> 5 geometric identification -> 6 attitude + verification
        -> AstrometricResult (verified solution, or failure_code + failure_reason)
"""

from .catalog import StarCatalog
from .config import AcceptanceConfig, CalibrationConfig, DetectionConfig, SolverConfig, VisionConfig
from .geometry import CameraModel
from .pipeline import StellarVisionPipeline
from .result import AstrometricResult, FailureCode

__version__ = "1.0.0"
__all__ = ["StellarVisionPipeline", "VisionConfig", "CameraModel", "StarCatalog", "AstrometricResult", "FailureCode",
           "CalibrationConfig", "DetectionConfig", "SolverConfig", "AcceptanceConfig"]
