"""
enduria - terrestrial celestial navigation engine (ENGR 570 capstone).

One public operation:

    result = solve_navigation_fix(observation_batch, config, references)

The algorithms live in the course module folders (module-1 ... module-8),
which remain the reference implementations; this package integrates them.
"""

from .astrometry import STANDARD_ATMOSPHERE, Atmosphere, CatalogStar
from .contracts import (
    DEGRADED,
    FIX_STATUSES,
    REJECTED,
    STATUSES,
    UNOBSERVABLE,
    VALID,
    NavigationConfig,
    NavigationInputError,
    NavigationResult,
    Observation,
    ObservationBatch,
)
from .frames import CameraModel
from .pipeline import solve_navigation_fix
from .references import ReferenceData

__version__ = "1.0.0"
__all__ = [
    "solve_navigation_fix", "NavigationResult", "NavigationConfig",
    "Observation", "ObservationBatch", "ReferenceData", "CameraModel",
    "CatalogStar", "Atmosphere", "STANDARD_ATMOSPHERE", "NavigationInputError",
    "VALID", "DEGRADED", "UNOBSERVABLE", "REJECTED", "STATUSES", "FIX_STATUSES",
]
