"""Stellar attitude determination (Module 4)."""

from . import _legacy  # noqa: F401
from star_field_simulation import (  # noqa: E402,F401
    centroid_sigma_to_angle,
    unit_square_positions,
)
from stellar_attitude import (  # noqa: E402,F401
    InsufficientGeometryError,
    angular_residuals,
    attitude_covariance,
    attitude_error,
    attitude_information,
    ransac_attitude,
    robust_wahba,
    wahba_svd,
)
