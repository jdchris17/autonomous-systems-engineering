"""Astronomical time scales and Earth rotation (Modules 1-2)."""

from . import _legacy  # noqa: F401
from astronomical_time import (  # noqa: E402,F401
    EARTH_ROTATION_RATE_UT1,
    configure_offline_iers,
    earth_rotation_angle_iau,
    iers_status,
    time_scales,
)
from compare_reference_frames import polar_motion, reference_matrix  # noqa: E402,F401
