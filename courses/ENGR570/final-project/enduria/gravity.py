"""Accelerometer calibration, local vertical and its quality (Module 5)."""

import numpy as np

from . import _legacy  # noqa: F401
from gravity_estimator import (  # noqa: E402,F401
    QUALITY_CAVEAT,
    assess_measurement_quality,
    calibrate_accelerometer,
    estimate_up_direction,
    tangent_basis,
)
from imu_alignment import perturb_calibration  # noqa: E402,F401

# Somigliana closed form for normal gravity on the WGS84 ellipsoid, with the
# linear free-air decrease in height. New in the capstone: Modules 5-8 used
# one illustrative value (9.81 m/s^2) everywhere, which is 0.03 m/s^2 wrong
# at the equator and would trip the specific-force magnitude check there.
_GAMMA_EQUATOR = 9.7803253359
_SOMIGLIANA_K = 0.00193185265241
_E2 = 0.00669437999013


def normal_gravity(latitude_deg, height_m=0.0):
    """Magnitude of normal gravity [m/s^2] at a geodetic latitude and height."""
    s2 = np.sin(np.radians(latitude_deg)) ** 2
    surface = _GAMMA_EQUATOR * (1.0 + _SOMIGLIANA_K * s2) / np.sqrt(1.0 - _E2 * s2)
    return float(surface - 3.086e-6 * height_m)
