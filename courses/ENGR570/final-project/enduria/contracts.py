"""
contracts.py - the data that crosses the public interface.

    ObservationBatch + NavigationConfig + ReferenceData  ->  NavigationResult

Units: pixels for centroids, m/s^2 for specific force (a stationary
accelerometer reads +g along local UP), rad/s for gyro rates, ISO UTC
strings for time, degrees for reported positions, metres squared (north,
east) for the reported covariance.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .astrometry import Atmosphere
from .frames import CameraModel

VALID = "VALID"
DEGRADED = "DEGRADED"
UNOBSERVABLE = "UNOBSERVABLE"
REJECTED = "REJECTED"
STATUSES = (VALID, DEGRADED, UNOBSERVABLE, REJECTED)
FIX_STATUSES = (VALID, DEGRADED)          # a position is reported


class NavigationInputError(ValueError):
    """The batch or the configuration cannot be processed at all."""


# ------------------------------------------------------------------- inputs
@dataclass
class Observation:
    """
    One exposure with its accelerometer window.

    The camera input is MATCHED star centroids: pixel coordinates together
    with the catalog identity of each star. Detection and plate solving
    (ENGR 560) happen before this interface.
    """
    time_utc: str                           # instrument-clock timestamp
    centroids_px: np.ndarray                # (N, 2) [u, v]
    star_ids: list                          # N catalog identities
    accel_samples: np.ndarray               # (M, 3) raw, IMU axes [m/s^2]
    gyro_samples: Optional[np.ndarray] = None   # (M, 3) [rad/s], if available
    label: str = ""


@dataclass
class ObservationBatch:
    observations: list
    metadata: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.observations)


@dataclass(frozen=True)
class NavigationConfig:
    """
    Instrument calibration, declared uncertainties and estimator settings.

    Every sigma is a one-sigma DECLARATION by whoever calibrated the
    instrument. The engine propagates them; it cannot check them.
    """
    # ---- calibration ---------------------------------------------------------
    camera: CameraModel = CameraModel()
    C_CI: object = None                     # camera-from-IMU, 3x3 or Transform
    accel_bias: tuple = (0.0, 0.0, 0.0)     # m/s^2, subtracted first
    accel_correction: object = None         # 3x3 K, f = K (raw - bias)
    # ---- declared random noise ----------------------------------------------
    centroid_sigma_px: float = 0.2
    accel_noise_sigma: float = 0.005        # per axis, per sample
    gyro_noise_sigma: Optional[float] = None
    # ---- declared fixed (systematic) uncertainties --------------------------
    accel_bias_sigma: float = 0.0           # m/s^2 per axis, after calibration
    misalignment_sigma_arcsec: float = 0.0  # camera-IMU, per axis
    principal_point_sigma_px: float = 0.0
    clock_sigma_s: float = 0.0
    deflection_sigma_arcsec: float = 0.0    # per component, after any model
    # ---- environment -----------------------------------------------------------
    apply_refraction: bool = True
    atmosphere: Optional[Atmosphere] = None  # None = no weather data supplied
    refraction_fraction_sigma: float = 0.02  # with measured weather
    assumed_atmosphere_fraction_sigma: float = 0.15   # with no weather data
    height_m: float = 0.0
    deflection_model_arcsec: Optional[tuple] = None   # (xi north, eta east)
    # ---- estimator ---------------------------------------------------------------
    initial_position: Optional[tuple] = None  # (lat_deg, lon_deg) guess
    estimate_calibration: bool = False
    estimate_clock: bool = False
    min_stars: int = 3
    min_star_altitude_deg: float = 3.0
    outlier_false_alarm: float = 1e-3
    max_rejected_star_fraction: float = 0.25
    attitude_sigma_limit_arcsec: float = 120.0
    gravity_tolerance: float = 0.02           # m/s^2 on |f| - normal gravity
    max_iterations: int = 10
    convergence_m: float = 0.01               # last position update


# ------------------------------------------------------------------- output
@dataclass
class NavigationResult:
    """
    The standard output of the navigation engine.

    latitude_deg / longitude_deg / covariance_ne_m2 are None unless status
    is VALID or DEGRADED. Longitude alone is None where it is not supported
    (at a pole). A position that was computed but not accepted is kept only
    in diagnostics["withheld_position"], for analysis.

    Four things are reported separately:
      precision             random uncertainty under the declared noise model
      accuracy_limitations  declared systematic and reference-model terms
      integrity             which checks ran and what they found
      observability         whether the measurements constrain the state

    VALID is conditional on `assumptions`, which the measurements cannot
    verify (in particular: that the instrument was not accelerating).
    """
    latitude_deg: Optional[float]
    longitude_deg: Optional[float]

    covariance_ne_m2: Optional[np.ndarray]
    status: str

    observation_time_utc: str
    observations_used: int
    stars_used: int

    residual_rms_arcsec: Optional[float]
    calibration_status: str
    reference_status: str

    diagnostics: dict = field(default_factory=dict)
    assumptions: list = field(default_factory=list)

    reasons: list = field(default_factory=list)
    precision: dict = field(default_factory=dict)
    accuracy_limitations: dict = field(default_factory=dict)
    integrity: dict = field(default_factory=dict)
    observability: dict = field(default_factory=dict)

    @property
    def has_fix(self):
        return self.status in FIX_STATUSES and self.latitude_deg is not None

    @property
    def horizontal_sigma_m(self):
        """sqrt(trace) of the reported covariance: one-sigma radial error."""
        if self.covariance_ne_m2 is None:
            return None
        return float(np.sqrt(np.trace(self.covariance_ne_m2)))

    def summary(self):
        if self.latitude_deg is None:
            position = "no position reported"
        elif self.longitude_deg is None:
            position = f"lat {self.latitude_deg:+.6f} deg, longitude undefined"
        else:
            position = (f"lat {self.latitude_deg:+.6f} deg, "
                        f"lon {self.longitude_deg:+.6f} deg")
        sigma = self.horizontal_sigma_m
        lines = [
            f"{self.status}: {position}"
            + ("" if sigma is None else f"  (1-sigma {sigma:,.1f} m)"),
            f"  time {self.observation_time_utc}; {self.observations_used} "
            f"observations, {self.stars_used} stars; calibration "
            f"{self.calibration_status}; reference {self.reference_status}",
        ]
        lines += [f"  - {reason}" for reason in self.reasons]
        return "\n".join(lines)
