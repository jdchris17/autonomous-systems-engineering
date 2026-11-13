"""
celestial_position.py - Space Systems Engineering, Module 6

Terrestrial position from the stars and the local vertical.

    u_I  (IMU: measured upward direction)
     |  C_CI   camera-from-IMU calibration            (Module 3 / 5)
    u_C
     |  C_CG^T celestial-from-camera, from the stars   (Module 4)
    u_G
     |  C_EG   Earth-fixed-from-celestial at time t    (Module 2)
    u_E  -> geodetic latitude, east-positive longitude

Nothing here re-implements attitude estimation, accelerometer calibration,
astronomical time or the Transform class; they are imported from Modules
2 to 5.

Model assumptions (also attached to every solution)
---------------------------------------------------
* Local up is the WGS84 ellipsoid normal. Deflection of the vertical
  (real gravity is not exactly normal to the ellipsoid) is not modelled.
* The instrument is unaccelerated, so averaged specific force points up.
* Star directions are GCRS-oriented unit vectors with no parallax; stellar
  aberration and atmospheric refraction are not modelled.
* C_EG is the full IAU 2006/2000A GCRS -> ITRS rotation with IERS UT1-UTC
  and polar motion at the supplied UTC timestamp.
* Height is not observable from a direction and is not estimated.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from geographiclib.geodesic import Geodesic

_COURSE = Path(__file__).resolve().parents[1]
for _module in ("module-1", "module-2", "module-3", "module-4", "module-5"):
    sys.path.insert(0, str(_COURSE / _module))

from astropy.time import Time  # noqa: E402

from astronomical_time import iers_status, time_scales  # noqa: E402
from compare_reference_frames import reference_matrix  # noqa: E402
from gravity_estimator import (  # noqa: E402
    assess_measurement_quality,
    calibrate_accelerometer,
    estimate_up_direction,
)
from rotation_engine import check_rotation_matrix  # noqa: E402
from sensor_frames import Transform  # noqa: E402
from stellar_attitude import angular_residuals, wahba_svd  # noqa: E402

# If the vertical is within this (as a sine) of the rotation axis, the
# observer is treated as exactly at a pole and longitude is not reported.
# 1e-12 corresponds to about 6 micrometres on the ground.
POLE_TOLERANCE = 1e-12

MODEL_ASSUMPTIONS = (
    "local up = WGS84 ellipsoid normal (no deflection of the vertical)",
    "instrument unaccelerated: mean specific force is taken as up",
    "stars are GCRS-oriented directions: no parallax, aberration or refraction",
    "Earth orientation: IAU 2006/2000A GCRS->ITRS with IERS UT1-UTC and polar motion",
    "height is not observable from a direction and is not estimated",
)


# --------------------------------------------------------------------- helpers
def _dcm(transform, to_frame, from_frame):
    """Accept a Module 3 Transform (frames checked) or a bare 3x3 rotation."""
    if isinstance(transform, Transform):
        if (transform.to_frame, transform.from_frame) != (to_frame, from_frame):
            raise ValueError(f"expected C_{to_frame}{from_frame}, "
                             f"got {transform.name}")
        return np.asarray(transform.matrix)
    return check_rotation_matrix(transform)


def _unit(vector, name):
    v = np.asarray(vector, dtype=float)
    if v.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {v.shape}")
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{name} contains NaN or infinite values")
    norm = np.linalg.norm(v)
    if norm == 0.0:
        raise ValueError(f"{name} is the zero vector")
    return v / norm


def wrap_longitude(longitude_deg):
    """Wrap to [-180, 180). 179.999 and -180.001 map 0.002 deg apart."""
    return (np.asarray(longitude_deg, dtype=float) + 180.0) % 360.0 - 180.0


def longitude_difference(lon_a_deg, lon_b_deg):
    """Signed a - b in degrees, taking the short way round the date line."""
    return float(wrap_longitude(lon_a_deg - lon_b_deg))


# --------------------------------------------------------- required interface
def earth_fixed_up_to_geodetic(up_ecef):
    """
    Convert a WGS84 ellipsoid-normal unit vector
    into geodetic latitude and east-positive longitude.

    Returns degrees.
    Assumes local up equals the ellipsoid normal.

    The ellipsoid normal at geodetic latitude phi, longitude lam is exactly
    [cos phi cos lam, cos phi sin lam, sin phi], so no iteration and no
    ellipsoid parameters are needed: latitude = atan2(z, hypot(x, y)) and
    longitude = atan2(y, x). (This is geodetic latitude BECAUSE the input is
    the normal; the same formula applied to a position vector would give
    geocentric latitude.)

    Longitude is returned in [-180, 180). At a pole the normal is along the
    rotation axis and longitude is undefined: NaN is returned rather than an
    arbitrary number.
    """
    x, y, z = _unit(up_ecef, "up_ecef")
    horizontal = np.hypot(x, y)
    latitude = float(np.degrees(np.arctan2(z, horizontal)))
    if horizontal < POLE_TOLERANCE:
        return latitude, float("nan")
    return latitude, float(wrap_longitude(np.degrees(np.arctan2(y, x))))


def celestial_vertical_to_position(
    up_celestial,
    celestial_to_earth_matrix
):
    """
    Transform the celestial upward vector into ITRS
    and return estimated geodetic latitude/longitude.

    celestial_to_earth_matrix is C_EG (a Transform or a 3x3 rotation).
    Returns (latitude_deg, longitude_deg).
    """
    up_ecef = _dcm(celestial_to_earth_matrix, "E", "G") @ _unit(up_celestial,
                                                               "up_celestial")
    return earth_fixed_up_to_geodetic(up_ecef)


@dataclass
class PositionSolution:
    """Result of solve_terrestrial_position(). Angles in degrees."""
    latitude_deg: float
    longitude_deg: float                # NaN when not observable (at a pole)
    up_ecef: np.ndarray                 # Earth-fixed upward unit vector
    up_celestial: np.ndarray
    up_camera: np.ndarray
    status: str                         # "ok" or a description of the problem
    valid: bool                         # False if the solution must not be used
    longitude_observable: bool
    height_m: None = None               # never estimated
    height_estimated: bool = False
    timestamp_utc: str = None           # time used for Earth orientation
    earth_orientation_source: str = None
    assumptions: tuple = MODEL_ASSUMPTIONS
    quality: object = None              # Module 5 QualityReport, if available
    diagnostics: dict = field(default_factory=dict)
    # Reserved for Module 7: 2x2 covariance of (north, east) position [m^2].
    position_covariance: object = None

    @property
    def lat_lon(self):
        return self.latitude_deg, self.longitude_deg


def solve_terrestrial_position(
    up_imu,
    C_CI,
    C_CG,
    C_EG,
    *,
    timestamp_utc=None,
    earth_orientation_source=None,
    quality=None,
    diagnostics=None,
):
    """
    Combine IMU gravity, extrinsic calibration,
    stellar attitude, and Earth orientation.

    C_CG is celestial-to-camera.
    Return terrestrial position and diagnostics.

        u_E = C_EG C_CG^T C_CI u_I

    up_imu: upward direction in IMU axes. C_CI: camera-from-IMU. C_CG:
    celestial-to-camera (Module 4's output). C_EG: celestial-to-Earth-fixed.
    Each matrix may be a Module 3 Transform (its frames are then checked)
    or a bare 3x3 rotation.

    Returns a PositionSolution. Height is never estimated. At a pole the
    longitude is NaN and longitude_observable is False; near a pole the
    solution is valid but diagnostics["longitude_dilution"] = 1/cos(lat)
    shows how strongly angular error is magnified in longitude.
    """
    up_i = _unit(up_imu, "up_imu")
    up_camera = _dcm(C_CI, "C", "I") @ up_i
    up_celestial = _dcm(C_CG, "C", "G").T @ up_camera
    up_ecef = _dcm(C_EG, "E", "G") @ up_celestial
    up_ecef = up_ecef / np.linalg.norm(up_ecef)

    latitude, longitude = earth_fixed_up_to_geodetic(up_ecef)
    observable = not np.isnan(longitude)
    cos_lat = float(np.hypot(up_ecef[0], up_ecef[1]))

    info = {
        "longitude_dilution": (1.0 / cos_lat) if observable else float("inf"),
        "angle_from_pole_deg": float(np.degrees(np.arctan2(cos_lat,
                                                           abs(up_ecef[2])))),
    }
    if diagnostics:
        info.update(diagnostics)

    status, valid = "ok", True
    if not observable:
        status = ("at a pole: latitude is valid, longitude is not observable")
    if quality is not None and quality.detectable_disturbance:
        status = (f"accelerometer quality check failed ({quality.verdict}): "
                  "the measured vertical is unreliable")
        valid = False

    return PositionSolution(
        latitude_deg=latitude, longitude_deg=longitude, up_ecef=up_ecef,
        up_celestial=up_celestial, up_camera=up_camera, status=status,
        valid=valid, longitude_observable=observable,
        timestamp_utc=timestamp_utc,
        earth_orientation_source=earth_orientation_source, quality=quality,
        diagnostics=info)


def horizontal_position_error(
    estimated_lat_lon,
    true_lat_lon
):
    """
    Compute horizontal geodesic error in meters
    on the WGS84 ellipsoid.

    Uses GeographicLib's inverse geodesic solution (Karney), not a
    spherical approximation. A NaN longitude is accepted for a point at a
    pole, where every longitude is the same place.
    """
    return _geodesic(estimated_lat_lon, true_lat_lon)["s12"]


def north_east_error(estimated_lat_lon, true_lat_lon):
    """
    (north_m, east_m): the estimate's displacement from the truth, resolved
    along the geodesic's departure direction at the true position.
    """
    line = _geodesic(estimated_lat_lon, true_lat_lon)
    azimuth = np.radians(line["azi1"])
    return line["s12"] * np.cos(azimuth), line["s12"] * np.sin(azimuth)


def _geodesic(estimated_lat_lon, true_lat_lon):
    def clean(point, name):
        lat, lon = (float(v) for v in point)
        if not -90.0 <= lat <= 90.0:
            raise ValueError(f"{name} latitude {lat} is outside [-90, 90]")
        if np.isnan(lon):
            if abs(lat) < 90.0 - 1e-9:
                raise ValueError(f"{name} longitude is NaN away from a pole")
            lon = 0.0
        return lat, lon

    est_lat, est_lon = clean(estimated_lat_lon, "estimated")
    true_lat, true_lon = clean(true_lat_lon, "true")
    # From the truth to the estimate, so azi1 is the error direction there.
    return Geodesic.WGS84.Inverse(true_lat, true_lon, est_lat, est_lon)


# ----------------------------------------------------------- Earth orientation
def earth_orientation(utc):
    """
    C_EG at a UTC timestamp, from Module 2's complete reference
    transformation, plus where its Earth-orientation data came from.

    Returns (Transform C_EG, info dict). `utc` is an ISO string or an
    astropy Time.
    """
    time = utc if isinstance(utc, Time) else Time(utc, scale="utc")
    status = iers_status(time)
    scales = time_scales(time)
    info = {
        "timestamp_utc": time.utc.isot,
        "earth_orientation_source": status["source"],
        "ut1_minus_utc_s": scales.dut1_seconds,
        "earth_rotation_angle_rad": scales.era_rad,
    }
    return Transform("E", "G", reference_matrix(time)), info


# ------------------------------------------------------- measurement pipeline
def solve_from_measurements(reference_stars, measured_stars, accel_samples,
                            C_CI, utc, *, star_sigma_rad=None,
                            accel_bias=None, accel_correction=None,
                            accel_noise_sigma=None, earth=None):
    """
    The complete navigation pipeline, from raw measurements to position.

      1. camera attitude C_CG from matched stars (Module 4 SVD);
      2. IMU-frame up from accelerometer samples (Module 5), after applying
         the supplied accelerometer calibration, if any;
      3. camera-IMU calibration C_CI;
      4. Earth orientation at `utc` (Module 2);
      5. inversion to latitude and longitude.

    reference_stars: (N, 3) catalog directions, celestial frame.
    measured_stars: (N, 3) the same stars in camera axes.
    accel_samples: (M, 3) raw specific force in IMU axes [m/s^2].
    star_sigma_rad: per-star angular noise, used as 1/sigma^2 weights.
    accel_noise_sigma: if given, Module 5's quality indicator is run.

    earth: optionally the (C_EG, info) pair already returned by
    earth_orientation(utc), to avoid recomputing it inside Monte Carlo
    loops. It must be the one for `utc`.

    No true position enters this function.
    """
    weights = None if star_sigma_rad is None else 1.0 / np.asarray(star_sigma_rad) ** 2
    C_CG = wahba_svd(reference_stars, measured_stars, weights)
    star_residuals = angular_residuals(C_CG, reference_stars, measured_stars)

    samples = np.asarray(accel_samples, dtype=float)
    if accel_bias is not None or accel_correction is not None:
        samples = calibrate_accelerometer(
            samples,
            np.zeros(3) if accel_bias is None else accel_bias,
            np.eye(3) if accel_correction is None else accel_correction)
    up_imu, up_uncertainty, statistics = estimate_up_direction(samples)

    quality = None
    if accel_noise_sigma is not None:
        quality = assess_measurement_quality(samples, accel_noise_sigma)

    C_EG, orientation_info = earth_orientation(utc) if earth is None else earth
    return solve_terrestrial_position(
        up_imu, C_CI, C_CG, C_EG,
        timestamp_utc=orientation_info["timestamp_utc"],
        earth_orientation_source=orientation_info["earth_orientation_source"],
        quality=quality,
        diagnostics={
            "star_count": int(len(star_residuals)),
            "star_residual_rms_rad": float(np.sqrt(np.mean(star_residuals**2))),
            "accel_sample_count": statistics.count,
            "specific_force_magnitude": statistics.magnitude,
            "up_imu_angular_rms_rad": up_uncertainty.angular_rms,
            "ut1_minus_utc_s": orientation_info["ut1_minus_utc_s"],
            "C_CG": C_CG,
        })
