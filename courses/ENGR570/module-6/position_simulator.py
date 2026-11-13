"""
position_simulator.py - Space Systems Engineering, Module 6

Synthetic star-camera and accelerometer measurements for an observer at a
known place and time. Everything is derived from one physical geometry:

    position (lat, lon) -> ellipsoid normal u_E -> u_G = C_EG(t)^T u_E
    camera attitude chosen relative to the local east/north/up axes
    stars and specific force generated from that same attitude

so the accelerometer's gravity direction is never chosen independently of
the camera attitude or the observer's location.

The true latitude and longitude are used here and in final validation only.
The estimator in celestial_position.py never sees them.
"""

from dataclasses import dataclass, field

import numpy as np
from astropy.time import Time

import celestial_position as cp  # noqa: F401  (sets up sys.path first)
from accelerometer_calibration import IDEAL_SENSOR
from rotation_engine import rotation_vector_to_matrix
from sensor_frames import PinholeCamera, Transform
from star_field_simulation import (
    centroid_sigma_to_angle,
    observe_with_centroid_noise,
    stars_in_field,
)

OBSERVATION_UTC = "2026-11-11T12:00:00"
STANDARD_GRAVITY = 9.81                      # m/s^2 at the ellipsoid
EARTH_RADIUS_M = 6371008.8                   # mean radius, for scaling only

# A fixed, known, non-trivial camera-from-IMU mounting.
DEFAULT_C_CI = Transform("C", "I", rotation_vector_to_matrix([0.02, -0.01, 0.03]))


def true_vertical_ecef(latitude_deg, longitude_deg):
    """WGS84 ellipsoid normal: [cos lat cos lon, cos lat sin lon, sin lat]."""
    lat, lon = np.radians(latitude_deg), np.radians(longitude_deg)
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon),
                     np.sin(lat)])


def local_axes_ecef(latitude_deg, longitude_deg):
    """
    C_EN: columns are the local east, north and up unit vectors in
    Earth-fixed coordinates. At a pole east and north are defined by the
    supplied longitude.
    """
    lat, lon = np.radians(latitude_deg), np.radians(longitude_deg)
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon),
                      np.cos(lat)])
    return np.column_stack((east, north, true_vertical_ecef(latitude_deg,
                                                            longitude_deg)))


def camera_axes_local(zenith_angle_deg, azimuth_deg=0.0, roll_deg=0.0):
    """
    C_NC: columns are the camera x, y, z axes in local east/north/up.

    The boresight (+z) is zenith_angle_deg from the vertical toward
    azimuth_deg (clockwise from north). With roll = 0 the image +y axis
    (down) points as nearly toward the ground as the boresight allows.
    """
    z, az = np.radians(zenith_angle_deg), np.radians(azimuth_deg)
    boresight = np.array([np.sin(z) * np.sin(az), np.sin(z) * np.cos(az),
                          np.cos(z)])
    up = np.array([0.0, 0.0, 1.0])
    down_in_image = -(up - (up @ boresight) * boresight)
    if np.linalg.norm(down_in_image) < 1e-9:      # looking straight up
        down_in_image = -np.array([np.sin(az), np.cos(az), 0.0])
    y = down_in_image / np.linalg.norm(down_in_image)
    x = np.cross(y, boresight)
    axes = np.column_stack((x, y, boresight))
    return axes @ rotation_vector_to_matrix([0.0, 0.0, np.radians(roll_deg)])


def gravity_magnitude(height_m, g0=STANDARD_GRAVITY):
    """Free-air decrease of gravity with height: g0 (1 - 2h/R)."""
    return g0 * (1.0 - 2.0 * height_m / EARTH_RADIUS_M)


@dataclass
class Observation:
    """Measurements handed to the estimator, plus the truth for validation."""
    reference_stars: np.ndarray        # (N, 3) celestial frame
    measured_stars: np.ndarray         # (N, 3) camera frame
    star_sigma_rad: np.ndarray         # (N,) assumed angular noise per star
    accel_samples: np.ndarray          # (M, 3) raw specific force, IMU axes
    C_CI: Transform                    # true camera-from-IMU calibration
    utc: str                           # true observation time
    truth: dict = field(default_factory=dict)


def simulate_observation(latitude_deg, longitude_deg, utc=OBSERVATION_UTC, *,
                         height_m=0.0, zenith_angle_deg=30.0, azimuth_deg=40.0,
                         roll_deg=15.0, C_CG=None, C_CI=DEFAULT_C_CI,
                         camera=None, n_stars=20, centroid_noise_px=0.0,
                         accel_model=IDEAL_SENSOR, n_accel_samples=100,
                         acceleration_enu=(0.0, 0.0, 0.0), rng=None,
                         C_EG=None):
    """
    Simulate one stationary (or accelerating) observation.

    The camera attitude is set relative to the local vertical by
    zenith_angle_deg / azimuth_deg / roll_deg, unless a celestial attitude
    C_CG (celestial-to-camera, 3x3) is supplied directly.

    acceleration_enu is the instrument's true acceleration in local
    east/north/up [m/s^2]; the accelerometer then senses
    f = a - g, i.e. C_IN (a_N + [0, 0, g]).

    C_EG may be passed to avoid recomputing Earth orientation in loops; it
    must be the matrix for `utc`.
    """
    rng = np.random.default_rng() if rng is None else rng
    camera = PinholeCamera() if camera is None else camera
    if C_EG is None:
        C_EG, _ = cp.earth_orientation(utc)
    c_eg = np.asarray(C_EG.matrix if isinstance(C_EG, Transform) else C_EG)

    # --- geometry: one consistent chain from position to every sensor -------
    up_ecef = true_vertical_ecef(latitude_deg, longitude_deg)
    c_en = local_axes_ecef(latitude_deg, longitude_deg)       # N -> E
    c_gn = c_eg.T @ c_en                                      # N -> G
    up_celestial = c_gn[:, 2]

    if C_CG is None:
        c_gc = c_gn @ camera_axes_local(zenith_angle_deg, azimuth_deg, roll_deg)
        c_cg = c_gc.T
    else:
        c_cg = np.asarray(C_CG, dtype=float)
        c_gc = c_cg.T
    c_ic = np.asarray(C_CI.matrix).T
    c_in = c_ic @ c_cg @ c_gn                                 # N -> I

    # --- star camera -----------------------------------------------------------
    camera_vectors = stars_in_field(camera, n_stars, rng)
    reference = camera_vectors @ c_gc.T                       # to celestial
    measured = observe_with_centroid_noise(camera, camera_vectors,
                                           centroid_noise_px, rng)
    sigma_px = centroid_noise_px if centroid_noise_px > 0 else 1.0
    star_sigma = centroid_sigma_to_angle(camera, camera_vectors, sigma_px)

    # --- accelerometer -----------------------------------------------------------
    g = gravity_magnitude(height_m)
    specific_force_n = np.asarray(acceleration_enu, dtype=float) + [0.0, 0.0, g]
    ideal = c_in @ specific_force_n
    accel = accel_model.measure(np.tile(ideal, (n_accel_samples, 1)), rng)

    truth = {
        "latitude_deg": float(latitude_deg),
        "longitude_deg": float(longitude_deg),
        "height_m": float(height_m),
        "up_ecef": up_ecef,
        "up_celestial": up_celestial,
        "up_camera": c_cg @ up_celestial,
        "up_imu": c_in[:, 2],
        "C_CG": c_cg,
        "C_EG": c_eg,
        "C_IN": c_in,
        "specific_force_imu": ideal,
        "gravity": g,
        "camera_vectors": camera_vectors,
    }
    time = utc if isinstance(utc, str) else Time(utc).utc.isot
    return Observation(reference_stars=reference, measured_stars=measured,
                       star_sigma_rad=star_sigma, accel_samples=accel,
                       C_CI=C_CI, utc=time, truth=truth)


def solve_observation(observation, *, assumed_utc=None, C_CI=None,
                      accel_bias=None, accel_correction=None,
                      accel_noise_sigma=None, earth=None):
    """
    Run the estimator on an Observation. Only measurements, the calibration
    the software believes (C_CI) and the time it believes (assumed_utc) are
    passed on; observation.truth is not touched.
    """
    return cp.solve_from_measurements(
        observation.reference_stars, observation.measured_stars,
        observation.accel_samples,
        observation.C_CI if C_CI is None else C_CI,
        observation.utc if assumed_utc is None else assumed_utc,
        star_sigma_rad=observation.star_sigma_rad,
        accel_bias=accel_bias, accel_correction=accel_correction,
        accel_noise_sigma=accel_noise_sigma, earth=earth)
