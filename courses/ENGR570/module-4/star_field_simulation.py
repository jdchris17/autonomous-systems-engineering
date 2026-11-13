"""
star_field_simulation.py - Space Systems Engineering, Module 4

Shared synthetic-observation helpers for the Module 4 experiments. Built on
the Module 3 camera and frame code; nothing here estimates attitude.

Limitations (as stated in the assignment): star positions are uniformly
random over the sensor. There is no real star density, magnitude limit,
aperture, signal-to-noise model or lens distortion.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "module-3"))
from sensor_frames import (  # noqa: E402,F401  (re-exported for experiments)
    PinholeCamera,
    attitude_to_transform,
    camera_attitude_from_boresight,
)

ARCSEC_PER_RAD = np.degrees(1.0) * 3600.0

# One true attitude used by every experiment: boresight RA 120, Dec 30,
# rolled 25 degrees so that no camera axis lines up with a celestial axis.
TRUE_ATTITUDE = camera_attitude_from_boresight(120.0, 30.0, roll_deg=25.0)


def true_attitude_matrix(q_attitude=TRUE_ATTITUDE):
    """C_CG (celestial -> camera) for an attitude quaternion."""
    return np.array(attitude_to_transform(q_attitude).inverse().matrix)


def unit_square_positions(n_stars, rng):
    """Random star positions as fractions of the sensor, each in (0, 1)."""
    return rng.uniform(0.02, 0.98, size=(n_stars, 2))


def camera_vectors_from_positions(camera, positions):
    """Camera-frame unit vectors for fractional sensor positions (N, 2)."""
    pixels = np.asarray(positions) * [camera.width, camera.height]
    return camera.unproject(pixels)


def stars_in_field(camera, n_stars, rng):
    """n random star directions inside the field of view, camera frame."""
    return camera_vectors_from_positions(camera,
                                         unit_square_positions(n_stars, rng))


def to_celestial(camera_vectors, q_attitude=TRUE_ATTITUDE):
    """True catalog directions (celestial frame) of camera-frame stars."""
    return attitude_to_transform(q_attitude).apply(camera_vectors)


def observe_with_centroid_noise(camera, camera_vectors, sigma_px, rng):
    """
    Measured camera-frame unit vectors after adding independent Gaussian
    noise of sigma_px to each star's u and v image coordinates.
    """
    pixels, _ = camera.project(camera_vectors)
    if sigma_px > 0.0:
        pixels = pixels + rng.normal(0.0, sigma_px, size=pixels.shape)
    return camera.unproject(pixels)


def observe_with_angular_noise(camera_vectors, sigma_rad, rng):
    """
    Measured unit vectors with isotropic angular noise: each direction is
    displaced by independent Gaussian angles of sigma_rad along two axes
    perpendicular to it.
    """
    v = np.asarray(camera_vectors, dtype=float)
    helper = np.where(np.abs(v[:, [0]]) < 0.9, [[1.0, 0.0, 0.0]],
                      [[0.0, 1.0, 0.0]])
    e1 = np.cross(v, helper)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(v, e1)
    noise = rng.normal(0.0, sigma_rad, size=(len(v), 2))
    noisy = v + noise[:, [0]] * e1 + noise[:, [1]] * e2
    return noisy / np.linalg.norm(noisy, axis=1, keepdims=True)


def centroid_sigma_to_angle(camera, camera_vectors, sigma_px):
    """
    Per-star angular noise [rad] equivalent to sigma_px of centroid noise.

    At an angle t off the boresight a pixel subtends cos^2(t)/f radially and
    cos(t)/f tangentially, so the noise is slightly anisotropic. The
    isotropic value returned is the geometric mean, sigma_px cos^1.5(t) / f.
    On the boresight it is simply sigma_px / f.
    """
    v = np.asarray(camera_vectors, dtype=float)
    cos_off_axis = v[:, 2] / np.linalg.norm(v, axis=1)
    return sigma_px * cos_off_axis**1.5 / camera.fx
