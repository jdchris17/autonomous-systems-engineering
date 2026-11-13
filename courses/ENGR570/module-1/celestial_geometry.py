"""
celestial_geometry.py - Space Systems Engineering, Module 1

A small, reusable geometry library for directions on the celestial sphere.

Conventions
-----------
* Frame: right-handed equatorial frame.
    +x -> RA = 0 deg,  Dec = 0 deg  (vernal equinox direction)
    +y -> RA = 90 deg, Dec = 0 deg
    +z -> Dec = +90 deg             (north celestial pole)
* Public angle arguments/returns are in degrees, except
  earth_rotation_angle(), which returns radians as specified.
* Right ascension is returned in [0, 360). Declination in [-90, +90].
* Pole convention: at the celestial poles RA is geometrically undefined;
  unit_to_radec() returns RA = 0.0 there.
* Hour angle is positive westward (H = 0 on the upper meridian, increasing
  with time).
"""

import numpy as np

# Earth's sidereal rotation period (one rotation relative to the stars).
SIDEREAL_DAY_SECONDS = 86164.0905

# Idealized, constant sidereal rotation rate [rad/s].
EARTH_ROTATION_RATE = 2.0 * np.pi / SIDEREAL_DAY_SECONDS

# If the equatorial-plane projection of a unit vector is shorter than this,
# the direction is treated as a celestial pole (RA undefined -> 0).
POLE_TOLERANCE = 1e-12

# How far |v| may deviate from 1 before radec_to_unit's self-check fails.
UNIT_TOLERANCE = 1e-12


def radec_to_unit(ra_deg, dec_deg):
    """
    Convert equatorial right ascension and
    declination, in degrees, into a 3D unit vector.

    Returns a NumPy array [x, y, z] with
        x = cos(dec) cos(ra)
        y = cos(dec) sin(ra)
        z = sin(dec)
    """
    if not -90.0 <= dec_deg <= 90.0:
        raise ValueError(f"declination must be in [-90, 90] deg, got {dec_deg}")

    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)

    cos_dec = np.cos(dec)
    vector = np.array([
        cos_dec * np.cos(ra),
        cos_dec * np.sin(ra),
        np.sin(dec),
    ])

    # Confirm unit magnitude (guards against future edits breaking the formula).
    norm = np.linalg.norm(vector)
    if abs(norm - 1.0) > UNIT_TOLERANCE:
        raise ArithmeticError(f"result is not a unit vector (|v| = {norm})")

    return vector


def unit_to_radec(vector):
    """
    Convert a 3D celestial direction vector
    into right ascension and declination.

    Returns (ra_deg, dec_deg) with RA wrapped into [0, 360) and Dec in
    [-90, +90]. The input is normalized first, so any non-zero direction
    vector is accepted.

    Pole convention: when the vector points at a celestial pole (its
    projection onto the equatorial plane is below POLE_TOLERANCE), RA is
    geometrically undefined and is returned as exactly 0.0.
    """
    v = np.asarray(vector, dtype=float)
    if v.shape != (3,):
        raise ValueError(f"expected a 3-vector, got shape {v.shape}")

    norm = np.linalg.norm(v)
    if norm == 0.0:
        raise ValueError("cannot convert the zero vector to a direction")
    x, y, z = v / norm

    # atan2(z, rho) is well conditioned everywhere, unlike asin(z) near the
    # poles where d(asin)/dz blows up.
    rho = np.hypot(x, y)
    dec_deg = np.degrees(np.arctan2(z, rho))

    if rho < POLE_TOLERANCE:
        return 0.0, float(dec_deg)

    # atan2 returns (-180, 180]; wrap into [0, 360).
    ra_deg = np.degrees(np.arctan2(y, x)) % 360.0
    if ra_deg >= 360.0:  # a tiny negative angle can round up to 360.0
        ra_deg = 0.0

    return float(ra_deg), float(dec_deg)


def angular_separation(vec1, vec2):
    """
    Return the angular separation between
    two celestial unit vectors in degrees.

    Uses the atan2 formulation
        theta = atan2(|v1 x v2|, v1 . v2)
    which stays accurate for very small and near-180 deg separations, where
    acos(v1 . v2) loses precision. The result is in [0, 180] and does not
    depend on the magnitudes of the inputs.
    """
    a = np.asarray(vec1, dtype=float)
    b = np.asarray(vec2, dtype=float)
    if a.shape != (3,) or b.shape != (3,):
        raise ValueError("both inputs must be 3-vectors")
    if not a.any() or not b.any():
        raise ValueError("angular separation is undefined for a zero vector")

    cross_norm = np.linalg.norm(np.cross(a, b))
    dot = np.dot(a, b)
    return float(np.degrees(np.arctan2(cross_norm, dot)))


def earth_rotation_angle(elapsed_seconds, initial_angle_rad=0.0):
    """
    Return the accumulated idealized sidereal
    rotation angle in radians.

    theta(t) = initial_angle_rad + omega * elapsed_seconds,
    with omega = 2*pi / SIDEREAL_DAY_SECONDS.

    This is rotation accumulated from a defined initial angle at t = 0. It
    is NOT Greenwich sidereal time and knows nothing about absolute UTC
    epochs, precession, nutation, or variations in Earth's rotation rate.
    The angle is deliberately left unwrapped (it grows past 2*pi) so the
    number of completed rotations is preserved; wrap at the call site if
    needed.
    """
    return initial_angle_rad + EARTH_ROTATION_RATE * elapsed_seconds


def star_altitude(
    latitude_deg,
    declination_deg,
    hour_angle_deg
):
    """
    Compute geometric stellar altitude
    for a spherical Earth observer.

    sin(h) = sin(lat) sin(dec) + cos(lat) cos(dec) cos(H)

    Returns altitude in degrees, in [-90, +90]. Negative means below the
    geometric horizon. No refraction, parallax, or horizon dip is modelled.
    Hour angle is positive westward; H = 0 is upper meridian transit.
    """
    lat = np.radians(latitude_deg)
    dec = np.radians(declination_deg)
    hour_angle = np.radians(hour_angle_deg)

    sin_alt = (np.sin(lat) * np.sin(dec)
               + np.cos(lat) * np.cos(dec) * np.cos(hour_angle))

    # Round-off can push sin_alt a hair outside [-1, 1] (e.g. at the zenith).
    sin_alt = np.clip(sin_alt, -1.0, 1.0)
    return np.degrees(np.arcsin(sin_alt))
