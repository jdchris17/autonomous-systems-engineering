"""
reference_frames.py - Space Systems Engineering, Module 2, Experiment B

Rotation about the pole, an idealized inertial -> Earth-fixed transformation,
and WGS84 geodetic -> Earth-centred Earth-fixed (ECEF) coordinates.

Conventions
-----------
* Right-handed frames, +z along the rotation axis / north pole.
* Earth-fixed +x pierces the equator at the Greenwich meridian (lon = 0),
  +y at lon = 90 deg East.
* rotation_z() is ACTIVE: it turns a vector counter-clockwise (seen from +z)
  within a fixed frame. Rotating the *frame* by +theta (a passive
  transformation) is therefore rotation_z(-theta), the R3(theta) of the
  astronomical literature and erfa.rz.
"""

import numpy as np

# WGS84 defining parameters.
WGS84_A = 6378137.0                    # semi-major axis [m]
WGS84_F = 1.0 / 298.257223563          # flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)    # semi-minor axis [m]
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)   # first eccentricity squared


def rotation_z(theta_rad):
    """
    Return an active right-handed rotation
    matrix about the positive z axis.

    rotation_z(theta) @ v turns v by +theta from +x toward +y.
    """
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def celestial_to_earth_fixed(
    celestial_vector,
    earth_angle_rad
):
    """
    Transform a direction from an idealized
    inertial frame into a rotating Earth frame.

    Passive transformation  v_E = R3(-theta) v_I,  where R3 is the active
    rotation_z above and theta is the angle from the inertial +x axis to the
    Earth-fixed +x axis (Greenwich), measured eastward. A star whose right
    ascension equals theta lands on the Greenwich meridian.

    IDEALIZED MODEL. This is a single rotation about a fixed z axis. It is
    not the ICRS -> ITRS transformation: there is no frame bias, precession,
    nutation or polar motion, and no aberration or light deflection. How
    wrong that is depends on what angle is supplied; see
    compare_reference_frames.py.

    Accepts one 3-vector or an (..., 3) array of them.
    """
    v = np.asarray(celestial_vector, dtype=float)
    if v.shape[-1] != 3:
        raise ValueError(f"expected (..., 3) vectors, got shape {v.shape}")
    # v @ R.T applies R to every row vector.
    return v @ rotation_z(-earth_angle_rad).T


def geodetic_to_ecef(
    latitude_deg,
    longitude_deg,
    height_m
):
    """
    Convert WGS84 geodetic coordinates
    into Earth-fixed Cartesian coordinates.

    Latitude is geodetic (angle of the ellipsoid normal), longitude is
    positive east, height is above the ellipsoid. Returns [x, y, z] in
    meters.

        N = a / sqrt(1 - e^2 sin^2(lat))       prime-vertical radius
        x = (N + h) cos(lat) cos(lon)
        y = (N + h) cos(lat) sin(lon)
        z = (N (1 - e^2) + h) sin(lat)
    """
    if not -90.0 <= latitude_deg <= 90.0:
        raise ValueError(f"latitude must be in [-90, 90] deg, got {latitude_deg}")

    lat = np.radians(latitude_deg)
    lon = np.radians(longitude_deg)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)

    prime_vertical = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)

    return np.array([
        (prime_vertical + height_m) * cos_lat * np.cos(lon),
        (prime_vertical + height_m) * cos_lat * np.sin(lon),
        (prime_vertical * (1.0 - WGS84_E2) + height_m) * sin_lat,
    ])
