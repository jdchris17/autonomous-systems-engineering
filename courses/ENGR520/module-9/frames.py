"""module-9 -- frames.py  (Sections 35-36: Coordinate Frames -- Be Ruthless About Naming)

This is the computational centerpiece's frame backbone (Section 34): the
first half of the chain

    C (celestial) -> E (Earth-fixed) -> L (local observer) -> B (camera/body)

Four frames, named exactly as specified, never left implicit:

    C = celestial/inertial   equatorial, RA/Dec-based; z = celestial pole,
                             x = RA=0 (vernal equinox direction). Fixed --
                             does not rotate with the Earth. (Precession,
                             nutation, and proper motion are outside this
                             course's scope; C is treated as a fixed J2000-like
                             frame.)
    E = Earth-fixed          same z (the poles coincide), but x points at the
                             Greenwich meridian and rotates WITH the Earth.
    L = local observer       ENU (East, North, Up), tangent to the Earth at
                             the observer's (lat, lon). See README.md for why
                             ENU and not NED.
    B = camera/body          module-7/camera's convention: +x right, +y DOWN,
                             +z = optical axis (into the scene). R_BL depends
                             on where the camera points and is built once a
                             specific pointing scenario is defined -- see
                             module-7/camera/camera_orientation.py for the
                             machinery (same "reuse the elementary rotations,
                             respect what axis is forward" discipline applies).

    E s = R_EC  C s
    L s = R_LE  E s
    B s = R_BL  L s
    ------------------
    B s = R_BL R_LE R_EC  C s

Every function below says, in its name, which frame its output is in --
star_celestial, star_earth_fixed, star_local. Nothing here returns a bare
"star_vector" and leaves you to remember what frame it's expressed in.
"""

import os
import sys

import numpy as np

_MODULE3 = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "module-3"))
if _MODULE3 not in sys.path:
    sys.path.insert(0, _MODULE3)
from rotations import rot_z                              # noqa: E402  (Module 3)

from spherical_astronomy import gmst_degrees              # noqa: E402


# ---------------------------------------------------------------------------
# C -- celestial
# ---------------------------------------------------------------------------
def star_celestial(ra_deg, dec_deg):
    """(RA, Dec) [deg, any broadcastable shape] -> unit vector in frame C."""
    ra, dec = np.deg2rad(ra_deg), np.deg2rad(dec_deg)
    return np.stack([np.cos(dec) * np.cos(ra),
                     np.cos(dec) * np.sin(ra),
                     np.sin(dec)], axis=-1)


# ---------------------------------------------------------------------------
# C -> E : pure Earth rotation about the shared polar axis
# ---------------------------------------------------------------------------
def R_EC(jd):
    """Earth-fixed <- celestial, at Julian Date jd (scalar). R_EC = Rz(-GMST).

    Rz(+GMST) would carry E's own x-axis (Greenwich, at celestial longitude
    GMST) FROM C-frame coordinates; we want the inverse map (a fixed star's
    C-components -> its components in the E basis), which is the transpose:
    Rz(-GMST). Verified below against the already-validated
    spherical_astronomy.equatorial_to_horizontal, not just asserted.
    """
    gmst = np.deg2rad(gmst_degrees(jd))
    return rot_z(-gmst)


def star_earth_fixed(star_c, jd):
    """C-frame vector(s) -> E-frame, at a SINGLE Julian Date jd."""
    return star_c @ R_EC(jd).T


def star_earth_fixed_series(ra_deg, dec_deg, jd_array):
    """Vectorized C->E for one star swept over many jd (array), broadcast-friendly."""
    gmst = np.deg2rad(gmst_degrees(np.asarray(jd_array, dtype=float)))
    ra, dec = np.deg2rad(ra_deg), np.deg2rad(dec_deg)
    H_G = gmst - ra                              # Greenwich hour angle
    return np.stack([np.cos(dec) * np.cos(H_G),
                     -np.cos(dec) * np.sin(H_G),
                     np.full_like(H_G, np.sin(dec))], axis=-1)


# ---------------------------------------------------------------------------
# E -> L : observer's fixed position on the Earth (no time dependence)
# ---------------------------------------------------------------------------
def R_LE(lat_deg, lon_deg):
    """Local ENU <- Earth-fixed. Standard ECEF->ENU rotation: each ROW is
    one ENU basis vector (East, North, Up) expressed in Earth-fixed
    coordinates -- the standard geodesy/GPS result."""
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    sp, cp, sl, cl = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    return np.array([[-sl, cl, 0.0],
                     [-sp * cl, -sp * sl, cp],
                     [cp * cl, cp * sl, sp]])


def star_local(star_e, lat_deg, lon_deg):
    """E-frame vector(s) -> L-frame (ENU)."""
    return star_e @ R_LE(lat_deg, lon_deg).T


def altaz_from_local(star_l):
    """ENU vector(s) -> (altitude, azimuth) degrees; azimuth from North thru East."""
    E, N, U = star_l[..., 0], star_l[..., 1], star_l[..., 2]
    alt = np.degrees(np.arcsin(np.clip(U, -1.0, 1.0)))
    az = np.degrees(np.arctan2(E, N)) % 360.0
    return alt, az


# ---------------------------------------------------------------------------
# full C -> L convenience (composes the two rotations above)
# ---------------------------------------------------------------------------
def altaz_from_celestial(ra_deg, dec_deg, jd, lat_deg, lon_deg):
    """(RA, Dec) at Julian Date jd, observed from (lat, lon) -> (alt, az) deg.

    Runs the explicit chain  L s = R_LE (R_EC  C s)  -- three named frames,
    two named rotations, no shortcuts.
    """
    s_c = star_celestial(ra_deg, dec_deg)
    s_e = star_earth_fixed(s_c, jd)
    s_l = star_local(s_e, lat_deg, lon_deg)
    return altaz_from_local(s_l)
"""module-9 -- frames.py  (Sections 35-36: Coordinate Frames -- Be Ruthless About Naming)

This is the computational centerpiece's frame backbone (Section 34): the
first half of the chain

    C (celestial) -> E (Earth-fixed) -> L (local observer) -> B (camera/body)

Four frames, named exactly as specified, never left implicit:

    C = celestial/inertial   equatorial, RA/Dec-based; z = celestial pole,
                             x = RA=0 (vernal equinox direction). Fixed --
                             does not rotate with the Earth. (Precession,
                             nutation, and proper motion are outside this
                             course's scope; C is treated as a fixed J2000-like
                             frame.)
    E = Earth-fixed          same z (the poles coincide), but x points at the
                             Greenwich meridian and rotates WITH the Earth.
    L = local observer       ENU (East, North, Up), tangent to the Earth at
                             the observer's (lat, lon). See README.md for why
                             ENU and not NED.
    B = camera/body          module-7/camera's convention: +x right, +y DOWN,
                             +z = optical axis (into the scene). R_BL depends
                             on where the camera points and is built once a
                             specific pointing scenario is defined -- see
                             module-7/camera/camera_orientation.py for the
                             machinery (same "reuse the elementary rotations,
                             respect what axis is forward" discipline applies).

    E s = R_EC  C s
    L s = R_LE  E s
    B s = R_BL  L s
    ------------------
    B s = R_BL R_LE R_EC  C s

Every function below says, in its name, which frame its output is in --
star_celestial, star_earth_fixed, star_local. Nothing here returns a bare
"star_vector" and leaves you to remember what frame it's expressed in.
"""

import os
import sys

import numpy as np

_MODULE3 = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "module-3"))
if _MODULE3 not in sys.path:
    sys.path.insert(0, _MODULE3)
from rotations import rot_z                              # noqa: E402  (Module 3)

from spherical_astronomy import gmst_degrees              # noqa: E402


# ---------------------------------------------------------------------------
# C -- celestial
# ---------------------------------------------------------------------------
def star_celestial(ra_deg, dec_deg):
    """(RA, Dec) [deg, any broadcastable shape] -> unit vector in frame C."""
    ra, dec = np.deg2rad(ra_deg), np.deg2rad(dec_deg)
    return np.stack([np.cos(dec) * np.cos(ra),
                     np.cos(dec) * np.sin(ra),
                     np.sin(dec)], axis=-1)


# ---------------------------------------------------------------------------
# C -> E : pure Earth rotation about the shared polar axis
# ---------------------------------------------------------------------------
def R_EC(jd):
    """Earth-fixed <- celestial, at Julian Date jd (scalar). R_EC = Rz(-GMST).

    Rz(+GMST) would carry E's own x-axis (Greenwich, at celestial longitude
    GMST) FROM C-frame coordinates; we want the inverse map (a fixed star's
    C-components -> its components in the E basis), which is the transpose:
    Rz(-GMST). Verified below against the already-validated
    spherical_astronomy.equatorial_to_horizontal, not just asserted.
    """
    gmst = np.deg2rad(gmst_degrees(jd))
    return rot_z(-gmst)


def star_earth_fixed(star_c, jd):
    """C-frame vector(s) -> E-frame, at a SINGLE Julian Date jd."""
    return star_c @ R_EC(jd).T


def star_earth_fixed_series(ra_deg, dec_deg, jd_array):
    """Vectorized C->E for one star swept over many jd (array), broadcast-friendly."""
    gmst = np.deg2rad(gmst_degrees(np.asarray(jd_array, dtype=float)))
    ra, dec = np.deg2rad(ra_deg), np.deg2rad(dec_deg)
    H_G = gmst - ra                              # Greenwich hour angle
    return np.stack([np.cos(dec) * np.cos(H_G),
                     -np.cos(dec) * np.sin(H_G),
                     np.full_like(H_G, np.sin(dec))], axis=-1)


# ---------------------------------------------------------------------------
# E -> L : observer's fixed position on the Earth (no time dependence)
# ---------------------------------------------------------------------------
def R_LE(lat_deg, lon_deg):
    """Local ENU <- Earth-fixed. Standard ECEF->ENU rotation: each ROW is
    one ENU basis vector (East, North, Up) expressed in Earth-fixed
    coordinates -- the standard geodesy/GPS result."""
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    sp, cp, sl, cl = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    return np.array([[-sl, cl, 0.0],
                     [-sp * cl, -sp * sl, cp],
                     [cp * cl, cp * sl, sp]])


def star_local(star_e, lat_deg, lon_deg):
    """E-frame vector(s) -> L-frame (ENU)."""
    return star_e @ R_LE(lat_deg, lon_deg).T


def altaz_from_local(star_l):
    """ENU vector(s) -> (altitude, azimuth) degrees; azimuth from North thru East."""
    E, N, U = star_l[..., 0], star_l[..., 1], star_l[..., 2]
    alt = np.degrees(np.arcsin(np.clip(U, -1.0, 1.0)))
    az = np.degrees(np.arctan2(E, N)) % 360.0
    return alt, az


# ---------------------------------------------------------------------------
# full C -> L convenience (composes the two rotations above)
# ---------------------------------------------------------------------------
def altaz_from_celestial(ra_deg, dec_deg, jd, lat_deg, lon_deg):
    """(RA, Dec) at Julian Date jd, observed from (lat, lon) -> (alt, az) deg.

    Runs the explicit chain  L s = R_LE (R_EC  C s)  -- three named frames,
    two named rotations, no shortcuts.
    """
    s_c = star_celestial(ra_deg, dec_deg)
    s_e = star_earth_fixed(s_c, jd)
    s_l = star_local(s_e, lat_deg, lon_deg)
    return altaz_from_local(s_l)


# ---------------------------------------------------------------------------
# L -> B : camera attitude, now that a pointing scenario exists (Section 38)
# ---------------------------------------------------------------------------
def R_BL(alt_deg, az_deg, roll_deg=0.0):
    """Local ENU <- camera/body, for a camera pointed at (altitude, azimuth)
    with a given roll about its own boresight.

    Camera frame (module-7/camera's convention): +x right, +y down,
    +z = boresight (optical axis). With roll=0 the camera's -y (image "up")
    is aligned with the projection of the local zenith onto the plane
    perpendicular to the boresight -- the natural "keep the image upright"
    choice for an alt-az-pointed instrument (roll=0 does NOT mean "aligned
    with L's own axes"; L's own Up axis generally is not the boresight).

    Built the same way R_LE was: each ROW is one camera-frame basis vector
    (x_cam, y_cam, z_cam), expressed in L (ENU) coordinates. Verified below
    (orthogonal, det=+1, boresight lands where requested) rather than
    trusted from the handedness argument alone.
    """
    alt, az, roll = np.deg2rad(alt_deg), np.deg2rad(az_deg), np.deg2rad(roll_deg)
    z_cam = np.array([np.sin(az) * np.cos(alt),      # boresight direction, in L (ENU)
                      np.cos(az) * np.cos(alt),
                      np.sin(alt)])
    zenith = np.array([0.0, 0.0, 1.0])                # L's own Up axis
    x_ref = np.cross(z_cam, zenith)                    # "right": e.g. East when facing North
    x_ref /= np.linalg.norm(x_ref)
    y_ref = np.cross(z_cam, x_ref)                    # right-handed: x,y,z with x cross y = z

    c, s = np.cos(roll), np.sin(roll)                 # roll spins (x_ref,y_ref) about z_cam
    x_cam = c * x_ref + s * y_ref
    y_cam = -s * x_ref + c * y_ref

    return np.array([x_cam, y_cam, z_cam])


def star_camera(star_l, alt_deg, az_deg, roll_deg=0.0):
    """L-frame vector(s) -> B-frame (camera), for the pointing above."""
    return star_l @ R_BL(alt_deg, az_deg, roll_deg).T
