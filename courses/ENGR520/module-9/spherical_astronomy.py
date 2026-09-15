"""module-9 -- spherical_astronomy.py

Shared library: calendar date -> Julian Date -> Greenwich/Local Sidereal
Time -> hour angle -> the equatorial-to-horizontal coordinate transform.

Definitions used throughout this module:

    phi      observer latitude          (+N)
    lambda   observer longitude         (+E)
    alpha    right ascension  (RA)       hours, 0..24
    delta    declination      (Dec)      degrees, -90..+90
    theta    local sidereal time (LST)   degrees
    H        hour angle = LST - alpha    degrees, +west of the meridian
    h        altitude                    degrees, 0 = horizon, 90 = zenith
    A        azimuth                     degrees, 0 = North, 90 = East (compass convention)

The equatorial -> horizontal transform is not a lookup table of special
cases (rise/set/circumpolar). It is one 3x3 rotation, exactly like every
other frame change in this course (Module 3's R_IB, Module 7's camera R).
Build the star's unit vector in the hour-angle/declination frame:

    r_eq = [cos(delta) cos(H), cos(delta) sin(H), sin(delta)]

and rotate it into the local horizontal frame (North, East, Up) by the
observer's colatitude. That rotation is

    R(phi) = [[-sin(phi), 0, cos(phi)],
              [    0,    -1,    0    ],
              [ cos(phi), 0, sin(phi)]]

(orthogonal, det = +1 -- check it yourself; the -1 in the middle row is not
a mistake, it is H increasing WEST meeting azimuth increasing EAST). Then

    h = asin(Up),   A = atan2(East, North)

fall out with no rise/set/circumpolar logic anywhere -- see
sky_over_one_day.py for the numerical proof that behavior emerges from
this alone.
"""

import numpy as np


# ---------------------------------------------------------------------------
# time
# ---------------------------------------------------------------------------
def julian_date(year, month, day, hour=0.0):
    """Julian Date for a Gregorian calendar date + UT hour (Meeus, ch. 7)."""
    year = np.asarray(year)
    month = np.asarray(month)
    y = np.where(month <= 2, year - 1, year)
    m = np.where(month <= 2, month + 12, month)
    A = np.floor(y / 100.0)
    B = 2 - A + np.floor(A / 4.0)
    jd = (np.floor(365.25 * (y + 4716)) + np.floor(30.6001 * (m + 1))
         + day + B - 1524.5 + hour / 24.0)
    return jd


def gmst_degrees(jd):
    """Greenwich Mean Sidereal Time, in degrees, at Julian Date jd (Meeus 12.4)."""
    T = (jd - 2451545.0) / 36525.0
    gmst = (280.46061837 + 360.98564736629 * (jd - 2451545.0)
           + 0.000387933 * T**2 - T**3 / 38710000.0)
    return np.mod(gmst, 360.0)


def local_sidereal_time_degrees(jd, lon_east_deg):
    """LST = GMST + east longitude (both in degrees)."""
    return np.mod(gmst_degrees(jd) + lon_east_deg, 360.0)


def hour_angle_degrees(lst_deg, ra_hours):
    """H = LST - RA, in degrees. Not wrapped -- sin/cos below don't care."""
    return lst_deg - ra_hours * 15.0


# ---------------------------------------------------------------------------
# the one rotation
# ---------------------------------------------------------------------------
def horizontal_rotation(lat_rad):
    sp, cp = np.sin(lat_rad), np.cos(lat_rad)
    return np.array([[-sp, 0.0, cp],
                     [0.0, -1.0, 0.0],
                     [cp, 0.0, sp]])


def equatorial_to_horizontal(H_deg, dec_deg, lat_deg):
    """(H, delta, phi) [[all degrees, any shape]] -> (altitude, azimuth) degrees.

    altitude: 0 horizon, 90 zenith.  azimuth: 0 North, 90 East, compass sense.
    """
    # H and dec may be arrays (e.g. H(t) for one star) or scalars; broadcast
    # them together. lat is the observer's (scalar) latitude -> one rotation.
    H, dec = np.broadcast_arrays(np.deg2rad(H_deg), np.deg2rad(dec_deg))
    lat = np.deg2rad(lat_deg)

    r_eq = np.stack([np.cos(dec) * np.cos(H),
                     np.cos(dec) * np.sin(H),
                     np.sin(dec)], axis=-1)
    R = horizontal_rotation(lat)
    # apply R to the last axis for arbitrarily-shaped H/dec
    horiz = r_eq @ R.T
    north, east, up = horiz[..., 0], horiz[..., 1], horiz[..., 2]
    alt = np.degrees(np.arcsin(np.clip(up, -1.0, 1.0)))
    az = np.degrees(np.arctan2(east, north)) % 360.0
    return alt, az


class Star:
    def __init__(self, name, ra_hours, dec_deg):
        self.name = name
        self.ra_hours = ra_hours
        self.dec_deg = dec_deg

    def __repr__(self):
        return f"Star({self.name!r}, ra={self.ra_hours}h, dec={self.dec_deg} deg)"
