"""
astrometric_corrections.py - Space Systems Engineering, Module 8

Where a catalog star actually appears to an observer on Earth.

ERFA is the physical reference implementation (the same routines Astropy
uses). Each level of the chain is a separate, explicitly labelled function:

  catalog          ICRS direction at the catalog's reference epoch
  propagated       ICRS (barycentric) direction at the observation epoch:
                   proper motion and radial velocity only (erfa.pmsafe)
  topocentric      what the observer would see with no atmosphere: adds
                   annual parallax, gravitational light deflection by the
                   Sun, annual and diurnal aberration (erfa.atco13 with zero
                   pressure)
  observed         the same with atmospheric refraction (erfa.atco13)

Frames
------
erfa.atco13 returns azimuth and zenith distance in the observer's local
horizon frame. Every function here also returns the same direction as a
unit vector in "GCRS-oriented axes": the local vector rotated into the
celestial frame with the SAME Earth-orientation matrix the navigation
solver uses (Module 6, C_GE = C_EG^T, and the local east/north/up axes).
That rotation contains no physics. It makes directions at different
correction levels comparable, and it is the form Module 4's attitude
solver expects for its reference vectors.

Not applied twice
-----------------
Precession, nutation, Earth rotation and polar motion are frame rotations
and live only in C_EG. They are deliberately absent from the list of
corrections above: the vectors stay in non-rotating celestial axes.
Aberration, parallax, light deflection and refraction change the direction
itself and are applied exactly once, inside erfa.atco13.
"""

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import erfa
import numpy as np
from astropy.time import Time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "module-6"))
import celestial_position as cp  # noqa: E402  (adds Modules 1-5 to sys.path)
import position_simulator as ps  # noqa: E402
from astronomical_time import time_scales  # noqa: E402
from compare_reference_frames import polar_motion  # noqa: E402

from atmospheric_refraction import VACUUM, Atmosphere  # noqa: E402

MAS = np.radians(1.0 / 3.6e6)
GAIA_DR3_EPOCH = 2016.0                # Julian year of the reference epoch
J2000_JD = (2451545.0, 0.0)


@dataclass(frozen=True)
class CatalogStar:
    """
    An astrometric catalog entry in the ICRS at a reference epoch.

    Proper motion in right ascension is mu_alpha* = d(RA)/dt * cos(Dec), the
    convention of Hipparcos and Gaia.
    """
    ra_deg: float
    dec_deg: float
    pm_ra_cosdec_mas_yr: float = 0.0
    pm_dec_mas_yr: float = 0.0
    parallax_mas: float = 0.0
    radial_velocity_km_s: float = 0.0
    epoch_jyear: float = GAIA_DR3_EPOCH
    name: str = ""


@dataclass(frozen=True)
class ObserverLocation:
    latitude_deg: float                # geodetic, WGS84
    longitude_deg: float               # east positive
    height_m: float = 0.0


@dataclass(frozen=True)
class StarDirection:
    """A direction together with an explicit statement of what it is."""
    celestial: np.ndarray              # unit vector, GCRS-oriented axes
    frame: str
    corrections: tuple                 # physical effects included
    enu: object = None                 # unit vector east/north/up, if local
    azimuth_deg: float = None          # north through east
    altitude_deg: float = None


# --------------------------------------------------------------------- helpers
def _time(observation_time):
    return (observation_time if isinstance(observation_time, Time)
            else Time(observation_time, scale="utc"))


def _epoch_jd(jyear):
    return erfa.epj2jd(jyear)


def _columns(stars):
    stars = [stars] if isinstance(stars, CatalogStar) else list(stars)
    dec = np.radians([s.dec_deg for s in stars])
    return {
        "ra": np.radians([s.ra_deg for s in stars]),
        "dec": dec,
        # ERFA wants d(RA)/dt, not the cos(Dec)-scaled value, in rad/year.
        "pmr": np.array([s.pm_ra_cosdec_mas_yr for s in stars]) * MAS / np.cos(dec),
        "pmd": np.array([s.pm_dec_mas_yr for s in stars]) * MAS,
        "px": np.array([s.parallax_mas for s in stars]) / 1000.0,       # arcsec
        "rv": np.array([s.radial_velocity_km_s for s in stars]),
        "epoch": np.array([s.epoch_jyear for s in stars]),
    }


def _unit(ra, dec):
    return np.column_stack((np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra),
                            np.sin(dec)))


def _to_epoch(columns, jd1, jd2):
    """Space-motion propagation of catalog columns to a TDB epoch."""
    e1, e2 = _epoch_jd(columns["epoch"])
    return erfa.pmsafe(columns["ra"], columns["dec"], columns["pmr"],
                       columns["pmd"], columns["px"], columns["rv"],
                       e1, e2, jd1, jd2)


def local_to_celestial_matrix(observation_time, location):
    """
    Rotation from local east/north/up to GCRS-oriented axes, C_GE C_EN,
    using Module 6's Earth orientation. A frame change only.
    """
    c_eg, _ = cp.earth_orientation(_time(observation_time))
    c_en = ps.local_axes_ecef(location.latitude_deg, location.longitude_deg)
    return np.asarray(c_eg.matrix).T @ c_en


def enu_from_az_zd(azimuth_rad, zenith_rad):
    sin_z = np.sin(zenith_rad)
    return np.column_stack((sin_z * np.sin(azimuth_rad),
                            sin_z * np.cos(azimuth_rad), np.cos(zenith_rad)))


# -------------------------------------------------------- array implementations
def catalog_directions(stars):
    """(N, 3) ICRS unit vectors at each star's catalog epoch. No corrections."""
    c = _columns(stars)
    return _unit(c["ra"], c["dec"])


def propagated_directions(stars, observation_time):
    """(N, 3) ICRS barycentric unit vectors at the observation epoch
    (proper motion and radial velocity; no parallax, no aberration)."""
    t = _time(observation_time).tdb
    ra, dec, *_ = _to_epoch(_columns(stars), t.jd1, t.jd2)
    return _unit(ra, dec)


def _observed(stars, observation_time, location, atmosphere):
    """erfa.atco13 for a list of stars. Returns (enu (N,3), az, zd)."""
    time = _time(observation_time)
    columns = _columns(stars)
    # atco13 expects star data at epoch J2000.0 and applies the space motion
    # to the date itself, so move the catalog entry to J2000.0 first.
    ra, dec, pmr, pmd, px, rv = _to_epoch(columns, *J2000_JD)
    xp, yp = polar_motion(time)
    dut1 = time_scales(time).dut1_seconds
    atmosphere = VACUUM if atmosphere is None else atmosphere
    az, zd, *_ = erfa.atco13(
        ra, dec, pmr, pmd, px, rv, time.utc.jd1, time.utc.jd2, dut1,
        np.radians(location.longitude_deg), np.radians(location.latitude_deg),
        location.height_m, xp, yp, atmosphere.pressure_hpa,
        atmosphere.temperature_c, atmosphere.relative_humidity,
        atmosphere.wavelength_um)
    return enu_from_az_zd(az, zd), az, zd


def topocentric_directions(stars, observation_time, location):
    """(celestial (N,3), enu (N,3)): airless topocentric apparent directions."""
    enu, _, _ = _observed(stars, observation_time, location, VACUUM)
    return enu @ local_to_celestial_matrix(observation_time, location).T, enu


def observed_directions(stars, observation_time, location, atmosphere):
    """(celestial (N,3), enu (N,3)): observed directions with refraction."""
    enu, _, _ = _observed(stars, observation_time, location, atmosphere)
    return enu @ local_to_celestial_matrix(observation_time, location).T, enu


# ---------------------------------------------------------- required interface
def propagate_catalog_star(star, observation_time):
    """
    Propagate an astrometric catalog entry
    to a specified observation epoch.

    Returns a new CatalogStar whose position, proper motion, parallax and
    radial velocity are those at the observation epoch (rigorous space
    motion, erfa.pmsafe). Frame: ICRS, barycentric. Not included: parallax
    displacement, light deflection, aberration, refraction.
    """
    t = _time(observation_time).tdb
    ra, dec, pmr, pmd, px, rv = (np.asarray(v).item() for v in
                                 _to_epoch(_columns(star), t.jd1, t.jd2))
    return replace(
        star, ra_deg=float(np.degrees(ra) % 360.0), dec_deg=float(np.degrees(dec)),
        pm_ra_cosdec_mas_yr=float(pmr * np.cos(dec) / MAS),
        pm_dec_mas_yr=float(pmd / MAS), parallax_mas=float(px * 1000.0),
        radial_velocity_km_s=float(rv), epoch_jyear=float(t.jyear))


def topocentric_star_direction(
    star,
    observation_time,
    observer_location
):
    """
    Return a topocentric geometric direction,
    using an astronomical reference implementation.

    Frame and included corrections must be explicit.

    Reference implementation: erfa.atco13 with zero pressure. The result is
    the direction from which the star's light arrives at the observer in
    the absence of an atmosphere.
    """
    celestial, enu = topocentric_directions([star], observation_time,
                                            observer_location)
    return _direction(celestial[0], enu[0],
                      "topocentric apparent, no atmosphere",
                      ("proper motion and radial velocity", "annual parallax",
                       "light deflection by the Sun", "annual aberration",
                       "diurnal aberration and parallax"))


def observed_star_direction(
    star,
    observation_time,
    observer_location,
    atmosphere
):
    """
    Return an observed direction with
    atmospheric refraction included.

    Reference implementation: erfa.atco13 with the supplied Atmosphere
    (pressure, temperature, humidity, wavelength).
    """
    celestial, enu = observed_directions([star], observation_time,
                                         observer_location, atmosphere)
    return _direction(celestial[0], enu[0], "topocentric observed",
                      ("proper motion and radial velocity", "annual parallax",
                       "light deflection by the Sun", "annual aberration",
                       "diurnal aberration and parallax",
                       "atmospheric refraction"))


def _direction(celestial, enu, frame, corrections):
    return StarDirection(
        celestial=celestial, enu=enu,
        frame=f"{frame}; unit vector in GCRS-oriented axes (and local ENU)",
        corrections=corrections,
        azimuth_deg=float(np.degrees(np.arctan2(enu[0], enu[1])) % 360.0),
        altitude_deg=float(np.degrees(np.arcsin(np.clip(enu[2], -1, 1)))))


# -------------------------------------------------------- inverse and synthesis
def icrs_from_observed_enu(vectors_enu, observation_time, location,
                           atmosphere=None, eop=None):
    """
    (N, 3) ICRS astrometric unit vectors of sources seen at given observed
    local directions (erfa.atoc13): the inverse of the observed-place chain
    for a star with no proper motion or parallax.

    eop = (dut1_seconds, xp_rad, yp_rad) may be supplied to avoid table
    lookups inside loops.
    """
    time = _time(observation_time)
    v = np.atleast_2d(np.asarray(vectors_enu, dtype=float))
    az = np.arctan2(v[:, 0], v[:, 1])
    zd = np.arctan2(np.hypot(v[:, 0], v[:, 1]), v[:, 2])
    if eop is None:
        xp, yp = polar_motion(time)
        eop = (time_scales(time).dut1_seconds, xp, yp)
    atmosphere = VACUUM if atmosphere is None else atmosphere
    ra, dec = erfa.atoc13(
        "A", az, zd, time.utc.jd1, time.utc.jd2, eop[0],
        np.radians(location.longitude_deg), np.radians(location.latitude_deg),
        location.height_m, eop[1], eop[2], atmosphere.pressure_hpa,
        atmosphere.temperature_c, atmosphere.relative_humidity,
        atmosphere.wavelength_um)
    return _unit(ra, dec)


def synthetic_catalog(camera_vectors, camera_axes_local, observation_time,
                      location, rng, *, proper_motion_mas_yr=100.0,
                      max_parallax_mas=80.0, radial_velocity_km_s=30.0,
                      epoch_jyear=GAIA_DR3_EPOCH):
    """
    A catalog of stars that fall (to a fraction of an arcsecond) at the
    given camera-frame directions for this observer, time and pointing.

    camera_axes_local is C_NC (camera axes in east/north/up). Each star gets
    a random proper motion (Gaussian, per axis), parallax (uniform up to the
    maximum) and radial velocity, and its entry is stated at epoch_jyear.
    The numbers are illustrative of bright nearby stars, not a real sky.
    """
    time = _time(observation_time)
    enu = np.asarray(camera_vectors) @ np.asarray(camera_axes_local).T
    icrs = icrs_from_observed_enu(enu, time, location)
    ra = np.arctan2(icrs[:, 1], icrs[:, 0])
    dec = np.arcsin(icrs[:, 2])
    n = len(icrs)
    at_date = [CatalogStar(
        ra_deg=float(np.degrees(ra[i])), dec_deg=float(np.degrees(dec[i])),
        pm_ra_cosdec_mas_yr=float(rng.normal(0.0, proper_motion_mas_yr)),
        pm_dec_mas_yr=float(rng.normal(0.0, proper_motion_mas_yr)),
        parallax_mas=float(rng.uniform(2.0, max_parallax_mas)),
        radial_velocity_km_s=float(rng.normal(0.0, radial_velocity_km_s)),
        epoch_jyear=float(time.tdb.jyear), name=f"S{i:02d}") for i in range(n)]
    # Run the space motion backwards to state each entry at the catalog epoch.
    e1, e2 = _epoch_jd(epoch_jyear)
    c = _columns(at_date)
    t = time.tdb
    ra0, dec0, pmr, pmd, px, rv = erfa.pmsafe(
        c["ra"], c["dec"], c["pmr"], c["pmd"], c["px"], c["rv"], t.jd1, t.jd2,
        e1, e2)
    return [CatalogStar(
        ra_deg=float(np.degrees(ra0[i]) % 360.0), dec_deg=float(np.degrees(dec0[i])),
        pm_ra_cosdec_mas_yr=float(pmr[i] * np.cos(dec0[i]) / MAS),
        pm_dec_mas_yr=float(pmd[i] / MAS), parallax_mas=float(px[i] * 1000.0),
        radial_velocity_km_s=float(rv[i]), epoch_jyear=epoch_jyear,
        name=f"S{i:02d}") for i in range(n)]
