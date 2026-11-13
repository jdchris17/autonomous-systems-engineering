"""
independent_reference.py - a second route to the same physical quantities.

Nothing here calls the engine's algorithms or the course modules. The only
things imported from the project are the plain data containers that the
public interface is defined in terms of (CatalogStar, Observation,
ObservationBatch).

  stars and Earth rotation  Astropy's coordinate frames (ICRS, AltAz, ITRS)
  rotations                 scipy.spatial.transform.Rotation
  camera, geodesy, gravity,
  distance, rotation angle  closed formulas written out in this file

How independent is it? Astropy's frames are assembled by Astropy but call
the same ERFA routines underneath, and both sides read the same bundled
IERS table. So this path detects errors in how the engine ARRANGES the
physics (frames, signs, conventions, units, time scales, iteration), which
is where integration errors live. It cannot detect an error inside ERFA
itself or in the IERS data.
"""

import numpy as np
from astropy import units as u
from astropy.coordinates import ICRS, AltAz, Distance, EarthLocation, SkyCoord
from astropy.time import Time
from scipy.spatial.transform import Rotation

from enduria.astrometry import CatalogStar
from enduria.contracts import Observation, ObservationBatch

# WGS84 defining constants, typed here rather than imported.
A = 6378137.0
F = 1.0 / 298.257223563
B = A * (1.0 - F)
E2 = F * (2.0 - F)

# The camera of Module 3, restated.
WIDTH, HEIGHT, HFOV_DEG = 1920, 1080, 20.0
CX, CY = 960.0, 540.0
FOCAL_PX = WIDTH / (2.0 * np.tan(np.radians(HFOV_DEG) / 2.0))

C_CI = Rotation.from_rotvec([0.02, -0.01, 0.03]).as_matrix()   # camera-from-IMU
CATALOG_EPOCH = Time("J2016.0")


# ------------------------------------------------------------ closed formulas
def geodetic_to_ecef(latitude_deg, longitude_deg, height_m=0.0):
    lat, lon = np.radians(latitude_deg), np.radians(longitude_deg)
    n = A / np.sqrt(1.0 - E2 * np.sin(lat) ** 2)
    return np.array([(n + height_m) * np.cos(lat) * np.cos(lon),
                     (n + height_m) * np.cos(lat) * np.sin(lon),
                     (n * (1.0 - E2) + height_m) * np.sin(lat)])


def ellipsoid_normal_from_ecef(point):
    """Unit normal of the ellipsoid x^2/a^2 + y^2/a^2 + z^2/b^2 = 1 (gradient)."""
    gradient = np.array([point[0] / A**2, point[1] / A**2, point[2] / B**2])
    return gradient / np.linalg.norm(gradient)


def earth_rotation_angle(ut1_jd1, ut1_jd2):
    """IAU 2000 ERA [rad], from the defining expression."""
    days = (ut1_jd1 - 2451545.0) + ut1_jd2
    fraction = (ut1_jd1 % 1.0) + (ut1_jd2 % 1.0)
    turns = 0.7790572732640 + 0.00273781191135448 * days + fraction
    return 2.0 * np.pi * (turns % 1.0)


def gravity_1980(latitude_deg):
    """International Gravity Formula 1980 [m/s^2]."""
    lat = np.radians(latitude_deg)
    return 9.780327 * (1.0 + 0.0053024 * np.sin(lat) ** 2
                       - 0.0000058 * np.sin(2.0 * lat) ** 2)


def vincenty_inverse(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
    """Geodesic distance [m] on WGS84 by Vincenty's inverse iteration."""
    u1 = np.arctan((1.0 - F) * np.tan(np.radians(lat1_deg)))
    u2 = np.arctan((1.0 - F) * np.tan(np.radians(lat2_deg)))
    big_l = np.radians(lon2_deg - lon1_deg)
    big_l = (big_l + np.pi) % (2.0 * np.pi) - np.pi
    lam = big_l
    sin_u1, cos_u1, sin_u2, cos_u2 = np.sin(u1), np.cos(u1), np.sin(u2), np.cos(u2)
    for _ in range(200):
        sin_lam, cos_lam = np.sin(lam), np.cos(lam)
        sin_sigma = np.hypot(cos_u2 * sin_lam,
                             cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lam)
        if sin_sigma == 0.0:
            return 0.0
        cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lam
        sigma = np.arctan2(sin_sigma, cos_sigma)
        sin_alpha = cos_u1 * cos_u2 * sin_lam / sin_sigma
        cos2_alpha = 1.0 - sin_alpha**2
        cos_2sm = (cos_sigma - 2.0 * sin_u1 * sin_u2 / cos2_alpha
                   if cos2_alpha != 0.0 else 0.0)
        c = F / 16.0 * cos2_alpha * (4.0 + F * (4.0 - 3.0 * cos2_alpha))
        new = big_l + (1.0 - c) * F * sin_alpha * (
            sigma + c * sin_sigma * (cos_2sm + c * cos_sigma
                                     * (-1.0 + 2.0 * cos_2sm**2)))
        if abs(new - lam) < 1e-14:
            lam = new
            break
        lam = new
    k2 = cos2_alpha * (A**2 - B**2) / B**2
    big_a = 1.0 + k2 / 16384.0 * (4096.0 + k2 * (-768.0 + k2 * (320.0 - 175.0 * k2)))
    big_b = k2 / 1024.0 * (256.0 + k2 * (-128.0 + k2 * (74.0 - 47.0 * k2)))
    delta = big_b * sin_sigma * (cos_2sm + big_b / 4.0 * (
        cos_sigma * (-1.0 + 2.0 * cos_2sm**2)
        - big_b / 6.0 * cos_2sm * (-3.0 + 4.0 * sin_sigma**2)
        * (-3.0 + 4.0 * cos_2sm**2)))
    return float(B * big_a * (sigma - delta))


# -------------------------------------------------------------------- camera
def camera_axes(zenith_deg, azimuth_deg, roll_deg):
    """
    Columns: camera x (image right), y (image down), z (boresight) in local
    east/north/up. Built its own way; the engine never sees this matrix, so
    it need not match the simulator's convention for azimuth or roll.
    """
    z, az = np.radians(zenith_deg), np.radians(azimuth_deg)
    boresight = np.array([np.sin(z) * np.sin(az), np.sin(z) * np.cos(az), np.cos(z)])
    helper = np.array([0.0, 0.0, 1.0]) if zenith_deg > 1e-6 else np.array([0.0, 1.0, 0.0])
    x = np.cross(boresight, helper)
    x /= np.linalg.norm(x)
    y = np.cross(boresight, x)
    axes = np.column_stack((x, y, boresight))
    return Rotation.from_rotvec(boresight * np.radians(roll_deg)).as_matrix() @ axes


def pixel_to_ray(pixels):
    rays = np.column_stack(((pixels[:, 0] - CX) / FOCAL_PX,
                            (pixels[:, 1] - CY) / FOCAL_PX, np.ones(len(pixels))))
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def ray_to_pixel(rays):
    return np.column_stack((CX + FOCAL_PX * rays[:, 0] / rays[:, 2],
                            CY + FOCAL_PX * rays[:, 1] / rays[:, 2]))


def enu_from_altaz(altaz):
    alt, az = altaz.alt.rad, altaz.az.rad
    return np.column_stack((np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az),
                            np.sin(alt)))


def altaz_frame(time, location, atmosphere=None):
    if atmosphere is None:
        return AltAz(obstime=time, location=location)
    return AltAz(obstime=time, location=location,
                 pressure=atmosphere.pressure_hpa * u.hPa,
                 temperature=atmosphere.temperature_c * u.deg_C,
                 relative_humidity=atmosphere.relative_humidity,
                 obswl=atmosphere.wavelength_um * u.micron)


# ------------------------------------------------------------------- the sky
def make_catalog(enu_directions, time, location, rng, prefix):
    """
    (ids, SkyCoord at the catalog epoch, {id: CatalogStar}) for stars that
    are near the given local directions at `time`, with random space motion.
    """
    az = np.arctan2(enu_directions[:, 0], enu_directions[:, 1])
    alt = np.arcsin(enu_directions[:, 2])
    at_date = SkyCoord(AltAz(az=az * u.rad, alt=alt * u.rad, obstime=time,
                             location=location)).transform_to(ICRS())
    n = len(az)
    moving = SkyCoord(
        ra=at_date.ra, dec=at_date.dec,
        distance=Distance(parallax=rng.uniform(2.0, 80.0, n) * u.mas),
        pm_ra_cosdec=rng.normal(0.0, 100.0, n) * u.mas / u.yr,
        pm_dec=rng.normal(0.0, 100.0, n) * u.mas / u.yr,
        radial_velocity=rng.normal(0.0, 30.0, n) * u.km / u.s,
        obstime=time, frame="icrs")
    at_epoch = moving.apply_space_motion(new_obstime=CATALOG_EPOCH)
    ids = [f"{prefix}-{i:02d}" for i in range(n)]
    catalog = {identity: CatalogStar(
        ra_deg=float(at_epoch.ra.deg[i]), dec_deg=float(at_epoch.dec.deg[i]),
        pm_ra_cosdec_mas_yr=float(at_epoch.pm_ra_cosdec.to_value(u.mas / u.yr)[i]),
        pm_dec_mas_yr=float(at_epoch.pm_dec.to_value(u.mas / u.yr)[i]),
        parallax_mas=float(at_epoch.distance.parallax.to_value(u.mas)[i]),
        radial_velocity_km_s=float(at_epoch.radial_velocity.to_value(u.km / u.s)[i]),
        epoch_jyear=2016.0, name=identity) for i, identity in enumerate(ids)}
    return ids, at_epoch, catalog


def observed_enu(catalog_coordinates, time, location, atmosphere=None):
    """Observed east/north/up unit vectors of catalog stars, by Astropy."""
    at_date = catalog_coordinates.apply_space_motion(new_obstime=time)
    return enu_from_altaz(at_date.transform_to(altaz_frame(time, location,
                                                           atmosphere)))


def catalog_skycoord(stars):
    """SkyCoord for a list of CatalogStar (all at epoch 2016.0)."""
    return SkyCoord(
        ra=[s.ra_deg for s in stars] * u.deg, dec=[s.dec_deg for s in stars] * u.deg,
        distance=Distance(parallax=[s.parallax_mas for s in stars] * u.mas),
        pm_ra_cosdec=[s.pm_ra_cosdec_mas_yr for s in stars] * u.mas / u.yr,
        pm_dec=[s.pm_dec_mas_yr for s in stars] * u.mas / u.yr,
        radial_velocity=[s.radial_velocity_km_s for s in stars] * u.km / u.s,
        obstime=CATALOG_EPOCH, frame="icrs")


# -------------------------------------------------------------- observations
def independent_batch(latitude_deg, longitude_deg, attitudes, seed, *,
                      height_m=0.0, atmosphere=None,
                      start_utc="2026-11-11T12:00:00", spacing_s=60.0, n_stars=20):
    """
    A noise-free ObservationBatch generated without the engine's code.
    Returns (batch, catalog).
    """
    rng = np.random.default_rng(seed)
    location = EarthLocation.from_geodetic(longitude_deg * u.deg,
                                           latitude_deg * u.deg, height_m * u.m)
    gravity = gravity_1980(latitude_deg)
    catalog, observations = {}, []
    for k, (zenith, azimuth, roll) in enumerate(attitudes):
        time = Time(start_utc, scale="utc", precision=9) + k * spacing_s * u.s
        c_nc = camera_axes(zenith, azimuth, roll)
        pixels = np.column_stack((rng.uniform(60, WIDTH - 60, n_stars),
                                  rng.uniform(60, HEIGHT - 60, n_stars)))
        ids, coordinates, entries = make_catalog(pixel_to_ray(pixels) @ c_nc.T,
                                                 time, location, rng, f"I{k:02d}")
        catalog.update(entries)
        enu = observed_enu(coordinates, time, location, atmosphere)
        centroids = ray_to_pixel(enu @ c_nc)
        # A stationary accelerometer reads +g along local up.
        up_imu = C_CI.T @ (c_nc.T @ np.array([0.0, 0.0, 1.0]))
        observations.append(Observation(
            time_utc=time.isot, centroids_px=centroids, star_ids=ids,
            accel_samples=np.tile(gravity * up_imu, (50, 1)),
            label=f"independent {k}"))
    return ObservationBatch(observations, {"source": "independent (Astropy)"}), catalog
