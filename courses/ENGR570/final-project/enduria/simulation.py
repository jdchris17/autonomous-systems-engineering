"""
simulation.py - a truth model that produces ObservationBatch objects.

The simulator knows the real world (where the instrument is, what its
sensors really do, the real atmosphere, the real Earth orientation, whether
it is accelerating). The engine knows only what is in the batch, the
config and the reference data. Every error studied in the course is a
DIFFERENCE between the two, never a number added to a position.

Two things are shared with the engine on purpose, so that a perfect,
correctly configured case closes to numerical precision: the ERFA
star-direction chain (references.observed_enu) and the camera model. An
independent generator that shares neither lives in
tests/independent_reference.py.

All default magnitudes are illustrative, not sensor specifications.
"""

from dataclasses import dataclass

import erfa
import numpy as np
from astropy.time import TimeDelta

from .astrometry import MAS, Atmosphere, CatalogStar
from .contracts import NavigationConfig, Observation, ObservationBatch
from .estimation import ARCSEC
from .frames import CameraModel
from .geodesy import DEFAULT_C_CI, camera_axes_local
from .gravity import normal_gravity
from .references import (
    ReferenceData,
    catalog_columns_j2000,
    icrs_from_observed_enu,
    observed_enu,
    utc_time,
)

DEFAULT_START_UTC = "2026-11-11T12:00:00"
CATALOG_EPOCH = 2016.0


@dataclass(frozen=True)
class TrueInstrument:
    """What the hardware really does."""
    camera: CameraModel = CameraModel()
    C_CI: object = None                      # true camera-from-IMU (3x3)
    accel_bias: tuple = (0.0, 0.0, 0.0)      # raw = M f + bias + noise
    accel_matrix: object = None              # M, default identity
    centroid_noise_px: float = 0.0
    accel_noise: float = 0.0                 # per axis, per sample
    accel_samples: int = 200
    sample_rate_hz: float = 100.0
    gyro_noise: object = None                # None: the IMU reports no gyro

    @property
    def c_ci(self):
        return np.asarray(DEFAULT_C_CI.matrix if self.C_CI is None else self.C_CI,
                          dtype=float)


@dataclass(frozen=True)
class World:
    """Where the instrument is and what is happening to it."""
    latitude_deg: float = 45.0
    longitude_deg: float = -75.0
    height_m: float = 0.0
    atmosphere: object = None                # None: no atmosphere
    deflection_arcsec: tuple = (0.0, 0.0)    # (xi north, eta east) of the plumb line
    acceleration_enu: tuple = (0.0, 0.0, 0.0)   # constant, m/s^2
    vibration_amplitude: float = 0.0         # sinusoidal, east, m/s^2
    vibration_hz: float = 7.0
    disturbed_observations: object = None    # indices; None = all
    clock_error_s: float = 0.0               # instrument clock minus true time
    eop_error: tuple = (0.0, 0.0, 0.0)       # truth minus reference: dUT1 s, xp, yp rad
    proper_motion_mas_yr: float = 100.0
    max_parallax_mas: float = 80.0
    radial_velocity_km_s: float = 30.0


def star_positions(n, rng, camera, layout="uniform", cluster_radius_deg=0.3):
    """Pixel positions of n stars. 'clustered' packs them into a small disc."""
    if layout == "uniform":
        fraction = 0.04 + 0.92 * rng.random((n, 2))
        return fraction * [camera.width, camera.height]
    if layout == "clustered":
        radius_px = np.radians(cluster_radius_deg) * camera.fx
        centre = np.array([camera.width, camera.height]) * (0.3 + 0.4 * rng.random(2))
        r = radius_px * np.sqrt(rng.random(n))
        angle = 2.0 * np.pi * rng.random(n)
        return centre + np.column_stack((r * np.cos(angle), r * np.sin(angle)))
    raise ValueError(f"unknown star layout {layout!r}")


def _catalog_at(camera_dirs, c_nc, eop, world, rng, prefix):
    """
    Catalog entries (epoch 2016.0, with random space motion) for stars that
    are observed, in this world at this instant, at the given camera
    directions to within a fraction of an arcsecond.
    """
    enu = camera_dirs @ c_nc.T
    ra, dec = icrs_from_observed_enu(
        enu, eop, world.latitude_deg, world.longitude_deg, world.height_m,
        world.atmosphere)
    n = len(ra)
    pmr = rng.normal(0.0, world.proper_motion_mas_yr, n) * MAS / np.cos(dec)
    pmd = rng.normal(0.0, world.proper_motion_mas_yr, n) * MAS
    px = rng.uniform(2.0, max(world.max_parallax_mas, 2.0), n) / 1000.0
    rv = rng.normal(0.0, world.radial_velocity_km_s, n)
    e1, e2 = erfa.epj2jd(CATALOG_EPOCH)
    ra0, dec0, pmr0, pmd0, px0, rv0 = erfa.pmsafe(
        ra, dec, pmr, pmd, px, rv, eop.tdb_jd[0], eop.tdb_jd[1], e1, e2)
    return {f"{prefix}-{i:02d}": CatalogStar(
        ra_deg=float(np.degrees(ra0[i]) % 360.0), dec_deg=float(np.degrees(dec0[i])),
        pm_ra_cosdec_mas_yr=float(pmr0[i] * np.cos(dec0[i]) / MAS),
        pm_dec_mas_yr=float(pmd0[i] / MAS), parallax_mas=float(px0[i] * 1000.0),
        radial_velocity_km_s=float(rv0[i]), epoch_jyear=CATALOG_EPOCH,
        name=f"{prefix}-{i:02d}") for i in range(n)}


def simulate_batch(world, instrument, attitudes, rng, *, start_utc=DEFAULT_START_UTC,
                   spacing_s=1.0, n_stars=20, star_layout="uniform",
                   cluster_radius_deg=0.3, mismatch_fraction=0.0,
                   mismatch_offset_deg=(0.1, 1.5), eop_source=None):
    """
    Simulate one instrument at one place making len(attitudes) observations.

    attitudes: list of (zenith_angle_deg, azimuth_deg, roll_deg) of the
        camera boresight.
    n_stars: int, or one int per observation.
    mismatch_fraction: fraction of centroids labelled with the identity of
        a different catalog star mismatch_offset_deg away.
    eop_source: ReferenceData whose table values are the reference; the
        truth is those values plus world.eop_error.

    Returns (ObservationBatch, catalog dict, truth dict).
    """
    eop_source = ReferenceData({}) if eop_source is None else eop_source
    counts = np.broadcast_to(np.asarray(n_stars, dtype=int), (len(attitudes),))
    start = utc_time(start_utc)
    camera = instrument.camera
    matrix = (np.eye(3) if instrument.accel_matrix is None
              else np.asarray(instrument.accel_matrix, dtype=float))
    xi, eta = np.asarray(world.deflection_arcsec, dtype=float) * ARCSEC
    plumb = np.array([eta, xi, 1.0])
    plumb /= np.linalg.norm(plumb)
    gravity = normal_gravity(world.latitude_deg, world.height_m)
    sample_times = np.arange(instrument.accel_samples) / instrument.sample_rate_hz
    disturbed = (set(range(len(attitudes))) if world.disturbed_observations is None
                 else set(world.disturbed_observations))

    catalog, observations, truths = {}, [], []
    for k, (zenith, azimuth, roll) in enumerate(attitudes):
        true_time = start + TimeDelta(k * spacing_s, format="sec")
        stamped = utc_time(true_time + TimeDelta(world.clock_error_s, format="sec"))
        eop = eop_source.eop(true_time).with_errors(*world.eop_error)
        c_nc = camera_axes_local(zenith, azimuth, roll)

        # ---- the sky: place stars, write their catalog entries, then compute
        #      where those catalog stars are really observed ----------------------
        n = int(counts[k])
        nominal = CameraModel(horizontal_fov_deg=camera.horizontal_fov_deg)
        placed = nominal.unproject(star_positions(n, rng, nominal, star_layout,
                                                  cluster_radius_deg))
        entries = _catalog_at(placed, c_nc, eop, world, rng, f"E{k:03d}")
        ids = list(entries)
        enu = observed_enu(catalog_columns_j2000(list(entries.values())), eop,
                           world.latitude_deg, world.longitude_deg, world.height_m,
                           world.atmosphere)
        camera_dirs = enu @ c_nc
        pixels, _ = camera.project(camera_dirs)
        pixels = pixels + rng.standard_normal(pixels.shape) * instrument.centroid_noise_px

        # ---- wrong catalog identities ----------------------------------------------
        mismatched = []
        n_wrong = int(round(mismatch_fraction * n))
        if n_wrong:
            wrong = rng.choice(n, size=n_wrong, replace=False)
            offset = np.radians(rng.uniform(*mismatch_offset_deg, n_wrong))
            angle = 2.0 * np.pi * rng.random(n_wrong)
            decoy_dirs = placed[wrong] + np.column_stack(
                (offset * np.cos(angle), offset * np.sin(angle), np.zeros(n_wrong)))
            decoy_dirs /= np.linalg.norm(decoy_dirs, axis=1, keepdims=True)
            decoys = _catalog_at(decoy_dirs, c_nc, eop, world, rng, f"D{k:03d}")
            for index, decoy_id in zip(wrong, decoys):
                mismatched.append((ids[index], decoy_id))
                ids[index] = decoy_id
            entries.update(decoys)
        catalog.update(entries)

        # ---- the accelerometer: specific force = gravity reaction + acceleration ---
        force_enu = np.tile(gravity * plumb, (len(sample_times), 1))
        if k in disturbed:
            force_enu = force_enu + np.asarray(world.acceleration_enu, dtype=float)
            force_enu[:, 0] += world.vibration_amplitude * np.sin(
                2.0 * np.pi * world.vibration_hz * sample_times)
        c_in = instrument.c_ci.T @ c_nc.T                       # ENU -> IMU
        raw = (force_enu @ c_in.T) @ matrix.T + np.asarray(instrument.accel_bias)
        raw = raw + rng.standard_normal(raw.shape) * instrument.accel_noise
        gyro = None
        if instrument.gyro_noise is not None:
            gyro = rng.standard_normal(raw.shape) * instrument.gyro_noise

        observations.append(Observation(
            time_utc=stamped.isot, centroids_px=pixels, star_ids=ids,
            accel_samples=raw, gyro_samples=gyro, label=f"exposure {k}"))
        truths.append({"true_utc": utc_time(true_time).isot, "C_NC": c_nc,
                       "attitude": (zenith, azimuth, roll),
                       "camera_directions": camera_dirs, "mismatched": mismatched})

    truth = {"position": (world.latitude_deg, world.longitude_deg),
             "world": world, "instrument": instrument, "observations": truths}
    batch = ObservationBatch(observations, metadata={
        "source": "enduria.simulation", "start_utc": start.isot,
        "spacing_s": spacing_s})
    return batch, catalog, truth


def matched_config(world, instrument, **overrides):
    """
    A NavigationConfig that describes the instrument and world exactly:
    correct calibration, the true atmosphere, the true deflection as the
    model. Declared noise equals the true noise (with a small floor so the
    covariances stay positive when the noise is zero).
    """
    atmosphere = world.atmosphere
    settings = dict(
        camera=instrument.camera, C_CI=instrument.c_ci,
        accel_bias=tuple(instrument.accel_bias),
        accel_correction=None if instrument.accel_matrix is None
        else np.linalg.inv(np.asarray(instrument.accel_matrix, dtype=float)),
        centroid_sigma_px=max(instrument.centroid_noise_px, 1e-3),
        accel_noise_sigma=max(instrument.accel_noise, 1e-6),
        gyro_noise_sigma=instrument.gyro_noise,
        apply_refraction=atmosphere is not None,
        atmosphere=atmosphere if isinstance(atmosphere, Atmosphere) else None,
        refraction_fraction_sigma=0.0, height_m=world.height_m,
        deflection_model_arcsec=tuple(world.deflection_arcsec)
        if any(world.deflection_arcsec) else None)
    settings.update(overrides)
    return NavigationConfig(**settings)
