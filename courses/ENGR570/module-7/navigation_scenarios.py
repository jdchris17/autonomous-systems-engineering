"""
navigation_scenarios.py - Space Systems Engineering, Module 7

Batches of simulated observations from ONE instrument, built on the Module 6
observation simulator. Shared by the Module 7 experiments and tests.

A batch is a list of VerticalMeasurement objects produced exactly as real
data would be: stars -> Module 4 attitude, accelerometer samples -> Module 5
vertical, the believed calibration, and the believed timestamp. The true
position is kept in a separate dictionary for validation only.

Fixed errors (held constant across the whole batch, as for a real device):
  accel_bias          accelerometer bias in IMU axes [m/s^2]
  misalignment_rad    camera-IMU error; the software believes
                      C_CI = R(misalignment) C_CI,true
  clock_offset_s      believed time = true time + clock_offset_s
"""

from functools import lru_cache

import numpy as np
from astropy.time import Time, TimeDelta

import navigation_estimator as ne
import position_simulator as ps
from accelerometer_calibration import AccelerometerErrorModel
from imu_alignment import perturb_calibration

cp = ne.cp

START_UTC = ps.OBSERVATION_UTC
TRUE_POSITION = (45.0, -75.0)

# Nominal random-noise model.
CENTROID_NOISE_PX = 0.2
ACCEL_NOISE = 0.005                  # m/s^2 per axis per sample
ACCEL_SAMPLES = 200
N_STARS = 20
SPACING_S = 1.0                      # time between observations

# Nominal sizes of the fixed errors (1 sigma, per axis).
ACCEL_BIAS_SIGMA = 0.005             # m/s^2
MISALIGNMENT_SIGMA = np.radians(0.05)

DEFAULT_ATTITUDE = (30.0, 40.0, 15.0)        # zenith angle, azimuth, roll [deg]


@lru_cache(maxsize=4096)
def earth_at(start_utc, offset_s):
    """Cached (C_EG, info) at start_utc + offset_s."""
    time = Time(start_utc, scale="utc") + TimeDelta(float(offset_s), format="sec")
    return cp.earth_orientation(time)


def simulate_batch(rng, n_observations=20, *, position=TRUE_POSITION,
                   start_utc=START_UTC, spacing_s=SPACING_S, attitudes=None,
                   centroid_noise_px=CENTROID_NOISE_PX, accel_noise=ACCEL_NOISE,
                   noise_multipliers=None, accel_bias=None,
                   misalignment_rad=None, clock_offset_s=0.0,
                   acceleration_enu=(0.0, 0.0, 0.0), n_accel_samples=ACCEL_SAMPLES,
                   n_stars=N_STARS, run_quality_check=True):
    """
    Simulate n_observations from one instrument.

    attitudes: None for a fixed instrument, or a list of (zenith_angle_deg,
        azimuth_deg, roll_deg), one per observation.
    noise_multipliers: optional per-observation factor applied to both the
        centroid noise and the accelerometer noise.

    Returns (measurements, truth). The random-error covariance attached to
    each measurement is computed by the estimator from the same noise model
    used to generate it; fixed errors are deliberately absent from it.
    """
    bias = np.zeros(3) if accel_bias is None else np.asarray(accel_bias, float)
    multipliers = (np.ones(n_observations) if noise_multipliers is None
                   else np.asarray(noise_multipliers, dtype=float))
    measurements, truths = [], []
    for k in range(n_observations):
        zenith, azimuth, roll = (DEFAULT_ATTITUDE if attitudes is None
                                 else attitudes[k])
        true_earth = earth_at(start_utc, k * spacing_s)
        model = AccelerometerErrorModel(
            bias=tuple(bias), scale_factor_error=0.0, cross_axis=0.0,
            noise_sigma=accel_noise * multipliers[k])
        obs = ps.simulate_observation(
            *position, true_earth[1]["timestamp_utc"], C_EG=true_earth[0],
            rng=rng, n_stars=n_stars,
            centroid_noise_px=centroid_noise_px * multipliers[k],
            accel_model=model, n_accel_samples=n_accel_samples,
            zenith_angle_deg=zenith, azimuth_deg=azimuth, roll_deg=roll,
            acceleration_enu=acceleration_enu)

        believed_calibration = obs.C_CI
        if misalignment_rad is not None:
            believed_calibration = perturb_calibration(obs.C_CI, misalignment_rad)
        believed_earth = (true_earth if clock_offset_s == 0.0
                          else earth_at(start_utc, k * spacing_s + clock_offset_s))

        measurements.append(ne.measurement_from_sensors(
            obs.reference_stars, obs.measured_stars, obs.star_sigma_rad,
            obs.accel_samples, believed_calibration, believed_earth[1]["timestamp_utc"],
            earth=believed_earth,
            accel_noise_sigma=(accel_noise * multipliers[k]
                               if run_quality_check else None)))
        truths.append(obs.truth)

    truth = {"position": tuple(position), "accel_bias": bias,
             "misalignment_rad": (np.zeros(3) if misalignment_rad is None
                                  else np.asarray(misalignment_rad, float)),
             "clock_offset_s": clock_offset_s, "observations": truths}
    return measurements, truth


def single_fix(measurement):
    """Module 6 direct inversion of one measurement: (lat_deg, lon_deg)."""
    return cp.celestial_vertical_to_position(measurement.up_celestial,
                                             measurement.C_GE.T)


def north_east_error(lat_lon, truth_position=TRUE_POSITION):
    return np.array(cp.north_east_error(lat_lon, truth_position))


def diverse_attitudes(n, rng=None, *, max_zenith=60.0):
    """
    A deliberately varied attitude sequence: zenith angles spread from 5 deg
    to max_zenith, azimuths and rolls stepped around the circle.
    """
    k = np.arange(n)
    zenith = 5.0 + (max_zenith - 5.0) * ((k * 0.618034) % 1.0)
    azimuth = (k * 137.508) % 360.0
    roll = (k * 97.0) % 360.0
    return list(zip(zenith, azimuth, roll))
