"""Shared helpers for the system verification suites."""

import numpy as np

from enduria import ReferenceData, solve_navigation_fix
from enduria.astrometry import STANDARD_ATMOSPHERE
from enduria.geodesy import horizontal_position_error, north_east_error
from enduria.simulation import (
    TrueInstrument,
    World,
    matched_config,
    simulate_batch,
)

DEFAULT_ATTITUDE = (30.0, 40.0, 15.0)


def position_error_m(result, truth_position):
    """Geodesic distance from the reported position to the truth [m]."""
    longitude = 0.0 if result.longitude_deg is None else result.longitude_deg
    truth_lat, truth_lon = truth_position
    if result.longitude_deg is None:
        # Longitude is only withheld next to a pole; measure along a meridian.
        truth_lon = 0.0 if abs(truth_lat) > 89.9999999 else truth_lon
    return float(horizontal_position_error((result.latitude_deg, longitude),
                                           (truth_lat, truth_lon)))


def withheld_error_m(result, truth_position):
    """Error of a position that was computed but not reported, or None."""
    record = result.diagnostics.get("withheld_position")
    if record is None or record["latitude_deg"] is None:
        return None
    lon = record["longitude_deg"] or 0.0
    return float(horizontal_position_error((record["latitude_deg"], lon),
                                           truth_position))


def ne_error(result, truth_position):
    return np.array(north_east_error((result.latitude_deg, result.longitude_deg),
                                     truth_position))


def ideal_case(latitude, longitude, attitudes, seed, *,
               atmosphere=STANDARD_ATMOSPHERE, n_stars=20, **config):
    """A perfect instrument, correctly configured. Returns (result, truth)."""
    world = World(latitude_deg=latitude, longitude_deg=longitude,
                  atmosphere=atmosphere)
    instrument = TrueInstrument()
    batch, catalog, truth = simulate_batch(
        world, instrument, list(attitudes), np.random.default_rng(seed),
        n_stars=n_stars)
    result = solve_navigation_fix(batch, matched_config(world, instrument, **config),
                                  ReferenceData(catalog))
    return result, truth


def noisy_instrument(**changes):
    settings = dict(centroid_noise_px=0.2, accel_noise=0.005)
    settings.update(changes)
    return TrueInstrument(**settings)


def run(world, instrument, attitudes, seed, config=None, *, references=None,
        **simulation):
    """Simulate and solve. Returns (result, truth, batch, catalog)."""
    batch, catalog, truth = simulate_batch(
        world, instrument, list(attitudes), np.random.default_rng(seed),
        **simulation)
    config = matched_config(world, instrument) if config is None else config
    refs = ReferenceData(catalog) if references is None else references(catalog)
    return solve_navigation_fix(batch, config, refs), truth, batch, catalog
