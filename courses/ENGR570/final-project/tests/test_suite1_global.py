"""
Test suite 1 - global geographic validation (system level).

Acceptance criterion: in the perfect, noise-free, correctly modelled case
the complete engine recovers the specified location to 1 mm, everywhere on
the ellipsoid and for any camera orientation. The synthetic observations
include proper motion, parallax, aberration and a standard atmosphere, so
this also verifies the iterated position-dependent corrections.
"""

import numpy as np
import pytest

from enduria import FIX_STATUSES
from helpers import ideal_case, position_error_m
from tolerances import SEED, tolerance

TOLERANCE_M = tolerance("ideal_position_m")

LATITUDES = [0.0, 23.44, -23.44, 45.0, -45.0, 66.56, -66.56, 80.0, -80.0,
             89.0, -89.0, 89.99, -89.99]
LONGITUDES = [0.0, 75.3, -120.7, 179.99999, -179.99999]
# (zenith angle, azimuth, roll) of the boresight, degrees
ORIENTATIONS = [(0.0, 0.0, 0.0), (25.0, 70.0, 40.0), (50.0, 200.0, 310.0),
                (60.0, 310.0, 120.0)]

GRID = [(lat, LONGITUDES[(i + j) % len(LONGITUDES)], ORIENTATIONS[j])
        for i, lat in enumerate(LATITUDES) for j in range(4)]


@pytest.mark.parametrize("latitude, longitude, orientation", GRID)
def test_single_observation_recovers_location(latitude, longitude, orientation):
    result, truth = ideal_case(latitude, longitude, [orientation], SEED + 1)
    assert result.status in FIX_STATUSES, result.summary()
    assert position_error_m(result, truth["position"]) < TOLERANCE_M


@pytest.mark.parametrize("longitude", [179.9999999, -179.9999999, 180.0, -180.0])
def test_longitude_wraps_at_the_date_line(longitude):
    result, truth = ideal_case(12.0, longitude, [ORIENTATIONS[1]], SEED + 2)
    assert -180.0 <= result.longitude_deg < 180.0
    assert position_error_m(result, truth["position"]) < TOLERANCE_M


@pytest.mark.parametrize("latitude", [90.0, -90.0])
@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_pole_has_latitude_but_no_longitude(latitude, orientation):
    result, truth = ideal_case(latitude, 33.0, [orientation], SEED + 3)
    assert result.latitude_deg is not None
    assert result.longitude_deg is None            # undefined, not zero
    assert result.status == "DEGRADED"
    assert any("longitude is not defined" in r for r in result.reasons)
    assert result.covariance_ne_m2 is not None     # north/east metres still valid
    assert position_error_m(result, truth["position"]) < TOLERANCE_M


@pytest.mark.parametrize("latitude, longitude", [(45.0, -75.0), (-33.9, 151.2),
                                                 (78.2, 15.6), (-89.5, -140.0)])
def test_batch_of_orientations_recovers_location(latitude, longitude):
    result, truth = ideal_case(latitude, longitude, ORIENTATIONS, SEED + 4)
    assert result.observations_used == len(ORIENTATIONS)
    assert position_error_m(result, truth["position"]) < TOLERANCE_M


@pytest.mark.parametrize("offset_deg", [0.0, 1.0, 20.0, 170.0])
def test_answer_does_not_depend_on_the_initial_position(offset_deg):
    truth_position = (38.0, -77.0)
    guess = (max(truth_position[0] - 0.4 * offset_deg, -89.0),
             (truth_position[1] + offset_deg + 180.0) % 360.0 - 180.0)
    result, truth = ideal_case(*truth_position, [ORIENTATIONS[2]], SEED + 5,
                               initial_position=guess)
    assert position_error_m(result, truth["position"]) < TOLERANCE_M


def test_world_without_atmosphere_also_closes():
    result, truth = ideal_case(51.5, -0.1, ORIENTATIONS, SEED + 6, atmosphere=None)
    assert position_error_m(result, truth["position"]) < TOLERANCE_M


def test_iteration_is_reported_and_converges_quickly():
    result, _ = ideal_case(45.0, -75.0, [ORIENTATIONS[2]], SEED + 7)
    updates = [step["update_m"] for step in result.diagnostics["iterations"]
               if "update_m" in step]
    assert updates[-1] < 0.01 and len(updates) <= 5
    assert all(b < a for a, b in zip(updates, updates[1:]))
    assert np.isfinite(result.horizontal_sigma_m)
