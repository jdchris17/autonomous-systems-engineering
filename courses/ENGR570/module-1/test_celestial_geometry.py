"""Tests for celestial_geometry.py. Run with:  python -m pytest -v"""

import numpy as np
import pytest

from celestial_geometry import (
    SIDEREAL_DAY_SECONDS,
    angular_separation,
    earth_rotation_angle,
    radec_to_unit,
    star_altitude,
    unit_to_radec,
)

TOL = 1e-12


# ---------------------------------------------------------------- Function 1
@pytest.mark.parametrize(
    "ra, dec, expected",
    [
        (0.0, 0.0, [1, 0, 0]),       # vernal equinox direction
        (90.0, 0.0, [0, 1, 0]),
        (180.0, 0.0, [-1, 0, 0]),
        (270.0, 0.0, [0, -1, 0]),
        (0.0, 90.0, [0, 0, 1]),      # north celestial pole
        (123.0, 90.0, [0, 0, 1]),    # RA is irrelevant at the pole
        (0.0, -90.0, [0, 0, -1]),    # south celestial pole
        (45.0, 0.0, [np.sqrt(0.5), np.sqrt(0.5), 0]),
        (0.0, 45.0, [np.sqrt(0.5), 0, np.sqrt(0.5)]),
    ],
)
def test_radec_to_unit_known_directions(ra, dec, expected):
    v = radec_to_unit(ra, dec)
    assert isinstance(v, np.ndarray) and v.shape == (3,)
    np.testing.assert_allclose(v, expected, atol=TOL)


def test_radec_to_unit_always_unit_magnitude():
    rng = np.random.default_rng(0)
    for ra, dec in zip(rng.uniform(-720, 720, 500), rng.uniform(-90, 90, 500)):
        assert abs(np.linalg.norm(radec_to_unit(ra, dec)) - 1.0) < TOL


def test_radec_to_unit_rejects_bad_declination():
    with pytest.raises(ValueError):
        radec_to_unit(0.0, 90.1)


# ---------------------------------------------------------------- Function 2
def test_round_trip_random_directions():
    rng = np.random.default_rng(1)
    for ra, dec in zip(rng.uniform(0, 360, 1000), rng.uniform(-89.9, 89.9, 1000)):
        ra2, dec2 = unit_to_radec(radec_to_unit(ra, dec))
        # Compare RA on the circle so 359.999.. vs 0 can't cause a false fail.
        d_ra = (ra2 - ra + 180.0) % 360.0 - 180.0
        assert abs(d_ra) < 1e-9
        assert abs(dec2 - dec) < 1e-10


@pytest.mark.parametrize("ra", [0.0, 90.0, 180.0, 270.0, 359.999999])
def test_round_trip_quadrant_boundaries(ra):
    ra2, dec2 = unit_to_radec(radec_to_unit(ra, 10.0))
    assert ra2 == pytest.approx(ra, abs=1e-10)
    assert dec2 == pytest.approx(10.0, abs=1e-10)


@pytest.mark.parametrize(
    "ra_in, ra_wrapped", [(-90.0, 270.0), (360.0, 0.0), (450.0, 90.0), (-1.0, 359.0)]
)
def test_ra_wraps_into_0_360(ra_in, ra_wrapped):
    ra, _ = unit_to_radec(radec_to_unit(ra_in, 0.0))
    assert 0.0 <= ra < 360.0
    d_ra = (ra - ra_wrapped + 180.0) % 360.0 - 180.0
    assert abs(d_ra) < 1e-10


def test_negative_y_gives_ra_above_180():
    ra, dec = unit_to_radec([0.0, -1.0, 0.0])
    assert ra == pytest.approx(270.0)
    assert dec == pytest.approx(0.0)


@pytest.mark.parametrize("ra_in", [0.0, 77.0, 200.0])
def test_pole_convention_ra_is_zero(ra_in):
    assert unit_to_radec(radec_to_unit(ra_in, 90.0)) == (0.0, 90.0)
    assert unit_to_radec(radec_to_unit(ra_in, -90.0)) == (0.0, -90.0)


def test_unit_to_radec_normalizes_input():
    ra, dec = unit_to_radec([0.0, 5.0, 5.0])
    assert ra == pytest.approx(90.0)
    assert dec == pytest.approx(45.0)


def test_unit_to_radec_rejects_zero_and_bad_shape():
    with pytest.raises(ValueError):
        unit_to_radec([0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        unit_to_radec([1.0, 0.0])


# ---------------------------------------------------------------- Function 3
def test_separation_identical():
    v = radec_to_unit(37.0, -12.0)
    assert angular_separation(v, v) == pytest.approx(0.0, abs=1e-12)


def test_separation_perpendicular():
    assert angular_separation([1, 0, 0], [0, 1, 0]) == pytest.approx(90.0)
    assert angular_separation([1, 0, 0], [0, 0, 1]) == pytest.approx(90.0)


def test_separation_antiparallel():
    v = radec_to_unit(37.0, -12.0)
    assert angular_separation(v, -v) == pytest.approx(180.0)


def test_separation_along_equator_and_meridian():
    assert angular_separation(
        radec_to_unit(10.0, 0.0), radec_to_unit(55.0, 0.0)
    ) == pytest.approx(45.0)
    assert angular_separation(
        radec_to_unit(80.0, -20.0), radec_to_unit(80.0, 40.0)
    ) == pytest.approx(60.0)


def test_separation_is_symmetric():
    a, b = radec_to_unit(10.0, 20.0), radec_to_unit(200.0, -50.0)
    assert angular_separation(a, b) == pytest.approx(angular_separation(b, a))


def test_separation_resolves_tiny_angles():
    # 1 milliarcsecond: acos(dot) would return 0 here; atan2 does not.
    mas = 1.0 / 3.6e6
    sep = angular_separation(radec_to_unit(0.0, 0.0), radec_to_unit(mas, 0.0))
    assert sep == pytest.approx(mas, rel=1e-6)


def test_separation_rejects_zero_vector():
    with pytest.raises(ValueError):
        angular_separation([0, 0, 0], [1, 0, 0])


# ---------------------------------------------------------------- Function 4
def test_rotation_zero_elapsed_returns_initial_angle():
    assert earth_rotation_angle(0.0) == 0.0
    assert earth_rotation_angle(0.0, initial_angle_rad=1.25) == 1.25


def test_rotation_one_sidereal_day_is_two_pi():
    assert earth_rotation_angle(SIDEREAL_DAY_SECONDS) == pytest.approx(2 * np.pi)


def test_rotation_quarter_day_and_accumulation():
    assert earth_rotation_angle(SIDEREAL_DAY_SECONDS / 4) == pytest.approx(np.pi / 2)
    # Accumulated, not wrapped: three days is 6*pi, not 0.
    assert earth_rotation_angle(3 * SIDEREAL_DAY_SECONDS) == pytest.approx(6 * np.pi)


def test_rotation_solar_day_exceeds_one_turn():
    # In 24 h of solar time Earth turns ~360.9856 deg relative to the stars.
    deg = np.degrees(earth_rotation_angle(86400.0))
    assert deg == pytest.approx(360.9856, abs=1e-3)


def test_rotation_offset_is_additive():
    t = 12345.0
    assert earth_rotation_angle(t, 0.5) == pytest.approx(earth_rotation_angle(t) + 0.5)


# ---------------------------------------------------------------- Function 5
@pytest.mark.parametrize(
    "lat, dec",
    [(0.0, 20.0), (45.0, 20.0), (75.0, 20.0), (45.0, -30.0), (-35.0, 10.0)],
)
def test_altitude_upper_meridian_transit(lat, dec):
    # On the meridian (H = 0): h = 90 - |lat - dec|
    assert star_altitude(lat, dec, 0.0) == pytest.approx(90.0 - abs(lat - dec))


@pytest.mark.parametrize("lat", [-60.0, 0.0, 28.5, 45.0, 90.0])
def test_altitude_zenith_passage(lat):
    # dec == lat and H = 0: the star is exactly overhead.
    assert star_altitude(lat, lat, 0.0) == pytest.approx(90.0)


def test_altitude_lower_transit():
    # H = 180 deg: h = lat + dec - 90 (northern observer)
    assert star_altitude(45.0, 20.0, 180.0) == pytest.approx(-25.0)
    assert star_altitude(75.0, 20.0, 180.0) == pytest.approx(5.0)   # circumpolar


def test_altitude_of_pole_equals_latitude():
    for h in (0.0, 73.0, 180.0):
        assert star_altitude(52.0, 90.0, h) == pytest.approx(52.0)


def test_altitude_equatorial_star_on_horizon_at_six_hours():
    # dec = 0 stars rise/set at H = +/-90 deg from every latitude.
    assert star_altitude(40.0, 0.0, 90.0) == pytest.approx(0.0, abs=1e-12)
    assert star_altitude(40.0, 0.0, -90.0) == pytest.approx(0.0, abs=1e-12)


def test_altitude_symmetric_about_meridian():
    assert star_altitude(33.0, 15.0, 40.0) == pytest.approx(
        star_altitude(33.0, 15.0, -40.0)
    )


def test_altitude_at_equator_dec_zero_tracks_hour_angle():
    # Star on the celestial equator seen from the equator: h = 90 - |H|
    assert star_altitude(0.0, 0.0, 30.0) == pytest.approx(60.0)


def test_altitude_accepts_arrays():
    h = star_altitude(45.0, 20.0, np.array([0.0, 180.0]))
    np.testing.assert_allclose(h, [65.0, -25.0], atol=1e-10)


def test_altitude_agrees_with_vector_geometry():
    # Cross-check against the library's own vector functions: altitude is
    # 90 deg minus the separation between the star and the observer's zenith.
    # With local sidereal angle LST, zenith = (RA=LST, Dec=lat), H = LST - RA.
    lat, dec, ra, lst = 38.0, -17.0, 101.0, 152.0
    zenith = radec_to_unit(lst, lat)
    star = radec_to_unit(ra, dec)
    expected = 90.0 - angular_separation(zenith, star)
    assert star_altitude(lat, dec, lst - ra) == pytest.approx(expected)
