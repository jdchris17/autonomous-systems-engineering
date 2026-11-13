"""Tests for reference_frames.py. Run with:  python -m pytest -v"""

import astropy.units as u
import erfa
import numpy as np
import pytest
from astropy.coordinates import EarthLocation

from reference_frames import (
    WGS84_A,
    WGS84_B,
    celestial_to_earth_fixed,
    geodetic_to_ecef,
    rotation_z,
)

ANGLES = [0.0, 0.3, np.pi / 2, np.pi, -1.1, 2 * np.pi, 7.5, 1e-9]


# ---------------------------------------------------------------- Function B1
@pytest.mark.parametrize("theta", ANGLES)
def test_rotation_is_orthogonal(theta):
    R = rotation_z(theta)
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-15)


@pytest.mark.parametrize("theta", ANGLES)
def test_rotation_has_determinant_plus_one(theta):
    assert np.linalg.det(rotation_z(theta)) == pytest.approx(1.0, abs=1e-15)


def test_rotation_is_active_and_right_handed():
    # +90 deg about +z carries x to y and y to -x.
    R = rotation_z(np.pi / 2)
    np.testing.assert_allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-15)
    np.testing.assert_allclose(R @ [0, 1, 0], [-1, 0, 0], atol=1e-15)


def test_rotation_leaves_z_axis_alone():
    np.testing.assert_allclose(rotation_z(1.234) @ [0, 0, 1], [0, 0, 1])


def test_rotation_zero_is_identity_and_inverse_is_transpose():
    np.testing.assert_array_equal(rotation_z(0.0), np.eye(3))
    np.testing.assert_allclose(rotation_z(-0.7), rotation_z(0.7).T, atol=1e-16)


def test_rotations_compose_by_adding_angles():
    np.testing.assert_allclose(rotation_z(0.4) @ rotation_z(0.9),
                               rotation_z(1.3), atol=1e-15)


# ---------------------------------------------------------------- Function B2
def test_star_with_ra_equal_to_earth_angle_is_on_greenwich_meridian():
    theta = np.radians(73.0)
    star = [np.cos(theta), np.sin(theta), 0.0]          # RA = theta, Dec = 0
    np.testing.assert_allclose(celestial_to_earth_fixed(star, theta),
                               [1, 0, 0], atol=1e-15)


def test_earth_fixed_longitude_is_ra_minus_earth_angle():
    ra, dec, theta = np.radians(120.0), np.radians(30.0), np.radians(50.0)
    star = [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]
    x, y, z = celestial_to_earth_fixed(star, theta)
    assert np.degrees(np.arctan2(y, x)) == pytest.approx(70.0)
    assert np.degrees(np.arcsin(z)) == pytest.approx(30.0)   # dec unchanged


def test_stars_drift_westward_as_earth_angle_grows():
    star = [1.0, 0.0, 0.0]
    _, y, _ = celestial_to_earth_fixed(star, 0.01)
    assert y < 0.0


def test_matches_erfa_frame_rotation():
    # ERFA's Rz(psi) rotates the frame by +psi: the same passive convention.
    rng = np.random.default_rng(3)
    for theta in rng.uniform(-10, 10, 20):
        v = rng.normal(size=3)
        expected = erfa.rz(theta, np.eye(3)) @ v
        np.testing.assert_allclose(celestial_to_earth_fixed(v, theta),
                                   expected, atol=1e-14)


def test_transformation_preserves_length_and_pole():
    v = np.array([0.3, -0.4, 0.5])
    out = celestial_to_earth_fixed(v, 2.2)
    assert np.linalg.norm(out) == pytest.approx(np.linalg.norm(v))
    np.testing.assert_allclose(celestial_to_earth_fixed([0, 0, 1], 2.2),
                               [0, 0, 1])


def test_zero_angle_and_full_turn_are_identity():
    v = np.array([0.3, -0.4, 0.5])
    np.testing.assert_allclose(celestial_to_earth_fixed(v, 0.0), v)
    np.testing.assert_allclose(celestial_to_earth_fixed(v, 2 * np.pi), v,
                               atol=1e-15)


def test_accepts_a_stack_of_vectors():
    vs = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    out = celestial_to_earth_fixed(vs, 0.6)
    assert out.shape == (3, 3)
    for v, row in zip(vs, out):
        np.testing.assert_allclose(row, celestial_to_earth_fixed(v, 0.6))


def test_rejects_non_3_vectors():
    with pytest.raises(ValueError):
        celestial_to_earth_fixed([1.0, 0.0], 0.1)


# ---------------------------------------------------------------- Function B3
@pytest.mark.parametrize(
    "lat, lon, h, expected",
    [
        (0.0, 0.0, 0.0, [WGS84_A, 0, 0]),                 # equator, Greenwich
        (0.0, 90.0, 0.0, [0, WGS84_A, 0]),                # equator, 90 E
        (0.0, 180.0, 0.0, [-WGS84_A, 0, 0]),
        (0.0, -90.0, 0.0, [0, -WGS84_A, 0]),
        (90.0, 0.0, 0.0, [0, 0, WGS84_B]),                # north pole
        (-90.0, 0.0, 0.0, [0, 0, -WGS84_B]),              # south pole
        (90.0, 123.0, 0.0, [0, 0, WGS84_B]),              # lon irrelevant
        (0.0, 0.0, 1000.0, [WGS84_A + 1000.0, 0, 0]),     # height is radial
        (90.0, 0.0, 1000.0, [0, 0, WGS84_B + 1000.0]),
    ],
)
def test_geodetic_axes_and_poles(lat, lon, h, expected):
    np.testing.assert_allclose(geodetic_to_ecef(lat, lon, h), expected,
                               atol=1e-6)


def test_polar_radius_value():
    assert WGS84_B == pytest.approx(6356752.314245, abs=1e-6)


@pytest.mark.parametrize(
    "lat, lon, h, expected",
    [
        # Published WGS84 values (e.g. 45N on the prime meridian).
        (45.0, 0.0, 0.0, [4517590.8788, 0.0, 4487348.4088]),
        (45.0, 90.0, 0.0, [0.0, 4517590.8788, 4487348.4088]),
        (-45.0, 0.0, 0.0, [4517590.8788, 0.0, -4487348.4088]),
    ],
)
def test_geodetic_midlatitude_known_values(lat, lon, h, expected):
    np.testing.assert_allclose(geodetic_to_ecef(lat, lon, h), expected,
                               atol=1e-3)


def test_geodetic_matches_astropy_and_erfa_everywhere():
    rng = np.random.default_rng(7)
    for lat, lon, h in zip(rng.uniform(-90, 90, 200),
                           rng.uniform(-180, 180, 200),
                           rng.uniform(-500, 40000e3, 200)):
        mine = geodetic_to_ecef(lat, lon, h)
        loc = EarthLocation.from_geodetic(lon * u.deg, lat * u.deg, h * u.m,
                                          ellipsoid="WGS84")
        astro = np.array([loc.x.to_value(u.m), loc.y.to_value(u.m),
                          loc.z.to_value(u.m)])
        ref = erfa.gd2gc(1, np.radians(lon), np.radians(lat), h)   # 1 = WGS84
        np.testing.assert_allclose(mine, astro, atol=1e-6)
        np.testing.assert_allclose(mine, ref, atol=1e-6)


def test_height_moves_along_the_ellipsoid_normal():
    # The geodetic normal at latitude 60 is tilted 60 deg from the equator,
    # which is NOT the geocentric direction of the point.
    low = geodetic_to_ecef(60.0, 25.0, 0.0)
    high = geodetic_to_ecef(60.0, 25.0, 1000.0)
    step = high - low
    assert np.linalg.norm(step) == pytest.approx(1000.0, abs=1e-6)
    lat, lon = np.radians(60.0), np.radians(25.0)
    normal = [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    np.testing.assert_allclose(step / 1000.0, normal, atol=1e-9)


def test_geodetic_rejects_bad_latitude():
    with pytest.raises(ValueError):
        geodetic_to_ecef(91.0, 0.0, 0.0)
