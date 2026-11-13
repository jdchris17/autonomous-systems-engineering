"""
Test suite 2 - independent astronomical verification.

Each physical transformation the engine relies on is compared with a
separately written or separately validated implementation, and one
end-to-end test feeds the engine observations it had no part in creating
(tests/independent_reference.py: Astropy frames, scipy rotations, closed
formulas). Tolerances and their justification are in tolerances.py.
"""

import erfa
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import (
    GCRS,
    ITRS,
    CartesianRepresentation,
    EarthLocation,
    SkyCoord,
)
from astropy.time import Time
from scipy.spatial.transform import Rotation

import independent_reference as ind
from enduria import (
    Atmosphere,
    CameraModel,
    NavigationConfig,
    ReferenceData,
    solve_navigation_fix,
)
from enduria.attitude import wahba_svd
from enduria.frames import geodetic_to_ecef
from enduria.geodesy import (
    camera_axes_local,
    earth_fixed_up_to_geodetic,
    horizontal_position_error,
)
from enduria.gravity import normal_gravity
from enduria.timekeeping import EARTH_ROTATION_RATE_UT1, reference_matrix, time_scales
from helpers import position_error_m
from tolerances import SEED, tolerance

MAS = np.radians(1.0 / 3.6e6)
TIMES = ["2024-03-20T03:06:00", "2025-07-04T18:30:15.250", "2026-11-11T12:00:00"]
SITES = [(45.0, -75.0, 0.0), (-33.9, 151.2, 50.0), (64.1, -21.9, 0.0),
         (1.3, 103.8, 15.0), (-77.8, 166.7, 2800.0)]


def _angle(a, b):
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1),
                      np.einsum("...i,...i->...", a, b))


# ----------------------------------------------- astronomical time, Earth rotation
@pytest.mark.parametrize("utc", TIMES)
def test_time_scales_against_known_offsets(utc):
    record = ReferenceData({}).eop(utc)
    tt_minus_utc = ((record.tt_jd[0] - record.utc_jd[0])
                    + (record.tt_jd[1] - record.utc_jd[1])) * 86400.0
    # 37 leap seconds since 2017-01-01, plus TT - TAI = 32.184 s.
    assert tt_minus_utc == pytest.approx(69.184, abs=1e-6)
    assert abs(record.dut1_s) < 0.9
    assert record.dut1_s == pytest.approx(time_scales(utc).dut1_seconds, abs=1e-12)


@pytest.mark.parametrize("utc", TIMES)
def test_earth_rotation_angle_three_ways(utc):
    scales = time_scales(utc)
    by_formula = ind.earth_rotation_angle(*scales.ut1_jd_parts)
    by_astropy = Time(utc, scale="utc").earth_rotation_angle(0 * u.deg).rad
    for other in (by_formula, by_astropy):
        difference = (scales.era_rad - other + np.pi) % (2 * np.pi) - np.pi
        assert abs(difference) < 1e-10          # 1e-10 rad = 0.6 mm at the equator


@pytest.mark.parametrize("utc", TIMES)
def test_celestial_to_terrestrial_rotation_three_ways(utc):
    engine = ReferenceData({}).eop(utc).earth_from_celestial()
    module2 = reference_matrix(Time(utc, scale="utc"))          # erfa.c2t06a
    time = Time(utc, scale="utc")
    basis = SkyCoord(CartesianRepresentation(np.eye(3)), frame=GCRS(obstime=time))
    astropy_matrix = basis.transform_to(ITRS(obstime=time)).cartesian.xyz.value
    limit = tolerance("earth_rotation_rad")
    assert np.abs(engine - module2).max() < limit
    assert np.abs(engine - astropy_matrix).max() < limit
    assert np.allclose(engine @ engine.T, np.eye(3), atol=1e-14)


def test_earth_turns_at_the_sidereal_rate():
    refs = ReferenceData({})
    first = refs.eop("2026-11-11T12:00:00").earth_from_celestial()
    later = refs.eop("2026-11-11T12:10:00").earth_from_celestial()
    turned = Rotation.from_matrix(later @ first.T).magnitude()
    assert turned == pytest.approx(EARTH_ROTATION_RATE_UT1 * 600.0, rel=1e-7)


# ------------------------------------------- catalog propagation, apparent places
def _engine_and_astropy_stars(site, utc, atmosphere, seed):
    lat, lon, height = site
    time = Time(utc, scale="utc", precision=9)
    location = EarthLocation.from_geodetic(lon * u.deg, lat * u.deg, height * u.m)
    rng = np.random.default_rng(seed)
    altitude = np.radians(rng.uniform(15.0, 88.0, 40))
    azimuth = rng.uniform(0.0, 2 * np.pi, 40)
    enu = np.column_stack((np.cos(altitude) * np.sin(azimuth),
                           np.cos(altitude) * np.cos(azimuth), np.sin(altitude)))
    ids, coordinates, catalog = ind.make_catalog(enu, time, location, rng, "T")
    references = ReferenceData(catalog)
    index, _ = references.indices(ids)
    return references, index, references.eop(time), coordinates, time, location


@pytest.mark.parametrize("utc", TIMES)
def test_catalog_propagation_against_astropy(utc):
    references, index, eop, coordinates, time, _ = _engine_and_astropy_stars(
        SITES[0], utc, None, SEED + 20)
    moved = coordinates.apply_space_motion(new_obstime=time)
    astropy_unit = moved.cartesian.xyz.value.T
    astropy_unit /= np.linalg.norm(astropy_unit, axis=1, keepdims=True)
    difference = _angle(references.propagated(index, eop), astropy_unit)
    assert difference.max() / MAS < 0.01
    # ... and the propagation is not a no-op: ten years of 100 mas/yr.
    start = coordinates.cartesian.xyz.value.T
    start /= np.linalg.norm(start, axis=1, keepdims=True)
    assert _angle(references.propagated(index, eop), start).max() / MAS > 500.0


@pytest.mark.parametrize("site", SITES)
@pytest.mark.parametrize("atmosphere", [None, Atmosphere(1005.0, 8.0, 0.4, 0.55)])
def test_observed_directions_against_astropy(site, atmosphere):
    references, index, eop, coordinates, time, location = \
        _engine_and_astropy_stars(site, TIMES[2], atmosphere, SEED + 21)
    _, engine_enu = references.observed(index, eop, site[0], site[1], site[2],
                                        atmosphere)
    astropy_enu = ind.observed_enu(coordinates, time, location, atmosphere)
    difference = _angle(engine_enu, astropy_enu) / MAS
    assert difference.max() < tolerance("independent_direction_mas")


def test_observed_celestial_and_local_directions_are_one_rotation_apart():
    references, index, eop, *_ = _engine_and_astropy_stars(SITES[1], TIMES[1],
                                                           None, SEED + 22)
    celestial, enu = references.observed(index, eop, *SITES[1][:2])
    gram_celestial = celestial @ celestial.T
    assert np.allclose(gram_celestial, enu @ enu.T, atol=1e-13)


# ----------------------------------------------------- camera-frame transformations
def test_camera_model_against_written_out_pinhole():
    camera = CameraModel()
    assert camera.fx == pytest.approx(ind.FOCAL_PX, rel=1e-14)
    rng = np.random.default_rng(SEED + 23)
    pixels = np.column_stack((rng.uniform(0, ind.WIDTH, 200),
                              rng.uniform(0, ind.HEIGHT, 200)))
    rays = ind.pixel_to_ray(pixels)
    assert np.abs(camera.unproject(pixels) - rays).max() < 1e-14
    projected, visible = camera.project(rays)
    assert visible.all() and np.abs(projected - pixels).max() < 1e-9


@pytest.mark.parametrize("zenith, azimuth, roll", [(0, 0, 0), (30, 40, 15),
                                                   (65, 250, 300), (10, 180, 90)])
def test_simulator_camera_axes_are_a_rotation_with_the_right_boresight(zenith, azimuth,
                                                                       roll):
    axes = camera_axes_local(zenith, azimuth, roll)
    assert np.allclose(axes.T @ axes, np.eye(3), atol=1e-13)
    assert np.linalg.det(axes) == pytest.approx(1.0)
    z, az = np.radians(zenith), np.radians(azimuth)
    expected = [np.sin(z) * np.sin(az), np.sin(z) * np.cos(az), np.cos(z)]
    assert np.allclose(axes[:, 2], expected, atol=1e-13)


def test_attitude_solver_against_scipy():
    rng = np.random.default_rng(SEED + 24)
    truth = Rotation.random(random_state=7).as_matrix()               # C_CG
    reference = rng.standard_normal((25, 3))
    reference /= np.linalg.norm(reference, axis=1, keepdims=True)
    measured = reference @ truth.T + 2e-5 * rng.standard_normal((25, 3))
    measured /= np.linalg.norm(measured, axis=1, keepdims=True)
    engine = wahba_svd(reference, measured)
    scipy_fit, _ = Rotation.align_vectors(measured, reference)
    assert Rotation.from_matrix(engine @ scipy_fit.as_matrix().T).magnitude() < 1e-12


# ------------------------------------------------------- terrestrial geodetic geometry
@pytest.mark.parametrize("lat, lon, height", SITES + [(0.0, 0.0, 0.0),
                                                      (89.999, 10.0, 0.0)])
def test_geodetic_geometry_three_ways(lat, lon, height):
    module2 = np.asarray(geodetic_to_ecef(lat, lon, height))
    written_out = ind.geodetic_to_ecef(lat, lon, height)
    location = EarthLocation.from_geodetic(lon * u.deg, lat * u.deg, height * u.m)
    astropy_xyz = np.array([location.x.to_value(u.m), location.y.to_value(u.m),
                            location.z.to_value(u.m)])
    limit = tolerance("geodesy_m")
    assert np.abs(module2 - written_out).max() < limit
    assert np.abs(module2 - astropy_xyz).max() < limit
    # The engine recovers geodetic coordinates from the ellipsoid normal;
    # here the normal comes from the gradient of the ellipsoid equation.
    normal = ind.ellipsoid_normal_from_ecef(ind.geodetic_to_ecef(lat, lon, 0.0))
    recovered = earth_fixed_up_to_geodetic(normal)
    assert recovered[0] == pytest.approx(lat, abs=1e-11)
    assert recovered[1] == pytest.approx(lon, abs=1e-8)


@pytest.mark.parametrize("latitude", [0.0, 45.0, 90.0, -90.0])
def test_normal_gravity_two_formulas(latitude):
    # Somigliana (engine) against the 1980 International Gravity Formula.
    assert normal_gravity(latitude) == pytest.approx(ind.gravity_1980(latitude),
                                                     abs=2e-5)


# ----------------------------------------------------------- position-distance
def test_distance_against_vincenty():
    rng = np.random.default_rng(SEED + 25)
    for _ in range(200):
        lat1, lon1 = rng.uniform(-85, 85), rng.uniform(-180, 180)
        scale = 10.0 ** rng.uniform(-7, 0.5)                 # 1 cm to 350 km
        lat2 = float(np.clip(lat1 + scale * rng.normal(), -89.0, 89.0))
        lon2 = lon1 + scale * rng.normal()
        engine = horizontal_position_error((lat1, lon1), (lat2, lon2))
        vincenty = ind.vincenty_inverse(lat1, lon1, lat2, lon2)
        assert engine == pytest.approx(
            vincenty, rel=tolerance("distance_relative"), abs=2e-4)


def test_distance_against_known_ellipsoid_lengths():
    # WGS84 meridian quadrant and one degree of the equator.
    assert horizontal_position_error((0.0, 30.0), (90.0, 30.0)) == pytest.approx(
        10001965.729, abs=1e-3)
    assert horizontal_position_error((0.0, 10.0), (0.0, 11.0)) == pytest.approx(
        ind.A * np.radians(1.0), abs=1e-6)
    assert horizontal_position_error((0.0, 179.5), (0.0, -179.5)) == pytest.approx(
        ind.A * np.radians(1.0), abs=1e-6)


# -------------------------------------------------- independent end-to-end tests
ATTITUDES = [(20.0, 15.0, 0.0), (45.0, 140.0, 70.0), (35.0, 290.0, 200.0)]


@pytest.mark.parametrize("site", SITES)
@pytest.mark.parametrize("atmosphere", [None, Atmosphere(990.0, -5.0, 0.3, 0.55)])
def test_engine_locates_independently_generated_observations(site, atmosphere):
    lat, lon, height = site
    batch, catalog = ind.independent_batch(lat, lon, ATTITUDES, SEED + 26,
                                           height_m=height, atmosphere=atmosphere)
    config = NavigationConfig(
        camera=CameraModel(), C_CI=ind.C_CI, centroid_sigma_px=1e-3,
        accel_noise_sigma=1e-6, apply_refraction=atmosphere is not None,
        atmosphere=atmosphere, refraction_fraction_sigma=0.0, height_m=height,
        gravity_tolerance=0.05)
    result = solve_navigation_fix(batch, config, ReferenceData(catalog))
    assert result.has_fix, result.summary()
    assert position_error_m(result, (lat, lon)) < tolerance("independent_position_m")


def test_independent_single_exposure_and_refraction_matters():
    site = SITES[0]
    atmosphere = Atmosphere(1013.0, 15.0, 0.5, 0.55)
    batch, catalog = ind.independent_batch(site[0], site[1], [(50.0, 100.0, 30.0)],
                                           SEED + 27, atmosphere=atmosphere)
    base = dict(camera=CameraModel(), C_CI=ind.C_CI, centroid_sigma_px=1e-3,
                accel_noise_sigma=1e-6, refraction_fraction_sigma=0.0,
                gravity_tolerance=0.05)
    right = solve_navigation_fix(batch, NavigationConfig(atmosphere=atmosphere, **base),
                                 ReferenceData(catalog))
    assert position_error_m(right, site[:2]) < tolerance("independent_position_m")
    # The independent data really contain refraction. Ignoring it is caught
    # when the centroids are declared precise (the differential refraction
    # across the field does not fit), and costs kilometres when they are not.
    strict = solve_navigation_fix(
        batch, NavigationConfig(apply_refraction=False, **base),
        ReferenceData(catalog))
    assert not strict.has_fix
    loose = solve_navigation_fix(
        batch, NavigationConfig(apply_refraction=False,
                                **{**base, "centroid_sigma_px": 0.5}),
        ReferenceData(catalog))
    assert position_error_m(loose, site[:2]) > 1000.0
