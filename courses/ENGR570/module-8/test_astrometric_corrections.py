"""Tests for astrometric_corrections.py and atmospheric_refraction.py.
Run with:  python -m pytest -v"""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

import astrometric_corrections as ac
import atmospheric_refraction as ar

UTC = "2026-11-11T12:00:00"
TIME = Time(UTC, scale="utc")
LOCATION = ac.ObserverLocation(45.0, -75.0, 0.0)
ARCSEC = np.radians(1 / 3600)
ATMOSPHERE = ar.STANDARD_ATMOSPHERE

# A star that is well above the horizon at this place and time.
STAR = ac.CatalogStar(ra_deg=150.0, dec_deg=40.0, pm_ra_cosdec_mas_yr=350.0,
                      pm_dec_mas_yr=-220.0, parallax_mas=75.0,
                      radial_velocity_km_s=-20.0, epoch_jyear=2016.0, name="test")


def separation_arcsec(a, b):
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), a @ b)) * 3600


def astropy_star(star):
    return SkyCoord(ra=star.ra_deg * u.deg, dec=star.dec_deg * u.deg,
                    pm_ra_cosdec=star.pm_ra_cosdec_mas_yr * u.mas / u.yr,
                    pm_dec=star.pm_dec_mas_yr * u.mas / u.yr,
                    distance=(1000.0 / star.parallax_mas) * u.pc,
                    radial_velocity=star.radial_velocity_km_s * u.km / u.s,
                    obstime=Time(star.epoch_jyear, format="jyear", scale="tdb"),
                    frame="icrs")


def astropy_altaz(star, atmosphere):
    site = EarthLocation.from_geodetic(LOCATION.longitude_deg * u.deg,
                                       LOCATION.latitude_deg * u.deg,
                                       LOCATION.height_m * u.m)
    frame = AltAz(obstime=TIME, location=site,
                  pressure=atmosphere.pressure_hpa * u.hPa,
                  temperature=atmosphere.temperature_c * u.deg_C,
                  relative_humidity=atmosphere.relative_humidity,
                  obswl=atmosphere.wavelength_um * u.micron)
    # Astropy frame transforms do not apply proper motion by themselves.
    at_date = astropy_star(star).apply_space_motion(new_obstime=TIME)
    altaz = at_date.transform_to(frame)
    az, alt = altaz.az.rad, altaz.alt.rad
    return np.array([np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az),
                     np.sin(alt)])


# ------------------------------------------------------------------ refraction
def test_refraction_at_standard_conditions_has_textbook_values():
    assert ar.refraction_arcsec(45.0) == pytest.approx(58.0, abs=2.0)
    assert ar.refraction_arcsec(10.0) == pytest.approx(318.0, abs=12.0)
    assert ar.refraction_arcsec(90.0) == pytest.approx(0.0, abs=1e-9)
    values = ar.refraction_arcsec(np.array([10, 20, 30, 45, 60, 80.0]))
    assert np.all(np.diff(values) < 0)             # less refraction higher up


def test_refraction_agrees_with_an_independent_formula():
    altitudes = np.array([15.0, 20.0, 30.0, 45.0, 60.0, 80.0])
    atmosphere = ar.Atmosphere(pressure_hpa=1010.0, temperature_c=10.0)
    np.testing.assert_allclose(ar.refraction_arcsec(altitudes, atmosphere),
                               ar.saemundsson_refraction_arcsec(altitudes, atmosphere),
                               rtol=0.0, atol=9.0)   # its stated accuracy, ~0.1 arcmin


def test_refraction_scales_with_pressure_and_temperature():
    base = ar.refraction_arcsec(30.0)
    half = ar.refraction_arcsec(30.0, ar.Atmosphere(pressure_hpa=506.5))
    cold = ar.refraction_arcsec(30.0, ar.Atmosphere(temperature_c=-15.0))
    assert half / base == pytest.approx(0.5, abs=0.01)
    assert cold / base == pytest.approx(288.15 / 258.15, abs=0.01)
    assert ar.refraction_arcsec(30.0, ar.VACUUM) == 0.0


def test_simple_tan_formula_is_only_good_high_up():
    assert abs(ar.simple_tan_refraction_arcsec(60.0) - ar.refraction_arcsec(60.0)) < 0.1
    assert abs(ar.simple_tan_refraction_arcsec(10.0) - ar.refraction_arcsec(10.0)) > 3.0


def test_refract_and_unrefract_are_inverse_and_preserve_azimuth():
    rng = np.random.default_rng(0)
    az = rng.uniform(0, 2 * np.pi, 50)
    alt = np.radians(rng.uniform(8, 90, 50))
    vacuum = np.column_stack((np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az),
                              np.sin(alt)))
    observed = ar.refract_enu(vacuum, ATMOSPHERE)
    np.testing.assert_allclose(ar.unrefract_enu(observed, ATMOSPHERE), vacuum,
                               atol=1e-13)
    assert np.all(observed[:, 2] >= vacuum[:, 2])          # raised toward zenith
    np.testing.assert_allclose(np.arctan2(observed[:, 0], observed[:, 1]),
                               np.arctan2(vacuum[:, 0], vacuum[:, 1]), atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(observed, axis=1), 1.0)
    lift = np.degrees(np.arcsin(observed[:, 2]) - alt) * 3600
    np.testing.assert_allclose(lift, ar.refraction_arcsec(np.degrees(alt)),
                               atol=1e-6)


def test_zenith_star_and_single_vector_input():
    np.testing.assert_allclose(ar.refract_enu([0, 0, 1.0], ATMOSPHERE), [0, 0, 1])
    out = ar.refract_enu([0.0, 1.0, 1.0], ATMOSPHERE)
    assert out.shape == (3,) and out[2] > np.sqrt(0.5)


def test_refraction_is_refused_too_close_to_the_horizon():
    with pytest.raises(ValueError, match="horizon"):
        ar.refraction_arcsec(1.0)
    with pytest.raises(ValueError):
        ar.refract_enu([1.0, 0.0, 0.01], ATMOSPHERE)


def test_coefficients_match_erfa():
    import erfa
    a, b = erfa.refco(1013.0, 15.0, 0.5, 0.55)
    assert ar.refraction_coefficients(ATMOSPHERE) == pytest.approx((a, b))
    assert ar.refraction_coefficients(None) == (0.0, 0.0)
    assert b < 0 < a


# --------------------------------------------------------------- catalog chain
def test_catalog_direction_is_the_plain_ra_dec_vector():
    v = ac.catalog_directions([STAR])[0]
    ra, dec = np.radians(STAR.ra_deg), np.radians(STAR.dec_deg)
    np.testing.assert_allclose(v, [np.cos(dec) * np.cos(ra),
                                   np.cos(dec) * np.sin(ra), np.sin(dec)])


def test_propagation_matches_astropy_space_motion():
    moved = ac.propagate_catalog_star(STAR, TIME)
    reference = astropy_star(STAR).apply_space_motion(new_obstime=TIME.tdb)
    assert moved.epoch_jyear == pytest.approx(TIME.tdb.jyear)
    assert (moved.ra_deg - reference.ra.deg) * 3600 == pytest.approx(0, abs=1e-4)
    assert (moved.dec_deg - reference.dec.deg) * 3600 == pytest.approx(0, abs=1e-4)
    # ~10.86 years of 0.41 arcsec/year.
    shift = separation_arcsec(ac.catalog_directions([moved])[0],
                              ac.catalog_directions([STAR])[0])
    years = TIME.tdb.jyear - STAR.epoch_jyear
    assert shift == pytest.approx(np.hypot(350, 220) / 1000 * years, rel=1e-3)
    np.testing.assert_allclose(ac.propagated_directions([STAR], TIME)[0],
                               ac.catalog_directions([moved])[0], atol=1e-14)


def test_propagating_to_the_catalog_epoch_changes_nothing():
    same = ac.propagate_catalog_star(
        STAR, Time(STAR.epoch_jyear, format="jyear", scale="tdb"))
    assert same.ra_deg == pytest.approx(STAR.ra_deg, abs=1e-12)
    assert same.pm_dec_mas_yr == pytest.approx(STAR.pm_dec_mas_yr, abs=1e-9)


def test_stationary_star_is_unchanged_by_propagation():
    fixed = ac.CatalogStar(ra_deg=10.0, dec_deg=-30.0)
    np.testing.assert_allclose(ac.propagated_directions([fixed], TIME),
                               ac.catalog_directions([fixed]), atol=1e-15)


def test_topocentric_direction_matches_astropy_without_atmosphere():
    mine = ac.topocentric_star_direction(STAR, TIME, LOCATION)
    assert separation_arcsec(mine.enu, astropy_altaz(STAR, ar.VACUUM)) < 0.002
    assert "no atmosphere" in mine.frame and "GCRS-oriented" in mine.frame
    assert "annual aberration" in mine.corrections
    assert "atmospheric refraction" not in mine.corrections
    assert mine.altitude_deg > 20.0
    np.testing.assert_allclose(np.linalg.norm(mine.celestial), 1.0)


def test_observed_direction_matches_astropy_with_refraction():
    mine = ac.observed_star_direction(STAR, TIME, LOCATION, ATMOSPHERE)
    assert separation_arcsec(mine.enu, astropy_altaz(STAR, ATMOSPHERE)) < 0.002
    assert "atmospheric refraction" in mine.corrections
    assert mine.frame.startswith("topocentric observed")


def test_refraction_is_the_only_difference_and_is_applied_once():
    vacuum = ac.topocentric_star_direction(STAR, TIME, LOCATION)
    observed = ac.observed_star_direction(STAR, TIME, LOCATION, ATMOSPHERE)
    assert observed.azimuth_deg == pytest.approx(vacuum.azimuth_deg, abs=1e-9)
    lift = (observed.altitude_deg - vacuum.altitude_deg) * 3600
    assert lift == pytest.approx(ar.refraction_arcsec(vacuum.altitude_deg), abs=0.01)
    # ERFA's own refraction and this module's ENU helper agree, so applying
    # the helper to the airless direction reproduces the observed one.
    np.testing.assert_allclose(ar.refract_enu(vacuum.enu, ATMOSPHERE),
                               observed.enu, atol=1e-7)
    none = ac.observed_star_direction(STAR, TIME, LOCATION, ar.VACUUM)
    np.testing.assert_allclose(none.enu, vacuum.enu, atol=1e-15)


def test_aberration_dominates_the_apparent_place_shift():
    propagated = ac.propagated_directions([STAR], TIME)[0]
    apparent = ac.topocentric_star_direction(STAR, TIME, LOCATION).celestial
    shift = separation_arcsec(propagated, apparent)
    assert 1.0 < shift < 20.8          # annual aberration is at most 20.5 arcsec


def test_celestial_and_local_vectors_are_the_same_direction():
    direction = ac.observed_star_direction(STAR, TIME, LOCATION, ATMOSPHERE)
    matrix = ac.local_to_celestial_matrix(TIME, LOCATION)
    np.testing.assert_allclose(matrix @ direction.enu, direction.celestial,
                               atol=1e-15)
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-14)


def test_array_functions_match_single_star_functions():
    stars = [STAR, ac.CatalogStar(ra_deg=140.0, dec_deg=50.0, parallax_mas=10.0)]
    celestial, enu = ac.observed_directions(stars, TIME, LOCATION, ATMOSPHERE)
    assert celestial.shape == enu.shape == (2, 3)
    for star, c, e in zip(stars, celestial, enu):
        single = ac.observed_star_direction(star, TIME, LOCATION, ATMOSPHERE)
        np.testing.assert_allclose(c, single.celestial, atol=1e-15)
        np.testing.assert_allclose(e, single.enu, atol=1e-15)


def test_higher_observer_and_other_places_change_the_local_direction():
    here = ac.observed_star_direction(STAR, TIME, LOCATION, ATMOSPHERE)
    there = ac.observed_star_direction(STAR, TIME,
                                       ac.ObserverLocation(44.0, -75.0), ATMOSPHERE)
    assert abs(there.altitude_deg - here.altitude_deg) > 0.3


# ------------------------------------------------------------ inverse / catalog
def test_inverse_chain_round_trips():
    rng = np.random.default_rng(1)
    alt = np.radians(rng.uniform(30, 85, 10))
    az = rng.uniform(0, 2 * np.pi, 10)
    enu = np.column_stack((np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az),
                           np.sin(alt)))
    icrs = ac.icrs_from_observed_enu(enu, TIME, LOCATION, ATMOSPHERE)
    stars = [ac.CatalogStar(ra_deg=float(np.degrees(np.arctan2(v[1], v[0]))),
                            dec_deg=float(np.degrees(np.arcsin(v[2]))),
                            epoch_jyear=2000.0) for v in icrs]
    _, back = ac.observed_directions(stars, TIME, LOCATION, ATMOSPHERE)
    for a, b in zip(enu, back):
        assert separation_arcsec(a, b) < 1e-4


def test_synthetic_catalog_lands_in_the_camera_field():
    rng = np.random.default_rng(2)
    camera_axes = ac.ps.camera_axes_local(30.0, 40.0, 15.0)
    camera_vectors = np.array([[0.0, 0.0, 1.0], [0.1, 0.05, 1.0], [-0.12, 0.02, 1.0]])
    camera_vectors /= np.linalg.norm(camera_vectors, axis=1, keepdims=True)
    stars = ac.synthetic_catalog(camera_vectors, camera_axes, TIME, LOCATION, rng)
    assert len(stars) == 3 and all(s.epoch_jyear == 2016.0 for s in stars)
    assert any(abs(s.pm_ra_cosdec_mas_yr) > 1 for s in stars)
    _, enu = ac.topocentric_directions(stars, TIME, LOCATION)
    seen = enu @ camera_axes                       # ENU -> camera
    for a, b in zip(seen, camera_vectors):
        assert separation_arcsec(a, b) < 0.3       # parallax-sized offsets only
