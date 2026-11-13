"""Tests for astronomical_time.py. Run with:  python -m pytest -v"""

import erfa
import numpy as np
import pytest
from astropy.time import Time
from astropy.utils import iers
from astropy.utils.data import conf as data_conf

from astronomical_time import (
    EARTH_ROTATION_RATE_UT1,
    JD_J2000,
    TWO_PI,
    earth_rotation_angle_iau,
    iers_status,
    time_scales,
)

STAMP = "2026-11-11T12:00:00"


def _angle_diff(a, b):
    """Smallest signed difference between two angles [rad]."""
    return (a - b + np.pi) % TWO_PI - np.pi


# ------------------------------------------------------------ offline policy
def test_downloads_are_disabled():
    assert iers.conf.auto_download is False
    assert data_conf.allow_internet is False


def test_iers_data_is_available_locally_for_the_observation_time():
    status = iers_status(STAMP)
    assert status["available"]
    assert status["table_mjd_min"] < status["mjd"] < status["table_mjd_max"]
    assert "IERS" in status["source"]


@pytest.mark.filterwarnings("ignore:ERFA function")
def test_time_outside_local_iers_range_raises_instead_of_downloading():
    assert not iers_status("2060-01-01T00:00:00")["available"]
    with pytest.raises(iers.IERSRangeError):
        time_scales("2060-01-01T00:00:00")


# ---------------------------------------------------------------- Function A1
def test_utc_julian_date_of_the_observation_time():
    # 2026-11-11 12:00 UTC is exactly MJD 61355.5
    assert time_scales(STAMP).utc_jd == 2461356.0


def test_tt_minus_utc_is_tai_offset_plus_32_184():
    ts = time_scales(STAMP)
    seconds = ((ts.tt_jd_parts[0] - ts.utc_jd_parts[0])
               + (ts.tt_jd_parts[1] - ts.utc_jd_parts[1])) * 86400.0
    # 37 leap seconds (TAI-UTC since 2017) + 32.184 s (TT-TAI)
    assert seconds == pytest.approx(69.184, abs=1e-6)


def test_ut1_minus_utc_is_consistent_and_bounded():
    ts = time_scales(STAMP)
    seconds = ((ts.ut1_jd_parts[0] - ts.utc_jd_parts[0])
               + (ts.ut1_jd_parts[1] - ts.utc_jd_parts[1])) * 86400.0
    assert seconds == pytest.approx(ts.dut1_seconds, abs=1e-9)
    assert abs(ts.dut1_seconds) < 0.9          # UTC is steered to keep this


def test_matches_astropy_directly():
    t = Time(STAMP, scale="utc")
    ts = time_scales(STAMP)
    assert ts.utc_jd == t.utc.jd
    assert ts.ut1_jd == t.ut1.jd
    assert ts.tt_jd == t.tt.jd
    assert ts.dut1_seconds == float(t.delta_ut1_utc)
    assert ts.era_rad == float(erfa.era00(t.ut1.jd1, t.ut1.jd2))


def test_accepts_time_objects_in_other_scales():
    from_string = time_scales(STAMP)
    from_tt = time_scales(Time(STAMP, scale="utc").tt)
    assert from_tt.utc_jd == pytest.approx(from_string.utc_jd, abs=1e-10)
    assert _angle_diff(from_tt.era_rad, from_string.era_rad) == pytest.approx(
        0.0, abs=1e-10)


def test_two_part_jd_sums_to_single_jd():
    ts = time_scales(STAMP)
    assert sum(ts.ut1_jd_parts) == pytest.approx(ts.ut1_jd, abs=1e-9)
    assert 0.0 <= ts.era_rad < TWO_PI


# ---------------------------------------------------------------- Function A2
def test_era_at_j2000():
    expected = TWO_PI * 0.7790572732640          # 280.46061837 deg
    assert earth_rotation_angle_iau(JD_J2000) == pytest.approx(expected, abs=1e-14)
    assert np.degrees(earth_rotation_angle_iau(JD_J2000)) == pytest.approx(
        280.46061837504, abs=1e-9)


def test_era_matches_erfa_for_the_observation_time():
    jd1, jd2 = time_scales(STAMP).ut1_jd_parts
    assert abs(_angle_diff(earth_rotation_angle_iau(jd1, jd2),
                           erfa.era00(jd1, jd2))) < 1e-14


def test_era_matches_erfa_over_two_centuries():
    rng = np.random.default_rng(42)
    days = np.floor(rng.uniform(2415020.5, 2488070.5, 5000)) + 0.5
    fractions = rng.uniform(0.0, 1.0, 5000)
    for jd1, jd2 in ((days, fractions), (fractions, days)):   # either order
        diff = _angle_diff(earth_rotation_angle_iau(jd1, jd2),
                           erfa.era00(jd1, jd2))
        # Same two-part JD in, same angle out, to floating-point precision.
        assert np.max(np.abs(diff)) < 1e-14


def test_era_split_of_the_date_does_not_matter():
    a = earth_rotation_angle_iau(2461355.5, 0.25)
    b = earth_rotation_angle_iau(2400000.5, 61355.25)
    assert abs(_angle_diff(a, b)) < 1e-12


def test_single_float_jd_is_measurably_less_precise():
    # Why ERFA takes two parts: one float JD quantises time to ~40 us,
    # i.e. ~3e-9 rad of rotation, far above the two-part agreement.
    jd1, jd2 = 2461355.5, 0.123456789012345
    two_part = earth_rotation_angle_iau(jd1, jd2)
    one_part = earth_rotation_angle_iau(jd1 + jd2)
    error = abs(_angle_diff(one_part, two_part))
    assert 1e-12 < error < 1e-8


def test_era_advances_one_turn_plus_excess_per_ut1_day():
    a = earth_rotation_angle_iau(2461355.5, 0.3)
    b = earth_rotation_angle_iau(2461356.5, 0.3)
    excess_deg = np.degrees(_angle_diff(b, a))
    assert excess_deg == pytest.approx(360.0 * 0.00273781191135448, abs=1e-9)


def test_era_repeats_after_one_stellar_day():
    stellar_day = TWO_PI / EARTH_ROTATION_RATE_UT1 / 86400.0   # in UT1 days
    assert stellar_day * 86400.0 == pytest.approx(86164.0989, abs=1e-4)
    a = earth_rotation_angle_iau(2461355.5, 0.1)
    b = earth_rotation_angle_iau(2461355.5, 0.1 + stellar_day)
    assert abs(_angle_diff(a, b)) < 1e-10


def test_era_returns_scalar_or_array_in_range():
    assert isinstance(earth_rotation_angle_iau(2461355.5, 0.5), float)
    out = earth_rotation_angle_iau(np.array([2451545.0, 2461355.5]),
                                   np.array([0.0, 0.9]))
    assert out.shape == (2,)
    assert np.all((out >= 0.0) & (out < TWO_PI))
