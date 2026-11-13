"""
astronomical_time.py - Space Systems Engineering, Module 2, Experiment A

Absolute astronomical time: UTC -> UT1 / TT, and the Earth Rotation Angle.

Astropy (astropy.time.Time) and ERFA are the authoritative references here.
The only astronomy implemented by hand is the IAU 2000 ERA formula
(Function A2), which is validated against erfa.era00.

Offline policy
--------------
UT1 needs Earth-orientation (IERS) data. Importing this module calls
configure_offline_iers(), which stops Astropy from downloading anything, so
results are reproducible and tests never touch the network. Astropy then
falls back to the IERS tables shipped in the astropy-iers-data package.
Use iers_status() to see whether a timestamp is covered by measured values,
predictions, or nothing at all.
"""

from dataclasses import dataclass

import erfa
import numpy as np
from astropy.time import Time
from astropy.utils import iers
from astropy.utils.data import conf as data_conf

TWO_PI = 2.0 * np.pi
JD_J2000 = 2451545.0
SECONDS_PER_DAY = 86400.0

# IAU 2000 Earth Rotation Angle (Capitaine et al. 2000; IERS Conventions 2010
# eq. 5.15):  ERA = 2*pi * (ERA_J2000_TURNS + ERA_TURNS_PER_UT1_DAY * Tu),
# with Tu = JD(UT1) - 2451545.0.
ERA_J2000_TURNS = 0.7790572732640
ERA_TURNS_PER_UT1_DAY = 1.00273781191135448
ERA_EXCESS_TURNS_PER_UT1_DAY = 0.00273781191135448   # the part beyond 1 turn/day

# Earth's rotation rate implied by the ERA definition [rad per UT1 second].
EARTH_ROTATION_RATE_UT1 = TWO_PI * ERA_TURNS_PER_UT1_DAY / SECONDS_PER_DAY

_IERS_STATUS_TEXT = {
    iers.FROM_IERS_B: "IERS-B final (measured)",
    iers.FROM_IERS_A: "IERS-A rapid (measured)",
    iers.FROM_IERS_A_PREDICTION: "IERS-A prediction",
    iers.TIME_BEFORE_IERS_RANGE: "before table range (no data)",
    iers.TIME_BEYOND_IERS_RANGE: "beyond table range (no data)",
}


def configure_offline_iers():
    """
    Forbid Astropy from fetching Earth-orientation or leap-second data.

    * iers.conf.auto_download = False  -> never fetch a newer IERS-A table
    * data conf.allow_internet = False -> block every astropy.utils.data
                                          download, as a second safety net

    With these set, Astropy uses the bundled astropy-iers-data tables.

    Caution: with auto_download off, Astropy does NOT raise for a timestamp
    beyond the end of the table; it quietly reuses the last tabulated
    UT1-UTC. time_scales() therefore checks coverage itself and raises.
    """
    iers.conf.auto_download = False
    data_conf.allow_internet = False


configure_offline_iers()


def iers_status(utc):
    """
    Report which local Earth-orientation data covers a UTC timestamp.

    Returns a dict with the table type, its MJD coverage, whether the
    timestamp is inside it, and whether UT1-UTC at that time is a measured
    value or a prediction.
    """
    t = _as_utc_time(utc)
    table = iers.earth_orientation_table.get()
    _, status = table.ut1_utc(t, return_status=True)
    status = int(status)
    return {
        "table": type(table).__name__,
        "auto_download": iers.conf.auto_download,
        "allow_internet": data_conf.allow_internet,
        "mjd": float(t.mjd),
        "table_mjd_min": float(table["MJD"][0].value),
        "table_mjd_max": float(table["MJD"][-1].value),
        "available": status in (iers.FROM_IERS_B, iers.FROM_IERS_A,
                                iers.FROM_IERS_A_PREDICTION),
        "is_prediction": status == iers.FROM_IERS_A_PREDICTION,
        "source": _IERS_STATUS_TEXT.get(status, f"unknown status {status}"),
    }


@dataclass(frozen=True)
class TimeScales:
    """Result of time_scales(). Julian Dates are in days, angles in radians."""
    utc_jd: float
    ut1_jd: float
    tt_jd: float
    dut1_seconds: float          # UT1 - UTC
    era_rad: float               # Earth Rotation Angle, [0, 2*pi)
    # Two-part Julian Dates (jd1 + jd2). A single float JD only resolves
    # ~40 microseconds; carry these when precision matters.
    utc_jd_parts: tuple
    ut1_jd_parts: tuple
    tt_jd_parts: tuple
    iers_source: str


def _as_utc_time(utc):
    if isinstance(utc, Time):
        return utc.utc
    return Time(utc, scale="utc")


def time_scales(utc):
    """
    Function A1 - convert an absolute UTC timestamp to the other time scales.

    `utc` is an ISO string such as "2026-11-11T12:00:00" (interpreted as
    UTC) or an astropy Time. Returns a TimeScales with the UTC, UT1 and TT
    Julian Dates, UT1-UTC in seconds, and the Earth Rotation Angle in
    radians computed by erfa.era00 from the two-part UT1 Julian Date.

    Raises astropy's IERSRangeError if no local Earth-orientation data
    covers the timestamp. Nothing is downloaded to fill the gap, and the
    last tabulated value is not silently reused.
    """
    t = _as_utc_time(utc)
    status = iers_status(t)
    if not status["available"]:
        raise iers.IERSRangeError(
            f"no local Earth-orientation data for MJD {status['mjd']:.2f} "
            f"({status['source']}); table covers MJD "
            f"{status['table_mjd_min']:.0f} to {status['table_mjd_max']:.0f} "
            "and downloads are disabled")
    ut1 = t.ut1
    tt = t.tt
    return TimeScales(
        utc_jd=float(t.jd),
        ut1_jd=float(ut1.jd),
        tt_jd=float(tt.jd),
        dut1_seconds=float(t.delta_ut1_utc),
        era_rad=float(erfa.era00(ut1.jd1, ut1.jd2)),
        utc_jd_parts=(float(t.jd1), float(t.jd2)),
        ut1_jd_parts=(float(ut1.jd1), float(ut1.jd2)),
        tt_jd_parts=(float(tt.jd1), float(tt.jd2)),
        iers_source=status["source"],
    )


def earth_rotation_angle_iau(ut1_jd1, ut1_jd2=0.0):
    """
    Function A2 - IAU 2000 Earth Rotation Angle from a UT1 Julian Date.

    The date is ut1_jd1 + ut1_jd2 (any split; pass 0.0 for the second part
    if you only have a single JD). Returns radians in [0, 2*pi).

    Precision: ERA = 2*pi * (0.779... + 1.00273781191135448 * Tu) evaluated
    naively multiplies a ~1e4-day number by 2*pi and then discards ~1e4 whole
    turns, losing the low bits. Instead, as ERFA does:
      * the "1 turn per day" part is taken from the fractional days of the
        two JD parts separately, so whole days never enter the arithmetic;
      * only the small excess rate (0.0027 turn/day) multiplies the full Tu.
    Given the same two-part JD this agrees with erfa.era00 to the last bit
    or two.
    """
    jd1 = np.asarray(ut1_jd1, dtype=float)
    jd2 = np.asarray(ut1_jd2, dtype=float)

    # Smaller-magnitude part first, so (large - J2000) is exact-ish.
    small = np.where(jd1 < jd2, jd1, jd2)
    large = np.where(jd1 < jd2, jd2, jd1)

    days_since_j2000 = small + (large - JD_J2000)
    day_fraction = np.fmod(small, 1.0) + np.fmod(large, 1.0)

    turns = (day_fraction + ERA_J2000_TURNS
             + ERA_EXCESS_TURNS_PER_UT1_DAY * days_since_j2000)

    theta = np.fmod(TWO_PI * turns, TWO_PI)
    theta = np.where(theta < 0.0, theta + TWO_PI, theta)
    return float(theta) if theta.ndim == 0 else theta


if __name__ == "__main__":
    stamp = "2026-11-11T12:00:00"
    status = iers_status(stamp)
    ts = time_scales(stamp)
    mine = earth_rotation_angle_iau(*ts.ut1_jd_parts)

    print(f"Observation time (UTC): {stamp}")
    print(f"IERS table: {status['table']}, MJD {status['table_mjd_min']:.0f}"
          f" to {status['table_mjd_max']:.0f}; auto_download="
          f"{status['auto_download']}")
    print(f"UT1-UTC source at MJD {status['mjd']:.1f}: {status['source']}")
    print(f"UTC JD : {ts.utc_jd:.9f}")
    print(f"UT1 JD : {ts.ut1_jd:.9f}")
    print(f"TT  JD : {ts.tt_jd:.9f}")
    print(f"UT1-UTC: {ts.dut1_seconds:+.7f} s")
    tt_minus_utc = ((ts.tt_jd_parts[0] - ts.utc_jd_parts[0])
                    + (ts.tt_jd_parts[1] - ts.utc_jd_parts[1])) * 86400
    print(f"TT-UTC : {tt_minus_utc:+.3f} s")
    print(f"ERA (erfa.era00): {ts.era_rad:.15f} rad"
          f" = {np.degrees(ts.era_rad):.10f} deg")
    print(f"ERA (own formula): {mine:.15f} rad,"
          f" difference {mine - ts.era_rad:+.2e} rad")
    naive = earth_rotation_angle_iau(ts.ut1_jd)
    print(f"ERA from single-float JD: differs by {naive - ts.era_rad:+.2e} rad"
          f" ({np.degrees(naive - ts.era_rad) * 3.6e6:+.3f} mas)")
