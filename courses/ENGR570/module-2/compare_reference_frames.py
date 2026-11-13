"""
compare_reference_frames.py - Space Systems Engineering, Module 2, Experiment C

Take one catalog direction (ICRS RA/Dec) and one UTC timestamp, express the
direction in the Earth-fixed frame with models of increasing fidelity, and
measure the angle between the results.

Models
------
A0  Module 1 rotation: constant *sidereal* rate (2*pi / 86164.0905 s),
    anchored to the ERA at J2000.0, counted in UTC. Rotation about z only.
A   Simple rotation: IAU ERA formula (constant rate, J2000 anchor) fed with
    UTC as if it were UT1. Rotation about z only.
B   Intermediate: ERA from true UT1, plus the celestial-pole treatment
    (IAU 2006/2000A bias-precession-nutation, CIO based: erfa.c2i06a).
    No polar motion.
C   Reference: full IAU 2006/2000A GCRS -> ITRS with IERS UT1-UTC and polar
    motion. Built two ways that must agree: directly from ERFA
    (erfa.c2t06a) and through Astropy's frame graph (GCRS -> ITRS).

What is being compared
----------------------
A0, A, B and C all rotate the SAME vector, the star's ICRS catalog
direction, so the A/B-versus-C numbers are pure frame-rotation differences.

The course's starting snippet, SkyCoord(icrs).transform_to(ITRS(obstime)),
is a different quantity: Astropy first converts the catalog direction to a
geocentric apparent direction (annual aberration and gravitational light
deflection by the Sun), then rotates it. That result is reported separately
as "C-apparent", and its offset from C is an apparent-place correction, not
a frame-rotation error.

Run:  python compare_reference_frames.py
"""

import sys
from pathlib import Path

import astropy.units as u
import erfa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, SkyCoord
from astropy.time import Time
from astropy.utils import iers

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "module-1"))
from celestial_geometry import (  # noqa: E402
    EARTH_ROTATION_RATE as MODULE1_SIDEREAL_RATE,
    angular_separation,
    radec_to_unit,
    unit_to_radec,
)

from astronomical_time import (  # noqa: E402
    ERA_J2000_TURNS,
    JD_J2000,
    SECONDS_PER_DAY,
    TWO_PI,
    EARTH_ROTATION_RATE_UT1,
    earth_rotation_angle_iau,
    iers_status,
    time_scales,
)
from reference_frames import celestial_to_earth_fixed  # noqa: E402

OBSERVATION_UTC = "2026-11-11T12:00:00"
STAR_RA_DEG = 120.0
STAR_DEC_DEG = 30.0

ARCSEC_PER_DEG = 3600.0
RAD_TO_ARCSEC = np.degrees(1.0) * ARCSEC_PER_DEG
EARTH_RADIUS_M = 6378137.0


def polar_motion(t):
    """IERS polar-motion coordinates (xp, yp) of the pole at time t [rad]."""
    xp, yp = iers.earth_orientation_table.get().pm_xy(t)
    return float(xp.to_value(u.rad)), float(yp.to_value(u.rad))


# ----------------------------------------------------------------- the models
def model_a0_module1_rate(vector, t):
    """Module 1's constant sidereal rate, anchored at J2000.0, run on UTC."""
    ts = time_scales(t)
    days = (ts.utc_jd_parts[0] - JD_J2000) + ts.utc_jd_parts[1]
    angle = (TWO_PI * ERA_J2000_TURNS
             + MODULE1_SIDEREAL_RATE * SECONDS_PER_DAY * days)
    return celestial_to_earth_fixed(vector, angle)


def model_a_simple_rotation(vector, t):
    """ERA formula evaluated on the UTC Julian Date; rotation about z only."""
    ts = time_scales(t)
    angle = earth_rotation_angle_iau(*ts.utc_jd_parts)
    return celestial_to_earth_fixed(vector, angle)


def model_b_intermediate(vector, t):
    """True-UT1 ERA plus bias-precession-nutation; no polar motion."""
    ts = time_scales(t)
    celestial_to_intermediate = erfa.c2i06a(*ts.tt_jd_parts)   # GCRS -> CIRS
    angle = earth_rotation_angle_iau(*ts.ut1_jd_parts)
    return celestial_to_earth_fixed(celestial_to_intermediate @ vector, angle)


def reference_matrix(t):
    """Full GCRS -> ITRS rotation matrix (IAU 2006/2000A + IERS EOP)."""
    ts = time_scales(t)
    xp, yp = polar_motion(Time(t, scale="utc") if isinstance(t, str) else t)
    return erfa.c2t06a(*ts.tt_jd_parts, *ts.ut1_jd_parts, xp, yp)


def model_c_reference(vector, t):
    """Reference frame rotation, straight from ERFA."""
    return reference_matrix(t) @ vector


def model_c_astropy_frame_only(vector, t):
    """
    The same rotation through Astropy: declare the vector to be a GCRS
    direction (GCRS axes are aligned with ICRS), so that no aberration or
    light deflection is applied, then transform GCRS -> ITRS.
    """
    time = Time(t, scale="utc") if isinstance(t, str) else t
    gcrs = SkyCoord(CartesianRepresentation(vector * u.one),
                    frame=GCRS(obstime=time))
    xyz = gcrs.transform_to(ITRS(obstime=time)).cartesian.xyz.value
    return xyz / np.linalg.norm(xyz)


def astropy_apparent(ra_deg, dec_deg, t):
    """
    The course's starting snippet: ICRS catalog position -> ITRS. Returns
    (earth_fixed_direction, geocentric_apparent_direction_in_GCRS).
    """
    time = Time(t, scale="utc") if isinstance(t, str) else t
    star = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    itrs = star.transform_to(ITRS(obstime=time)).cartesian.xyz.value
    gcrs = star.transform_to(GCRS(obstime=time)).cartesian.xyz.value
    return itrs / np.linalg.norm(itrs), gcrs / np.linalg.norm(gcrs)


# ------------------------------------------------------------------ reporting
def sep_arcsec(a, b):
    return angular_separation(a, b) * ARCSEC_PER_DEG


def main():
    t = Time(OBSERVATION_UTC, scale="utc")
    ts = time_scales(t)
    status = iers_status(t)
    star = radec_to_unit(STAR_RA_DEG, STAR_DEC_DEG)

    a0 = model_a0_module1_rate(star, t)
    a = model_a_simple_rotation(star, t)
    b = model_b_intermediate(star, t)
    c = model_c_reference(star, t)
    c_astropy = model_c_astropy_frame_only(star, t)
    c_apparent, apparent_gcrs = astropy_apparent(STAR_RA_DEG, STAR_DEC_DEG, t)

    print(f"Star (ICRS): RA = {STAR_RA_DEG} deg, Dec = {STAR_DEC_DEG} deg")
    print(f"Time (UTC):  {OBSERVATION_UTC}")
    print(f"UT1-UTC = {ts.dut1_seconds:+.4f} s  [{status['source']}]")
    xp, yp = polar_motion(t)
    print(f"Polar motion xp, yp = {xp * RAD_TO_ARCSEC:+.4f}, "
          f"{yp * RAD_TO_ARCSEC:+.4f} arcsec\n")

    print("Earth-fixed direction (longitude, latitude of the sub-stellar "
          "point, degrees):")
    rows = [("A0 Module 1 sidereal rate", a0), ("A  simple rotation", a),
            ("B  intermediate", b), ("C  reference (ERFA)", c),
            ("C  reference (Astropy GCRS->ITRS)", c_astropy),
            ("C-apparent (Astropy ICRS->ITRS)", c_apparent)]
    for label, v in rows:
        lon, lat = unit_to_radec(v)
        lon = (lon + 180.0) % 360.0 - 180.0
        print(f"  {label:<36}{lon:>14.7f}{lat:>13.7f}")

    # ---- 1. frame-rotation differences (same input vector everywhere) ------
    print("\n1. Frame-rotation differences from the reference C "
          "(same catalog vector rotated):")
    frame_rows = [
        ("A0 Module 1 sidereal rate", sep_arcsec(a0, c)),
        ("A  simple rotation", sep_arcsec(a, c)),
        ("B  intermediate", sep_arcsec(b, c)),
        ("C  ERFA vs Astropy", sep_arcsec(c, c_astropy)),
    ]
    for label, arcsec in frame_rows:
        ground = np.radians(arcsec / ARCSEC_PER_DEG) * EARTH_RADIUS_M
        print(f"  {label:<28}{arcsec:>14.6f} arcsec   "
              f"(~{ground:,.3f} m on the ground)")

    # ---- 2. what each neglected effect is worth ----------------------------
    x_cip, y_cip, _ = erfa.xys06a(*ts.tt_jd_parts)
    pole_offset = np.hypot(x_cip, y_cip) * RAD_TO_ARCSEC
    years = (ts.tt_jd - JD_J2000) / 365.25
    cos_dec = np.cos(np.radians(STAR_DEC_DEG))

    rate_gap = MODULE1_SIDEREAL_RATE - EARTH_ROTATION_RATE_UT1
    elapsed_s = ((ts.utc_jd_parts[0] - JD_J2000) + ts.utc_jd_parts[1]) * 86400
    a0_angle_excess = rate_gap * elapsed_s * RAD_TO_ARCSEC
    dut1_angle = EARTH_ROTATION_RATE_UT1 * ts.dut1_seconds * RAD_TO_ARCSEC
    pm_size = np.hypot(xp, yp) * RAD_TO_ARCSEC

    # A with true UT1 isolates the missing pole treatment from the DUT1 part.
    a_with_ut1 = celestial_to_earth_fixed(
        star, earth_rotation_angle_iau(*ts.ut1_jd_parts))

    print(f"\n2. Size of each neglected effect ({years:.2f} yr after J2000):")
    print(f"  Sidereal-vs-stellar rate, accumulated   {a0_angle_excess:>12.3f}"
          f" arcsec of rotation angle ({a0_angle_excess / 3600:.4f} deg)")
    print(f"  Celestial pole offset from ICRS z axis  {pole_offset:>12.3f}"
          f" arcsec (precession + nutation of the pole)")
    print(f"     -> effect on this star (A with true UT1 vs B) "
          f"{sep_arcsec(a_with_ut1, b):>9.3f} arcsec")
    print(f"  UT1-UTC as rotation angle               {dut1_angle:>12.3f}"
          f" arcsec ({abs(dut1_angle) * cos_dec:.3f} on this star)")
    print(f"     -> effect on this star (A vs A with true UT1) "
          f"{sep_arcsec(a, a_with_ut1):>9.3f} arcsec")
    print(f"  Polar motion                            {pm_size:>12.3f}"
          f" arcsec (B vs C: {sep_arcsec(b, c):.3f})")

    # ---- 3. apparent place: a different quantity, not a rotation error -----
    _, earth_bary = erfa.epv00(*ts.tt_jd_parts)        # [pos, vel], au/day
    velocity = earth_bary[1]
    speed_over_c = np.linalg.norm(velocity) / erfa.DC   # DC = c in au/day
    apex_angle = angular_separation(star, velocity)
    predicted_aberration = (np.degrees(speed_over_c
                                       * np.sin(np.radians(apex_angle)))
                            * ARCSEC_PER_DEG)

    print("\n3. Apparent-place correction (Astropy ICRS -> ITRS versus the "
          "frame-only reference):")
    print(f"  C-apparent vs C (Earth-fixed)           "
          f"{sep_arcsec(c_apparent, c):>12.3f} arcsec")
    print(f"  Catalog vs apparent direction (GCRS)    "
          f"{sep_arcsec(apparent_gcrs, star):>12.3f} arcsec")
    print(f"  First-order aberration, (v/c) sin(angle to Earth's velocity "
          f"= {apex_angle:.1f} deg): {predicted_aberration:.3f} arcsec")
    print(f"  Rotating the apparent direction with the reference matrix "
          f"reproduces C-apparent to "
          f"{sep_arcsec(reference_matrix(t) @ apparent_gcrs, c_apparent):.2e}"
          f" arcsec")

    print(f"""
Physical reading
----------------
* A0 uses the sidereal day (86164.0905 s), the rotation period relative to
  the precessing equinox. The ICRS axes do not precess, and relative to them
  Earth turns once per 86164.0989 s. The 8.4 ms per turn accumulates to
  {a0_angle_excess / 3600:.3f} deg since J2000, which is essentially the precession of the
  equinox in right ascension (plus ~5 arcsec from the rounded period). A sidereal angle is only correct together with a
  precession matrix; used alone on an ICRS vector it is the largest error.
* A uses the ERA, which is measured from a non-rotating origin, so its
  rotation about the pole is right apart from UT1-UTC ({dut1_angle:+.2f} arcsec of
  angle). What it lacks is the pole itself: Earth's spin axis has precessed
  and nutated {pole_offset:.0f} arcsec ({pole_offset / 3600:.3f} deg) away from the ICRS z axis since
  J2000, and A rotates about the wrong axis. That is a tilt, so the error
  depends on where the star is relative to the direction of the tilt.
* B rotates about the true celestial pole with true UT1. Only polar motion
  remains: the crust is offset from the rotation axis by {pm_size:.2f} arcsec,
  about {np.radians(pm_size / 3600) * EARTH_RADIUS_M:.0f} m at the surface.
* C from ERFA and C from Astropy are the same model and agree to numerical
  noise.
* C-apparent differs from C by {sep_arcsec(c_apparent, c):.1f} arcsec, but not because of any
  rotation. Earth's {speed_over_c * erfa.CMPS / 1000:.1f} km/s orbital velocity tilts incoming starlight
  (annual aberration, up to 20.5 arcsec), plus milliarcseconds of solar
  light deflection. It belongs to the direction, not the frame.
""")

    # ---- figure -------------------------------------------------------------
    labels = ["A0: Module 1\nsidereal rate", "A: simple\nrotation (ERA)",
              "B: intermediate\n(no polar motion)",
              "C-apparent\n(aberration, not\na frame error)"]
    values = [sep_arcsec(a0, c), sep_arcsec(a, c), sep_arcsec(b, c),
              sep_arcsec(c_apparent, c)]
    colors = ["tab:red", "tab:orange", "tab:green", "tab:gray"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_yscale("log")
    ax.set_ylim(0.01, 10 * max(values))
    ax.set_ylabel("Angular difference from reference C [arcsec]")
    ax.set_title(f"Earth-fixed direction of RA {STAR_RA_DEG:.0f}°, "
                 f"Dec {STAR_DEC_DEG:.0f}° at {OBSERVATION_UTC} UTC")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.15,
                f'{value:,.2f}"', ha="center", va="bottom")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    out = Path(__file__).with_name("compare_reference_frames.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out.name}")


if __name__ == "__main__":
    main()
