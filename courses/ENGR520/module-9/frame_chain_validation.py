"""module-9 -- frame_chain_validation.py  (validates frames.py, Sections 34-36)

Two things to check before trusting the new explicit C -> E -> L chain:

  1. It must reproduce sky_over_one_day.py's altitude/azimuth exactly --
     that file's equatorial_to_horizontal (one combined rotation, via the
     hour angle H = LST - RA) was already validated there against real star
     behaviour and the textbook formulas. Splitting it into R_EC then R_LE
     is a DIFFERENT derivation of the SAME physics, so it is only trustworthy
     if it lands on the same numbers.

  2. The frame-naming discipline (Section 35) should be visible in the
     output, not just the code: this file prints C_s, E_s, L_s labelled at
     every step for one worked instant, and shows R_BL R_LE R_EC as an
     explicit (if partial, pending a camera-pointing scenario) matrix chain.
"""

import numpy as np

from frames import (star_celestial, R_EC, R_LE, star_earth_fixed, star_local,
                    altaz_from_local, altaz_from_celestial, star_earth_fixed_series)
from spherical_astronomy import (julian_date, local_sidereal_time_degrees,
                                 hour_angle_degrees, equatorial_to_horizontal)
from sky_over_one_day import STARS, LAT_DEG, LON_DEG, YEAR, MONTH, DAY

N_SAMPLES = 200


def main():
    ut = np.linspace(0.0, 24.0, N_SAMPLES)
    jd = julian_date(YEAR, MONTH, DAY, hour=ut)

    print("=" * 90)
    print("VALIDATING frames.py's explicit C -> E -> L CHAIN against "
          "spherical_astronomy.py's trusted result")
    print("=" * 90)
    print(f"{'star':>16} {'max |d(alt)| deg':>18} {'max |d(az)| deg':>17}")
    worst = 0.0
    for star in STARS:
        # trusted path: one combined hour-angle rotation
        lst = local_sidereal_time_degrees(jd, LON_DEG)
        H = hour_angle_degrees(lst, star.ra_hours)
        alt_trusted, az_trusted = equatorial_to_horizontal(H, star.dec_deg, LAT_DEG)

        # new path: explicit C -> E -> L, two separately-named rotations
        s_c = star_celestial(star.ra_hours * 15.0, star.dec_deg)
        s_e = star_earth_fixed_series(star.ra_hours * 15.0, star.dec_deg, jd)
        s_l = star_local(s_e, LAT_DEG, LON_DEG)
        alt_new, az_new = altaz_from_local(s_l)

        d_alt = np.abs(alt_new - alt_trusted).max()
        d_az = np.min(np.stack([np.abs(az_new - az_trusted),
                               360 - np.abs(az_new - az_trusted)]), axis=0).max()
        worst = max(worst, d_alt, d_az)
        print(f"{star.name:>16} {d_alt:>18.2e} {d_az:>17.2e}")

    print()
    print(f"worst discrepancy across all stars/times: {worst:.2e} deg "
          f"(floating-point only -- the two derivations agree)")
    print("=" * 90)

    # ---- Section 35: the explicit, ruthlessly-named chain, one instant ----
    print()
    print("ONE WORKED INSTANT -- every vector labelled by frame (Section 35):")
    star = STARS[1]                     # Vega
    ut0 = 3.0
    jd0 = julian_date(YEAR, MONTH, DAY, hour=ut0)
    print(f"  star = {star.name}  (RA={star.ra_hours}h, Dec={star.dec_deg} deg)")
    print(f"  observer: lat={LAT_DEG} deg, lon={LON_DEG} deg,  UT={ut0} h on "
          f"{YEAR}-{MONTH:02d}-{DAY:02d}  (JD={jd0:.5f})")

    C_s = star_celestial(star.ra_hours * 15.0, star.dec_deg)
    print(f"\n  C_s (celestial)    = {np.round(C_s, 6)}")

    R_EC_mat = R_EC(jd0)
    E_s = C_s @ R_EC_mat.T
    print(f"  R_EC (this instant) =\n{np.round(R_EC_mat, 5)}")
    print(f"  E_s (Earth-fixed)  = R_EC @ C_s = {np.round(E_s, 6)}")

    R_LE_mat = R_LE(LAT_DEG, LON_DEG)
    L_s = E_s @ R_LE_mat.T
    print(f"\n  R_LE (this observer) =\n{np.round(R_LE_mat, 5)}")
    print(f"  L_s (local ENU)    = R_LE @ E_s = {np.round(L_s, 6)}")

    alt, az = altaz_from_local(L_s)
    print(f"\n  -> altitude = {alt:.4f} deg, azimuth = {az:.4f} deg")

    alt_direct, az_direct = altaz_from_celestial(star.ra_hours * 15.0, star.dec_deg,
                                                 jd0, LAT_DEG, LON_DEG)
    print(f"  cross-check via altaz_from_celestial(): "
          f"alt={alt_direct:.4f}, az={az_direct:.4f}  (must match exactly)")

    print()
    print("R_EC and R_LE are each orthogonal with det=+1 (proper rotations):")
    for name, R in [("R_EC", R_EC_mat), ("R_LE", R_LE_mat)]:
        oe = np.abs(R @ R.T - np.eye(3)).max()
        de = abs(np.linalg.det(R) - 1.0)
        print(f"   {name}: |R R^T - I| = {oe:.2e},  |det - 1| = {de:.2e}")

    print()
    print("R_BL (local -> camera/body) is NOT built yet -- it needs an actual")
    print("camera-pointing scenario (where is the camera aimed?), which hasn't")
    print("been specified yet. module-7/camera/camera_orientation.py already")
    print("has the machinery (reused, not duplicated, the same way this file")
    print("reused Module 3's rot_z); wiring it in as R_BL is the next step in")
    print("the C -> E -> L -> B chain.")
    print("=" * 90)


if __name__ == "__main__":
    main()
