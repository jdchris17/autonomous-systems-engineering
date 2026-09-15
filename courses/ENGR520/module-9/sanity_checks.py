"""module-9 -- sanity_checks.py  (Known sanity checks for the coordinate-transform system)

Five predictions, made from geometry/physics alone, BEFORE any number is
computed -- then checked against frames.py's R_EC/R_LE chain.

  1. North celestial pole altitude  ~= observer latitude phi (northern obs.)
  2. At the equator (phi=0), the celestial poles sit on the horizon.
  3. At the north pole (phi=90), the NCP sits at zenith.
  4. Meridian transit (H=0): the star lies on the observer's meridian
     (azimuth = 0 deg if it culminates north of zenith, 180 deg if south).
  5. Zenith passage: a star with declination = phi, observed at H=0,
     passes through the zenith (altitude = 90 deg).
"""

import numpy as np

from frames import star_celestial, star_earth_fixed, star_local, altaz_from_local
from spherical_astronomy import julian_date

JD0 = julian_date(2026, 9, 14, hour=12.0)   # an arbitrary but fixed instant


def altaz(ra_deg, dec_deg, lat_deg, lon_deg, jd=JD0):
    s_c = star_celestial(ra_deg, dec_deg)
    s_e = star_earth_fixed(s_c, jd)
    s_l = star_local(s_e, lat_deg, lon_deg)
    return altaz_from_local(s_l)


def check(label, predicted, measured, tol, unit="deg", wrap360=False):
    diff = abs(predicted - measured)
    if wrap360:
        diff = min(diff, 360.0 - diff)          # 0 and 360 are the same azimuth
    ok = diff < tol
    print(f"  [{'OK' if ok else 'FAIL'}] {label}")
    print(f"        predicted {predicted:+.4f} {unit},  measured {measured:+.4f} {unit},  "
          f"diff {diff:.2e}")
    return ok


def main():
    print("=" * 82)
    print("SANITY CHECKS -- predict from geometry first, THEN run the code")
    print("=" * 82)
    all_ok = True

    # ---- 1: NCP altitude ~= phi, for a few northern latitudes -----------
    print("\n1. North celestial pole altitude ~= observer latitude phi")
    print("   (the NCP is essentially fixed at dec=90; H is undefined there,")
    print("   but harmless -- cos(dec)=0 kills any H-dependence in the formula)")
    for lat in (10.0, 40.0, 60.0, 89.0):
        alt, az = altaz(ra_deg=0.0, dec_deg=90.0, lat_deg=lat, lon_deg=0.0)
        ok = check(f"lat = {lat} deg", predicted=lat, measured=alt, tol=1e-9)
        all_ok &= ok

    # ---- 2: at the equator, celestial poles on the horizon ----------------
    print("\n2. At the equator (phi=0), BOTH celestial poles sit on the horizon")
    for dec, name in [(90.0, "north celestial pole"), (-90.0, "south celestial pole")]:
        alt, az = altaz(ra_deg=123.0, dec_deg=dec, lat_deg=0.0, lon_deg=0.0)
        ok = check(f"{name}, phi=0", predicted=0.0, measured=alt, tol=1e-9)
        all_ok &= ok

    # ---- 3: at the north pole, the NCP is at zenith ------------------------
    print("\n3. At the north pole (phi=90), the NCP sits at zenith")
    alt, az = altaz(ra_deg=77.0, dec_deg=90.0, lat_deg=90.0, lon_deg=0.0)
    all_ok &= check("NCP altitude at phi=90", predicted=90.0, measured=alt, tol=1e-9)

    # ---- 4: meridian transit (H=0) -> on the meridian (az = 0 or 180) -----
    print("\n4. Meridian transit (H=0): star lies on the meridian (az=0 if it")
    print("   culminates north of zenith i.e. dec>phi, az=180 if south, dec<phi)")
    lat = 40.0
    for dec, expected_az, tag in [(70.0, 0.0, "dec > phi -> culminates north of zenith"),
                                  (10.0, 180.0, "dec < phi -> culminates south of zenith")]:
        # H = 0  <=>  LST = RA; pick RA = LST at JD0 for this longitude directly
        from spherical_astronomy import local_sidereal_time_degrees
        lst = local_sidereal_time_degrees(JD0, lon_east_deg=0.0)
        alt, az = altaz(ra_deg=lst, dec_deg=dec, lat_deg=lat, lon_deg=0.0)
        print(f"   {tag}")
        all_ok &= check("azimuth at H=0", predicted=expected_az, measured=az,
                        tol=1e-7, wrap360=True)

    # ---- 5: a star with dec=phi, at H=0, passes through the zenith -------
    print("\n5. Zenith passage: dec = phi, observed at H=0 -> altitude = 90 deg")
    print("   (loose tolerance here on purpose: altitude = arcsin(Up), and")
    print("   d(arcsin)/dx -> infinity as x -> 1, so ordinary ~1e-16 floating-")
    print("   point noise in Up gets amplified to ~sqrt(noise) in the angle --")
    print("   about 1e-8 rad ~ 3 milliarcsec here. Not a bug: the zenith is a")
    print("   genuine ill-conditioned point of the (alt, az) chart, the same")
    print("   flavour of issue as azimuth being undefined there.)")
    from spherical_astronomy import local_sidereal_time_degrees
    for lat in (15.0, 40.0, 65.0):
        lst = local_sidereal_time_degrees(JD0, lon_east_deg=0.0)
        alt, az = altaz(ra_deg=lst, dec_deg=lat, lat_deg=lat, lon_deg=0.0)
        all_ok &= check(f"dec=phi={lat} deg at H=0", predicted=90.0, measured=alt, tol=1e-5)

    print()
    print("=" * 82)
    print("ALL CHECKS PASSED" if all_ok else "AT LEAST ONE CHECK FAILED")
    print("These are exactly the checks Section 34's pipeline leans on before")
    print("trusting it with a real catalog: get the degenerate/special cases")
    print("right by hand first, because a subtle sign error usually still")
    print("produces a plausible-looking, completely wrong picture.")
    print("=" * 82)


if __name__ == "__main__":
    main()
