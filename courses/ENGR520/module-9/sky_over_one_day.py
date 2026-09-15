"""module-9 -- sky_over_one_day.py  (Computational Exercise: The Sky Over One Day)

Pick an observer (latitude, longitude), a date, and several real stars
(RA/Dec). Sweep UT over 24 hours, compute each star's altitude(t) and
azimuth(t) from nothing but the coordinate transform in
spherical_astronomy.py, and watch rise / culminate / set / circumpolar /
never-rises fall out on their own -- nothing here branches on "if dec > 90-phi".
"""

import numpy as np
import matplotlib.pyplot as plt

from spherical_astronomy import (Star, julian_date, local_sidereal_time_degrees,
                                 hour_angle_degrees, equatorial_to_horizontal)

# ---------------------------------------------------------------------------
# observer + date
# ---------------------------------------------------------------------------
LAT_DEG = 40.0          # roughly New York City
LON_DEG = -74.0         # degrees EAST (so this is 74 W)
YEAR, MONTH, DAY = 2026, 9, 14

# a handful of real bright stars spanning the full range of behaviour at
# phi = 40 N: circumpolar, ordinary rise/set (high and low), and never-rises
STARS = [
    Star("Polaris", ra_hours=2.530, dec_deg=89.264),
    Star("Vega", ra_hours=18.615, dec_deg=38.784),
    Star("Betelgeuse", ra_hours=5.919, dec_deg=7.407),
    Star("Sirius", ra_hours=6.752, dec_deg=-16.716),
    Star("Alpha Centauri", ra_hours=14.660, dec_deg=-60.834),
]

N_SAMPLES = 1441                     # every minute over 24 h


def sky_path(star, ut_hours):
    jd = julian_date(YEAR, MONTH, DAY, hour=ut_hours)
    lst = local_sidereal_time_degrees(jd, LON_DEG)
    H = hour_angle_degrees(lst, star.ra_hours)
    alt, az = equatorial_to_horizontal(H, star.dec_deg, LAT_DEG)
    return alt, az


def find_crossings(t, alt):
    """Zero-crossing times of altitude(t), tagged 'rise' (neg->pos) or 'set'."""
    s = np.sign(alt)
    idx = np.where(np.diff(s) != 0)[0]
    events = []
    for i in idx:
        t0, t1, a0, a1 = t[i], t[i + 1], alt[i], alt[i + 1]
        tc = t0 - a0 * (t1 - t0) / (a1 - a0)
        events.append((tc, "rise" if a1 > a0 else "set"))
    return events


def main():
    ut = np.linspace(0.0, 24.0, N_SAMPLES)

    print("=" * 90)
    print("THE SKY OVER ONE DAY")
    print("=" * 90)
    print(f"observer: latitude {LAT_DEG:+.1f} deg, longitude {LON_DEG:+.1f} deg  "
          f"(east positive)")
    print(f"date: {YEAR}-{MONTH:02d}-{DAY:02d} UT, sampled every "
          f"{24*60/(N_SAMPLES-1):.1f} min")
    print()
    print("Nothing below tests 'if dec > 90-lat'. Every classification comes")
    print("from reading the computed altitude(t) array itself.")
    print()
    header = (f"{'star':>16} {'dec':>7} {'max alt':>8} {'min alt':>8} "
             f"{'hrs above horizon':>18}   classification")
    print(header)
    print("-" * len(header))

    results = {}
    for star in STARS:
        alt, az = sky_path(star, ut)
        results[star.name] = (alt, az)
        above = alt > 0.0
        hrs_above = above.mean() * 24.0

        if np.all(above):
            cls = "circumpolar (never sets)"
        elif not np.any(above):
            cls = "never rises"
        else:
            cls = "rises and sets"

        print(f"{star.name:>16} {star.dec_deg:>+7.2f} {alt.max():>8.2f} "
              f"{alt.min():>8.2f} {hrs_above:>18.2f}   {cls}")

    print()
    print("Cross-check against the textbook criterion (not used to produce the")
    print("numbers above -- only to confirm them): a star is circumpolar if")
    print("dec > 90-lat, and never rises if dec < -(90-lat).")
    limit = 90.0 - LAT_DEG
    for star in STARS:
        alt, _ = results[star.name]
        predicted = ("circumpolar (never sets)" if star.dec_deg > limit else
                    "never rises" if star.dec_deg < -limit else "rises and sets")
        observed = ("circumpolar (never sets)" if np.all(alt > 0) else
                   "never rises" if not np.any(alt > 0) else "rises and sets")
        match = "OK" if predicted == observed else "MISMATCH"
        print(f"   {star.name:>16}: dec={star.dec_deg:+.2f}, limit=+/-{limit:.2f}"
              f" -> predicted [{predicted}], computed [{observed}]  [{match}]")

    print()
    print("Rise / culmination / set times for the stars that do both:")
    for star in STARS:
        alt, az = results[star.name]
        events = find_crossings(ut, alt)
        t_cul = ut[np.argmax(alt)]
        alt_cul = alt.max()
        if events:
            ev_str = ", ".join(f"{kind} at {t:5.2f}h UT" for t, kind in events)
        else:
            ev_str = "(no rise/set in this window)"
        print(f"   {star.name:>16}: culminates {t_cul:5.2f}h UT at alt="
              f"{alt_cul:6.2f} deg;  {ev_str}")
    print()
    print("A 24 solar-hour window is slightly longer than one sidereal day (by")
    print("~3m56s), so an ordinary star can show a 3rd crossing near the very")
    print("end of the window -- it is starting its NEXT rise, not a bug.")
    print("=" * 90)

    _plot(ut, results)


def _plot(ut, results):
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0])
    ax_alt = fig.add_subplot(gs[0, 0])
    ax_az = fig.add_subplot(gs[1, 0])
    ax_polar = fig.add_subplot(gs[:, 1], projection="polar")

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(STARS)))
    for star, color in zip(STARS, colors):
        alt, az = results[star.name]
        ax_alt.plot(ut, alt, color=color, label=star.name)
        ax_az.plot(ut, az, ".", color=color, ms=1.5, label=star.name)

        # polar sky chart: radius = zenith distance (90-alt), angle = azimuth.
        # ax_polar.set_theta_zero_location("N") + set_theta_direction(-1)
        # (below) already make theta=azimuth plot as a proper compass --
        # north up, east clockwise -- so no extra 90-az transform is needed.
        above = alt > 0
        theta = np.deg2rad(az[above])
        r = 90.0 - alt[above]
        ax_polar.plot(theta, r, ".", color=color, ms=2, label=star.name)

    ax_alt.axhline(0, color="k", lw=1)
    ax_alt.set_xlabel("UT (hours)"); ax_alt.set_ylabel("altitude (deg)")
    ax_alt.set_title("Altitude over 24 h -- rise/set/circumpolar/never-rises,\n"
                     "all from the same formula, no special-casing")
    ax_alt.set_xlim(0, 24); ax_alt.legend(fontsize=8, loc="upper right")
    ax_alt.grid(True, alpha=0.3)

    ax_az.set_xlabel("UT (hours)"); ax_az.set_ylabel("azimuth (deg, 0=N,90=E)")
    ax_az.set_title("Azimuth over 24 h")
    ax_az.set_xlim(0, 24); ax_az.set_ylim(0, 360)
    ax_az.grid(True, alpha=0.3)

    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)          # clockwise, compass sense
    ax_polar.set_rlim(0, 90)
    ax_polar.set_rgrids([0, 30, 60, 90], labels=["90", "60", "30", "0 (horiz)"])
    ax_polar.set_title("Sky chart: each star's path across the dome\n"
                       "(centre = zenith, edge = horizon)", pad=20)
    ax_polar.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.05, 1.0))

    fig.suptitle(f"Sky over {YEAR}-{MONTH:02d}-{DAY:02d}, "
                f"observer at lat={LAT_DEG:+.1f}, lon={LON_DEG:+.1f}", fontsize=13)
    fig.tight_layout()
    fig.savefig("sky_over_one_day.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
