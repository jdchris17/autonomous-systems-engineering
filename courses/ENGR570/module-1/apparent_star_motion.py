"""
apparent_star_motion.py - Space Systems Engineering, Module 1, Section 9

Propagate a star's hour angle across one sidereal day and plot its
geometric altitude for observers at three latitudes.

The star and the initial hour angle are identical for every run; only the
observer latitude changes.

Run:  python apparent_star_motion.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from celestial_geometry import (
    SIDEREAL_DAY_SECONDS,
    earth_rotation_angle,
    star_altitude,
)

# ---- experiment setup -------------------------------------------------------
STAR_DECLINATION_DEG = 20.0
INITIAL_HOUR_ANGLE_DEG = 180.0      # start at lower transit -> peak mid-run
LATITUDES_DEG = {
    "Equator (0°)": 0.0,
    "Midlatitude (45°N)": 45.0,
    "High latitude (75°N)": 75.0,
}
N_SAMPLES = 481                     # every ~3 minutes


def propagate_altitude(latitude_deg, declination_deg, initial_hour_angle_deg,
                       elapsed_seconds):
    """Altitude [deg] at each elapsed time. H(t) = H0 + accumulated rotation."""
    rotation_deg = np.degrees(earth_rotation_angle(elapsed_seconds))
    hour_angle_deg = initial_hour_angle_deg + rotation_deg
    return star_altitude(latitude_deg, declination_deg, hour_angle_deg)


def main():
    t = np.linspace(0.0, SIDEREAL_DAY_SECONDS, N_SAMPLES)
    t_hours = t / 3600.0

    fig, ax = plt.subplots(figsize=(10, 5.5))

    print(f"Star: dec = {STAR_DECLINATION_DEG:+.1f} deg, "
          f"initial hour angle = {INITIAL_HOUR_ANGLE_DEG:.1f} deg")
    print(f"{'observer':<24}{'min alt':>10}{'max alt':>10}{'hours up':>10}  note")

    for label, lat in LATITUDES_DEG.items():
        alt = propagate_altitude(lat, STAR_DECLINATION_DEG,
                                 INITIAL_HOUR_ANGLE_DEG, t)
        ax.plot(t_hours, alt, linewidth=2, label=label)

        hours_up = np.mean(alt[:-1] > 0.0) * SIDEREAL_DAY_SECONDS / 3600.0
        if alt.min() > 0.0:
            note = "circumpolar (never sets)"
        elif alt.max() < 0.0:
            note = "never rises"
        else:
            note = "rises and sets"
        print(f"{label:<24}{alt.min():>10.2f}{alt.max():>10.2f}"
              f"{hours_up:>10.2f}  {note}")

    ax.axhline(0.0, color="black", linewidth=1, linestyle="--", label="Horizon")
    ax.set_xlim(0.0, t_hours[-1])
    ax.set_ylim(-90, 90)
    ax.set_xticks(np.arange(0, 24, 2))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.set_xlabel("Elapsed time [hours] (one sidereal day = 23.934 h)")
    ax.set_ylabel("Geometric altitude [deg]")
    ax.set_title(f"Apparent stellar motion: dec = {STAR_DECLINATION_DEG:+.0f}°, "
                 f"initial hour angle = {INITIAL_HOUR_ANGLE_DEG:.0f}°")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center", ncol=4, fontsize=9)
    fig.tight_layout()

    out = Path(__file__).with_name("apparent_star_motion.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out.name}")


if __name__ == "__main__":
    main()
