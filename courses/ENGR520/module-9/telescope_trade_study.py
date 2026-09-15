"""module-9 -- telescope_trade_study.py  (Computational Exercise: Telescope/Camera Trade Study)

Aperture buys two things at once, not one:

    light-gathering area        A         = pi D^2 / 4        ~ D^2
    diffraction-limited angle   theta_min = 1.22 lambda / D    ~ 1/D

Every doubling of aperture: 4x the photons, HALF the blur, simultaneously.
D = 5, 10, 25, 50 mm at lambda = 550 nm (a camera-lens / small-telescope
range: D=5mm is roughly a phone-camera aperture, D=50mm a small finder scope).

Then the connection to the rest of this module: more collecting area is
*exactly* like the star getting brighter (magnitude_flux.py's relative_flux,
run backward). A bigger mirror doesn't just give "more signal" -- it buys a
quantified, calculable number of additional magnitudes of reach.
"""

import numpy as np
import matplotlib.pyplot as plt

D_MM = np.array([5.0, 10.0, 25.0, 50.0])
WAVELENGTH_NM = 550.0


def area_mm2(D_mm):
    return np.pi * D_mm**2 / 4.0


def diffraction_limit_rad(D_mm, wavelength_nm=WAVELENGTH_NM):
    return 1.22 * (wavelength_nm * 1e-9) / (D_mm * 1e-3)


def main():
    A = area_mm2(D_MM)
    theta_rad = diffraction_limit_rad(D_MM)
    theta_deg = np.degrees(theta_rad)
    theta_arcsec = theta_deg * 3600.0

    print("=" * 88)
    print(f"TELESCOPE/CAMERA TRADE STUDY   lambda = {WAVELENGTH_NM:.0f} nm")
    print("=" * 88)
    print(f"{'D (mm)':>7} {'A (mm^2)':>10} {'A (cm^2)':>10} "
          f"{'theta_min (rad)':>16} {'theta_min (deg)':>16} {'theta_min (arcsec)':>19}")
    for d, a, tr, td, ta in zip(D_MM, A, theta_rad, theta_deg, theta_arcsec):
        print(f"{d:>7.0f} {a:>10.2f} {a/100:>10.4f} {tr:>16.4e} "
              f"{td:>16.6f} {ta:>19.4f}")

    print()
    print("Photon-collection scaling and resolution scaling, both relative")
    print(f"to the D = {D_MM[0]:.0f} mm baseline:")
    print(f"{'D (mm)':>7} {'D/D0':>6} {'A/A0':>8} {'(D/D0)^2':>10} "
          f"{'theta/theta0':>13} {'D0/D':>8} {'delta_m_lim':>12}")
    A0, theta0, D0 = A[0], theta_rad[0], D_MM[0]
    for d, a, tr in zip(D_MM, A, theta_rad):
        dm = 5.0 * np.log10(d / D0)          # limiting-magnitude gain from area
        print(f"{d:>7.0f} {d/D0:>6.1f} {a/A0:>8.2f} {(d/D0)**2:>10.2f} "
              f"{tr/theta0:>13.4f} {D0/d:>8.4f} {dm:>+12.3f}")

    print()
    print("A/A0 matches (D/D0)^2 exactly, and theta/theta0 matches D0/D exactly")
    print("(both to floating-point precision) -- they are the same D, read two")
    print("ways.")
    print()
    print("The magnitude column is the double benefit made concrete: going")
    print(f"from D={D_MM[0]:.0f} mm to D={D_MM[-1]:.0f} mm is a {(D_MM[-1]/D0)**2:.0f}x gain in collecting")
    print(f"area -- via relative_flux(m)=10^(-0.4m) (magnitude_flux.py), that is")
    print(f"the same as every star becoming {5*np.log10(D_MM[-1]/D0):.2f} magnitudes brighter, so the")
    print(f"SAME exposure now reaches {5*np.log10(D_MM[-1]/D0):.2f} magnitudes fainter into the sky --")
    print(f"AND the diffraction limit tightens from {theta_arcsec[0]:.2f} arcsec to "
          f"{theta_arcsec[-1]:.2f} arcsec,")
    print(f"a {D_MM[-1]/D0:.0f}x sharper image, for the same physical change. Bigger glass")
    print("is not a single knob -- it turns two dials at once, in opposite")
    print("but equally valuable directions (more light in, less blur out).")
    print("=" * 88)

    _plot(D_MM, A, theta_arcsec)


def _plot(D_MM, A, theta_arcsec):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    ax1.loglog(D_MM, A, "o-", color="tab:blue", label="collecting area A (mm^2)")
    ax1b = ax1.twinx()
    ax1b.loglog(D_MM, theta_arcsec, "s-", color="tab:red", label="diffraction limit (arcsec)")
    ax1.set_xlabel("aperture D (mm)")
    ax1.set_ylabel("A (mm^2)", color="tab:blue")
    ax1b.set_ylabel("theta_min (arcsec)", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1b.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_title("Two log-log lines, opposite slopes\n"
                  "area rises as D^2, blur falls as D^-1 -- same D, both benefits")
    ax1.grid(True, which="both", alpha=0.3)
    lines = ax1.get_lines() + ax1b.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center left", fontsize=8)

    D0 = D_MM[0]
    dm = 5.0 * np.log10(D_MM / D0)
    ax2.plot(D_MM, dm, "D-", color="tab:green")
    for d, m in zip(D_MM, dm):
        ax2.annotate(f"+{m:.2f} mag", (d, m), textcoords="offset points",
                     xytext=(4, 6), fontsize=8)
    ax2.set_xlabel("aperture D (mm)")
    ax2.set_ylabel(f"limiting-magnitude gain over D={D0:.0f} mm")
    ax2.set_title("Same area gain, read as reach:\n"
                  "delta_m_lim = 5 log10(D/D0) = 2.5 log10(area ratio)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("telescope_trade_study.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
