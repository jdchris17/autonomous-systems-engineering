"""module-6 -- resolution.py  (Small Computational Experiment: Resolution)

Diffraction sets the angular resolution of any circular aperture (Rayleigh
criterion):

    theta_min = 1.22 * lambda / D            (radians)

Sweep a few wavelengths and apertures, convert theta_min to degrees,
arcminutes and arcseconds, then run the question the other way: given a
target angular separation (a binary star, a feature on a planet, a satellite
image), what aperture D would you need to just resolve it?
"""

import numpy as np
import matplotlib.pyplot as plt

WAVELENGTHS = {"450 nm (blue)": 450e-9, "550 nm (green)": 550e-9,
              "700 nm (red)": 700e-9}

# real apertures, metres -- human eye pupil up to an extremely large telescope
APERTURES = {"human eye (5 mm)": 0.005, "amateur scope (10 cm)": 0.10,
            "amateur scope (25 cm)": 0.25, "Hubble (2.4 m)": 2.4,
            "JWST (6.5 m)": 6.5, "ELT (39.3 m)": 39.3}

# targets to resolve, arcseconds -- used to invert the formula for D
TARGETS_ARCSEC = {
    "typical visual binary star (1.0 arcsec)": 1.0,
    "Pluto surface feature from Earth (0.10 arcsec)": 0.10,
    "directly-imaged exoplanet separation (0.05 arcsec)": 0.05,
    "stellar-surface imaging (1 milliarcsec)": 1e-3,
    "10 cm license plate digit from 400 km orbit": None,   # computed below
}
# angular size of a 0.10 m feature at 400 km (LEO): theta = size / distance
TARGETS_ARCSEC["10 cm license plate digit from 400 km orbit"] = \
    np.degrees(0.10 / 400e3) * 3600


def theta_min_rad(wavelength, D):
    return 1.22 * wavelength / D


def rad_to_units(theta_rad):
    deg = np.degrees(theta_rad)
    return deg, deg * 60.0, deg * 3600.0        # deg, arcmin, arcsec


def required_aperture(wavelength, theta_target_arcsec):
    theta_rad = np.deg2rad(theta_target_arcsec / 3600.0)
    return 1.22 * wavelength / theta_rad


def main():
    print("=" * 84)
    print("RAYLEIGH RESOLUTION   theta_min = 1.22 lambda / D")
    print("=" * 84)

    print("\n1. theta_min (arcsec) for every (wavelength, aperture) pair:")
    header = f"{'aperture':>24}" + "".join(f"{k:>16}" for k in WAVELENGTHS)
    print(header)
    for name, D in APERTURES.items():
        row = f"{name:>24}"
        for lam in WAVELENGTHS.values():
            _, _, arcsec = rad_to_units(theta_min_rad(lam, D))
            row += f"{arcsec:>16.4f}"
        print(row)

    name, D = "Hubble (2.4 m)", APERTURES["Hubble (2.4 m)"]
    lam_name, lam = "550 nm (green)", WAVELENGTHS["550 nm (green)"]
    th = theta_min_rad(lam, D)
    deg, arcmin, arcsec = rad_to_units(th)
    print(f"\n2. Full unit chain for {name}, {lam_name}:")
    print(f"   theta_min = 1.22 * {lam:.2e} / {D} = {th:.4e} rad")
    print(f"             = {deg:.6e} deg")
    print(f"             = {arcmin:.6e} arcmin")
    print(f"             = {arcsec:.6f} arcsec")

    print("\n3. Inverting it: what aperture D resolves a GIVEN angular "
          "separation?")
    header = f"{'target':>48} {'theta (arcsec)':>15}" + \
             "".join(f"{k:>16}" for k in WAVELENGTHS)
    print(header)
    for name, theta in TARGETS_ARCSEC.items():
        row = f"{name:>48} {theta:>15.4f}"
        for lam in WAVELENGTHS.values():
            D_req = required_aperture(lam, theta)
            row += f"{D_req:>16.3f}"
        print(row)

    print()
    print("Notes:")
    print(" - Hubble's real diffraction limit (~0.05-0.1 arcsec, visible) falls")
    print("   right out of the table above -- the formula alone predicts it.")
    print(" - Resolving 1 milliarcsecond needs a ~100+ m aperture at visible")
    print("   wavelengths -- bigger than any single mirror ever built. That gap")
    print("   is exactly why radio/optical INTERFEROMETRY (arrays of telescopes")
    print("   combined, e.g. the EHT, VLTI, ALMA) exists: it synthesizes the")
    print("   resolution of one huge aperture without building one huge mirror.")
    print(" - The 'read a license plate from orbit' aperture comes out a few")
    print("   metres -- Hubble-class -- which is the real reason that trope is")
    print("   physically plausible, not movie magic.")
    print("=" * 84)

    _plot()


def _plot():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    D = np.geomspace(0.003, 50, 400)
    for name, lam in WAVELENGTHS.items():
        _, _, arcsec = rad_to_units(theta_min_rad(lam, D))
        ax1.loglog(D, arcsec, label=name)
    for name, Dv in APERTURES.items():
        _, _, a = rad_to_units(theta_min_rad(WAVELENGTHS["550 nm (green)"], Dv))
        ax1.plot(Dv, a, "ko", ms=4)
        ax1.annotate(name, (Dv, a), fontsize=7, xytext=(3, 3),
                     textcoords="offset points")
    ax1.set_xlabel("aperture D (m)")
    ax1.set_ylabel("theta_min (arcsec)")
    ax1.set_title("Diffraction limit vs aperture\n"
                  "(points: real apertures at 550 nm)")
    ax1.legend(fontsize=8); ax1.grid(True, which="both", alpha=0.3)

    theta = np.geomspace(3e-4, 3, 400)
    for name, lam in WAVELENGTHS.items():
        D_req = required_aperture(lam, theta)
        ax2.loglog(theta, D_req, label=name)
    offsets = [(4, 6), (4, -12), (10, 16), (4, -12), (-150, -14)]
    for (name, th), off in zip(TARGETS_ARCSEC.items(), offsets):
        Dv = required_aperture(WAVELENGTHS["550 nm (green)"], th)
        ax2.plot(th, Dv, "ko", ms=4)
        short = name.split("(")[0].strip()
        ax2.annotate(short, (th, Dv), fontsize=7, xytext=off,
                     textcoords="offset points")
    ax2.set_xlabel("target angular separation (arcsec)")
    ax2.set_ylabel("required aperture D (m)")
    ax2.set_title("Aperture needed to resolve a given angle\n"
                  "(points: named targets at 550 nm)")
    ax2.legend(fontsize=8); ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig("resolution.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
