"""module-9 -- magnitude_flux.py  (Computational Exercise: Magnitude -> Relative Photon Flux)

The magnitude system is logarithmic, and Pogson's ratio fixes the base: a
difference of exactly 5 magnitudes is defined to be exactly a factor of 100
in flux, so one magnitude is a factor of 100^(1/5) = 10^0.4 = 2.512:

    relative_flux(m, m_ref) = 10 ^ ( -0.4 (m - m_ref) )

Fainter objects have LARGER m and SMALLER flux -- the minus sign is the whole
reason the magnitude scale feels backwards until you've used it a few times.

Then the connection to Module 9 (real detectors looking at a real sky): flux
ratios ARE photon-count ratios, because a detector counts photons in
proportion to the flux it receives. If a magnitude-2 star gives an expected
count N2, a magnitude-m star gives

    N(m) = N2 * relative_flux(m, reference_magnitude=2)

and if the detector is photon-noise-limited (Poisson statistics), the
signal-to-noise ratio SNR = sqrt(N) falls off HALF as fast, in log terms,
as the flux itself: a 1-magnitude-fainter star costs a factor of 10^0.4 in
photons but only 10^0.2 in SNR.
"""

import numpy as np
import matplotlib.pyplot as plt


def relative_flux(magnitude, reference_magnitude=0.0):
    """F_rel = 10^(-0.4 (m - m_ref)) -- flux relative to an arbitrary reference."""
    m = np.asarray(magnitude, dtype=float)
    return 10.0 ** (-0.4 * (m - reference_magnitude))


def main():
    mags = np.arange(0, 9)
    flux = relative_flux(mags)             # reference_magnitude=0 (default)

    print("=" * 74)
    print("MAGNITUDE -> RELATIVE PHOTON FLUX")
    print("=" * 74)
    print("relative_flux(m) = 10^(-0.4 m),  reference m_ref = 0")
    print()
    print(f"{'m':>3} {'F_rel':>12} {'F_rel(m)/F_rel(m-1)':>22}")
    for i, m in enumerate(mags):
        ratio = f"{flux[i]/flux[i-1]:.5f}" if i > 0 else "--"
        print(f"{m:>3} {flux[i]:>12.6f} {ratio:>22}")
    print()
    print(f"per-magnitude ratio is constant: 10^-0.4 = {10**-0.4:.6f}")
    print(f"check: F_rel(5)/F_rel(0) = {flux[5]/flux[0]:.6f}  "
          f"(Pogson's ratio: 5 mag is defined to be exactly 100x)")
    print()

    # ---- connect to Module 9: a real detector counting real photons -----
    N2 = 10_000.0                                   # expected counts at m=2
    connect_mags = np.array([2, 3, 4, 5, 6])
    N = N2 * relative_flux(connect_mags, reference_magnitude=2)
    SNR = np.sqrt(N)                                # photon-noise-limited

    print("Connecting to Module 9: a magnitude-2 star gives an expected "
          f"count N2 = {N2:.0f} photons.")
    print(f"{'m':>3} {'N(m) = N2*F_rel':>16} {'SNR = sqrt(N)':>14} "
          f"{'SNR(m)/SNR(m-1)':>16}")
    for i, m in enumerate(connect_mags):
        ratio = f"{SNR[i]/SNR[i-1]:.5f}" if i > 0 else "--"
        print(f"{m:>3} {N[i]:>16.2f} {SNR[i]:>14.3f} {ratio:>16}")
    print()
    print(f"flux drops by 10^-0.4 = {10**-0.4:.4f} per magnitude;")
    print(f"SNR drops by only sqrt of that, 10^-0.2 = {10**-0.2:.4f} per "
          f"magnitude, because SNR = sqrt(N) and sqrt halves the log-slope.")
    print("Stellar magnitude has become exactly what a real observation plan")
    print("cares about: how many photons land, and how noisy the resulting")
    print("measurement is -- not just 'how bright does it look.'")
    print("=" * 74)

    _plot(mags, flux, connect_mags, N, SNR)


def _plot(mags, flux, connect_mags, N, SNR):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    ax1.semilogy(mags, flux, "o-", color="tab:blue")
    ax1.annotate("", xy=(5, flux[5]), xytext=(0, flux[0]),
                 arrowprops=dict(arrowstyle="<->", color="grey", lw=1))
    ax1.text(2.3, 0.18, "exactly\n100x drop\n(5 mag, by\ndefinition)",
             fontsize=8, color="0.3")
    ax1.set_xlabel("apparent magnitude m")
    ax1.set_ylabel("relative flux  F_rel(m)  (m_ref = 0)")
    ax1.set_title("Relative flux vs magnitude\n"
                  "one magnitude = a factor of 10^0.4 = 2.512")
    ax1.grid(True, which="both", alpha=0.3)

    # both normalised to their m=2 value and put on ONE log axis, so the
    # different slopes are directly, honestly comparable (a dual-axis plot
    # with independently auto-scaled axes would hide this)
    ax2.semilogy(connect_mags, N / N[0], "o-", color="tab:red",
                 label="N(m) / N(2)  (photon count)")
    ax2.semilogy(connect_mags, SNR / SNR[0], "s-", color="tab:green",
                 label="SNR(m) / SNR(2)")
    ax2.set_xlabel("apparent magnitude m")
    ax2.set_ylabel("ratio to the m=2 value")
    ax2.set_title("Same drop, two speeds (both normalised to m=2)\n"
                  "count falls as 10^-0.4/mag, SNR only as 10^-0.2/mag")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig("magnitude_flux.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
