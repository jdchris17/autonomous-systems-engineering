"""module-8 -- single_slit.py  (Start with a Single Slit)

Fraunhofer diffraction from a slit of width a:

    I(theta) = I0 (sin(beta) / beta)^2         beta = pi a sin(theta) / lambda

Minima at a sin(theta) = m lambda  (m = +/-1, +/-2, ...) -- exact zeros of
sin(beta), not an approximation of this formula (the word "approximately" in
the exercise is about Fraunhofer diffraction itself being the far-field limit
of the full near-field/Fresnel picture, the same distinction Module 8's
interference_revisited.py drew for the double slit).

Plot I(theta) for several a, several wavelength; confirm the inverse
relationship: smaller aperture -> wider pattern, longer wavelength -> wider
pattern -- both live in the SAME ratio, lambda/a.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar

A0 = 10.0            # um, baseline slit width
WAVELENGTH0 = 0.5     # um, baseline wavelength


def single_slit_intensity(theta, a, wavelength, I0=1.0):
    beta = np.pi * a * np.sin(theta) / wavelength
    return I0 * np.sinc(beta / np.pi) ** 2     # sinc(x) = sin(pi x)/(pi x); safe at beta=0


def predicted_minima(a, wavelength, m_max=30):
    m = np.arange(-m_max, m_max + 1)
    m = m[m != 0]
    s = m * wavelength / a
    keep = np.abs(s) <= 1.0
    return np.arcsin(s[keep]), m[keep]


def first_null_angle(a, wavelength):
    s = wavelength / a
    return np.arcsin(s) if abs(s) <= 1.0 else np.nan


def main():
    theta = np.linspace(-np.pi / 2 * 0.999, np.pi / 2 * 0.999, 4000)

    print("=" * 78)
    print("SINGLE-SLIT DIFFRACTION   I(theta) = I0 (sin(beta)/beta)^2")
    print("=" * 78)
    print(f"baseline: a = {A0} um, wavelength = {WAVELENGTH0} um")
    print()

    # -- validation: numerically found minima vs a sin(theta) = m*wavelength -
    I0_curve = single_slit_intensity(theta, A0, WAVELENGTH0)
    dips, _ = find_peaks(-I0_curve, height=-0.01)          # coarse-grid minima
    dtheta = theta[1] - theta[0]
    theta_meas = []
    for idx in dips:                                        # refine each with a
        res = minimize_scalar(                               # local bounded search
            lambda t: single_slit_intensity(t, A0, WAVELENGTH0),
            bounds=(theta[idx] - 2 * dtheta, theta[idx] + 2 * dtheta),
            method="bounded", options={"xatol": 1e-12})
        theta_meas.append(res.x)
    theta_meas = np.sort(np.array(theta_meas))
    theta_pred, m_pred = predicted_minima(A0, WAVELENGTH0)
    theta_pred = np.sort(theta_pred)
    # the m = +/-(a/wavelength) minima sit exactly at theta=+/-90 deg, just
    # outside the sampled range (theta stops just short of +/-90) -- drop
    # those two boundary cases so the comparison is apples-to-apples
    theta_pred_in_range = theta_pred[np.abs(theta_pred) < theta.max()]
    print(f"VALIDATION -- numerically located minima vs a*sin(theta) = m*wavelength")
    print(f"   found {len(theta_meas)} minima in the sampled range numerically; "
          f"predicted {len(theta_pred_in_range)} in-range "
          f"({len(theta_pred)} total including the two grazing m=+/-{abs(m_pred).max()} "
          f"cases right at +/-90 deg)")
    if len(theta_meas) == len(theta_pred_in_range):
        err = np.abs(theta_meas - theta_pred_in_range)
        print(f"   max angular discrepancy = {np.degrees(err).max():.2e} deg "
              f"-- exact zeros, not an approximation")
    print()

    # -- first-null half-angle: exact arcsin vs small-angle theta1~=wavelength/a
    print("First-null half-angle theta1 = arcsin(wavelength/a): exact vs "
          "small-angle theta1~=wavelength/a")
    print(f"{'a (um)':>8} {'wavelength/a':>13} {'theta1 exact (deg)':>19} "
          f"{'theta1~=wavelength/a (deg)':>26} {'rel err':>9}")
    for a in [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]:
        s = WAVELENGTH0 / a
        exact = np.degrees(np.arcsin(s))
        approx = np.degrees(s)
        print(f"{a:>8.1f} {s:>13.4f} {exact:>19.4f} {approx:>26.4f} "
              f"{abs(exact-approx)/exact:>9.2e}")
    print()
    print("smaller a -> larger wavelength/a -> wider pattern; the small-angle")
    print("theta1~=wavelength/a formula only holds while wavelength/a stays small")
    print("(same story as every paraxial approximation in this course).")
    print()

    print("Link back to Module 6's resolution.py: a RECTANGULAR slit's first")
    print("null is at sin(theta)=wavelength/a (no 1.22); a CIRCULAR aperture's")
    print("first null (the Airy pattern) is at sin(theta)=1.22 wavelength/D --")
    print("same physics, the 1.22 is purely the geometry of a circular vs a")
    print("slit-shaped opening.")
    print("=" * 78)

    _plot(theta)


def _plot(theta):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (0,0) baseline with minima marked
    ax = axes[0, 0]
    I = single_slit_intensity(theta, A0, WAVELENGTH0)
    ax.plot(np.degrees(theta), I, lw=1.2)
    th_pred, m_pred = predicted_minima(A0, WAVELENGTH0)
    ax.plot(np.degrees(th_pred), np.zeros_like(th_pred), "rx", ms=6,
            label="predicted minima  a sin(theta) = m*wavelength")
    ax.set_xlabel("theta (deg)"); ax.set_ylabel("I / I0")
    ax.set_title(f"baseline: a={A0} um, wavelength={WAVELENGTH0} um")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (0,1) aperture sweep
    ax = axes[0, 1]
    for a in [5.0, 10.0, 20.0, 40.0]:
        ax.plot(np.degrees(theta), single_slit_intensity(theta, a, WAVELENGTH0),
                label=f"a={a} um")
    ax.set_xlim(-30, 30)
    ax.set_xlabel("theta (deg)"); ax.set_ylabel("I / I0")
    ax.set_title(f"aperture sweep (wavelength={WAVELENGTH0} um fixed)\n"
                 "smaller a -> wider pattern")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (1,0) wavelength sweep
    ax = axes[1, 0]
    for wl in [0.4, 0.5, 0.6, 0.7]:
        ax.plot(np.degrees(theta), single_slit_intensity(theta, A0, wl),
                label=f"wavelength={wl} um")
    ax.set_xlim(-6, 6)
    ax.set_xlabel("theta (deg)"); ax.set_ylabel("I / I0")
    ax.set_title(f"wavelength sweep (a={A0} um fixed)\nlonger wavelength -> wider pattern")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (1,1) exact vs small-angle first-null, and where the approx breaks
    ax = axes[1, 1]
    ratio = np.geomspace(0.01, 1.0, 300)
    exact_deg = np.degrees(np.arcsin(ratio))
    approx_deg = np.degrees(ratio)
    ax.plot(ratio, exact_deg, label="exact: arcsin(wavelength/a)")
    ax.plot(ratio, approx_deg, "--", label="small-angle: wavelength/a (rad->deg)")
    ax.set_xlabel("wavelength / a"); ax.set_ylabel("theta1 (deg)")
    ax.set_title("First-null half-angle: exact vs small-angle\n"
                 "diverge once wavelength/a is not small")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle("Single-slit diffraction: I(theta) = I0 (sin(beta)/beta)^2, "
                 "beta = pi a sin(theta)/wavelength", fontsize=13)
    fig.tight_layout()
    fig.savefig("single_slit.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
