"""module-6 -- double_slit.py  (Computational Exercise: Double-Slit Interference)

Two coherent point sources separated by d. At a distant observation angle
theta (far field / Fraunhofer), the path difference between the two sources
is

    dL(theta) = d sin(theta)

so the phase difference and intensity are

    dphi(theta) = (2 pi / lambda) dL(theta)
    I(theta)    = cos^2( dphi / 2 )            (normalised, 2 equal sources)

A screen at distance L converts angle to position via  y = L tan(theta)  --
that mapping is exact given theta; the only physics approximation is the far-
field path difference d sin(theta) itself (excellent whenever d << L, which
holds throughout this file).

We vary wavelength, slit separation, and screen distance, measure the fringe
spacing NUMERICALLY (find the peaks of I(y) and diff their positions), and
test it against the small-angle formula

    dy ~= lambda L / d

including a case chosen to make that approximation visibly fail.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

LAMBDA0 = 600e-9      # m  (600 nm, red)
D0 = 0.20e-3          # m  (0.20 mm slit separation)
L0 = 1.0              # m  (screen distance)


def intensity(y, L, d, wavelength):
    theta = np.arctan2(y, L)
    dL = d * np.sin(theta)
    dphi = 2 * np.pi * dL / wavelength
    return np.cos(dphi / 2.0) ** 2


def measure_spacing(y, I, min_height=0.9):
    """Numerically find fringe (maxima) positions and their spacing."""
    peaks, _ = find_peaks(I, height=min_height)
    y_peaks = y[peaks]
    spacing = np.diff(y_peaks)
    return y_peaks, spacing


def screen_range(wavelength, d, L, half_fringes=10):
    dy_pred = wavelength * L / d
    y = np.linspace(-half_fringes * dy_pred, half_fringes * dy_pred, 40001)
    return y, dy_pred


def main():
    print("=" * 78)
    print("DOUBLE-SLIT INTERFERENCE: analytic dy ~ lambda L / d  vs  numerical model")
    print("=" * 78)
    print(f"base case: lambda = {LAMBDA0*1e9:.0f} nm, d = {D0*1e3:.2f} mm, "
          f"L = {L0:.2f} m")
    y0, dy0_pred = screen_range(LAMBDA0, D0, L0)
    I0 = intensity(y0, L0, D0, LAMBDA0)
    peaks0, spacing0 = measure_spacing(y0, I0)
    print(f"   predicted dy = lambda L / d = {dy0_pred*1e3:.4f} mm")
    print(f"   measured  dy (mean peak spacing, {len(peaks0)} fringes found) = "
          f"{spacing0.mean()*1e3:.4f} mm")
    print(f"   relative error = {abs(spacing0.mean()-dy0_pred)/dy0_pred:.2e}")
    print()

    print("Sweep 1 -- wavelength (d, L fixed at base):")
    print(f"   {'lambda(nm)':>11} {'dy pred (mm)':>13} {'dy meas (mm)':>13} "
          f"{'rel err':>9}")
    lam_rows = []
    for lam in (400e-9, 500e-9, 600e-9, 700e-9):
        y, dy_pred = screen_range(lam, D0, L0)
        I = intensity(y, L0, D0, lam)
        _, sp = measure_spacing(y, I)
        lam_rows.append((lam, dy_pred, sp.mean()))
        print(f"   {lam*1e9:>11.0f} {dy_pred*1e3:>13.4f} {sp.mean()*1e3:>13.4f} "
              f"{abs(sp.mean()-dy_pred)/dy_pred:>9.2e}")

    print()
    print("Sweep 2 -- slit separation d (lambda, L fixed at base):")
    print(f"   {'d(mm)':>7} {'dy pred (mm)':>13} {'dy meas (mm)':>13} "
          f"{'rel err':>9}")
    d_rows = []
    for d in (0.10e-3, 0.20e-3, 0.30e-3, 0.40e-3):
        y, dy_pred = screen_range(LAMBDA0, d, L0)
        I = intensity(y, L0, d, LAMBDA0)
        _, sp = measure_spacing(y, I)
        d_rows.append((d, dy_pred, sp.mean()))
        print(f"   {d*1e3:>7.2f} {dy_pred*1e3:>13.4f} {sp.mean()*1e3:>13.4f} "
              f"{abs(sp.mean()-dy_pred)/dy_pred:>9.2e}")

    print()
    print("Sweep 3 -- screen distance L (lambda, d fixed at base):")
    print(f"   {'L(m)':>6} {'dy pred (mm)':>13} {'dy meas (mm)':>13} "
          f"{'rel err':>9}")
    L_rows = []
    for L in (0.5, 1.0, 1.5, 2.0):
        y, dy_pred = screen_range(LAMBDA0, D0, L)
        I = intensity(y, L, D0, LAMBDA0)
        _, sp = measure_spacing(y, I)
        L_rows.append((L, dy_pred, sp.mean()))
        print(f"   {L:>6.2f} {dy_pred*1e3:>13.4f} {sp.mean()*1e3:>13.4f} "
              f"{abs(sp.mean()-dy_pred)/dy_pred:>9.2e}")

    print()
    print("dy scales linearly with lambda, linearly with L, and as 1/d -- exactly")
    print("what lambda L / d says -- and the numerical model agrees to << 0.1%")
    print("whenever lambda/d is small (paraxial: sin(theta) ~= tan(theta)).")
    print()

    # ---- break the approximation on purpose ----------------------------
    d_break = LAMBDA0 / 0.30                 # lambda/d = 0.30 -> not small
    m_max = int(np.floor(d_break / LAMBDA0))
    print(f"Now push it: d = {d_break*1e6:.2f} micron so lambda/d = 0.30 "
          f"(only m = 0..{m_max} orders exist,")
    print("since sin(theta) = m lambda/d cannot exceed 1). Small-angle stops "
          "being a good approximation:")
    y_b = np.linspace(-1.05, 1.05, 20001)
    I_b = intensity(y_b, L0, d_break, LAMBDA0)
    peaks_b, spacing_b = measure_spacing(y_b, I_b, min_height=0.95)
    dy_pred_b = LAMBDA0 * L0 / d_break
    print(f"   paraxial prediction dy = lambda L/d = {dy_pred_b*1e3:.2f} mm "
          f"(would-be constant spacing)")
    print(f"   {'gap #':>6} {'centre y (m)':>13} {'measured dy (mm)':>17} "
          f"{'vs paraxial':>12}")
    mids = 0.5 * (peaks_b[1:] + peaks_b[:-1])
    for i, (yc, sp) in enumerate(zip(mids, spacing_b)):
        print(f"   {i+1:>6} {yc:>13.4f} {sp*1e3:>17.2f} "
              f"{sp/dy_pred_b:>11.2f}x")
    print("   spacing grows toward the edges: sin(theta) is linear in the")
    print("   fringe order but y = L tan(theta) is not, so equal-angle fringes")
    print("   are NOT equally spaced on the screen once theta is not small.")
    print("=" * 78)

    _plot(y0, I0, peaks0, dy0_pred, lam_rows, d_rows, L_rows,
          y_b, I_b, peaks_b, mids, spacing_b, dy_pred_b)


def _plot(y0, I0, peaks0, dy0_pred, lam_rows, d_rows, L_rows,
         y_b, I_b, peaks_b, mids, spacing_b, dy_pred_b):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) base-case fringe pattern
    ax = axes[0, 0]
    ax.plot(y0 * 1e3, I0, lw=0.8)
    ax.plot(peaks0 * 1e3, np.ones_like(peaks0), "rv", ms=5)
    ax.set_xlabel("screen position y (mm)"); ax.set_ylabel("I (normalised)")
    ax.set_title(f"Base case: lambda={LAMBDA0*1e9:.0f} nm, d={D0*1e3:.2f} mm, "
                 f"L={L0:.1f} m\n"
                 f"measured dy = {np.diff(peaks0).mean()*1e3:.3f} mm  "
                 f"(predicted {dy0_pred*1e3:.3f} mm)")
    ax.grid(True, alpha=0.3)

    # (0,1) parity plot: predicted vs measured across all sweeps
    ax = axes[0, 1]
    all_rows = [("lambda sweep", lam_rows), ("d sweep", d_rows), ("L sweep", L_rows)]
    for label, rows in all_rows:
        pred = np.array([r[1] for r in rows]) * 1e3
        meas = np.array([r[2] for r in rows]) * 1e3
        ax.plot(pred, meas, "o", label=label, ms=8)
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="measured = predicted")
    ax.set_xlabel("predicted dy = lambda L / d  (mm)")
    ax.set_ylabel("measured dy  (mm)")
    ax.set_title("Every sweep collapses onto the same line:\n"
                 "measured fringe spacing = lambda L / d")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (1,0) breaking case fringe pattern
    ax = axes[1, 0]
    ax.plot(y_b, I_b, lw=0.9)
    ax.plot(peaks_b, np.ones_like(peaks_b), "rv", ms=6)
    ax.set_xlabel("screen position y (m)"); ax.set_ylabel("I (normalised)")
    ax.set_title(r"Pushed case: $\lambda/d = 0.30$ (only a few orders exist)"
                 "\nspacing visibly widens away from the centre")
    ax.grid(True, alpha=0.3)

    # (1,1) local spacing vs order for the breaking case
    ax = axes[1, 1]
    ax.plot(mids, spacing_b * 1e3, "o-", label="measured local spacing")
    ax.axhline(dy_pred_b * 1e3, color="grey", ls="--",
              label="paraxial dy = lambda L / d")
    ax.set_xlabel("gap centre position y (m)")
    ax.set_ylabel("local fringe spacing (mm)")
    ax.set_title("Paraxial formula predicts constant spacing;\n"
                 "the real (exact-angle) spacing grows toward the edges")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("double_slit.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
