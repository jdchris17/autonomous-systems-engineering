"""module-8 -- star_centroid.py  (Computational Exercise: Synthetic Star Centroid)

An Airy PSF centered at a deliberately non-integer pixel position:

    (u0, v0) = (250.37, 174.62)

Sample it onto a pixel grid (each pixel gets the PSF's flux properly
INTEGRATED over its area -- a fine sub-pixel grid, block-averaged down, not
just the PSF point-sampled at pixel centers). Estimate the centroid with the
standard intensity-weighted "centre of mass":

    u_est = sum(I * u) / sum(I)          v_est = sum(I * v) / sum(I)

and compare to the truth. No noise anywhere in this file -- this isolates
the OPTICS + SAMPLING problem: can pixels of finite size, on their own,
recover a location the eye never gets to see directly?

The answer depends entirely on one ratio: how many pixels fit across the
PSF. Well sampled, the centroid is unbiased to a tiny fraction of a pixel no
matter where the true centre falls within a pixel. Undersampled (PSF barely
wider than a pixel), the estimate gets pulled toward the nearest pixel
centre -- a real, well known effect in astrometry called "pixel-phase
error," and it shows up here from pure geometry, nothing statistical.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1

D0 = 50.0e-3                 # m, baseline aperture
WAVELENGTH0 = 0.5e-6          # m
PIXEL_SCALE0 = 3.0e-6         # rad/pixel
U0, V0 = 250.37, 174.62       # the example from the exercise


def airy_value(du, dv, D, wavelength, pixel_scale):
    theta = np.hypot(du, dv) * pixel_scale
    v = (np.pi * D / wavelength) * theta
    v_safe = np.where(np.abs(v) < 1e-8, 1.0, v)
    amp = np.where(np.abs(v) < 1e-8, 1.0, 2.0 * j1(v_safe) / v_safe)
    return amp ** 2


def first_null_px(D, wavelength, pixel_scale):
    return 1.22 * wavelength / D / pixel_scale


# ---------------------------------------------------------------------------
# sample the PSF onto a pixel grid: proper per-pixel flux integration
# ---------------------------------------------------------------------------
def sample_star(u0, v0, D, wavelength, pixel_scale, half_window, oversample=16):
    # centre the window on the NEAREST pixel to the true position (not floor),
    # so the window is as close to symmetric about u0,v0 as an integer pixel
    # grid allows -- an asymmetric window truncates the Airy rings unevenly
    # and would otherwise contaminate the subpixel-phase sweep below with a
    # window artefact rather than the real sampling effect being measured
    x0, x1 = int(np.round(u0)) - half_window, int(np.round(u0)) + half_window + 1
    y0, y1 = int(np.round(v0)) - half_window, int(np.round(v0)) + half_window + 1
    nx, ny = x1 - x0, y1 - y0

    xs_fine = x0 - 0.5 + (np.arange(nx * oversample) + 0.5) / oversample
    ys_fine = y0 - 0.5 + (np.arange(ny * oversample) + 0.5) / oversample
    X, Y = np.meshgrid(xs_fine, ys_fine)
    fine = airy_value(X - u0, Y - v0, D, wavelength, pixel_scale)
    img = fine.reshape(ny, oversample, nx, oversample).mean(axis=(1, 3))

    xs_pix = x0 + np.arange(nx)
    ys_pix = y0 + np.arange(ny)
    return img, xs_pix, ys_pix


def centroid(img, xs_pix, ys_pix):
    total = img.sum()
    u_est = np.sum(img.sum(axis=0) * xs_pix) / total
    v_est = np.sum(img.sum(axis=1) * ys_pix) / total
    return u_est, v_est


def main():
    print("=" * 78)
    print("SYNTHETIC STAR CENTROID -- optics + sampling, no noise yet")
    print("=" * 78)

    # ---- baseline: well-sampled -----------------------------------------
    fn0 = first_null_px(D0, WAVELENGTH0, PIXEL_SCALE0)
    img, xs, ys = sample_star(U0, V0, D0, WAVELENGTH0, PIXEL_SCALE0, half_window=40)
    u_est, v_est = centroid(img, xs, ys)
    print(f"baseline: D={D0*1e3:.0f} mm, wavelength={WAVELENGTH0*1e9:.0f} nm, "
          f"pixel scale gives first-null = {fn0:.2f} px  (well sampled)")
    print(f"   true       (u0, v0)             = ({U0:.5f}, {V0:.5f})")
    print(f"   estimated  (u_est, v_est)        = ({u_est:.5f}, {v_est:.5f})")
    print(f"   error                            = ({u_est-U0:+.2e}, {v_est-V0:+.2e}) px")
    print("   (this small residual is NOT sampling bias -- it's the finite")
    print("   40-pixel window truncating the Airy pattern's slowly-decaying")
    print("   outer rings. It shrinks roughly as 1/window, not exponentially;")
    print("   real pipelines use a tapered/weighted window to suppress it")
    print("   faster than growing the window ever could.)")
    print()

    # ---- undersampled comparison ------------------------------------
    D_under = D0 * 4.0
    fn_under = first_null_px(D_under, WAVELENGTH0, PIXEL_SCALE0)
    img_u, xs_u, ys_u = sample_star(U0, V0, D_under, WAVELENGTH0, PIXEL_SCALE0,
                                    half_window=10)
    u_est_u, v_est_u = centroid(img_u, xs_u, ys_u)
    print(f"undersampled: D={D_under*1e3:.0f} mm (4x baseline aperture -> "
          f"SHARPER psf, but only {fn_under:.2f} px across first null)")
    print(f"   true       (u0, v0)             = ({U0:.5f}, {V0:.5f})")
    print(f"   estimated  (u_est, v_est)        = ({u_est_u:.5f}, {v_est_u:.5f})")
    print(f"   error                            = ({u_est_u-U0:+.3f}, "
          f"{v_est_u-V0:+.3f}) px  -- pulled toward the nearest pixel centre")
    print()
    print("A bigger aperture made the PSF sharper (better resolution) but WORSE")
    print("sampled by these pixels -- and centroiding got worse, not better.")
    print("Angular resolution and astrometric precision are not the same axis.")
    print()

    # ---- sweep 1: error vs sampling ratio, two independent knobs ---------
    print("SWEEP -- centroid error vs samples-per-first-null (fixed subpixel "
          "phase, frac=(0.37, 0.62))")
    print(f"{'knob varied':>14} {'first-null (px)':>16} {'|error| (px)':>14}")
    D_list = np.geomspace(D0 * 0.4, D0 * 6, 14)
    err_vs_D = []
    for D in D_list:
        fn = first_null_px(D, WAVELENGTH0, PIXEL_SCALE0)
        hw = max(6, int(np.ceil(5 * fn)))
        img_i, xs_i, ys_i = sample_star(U0, V0, D, WAVELENGTH0, PIXEL_SCALE0, hw)
        ue, ve = centroid(img_i, xs_i, ys_i)
        err = np.hypot(ue - U0, ve - V0)
        err_vs_D.append((fn, err))
    for fn, err in err_vs_D[::3]:
        print(f"{'aperture D':>14} {fn:>16.3f} {err:>14.2e}")

    scale_list = np.geomspace(PIXEL_SCALE0 * 0.15, PIXEL_SCALE0 * 2.5, 14)
    err_vs_scale = []
    for ps in scale_list:
        fn = first_null_px(D0, WAVELENGTH0, ps)
        hw = max(6, int(np.ceil(5 * fn)))
        img_i, xs_i, ys_i = sample_star(U0, V0, D0, WAVELENGTH0, ps, hw)
        ue, ve = centroid(img_i, xs_i, ys_i)
        err = np.hypot(ue - U0, ve - V0)
        err_vs_scale.append((fn, err))
    for fn, err in err_vs_scale[::3]:
        print(f"{'pixel scale':>14} {fn:>16.3f} {err:>14.2e}")
    print("   both knobs collapse onto the same curve -- only the RATIO")
    print("   (pixels per first null) sets the centroiding error.")
    print("=" * 78)

    _plot(img, xs, ys, img_u, xs_u, ys_u, err_vs_D, err_vs_scale)


def _subpixel_phase_sweep(D, hw):
    # avoid frac=0 exactly: a star placed dead-on a pixel centre makes the
    # (already near-symmetric) window PERFECTLY symmetric about it, driving
    # the residual to machine precision -- a real but non-generic special
    # case that would otherwise dominate a log-scale plot with a misleading
    # cliff. Every other phase is representative of "a star doesn't know
    # about the pixel grid."
    fracs = np.linspace(0.03, 0.97, 20)
    errs = []
    for f in fracs:
        u0, v0 = 250.0 + f, 174.0 + f
        img, xs, ys = sample_star(u0, v0, D, WAVELENGTH0, PIXEL_SCALE0, hw)
        ue, ve = centroid(img, xs, ys)
        errs.append(np.hypot(ue - u0, ve - v0))
    return fracs, np.array(errs)


def _plot(img, xs, ys, img_u, xs_u, ys_u, err_vs_D, err_vs_scale):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    def show_stamp(ax, img, xs, ys, u_true, v_true, u_est, v_est, title):
        ax.imshow(img, cmap="inferno", origin="lower",
                  extent=[xs[0] - 0.5, xs[-1] + 0.5, ys[0] - 0.5, ys[-1] + 0.5])
        ax.plot(u_true, v_true, "c+", ms=16, mew=2, label="true")
        ax.plot(u_est, v_est, "gx", ms=12, mew=2, label="estimated")
        ax.set_title(title, fontsize=9.5)
        ax.legend(fontsize=7)
        ax.set_xlabel("u (px)"); ax.set_ylabel("v (px)")

    u_est, v_est = centroid(img, xs, ys)
    show_stamp(axes[0, 0], img, xs, ys, U0, V0, u_est, v_est,
              f"well sampled ({first_null_px(D0, WAVELENGTH0, PIXEL_SCALE0):.1f} "
              f"px/first-null)\nerror = {np.hypot(u_est-U0, v_est-V0):.1e} px")

    u_est_u, v_est_u = centroid(img_u, xs_u, ys_u)
    show_stamp(axes[0, 1], img_u, xs_u, ys_u, U0, V0, u_est_u, v_est_u,
              f"undersampled ({first_null_px(D0*4, WAVELENGTH0, PIXEL_SCALE0):.2f} "
              f"px/first-null)\nerror = {np.hypot(u_est_u-U0, v_est_u-V0):.3f} px")

    ax = axes[0, 2]
    fracs_w, err_w = _subpixel_phase_sweep(D0, 20)
    fracs_u, err_uu = _subpixel_phase_sweep(D0 * 4.0, 10)
    ax.plot(fracs_w, err_w, "o-", label="well sampled")
    ax.plot(fracs_u, err_uu, "o-", label="undersampled")
    ax.set_xlabel("subpixel phase (fractional part of u0, v0)")
    ax.set_ylabel("|centroid error| (px)")
    ax.set_yscale("log")
    ax.set_title("Error vs where the star falls\nwithin a pixel")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)

    ax = axes[1, 0]
    fn_D, e_D = zip(*err_vs_D)
    ax.loglog(fn_D, e_D, "o-")
    ax.set_xlabel("first null (px)  [varying aperture D]")
    ax.set_ylabel("|centroid error| (px)")
    ax.set_title("Error vs sampling ratio\n(knob: aperture)")
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1, 1]
    fn_s, e_s = zip(*err_vs_scale)
    ax.loglog(fn_s, e_s, "o-", color="tab:orange")
    ax.set_xlabel("first null (px)  [varying pixel scale]")
    ax.set_ylabel("|centroid error| (px)")
    ax.set_title("Error vs sampling ratio\n(knob: pixel size)")
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1, 2]
    ax.loglog(fn_D, e_D, "o-", label="varying aperture D")
    ax.loglog(fn_s, e_s, "s--", label="varying pixel scale")
    ax.axvline(2.0, color="grey", ls=":", lw=1, label="~Nyquist, 2 px/first-null")
    ax.set_xlabel("first null (px)")
    ax.set_ylabel("|centroid error| (px)")
    ax.set_title("Same curve either way:\nonly the RATIO matters")
    ax.legend(fontsize=7); ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Synthetic star centroid: optics + sampling only, no noise",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig("star_centroid.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
