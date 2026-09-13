"""camera/measurement_error.py  (Add Measurement Error)

The forward model so far (Phase II - V) has produced perfect centroids
(u_i, v_i). No real detector does that -- every measured centroid carries
noise from the sensor, the PSF, the centroiding algorithm:

    u~_i = u_i + eps_u,i        eps ~ N(0, sigma_px)
    v~_i = v_i + eps_v,i

Try sigma_px = 0.1, 0.5, 1.0 pixels, then convert each into an ANGULAR
uncertainty using the camera's own plate scale (Phase I's angular_scale_x/y,
arcsec/pixel):

    sigma_angular = sigma_px * angular_scale

This is a unit conversion, nothing more -- but it is the conversion that
turns "how good is my centroiding" into "how good can my attitude solution
possibly be," which is exactly what Phase VIII (attitude_determination.py)
measures.
"""

import numpy as np
import matplotlib.pyplot as plt

from camera_model import Camera
from star_field import make_catalog, look_at, render

SIGMAS_PX = [0.1, 0.5, 1.0]


def add_centroid_noise(uv, sigma_px, rng):
    return uv + rng.normal(0.0, sigma_px, uv.shape)


def main():
    star_cam = Camera(f=8.0, Ws=10.0, Hs=8.0, Nx=1024, Ny=820, name="star camera")
    photo_cam = Camera(f=50.0, Ws=36.0, Hs=24.0, Nx=6000, Ny=4000, name="50mm photo camera")
    scope_cam = Camera(f=2000.0, Ws=13.2, Hs=8.8, Nx=5496, Ny=3672, name="2000mm scope")

    print("=" * 78)
    print("MEASUREMENT ERROR: pixel centroid noise -> angular uncertainty")
    print("=" * 78)
    print(f"star camera plate scale: {star_cam.angular_scale_x:.4f} arcsec/px (x), "
          f"{star_cam.angular_scale_y:.4f} arcsec/px (y)")
    print()
    print(f"{'sigma (px)':>12} {'sigma_x (arcsec)':>18} {'sigma_y (arcsec)':>18}")
    print("-" * 50)
    for s in SIGMAS_PX:
        print(f"{s:>12.2f} {s*star_cam.angular_scale_x:>18.4f} "
              f"{s*star_cam.angular_scale_y:>18.4f}")

    print()
    print("Same pixel error, three different cameras (why this 'matters a lot")
    print("later': attitude accuracy is bought with plate scale, not just")
    print("centroiding cleverness):")
    print(f"{'camera':>22} {'arcsec/px':>12} {'sigma=0.5px -> arcsec':>22}")
    for cam in (photo_cam, star_cam, scope_cam):
        print(f"{cam.name:>22} {cam.angular_scale_x:>12.4f} "
              f"{0.5*cam.angular_scale_x:>22.4f}")
    ratio = (0.5 * photo_cam.angular_scale_x) / (0.5 * scope_cam.angular_scale_x)
    print()
    print(f"The 2000mm scope turns half a pixel of centroiding noise into "
          f"{0.5*scope_cam.angular_scale_x:.3f} arcsec; the 50mm photo lens")
    print(f"turns the SAME half pixel into {0.5*photo_cam.angular_scale_x:.2f} "
          f"arcsec -- {ratio:.0f}x worse, purely from plate scale (focal")
    print("length, mainly -- the scope's 40x longer focal length dominates "
          "even though its pixels are also smaller).")
    print("=" * 78)

    _plot(star_cam, photo_cam, scope_cam)


def _plot(star_cam, photo_cam, scope_cam):
    rng = np.random.default_rng(3)
    s_hat, mag = make_catalog()
    R_CI = look_at(s_hat[0])
    uv_true, valid, _ = render(s_hat, R_CI, star_cam)
    # a single, near-centre star to visualise the noise cloud on
    dist = np.hypot(uv_true[:, 0] - star_cam.cx, uv_true[:, 1] - star_cam.cy)
    dist[~valid] = np.inf
    i0 = np.argmin(dist)
    u0, v0 = uv_true[i0]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5.5))

    colors = ["tab:blue", "tab:orange", "tab:red"]
    for s, c in zip(SIGMAS_PX, colors):
        cloud = np.column_stack([u0, v0]) + rng.normal(0, s, (300, 2))
        ax1.scatter(cloud[:, 0], cloud[:, 1], s=6, alpha=0.4, color=c,
                   label=f"sigma={s} px")
    ax1.plot(u0, v0, "k+", ms=14, mew=2, label="true centroid")
    ax1.set_xlabel("u (px)"); ax1.set_ylabel("v (px)")
    ax1.set_title("300 noisy centroid draws per sigma\n"
                  "(one star, zoomed in)")
    ax1.set_aspect("equal"); ax1.legend(fontsize=8); ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)

    s_range = np.geomspace(0.02, 1.5, 50)
    ax2.loglog(s_range, s_range * star_cam.angular_scale_x, label="star camera")
    ax2.loglog(s_range, s_range * photo_cam.angular_scale_x, label="50mm photo")
    ax2.loglog(s_range, s_range * scope_cam.angular_scale_x, label="2000mm scope")
    for s in SIGMAS_PX:
        ax2.axvline(s, color="0.8", lw=0.8, zorder=0)
    ax2.set_xlabel("pixel centroid sigma (px)")
    ax2.set_ylabel("angular sigma (arcsec)")
    ax2.set_title("Angular error = pixel error x plate scale\n"
                  "(parallel lines on log-log: same linear law, offset by "
                  "plate scale)")
    ax2.legend(fontsize=8); ax2.grid(True, which="both", alpha=0.3)

    cams = [photo_cam, star_cam, scope_cam]
    vals = [0.5 * c.angular_scale_x for c in cams]
    ax3.bar([c.name for c in cams], vals, color=["tab:green", "tab:blue", "tab:purple"])
    ax3.set_yscale("log")
    ax3.set_ylabel("angular error at sigma=0.5 px (arcsec)")
    ax3.set_title("Same centroiding precision,\nthree very different plate scales")
    ax3.tick_params(axis="x", labelrotation=15)
    for i, v in enumerate(vals):
        ax3.text(i, v * 1.15, f"{v:.3f}\"", ha="center", fontsize=8)
    ax3.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig("measurement_error.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
