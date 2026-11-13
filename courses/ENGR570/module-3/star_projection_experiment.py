"""
star_projection_experiment.py - Space Systems Engineering, Module 3, Experiment C

Catalog RA/Dec + camera attitude quaternion -> predicted star pixels.

Pipeline
--------
1. catalog RA/Dec          -> celestial unit vectors s_G     (Module 1)
2. s_C = C_CG s_G, with C_CG = R(q)^T from the attitude quaternion q
3. reject stars behind the camera or off the sensor
4. pinhole projection      -> pixel coordinates (u, v)
5. repeat with the camera rotated 5 deg about its own y axis

Then the inverse attitude transformation recovers the catalog directions.

Run:  python star_projection_experiment.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rotation_engine import axis_angle_to_quaternion, quaternion_multiply
from sensor_frames import (
    PinholeCamera,
    attitude_to_transform,
    camera_attitude_from_boresight,
    radec_to_unit,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "module-1"))
from celestial_geometry import angular_separation, unit_to_radec  # noqa: E402

BORESIGHT_RA_DEG = 120.0
BORESIGHT_DEC_DEG = 30.0
N_STARS = 40
SLEW_DEG = 5.0
SEED = 570


def make_catalog(camera, q_reference, n_stars, seed):
    """
    Synthetic catalog of (RA, Dec) in degrees, scattered around the
    reference boresight over an area somewhat larger than the field of
    view, so that some stars start outside the image.
    """
    rng = np.random.default_rng(seed)
    half_width = np.radians(camera.horizontal_fov_deg / 2.0 + SLEW_DEG)
    half_height = np.radians(camera.vertical_fov_deg / 2.0 + 2.0)
    tan_x = np.tan(rng.uniform(-half_width, half_width, n_stars))
    tan_y = np.tan(rng.uniform(-half_height, half_height, n_stars))
    camera_dirs = np.column_stack((tan_x, tan_y, np.ones(n_stars)))
    camera_dirs /= np.linalg.norm(camera_dirs, axis=1, keepdims=True)
    celestial = attitude_to_transform(q_reference).apply(camera_dirs)
    return np.array([unit_to_radec(v) for v in celestial])


def predict_image(catalog_radec, q_attitude, camera):
    """Steps 1-4. Returns (camera_vectors, pixels, visible)."""
    celestial = np.array([radec_to_unit(ra, dec) for ra, dec in catalog_radec])
    c_cg = attitude_to_transform(q_attitude).inverse()        # celestial -> camera
    camera_vectors = c_cg.apply(celestial)
    pixels, visible = camera.project(camera_vectors)
    return camera_vectors, pixels, visible


def main():
    camera = PinholeCamera()
    print(f"Camera: {camera.width} x {camera.height} px, "
          f"HFOV {camera.horizontal_fov_deg} deg, "
          f"VFOV {camera.vertical_fov_deg:.2f} deg, "
          f"fx = fy = {camera.fx:.2f} px, "
          f"{3600 * camera.horizontal_fov_deg / camera.width:.1f} arcsec/px")

    # Reference attitude: boresight at the chosen RA/Dec, north up.
    q_ref = camera_attitude_from_boresight(BORESIGHT_RA_DEG, BORESIGHT_DEC_DEG)
    catalog = make_catalog(camera, q_ref, N_STARS, SEED)
    # Add one star exactly on the reference boresight as a check target.
    catalog = np.vstack(([BORESIGHT_RA_DEG, BORESIGHT_DEC_DEG], catalog))

    # New attitude: rotate the camera +5 deg about its OWN y axis.
    # A rotation expressed in the camera's local axes multiplies on the right.
    q_slew = axis_angle_to_quaternion([0.0, 1.0, 0.0], np.radians(SLEW_DEG))
    q_new = quaternion_multiply(q_ref, q_slew)

    vec_ref, pix_ref, vis_ref = predict_image(catalog, q_ref, camera)
    vec_new, pix_new, vis_new = predict_image(catalog, q_new, camera)

    print(f"Attitude quaternion, reference [w x y z]: {np.round(q_ref, 6)}")
    print(f"Attitude quaternion, after slew [w x y z]: {np.round(q_new, 6)}")
    print(f"\nCatalog: {len(catalog)} stars around RA {BORESIGHT_RA_DEG} deg, "
          f"Dec {BORESIGHT_DEC_DEG} deg")
    print(f"Visible at reference attitude: {vis_ref.sum()}")
    print(f"Visible after +{SLEW_DEG:g} deg about camera y: {vis_new.sum()}")
    both = vis_ref & vis_new
    print(f"Visible in both: {both.sum()}  "
          f"(left the image: {(vis_ref & ~vis_new).sum()}, "
          f"entered: {(~vis_ref & vis_new).sum()})")

    print("\n  star     RA [deg]   Dec [deg]     u_ref     v_ref     u_new"
          "     v_new        du        dv")
    for i in np.flatnonzero(both)[:10]:
        du, dv = pix_new[i] - pix_ref[i]
        print(f"  {i:>4}{catalog[i, 0]:>13.4f}{catalog[i, 1]:>12.4f}"
              f"{pix_ref[i, 0]:>10.2f}{pix_ref[i, 1]:>10.2f}"
              f"{pix_new[i, 0]:>10.2f}{pix_new[i, 1]:>10.2f}"
              f"{du:>10.2f}{dv:>10.2f}")

    shift = pix_new[both] - pix_ref[both]
    predicted_centre_shift = -camera.fx * np.tan(np.radians(SLEW_DEG))
    print(f"\nMean shift of stars visible in both images: "
          f"du = {shift[:, 0].mean():+.2f} px, dv = {shift[:, 1].mean():+.2f} px")
    print(f"Boresight star: u {pix_ref[0, 0]:.2f} -> {pix_new[0, 0]:.2f} "
          f"(du = {pix_new[0, 0] - pix_ref[0, 0]:+.3f} px); "
          f"predicted -fx tan({SLEW_DEG:g} deg) = {predicted_centre_shift:+.3f} px")

    print(f"""
Which way the image moves, and why
----------------------------------
The attitude quaternion q is active: R(q) carries the celestial axes onto
the camera axes, so R(q) = C_GC and a star's camera coordinates are
s_C = R(q)^T s_G.
Rotating the camera about its own y axis is a right-multiplication,
q_new = q (x) q_y(+5 deg), so R(q_new) = R(q) R_y(+5 deg) and
    s_C,new = R_y(+5 deg)^T s_C = R_y(-5 deg) s_C.
The camera turns one way; star coordinates in the camera frame turn the
opposite way. R_y(+5 deg) swings the boresight (+z) toward +x, image right.
A star that was on the boresight, (0, 0, 1), becomes
(-sin 5, 0, cos 5): X/Z = -tan 5 deg, so it lands fx tan(5 deg) = {-predicted_centre_shift:.1f} px
to the LEFT. The camera pans right, the star field slides left. Rows barely
change: +y (image down) is the rotation axis. The shift is not the same for
every star because the projection is tan(angle), which stretches toward
the image edges, and stars off the horizontal centre line also move slightly
in v.""")

    # Check against the closed form for every star.
    ry_minus = np.array([[np.cos(np.radians(SLEW_DEG)), 0, -np.sin(np.radians(SLEW_DEG))],
                         [0, 1, 0],
                         [np.sin(np.radians(SLEW_DEG)), 0, np.cos(np.radians(SLEW_DEG))]])
    closed_form = vec_ref @ ry_minus.T
    print(f"\nPipeline vs closed form R_y(-5 deg) s_C, all stars: "
          f"max difference {np.max(np.abs(closed_form - vec_new)):.1e}")

    # ---- inverse: camera-frame vectors back to celestial directions --------
    worst = 0.0
    for q, camera_vectors in ((q_ref, vec_ref), (q_new, vec_new)):
        recovered = attitude_to_transform(q).apply(camera_vectors)   # C_GC
        for (ra, dec), v in zip(catalog, recovered):
            worst = max(worst, angular_separation(v, radec_to_unit(ra, dec)))
    print(f"Inverse attitude transform recovers the catalog directions to "
          f"{worst * 3600:.1e} arcsec (worst star, both attitudes)")

    # Full loop from pixels for the visible stars.
    back = attitude_to_transform(q_new).apply(camera.unproject(pix_new[vis_new]))
    worst_px = max(angular_separation(v, radec_to_unit(ra, dec))
                   for v, (ra, dec) in zip(back, catalog[vis_new]))
    print(f"Pixels -> rays -> celestial for visible stars: "
          f"{worst_px * 3600:.1e} arcsec")

    # ---- figure -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6.4))
    ax.scatter(pix_ref[vis_ref, 0], pix_ref[vis_ref, 1], s=70,
               facecolors="none", edgecolors="tab:blue", linewidths=1.6,
               label=f"Reference attitude ({vis_ref.sum()} stars)")
    ax.scatter(pix_new[vis_new, 0], pix_new[vis_new, 1], s=45,
               color="tab:orange",
               label=f"After +{SLEW_DEG:g}° about camera y ({vis_new.sum()} stars)")
    for i in np.flatnonzero(both):
        ax.annotate("", xy=pix_new[i], xytext=pix_ref[i],
                    arrowprops=dict(arrowstyle="->", color="gray", alpha=0.6))
    ax.plot(camera.cx, camera.cy, "k+", markersize=14, label="Principal point")
    ax.set_xlim(0, camera.width)
    ax.set_ylim(camera.height, 0)            # row 0 at the top
    ax.set_aspect("equal")
    ax.set_xlabel("u [px]  (camera +x, image right)")
    ax.set_ylabel("v [px]  (camera +y, image down)")
    ax.set_title(f"Predicted star positions: camera pans right {SLEW_DEG:g}°, "
                 f"stars move left ~{-predicted_centre_shift:.0f} px")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.tight_layout()
    out = Path(__file__).with_name("star_projection_experiment.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out.name}")


if __name__ == "__main__":
    main()
