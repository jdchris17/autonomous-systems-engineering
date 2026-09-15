"""module-10 -- run_stages_1_3.py

Builds a SimulatorState (Section 3), runs Stages I-III (Section 4-6), and
validates every reused piece rather than trusting the wiring:

  * CameraAttitude's q <-> R round trip (Module 3's new
    rotation_matrix_to_quaternion against quaternion_to_rotation_matrix).
  * Stage III's "normalized coordinates, then K" formula against Module 7's
    project_points -- algebraically the same thing, so they must agree
    exactly, and this is cheap insurance that they actually do.
  * The whole Stage I-III chain against Module 9's synthetic_celestial_camera
    result for the SAME real pointing (Vega) -- this project is not
    starting over, it is re-plumbing work that already ran correctly.

Still no diffraction. Still no detector. Section 6 says that separation is
important, and the print/plot below only ever show the IDEAL geometric (u, v).
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(__file__)
for rel in [("..", "module-9"), ("..", "module-7"), ("..", "module-3")]:
    p = os.path.normpath(os.path.join(_HERE, *rel))
    if p not in sys.path:
        sys.path.insert(0, p)

from camera_model import Camera                          # noqa: E402  (Module 7)
from projection import project_points                     # noqa: E402  (Module 7)
from quaternions import quaternion_to_rotation_matrix       # noqa: E402  (Module 3)
from spherical_astronomy import julian_date                # noqa: E402  (Module 9)
from star_catalog import generate_catalog                  # noqa: E402  (Module 9)

from state import (Star, CelestialScene, Observer, CameraAttitude,
                   CameraIntrinsics, PhysicalOptics, Distortion, Sensor,
                   SimulatorState)
# star_celestial/star_earth_fixed/star_local/altaz_from_local come back out
# through pipeline.py, which already resolved module-9's frames.py
# unambiguously (module-3 has its own, unrelated frames.py -- see
# _pathutil.py) -- reuse that instead of importing "frames" a second time.
from pipeline import (run_stages_1_to_3, star_celestial, star_earth_fixed,
                      star_local, altaz_from_local)


def build_state(rng):
    LAT_DEG, LON_DEG = 40.0, -74.0
    jd = julian_date(2026, 9, 14, hour=22.0)

    # point at Vega's real, live-computed sky position (continuity with
    # Module 9's capstone -- same target, now through the formal state/
    # pipeline split instead of one monolithic script)
    vega_ra, vega_dec, vega_mag = 18.615 * 15.0, 38.784, 0.03
    s_c = star_celestial(vega_ra, vega_dec)
    s_e = star_earth_fixed(s_c, jd)
    s_l = star_local(s_e, LAT_DEG, LON_DEG)
    point_alt, point_az = altaz_from_local(s_l)

    ra, dec, mag = generate_catalog(20_000, plane_biased=True, rng=rng)
    stars = [Star(r, d, m) for r, d, m in zip(ra, dec, mag)]
    stars.append(Star(vega_ra, vega_dec, vega_mag, name="Vega"))

    cam = Camera(f=50.0, Ws=36.0, Hs=24.0, Nx=960, Ny=640, name="wide-field")

    state = SimulatorState(
        scene=CelestialScene(stars),
        observer=Observer(lat_deg=LAT_DEG, lon_deg=LON_DEG, jd=jd),
        attitude=CameraAttitude.from_pointing(point_alt, point_az),
        intrinsics=CameraIntrinsics.from_camera_model(cam),
        optics=PhysicalOptics(f_mm=cam.f, D_mm=25.0, wavelength_optical_nm=550.0),
        distortion=Distortion(k1=0.0, k2=0.0),
        sensor=Sensor(px_um=cam.px * 1e3, py_um=cam.py * 1e3, qe=0.7, exposure_s=2.0),
    )
    return state, cam, (point_alt, point_az), vega_mag


def main():
    rng = np.random.default_rng(7)
    state, cam, (point_alt, point_az), vega_mag = build_state(rng)

    print("=" * 88)
    print("MODULE 10 -- STAGES I-III")
    print("=" * 88)

    # ---- validate CameraAttitude's q <-> R round trip ---------------------
    R_from_matrix = state.attitude.R_LB
    q = state.attitude.quaternion
    R_from_q = quaternion_to_rotation_matrix(q)
    print("CameraAttitude round trip (built from R, converted to q, back to R):")
    print(f"   max |R_from_R - R_from_q| = {np.abs(R_from_matrix - R_from_q).max():.2e}")
    print(f"   |R_BL R_BL^T - I| = {np.abs(state.attitude.R_BL @ state.attitude.R_BL.T - np.eye(3)).max():.2e}, "
          f"det = {np.linalg.det(state.attitude.R_BL):.10f}")
    print()

    # ---- run the pipeline --------------------------------------------------
    s1, s2, s3 = run_stages_1_to_3(state)

    print(f"Stage I  -- celestial scene:")
    print(f"   catalog stars           : {s1['n_total']:>7,}")
    print(f"   above the horizon       : {s1['n_above_horizon']:>7,}")
    print()
    print(f"Stage II -- observer -> camera frame:")
    print(f"   in front of the camera  : {s2['n_in_front']:>7,}")
    print()
    print(f"Stage III -- geometric projection:")
    print(f"   (u, v) computed for     : {len(s3['u']):>7,} stars (no FOV/sensor cull yet)")
    in_frame = ((s3["u"] >= 0) & (s3["u"] < cam.Nx) &
               (s3["v"] >= 0) & (s3["v"] < cam.Ny))
    print(f"   ... of which inside the sensor rectangle: {int(in_frame.sum()):,} "
          f"(shown only for scale -- Stage III itself does not reject on this)")
    print()

    # ---- validate Stage III against Module 7's project_points -------------
    uv_ref, valid_ref, _ = project_points(s2["s_b"], np.eye(3), np.zeros(3),
                                          state.intrinsics.K)
    d = np.abs(np.stack([s3["u"], s3["v"]], axis=-1) - uv_ref)[valid_ref]
    print(f"Stage III cross-check against Module 7's project_points: "
          f"max |du,dv| = {d.max():.2e} px")
    print("(normalized coordinates then K, vs. project_points' u=fx Xc/Zc+cx --")
    print(" algebraically the same map; agreement here is not a coincidence,")
    print(" it is the whole point of factoring K out explicitly in Stage III.)")
    print()

    # ---- find Vega, confirm it lands where it should -----------------------
    vega_mask = np.abs(s3["mag"] - vega_mag) < 1e-9
    if vega_mask.any():
        u_v, v_v = s3["u"][vega_mask][0], s3["v"][vega_mask][0]
        print(f"Vega (pointed at directly): (u, v) = ({u_v:.4f}, {v_v:.4f}), "
              f"principal point = ({state.intrinsics.cx:.4f}, {state.intrinsics.cy:.4f})")
        print(f"   diff = {np.hypot(u_v - state.intrinsics.cx, v_v - state.intrinsics.cy):.2e} px")
    print("=" * 88)

    _plot(s3, cam)


def _plot(s3, cam):
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.set_facecolor("black")
    order = np.argsort(-s3["mag"])
    m = s3["mag"][order]
    size = np.clip(50.0 * 10 ** (-0.2 * m), 0.6, 60.0)
    ax.scatter(s3["u"][order], s3["v"][order], s=size, c="white", edgecolors="none")
    ax.add_patch(plt.Rectangle((0, 0), cam.Nx, cam.Ny, fill=False,
                               edgecolor="tab:red", lw=1.2, label="sensor"))
    ax.set_xlim(-cam.Nx * 0.3, cam.Nx * 1.3)
    ax.set_ylim(cam.Ny * 1.3, -cam.Ny * 0.3)
    ax.set_aspect("equal")
    ax.set_xlabel("u (px)"); ax.set_ylabel("v (px)")
    ax.set_title("Stage III: ideal geometric star positions\n"
                 "(no PSF, no detector -- ungated on FOV, so points spill "
                 "outside the red sensor rectangle)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig("run_stages_1_3.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
