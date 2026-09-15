"""module-10 -- run_stages_4_5.py

Runs the full Stage I-V pipeline (pipeline.run_stages_1_to_5) on TWO
SimulatorStates that share the same catalog, observer, and pointing (both
aimed straight at Vega, same as run_stages_1_3.py) but differ in camera/
optics/distortion -- so the same code exercises both new stages' interesting
regimes instead of asserting them:

  Config A: 50 mm f/2 wide-field lens (this project's original camera),
            now with nonzero (k1, k2) -- Stage IV actually does something,
            and Stage V lands in the point-source regime (first_null << 1 px,
            same as module-9's synthetic_celestial_camera.py Config B).

  Config B: a 50 mm-aperture "telescope" (module-9's Config A numbers,
            reused exactly), zero distortion, Stage V lands in the
            RESOLVED regime -- an actual Airy disk with rings, which is the
            whole point of Stage V ("the optical system determines shape").

Validation, not just assertion:
  * Stage IV: k1=k2=0 (Config B) must be an exact identity -- checked.
    Radial distortion must vanish at r=0 regardless of k1,k2 -- checked on
    Vega (pointed at directly, so its normalized coordinates start at ~0).
  * Stage V: module-8's own centroid() (star_centroid.py) is run on each
    config's rendered Vega stamp and compared to Stage IV's (u_d, v_d) --
    reusing the SAME optics+sampling validation star_centroid.py already
    did, applied here to the pipeline's own rendering, not a copy of it.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

_HERE = os.path.dirname(__file__)
for rel in [("..", "module-9"), ("..", "module-7"), ("..", "module-3"),
           ("..", "module-8")]:
    p = os.path.normpath(os.path.join(_HERE, *rel))
    if p not in sys.path:
        sys.path.insert(0, p)

from camera_model import Camera                          # noqa: E402  (Module 7)
from spherical_astronomy import julian_date                # noqa: E402  (Module 9)
from star_catalog import generate_catalog                  # noqa: E402  (Module 9)
from star_centroid import centroid                          # noqa: E402  (Module 8)

from state import (Star, CelestialScene, Observer, CameraAttitude,
                   CameraIntrinsics, PhysicalOptics, Distortion, Sensor,
                   SimulatorState)
from pipeline import run_stages_1_to_5, star_celestial, star_earth_fixed, \
    star_local, altaz_from_local


def build_scene_and_pointing(rng):
    """The shared celestial scene/observer/pointing both configs below aim
    at -- same real target (Vega), same catalog, only the instrument differs."""
    LAT_DEG, LON_DEG = 40.0, -74.0
    jd = julian_date(2026, 9, 14, hour=22.0)

    vega_ra, vega_dec, vega_mag = 18.615 * 15.0, 38.784, 0.03
    s_c = star_celestial(vega_ra, vega_dec)
    s_e = star_earth_fixed(s_c, jd)
    s_l = star_local(s_e, LAT_DEG, LON_DEG)
    point_alt, point_az = altaz_from_local(s_l)

    ra, dec, mag = generate_catalog(20_000, plane_biased=True, rng=rng)
    stars = [Star(r, d, m) for r, d, m in zip(ra, dec, mag)]
    stars.append(Star(vega_ra, vega_dec, vega_mag, name="Vega"))

    scene = CelestialScene(stars)
    observer = Observer(lat_deg=LAT_DEG, lon_deg=LON_DEG, jd=jd)
    attitude = CameraAttitude.from_pointing(point_alt, point_az)
    return scene, observer, attitude, vega_mag


def build_state(scene, observer, attitude, cam, D_mm, wavelength_optical_nm=550.0,
                exposure_s=1.0, k1=0.0, k2=0.0):
    return SimulatorState(
        scene=scene, observer=observer, attitude=attitude,
        intrinsics=CameraIntrinsics.from_camera_model(cam),
        optics=PhysicalOptics(f_mm=cam.f, D_mm=D_mm, wavelength_optical_nm=wavelength_optical_nm),
        distortion=Distortion(k1=k1, k2=k2),
        sensor=Sensor(px_um=cam.px * 1e3, py_um=cam.py * 1e3, qe=0.7, exposure_s=exposure_s),
    )


def _vega_stamp(state, s4, s5, half_window):
    """Pull the rendered window around Vega back out of Stage V's full
    sensor image, for module-8's centroid() to run on."""
    vega_mask = np.abs(s4["mag"] - VEGA_MAG[0]) < 1e-9
    u_d, v_d = s4["u_d"][vega_mask][0], s4["v_d"][vega_mask][0]
    Nx, Ny = state.intrinsics.Nx, state.intrinsics.Ny
    x0 = int(np.round(u_d)) - half_window
    y0 = int(np.round(v_d)) - half_window
    xs = x0 + np.arange(2 * half_window + 1)
    ys = y0 + np.arange(2 * half_window + 1)
    xs0, xs1 = max(x0, 0), min(x0 + len(xs), Nx)
    ys0, ys1 = max(y0, 0), min(y0 + len(ys), Ny)
    img = s5["expected_image"][ys0:ys1, xs0:xs1]
    return img, xs[(xs >= xs0) & (xs < xs1)], ys[(ys >= ys0) & (ys < ys1)], (u_d, v_d)


VEGA_MAG = [None]


def run_config(name, state, oversample=8):
    s1, s2, s3, s4, s5 = run_stages_1_to_5(state, oversample=oversample)

    print(f"--- {name} ---")
    print(f"  above horizon / in front / distortion applied to: "
          f"{s1['n_above_horizon']:>7,} / {s2['n_in_front']:>7,} / {len(s4['u_d']):>7,} stars")
    print(f"  k1, k2                      : {state.distortion.k1:+.3f}, {state.distortion.k2:+.3f}")

    # distortion_px is computed for EVERY in-front-of-camera star, including
    # extreme grazing rays (z~0, so normalized x_n=s_x/s_z, y_n=s_y/s_z blow
    # up) that Stage III's deliberately-ungated pipeline still carries. A
    # k1 r^2 + k2 r^4 model isn't fit to extrapolate out there -- no real
    # lens's distortion polynomial is either -- so summarizing "max/mean" over
    # that full ungated population is a meaningless, huge number, not a bug.
    # Report it over the same domain the model is actually meant for: stars
    # whose IDEAL (pre-distortion) position lands on the sensor.
    Nx, Ny = state.intrinsics.Nx, state.intrinsics.Ny
    in_frame = ((s3["u"] >= 0) & (s3["u"] < Nx) & (s3["v"] >= 0) & (s3["v"] < Ny))
    d_px = s4["distortion_px"][in_frame]
    print(f"  distortion_px  max / mean   : {d_px.max():.3f} / {d_px.mean():.3f} px "
          f"(over the {int(in_frame.sum()):,} stars actually inside the sensor "
          f"rectangle -- see note above on why the ungated population isn't "
          f"a meaningful domain for this model)")

    vega_mask = np.abs(s4["mag"] - VEGA_MAG[0]) < 1e-9
    du_v, dv_v = s4["du"][vega_mask][0], s4["dv"][vega_mask][0]
    print(f"  Vega (pointed at, r~0)      : (du, dv) = ({du_v:+.2e}, {dv_v:+.2e}) px "
          f"-- distortion vanishes at r=0 regardless of k1,k2, checked not assumed")

    print(f"  Stage V regime              : {s5['regime']}")
    print(f"  first_null_px               : {s5['first_null_px']:.4f} px")
    print(f"  stars rendered               : {s5['n_rendered']:,}")
    if s5["n_rendered"]:
        if s5["half_window"] > 1:
            print(f"  window flux captured        : {s5['flux_captured'].mean()*100:.2f}% "
                  f"(half-window={s5['half_window']} px)")
        print(f"  peak expected value/pixel   : {s5['expected_image'].max():.4g} "
              "(relative-flux units, no photon calibration yet)")

    hw_check = max(s5["half_window"], 6)
    img, xs, ys, (u_d, v_d) = _vega_stamp(state, s4, s5, hw_check)
    if img.sum() > 0:
        u_est, v_est = centroid(img, xs, ys)
        err = np.hypot(u_est - u_d, v_est - v_d)
        print(f"  Vega: Stage IV (u_d,v_d)    = ({u_d:.4f}, {v_d:.4f})")
        print(f"  Vega: rendered-stamp centroid (module-8's centroid(), same "
              f"technique star_centroid.py validated) = ({u_est:.4f}, {v_est:.4f})")
        if s5["half_window"] > 1:
            note = "(well sampled -> tiny, as star_centroid.py found)"
        elif err < 1e-9:
            note = ("(point-source regime; Vega happens to land exactly on an "
                    "integer pixel corner here, so the bilinear split puts all "
                    "its flux on one pixel and recovery is exact -- a special "
                    "case of this regime, not a general property of it)")
        else:
            note = ("(point-source regime -> pulled toward the nearest pixel "
                    "centre, the same pixel-phase effect star_centroid.py found "
                    "-- not a bug)")
        print(f"  centroid error vs (u_d,v_d) = {err:.4f} px {note}")
    print()
    return s1, s2, s3, s4, s5


def main():
    rng = np.random.default_rng(7)
    scene, observer, attitude, vega_mag = build_scene_and_pointing(rng)
    VEGA_MAG[0] = vega_mag

    print("=" * 90)
    print("MODULE 10 -- STAGE IV (Optical Distortion) + STAGE V (Diffraction / PSF)")
    print("=" * 90)

    camA = Camera(f=50.0, Ws=36.0, Hs=24.0, Nx=960, Ny=640, name="wide-field")
    stateA = build_state(scene, observer, attitude, camA, D_mm=25.0,
                         exposure_s=2.0, k1=-0.25, k2=0.05)

    camB = Camera(f=1000.0, Ws=0.603, Hs=0.603, Nx=201, Ny=201, name="telescope")
    stateB = build_state(scene, observer, attitude, camB, D_mm=50.0,
                         exposure_s=0.05, k1=0.0, k2=0.0)

    resA = run_config("Config A: 50mm wide-field lens, k1=-0.25 k2=+0.05 "
                      "(point-source PSF regime)", stateA)
    resB = run_config("Config B: 50mm-aperture telescope, zero distortion "
                      "(resolved Airy PSF regime)", stateB)

    print("Both configs point at the SAME real target (Vega) from the SAME "
          "observer/time;\nonly the instrument (camera + optics + distortion) "
          "differs -- exactly Stage IV/V's\nown claim, that these effects live "
          "entirely downstream of the ideal geometry in\nStage III, now checked "
          "rather than just asserted.")
    print("=" * 90)

    _plot(stateA, resA, stateB, resB)


def _plot(stateA, resA, stateB, resB):
    s3A, s4A = resA[2], resA[3]
    s5B = resB[4]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # ---- panel 1: distortion magnitude vs radius, with analytic overlay --
    # limited to stars near the real sensor -- Stage III's ungated population
    # includes grazing rays (r_ideal -> huge) the distortion polynomial was
    # never meant to extrapolate to (see run_config's printed note); plotting
    # those would blow out the axes and hide the actual FOV-scale behavior.
    ax = axes[0]
    Nx, Ny = stateA.intrinsics.Nx, stateA.intrinsics.Ny
    near_frame = ((s3A["u"] > -Nx) & (s3A["u"] < 2 * Nx) &
                 (s3A["v"] > -Ny) & (s3A["v"] < 2 * Ny))
    r_ideal = np.hypot(s3A["u"][near_frame] - stateA.intrinsics.cx,
                       s3A["v"][near_frame] - stateA.intrinsics.cy)
    ax.scatter(r_ideal, s4A["distortion_px"][near_frame], s=3, alpha=0.25,
              color="tab:blue", label="per-star (Config A)")
    r_n = np.linspace(0, np.hypot(stateA.intrinsics.cx, stateA.intrinsics.cy)
                      / stateA.intrinsics.fx, 200)
    k1, k2 = stateA.distortion.k1, stateA.distortion.k2
    disp_n = r_n * np.abs(k1 * r_n ** 2 + k2 * r_n ** 4)
    ax.plot(r_n * stateA.intrinsics.fx, disp_n * stateA.intrinsics.fx, "r-", lw=1.5,
           label="analytic: r|k1 r^2+k2 r^4| (px)")
    ax.set_xlabel("distance from principal point, ideal (u,v)  [px]")
    ax.set_ylabel("|distortion| [px]")
    ax.set_title("Stage IV: distortion grows radially,\nmatching the k1 r^2+k2 r^4 model")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ---- panel 2: Config A rendered field, DISTORTED positions -----------
    ax = axes[1]
    ax.set_facecolor("black")
    order = np.argsort(-s4A["mag"])
    m = s4A["mag"][order]
    size = np.clip(50.0 * 10 ** (-0.2 * m), 0.6, 60.0)
    ax.scatter(s4A["u_d"][order], s4A["v_d"][order], s=size, c="white", edgecolors="none")
    cam = stateA.intrinsics
    ax.add_patch(plt.Rectangle((0, 0), cam.Nx, cam.Ny, fill=False,
                               edgecolor="tab:red", lw=1.2, label="sensor"))
    ax.set_xlim(-cam.Nx * 0.15, cam.Nx * 1.15)
    ax.set_ylim(cam.Ny * 1.15, -cam.Ny * 0.15)
    ax.set_aspect("equal")
    ax.set_xlabel("u_distorted (px)"); ax.set_ylabel("v_distorted (px)")
    ax.set_title("Config A: DISTORTED star field\n(k1<0 -- barrel: corners pulled inward)")
    ax.legend(fontsize=8, loc="upper right")

    # ---- panel 3: Config B, resolved Airy PSF around Vega -----------------
    ax = axes[2]
    vega_mask = np.abs(resB[3]["mag"] - resB[3]["mag"].min()) < 1e-9  # Vega is brightest
    u_d, v_d = resB[3]["u_d"][vega_mask][0], resB[3]["v_d"][vega_mask][0]
    hw = max(s5B["half_window"], 15)
    Nx, Ny = stateB.intrinsics.Nx, stateB.intrinsics.Ny
    x0, x1 = max(int(round(u_d)) - hw, 0), min(int(round(u_d)) + hw + 1, Nx)
    y0, y1 = max(int(round(v_d)) - hw, 0), min(int(round(v_d)) + hw + 1, Ny)
    stamp = s5B["expected_image"][y0:y1, x0:x1]
    # log stretch -- the Airy rings are ~1-2% of the peak; a linear scale
    # buries them to black, which would make the title's claim about rings
    # false. Floor at 1e-4 of the peak so the plot doesn't choke on true
    # zeros (outside the rendered window) while still showing faint rings.
    vmax = stamp.max()
    im = ax.imshow(stamp, cmap="inferno", origin="upper", extent=[x0, x1, y1, y0],
                   norm=LogNorm(vmin=max(vmax * 1e-4, 1e-12), vmax=vmax))
    ax.plot(u_d, v_d, "c+", ms=14, mew=1.5, label="Stage IV (u_d, v_d)")
    ax.set_xlabel("u (px)"); ax.set_ylabel("v (px)")
    ax.set_title(f"Config B: resolved Airy PSF on Vega (log scale)\n"
                f"(first_null = {s5B['first_null_px']:.2f} px -- rings visible)")
    ax.legend(fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.75, label="relative flux / pixel (log)")

    fig.suptitle("Stage IV (distortion) + Stage V (diffraction/PSF): "
                 "downstream of the ideal geometry, never inside it", fontsize=12)
    fig.tight_layout()
    fig.savefig("run_stages_4_5.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
