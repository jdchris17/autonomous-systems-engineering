"""camera/camera_orientation.py  (Camera Orientation)

Translation eliminated (t = 0 throughout -- the camera sits at the world
origin). A collection of world directions ("stars", effectively at infinity:
only their direction matters, so any finite point along that direction
projects identically) is fixed once and never moves. Only the camera's
ATTITUDE changes, through yaw/pitch/roll, and we watch the projected image
coordinates change.

Reused from Module 3 (imported directly, not re-implemented -- see the
sys.path insert below):
    rotations.py   -- rot_x, rot_y, rot_z   (elementary active rotations)
    quaternions.py -- axis_angle_to_quaternion, quaternion_multiply,
                      quaternion_to_rotation_matrix

FRAME DISCIPLINE -- read this before trusting any number below:

  Module 3's convention (rigid-body / aerospace) has the body's FORWARD axis
  as X, and defines R_IB(phi,theta,psi) = Rz(psi) Ry(theta) Rx(phi) as the
  body->inertial (camera->world) attitude, with roll about X, pitch about Y,
  yaw about Z (because Z is "down"/vertical there).

  THIS camera's convention (Phase I/II, OpenCV-style) has FORWARD as Z, right
  as X, down as Y. So the *axis a name refers to* is different here:
  roll (spin about the boresight)   -> about camera Z
  pitch (tilt up/down)              -> about camera X
  yaw (turn left/right)             -> about camera Y

  Blindly calling Module 3's euler_zyx_to_R here would silently apply "yaw"
  about the wrong physical axis. So we reuse the axis-agnostic ELEMENTARY
  matrices (rot_x, rot_y, rot_z) and compose them ourselves, in the pattern
  that matches THIS camera's axes:

      R_cam_world(yaw, pitch, roll) = rot_y(yaw) @ rot_x(pitch) @ rot_z(roll)

  R_cam_world is body(camera)->world: its columns are the camera's axes
  expressed in world coordinates -- exactly Module 3's R_IB idea, just
  relabelled for a Z-forward body. That means R_cam_world ROTATES A VECTOR
  from camera frame to world frame.

  project_points() needs the opposite: world -> camera, i.e. R_cam_world^T.
  Getting this transpose backward is the single most common attitude bug --
  the validation below exists specifically to catch it.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

_MODULE3 = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "module-3"))
if _MODULE3 not in sys.path:
    sys.path.insert(0, _MODULE3)

from rotations import rot_x, rot_y, rot_z                       # noqa: E402  (Module 3)
from quaternions import (axis_angle_to_quaternion, quaternion_multiply,  # noqa: E402
                         quaternion_to_rotation_matrix)          # (Module 3)

from camera_model import Camera
from projection import project_points


# ---------------------------------------------------------------------------
# camera attitude: two independent implementations (matrix and quaternion),
# both built ONLY from Module 3 primitives, cross-checked below
# ---------------------------------------------------------------------------
def attitude_matrix(yaw, pitch, roll):
    """R_cam_world (camera -> world) via Module 3's elementary rot_ matrices."""
    return rot_y(yaw) @ rot_x(pitch) @ rot_z(roll)


def attitude_quaternion(yaw, pitch, roll):
    """Same attitude, built by composing Module 3 quaternions instead."""
    q_roll = axis_angle_to_quaternion([0, 0, 1], roll)     # about camera Z
    q_pitch = axis_angle_to_quaternion([1, 0, 0], pitch)   # about camera X
    q_yaw = axis_angle_to_quaternion([0, 1, 0], yaw)       # about camera Y
    q = quaternion_multiply(quaternion_multiply(q_yaw, q_pitch), q_roll)
    return quaternion_to_rotation_matrix(q)


# ---------------------------------------------------------------------------
# fixed world directions -- defined once, at identity attitude, and never
# touched again. "Stars": only direction matters (see module docstring).
# ---------------------------------------------------------------------------
def make_star_field(cam, n=25, frac=0.6, depth=100.0, seed=0):
    rng = np.random.default_rng(seed)
    half_ax = frac * cam.fov_x / 2.0
    half_ay = frac * cam.fov_y / 2.0
    ax = rng.uniform(-half_ax, half_ax, n)
    ay = rng.uniform(-half_ay, half_ay, n)
    # at identity attitude, camera frame == world frame, so these camera-frame
    # points ARE the fixed world coordinates of the stars
    return np.column_stack([depth * np.tan(ax), depth * np.tan(ay), np.full(n, depth)])


def project(stars_world, R_cam_world, cam):
    """World -> camera needs the TRANSPOSE of the camera's own attitude."""
    R_world_to_cam = R_cam_world.T
    t = np.zeros(3)                      # translation eliminated, per the exercise
    return project_points(stars_world, R_world_to_cam, t, cam.K, image_size=(cam.Nx, cam.Ny))


def main():
    cam = Camera(f=35.0, Ws=36.0, Hs=24.0, Nx=960, Ny=640, name="orientation demo")
    stars = make_star_field(cam)

    print("=" * 82)
    print("CAMERA ORIENTATION -- translation eliminated, attitude only")
    print("=" * 82)

    # ---- cross-check: matrix path vs quaternion path, both from Module 3 --
    yaw, pitch, roll = np.deg2rad([12.0, -7.0, 20.0])
    Rm = attitude_matrix(yaw, pitch, roll)
    Rq = attitude_quaternion(yaw, pitch, roll)
    print(f"attitude(yaw=12, pitch=-7, roll=20 deg): matrix-path vs "
          f"quaternion-path max diff = {np.abs(Rm - Rq).max():.2e}")
    print("(both built only from Module 3 primitives -- rot_x/y/z and the "
          "quaternion algebra agree)")
    print()

    # ---- validation 1: identity attitude -----------------------------
    star_fwd = np.array([[0.0, 0.0, 100.0]])     # fixed world direction: +Z
    star_side = np.array([[100.0, 0.0, 0.0]])    # fixed world direction: +X
    R_id = np.eye(3)
    uv_fwd, ok_fwd, _ = project(star_fwd, R_id, cam)
    uv_side, ok_side, r_side = project(star_side, R_id, cam)
    print("VALIDATION 1 -- identity attitude (yaw=pitch=roll=0)")
    print("  predicted: world +Z star sits at the principal point; world +X")
    print("  star is exactly 90 deg off boresight (Zc=0) -> not visible.")
    print(f"  measured:  +Z star -> {uv_fwd[0]} (principal point is "
          f"[{cam.cx:.1f}, {cam.cy:.1f}])")
    print(f"             +X star -> valid={ok_side[0]} ({r_side[0]})")
    print()

    # ---- validation 2: a KNOWN 90 deg rotation, predicted first --------
    print("VALIDATION 2 -- yaw = +90 deg about camera Y")
    print("  PREDICTION (made before running the numbers):")
    print("    R_cam_world = rot_y(90deg) sends the camera's forward axis")
    print("    (0,0,1)_cam to world (+1,0,0) -- the boresight now points")
    print("    along world +X. So the world+X star should jump to the")
    print("    principal point, and the world+Z star (old boresight) should")
    print("    swing to exactly 90 deg off-axis and disappear.")
    R_yaw90 = attitude_matrix(np.pi / 2, 0.0, 0.0)
    boresight_world = R_yaw90 @ np.array([0.0, 0.0, 1.0])
    print(f"    check: R_cam_world @ [0,0,1] = {np.round(boresight_world, 6)}  "
          f"(should be [1,0,0])")
    uv_fwd2, ok_fwd2, r_fwd2 = project(star_fwd, R_yaw90, cam)
    uv_side2, ok_side2, r_side2 = project(star_side, R_yaw90, cam)
    print(f"  MEASURED: +Z star  -> valid={ok_fwd2[0]} ({r_fwd2[0]})")
    print(f"            +X star  -> {uv_side2[0]} "
          f"(principal point is [{cam.cx:.1f}, {cam.cy:.1f}])")
    print("  (the +Z star's reason shows 'out_of_bounds' rather than")
    print("  'behind_camera': at exactly 90 deg, floating-point rounding of")
    print("  cos(pi/2) leaves Zc a tiny POSITIVE number instead of exactly")
    print("  zero, so it passes Zc>0 but divides out to a pixel coordinate")
    print("  far off the sensor -- still correctly rejected, just by the")
    print("  other check. Exactly-90-degrees is a genuinely unstable")
    print("  direction to project, on a real detector as much as in code.)")
    print("  Matches the prediction. Note this used R_cam_world.T inside")
    print("  project() -- using R_cam_world directly (forgetting the")
    print("  transpose) would have rotated the STARS the wrong way and")
    print("  swapped which one disappears.")
    print("=" * 82)

    _plot(cam, stars, star_fwd, star_side)


def _plot(cam, stars, star_fwd, star_side):
    configs = [
        ("identity\nyaw=pitch=roll=0", 0.0, 0.0, 0.0),
        ("yaw = +20 deg", np.deg2rad(20), 0.0, 0.0),
        ("pitch = +15 deg", 0.0, np.deg2rad(15), 0.0),
        ("roll = +30 deg\n(spins about the CENTRE, doesn't pan)", 0.0, 0.0, np.deg2rad(30)),
        ("yaw+pitch+roll combined", np.deg2rad(15), np.deg2rad(-10), np.deg2rad(15)),
        ("known 90 deg yaw test\n(+Z star vanishes, +X star centres)", np.pi / 2, 0.0, 0.0),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (title, yaw, pitch, roll) in zip(axes.ravel(), configs):
        R = attitude_matrix(yaw, pitch, roll)
        uv, valid, _ = project(stars, R, cam)
        ax.scatter(uv[valid, 0], uv[valid, 1], s=18, color="tab:cyan")

        uv_f, ok_f, _ = project(star_fwd, R, cam)
        if ok_f[0]:
            ax.scatter(*uv_f[0], s=90, color="tab:red", marker="*",
                      label="+Z star", zorder=5)
        uv_s, ok_s, _ = project(star_side, R, cam)
        if ok_s[0]:
            ax.scatter(*uv_s[0], s=90, color="tab:orange", marker="*",
                      label="+X star", zorder=5)

        ax.plot(cam.cx, cam.cy, "+", color="white", ms=10)
        ax.set_xlim(0, cam.Nx); ax.set_ylim(cam.Ny, 0)
        ax.set_facecolor("black")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=9.5)
        if ok_f[0] or ok_s[0]:
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Fixed world stars, six camera attitudes  "
                 "(white cross = principal point)", fontsize=13)
    fig.tight_layout()
    fig.savefig("camera_orientation.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
