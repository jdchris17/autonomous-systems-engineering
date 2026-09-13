"""camera/projection.py  (Camera Model, Phase II: Project 3-D Points)

    project_points(points_3d, R, t, K, image_size=None)

Take world points P_W, move them into camera coordinates

    P_C = R P_W + t

(R, t is the world-to-camera rigid transform: if the camera sits at world
position C with orientation R, then t = -R @ C), and project with the
pinhole model

    u = fx Xc/Zc + cx
    v = fy Yc/Zc + cy

Two ways a point can fail to appear in the image, both handled explicitly:

  * Zc <= 0  -- the point is behind the camera. Naively plugging in the
    formula anyway is actively wrong, not just out of range: a point at
    (-Xc,-Yc,-Zc) has the SAME ratios Xc/Zc, Yc/Zc as its mirror image
    (Xc,Yc,Zc) in front of the camera, so it would land on the exact same
    pixel if you forgot to check Zc. The demo below constructs exactly that
    ghost point to show why the check matters.

  * the projected (u, v) falls outside the sensor -- rejected if
    `image_size = (Nx, Ny)` is given.
"""

import numpy as np
import matplotlib.pyplot as plt

from camera_model import Camera


def project_points(points_3d, R, t, K, image_size=None):
    """Project world points through a pinhole camera.

    points_3d : (N, 3) array of world points
    R, t      : world -> camera rigid transform,  P_C = R @ P_W + t
    K         : (3, 3) intrinsic matrix
    image_size: optional (Nx, Ny) -- if given, points landing outside
                [0, Nx) x [0, Ny) are also rejected

    Returns
    uv     : (N, 2) pixel coordinates (NaN where invalid)
    valid  : (N,) bool, True where the point is in front of the camera AND
             (if image_size given) inside the sensor
    reason : (N,) array of strings: "ok", "behind_camera", or "out_of_bounds"
    """
    P_W = np.atleast_2d(np.asarray(points_3d, dtype=float))
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float)

    P_C = P_W @ R.T + t                       # (N,3):  R @ p + t for every row
    Xc, Yc, Zc = P_C[:, 0], P_C[:, 1], P_C[:, 2]

    in_front = Zc > 0
    uv = np.full((len(P_W), 2), np.nan)
    reason = np.full(len(P_W), "behind_camera", dtype=object)

    # only divide where it's safe (Zc > 0) -- never evaluate X/Z, Y/Z for
    # points behind the camera, even to discard them afterward
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * Xc[in_front] / Zc[in_front] + cx
    v = fy * Yc[in_front] / Zc[in_front] + cy
    uv[in_front, 0] = u
    uv[in_front, 1] = v
    reason[in_front] = "ok"

    valid = in_front.copy()
    if image_size is not None:
        Nx, Ny = image_size
        in_bounds = in_front & (uv[:, 0] >= 0) & (uv[:, 0] < Nx) \
                             & (uv[:, 1] >= 0) & (uv[:, 1] < Ny)
        reason[in_front & ~in_bounds] = "out_of_bounds"
        uv[in_front & ~in_bounds] = np.nan
        valid = in_bounds

    return uv, valid, reason


def _naive_project(points_3d, R, t, K):
    """The WRONG version: no Zc check at all. Used only to show the failure."""
    P_C = np.atleast_2d(points_3d) @ np.asarray(R).T + np.asarray(t)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * P_C[:, 0] / P_C[:, 2] + cx
    v = fy * P_C[:, 1] / P_C[:, 2] + cy
    return np.column_stack([u, v])


def rot_y(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]])


def main():
    cam = Camera(f=35.0, Ws=36.0, Hs=24.0, Nx=1920, Ny=1280, name="35mm demo camera")
    K = cam.K

    # camera sits at world origin, yawed 10 deg so the boresight is not
    # aligned with a world axis -- R, t actually have to do something
    R = rot_y(np.deg2rad(10.0))
    C_world = np.array([0.0, 0.0, 0.0])
    t = -R @ C_world

    # a point dead-centre in the CAMERA frame, at depth 20, converted back to
    # world coordinates (P_C = R(P_W - C_world)  ->  P_W = R^T P_C + C_world)
    boresight_world = R.T @ np.array([0.0, 0.0, 20.0]) + C_world

    # a handful of test points, plus the two purpose-built edge cases
    pts = {
        "on boresight, far":        list(boresight_world),
        "inside FOV, upper-right":  [3.0, -2.0, 12.0],
        "inside FOV, lower-left":   [-3.0, 2.0, 12.0],
        "just outside FOV (side)":  [30.0, 0.0, 10.0],
        "behind camera":            [0.0, 0.0, -8.0],
    }
    names = list(pts.keys())
    P = np.array(list(pts.values()))

    # the deliberate "ghost" case: mirror the boresight point through the
    # camera centre in world space. Because P_C = R(P_W - C_world), mirroring
    # in world space (P_W -> 2 C_world - P_W) exactly negates the camera-frame
    # coordinates too: same X/Z, Y/Z ratios, opposite (behind-camera) Zc.
    ghost_world = 2 * C_world - P[0]
    names.append("ghost: mirror of boresight pt, behind camera")
    P = np.vstack([P, ghost_world])

    uv, valid, reason = project_points(P, R, t, K, image_size=(cam.Nx, cam.Ny))
    uv_naive = _naive_project(P, R, t, K)

    print("=" * 92)
    print("CAMERA MODEL -- PHASE II: PROJECT 3-D POINTS")
    print("=" * 92)
    print(cam.summary())
    print()
    print(f"{'point':>42} {'Zc':>8} {'u,v (correct)':>18} {'u,v (naive, no Zc check)':>26} {'reason':>14}")
    P_C = P @ R.T + t
    for name, p, pc, (u, v), (un, vn), r in zip(names, P, P_C, uv, uv_naive, reason):
        uv_str = f"({u:7.1f},{v:7.1f})" if np.isfinite(u) else "  --  invalid  --"
        naive_str = f"({un:7.1f},{vn:7.1f})"
        print(f"{name:>42} {pc[2]:>8.2f} {uv_str:>18} {naive_str:>26} {r:>14}")

    print()
    ghost_uv = uv_naive[-1]
    boresight_uv = uv_naive[0]
    print("The point directly behind the camera (the 'ghost') naively projects")
    print(f"to pixel ({ghost_uv[0]:.1f}, {ghost_uv[1]:.1f}) -- EXACTLY on top of")
    print(f"the real point in front of the camera at ({boresight_uv[0]:.1f}, "
          f"{boresight_uv[1]:.1f}).")
    print("Same ray, opposite direction, same (u,v) if you skip the Zc <= 0 check.")
    print("The Zc > 0 test in project_points() is what tells them apart.")
    print("=" * 92)

    _plot(cam, R, P, P_C, uv, valid, reason, names)


def _plot(cam, R, P, P_C, uv, valid, reason, names):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))

    # (1) 3-D-ish top-down view of the scene (world X-Z plane)
    colors = {"ok": "tab:green", "behind_camera": "tab:red", "out_of_bounds": "tab:orange"}
    for p, r, name in zip(P, reason, names):
        ax1.scatter(p[0], p[2], color=colors[r], s=60, zorder=3)
        ax1.annotate(name, (p[0], p[2]), fontsize=7, xytext=(4, 4),
                     textcoords="offset points")
    ax1.plot(0, 0, "k^", ms=12)
    ax1.annotate("camera", (0, 0), fontsize=9, xytext=(4, -12),
                 textcoords="offset points")
    # FOV cone edges, rotated into world coordinates by the camera's own
    # orientation R (P_C = R P_W  ->  a camera-frame direction d_C corresponds
    # to world direction R^T d_C)
    half_fov = cam.fov_x / 2.0
    L = 35.0
    for sign in (+1, -1):
        d_cam = np.array([sign * np.sin(half_fov), 0.0, np.cos(half_fov)])
        d_world = R.T @ d_cam
        ax1.plot([0, L * d_world[0]], [0, L * d_world[2]], "k--", lw=1)
    ax1.axhline(0, color="0.85", lw=1)
    ax1.set_xlabel("world X"); ax1.set_ylabel("world Z (depth)")
    ax1.set_title("Scene, top-down (X-Z)\ndashed = horizontal FOV cone")
    ax1.set_aspect("equal"); ax1.grid(True, alpha=0.3)

    # (2) the image plane itself
    ax2.add_patch(plt.Rectangle((0, 0), cam.Nx, cam.Ny, fill=False, edgecolor="k", lw=1.5))
    for (u, v), r, name in zip(uv, reason, names):
        if np.isfinite(u):
            ax2.scatter(u, v, color=colors[r], s=70, zorder=3)
            ax2.annotate(name, (u, v), fontsize=7, xytext=(4, 4),
                         textcoords="offset points")
    ax2.set_xlim(-cam.Nx * 0.15, cam.Nx * 1.15)
    ax2.set_ylim(cam.Ny * 1.15, -cam.Ny * 0.15)     # v grows downward
    ax2.set_xlabel("u (px)"); ax2.set_ylabel("v (px)")
    ax2.set_title("Image plane\ngreen = imaged, orange = outside sensor, "
                  "red = behind camera (not shown)")
    ax2.set_aspect("equal"); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("projection.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
