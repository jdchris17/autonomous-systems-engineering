"""camera/star_field.py  (Simulation: Synthetic Star Field)

Bring Phases I-IV together into the core forward model of a star camera:

    s_hat_i          N unit vectors, arbitrary "stars" fixed in the inertial
                     (world) frame, ||s_hat_i|| = 1
    R_CI             camera attitude, INERTIAL -> CAMERA directly (this is
                     the one place in the project where the rotation handed
                     to project_points needs NO transpose -- R_CI is already
                     "world -> camera" by definition, unlike Module 3's
                     camera->world R_IB convention used in
                     camera_orientation.py. Same physics, different label.)
    s_C,i = R_CI @ s_hat_i        rotate every star into the camera frame
    keep s_Cz,i > 0                in front of the camera
    project onto the sensor        pinhole model, Phase II

Plot a black frame with the surviving stars, then rotate the camera (a
"slew") and regenerate. Since these are direction vectors (effectively at
infinity), translation never enters -- only R_CI matters, exactly as in
camera_orientation.py.
"""

import numpy as np
import matplotlib.pyplot as plt

from camera_model import Camera
from projection import project_points
from camera_orientation import attitude_matrix   # reuse the yaw/pitch/roll builder

N_STARS = 80
SEED = 7


# ---------------------------------------------------------------------------
# the star catalog: fixed once, in the inertial (world) frame
# ---------------------------------------------------------------------------
def make_catalog(n=N_STARS, seed=SEED):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))              # isotropic on the sphere
    s_hat = v / np.linalg.norm(v, axis=1, keepdims=True)
    mag = rng.uniform(0.0, 6.0, n)           # a fake but plausible magnitude
    return s_hat, mag


def magnitude_to_size(mag):
    return np.interp(mag, [0.0, 6.0], [90.0, 4.0])


# ---------------------------------------------------------------------------
# pointing: build R_CI so the boresight looks at a chosen world direction
# ---------------------------------------------------------------------------
def look_at(forward_world, ref=(0.0, -1.0, 0.0)):
    """R_cam_world (columns = camera axes in world coords) whose Z axis is
    `forward_world`. Gram-Schmidt against `ref` fixes the roll about that
    axis; falls back to a different reference if forward is nearly parallel
    to it. Returns R_CI = R_cam_world.T (world -> camera) directly, since
    that is what every caller here actually needs."""
    z = np.asarray(forward_world, dtype=float)
    z = z / np.linalg.norm(z)
    ref = np.asarray(ref, dtype=float)
    if abs(np.dot(ref, z)) > 0.999:
        ref = np.array([1.0, 0.0, 0.0])
    x = ref - np.dot(ref, z) * z
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)                        # x (x) y = z by construction
    R_cam_world = np.column_stack([x, y, z])
    return R_cam_world.T                      # R_CI


def slew(R_CI_base, dyaw, dpitch, droll):
    """Apply a small yaw/pitch/roll rotation IN THE CURRENT CAMERA'S OWN
    axes on top of an existing pointing (a spacecraft-style slew)."""
    R_cam_world_base = R_CI_base.T
    delta = attitude_matrix(dyaw, dpitch, droll)     # small rotation, camera axes
    R_cam_world_new = R_cam_world_base @ delta
    return R_cam_world_new.T


# ---------------------------------------------------------------------------
# forward model: stars -> camera frame -> sensor
# ---------------------------------------------------------------------------
def render(s_hat, R_CI, cam):
    """s_C = R_CI @ s_hat for every star; keep s_Cz > 0; project.

    Implemented via project_points (t=0, since only direction matters) so
    there is exactly one place in the project that does pinhole division --
    but the algebra is identical to the literal s_C = R_CI @ s_hat_i,
    keep s_Cz > 0, project sequence.
    """
    t = np.zeros(3)
    uv, valid, reason = project_points(s_hat, R_CI, t, cam.K, image_size=(cam.Nx, cam.Ny))
    return uv, valid, reason


def main():
    s_hat, mag = make_catalog()
    sizes = magnitude_to_size(mag)

    print("=" * 78)
    print("SYNTHETIC STAR FIELD -- the forward model of a star camera")
    print("=" * 78)
    print(f"{N_STARS} stars, fixed unit vectors in the inertial frame "
          f"(seed={SEED})")
    print(f"normalisation check: max | ||s_i|| - 1 | = "
          f"{np.abs(np.linalg.norm(s_hat, axis=1) - 1.0).max():.2e}")

    cam = Camera(f=8.0, Ws=10.0, Hs=8.0, Nx=1024, Ny=820, name="star camera")
    solid_angle = cam.fov_x * cam.fov_y             # small-field approx, steradians
    expected = N_STARS * solid_angle / (4 * np.pi)
    print()
    print(f"star camera: f={cam.f} mm, sensor={cam.Ws}x{cam.Hs} mm, "
          f"{cam.Nx}x{cam.Ny} px")
    print(f"   FOV = {np.degrees(cam.fov_x):.2f} x {np.degrees(cam.fov_y):.2f} deg "
          f"(wide-field, so a handful of the {N_STARS} catalog stars share")
    print(f"   any one frame -- a real narrow star-tracker FOV would need a "
          f"catalog of thousands to match)")
    print(f"   expected stars/frame for an isotropic catalog = N * FOV_solid_angle"
          f" / 4pi = {expected:.2f}")
    print()

    # a sequence of attitudes: start aimed at star 0, then slew around
    R0 = look_at(s_hat[0])
    attitudes = [
        ("aimed at star #0", R0),
        ("small yaw slew (+3 deg)", slew(R0, np.deg2rad(3), 0, 0)),
        ("small pitch slew (+2 deg)", slew(R0, 0, np.deg2rad(2), 0)),
        ("combined slew (yaw+pitch+roll)",
         slew(R0, np.deg2rad(3), np.deg2rad(-2), np.deg2rad(5))),
        ("slewed to star #20", look_at(s_hat[20])),
        ("slewed to star #45", look_at(s_hat[45])),
    ]

    print(f"{'attitude':>34} {'stars in frame':>15}")
    print("-" * 52)
    results = []
    for name, R_CI in attitudes:
        uv, valid, _ = render(s_hat, R_CI, cam)
        print(f"{name:>34} {valid.sum():>15d} / {N_STARS}")
        results.append((name, R_CI, uv, valid))

    print()
    print("Every attitude is a fresh rigid rotation of the SAME fixed catalog --")
    print("no star ever moves in the inertial frame; only R_CI changes, exactly")
    print("as a real star tracker sees a different patch of sky as the")
    print("spacecraft/telescope turns. This loop (rotate -> keep s_Cz>0 ->")
    print("project -> render) is the whole forward model; attitude ESTIMATION")
    print("(matching the observed dots back to catalog stars to recover R_CI)")
    print("is the inverse problem this sets up for.")
    print("=" * 78)

    _plot(results, cam, sizes)


def _plot(results, cam, sizes):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.5))
    for ax, (name, R_CI, uv, valid) in zip(axes.ravel(), results):
        ax.scatter(uv[valid, 0], uv[valid, 1], s=sizes[valid],
                  color="white", edgecolor="none")
        ax.set_xlim(0, cam.Nx); ax.set_ylim(cam.Ny, 0)
        ax.set_facecolor("black")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{name}\n{valid.sum()} stars in frame", fontsize=9.5,
                     color="black")

    fig.suptitle(f"Synthetic star camera: {N_STARS}-star catalog, six "
                 f"attitudes", fontsize=13)
    fig.tight_layout()
    fig.savefig("star_field.png", dpi=120, facecolor="white")


if __name__ == "__main__":
    main()
    plt.show()
