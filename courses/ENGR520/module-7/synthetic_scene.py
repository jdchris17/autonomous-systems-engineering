"""camera/synthetic_scene.py  (Simulation: Build a Synthetic Scene)

A simple 3-D scene -- a cube, the world coordinate axes, a floor grid, and a
handful of arbitrary points -- rendered through the Phase I/II camera model
(camera_model.Camera, projection.project_points). Then the same scene is
re-rendered after changing, one at a time:

  * camera position
  * camera attitude
  * focal length
  * sensor size

so the camera equation stops being an abstract formula and starts being
"why does the picture look like that."
"""

import numpy as np
import matplotlib.pyplot as plt

from camera_model import Camera
from projection import project_points, rot_y

# ---------------------------------------------------------------------------
# the scene, in WORLD coordinates (fixed for the whole file)
# ---------------------------------------------------------------------------
def _cube_edges(centre, half):
    c = np.array(centre)
    verts = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half + c
    edges = [(verts[i], verts[j]) for i in range(8) for j in range(i + 1, 8)
            if np.sum(np.abs(verts[i] - verts[j]) > 1e-9) == 1]
    return edges


CUBE_EDGES = _cube_edges(centre=(0.0, 0.0, 8.0), half=1.5)

AXES = [
    (np.array([0, 0, 0]), np.array([4, 0, 0]), "tab:red", "X"),
    (np.array([0, 0, 0]), np.array([0, 4, 0]), "tab:green", "Y"),
    (np.array([0, 0, 0]), np.array([0, 0, 4]), "tab:blue", "Z"),
]

def _floor_grid(y=3.0, x_range=(-8, 8), z_range=(0, 16), step=2.0):
    lines = []
    for x in np.arange(x_range[0], x_range[1] + 1e-9, step):
        lines.append((np.array([x, y, z_range[0]]), np.array([x, y, z_range[1]])))
    for z in np.arange(z_range[0], z_range[1] + 1e-9, step):
        lines.append((np.array([x_range[0], y, z]), np.array([x_range[1], y, z])))
    return lines


FLOOR = _floor_grid()

POINTS = np.array([
    [-4.0, -1.0, 10.0],
    [5.0, 0.0, 6.0],
    [2.0, -2.5, 14.0],
    [-2.0, 1.0, 5.0],
    [0.0, -3.0, 9.0],
])


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def _project_segments(segments, R, t, K):
    """Project a list of (p0, p1) world pairs; drop a segment if either end
    is behind the camera (Zc <= 0). No sensor-bound rejection here -- letting
    the axes' xlim/ylim clip the drawing is simpler and shows partial lines
    exiting the frame, which is what a real camera view looks like."""
    ends = np.array([p for seg in segments for p in seg])
    uv, valid, _ = project_points(ends, R, t, K)
    uv = uv.reshape(-1, 2, 2)
    ok = valid.reshape(-1, 2).all(axis=1)
    return uv[ok]


def render(ax, cam, R, t, title):
    K = cam.K

    for p0, p1 in _project_segments(CUBE_EDGES, R, t, K):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="white", lw=1.2)

    for p0, p1, color, label in AXES:
        uv, valid, _ = project_points(np.array([p0, p1]), R, t, K)
        if valid.all():
            ax.plot([uv[0, 0], uv[1, 0]], [uv[0, 1], uv[1, 1]], color=color, lw=2)
            ax.annotate(label, uv[1], color=color, fontsize=9)

    for p0, p1 in _project_segments(FLOOR, R, t, K):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="0.4", lw=0.6)

    uv, valid, _ = project_points(POINTS, R, t, K, image_size=(cam.Nx, cam.Ny))
    ax.scatter(uv[valid, 0], uv[valid, 1], color="yellow", s=40, zorder=5,
              edgecolor="k", linewidth=0.5)

    ax.set_xlim(0, cam.Nx)
    ax.set_ylim(cam.Ny, 0)               # v grows downward
    ax.set_facecolor("black")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def main():
    base_cam = Camera(f=35.0, Ws=36.0, Hs=24.0, Nx=960, Ny=640, name="baseline")
    C0 = np.array([0.0, 0.0, -6.0])
    R0 = np.eye(3)

    configs = [
        ("baseline\nf=35mm, C=(0,0,-6), R=I",
         base_cam, C0, R0),
        ("camera MOVED closer + to the side\nC=(3,-1,-9)",
         base_cam, np.array([3.0, -1.0, -9.0]), R0),
        ("camera ROTATED (yaw +15 deg)",
         base_cam, C0, rot_y(np.deg2rad(15.0))),
        ("focal length DOUBLED (f=70mm)\nnarrower FOV, everything magnified",
         Camera(f=70.0, Ws=36.0, Hs=24.0, Nx=960, Ny=640, name="long f"), C0, R0),
        ("sensor size HALVED (18x12mm)\nsame f -- also narrows the FOV",
         Camera(f=35.0, Ws=18.0, Hs=12.0, Nx=960, Ny=640, name="small sensor"), C0, R0),
        ("all four changes at once",
         Camera(f=70.0, Ws=18.0, Hs=12.0, Nx=960, Ny=640, name="combined"),
         np.array([3.0, -1.0, -9.0]), rot_y(np.deg2rad(15.0))),
    ]

    print("=" * 78)
    print("SYNTHETIC SCENE: cube + axes + floor grid + points, six camera setups")
    print("=" * 78)
    for title, cam, C, R in configs:
        t = -R @ C
        print(f"\n{title.splitlines()[0]}")
        print(f"   f={cam.f} mm, sensor={cam.Ws}x{cam.Hs} mm, "
              f"FOVx={np.degrees(cam.fov_x):.1f} deg,  camera at "
              f"({C[0]:.0f}, {C[1]:.0f}, {C[2]:.0f})")

    print()
    print("Moving the camera changes PERSPECTIVE (what's near looms larger,")
    print("relative positions of objects at different depths shift).")
    print("Rotating the camera PANS the whole scene across the frame rigidly --")
    print("nothing changes shape or relative size, everything just translates")
    print("in (u,v).")
    print("Changing f or sensor size both narrow/widen the FOV and rescale the")
    print("whole image the same way (2 atan(W/2f) is symmetric in W and f) --")
    print("but they are not physically the same lens change (a smaller sensor")
    print("with fixed resolution also changes pixel pitch, hence plate scale).")
    print("=" * 78)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, (title, cam, C, R) in zip(axes.ravel(), configs):
        t = -R @ C
        render(ax, cam, R, t, title)
    fig.suptitle("Same scene, six cameras", fontsize=13)
    fig.tight_layout()
    fig.savefig("synthetic_scene.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
