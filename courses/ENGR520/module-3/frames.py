"""module-3 -- frames.py  (Body Frame vs Inertial Frame)

Two frames, always:

    I  inertial  -- non-rotating; we describe overall motion here.
    B  body      -- glued to the rigid body; rotates with it.

The same physical vector has different coordinates in each:

    v_I = R_IB @ v_B          (body -> inertial)
    v_B = R_IB.T @ v_I        (inertial -> body,  because R^-1 = R^T)

R_IB is the body's attitude. Its columns are the body axes in inertial
coordinates. This file just hammers one habit: for every vector, ask
"what frame is this expressed in?"
"""

import numpy as np
import matplotlib.pyplot as plt

from euler_angles import euler_zyx_to_R


def demo_single_attitude():
    print("=" * 72)
    print("ONE ATTITUDE")
    print("=" * 72)
    R_IB = euler_zyx_to_R(np.deg2rad(15), np.deg2rad(30), np.deg2rad(-50))

    # a sensor mounted on the body, pointing along body +z
    boresight_B = np.array([0.0, 0.0, 1.0])
    boresight_I = R_IB @ boresight_B
    print(f"sensor boresight, body coords     v_B = {boresight_B}")
    print(f"same vector, inertial coords       v_I = R_IB v_B = "
          f"({boresight_I[0]:+.4f}, {boresight_I[1]:+.4f}, {boresight_I[2]:+.4f})")
    print(f"round trip R_IB^T v_I              = {np.round(R_IB.T @ boresight_I, 12)}")
    print()

    # something known in inertial: gravity, pointing down -z_I
    g_I = np.array([0.0, 0.0, -9.81])
    g_B = R_IB.T @ g_I
    print(f"gravity, inertial coords           g_I = {g_I}")
    print(f"gravity as the body feels it        g_B = R_IB^T g_I = "
          f"({g_B[0]:+.4f}, {g_B[1]:+.4f}, {g_B[2]:+.4f})")
    print(f"|g_I| = |g_B| = {np.linalg.norm(g_B):.4f}   -- coordinates change, "
          f"the physical vector does not")
    print()


def demo_spinning_body():
    print("=" * 72)
    print("A SPINNING BODY  --  same vector, two very different coordinate histories")
    print("=" * 72)
    omega = np.deg2rad(90.0)         # body spins about its own z at 90 deg/s
    tilt = np.deg2rad(20.0)          # spin axis tilted from inertial z
    R_tilt = euler_zyx_to_R(tilt, 0.0, 0.0)

    marker_B = np.array([1.0, 0.0, 0.0])   # a painted dot on the body equator
    times = np.linspace(0, 4, 9)
    print(f"{'t (s)':>6} | {'marker in BODY coords':>26} | "
          f"{'marker in INERTIAL coords':>28}")
    print("-" * 72)
    traj_I = []
    for t in times:
        R_IB = R_tilt @ euler_zyx_to_R(0.0, 0.0, omega * t)   # spin about body z
        m_I = R_IB @ marker_B
        traj_I.append(m_I)
        print(f"{t:>6.1f} | ({marker_B[0]:+.3f},{marker_B[1]:+.3f},{marker_B[2]:+.3f})"
              f"{'':>7} | ({m_I[0]:+.3f}, {m_I[1]:+.3f}, {m_I[2]:+.3f})")
    print()
    print("In BODY coordinates the marker never moves -- it is bolted on.")
    print("In INERTIAL coordinates it traces a circle. Same dot, same physics,")
    print("different frame. Always know which one you are holding.")
    print("=" * 72)
    return np.array(traj_I), R_tilt


def _plot(traj_I, R_tilt):
    ts = np.linspace(0, 4, 200)
    omega = np.deg2rad(90.0)
    pts = np.array([R_tilt @ euler_zyx_to_R(0, 0, omega * t) @ [1, 0, 0] for t in ts])

    fig = plt.figure(figsize=(6.5, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="tab:blue",
            label="marker in inertial coords")
    ax.scatter(*traj_I.T, color="tab:red", s=20)
    for axis, col in zip("xyz", ["tab:red", "tab:green", "tab:blue"]):
        v = R_tilt[:, "xyz".index(axis)] * 1.4
        ax.quiver(0, 0, 0, *v, color=col, lw=2)
    ax.quiver(0, 0, 0, 0, 0, 1.6, color="0.4", lw=1, ls="--")
    ax.text(0, 0, 1.7, "z_I")
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_zlim(-1.5, 1.5)
    ax.set_xlabel("x_I"); ax.set_ylabel("y_I"); ax.set_zlabel("z_I")
    ax.set_title("Body-fixed marker, viewed from the inertial frame")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("frames.png", dpi=120)


def main():
    demo_single_attitude()
    traj_I, R_tilt = demo_spinning_body()
    _plot(traj_I, R_tilt)


if __name__ == "__main__":
    main()
    plt.show()
