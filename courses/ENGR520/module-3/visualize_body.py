"""module-3 -- visualize_body.py  (Visualize the Body)

Run a torque-free tumble, then look at it three ways:

  1. omega_x(t), omega_y(t), omega_z(t)               -- body angular velocity
  2. R_IB(t) applied to the body axes xhat_B, yhat_B, zhat_B, plotted as
     inertial-frame components through time
  3. a rectangular box drawn at a sequence of times, and (if a writer is
     available) an animated GIF

The visualization is here to make the physics legible, nothing more.
"""

import sys

import numpy as np
import matplotlib.pyplot as plt

from inertia_tensor import box_principal_moments
from rigid_body import RigidBody, simulate, initial_state, W, Q
from quaternions import quaternion_to_rotation_matrix

M, A, B, C = 12.0, 2.0, 1.4, 1.0
OMEGA0 = np.array([0.15, 5.0, 0.15])        # nearly about the intermediate axis
DT, T_END = 1e-3, 18.0
HALF = np.array([A, B, C]) / 2


def cuboid_edges():
    s = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    v = s * HALF
    return [(v[i], v[j]) for i in range(8) for j in range(i + 1, 8)
            if np.sum(np.abs(s[i] - s[j])) == 2]


def main():
    body = RigidBody(M, box_principal_moments(M, A, B, C))
    t, X = simulate(body, initial_state(omega=OMEGA0), T_END, DT)

    Rs = np.array([quaternion_to_rotation_matrix(x[Q]) for x in X])   # (N,3,3)
    body_axes_I = Rs                    # columns are xhat_B, yhat_B, zhat_B in I

    print("=" * 70)
    print("VISUALIZE THE BODY -- torque-free tumble")
    print("=" * 70)
    print(f"box dims ({A},{B},{C}), omega0 = {OMEGA0} rad/s (near intermediate axis)")
    print(f"{len(t)} steps, dt = {DT}")
    print(f"T   drift : {np.ptp([body.rotational_KE(x) for x in X]):.2e} J")
    print(f"|L| drift : "
          f"{np.ptp(np.linalg.norm([body.angular_momentum_body(x) for x in X], axis=1)):.2e}")
    print("saved: visualize_body_timeseries.png, visualize_body_snapshots.png")

    _plot_timeseries(t, X[:, W], body_axes_I)
    _plot_snapshots(t, Rs)

    if "--gif" in sys.argv:
        _try_gif(t, Rs)
    else:
        print("(run with  --gif  to also render visualize_body.gif -- slower)")
    print("=" * 70)


def _plot_timeseries(t, w, axes_I):
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0, 0].plot(t, w)
    ax[0, 0].set_title("body angular velocity omega(t)")
    ax[0, 0].set_ylabel("rad/s"); ax[0, 0].legend(["x", "y", "z"], fontsize=8)
    ax[0, 0].grid(True)

    names = ["xhat_B", "yhat_B", "zhat_B"]
    slots = [(0, 1), (1, 0), (1, 1)]
    for a, (r, c), name in zip(range(3), slots, names):
        ax[r, c].plot(t, axes_I[:, :, a])
        ax[r, c].set_title(f"{name} in inertial components")
        ax[r, c].set_ylabel("component"); ax[r, c].set_xlabel("t (s)")
        ax[r, c].legend(["I_x", "I_y", "I_z"], fontsize=8)
        ax[r, c].grid(True)
        ax[r, c].set_ylim(-1.1, 1.1)
    fig.tight_layout()
    fig.savefig("visualize_body_timeseries.png", dpi=120)


def _draw_box(ax, R):
    for p, q in cuboid_edges():
        pp, qq = R @ p, R @ q
        ax.plot(*zip(pp, qq), color="0.4", lw=1)
    for k, col in enumerate(["tab:red", "tab:green", "tab:blue"]):
        v = R[:, k] * (HALF[k] + 0.6)
        ax.quiver(0, 0, 0, *v, color=col, lw=2)
    lim = HALF.max() + 0.8
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def _plot_snapshots(t, Rs):
    idx = np.linspace(0, len(t) - 1, 8).astype(int)
    fig = plt.figure(figsize=(15, 8))
    for n, i in enumerate(idx):
        ax = fig.add_subplot(2, 4, n + 1, projection="3d")
        _draw_box(ax, Rs[i])
        ax.set_title(f"t = {t[i]:.2f} s", fontsize=9)
    fig.suptitle("Body orientation over one tumble cycle "
                 "(red/green/blue = body x/y/z axes)")
    fig.tight_layout()
    fig.savefig("visualize_body_snapshots.png", dpi=120)


def _try_gif(t, Rs):
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except Exception as e:                       # pragma: no cover
        print(f"(animation skipped: {e})")
        return
    step = max(1, len(t) // 120)
    frames = range(0, len(t), step)
    fig = plt.figure(figsize=(4.5, 4.5))
    ax = fig.add_subplot(111, projection="3d")

    def update(i):
        ax.cla()
        _draw_box(ax, Rs[i])
        ax.set_title(f"t = {t[i]:.2f} s")

    anim = FuncAnimation(fig, update, frames=frames, interval=40)
    try:
        anim.save("visualize_body.gif", writer=PillowWriter(fps=25))
        print("saved: visualize_body.gif")
    except Exception as e:                       # pragma: no cover
        print(f"(gif not written: {e})")
    plt.close(fig)


if __name__ == "__main__":
    main()
    plt.show()
