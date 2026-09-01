"""module-3 -- sim_torque_free.py  (Simulation Experiment 2: Torque-Free Rotation)

tau = 0, arbitrary initial angular velocity. Two quantities must be conserved:

    rotational kinetic energy   T = 1/2 omega^T I omega
    angular momentum            L = I omega   (body frame; its magnitude, and
                                the whole vector once rotated into inertial)

The body-frame components of L and omega move around (the body frame is
rotating), but |L|, T, and the inertial vector L_I = R_IB(q) (I omega) are
fixed. Any drift is integrator error -- the same invariant check used on the
orbits in Module 2.
"""

import numpy as np
import matplotlib.pyplot as plt

from inertia_tensor import box_principal_moments
from rigid_body import RigidBody, simulate, initial_state, W, Q

M, A, B, C = 6.0, 0.40, 0.30, 0.20
OMEGA0 = np.array([1.4, -2.1, 0.8])          # rad/s, not aligned with any axis
DT, T_END = 0.001, 30.0


def main():
    Ip = box_principal_moments(M, A, B, C)
    body = RigidBody(M, Ip)
    print("=" * 72)
    print("EXPERIMENT 2 -- torque-free rotation, invariants must be conserved")
    print("=" * 72)
    print(f"principal moments I = {np.array2string(Ip, precision=5)}")
    print(f"omega0 = {OMEGA0} rad/s")

    t, X = simulate(body, initial_state(omega=OMEGA0), T_END, DT)

    T_rot = np.array([body.rotational_KE(x) for x in X])
    L_body = np.array([body.angular_momentum_body(x) for x in X])
    L_in = np.array([body.angular_momentum_inertial(x) for x in X])
    L_mag = np.linalg.norm(L_body, axis=1)

    def drift(a):
        return np.ptp(a) / np.abs(a).mean()

    print()
    print(f"T   : mean {T_rot.mean():.6f} J   fractional drift {drift(T_rot):.2e}")
    print(f"|L| : mean {L_mag.mean():.6f}     fractional drift {drift(L_mag):.2e}")
    rel_range = np.ptp(L_in, axis=0) / L_mag.mean()
    print(f"L_inertial vector : component ranges "
          f"{np.array2string(np.ptp(L_in, axis=0), precision=3, suppress_small=True)}")
    print(f"                   relative to |L| = {L_mag.mean():.3f}: "
          f"{np.array2string(rel_range, precision=3, suppress_small=True)}")
    print()
    print("T and |L| hold to integrator precision; the inertial angular-momentum")
    print("vector is a fixed arrow in space even though its BODY components")
    print("oscillate as the body tumbles under itself.")
    print("=" * 72)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    ax = axes[0, 0]
    ax.plot(t, X[:, W])
    ax.set_title("omega components, BODY frame (they move)")
    ax.set_xlabel("t (s)"); ax.set_ylabel("rad/s")
    ax.legend(["x", "y", "z"], fontsize=8); ax.grid(True)

    ax = axes[0, 1]
    ax.plot(t, L_body)
    ax.set_title("L = I omega, BODY frame (also moves)")
    ax.set_xlabel("t (s)"); ax.set_ylabel("kg m^2 / s")
    ax.legend(["x", "y", "z"], fontsize=8); ax.grid(True)

    ax = axes[1, 0]
    ax.plot(t, L_in)
    ax.set_title("L in the INERTIAL frame (conserved -> flat)")
    ax.set_xlabel("t (s)"); ax.set_ylabel("kg m^2 / s")
    ax.legend(["x", "y", "z"], fontsize=8); ax.grid(True)

    ax = axes[1, 1]
    ax.plot(t, T_rot / T_rot[0] - 1.0, label="T / T0 - 1")
    ax.plot(t, L_mag / L_mag[0] - 1.0, label="|L| / |L0| - 1")
    ax.set_title("Invariant drift (integrator error only)")
    ax.set_xlabel("t (s)"); ax.set_ylabel("fractional drift")
    ax.legend(fontsize=8); ax.grid(True)

    fig.tight_layout()
    fig.savefig("sim_torque_free.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
