"""module-3 -- sim_intermediate_axis.py  (Simulation Experiment 3: Intermediate-Axis Theorem)

Rigid body with I1 < I2 < I3. Spin it almost perfectly about each principal
axis in turn, with a tiny perturbation on the other two components.

  * about I1 (smallest)      -> stable: perturbation stays small, bounded wobble
  * about I3 (largest)       -> stable: same
  * about I2 (intermediate)  -> UNSTABLE: perturbation grows exponentially, the
                               spin axis flips over and over (tennis-racket /
                               Dzhanibekov effect)

Nothing special is coded for tumbling. We integrate Euler's equations

    omega_dot = I^-1 ( -omega x (I omega) )        (tau = 0)

and the behaviour falls out of the physics.

Linear stability about axis k gives an exponent sigma with

    sigma^2 = Omega^2 * (I_i - I_k)(I_j - I_k) / (I_i I_j)

positive (growth) only for the intermediate axis.
"""

import numpy as np
import matplotlib.pyplot as plt

from inertia_tensor import box_principal_moments
from rigid_body import RigidBody, simulate, initial_state, W

# a > b > c  =>  I_xx < I_yy < I_zz, all distinct
M, A, B, C = 12.0, 2.0, 1.4, 1.0
SPIN = 6.0            # rad/s about the chosen axis
EPS = 0.02           # rad/s seed on the other two components
DT, T_END = 5e-4, 25.0


def predicted_sigma(Ip, axis):
    """Linear-stability exponent for a steady spin about principal `axis`.

    Perturbing Euler's equations about omega = Omega * e_axis gives
        d2(delta)/dt2 = sigma^2 * delta,
        sigma^2 = -Omega^2 (I_i - I_axis)(I_j - I_axis) / (I_i I_j)
    with i, j the other two axes. sigma^2 > 0 (growth) only for the
    intermediate axis.
    """
    i, j = [k for k in range(3) if k != axis]
    val = -SPIN**2 * (Ip[i] - Ip[axis]) * (Ip[j] - Ip[axis]) / (Ip[i] * Ip[j])
    return (np.sqrt(val) if val > 0 else 0.0), val


def main():
    Ip = box_principal_moments(M, A, B, C)
    body = RigidBody(M, Ip)
    print("=" * 74)
    print("EXPERIMENT 3 -- the intermediate-axis (tennis-racket) theorem")
    print("=" * 74)
    print(f"principal moments  I1,I2,I3 = {np.array2string(Ip, precision=4)}  "
          f"(I1 < I2 < I3)")
    print(f"spin {SPIN} rad/s about each axis, {EPS} rad/s seed on one "
          f"perpendicular component\n")

    cases = [(0, "I1  (smallest)"), (2, "I3  (largest)"), (1, "I2  (intermediate)")]
    results = []
    for axis, name in cases:
        w0 = np.zeros(3)
        w0[axis] = SPIN
        w0[(axis + 1) % 3] = EPS          # seed one perpendicular component
        t, X = simulate(body, initial_state(omega=w0), T_END, DT)
        w = X[:, W]
        perp = np.sqrt(np.sum(np.delete(w, axis, axis=1) ** 2, axis=1))
        sigma_pred, val = predicted_sigma(Ip, axis)

        # measure growth/wobble from the perpendicular envelope
        if val > 0:                       # unstable: fit the FIRST exp-growth segment
            lo = int(np.argmax(perp > 3 * perp[0]))
            hi = int(np.argmax(perp > 0.2 * SPIN))
            seg = slice(lo, hi)
            k = np.polyfit(t[seg], np.log(perp[seg]), 1)[0]
            verdict = (f"UNSTABLE  predicted sigma = {sigma_pred:.3f}/s, "
                       f"measured growth = {k:.3f}/s   (max |perp| = "
                       f"{perp.max():.2f} rad/s -- full tumble)")
        else:
            verdict = (f"stable    (sigma^2 = {val:.3f} < 0; bounded wobble, "
                       f"max |perp| = {perp.max():.3f} rad/s)")
        print(f"  spin about {name:<18}: {verdict}")
        results.append((axis, name, t, w, perp))

    # count flips for the intermediate case
    axis, name, t, w, perp = results[2]
    sign = np.sign(w[:, 1])
    flips = np.sum(np.abs(np.diff(sign)) > 1)
    print(f"\n  intermediate-axis run: omega_2 changes sign {flips} times in "
          f"{T_END:g} s -- the body keeps flipping over. Energy and |L| stay")
    print("  conserved throughout (checked in sim_torque_free.py); the tumble is")
    print("  not numerical noise, it is what Euler's equations do here.")
    print("=" * 74)

    _plot(results, Ip)


def _plot(results, Ip):
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    labels = ["omega_1", "omega_2", "omega_3"]
    for ax, (axis, name, t, w, perp) in zip(axes, results):
        for k in range(3):
            ax.plot(t, w[:, k], label=labels[k], lw=1)
        ax.set_ylabel("rad/s")
        ax.set_title(f"spin about {name}")
        ax.legend(ncol=3, fontsize=8, loc="upper right")
        ax.grid(True)
    axes[-1].set_xlabel("t (s)")
    fig.suptitle("Same equations, three initial conditions: only the "
                 "intermediate axis tumbles")
    fig.tight_layout()
    fig.savefig("sim_intermediate_axis.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
