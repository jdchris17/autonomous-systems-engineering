"""Module 2 -- Coding Challenge: break the circular orbit.

Same governing equation as orbit_euler.py / orbit_rk4.py:

    r_dot = v
    v_dot = -mu * r / |r|^3

We change *only* the initial speed, always launched perpendicular to the
radius at r0 = R_E + 400 km, as a multiple of the circular speed
v_c = sqrt(mu / r0):

    0.9  v_c    1.0 v_c    1.1 v_c    1.3 v_c    sqrt(2) v_c

and watch the trajectory move through

    collision / ellipse  ->  circle  ->  ellipse  ->  escape.

Why sqrt(2)?  A trajectory escapes when its specific orbital energy is
non-negative:

    epsilon = v^2/2 - mu/r >= 0   ->   v >= sqrt(2 mu / r) = v_esc

and since v_c = sqrt(mu / r) at the same radius,

    v_esc = sqrt(2) * v_c.

So sqrt(2) v_c is exactly the parabolic escape speed -- the boundary between
a closed ellipse and an open, never-returning path.

Integrator: RK4 from orbit_rk4.py. Model + constants: orbit_euler.py.
No other file changes.
"""

import numpy as np
import matplotlib.pyplot as plt

from orbit_euler import (
    MU_EARTH, R_EARTH, R0, dynamics, circular_speed, orbital_period,
)
from orbit_rk4 import rk4_step


# ---------------------------------------------------------------------------
# Local propagator with events (keeps orbit_euler.simulate untouched)
# ---------------------------------------------------------------------------
def propagate(state0, dt, t_max, r_collide, r_escape):
    """RK4 propagation of the shared model that stops on impact or escape."""
    states = [np.asarray(state0, dtype=float)]
    t, event = 0.0, "max_time"
    while t < t_max:
        s = rk4_step(dynamics, states[-1], t, dt)
        r = np.hypot(s[0], s[1])
        states.append(s)
        t += dt
        if r <= r_collide:
            event = "collision"
            break
        if r >= r_escape:
            event = "escape"
            break
    return np.array(states), event


# ---------------------------------------------------------------------------
# Orbit classification from the invariants (no trajectory needed)
# ---------------------------------------------------------------------------
def classify(r0, v0):
    """Return a dict describing the conic for a perpendicular launch at r0."""
    eps = 0.5 * v0 ** 2 - MU_EARTH / r0          # specific energy
    h = r0 * v0                                  # specific ang. momentum
    e = np.sqrt(max(0.0, 1.0 + 2.0 * eps * h ** 2 / MU_EARTH ** 2))
    scale = MU_EARTH / r0

    out = {"eps": eps, "e": e}
    if eps < -1e-9 * scale:                      # bound
        a = -MU_EARTH / (2.0 * eps)
        r_peri, r_apo = a * (1 - e), a * (1 + e)
        out.update(a=a, r_peri=r_peri, r_apo=r_apo,
                   period=orbital_period(MU_EARTH, a))
        if e < 1e-6:
            out["kind"] = "circle"
        elif r_peri <= R_EARTH:
            out["kind"] = "ellipse -> hits Earth"
        else:
            out["kind"] = "ellipse"
    elif eps > 1e-9 * scale:
        out["kind"] = "hyperbola -> escape"
    else:
        out["kind"] = "parabola -> escape (v = v_esc)"
    return out


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
DT = 2.0
CASES = [
    (0.9, "too slow: r0 becomes apogee, falls to a lower perigee -- "
          "perigee is below the surface, so it re-enters / collides"),
    (1.0, "exactly circular: closed circle at constant radius"),
    (1.1, "a bit fast: r0 becomes perigee, small ellipse bulging outward"),
    (1.3, "much faster but eps still < 0: large, highly eccentric ellipse "
          "-- still bound, still returns"),
    (np.sqrt(2), "eps = 0 exactly: parabolic trajectory, just barely escapes "
                 "and never comes back"),
]


def main():
    v_c = circular_speed(MU_EARTH, R0)
    T_c = orbital_period(MU_EARTH, R0)

    print("=" * 74)
    print("BREAK THE CIRCULAR ORBIT  --  vary only the initial speed")
    print("=" * 74)
    print(f"r0 = {R0/1e3:.1f} km   v_c = sqrt(mu/r0) = {v_c:.2f} m/s")
    print(f"escape speed  v_esc = sqrt(2 mu/r0) = sqrt(2) v_c = {np.sqrt(2)*v_c:.2f} m/s")
    print()
    print(f"{'v0 / v_c':>9} {'v0 (m/s)':>11} {'eps (J/kg)':>13} {'e':>7}  outcome")
    print("-" * 74)

    trajectories = []
    for factor, prediction in CASES:
        v0 = factor * v_c
        info = classify(R0, v0)

        if "period" in info:
            t_max = 2.3 * info["period"]
        else:
            t_max = 7.0 * T_c            # escape: just run long enough to see it leave

        traj, event = propagate([R0, 0.0, 0.0, v0], DT, t_max,
                                r_collide=R_EARTH, r_escape=12.0 * R0)
        trajectories.append((factor, traj, info))

        tag = f"{factor:.3f}" if abs(factor - np.sqrt(2)) > 1e-9 else "sqrt(2)"
        print(f"{tag:>9} {v0:>11.1f} {info['eps']:>13.3e} {info['e']:>7.3f}  "
              f"{info['kind']}  [sim: {event}]")
        print(f"{'':>9} predicted: {prediction}")
        if "r_apo" in info:
            print(f"{'':>9}            perigee alt {info['r_peri']/1e3 - R_EARTH/1e3:8.1f} km"
                  f"   apogee alt {info['r_apo']/1e3 - R_EARTH/1e3:9.1f} km")
        print()

    print("One governing equation, five initial conditions, four qualitatively")
    print("different fates. The dividing lines are set entirely by the specific")
    print("energy eps = v^2/2 - mu/r:  eps < 0 bound, eps = 0 parabolic escape,")
    print("eps > 0 hyperbolic escape. v0 = sqrt(2) v_c lands exactly on eps = 0,")
    print("which is why that number is not arbitrary.")
    print("=" * 74)

    _plot(trajectories, v_c)


def _plot(trajectories, v_c):
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(trajectories)))
    fig, (ax_wide, ax_zoom) = plt.subplots(1, 2, figsize=(13, 6.5))
    theta = np.linspace(0, 2 * np.pi, 200)

    for ax in (ax_wide, ax_zoom):
        ax.fill(R_EARTH / 1e3 * np.cos(theta), R_EARTH / 1e3 * np.sin(theta),
                color="tab:blue", alpha=0.3)
        ax.plot(0, 0, "+", color="k")
        ax.set_aspect("equal")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        ax.grid(True)

    for (factor, traj, info), c in zip(trajectories, colors):
        tag = "sqrt(2)" if abs(factor - np.sqrt(2)) < 1e-9 else f"{factor:g}"
        label = f"{tag} v_c  ({info['kind']})"
        for ax in (ax_wide, ax_zoom):
            ax.plot(traj[:, 0] / 1e3, traj[:, 1] / 1e3, color=c, lw=1.3, label=label)

    ax_wide.set_title("All five trajectories")
    lim = 9 * R0 / 1e3
    ax_wide.set_xlim(-lim, lim)
    ax_wide.set_ylim(-lim, lim)
    ax_wide.legend(fontsize=8, loc="upper right")

    ax_zoom.set_title("Zoom: circle vs. the two bound ellipses vs. re-entry")
    z = 2.2 * R0 / 1e3
    ax_zoom.set_xlim(-z, z)
    ax_zoom.set_ylim(-z, z)

    fig.tight_layout()
    fig.savefig("break_orbit.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
