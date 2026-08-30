"""Coding Challenge 3 -- change the physics.

Model A (vacuum):      v_dot = -g
Model B (atmosphere):  m v_dot = -m g + F_D

Aerodynamic drag has magnitude

    |F_D| = 1/2 rho C_D A v^2

and acts opposite to the velocity. Writing v^2 would lose that sign, so we use
v*|v|, which keeps the magnitude v^2 but carries the sign of v:

    F_D = -1/2 rho C_D A v|v|          (always opposes motion)
    v_dot = -g - (rho C_D A / 2m) v|v|

Falling (v < 0):  v|v| = -v^2, so the drag term is +k v^2 -> slows the descent
and the speed levels off at the terminal velocity  v_t = -sqrt(g / k).

Only the dynamics function changes. The integrator and driver are imported
unchanged from falling_object.py -- that is the point of the architecture.
"""

import numpy as np
import matplotlib.pyplot as plt

from falling_object import G, STATE0, dynamics as vacuum_dynamics, simulate


# ---------------------------------------------------------------------------
# Model B -- free fall with quadratic drag
# ---------------------------------------------------------------------------
def make_drag_dynamics(m, rho, c_d, area):
    """Return a dynamics function v_dot = f(state, t) with quadratic drag baked in."""
    k = 0.5 * rho * c_d * area / m          # drag parameter, 1/m

    def drag_dynamics(state, t):
        h, v = state
        v_dot = -G - k * v * abs(v)         # v*|v| -> magnitude v^2, sign of v
        return np.array([v, v_dot])

    drag_dynamics.k = k
    drag_dynamics.v_terminal = -np.sqrt(G / k)
    return drag_dynamics


# ---------------------------------------------------------------------------
# Object + air (any convenient but physical values)
# ---------------------------------------------------------------------------
MASS = 0.05        # kg     -- light plastic ball
RADIUS = 0.05      # m
AREA = np.pi * RADIUS ** 2
C_D = 0.47         # sphere
RHO = 1.225        # kg/m^3 -- sea-level air

model_b = make_drag_dynamics(MASS, RHO, C_D, AREA)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
DT = 0.01


def impact(times, states):
    """Linear-interpolate the crossing h = 0: returns (t_impact, speed)."""
    h = states[:, 0]
    i = np.argmax(h <= 0.0)                 # first index at/below ground
    if i == 0:
        return times[-1], abs(states[-1, 1])
    frac = h[i - 1] / (h[i - 1] - h[i])
    t_hit = times[i - 1] + frac * (times[i] - times[i - 1])
    v_hit = states[i - 1, 1] + frac * (states[i, 1] - states[i - 1, 1])
    return t_hit, abs(v_hit)


def run():
    t_a, s_a = simulate(STATE0, DT, f=vacuum_dynamics)
    t_b, s_b = simulate(STATE0, DT, f=model_b)

    ti_a, vi_a = impact(t_a, s_a)
    ti_b, vi_b = impact(t_b, s_b)

    print(f"{'':<20}{'impact time (s)':>18}{'impact speed (m/s)':>22}")
    print(f"{'Model A  vacuum':<20}{ti_a:>18.3f}{vi_a:>22.3f}")
    print(f"{'Model B  atmosphere':<20}{ti_b:>18.3f}{vi_b:>22.3f}")
    print(f"\nModel B terminal velocity  v_t = {model_b.v_terminal:.3f} m/s"
          f"   (k = {model_b.k:.4f} /m)")

    fig, (ax_h, ax_v) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    ax_h.plot(t_a, s_a[:, 0], label="A  vacuum")
    ax_h.plot(t_b, s_b[:, 0], label="B  atmosphere")
    ax_h.set_ylabel("altitude h (m)")
    ax_h.set_title("Same object, same drop, different physics")
    ax_h.grid(True)
    ax_h.legend()

    ax_v.plot(t_a, s_a[:, 1], label="A  vacuum")
    ax_v.plot(t_b, s_b[:, 1], label="B  atmosphere")
    ax_v.axhline(model_b.v_terminal, ls="--", color="grey", lw=1,
                 label="terminal velocity")
    ax_v.set_ylabel("velocity v (m/s)")
    ax_v.set_xlabel("time t (s)")
    ax_v.grid(True)
    ax_v.legend()

    fig.tight_layout()
    fig.savefig("plot_drag_compare.png", dpi=120)


EXPLANATION = """\
Explanation
-----------
Only one function changed -- the dynamics -- and the same integrator and driver
produced a qualitatively different prediction. In vacuum the ball accelerates
the whole way down and hits fast. In air, drag grows with v^2 until it balances
gravity; the speed then stops increasing and holds at the terminal velocity, so
the ball lands later and much slower.

The sign matters: drag opposes velocity, not "down". Writing v^2 would push the
ball the same way regardless of travel direction. Using v*|v| keeps the v^2
magnitude while carrying the sign of v, so the force flips correctly if the
object ever moves upward.

A model is a set of assumptions. Change the assumptions and you change the
prediction -- the code architecture just makes that swap cheap and contained."""


if __name__ == "__main__":
    run()
    print()
    print(EXPLANATION)
    plt.show()
