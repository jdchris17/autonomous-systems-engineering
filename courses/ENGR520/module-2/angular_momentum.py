"""Module 2 -- Coding Challenge: angular momentum and Kepler's second law.

Specific angular momentum (per unit mass):

    h = r x v          ->   planar:   h_z = x*vy - y*vx

Gravity is a central force: F points along r, so the torque about the central
body is tau = r x F = 0, hence dh/dt = 0 and h_z is constant.

The geometric payoff: the area swept by the radius vector per unit time is

    dA/dt = 1/2 |r x v| = h_z / 2

so a constant h_z means the orbit sweeps *equal areas in equal times* --
Kepler's second law. This file shows all three links in one run:

    r x F = 0   ->   dh/dt = 0   ->   equal areas in equal times

on a deliberately eccentric orbit (1.3 v_c, e ~ 0.69) where the satellite
visibly races through perigee and crawls through apogee -- yet each equal-time
wedge has the same area.
"""

import numpy as np
import matplotlib.pyplot as plt

from orbit_euler import (
    MU_EARTH, R_EARTH, R0, dynamics, euler_step,
    circular_speed, orbital_period,
)
from orbit_rk4 import rk4_step


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def h_z(states):
    """Specific angular momentum (z-component) for each [x, y, vx, vy] row."""
    x, y, vx, vy = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
    return x * vy - y * vx


def swept_area(states):
    """Cumulative area swept by the radius vector from the origin (triangle fan)."""
    x, y = states[:, 0], states[:, 1]
    cross = x[:-1] * y[1:] - y[:-1] * x[1:]          # 2 * triangle area, signed
    return np.concatenate([[0.0], np.cumsum(0.5 * np.abs(cross))])


# ---------------------------------------------------------------------------
# Propagator (shared model + RK4; Euler kept for the contrast)
# ---------------------------------------------------------------------------
def propagate(state0, dt, n_steps, step):
    states = np.empty((n_steps + 1, 4))
    states[0] = state0
    t = 0.0
    for k in range(n_steps):
        states[k + 1] = step(dynamics, states[k], t, dt)
        t += dt
    times = np.arange(n_steps + 1) * dt
    return times, states


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
FACTOR = 1.3        # eccentric bound orbit
N_WEDGES = 6        # equal-time intervals for the Kepler check
DT = 2.0           # small enough that the polygon area ~ the true swept sector


def main():
    v_c = circular_speed(MU_EARTH, R0)
    v0 = FACTOR * v_c
    eps = 0.5 * v0 ** 2 - MU_EARTH / R0
    a = -MU_EARTH / (2.0 * eps)
    T = orbital_period(MU_EARTH, a)
    n_steps = int(round(T / DT))
    state0 = np.array([R0, 0.0, 0.0, v0])

    t_r, s_r = propagate(state0, DT, n_steps, rk4_step)
    t_e, s_e = propagate(state0, DT, n_steps, euler_step)

    hz_r, hz_e = h_z(s_r), h_z(s_e)
    hz0 = hz_r[0]

    print("=" * 74)
    print("ANGULAR MOMENTUM  &  KEPLER'S SECOND LAW")
    print("=" * 74)
    print(f"Orbit: v0 = {FACTOR} v_c = {v0:.1f} m/s   (e ~ {np.sqrt(1+2*eps*(R0*v0)**2/MU_EARTH**2):.3f}, "
          f"period {T/3600:.2f} h)")
    print(f"h_z(0) = x*vy - y*vx = {hz0:.6e} m^2/s")
    print()
    print("Is h_z constant?  (max fractional deviation over one full orbit)")
    print(f"    RK4          : {np.ptp(hz_r)/abs(hz0):.2e}   -> constant to round-off")
    print(f"    forward Euler: {np.ptp(hz_e)/abs(hz0):.2e}   -> leaks, like the energy did")
    print()

    # ---- Kepler II: equal areas in equal times --------------------------
    edges = np.linspace(0, n_steps, N_WEDGES + 1, dtype=int)
    A = swept_area(s_r)
    print(f"Split one orbit into {N_WEDGES} equal time intervals of "
          f"{T/N_WEDGES/60:.1f} min each.")
    print(f"{'interval':>9} {'r_start (km)':>13} {'speed (km/s)':>13} "
          f"{'angle swept':>13} {'area swept (km^2)':>19}")
    areas = []
    for i in range(N_WEDGES):
        lo, hi = edges[i], edges[i + 1]
        area = (A[hi] - A[lo]) / 1e6
        areas.append(area)
        r_lo = np.hypot(s_r[lo, 0], s_r[lo, 1]) / 1e3
        spd = np.hypot(s_r[lo, 2], s_r[lo, 3]) / 1e3
        ang = np.degrees(np.arctan2(s_r[hi, 1], s_r[hi, 0])
                         - np.arctan2(s_r[lo, 1], s_r[lo, 0])) % 360
        print(f"{i+1:>9} {r_lo:>13.0f} {spd:>13.2f} {ang:>12.1f} deg {area:>19.3e}")
    areas = np.array(areas)
    print()
    print(f"area spread: (max-min)/mean = {np.ptp(areas)/areas.mean():.2e}")
    print(f"predicted sweep rate  dA/dt = h_z/2 = {hz0/2:.4e} m^2/s")
    print(f"measured   sweep rate  A_total/T   = {A[-1]/T:.4e} m^2/s")
    print()
    print("The satellite covers wildly different angles in each interval --")
    print("fast and tight near perigee, slow and wide near apogee -- but the")
    print("swept AREAS match to a part in 1e-"
          f"{int(-np.log10(max(np.ptp(areas)/areas.mean(), 1e-16)))}. That is Kepler's second law, and it is")
    print("nothing more than dh/dt = 0 written geometrically:")
    print("    r x F = 0   ->   dh/dt = 0   ->   equal areas in equal times")
    print("=" * 74)

    _plot(t_r, s_r, hz_r, t_e, hz_e, edges, areas, hz0, T)


def _plot(t_r, s_r, hz_r, t_e, hz_e, edges, areas, hz0, T):
    fig = plt.figure(figsize=(13, 6))
    ax_orbit = fig.add_subplot(1, 2, 1)
    ax_hz = fig.add_subplot(2, 2, 2)
    ax_area = fig.add_subplot(2, 2, 4)

    # orbit with equal-time wedges
    theta = np.linspace(0, 2 * np.pi, 200)
    ax_orbit.fill(R_EARTH / 1e3 * np.cos(theta), R_EARTH / 1e3 * np.sin(theta),
                  color="tab:blue", alpha=0.3)
    ax_orbit.plot(s_r[:, 0] / 1e3, s_r[:, 1] / 1e3, color="0.4", lw=0.8)
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(areas)))
    for i, c in enumerate(cmap):
        lo, hi = edges[i], edges[i + 1]
        poly_x = np.concatenate([[0.0], s_r[lo:hi + 1, 0] / 1e3, [0.0]])
        poly_y = np.concatenate([[0.0], s_r[lo:hi + 1, 1] / 1e3, [0.0]])
        ax_orbit.fill(poly_x, poly_y, color=c, alpha=0.7,
                      label=f"wedge {i+1}: {areas[i]:.3e} km$^2$")
    ax_orbit.plot(0, 0, "+k")
    ax_orbit.set_aspect("equal")
    ax_orbit.set_xlabel("x (km)")
    ax_orbit.set_ylabel("y (km)")
    ax_orbit.set_title(f"{len(areas)} equal-time wedges -> equal area, different shape")
    ax_orbit.legend(fontsize=7, loc="lower left")
    ax_orbit.grid(True)

    ax_hz.axhline(hz0, color="grey", ls="--", lw=1)
    ax_hz.plot(t_r / T, hz_r, label="RK4")
    ax_hz.plot(t_e / T, hz_e, label="forward Euler")
    ax_hz.set_ylabel(r"$h_z(t)$  (m$^2$/s)")
    ax_hz.set_xlabel("orbits completed")
    ax_hz.set_title(r"$h_z = x v_y - y v_x$")
    ax_hz.legend(fontsize=8)
    ax_hz.grid(True)

    ax_area.bar(np.arange(1, len(areas) + 1), areas / 1e6, color=cmap)
    ax_area.set_xlabel("wedge #")
    ax_area.set_ylabel(r"area ($10^6$ km$^2$)")
    ax_area.set_title("Swept area per equal-time interval")
    ax_area.grid(True, axis="y")

    fig.tight_layout()
    fig.savefig("angular_momentum.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
