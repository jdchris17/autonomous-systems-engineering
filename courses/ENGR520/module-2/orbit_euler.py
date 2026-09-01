"""Module 2 -- circular low-Earth orbit, integrated with forward Euler.

Initial system
    mu_E = 3.986004418e14 m^3/s^2      (Earth gravitational parameter)
    R_E  = 6.371e6 m                   (Earth radius)
    h    = 400 km                      (satellite altitude)
    r0   = R_E + h

Required circular speed -- derived, not looked up.
For a circular orbit the gravitational acceleration supplies exactly the
centripetal acceleration:

    v^2 / r = mu / r^2        ->        v = sqrt(mu / r)

Architecture mirrors Module 1: model (dynamics) / integrator (euler_step) /
driver (simulate). Only the model is new -- 2-D inverse-square gravity.

State:  x = [rx, ry, vx, vy]     (planar, Earth at the origin)
        r_dot = v
        v_dot = -mu * r / |r|^3
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MU_EARTH = 3.986004418e14   # m^3/s^2
R_EARTH = 6.371e6           # m
ALTITUDE = 400e3            # m
R0 = R_EARTH + ALTITUDE     # m, initial orbital radius


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def circular_speed(mu, r):
    """Circular orbital speed from v^2/r = mu/r^2  ->  v = sqrt(mu/r)."""
    return np.sqrt(mu / r)


def dynamics(state, t, mu=MU_EARTH):
    """x_dot = [vx, vy, ax, ay] for inverse-square gravity toward the origin."""
    r = state[:2]
    v = state[2:]
    dist = np.hypot(*r)
    acc = -mu * r / dist ** 3
    return np.concatenate([v, acc])


def orbital_period(mu, a):
    """Keplerian period T = 2 pi sqrt(a^3 / mu)."""
    return 2.0 * np.pi * np.sqrt(a ** 3 / mu)


# ---------------------------------------------------------------------------
# Integrator
# ---------------------------------------------------------------------------
def euler_step(f, state, t, dt):
    """One explicit (forward) Euler step of x_dot = f(x, t)."""
    return state + dt * f(state, t)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def simulate(state0, dt, t_max, step=euler_step, f=dynamics):
    """Fixed-step propagation. Returns times (N,) and states (N, 4)."""
    n = int(round(t_max / dt))
    times = np.linspace(0.0, n * dt, n + 1)
    states = np.empty((n + 1, len(state0)))
    states[0] = state0
    for k in range(n):
        states[k + 1] = step(f, states[k], times[k], dt)
    return times, states


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def radius(states):
    return np.hypot(states[:, 0], states[:, 1])


def specific_energy(states, mu=MU_EARTH):
    """epsilon = v^2/2 - mu/r  (energy per unit mass); constant for a real orbit."""
    r = radius(states)
    speed2 = states[:, 2] ** 2 + states[:, 3] ** 2
    return 0.5 * speed2 - mu / r


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
DT = 10.0          # s
N_ORBITS = 5


def main():
    v_circ = circular_speed(MU_EARTH, R0)
    T = orbital_period(MU_EARTH, R0)

    state0 = np.array([R0, 0.0, 0.0, v_circ])
    times, states = simulate(state0, DT, N_ORBITS * T)

    r = radius(states)
    eps = specific_energy(states)

    print("=" * 68)
    print("CIRCULAR LOW-EARTH ORBIT  --  forward Euler")
    print("=" * 68)
    print(f"Altitude h                 : {ALTITUDE/1e3:8.1f} km")
    print(f"Orbital radius r0 = R_E + h : {R0/1e3:8.1f} km")
    print()
    print("Required circular speed, derived from  v^2/r = mu/r^2 :")
    print(f"    v = sqrt(mu / r0)      = {v_circ:8.2f} m/s   ({v_circ/1e3:.3f} km/s)")
    print(f"Orbital period  T = 2*pi*sqrt(r0^3/mu) = {T:8.1f} s   ({T/60:.1f} min)")
    print(f"Integrated {N_ORBITS} orbits at dt = {DT:.0f} s  "
          f"({len(times)-1} steps).")
    print()
    print("DIAGNOSTIC -- a real circular orbit keeps r and epsilon constant:")
    print(f"    radius   : start {r[0]/1e3:9.1f} km   end {r[-1]/1e3:9.1f} km   "
          f"change {(r[-1]-r[0])/1e3:+.1f} km")
    print(f"    epsilon  : start {eps[0]:12.1f}   end {eps[-1]:12.1f}   J/kg   "
          f"change {abs((eps[-1]-eps[0])/eps[0])*100:.1f} %")
    print("    (epsilon is negative = bound; it rises toward 0 here, so the")
    print("     satellite is being handed energy it should never receive.)")
    print()
    print("Forward Euler injects energy every step, so the orbit spirals")
    print("outward instead of closing on itself. The method is stable enough")
    print("to run but not accurate enough to trust -- that is why the next")
    print("file switches to RK4, and orbital_energy.py makes the drift the")
    print("headline result rather than a footnote.")
    print("=" * 68)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.fill(R_EARTH / 1e3 * np.cos(theta), R_EARTH / 1e3 * np.sin(theta),
            color="tab:blue", alpha=0.3, label="Earth")
    ax.plot(states[:, 0] / 1e3, states[:, 1] / 1e3, lw=0.8, color="tab:red",
            label=f"Euler orbit ({N_ORBITS} periods)")
    ax.plot(0, 0, "+", color="k")
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title(f"Forward Euler, dt = {DT:.0f} s -- orbit spirals out")
    ax.grid(True)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig("orbit_euler.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
