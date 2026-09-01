"""Module 2 -- the same circular orbit, integrated with classical RK4.

Nothing about the physics changes. We import the model, the driver, and the
diagnostics from orbit_euler.py and swap in a better integrator.

Forward Euler uses the slope at the start of the step. RK4 samples the slope
four times across the step and takes a weighted average:

    k1 = f(x,           t)
    k2 = f(x + dt/2 k1, t + dt/2)
    k3 = f(x + dt/2 k2, t + dt/2)
    k4 = f(x + dt   k3, t + dt)
    x_next = x + dt/6 (k1 + 2 k2 + 2 k3 + k4)

That makes it 4th-order: the per-step error scales as dt^5, the accumulated
error as dt^4. Same step size as the Euler run, dramatically less drift.
"""

import numpy as np
import matplotlib.pyplot as plt

from orbit_euler import (
    MU_EARTH, R_EARTH, R0,
    circular_speed, orbital_period, simulate, euler_step,
    radius, specific_energy,
    DT, N_ORBITS,
)


# ---------------------------------------------------------------------------
# Integrator
# ---------------------------------------------------------------------------
def rk4_step(f, state, t, dt):
    """One classical 4th-order Runge-Kutta step of x_dot = f(x, t)."""
    k1 = f(state, t)
    k2 = f(state + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(state + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(state + dt * k3, t + dt)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def main():
    v_circ = circular_speed(MU_EARTH, R0)
    T = orbital_period(MU_EARTH, R0)
    state0 = np.array([R0, 0.0, 0.0, v_circ])

    t_e, s_e = simulate(state0, DT, N_ORBITS * T, step=euler_step)
    t_r, s_r = simulate(state0, DT, N_ORBITS * T, step=rk4_step)

    r_e, r_r = radius(s_e), radius(s_r)
    eps_e, eps_r = specific_energy(s_e), specific_energy(s_r)

    print("=" * 68)
    print("CIRCULAR LOW-EARTH ORBIT  --  RK4 vs forward Euler (same dt)")
    print("=" * 68)
    print(f"Derived circular speed  v = sqrt(mu/r0) = {v_circ:.2f} m/s")
    print(f"Period T = {T:.1f} s   |   dt = {DT:.0f} s   |   {N_ORBITS} orbits "
          f"({len(t_r)-1} steps)")
    print()
    print("A correct circular orbit returns to its start each period and keeps")
    print("radius and specific energy fixed. Comparing the two integrators:")
    print()
    print(f"{'':<16}{'radius change':>18}{'|epsilon| change':>20}")
    print(f"{'forward Euler':<16}{(r_e[-1]-r_e[0])/1e3:>15.1f} km"
          f"{abs((eps_e[-1]-eps_e[0])/eps_e[0])*100:>17.2f} %")
    print(f"{'RK4':<16}{(r_r[-1]-r_r[0])/1e3:>15.3f} km"
          f"{abs((eps_r[-1]-eps_r[0])/eps_r[0])*100:>17.5f} %")
    print()
    print("Same model, same step size. RK4 costs ~4x the force evaluations")
    print("per step but spends them across the interval instead of all at the")
    print("start. The Euler orbit spirals thousands of km outward; the RK4")
    print("orbit stays closed to sub-metre level. orbital_energy.py scores")
    print("both against dt.")
    print("=" * 68)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.fill(R_EARTH / 1e3 * np.cos(theta), R_EARTH / 1e3 * np.sin(theta),
            color="tab:blue", alpha=0.3, label="Earth")
    ax.plot(s_e[:, 0] / 1e3, s_e[:, 1] / 1e3, lw=0.8, color="tab:red",
            label="forward Euler")
    ax.plot(s_r[:, 0] / 1e3, s_r[:, 1] / 1e3, lw=1.0, color="tab:green",
            label="RK4")
    ax.plot(0, 0, "+", color="k")
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title(f"Same dt = {DT:.0f} s, {N_ORBITS} orbits: Euler drifts, RK4 holds")
    ax.grid(True)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig("orbit_rk4.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
