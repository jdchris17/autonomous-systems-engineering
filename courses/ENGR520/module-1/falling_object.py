"""Falling-object model.

The physics is deliberately trivial so the focus stays on architecture:

    model      -> the dynamics x_dot = f(x, t)
    integrator -> a generic forward-Euler step, unaware of the model
    driver     -> propagates a model with an integrator until a stop event
    reference  -> the closed-form solution, for error analysis
    analysis   -> baseline run, error vs. analytical, and a dt convergence sweep

Swapping the model, the integrator, or the stop condition touches exactly one
of those pieces.

State:  x = [h, v],  h = altitude (m), v = vertical velocity (m/s), up positive.
        h_dot = v
        v_dot = -g
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
G = 9.81  # m/s^2


def dynamics(state, t):
    """State derivative x_dot = [h_dot, v_dot] for a body in free fall."""
    h, v = state
    return np.array([v, -G])


# ---------------------------------------------------------------------------
# Integrator (knows nothing about falling objects)
# ---------------------------------------------------------------------------
def euler_step(f, state, t, dt):
    """One explicit (forward) Euler step of x_dot = f(x, t)."""
    return state + dt * f(state, t)


# ---------------------------------------------------------------------------
# Simulation driver
# ---------------------------------------------------------------------------
def simulate(state0, dt, step=euler_step, f=dynamics, stop=None, t_max=1000.0):
    """Propagate `f` from `state0` with `step` until `stop(state)` or t_max.

    Returns
        times  : (N,) array
        states : (N, 2) array of [h, v] rows
    """
    if stop is None:
        stop = lambda s: s[0] <= 0.0  # reached the ground

    t = 0.0
    state = np.asarray(state0, dtype=float)
    times = [t]
    states = [state.copy()]

    while not stop(state) and t < t_max:
        state = step(f, state, t, dt)
        t += dt
        times.append(t)
        states.append(state.copy())

    return np.array(times), np.array(states)


# ---------------------------------------------------------------------------
# Analytical reference
# ---------------------------------------------------------------------------
def analytical(times, state0):
    """Closed-form [h, v] at each time in `times`."""
    h0, v0 = state0
    h = h0 + v0 * times - 0.5 * G * times ** 2
    v = v0 - G * times
    return np.column_stack([h, v])


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
STATE0 = np.array([100.0, 0.0])   # h0 = 100 m, v0 = 0 m/s
DT_BASELINE = 0.01
DT_SWEEP = [1.0, 0.1, 0.01, 0.001]


def plot_baseline():
    times, states = simulate(STATE0, DT_BASELINE)
    heights, velocities = states[:, 0], states[:, 1]

    fig, (ax_h, ax_v) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    ax_h.plot(times, heights)
    ax_h.set_ylabel("altitude h (m)")
    ax_h.set_title(f"Falling object (forward Euler, dt = {DT_BASELINE} s)")
    ax_h.grid(True)

    ax_v.plot(times, velocities, color="tab:orange")
    ax_v.set_ylabel("velocity v (m/s)")
    ax_v.set_xlabel("time t (s)")
    ax_v.grid(True)

    fig.tight_layout()
    fig.savefig("plot_trajectory.png", dpi=120)


def plot_error():
    times, states = simulate(STATE0, DT_BASELINE)
    e_h = states[:, 0] - analytical(times, STATE0)[:, 0]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, e_h)
    ax.set_xlabel("time t (s)")
    ax.set_ylabel(r"$e_h(t) = h_{num} - h_{analytical}$  (m)")
    ax.set_title(f"Integration error, dt = {DT_BASELINE} s")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig("plot_error.png", dpi=120)


def plot_dt_sweep():
    fig, ax = plt.subplots(figsize=(8, 4))
    print(f"{'dt (s)':>8} {'steps':>8} {'max |e_h| (m)':>16} {'final |e_h| (m)':>16}")
    for dt in DT_SWEEP:
        times, states = simulate(STATE0, dt)
        e_h = states[:, 0] - analytical(times, STATE0)[:, 0]
        ax.plot(times, e_h, label=f"dt = {dt} s")
        print(f"{dt:>8} {len(times):>8} {np.abs(e_h).max():>16.4f} {abs(e_h[-1]):>16.4f}")

    ax.set_xlabel("time t (s)")
    ax.set_ylabel(r"$e_h(t)$  (m)")
    ax.set_title("Error vs. analytical for decreasing dt")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("plot_dt_sweep.png", dpi=120)


if __name__ == "__main__":
    plot_baseline()
    plot_error()
    plot_dt_sweep()
    plt.show()
