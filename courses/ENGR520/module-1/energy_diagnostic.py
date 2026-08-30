"""Coding Challenge 2 -- energy as a diagnostic.

Engineering technique: use a known physical invariant to test a numerical
model. For the idealized falling object no forces do work over the fall, so

    E_total = 1/2 m v^2 + m g h

must stay constant. Any drift is integration error, not physics. We watch that
drift grow as the timestep grows.

The model, integrator, and driver are reused unchanged from falling_object.py;
this file only adds the diagnostic and its plots.
"""

import numpy as np
import matplotlib.pyplot as plt

from falling_object import G, STATE0, simulate


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------
MASS = 1.0  # kg -- any convenient value; total energy scales linearly with it


def energy(states, m=MASS):
    """Kinetic, potential, and total mechanical energy for each [h, v] row."""
    h, v = states[:, 0], states[:, 1]
    e_k = 0.5 * m * v ** 2
    e_p = m * G * h
    return e_k, e_p, e_k + e_p


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
DT_BASELINE = 0.01
DT_SWEEP = [1.0, 0.5, 0.1, 0.01]


def plot_energy_baseline():
    times, states = simulate(STATE0, DT_BASELINE)
    e_k, e_p, e_tot = energy(states)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, e_k, label="kinetic  $E_k$")
    ax.plot(times, e_p, label="potential  $E_p$")
    ax.plot(times, e_tot, label="total  $E_{total}$", color="k", lw=2)
    ax.axhline(e_tot[0], ls="--", color="grey", lw=1, label="$E_{total}(0)$")
    ax.set_xlabel("time t (s)")
    ax.set_ylabel(f"energy (J), m = {MASS} kg")
    ax.set_title(f"Energy budget (forward Euler, dt = {DT_BASELINE} s)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("plot_energy.png", dpi=120)


def plot_energy_dt_sweep():
    fig, ax = plt.subplots(figsize=(8, 4))
    print(f"{'dt (s)':>8} {'steps':>8} {'E0 (J)':>12} {'E_end (J)':>12} {'drift %':>10}")
    for dt in DT_SWEEP:
        times, states = simulate(STATE0, dt)
        _, _, e_tot = energy(states)
        drift = 100.0 * (e_tot[-1] - e_tot[0]) / e_tot[0]
        ax.plot(times, e_tot, label=f"dt = {dt} s")
        print(f"{dt:>8} {len(times):>8} {e_tot[0]:>12.3f} {e_tot[-1]:>12.3f} {drift:>10.2f}")

    ax.axhline(energy(STATE0[None, :])[2][0], ls="--", color="grey", lw=1,
               label="true $E_{total}$")
    ax.set_xlabel("time t (s)")
    ax.set_ylabel(f"total energy $E_{{total}}$ (J), m = {MASS} kg")
    ax.set_title("Energy drift grows with timestep")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("plot_energy_dt_sweep.png", dpi=120)


EXPLANATION = """\
Explanation
-----------
Nothing does work on the idealized falling object, so its total mechanical
energy 1/2 m v^2 + m g h is a physical invariant: the true value never changes.
Forward Euler does not know that. It updates velocity exactly but lets height
lag, so the model keeps a little too much potential energy and E_total creeps
upward -- roughly linearly in time, with a slope proportional to dt. Halving
the timestep roughly halves the drift; at dt = 1 s the error is glaring, at
dt = 0.01 s it is negligible on this plot.

The takeaway is the method, not the number: when a model should conserve a
known quantity, track that quantity. It is a cheap, assumption-free check that
flags integration error (and outright bugs) without needing the analytical
solution. The same trick reappears later with orbital energy and angular
momentum."""


if __name__ == "__main__":
    plot_energy_baseline()
    plot_energy_dt_sweep()
    print()
    print(EXPLANATION)
    plt.show()
