"""Module 2 -- Coding Challenge: energy classification.

Specific orbital energy (energy per unit mass -- keeps the algebra clean):

    epsilon = v^2 / 2  -  mu / r

For the two-body problem epsilon is conserved, and its sign alone fixes the
fate of the orbit:

    epsilon < 0   ->  bound      (ellipse; a = -mu / 2 epsilon, finite period)
    epsilon = 0   ->  parabolic  (escape boundary, v = v_esc = sqrt(2) v_c)
    epsilon > 0   ->  unbound    (hyperbola; leaves and never returns)

The point of this file: classify every initial condition from that one scalar
*before* integrating anything. Brute force would mean propagating each orbit
for a full period -- hours of wall-clock and simulated time for the eccentric
cases -- just to see whether it comes back. epsilon answers in one subtraction.

Part 2 reuses the same scalar the other way round: because epsilon *should*
never change, any drift an integrator introduces is pure numerical error, so
it doubles as a scorecard for forward Euler vs RK4.

Model, constants, integrators: imported unchanged from orbit_euler / orbit_rk4.
"""

import numpy as np
import matplotlib.pyplot as plt

from orbit_euler import (
    MU_EARTH, R_EARTH, R0,
    dynamics, circular_speed, orbital_period, simulate, euler_step, radius,
    specific_energy,
)
from orbit_rk4 import rk4_step


# ===========================================================================
# Part 1 -- classify orbits from epsilon alone
# ===========================================================================
def specific_angular_momentum(states):
    """L = r x v (z-component); constant for any central force."""
    rx, ry, vx, vy = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
    return rx * vy - ry * vx


def classify(r, v):
    """Everything the scalar epsilon (plus L) tells us, with zero integration.

    Assumes a perpendicular launch at radius r (so r is an apsis).
    """
    eps = 0.5 * v ** 2 - MU_EARTH / r
    h = r * v
    scale = MU_EARTH / r                       # ~ |epsilon| of the circular orbit
    e = np.sqrt(max(0.0, 1.0 + 2.0 * eps * h ** 2 / MU_EARTH ** 2))

    info = {"eps": eps, "e": e}
    if eps < -1e-9 * scale:
        a = -MU_EARTH / (2.0 * eps)
        info.update(kind="bound (ellipse)", a=a,
                    period=orbital_period(MU_EARTH, a),
                    r_peri=a * (1 - e), r_apo=a * (1 + e))
        if info["r_peri"] <= R_EARTH:
            info["kind"] = "bound, but perigee < R_E -> re-entry"
    elif eps > 1e-9 * scale:
        info["kind"] = "unbound (hyperbola) -> escape"
    else:
        info["kind"] = "parabolic (epsilon = 0) -> escape boundary"
    return info


def propagate(state0, dt, t_max, r_escape):
    """Minimal RK4 propagation used only to *confirm* the classification."""
    s = np.asarray(state0, dtype=float)
    traj = [s]
    t, event = 0.0, "completed"
    while t < t_max:
        s = rk4_step(dynamics, s, t, dt)
        traj.append(s)
        t += dt
        r = np.hypot(s[0], s[1])
        if r <= R_EARTH:
            event = "hit surface"
            break
        if r >= r_escape:
            event = "left system"
            break
    return np.array(traj), event


FACTORS = [0.5, 0.9, 1.0, 1.1, 1.3, np.sqrt(2), 1.6]


def _tag(k):
    return "sqrt2" if abs(k - np.sqrt(2)) < 1e-9 else f"{k:.2f}"


def part1_classification():
    v_c = circular_speed(MU_EARTH, R0)
    T_c = orbital_period(MU_EARTH, R0)

    print("=" * 78)
    print("PART 1 -- CLASSIFY EACH ORBIT FROM epsilon, BEFORE PLOTTING ANYTHING")
    print("=" * 78)
    print(f"launch radius r0 = {R0/1e3:.1f} km    v_c = sqrt(mu/r0) = {v_c:.1f} m/s")
    print(f"parabolic escape speed  v_esc = sqrt(2) v_c = {np.sqrt(2)*v_c:.1f} m/s")
    print()
    print(f"{'v0/v_c':>7} {'v0 (m/s)':>10} {'epsilon (J/kg)':>16} {'sign':>5} "
          f"{'e':>6}  classification (scalar only)")
    print("-" * 78)

    results = []
    for k in FACTORS:
        v0 = k * v_c
        info = classify(R0, v0)
        sign = "< 0" if info["eps"] < -1 else ("> 0" if info["eps"] > 1 else "= 0")
        print(f"{_tag(k):>7} {v0:>10.1f} {info['eps']:>16.3e} {sign:>5} "
              f"{info['e']:>6.3f}  {info['kind']}")
        results.append((k, v0, info))

    print()
    print("Cost of the alternative -- integrating until the orbit reveals itself:")
    for k, v0, info in results:
        if "period" in info:
            print(f"  {_tag(k):>7} v_c : one full period = {info['period']:>9.0f} s "
                  f"= {info['period']/3600:5.2f} h of propagation to see it return")
        else:
            print(f"  {_tag(k):>7} v_c : never returns -- you would integrate forever "
                  f"waiting for a period that does not exist")
    print("  epsilon gave every one of these answers with one multiply and one "
          "subtract.")
    print()

    # ---- confirm the predictions by actually integrating -------------------
    print("Confirming each prediction with a short RK4 run:")
    trajectories = []
    for k, v0, info in results:
        t_max = 2.3 * info["period"] if "period" in info else 6.0 * T_c
        traj, event = propagate([R0, 0.0, 0.0, v0], dt=5.0, t_max=t_max,
                                r_escape=15.0 * R0)
        if "re-entry" in info["kind"]:
            expected = {"hit surface"}
        elif "period" in info:                 # bound ellipse / circle
            expected = {"completed"}
        else:                                  # parabolic / hyperbolic
            expected = {"left system"}
        ok = "OK" if event in expected else "CHECK"
        print(f"  {_tag(k):>7} v_c : predicted {info['kind']:<44} "
              f"sim -> {event:<12} [{ok}]")
        trajectories.append((k, traj, info))
    print("=" * 78)

    _plot_classification(results, trajectories, v_c)
    return v_c, T_c


def _plot_classification(results, trajectories, v_c):
    ks = np.array([r[0] for r in results])
    eps = np.array([r[2]["eps"] for r in results])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    colors = ["tab:green" if e < 0 else ("tab:orange" if abs(e) < 1 else "tab:red")
              for e in eps]
    ax1.axhline(0.0, color="k", lw=1)
    ax1.axvline(np.sqrt(2), color="grey", ls="--", lw=1)
    ax1.plot(ks, eps / 1e6, "-", color="0.7", lw=1, zorder=1)
    ax1.scatter(ks, eps / 1e6, c=colors, s=70, zorder=3)
    ax1.text(np.sqrt(2), ax1.get_ylim()[1], r"  $v_0=\sqrt{2}\,v_c$: $\epsilon=0$",
             va="top", fontsize=9)
    ax1.text(0.55, eps.max() / 1e6 * 0.5, "bound\n$\\epsilon<0$", color="tab:green",
             fontsize=9)
    ax1.text(1.55, eps.max() / 1e6 * 0.5, "unbound\n$\\epsilon>0$", color="tab:red",
             fontsize=9)
    ax1.set_xlabel(r"$v_0 / v_c$")
    ax1.set_ylabel(r"specific orbital energy $\epsilon$  (MJ/kg)")
    ax1.set_title(r"One scalar fixes the class: the sign of $\epsilon$")
    ax1.grid(True)

    theta = np.linspace(0, 2 * np.pi, 200)
    ax2.fill(R_EARTH / 1e3 * np.cos(theta), R_EARTH / 1e3 * np.sin(theta),
             color="tab:blue", alpha=0.3)
    ax2.plot(0, 0, "+", color="k")
    for (k, traj, info), c in zip(trajectories,
                                  ["tab:green" if info["eps"] < 0 else
                                   ("tab:orange" if abs(info["eps"]) < 1 else "tab:red")
                                   for _, _, info in trajectories]):
        tag = r"$\sqrt{2}$" if abs(k - np.sqrt(2)) < 1e-9 else f"{k:g}"
        ax2.plot(traj[:, 0] / 1e3, traj[:, 1] / 1e3, color=c, lw=1.3,
                 label=f"{tag} $v_c$")
    lim = 6 * R0 / 1e3
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.set_aspect("equal")
    ax2.set_xlabel("x (km)")
    ax2.set_ylabel("y (km)")
    ax2.set_title("...and the trajectories only confirm it\n"
                  "green = bound, orange = parabolic, red = unbound")
    ax2.grid(True)
    ax2.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig("orbital_energy.png", dpi=120)


# ===========================================================================
# Part 2 -- the same scalar as an integrator scorecard
# ===========================================================================
def drift(series):
    return abs((series[-1] - series[0]) / series[0])


def part2_integrator_scorecard(v_c, T_c):
    state0 = np.array([R0, 0.0, 0.0, v_c])       # the clean circular orbit
    integrators = {"forward Euler": euler_step, "RK4": rk4_step}
    dt_sweep = [8.0, 4.0, 2.0, 1.0]
    n_orbits = 2

    print()
    print("=" * 78)
    print("PART 2 -- epsilon SHOULD NOT MOVE, SO ITS DRIFT SCORES THE INTEGRATOR")
    print("=" * 78)
    print(f"{'dt (s)':>7} | {'Euler d(eps)':>14} {'Euler d(L)':>12} | "
          f"{'RK4 d(eps)':>14} {'RK4 d(L)':>12}")
    print("-" * 78)
    for dt in dt_sweep:
        cells = []
        for step in integrators.values():
            _, s = simulate(state0, dt, n_orbits * T_c, step=step)
            cells.append(f"{drift(specific_energy(s)):>14.2e} "
                         f"{drift(specific_angular_momentum(s)):>12.2e}")
        print(f"{dt:>7.0f} | " + " | ".join(cells))
    print()
    print("Euler's epsilon error ~halves when dt halves (1st order) and is one-")
    print("signed, so it compounds every orbit -> the spiral in orbit_euler.png.")
    print("RK4's falls ~30x per halving until it hits round-off near 1e-14.")
    print("Same check, no analytical solution required.")
    print("=" * 78)

    dt0 = 10.0
    fig, (ax_e, ax_l) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    for name, step in integrators.items():
        t, s = simulate(state0, dt0, n_orbits * T_c, step=step)
        orbits = t / T_c
        eps = specific_energy(s)
        angm = specific_angular_momentum(s)
        ax_e.plot(orbits, (eps - eps[0]) / abs(eps[0]), label=name)
        ax_l.plot(orbits, (angm - angm[0]) / abs(angm[0]), label=name)
    ax_e.set_ylabel(r"$(\epsilon - \epsilon_0)/|\epsilon_0|$")
    ax_e.set_title(f"Invariant drift vs orbit number (dt = {dt0:.0f} s)")
    ax_e.grid(True); ax_e.legend()
    ax_l.set_ylabel(r"$(L - L_0)/|L_0|$")
    ax_l.set_xlabel("orbits completed")
    ax_l.grid(True); ax_l.legend()
    fig.tight_layout()
    fig.savefig("orbital_energy_drift.png", dpi=120)


if __name__ == "__main__":
    v_c, T_c = part1_classification()
    part2_integrator_scorecard(v_c, T_c)
    plt.show()
