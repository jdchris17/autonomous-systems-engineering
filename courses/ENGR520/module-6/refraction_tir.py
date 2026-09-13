"""module-6 -- refraction_tir.py  (Computational Exercise: Refraction and TIR)

A small ray-interface model. Given n1, n2 and an incidence angle theta_i,
Snell's law fixes the transmitted angle:

    n1 sin(theta_i) = n2 sin(theta_t)   ->   theta_t = arcsin( (n1/n2) sin(theta_i) )

Sweep 0 <= theta_i < 90 deg and plot theta_t against theta_i.

When n1 > n2 (light trying to leave the denser medium), (n1/n2) sin(theta_i)
can exceed 1 -- there is no real theta_t, no propagating transmitted ray, and
all the light reflects. That onset is the critical angle

    theta_c = arcsin(n2 / n1)

This is the physics a fiber-optic core relies on to keep light in.
"""

import numpy as np
import matplotlib.pyplot as plt

N_GLASS, N_AIR = 1.50, 1.00


def theta_t(n1, n2, theta_i):
    """Transmitted angle (rad); NaN wherever Snell's law has no real solution."""
    s = (n1 / n2) * np.sin(theta_i)
    return np.where(np.abs(s) <= 1.0, np.arcsin(np.clip(s, -1.0, 1.0)), np.nan)


def critical_angle(n1, n2):
    """theta_c = arcsin(n2/n1), only meaningful (real) when n1 > n2."""
    if n1 <= n2:
        return None
    return np.arcsin(n2 / n1)


def main():
    theta_i = np.deg2rad(np.linspace(0.0, 89.99, 2000))

    tc_out = critical_angle(N_GLASS, N_AIR)     # glass -> air: TIR exists
    tt_out = theta_t(N_GLASS, N_AIR, theta_i)   # dense -> rare
    tt_in = theta_t(N_AIR, N_GLASS, theta_i)    # rare -> dense: always defined

    print("=" * 74)
    print("REFRACTION AND TOTAL INTERNAL REFLECTION")
    print("=" * 74)
    print(f"n_glass = {N_GLASS}, n_air = {N_AIR}")
    print()
    print(f"glass -> air (n1={N_GLASS} > n2={N_AIR}):")
    print(f"   critical angle  theta_c = arcsin(n2/n1) = {np.rad2deg(tc_out):.4f} deg")
    n_prop = np.sum(~np.isnan(tt_out))
    print(f"   {n_prop}/{len(theta_i)} sampled incidence angles still have a "
          f"propagating refracted ray;")
    print(f"   beyond theta_c, sin(theta_t) would exceed 1 -- no such ray "
          f"exists, so ALL the")
    print(f"   light reflects (total internal reflection).")
    print()
    print(f"air -> glass (n1={N_AIR} < n2={N_GLASS}):")
    print(f"   theta_t is defined for every theta_i in [0, 90) deg -- no critical")
    print(f"   angle this direction. As theta_i -> 90 deg (grazing entry),")
    print(f"   theta_t -> {np.rad2deg(tt_in[-1]):.4f} deg, "
          f"which equals theta_c above: the same interface,")
    print(f"   the same angle, seen from the two sides.")
    print()
    for name, ti in [("well below theta_c", 20.0),
                     ("just below theta_c", tc_out and np.rad2deg(tc_out) - 2),
                     ("at theta_c (grazing exit)", tc_out and np.rad2deg(tc_out)),
                     ("above theta_c (TIR)", tc_out and np.rad2deg(tc_out) + 10)]:
        ti_r = np.deg2rad(ti)
        tt = theta_t(N_GLASS, N_AIR, ti_r)
        tag = f"{np.rad2deg(tt):.2f} deg" if not np.isnan(tt) else "no real solution -> TIR"
        print(f"   theta_i = {ti:6.2f} deg ({name:<26}) : theta_t = {tag}")
    print()
    print("Typical step-index fiber (core 1.4600 / cladding 1.4500):")
    tc_fiber = critical_angle(1.4600, 1.4500)
    print(f"   theta_c = {np.rad2deg(tc_fiber):.3f} deg -- any core ray hitting "
          f"the cladding steeper than")
    print(f"   this (measured from the normal) is trapped by TIR and guided "
          f"down the fiber.")
    print("=" * 74)

    _plot(theta_i, tt_out, tt_in, tc_out)


def _plot(theta_i, tt_out, tt_in, tc_out):
    fig = plt.figure(figsize=(15, 6.5))
    ax1 = fig.add_subplot(1, 2, 1)

    ti_deg = np.rad2deg(theta_i)
    ax1.plot(ti_deg, np.rad2deg(tt_in), color="tab:blue",
             label=f"air (n={N_AIR}) -> glass (n={N_GLASS})")
    ax1.plot(ti_deg, np.rad2deg(tt_out), color="tab:red",
             label=f"glass (n={N_GLASS}) -> air (n={N_AIR})")
    tc_deg = np.rad2deg(tc_out)
    ax1.axvline(tc_deg, color="grey", ls="--", lw=1)
    ax1.axhline(90, color="0.85", lw=6, zorder=0)
    ax1.axvspan(tc_deg, 90, color="red", alpha=0.08)
    ax1.plot(tc_deg, 90, "ko", ms=6)
    ax1.annotate(f"theta_c = {tc_deg:.2f} deg\n(same angle, both directions)",
                 (tc_deg, 90), xytext=(tc_deg - 45, 70), fontsize=9,
                 arrowprops=dict(arrowstyle="->"))
    ax1.text(tc_deg + 10, 40, "TIR region (glass->air):\nno real theta_t,\n"
             "all light reflects", color="tab:red", fontsize=9)
    ax1.text(15, 60, "air->glass: theta_t\ndefined everywhere,\n"
             "asymptotes to theta_c", color="tab:blue", fontsize=9)
    ax1.set_xlim(0, 90); ax1.set_ylim(0, 95)
    ax1.set_xlabel("theta_i (deg)"); ax1.set_ylabel("theta_t (deg)")
    ax1.set_title("Transmitted angle vs incidence angle\n"
                  "same interface, both directions -- one shared critical angle")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ray diagrams: glass (y>0, n1) -> air (y<0, n2), three incidence angles
    cases = [("below theta_c", tc_deg - 15, True),
             ("at theta_c", tc_deg, True),
             ("above theta_c: TIR", tc_deg + 15, False)]
    gs = fig.add_gridspec(3, 2)
    for k, (label, ti_deg_, has_t) in enumerate(cases):
        ax = fig.add_subplot(gs[k, 1])
        _draw_rays(ax, ti_deg_, label, has_t)

    fig.tight_layout()
    fig.savefig("refraction_tir.png", dpi=120)


def _draw_rays(ax, ti_deg, label, has_transmitted):
    ti = np.deg2rad(ti_deg)
    tt = theta_t(N_GLASS, N_AIR, ti)

    ax.axhline(0, color="k", lw=1)
    ax.fill_between([-1.3, 1.3], 0, 1.3, color="tab:blue", alpha=0.12)
    ax.fill_between([-1.3, 1.3], -1.3, 0, color="tab:orange", alpha=0.12)
    ax.plot([0, 0], [-1.15, 1.15], "g:", lw=1)

    # incident ray: from upper-left down to origin
    ax.plot([-np.sin(ti), 0], [np.cos(ti), 0], "b-", lw=2)
    # reflected ray: back up to the upper-right, mirrored angle
    ax.plot([0, np.sin(ti)], [0, np.cos(ti)], color="b", lw=2, ls=(0, (5, 2)))
    if has_transmitted and not np.isnan(tt):
        ax.plot([0, np.sin(tt)], [0, -np.cos(tt)], "r-", lw=2)
        ax.annotate(f"theta_t={np.rad2deg(tt):.1f} deg", (np.sin(tt) * 0.55, -0.55),
                    fontsize=7, color="r")
    else:
        ax.text(0.05, -0.5, "no transmitted ray\n(evanescent only)",
                fontsize=7.5, color="0.3")

    ax.annotate(f"theta_i={ti_deg:.1f} deg", (-np.sin(ti) * 0.6, 0.35),
                fontsize=7, color="b")
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(label, fontsize=9)


if __name__ == "__main__":
    main()
    plt.show()
