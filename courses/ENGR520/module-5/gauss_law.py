"""module-5 -- gauss_law.py  (Numerical Gauss-Law Experiment)

Test Gauss's law directly by integrating the point-charge field over a closed
surface:

    Phi_E = closed_integral  E . dA        should equal    q_enclosed / eps0

The Gaussian surface is a sphere of radius R centred at the origin, sampled at
near-uniform points with a Fibonacci spiral. Each point carries area
dA = 4 pi R^2 / N and outward normal n = p / |p|.

Experiments (predict before reading the numbers):

  1. charge at the centre            -> Phi = q/eps0   (exact, any N, by symmetry)
  2. charge off-centre but INSIDE    -> Phi = q/eps0   (field lopsided, integral
                                        unchanged -- flux sees only enclosed charge)
  3. charge OUTSIDE the sphere       -> Phi = 0        (every line in also goes out)
  4. two charges inside              -> Phi = 2q/eps0
  5. one in, one out                 -> Phi = q/eps0

Units SI. Only the physics constants come from fields.py.
"""

import numpy as np
import matplotlib.pyplot as plt

from fields import K_E, EPS0

Q = 1e-9
R_SPHERE = 1.0


def fib_sphere(n, R=R_SPHERE):
    """n near-uniform points on a sphere of radius R (Fibonacci spiral)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)                 # polar angle
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i             # azimuth (golden angle)
    p = np.stack([np.sin(phi) * np.cos(theta),
                  np.sin(phi) * np.sin(theta),
                  np.cos(phi)], axis=1)
    return R * p


def E_field(P, charges):
    """E at points P (n,3) from charges [(q, x, y, z), ...]."""
    E = np.zeros_like(P)
    for q, *rc in charges:
        d = P - np.asarray(rc, float)
        r = np.linalg.norm(d, axis=1, keepdims=True)
        E += K_E * q * d / r**3
    return E


def flux(charges, n=20000, R=R_SPHERE):
    P = fib_sphere(n, R)
    normals = P / R
    dA = 4.0 * np.pi * R**2 / n
    E = E_field(P, charges)
    contrib = np.sum(E * normals, axis=1) * dA         # E . dA at each patch
    return contrib.sum(), P, contrib


CONFIGS = {
    "1  centre":        ([(Q, 0.0, 0.0, 0.0)],               Q),
    "2  off-centre in": ([(Q, 0.6, -0.3, 0.2)],              Q),
    "3  outside":       ([(Q, 2.0, 0.0, 0.0)],               0.0),
    "4  two inside":    ([(Q, 0.5, 0, 0), (Q, -0.4, 0.3, 0)], 2 * Q),
    "5  one in one out":([(Q, 0.3, 0, 0), (Q, 0.0, 0.0, 3.0)], Q),
}


def main():
    print("=" * 74)
    print("NUMERICAL GAUSS-LAW EXPERIMENT   Phi_E = closed_int E.dA  vs  q_enc/eps0")
    print("=" * 74)
    print(f"Gaussian surface: sphere R = {R_SPHERE} m, {20000} Fibonacci points")
    print(f"q = {Q:.1e} C   ->   q/eps0 = {Q/EPS0:.4f}  (V m)")
    print()
    print("PREDICTIONS: config 1,2 -> 1.0 ;  config 3 -> 0.0 ;  "
          "config 4 -> 2.0 ;  config 5 -> 1.0   (in units of q/eps0)")
    print()
    print(f"{'config':>18} {'q_enc/eps0':>12} {'Phi (numeric)':>15} "
          f"{'Phi / (q_enc/eps0)':>20}")
    print("-" * 74)
    results = {}
    for name, (charges, q_enc) in CONFIGS.items():
        Phi, P, contrib = flux(charges)
        ratio = Phi / (q_enc / EPS0) if q_enc != 0 else Phi / (Q / EPS0)
        tgt = f"{q_enc/EPS0:.4f}" if q_enc != 0 else "0 (rel to q/eps0)"
        print(f"{name:>18} {tgt:>12} {Phi:>15.4f} {ratio:>20.6f}")
        results[name] = (charges, q_enc, Phi, P, contrib)
    print()
    print("Config 2 is the headline: the charge is way off to one side, so E")
    print("is many times stronger on the near face of the sphere than the far")
    print("face -- yet the surface integral is the same q/eps0 as the centred")
    print("case. Flux counts enclosed charge, not where it sits. Move it just")
    print("outside (config 5's second charge) and its contribution is exactly 0.")
    print("=" * 74)

    _plot(results)


def _plot(results):
    fig = plt.figure(figsize=(15, 5.5))

    # (1) bar chart of flux ratios
    ax1 = fig.add_subplot(1, 3, 1)
    names = list(results)
    ratios = []
    for name in names:
        charges, q_enc, Phi, _, _ = results[name]
        ratios.append(Phi / (Q / EPS0))
    ax1.bar(range(len(names)), ratios, color="tab:blue")
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels([n.split()[0] for n in names])
    ax1.set_ylabel("Phi / (q/eps0)")
    ax1.set_title("Flux = enclosed charge (in units of q/eps0)\n"
                  "1: centre  2: off-centre in  3: outside  4: 2 in  5: 1 in 1 out")
    for i, r in enumerate(ratios):
        ax1.text(i, r + 0.05, f"{r:.3f}", ha="center", fontsize=9)
    ax1.axhline(0, color="k", lw=0.8)
    ax1.grid(True, axis="y", alpha=0.3)

    # (2) convergence vs N for the three key cases
    ax2 = fig.add_subplot(1, 3, 2)
    Ns = np.unique(np.geomspace(20, 40000, 22).astype(int))
    for name, style in [("1  centre", "-o"), ("2  off-centre in", "-s"),
                        ("3  outside", "-^")]:
        charges = CONFIGS[name][0]
        vals = [flux(charges, n=n)[0] / (Q / EPS0) for n in Ns]
        ax2.plot(Ns, vals, style, ms=4, label=name)
    ax2.set_xscale("log")
    ax2.axhline(1.0, color="grey", ls=":", lw=1)
    ax2.axhline(0.0, color="grey", ls=":", lw=1)
    ax2.set_xlabel("number of surface samples N")
    ax2.set_ylabel("Phi / (q/eps0)")
    ax2.set_title("Centre case is exact at any N;\noff-centre & outside "
                  "converge as N grows")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # (3) 3-D sphere for the off-centre case, coloured by local E.dA
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    charges, q_enc, Phi, P, contrib = results["2  off-centre in"]
    sc = ax3.scatter(P[:, 0], P[:, 1], P[:, 2], c=contrib, cmap="coolwarm",
                     s=6, alpha=0.55,
                     vmin=-np.abs(contrib).max(), vmax=np.abs(contrib).max())
    cx, cy, cz = charges[0][1:]
    ax3.scatter([cx], [cy], [cz], color="k", s=140, marker="*",
                label="charge (inside)")
    ax3.legend(fontsize=8, loc="upper left")
    ax3.set_title("Config 2: local E.dA over the sphere\n"
                  "concentrated near the charge, sum still q/eps0")
    ax3.set_box_aspect((1, 1, 1))
    fig.colorbar(sc, ax=ax3, shrink=0.6, label="E . dA per patch")

    fig.tight_layout()
    fig.savefig("gauss_law.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
