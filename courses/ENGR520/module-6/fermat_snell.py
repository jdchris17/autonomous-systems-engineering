"""module-6 -- fermat_snell.py  (Computational Exercise: Fermat Finds Snell)

Point A sits in medium 1 (refractive index n1), point B in medium 2 (n2),
separated by the flat interface y = 0. A ray leaves A, crosses the interface
at (x, 0), and continues to B. The travel time is

    T(x) = n1 L1(x) / c  +  n2 L2(x) / c

    L1(x) = |A - (x, 0)|      L2(x) = |(x, 0) - B|

We plot T(x), find its minimum numerically, read off the incidence and
refraction angles there, and check

    n1 sin(theta1)  ==  n2 sin(theta2).

Nothing in the code knows Snell's law -- it only knows travel time. Snell's
law is what "fastest path" turns into.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

C = 299_792_458.0        # m/s
N1, N2 = 1.00, 1.50      # e.g. air -> glass
A = np.array([0.0, 1.0])     # medium 1  (y > 0)
B = np.array([2.5, -1.2])    # medium 2  (y < 0)


def L1(x):
    return np.hypot(x - A[0], A[1])          # A[1] = height above interface


def L2(x):
    return np.hypot(B[0] - x, B[1])          # B[1] < 0


def travel_time(x):
    return (N1 * L1(x) + N2 * L2(x)) / C


def angles(x):
    """Incidence / refraction angles from the normal (the y-axis), in radians."""
    theta1 = np.arctan2(abs(x - A[0]), abs(A[1]))
    theta2 = np.arctan2(abs(B[0] - x), abs(B[1]))
    return theta1, theta2


def main():
    # coarse search first (this is "the method"), then refine
    xs = np.linspace(-1.0, 3.5, 4001)
    T = travel_time(xs)
    x_coarse = xs[np.argmin(T)]
    res = minimize_scalar(travel_time, bracket=(x_coarse - 0.2, x_coarse + 0.2))
    x_min = res.x

    th1, th2 = angles(x_min)
    s1, s2 = N1 * np.sin(th1), N2 * np.sin(th2)

    # reference paths
    x_straight = A[0] + (B[0] - A[0]) * A[1] / (A[1] - B[1])   # geometric straight line
    x_equal = 0.5 * (A[0] + B[0])                              # crude "equal angle" guess

    print("=" * 74)
    print("FERMAT FINDS SNELL")
    print("=" * 74)
    print(f"n1 = {N1}, n2 = {N2}   A = ({A[0]:g}, {A[1]:g}) [medium 1],  "
          f"B = ({B[0]:g}, {B[1]:g}) [medium 2]")
    print(f"interface at y = 0,  ray crosses at x")
    print()
    print(f"minimum-time crossing:  x* = {x_min:.6f} m")
    print(f"   T(x*)          = {travel_time(x_min)*1e9:.6f} ns")
    print(f"   T(straight)    = {travel_time(x_straight)*1e9:.6f} ns   "
          f"(x = {x_straight:.4f})  -- slower")
    print(f"   dT/dx at x*    = {np.gradient(T, xs)[np.argmin(np.abs(xs - x_min))]:.2e}"
          f"  (~0: it is a stationary point)")
    print()
    print(f"angles at x*:  theta1 = {np.degrees(th1):.4f} deg,  "
          f"theta2 = {np.degrees(th2):.4f} deg")
    print(f"   n1 sin(theta1) = {s1:.8f}")
    print(f"   n2 sin(theta2) = {s2:.8f}")
    print(f"   relative mismatch = {abs(s1 - s2) / s1:.2e}   ->  Snell's law holds")
    print()

    # sweep: does it hold for other geometries / indices?
    print("Same check for a range of setups (B_x moved, n2 changed):")
    print(f"   {'B_x':>6} {'n2':>5} {'x*':>9} {'th1(deg)':>9} {'th2(deg)':>9} "
          f"{'n1 sin1':>9} {'n2 sin2':>9} {'mismatch':>10}")
    for bx, n2 in [(1.0, 1.33), (2.5, 1.50), (4.0, 1.50), (2.5, 2.42), (-1.0, 1.5)]:
        _sweep_row(bx, n2)
    print()
    print("The optimiser was handed T(x) and nothing else. n1 sin(theta1) =")
    print("n2 sin(theta2) drops out of dT/dx = 0 every time -- that derivative")
    print("is literally  (n1 sin theta1 - n2 sin theta2) / c.")
    print("=" * 74)

    _plot(xs, T, x_min, x_straight, th1, th2)


def _sweep_row(bx, n2):
    b = np.array([bx, B[1]])

    def tt(x):
        return (N1 * np.hypot(x - A[0], A[1]) + n2 * np.hypot(bx - x, b[1])) / C

    xg = np.linspace(min(A[0], bx) - 2, max(A[0], bx) + 2, 4001)
    x0 = xg[np.argmin(tt(xg))]
    xm = minimize_scalar(tt, bracket=(x0 - 0.2, x0 + 0.2)).x
    t1 = np.arctan2(abs(xm - A[0]), abs(A[1]))
    t2 = np.arctan2(abs(bx - xm), abs(b[1]))
    s1, s2 = N1 * np.sin(t1), n2 * np.sin(t2)
    print(f"   {bx:>6.1f} {n2:>5.2f} {xm:>9.4f} {np.degrees(t1):>9.3f} "
          f"{np.degrees(t2):>9.3f} {s1:>9.5f} {s2:>9.5f} {abs(s1-s2)/s1:>10.1e}")


def _plot(xs, T, x_min, x_straight, th1, th2):
    fig, (axT, axG) = plt.subplots(1, 2, figsize=(14, 6))

    # T(x)
    axT.plot(xs, T * 1e9, color="tab:blue")
    axT.plot(x_min, travel_time(x_min) * 1e9, "r*", ms=15,
             label=f"minimum  x* = {x_min:.3f}")
    axT.axvline(x_straight, color="grey", ls="--", lw=1,
                label=f"straight-line crossing x = {x_straight:.3f}")
    axT.plot(x_straight, travel_time(x_straight) * 1e9, "ko", ms=5)
    axT.set_xlabel("interface crossing point  x (m)")
    axT.set_ylabel("travel time  T(x)  (ns)")
    axT.set_title("T(x): one smooth minimum\n"
                  "the geometric straight line is NOT the fastest path")
    axT.legend(); axT.grid(True, alpha=0.3)

    # inset: n1 sin th1(x) and n2 sin th2(x) crossing at x_min
    axins = axT.inset_axes([0.55, 0.15, 0.4, 0.4])
    t1 = np.arctan2(np.abs(xs - A[0]), abs(A[1]))
    t2 = np.arctan2(np.abs(B[0] - xs), abs(B[1]))
    axins.plot(xs, N1 * np.sin(t1), label="n1 sin th1")
    axins.plot(xs, N2 * np.sin(t2), label="n2 sin th2")
    axins.axvline(x_min, color="r", lw=1)
    axins.set_title("they cross exactly at x*", fontsize=8)
    axins.tick_params(labelsize=7)
    axins.legend(fontsize=7)

    # geometry
    axG.axhline(0, color="k", lw=1)
    axG.fill_between([-1.5, 4.5], 0, 2.5, color="tab:cyan", alpha=0.12)
    axG.fill_between([-1.5, 4.5], -2.0, 0, color="tab:orange", alpha=0.12)
    axG.text(-1.3, 1.6, f"medium 1\n n1 = {N1}", fontsize=9)
    axG.text(-1.3, -1.4, f"medium 2\n n2 = {N2}", fontsize=9)

    P = np.array([x_min, 0.0])
    axG.plot(*A, "ko"); axG.annotate(" A", A)
    axG.plot(*B, "ko"); axG.annotate(" B", B)
    axG.plot([A[0], P[0]], [A[1], P[1]], "b-", lw=2)
    axG.plot([P[0], B[0]], [P[1], B[1]], "b-", lw=2, label="least-time path")
    axG.plot([A[0], B[0]], [A[1], B[1]], "k--", lw=1, label="straight line")
    axG.plot([x_min, x_min], [-0.9, 0.9], "g:", lw=1)      # normal
    axG.annotate("normal", (x_min + 0.05, 0.85), fontsize=8, color="g")

    # angle arcs: theta1 opens toward A (upper left), theta2 toward B (lower right)
    a1 = np.linspace(0, th1, 30)
    axG.plot(x_min - 0.4 * np.sin(a1), 0.4 * np.cos(a1), "g-", lw=1.2)
    axG.annotate(f"theta1 = {np.degrees(th1):.1f} deg", (x_min - 0.95, 0.55),
                 fontsize=8, color="g")
    a2 = np.linspace(0, th2, 30)
    axG.plot(x_min + 0.4 * np.sin(a2), -0.4 * np.cos(a2), "g-", lw=1.2)
    axG.annotate(f"theta2 = {np.degrees(th2):.1f} deg", (x_min + 0.1, -0.62),
                 fontsize=8, color="g")

    axG.set_xlim(-1.5, 4.5); axG.set_ylim(-2.0, 2.5)
    axG.set_aspect("equal")
    axG.set_xlabel("x (m)"); axG.set_ylabel("y (m)")
    axG.set_title("The path the minimum picks -- bent toward the normal in the\n"
                  "slower (higher-n) medium, exactly as Snell's law requires")
    axG.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig("fermat_snell.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
