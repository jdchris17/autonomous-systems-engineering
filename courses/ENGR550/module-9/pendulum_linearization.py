"""
Module 9, Part 4: local linearization of a damped pendulum.

    theta' = omega,   omega' = -(g/L) sin(theta) - c omega,   g = 9.81, L = 1 m, c = 0.5 1/s.

Equilibria: theta' = 0 -> omega = 0;  omega' = 0 -> sin(theta) = 0 -> theta = n pi.
Physically two: hanging down (0, 0) and upright (pi, 0).

Jacobian  A(theta*) = [[0, 1], [-(g/L) cos(theta*), -c]],  char. eq.  lambda^2 + c lambda + (g/L) cos(theta*) = 0.

Down, cos = +1:  lambda^2 + 0.5 lambda + 9.81 = 0
    lambda = -0.25 +- j sqrt(9.81 - 0.0625) = -0.25 +- 3.122j
    -> stable focus: oscillation at ~3.12 rad/s (period 2.0 s) decaying as e^{-0.25 t}.
Up, cos = -1:    lambda^2 + 0.5 lambda - 9.81 = 0
    lambda = -0.25 +- sqrt(0.0625 + 9.81) = +2.892, -3.392
    -> saddle, unstable: one direction falls away as e^{2.89 t} (a small tilt grows x18 in 1 s);
       the other direction (a precise swing toward vertical) decays.
The product of eigenvalues equals (g/L) cos(theta*): positive (same sign, focus/node) at the bottom,
negative (opposite signs, saddle) at the top, whatever the damping.
"""

import numpy as np
from scipy.linalg import expm

G_ACC, L, C_DAMP = 9.81, 1.0, 0.5
H_STEP = 0.02


def jacobian(theta_star):
    return np.array([[0.0, 1.0], [-(G_ACC / L) * np.cos(theta_star), -C_DAMP]])


def f(x):
    return np.array([x[1], -(G_ACC / L) * np.sin(x[0]) - C_DAMP * x[1]])


def numeric_jacobian(x, eps=1e-6):
    J = np.zeros((2, 2))
    for i in range(2):
        dx = np.zeros(2)
        dx[i] = eps
        J[:, i] = (f(x + dx) - f(x - dx)) / (2 * eps)
    return J


def classify(lams):
    if np.all(lams.real < 0):
        return "stable focus" if np.any(np.abs(lams.imag) > 0) else "stable node"
    if np.all(np.isreal(lams)) and lams.real.min() < 0 < lams.real.max():
        return "saddle (unstable)"
    return "unstable"


def main():
    print("Equilibria: omega = 0 and sin(theta) = 0 -> theta = 0 (down), pi (up)")
    print(f"  {'equilibrium':<16}{'lambda_1':>18}{'lambda_2':>18}   {'|mu| = |e^(lambda h)|, h=0.02':<30} classification")
    for name, th in (("down (0, 0)", 0.0), ("up (pi, 0)", np.pi)):
        A = jacobian(th)
        assert np.allclose(A, numeric_jacobian(np.array([th, 0.0])), atol=1e-6)
        assert np.allclose(f(np.array([th, 0.0])), 0.0, atol=1e-12)
        lams = np.sort_complex(np.linalg.eigvals(A))
        mus = np.abs(np.linalg.eigvals(expm(A * H_STEP)))
        lam_s = [f"{z.real:+.3f}{z.imag:+.3f}j" if abs(z.imag) > 1e-12 else f"{z.real:+.3f}" for z in lams]
        print(f"  {name:<16}{lam_s[0]:>18}{lam_s[1]:>18}   {', '.join(f'{m:.4f}' for m in sorted(mus)):<30} "
              f"{classify(lams)}")
    print("  (analytic Jacobian matches a central-difference Jacobian at both points)")


if __name__ == "__main__":
    main()
