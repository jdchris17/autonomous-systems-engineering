"""
Module 12, Parts 1-2: reachability of the integral-augmented speed plant, and the LQI design.

Plant (nominal, sampled, no delay):  v_{k+1} = a v_k + b u_k + b d_k,  a = e^{-ch},  b = (1-a)/c.
Feedforward u_ff = c r holds v = r at equilibrium: (1 - a) r = b c r. With u = c r + du:
    e_{k+1}  = a e_k + b du_k + b d_k             (e = v - r)
    xi_{k+1} = xi_k - h e_k                       (xi = running integral of r - v, in metres)
    A_a = [[a, 0], [-h, 1]],  B_a = [b, 0]^T.

Part 1, reachability:
    A_a B_a = [a b, -h b]^T,   C = [B_a  A_a B_a] = [[b, a b], [0, -h b]],
    det C = b (-h b) - (a b)(0) = -h b^2  != 0   ->  rank 2 whenever b != 0 and h != 0.
    The command cannot touch xi directly (B_a has a zero there). But it moves the speed, the speed
    error is what gets integrated, so one step later xi changes by -h e. Two steps are enough to
    place both: the first command sets the speed error, the integral then accumulates it, and a
    second command can set the speed error independently. That chain (u -> e -> xi) is exactly the
    second column A_a B_a, and it is linearly independent of the first because the -h entry
    appears only after the speed has been moved.

Part 2, weights (with fixed scales 1 m/s, 1 m, 1 m/s^2):
    J = sum e^2 + 0.25 xi^2 + R du^2. A 1 m/s speed error costs 1 per sample. A 1 m accumulated error
    costs 0.25, so 2 m of accumulated error is "as bad" as 1 m/s of speed error. A 1 m/s^2 feedback
    command costs R: R = 0.25 makes command cheap (2 m/s^2 of push is worth the same as 1 m/s of
    error), R = 1 treats 1 m/s^2 and 1 m/s alike, R = 4 makes command expensive (1 m/s^2 is as bad
    as 2 m/s of error). Only the ratios matter: scaling Q and R together leaves K unchanged.

Gains: with K = [k_e, k_xi], u = c r - k_e (v - r) - k_xi xi = c r + k_e (r - v) - k_xi xi.
    Since xi_{k+1} = xi_k + h (r_k - v_k), the term I_k = -k_xi xi_k obeys I_{k+1} = I_k + (-k_xi) h (r - v):
    an LQI design IS a PI controller (plus feedforward) with Kp = k_e and Ki = -k_xi, updated in the
    same compute-then-integrate order. LQR just chooses the two gains from the weights.
"""

import numpy as np
from scipy.linalg import solve_discrete_are

C_DRAG, H_STEP = 0.2, 0.02
A_PLANT = np.exp(-C_DRAG * H_STEP)
B_PLANT = (1 - A_PLANT) / C_DRAG
A_aug = np.array([[A_PLANT, 0.0], [-H_STEP, 1.0]])
B_aug = np.array([[B_PLANT], [0.0]])
Q_cost = np.diag([1.0, 0.25])
R_COSTS = (0.25, 1.0, 4.0)
PI_GAINS = (1.5, 0.5)            # Kp, Ki of the comparison PI (Module 10 final gains)


def lqi_gain(R_cost, Q=Q_cost):
    R = np.atleast_2d(R_cost)
    X = solve_discrete_are(A_aug, B_aug, Q, R)
    K = np.linalg.solve(R + B_aug.T @ X @ B_aug, B_aug.T @ X @ A_aug)
    # residual of the Riccati equation as a check
    res = A_aug.T @ X @ A_aug - A_aug.T @ X @ B_aug @ K + Q - X
    return K.ravel(), X, np.max(np.abs(res))


def designs():
    return {f"LQI R={R:g}": lqi_gain(R)[0] for R in R_COSTS}


def fmt(z):
    return f"{z.real:.5f}{z.imag:+.5f}j"


def main():
    print("Part 1: reachability")
    Ctrb = np.hstack([B_aug, A_aug @ B_aug])
    print(f"  a = {A_PLANT:.6f}, b = {B_PLANT:.6f}")
    print(f"  C = [B_a, A_a B_a] = {np.array2string(Ctrb, precision=6)}")
    print(f"  rank {np.linalg.matrix_rank(Ctrb)},  det = {np.linalg.det(Ctrb):.4e},  -h b^2 = {-H_STEP * B_PLANT**2:.4e}\n")

    print("Part 2: LQI gains, Q_cost = diag(1, 0.25)")
    print(f"  {'R_cost':>6}  {'k_e':>8}{'k_xi':>9}   {'Kp=k_e':>7}{'Ki=-k_xi':>9}   closed-loop eigenvalues (max |mu|)"
          f"   s = ln(mu)/h   Riccati residual")
    for R in R_COSTS:
        K, X, res = lqi_gain(R)
        mu = np.linalg.eigvals(A_aug - B_aug @ K[None, :])
        s = np.log(mu.astype(complex)) / H_STEP
        print(f"  {R:6.2f}  {K[0]:8.4f}{K[1]:9.4f}   {K[0]:7.4f}{-K[1]:9.4f}   "
              f"{', '.join(fmt(m) for m in mu)} ({np.max(np.abs(mu)):.5f})   "
              f"{', '.join(f'{z.real:.3f}{z.imag:+.3f}j' for z in s)}   {res:.1e}")
    Kpi = np.array([PI_GAINS[0], -PI_GAINS[1]])
    mu = np.linalg.eigvals(A_aug - B_aug @ Kpi[None, :])
    s = np.log(mu.astype(complex)) / H_STEP
    print(f"  {'PI':>6}  {Kpi[0]:8.4f}{Kpi[1]:9.4f}   {PI_GAINS[0]:7.4f}{PI_GAINS[1]:9.4f}   "
          f"{', '.join(fmt(m) for m in mu)} ({np.max(np.abs(mu)):.5f})   "
          f"{', '.join(f'{z.real:.3f}{z.imag:+.3f}j' for z in s)}")
    print("  All max |mu| < 1: every design is stable. (PI written as K = [Kp, -Ki] for comparison.)")


if __name__ == "__main__":
    main()
