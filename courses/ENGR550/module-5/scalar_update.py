"""
Module 5, Part 2: one scalar update by hand.

    p_hat^- = 10 m, P^- = 4 m^2, z = 12 m, R = 1 m^2

    nu    = z - p_hat^-        = 12 - 10       = 2 m
    S     = P^- + R            = 4 + 1         = 5 m^2
    K     = P^- / S            = 4 / 5         = 0.8
    p^+   = p_hat^- + K nu     = 10 + 0.8 * 2  = 11.6 m
    P^+   = (1 - K) P^-        = 0.2 * 4       = 0.8 m^2
          (Joseph: (1 - K)^2 P^- + K^2 R = 0.16 + 0.64 = 0.8 m^2)

Relation to Module 2 inverse-variance weighting:
    p^+ = (1 - K) p_hat^- + K z, with K = P^- / (P^- + R) = (1/R) / (1/P^- + 1/R).
    That is exactly the Module 2 fused estimate with the prediction acting as one "sensor"
    (variance 4) and the measurement as the other (variance 1): weights 0.2 and 0.8, and
    1/P^+ = 1/P^- + 1/R = 1.25, so P^+ = 0.8 m^2. It is even the same numbers as the
    Module 2 hand check (10.0 and 12.0 -> 10.4 there with the weights swapped, because
    the 10 m reading was the precise one).
    What the Kalman filter adds is where the second "sensor" comes from: the model
    carries the previous estimate forward, and P^- = F P^+ F^T + Q says how much to
    trust it now.
"""

import numpy as np

from kalman import correct

P_HAT_PRED, P_PRED, Z, R = 10.0, 4.0, 12.0, 1.0
EXPECTED = {"nu": 2.0, "S": 5.0, "K": 0.8, "p_corr": 11.6, "P_corr": 0.8}


def hand_check(verbose=True):
    """Run the scalar case through the general filter code; return True if all values match."""
    x, P, nu, S, K = correct(np.array([P_HAT_PRED]), np.array([[P_PRED]]),
                             np.array([Z]), np.array([[1.0]]), np.array([[R]]))
    got = {"nu": nu[0], "S": S[0, 0], "K": K[0, 0], "p_corr": x[0], "P_corr": P[0, 0]}
    ok = all(np.isclose(got[key], EXPECTED[key]) for key in EXPECTED)
    if verbose:
        for key, val in got.items():
            print(f"  {key:<7} = {val:7.4f}   expected {EXPECTED[key]}")
    return ok


def main():
    print(f"Scalar update: p^- = {P_HAT_PRED} m, P^- = {P_PRED} m^2, z = {Z} m, R = {R} m^2")
    ok = hand_check()
    print("Matches the hand calculation." if ok else "MISMATCH.")

    w_meas = (1 / R) / (1 / P_PRED + 1 / R)
    fused = (1 - w_meas) * P_HAT_PRED + w_meas * Z
    print(f"\nModule 2 inverse-variance fusion of prediction and measurement:")
    print(f"  weight on measurement = {w_meas:.2f} (= K),  fused = {fused:.2f} m,  "
          f"variance = 1/(1/{P_PRED} + 1/{R}) = {1 / (1 / P_PRED + 1 / R):.2f} m^2")


if __name__ == "__main__":
    main()
