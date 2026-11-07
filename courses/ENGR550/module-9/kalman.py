"""
Module 5, Part 1: the Kalman filter predict / correct steps, with the derivation.

Model:  x_{k+1} = F x_k + G u_k + w_k,   z_k = H x_k + n_k
        w_k ~ (0, Q), n_k ~ (0, R), independent of each other, across time, and of x_0.
Notation: superscript - = prediction before z_k, + = corrected estimate after z_k.
Errors:   e^- = x_k - x_hat^-,   e^+ = x_k - x_hat^+,   P^- = E[e^- e^-T],   P^+ = E[e^+ e^+T].

Predict
    x_hat^-_k = F x_hat^+_{k-1} + G u_{k-1}
    P^-_k     = F P^+_{k-1} F^T + Q
    (the error obeys e^-_k = F e^+_{k-1} + w_{k-1}; the known input cancels.)

Innovation and its covariance
    nu_k = z_k - H x_hat^-_k = (H x_k + n_k) - H x_hat^-_k = H e^-_k + n_k
    E[nu] = 0 when the prediction is unbiased.
    S_k = E[nu nu^T] = H E[e^- e^-T] H^T + H E[e^- n^T] + E[n e^-T] H^T + E[n n^T]
        = H P^-_k H^T + R,
    because n_k is independent of e^-_k (e^- depends only on x_0, past w and past n).

Corrected covariance for ANY gain K  (x_hat^+ = x_hat^- + K nu)
    e^+ = x - x_hat^- - K (H e^- + n) = (I - K H) e^- - K n
    P^+ = (I - K H) P^- (I - K H)^T + K R K^T                    <- Joseph form
    (the cross terms vanish for the same independence reason.)

Gain that minimizes the total error variance, trace(P^+)
    Expanding:  tr P^+ = tr P^- - 2 tr(K H P^-) + tr(K S K^T)
    d/dK:       -2 P^- H^T + 2 K S = 0   ->   K = P^- H^T S^{-1}
    The second derivative is 2 S > 0, so this is a minimum. Substituting it back gives
    the short form P^+ = (I - K H) P^-, which equals the Joseph form only at this K.
    The Joseph form is used in code because it stays symmetric and positive semidefinite
    under roundoff and is correct for any K.

One position sensor, H = [1 0]:
    P^- H^T = [P_pp, P_vp]^T,   S = P_pp + R,   K = [P_pp, P_vp]^T / (P_pp + R)
    K_p is dimensionless (m per m of innovation); K_v has units 1/s (m/s per m).
    K_v is nonzero only through P_vp: the position reading corrects velocity because the
    prediction's position and velocity errors are correlated (Module 4, Part 2).

Running this file checks the derivation numerically.
"""

import numpy as np


def predict(x, P, F, G, u, Q):
    """Propagate the estimate and its covariance one step."""
    return F @ x + G * u, F @ P @ F.T + Q


def correct(x_pred, P_pred, z, H, R):
    """Measurement update with the Joseph-form covariance.

    x_pred (n,), P_pred (n, n), z (m,), H (m, n), R (m, m).
    Returns x_corr, P_corr, innovation nu (m,), its covariance S (m, m), and gain K (n, m).
    """
    H = np.atleast_2d(H)
    R = np.atleast_2d(R)
    z = np.atleast_1d(z)
    nu = z - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = np.linalg.solve(S, H @ P_pred).T          # = P^- H^T S^{-1}, without forming S^{-1}
    x_corr = x_pred + K @ nu
    I_KH = np.eye(len(x_pred)) - K @ H
    P_corr = I_KH @ P_pred @ I_KH.T + K @ R @ K.T
    return x_corr, P_corr, nu, S, K


def joseph(P_pred, K, H, R):
    """Corrected covariance for an arbitrary gain."""
    I_KH = np.eye(P_pred.shape[0]) - K @ H
    return I_KH @ P_pred @ I_KH.T + K @ R @ K.T


def main():
    rng = np.random.default_rng(0)
    H = np.array([[1.0, 0.0]])
    R = np.array([[0.25]])
    L = rng.normal(size=(2, 2))
    P_pred = L @ L.T + 0.1 * np.eye(2)            # an arbitrary valid prior covariance
    x_pred = np.zeros(2)

    _, P_corr, _, S, K = correct(x_pred, P_pred, np.array([0.0]), H, R)
    print(f"P^- =\n{np.round(P_pred, 4)}\nR = {R[0, 0]}")
    print(f"K = {np.round(K.ravel(), 4)},  closed form [P_pp, P_vp]/(P_pp + R) = "
          f"{np.round(P_pred[:, 0] / (P_pred[0, 0] + R[0, 0]), 4)}")

    # Monte Carlo check of S and of the Joseph form for an arbitrary (suboptimal) gain.
    n = 400_000
    e_pred = rng.multivariate_normal(np.zeros(2), P_pred, size=n)
    noise = rng.normal(0, np.sqrt(R[0, 0]), size=(n, 1))
    nu = e_pred @ H.T + noise
    K_any = np.array([[0.3], [-0.7]])
    e_corr = e_pred - (nu @ K_any.T)
    print(f"\nS: formula {S[0, 0]:.4f}, Monte Carlo {nu.var():.4f}")
    print(f"Joseph form with arbitrary K = {K_any.ravel()}:\n  formula     {np.round(joseph(P_pred, K_any, H, R).ravel(), 4)}"
          f"\n  Monte Carlo {np.round(np.cov(e_corr, rowvar=False).ravel(), 4)}")

    # Trace is minimized at the Kalman gain: every perturbation increases it.
    base = np.trace(P_corr)
    increases = [np.trace(joseph(P_pred, K + 1e-3 * d.reshape(2, 1), H, R)) - base
                 for d in rng.normal(size=(1000, 2))]
    print(f"\ntrace(P^+) at Kalman gain = {base:.6f}; smallest change over 1000 random gain "
          f"perturbations = {min(increases):+.2e} (all positive: {min(increases) > 0})")
    short = (np.eye(2) - K @ H) @ P_pred
    print(f"Joseph form = (I - KH)P^- at the Kalman gain: {np.allclose(P_corr, short)}; "
          f"at the arbitrary gain: {np.allclose(joseph(P_pred, K_any, H, R), (np.eye(2) - K_any @ H) @ P_pred)}")


if __name__ == "__main__":
    main()
