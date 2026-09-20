"""Numerical Fourier transform for the ENGR 530 toolkit."""

import numpy as np


def fourier_transform(t, x, omega, chunk=100):
    """Approximate X(w) = integral x(t) exp(-j w t) dt on a grid of frequencies.

    t      : uniformly or nonuniformly spaced sample times covering the whole
             signal (x should be ~0 outside the window)
    x      : samples of x(t)
    omega  : array of angular frequencies (rad/s) at which to evaluate X
    The integral is the trapezoid rule, done directly (no FFT). Frequencies are
    processed in chunks to keep memory bounded.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x)
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    dt = np.diff(t)
    X = np.zeros(len(omega), dtype=complex)
    for start in range(0, len(omega), chunk):
        w = omega[start:start + chunk]
        integrand = x[None, :] * np.exp(-1j * np.outer(w, t))
        X[start:start + chunk] = np.sum(0.5 * (integrand[:, 1:] + integrand[:, :-1]) * dt, axis=1)
    return X


def dtft(x, n, omega):
    """Discrete-time Fourier transform, computed directly from the sum:
        X(e^{j w}) = sum_n x[n] exp(-j w n)

    x      : samples of the signal
    n      : the integer index of each sample (may start away from zero or be negative)
    omega  : array of frequencies in rad/sample (typically -pi .. pi; X is 2 pi periodic)
    """
    x = np.asarray(x)
    n = np.asarray(n, dtype=float)
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    return np.exp(-1j * np.outer(omega, n)) @ x
