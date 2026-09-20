"""Fourier series synthesis for the ENGR 530 toolkit.

Coefficients are passed as a dict {k: a_k} (k = -K..K), meaning
    x(t) = sum_k a_k * exp(j * k * w0 * t),   w0 = 2*pi*f0.
"""

import numpy as np


def fourier_synthesis(coefficients, fundamental_freq, t, real=True):
    """Build x(t) = sum_{k=-K}^{K} a_k exp(j k w0 t) on the time array `t`.

    fundamental_freq in Hz. With real=True (the default, for real signals) the
    coefficients must be conjugate-symmetric (a_{-k} = conj(a_k)); the
    imaginary part then cancels to rounding error and the real part is
    returned. A ValueError is raised if it does not cancel.
    """
    t = np.asarray(t, dtype=float)
    w0 = 2 * np.pi * fundamental_freq
    x = np.zeros(t.shape, dtype=complex)
    for k, a_k in coefficients.items():
        x += a_k * np.exp(1j * k * w0 * t)
    if not real:
        return x
    if np.max(np.abs(x.imag)) > 1e-9 * (1 + np.max(np.abs(x.real))):
        raise ValueError("coefficients are not conjugate-symmetric, so x(t) is not real")
    return x.real


def fourier_coefficients(t, x, period, K):
    """Numerical Fourier analysis: a_k = (1/T0) * integral_{T0} x(t) exp(-j k w0 t) dt.

    `t` must span exactly one period (t[-1] - t[0] == period, endpoints
    included) and `x` holds the samples. The integral is done with the
    trapezoid rule. Returns {k: a_k} for k = -K..K, in the same form
    fourier_synthesis() accepts, so analysis and synthesis round-trip.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x)
    if abs((t[-1] - t[0]) - period) > 1e-6 * period:
        raise ValueError("t must span exactly one period")
    w0 = 2 * np.pi / period
    dt = np.diff(t)
    coefficients = {}
    for k in range(-K, K + 1):
        integrand = x * np.exp(-1j * k * w0 * t)
        coefficients[k] = np.sum(0.5 * (integrand[1:] + integrand[:-1]) * dt) / period
    return coefficients


def _harmonics(K):
    return range(-K, K + 1)


def square_wave_coeffs(K):
    """+/-1 square wave, odd about t = 0: a_k = 2/(j pi k) for odd k, else 0."""
    return {k: (2 / (1j * np.pi * k) if k % 2 else 0.0) for k in _harmonics(K)}


def triangle_wave_coeffs(K):
    """Triangle from -1 to +1, peak at t = 0: a_k = 4/(pi^2 k^2) for odd k, else 0."""
    return {k: (4 / (np.pi**2 * k**2) if k % 2 else 0.0) for k in _harmonics(K)}


def sawtooth_wave_coeffs(K):
    """Ramp from -1 to +1 over one period, zero at t = 0: a_k = (-1)^(k+1) / (j pi k)."""
    return {k: (0.0 if k == 0 else (-1) ** (k + 1) / (1j * np.pi * k)) for k in _harmonics(K)}


def pulse_train_coeffs(K, duty=0.25):
    """0/1 pulse train, centered on t = 0, fraction `duty` of the period high.

    a_0 = duty and a_k = sin(pi k duty) / (pi k).
    """
    return {k: (duty if k == 0 else np.sin(np.pi * k * duty) / (np.pi * k)) for k in _harmonics(K)}
