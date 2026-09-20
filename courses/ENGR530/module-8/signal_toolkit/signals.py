"""Reusable signal generators for the ENGR 530 toolkit.

Continuous-time signals are sampled on t = 0, 1/fs, 2/fs, ... (the endpoint
t = duration is excluded, so a `duration` of D seconds gives D * fs samples).
"""

import numpy as np


def time_axis(duration, sampling_rate):
    """Sample times [0, 1/fs, ..., duration - 1/fs]."""
    n_samples = int(round(duration * sampling_rate))
    return np.arange(n_samples) / sampling_rate


def sinusoid(amplitude, frequency, phase, duration, sampling_rate):
    """x(t) = amplitude * cos(2*pi*frequency*t + phase).

    frequency in Hz, phase in radians. Returns (time_array, signal_array).
    """
    t = time_axis(duration, sampling_rate)
    return t, amplitude * np.cos(2 * np.pi * frequency * t + phase)


def exponential(amplitude, rate, duration, sampling_rate):
    """x(t) = amplitude * exp(rate * t). rate < 0 decays, rate > 0 grows (1/s)."""
    t = time_axis(duration, sampling_rate)
    return t, amplitude * np.exp(rate * t)


def unit_step(duration, sampling_rate, t0=0.0, amplitude=1.0):
    """x(t) = amplitude for t >= t0, else 0."""
    t = time_axis(duration, sampling_rate)
    return t, amplitude * (t >= t0).astype(float)


def discrete_impulse(n_samples, n0=0):
    """delta[n - n0] for n = 0, ..., n_samples - 1. Returns (n_array, signal_array)."""
    n = np.arange(n_samples)
    return n, (n == n0).astype(float)
