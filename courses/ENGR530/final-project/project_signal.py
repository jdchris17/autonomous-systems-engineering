"""Final-project signal model: known truth plus known contamination.

    x_true(t)   = A1 sin(2 pi f1 t) + A2 sin(2 pi f2 t + phi)     desired motion
    z_analog(t) = x_true + v(t) + h(t) + b(t) + n(t)              what the sensor produces before the ADC

    v(t) : in-band vibration      A_v sin(2 pi f_v t)       (15 Hz, below any sensible Nyquist)
    h(t) : high-frequency interference A_h sin(2 pi f_h t)  (80 Hz, above a 100 Hz ADC's Nyquist)
    b(t) : bias + slow drift      b0 + B sin(2 pi f_b t)
    n(t) : white Gaussian noise, independent sample by sample at the analog rate

"z_analog" is simulated on a very fine grid (FS_ANALOG) to stand in for continuous time.
The signal generator is the toolkit's sinusoid() from module-8/signal_toolkit.
"""

import os
import sys

import numpy as np

_TOOLKIT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "module-8", "signal_toolkit")
if _TOOLKIT not in sys.path:
    sys.path.insert(0, _TOOLKIT)
from signals import sinusoid  # noqa: E402

# Given in the assignment
A1, F1 = 10.0, 0.5          # primary platform motion
A2, F2 = 4.0, 2.0           # smaller maneuvering component
PHI = np.pi / 3             # phase of the maneuvering component
F_V = 15.0                  # vibration frequency
# Chosen here (not given): change these to explore
A_V = 3.0                   # vibration amplitude
A_H, F_H = 2.0, 80.0        # high-frequency interference (aliases to 20 Hz at a 100 Hz ADC)
B0 = 1.0                    # constant bias
B_DRIFT, F_B = 3.0, 0.05    # slow drift: amplitude and frequency
SIGMA_N = 1.0               # noise standard deviation

FS_ANALOG = 2000            # Hz: the "continuous-time" simulation grid
DURATION = 20.0             # s: a whole number of periods of every tone above (0.05 Hz -> 1 period)


def sine(amplitude, frequency, phase=0.0, duration=DURATION, fs=FS_ANALOG):
    """amplitude * sin(2 pi f t + phase), built with the toolkit's cosine generator."""
    return sinusoid(amplitude, frequency, phase - np.pi / 2, duration, fs)


def build_analog(seed=0, drift=True, interference=True):
    """Return every component of the analog measurement, all on the FS_ANALOG grid."""
    t, m1 = sine(A1, F1)
    _, m2 = sine(A2, F2, PHI)
    x_true = m1 + m2
    _, vib = sine(A_V, F_V)
    _, hf = sine(A_H, F_H)
    hf = hf if interference else 0.0 * hf
    _, slow = sine(B_DRIFT, F_B)
    bias = B0 + (slow if drift else 0.0 * slow)
    noise = np.random.default_rng(seed).normal(0.0, SIGMA_N, size=len(t))
    z = x_true + vib + hf + bias + noise
    return {"t": t, "x_true": x_true, "vib": vib, "hf": hf, "bias": bias, "noise": noise, "z_analog": z}


def amplitude_spectrum(x, fs):
    """One-sided amplitude spectrum: a sinusoid of amplitude A shows up as a peak of height A."""
    N = len(x)
    amp = 2 * np.abs(np.fft.rfft(x)) / N
    amp[0] /= 2
    return np.fft.rfftfreq(N, 1 / fs), amp


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
