# Module 11 — Fourier transforms and frequency response

From the Fourier series (periodic signals, discrete `a_k`) to the transform of
finite signals, the frequency response of LTI systems, and practical spectral
analysis. Ends by resolving the IMU problem from Module 8.

Reusable pieces live in the toolkit, `../module-8/signal_toolkit/`:

| File | What it provides |
|---|---|
| `transforms.py` | `fourier_transform(t, x, omega)`: `X(w) = integral x(t) exp(-jwt) dt` by direct trapezoid integration on a frequency array (no FFT). `dtft(x, n, omega)`: `X(e^jw) = sum x[n] exp(-jwn)`, with any integer index array `n`. |
| `m11_numerical_transform.py` | Toolkit demo: a rectangular pulse's numerical transform vs. the exact sinc (`|X|` and phase plotted separately, error ~2e-7), plus width and delay effects. |

## Files in this folder

| File | What it does |
|---|---|
| `m11_time_frequency_scaling.py` | Pulses of width `T = 0.1, 0.5, 1, 5 s`. Measures each spectrum's main lobe: the first null is at `2 pi / T`, so null-to-null width is `4 pi / T` (width x T = 12.566 in every row). All four collapse onto one sinc when plotted against `w T / 2 pi`. |
| `m11_frequency_response.py` | Numerically transforms `h(t) = (1/tau) e^{-t/tau} u(t)` for `tau = 0.01, 0.1, 1 s` and matches `1/(1 + j w tau)`. -3 dB at `w = 1/tau`, -45 degrees there, -20 dB/decade after; bandwidth x tau = 1. |
| `m11_dtft.py` | DTFT of `d[n]` (flat, phase 0), `d[n-5]` (flat, phase slope -5), and the 5-point moving average (Dirichlet-kernel magnitude, 2-sample delay, nulls at `2 pi k / 5`). Its gains reproduce the Module 8 measurements (0.984 at 2 Hz, 0.647 at 10 Hz at 100 Hz sampling), and mean `|X|^2 = 1/5`. |
| `m11_spectral_leakage.py` | FFT of a sinusoid with 10 whole periods (one clean bin) vs. 10.5 periods (peak -3.7 dB, energy in every bin), then rectangular / Hann / Hamming windows: first sidelobes -13.3 / -31.5 / -42.6 dB, main lobes 2 / 4 / 4 bins. |
| `m11_imu_spectrum.py` | The Module 8 IMU signal in frequency: a 20-amplitude line at 0.5 Hz, a 3-amplitude line at 8 Hz, and a flat noise floor (~0.08). Power splits 200 / 4.2 / 4.0, matching the time-domain mean square (Parseval). |
| `m11_remove_vibration.py` | Idealized `Y(f) = H(f) X(f)`: an ideal 7-9 Hz notch removes the 8 Hz line and leaves 0.5 Hz untouched. Shows raw signal, raw spectrum, filtered spectrum and filtered signal, and compares with a 3 Hz low-pass (RMSE vs. true motion: raw 2.86, notch 1.98, low-pass 0.37). |
| `m11_sensor_dynamics_ode.py` | `tau y' + y = x` with `tau = 0.1 s`: derives `H(w) = 1/(1 + j w tau)`, then integrates the ODE with RK4 for 0.1, 1, 10, 100 Hz inputs. Simulated steady-state amplitudes (0.998, 0.847, 0.157, 0.016) and phases match the prediction to better than 1e-6. |

Plots are saved as PNGs next to each script.

## Running

```bash
cd probability/module-11
python m11_time_frequency_scaling.py
python m11_frequency_response.py
python m11_dtft.py
python m11_spectral_leakage.py
python m11_imu_spectrum.py
python m11_remove_vibration.py     # imports make_signal from m11_imu_spectrum.py
python m11_sensor_dynamics_ode.py
```

## Key ideas

**Time-frequency trade-off.** Stretching a signal in time shrinks its spectrum by
the same factor: `duration x bandwidth` is fixed (about `4 pi` for a pulse). The
first-order system shows the same law from the system side: `tau` is both the
time constant and the inverse bandwidth, so a slower sensor is a narrower-band
sensor.

**Frequency response is the impulse response, transformed.** `H(w)` scales each
input frequency by `|H|` and delays it by `angle H`. Its magnitude and phase
explain what earlier modules only measured: why the moving average smooths, why
a sensor under-reads fast motion, and where the delay comes from.

**The DTFT lives on a circle.** `X(e^jw)` is `2 pi` periodic, so `w = pi` is the
highest frequency a sampled signal can have. For the moving average, `|X|` is 1
at DC and falls to nulls at `2 pi / M`: a low-pass filter, and the average
`|X|^2 = 1/M` is why white noise loses `1 - 1/M` of its power.

**Finite observation causes leakage.** We only ever see a windowed piece of a
signal: time multiplication by a window, which is frequency convolution with the
window's spectrum. The FFT samples that spectrum at its bins. A whole number of
periods lands every bin on a zero of the sinc (clean by luck); otherwise energy
smears across all bins. Tapering windows trade sidelobe level against
main-lobe width, so no window is free.

**Decompose, select, rebuild.** In the time domain the IMU's motion, vibration
and noise were hard to separate by eye. In the spectrum they are three distinct
things: two lines and a flat floor. Filtering is then multiplication,
`Y(f) = H(f) X(f)`, followed by an inverse transform. A notch removes the
vibration but almost no noise; a low-pass removes both, at the cost of any
genuine content above its cutoff. Ideal brick-wall filters are an offline
idealization; the moving averages and first-order sensors of Modules 8-9 are the
practical versions.

**Physical model and frequency analysis are one object.** Substituting a complex
sinusoid into `tau y' + y = x` gives `H = 1/(1 + j w tau)`, and integrating the
ODE in time reproduces that response. The differential equation and the
frequency response are two views of the same sensor.
