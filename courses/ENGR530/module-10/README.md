# Module 10 — Fourier series

A periodic signal is a list of harmonic coefficients `a_k`. This module builds
both directions (analysis and synthesis) and then uses them for
frequency-selective filtering.

The reusable pieces live in the toolkit, `../module-8/signal_toolkit/`:

| File | What it provides |
|---|---|
| `fourier.py` | `fourier_synthesis(coefficients, f0, t)` builds `x(t) = sum a_k exp(j k w0 t)` and returns the real part after checking `a_{-k} = conj(a_k)`. `fourier_coefficients(t, x, period, K)` estimates `a_k = (1/T0) integral x(t) exp(-j k w0 t) dt` by the trapezoid rule. Also closed-form coefficients for square, triangle, sawtooth and pulse-train waves. |
| `m10_fourier_synthesis.py` | Toolkit demo: a cosine from two complex coefficients, then square / triangle / sawtooth / pulse train. |

## Files in this folder

| File | What it does |
|---|---|
| `m10_square_wave.py` | Rebuilds a square wave from `N = 1, 3, 5, 10, 25, 50, 100` odd harmonics and tracks edge sharpness (10-90% rise time), error away from the jumps, and Gibbs overshoot. Figures: `square_wave_reconstructions.png` (one panel per N), `square_wave_edge_zoom.png`, `square_wave_metrics.png`, `square_wave_assembly.png` (harmonics k = 1, 3, 5, 7 beside their running sum). |
| `m10_fourier_analysis.py` | Numerical analysis then synthesis. Coefficients of a cosine, square, triangle, sawtooth and pulse train match the closed forms; reconstruction RMSE vs. K; integration accuracy vs. sample count; and a noisy IMU-style record whose two spectral peaks stand out from the noise floor. Figures: `fourier_analysis.png`, `fourier_analysis_noisy_record.png`. |
| `m10_imu_decomposition.py` | Decomposes `w(t) = 20 sin(2 pi 0.5 t) + 3 sin(2 pi 8 t)` over its 2 s period (`a_1 = -10j`, `a_16 = -1.5j`), then filters by scaling each coefficient with a 4th-order Butterworth low-pass `H` (cutoff 2 Hz) and re-synthesizing. Compares the causal filter (delays the motion 0.21 s) with the zero-phase version, and repeats with noise added. Figure: `imu_decomposition.png`. |
| `m10_sensor_bandwidth.py` | Given `|H| = 1.0, 0.98, 0.60, 0.10` at `1, 5, 20, 50 Hz`, computes `|b_k| = |H| |a_k|`, the dB gain and percent amplitude lost, and interpolates a rough -3 dB point (~14 Hz). Figure: `sensor_bandwidth.png`. |

## Running

```bash
cd probability/module-10
python m10_square_wave.py
python m10_fourier_analysis.py
python m10_imu_decomposition.py
python m10_sensor_bandwidth.py
```

## Key ideas

**Frequencies assemble into shape.** Each odd harmonic of a square wave is a
plain sine of amplitude `4/(pi k)`. The fundamental alone is a lump, and every
extra harmonic squares the shoulders and steepens the edge. Edge rise time and
error away from the jumps both fall roughly as `1/N`.

**Gibbs overshoot does not go away.** It stays at about 8.95% of the jump (a
peak near 1.18 for a +/-1 wave) for any N. More harmonics only squeeze it into
a narrower region at the edge. A finite sum of smooth sines never reproduces a
discontinuity exactly.

**Smoothness sets convergence, twice.** Coefficients of a smooth signal decay
fast (triangle `1/k^2`, square `1/k`), so few harmonics suffice. The numerical
integral converges the same way: a cosine is exact to rounding error, a
triangle's error falls like `1/N^2` in the sample count, a square's like `1/N`.

**Analysis and synthesis are inverses.** `fourier_coefficients` then
`fourier_synthesis` reproduces the signal; the leftover RMSE is truncation at K
harmonics, not the analysis.

**Filtering is multiplication.** Once a signal is a list of coefficients, a
linear time-invariant system just scales each one: `b_k = H(j k w0) a_k`. `|H|`
sets how much each frequency is kept and `angle(H)` delays it. A filter that
removes the 8 Hz vibration and keeps the 0.5 Hz motion (`|H| = 1` and `0.004`)
does so cleanly, but a causal filter also delays the motion (0.21 s here). Using
`|H|` alone gives the zero-phase, offline version, the same idea as the
noncausal filter in Module 9. The same filter removes most of the noise too,
because noise is spread over all frequencies and the filter keeps a narrow band.

**Frequency response is sensor capability.** With `|H|` of 1.00, 0.98, 0.60 and
0.10, the sensor reproduces 1 and 5 Hz faithfully (at most 2% amplitude lost),
under-reads 20 Hz by 40%, and reports 50 Hz at a tenth of its real size. Only
`|H|` was given, so timing distortion (phase) is not captured, and equal input
amplitudes were assumed.
