# Final project: an IMU-style measurement, from physics to a distribution of performance

One signal carried through a complete acquisition and estimation chain, scored against a
known truth at every step, and finally repeated many times so the performance becomes a
probability distribution.

```
physical signal --> analog anti-alias filter --> sampling --> analysis --> digital filter
   --> estimate --> error --> Monte Carlo --> (correlated-noise case)
```

## The signal (`project_signal.py`)

| Component | Definition | Value |
|---|---|---|
| desired motion `x_true` | `A1 sin(2 pi f1 t) + A2 sin(2 pi f2 t + phi)` | A1 = 10, f1 = 0.5 Hz, A2 = 4, f2 = 2 Hz, phi = pi/3 |
| vibration | `A_v sin(2 pi f_v t)` | 3 at 15 Hz |
| HF interference | `A_h sin(2 pi f_h t)` | 2 at 80 Hz (aliases to 20 Hz at a 100 Hz ADC) |
| bias and drift | `b0 + B sin(2 pi f_b t)` | b0 = 1, B = 3 at 0.05 Hz |
| noise | white Gaussian, independent per sample | sigma = 1 on a 2000 Hz analog grid |

The analog measurement `z_analog(t) = x_true + vibration + interference + bias/drift + noise` is
simulated on a 2000 Hz grid over 20 s (every tone fits a whole number of periods). Parameters
not given in the assignment (phi, A_v, the interference, drift, sigma) are constants at the top of
`project_signal.py`. Set `A_H = 0` there to remove the 80 Hz line.

The signal is built with `sinusoid()` from `../module-8/signal_toolkit/signals.py`, and Stage 4
reuses the analog Butterworth response from `../module-12/m12_sampling_rate_selection.py`. Keep
those folders where they are.

## Files

| File | Stage | What it does |
|---|---|---|
| `project_signal.py` | | the signal model above; imported by every stage, not run directly |
| `stage3_spectrum.py` | 3 | spectra of `x_true` and `z_analog` before sampling: two lines vs. six lines plus a flat noise floor; the physical bandwidth drives the sampling choice |
| `stage4_antialias.py` | 4 | fs = 100 Hz (Nyquist 50 Hz), 4th-order Butterworth anti-alias filter with a 20 Hz corner (geometric mean of 5 and 80 Hz); sampling with and without it |
| `stage5_sample.py` | 5 | `z_analog(t) -> z[n]`; prints the sampling rate, T_s, Nyquist, desired bandwidth, highest disturbance and anti-alias cutoff, plus six consistency checks; `acquire()` is reused by later stages |
| `stage6_analyze.py` | 6 | FFT of `z[n]`; detects the dominant peaks with a local noise floor; compares each against the truth; flags the alias if the filter is left out |
| `stage7_filter.py` | 7 | zero-phase Butterworth band-pass (0.15 Hz high-pass + 5 Hz low-pass, order 4 each) with the design justified from the spectrum; divides out the known analog filter delay; `estimate_waveform()` is reused |
| `stage8_estimate.py` | 8 | recovers the waveform and estimates A1, A2, f1, f2 (and phases) by FFT peak and by least-squares fit; record-length study against the Cramer-Rao bound |
| `stage9_error.py` | 9 | `e[n] = estimate - truth`: bias, variance, std, RMSE, SNR for raw, filtered and parametric estimates, with an error budget by source |
| `stage30_monte_carlo.py` | 30 | repeats the whole experiment with fresh noise (default 1000 trials): E[RMSE], its spread, and the distribution of the parameter estimates |
| `stage34_correlated_noise.py` | 34 | AR(1) noise, rho = 0 vs. 0.95, same raw std, same filters and estimator: total variance does not tell the story, the spectrum does |
| `run_all.py` | | runs everything in the order below |

## Results at a glance

| Measure | Raw `z[n]` | Filtered `x_hat[n]` | Parametric |
|---|---|---|---|
| RMSE, one run (Stage 9) | 3.22 | 0.40 | 0.015 |
| E[RMSE] over 1000 runs (Stage 30) | 3.220 | 0.343 (std 0.022) | 0.019 (std 0.005) |
| SNR, one run | 7.5 dB | 25.6 dB | 54 dB |

- The frequency estimates reach the Cramer-Rao bound (100% and 102% efficiency for f1 and f2 over
  1000 trials), with a small but statistically detectable bias (0.05% on A1).
- Correlated noise (rho = 0.95) at the same raw std leaves 2.9x more noise after a 10-point
  moving average and makes the estimator's RMSE 2.3x larger and the f1 scatter 5x larger.

Things worth knowing when reading the results:

- **Edge error dominates the filtered estimate.** About 96% of its mean-square error is
  deterministic and concentrated in the first and last 2 s of the record, where the 0.15 Hz
  high-pass has not settled. A longer record, or the parametric estimate, is the fix. Away from the
  ends the filtered RMSE is about 0.11.
- **Stage 9's single run was an unlucky draw.** Its RMSE of 0.401 sits near the top of the Monte
  Carlo distribution (mean 0.343), so quote the Monte Carlo mean.
- **The analog anti-alias filter's 21 ms delay** would otherwise be the largest error term (RMSE
  0.97 uncorrected), so Stage 7 divides out its known response over 0-8 Hz.
- **The 80 Hz line aliases to 20 Hz, outside the 0-5 Hz band**, so it is not fatal by itself at
  fs = 100 Hz. The larger damage from sampling without the filter is the noise that folds in
  (4x larger inside the useful band). A line at 98 Hz would land on the 2 Hz motion.

## Best order to run

Each stage imports the ones before it, and each one's printout explains why the next is needed.

```bash
cd probability/final-project
python run_all.py            # everything in order, about 2.5 minutes
python run_all.py --quick    # fewer Monte Carlo trials, about 1 minute
```

Or one at a time (about 2-3 s each unless noted), reading the printout before moving on:

1. `python stage3_spectrum.py` - **look first.** See what is in the signal and where. Every later
   design decision is read off this spectrum, so do this before choosing a sample rate.
2. `python stage4_antialias.py` - **decide the acquisition.** Pick fs and the analog filter from that
   spectrum; see what aliasing does with and without the filter.
3. `python stage5_sample.py` - **acquire.** Produce `z[n]` and read the printed assumptions; the six
   checks must all pass before the digital analysis means anything.
4. `python stage6_analyze.py` - **inspect the sampled data.** FFT and peak finding against the truth.
5. `python stage7_filter.py` - **filter.** Design the digital band-pass from what stages 3 and 6 showed.
6. `python stage8_estimate.py` - **estimate** the waveform and the four parameters (about 10 s).
7. `python stage9_error.py` - **score it once.** Error statistics and the error budget for a single
   noise realization.
8. `python stage30_monte_carlo.py [N]` - **score it properly.** N noise realizations (default 1000,
   about 15-60 s) turn the single-run score into a distribution.
9. `python stage34_correlated_noise.py [N]` - **stress the noise model.** Optional experiment
   (default 500 records per noise type, about a minute) showing why total variance is not enough.

The order is the order of decisions: inspect before designing, design before sampling, sample
before filtering, filter before estimating, estimate before scoring, and score once before
scoring statistically. Run stages 3-9 first even if you only care about Stage 30, since its
result is only interpretable against what they print.
