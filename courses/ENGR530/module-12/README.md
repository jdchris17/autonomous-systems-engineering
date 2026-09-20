# Module 12 — Filtering, sampling, and system design

From ideal frequency-domain filters to realizable ones, then to what sampling
does to a signal and how to design a whole acquisition chain. All scripts score
themselves against a known clean signal.

## Files

| File | What it does |
|---|---|
| `m12_lowpass.py` | `5 sin(2 pi 2t) + 2 sin(2 pi 20t) + noise`. An idealized brick-wall filter and a realizable 121-tap windowed-sinc FIR (6 Hz cutoff, 0.3 s delay). RMSE vs. the clean 2 Hz signal: raw 1.73, ideal 0.24, FIR 0.23 (6.77 if its delay is not accounted for). |
| `m12_highpass.py` | Drift + faster signal + noise. RMSE vs. cutoff for an ideal filter and a 2nd-order Butterworth (causal and zero-phase). Best cutoffs: ideal about 0.96 Hz (0.49), zero-phase 0.51 Hz (0.51), causal 0.33 Hz (1.05). Too low leaves drift; too high removes the desired signal. |
| `m12_aliasing.py` | One 70 Hz sinusoid sampled at 500, 200, 140, 120, 100, 80 Hz: waveform, samples, sampled spectrum, apparent frequency (70, 70, 70, 50, 30, 10 Hz) and the fold map. |
| `m12_antialias.py` | 5 Hz motion + 90 Hz vibration into a 100 Hz ADC. Downsample only leaves a 0.495 alias at 10 Hz; low-pass first leaves 0.004; filtering after the ADC leaves it untouched. |
| `m12_noise_bandwidth.py` | Signal below 5 Hz plus white noise, low-pass cutoffs 5 to 100 Hz. Output noise grows as sqrt(cutoff); a 5 Hz cutoff distorts the signal. Total SNR peaks near 9-10 Hz (25 dB). |
| `m12_imu_chain.py` | Full chain for the IMU signal: analog anti-alias filter, 200 Hz ADC, digital band-pass. Drift, an 8 Hz vibration, a broadband structural vibration and a 201 Hz motor tone are added. Every choice is read off the printed spectrum. Residual contamination 2.01 without the analog filter vs. 0.13 with it. |
| `m12_sampling_rate_selection.py` | Camera-stabilization gyro: 25 Hz motion, 300 Hz vibration, ADC 100 / 200 / 500 / 1000 Hz. Compares fold locations, required filter order and measured aliasing, and analog-filter delay. Recommends 500 Hz. |

Each script prints its numbers and a "What this means" block, and saves PNGs next to it.

## Running

```bash
cd probability/module-12
python m12_lowpass.py
python m12_highpass.py
python m12_aliasing.py
python m12_antialias.py
python m12_noise_bandwidth.py
python m12_imu_chain.py
python m12_sampling_rate_selection.py
```

## Key ideas

**Filtering is a tradeoff, and RMSE against a known truth makes it measurable.**
A high-pass cutoff too low leaves drift, too high removes the signal. A low-pass
cutoff too wide admits noise (proportional to sqrt of bandwidth), too narrow
distorts the signal. In both cases the best cutoff sits in the gap between what
you want and what you do not.

**Ideal is not realizable.** A brick-wall filter needs an infinite, non-causal
impulse response. Real filters have gradual transitions and delay or phase shift;
zero-phase (forward-backward) filtering removes that, but only offline.

**Aliasing is irreversible.** A tone above fs/2 reappears at `|f - k fs|` with
full amplitude and is indistinguishable from a real signal at that frequency. A
digital filter applied after sampling cannot remove it, so the protection must be
an analog filter before the ADC.

**What must be kept out is everything above `fs - f_max`,** not everything above
fs/2, because that is what folds into the wanted band. The transition band is
`f_max` to `fs - f_max`, so a higher rate buys a wider transition band and a
simpler analog filter.

**Rate, analog filter order and delay are one trade.** In the rate study a 1%
aliasing tolerance needs analog order 4 at 100 Hz, 3 at 200 Hz and 1 at 500 Hz,
while the filter's delay grows from 3 to 14 ms with order. 1000 Hz adds no
measurable benefit over 500 Hz for a 300 Hz vibration and doubles the data rate.
Measured need is lower than the worst-case spec (order 3 for 60 dB at fs - 25 Hz),
because the modeled vibration stops at 300 Hz; the spec protects against
out-of-band content the model leaves out.

**The chain order is fixed by what can be undone.** Analog filter protects,
sampling digitizes, digital filter selects. Anything folded by sampling is gone
for good; anything left in-band after sampling can still be filtered digitally.
