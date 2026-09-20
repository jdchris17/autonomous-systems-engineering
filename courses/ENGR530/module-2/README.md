# Module 2 — Bayesian updating

A single running example — a noisy target sensor — carried through three
builds: derive the update rule, validate it by simulation, then apply it
sequentially as evidence streams in.

## The sensor model

| Quantity | Symbol | Meaning |
|---|---|---|
| Prior | `P(T)` | probability the target is present before any report |
| Detection probability | `P(+\|T)` | probability of a positive report given the target *is* present |
| False alarm probability | `P(+\|~T)` | probability of a positive report given the target is *not* present |
| Evidence | `P(+)` | total probability of a positive report, marginalizing over the target |
| Posterior | `P(T\|+)` | updated belief after seeing a positive report |

Bayes' rule: `P(T|+) = P(+|T) P(T) / P(+)`, with
`P(+) = P(+|T) P(T) + P(+|~T) P(~T)`.

## Files

| File | What it does |
|---|---|
| `m02_bayes_sensor.py` | **Build 1.** `bayes_update(prior, detection_probability, false_alarm_probability)` computes the four quantities above for one positive report. Demo reproduces prior 0.01 → posterior 0.1610. Then sweeps the prior from 0.001 to 0.5 with the sensor held fixed (detection 0.95, false alarm 0.05) and plots `P(T\|+)` against `P(T)`, with a `y = x` reference line — `bayes_prior_sweep.png`. |
| `m02_bayes_monte_carlo.py` | **Build 2.** Validates Build 1 by brute force: simulates `N = 1,000,000` independent trials (draw whether the target is present from the prior, draw the sensor's report from the appropriate detection/false-alarm rate), then estimates `P̂(T\|+) = true positives / all positives` and compares it against the analytical result from `bayes_update`. |
| `m02_sequential_bayes.py` | **Build 3.** Starts at prior `P(T) = 0.02` and folds in a stream of 40 simulated +/- reports one at a time, using each posterior as the next prior. Prints the full `k, observation, P(T\|E1..Ek)` trace and plots it — `sequential_bayes_belief.png`. |

## Running

```bash
cd probability/module-2
python m02_bayes_sensor.py
python m02_bayes_monte_carlo.py
python m02_sequential_bayes.py
```

## Key ideas

**Build 1** makes the role of the prior visible: with a good sensor (95%
detection, 5% false alarm) a positive report still leaves you *more likely
wrong than right* whenever the prior is small enough — a rare-event problem
needs either a very reliable sensor or several independent reports before a
single positive means much. The sweep plot shows the posterior sitting well
above the `P(T|+) = P(T)` line everywhere, but the two only get close once
the false-alarm rate stops dominating the low-prior regime.

**Build 2** is the sanity check underneath every closed-form Bayesian result:
simulate the generative process directly, count outcomes, and the frequency
converges to the same number algebra produced (empirical 0.1604 vs.
analytical 0.1610 over 1M trials). If a derived formula and a simulation of
the same story disagree, trust the simulation and go find the bug in the
algebra.

**Build 3** is sequential updating: `P(T|E1,...,Ek)` becomes the prior for
observation `k+1`, so belief is a running product of likelihood ratios. With
a strong sensor (95%/5%) that product saturates fast — belief hits `1.0` (to
float64 precision) after only ~15 positive reports, which would make the
raw probability plot go flat for the rest of the stream. The script also
tracks belief in **log-odds** space, accumulated directly as a running sum
of log-likelihood-ratios rather than recovered from the (already-saturated)
probability — since `log(p/(1-p))` divides by zero once `p` rounds to
exactly `1.0`, but the evidence itself never stops accumulating. In log-odds
space every single observation stays visible, including the two misses
(steps 17 and 20) that briefly nudge belief back down before the stream
resumes: that's the concrete meaning of "watching belief evolve."
