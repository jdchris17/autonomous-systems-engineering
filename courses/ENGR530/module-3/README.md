# Module 3 — Discrete distributions

Bernoulli and Binomial from first principles: implement the PMF, see how its
shape depends on `n` and `p`, then validate it by simulation.

## Files

| File | What it does |
|---|---|
| `m03_pmf_visualizer.py` | **Build 1.** `bernoulli_pmf(k, p)` and `binomial_pmf(k, n, p)` (via `math.comb`, no `scipy`), plus `binomial_pmf_table(n, p)` for every `k = 0, ..., n`. Plots four cases — `(10, 0.1)`, `(10, 0.5)`, `(10, 0.9)`, `(100, 0.1)` — in a 2x2 grid, each annotated with its mean and std — `pmf_visualizer.png`. |
| `m03_binomial_simulation.py` | **Build 2.** Fixes `X ~ Binomial(20, 0.3)` and draws `100 / 1,000 / 10,000 / 100,000` samples with `rng.binomial`. For each sample size, bins the outcomes into empirical frequencies (`np.bincount(samples) / len(samples)`) and overlays the theoretical PMF from Build 1 — `binomial_convergence.png`. Prints the max and mean absolute error between empirical and theoretical at each size. |

## Running

```bash
cd probability/module-3
python m03_pmf_visualizer.py
python m03_binomial_simulation.py
```

## Key ideas

**Build 1 — why the shapes look the way they do.** A Binomial(n, p) is `n`
independent Bernoulli(p) trials summed, so its mean is `np` and its std is
`sqrt(np(1-p))`:

- **`p` sets the skew.** At `p = 0.1` most trials fail, so `k = 0` and `k = 1`
  are the likely outcomes and the tail stretches right toward `n` — right
  (positive) skew. At `p = 0.9` the mirror image holds — left skew, mass
  piled near `n`. At `p = 0.5` the distribution is exactly symmetric, since
  swapping success/failure labels maps the distribution onto itself
  (`P(k) = P(n-k)`).
- **`n` sets the scale and, via the CLT, the shape.** `Binomial(100, 0.1)`
  has the same `p` as `Binomial(10, 0.1)` but a mean 10x larger (`10` vs
  `1`) and a std only ~3.2x larger (`3.0` vs `0.95`, since std grows with
  `sqrt(n)` not `n`). That relative narrowing is exactly the Central Limit
  Theorem in miniature: a sum of more independent trials concentrates more
  tightly (in relative terms) around its mean and looks increasingly
  Gaussian, even though the underlying `p` is small enough that `n = 10`
  alone would still look sharply skewed.

**Build 2 — simulation is a quantitative tool, not just a picture.** The
empirical histogram is visibly ragged at 100 samples (max error ~0.045 —
individual bars are off by several percentage points) and by 100,000 samples
tracks the theoretical curve closely (max error ~0.002). That's the Law of
Large Numbers made concrete: empirical frequency is itself a random
variable, and its spread around the true probability shrinks as
`1/sqrt(n_samples)` — going from 100 to 10,000 samples (100x more data) cuts
the error by roughly `sqrt(100) = 10x`, which is exactly what the printed
error table shows.
