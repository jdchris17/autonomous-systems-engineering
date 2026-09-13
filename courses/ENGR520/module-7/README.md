# Module 7 — ABCD matrix (paraxial ray) optics

Ray optics as linear algebra: a ray is a 2-vector `[y, theta]`, every optical
element is a 2x2 matrix, and a whole system is one matrix product (Hecht 6.2).

## Files

| File | Section | What it does |
|---|---|---|
| `abcd_optics.py` | 7 | Library: `propagation_matrix(d)`, `thin_lens_matrix(f)`, `system_matrix(elements)` (= `M_n ... M_2 M_1`), `propagate` (apply the system to one ray), `trace_ray` (finely-sampled `[y,theta]` through the system, for plotting), `lens_positions`. |
| `ray_tracer.py` | 8 | **System 1:** `space -> lens -> space`. Five parallel rays (`theta=0`) at different heights; converge exactly at `z = d1 + f`. **System 2:** `lens -> space(f1+f2) -> lens`, an afocal two-lens system. Predicts `A = -f2/f1`, `C = 0` from the system matrix alone, then confirms it ray by ray. Figure `ray_tracer.png`. |

## Running

```bash
cd physics/module-7
python ray_tracer.py
```

## Results

**System 1** (`f = 100 mm` at `z = 30 mm`, rays at `y0 = -8..8 mm`):

```
predicted focus at z = d1 + f = 130.0 mm
   ray heights there: all ~0 (1e-16 level) -- exact in this ideal-lens model
grid search (no f used): minimum spread 3.6e-03 mm found at z = 130.02 mm
```

**System 2** (`f1 = 50 mm`, `f2 = 150 mm`, spacing `= f1+f2 = 200 mm`):

```
system matrix M = [[-3.000000, 123.3333], [0.00e+00, -0.333333]]
predicted (by hand): A = -f2/f1 = -3.000000,  C = 0

 y_in (mm)   y_out predicted   y_out traced   theta_out traced
     -3.00            9.0000         9.0000           0.00e+00
      1.50           -4.5000        -4.5000           0.00e+00
      3.00           -9.0000        -9.0000           0.00e+00
```

Every output ray is parallel (`theta_out = 0`) and scaled by exactly `A`,
matching the by-hand matrix algebra before a single ray was traced.

## Key ideas

- **`M = M_n ... M_2 M_1` is not an analogy to linear algebra — it *is* linear
  algebra.** Composing optical elements is matrix multiplication, ray tracing
  is matrix-vector multiplication, and every question ("where do parallel
  rays focus?", "does this system stay collimated?") is a question about the
  entries of one 2x2 matrix.

- **System 1's convergence is exact, not approximate, inside this model.**
  `y(z)` past the lens is an affine function of `y0` and `z`, so it hits zero
  at precisely `z = d1 + f` for every input height. The word "approximately"
  in the exercise is about *real* lenses, whose aberrations live outside the
  paraxial `[[1,0],[-1/f,1]]` description — not about this simulation.

- **`C = 0` is the definition of an afocal system**: parallel rays in stay
  parallel out (`theta_out = C y_in + D theta_in = D theta_in` when `C=0`).
  Placing two lenses a distance `f1+f2` apart forces `C = 0` algebraically,
  regardless of how much free space sits before or after the pair — that's
  the whole design principle of a Keplerian telescope / beam expander, and it
  falls out of a 2x2 matrix product rather than a lens-maker's derivation.
