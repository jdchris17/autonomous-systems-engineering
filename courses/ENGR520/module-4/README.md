# Module 4 — Linear systems and state space

Scalar second-order mechanical systems written in state-space form and
propagated with the same RK4 tools as the earlier modules.

## State-space form

```
m x'' + c x' + k x = F(t)
```

With state `x = [position, velocity]`:

```
x' = A x + B u,     u = F(t)

A = [[   0  ,   1   ]      B = [[  0  ]
     [ -k/m , -c/m  ]]          [ 1/m ]]
```

`x' = A x + B u` is the canonical linear dynamical-system form. The eigenvalues
of `A` are the roots of

```
lambda^2 + (c/m) lambda + k/m = 0
```

- **real part** = decay rate (negative → decays, zero → sustained, positive → grows)
- **imaginary part** = oscillation frequency

so the damping behaviour you observe physically is read straight off `A`.

## Files

| File | Block | What it does |
|---|---|---|
| `state_space.py` | — | Library: `oscillator_AB(m, k, c)`, `linear_dynamics(A, B, u)`, `eigenvalues`, RK4 `simulate`, `zero_crossing_period`. |
| `free_oscillator.py` | I | `m=1, k=4, c=0` → `omega_0 = 2 rad/s`, `T = pi s`. Propagates `x(t), v(t)`; measures the period from zero crossings (matches analytic to ~1e-12); checks `E = K + U` is constant (drift ~1e-14 J). Figure `free_oscillator.png`. |

## Running

```bash
cd physics/module-4
python free_oscillator.py
```

## Key idea

The undamped oscillator has eigenvalues `lambda = ±i omega_0` — purely
imaginary, zero real part, so nothing decays. The phase portrait is a closed
ellipse and total energy is flat: kinetic and potential just trade back and
forth. This is the `c = 0` baseline that damping (`c > 0`, eigenvalues move
into the left half-plane) and forcing (`u != 0`) are measured against.
