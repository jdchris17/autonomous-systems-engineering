# Module 6 — Light as an electromagnetic phenomenon

Optics built up from more primitive principles: least-time paths, wave
superposition, and Maxwell's equations, rather than from memorised laws.

## Files

| File | Exercise | What it does |
|---|---|---|
| `fermat_snell.py` | 2 | Point `A` in medium 1, `B` in medium 2, flat interface at `y = 0`. Ray crosses at `x`; travel time `T(x) = (n1 L1(x) + n2 L2(x)) / c`. Plot `T(x)`, minimise it numerically, read off `theta1, theta2` at the minimum, and verify `n1 sin(theta1) = n2 sin(theta2)`. Figure `fermat_snell.png`. |
| `refraction_tir.py` | 7 | Small ray-interface model: `theta_t = arcsin((n1/n2) sin(theta_i))`, swept over `0 <= theta_i < 90 deg`, both directions of a glass/air interface plotted together. For `n1 > n2` (glass -> air) identifies the critical angle `theta_c = arcsin(n2/n1) = 41.81 deg` and marks the total-internal-reflection region beyond it; ray diagrams at angles below / at / above `theta_c`. Figure `refraction_tir.png`. |
| `double_slit.py` | 9 | Two coherent point sources, `I(theta) = cos^2(pi d sin(theta)/lambda)` mapped to a screen at distance `L`. Sweeps `lambda`, `d`, and `L`; numerically finds the fringe peaks and measures their spacing against `dy ~= lambda L / d` (agrees to `< 0.15%` whenever `lambda/d` is small). A `lambda/d = 0.30` case is pushed on purpose to show the paraxial formula fail — measured spacing grows from `1.05x` to `1.45x` the prediction from centre to edge. Figure `double_slit.png`. |
| `resolution.py` | 11 | Rayleigh criterion `theta_min = 1.22 lambda / D` for `450/550/700 nm` across apertures from a 5 mm eye to a 39.3 m ELT (converted rad -> deg -> arcmin -> arcsec), then inverted: what aperture resolves a 1 arcsec binary star, a 0.1 arcsec Pluto feature, a 0.05 arcsec exoplanet separation, 1 milliarcsecond stellar-surface detail, or a 10 cm license-plate digit from 400 km orbit. Figure `resolution.png`. |

## Running

```bash
cd physics/module-6
python fermat_snell.py
python refraction_tir.py
python double_slit.py
python resolution.py
```

## Results

`n1 = 1.0`, `n2 = 1.5`, `A = (0, 1)`, `B = (2.5, -1.2)`:

```
minimum-time crossing:  x* = 1.664373 m
   T(x*)       = 13.793 ns
   T(straight) = 14.138 ns      <- the geometric straight line is slower
   dT/dx at x* = -1e-12          (stationary point)

theta1 = 59.00 deg,  theta2 = 34.85 deg
   n1 sin(theta1) = 0.85718012
   n2 sin(theta2) = 0.85718011
   relative mismatch = 1.6e-08
```

and it holds across a sweep of geometries and indices (`n2 = 1.33, 1.5, 2.42`,
`B_x` from `-1` to `4`) — mismatch always `< 1e-7`.

## Key ideas

- The program is handed `T(x)` and nothing else. Setting `dT/dx = 0` gives

  ```
  n1 (x - xA) / L1  =  n2 (xB - x) / L2
  ```

  and the two sides are exactly `n1 sin(theta1)` and `n2 sin(theta2)`. Snell's
  law is not an input — it is the condition for a light path to be the fastest
  one. The ray bends toward the normal on entering the slower (higher-`n`)
  medium because that trades a little extra distance in the fast medium for a
  shortcut through the slow one.

- **Total internal reflection is just Snell's law running out of room.**
  `sin(theta_t) = (n1/n2) sin(theta_i)`; once `n1 > n2`, the right side passes
  1 before `theta_i` reaches 90 deg, and there is no real angle left to solve
  for. The critical angle `theta_c = arcsin(n2/n1)` is exactly where that
  happens — and it is the *same* angle you get approaching from the other
  side (`air -> glass`, `theta_i -> 90 deg`), because it's one interface
  looked at from two directions. Past `theta_c`, energy conservation has
  nowhere to send a transmitted wave, so all of it reflects. That's the
  guiding mechanism in a fiber-optic core.

- **`dy ~= lambda L / d` is itself an approximation, and it is worth knowing
  which one.** The far-field path difference `d sin(theta)` is essentially
  exact whenever `d << L` (true throughout this file). Converting angle to
  screen position via `y = L tan(theta)` is exact too — but *predicting* `y`
  with the small-angle swap `tan(theta) ~= sin(theta) ~= theta` only holds
  while `lambda/d` is small. Push `lambda/d` up to 0.30 and the fringes
  measurably spread out away from the centre instead of staying evenly
  spaced, because `tan` grows faster than `sin` as the angle grows.

- **Resolution is set by the aperture, not the eyepiece.** `1.22 lambda/D`
  alone reproduces Hubble's real diffraction limit (~0.06 arcsec at 550 nm)
  and the ELT's design goal (~3.5 mas) with no fitting. Run the formula
  backward and a milliarcsecond of resolution needs a ~140 m mirror — bigger
  than anything ever built — which is exactly why radio/optical
  interferometry (arrays of smaller telescopes combined, e.g. the EHT, VLTI)
  exists: it buys the resolving power of one huge aperture without building
  the huge aperture.
