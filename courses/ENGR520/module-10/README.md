# Module 10 -- Final Project: Star-Tracker Simulator

The course's final capstone project (shown in some course materials as
"Module 11" -- filed here as `module-10` per the same numbering quirk as
[`module-9`](../module-9/README.md), which is itself the course's real
"Module 10"). Goal: build a synthetic star-camera simulator, stage by
stage, reusing as much of modules 3/7/9 as legitimately applies rather than
re-deriving any of it.

This module is being built incrementally as the course exercise reveals
each stage. **Currently implemented: Section 3 (state definition) and
Stages I-VII of the pipeline (Sections 4-10) -- the full simulator, start
to detector.** Further stages will be added if the course assigns more.

## Files

| File | Purpose |
|---|---|
| [`state.py`](state.py) | Section 3: the simulator's state as a set of small dataclasses -- `Star`/`CelestialScene`, `Observer`, `CameraAttitude` (q or R, either direction on demand), `CameraIntrinsics`, `PhysicalOptics`, `Distortion`, `Sensor`, and the top-level `SimulatorState`. |
| [`pipeline.py`](pipeline.py) | Stages I-VII as pure functions on `SimulatorState`: `stage1_celestial_scene` through `stage7_sensor_measurement`, and the convenience `run_stages_1_to_3` / `run_stages_1_to_5` / `run_stages_1_to_7`. |
| [`run_stages_1_3.py`](run_stages_1_3.py) | Builds a real `SimulatorState` (catalog + Vega, live geometry for a real date/site), runs Stages I-III, validates every reused piece numerically, prints results, and saves `run_stages_1_3.png`. |
| [`run_stages_4_5.py`](run_stages_4_5.py) | Runs the full Stage I-V pipeline on two `SimulatorState`s (same catalog/observer/pointing, different instruments) so Stage IV's distortion and both of Stage V's PSF regimes are exercised for real, not asserted; prints diagnostics and saves `run_stages_4_5.png`. |
| [`run_stages_6_7.py`](run_stages_6_7.py) | Runs the full Stage I-VII pipeline on the same two instruments, exercising Stage VI's photon budget and Stage VII's Poisson sensor realization in both PSF regimes; prints diagnostics and saves `run_stages_6_7.png`. |
| [`_pathutil.py`](_pathutil.py) | One helper, `load_module(alias, file_path)`: loads a `.py` file by explicit path under a private name, bypassing `sys.path`/`sys.modules` name lookup. Used only where it's actually needed -- see "A `frames.py` naming collision" below. |

## Running

```bash
python run_stages_1_3.py
python run_stages_4_5.py
python run_stages_6_7.py
```

Requires `numpy`, `scipy`, and `matplotlib`, and (via reuse) the sibling
`module-3`, `module-7/camera`, `module-8`, and `module-9` folders in place
alongside `module-10`.

## Results

A wide-field camera (50 mm, 36x24 mm sensor, 960x640 px) pointed directly
at Vega's real live-computed sky position, from 40N/74W at a specific
date/time, against a synthetic 20,000-star catalog:

```
CameraAttitude round trip (built from R, converted to q, back to R):
   max |R_from_R - R_from_q| = 3.47e-16
   |R_BL R_BL^T - I| = 2.22e-16, det = 1.0000000000

Stage I  -- celestial scene:
   catalog stars           :  20,001
   above the horizon       :  10,033

Stage II -- observer -> camera frame:
   in front of the camera  :   9,223

Stage III -- geometric projection:
   (u, v) computed for     :   9,223 stars (no FOV/sensor cull yet)
   ... of which inside the sensor rectangle: 607

Stage III cross-check against Module 7's project_points: max |du,dv| = 3.73e-09 px

Vega (pointed at directly): (u, v) = (480.0000, 320.0000), principal point = (480.0000, 320.0000)
   diff = 2.54e-13 px
```

`run_stages_1_3.png` -- the ideal (u, v) star field, no PSF or detector yet,
deliberately plotted beyond the sensor rectangle to show the full
projection before any FOV cut is applied.

Stages IV-V, run on two instruments sharing the same catalog/observer/
pointing (both aimed straight at Vega):

```
--- Config A: 50mm wide-field lens, k1=-0.25 k2=+0.05 (point-source PSF regime) ---
  distortion_px  max / mean   : 22.983 / 6.122 px (over the 607 stars actually inside the sensor rectangle)
  Vega (pointed at, r~0)      : (du, dv) = (+0.00e+00, +0.00e+00) px -- distortion vanishes at r=0
  Stage V regime              : point-source (first_null < 0.5 px) -- bilinear split onto 4 nearest pixels
  first_null_px               : 0.0358 px
  stars rendered               : 651
  Vega: Stage IV (u_d,v_d)    = (480.0000, 320.0000)
  Vega: rendered-stamp centroid = (480.0000, 320.0000)  -- error 0.0000 px

--- Config B: 50mm-aperture telescope, zero distortion (resolved Airy PSF regime) ---
  distortion_px  max / mean   : 0.000 / 0.000 px  (k1=k2=0 -- exact identity, checked)
  Stage V regime              : resolved Airy PSF (first_null = 4.47 px)
  window flux captured        : 98.65% (half-window=27 px)
  Vega: Stage IV (u_d,v_d)    = (100.5000, 100.5000)
  Vega: rendered-stamp centroid = (100.4928, 100.4928)  -- error 0.0102 px
```

`run_stages_4_5.png` -- left: distortion magnitude vs. radius from the
principal point, data on top of the analytic `r|k1 r^2+k2 r^4|` curve;
middle: Config A's star field after distortion (barrel: corners pulled
inward); right: Config B's rendered Airy PSF on Vega, log-stretched so the
diffraction rings are actually visible instead of buried under the
central peak.

Stages VI-VII, same two instruments:

```
--- Config A: 50mm wide-field lens (point-source PSF regime) ---
  stars actually rendered       : 651
  normalized PSF sum (should be ~1): mean=1.0000, min=1.0000
  flux-ratio check: mag 0.03 vs 8.51 -> N_i/N_j measured=2461.696891, 10^(-0.4 dm)=2461.696891, diff=2.73e-12
  Stage VII Poisson check (brightest pixel, lambda=9549923.538):
     20,000-draw Monte Carlo: mean=9549924.190, var=9704825.018 (Poisson requires mean==var==lambda)
  one real Poisson realization: total photons = 13,777,729 vs expected 13,777,711.1 +/- 3711.8 (+0.00 sigma)
  N_e = QE * N_gamma holds exactly, every pixel: True (QE=0.7)

--- Config B: 50mm-aperture telescope (resolved Airy PSF regime) ---
  stars actually rendered       : 1
  normalized PSF sum (should be ~1): mean=0.9865, min=0.9865
  Stage VII Poisson check (brightest pixel, lambda=50003.194):
     20,000-draw Monte Carlo: mean=50003.093, var=49486.220
  one real Poisson realization: total photons = 943,203 vs expected 942,110.9 +/- 970.6 (+1.13 sigma)
  N_e = QE * N_gamma holds exactly, every pixel: True (QE=0.7)
```

`run_stages_6_7.png` -- top-left: expected photon count `N_i` vs.
magnitude for every Config A star, on top of the analytic
`N_ref * 10^(-0.4 dm)` curve; top-right: Config A's whole realized sensor
image (log-stretched -- Vega alone carries ~9.5e6 of the ~1.4e7 total
photons, saturating the display, while the other 650 stars' much fainter
real counts are still visible); bottom row: Config B's Vega, Stage VI's
smooth `lambda_ij` next to ONE Stage VII Poisson realization of it --
same Airy rings, now with visible shot-noise texture.

## Key ideas

- **State first, pipeline second.** Section 3 asked for a state
  definition before any pipeline logic -- so `state.py` has zero pipeline
  code in it, and `pipeline.py` has zero dataclasses in it. Every stage
  function takes `(state, ...)` and returns a plain dict; nothing is
  hidden in module-level globals or object mutation.

- **`CameraAttitude` stores one thing, exposes both.** Give it a
  quaternion or an `R_LB` matrix (never both -- `__post_init__` enforces
  exactly one), and `.R_LB`, `.R_BL`, `.quaternion` are all available as
  properties, converting on demand via Module 3's
  `quaternion_to_rotation_matrix` / the new `rotation_matrix_to_quaternion`
  (added to `module-3/quaternions.py` this module, validated to ~1e-16
  over 2000 random rotations plus the near-180-degree edge case). This is
  exactly the "R or q, your choice" flexibility Section 3 asked for,
  without silently storing a matrix and a quaternion that could drift out
  of sync with each other.

- **Stage I-III is a direct re-plumbing of Module 9's capstone, not a
  restart.** `star_celestial -> star_earth_fixed -> star_local` are
  Module 9's `frames.py` functions, called unmodified. `R_BL` -- camera
  attitude -- is Module 3 rigid-body rotation math (Section 5 says this
  outright: "This is fundamentally your Module 3 rigid-body rotation
  mathematics"). Stage III's `[u,v,1]^T ~ K[x_n,y_n,1]^T` is Module 7's
  pinhole model, and the pipeline is cross-checked line-for-line against
  Module 7's own `project_points` (agreement to 3.7e-9 px -- not a
  coincidence, since they're algebraically the same map, but cheap
  insurance that the wiring is right).

- **A `frames.py` naming collision, and why it's worth documenting.**
  `module-3/frames.py` (an early "body vs. inertial frame" demo) and
  `module-9/frames.py` (the astronomy C->E->L->B chain) happen to share a
  filename. A naive `sys.path.insert` for both directories followed by
  `from frames import ...` is a real bug, not a hypothetical one -- it
  broke `run_stages_1_3.py` on the first run here, silently resolving
  `frames` to whichever directory landed at the front of `sys.path` last.
  The fix isn't "get the insertion order right" (that's one accidental
  fix away from breaking again the next time a file is reordered); it's
  `_pathutil.load_module`, which loads `module-9/frames.py` by its exact
  file path under a private alias, so the ambiguous bare name `frames` is
  never looked up off `sys.path` at all. `pipeline.py` does this once;
  `state.py` and `run_stages_1_3.py` both get the resulting functions
  from already-resolved modules rather than re-importing `frames` a
  second and third time.

- **Separating the geometric projection from everything downstream is
  deliberate, and enforced by what Stage III does *not* do.** Section 6
  says explicitly: "Still no diffraction. Still no detector." Stage III
  computes an ideal, continuous (u, v) for every star still in front of
  the camera -- it does not cull to the sensor rectangle (that's shown in
  the printed output only "for scale," and the plot deliberately draws
  points spilling past the red sensor border) and it carries
  `state.distortion` without ever applying it. Whatever stage comes next
  (PSF, sensor response, distortion) operates *on top of* this ideal
  geometry, never inside it -- keeping each physical effect isolated and
  independently testable, the same discipline `module-9`'s PSF/photon-budget
  work already established.

- **A radial distortion model isn't meant to be extrapolated to grazing
  rays, and the code didn't originally protect against that.** Because
  Stage III deliberately never culls on FOV, `stage4_distortion` inherits
  every in-front-of-camera star, including ones with `z` near zero (rays
  almost perpendicular to the boresight) where the normalized-coordinate
  radius `r` blows up and `k2 r^4` sends the distorted position to
  nonsense values (the first run of `run_stages_4_5.py` printed a "max
  distortion" of `~2e22 px` before this was caught). That's not a code bug
  -- the formula did exactly what it says -- it's a modeling-domain bug:
  no real lens's calibrated distortion polynomial is valid that far off
  axis either. `pipeline.py` still computes `distortion_px` for every
  star (the pipeline stays complete and honest about what the model
  predicts everywhere), but `run_stages_4_5.py`'s printed summary and plot
  restrict themselves to stars whose *ideal* position actually lands on
  the sensor -- the domain the model is meant for.

- **Stage V's two regimes are exercised on purpose, not by coincidence.**
  One `SimulatorState` (Config A, the original wide-field lens) and one
  new one (Config B, `module-9`'s already-validated 50mm-telescope
  numbers) share the exact same catalog, observer, and pointing --
  same real sky, same target, only the instrument differs. Config A's
  `first_null_px = 0.036` lands deep in the point-source limit (same
  finding `module-9/synthetic_celestial_camera.py` Config B already made,
  now reached through this module's own state/pipeline split); Config B's
  `first_null_px = 4.47` resolves an actual Airy disk with visible rings.
  Both are checked, not just labelled: Module 8's `star_centroid.py`
  `centroid()` function is run on each config's *rendered* Vega stamp and
  compared back to Stage IV's `(u_d, v_d)` -- the same optics+sampling
  validation `star_centroid.py` already did, now applied to this
  pipeline's own output rather than to a copy of the logic.

- **Reuse, once more, stayed real reuse.** Stage V's Airy PSF is Module
  8's `airy_value` / `first_null_px` (`star_centroid.py`), called
  directly; the fine-grid block-averaged rendering technique (avoid
  aliasing a rapidly-oscillating pattern by point-sampling it) is the
  same one `star_centroid.py` and `module-9/synthetic_celestial_camera.py`
  already validated, adapted here rather than re-derived. The only new
  arithmetic in this module's Stage V is the regime split's bookkeeping
  and the relative-brightness weighting (`module-9`'s `relative_flux`,
  also reused) -- everything that determines *what a diffraction-limited
  image actually looks like* was already built and checked in an earlier
  module.

- **Stage VI is Stage V, called with a different number.** The exercise's
  own framing -- "you've joined Carroll & Ostlie magnitude, Module 8 PSF,
  and Module 9 photon statistics into one model" -- is implemented
  literally as composition, not as new rendering code: `stage5_psf`
  gained one optional `weights` argument (default: Stage V's original
  relative-brightness placeholder, so nothing about Stage V's own
  behavior or its earlier results changed), and `stage6_photon_distribution`
  calls it with real expected photon counts (`expected_photons`, Carroll &
  Ostlie's `F propto 10^-0.4m` scaled from a chosen m=0 reference) instead.
  `lambda_ij = N_i PSF_ij` falls out of the *same* fine-grid Airy kernel
  Stage V already built and validated -- there is no second PSF
  implementation to keep in sync with the first.

- **`expected_photons`'s zero-point constant is given directly in
  `pipeline.py`, not imported from `module-9/synthetic_celestial_camera.py`
  -- on purpose.** That file already has this exact formula (same
  `ZERO_MAG_PHOTON_RATE`, same "order of magnitude, not a calibration"
  caveat), and importing it would have been the more obvious reuse move.
  But its own top-level code does `from frames import (star_celestial,
  ...)` -- the identical ambiguous bare import `_pathutil.load_module`
  was built to route around for `module-9/frames.py` itself. Importing
  that whole file here would silently reintroduce the collision this
  module already fixed once. Four lines of arithmetic and two named
  constants, copied with attribution, was the lower-risk call than a
  second workaround for the same landmine.

- **QE is deliberately split across two stages, matching where the
  exercise puts it.** `expected_photons` (Stage VI) computes *incident*
  photons only -- no QE factor -- because Stage VII's own text introduces
  `N_e = QE * N_gamma` as the "slightly more realistic" optional next
  step, implying Stage VI's `N_i` is pre-QE. `stage7_sensor_measurement`
  is where `state.sensor.qe` (carried since Section 3, unused until now --
  the same pattern `state.distortion` followed through Stages III-IV)
  finally multiplies in, and nothing past it: no read noise, dark current,
  ADC modeling, sensor nonlinearity, thermal modeling, or rolling-shutter
  behavior, exactly the exercise's own "stop there... those belong in
  later engineering work."

- **Two bugs, both caught by looking at the numbers rather than trusting
  the formula.** First, `run_stages_6_7.py`'s Config A realized-image
  panel came out solid black under a linear/sqrt stretch -- Vega's ~9.5e6
  photons (a real consequence of the point-source regime dumping an
  entire exposure's flux onto one undersampled pixel, not a bug) swamped
  the other 650 stars' much smaller counts on any reasonable color scale;
  fixed with a log stretch instead of hiding or rescaling the underlying
  data. Second, the *fix* for that then revealed a real rendering bug:
  `matplotlib`'s `LogNorm` cannot take `log(0)`, and pixels at exactly
  zero (everywhere Stage V/VI's kernel window never touched) were
  rendering as transparent -- the white figure background showing through
  -- rather than black, which looked exactly like scattered noise outside
  the Airy rings until traced back to the zeros. Fixed by flooring every
  log-stretched image at a small positive value below its display `vmin`
  before plotting (the underlying data arrays are untouched -- this is a
  display-only fix).
