# Module 13 — Star identification and camera attitude

All numbers come from `python run_all.py`, logged in `outputs/run_all_log.txt` (CPU only, Windows 11 on ARM). Experiment C also uses Astrometry.net 0.98, installed in WSL Ubuntu.

| File | Contents |
|---|---|
| `sky13.py` | Synthetic catalogue, pinhole camera, projection / back-projection, **weighted SVD (Wahba)**, attitude error |
| `exp_a_wahba.py` | Experiment A: attitude error vs centroid noise; weighted vs unweighted |
| `solver13.py` | `TriangleSolver`: triangle signatures → hypotheses → verification → best solution / rejection |
| `exp_b_identify.py` | Experiment B: the solver without associations; hard conditions; controls |
| `render13.py` | Experiment C rendering: catalogue positions first → camera → Module 11/12 PSF and noise machinery |
| `adapter13.py` | Module 12 detector catalogue → astrometric source list (coordinates untouched) and `.xyls` writer |
| `exp_c_integrate.py` | Experiment C: detector → our solver and Astrometry.net on the same lists |

**Screenshots.**
- Some screenshots were cut off at the bottom or partly hidden by the page's scroll button. Each time, the next screenshot repeated the hidden part, so nothing appears to be missing.
- The set refers to "the weighted SVD solution from Section 4", which was not included. I implemented the standard SVD solution of Wahba's problem (below).

## Common setup

**Catalogue.**
- 180 synthetic stars, uniform on a 7° radius cap around RA 150°, Dec +20°.
- Magnitudes 3–8, with counts rising ∝ 10^(0.35 m).
- Each star has an ID, RA, Dec, inertial unit vector **s** = (cos δ cos α, cos δ sin α, sin δ) and magnitude. Example rows:

| Star ID | RA | Dec | Unit vector s | Mag |
|---|---|---|---|---|
| 001 | 150.7264 | +23.4742 | (−0.8001, +0.4485, +0.3983) | 3.07 |
| 002 | 151.4279 | +18.3664 | (−0.8335, +0.4539, +0.3151) | 3.17 |
| 003 | 155.5721 | +22.7952 | (−0.8394, +0.3812, +0.3874) | 3.85 |

**Camera.**
- 512 × 512 pinhole, 30″ per pixel at the centre, so f = 6875.5 px. The field is 4.26° square.
- Camera axes: +z is the boresight, +x points along increasing column u, +y along increasing row v.
- **c = R_CI s**, then u = c_x + f c_x/c_z and v = c_y + f c_y/c_z.
- Pixels use the **zero-based pixel-centre** convention of Modules 11 and 12: the image centre is (255.5, 255.5).
- True boresights are random within 2° of the patch centre, with random roll. That puts 16–33 stars in the field (median 24).

**Wahba / weighted SVD.**
- B = Σ wᵢ **b**ᵢ **r**ᵢᵀ, where **b** are measured camera-frame directions and **r** are catalogue inertial directions.
- B = U S Vᵀ, and R = U diag(1, 1, det U · det V) Vᵀ. The determinant fix guarantees a proper rotation.
- Error: R_err = R_est R_trueᵀ and Δθ = arccos(clip((tr R_err − 1)/2, −1, 1)).
- **No machine learning is involved in the attitude.** It comes from measured directions and the geometry model.

---

## Experiment A — known field, known geometry, correct associations

Setup: 1,000 random attitudes per noise level.

Noise models:
- **Uniform:** every star has the same σ.
- **Star-dependent:** σᵢ = σ·g(mᵢ), with g running log-linearly from 0.5 (m = 3) to 2 (m = 8). At σ = 0.1 px, the stars span 0.05–0.2 px, the range the problem set asks for.

Weights: wᵢ = 1/σ_θ,ᵢ², where σ_θ = σ_px · cos^1.5(α)/f. That is the pixel-to-angle mapping of a pinhole camera; at α ≤ 3°, it differs from 1/f by 0.4%.

The prediction comes from the Fisher information J = Σ wᵢ (I − **b**ᵢ**b**ᵢᵀ), with the error vector distributed as N(0, J⁻¹).

**Required plot:** `outputs/exp_a_attitude_error.png`. The grey band is the 0.05–0.2 px range from the problem set.

| σ (px) | Uniform | Uniform, predicted | Star-dep. unweighted | Star-dep. weighted | Weighted, predicted | p90 (weighted) |
|---|---|---|---|---|---|---|
| 0.05 | 6.9″ | 6.9″ | 10.5″ | 8.0″ | 8.9″ | 21.7″ |
| 0.10 | 14.3″ | 14.4″ | 21.0″ | 18.2″ | 17.0″ | 44.3″ |
| 0.20 | 29.4″ | 27.6″ | 41.0″ | 34.7″ | 38.2″ | 86.0″ |
| 0.50 | 71.8″ | 69.5″ | 103.6″ | 85.4″ | 84.9″ | 214.9″ |

(Median attitude error over 1,000 trials; 0.075, 0.15 and 0.3 px are in the log.)

**Compact weighted vs unweighted comparison at σ = 0.1 px:**

| Case | Median | p90 | Cross-boresight | Roll |
|---|---|---|---|---|
| Uniform noise (weighted = unweighted) | 14.3″ | 34.6″ | 0.74″ | 14.2″ |
| 0.05–0.2 px spread, unweighted | 21.0″ | 53.0″ | 1.15″ | 20.9″ |
| 0.05–0.2 px spread, **weighted** | **18.2″** | 44.3″ | 0.92″ | 18.2″ |
| 0.025–0.4 px spread, unweighted | 36.2″ | 84.6″ | 1.85″ | 36.2″ |
| 0.025–0.4 px spread, **weighted** | **17.9″** | 44.3″ | 0.84″ | 17.8″ |

**What this shows:**
- **The error is linear in centroid noise.** The log-log slope is 1.008, and the weighted errors agree with the Fisher prediction to within Monte Carlo noise. With correct associations and calibrated geometry, this is a well-posed rotation fit whose accuracy follows directly from the measurement quality.
- **Roll dominates.** Cross-boresight pointing is about 20× more precise than roll: 0.7″ against 14″, or 0.025 px. Every star constrains the boresight directly, but roll only through its lever arm, the star's distance from the axis. That arm is at most about 1.5° here, against 206,265″ per radian. A wider field or more stars far from the axis would reduce the roll error.
- **Weighting matters only when the uncertainties actually differ.**
  - With uniform noise the two solutions are identical: they differ by at most 0.07″, from the small cos^1.5 factor.
  - With a 4× spread in uncertainty, weighting reduces the median by ×1.15 and wins in 61% of trials.
  - With a 16× spread it halves the error (×2.03) and wins in 73% of trials; the weighted result stays at the Fisher bound.
  - The weights must be *calibrated*. That is why Module 11 calibrated its centroid uncertainties empirically.

---

## Experiment B — unknown identities and roll (constrained pattern matching)

**What the solver receives (`solver13.py`):**
- Only pixel coordinates, shuffled: no IDs and no brightness order.
- A focal length that is **1% wrong** (6944 px instead of 6875).
- A search region: the boresight lies within 2.5° of (150°, +20°). The truth was drawn within 2°.
- An assumed centroid σ.
- Roll is unknown.

**Method:**
1. **Catalogue signatures, precomputed in 0.3 s.**
   - 124 catalogue stars can enter the field.
   - That gives 115,773 triangles with every side between 0.15° and the field diagonal, and a longest side of at least 0.6°.
   - Each triangle's signature is its sorted sides a ≤ b ≤ c, stored as **(a/c, b/c)** in a KD-tree.
   - The ratios are scale-free, so the focal-length error does not move them. c is checked separately, with a tolerance of 3σ (both triangle ends) + 3% scale.
2. **Image triangles** come from up to 14 detections. Each one queries the KD-tree within a tolerance set by the centroid uncertainty.
   - Vertices are identified by their opposite sides. Near-isosceles triangles try both orderings.
3. **Hypotheses.**
   - Each compatible pair gives 3 correspondences, then weighted SVD.
   - The 3-star residual check rejects mirror-image matches, and the boresight must lie in the search region.
   - Near-duplicate rotations (within 0.5°) are skipped, and a quick KD-tree count discards hypotheses with too few nearby detections.
4. **Verification against the whole catalogue subset.**
   - Project, then do a one-to-one (Hungarian) match to *all* detections.
   - Re-fit R (weighted SVD) and f (linear least squares), and repeat.
   - **Only matches beyond the 3 stars that generated the hypothesis count as verification.**
5. **Every hypothesis is verified, and the best one is kept** (most inliers, then lowest RMS).
   - It is accepted only if it has **≥ 6 extra matches** *and* matches **≥ 50%** of the catalogue stars it predicts in the frame.
   - Otherwise the result is "no solution".

**Results** (6 independent fields per condition, fixed seeds):

| Condition | Solved | Correct-ID fraction | Wrong associations | Verified matches (beyond hypothesis) | Reprojection median / p90 | Attitude median / max | f error | s / field |
|---|---|---|---|---|---|---|---|---|
| Nominal (roll 0, σ 0.1 px) | 6/6 | 1.000 | 0 | 21.8 (18.8) | 0.108 / 0.197 px | 4.0″ / 25.0″ | 1.4e-4 | 0.15 |
| Unknown image rotation | 6/6 | 1.000 | 0 | 22.8 (19.8) | 0.123 / 0.208 | 10.6″ / 18.0″ | 1.1e-4 | 0.15 |
| 20% missing stars | 6/6 | 1.000 | 0 | 17.8 (14.8) | 0.119 / 0.197 | 14.4″ / 38.7″ | 1.1e-4 | 0.14 |
| 20% false detections | 6/6 | 1.000 | 0 | 22.8 (19.8) | 0.123 / 0.208 | 10.6″ / 18.0″ | 1.1e-4 | 0.13 |
| Centroid noise 1 px | 6/6 | 1.000 | 0 | 22.8 (19.8) | 1.23 / 2.08 | 107″ / 180″ | 1.1e-3 | 1.70 |
| Centroid noise 3 px | 6/6 | 1.000 | 0 | 22.8 (19.8) | 3.69 / 6.23 | 321″ / 538″ | 3.2e-3 | 4.70 |
| Missing + false + 1 px | 6/6 | 1.000 | 0 | 17.8 (14.8) | 1.19 / 1.97 | 144″ / 387″ | 1.1e-3 | 1.72 |

**Controls, which must be rejected:**

| Control | False solutions accepted | Best hypothesis (extra / total matches) |
|---|---|---|
| Random points (24 per field) | **0/6** | 0–2 / 1–5 |
| Wrong sky (field from a different catalogue) | **0/6** | 1–2 / 4–5 |
| Random points, solver set to 3 px tolerance | **0/6** | **5–8 / 8–9** |

**What I found:**
- **Rotational invariance.** Random roll gives exactly the same success, identification and timing as roll 0. Triangle side lengths do not depend on roll.
- **Missing and false detections are handled by verification.** Missing stars only reduce the number of verified matches. Not one random false detection was associated with a star.
  - The "false" and "rotation" rows are identical because the same seeds give the same real stars, and the extra points never entered the solution.
- **Noise.** Identification stays perfect up to 3 px. The attitude error grows linearly (about 107″ per px), as in Experiment A, and so does the solve time, because the wider tolerances admit more hypotheses.
- **No field failed, so I investigated the closest call** (the 3 px-tolerance control).
  - With a loose tolerance, a search over thousands of hypotheses finds **5–8 coincidental "verifying" matches among pure random points**. In 4 of 6 fields that met the extra-match count on its own.
  - Those false hypotheses were rejected only because they matched less than 50% of the catalogue stars they predicted. A real solution matches 80–100%.
  - **Lesson:** a verification count is meaningless without its false-match rate, which rises with tolerance × detection density × number of hypotheses tried. That is why Astrometry.net uses a log-odds test. The fraction-of-predicted-stars test is our simple stand-in.
- **The 1% focal-length error is recovered** to about 1e-4 at nominal noise. Verification refits f, so approximate intrinsics are enough.

**Required figure:** `outputs/exp_b_matches.png`. It shows the hardest combined condition (20% missing, 20% false, 1 px noise):
- detections (+ inliers, × rejected);
- catalogue stars projected with the recovered solution (blue: matched; orange: unmatched, i.e. the deleted stars);
- green connecting lines for accepted matches, and the hypothesis triangle.

Result: 15 correct IDs, 0 wrong, all 4 false detections rejected.

---

## Experiment C — integrate the Module 12 detector and compare with Astrometry.net

### Part 1 — architectural change and adapter

- **Simulation order changed.** Module 12 placed stars at random pixels with no sky identity. `render13.py` now works in this order:
  1. catalogue RA/Dec;
  2. project through the true camera;
  3. render with the unchanged Module 11/12 PSF (FWHM 3), sky and noise model, plus hot pixels and streaks;
  4. add **40 faint uncatalogued field stars** (real sources the catalogue does not contain);
  5. run the image through the **Module 12 hybrid detector** (`HybridDetector`, `outputs/cnn_best.pt`).
- **Flux scale:** 1,000 e⁻ at m = 8. Saturation is not modelled.
- **Adapter (`adapter13.py`).**
  - It copies `x, y, flux, snr, centroid_uncertainty, quality_flags, cnn_score` and only reorders rows, brightest first. Each row keeps the index of its source row.
  - `check_unmodified()` asserts, for every list, that the coordinates are bit-identical to the detector's.
  - Per-axis σ = radial uncertainty/√2, used as weights.
  - The only coordinate change is the documented +1.0, applied when the FITS `.xyls` file is written.

### Part 2 — our solver on detector output (6 rendered fields)

**Hybrid source lists (CNN):**
- Each list held 18–28 catalogue stars, 31–37 uncatalogued field stars, and 0–1 artifacts.
- **All 6 fields solved, every detected catalogue star was identified (22.2 per field on average), and there were 0 wrong associations.**
- Attitude error: median 4.5″, max 7.4″.
- Reprojection RMS on the true correspondences: 0.100 px.
- The uncatalogued stars (60% of the list) were all correctly left unmatched.

**Effect of false detections.** The raw Module 11 candidate list (no CNN) adds 14–26 hot pixels and streaks per field.
- It still solved 6/6 with the same attitude, but produced **1 wrong association**: in field 200, an artifact was matched to a catalogue star.
- The RMS grew to 0.118 px.
- The CNN's artifact rejection matters for clean *identifications* more than for solve success, because verification already tolerates many extra points.

### Part 3 — Astrometry.net on the same lists

**Index.** Our stars are synthetic and do not exist in the real sky, so the real index files installed in WSL (4110–4118) cannot match them.
- I built **custom indexes from our own catalogue** with `build-astrometry-index -P 8..11`. P7 failed: its quads are too small for a catalogue this sparse.
- The config file lists only those indexes, so no real-sky solution is possible.

**Command:** `solve-field` with the problem set's flags, plus:
- `--scale-low 28 --scale-high 32` (the camera is 30″/px, which replaces the placeholder 20–60);
- `--ra 150 --dec 20 --radius 5.5` (the same search region as our solver);
- `-s FLUX` and `--crpix-center`.

**Pixel-convention check with a known reference.** I solved an exact (noiseless) list of projected positions twice:

| `.xyls` written | WCS-predicted − true position |
|---|---|
| **With +1.0** | (+0.0000, −0.0000) px, RMS 0.0000 |
| As-is (zero-based) | (−1.0000, −1.0000) px, RMS 1.414 |

This confirms that the +1.0 is required and correct.

**An issue I hit and fixed:** a config file written from Windows Python had CRLF line endings. `astrometry-engine` then looked for an index called `index-13-8\r` and every solve failed. The file is now written with LF.

### Part 4 — comparison (same hybrid source lists, 6 fields)

| Metric | Our solver | Astrometry.net (default: TAN + 2nd-order SIP) | Astrometry.net, linear fit only (TAN) |
|---|---|---|---|
| Solve success | 6/6 | 6/6 | 6/6 |
| Verified star matches (mean) | 22.2 | 22.2 (`.corr`); log-odds 69–120 | 22.2 |
| Wrong associations | 0 | — (not exposed; its matches agree in count) | — |
| Reprojection RMS (same true correspondences) | 0.100 px | **0.082 px** | 0.092 px |
| Reprojection median / p90 | 0.060 / 0.167 px | 0.058 / 0.129 px | 0.062 / 0.159 px |
| Camera attitude error, median / max | **4.5″ / 7.4″** | 21.6″ / 30.3″ | 11.9″ / 18.4″ |
| Processing time per field | **0.15 s** (Python; 0.3 s one-off index build) | 1.2–1.4 s wall-clock, including a WSL process launch | 1.23 s |
| False-solution rejection (3 random + 3 wrong-sky lists) | 0/6 accepted | 0/6 accepted | — |

On the raw (no-CNN) lists, Astrometry.net also solved 6/6, with RMS 0.099 px and attitude max 61.8″.

**Measured with pinhole (TAN) geometry:** both solvers agree to a fraction of a pixel. Their identifications agree, both reject false lists, and the residual distributions overlap (`outputs/exp_c_solutions.png`, right panel).

**Lower residuals with a larger attitude error is not a contradiction.**
- Astrometry.net fits a general WCS: a linear sky mapping (4 parameters) plus, by default, 2nd-order SIP distortion terms, fitted to about 20 stars.
- Those extra degrees of freedom absorb noise, which lowers residuals on the *fitted* stars.
- They also trade off against rotation, so the rotation recovered from that mapping is less accurate.
- Switching to a linear fit only halves the attitude error (21.6″ → 11.9″).
- Our solver assumes the true physical model: a rigid rotation, square pixels, a known principal point and one focal length. It has fewer free parameters, so it gives the best attitude here.
- With a real lens, which does have distortion, the comparison could reverse. Our solver would then need a distortion model too.

**Timing.**
- Astrometry.net's own solve is fast. Most of its 1.2–1.4 s is the `wsl.exe` round trip plus file I/O.
- The first call into a stopped WSL VM took 5.0 s; warm launches measured about 0.15 s in another run.
- So the speed comparison says more about integration overhead than algorithmic speed.

**Not done, because the inputs are synthetic:**
- Comparing WCS reference-frame conventions and coordinate accuracy against a real sky catalogue and real observations.
- The real indexes installed in WSL (4110–4118) could be used for that once a real frame is available.

## Conclusions

1. **With correct associations, attitude is pure geometry.** The weighted SVD error is linear in centroid noise, matches the Fisher prediction, and is dominated by roll in a narrow field. Weighting helps in proportion to how much the per-star uncertainties differ, and only if they are calibrated.
2. **A triangle-signature solver with full-catalogue verification** identified every star in every test (42 fields, 0 wrong associations), including random roll, 20% missing, 20% false and up to 3 px noise.
   - It rejected all random and wrong-sky controls.
   - Verification needs a false-match criterion, not just a count: with loose tolerances, coincidental "verifying" matches reach 5–8 on random points.
3. **The real detector output plugs straight in.** The adapter never touches coordinates. Our solver and Astrometry.net solve the same lists, agree on identifications and pixel residuals, and both reject false lists.
4. **Our solver's median attitude error is 2.6–4.8× smaller than Astrometry.net's (linear-only fit and default fit respectively)** on this distortion-free camera, because its model is physically constrained.
5. **The CNN mostly protects identification purity.** Without it, solves still succeed, but one artifact became a wrong association.
