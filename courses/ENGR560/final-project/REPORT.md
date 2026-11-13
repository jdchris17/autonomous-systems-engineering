# Final project: `stellar_vision`, an astronomical perception pipeline

`stellar_vision` is a package built from the stable code of Modules 11–13. From a night-sky image it produces a verified camera attitude, or an explicit, coded failure.

- **To run everything:** `python run_all.py` (about 6 minutes on CPU, Windows on ARM).
- **To run the tests:** `python -m pytest tests` (56 tests, about 70 s).
- **Logs and figures** are in `outputs/`.
- **Original module directories:** `module-11` … `module-14` are untouched. They stay as historical references.

**Data limitation.** Every result below comes from the course's synthetic simulator. **No result here establishes real-camera performance.** See the end of this report for the open validation requirement.

**Screenshots.** A few lines were partly hidden by the page's scroll button. Each was readable in the next screenshot, so nothing appears to be missing.

---

## 1. Architecture

```
1 image (ndarray | FITS) + camera JSON + VisionConfig
2 calibrate()             bias / dark / flat / gain (identity by default), background + noise          stages.py
3 detect_candidates()     Module 11 matched filter, 5 sigma, local maxima                              _m11.py (verbatim)
3' screen                 ClassicalScreen: keep all (shape rules need the fit)  |  CNNScreen: Module 12
                          CNN on 32x32 patches, batched, rejects BEFORE fitting                        _m12_cnn.py (verbatim)
4 fit_centroids()         Module 11 Gaussian PSF fit -> x, y, flux, SNR, calibrated uncertainty, flags;
                          quality rules (SHARP, FIT_FAILED) in BOTH modes                              stages.py
  to_source_list()        Module 13 adapter: accepted rows, brightest first, coordinates untouched
5 TriangleSolver          Module 13 triangle signatures -> hypotheses -> weighted SVD -> refinement     solver.py
  polish()                NEW: rotation / focal-length alternation iterated to convergence
6 verify()                8 configurable acceptance criteria + attitude covariance                     verification.py
-> AstrometricResult      verified solution (IDs, attitude, quaternion, covariance, WCS, residuals,
                          diagnostics)  OR  failure_code + failure_reason                              result.py
```

**Classical detection and CNN classification are one modular pipeline, not two systems.**
- The two modes differ only in the screen object at step 3′.
- Everything downstream receives the same `SourceTable`.
- **The solver never knows which mode produced its sources.** `solve_sources()` accepts any source list, which is how the random-detection control is injected.
- **`detection_mode="classical"` never imports torch.** A test runs it with `sys.modules['torch'] = None`. Machine learning can therefore be switched off without switching off astrometric solving.

### Public interface

```python
from stellar_vision import StellarVisionPipeline, VisionConfig
config = VisionConfig(detection_mode="hybrid", verify_solution=True, offline=True)
pipeline = StellarVisionPipeline.from_config(config)
result = pipeline.solve(image="night_sky.fits", camera="camera_config.json")   # or a 2-D NumPy array
if result.success:
    print(result.attitude, result.n_matched, result.reprojection_rms_px, result.attitude_sigma_arcsec)
else:
    print(result.failure_code, result.failure_reason)
```

### Configuration

`VisionConfig` is a JSON-serialisable dataclass with a SHA-256 fingerprint that is stored in every result. It declares:
- **camera intrinsics:** `CameraModel` with size, focal length, principal point, PSF FWHM and the permitted focal-length tolerance;
- **calibration:** `CalibrationConfig` with version, read noise, gain, bias, optional dark and flat frames, and the empirical centroid-uncertainty scale 1.118 from Module 11;
- **star catalogue:** `SolverConfig.catalog`, together with the search region, tolerances and polishing;
- **detector and CNN:** `DetectionConfig`, including the CNN checkpoint and threshold;
- **acceptance thresholds:** `AcceptanceConfig`.

**Offline reference data** lives in `stellar_vision/data/`, with SHA-256 hashes in `MANIFEST.json`:
- the Module 12 checkpoint (byte-identical to the original);
- the Module 13 catalogue, stored as unit vectors so results are bit-reproducible, plus a CSV copy;
- the camera JSON and the default config.

No internet access, external service or truth data is used during inference. Astrometry.net remains an optional external comparison tool, reached through the `.xyls` adapter.

### Conventions, stated explicitly in code and results

- **Pixels:** zero-based pixel centres, so a 512 × 512 image's centre is (255.5, 255.5). FITS/Astrometry.net export adds exactly +1.0, as validated in Module 13.
- **Attitude:** R_CI (camera ← inertial). Quaternion is scalar-first. Boresight RA/Dec and roll (camera +x from local east towards north) are in degrees.
- **Covariance:** 3 × 3 in rad², for the small-angle error in the **camera frame** (cross-x, cross-y, roll). It is also summarised as 1σ values in arcsec.
- **WCS:** the exact TAN equivalent of the pinhole solution, with a 1-based CRPIX. Through astropy's origin-0 API it agrees with the camera model to below 1e-8 px (tested).
- **Serialisation:** `AstrometricResult.to_dict()` produces JSON, with the WCS stored as FITS header cards.

---

## 2. Experiment A — integration and reproducibility (`outputs/exp_a_log.txt`)

**One end-to-end execution** through the proposed API: Module 13 field 200, written to `outputs/night_sky.fits` plus `outputs/camera_config.json`.
- **VERIFIED:** boresight RA 149.12622, Dec +18.61901, roll 10.7633° (truth 149.12616, +18.61896, 10.7646°).
- **Attitude error 4.60″.** Reported 1σ: 0.12″ / 0.11″ cross-boresight and 3.38″ roll.
- 74 candidates → 54 sources → 23 matched (20 independent), catalogue fraction 1.00, reprojection RMS 0.098 px, reduced χ² 1.01.
- The full serialised result is in `outputs/exp_a_result.json`.

**Regression against Module 13** (`tests/test_regression.py`). The reference in `tests/reference/m13_reference.json` was frozen by running the *original* module code (`tests/make_reference.py`).
- **The copied routines are verbatim.** Their abstract syntax trees are compared with `module-11/starfield.py`, `module-12/cnn.py` and `module-12/data12.py`, and the checkpoint hash matches.
- **The six Module 13 Experiment C fields reproduce bit for bit** (module-13 settings: focal length +1%, no quality rules, no polishing):
  - identical rendered image hash;
  - identical source coordinates, σ and CNN scores;
  - identical R_CI bits, focal length and matched star IDs;
  - matches and attitude errors equal to the Module 13 log (23/19/18/22/28/23; 4.61, 5.29, 1.68, 0.84, 4.41, 7.45″).
- **The raw-candidate (no CNN) path** reproduces Module 13's raw Module 11 list bit for bit.
- **The solver alone** reproduces 18 Module 13 Experiment B source lists bit for bit: hypothesis counts, R, f and correspondences.

**Determinism.** Fresh pipelines on the same input give bit-identical attitudes, IDs, residuals and covariances, in both modes. NumPy and FITS inputs give identical results. torch inference runs single-threaded (`deterministic=True`).

**One deliberate numerical change**, enabled by default and disabled for regression: `SolverConfig.polish_iterations`.
- The new residual criterion exposed it. Module 13's solver alternates between fitting the rotation and the focal length only **4 times**, which converges linearly.
- With a 2% focal-length error, 0.08% scale error remains: about 0.25 px at the image edge, χ² 11.8. A correct field inside the declared tolerance was therefore rejected.
- Iterating to convergence (15–24 iterations) makes the result independent of the focal-length error: 5.1″ and χ² 1.08 for 0%, ±2% and +2.8%.

---

## 3. Experiment B — verified solutions and explicit failure (`outputs/exp_b_log.txt`, `outputs/exp_b_table.md`)

### 3.1 Acceptance criteria

The criteria live in `AcceptanceConfig` and are evaluated in order. The first failure sets the `FailureCode`, and every criterion's value is reported in `diagnostics["criteria"]`.

| Criterion | Statistic | Default | Failure code |
|---|---|---|---|
| Usable sources | number of accepted sources | ≥ 5 | NO_SOURCES / INSUFFICIENT_SOURCES |
| Hypothesis exists | compatible catalogue triangle within the search region | — | NO_HYPOTHESIS |
| Independent star matches | matches beyond the 3 hypothesis stars | ≥ 6 | INSUFFICIENT_MATCHES |
| Catalogue consistency | matched / catalogue stars predicted in the frame | ≥ 0.5 | LOW_CATALOG_FRACTION |
| Camera-model validity | fitted focal length not pinned at the permitted ±3% bound | margin > 0.002 | CAMERA_MODEL_INVALID |
| Residuals vs uncertainties | reduced χ² = Σ dᵢ² / (2N − 4), with dᵢ² = rᵢᵀ Σᵢ⁻¹ rᵢ; plus an absolute RMS bound | ≤ 5; ≤ 1.5 px | RESIDUALS_INCONSISTENT |
| Attitude geometry | Fisher covariance J⁻¹ · max(1, χ²_red); roll and cross-boresight 1σ | ≤ 200″; ≤ 6″ | WEAK_GEOMETRY |
| Competing hypotheses | best *distinct* (> 0.25° away) hypothesis with ≥ 6 independent matches | < 0.8 × best's matches | AMBIGUOUS |

**Where the thresholds came from.**
- The first three are Module 13's reference values, now configurable.
- The residual and geometry thresholds are about **3× the largest value seen for correct solutions** on separate calibration fields (seeds 300–311), never on the evaluation fields. The calibration covered nominal, low-SNR, defocus and artifact conditions, in both modes (`experiments/calibrate_thresholds.py`, `outputs/calibrate_thresholds_log.txt`).
- **These are working thresholds for this simulator, not universal statistical guarantees.**

**How dᵢ² is used.** Σᵢ = σᵢ² I, with σᵢ the per-axis calibrated centroid uncertainty. The bound is *not* applied as an exact χ² test:
- matching truncates large residuals;
- σᵢ are estimates;
- rotation and focal length are fitted.

It works as a plausibility bound. On the nominal field, mean dᵢ² is 1.85 (2 expected) and no star exceeds the 99.9% point. On the calibration fields, correct solutions had reduced χ² of 0.8–1.6.

**The reported covariance is honest.** On the calibration fields, z² = φᵀ P⁻¹ φ for the *true* attitude error φ averaged 2.3–3.2 per condition, against the expected 3. **The anisotropy is preserved:** at nominal, σ is 0.12″ / 0.11″ cross-boresight and **3.4″ roll**, 28× worse about the boresight, as found in Module 13. A single scalar would hide this.

### 3.2 Controlled outcomes (both modes)

| Scenario | Classical | Hybrid |
|---|---|---|
| **Nominal field** (positive) | VERIFIED: 24 matched, χ² 0.97, error 4.6″ | VERIFIED: 23 matched, χ² 1.01, error 4.6″ |
| **Focal length +2%**, inside tolerance (positive) | VERIFIED: χ² 1.08, error 5.1″ | VERIFIED: χ² 1.08, error 5.1″ |
| Blank image | NO_SOURCES | NO_SOURCES |
| Near-blank: hot pixels + streaks only | NO_HYPOTHESIS (15 streak/hot sources survive the shape rules) | NO_SOURCES (the CNN removes them all) |
| Insufficient stars (4) | NO_HYPOTHESIS | INSUFFICIENT_SOURCES (4 < 5) |
| Random false detections (30 points) | NO_HYPOTHESIS | NO_HYPOTHESIS |
| Wrong catalogue | INSUFFICIENT_MATCHES (1 of 6 needed) | INSUFFICIENT_MATCHES |
| Wrong sky region (5.5° off-centre) | NO_HYPOTHESIS | NO_HYPOTHESIS |
| Focal length +6%, outside tolerance | NO_HYPOTHESIS | NO_HYPOTHESIS |
| Duplicated pattern (two distinct solutions) | **AMBIGUOUS** (runner-up 120° away, 23 vs 23 matches) | **AMBIGUOUS** |
| Tight faint cluster (9 stars within 0.42°) | **WEAK_GEOMETRY** (roll σ 457″) | **WEAK_GEOMETRY** |

**22 of 22 runs ended in the expected outcome, and 0 false verified solutions were produced.**
- The ambiguous and weak-geometry cases are the important ones. They pass *every* Module 13 criterion: 20 independent matches, catalogue fraction 0.68–1.0 and χ² below 1. Only the new competing-hypothesis and observability checks reject them.
- **CAMERA_MODEL_INVALID is exercised only in the unit test.** An out-of-tolerance focal length never produced a triangle hypothesis, so the earlier NO_HYPOTHESIS check catches it first.
- With `verify_solution=False`, the pipeline returns the best hypothesis but marks it `verified=False`. A rejected hypothesis is never presented as an attitude: `attitude`, `wcs` and the covariance are `None`, and its summary goes to `diagnostics["rejected_hypothesis"]`.

---

## 4. Experiment C — classical vs hybrid complete systems (`outputs/exp_c_log.txt`, `exp_c_table.md`)

**Setup.**
- 6 fields per condition. Both configurations use the same images, camera model, catalogue and acceptance rules. The only difference is CNN rejection before the PSF fit.
- **Nominal** fields are the Module 13 images (seeds 200–205). The stress cases are new seeds.
- Each mode runs in its own process, so its memory can be measured.

### Results table (30 solvable fields + 6 incorrect-sky/catalogue fields)

| Metric | Classical | Hybrid |
|---|---|---|
| Verified solve success rate | **30/30** | 29/30 |
| False solutions accepted | 0/6 | 0/6 |
| Correct star IDs (of catalogue stars in frame) | **620/670** | 590/670 |
| Incorrect star associations | 9 | **1** |
| Median centroid error, matched stars | 0.073 px | 0.070 px |
| Reprojection RMS | 0.294 px | 0.249 px |
| Attitude error vs truth, median / p90 | **4.1″ / 15.0″** | 4.5″ / 19.7″ |
| Mean time: calibration / detection | 41 / 104 ms | 35 / 94 ms |
| Mean time: CNN classification | — | 68 ms |
| Mean time: centroid fitting | 1,981 ms | **312 ms** |
| Mean time: geometric solving + verification | 130 ms | 133 ms |
| **Mean time end-to-end** | 2,261 ms | **648 ms (3.5× faster)** |
| Peak memory: tracemalloc during a solve / process peak working set | 5.0 MB / 251 MB | 5.0 MB / 536 MB |

Notes on the table:
- The first solve per pipeline (catalogue index build, about 0.3 s, plus lazy imports) is excluded from the timings as warm-up.
- tracemalloc covers the Python and NumPy heap only. torch's native allocations show up only in the process working set, which also includes the simulator.

### By condition

| Condition | Question | Classical | Hybrid |
|---|---|---|---|
| **Nominal** | Accurate attitudes? | 6/6; 134/135 IDs; 1 wrong; 4.5″ | 6/6; 133/135; **0 wrong**; 4.5″ |
| **Low SNR** | Useful stars surviving? | 6/6; 96/144 IDs; 16.0 useful sources/image | **5/6**; 83/144; 15.0 per image; CNN discarded 6 true stars; one field INSUFFICIENT_MATCHES |
| **Strong defocus** | Does the CNN discard valid stars? | 6/6; 132/132; median 12.8″ | 6/6; **121/132: CNN discarded 10 true stars**; median **18.1″** |
| **Hot pixels & streaks** | Fewer false associations? | 6/6; **31 artifacts/image reach the solver**; 3 wrong; 5.6 s/image | 6/6; 0.2 artifacts/image; **0 wrong**; 0.7 s/image |
| **Missing & uncatalogued** | Tolerates incomplete catalogues? | 6/6; 121/121; 0 wrong | 6/6; 120/121; 0 wrong |
| **Incorrect sky / catalogue** | Rejects false solutions? | 0/6 accepted | 0/6 accepted |

The missing-and-uncatalogued case uses a catalogue missing 20% of its stars, plus 120 uncatalogued stars per image.

**Figures.**
- `outputs/exp_c_associations.png`: one hot-pixel/streak field, both modes side by side. It shows verified associations, candidates the CNN rejected before fitting, candidates rejected by the quality rules, and accepted but unmatched sources.
- `outputs/exp_c_system_comparison.png`: solve rate, artifacts reaching the solver, attitude error, and time by stage.

### Where does machine learning actually help?

- **Processing cost.** This is the largest and most consistent effect: 3.5× faster end to end, and 8× faster in artifact-heavy images.
  - The classical path fits every candidate, and a hot-pixel fit costs about 50 ms against about 5 ms for a star (measured in Module 12).
  - The CNN costs 68 ms per image and removes most of that work.
- **Cleaner inputs and fewer wrong associations.** About 5–31 artifacts per image reach the solver without the CNN, against 0–0.8 with it, and wrong associations fall from 9 to 1.
  - The classical shape rule rejects hot pixels (SHARP) but **not streaks**, whose cross-section fits a normal PSF width.
- **Not attitude accuracy.** Verification already rejects nearly every false detection, so attitude accuracy is unchanged at nominal (4.5″ for both).
- **Where the CNN costs something:**
  - Outside its training distribution, it **discards valid faint stars**: 10 under defocus (median attitude error 12.8″ → 18.1″), and 6 at low SNR, which cost one solve.
  - The classical path solved all 30 solvable fields, while the hybrid solved 29.
  - The hybrid also needs about 285 MB more process memory, which is the torch runtime.
- **The trade:** speed and association purity, against completeness in degraded imaging.
  - A sensible deployment default is the hybrid, with an automatic classical retry when the hybrid returns INSUFFICIENT_MATCHES or NO_HYPOTHESIS.
  - That costs nothing in the common case and would have recovered the one low-SNR field. I note this as a recommendation; it is not implemented or measured here.

---

## 5. A crucial limitation: real-image validation is **open**

**What these results don't cover.** The simulator provides exact truth, but it reproduces only its own assumptions:
- a Gaussian PSF with no optical distortion;
- a perfect pinhole camera;
- simple hot pixels and cosmic-ray streaks;
- a catalogue that *is* the sky.

**Not established:**
- real-camera performance;
- the real-world value of the CNN, which was trained only on this simulator;
- the transfer of the acceptance thresholds;
- whether the covariance is calibrated on real data.

**The 30/30 and 29/30 solve rates are simulated results and must not be read as operational reliability.**

**Open requirements before any real-world claim:**
1. Real frames with independent reference solutions, ideally Astrometry.net solutions against a real catalogue (Gaia/Tycho), with known camera calibration including distortion.
2. Re-derive the acceptance thresholds and the centroid-uncertainty scale on held-out real calibration frames.
3. Re-validate the CNN, and retrain if necessary, on real artifacts: satellites, bleed trails, diffraction spikes, bad columns.
4. Compare WCS conventions and coordinate accuracy against the reference solutions, adding a distortion model if residuals show one.

---

## 6. Files

| Path | Purpose |
|---|---|
| `stellar_vision/` | The package: `pipeline.py`, `config.py`, `stages.py`, `solver.py`, `verification.py`, `result.py`, `geometry.py`, `catalog.py`, `simulation.py` (test data only), `_m11.py` and `_m12_cnn.py` (verbatim copies), `data/` |
| `tests/` | 56 tests: regression (17), stages (14), determinism and I/O (4), positive and negative controls (21); `make_reference.py` plus the frozen reference; `scenarios.py` |
| `experiments/` | `exp_a_integration.py`, `exp_b_verification.py`, `exp_c_comparison.py`, `conditions.py`, `calibrate_thresholds.py` |
| `outputs/` | Logs, tables (`exp_b_table.md`, `exp_c_table.md`), figures, serialised result, example FITS, camera and config JSON |
| `pyproject.toml` | Installable package (`pip install -e .[hybrid,test]`); torch is an optional extra |
