# Module 12 — CNNs on astronomical image patches

All numbers come from `python run_all.py` (CPU only: Windows 11 on ARM, 12 logical cores; torch 2.15 CPU build).
Figures and logs are in `outputs/`.

| File | Contents |
|---|---|
| `data12.py` | Module 11 simulator plus cosmic-ray streaks; detector-proposed patches; one-to-one labelling; image-level splits |
| `cnn.py` | `StarCNN`, the preprocessing (`transform`), augmentation, checkpoint loader |
| `exp_a_train_cnn.py` | Experiment A: training, threshold chosen on validation, test table, loss plot, checkpoint |
| `exp_b_filters.py` | Experiment B: filters, activation maps, controlled probes |
| `exp_c_pipelines.py` | Experiment C: classical vs hybrid vs combined pipelines, three conditions, timing |
| `hybrid.py` | **Final deliverable**: `HybridDetector` / `hybrid_detect_sources()` (classify before fitting) |
| `outputs/cnn_best.pt` | Best weights + preprocessing config + frozen threshold (load with `weights_only=True`) |

## Data (shared by all experiments)

**Simulator.**
- Images come from the Module 11 simulator (512×512 pixels, Gaussian PSF, Poisson plus read noise, 50 stars, 10–20 hot pixels).
- I added one artifact type, **elongated cosmic-ray streaks**: 2–6 per image, 6–20 px long, cross-section σ 0.6 px, 500–5,000 e⁻.
- Without the streaks, Module 11's detector proposes almost nothing besides stars and hot pixels, and Experiment B could not ask about elongated features.
- The simulator was deliberately not developed any further.

**Candidates and labels.**
- Candidates come from the **unchanged** Module 11 detector (matched filter for FWHM 3, threshold 5σ).
- Each candidate gives a 36×36 patch: background-subtracted and divided by the image's measured background RMS, so pixels are in units of the detector's own noise.
- Candidates are matched one-to-one (Hungarian matching, within 2 px) to true stars.
  - A matched candidate is a *bright star* (true SNR ≥ 20) or a *faint star*.
  - An unmatched one is a *hot pixel* (within 1.5 px of one), *elongated* (near a streak) or *noise / other*.

**Splits by image.**

| Split | Seeds | Conditions | Patches | Star fraction |
|---|---|---|---|---|
| Training | 100 images, 10000–10099 | Mixed: background U(80, 320), read noise U(4, 10), flux floor U(100, 300), flux ceiling U(3k, 30k), FWHM U(2.7, 3.3) | 6,324 | 0.67 |
| Validation | 30 images, 20000–20029 | Same mix | 1,900 | 0.68 |
| Test | 40 images per condition, 30000–30039 | Each fixed condition | 2,846 (nominal) | 0.69 |

- No image appears in more than one split.
- **Defocus never appears in training.**

**Noise peaks are rare.** At a 5σ threshold on the matched-filtered image, pure noise peaks are rare: only 24 of the 6,324 training candidates are "noise / other". Most of those are star wings or blends. Results for that subtype rest on 11–31 test candidates and are only indicative.

---

## Experiment A — CNN classifier

**Model.**
- Three conv blocks (5×5 → 8 channels, 3×3 → 16, 3×3 → 32; each followed by ReLU and 2×2 max-pool), then 512 → 32 → 1 logit.
- **22,465 parameters**, float32 throughout.

**Preprocessing.** The steps are stored in the checkpoint.
1. Patch pixels are in noise (RMS) units, computed per image, with no learned parameters.
2. `asinh(x/3)` compresses bright cores while keeping faint contrast. It is monotonic, so brightness order is kept.
3. One scalar mean/std, fitted on the **training** patches only (mean and std of the centre-cropped asinh values).

**Augmentation.**
- Training only: random rot90, horizontal/vertical flips, and ±2 px translation (a random 32×32 crop of the 36×36 patch). All of these are valid symmetries for a circular PSF.
- Augmentation never changes intensities, so normalisation is unaffected.
- Evaluation uses the centre crop.

**Training.**
- Adam with learning rate 0.001, batch 64, `BCEWithLogitsLoss`, early stopping on validation loss (patience 10, best weights restored), at most 60 epochs.
- Took 128 s.
- **Validation loss was still slowly improving.** The best epoch was 53 (validation BCE 0.0193), and the run ended at the 60-epoch cap rather than by early stopping. The epoch-53 weights were restored. More epochs might help a little.
- There is one validation spike at epoch 30, the kind of transient Adam produces; the restore guards against it.
- Plot: `outputs/exp_a_training.png`.

**Threshold.**
- On the **validation** images, I took the largest threshold that keeps ≥ 98% of the detector-proposed true stars: **P(star) ≥ 0.902**.
- It was then frozen and stored in the checkpoint. Nothing was tuned on test data.

**Test results (40 nominal images, 2,846 candidates, used once):**

| Metric | Value |
|---|---|
| ROC AUC | **0.9998** |
| Precision at the frozen threshold | **0.9974** |
| Recall of proposed true stars | **0.9944** (above the 0.98 target, since test is easier than the mixed validation set) |
| False-positive rate | **0.0056** |

| Subtype | Candidates | Accepted as star | Median P(star) |
|---|---|---|---|
| Bright star | 1,380 | 99.9% | 1.000 |
| Faint star | 573 | 98.3% | 0.999 |
| Hot pixel | 607 | 0.0% | 0.000 |
| Elongated | 275 | 0.4% | 0.000 |
| Noise / other | 11 | 36.4% | 0.068 |

The classifier is close to perfect on stars, hot pixels and streaks. The only weak class is "noise / other", which is rare and ambiguous (several of these are star wings).

---

## Experiment B — what did it learn? (`exp_b_filters.png`, `exp_b_probes.png`)

**First-layer filters.** I compared each 5×5 filter with Module 11's zero-mean Gaussian matched filter (σ 1.27 px) using cosine similarity:

| Filter | f0 | f1 | f2 | f3 | f4 | f5 | f6 | f7 |
|---|---|---|---|---|---|---|---|---|
| Cosine similarity | −0.13 | +0.35 | +0.46 | **+0.71** | −0.01 | **+0.72** | −0.44 | −0.16 |

- **f3 and f5 are compact centre-positive blobs that closely resemble the matched filter.**
- f2 is a broader positive lobe.
- f0, f4, f6 and f7 are edge-like or oriented (positive on one side, negative on the other).
- f6 is anti-correlated with the Gaussian: it responds to a bright pixel surrounded by fainter ones.

**Responses by subtype.** Mean peak response near the candidate on test patches:
- Stars drive f2, f3 and f5 hardest. For bright stars f2 reaches 12.8, against 2.6 for hot pixels.
- **Hot pixels** give their *highest* responses in **f6, f1 and f4** (4.9, 4.4 and 3.1, against 1.5, 0.3 and 0.7 for bright stars). These filters effectively detect "a single-pixel spike".
- **Streaks** light up f2, f3 and f5 along their whole length, as the activation maps show. The oriented filters f0, f4 and f6 trace the edges of the line, f4 as a double line along each side. Rejecting a streak therefore seems to happen in the deeper layers, which see the extended shape.

So yes, different filters respond to different artifacts. The first layer is a small bank of matched-filter-like blobs plus spike and edge detectors.

**Morphology or brightness?** I probed with synthetic noisy patches: 200 per point, accepted at the frozen threshold.

- **Single-pixel spikes** are rejected at **every** brightness from 100 to 30,000 e⁻ (acceptance ≤ 1%).
- A **star-shaped source** (σ 1.27) is accepted from about 280 e⁻ (81%), and at ≥ 99% from about 470 e⁻ up.
- So the decision is **not** "bright = star": the same flux in a different shape is rejected.

Acceptance against source width:

| Flux | σ ≤ 0.5 | σ 0.7 | σ 0.9–1.5 | σ 2.2 | σ 2.76 (defocus) | σ 3.3 |
|---|---|---|---|---|---|---|
| 500 e⁻ | 0–1% | 39% | 92–99% | 74% | 51% | 41% |
| 3,000 e⁻ | 0% | 51% | 100% | 100% | 100% | 100% |

- The network has learned a **lower width limit**: sharper than about 0.7 px is not a star, the same physics as the SHARP rule.
- Wide sources are accepted when bright, but less often when faint. That is a dependence on both shape and SNR, and it predicts the defocus behaviour seen in Experiment C.

**The caveat.**
- **11.6% of pure-noise patches** (no source at all) are accepted. The network has seen almost no such patches, because the detector rarely proposes them, so its behaviour there is unconstrained.
- It is safe *only* behind the detector. **The CNN is a candidate classifier, not a detector.**
- Interpretations from 8 filters and a few probes are suggestive, not proof of what the deeper layers compute.

---

## Experiment C — classical vs hybrid vs combined (the main result)

**Setup.**
- Every pipeline receives the **same** candidates from the same detector run.
- **A classical** is Module 11's `detect_sources()`: it fits every candidate, then rejects SHARP (σ < 0.6) and failed fits.
- **A′ classical strict** additionally rejects BAD_FIT (σ > 1.6 × nominal), to show what that rule does.
- **B hybrid** runs the CNN on all patches, rejects P < 0.902, and fits only the candidates it keeps.
- **C combined** is the hybrid followed by the SHARP rule.

**Evaluation.**
- 40 test images per condition.
- Matching of the final catalogue to truth is one-to-one, within 2 px.
- Times are wall-clock per image: same machine, same images, a single process, torch on 1 thread.
- The 95% intervals are bootstraps over images.

### Part 1 — threshold

The threshold P ≥ 0.902 was selected in Experiment A on the validation images, to keep ≥ 98% of proposed true stars. It was frozen and reused unchanged for all three test conditions, including defocus, with no retraining.

### Part 2 — nominal and low SNR

| Condition | Pipeline | Precision | Recall (all stars) [95% CI] | Recall (proposed) | False detections / image | Median / p90 centroid error (px) | ms / image |
|---|---|---|---|---|---|---|---|
| **Nominal** (detector proposes 97.7% of stars) | A classical | 0.866 | 0.978 [0.972, 0.984] | 0.999 | 7.60 | 0.048 / 0.279 | 1,430 |
| | A′ strict | 0.872 | 0.977 | 0.997 | 7.15 | 0.048 / 0.277 | 1,430 |
| | **B hybrid** | **0.997** | 0.971 [0.964, 0.978] | 0.994 | **0.12** | 0.047 / 0.275 | **348** |
| | C combined | 0.997 | 0.971 | 0.993 | 0.12 | 0.047 / 0.274 | 348 |
| **Low SNR** (detector proposes 55.7%) | A classical | 0.824 | 0.556 [0.535, 0.576] | 0.995 | 5.92 | 0.179 / 0.485 | 710 |
| | A′ strict | 0.825 | 0.553 | 0.990 | 5.85 | 0.179 / 0.477 | 710 |
| | **B hybrid** | **0.996** | 0.539 [0.518, 0.558] | 0.965 | **0.10** | 0.177 / 0.443 | **273** |
| | C combined | 0.996 | 0.537 | 0.962 | 0.10 | 0.176 / 0.441 | 273 |

**What I found:**
- **Precision.** The hybrid removes about 98% of the false detections (7.6 → 0.12 per image at nominal). The classical precision here (0.87) is lower than in Module 11 (0.97) **only because this module adds streaks**. The SHARP rule cannot reject them: their cross-section fits a "normal" σ, and the classical rules reject only 1.5% of them. The CNN rejects 99.6% of them.
- **Recall cost.**
  - The hybrid loses **0.7 percentage points** of recall at nominal (CI for the difference −1.0 to −0.4) and **1.7 points** at low SNR (−2.2 to −1.2).
  - These are all faint stars: the CNN rejects 1.7% of proposed faint stars at nominal and 4.3% at low SNR, against 0.3% and 0.6% for the rules.
  - This is the price of the validation-chosen 98% operating point.
  - In exchange, there are 63× (nominal) and 59× (low SNR) fewer false detections.
- **Detection is the bigger limit.** At low SNR, recall against all stars (about 0.55) is set by the **detector**, which proposes only 55.7% of the stars. Both pipelines keep 96–99% of what they are given.
- **Centroid errors are unchanged.** The same fit is used, so the hybrid's slightly lower p90 simply reflects that the stars it drops are the faintest, worst-centroided ones.
- **The combined pipeline (C) adds nothing.** After the CNN, the SHARP rule removes no false detections and only 1–3 faint stars. The CNN has already absorbed that rule.

### Part 3 — strong defocus (FWHM 6.5, no retraining)

| Pipeline | Precision | Recall (all) [95% CI] | Recall (proposed) | False detections / image | Median / p90 centroid error (px) | ms / image |
|---|---|---|---|---|---|---|
| A classical | 0.842 | **0.813** [0.798, 0.827] | **0.996** | 7.65 | 0.129 / 0.670 | 1,492 |
| A′ strict | 0.226 | 0.038 | 0.046 | 6.53 | 0.676 / 1.624 | 1,492 |
| **B hybrid** | **0.996** | 0.756 [0.740, 0.771] | 0.935 | **0.15** | 0.119 / 0.577 | **376** |
| C combined | 0.996 | 0.756 | 0.935 | 0.15 | 0.119 / 0.577 | 376 |

Fraction of proposed true stars that each method rejects:

| Subtype | Classical (SHARP / failed) | Strict (+ BAD_FIT) | CNN |
|---|---|---|---|
| Bright | 0.0% | 99.8% | 1.6% |
| Faint | 0.0% | 85.9% | 15.9% |

- **Does the CNN degrade?** Yes, moderately, and only for faint stars.
  - It rejects 15.9% of proposed faint defocused stars, against 1.7% in focus.
  - Recall against all stars falls 5.7 points below the classical pipeline (CI −6.7 to −4.8).
  - The width probe in Experiment B predicted this: a faint, wide source looks unlike anything in training.
  - Bright defocused stars are still accepted (98.4%), and precision stays at 0.996.
- **Does the hybrid keep more useful candidates than the rules?** It depends which rule.
  - The **SHARP rule alone** is harmless under defocus: defocused stars are *wider*, never sharp. It keeps 99.6% of proposed stars, *more* than the CNN, but with 7.65 false detections per image.
  - The **BAD_FIT width rule** (σ > 1.6σ₀) is catastrophic: it rejects 99.8% of bright defocused stars and recall collapses to 0.038.
  - Against a rule set tuned to nominal focus, the CNN is much more robust. Against the minimal SHARP rule, it trades about 6 points of recall for 50× fewer false detections.
- **Most of the loss happens before classification.** The fixed FWHM-3 matched filter proposes only **80.2%** of the defocused stars (dashed ceiling in `exp_c_pipelines.png`). That 20% is lost before any classifier runs, and neither the CNN nor the rules can recover it.
  - Fix for the detector: a defocus-aware or multi-scale detector.
  - Fix for the CNN: train with a range of PSF widths. Both are out of scope here.
- `exp_c_kept_vs_snr.png` shows the effect is concentrated at true SNR below 15.

### Part 4 — real pipeline cost (classify **before** `gaussian_fit()`)

`hybrid.py` runs these steps in order: detect → extract 32×32 patches → **one batched CNN call** → reject → fit only the retained candidates.

| Condition | Detect | Patches | CNN (batch) | Fits | Candidates | Fitted | **Hybrid total** | **Classical total** |
|---|---|---|---|---|---|---|---|---|
| Nominal | 85 ms | 2.3 ms | 47 ms | 213 ms | 71.2 | 48.7 | **348 ms** | **1,430 ms** (4.1×) |
| Low SNR | 86 | 1.7 | 35 | 150 | 49.2 | 27.0 | **273** | **710** (2.6×) |
| Defocus | 84 | 2.0 | 39 | 251 | 62.6 | 37.9 | **376** | **1,492** (4.0×) |

**The speed-up is bigger than the 30% drop in fit count suggests.** The fits the CNN removes are the expensive ones. I timed `gaussian_fit` alone on 8 nominal images:

| Fit target | Mean time per fit |
|---|---|
| Star | 4–5 ms |
| Streak | 36 ms |
| Hot pixel | 59 ms (the least-squares fit fights a point source as σ → 0 and takes many iterations) |

- The classical pipeline spends most of its time fitting artifacts it then throws away.
- The CNN's own cost (patches plus inference, about 40–50 ms per image on 1 CPU thread) is small next to that.

### Real-image validation — **not completed**

- No labelled real images are available in this course environment.
- Every number above is on simulated images whose noise model, PSF and artifacts match the simulator the CNN was trained on.
- On real data I would expect the following to need checking:
  - CCD-specific artifacts: bleed trails, satellite trails, diffraction spikes, bad columns;
  - non-Gaussian, spatially varying PSFs;
  - blends;
  - the CNN's unconstrained behaviour on unfamiliar patches (Experiment B: 11.6% of pure-noise patches accepted).
- **Next step:** a labelled set of real frames, for example cross-matched against a deeper catalogue, split by frame or by night. Re-check the 98% threshold on it before trusting the reported precision and recall.

## Final deliverable

```python
from hybrid import HybridDetector, hybrid_detect_sources
hd = HybridDetector()                     # loads outputs/cnn_best.pt and the frozen threshold (P >= 0.902)
cat = hd(image, read_noise=5.0)           # optional: reject_flags=FLAG_SHARP, i.e. pipeline C
```

- **Returns** Module 11's catalogue fields: `x, y, flux, snr, sigma, sx, sy, centroid_uncertainty` (with the empirical ×1.118 calibration), `quality_flags` (SHARP / EDGE / CROWDED / LOW_SNR / BAD_FIT, kept as information), `background, background_rms`.
- **Adds** `cnn_score` (P(star)), `n_candidates`, `n_fitted` and a per-stage `timing` dict.
- A demo on test image 30000 prints the five brightest sources (all scores 1.000, flags OK).

## Conclusions

1. A 22k-parameter CNN trained only on detector proposals separates stars from hot pixels and streaks almost perfectly: test AUC 0.9998.
2. End to end, the hybrid cuts false detections per image by about 98% in every condition, and the run is about 2.6–4× faster, because expensive artifact fits are no longer attempted.
3. **The cost is recall on the faintest stars:** −0.7 points (nominal), −1.7 (low SNR) and −5.7 (strong defocus, outside the training distribution).
4. In this setup the remaining SHARP rule adds nothing on top of the CNN.
5. The quality-rule lesson from Module 11 still holds: the width rule BAD_FIT must never be used as a reject rule under defocus.
6. **The biggest limit on recall is detection, not classification:** 55.7% of stars are proposed at low SNR and 80.2% under defocus.
