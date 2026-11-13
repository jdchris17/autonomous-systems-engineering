# Enduria navigation engine — ENGR 570 capstone

A terrestrial celestial fix from matched star centroids and accelerometer
data, through one public function:

```python
from enduria import solve_navigation_fix

result = solve_navigation_fix(observation_batch, config, references)
print(result.summary())
```

`final-project/` integrates the code of Modules 1–8. The module folders are
not copied or changed: they remain the reference implementations and the
only source of each algorithm.

## Layout

| Path | What it is |
|---|---|
| `enduria/_legacy.py` | The one place the module folders are put on the import path; applies the offline policy. |
| `enduria/timekeeping.py`, `frames.py`, `attitude.py`, `gravity.py`, `geodesy.py`, `estimation.py`, `astrometry.py` | Each subsystem of Modules 1–8 re-exported under one namespace. |
| `enduria/contracts.py` | `ObservationBatch`, `NavigationConfig`, `NavigationResult`. |
| `enduria/references.py` | `ReferenceData`: star catalog and Earth-orientation data, with provenance and age. |
| `enduria/pipeline.py` | `solve_navigation_fix`: stages 1–5. |
| `enduria/integrity.py` | Stage 6: status, composed uncertainty, assumptions. |
| `enduria/simulation.py` | Truth model that produces observation batches. |
| `enduria/qualification.py` | Operational scenarios and the Monte Carlo runner. |
| `tests/` | System verification suites 1–5, architecture and contract tests, the independent (Astropy) reference path, and the table of tolerances. |
| `run_verification.py` | Runs every test in the course by verification level; writes `verification_report.txt`. |
| `run_qualification.py` | Scenarios A–F and the geographic sweep; writes `qualification_report.txt` and figures. |

## Pipeline

1. **Ingest** — validate each exposure; set aside what cannot be used.
2. **Reference directions** — catalog to observation epoch, topocentric apparent place, refraction (ERFA).
3. **Camera attitude** — Wahba/SVD with outlier rejection, RANSAC fallback (Module 4).
4. **Local vertical** — calibrated accelerometer mean and quality checks (Module 5), carried into celestial axes.
5. **Position** — batch estimator with shared-error covariance and optional calibration or clock states (Modules 6–7).
6. **Integrity** — acceptance policy, uncertainty composition, assumptions.

Stages 2–5 repeat until the position stops moving, because the reference
directions depend on where the observer is.

## Result status

| Status | Meaning |
|---|---|
| `VALID` | Every available check passed. Conditional on `result.assumptions`. |
| `DEGRADED` | A position is reported, with a stated reason to trust it less. |
| `UNOBSERVABLE` | The measurements do not determine the position. No position reported. |
| `REJECTED` | A validity check failed. No position reported. |

The function does not raise on bad data. A position never appears without
its covariance, and `VALID` never means "the instrument was not
accelerating": that cannot be checked from these sensors.

## Running

```bash
python -m pytest -q -p no:cacheprovider
```

```bash
python run_verification.py
```

```bash
python run_qualification.py
```

The statistical suite takes several minutes; `ENDURIA_TRIALS` sets its
trial count (default 500). The qualification Monte Carlo is cached by stage
in `qualification_cache.json`; `--fresh` recomputes all of it (about 20
minutes).

All error magnitudes are illustrative assumptions, not sensor
specifications. Nothing is downloaded: Earth-orientation and leap-second
data come from the tables bundled with Astropy.
