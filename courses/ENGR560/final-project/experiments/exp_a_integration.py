"""
Experiment A - integration and reproducibility.

One end-to-end execution through the public API exactly as proposed (FITS image + camera JSON on disk),
then the regression suite against the frozen Module 13 reference, and a repeat-run determinism check.
"""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs"

from stellar_vision import StellarVisionPipeline, VisionConfig  # noqa: E402
from stellar_vision.geometry import ARCSEC, rotation_angle  # noqa: E402
from stellar_vision.simulation import random_pointing, render  # noqa: E402


def main():
    OUT.mkdir(exist_ok=True)
    from astropy.io import fits
    seed = 200                                                   # Module 13 Experiment C field
    config = VisionConfig(detection_mode="hybrid", verify_solution=True, offline=True)
    pipeline = StellarVisionPipeline.from_config(config)
    R_true, pose = random_pointing(np.random.default_rng(seed))
    image, truth = render(pipeline.catalog, R_true, seed)
    fits.PrimaryHDU(image).writeto(OUT / "night_sky.fits", overwrite=True)
    (OUT / "camera_config.json").write_text(json.dumps(config.camera.to_dict(), indent=2))
    config.to_json(OUT / "vision_config.json")

    result = pipeline.solve(image=str(OUT / "night_sky.fits"), camera=str(OUT / "camera_config.json"))
    print("Experiment A: one end-to-end execution through stellar_vision (Module 13 field, seed 200)")
    print(f"  config fingerprint {result.diagnostics['config_fingerprint']}, mode {config.detection_mode},"
          f" calibration {config.calibration.version}, catalogue {pipeline.catalog.name} ({len(pipeline.catalog)} stars)")
    if result.success:
        ra, dec = result.boresight_radec_deg
        s = result.attitude_sigma_arcsec
        print(f"  VERIFIED: boresight RA {ra:.5f} Dec {dec:+.5f} roll {result.roll_deg:.4f} deg"
              f" (truth {pose[0]:.5f} {pose[1]:+.5f} {pose[2]:.4f})")
        print(f"  attitude error vs truth {rotation_angle(result.attitude, R_true) / ARCSEC:.2f}\";"
              f" 1-sigma cross-boresight {s['cross_x']:.2f}\" / {s['cross_y']:.2f}\", roll {s['roll']:.2f}\"")
        print(f"  quaternion (w,x,y,z) {np.round(result.quaternion, 8).tolist()}")
        print(f"  {result.n_detected} candidates -> {result.n_sources} sources -> {result.n_matched} matched"
              f" ({result.n_verify} beyond the hypothesis), catalogue fraction {result.catalog_match_fraction:.2f},"
              f" reprojection RMS {result.reprojection_rms_px:.3f} px, reduced chi2 {result.diagnostics['reduced_chi2']:.2f}")
        print("  criteria: " + "; ".join(f"{c['name']} {c['value']} ({'ok' if c['passed'] else 'FAIL'})"
                                         for c in result.diagnostics["criteria"]))
        print("  stage times (ms): " + ", ".join(f"{k} {1000 * v:.0f}" for k, v in result.diagnostics["timing_s"].items()))
        print(f"  WCS: CRVAL {np.round(result.wcs.wcs.crval, 6).tolist()}, CRPIX {result.wcs.wcs.crpix.tolist()} (1-based)")
    json.dump(result.to_dict(), open(OUT / "exp_a_result.json", "w"), indent=1)
    print("  serialised result ->", OUT / "exp_a_result.json")

    # repeat-run determinism, both modes, fresh pipelines
    for mode in ("classical", "hybrid"):
        a = StellarVisionPipeline.from_config(VisionConfig(detection_mode=mode)).solve(image)
        b = StellarVisionPipeline.from_config(VisionConfig(detection_mode=mode)).solve(image.copy())
        same = (a.attitude.tobytes() == b.attitude.tobytes() and a.matched_star_ids.tolist() == b.matched_star_ids.tolist()
                and a.residuals_px.tobytes() == b.residuals_px.tobytes())
        print(f"  repeat run, {mode:<9}: bit-identical {same}  (attitude error {rotation_angle(a.attitude, R_true) / ARCSEC:.2f}\","
              f" {a.n_matched} matches)")

    # regression suite against the Module 13 reference
    print("\n  regression / unit / negative test suite (pytest):")
    p = subprocess.run([sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q", "-p", "no:cacheprovider"],
                       capture_output=True, text=True, cwd=ROOT)
    print("    " + p.stdout.strip().splitlines()[-1])
    ref = json.loads((ROOT / "tests/reference/m13_reference.json").read_text())
    print("\n  Module 13 Experiment C reproduced through the package (Module 13 settings: f +1%, no rules, no polish):")
    print(f"    {'seed':<6}{'M13 matches':>12}{'pkg matches':>12}{'M13 att':>9}{'pkg att':>9}{'attitude bits':>15}")
    sys.path.insert(0, str(ROOT / "tests"))
    from conftest import module13_config
    pm13 = StellarVisionPipeline.from_config(module13_config("hybrid"))
    for sd in range(200, 206):
        R, _ = random_pointing(np.random.default_rng(sd))
        img, _ = render(pm13.catalog, R, sd)
        r = pm13.solve(img)
        rs = ref["expc"][str(sd)]["hybrid"]["solve"]
        print(f"    {sd:<6}{rs['n_match']:>12}{r.n_matched:>12}{rs['att_arcsec']:>8.2f}\"{rotation_angle(r.attitude, R) / ARCSEC:>8.2f}\""
              f"{'identical' if np.array_equal(r.attitude.ravel(), np.asarray(rs['R'])) else 'DIFFERENT':>15}")


if __name__ == "__main__":
    main()
