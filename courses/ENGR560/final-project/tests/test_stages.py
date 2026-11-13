"""Each stage tested on its own (no full pipeline needed)."""

import json
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import ROOT

from stellar_vision import CalibrationConfig, DetectionConfig, VisionConfig, stages
from stellar_vision._m11 import FWHM_TO_SIGMA, pixel_integrated_gaussian
from stellar_vision.geometry import (ARCSEC, attitude_from_pointing, pointing_from_attitude, quaternion_from_rotation,
                                     rotation_angle, tan_wcs, wahba_svd)
from stellar_vision.result import AstrometricResult, FailureCode
from stellar_vision.verification import verify


def star_image(x0=200.3, y0=150.7, flux=20000.0, sigma=3.0 * FWHM_TO_SIGMA, seed=0, hot=None):
    rng = np.random.default_rng(seed)
    exp = np.full((512, 512), 100.0) + pixel_integrated_gaussian((512, 512), x0, y0, sigma, flux)
    if hot:
        exp[hot[1], hot[0]] += 4000
    return (rng.poisson(exp) + rng.normal(0, 5, exp.shape)).astype(np.float32)


# ---------------------------------------------------------------------------------------- geometry
def test_projection_roundtrip_and_pixel_convention(camera):
    R = attitude_from_pointing(150.3, 19.2, 33.0)
    assert abs(np.linalg.det(R) - 1) < 1e-12
    u, v, front = camera.project(R, R[2][None])            # boresight -> principal point (zero-based centre)
    assert np.allclose([u[0], v[0]], [255.5, 255.5])
    s = camera.pixel_to_vec([10.0, 400.25], [3.5, 500.0]) @ R
    u2, v2, _ = camera.project(R, s)
    assert np.allclose(u2, [10.0, 400.25], atol=1e-9) and np.allclose(v2, [3.5, 500.0], atol=1e-9)


def test_pointing_and_quaternion_inverse():
    R = attitude_from_pointing(151.7, 21.4, 287.5)
    ra, dec, roll = pointing_from_attitude(R)
    assert np.allclose([ra, dec, roll], [151.7, 21.4, 287.5])
    w, x, y, z = quaternion_from_rotation(R)
    Rq = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                   [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                   [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    assert np.allclose(Rq, R, atol=1e-12)


def test_wahba_recovers_rotation_exactly(catalog):
    R = attitude_from_pointing(150.0, 20.0, 45.0)
    r = catalog.vec[:20]
    assert rotation_angle(wahba_svd(r @ R.T, r), R) < 1e-7
    assert np.linalg.det(wahba_svd(r @ R.T * [1, 1, -1], r)) > 0            # mirrored data still gives a rotation


def test_tan_wcs_matches_pinhole_camera(camera, catalog):
    R = attitude_from_pointing(149.2, 20.9, 123.0)
    w = tan_wcs(R, camera)
    u, v, front = camera.project(R, catalog.vec)
    m = camera.in_frame(u, v, front)
    x, y = w.all_world2pix(catalog.ra[m], catalog.dec[m], 0)                  # astropy origin 0 = our convention
    assert np.max(np.hypot(x - u[m], y - v[m])) < 1e-8
    assert np.allclose(w.wcs.crpix, [256.5, 256.5])                           # FITS 1-based reference pixel


# ------------------------------------------------------------------------------- calibration
def test_calibration_identity_passthrough_and_corrections():
    img = star_image()
    cal = stages.calibrate(img, CalibrationConfig())
    assert cal.data is img and cal.corrections == ()
    cal2 = stages.calibrate(img, CalibrationConfig(gain_e_per_adu=2.0, bias_adu=10.0))
    assert np.allclose(cal2.data, (img.astype(float) - 10) * 2) and abs(cal2.background - 2 * (cal.background - 10)) < 1.0
    with pytest.raises(ValueError):
        stages.calibrate(np.full((4, 4), np.nan), CalibrationConfig())


# --------------------------------------------------------------------------- detection / screening / fit
def test_detection_finds_star_and_ignores_blank(camera):
    cal = stages.calibrate(star_image(), CalibrationConfig())
    c = stages.detect_candidates(cal, DetectionConfig(), camera)
    assert np.min(np.hypot(c.xy[:, 0] - 200.3, c.xy[:, 1] - 150.7)) <= 1.0
    blank = stages.calibrate(star_image(flux=0.0, seed=3), CalibrationConfig())
    assert len(stages.detect_candidates(blank, DetectionConfig(), camera).xy) <= 1


def test_screens_and_centroid_fit(camera):
    img = star_image(hot=(400, 300))
    cal = stages.calibrate(img, CalibrationConfig())
    cand = stages.detect_candidates(cal, DetectionConfig(), camera)
    is_star = np.hypot(cand.xy[:, 0] - 200.3, cand.xy[:, 1] - 150.7) <= 1.5
    is_hot = np.hypot(cand.xy[:, 0] - 400, cand.xy[:, 1] - 300) <= 1.0
    cls = stages.ClassicalScreen()(cal, cand)
    assert cls.keep.all() and np.isnan(cls.score).all()
    cnn = stages.CNNScreen("cnn_best.pt")(cal, cand)
    assert cnn.keep[is_star].all() and not cnn.keep[is_hot].any()
    table = stages.fit_centroids(cal, cand, cls, DetectionConfig(), CalibrationConfig(), camera)
    i_star = np.nonzero(is_star[table.candidate_index])[0][0]
    i_hot = np.nonzero(is_hot[table.candidate_index])[0][0]
    assert np.hypot(table.x[i_star] - 200.3, table.y[i_star] - 150.7) < 0.05
    assert table.quality_flags[i_hot] & stages.FLAGS["SHARP"] and not table.accepted[i_hot]
    assert table.accepted[i_star] and 0 < table.centroid_uncertainty[i_star] < 0.05


def test_adapter_preserves_coordinates_and_fits_convention(tmp_path):
    t = SimpleNamespace(x=np.array([1.25, 7.5, 3.0]), y=np.array([2.0, 4.125, 9.5]), flux=np.array([10.0, 30.0, 20.0]),
                        snr=np.ones(3), centroid_uncertainty=np.array([0.1, 0.2, np.nan]), quality_flags=np.zeros(3, int),
                        cnn_score=np.ones(3), accepted=np.array([True, True, True]))
    src = stages.to_source_list(t)
    assert src["source_row"].tolist() == [1, 2, 0]                     # brightest first
    assert np.array_equal(src["x"], t.x[[1, 2, 0]]) and np.array_equal(src["y"], t.y[[1, 2, 0]])
    assert np.allclose(src["sigma_px"], [0.2 / np.sqrt(2), 0.3, 0.1 / np.sqrt(2)])
    from astropy.io import fits
    stages.write_xyls(src, tmp_path / "s.xyls")
    d = fits.getdata(tmp_path / "s.xyls")
    assert np.array_equal(d["X"], src["x"] + 1.0) and np.array_equal(d["Y"], src["y"] + 1.0)


def test_classical_mode_never_imports_torch():
    code = ("import sys; sys.modules['torch'] = None\n"
            f"sys.path.insert(0, r'{ROOT}'); sys.path.insert(0, r'{ROOT / 'tests'}')\n"
            "import scenarios as S\nfrom stellar_vision import StellarVisionPipeline, VisionConfig\n"
            "r = StellarVisionPipeline.from_config(VisionConfig(detection_mode='classical')).solve(S.nominal().image)\n"
            "print(r.success, r.n_matched)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split()[0] == "True"


# ----------------------------------------------------------------------------------- verification
def _synthetic_best(camera, catalog, R, idx, noise=0.05, seed=0, f=None):
    f = camera.focal_length_px if f is None else f
    rng = np.random.default_rng(seed)
    u, v, _ = camera.project(R, catalog.vec[idx], f)
    src = {"x": u + rng.normal(0, noise, len(u)), "y": v + rng.normal(0, noise, len(u)), "sigma_px": np.full(len(u), noise)}
    best = {"R": R, "f": f, "det_idx": np.arange(len(idx)), "cat_idx": idx, "n_match": len(idx), "n_verify": len(idx) - 3,
            "n_pred": len(idx), "pred_u": u, "pred_v": v, "rms_px": noise}
    return best, src


def test_verification_accepts_good_and_names_each_failure(camera, catalog):
    from stellar_vision import AcceptanceConfig
    acc = AcceptanceConfig()
    R = attitude_from_pointing(150.0, 20.0, 10.0)
    u, v, front = camera.project(R, catalog.vec)
    vis = np.nonzero(camera.in_frame(u, v, front))[0]
    best, src = _synthetic_best(camera, catalog, R, vis)
    ok, code, _, info = verify(best, [dict(best)], src, camera, acc)
    assert ok and 0.3 < info["chi2_red"] < 3
    # too few independent matches
    b2 = dict(best, n_verify=3)
    assert verify(b2, [], src, camera, acc)[1] == FailureCode.INSUFFICIENT_MATCHES
    # residuals 10x larger than the stated sigma
    s3 = dict(src, x=src["x"] + np.random.default_rng(1).normal(0, 0.5, len(vis)))
    assert verify(best, [], s3, camera, acc)[1] == FailureCode.RESIDUALS_INCONSISTENT
    # focal length pinned at the tolerance bound
    b4 = dict(best, f=camera.focal_length_px * 1.03)
    assert verify(b4, [], src, camera, acc)[1] == FailureCode.CAMERA_MODEL_INVALID
    # competing distinct hypothesis with equal support
    rival = {"R": attitude_from_pointing(151.0, 20.0, 10.0), "n_match": best["n_match"], "n_verify": best["n_verify"], "rms_px": 0.1}
    assert verify(best, [rival], src, camera, acc)[1] == FailureCode.AMBIGUOUS
    # tight cluster -> roll unobservable
    cl = vis[np.argsort(np.hypot(u[vis] - 255.5, v[vis] - 255.5))[:3]]
    b5, s5 = _synthetic_best(camera, catalog, R, cl, noise=0.3)
    b5["n_verify"] = 10
    assert verify(b5, [], s5, camera, acc)[1] in (FailureCode.WEAK_GEOMETRY,)


# -------------------------------------------------------------------------------- config / result
def test_config_roundtrip_and_fingerprint(tmp_path):
    cfg = VisionConfig(detection_mode="classical")
    cfg.to_json(tmp_path / "c.json")
    back = VisionConfig.from_json(tmp_path / "c.json")
    assert back.to_dict() == cfg.to_dict() and back.fingerprint() == cfg.fingerprint()
    assert VisionConfig().fingerprint() != cfg.fingerprint()
    with pytest.raises(ValueError):
        VisionConfig(detection_mode="magic")


def test_result_serialises_to_json(default_hybrid):
    import scenarios as S
    r = default_hybrid.solve(S.nominal().image)
    d = json.loads(json.dumps(r.to_dict()))
    assert d["success"] and len(d["attitude"]) == 3 and d["wcs"]["CTYPE1"] == "RA---TAN"
    f = AstrometricResult(success=False, verified=False, failure_code=FailureCode.NO_SOURCES, failure_reason="x")
    assert json.loads(json.dumps(f.to_dict()))["failure_code"] == "NO_SOURCES"
