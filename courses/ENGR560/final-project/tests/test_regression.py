"""Regression: restructuring must not change numerical behaviour (reference frozen from the ORIGINAL
Module 11-13 code by tests/make_reference.py)."""

import ast
import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import ROOT, arr

from stellar_vision import stages
from stellar_vision.geometry import ARCSEC, rotation_angle
from stellar_vision.simulation import random_pointing, render
from stellar_vision.solver import TriangleSolver

ML = ROOT.parent


def _segments(path, names):
    src = Path(path).read_text(encoding="utf-8")
    out = {}
    for node in ast.parse(src).body:
        nm = getattr(node, "name", None)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            nm = node.targets[0].id
        if nm in names:
            out[nm] = ast.get_source_segment(src, node)
    return out


@pytest.mark.parametrize("pkg,orig,names", [
    ("_m11.py", "module-11/starfield.py", ["FWHM_TO_SIGMA", "pixel_integrated_gaussian", "background_stats", "detect",
                                          "gaussian_fit", "match"]),
    ("_m12_cnn.py", "module-12/cnn.py", ["ASINH_A", "StarCNN", "transform", "center_crop"]),
    ("_m12_cnn.py", "module-12/data12.py", ["extract"]),
])
def test_copied_code_is_verbatim(pkg, orig, names):
    a = _segments(ROOT / "stellar_vision" / pkg, names)
    b = _segments(ML / orig, names)
    assert set(a) == set(names) and a == b


def test_checkpoint_is_module12_checkpoint():
    h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    assert h(ROOT / "stellar_vision/data/cnn_best.pt") == h(ML / "module-12/outputs/cnn_best.pt")


@pytest.mark.parametrize("seed", range(200, 206))
def test_module13_expc_hybrid_bit_identical(seed, reference, catalog, m13_hybrid_pipeline):
    ref = reference["expc"][str(seed)]
    R, _ = random_pointing(np.random.default_rng(seed))
    assert np.array_equal(R.ravel(), arr(ref["R_true"]))
    image, _ = render(catalog, R, seed)
    assert hashlib.sha256(image.tobytes()).hexdigest() == ref["image_sha256"]          # same rendered image
    res = m13_hybrid_pipeline.solve(image)
    src = stages.to_source_list(SimpleNamespace(**res.sources))
    h = ref["hybrid"]
    assert res.n_detected == h["n_candidates"]
    for k in ("x", "y", "sigma_px", "cnn_score"):                                       # coordinates untouched
        assert np.array_equal(src[k], arr(h[k])), k
    s = h["solve"]
    assert res.success and res.verified
    assert np.array_equal(res.attitude.ravel(), arr(s["R"]))                             # bit-identical attitude
    assert res.focal_length_px == s["f"]
    assert res.matched_star_ids.tolist() == s["cat_id"]
    assert np.array_equal(res.measured_centroids[:, 0], arr(h["x"])[s["det_idx"]])
    assert res.n_matched == s["n_match"] and res.n_verify == s["n_verify"]
    logged = reference["logged_module13"][str(seed)]                                     # Module 13 report numbers
    assert res.n_matched == logged["n_match"]
    assert abs(rotation_angle(res.attitude, R) / ARCSEC - logged["att_arcsec"]) < 0.05


@pytest.mark.parametrize("seed", range(200, 206))
def test_module13_raw_candidates_classical(seed, reference, catalog):
    """Classical mode without quality rules = Module 13's raw Module 11 candidate list."""
    from conftest import module13_config
    from stellar_vision import StellarVisionPipeline
    ref = reference["expc"][str(seed)]["raw"]
    R, _ = random_pointing(np.random.default_rng(seed))
    image, _ = render(catalog, R, seed)
    pipe = StellarVisionPipeline.from_config(module13_config("classical"), catalog=catalog)
    res = pipe.solve(image)
    src = stages.to_source_list(SimpleNamespace(**res.sources))
    assert np.array_equal(src["x"], arr(ref["x"])) and np.array_equal(src["y"], arr(ref["y"]))
    assert np.array_equal(res.attitude.ravel(), arr(ref["solve"]["R"]))
    assert res.matched_star_ids.tolist() == ref["solve"]["cat_id"]


def test_module13_expb_solver_bit_identical(reference, catalog, camera):
    solvers = {}
    for key, rec in reference["expb"].items():
        tol = rec["tol_sigma"]
        if tol not in solvers:
            cam = replace(camera, focal_length_px=camera.focal_length_px * 1.01)
            solvers[tol] = TriangleSolver(catalog, cam, (150.0, 20.0), 2.5, 0.03, tol)
        out = solvers[tol].solve(arr(rec["u"]), arr(rec["v"]))
        b, s = out["best"], rec["solve"]
        assert out["n_hypotheses"] == s["n_hypotheses"], key
        assert np.array_equal(b["R"].ravel(), arr(s["R"])), key
        assert b["f"] == s["f"] and b["det_idx"].tolist() == s["det_idx"], key
        assert (b["cat_idx"] + 1).tolist() == s["cat_id"], key
