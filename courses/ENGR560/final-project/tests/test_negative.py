"""Experiment B suite as tests: controlled positives must verify; negative controls must be rejected with
an acceptable, specific failure code - never a manufactured attitude."""

import numpy as np
import pytest

import scenarios as S
from stellar_vision import StellarVisionPipeline, VisionConfig
from stellar_vision.geometry import ARCSEC, rotation_angle


def run(sc, mode):
    pipe = StellarVisionPipeline.from_config(VisionConfig(detection_mode=mode), catalog=sc.catalog)
    return pipe.solve(sc.image, camera=sc.camera) if sc.image is not None else pipe.solve_sources(sc.sources, sc.camera)


@pytest.mark.parametrize("mode", ["classical", "hybrid"])
@pytest.mark.parametrize("builder", S.ALL_POSITIVE, ids=lambda b: b.__name__)
def test_positive_controls_verify(builder, mode):
    sc = builder()
    r = run(sc, mode)
    assert r.success and r.verified and r.failure_code is None
    assert rotation_angle(r.attitude, sc.R_true) < 30 * ARCSEC
    assert r.wcs is not None and r.attitude_covariance.shape == (3, 3)
    assert r.attitude_sigma_arcsec["roll"] > r.attitude_sigma_arcsec["cross_x"]       # roll is the weak axis


@pytest.mark.parametrize("mode", ["classical", "hybrid"])
@pytest.mark.parametrize("builder", S.ALL_NEGATIVE, ids=lambda b: b.__name__)
def test_negative_controls_rejected(builder, mode):
    sc = builder()
    r = run(sc, mode)
    assert not r.success and not r.verified
    assert r.failure_code in sc.acceptable_codes, (r.failure_code, r.failure_reason)
    assert r.failure_reason and r.attitude is None and r.wcs is None


def test_verify_disabled_returns_unverified_hypothesis():
    sc = S.ambiguous_twin()
    cfg = VisionConfig(detection_mode="hybrid", verify_solution=False)
    r = StellarVisionPipeline.from_config(cfg, catalog=sc.catalog).solve(sc.image)
    assert r.success and not r.verified and np.isfinite(r.attitude).all()
