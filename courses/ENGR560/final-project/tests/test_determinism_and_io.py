"""Repeated runs give identical results; NumPy and FITS inputs give identical results."""

import numpy as np
import pytest

import scenarios as S
from stellar_vision import StellarVisionPipeline, VisionConfig


def _key(r):
    return (r.success, r.attitude.tobytes(), r.matched_star_ids.tolist(), r.residuals_px.tobytes(),
            r.measured_centroids.tobytes(), r.focal_length_px, r.attitude_covariance.tobytes())


@pytest.mark.parametrize("mode", ["classical", "hybrid"])
def test_repeated_runs_identical(mode):
    img = S.nominal(seed=203).image
    a = StellarVisionPipeline.from_config(VisionConfig(detection_mode=mode)).solve(img)
    b = StellarVisionPipeline.from_config(VisionConfig(detection_mode=mode)).solve(img.copy())
    assert a.success and _key(a) == _key(b)


def test_numpy_and_fits_inputs_identical(tmp_path, default_hybrid):
    from astropy.io import fits
    img = S.nominal(seed=204).image
    fits.PrimaryHDU(img).writeto(tmp_path / "night_sky.fits")
    a = default_hybrid.solve(img)
    b = default_hybrid.solve(str(tmp_path / "night_sky.fits"), camera=str(S.ROOT / "stellar_vision/data/camera_m13.json"))
    assert _key(a) == _key(b)


def test_invalid_input_is_reported_not_raised(default_hybrid):
    r = default_hybrid.solve(np.zeros((100, 100), np.float32))
    assert not r.success and r.failure_code.value == "INVALID_INPUT"
