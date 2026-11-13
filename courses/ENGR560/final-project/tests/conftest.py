import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stellar_vision import StarCatalog, StellarVisionPipeline, VisionConfig  # noqa: E402
from stellar_vision.geometry import CameraModel  # noqa: E402

DATA = ROOT / "stellar_vision" / "data"


@pytest.fixture(scope="session")
def reference():
    return json.loads((Path(__file__).resolve().parent / "reference" / "m13_reference.json").read_text())


@pytest.fixture(scope="session")
def catalog():
    return StarCatalog.load(DATA / "catalog_m13.npz")


@pytest.fixture(scope="session")
def camera():
    return CameraModel.from_json(DATA / "camera_m13.json")


def module13_config(mode="hybrid"):
    """Exactly the Module 13 Experiment C setup: approximate focal length (+1 %), solver tolerance 0.2 px,
    no post-fit rejection (Module 13 passed the hybrid output unfiltered), no polishing."""
    cfg = VisionConfig(detection_mode=mode)
    cam = CameraModel.from_json(DATA / "camera_m13.json")
    cfg.camera = replace(cam, focal_length_px=cam.focal_length_px * 1.01)
    cfg.detection = replace(cfg.detection, post_fit_reject=())
    cfg.solver = replace(cfg.solver, polish_iterations=0)
    return cfg


@pytest.fixture(scope="session")
def m13_hybrid_pipeline():
    return StellarVisionPipeline.from_config(module13_config("hybrid"))


@pytest.fixture(scope="session")
def default_hybrid():
    return StellarVisionPipeline.from_config(VisionConfig(detection_mode="hybrid"))


@pytest.fixture(scope="session")
def default_classical():
    return StellarVisionPipeline.from_config(VisionConfig(detection_mode="classical"))


def arr(x):
    return np.asarray(x, float)
