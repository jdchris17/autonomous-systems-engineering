"""
Work Package A - the Module 1-8 components as one software system.

The integration package must expose every subsystem under one namespace
without copying it: one source for each algorithm, one place where the
module folders are put on the path, no downloads, and a top-level function
that coordinates rather than computes.
"""

import ast
import importlib
from pathlib import Path

import pytest
from astropy.utils import iers
from astropy.utils.data import conf as data_conf

import enduria
from enduria import _legacy

PACKAGE = Path(enduria.__file__).resolve().parent
SUBJECTS = ["timekeeping", "frames", "attitude", "gravity", "geodesy",
            "estimation", "astrometry"]

# (enduria module, name) -> course module that owns the implementation
OWNERS = {
    ("timekeeping", "time_scales"): "astronomical_time",
    ("timekeeping", "reference_matrix"): "compare_reference_frames",
    ("frames", "rotation_vector_to_matrix"): "rotation_engine",
    ("frames", "Transform"): "sensor_frames",
    ("frames", "CameraModel"): "navigation_error_budget",
    ("attitude", "wahba_svd"): "stellar_attitude",
    ("attitude", "robust_wahba"): "stellar_attitude",
    ("gravity", "estimate_up_direction"): "gravity_estimator",
    ("gravity", "assess_measurement_quality"): "gravity_estimator",
    ("geodesy", "earth_fixed_up_to_geodetic"): "celestial_position",
    ("geodesy", "horizontal_position_error"): "celestial_position",
    ("estimation", "batch_position_estimate"): "navigation_estimator",
    ("estimation", "assess_batch_result"): "navigation_estimator",
    ("astrometry", "Atmosphere"): "atmospheric_refraction",
    ("astrometry", "CatalogStar"): "astrometric_corrections",
}


def test_all_eight_module_folders_are_present_and_untouched_names():
    assert [d.name for d in _legacy.MODULE_DIRS] == [f"module-{n}" for n in range(1, 9)]
    assert all(d.is_dir() for d in _legacy.MODULE_DIRS)


@pytest.mark.parametrize("subject", SUBJECTS)
def test_every_subsystem_imports_under_one_namespace(subject):
    module = importlib.import_module(f"enduria.{subject}")
    public = [n for n in vars(module) if not n.startswith("_")]
    assert len(public) >= 4


@pytest.mark.parametrize("subject, name", list(OWNERS))
def test_one_source_of_truth_for_each_algorithm(subject, name):
    exported = getattr(importlib.import_module(f"enduria.{subject}"), name)
    owner = importlib.import_module(OWNERS[(subject, name)])
    assert exported is getattr(owner, name)                 # same object, not a copy
    assert Path(owner.__file__).resolve().parent in _legacy.MODULE_DIRS


def test_no_course_file_was_copied_into_the_package():
    course_files = {p.name for d in _legacy.MODULE_DIRS for p in d.glob("*.py")}
    assert not course_files & {p.name for p in PACKAGE.glob("*.py")}


def test_only_one_file_touches_the_import_path():
    offenders = [p.name for p in PACKAGE.glob("*.py")
                 if "sys.path" in p.read_text(encoding="utf-8")
                 and p.name != "_legacy.py"]
    assert offenders == []


def test_the_pipeline_coordinates_and_does_not_compute():
    """No astronomy library, optimizer or decomposition in the top-level file."""
    tree = ast.parse((PACKAGE / "pipeline.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert imported <= {"numpy", "dataclasses", "traceback"}
    source = (PACKAGE / "pipeline.py").read_text(encoding="utf-8")
    for forbidden in ("least_squares", "erfa.", "np.linalg.svd(", "astropy"):
        # _data_observability takes singular values of Module 7's Jacobian.
        allowed = 1 if forbidden == "np.linalg.svd(" else 0
        assert source.count(forbidden) <= allowed, forbidden


def test_public_interface_is_one_function_and_its_data_types():
    assert callable(enduria.solve_navigation_fix)
    for name in ("NavigationResult", "NavigationConfig", "ObservationBatch",
                 "Observation", "ReferenceData"):
        assert name in enduria.__all__


def test_offline_policy_is_in_force():
    assert iers.conf.auto_download is False
    assert data_conf.allow_internet is False
    described = enduria.ReferenceData({}).describe()
    assert described["downloads"] == "disabled"
