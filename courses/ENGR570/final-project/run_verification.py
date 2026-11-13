"""
run_verification.py - Deliverable C: the consolidated verification suite.

Runs every automated test in the course, grouped by verification level, and
writes verification_report.txt: what ran, what passed, with which software
versions, seeds and tolerances, and - kept separate from the passing
tests - what the system is known NOT to be able to do.

    python run_verification.py                     everything (about 10 minutes)
    python run_verification.py --skip-statistical  leave out the Monte Carlo suite

The number of passing tests is evidence that the software does what its
models say. It is not evidence of real-world navigation accuracy: no test
here uses a real image, a real IMU or a real sky.
"""

import argparse
import json
import platform
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ElementTree
from pathlib import Path

HERE = Path(__file__).resolve().parent
COURSE = HERE.parent
TESTS = HERE / "tests"
REPORT = HERE / "verification_report.txt"
STATISTICS = HERE / "verification_statistics.json"
sys.path.insert(0, str(TESTS))
sys.path.insert(0, str(HERE))

# level, purpose, [(label, working directory, pytest target)]
LEVELS = [
    ("Unit", "individual algorithms",
     [(f"Module {n}", COURSE / f"module-{n}", ".") for n in range(1, 9)]),
    ("Integration", "subsystems as one package; the result contract",
     [("Work Package A: architecture", HERE, "tests/test_architecture.py"),
      ("Work Package B: pipeline contract", HERE, "tests/test_pipeline_contract.py")]),
    ("System", "complete navigation operation",
     [("Suite 1: global geographic validation", HERE, "tests/test_suite1_global.py")]),
    ("Independent reference", "detect shared modelling errors",
     [("Suite 2: independent astronomical verification", HERE,
       "tests/test_suite2_independent.py")]),
    ("Robustness", "degraded and invalid conditions",
     [("Suite 3: calibration and orientation invariance", HERE,
       "tests/test_suite3_invariance.py"),
      ("Suite 4: navigation failure modes", HERE,
       "tests/test_suite4_failure_modes.py")]),
    ("Statistical", "error and uncertainty claims",
     [("Suite 5: statistical calibration", HERE, "tests/test_suite5_statistics.py")]),
]

KNOWN_LIMITATIONS = [
    ("Constant acceleration is invisible.",
     "A steady horizontal acceleration is indistinguishable from a tilt. Every "
     "check passes and the fix is wrong by about 650 m per mm/s^2. The result "
     "stays VALID and carries the limitation in `assumptions`, "
     "`integrity.unverified_assumptions` and `accuracy_limitations`. "
     "Demonstrated by suite 4, test_constant_horizontal_acceleration_is_not_"
     "detected_and_not_denied."),
    ("A wrong star match inside the centroid noise is invisible.",
     "A catalog identity error smaller than about 3.7 sigma of the centroid "
     "noise passes residual screening. Demonstrated by suite 4, "
     "test_a_mismatch_inside_the_noise_cannot_be_detected."),
    ("A fixed calibration error is not observable at one attitude.",
     "The engine detects this, withholds the calibration estimate and reports "
     "the declared calibration uncertainty in full (suite 3). It cannot detect "
     "a calibration error LARGER than declared."),
    ("Declared uncertainties are taken on trust.",
     "Coverage is honest only when the declared sigmas describe the real "
     "errors. Suite 5 shows coverage collapsing to a few percent when the same "
     "errors are present but not declared."),
    ("A camera focal-length error has no declared-uncertainty parameter.",
     "It shows up in the star residuals and degrades the status, but it is not "
     "in the covariance."),
    ("Refraction is a standard-atmosphere model.",
     "Temperature inversions and horizontal gradients are not modelled. Stars "
     "below 3 degrees altitude are not used."),
    ("The independent reference path is not fully independent.",
     "Astropy assembles its frames differently but calls the same ERFA "
     "routines and reads the same bundled IERS table. Suite 2 detects errors "
     "in how the engine arranges the physics, not errors inside ERFA or in the "
     "Earth-orientation data."),
    ("Earth-orientation prediction uncertainty uses a recalled formula.",
     "The growth law attributed to IERS Bulletin A (Module 8) has not been "
     "checked against the bulletin."),
    ("No real data.",
     "Star detection, centroiding and catalog matching happen before this "
     "interface (ENGR 560). No real image, IMU record or site has been used. "
     "All error magnitudes are illustrative assumptions."),
]


def run_pytest(directory, target, extra=()):
    """Run pytest in a subprocess. Returns a dict of counts and names."""
    with tempfile.TemporaryDirectory() as scratch:
        xml_path = Path(scratch) / "results.xml"
        started = time.time()
        process = subprocess.run(
            [sys.executable, "-m", "pytest", target, "-q", "-p", "no:cacheprovider",
             f"--junitxml={xml_path}", *extra],
            cwd=directory, capture_output=True, text=True)
        seconds = time.time() - started
        outcome = {"tests": 0, "failed": 0, "errors": 0, "skipped": 0,
                   "seconds": seconds, "failures": [], "returncode": process.returncode}
        if xml_path.exists():
            root = ElementTree.parse(xml_path).getroot()
            for case in root.iter("testcase"):
                outcome["tests"] += 1
                name = f"{case.get('classname')}::{case.get('name')}"
                if case.find("failure") is not None:
                    outcome["failed"] += 1
                    outcome["failures"].append(name)
                elif case.find("error") is not None:
                    outcome["errors"] += 1
                    outcome["failures"].append(name)
                elif case.find("skipped") is not None:
                    outcome["skipped"] += 1
        else:
            outcome["errors"] = 1
            outcome["failures"].append(process.stdout[-400:] + process.stderr[-400:])
        outcome["passed"] = (outcome["tests"] - outcome["failed"] - outcome["errors"]
                             - outcome["skipped"])
        return outcome


def collect_known_limitation_tests():
    process = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-m", "known_limitation",
         "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=HERE, capture_output=True, text=True)
    return [line.strip() for line in process.stdout.splitlines() if "::" in line]


def versions():
    import astropy
    import astropy_iers_data
    import erfa
    import geographiclib
    import numpy
    import pytest
    import scipy

    import enduria
    return {"enduria": enduria.__version__, "python": platform.python_version(),
            "platform": platform.platform(), "numpy": numpy.__version__,
            "scipy": scipy.__version__, "astropy": astropy.__version__,
            "pyerfa": erfa.__version__,
            "astropy-iers-data": astropy_iers_data.__version__,
            "geographiclib": geographiclib.__version__, "pytest": pytest.__version__}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--skip-statistical", action="store_true")
    args = parser.parse_args()
    import tolerances

    lines = []

    def out(text=""):
        lines.append(text)
        print(text, flush=True)

    out("ENDURIA NAVIGATION ENGINE - CONSOLIDATED VERIFICATION REPORT")
    out("=" * 76)
    out(time.strftime("Generated %Y-%m-%d %H:%M:%S local time"))
    out()
    out("SOFTWARE VERSIONS")
    for name, value in versions().items():
        out(f"  {name:<18} {value}")
    out("  Earth-orientation and leap-second data: bundled tables, downloads disabled")
    out()
    out("REPRODUCIBILITY")
    out(f"  Base seed {tolerances.SEED}; each test adds a fixed offset "
        "(numpy.random.default_rng).")
    out(f"  Monte Carlo trials per population: {tolerances.MONTE_CARLO_TRIALS} "
        "(environment variable ENDURIA_TRIALS).")
    out()
    out("1. PASSING TESTS BY VERIFICATION LEVEL")
    out("-" * 76)
    out(f"{'Level':<22} {'Test set':<48} passed  failed  time")
    totals = {"tests": 0, "passed": 0, "failed": 0}
    failures = []
    for level, purpose, sets in LEVELS:
        for label, directory, target in sets:
            if level == "Statistical" and args.skip_statistical:
                out(f"{level:<22} {label:<48} NOT RUN in this invocation")
                continue
            outcome = run_pytest(directory, target)
            bad = outcome["failed"] + outcome["errors"]
            totals["tests"] += outcome["tests"]
            totals["passed"] += outcome["passed"]
            totals["failed"] += bad
            failures += [f"{label}: {name}" for name in outcome["failures"]]
            out(f"{level:<22} {label:<48} {outcome['passed']:>6}  {bad:>6}  "
                f"{outcome['seconds']:>4.0f} s")
        out(f"{'':<22} purpose: {purpose}")
    out("-" * 76)
    out(f"{'TOTAL':<22} {'':<48} {totals['passed']:>6}  {totals['failed']:>6}")
    if failures:
        out()
        out("FAILURES")
        for failure in failures:
            out(f"  {failure}")
    out()
    limitation_tests = collect_known_limitation_tests()
    out(f"Of the passing tests, {len(limitation_tests)} are marked "
        "`known_limitation`: they pass when the software")
    out("correctly declines to claim something it cannot know. They document "
        "limits, not capability:")
    for name in limitation_tests:
        out(f"  {name}")
    out()

    out("2. TOLERANCES")
    out("-" * 76)
    for name, (value, reason) in tolerances.TOLERANCES.items():
        out(f"  {name} = {value:g}")
        words, line = reason.split(), "     "
        for word in words:
            if len(line) + len(word) > 76:
                out(line.rstrip())
                line = "     "
            line += word + " "
        out(line.rstrip())
    out()

    out("3. STATISTICAL CALIBRATION (suite 5)")
    out("-" * 76)
    if STATISTICS.exists():
        statistics = json.loads(STATISTICS.read_text(encoding="utf-8"))
        if args.skip_statistical:
            out("  (numbers from the last run of suite 5; the suite was not re-run now)")
        out(f"  {'Population':<40} fixes   median     RMS    95th  reported  "
            "NEES   95% cover (allowed)")
        for summary in statistics.values():
            out(f"  {summary['name']:<40} {summary['fixes']:>5} "
                f"{summary['median_m']:>8,.0f} {summary['rms_m']:>7,.0f} "
                f"{summary['p95_m']:>7,.0f} {summary['reported_rms_m']:>9,.0f} "
                f"{summary['mean_nees']:>6.2f}   {100 * summary['coverage95']:5.1f}% "
                f"(95 +/- {100 * summary['coverage_tolerance']:.1f})")
        out("  Distances in metres. 'reported' is the RMS of the engine's own "
            "1-sigma. Theory: NEES 2.00, cover 95 %.")
    else:
        out("  Suite 5 has not been run.")
    out()

    out("4. KNOWN LIMITATIONS (not failures; things the system cannot do)")
    out("-" * 76)
    for number, (title, text) in enumerate(KNOWN_LIMITATIONS, 1):
        out(f"  {number}. {title}")
        words, line = text.split(), "     "
        for word in words:
            if len(line) + len(word) > 76:
                out(line.rstrip())
                line = "     "
            line += word + " "
        out(line.rstrip())
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved {REPORT.name}")
    return 1 if totals["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
