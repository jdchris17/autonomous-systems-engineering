"""
Freeze the regression reference by running the ORIGINAL Module 11-13 code (not the package).

  * Module 13 Experiment C fields (seeds 200-205): render13 -> module-12 HybridDetector -> adapter13 ->
    solver13.TriangleSolver(f_approx = 1.01 f, sigma 0.2) - exactly the Module 13 configuration;
    also the raw Module 11 candidate list (detect_sources, no CNN, no rejection).
  * Module 13 Experiment B source lists (seeds 100-105, nominal / 20% false / 1 px noise): solver13 only.
Output: tests/reference/m13_reference.json (full-precision floats via repr). Run once; commit the JSON.
torch is limited to 1 thread here, as in the package's deterministic mode.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]          # machine_learning/
for m in ("module-13", "module-12", "module-11"):
    sys.path.insert(0, str(ROOT / m))

import torch  # noqa: E402

torch.set_num_threads(1)

import adapter13  # noqa: E402
import catalog as m11cat  # noqa: E402
import exp_b_identify as m13b  # noqa: E402
import render13  # noqa: E402
import sky13 as s  # noqa: E402
from hybrid import HybridDetector  # noqa: E402
from solver13 import TriangleSolver  # noqa: E402

LOGGED_M13 = {  # from module-13/outputs/run_all_log.txt (Experiment C, hybrid lists, our solver)
    200: {"n_match": 23, "att_arcsec": 4.6}, 201: {"n_match": 19, "att_arcsec": 5.3}, 202: {"n_match": 18, "att_arcsec": 1.7},
    203: {"n_match": 22, "att_arcsec": 0.8}, 204: {"n_match": 28, "att_arcsec": 4.4}, 205: {"n_match": 23, "att_arcsec": 7.4}}


def fl(a):
    return [float(x) for x in np.asarray(a, float).ravel()]


def solve_record(solver, src, R_true):
    r = solver.solve(src["x"], src["y"], order=np.arange(len(src["x"])), sigma_px=src.get("sigma_px"))
    rec = {"success": bool(r["success"]), "n_hypotheses": int(r["n_hypotheses"])}
    if "R" in r:
        rec.update({"R": fl(r["R"]), "f": float(r["f"]), "n_match": int(r["n_match"]), "n_verify": int(r["n_verify"]),
                    "det_idx": [int(i) for i in r["det_idx"]], "cat_id": [int(i) + 1 for i in r["cat_idx"]],
                    "resid_px": fl(r["resid_px"]), "att_arcsec": s.rotation_angle(r["R"], R_true) / s.ARCSEC})
    return rec


def main():
    cat = s.make_catalog()
    hd = HybridDetector()
    solver = TriangleSolver(cat, f_approx=s.F_TRUE * 1.01, sigma_px=0.2)
    ref = {"description": __doc__, "expc": {}, "expb": {}, "logged_module13": LOGGED_M13}
    for sd in range(200, 206):
        rng = np.random.default_rng(sd)
        R, pose = s.random_pointing(rng)
        image, truth = render13.render(cat, R, sd)
        hc = hd(image, read_noise=5.0)
        src = adapter13.to_source_list(hc)
        raw = m11cat.detect_sources(image, read_noise=5.0, uncertainty_scale=1.118)
        raw = {k: v for k, v in raw.items() if isinstance(v, np.ndarray)}
        src_raw = adapter13.to_source_list(raw)
        ref["expc"][sd] = {"R_true": fl(R), "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                           "hybrid": {"n_candidates": int(hc["n_candidates"]), "x": fl(src["x"]), "y": fl(src["y"]),
                                      "sigma_px": fl(src["sigma_px"]), "cnn_score": fl(src["cnn_score"]),
                                      "solve": solve_record(solver, src, R)},
                           "raw": {"x": fl(src_raw["x"]), "y": fl(src_raw["y"]), "sigma_px": fl(src_raw["sigma_px"]),
                                   "solve": solve_record(solver, src_raw, R)}}
        print(sd, ref["expc"][sd]["hybrid"]["solve"]["n_match"], round(ref["expc"][sd]["hybrid"]["solve"]["att_arcsec"], 2))
    solver_b = {}
    for cname in ("nominal (roll 0, sigma 0.1 px)", "20% false detections", "increased noise 1 px"):
        kw = m13b.CONDITIONS[cname]
        sig = kw.get("sigma", 0.1)
        if sig not in solver_b:
            solver_b[sig] = TriangleSolver(cat, f_approx=m13b.F_APPROX, sigma_px=max(0.2, sig))
        for sd in range(100, 106):
            fld = m13b.make_field(cat, sd, **kw)
            ref["expb"][f"{cname}|{sd}"] = {"u": fl(fld["u"]), "v": fl(fld["v"]), "truth": [int(t) for t in fld["truth"]],
                                           "R_true": fl(fld["R"]), "tol_sigma": max(0.2, sig),
                                           "solve": solve_record(solver_b[sig], {"x": fld["u"], "y": fld["v"]}, fld["R"])}
    out = Path(__file__).resolve().parent / "reference" / "m13_reference.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(ref, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
