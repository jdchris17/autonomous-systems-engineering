"""
Experiment C - final classical vs hybrid pipeline comparison (complete systems).

Configuration A (classical): calibration, matched filter, PSF fit, classical quality rules (SHARP, failed
fit), identification, attitude, verification.  Configuration B (hybrid): identical, plus Module 12 CNN
rejection BEFORE the PSF fit. Same images, camera model, catalogue and acceptance rules.

Each configuration runs in its own worker process (python exp_c_comparison.py --worker MODE) so that peak
memory is measured per configuration: (1) tracemalloc peak during a solve (Python/NumPy heap; torch's
native allocator is NOT included), (2) process peak working set (whole process, incl. libraries).
"""

import json
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = ROOT / "outputs"

from conditions import CONDITIONS, config_for, make_field, truth_labels  # noqa: E402
from stellar_vision import StellarVisionPipeline  # noqa: E402
from stellar_vision.geometry import ARCSEC, rotation_angle  # noqa: E402

STAGES = ("calibration", "detection", "classification", "centroiding", "identification", "verification", "total")


def peak_working_set_mb():
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t), ("a", ctypes.c_size_t), ("b", ctypes.c_size_t),
                        ("c", ctypes.c_size_t), ("d", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        k32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            raise OSError(ctypes.get_last_error())
        return pmc.PeakWorkingSetSize / 2**20
    except Exception:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def evaluate(r, truth, R_true, cat):
    """Per-field metrics against truth."""
    lab = truth_labels(r, truth)
    cand = r.diagnostics.get("candidates", {})
    rec = {"success": bool(r.success), "code": None if r.success else r.failure_code.value,
           "n_candidates": int(r.n_detected), "n_sources": int(r.n_sources),
           "n_cat_in_frame": int(np.sum(truth["cat_idx"] >= 0)),
           "useful_sources": int(np.sum((lab >= 0) & np.asarray(r.sources["accepted"]))) if r.sources else 0,
           "timing": {k: float(v) for k, v in r.diagnostics["timing_s"].items()}}
    # true catalogue stars among candidates discarded BEFORE fitting (CNN) or by the quality rules
    if r.sources is not None and len(cand.get("x", [])):
        from stellar_vision._m11 import match
        cx = np.column_stack([cand["x"], cand["y"]])
        pairs, _ = match(cx, np.column_stack([truth["x"], truth["y"]]), 2.0)
        star_cand = np.zeros(len(cx), bool)
        star_cand[[p for p, t in pairs if truth["cat_idx"][t] >= 0]] = True
        rec["stars_screen_rejected"] = int(np.sum(star_cand & ~np.asarray(cand["kept_by_screen"])))
        rec["stars_rule_rejected"] = int(np.sum((lab >= 0) & ~np.asarray(r.sources["accepted"])))
        rec["artifacts_to_solver"] = int(np.sum((lab == -2) & np.asarray(r.sources["accepted"])))
    if r.success:
        rows = np.asarray(r.matched_source_index)
        assigned = np.searchsorted(cat.id, r.matched_star_ids)
        tl = lab[rows]
        good = tl == assigned
        dx = np.asarray(r.sources["x"])[rows][good]
        dy = np.asarray(r.sources["y"])[rows][good]
        tx = np.array([truth["x"][np.nonzero(truth["cat_idx"] == a)[0][0]] for a in assigned[good]]) if good.any() else np.zeros(0)
        ty = np.array([truth["y"][np.nonzero(truth["cat_idx"] == a)[0][0]] for a in assigned[good]]) if good.any() else np.zeros(0)
        rec.update({"n_matched": int(r.n_matched), "correct_ids": int(good.sum()), "wrong_ids": int((~good).sum()),
                    "wrong_kinds": [int(k) for k in tl[~good]], "centroid_err": np.hypot(dx - tx, dy - ty).tolist(),
                    "rms": float(r.reprojection_rms_px), "att": rotation_angle(r.attitude, R_true) / ARCSEC,
                    "sigma_roll": r.attitude_sigma_arcsec["roll"]})
    return rec


def worker(mode, out_path):
    base = peak_working_set_mb()
    results = {}
    pipes = {}
    for cname, c in CONDITIONS.items():
        for sd in c["seeds"]:
            image, truth, R, cat, expect = make_field(cname, sd)
            key = (cname, cat.name)
            if key not in pipes:
                pipes[key] = StellarVisionPipeline.from_config(config_for(mode, cname), catalog=cat)
                pipes[key].solve(image)                      # warm-up: catalogue index build + lazy imports
            r = pipes[key].solve(image)
            rec = evaluate(r, truth, R, cat)
            rec["expect"] = expect
            results.setdefault(cname, []).append(rec)
    # memory: tracemalloc during two nominal solves (separate pass - tracemalloc slows execution)
    pipe = pipes[("nominal", "module13-synthetic-seed13")]
    peaks = []
    for sd in (200, 201):
        image, *_ = make_field("nominal", sd)
        tracemalloc.start()
        pipe.solve(image)
        peaks.append(tracemalloc.get_traced_memory()[1] / 2**20)
        tracemalloc.stop()
    json.dump({"results": results, "tracemalloc_peak_mb": max(peaks), "process_peak_mb": peak_working_set_mb(),
               "process_peak_after_imports_mb": base}, open(out_path, "w"))


def summarise(res):
    rows = {}
    for mode in ("classical", "hybrid"):
        R = res[mode]["results"]
        for cname, recs in R.items():
            exp = recs[0]["expect"]
            s = [r for r in recs if r["success"]]
            ok_correct = [r for r in s if r["att"] < 60]
            d = {"n": len(recs), "solved": len(s), "false_accept": sum(1 for r in s if not r["expect"] or r["att"] >= 60),
                 "correct_ids": sum(r.get("correct_ids", 0) for r in s), "wrong_ids": sum(r.get("wrong_ids", 0) for r in s),
                 "cat_in_frame": sum(r["n_cat_in_frame"] for r in recs if r["expect"]),
                 "useful": np.mean([r["useful_sources"] for r in recs]), "cand": np.mean([r["n_candidates"] for r in recs]),
                 "sources": np.mean([r["n_sources"] for r in recs]),
                 "stars_screen_rej": sum(r.get("stars_screen_rejected", 0) for r in recs),
                 "stars_rule_rej": sum(r.get("stars_rule_rejected", 0) for r in recs),
                 "artifacts": np.mean([r.get("artifacts_to_solver", 0) for r in recs]),
                 "cent": np.median(sum((r["centroid_err"] for r in s), [])) if s else np.nan,
                 "rms": np.sqrt(np.mean([r["rms"] ** 2 for r in s])) if s else np.nan,
                 "att_med": np.median([r["att"] for r in ok_correct]) if ok_correct else np.nan,
                 "att_max": np.max([r["att"] for r in ok_correct]) if ok_correct else np.nan,
                 "time": {k: np.mean([r["timing"].get(k, 0.0) for r in recs]) for k in STAGES},
                 "codes": sorted({r["code"] for r in recs if r["code"]}), "expect": exp}
            rows[(cname, mode)] = d
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    res = {}
    for mode in ("classical", "hybrid"):
        p = OUT / f"exp_c_worker_{mode}.json"
        t0 = time.perf_counter()
        pr = subprocess.run([sys.executable, __file__, "--worker", mode, str(p)], capture_output=True, text=True)
        if pr.returncode:
            raise RuntimeError(pr.stderr)
        res[mode] = json.loads(p.read_text())
        print(f"  worker {mode}: {time.perf_counter() - t0:.0f} s")
    rows = summarise(res)
    print("\nExperiment C: classical vs hybrid complete systems (6 fields per condition, same images / camera / catalogue / rules)")
    hdr = (f"  {'condition':<24}{'mode':<10}{'solved':>7}{'false acc':>10}{'correct IDs':>12}{'wrong':>6}{'useful src':>11}"
           f"{'artifacts':>10}{'stars lost':>11}{'cent err':>9}{'RMS':>7}{'att med':>8}{'att max':>8}{'ms/img':>8}")
    print(hdr)
    for cname in CONDITIONS:
        for mode in ("classical", "hybrid"):
            d = rows[(cname, mode)]
            lost = d["stars_screen_rej"] + d["stars_rule_rej"]
            cid = f"{d['correct_ids']}/{d['cat_in_frame']}" if d["expect"] else "-"
            print(f"  {cname:<24}{mode:<10}{d['solved']:>4}/{d['n']:<2}{d['false_accept']:>10}{cid:>12}{d['wrong_ids']:>6}"
                  f"{d['useful']:>11.1f}{d['artifacts']:>10.1f}{lost:>11}{d['cent']:>9.3f}{d['rms']:>7.3f}{d['att_med']:>7.1f}\""
                  f"{d['att_max']:>7.1f}\"{1000 * d['time']['total']:>8.0f}"
                  + (f"   codes: {', '.join(d['codes'])}" if d["codes"] else ""))
    print("\n  stage times, mean over all 36 fields (ms):")
    print("    " + f"{'mode':<10}" + "".join(f"{k:>15}" for k in STAGES))
    for mode in ("classical", "hybrid"):
        R = res[mode]["results"]
        allr = [r for recs in R.values() for r in recs]
        print("    " + f"{mode:<10}" + "".join(f"{1000 * np.mean([r['timing'].get(k, 0.0) for r in allr]):>15.1f}" for k in STAGES))
    print("\n  peak memory:")
    for mode in ("classical", "hybrid"):
        m = res[mode]
        print(f"    {mode:<10} tracemalloc peak during one solve {m['tracemalloc_peak_mb']:.1f} MB (Python/NumPy heap);"
              f" process peak working set {m['process_peak_mb']:.0f} MB (before the first pipeline is built: {m["process_peak_after_imports_mb"]:.0f} MB)")

    # totals table for the report
    tot = {}
    for mode in ("classical", "hybrid"):
        R = res[mode]["results"]
        pos = [r for c, recs in R.items() for r in recs if r["expect"]]
        neg = [r for c, recs in R.items() for r in recs if not r["expect"]]
        sp = [r for r in pos if r["success"] and r["att"] < 60]
        allr = [r for recs in R.values() for r in recs]
        tot[mode] = {"success": f"{len(sp)}/{len(pos)}", "false": f"{sum(r['success'] for r in neg) + sum(1 for r in pos if r['success'] and r['att'] >= 60)}/{len(neg)}",
                     "correct": f"{sum(r.get('correct_ids', 0) for r in sp)}/{sum(r['n_cat_in_frame'] for r in pos)}",
                     "wrong": sum(r.get("wrong_ids", 0) for r in pos if r["success"]),
                     "cent": np.median(sum((r["centroid_err"] for r in sp), [])),
                     "rms": np.sqrt(np.mean([r["rms"] ** 2 for r in sp])),
                     "att": (np.median([r["att"] for r in sp]), np.percentile([r["att"] for r in sp], 90)),
                     "time": {k: 1000 * np.mean([r["timing"].get(k, 0.0) for r in allr]) for k in STAGES},
                     "mem": (res[mode]["tracemalloc_peak_mb"], res[mode]["process_peak_mb"])}
    lines = ["| Metric | Classical | Hybrid |", "|---|---|---|",
             f"| Verified solve success rate (30 solvable fields) | {tot['classical']['success']} | {tot['hybrid']['success']} |",
             f"| False solutions accepted (6 incorrect-sky/catalogue fields) | {tot['classical']['false']} | {tot['hybrid']['false']} |",
             f"| Correct star IDs (of catalogue stars in frame) | {tot['classical']['correct']} | {tot['hybrid']['correct']} |",
             f"| Incorrect star associations | {tot['classical']['wrong']} | {tot['hybrid']['wrong']} |",
             f"| Median centroid error, matched stars (px) | {tot['classical']['cent']:.3f} | {tot['hybrid']['cent']:.3f} |",
             f"| Reprojection RMS (px) | {tot['classical']['rms']:.3f} | {tot['hybrid']['rms']:.3f} |",
             f"| Attitude error vs truth, median / p90 (arcsec) | {tot['classical']['att'][0]:.1f} / {tot['classical']['att'][1]:.1f}"
             f" | {tot['hybrid']['att'][0]:.1f} / {tot['hybrid']['att'][1]:.1f} |"]
    for k in STAGES:
        lines.append(f"| Mean time: {k} (ms) | {tot['classical']['time'][k]:.1f} | {tot['hybrid']['time'][k]:.1f} |")
    lines.append(f"| Peak memory: tracemalloc during a solve / process working set (MB) | {tot['classical']['mem'][0]:.1f} /"
                 f" {tot['classical']['mem'][1]:.0f} | {tot['hybrid']['mem'][0]:.1f} / {tot['hybrid']['mem'][1]:.0f} |")
    (OUT / "exp_c_table.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    json.dump({f"{k[0]}|{k[1]}": {kk: (vv if not isinstance(vv, np.floating) else float(vv)) for kk, vv in v.items()}
               for k, v in rows.items()}, open(OUT / "exp_c_rows.json", "w"), default=float, indent=1)
    make_figures(rows)


def make_figures(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from conditions import CAT
    from stellar_vision.geometry import CameraModel

    # ---------- figure 1: accepted / rejected associations, hot-pixel & streak field, both modes
    cname, sd = "hot pixels & streaks", 431
    image, truth, R, cat, _ = make_field(cname, sd)
    fig, axs = plt.subplots(1, 2, figsize=(16, 8.4))
    for ax, mode in zip(axs, ("classical", "hybrid")):
        pipe = StellarVisionPipeline.from_config(config_for(mode, cname), catalog=cat)
        r = pipe.solve(image)
        lab = truth_labels(r, truth)
        ax.imshow(np.arcsinh((image - np.median(image)) / 25), cmap="gray_r", vmin=-1, vmax=5)
        c = r.diagnostics["candidates"]
        scr = ~np.asarray(c["kept_by_screen"])
        ax.plot(np.asarray(c["x"])[scr], np.asarray(c["y"])[scr], "x", color="tab:purple", ms=7, mew=1.5,
                label=f"rejected by CNN before fit ({scr.sum()})")
        src = r.sources
        acc = np.asarray(src["accepted"])
        ax.plot(np.asarray(src["x"])[~acc], np.asarray(src["y"])[~acc], "x", color="tab:orange", ms=7, mew=1.5,
                label=f"rejected by quality rules ({(~acc).sum()})")
        matched = np.zeros(len(src["x"]), bool)
        if r.success:
            matched[np.asarray(r.matched_source_index)] = True
            cam = CameraModel()
            idx = np.searchsorted(cat.id, r.matched_star_ids)
            pu, pv, _ = cam.project(r.attitude, cat.vec[idx], r.focal_length_px)
            wrong = lab[np.asarray(r.matched_source_index)] != idx
            for (x, y), (px, py), w in zip(r.measured_centroids, np.column_stack([pu, pv]), wrong):
                ax.plot([x, px], [y, py], "-", color="red" if w else "lime", lw=2)
            ax.scatter(pu, pv, s=160, facecolors="none", edgecolors="tab:green", lw=1.5,
                       label=f"verified association ({len(idx)}; wrong: {wrong.sum()})")
            allu, allv, front = cam.project(r.attitude, cat.vec, r.focal_length_px)
            inf = np.nonzero(cam.in_frame(allu, allv, front))[0]
            um = np.setdiff1d(inf, idx)
            ax.scatter(allu[um], allv[um], s=160, facecolors="none", edgecolors="tab:red", lw=1.2, ls="--",
                       label=f"catalogue star predicted, not matched ({len(um)})")
        unm = acc & ~matched
        ax.plot(np.asarray(src["x"])[unm], np.asarray(src["y"])[unm], "+", color="tab:blue", ms=8,
                label=f"accepted source, unmatched ({unm.sum()})")
        ax.set_xlim(-0.5, 511.5)
        ax.set_ylim(511.5, -0.5)
        att = rotation_angle(r.attitude, R) / ARCSEC if r.success else np.nan
        ax.set_title(f"{mode}: {'VERIFIED' if r.success else r.failure_code.value}, {r.n_detected} candidates -> "
                     f"{r.n_sources} to solver, attitude error {att:.1f}\"", fontsize=10)
        ax.legend(fontsize=7.5, loc="lower left", framealpha=0.85)
    fig.suptitle(f"Accepted and rejected star associations - '{cname}' field (seed {sd}), same image, same verification")
    fig.tight_layout()
    fig.savefig(OUT / "exp_c_associations.png", dpi=105)
    plt.close(fig)

    # ---------- figure 2: system-level comparison
    names = list(CONDITIONS)
    short = ["nominal", "low\nSNR", "defocus", "hot px\nstreaks", "missing\nuncat.", "wrong\nsky/cat."]
    fig, axs = plt.subplots(1, 4, figsize=(19, 4.6))
    w = 0.38
    x = np.arange(len(names))
    for j, (mode, col) in enumerate((("classical", "tab:gray"), ("hybrid", "tab:blue"))):
        sol = [rows[(c, mode)]["solved"] / rows[(c, mode)]["n"] for c in names]
        axs[0].bar(x + (j - 0.5) * w, sol, w, color=col, label=mode)
        axs[1].bar(x + (j - 0.5) * w, [rows[(c, mode)]["artifacts"] for c in names], w, color=col, label=mode)
        axs[2].bar(x + (j - 0.5) * w, [rows[(c, mode)]["att_med"] for c in names], w, color=col, label=mode)
        bottom = np.zeros(len(names))
        for k, stg in enumerate(("detection", "classification", "centroiding", "identification")):
            v = np.array([1000 * rows[(c, mode)]["time"][stg] for c in names])
            axs[3].bar(x + (j - 0.5) * w, v, w, bottom=bottom, color=col, alpha=[1.0, 0.45, 0.75, 0.25][k],
                       edgecolor="k", lw=0.4, label=f"{mode}: {stg}")
            bottom += v
    axs[0].set_title("verified solves (incorrect sky: lower is right)")
    axs[1].set_title("artifacts passed to the solver per image")
    axs[2].set_title("median attitude error (arcsec)")
    axs[3].set_title("mean time per image by stage (ms)")
    for a in axs:
        a.set_xticks(x, short, fontsize=8)
        a.grid(alpha=0.3, axis="y")
    axs[0].legend(fontsize=8)
    axs[3].legend(fontsize=6.5, ncol=2)
    fig.suptitle("Experiment C: classical vs hybrid - complete-system comparison (6 synthetic fields per condition)")
    fig.tight_layout()
    fig.savefig(OUT / "exp_c_system_comparison.png", dpi=110)
    plt.close(fig)
    print("  saved", OUT / "exp_c_associations.png", "and", OUT / "exp_c_system_comparison.png")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3])
    else:
        main()
