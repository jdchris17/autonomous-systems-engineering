"""
Experiment C: solve rendered images from the Module 12 detector output, and compare with Astrometry.net.

Pipeline per field (6 fields, fixed seeds):
  catalogue (RA/Dec) -> true camera -> render (PSF, noise, hot pixels, streaks, 40 uncatalogued faint
  field stars) -> Module 12 hybrid detector -> adapter (coordinates untouched) -> source list
  -> (a) our constrained triangle solver      (b) Astrometry.net solve-field on the SAME list
Also run on the raw Module 11 candidate list (no CNN: hot pixels and streaks included) to see the effect of
false detections, and on control lists that must NOT solve (random points; a field from another catalogue).
Astrometry.net 0.98 in WSL, using indexes built from OUR synthetic catalogue (build-astrometry-index -P 8..11),
given the same search region (RA 150, Dec 20, radius 5.5 deg) and scale range (28-32 arcsec/px).
Pixel convention: our coordinates are zero-based pixel centres; the .xyls gets +1.0 (FITS 1-based). This is
checked with an exact (noiseless) reference list written with and without the +1.
solve-field default: --tweak-order 2 (TAN + 2nd-order SIP); the hybrid lists are also solved with --tweak-order 1.
Attitude from a WCS: a 9 x 9 pixel grid is mapped to the sky with the WCS (including SIP), and R_CI is the
Wahba fit between those sky directions and pinhole camera directions using the WCS centre pixel scale.
"""

import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

import render13  # noqa: F401  (sets up module-11/12 import paths)
import catalog as m11  # Module 11 detect_sources (raw candidate list)
import sky13 as s
from adapter13 import check_unmodified, to_source_list, write_xyls
from data12 import match
from hybrid import HybridDetector
from render13 import render
from solver13 import TriangleSolver

OUT = Path(__file__).resolve().parent / "outputs"
AST = OUT / "astrometry"
SEEDS = range(200, 206)
F_APPROX = s.F_TRUE * 1.01


def wsl_path(p):
    p = str(Path(p).resolve()).replace("\\", "/")
    return "/mnt/" + p[0].lower() + p[2:]


def build_indexes(cat):
    AST.mkdir(parents=True, exist_ok=True)
    fits.BinTableHDU.from_columns([fits.Column(name="RA", format="D", array=cat["ra"]),
                                   fits.Column(name="DEC", format="D", array=cat["dec"]),
                                   fits.Column(name="MAG", format="D", array=cat["mag"])]).writeto(AST / "catalog13.fits", overwrite=True)
    names = []
    for P in (8, 9, 10, 11):
        name = f"index-13-{P}"
        if not (AST / f"{name}.fits").exists():
            subprocess.run(["wsl", "-e", "bash", "-c", f"cd '{wsl_path(AST)}' && build-astrometry-index -i catalog13.fits"
                            f" -o {name}.fits -P {P} -S MAG -E -j 1 -I 1300{P}"], capture_output=True, text=True)
        names.append(name)
    # newline="\n": a Windows CRLF config makes astrometry-engine look for "index-13-8\r" and fail
    (AST / "astrometry13.cfg").write_text("cpulimit 60\nadd_path " + wsl_path(AST) + "\n" + "".join(f"index {n}\n" for n in names),
                                          newline="\n")


def solve_field(src, tag, one_based=True, tweak=2):
    d = AST / "runs"
    d.mkdir(exist_ok=True)
    write_xyls(src, d / f"{tag}.xyls", one_based)
    cmd = (f"cd '{wsl_path(d)}' && solve-field {tag}.xyls --config ../astrometry13.cfg --width 512 --height 512"
           f" -X X -Y Y -s FLUX --scale-units arcsecperpix --scale-low 28 --scale-high 32 --ra 150 --dec 20"
           f" --radius 5.5 --crpix-center --tweak-order {tweak} --no-plots --overwrite --dir out_{tag}")
    t0 = time.perf_counter()
    p = subprocess.run(["wsl", "-e", "bash", "-c", cmd], capture_output=True, text=True)
    dt = time.perf_counter() - t0
    od = d / f"out_{tag}"
    res = {"time": dt, "success": (od / f"{tag}.solved").exists(), "log": p.stdout}
    if res["success"]:
        res["wcs"] = WCS(fits.getheader(od / f"{tag}.wcs"))
        corr = fits.getdata(od / f"{tag}.corr")
        res["n_match"] = len(corr)
        res["corr_rms"] = float(np.sqrt(np.mean((corr["field_x"] - corr["index_x"]) ** 2 + (corr["field_y"] - corr["index_y"]) ** 2)))
        lo = [l for l in p.stdout.splitlines() if "log-odds ratio" in l]
        res["logodds"] = lo[-1].split("log-odds ratio")[1].split()[0] if lo else "?"
    return res


def wcs_attitude(w):
    g = np.linspace(0, 511, 9)
    X, Y = np.meshgrid(g, g)
    ra, dec = w.all_pix2world(X.ravel() + 1, Y.ravel() + 1, 1)
    f = 1 / np.radians(np.mean(proj_plane_pixel_scales(w)))
    return s.wahba_svd(s.pixel_to_vec(X.ravel(), Y.ravel(), f), s.radec_to_vec(ra, dec)), f


def wcs_project(w, cat_idx, cat):
    x, y = w.all_world2pix(cat["ra"][cat_idx], cat["dec"][cat_idx], 1)
    return x - 1, y - 1


def label(src, truth):
    det = np.column_stack([src["x"], src["y"]])
    ok = np.isfinite(det).all(1)
    t = np.full(len(det), -2)                                                  # -2 artifact / spurious
    pairs, _ = match(det[ok], np.column_stack([truth["x"], truth["y"]]), 2.0)
    okidx = np.nonzero(ok)[0]
    for di, ti in pairs:
        t[okidx[di]] = truth["cat_idx"][ti]                                    # >= 0 catalogue, -1 field star
    return t


def evaluate_true(xp, yp, src, t, cat_idx_used):
    """Reprojection residuals on the TRUE detection <-> catalogue correspondences (same set for both solvers)."""
    m = {c: i for i, c in enumerate(cat_idx_used)}
    dets = [d for d in range(len(t)) if t[d] >= 0 and t[d] in m]
    r = np.array([np.hypot(src["x"][d] - xp[m[t[d]]], src["y"][d] - yp[m[t[d]]]) for d in dets])
    return r


def main():
    OUT.mkdir(exist_ok=True)
    cat = s.make_catalog()
    other = s.make_catalog(seed=999)
    build_indexes(cat)
    hd = HybridDetector()
    solver = TriangleSolver(cat, f_approx=F_APPROX, sigma_px=0.2)
    t0 = time.perf_counter()
    subprocess.run(["wsl", "-e", "true"], capture_output=True)
    wsl_overhead = time.perf_counter() - t0
    print("Experiment C: Module 12 detector -> astrometric solve; our solver vs Astrometry.net 0.98 (WSL)")

    # --- pixel convention test -----------------------------------------------------------------
    rng = np.random.default_rng(7)
    R, _ = s.random_pointing(rng)
    o = s.observe(cat, R, rng, 0.0)
    fl = render13.mag_to_flux(cat["mag"][o["cat_idx"]])
    ref = {"x": o["u"], "y": o["v"], "flux": fl}
    print("  pixel-convention test (exact projected positions, zero-based):")
    for one_based in (True, False):
        r = solve_field(ref, f"convention_{int(one_based)}", one_based)
        if r["success"]:
            xp, yp = wcs_project(r["wcs"], o["cat_idx"], cat)
            dx, dy = np.mean(xp - o["u"]), np.mean(yp - o["v"])
            print(f"    written {'with +1.0 (1-based)' if one_based else 'as-is (0-based)   '}: WCS-predicted - true position"
                  f" mean ({dx:+.4f}, {dy:+.4f}) px, rms {np.sqrt(np.mean((xp - o['u'])**2 + (yp - o['v'])**2)):.4f} px")
        else:
            print(f"    written {'with +1' if one_based else 'as-is'}: did not solve")

    # --- fields ---------------------------------------------------------------------------------
    rows, fig_data = [], None
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        R, pose = s.random_pointing(rng)
        image, truth = render(cat, R, sd)
        hc = hd(image, read_noise=5.0)
        raw = m11.detect_sources(image, read_noise=5.0, uncertainty_scale=1.118)
        raw = {k: v for k, v in raw.items() if isinstance(v, np.ndarray)}
        for name, cat_out in (("hybrid (CNN)", hc), ("raw candidates (no CNN)", raw)):
            src = to_source_list(cat_out)
            assert check_unmodified(cat_out, src)
            t = label(src, truth)
            ours = solver.solve(src["x"], src["y"], order=np.arange(len(src["x"])), sigma_px=src["sigma_px"])
            ast = solve_field(src, f"field{sd}_{'h' if name.startswith('h') else 'r'}")
            row = {"seed": sd, "list": name, "n_src": len(t), "n_cat": int(np.sum(t >= 0)), "n_field": int(np.sum(t == -1)),
                   "n_art": int(np.sum(t == -2)), "ours": ours, "ast": ast}
            vis = np.nonzero(s.in_frame(*s.project(R, cat["vec"])[:2], s.project(R, cat["vec"])[2]))[0]
            if ours["success"]:
                assigned = dict(zip(ours["det_idx"].tolist(), ours["cat_idx"].tolist()))
                row["o_correct"] = sum(1 for d, c in assigned.items() if t[d] == c)
                row["o_wrong"] = len(assigned) - row["o_correct"]
                row["o_wrong_kind"] = [int(t[d]) for d, c in assigned.items() if t[d] != c]
                row["o_att"] = s.rotation_angle(ours["R"], R) / s.ARCSEC
                pu, pv, _ = s.project(ours["R"], cat["vec"][vis], ours["f"])
                row["o_true_res"] = evaluate_true(pu, pv, src, t, vis)
            if ast["success"]:
                Rw, fw = wcs_attitude(ast["wcs"])
                row["a_att"] = s.rotation_angle(Rw, R) / s.ARCSEC
                xp, yp = wcs_project(ast["wcs"], vis, cat)
                row["a_true_res"] = evaluate_true(xp, yp, src, t, vis)
            if name.startswith("h"):                                   # same list, linear (TAN) tweak only
                a1 = solve_field(src, f"field{sd}_h_t1", tweak=1)
                row["ast1"] = a1
                if a1["success"]:
                    Rw, _ = wcs_attitude(a1["wcs"])
                    row["a1_att"] = s.rotation_angle(Rw, R) / s.ARCSEC
                    xp, yp = wcs_project(a1["wcs"], vis, cat)
                    row["a1_true_res"] = evaluate_true(xp, yp, src, t, vis)
            rows.append(row)
            if sd == SEEDS[0] and name.startswith("h"):
                fig_data = (image, src, t, ours, ast, R, vis)

    print(f"\n  per field (source lists: n = catalogue stars / uncatalogued field stars / artifacts):")
    print(f"    {'seed':<5}{'list':<25}{'n':>14}{'ours':>6}{'match':>6}{'ok':>4}{'bad':>4}{'att\"':>7}{'t s':>6}"
          f"{'  a.net':>7}{'match':>6}{'att\"':>7}{'log-odds':>9}{'t s':>6}")
    for r in rows:
        o, a = r["ours"], r["ast"]
        ostr = (f"{'yes':>6}{o['n_match']:>6}{r['o_correct']:>4}{r['o_wrong']:>4}{r['o_att']:>7.1f}" if o["success"]
                else f"{'NO':>6}{'':>6}{'':>4}{'':>4}{'':>7}")
        astr = (f"{'yes':>7}{a['n_match']:>6}{r['a_att']:>7.1f}{a['logodds']:>9}" if a["success"] else f"{'NO':>7}{'':>6}{'':>7}{'':>9}")
        counts = f"{r['n_cat']}/{r['n_field']}/{r['n_art']}"
        print(f"    {r['seed']:<5}{r['list']:<25}{counts:>14}{ostr}{o['time']:>6.2f}{astr}{a['time']:>6.2f}")

    print("\n  summary over the 6 fields (residuals: TRUE correspondences, identical star set for both solvers)")
    print(f"    {'list':<25}{'solver':<16}{'solved':>7}{'verified':>9}{'wrong IDs':>10}{'reproj RMS px':>14}"
          f"{'median px':>10}{'p90 px':>8}{'att med\"':>9}{'att max\"':>9}{'time s':>8}")
    for lst in ("hybrid (CNN)", "raw candidates (no CNN)"):
        rr = [r for r in rows if r["list"] == lst]
        for sv, key in (("ours", "o"), ("Astrometry.net", "a"), ("A.net tweak 1", "a1")):
            if key == "a1" and "ast1" not in rr[0]:
                continue
            kk = {"o": "ours", "a": "ast", "a1": "ast1"}[key]
            ok = [r for r in rr if r[kk]["success"]]
            res = np.concatenate([r[f"{key}_true_res"] for r in ok]) if ok else np.array([np.nan])
            ver = np.mean([r[kk]["n_match"] for r in ok]) if ok else np.nan
            wrong = sum(r["o_wrong"] for r in ok) if key == "o" and ok else ("n/a" if key != "o" else 0)
            att = [r[f"{key}_att"] for r in ok]
            tm = np.mean([r[kk]["time"] for r in rr])
            print(f"    {lst:<25}{sv:<16}{len(ok):>4}/{len(rr):<2}{ver:>9.1f}{str(wrong):>10}{np.sqrt(np.mean(res**2)):>14.3f}"
                  f"{np.median(res):>10.3f}{np.percentile(res, 90):>8.3f}{np.median(att) if att else np.nan:>9.1f}"
                  f"{np.max(att) if att else np.nan:>9.1f}{tm:>8.2f}")
    kinds = [k for r in rows if r["ours"]["success"] for k in r.get("o_wrong_kind", [])]
    print(f"    our wrong associations by true type: catalogue-other {sum(k >= 0 for k in kinds)},"
          f" uncatalogued field star {sum(k == -1 for k in kinds)}, artifact {sum(k == -2 for k in kinds)}")
    print(f"    Astrometry.net self-reported match RMS (its .corr file): "
          f"{np.mean([r['ast']['corr_rms'] for r in rows if r['ast']['success']]):.3f} px;"
          f" WSL process start-up alone costs {wsl_overhead:.2f} s of its wall-clock time")

    print("\n  false-solution tests (must NOT solve):")
    for name in ("random points", "wrong sky (other catalogue)"):
        n_o = n_a = 0
        for k, sd in enumerate(SEEDS[:3]):
            rng = np.random.default_rng(sd + 900)
            if name == "random points":
                src = {"x": rng.uniform(-0.5, 511.5, 30), "y": rng.uniform(-0.5, 511.5, 30), "flux": rng.uniform(1e3, 1e5, 30)}
            else:
                Rr, _ = s.random_pointing(rng)
                ob = s.observe(other, Rr, rng, 0.1)
                fl = render13.mag_to_flux(other["mag"][ob["cat_idx"]])
                oo = np.argsort(-fl)
                src = {"x": ob["u"][oo], "y": ob["v"][oo], "flux": fl[oo]}
            n_o += solver.solve(src["x"], src["y"])["success"]
            n_a += solve_field(src, f"false_{name[:4]}_{k}")["success"]
        print(f"    {name:<30} ours accepted {n_o}/3, Astrometry.net accepted {n_a}/3")

    # figure
    image, src, t, ours, ast, R, vis = fig_data
    fig, axs = plt.subplots(1, 2, figsize=(15, 7.2), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0]
    ax.imshow(np.arcsinh((image - np.median(image)) / 30), cmap="gray_r", origin="upper", vmin=-1, vmax=6)
    pu, pv, _ = s.project(ours["R"], cat["vec"][vis], ours["f"])
    ax.scatter(pu, pv, s=140, facecolors="none", edgecolors="tab:blue", lw=1.3, label="catalogue projected (our solution)")
    xa, ya = wcs_project(ast["wcs"], vis, cat)
    ax.scatter(xa, ya, s=40, marker="s", facecolors="none", edgecolors="tab:orange", lw=1.2, label="catalogue projected (Astrometry.net WCS)")
    inl = np.zeros(len(src["x"]), bool)
    inl[ours["det_idx"]] = True
    ax.plot(src["x"][inl], src["y"][inl], "g+", ms=8, label=f"detection matched ({inl.sum()})")
    ax.plot(src["x"][~inl], src["y"][~inl], "rx", ms=6, label=f"detection unmatched ({(~inl).sum()}: uncatalogued / artifacts)")
    ax.set_xlim(-0.5, 511.5)
    ax.set_ylim(511.5, -0.5)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title(f"rendered field seed {SEEDS[0]} with hybrid-detector sources")
    ax = axs[1]
    for key, col, lab in (("o", "tab:blue", "ours"), ("a", "tab:orange", "Astrometry.net")):
        rr = [r for r in rows if r["list"] == "hybrid (CNN)" and r[{"o": "ours", "a": "ast"}[key]]["success"]]
        res = np.concatenate([r[f"{key}_true_res"] for r in rr])
        ax.hist(res, bins=np.linspace(0, 0.6, 31), histtype="step", lw=2, color=col,
                label=f"{lab}: RMS {np.sqrt(np.mean(res**2)):.3f} px, median {np.median(res):.3f}")
    ax.set_xlabel("reprojection error on true correspondences (px)")
    ax.set_ylabel("stars (6 fields)")
    ax.set_title("same hybrid source lists, both solvers")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "exp_c_solutions.png", dpi=115)
    plt.close(fig)
    print("  saved", OUT / "exp_c_solutions.png")


if __name__ == "__main__":
    main()
