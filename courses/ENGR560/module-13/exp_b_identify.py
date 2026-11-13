"""
Experiment B: unknown star identities and camera rotation (constrained pattern matching).

The solver gets ONLY detected pixel coordinates (shuffled - no IDs, no brightness order), an approximate
focal length (+1 % error), the search region (boresight within 2.5 deg of RA 150, Dec 20; the truth is
drawn within 2 deg) and an assumed centroid sigma. Roll is unknown.
6 independent fields (fixed seeds) per condition. Controls that MUST be rejected:
  random points      the same number of uniformly random "detections"
  wrong sky          a real-looking star field from a DIFFERENT catalogue (pattern not in our catalogue)
"""

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sky13 as s
from solver13 import TriangleSolver

OUT = Path(__file__).resolve().parent / "outputs"
F_APPROX = s.F_TRUE * 1.01
SEEDS = range(100, 106)
CONDITIONS = {
    "nominal (roll 0, sigma 0.1 px)": dict(roll=0.0),
    "unknown image rotation": dict(),
    "20% missing stars": dict(missing=0.2),
    "20% false detections": dict(false=0.2),
    "increased noise 1 px": dict(sigma=1.0),
    "increased noise 3 px": dict(sigma=3.0),
    "missing + false + 1 px": dict(missing=0.2, false=0.2, sigma=1.0),
}


def make_field(cat, seed, roll=None, sigma=0.1, missing=0.0, false=0.0, other_cat=None):
    rng = np.random.default_rng(seed)
    R, pose = s.random_pointing(rng, roll=roll)
    src = cat if other_cat is None else other_cat
    o = s.observe(src, R, rng, sigma)
    u, v, truth = o["u"], o["v"], o["cat_idx"].copy()
    if other_cat is not None:
        truth[:] = -1
    keep = rng.random(len(u)) >= missing
    u, v, truth = u[keep], v[keep], truth[keep]
    nf = int(round(false * len(u) / (1 - false))) if false else 0        # false = 20% of the final list
    u = np.concatenate([u, rng.uniform(-0.5, s.SIZE - 0.5, nf)])
    v = np.concatenate([v, rng.uniform(-0.5, s.SIZE - 0.5, nf)])
    truth = np.concatenate([truth, -np.ones(nf, int)])
    p = rng.permutation(len(u))
    return {"u": u[p], "v": v[p], "truth": truth[p], "R": R, "pose": pose, "sigma": sigma}


def score(field, r):
    t = field["truth"]
    n_star = int(np.sum(t >= 0))
    if not r.get("success"):
        return {"solved": False, "n_star": n_star}
    assigned = dict(zip(r["det_idx"].tolist(), r["cat_idx"].tolist()))
    correct = sum(1 for d, c in assigned.items() if t[d] == c)
    wrong = len(assigned) - correct
    return {"solved": True, "n_star": n_star, "correct": correct, "wrong": wrong, "n_match": r["n_match"],
            "n_verify": r["n_verify"], "med": float(np.median(r["resid_px"])), "p90": float(np.percentile(r["resid_px"], 90)),
            "att": s.rotation_angle(r["R"], field["R"]) / s.ARCSEC, "f_err": r["f"] / s.F_TRUE - 1, "time": r["time"]}


def main():
    OUT.mkdir(exist_ok=True)
    cat = s.make_catalog()
    other = s.make_catalog(seed=999)
    print("Experiment B: star identification without associations (6 fields per condition)")
    print(f"  approximate focal length {F_APPROX:.1f} px (true {s.F_TRUE:.1f}, +1.0 %); boresight search radius 2.5 deg")
    rows, keep_fig = {}, None
    for cname, kw in CONDITIONS.items():
        sig = kw.get("sigma", 0.1)
        solver = TriangleSolver(cat, f_approx=F_APPROX, sigma_px=max(0.2, sig))
        res = []
        for sd in SEEDS:
            fld = make_field(cat, sd, **kw)
            r = solver.solve(fld["u"], fld["v"])
            res.append((fld, r, score(fld, r)))
            if cname == "missing + false + 1 px" and sd == SEEDS[0]:
                keep_fig = (fld, r)
        rows[cname] = res
    print(f"  catalogue subset {len(solver.idx)} stars, {len(solver.tri_s)} catalogue triangles (built in {solver.build_time:.2f} s)")
    print(f"\n    {'condition':<32}{'solved':>7}{'correct ID':>11}{'wrong':>6}{'matches':>8}{'verify':>7}"
          f"{'med px':>8}{'p90 px':>8}{'att med':>9}{'att max':>9}{'f err':>8}{'s/field':>8}")
    for cname, res in rows.items():
        sc = [x[2] for x in res]
        ok = [q for q in sc if q["solved"]]
        if ok:
            ci = sum(q["correct"] for q in ok) / sum(q["n_star"] for q in sc)
            print(f"    {cname:<32}{len(ok):>4}/{len(sc):<2}{ci:>11.3f}{sum(q['wrong'] for q in ok):>6}"
                  f"{np.mean([q['n_match'] for q in ok]):>8.1f}{np.mean([q['n_verify'] for q in ok]):>7.1f}"
                  f"{np.median([q['med'] for q in ok]):>8.3f}{np.median([q['p90'] for q in ok]):>8.3f}"
                  f"{np.median([q['att'] for q in ok]):>8.1f}\"{np.max([q['att'] for q in ok]):>8.1f}\""
                  f"{np.median([abs(q['f_err']) for q in ok]):>8.1e}{np.mean([x[1]['time'] for x in res]):>8.2f}")
        else:
            print(f"    {cname:<32}{0:>4}/{len(sc):<2}")
        for fld, r, q in res:
            if not q["solved"]:
                print(f"      FAILED field: {r.get('reason')}  ({len(fld['u'])} detections, {q['n_star']} real)")

    print("\n  controls (must be rejected):")
    solver = TriangleSolver(cat, f_approx=F_APPROX, sigma_px=0.2)
    for name in ("random points", "wrong sky (other catalogue)", "random points, 3 px tolerance"):
        acc, info = 0, []
        for sd in SEEDS:
            rng = np.random.default_rng(sd + 500)
            if name.startswith("random"):
                n = 24
                u, v = rng.uniform(-0.5, 511.5, n), rng.uniform(-0.5, 511.5, n)
                slv = solver if "3 px" not in name else TriangleSolver(cat, f_approx=F_APPROX, sigma_px=3.0)
            else:
                fld = make_field(cat, sd, other_cat=other)
                u, v, slv = fld["u"], fld["v"], solver
            r = slv.solve(u, v)
            acc += bool(r["success"])
            info.append(f"{r.get('n_verify', 0)}/{r.get('n_match', 0)}")
        print(f"    {name:<32} false solutions accepted: {acc}/{len(SEEDS)};  best hypothesis extra/total matches per"
              f" field: {', '.join(info)}  (acceptance needs >= {solver.min_verify} extra and >= {solver.min_frac:.0%})")

    # figure
    fld, r = keep_fig
    pu, pv, front = s.project(r["R"], cat["vec"], r["f"])
    inf = s.in_frame(pu, pv, front)
    fig, ax = plt.subplots(figsize=(8, 8))
    matched_cat = set(r["cat_idx"].tolist())
    um = [i for i in np.nonzero(inf)[0] if i not in matched_cat]
    ax.scatter(pu[um], pv[um], s=90, facecolors="none", edgecolors="tab:orange", lw=1.5, label="unmatched catalogue star (projected)")
    ax.scatter(pu[r["cat_idx"]], pv[r["cat_idx"]], s=90, facecolors="none", edgecolors="tab:blue", lw=1.5,
               label="matched catalogue star (projected)")
    inl = np.zeros(len(fld["u"]), bool)
    inl[r["det_idx"]] = True
    ax.plot(fld["u"][inl], fld["v"][inl], "k+", ms=9, label=f"detection, inlier ({inl.sum()})")
    ax.plot(fld["u"][~inl], fld["v"][~inl], "rx", ms=8, label=f"detection, rejected ({(~inl).sum()})")
    for d, c in zip(r["det_idx"], r["cat_idx"]):
        ax.plot([fld["u"][d], pu[c]], [fld["v"][d], pv[c]], "g-", lw=1)
    hd = r["hyp_det"]
    ax.plot(fld["u"][list(hd) + [hd[0]]], fld["v"][list(hd) + [hd[0]]], "m--", lw=1, label="hypothesis triangle")
    for c in r["cat_idx"]:
        ax.annotate(f"{cat['id'][c]:03d}", (pu[c], pv[c]), xytext=(6, 6), textcoords="offset points", fontsize=7)
    ax.set_xlim(-0.5, 511.5)
    ax.set_ylim(511.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("u (column, px)")
    ax.set_ylabel("v (row, px)")
    q = score(fld, r)
    ax.set_title(f"Experiment B, 20% missing + 20% false + 1 px noise (seed {SEEDS[0]}): {q['correct']} correct IDs, {q['wrong']} wrong,"
                 f"\nattitude error {q['att']:.1f}\", reprojection median {q['med']:.3f} px", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "exp_b_matches.png", dpi=120)
    plt.close(fig)
    print("  saved", OUT / "exp_b_matches.png")


if __name__ == "__main__":
    main()
