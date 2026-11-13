"""
Experiment C: our detector vs photutils DAOStarFinder, and the integrated source catalog.

Two configurations, 10 independent images each (same images and truth catalogue for both methods):
    nominal   flux 300-30000 e-, background 100 e-, read noise 5 e-, FWHM 3 px
    low SNR   flux 100-3000 e-,  background 300 e-, read noise 10 e-, FWHM 3 px
Ours:  detect() + Gaussian PSF fit (catalog.detect_sources); reported for all detections and for
       detections without the SHARP flag (DAOStarFinder rejects sharp sources by default too).
DAOStarFinder(threshold = 5 * background_rms, fwhm = 3.0) on the background-subtracted image, using the
same sigma-clipped background and scalar RMS; a None result (no sources) is handled.
Matching: one-to-one Hungarian, max 2 px. Timing: median of 5 repeats per image, whole pipeline from
the raw image (background estimation included for both).

Centroid uncertainty: the fit covariance gives a model-based sigma; its scale is CALIBRATED on 10
separate nominal calibration images (std of the normalised errors), then checked on the evaluation
images (coverage of the 1-sigma and 2-sigma radial intervals).
"""

import time
from pathlib import Path

import numpy as np
from photutils.detection import DAOStarFinder

from catalog import FLAG_NAMES, detect_sources, flag_string
from starfield import background_stats, match, simulate, summarize

OUT = Path(__file__).parent / "outputs"
CONFIGS = {"nominal": dict(), "low SNR": dict(flux_range=(100.0, 3000.0), background=300.0, read_noise=10.0)}
N_IMAGES = 10


def dao(image, k=5.0):
    bkg, rms = background_stats(image)
    finder = DAOStarFinder(threshold=k * rms, fwhm=3.0)
    sources = finder(image - bkg)
    if sources is None or len(sources) == 0:
        return np.zeros((0, 2))
    # photutils >= 3.0 names the columns x_centroid / y_centroid (the old xcentroid / ycentroid are deprecated)
    xc, yc = ("x_centroid", "y_centroid") if "x_centroid" in sources.colnames else ("xcentroid", "ycentroid")
    return np.column_stack([np.asarray(sources[xc], float), np.asarray(sources[yc], float)])


def timed(fn, *a, reps=5, **kw):
    ts, r = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        r = fn(*a, **kw)
        ts.append(time.perf_counter() - t0)
    return r, float(np.median(ts))


def evaluate(pos, truth):
    tr = np.column_stack([truth["x"], truth["y"]])
    pairs, dist = match(pos, tr, 2.0)
    return {"tp": len(pairs), "fp": len(pos) - len(pairs), "fn": len(tr) - len(pairs), "err": dist,
            "pairs": pairs}


def calibrate(scale_seeds):
    zx, zy = [], []
    for s in scale_seeds:
        img, truth, _, meta = simulate(seed=s)
        cat = detect_sources(img, read_noise=meta["read_noise"])
        pos = np.column_stack([cat["x"], cat["y"]])
        ok = (cat["quality_flags"] & (1 | 16)) == 0
        ev = evaluate(pos, truth)
        for di, ti in ev["pairs"]:
            if ok[di]:
                zx.append((cat["x"][di] - truth["x"][ti]) / cat["sx"][di])
                zy.append((cat["y"][di] - truth["y"][ti]) / cat["sy"][di])
    z = np.r_[zx, zy]
    return float(np.std(z)), len(z)


def main():
    OUT.mkdir(exist_ok=True)
    print("Experiment C: our detector vs photutils DAOStarFinder")
    scale, nz = calibrate(range(500, 510))
    # DAOStarFinder threshold semantics differ from ours (photutils compares a differently normalised
    # convolved image), so besides the course's 5 * rms, pick its multiplier on the SAME calibration
    # images: highest recall with precision >= 0.97.
    cal = [simulate(seed=sd) for sd in range(500, 510)]
    k_tab = {}
    for kk in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
        tp = fp = fn = 0
        for img_c, truth_c, _, _ in cal:
            ev = evaluate(dao(img_c, kk), truth_c)
            tp, fp, fn = tp + ev["tp"], fp + ev["fp"], fn + ev["fn"]
        k_tab[kk] = (tp / max(tp + fp, 1), tp / (tp + fn))
    ok_k = [kk for kk, (pr, rc) in k_tab.items() if pr >= 0.97]
    k_dao = max(ok_k, key=lambda kk: k_tab[kk][1])
    print("  DAOStarFinder threshold multiplier on the calibration images (precision / recall): "
          + ", ".join(f"{kk}: {pr:.3f}/{rc:.3f}" for kk, (pr, rc) in k_tab.items()) + f"  -> selected k = {k_dao}")
    print(f"  centroid-uncertainty calibration on 10 separate nominal images: std of (error / model sigma) per axis"
          f" = {scale:.3f} over {nz} values -> uncertainty_scale = {scale:.3f}")

    for cname, kw in CONFIGS.items():
        names = ("ours (all)", "ours (not SHARP)", "DAO, 5 x rms") + ((f"DAO, {k_dao} x rms",) if k_dao != 5.0 else ())
        acc = {m: {"tp": 0, "fp": 0, "fn": 0, "err": [], "t": []} for m in names}
        common_o, common_d, t_det = [], [], []
        cover1, cover2, flags_fp = [], [], {}
        for s in range(N_IMAGES):
            img, truth, hot, meta = simulate(seed=700 + s, **kw)
            cat, t_ours = timed(detect_sources, img, read_noise=meta["read_noise"], uncertainty_scale=scale)
            pos_d, t_dao = timed(dao, img)
            pos_d2, t_dao2 = timed(dao, img, k_dao)
            from starfield import detect as _detect
            t_det.append(timed(_detect, img)[1])
            pos_o = np.column_stack([cat["x"], cat["y"]])
            keep = (cat["quality_flags"] & 1) == 0
            for name, pos, t in (("ours (all)", pos_o, t_ours), ("ours (not SHARP)", pos_o[keep], t_ours),
                                 ("DAO, 5 x rms", pos_d, t_dao), (f"DAO, {k_dao} x rms", pos_d2, t_dao2)):
                if name not in acc:
                    continue
                ev = evaluate(pos, truth)
                a = acc[name]
                a["tp"] += ev["tp"]
                a["fp"] += ev["fp"]
                a["fn"] += ev["fn"]
                a["err"].extend(ev["err"])
                a["t"].append(t)
            ev = evaluate(pos_o, truth)
            matched = set(ev["pairs"][:, 0]) if len(ev["pairs"]) else set()
            evd = evaluate(pos_d2, truth)
            eo = {ti: e for (di, ti), e in zip(ev["pairs"], ev["err"])}
            ed = {ti: e for (di, ti), e in zip(evd["pairs"], evd["err"])}
            for ti in set(eo) & set(ed):
                common_o.append(eo[ti])
                common_d.append(ed[ti])
            for di, ti in ev["pairs"]:
                if (cat["quality_flags"][di] & (1 | 16)) == 0:
                    e = np.hypot(cat["x"][di] - truth["x"][ti], cat["y"][di] - truth["y"][ti])
                    # radial error of a 2-D Gaussian: P(r < sigma_r * c) with sigma_r = sqrt(sx^2+sy^2)
                    cover1.append(e <= cat["centroid_uncertainty"][di])
                    cover2.append(e <= 2 * cat["centroid_uncertainty"][di])
            for di in range(len(pos_o)):
                if di not in matched:
                    key = flag_string(cat["quality_flags"][di])
                    flags_fp[key] = flags_fp.get(key, 0) + 1
        print(f"\n  Configuration: {cname} ({N_IMAGES} images x 50 stars)")
        print(f"    {'metric':<28}" + "".join(f"{m:>19}" for m in acc))
        rows = (("detection precision", lambda a: a["tp"] / max(a["tp"] + a["fp"], 1), "{:.3f}"),
                ("detection recall", lambda a: a["tp"] / (a["tp"] + a["fn"]), "{:.3f}"),
                ("false positives (total)", lambda a: a["fp"], "{}"),
                ("median centroid error [px]", lambda a: summarize(a["err"])["median"], "{:.3f}"),
                ("90th-pct centroid error", lambda a: summarize(a["err"])["p90"], "{:.3f}"),
                ("processing time per image", lambda a: 1000 * np.median(a["t"]), "{:.0f} ms"))
        for label, fn, fmt in rows:
            print(f"    {label:<28}" + "".join(f"{fmt.format(fn(a)):>19}" for a in acc.values()))
        so, sd_ = summarize(common_o), summarize(common_d)
        print(f"    same stars only ({len(common_o)} detected by both ours and DAO {k_dao} x rms): median / p90 centroid error"
              f" ours {so['median']:.3f} / {so['p90']:.3f} px, DAO {sd_['median']:.3f} / {sd_['p90']:.3f} px")
        print(f"    time breakdown (ours): detection alone {1000 * np.median(t_det):.0f} ms; the rest is one PSF fit per detection")
        print(f"    our false positives by quality flag: {dict(sorted(flags_fp.items(), key=lambda kv: -kv[1]))}")
        print(f"    calibrated uncertainty, matched unflagged (no SHARP/BAD_FIT) stars: radial error within 1 sigma_r"
              f" {np.mean(cover1):.1%} (2-D Gaussian expectation {1 - np.exp(-0.5 * 2):.1%} for sigma_r^2 = sx^2 + sy^2),"
              f" within 2 sigma_r {np.mean(cover2):.1%} (expectation {1 - np.exp(-0.5 * 8):.1%})")

    img, truth, _, meta = simulate(seed=700)
    cat = detect_sources(img, read_noise=meta["read_noise"], uncertainty_scale=scale)
    order = np.argsort(-cat["snr"])
    print("\n  Final output - first rows of the integrated catalogue (nominal image, brightest first):")
    print(f"    {'x':>9}{'y':>9}{'flux [e-]':>11}{'snr':>8}{'centroid_uncertainty [px]':>27}  quality_flags")
    for i in list(order[:4]) + list(order[-4:]):
        print(f"    {cat['x'][i]:>9.3f}{cat['y'][i]:>9.3f}{cat['flux'][i]:>11.0f}{cat['snr'][i]:>8.1f}"
              f"{cat['centroid_uncertainty'][i]:>27.3f}  {cat['quality_flags'][i]} ({flag_string(cat['quality_flags'][i])})")
    print(f"    keys: {sorted(k for k in cat if not k.startswith('background'))}; flag bits {FLAG_NAMES}")


if __name__ == "__main__":
    main()
