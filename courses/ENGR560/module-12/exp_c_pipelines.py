"""
Experiment C: classical vs hybrid vs combined pipelines, end to end, on the same candidate proposals.

All pipelines start from the identical Module 11 detector proposals (matched filter FWHM 3, 5 sigma).
  A  classical   Module 11 detect_sources(): fit EVERY candidate, then reject SHARP (sigma < 0.6) and
                 failed fits.   A' classical strict: also reject BAD_FIT (sigma > 1.6 x nominal).
  B  hybrid      CNN on all candidate patches -> reject P < threshold -> fit only the retained ones
  C  combined    hybrid + reject SHARP after the fit
The CNN threshold was frozen in Experiment A on the validation images (>= 98% of detector-proposed true
stars kept); nothing here is tuned on test images. No retraining for the defocus condition.
40 independent test images per condition (seeds 30000-30039; same star fields, different conditions).
Recall is reported against ALL simulated stars and against the stars the detector actually proposed.
Times are wall-clock per image on this machine (single process, torch 1 thread, CPU).
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data12 import CONDITIONS, OUT, SPLITS, detect, label_candidates, match, simulate
import catalog  # Module 11 (on sys.path via data12)  # noqa: E402
from hybrid import FLAG_BAD_FIT, FLAG_SHARP, HybridDetector

PIPES = ("A classical", "A' classical strict", "B hybrid (CNN)", "C combined (CNN + SHARP)")


def evaluate(pos, truth):
    tr = np.column_stack([truth["x"], truth["y"]])
    pairs, d = match(pos, tr, 2.0)
    return len(pairs), len(pos) - len(pairs), d, pairs[:, 1] if len(pairs) else np.zeros(0, int)


def boot_ci(num, den, rng, B=2000):
    idx = rng.integers(0, len(num), (B, len(num)))
    v = num[idx].sum(1) / den[idx].sum(1)
    return np.percentile(v, [2.5, 97.5])


def main():
    torch.set_num_threads(1)
    hd = HybridDetector()
    print(f"Experiment C: end-to-end pipelines (frozen CNN threshold P >= {hd.threshold:.3f}, chosen on validation)")
    rng = np.random.default_rng(0)
    results, cand_tables = {}, {}
    seeds = list(SPLITS["test"])
    for cname, cond in CONDITIONS.items():
        rn = cond["read_noise"]
        per = {p: {"tp": [], "fp": [], "tp_prop": [], "err": [], "time": []} for p in PIPES}
        ntrue, nprop, stage = [], [], {"detect": [], "patches": [], "cnn": [], "fit": [], "n_fit": [], "n_cand": []}
        ct = {"y": [], "sub": [], "snr": [], "score": [], "sharp": [], "badfit": [], "failed": []}
        for s in seeds:
            image, truth, art, meta = simulate(seed=s, **cond)
            det, diag = detect(image, fwhm_assumed=3.0, return_diagnostics=True)
            y, sub, snr, tidx = label_candidates(det, truth, art)
            proposed = set(tidx[tidx >= 0].tolist())
            ntrue.append(len(truth["x"]))
            nprop.append(len(proposed))
            t0 = time.perf_counter()
            cat = catalog.detect_sources(image, read_noise=rn, uncertainty_scale=1.118)
            t_cls = time.perf_counter() - t0
            hyb = hd(image, read_noise=rn)
            for k_ in ("detect", "patches", "cnn", "fit"):
                stage[k_].append(hyb["timing"][k_])
            stage["n_fit"].append(hyb["n_fitted"])
            stage["n_cand"].append(hyb["n_candidates"])
            fl = cat["quality_flags"]
            failed = ~np.isfinite(cat["sigma"])
            sel = {"A classical": ((fl & FLAG_SHARP) == 0) & ~failed,
                   "A' classical strict": (fl & (FLAG_SHARP | FLAG_BAD_FIT)) == 0}
            pos_c = np.column_stack([cat["x"], cat["y"]])
            pos_h = np.column_stack([hyb["x"], hyb["y"]])
            catalogs = {p: (pos_c[m], t_cls) for p, m in sel.items()}
            catalogs["B hybrid (CNN)"] = (pos_h, hyb["timing"]["total"])
            catalogs["C combined (CNN + SHARP)"] = (pos_h[(hyb["quality_flags"] & FLAG_SHARP) == 0], hyb["timing"]["total"])
            for p, (pos, t) in catalogs.items():
                tp, fp, d, ti = evaluate(pos, truth)
                per[p]["tp"].append(tp)
                per[p]["fp"].append(fp)
                per[p]["tp_prop"].append(sum(int(i) in proposed for i in ti))
                per[p]["err"].append(d)
                per[p]["time"].append(t)
            ct["y"].append(y)
            ct["sub"].append(sub)
            ct["snr"].append(snr)
            ct["score"].append(1 / (1 + np.exp(-hd.score(image, det, diag["background"], diag["rms"]).astype(float))))
            ct["sharp"].append((fl & FLAG_SHARP) > 0)
            ct["badfit"].append((fl & FLAG_BAD_FIT) > 0)
            ct["failed"].append(failed)
        ntrue, nprop = np.array(ntrue), np.array(nprop)
        results[cname] = {"per": per, "ntrue": ntrue, "nprop": nprop, "stage": stage}
        cand_tables[cname] = {k_: np.concatenate(v) for k_, v in ct.items()}

        print(f"\n  {cname}: {len(seeds)} images, {ntrue.sum()} simulated stars, detector proposed"
              f" {nprop.sum()} of them ({nprop.sum() / ntrue.sum():.3f}), {sum(stage['n_cand'])} candidates in total")
        print(f"    {'pipeline':<27}{'precision':>10}{'recall(all)':>13}{'95% CI':>15}{'recall(prop.)':>15}"
              f"{'FP/img':>8}{'med err':>9}{'p90 err':>9}{'ms/img':>8}")
        for p in PIPES:
            r = per[p]
            tp, fp, tpp = np.array(r["tp"]), np.array(r["fp"]), np.array(r["tp_prop"])
            e = np.concatenate(r["err"])
            lo, hi = boot_ci(tp, ntrue, rng)
            print(f"    {p:<27}{tp.sum() / (tp.sum() + fp.sum()):>10.4f}{tp.sum() / ntrue.sum():>13.4f}"
                  f"{f'[{lo:.3f},{hi:.3f}]':>15}{tpp.sum() / nprop.sum():>15.4f}{fp.mean():>8.2f}"
                  f"{np.median(e):>9.3f}{np.percentile(e, 90):>9.3f}{1000 * np.mean(r['time']):>8.0f}")
        dB = np.array(per["B hybrid (CNN)"]["tp"]) - np.array(per["A classical"]["tp"])
        idx = rng.integers(0, len(dB), (2000, len(dB)))
        lo, hi = np.percentile(dB[idx].sum(1) / ntrue[idx].sum(1), [2.5, 97.5])
        print(f"    recall(all) difference hybrid - classical: {dB.sum() / ntrue.sum():+.4f}  (image bootstrap 95% CI"
              f" [{lo:+.4f}, {hi:+.4f}])")

        c = cand_tables[cname]
        acc = c["score"] >= hd.threshold
        cls_rej = c["sharp"] | c["failed"]
        print("    candidate level (same proposals):  subtype: n | rejected by classical rules | strict rules | CNN")
        for st in ("bright star", "faint star", "hot pixel", "elongated", "noise / other"):
            m = c["sub"] == st
            if m.sum():
                print(f"      {st:<15}{m.sum():>6} | {np.mean(cls_rej[m]):>6.1%} | {np.mean((cls_rej | c['badfit'])[m]):>6.1%}"
                      f" | {np.mean(~acc[m]):>6.1%}")

    print("\n  Part 4: per-image cost breakdown of the hybrid pipeline (mean ms; fits only on retained candidates)")
    print(f"    {'condition':<16}{'detect':>8}{'patches':>9}{'CNN':>7}{'fits':>8}{'n cand':>8}{'n fitted':>9}"
          f"{'hybrid total':>14}{'classical total':>17}{'ms per fit':>12}")
    for cname, r in results.items():
        st = r["stage"]
        print(f"    {cname:<16}{1000 * np.mean(st['detect']):>8.1f}{1000 * np.mean(st['patches']):>9.1f}"
              f"{1000 * np.mean(st['cnn']):>7.1f}{1000 * np.mean(st['fit']):>8.0f}{np.mean(st['n_cand']):>8.1f}"
              f"{np.mean(st['n_fit']):>9.1f}{1000 * np.mean(r['per']['B hybrid (CNN)']['time']):>14.0f}"
              f"{1000 * np.mean(r['per']['A classical']['time']):>17.0f}"
              f"{1000 * np.sum(st['fit']) / np.sum(st['n_fit']):>12.1f}")

    # figures
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
    w = 0.2
    names = list(CONDITIONS)
    for j, p in enumerate(PIPES):
        prec = [np.sum(results[c]["per"][p]["tp"]) / (np.sum(results[c]["per"][p]["tp"]) + np.sum(results[c]["per"][p]["fp"]))
                for c in names]
        rec = [np.sum(results[c]["per"][p]["tp"]) / results[c]["ntrue"].sum() for c in names]
        tms = [1000 * np.mean(results[c]["per"][p]["time"]) for c in names]
        xs = np.arange(3) + (j - 1.5) * w
        bars = axs[0].bar(xs, prec, w, label=p)
        axs[0].bar_label(bars, fmt="%.2f", fontsize=6)
        axs[1].bar(xs, rec, w, label=p)
        axs[2].bar(xs, tms, w, label=p)
    for c_i, c in enumerate(names):
        axs[1].plot([c_i - 0.45, c_i + 0.45], [results[c]["nprop"].sum() / results[c]["ntrue"].sum()] * 2, "k--", lw=1)
    axs[0].set_ylim(0, 1.08)
    axs[1].set_ylim(0, 1.05)
    axs[0].set_title("precision")
    axs[1].set_title("recall vs ALL simulated stars\n(dashed: detector proposal ceiling)")
    axs[2].set_title("processing time per image (ms)")
    for a in axs:
        a.set_xticks(range(3), names)
        a.grid(alpha=0.3, axis="y")
    axs[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "exp_c_pipelines.png", dpi=120)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    bins = np.array([0, 5, 7, 10, 15, 20, 40, 100, 1e4])
    for a, c in zip(axs, names):
        t = cand_tables[c]
        star = t["y"] == 1
        k = np.digitize(t["snr"][star], bins) - 1
        mid = np.sqrt(bins[:-1] * np.maximum(bins[1:], 1))
        mid[0] = 4
        for lab, keep in (("classical (not SHARP, fit ok)", ~(t["sharp"] | t["failed"])),
                          ("classical strict (+ not BAD_FIT)", ~(t["sharp"] | t["failed"] | t["badfit"])),
                          ("CNN >= frozen threshold", t["score"] >= hd.threshold)):
            frac = [np.mean(keep[star][k == b]) if np.sum(k == b) >= 5 else np.nan for b in range(len(bins) - 1)]
            a.semilogx(mid, frac, "o-", label=lab)
        a.set_title(f"{c}: proposed true stars kept")
        a.set_xlabel("true SNR")
        a.grid(alpha=0.3)
    axs[0].set_ylabel("fraction kept")
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "exp_c_kept_vs_snr.png", dpi=120)
    plt.close(fig)
    print("  saved", OUT / "exp_c_pipelines.png", "and", OUT / "exp_c_kept_vs_snr.png")

    # deliverable demo
    image, truth, _, _ = simulate(seed=SPLITS["test"][0], **CONDITIONS["nominal"])
    out = hd(image, reject_flags=FLAG_SHARP)
    print(f"\n  hybrid_detect_sources() demo on test image {SPLITS['test'][0]}: {len(out['x'])} sources;"
          f" columns: {', '.join(k_ for k_ in out if isinstance(out[k_], np.ndarray))}")
    for i in np.argsort(-out["flux"])[:5]:
        print(f"    x {out['x'][i]:8.3f}  y {out['y'][i]:8.3f}  flux {out['flux'][i]:8.0f}  snr {out['snr'][i]:6.1f}"
              f"  sigma_r {out['centroid_uncertainty'][i]:.3f}  flags {catalog.flag_string(out['quality_flags'][i]):<8}"
              f"  cnn_score {out['cnn_score'][i]:.4f}")


if __name__ == "__main__":
    main()
