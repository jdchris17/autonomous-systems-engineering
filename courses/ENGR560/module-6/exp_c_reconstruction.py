"""
Experiment C: image reconstruction and information loss.

PCA is fit on the development images only; held-out patches come from the untouched test images.
k in {1, 2, 4, 8, 16, 32, 64, 128}.

Preprocessing (Part 4; nothing below is learned except the PCA mean and components):
  A  raw pixels, feature-wise centring (PCA mean)
  B  subtract a per-patch background (median of the 3-px border), then centre
  C  background subtraction, then divide by a per-patch flux (aperture flux within 4 px of the
     measured centroid, floored at 3 x noise x sqrt(aperture area) so empty patches do not explode),
     then centre
Reconstructions are mapped back to pixel units (C: multiply by the flux, B and C: add the background)
so every approach is scored on the same original pixel grid.

Physical-detail preservation (Part 3) compares measurements on the RECONSTRUCTION with the same
measurements on the NOISE-FREE source (patches.measure, adaptive Gaussian-weighted moments), so the
numbers isolate information lost by PCA rather than estimator bias:
  centroid error, centroid "slope" (regression of measured on true offset: 1 = unbiased,
  < 1 = pulled toward the training-mean position), width and roundness error, peak ratio,
  faint-star visibility (SNR >= 5 using the ORIGINAL patch's noise, so denoising cannot inflate it).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pca_lda import PCAFromScratch
from patches import CENTRE, SIZE, border_background, development_set, measure, untouched_test_set

OUT = __import__("pathlib").Path(__file__).parent / "outputs"
KS = (1, 2, 4, 8, 16, 32, 64, 128)
APPROACHES = ("A raw", "B background-subtracted", "C bg-sub + flux-normalised")


def preprocess(P, approach):
    """Returns flattened features and the per-patch parameters needed to invert them."""
    N = len(P)
    bg, noise = border_background(P)
    if approach.startswith("A"):
        return P.reshape(N, -1), (np.zeros(N), np.ones(N))
    Q = P - bg[:, None, None]
    if approach.startswith("B"):
        return Q.reshape(N, -1), (bg, np.ones(N))
    m = measure(P, bg, noise)
    floor = 3 * noise * np.sqrt(np.pi * 16)
    scale = np.maximum(m["flux"], floor)
    return (Q / scale[:, None, None]).reshape(N, -1), (bg, scale)


def postprocess(F, inv):
    bg, scale = inv
    return F.reshape(-1, SIZE, SIZE) * scale[:, None, None] + bg[:, None, None]


def main():
    OUT.mkdir(exist_ok=True)
    d, t = development_set(), untouched_test_set()
    P, Pt = d["X"], t["X"]
    _, noise_t = border_background(Pt)
    truth = measure(t["expected"], t["B"], np.sqrt(t["B"] + t["RN"] ** 2))
    tx, ty = t["x0"], t["y0"]
    sub = t["subtype"]
    print("Experiment C: reconstruction and information loss (PCA fit on development images only)")

    curves, recs, evs = {}, {}, {}
    for ap in APPROACHES:
        F, _ = preprocess(P, ap)
        Ft, inv_t = preprocess(Pt, ap)
        pca = PCAFromScratch().fit(F)
        evs[ap] = pca.cumulative_explained_variance_
        V, mu = pca.components_, pca.mean_
        for k in KS:
            Vk = V[:k]
            rec_t = postprocess((Ft - mu) @ Vk.T @ Vk + mu, inv_t)
            Ftr, inv_tr = preprocess(P, ap)
            rec_tr = postprocess((Ftr - mu) @ Vk.T @ Vk + mu, inv_tr)
            curves[(ap, k)] = (np.mean((P - rec_tr) ** 2), np.mean((Pt - rec_t) ** 2),
                               np.mean((t["expected"] - rec_t) ** 2))
            recs[(ap, k)] = rec_t

    print("\n  Part 2 - reconstruction MSE in original pixel units [e-^2] and cumulative explained variance")
    print(f"    {'k':>4}" + "".join(f"{ap[:1] + ': train / test / vs noise-free / cum.EV':>44}" for ap in APPROACHES))
    for k in KS:
        print(f"    {k:>4}" + "".join(f"{curves[(ap, k)][0]:>10.1f} /{curves[(ap, k)][1]:>8.1f} /{curves[(ap, k)][2]:>8.1f} /"
                                      f"{evs[ap][k - 1]:>7.3f}".rjust(44) for ap in APPROACHES))
    print(f"    noise variance of the test patches (mean B + RN^2): {np.mean(t['B'] + t['RN'] ** 2):.1f}")

    # ---------------------------------------------------------------- Part 3: physical details
    print("\n  Part 3 - physical-detail preservation on held-out patches (vs the same measurement on the"
          " noise-free source)")
    groups = {"bright": sub == "bright star", "faint": sub == "faint star", "hot": sub == "hot pixel",
              "elong": sub == "elongated"}

    def stats(m):
        cen_b = np.median(np.hypot(m["x"] - truth["x"], m["y"] - truth["y"])[groups["bright"]])
        cen_f = np.median(np.hypot(m["x"] - truth["x"], m["y"] - truth["y"])[groups["faint"]])
        b = groups["bright"]
        slope = np.polyfit(tx[b] - CENTRE, m["x"][b] - CENTRE, 1)[0]
        werr = np.nanmedian(np.abs(m["width"][b] / truth["width"][b] - 1))
        rerr = np.nanmedian(np.abs(m["roundness"][b] - truth["roundness"][b]))
        peak = np.median(m["peak"][b] / truth["peak"][b])
        hot_peak = np.median(m["peak"][groups["hot"]] / truth["peak"][groups["hot"]])
        el_round = np.nanmedian(m["roundness"][groups["elong"]])
        vis = np.mean(m["snr"][groups["faint"]] >= 5)
        return cen_b, cen_f, slope, werr, rerr, peak, hot_peak, el_round, vis

    hdr = (f"    {'approach':<26}{'k':>5}{'centroid bright':>16}{'centroid faint':>15}{'offset slope':>13}"
           f"{'width err':>10}{'round err':>10}{'peak ratio':>11}{'hot-px peak':>12}{'elong round':>12}{'faint SNR>=5':>13}")
    print(hdr)
    orig = measure(Pt, noise=noise_t)
    s0 = stats(orig)
    print(f"    {'original noisy patch':<26}{'-':>5}{s0[0]:>13.3f} px{s0[1]:>12.3f} px{s0[2]:>13.3f}{s0[3]:>10.1%}"
          f"{s0[4]:>10.3f}{s0[5]:>11.2f}{s0[6]:>12.2f}{s0[7]:>12.2f}{s0[8]:>13.1%}")
    print(f"    {'noise-free truth':<26}{'-':>5}{0:>13.3f} px{0:>12.3f} px{np.polyfit(tx[groups['bright']] - CENTRE, truth['x'][groups['bright']] - CENTRE, 1)[0]:>13.3f}"
          f"{0:>10.1%}{0:>10.3f}{1:>11.2f}{1:>12.2f}{np.nanmedian(truth['roundness'][groups['elong']]):>12.2f}"
          f"{np.mean(truth['snr'][groups['faint']] * np.sqrt(t['B'][groups['faint']] + t['RN'][groups['faint']] ** 2) / noise_t[groups['faint']] >= 5):>13.1%}")
    table = {}
    for ap in APPROACHES:
        for k in KS:
            m = measure(recs[(ap, k)], noise=noise_t)
            table[(ap, k)] = stats(m)
            s = table[(ap, k)]
            print(f"    {ap:<26}{k:>5}{s[0]:>13.3f} px{s[1]:>12.3f} px{s[2]:>13.3f}{s[3]:>10.1%}{s[4]:>10.3f}"
                  f"{s[5]:>11.2f}{s[6]:>12.2f}{s[7]:>12.2f}{s[8]:>13.1%}")
    print("    offset slope: regression of measured x-offset on the true x-offset for bright stars (1 = unbiased)")

    plots(curves, evs, recs, table, s0, t, Pt)


def plots(curves, evs, recs, table, s0, t, Pt):
    # gallery for each approach
    rng = np.random.default_rng(3)
    picks = []
    for s in ("bright star", "bright star", "faint star", "hot pixel", "diffuse", "elongated", "noise peak"):
        picks.append(rng.choice(np.nonzero(t["subtype"] == s)[0]))
    for ap in APPROACHES:
        fig, axes = plt.subplots(len(picks), len(KS) + 2, figsize=(1.45 * (len(KS) + 2), 1.55 * len(picks)))
        for r, i in enumerate(picks):
            row = [t["expected"][i], Pt[i]] + [recs[(ap, k)][i] for k in KS]
            lo, hi = np.percentile(Pt[i], 1), np.percentile(t["expected"][i], 99.9)
            for c, img in enumerate(row):
                ax = axes[r, c]
                ax.imshow(img, cmap="gray", origin="lower", vmin=lo, vmax=hi)
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(["noise-free", "observed"][c] if c < 2 else f"k={KS[c - 2]}", fontsize=8)
            axes[r, 0].set_ylabel(t["subtype"][i], fontsize=8)
        fig.suptitle(f"Reconstruction gallery, approach {ap} (held-out test patches; linear display range set by the"
                     " noise-free source)", fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / f"exp_c_gallery_{ap[0]}.png", dpi=110)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    ax = axes[0]
    for ap, col in zip(APPROACHES, ("tab:red", "tab:blue", "tab:green")):
        ax.semilogx(np.arange(1, 129), evs[ap][:128], color=col, label=ap)
    ax.set_xlabel("components k")
    ax.set_ylabel("cumulative explained variance (own feature space)")
    ax.set_title("Cumulative explained variance")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    ax = axes[1]
    for ap, col in zip(APPROACHES, ("tab:red", "tab:blue", "tab:green")):
        ax.loglog(KS, [curves[(ap, k)][1] for k in KS], "o-", color=col, label=f"{ap}: vs observed")
        ax.loglog(KS, [curves[(ap, k)][2] for k in KS], "s--", color=col, alpha=0.7, label=f"{ap}: vs noise-free")
    ax.axhline(np.mean(t["B"] + t["RN"] ** 2), color="k", ls=":", label="noise variance")
    ax.set_xlabel("components k")
    ax.set_ylabel("test reconstruction MSE [e-^2]")
    ax.set_title("Held-out reconstruction error (pixel units)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    ax = axes[2]
    for ap, col in zip(APPROACHES, ("tab:red", "tab:blue", "tab:green")):
        ax.semilogx(KS, [table[(ap, k)][0] for k in KS], "o-", color=col, label=f"{ap}: bright stars")
        ax.semilogx(KS, [table[(ap, k)][1] for k in KS], "s--", color=col, alpha=0.7, label=f"{ap}: faint stars")
    ax.axhline(s0[0], color="k", ls=":", label="original patch, bright")
    ax.axhline(s0[1], color="gray", ls=":", label="original patch, faint")
    ax.set_xlabel("components k")
    ax.set_ylabel("median centroid error [px]")
    ax.set_title("Centroid error after reconstruction")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "exp_c_curves.png", dpi=130)
    plt.close(fig)
    for f in ("exp_c_gallery_A.png", "exp_c_gallery_B.png", "exp_c_gallery_C.png", "exp_c_curves.png"):
        print("  saved", OUT / f)


if __name__ == "__main__":
    main()
