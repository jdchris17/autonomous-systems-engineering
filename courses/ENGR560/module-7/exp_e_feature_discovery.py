"""
Experiment E: unsupervised astronomical feature discovery.

Representations (fit on DEVELOPMENT images only; every column standardised with development stats):
  engineered 3    log width, roundness, log SNR
  PCA-B(8/16)     background-subtracted PCA scores (Module 6 approach B)
  PCA-C(8/16)     flux-normalised PCA scores       (Module 6 approach C)
  hybrid          engineered 3 + PCA-C(8)
Models (no labels): K-means (K = 6, best of 10 K-means++ starts), full-covariance GMM (K = 6, covariance
floor 0.05 as in Experiment C, best of 3), DBSCAN (min_samples 10, eps = 90th percentile of the
10th-neighbour distance in that representation).
K = 6 is a deliberate, label-free choice to make the representations comparable (it is also the
number of simulated subtypes; Experiment C's likelihood-based choice for the engineered features was 8).

Labels and metadata are used only afterwards: contingency tables, ARI, AMI, purity (with cluster
count), stability across seeds, generalisation to the untouched test images (K-means / GMM predict),
and associations with nuisance variables (SNR, background, focus, centroid offset, image id).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.neighbors import NearestNeighbors

from clustering import GaussianMixtureScratch, KMeansScratch
from data7 import ENG_NAMES, OUT, SUBTYPES, load, standardize

K = 6


def reps(z):
    out = {}
    for tag in ("dev", "test"):
        e = z[f"{tag}_eng"]
        out.setdefault("engineered 3", []).append(e)
        for ap in ("B", "C"):
            for k in (8, 16):
                out.setdefault(f"PCA-{ap}({k})", []).append(z[f"{tag}_pca{ap}"][:, :k])
        out.setdefault("hybrid (eng 3 + PCA-C 8)", []).append(np.hstack([e, z[f"{tag}_pcaC"][:, :8]]))
    return {name: standardize(dev, test) for name, (dev, test) in out.items()}


def purity(truth, lab):
    keep = lab != -1
    t, l = truth[keep], lab[keep]
    return sum(np.max(np.unique(t[l == c], return_counts=True)[1]) for c in np.unique(l)) / len(truth)


def quint(v):
    return np.digitize(v, np.quantile(v, [0.2, 0.4, 0.6, 0.8]))


def fit_rep(name, Xd, Xt):
    km_runs = [KMeansScratch(K, seed=s).fit(Xd) for s in range(10)]
    km = min(km_runs, key=lambda m: m.inertia_)
    km_stab = np.mean([ari(km_runs[i].labels_, km_runs[j].labels_) for i in range(10) for j in range(i + 1, 10)])
    gm_runs = [GaussianMixtureScratch(K, "full", reg_covar=0.05, seed=s, max_iter=300).fit(Xd) for s in range(3)]
    gm = max(gm_runs, key=lambda m: m.lower_bound_)
    gm_stab = np.mean([ari(gm_runs[i].predict(Xd), gm_runs[j].predict(Xd)) for i in range(3) for j in range(i + 1, 3)])
    eps = float(np.quantile(NearestNeighbors(n_neighbors=10).fit(Xd).kneighbors(Xd)[0][:, -1], 0.9))
    db = DBSCAN(eps=eps, min_samples=10).fit_predict(Xd)
    return {"K-means": (km.labels_, km.predict(Xt), km_stab, km),
            "GMM": (gm.predict(Xd), gm.predict(Xt), gm_stab, gm),
            "DBSCAN": (db, None, np.nan, eps)}


def main():
    OUT.mkdir(exist_ok=True)
    z = load()
    R = reps(z)
    sub, sub_t, y = z["dev_subtype"], z["test_subtype"], z["dev_y"]
    print("Experiment E: unsupervised feature discovery (labels used only after fitting)")
    print(f"  PCA explained variance of the first 8 / 16 components (development): B {z['evrB'][:8].sum():.3f} /"
          f" {z['evrB'][:16].sum():.3f}, C {z['evrC'][:8].sum():.3f} / {z['evrC'][:16].sum():.3f}")
    fits = dict(zip(R, Parallel(n_jobs=len(R))(delayed(fit_rep)(n, *R[n]) for n in R)))

    # ---------------------------------------------------------------- Part 3: agreement
    print("\n  Part 3 - external agreement (development patches unless noted)")
    print(f"    {'representation':<26}{'model':<9}{'clusters':>9}{'noise':>7}{'ARI':>7}{'AMI':>7}{'purity':>8}"
          f"{'AMI star/art':>13}{'stability':>11}{'test ARI':>10}")
    summary = {}
    for name in R:
        for model, (lab, lab_t, stab, _) in fits[name].items():
            nc = len(set(lab)) - (-1 in lab)
            row = (ari(sub, lab), ami(sub, lab), purity(sub, lab), ami(y, lab))
            summary[(name, model)] = row
            ta = f"{ari(sub_t, lab_t):.3f}" if lab_t is not None else "-"
            st = f"{stab:.3f}" if not np.isnan(stab) else "-"
            print(f"    {name:<26}{model:<9}{nc:>9}{np.mean(lab == -1):>7.1%}{row[0]:>7.3f}{row[1]:>7.3f}{row[2]:>8.3f}"
                  f"{row[3]:>13.3f}{st:>11}{ta:>10}")
    rng = np.random.default_rng(0)
    rand = rng.integers(0, 60, len(sub))
    print(f"    purity reference: 60 RANDOM clusters give purity {purity(sub, rand):.3f} (ARI {ari(sub, rand):.3f});"
          f" assigning every patch its own cluster gives purity 1.000 - purity alone rewards many small clusters")

    # ---------------------------------------------------------------- Part 4: morphology vs nuisance
    nuis = {"subtype": sub, "star / artifact": y, "true SNR quintile": quint(np.log(z["dev_snr_true"])),
            "background quintile": quint(z["dev_B"]), "focus (PSF sigma) quintile": quint(z["dev_img_sigma"]),
            "centroid offset quintile": quint(np.hypot(z["dev_x0"] - 15.5, z["dev_y0"] - 15.5)),
            "image id": z["dev_group"]}
    print("\n  Part 4 - what do the K-means clusters track? AMI between cluster membership and each variable")
    print(f"    {'representation':<26}" + "".join(f"{k[:18]:>20}" for k in nuis))
    for name in R:
        lab = fits[name]["K-means"][0]
        print(f"    {name:<26}" + "".join(f"{ami(v, lab):>20.3f}" for v in nuis.values()))

    # ---------------------------------------------------------------- Part 2: inspect structure
    best = max(R, key=lambda n: summary[(n, "K-means")][1])
    for name in sorted({best, "engineered 3", "PCA-B(8)"}):
        lab = fits[name]["K-means"][0]
        print(f"\n  Part 2 - K-means on {name}: contingency table and standardised engineered-feature means")
        print(f"    {'cluster':>8}{'n':>6}" + "".join(f"{s[:11]:>12}" for s in SUBTYPES)
              + "".join(f"{n[:10]:>11}" for n in ENG_NAMES) + f"{'median SNR':>12}")
        Eng, _ = standardize(z["dev_eng"])
        for c in np.argsort([-np.sum(lab == c) for c in range(K)]):
            m = lab == c
            print(f"    {c:>8}{m.sum():>6}" + "".join(f"{np.mean(sub[m] == s):>12.0%}" for s in SUBTYPES)
                  + "".join(f"{v:>11.2f}" for v in Eng[m].mean(0)) + f"{np.median(z['dev_snr_true'][m]):>12.1f}")

    plot_patches(z, fits, best)
    plot_summary(summary, R)


def plot_patches(z, fits, best):
    Xp = z["dev_X"].astype(float)
    for name in (best, "PCA-B(8)"):
        lab = fits[name]["K-means"][0]
        rng = np.random.default_rng(1)
        fig, axes = plt.subplots(K, 8, figsize=(12, 1.6 * K))
        for r, c in enumerate(np.argsort([-np.sum(lab == c) for c in range(K)])):
            idx = rng.choice(np.nonzero(lab == c)[0], min(8, np.sum(lab == c)), replace=False)
            for j in range(8):
                ax = axes[r, j]
                ax.set_xticks([])
                ax.set_yticks([])
                if j < len(idx):
                    p = Xp[idx[j]]
                    b = np.median(np.r_[p[:3].ravel(), p[-3:].ravel()])
                    q = p - b
                    s = max(np.percentile(np.abs(q), 99.5), 1e-6)
                    ax.imshow(np.arcsinh(q / (0.1 * s)), cmap="gray", origin="lower")
                    ax.set_title(z["dev_subtype"][idx[j]][:10], fontsize=6)
            axes[r, 0].set_ylabel(f"cluster {c}\nn={np.sum(lab == c)}", fontsize=8)
        fig.suptitle(f"Representative patches per K-means cluster: {name} (titles = true subtype, for interpretation only)")
        fig.tight_layout()
        safe = name.split(" ")[0].replace("(", "").replace(")", "")
        fig.savefig(OUT / f"exp_e_cluster_patches_{safe}.png", dpi=110)
        plt.close(fig)
        print(f"  saved {OUT / f'exp_e_cluster_patches_{safe}.png'}")


def plot_summary(summary, R):
    names = list(R)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    x = np.arange(len(names))
    for j, (model, col) in enumerate((("K-means", "tab:blue"), ("GMM", "tab:green"), ("DBSCAN", "tab:red"))):
        ax.bar(x + 0.27 * (j - 1), [summary[(n, model)][1] for n in names], 0.27, color=col, label=model)
    ax.set_xticks(x, names, rotation=15)
    ax.set_ylabel("AMI with physical subtype")
    ax.set_title("Which representation lets unsupervised models find the physical populations?")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "exp_e_representations.png", dpi=130)
    plt.close(fig)
    print(f"  saved {OUT / 'exp_e_representations.png'}")


if __name__ == "__main__":
    main()
