"""
Experiment A: generate star and artifact features and look at where the classes overlap.

Outputs: per-subtype feature summary, best single-feature threshold accuracy, histograms by class
and subtype, correlation matrices (all / stars / artifacts), and three class-separation scatters.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from classifiers import OUT
from star_data import (ARTIFACT_TYPES, DETECT_SNR, ENG_NAMES, FEATURE_NAMES, STAR_TYPES,
                       engineered, generate_candidates)

SUB_COLORS = {"bright star": "tab:blue", "faint star": "tab:cyan", "hot pixel": "tab:red",
              "diffuse": "tab:orange", "noise peak": "tab:gray", "compact irregular": "tab:purple",
              "ghost": "tab:brown"}


def best_threshold_accuracy(f, y):
    qs = np.quantile(f, np.linspace(0.005, 0.995, 199))
    acc = [max(np.mean((f > t) == y), np.mean((f <= t) == y)) for t in qs]
    i = int(np.argmax(acc))
    return acc[i], qs[i]


def main():
    OUT.mkdir(exist_ok=True)
    X, y, sub = generate_candidates(20000, np.random.default_rng(400), return_subtypes=True)
    E = engineered(X)
    print(f"Experiment A: {X.shape[0]} detected candidates, star fraction {y.mean():.3f},"
          f" detection threshold SNR >= {DETECT_SNR}")
    print("  median [10th - 90th percentile] of each measured feature by subtype:")
    print(f"    {'subtype':<18}{'n':>6}" + "".join(f"{n:>24}" for n in FEATURE_NAMES))
    for t in STAR_TYPES + ARTIFACT_TYPES:
        m = sub == t
        if not m.any():
            continue
        cells = "".join(f"{np.median(X[m, i]):>9.2f} [{np.percentile(X[m, i], 10):.2f}-"
                        f"{np.percentile(X[m, i], 90):.2f}]".rjust(24) for i in range(5))
        print(f"    {t:<18}{m.sum():>6}{cells}")

    print("\n  Best single-feature threshold accuracy (majority-class rate"
          f" = {max(y.mean(), 1 - y.mean()):.3f}):")
    for i, n in enumerate(ENG_NAMES):
        a, t = best_threshold_accuracy(E[:, i], y)
        print(f"    {n:<16}{a:.3f}")

    # Overlap regions, quantified
    faint, noise = sub == "faint star", sub == "noise peak"
    snr = X[:, 4]
    print("\n  Where the populations overlap:")
    print(f"    faint stars with SNR < 6: {np.mean(snr[faint] < 6):.0%};"
          f" noise peaks with SNR < 6: {np.mean(snr[noise] < 6):.0%}")
    w = X[:, 1]
    band = (w > 0.9) & (w < 1.8)
    print(f"    width in the stellar band 0.9-1.8 px: stars {np.mean(band[y == 1]):.0%},"
          f" noise peaks {np.mean(band[noise]):.0%}, compact irregular"
          f" {np.mean(band[sub == 'compact irregular']):.0%}, hot pixels"
          f" {np.mean(band[sub == 'hot pixel']):.0%}, diffuse {np.mean(band[sub == 'diffuse']):.0%}")
    r = X[:, 2]
    print(f"    roundness > 0.8: stars {np.mean(r[y == 1] > 0.8):.0%}, hot pixels"
          f" {np.mean(r[sub == 'hot pixel'] > 0.8):.0%}, noise peaks {np.mean(r[noise] > 0.8):.0%},"
          f" faint stars {np.mean(r[faint] > 0.8):.0%}, bright stars"
          f" {np.mean(r[sub == 'bright star'] > 0.8):.0%}")
    hot = sub == "hot pixel"
    print(f"    measured SNR > 20: bright stars {np.mean(snr[sub == 'bright star'] > 20):.0%}"
          f" (subtype defined on TRUE SNR >= 20), faint stars {np.mean(snr[faint] > 20):.0%},"
          f" hot pixels"
          f" {np.mean(snr[hot] > 20):.0%}; compact irregular"
          f" {np.mean(snr[sub == 'compact irregular'] > 20):.0%}")

    # ------------------------------------------------------------------ histograms
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    for i, n in enumerate(ENG_NAMES):
        lo, hi = np.percentile(E[:, i], [0.5, 99.5])
        bins = np.linspace(lo, hi, 50)
        ax = axes[0, i]
        ax.hist(E[y == 1, i], bins=bins, alpha=0.55, density=True, color="tab:blue", label="stars")
        ax.hist(E[y == 0, i], bins=bins, alpha=0.55, density=True, color="tab:red", label="artifacts")
        ax.set_title(f"{n} by class")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax = axes[1, i]
        for t in STAR_TYPES + ARTIFACT_TYPES[:-1]:
            m = sub == t
            ax.hist(E[m, i], bins=bins, histtype="step", lw=1.6, density=True,
                    color=SUB_COLORS[t], label=t)
        ax.set_title(f"{n} by subtype (normalized)")
        ax.set_xlabel(n)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)
    fig.suptitle("Experiment A: feature distributions (log features are what the classifiers use)")
    fig.tight_layout()
    fig.savefig(OUT / "exp_a_histograms.png", dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------------ correlations
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for ax, (lab, m) in zip(axes, (("all candidates", np.ones_like(y, bool)), ("stars", y == 1),
                                   ("artifacts", y == 0))):
        C = np.corrcoef(E[m].T)
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
        for a in range(5):
            for b in range(5):
                ax.text(b, a, f"{C[a, b]:.2f}", ha="center", va="center", fontsize=9)
        ax.set_xticks(range(5), ENG_NAMES, rotation=30, ha="right")
        ax.set_yticks(range(5), ENG_NAMES)
        ax.set_title(f"correlation: {lab}")
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(OUT / "exp_a_correlation.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    C = np.corrcoef(E.T)
    print("\n  Strongest correlations (all candidates, engineered features):")
    pairs = sorted(((abs(C[a, b]), a, b) for a in range(5) for b in range(a + 1, 5)), reverse=True)
    for v, a, b in pairs[:4]:
        print(f"    {ENG_NAMES[a]} ~ {ENG_NAMES[b]}: {C[a, b]:+.2f}")

    # ------------------------------------------------------------------ scatterplots
    rng = np.random.default_rng(0)
    show = rng.choice(X.shape[0], 6000, replace=False)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))
    specs = ((1, 4, "log width", "log SNR"), (1, 2, "log width", "roundness"),
             (0, 4, "log peak", "log SNR"))
    for ax, (a, b, xl, yl) in zip(axes, specs):
        for t in ("noise peak", "diffuse", "compact irregular", "hot pixel", "faint star",
                  "bright star"):
            m = show[sub[show] == t]
            ax.scatter(E[m, a], E[m, b], s=4, alpha=0.45, color=SUB_COLORS[t], label=t)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(alpha=0.3)
    axes[0].axhline(np.log(DETECT_SNR), color="k", ls=":", lw=1)
    axes[0].set_title("Width vs SNR: stars occupy a middle WIDTH band\n(hot pixels narrower,"
                      " diffuse broader); faint stars sit on the noise peaks")
    axes[1].set_title("Width vs roundness: irregular artifacts are elongated;\nfaint-star"
                      " roundness is smeared by measurement noise")
    axes[2].set_title("Peak vs SNR: physically linked (SNR ~ w (P - B) / sigma),\nthe band"
                      " width comes from background and source width")
    axes[0].legend(fontsize=8, markerscale=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "exp_a_scatter.png", dpi=130)
    plt.close(fig)
    for f in ("exp_a_histograms.png", "exp_a_correlation.png", "exp_a_scatter.png"):
        print(f"  saved {OUT / f}")


if __name__ == "__main__":
    main()
