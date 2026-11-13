"""
Experiment A: generate astronomical image patches.

Outputs: contact sheet per subtype, class proportions (overall and per image), parameter
distributions, the grouped split, and a variance decomposition of the pixel data into
background, source signal (and within it: brightness, position, shape) and noise.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from patches import (CENTRE, SIZE, SUBTYPES, development_set, elliptical_source, untouched_test_set)

OUT = __import__("pathlib").Path(__file__).parent / "outputs"


def stretch(p):
    """Display stretch: background-subtract by the border median, then asinh."""
    b = np.median(np.concatenate([p[:3].ravel(), p[-3:].ravel(), p[:, :3].ravel(), p[:, -3:].ravel()]))
    q = p - b
    s = max(np.percentile(np.abs(q), 99.5), 1e-6)
    return np.arcsinh(q / (0.1 * s))


def centred_source(d, i):
    """Re-render source i at the patch centre (same flux and shape) - a position counterfactual."""
    if d["subtype"][i] == "hot pixel":
        img = np.zeros((SIZE, SIZE))
        img[16, 16] = d["amp"][i]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            img[16 + dy, 16 + dx] = 0.05 * d["amp"][i]
        return img
    return elliptical_source(d["amp"][i], d["s_major"][i], d["s_minor"][i], d["theta"][i], CENTRE, CENTRE)


def main():
    OUT.mkdir(exist_ok=True)
    d = development_set()
    t = untouched_test_set()
    print("Experiment A: synthetic 32 x 32 patches")
    for name, ds in (("development", d), ("untouched test", t)):
        print(f"  {name}: {len(np.unique(ds['group']))} images, {ds['y'].size} patches, star fraction"
              f" {ds['y'].mean():.3f}; per-image star fraction"
              f" {min(np.mean(ds['y'][ds['group'] == g]) for g in np.unique(ds['group'])):.2f}-"
              f"{max(np.mean(ds['y'][ds['group'] == g]) for g in np.unique(ds['group'])):.2f}")
    print("  The two sets share no image ids: every split in this module is by image.")
    print("\n  Class / subtype proportions (development):")
    for s in SUBTYPES:
        m = d["subtype"] == s
        print(f"    {s:<12}{m.sum():>6} ({m.mean():.1%})   true SNR median {np.median(d['snr_true'][m]):6.1f}"
              f" [10-90%: {np.percentile(d['snr_true'][m], 10):.1f}-{np.percentile(d['snr_true'][m], 90):.1f}],"
              f" sigma_major {np.median(d['s_major'][m]):.2f}, axis ratio {np.median(d['s_minor'][m] / d['s_major'][m]):.2f}")

    # ---------------------------------------------------------------- contact sheet
    rng = np.random.default_rng(0)
    n_show = 8
    fig, axes = plt.subplots(len(SUBTYPES), n_show, figsize=(n_show * 1.5, len(SUBTYPES) * 1.65))
    for r, s in enumerate(SUBTYPES):
        idx = rng.choice(np.nonzero(d["subtype"] == s)[0], n_show, replace=False)
        idx = idx[np.argsort(-d["snr_true"][idx])]
        for c, i in enumerate(idx):
            ax = axes[r, c]
            ax.imshow(stretch(d["X"][i]), cmap="gray", origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"SNR {d['snr_true'][i]:.0f}", fontsize=7)
            if c == 0:
                ax.set_ylabel(s, fontsize=9)
    fig.suptitle("Contact sheet: measured 32x32 patches (asinh stretch, sorted by true SNR)")
    fig.tight_layout()
    fig.savefig(OUT / "exp_a_contact_sheet.png", dpi=120)
    plt.close(fig)

    # ---------------------------------------------------------------- distributions
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    cols = dict(zip(SUBTYPES, ("tab:blue", "tab:cyan", "tab:red", "tab:orange", "tab:purple", "tab:gray")))
    specs = (("log10 flux [e-]", np.log10(d["flux"])), ("log10 true SNR", np.log10(d["snr_true"])),
             ("sigma major [px]", d["s_major"]), ("axis ratio", d["s_minor"] / d["s_major"]),
             ("centroid offset x [px]", d["x0"] - CENTRE))
    for ax, (lab, v) in zip(axes.ravel()[:5], specs):
        bins = np.linspace(np.percentile(v, 0.5), np.percentile(v, 99.5), 40)
        for s in SUBTYPES:
            ax.hist(v[d["subtype"] == s], bins=bins, histtype="step", lw=1.5, density=True, color=cols[s], label=s)
        ax.set_title(lab)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    ids = np.unique(d["group"])
    first = np.array([np.nonzero(d["group"] == g)[0][0] for g in ids])
    for ax, (lab, key) in zip(axes.ravel()[5:], (("image background B [e-/px]", "B"),
                                                 ("image read noise [e-]", "RN"),
                                                 ("image PSF sigma [px]", "img_sigma"))):
        ax.hist(d[key][first], bins=20, color="tab:green", alpha=0.7)
        ax.set_title(lab + " (one value per image)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "exp_a_parameters.png", dpi=120)
    plt.close(fig)

    # ---------------------------------------------------------------- variance decomposition
    N = d["y"].size
    Xf = d["X"].reshape(N, -1)
    B = np.repeat(d["B"][:, None], SIZE * SIZE, axis=1)
    S = d["source"].reshape(N, -1)
    eps = Xf - d["expected"].reshape(N, -1)
    def ss(A):
        return float(np.sum((A - A.mean(axis=0)) ** 2))
    tot = ss(Xf)
    print("\n  Engineering question - where does the pixel variance come from? (development set)")
    print("    Share of the total sum of squares around the mean image (cross terms in brackets):")
    for lab, A in (("background level (per image/patch)", B), ("source signal", S), ("noise", eps)):
        print(f"      {lab:<36}{ss(A) / tot:>7.1%}")
    cross = (tot - ss(B) - ss(S) - ss(eps)) / tot
    print(f"      {'(cross terms)':<36}{cross:>7.1%}")
    Sc = np.array([centred_source(d, i).ravel() for i in range(N)])
    flux = S.sum(axis=1)
    shape_only = Sc / np.where(flux > 0, flux, 1)[:, None] * np.median(flux)
    print("    Inside the source signal (counterfactuals):")
    print(f"      position (source minus the same source re-rendered at the centre): {ss(S - Sc) / ss(S):.1%} of source SS")
    print(f"      shape only (centred, every source scaled to the median flux):      {ss(shape_only) / ss(S):.1%} of source SS")
    print("      -> the remainder is dominated by brightness differences (flux spans"
          f" {np.percentile(flux, 1):.0f}-{np.percentile(flux, 99):.0f} e-, 1-99%)")
    print("  saved", OUT / "exp_a_contact_sheet.png")
    print("  saved", OUT / "exp_a_parameters.png")


if __name__ == "__main__":
    main()
