"""
Experiment B: what did the CNN learn? (kept short, interpreted cautiously)

1. First-layer 5x5 filters, compared with Module 11's Gaussian matched filter (FWHM 3 -> sigma 1.27 px,
   zero-mean over the 5x5 support): cosine similarity of each filter with that kernel.
2. Layer-1 activation maps for four held-out TEST patches: bright star, faint star, hot pixel, elongated.
3. Mean layer-1 response at the patch centre by subtype (do different filters prefer different artifacts?).
4. Controlled probes on synthetic patches (Gaussian noise of the right RMS plus one source):
   P(star) versus source width sigma at fixed integrated flux (morphology), and versus flux for a
   fixed-width star and for a single-pixel spike (brightness).  These isolate morphology from brightness.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cnn import center_crop, load, transform
from data12 import FWHM_TO_SIGMA, OUT, SPLITS, SUBTYPES, build, pixel_integrated_gaussian


def gaussian_kernel(size=5, sigma=3.0 * FWHM_TO_SIGMA):
    r = np.arange(size) - size // 2
    k = np.exp(-0.5 * (r[:, None] ** 2 + r[None, :] ** 2) / sigma**2)
    return k - k.mean()


def cos(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def probe_scores(model, pre, patches):
    x = torch.from_numpy(transform(center_crop(patches), pre["mean"], pre["std"])[:, None])
    with torch.inference_mode():
        return torch.sigmoid(model(x)).numpy()


def synth(rng, n, sigma, flux, spike=False, bkg=100.0, rn=5.0):
    rms = np.sqrt(bkg + rn**2)
    P = rng.normal(0, 1, (n, 36, 36)).astype(np.float32)
    for k in range(n):
        x0, y0 = 18 + rng.uniform(-0.5, 0.5), 18 + rng.uniform(-0.5, 0.5)
        if spike:
            P[k, 18, 18] += flux / rms
        else:
            P[k] += pixel_integrated_gaussian((36, 36), x0, y0, sigma, flux) / rms
    return P


def main():
    model, pre, ck = load(OUT / "cnn_best.pt")
    thr_p = 1 / (1 + np.exp(-ck["threshold_logit"]))
    W = model.features[0].weight.detach().numpy()[:, 0]          # (8, 5, 5)
    g = gaussian_kernel()
    sims = [cos(w, g) for w in W]
    print("Experiment B: what the CNN learned")
    print("  first-layer filters, cosine similarity with Module 11's zero-mean Gaussian matched filter (sigma 1.27 px):")
    print("   " + "  ".join(f"f{i}:{s:+.2f}" for i, s in enumerate(sims)))

    te = build(SPLITS["test"], "nominal")
    rng = np.random.default_rng(3)
    picks = {}
    for s in ("bright star", "faint star", "hot pixel", "elongated"):
        idx = np.nonzero(te["subtype"] == s)[0]
        if s == "faint star":
            idx = idx[(te["snr"][idx] > 8) & (te["snr"][idx] < 12)]
        picks[s] = int(rng.choice(idx))
    x_all = torch.from_numpy(transform(center_crop(te["X"]), pre["mean"], pre["std"])[:, None])
    with torch.inference_mode():
        a1_all = torch.relu(model.features[0](x_all)).numpy()       # (N, 8, 32, 32)
        p_all = torch.sigmoid(model(x_all)).numpy()
    centre = a1_all[:, :, 14:19, 14:19].max(axis=(2, 3))            # peak response near the candidate
    print("\n  mean layer-1 peak response near the candidate, by subtype (test patches):")
    print("    " + f"{'subtype':<15}" + "".join(f"{'f' + str(i):>7}" for i in range(8)))
    for s in SUBTYPES:
        m = te["subtype"] == s
        if m.sum():
            print("    " + f"{s:<15}" + "".join(f"{v:>7.2f}" for v in centre[m].mean(0)))

    # probes
    sig_grid = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.27, 1.5, 1.8, 2.2, 2.76, 3.3])
    flux_grid = np.geomspace(100, 30000, 12)
    width_curves = {}
    for F in (500.0, 3000.0):
        width_curves[F] = [np.mean(probe_scores(model, pre, synth(rng, 200, s, F)) >= thr_p) for s in sig_grid]
    star_flux = [np.mean(probe_scores(model, pre, synth(rng, 200, 1.27, F)) >= thr_p) for F in flux_grid]
    spike_flux = [np.mean(probe_scores(model, pre, synth(rng, 200, 1.27, F, spike=True)) >= thr_p) for F in flux_grid]
    noise_only = np.mean(probe_scores(model, pre, synth(rng, 500, 1.27, 0.0)) >= thr_p)
    print(f"\n  probes (fraction accepted at the frozen threshold P >= {thr_p:.3f}; 200 noisy synthetic patches each):")
    print("    source sigma (px)   " + "".join(f"{s:>6.2f}" for s in sig_grid))
    for F, c in width_curves.items():
        print(f"    flux {F:>6.0f} e-       " + "".join(f"{v:>6.2f}" for v in c))
    print("    flux (e-)           " + "".join(f"{f:>6.0f}" for f in flux_grid))
    print("    star sigma 1.27     " + "".join(f"{v:>6.2f}" for v in star_flux))
    print("    single-pixel spike  " + "".join(f"{v:>6.2f}" for v in spike_flux))
    print(f"    pure noise patch accepted: {noise_only:.3f}")

    # figure
    fig = plt.figure(figsize=(14, 9.5))
    gs = fig.add_gridspec(5, 10, hspace=0.45, wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(g, cmap="RdBu_r")
    ax.set_title("M11 Gaussian\n(zero-mean)", fontsize=8)
    ax.axis("off")
    for i in range(8):
        ax = fig.add_subplot(gs[0, i + 1])
        v = np.abs(W[i]).max()
        ax.imshow(W[i], cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_title(f"f{i}  cos {sims[i]:+.2f}", fontsize=8)
        ax.axis("off")
    for r, (s, k) in enumerate(picks.items(), start=1):
        ax = fig.add_subplot(gs[r, 0])
        ax.imshow(center_crop(te["X"][k]), cmap="gray", vmin=-3, vmax=np.percentile(te["X"][k], 99.5))
        ax.set_title(f"{s}\nP={p_all[k]:.3f}", fontsize=8)
        ax.axis("off")
        for i in range(8):
            ax = fig.add_subplot(gs[r, i + 1])
            ax.imshow(a1_all[k, i], cmap="magma")
            ax.axis("off")
    ax = fig.add_subplot(gs[1:3, 9])
    ax.axis("off")
    ax.text(0, 0.5, "rows 2-5:\nlayer-1\nactivation\nmaps (ReLU)\nper filter", fontsize=8, va="center")
    ax = fig.add_subplot(gs[3:5, 9])
    ax.axis("off")
    fig.suptitle("Experiment B: first-layer filters (top) and their activation maps on held-out patches", fontsize=11)
    fig.savefig(OUT / "exp_b_filters.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for F, c in width_curves.items():
        axs[0].plot(sig_grid, c, "o-", label=f"flux {F:.0f} e-")
    axs[0].axvline(3.0 * FWHM_TO_SIGMA, color="k", ls=":", label="training PSF (FWHM 3)")
    axs[0].axvline(6.5 * FWHM_TO_SIGMA, color="r", ls=":", label="strong defocus (FWHM 6.5)")
    axs[0].set_xlabel("source Gaussian sigma (px)")
    axs[0].set_ylabel("fraction accepted as star")
    axs[0].set_title("morphology probe")
    axs[0].legend(fontsize=8)
    axs[1].semilogx(flux_grid, star_flux, "o-", label="star (sigma 1.27)")
    axs[1].semilogx(flux_grid, spike_flux, "s-", label="single-pixel spike")
    axs[1].set_xlabel("integrated flux (e-)")
    axs[1].set_title("brightness probe")
    axs[1].legend(fontsize=8)
    for a in axs:
        a.grid(alpha=0.3)
        a.set_ylim(-0.03, 1.03)
    fig.tight_layout()
    fig.savefig(OUT / "exp_b_probes.png", dpi=120)
    plt.close(fig)
    print("  saved", OUT / "exp_b_filters.png", "and", OUT / "exp_b_probes.png")


if __name__ == "__main__":
    main()
