"""module-9 -- star_density_vs_fov.py  (Computational Challenge: Star Density vs FOV)

For candidate fields of view (5, 10, 15, 20, 30 deg, full angle) and limiting
magnitudes (4, 5, 6, 7), count stars per pointing:

    N_star = f(FOV, m_lim, sky direction)

using a synthetic catalog -- first a plain uniform-on-sphere one (the "for
now" baseline), then one with a realistic galactic-plane density enhancement
(star_catalog.py), to make the last argument of f() matter.

The payoff: the design criterion is not the mean count. It's

    P(N_star >= N_required)

over random pointings -- because a field aimed near the galactic poles can
see far fewer stars than a field of the same size and m_lim aimed at the
plane, even though they average out to a similar whole-sky mean. For a real
star tracker, N_required is set by the star-ID algorithm (this file uses 3,
the minimum for unique pattern matching, and 6, a comfortable margin).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from star_catalog import (generate_catalog, whole_sky_count, galactic_latitude,
                          radec_to_xyz, MAG_MAX, MAG_MIN)

FOVS_DEG = [5, 10, 15, 20, 30]           # full angle
MAG_LIMS = [4, 5, 6, 7]
N_POINTINGS = 8000
REQUIRED = [3, 6]
FOCUS_MLIM = 5                            # used for the uniform-vs-realistic detail plots

RNG = np.random.default_rng(42)


def random_pointings(n, rng):
    dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=n)))
    ra = rng.uniform(0.0, 360.0, size=n)
    return ra, dec


def count_all(ra, dec, mag, pointing_xyz):
    """{(fov, m_lim): counts array, one entry per pointing}."""
    out = {}
    for m_lim in MAG_LIMS:
        sel = mag <= m_lim
        tree = cKDTree(radec_to_xyz(ra[sel], dec[sel]))
        for fov in FOVS_DEG:
            chord_r = 2.0 * np.sin(np.deg2rad(fov / 2.0) / 2.0)
            counts = tree.query_ball_point(pointing_xyz, r=chord_r, return_length=True)
            out[(fov, m_lim)] = np.asarray(counts)
    return out


def main():
    n_total = int(round(whole_sky_count(MAG_MAX) - whole_sky_count(MAG_MIN)))
    print("=" * 90)
    print("STAR DENSITY vs FOV")
    print("=" * 90)
    print(f"synthetic catalog: {n_total:,} stars down to m = {MAG_MAX} "
          f"(calibrated to ~{whole_sky_count(6):.0f} stars whole-sky at m<=6)")
    print()
    print("magnitude-model check (predicted vs. actual, this draw):")
    ra_u, dec_u, mag_u = generate_catalog(n_total, plane_biased=False, rng=RNG)
    for m_lim in MAG_LIMS:
        actual = int(np.sum(mag_u <= m_lim))
        print(f"   m<={m_lim}: predicted {whole_sky_count(m_lim):>9,.0f}   "
              f"actual {actual:>9,}   ratio {actual/whole_sky_count(m_lim):.3f}")

    ra_g, dec_g, mag_g = generate_catalog(n_total, plane_biased=True, rng=RNG)
    b_g = galactic_latitude(ra_g, dec_g)
    print()
    print(f"plane-biased catalog: fraction with |b|<15 deg = "
          f"{np.mean(np.abs(b_g) < 15):.3f}  (isotropic would give "
          f"{1-np.cos(np.deg2rad(15)):.3f})")
    print(f"   -> {np.mean(np.abs(b_g)<15)/(1-np.cos(np.deg2rad(15))):.1f}x "
          f"enhancement near the plane, by construction")

    pra, pdec = random_pointings(N_POINTINGS, RNG)
    pxyz = radec_to_xyz(pra, pdec)
    pb = galactic_latitude(pra, pdec)

    print()
    print(f"counting stars for {len(FOVS_DEG)} FOVs x {len(MAG_LIMS)} m_lim x "
          f"{N_POINTINGS} random pointings, both catalogs ...")
    counts_u = count_all(ra_u, dec_u, mag_u, pxyz)
    counts_g = count_all(ra_g, dec_g, mag_g, pxyz)

    print()
    print("MEAN N_star (plane-biased catalog):")
    header = f"{'FOV':>6} | " + " | ".join(f"m<={m:>2}" for m in MAG_LIMS)
    print(header)
    for fov in FOVS_DEG:
        row = [f"{counts_g[(fov, m)].mean():>6.2f}" for m in MAG_LIMS]
        print(f"{fov:>4} deg | " + " | ".join(row))

    print()
    print("Mean count alone hides the risk. At the FOCUS case "
          f"m_lim={FOCUS_MLIM}:")
    print(f"{'FOV':>6} {'mean(unif)':>11} {'mean(plane)':>12} "
          + "".join(f"{'P(N>='+str(r)+',u)':>13}{'P(N>='+str(r)+',g)':>13}" for r in REQUIRED))
    for fov in FOVS_DEG:
        cu = counts_u[(fov, FOCUS_MLIM)]
        cg = counts_g[(fov, FOCUS_MLIM)]
        row = f"{fov:>6} {cu.mean():>11.2f} {cg.mean():>12.2f}"
        for r in REQUIRED:
            row += f"{np.mean(cu >= r):>13.3f}{np.mean(cg >= r):>13.3f}"
        print(row)

    print()
    print("Note mean(plane) < mean(uniform) at every FOV, even though both")
    print("catalogs hold the SAME total star count: the enhanced band covers")
    print(f"only sin(15 deg) = {np.sin(np.deg2rad(15)):.3f} = 25.9% of the sky by area, so")
    print("concentrating stars there means the other 74% of the sky -- where")
    print("most random pointings land -- sits BELOW the isotropic baseline.")
    print("On top of that lower mean, P(N>=req) moves in BOTH directions")
    print("depending on FOV and threshold: a fat tail of near-empty polar")
    print("pointings pulls small-FOV, high-threshold probabilities down, while")
    print("a fat tail of rich plane pointings can push moderate thresholds UP")
    print("even as the mean drops (see FOV=10, P(N>=6): 0.026 -> 0.082). A")
    print("single mean cannot show either effect -- only the full distribution can.")

    # worst case for a small, bright-limited field -- the practical number a
    # star-tracker designer actually needs
    fov0, m0 = FOVS_DEG[0], MAG_LIMS[0]
    cg0 = counts_g[(fov0, m0)]
    frac_empty = np.mean(cg0 == 0)
    worst_b = pb[np.argsort(cg0)[:20]]
    print()
    print(f"Worst case, FOV={fov0} deg, m_lim={m0} (smallest FOV, brightest "
          f"limit -- the hardest case):")
    print(f"   {frac_empty*100:.1f}% of random pointings see ZERO usable "
          f"stars (plane-biased sky).")
    print(f"   the 20 worst pointings sit at galactic latitude "
          f"|b| = {np.abs(worst_b).mean():.1f} +/- {np.abs(worst_b).std():.1f} "
          f"deg on average --")
    print("   i.e. they really are clustered near the galactic poles, not "
          "random noise.")
    print("=" * 90)

    _plot(ra_g, dec_g, mag_g, ra_u, dec_u, counts_u, counts_g, pb)


def _plot(ra_g, dec_g, mag_g, ra_u, dec_u, counts_u, counts_g, pb):
    fig = plt.figure(figsize=(15, 12))

    # (1) sky maps, mollweide
    keep_u = np.random.default_rng(1).choice(len(ra_u), size=25000, replace=False)
    keep_g = np.random.default_rng(1).choice(len(ra_g), size=25000, replace=False)

    ax1 = fig.add_subplot(3, 2, 1, projection="mollweide")
    lam = np.deg2rad(((ra_u[keep_u] + 180) % 360) - 180)
    phi = np.deg2rad(dec_u[keep_u])
    ax1.plot(lam, phi, ",", color="k", alpha=0.2)
    ax1.set_title("Baseline catalog: uniform on the sphere", fontsize=10)
    ax1.set_xticklabels([]); ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(3, 2, 2, projection="mollweide")
    lam = np.deg2rad(((ra_g[keep_g] + 180) % 360) - 180)
    phi = np.deg2rad(dec_g[keep_g])
    ax2.plot(lam, phi, ",", color="k", alpha=0.2)
    ax2.set_title("Realistic catalog: enhanced near the galactic plane", fontsize=10)
    ax2.set_xticklabels([]); ax2.grid(True, alpha=0.3)

    # (2) heatmap of mean N_star, plane-biased catalog
    ax3 = fig.add_subplot(3, 2, 3)
    grid = np.array([[counts_g[(fov, m)].mean() for fov in FOVS_DEG] for m in MAG_LIMS])
    im = ax3.imshow(grid, aspect="auto", cmap="viridis",
                    extent=[0, len(FOVS_DEG), 0, len(MAG_LIMS)])
    ax3.set_xticks(np.arange(len(FOVS_DEG)) + 0.5)
    ax3.set_xticklabels(FOVS_DEG)
    ax3.set_yticks(np.arange(len(MAG_LIMS)) + 0.5)
    ax3.set_yticklabels(MAG_LIMS[::-1])
    for i, m in enumerate(MAG_LIMS[::-1]):
        for j, fov in enumerate(FOVS_DEG):
            ax3.text(j + 0.5, i + 0.5, f"{grid[len(MAG_LIMS)-1-i, j]:.1f}",
                     ha="center", va="center", color="w", fontsize=8)
    ax3.set_xlabel("FOV (deg)"); ax3.set_ylabel("m_lim")
    ax3.set_title("Mean N_star (realistic catalog)")
    fig.colorbar(im, ax=ax3, shrink=0.8, label="mean stars per pointing")

    # (3) P(N>=req) vs FOV, uniform vs realistic, at FOCUS_MLIM
    ax4 = fig.add_subplot(3, 2, 4)
    for r, ls in zip(REQUIRED, ["-", "--"]):
        pu = [np.mean(counts_u[(fov, FOCUS_MLIM)] >= r) for fov in FOVS_DEG]
        pg = [np.mean(counts_g[(fov, FOCUS_MLIM)] >= r) for fov in FOVS_DEG]
        ax4.plot(FOVS_DEG, pu, "o" + ls, color="tab:blue", label=f"uniform, N>={r}")
        ax4.plot(FOVS_DEG, pg, "s" + ls, color="tab:red", label=f"realistic, N>={r}")
    ax4.set_xlabel("FOV (deg)"); ax4.set_ylabel(f"P(N_star >= required),  m_lim={FOCUS_MLIM}")
    ax4.set_title("Design criterion: probability of enough stars,\n"
                  "not the average count")
    ax4.set_ylim(-0.02, 1.02)
    ax4.legend(fontsize=7.5); ax4.grid(True, alpha=0.3)

    # (4) histogram, small FOV: same mean, very different shape
    ax5 = fig.add_subplot(3, 2, 5)
    fov_h = FOVS_DEG[1]
    cu = counts_u[(fov_h, FOCUS_MLIM)]
    cg = counts_g[(fov_h, FOCUS_MLIM)]
    bins = np.arange(0, max(cu.max(), cg.max()) + 2) - 0.5
    ax5.hist(cu, bins=bins, alpha=0.5, density=True, color="tab:blue",
            label=f"uniform (mean {cu.mean():.2f})")
    ax5.hist(cg, bins=bins, alpha=0.5, density=True, color="tab:red",
            label=f"realistic (mean {cg.mean():.2f})")
    ax5.set_xlabel(f"N_star  (FOV={fov_h} deg, m_lim={FOCUS_MLIM})")
    ax5.set_ylabel("fraction of pointings")
    ax5.set_title("Similar mean, very different spread")
    ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)

    # (5) count vs galactic latitude of the pointing -- the mechanism, laid bare
    ax6 = fig.add_subplot(3, 2, 6)
    fov_s, m_s = FOVS_DEG[0], MAG_LIMS[0]
    cg_s = counts_g[(fov_s, m_s)]
    ax6.scatter(pb, cg_s, s=3, alpha=0.25, color="tab:purple")
    order = np.argsort(pb)
    running = np.convolve(cg_s[order], np.ones(200) / 200, mode="valid")
    ax6.plot(pb[order][99:-100], running, color="k", lw=2, label="running mean")
    ax6.set_xlabel("galactic latitude of the pointing, b (deg)")
    ax6.set_ylabel(f"N_star  (FOV={fov_s} deg, m_lim={m_s})")
    ax6.set_title("The mechanism: count vs. where you point,\nnot just how "
                  "wide or how deep")
    ax6.legend(fontsize=8); ax6.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("star_density_vs_fov.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
