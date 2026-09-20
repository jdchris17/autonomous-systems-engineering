"""Square-wave reconstruction: watch harmonics assemble into a waveform, and
track edge sharpness, error away from the jumps, and Gibbs overshoot.

N below is the NUMBER of odd harmonics used, i.e. harmonics 1, 3, ..., 2N-1.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "module-8", "signal_toolkit"))
from fourier import fourier_synthesis, square_wave_coeffs

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
F0 = 1.0                       # 1 Hz, so time is measured in periods
N_LIST = [1, 3, 5, 10, 25, 50, 100]
GRID = 40000
AWAY = 0.05                    # "away from a jump" = more than 5% of a period from it
COLORS = plt.cm.viridis(np.linspace(0.0, 0.9, len(N_LIST)))


def reconstruct(n_harmonics, t):
    """Square wave from its first n_harmonics odd harmonics (k = 1, 3, ..., 2n-1)."""
    return fourier_synthesis(square_wave_coeffs(2 * n_harmonics - 1), F0, t)


def ideal_square(t):
    return np.sign(np.sin(2 * np.pi * F0 * t))


def rise_time(t, y):
    """10-90% rise time across the upward jump at t = 0, in periods."""
    win = (t > -0.25) & (t < 0.25)
    tw, yw = t[win], y[win]
    t_lo = tw[np.argmax(yw >= -0.8)]
    t_hi = tw[np.argmax(yw >= 0.8)]
    return t_hi - t_lo


if __name__ == "__main__":
    # Fine grid over one period, offset so no sample sits exactly on a jump.
    t = (np.arange(GRID) + 0.5) / GRID - 0.5
    ideal = ideal_square(t)
    dist_to_jump = np.minimum(np.abs(t), 0.5 - np.abs(t))
    away = dist_to_jump > AWAY

    results = []
    print(f"{'N':>4}{'top k':>7}{'rise 10-90% (period)':>22}{'RMSE away':>11}{'max err away':>14}{'overshoot (% of jump)':>23}")
    for N in N_LIST:
        y = reconstruct(N, t)
        rt = rise_time(t, y)
        err = y[away] - ideal[away]
        rmse = float(np.sqrt(np.mean(err**2)))
        max_err = float(np.max(np.abs(err)))
        overshoot = (y[(t > 0) & (t < 0.25)].max() - 1.0) / 2.0 * 100
        results.append((N, rt, rmse, max_err, overshoot))
        print(f"{N:>4}{2 * N - 1:>7}{rt:>22.4f}{rmse:>11.4f}{max_err:>14.4f}{overshoot:>23.2f}")
    print("Gibbs limit for the overshoot as N -> infinity: 8.95% of the jump (0.179 for a +/-1 wave)")

    # Figure 1: each reconstruction in its own panel.
    tp = (np.arange(3000) + 0.5) / 3000 * 1.5 - 0.75
    fig, axes = plt.subplots(4, 2, figsize=(14, 14), sharex=True, sharey=True)
    for ax, N, c in zip(axes.flatten(), N_LIST, COLORS):
        ax.plot(tp, ideal_square(tp), color="lightgray", linewidth=4, label="ideal")
        ax.plot(tp, reconstruct(N, tp), color=c, linewidth=1.5, label="reconstruction")
        ax.set_title(f"{N} odd harmonic{'s' if N > 1 else ''}  (k = 1 ... {2 * N - 1})")
        ax.grid(alpha=0.3)
    axes.flatten()[-1].axis("off")
    axes.flatten()[-2].legend(loc="lower right", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("t (periods)")
    fig.suptitle("Square wave from N odd harmonics")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "square_wave_reconstructions.png"), dpi=150)

    # Figure 2: zoom on the edge, all N overlaid.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    tz = (np.arange(6000) + 0.5) / 6000 * 0.1 - 0.05
    for N, c in zip(N_LIST, COLORS):
        axes[0].plot(tz, reconstruct(N, tz), color=c, linewidth=1.6, label=f"N = {N}")
    axes[0].plot(tz, ideal_square(tz), "k--", linewidth=1, label="ideal")
    axes[0].set_title("Zoom on the jump at t = 0: edge sharpens, overshoot stays")
    axes[0].set_xlabel("t (periods)")
    axes[0].legend(fontsize=8, ncol=2)
    tz2 = (np.arange(6000) + 0.5) / 6000 * 0.1 + 0.15
    for N, c in zip(N_LIST, COLORS):
        axes[1].plot(tz2, reconstruct(N, tz2) - 1.0, color=c, linewidth=1.6, label=f"N = {N}")
    axes[1].set_title("Error on the flat top (t = 0.15 ... 0.25, well away from the edge)")
    axes[1].set_xlabel("t (periods)")
    axes[1].set_ylabel("reconstruction - 1")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "square_wave_edge_zoom.png"), dpi=150)

    # Figure 3: the three tracked quantities vs N.
    Ns = np.array([r[0] for r in results])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].loglog(Ns, [r[1] for r in results], "o-")
    axes[0].set_title("Edge sharpness: 10-90% rise time (periods)")
    axes[1].loglog(Ns, [r[2] for r in results], "o-", label="RMSE")
    axes[1].loglog(Ns, [r[3] for r in results], "s--", label="max error")
    axes[1].set_title("Error away from the jumps")
    axes[1].legend()
    axes[2].semilogx(Ns, [r[4] for r in results], "o-")
    axes[2].axhline(8.95, color="firebrick", linestyle="--", label="Gibbs limit 8.95%")
    axes[2].set_ylim(0, 12)
    axes[2].set_title("Gibbs overshoot (% of jump)")
    axes[2].legend()
    for ax in axes:
        ax.set_xlabel("N (number of odd harmonics)")
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "square_wave_metrics.png"), dpi=150)

    # Figure 4: assembly. Individual harmonics on the left, running sum on the right.
    t_a = (np.arange(3000) + 0.5) / 3000 * 1.5 - 0.75
    fig, axes = plt.subplots(4, 2, figsize=(13, 11), sharex=True)
    running = np.zeros_like(t_a)
    for row, k in enumerate([1, 3, 5, 7]):
        c = square_wave_coeffs(k)
        comp = fourier_synthesis({k: c[k], -k: c[-k]}, F0, t_a)   # this harmonic alone
        running = running + comp
        axes[row, 0].plot(t_a, comp, color="darkorange", linewidth=1.8)
        axes[row, 0].set_title(f"harmonic k = {k}:  (4/({k}$\\pi$)) sin(2$\\pi${k}t)", fontsize=10)
        axes[row, 0].set_ylim(-1.4, 1.4)
        axes[row, 1].plot(t_a, ideal_square(t_a), color="lightgray", linewidth=4)
        axes[row, 1].plot(t_a, running, color="steelblue", linewidth=1.8)
        axes[row, 1].set_title(f"sum of harmonics 1 ... {k}", fontsize=10)
        axes[row, 1].set_ylim(-1.4, 1.4)
        for ax in axes[row]:
            ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("t (periods)")
    fig.suptitle("Frequencies assemble into waveform shape")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "square_wave_assembly.png"), dpi=150)
    print("\nSaved square_wave_reconstructions.png, square_wave_edge_zoom.png, "
          "square_wave_metrics.png, square_wave_assembly.png")

    print(
        """
What this means
---------------
Each odd harmonic is a plain sine, and the square wave is what you get by
stacking them with amplitudes 4/(pi k). The fundamental alone gives a rounded
lump; adding k = 3 starts flattening the top; each further harmonic adds
finer detail that squares the shoulders and steepens the edge.
The three things we tracked behave very differently as N grows:
  - Edge sharpness improves steadily: the 10-90% rise time shrinks roughly as
    1/N. A sharp edge is made of high frequencies, so a sharper edge needs
    higher harmonics.
  - Error away from the jumps keeps shrinking as well (the flat tops settle to
    +/-1).
  - Gibbs overshoot does NOT shrink. It sits at about 9% of the jump (peak
    ~1.18 for a +/-1 wave) whether N is 10 or 100. Adding harmonics only
    squeezes the overshoot into a narrower region hugging the edge.
So more harmonics buy you a sharper edge and a cleaner flat top, but a
discontinuity is something a finite sum of smooth sines never reproduces
exactly."""
    )
