"""
Experiment E: batch vs stochastic vs mini-batch gradient descent.

Data: 5000 observations of y = 3 + 2x + eps (x ~ U(-5,5), sigma = 1.5); a new dataset per seed.
Within a seed, every method shares the dataset and the start theta = [0, 0].

  batch       batch_size = 5000  (one update = 5000 example-gradient evaluations)
  mini-batch  batch_size = 16
  stochastic  batch_size = 1

Controlled comparison: one learning rate, eta = 0.02, for all three, so only the batching differs.
It must be safe for the WORST single example: one example's Hessian is 2 [1 x][1 x]^T with
eigenvalue 2(1 + x^2) <= 52, so single-example steps are only stable for eta < 2/52 ~= 0.038.
Two extra runs: batch at eta = 0.1 (near the full-batch optimum eta* found in Experiment D) and
SGD at eta = 0.002, to show how the step size trades speed for noise.

Cost is counted two ways:
  updates   number of parameter updates (sequential steps)
  examples  number of per-example gradient evaluations (the actual arithmetic work)
The full-dataset loss J(theta) is evaluated after every update for analysis only; the optimizer
never uses it. The loss of the batch actually used for each update is recorded too.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gd_core import design_matrix, generate_data, minibatch_gradient_descent, mse_loss

OUT = Path(__file__).parent / "outputs"
N = 5000
ETA = 0.02
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 20                         # equal-work budget: 20 x 5000 = 100,000 example evaluations
UPDATES = 2000                      # equal-step budget
STAY_BELOW = 0.1
# name: (batch size, learning rate)
METHODS = {"batch (5000)": (N, ETA), "mini-batch (16)": (16, ETA), "stochastic (1)": (1, ETA),
           "batch, eta=0.1": (N, 0.1), "stochastic, eta=0.002": (1, 0.002)}
CORE = ["batch (5000)", "mini-batch (16)", "stochastic (1)"]
COLORS = {"batch (5000)": "tab:blue", "mini-batch (16)": "tab:green", "stochastic (1)": "tab:red",
          "batch, eta=0.1": "tab:cyan", "stochastic, eta=0.002": "tab:orange"}


def run_seed(seed):
    rng = np.random.default_rng([seed, 2])
    x, y = generate_data(N, rng)
    X = design_matrix(x)
    th_ls = np.linalg.solve(X.T @ X, X.T @ y)
    j_ls = mse_loss(X, y, th_ls)
    out = {}
    for name, (b, eta) in METHODS.items():
        max_updates = max(UPDATES, int(np.ceil(EPOCHS * N / b)))
        _, log = minibatch_gradient_descent(X, y, np.zeros(2), eta, b, max_updates,
                                            np.random.default_rng([seed, 3]), eval_every=1)
        log["gap"] = log["full_loss"] - j_ls
        log["dist"] = np.linalg.norm(log["theta"] - th_ls, axis=1)
        out[name] = log
    return out, j_ls


def at(log, key, value, field):
    """Value of `field` at the first record where log[key] >= value."""
    i = int(np.searchsorted(log[key], value))
    return log[field][min(i, log[field].size - 1)]


def main():
    OUT.mkdir(exist_ok=True)
    runs = [run_seed(s) for s in SEEDS]

    checkpoints = [("examples", N, "1 epoch = 5,000 examples"),
                   ("examples", EPOCHS * N, f"{EPOCHS} epochs = {EPOCHS * N:,} examples"),
                   ("updates", 20, "20 updates"),
                   ("updates", UPDATES, f"{UPDATES:,} updates")]
    print(f"Experiment E: N = {N}, start [0, 0], {len(SEEDS)} seeds;"
          f" controlled comparison at eta = {ETA}, plus two extra runs")
    print("  Excess full-dataset loss J(theta) - J*  and  |theta - theta_LS|   (mean +- std over seeds)")
    for key, val, label in checkpoints:
        print(f"\n  after {label}:")
        print(f"    {'method':<23}{'updates':>9}{'examples':>12}{'J - J*':>24}{'|theta - LS|':>24}")
        for name in METHODS:
            gaps = np.array([at(r[name], key, val, "gap") for r, _ in runs])
            dists = np.array([at(r[name], key, val, "dist") for r, _ in runs])
            upd = at(runs[0][0][name], key, val, "updates")
            ex = at(runs[0][0][name], key, val, "examples")
            print(f"    {name:<23}{upd:>9,}{ex:>12,}"
                  f"{gaps.mean():>14.3e} +- {gaps.std():<7.1e}{dists.mean():>14.3e} +- {dists.std():<7.1e}")

    # Reliability: first time below the level vs how much of the late phase is spent below it
    # (a stochastic method can dip below a level by luck and bounce back out).
    print(f"\n  Reaching J - J* < {STAY_BELOW}: first example count where it happens (median), and"
          f" fraction of updates in epochs 10-{EPOCHS} spent below it (mean over seeds):")
    for name in METHODS:
        first, frac = [], []
        for r, _ in runs:
            lg = r[name]
            hit = np.nonzero(lg["gap"] < STAY_BELOW)[0]
            first.append(lg["examples"][hit[0]] if hit.size else np.nan)
            m = (lg["examples"] > 10 * N) & (lg["examples"] <= EPOCHS * N)
            frac.append(np.mean(lg["gap"][m] < STAY_BELOW))
        first = np.array(first, dtype=float)
        txt = f"{np.nanmedian(first):>12,.0f}" if np.isfinite(first).any() else f"{'never':>12}"
        print(f"    {name:<23}{txt}   time below: {np.mean(frac):6.1%}")

    # Late-phase behaviour: constant-step stochastic methods hover at a noise floor
    print(f"\n  Mean J - J* over the last 2 epochs of the {EPOCHS}-epoch budget (noise floor):")
    for name in METHODS:
        vals = []
        for r, _ in runs:
            lg = r[name]
            m = (lg["examples"] > (EPOCHS - 2) * N) & (lg["examples"] <= EPOCHS * N)
            vals.append(lg["gap"][m].mean())
        print(f"    {name:<23}{np.mean(vals):>12.3e}")
    print(f"  J* (irreducible training loss, mean over seeds) = {np.mean([j for _, j in runs]):.4f}")

    # ---- figure 1: equal work vs equal updates
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for ax, key, xlabel, xmax in ((axes[0], "examples", "example-gradient evaluations (work)",
                                   EPOCHS * N),
                                  (axes[1], "updates", "parameter updates (sequential steps)",
                                   UPDATES)):
        for name in METHODS:
            for k, (r, _) in enumerate(runs):
                lg = r[name]
                m = (lg[key] >= 1) & (lg[key] <= xmax)
                ax.loglog(lg[key][m], np.maximum(lg["gap"][m], 1e-12), color=COLORS[name],
                          lw=0.8, alpha=0.6 if name in CORE else 0.45,
                          ls="-" if name in CORE else "--",
                          label=(name if name in CORE else name + " (extra)") if k == 0 else None)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("full-dataset J(theta) - J*")
    axes[0].set_title(f"Equal work: first {EPOCHS} epochs")
    axes[1].set_title(f"Equal number of updates: first {UPDATES:,}")
    fig.suptitle(f"Experiment E: batch vs mini-batch vs stochastic (solid: eta = {ETA}; "
                 f"{len(SEEDS)} seeds, one line per seed)")
    fig.tight_layout()
    fig.savefig(OUT / "exp_e_comparison.png", dpi=150)
    plt.close(fig)

    # ---- figure 2: batch loss (what the optimizer sees) vs full-dataset loss
    r, j_ls = runs[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    for ax, name in zip(axes, ("stochastic (1)", "mini-batch (16)")):
        lg = r[name]
        n_show = 3000
        ax.semilogy(np.arange(1, n_show + 1), lg["batch_loss"][:n_show], ".", ms=1.5, alpha=0.4,
                    color=COLORS[name], label="loss of the batch used for the update")
        ax.semilogy(lg["updates"][:n_show + 1], lg["full_loss"][:n_show + 1], "k", lw=1.5,
                    label="full-dataset loss")
        ax.axhline(j_ls, color="gray", ls=":", label=f"J* = {j_ls:.3f}")
        ax.set_xlabel("update")
        ax.set_title(f"{name}, eta = {ETA}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("MSE")
    fig.suptitle("Batch loss is a noisy measurement of training progress (seed 0)")
    fig.tight_layout()
    fig.savefig(OUT / "exp_e_batch_vs_full_loss.png", dpi=150)
    plt.close(fig)
    print(f"\n  saved {OUT / 'exp_e_comparison.png'}")
    print(f"  saved {OUT / 'exp_e_batch_vs_full_loss.png'}")


if __name__ == "__main__":
    main()
