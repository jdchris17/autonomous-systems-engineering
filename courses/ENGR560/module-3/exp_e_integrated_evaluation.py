"""
Experiment E: integrated regression model evaluation and the final integration challenge.

Imperfect calibration data with all three effects:
  1. systematic radial bias  delta_u(r) = 0.15 + 0.4 r^2
  2. intensity-dependent precision  sigma(I) = 0.05 + 0.8/sqrt(I)
  3. 5% corrupted measurements (offset ~ N(1, 2^2) px, unrecognized: sigma not inflated)
Splits per seed: train 300, validation 300, test 5000 - all drawn from the same imperfect
process (the test set is "unseen observations"; its corruption labels are used only for metrics).

Candidates: features {r, r2, quad, quartic} x estimators {OLS, WLS, Huber, RANSAC}, plus the
baseline that applies no correction.

Selection rule (validation only, no labels needed):
  score = mean of the smallest 90% of z_i^2,  z_i = (y_i - y_hat_i) / sigma_i
          (trimming makes it robust to the few corrupted points; using squares keeps it efficient)
  pick the lowest score; among candidates within 2% of it, pick the one with fewest parameters.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from calibration_eval import CalibrationModel, evaluate, features, residual_plots
from reg_core import CAL_TRUE, OUT, calibration_truth, corrupt, generate_calibration_data

FRACTION = 0.05
N_TRAIN, N_VAL, N_TEST = 300, 300, 5000
SEEDS = range(50)
FEATS = ("r", "r2", "quad", "quartic")
METHODS = ("ols", "wls", "huber", "ransac")
TIE = 1.02


def make_splits(seed):
    rng = np.random.default_rng([seed, 41])
    return {name: corrupt(generate_calibration_data(n, rng), FRACTION, rng)
            for name, n in (("train", N_TRAIN), ("val", N_VAL), ("test", N_TEST))}


TRIM = 0.10


def val_score(model, d):
    """Trimmed mean of squared standardized residuals: drop the largest 10% (covers the ~5%
    corrupted points), average the rest. Far more statistically efficient than a median."""
    z2 = np.sort(((d["observed_bias"] - model.predict(d["r"])) / d["sigma"]) ** 2)
    return float(np.mean(z2[: int(np.floor((1 - TRIM) * z2.size))]))


def select(splits, seed, verbose=False):
    rows = []
    for f in FEATS:
        for m in METHODS:
            mdl = CalibrationModel(f, m, seed).fit(splits["train"])
            rows.append((val_score(mdl, splits["val"]), features(f, np.zeros(1)).shape[1], mdl))
    best = min(r[0] for r in rows)
    ok = [r for r in rows if r[0] <= TIE * best]
    chosen = min(ok, key=lambda r: (r[1], r[0]))[2]
    if verbose:
        print(f"  Validation scores, seed {seed} (10%-trimmed mean of z^2; lower is better):")
        print("    " + f"{'':<10}" + "".join(f"{m.upper():>10}" for m in METHODS))
        for f in FEATS:
            vals = [r[0] for r in rows if r[2].feats == f]
            print(f"    {f:<10}" + "".join(f"{v:>10.4f}" for v in vals))
        print(f"  best score {best:.4f}; within {TIE - 1:.0%}: "
              + ", ".join(r[2].name for r in ok) + f"  ->  selected {chosen.name}")
    return chosen


def print_metrics(results):
    keys = ["RMSE", "MAE", "MedAE", "RMSE_truth", "param_err", "z_std", "frac_|z|>3",
            "CI95_mean_cover", "PI95_cover", "PI95_faint", "PI95_bright"]
    print(f"    {'model':<16}" + "".join(f"{k:>16}" for k in keys))
    for name, ev in results:
        t = ev["test"]
        print(f"    {name:<16}" + "".join(
            f"{t[k]:>16.4f}" if k in t else f"{'-':>16}" for k in keys))


def main():
    OUT.mkdir(exist_ok=True)
    print("Experiment E: integrated evaluation on deliberately imperfect calibration data")
    print(f"  {FRACTION:.0%} corrupted, N_train/val/test = {N_TRAIN}/{N_VAL}/{N_TEST}\n")

    splits = make_splits(0)
    chosen = select(splits, 0, verbose=True)
    compare = [CalibrationModel("none", "ols"),
               CalibrationModel("r", "ols"),
               CalibrationModel("r2", "ols"),
               CalibrationModel("r2", "wls"),
               CalibrationModel("r2", "huber"),
               CalibrationModel("r2", "ransac", 0)]
    if chosen.name not in [c.name for c in compare]:
        compare.append(chosen)
    results = [(m.fit(splits["train"]).name, evaluate(m, splits, calibration_truth, CAL_TRUE))
               for m in compare]
    print("\n  Test-set metrics, seed 0 (RMSE/MAE/MedAE vs noisy measurements incl. corrupted ones;"
          " RMSE_truth vs noise-free calibration;\n  z = standardized residual on clean points;"
          " CI95 = mean-function band covers the truth; PI95 = prediction interval covers a clean"
          " measurement)")
    print_metrics(results)

    residual_plots([compare[0], compare[1], compare[3], chosen], splits["test"],
                   OUT / "exp_e_residual_diagnostics.png", calibration_truth,
                   "Test-set residual diagnostics, seed 0 (red x = corrupted measurements)")
    print(f"  saved {OUT / 'exp_e_residual_diagnostics.png'}")

    # ---------------------------------------------------------------- consistency over seeds
    track = ["no correction", "OLS [r2]", "WLS [r2]", "RANSAC [r2]", "selected"]
    agg = {k: {"RMSE_truth": [], "CI95_mean_cover": [], "PI95_cover": [], "PI95_faint": [],
               "PI95_bright": [], "MedAE": []} for k in track}
    picks = []
    for s in SEEDS:
        sp = make_splits(s)
        sel = select(sp, s)
        picks.append(sel.name)
        models = {"no correction": CalibrationModel("none", "ols"),
                  "OLS [r2]": CalibrationModel("r2", "ols"),
                  "WLS [r2]": CalibrationModel("r2", "wls"),
                  "RANSAC [r2]": CalibrationModel("r2", "ransac", s),
                  "selected": sel}
        for k, mdl in models.items():
            if k != "selected":
                mdl.fit(sp["train"])
            t = evaluate(mdl, {"test": sp["test"]}, calibration_truth)["test"]
            for key in agg[k]:
                agg[k][key].append(t[key])

    u, c = np.unique(picks, return_counts=True)
    print(f"\n  Over {len(SEEDS)} seeds the validation rule selected: "
          + ", ".join(f"{a} x{b}" for a, b in sorted(zip(u, c), key=lambda t: -t[1])))
    print(f"    {'model':<16}{'RMSE_truth median':>18}{'95th pct':>10}{'worst':>9}{'MedAE':>9}"
          f"{'CI95 mean':>11}{'PI95 all':>10}{'PI95 faint':>12}{'PI95 bright':>13}")
    for k in track:
        a = {key: np.array(v) for key, v in agg[k].items()}
        print(f"    {k:<16}{np.median(a['RMSE_truth']):>18.4f}{np.percentile(a['RMSE_truth'], 95):>10.4f}"
              f"{a['RMSE_truth'].max():>9.4f}{np.median(a['MedAE']):>9.4f}"
              f"{a['CI95_mean_cover'].mean():>11.1%}{a['PI95_cover'].mean():>10.1%}"
              f"{a['PI95_faint'].mean():>12.1%}{a['PI95_bright'].mean():>13.1%}")

    plot_final(splits, chosen, agg, track)


def plot_final(splits, chosen, agg, track):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    d = splits["train"]
    rs = np.linspace(0, 1, 200)
    ok = ~d["outlier"]
    ax.scatter(d["r"][ok], d["observed_bias"][ok], s=6, color="tab:gray", alpha=0.5,
               label="training measurements")
    ax.scatter(d["r"][~ok], d["observed_bias"][~ok], s=20, marker="x", color="tab:red",
               label="corrupted (unknown to the fit)")
    mu = chosen.predict(rs)
    sm = np.sqrt(chosen.mean_var(rs))
    ax.plot(rs, calibration_truth(rs), "k", lw=2.5, label="true calibration")
    ax.plot(rs, mu, color="tab:blue", lw=2, label=f"selected: {chosen.name}")
    ax.fill_between(rs, mu - 1.96 * sm, mu + 1.96 * sm, color="tab:blue", alpha=0.35,
                    label="95% band for the MEAN calibration")
    for I, col in ((5.0, "tab:orange"), (80.0, "tab:green")):
        sp = np.sqrt(chosen.mean_var(rs) + chosen.noise_var(np.full_like(rs, 0.05 + 0.8 / np.sqrt(I))))
        ax.plot(rs, mu + 1.96 * sp, "--", color=col, lw=1)
        ax.plot(rs, mu - 1.96 * sp, "--", color=col, lw=1,
                label=f"95% interval for ONE measurement, I = {I:g}")
    ax.plot(rs, np.zeros_like(rs), ":", color="tab:red", lw=1.5, label="baseline: no correction")
    ax.set_ylim(-1.5, 2.5)
    ax.set_xlabel("r")
    ax.set_ylabel("delta_u [px]")
    ax.set_title("Final calibrated predictor (seed 0)")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)

    ax = axes[1]
    data = [np.array(agg[k]["RMSE_truth"]) for k in track]
    ax.boxplot(data, tick_labels=track, whis=(5, 95), showfliers=True)
    ax.set_yscale("log")
    ax.set_ylabel("RMSE vs true calibration on unseen data [px]")
    ax.set_title(f"Consistency over {len(SEEDS)} independent datasets")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "exp_e_final_predictor.png", dpi=150)
    plt.close(fig)
    print(f"  saved {OUT / 'exp_e_final_predictor.png'}")


if __name__ == "__main__":
    main()
