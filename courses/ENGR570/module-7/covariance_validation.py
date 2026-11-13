"""
covariance_validation.py - Space Systems Engineering, Module 7, Experiment B

Does the navigation estimator report realistic uncertainty?

Part 1  1,000 batches of 20 observations with random noise only.
Part 2  The same, but every simulated instrument has its own fixed
        accelerometer bias and camera-IMU misalignment, constant within
        its batch and drawn afresh for the next instrument. Two estimators
        process identical data:
          Estimator 1  assumes all observations are independent and uses
                       only the random measurement covariance;
          Estimator 2  adds the cross-observation covariance of the shared
                       calibration errors.
Part 3  How each estimator's stated confidence changes with the number of
        observations.

Run:  python covariance_validation.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.stats import chi2

import navigation_estimator as ne
import navigation_scenarios as ns

SEED = 77
HERE = Path(__file__).parent
BATCHES = 1000
N_OBS = 20
CHI2_95 = chi2.ppf(0.95, 2)

SHARED = {"accelerometer_bias": ns.ACCEL_BIAS_SIGMA**2 * np.eye(3),
          "calibration": ns.MISALIGNMENT_SIGMA**2 * np.eye(3)}


def run(rng, batches, n_obs, systematic):
    """Returns {estimator: (errors (B,2), covariances (B,2,2))}."""
    out = {"Estimator 1": ([], [])}
    if systematic:
        out["Estimator 2"] = ([], [])
    for _ in range(batches):
        kwargs = {}
        if systematic:
            kwargs = dict(accel_bias=rng.normal(0.0, ns.ACCEL_BIAS_SIGMA, 3),
                          misalignment_rad=rng.normal(0.0, ns.MISALIGNMENT_SIGMA, 3))
        measurements, _ = ns.simulate_batch(rng, n_obs, run_quality_check=False,
                                            **kwargs)
        start = ns.single_fix(measurements[0])
        results = {"Estimator 1": ne.batch_position_estimate(measurements, start)}
        if systematic:
            results["Estimator 2"] = ne.batch_position_estimate(
                measurements, start, shared_errors=SHARED)
        for name, result in results.items():
            out[name][0].append(ns.north_east_error(result.lat_lon))
            out[name][1].append(result.position_covariance_ne)
    return {k: (np.array(e), np.array(c)) for k, (e, c) in out.items()}


def report(name, errors, covariances):
    nees = np.einsum("bi,bij,bj->b", errors, np.linalg.inv(covariances), errors)
    empirical = np.cov(errors.T)
    predicted = covariances.mean(axis=0)
    mean = errors.mean(axis=0)
    standard_error = np.sqrt(np.diag(empirical) / len(errors))
    coverage = 100 * np.mean(nees < CHI2_95)
    print(f"  {name}")
    print(f"    mean error            north {mean[0]:+8.1f} m, east {mean[1]:+8.1f} m "
          f"(standard error {standard_error[0]:.1f}, {standard_error[1]:.1f} m)")
    print(f"    empirical covariance  sigma_N {np.sqrt(empirical[0, 0]):8.1f} m, "
          f"sigma_E {np.sqrt(empirical[1, 1]):8.1f} m, correlation "
          f"{empirical[0, 1] / np.sqrt(empirical[0, 0] * empirical[1, 1]):+.2f}")
    print(f"    predicted covariance  sigma_N {np.sqrt(predicted[0, 0]):8.1f} m, "
          f"sigma_E {np.sqrt(predicted[1, 1]):8.1f} m, correlation "
          f"{predicted[0, 1] / np.sqrt(predicted[0, 0] * predicted[1, 1]):+.2f}")
    print(f"    RMS horizontal error  {np.sqrt(np.mean(np.sum(errors**2, axis=1))):8.1f} m "
          f"(reported {np.sqrt(np.trace(predicted)):.1f} m)")
    print(f"    mean NEES             {nees.mean():8.2f}   (2 expected)")
    print(f"    inside 95% ellipse    {coverage:8.1f} %  (95 expected)")
    return {"nees": nees, "empirical": empirical, "predicted": predicted,
            "coverage": coverage, "errors": errors}


def main():
    rng = np.random.default_rng(SEED)
    print("=" * 78)
    print(f"Part 1 - random noise only ({BATCHES} batches of {N_OBS})")
    print("=" * 78)
    random_only = run(rng, BATCHES, N_OBS, systematic=False)
    stats_random = report("Batch estimator", *random_only["Estimator 1"])
    binomial = 100 * np.sqrt(0.95 * 0.05 / BATCHES)
    print(f"""
  With correctly modelled independent noise the estimator is unbiased to
  within its sampling error, its predicted covariance matches the scatter
  of its errors to a few percent, the mean NEES is {stats_random['nees'].mean():.2f} and the 95 %
  ellipse contains the truth {stats_random['coverage']:.1f} % of the time (sampling error of that
  percentage over {BATCHES} batches: +/-{binomial:.1f}). The reported sigma is about
  {100 * (np.sqrt(np.trace(stats_random['empirical']) / np.trace(stats_random['predicted'])) - 1):.0f} % smaller than the actual scatter, a small optimism from the
  first-order measurement covariance. Its confidence can be trusted under
  these conditions.""")

    print("=" * 78)
    print(f"Part 2 - fixed instrument errors ({BATCHES} instruments, {N_OBS} "
          f"observations each)")
    print("=" * 78)
    print(f"  Per instrument: accelerometer bias {ns.ACCEL_BIAS_SIGMA} m/s^2 per axis "
          f"and misalignment {np.degrees(ns.MISALIGNMENT_SIGMA):.2f} deg per axis "
          f"(1 sigma),\n  constant within the batch, redrawn for each instrument.\n")
    systematic = run(rng, BATCHES, N_OBS, systematic=True)
    stats = {name: report(f"{name}: " + ("independent observations, random covariance only"
                                         if name.endswith("1") else
                                         "shared calibration errors included"),
                          *data)
             for name, data in systematic.items()}
    e1, e2 = stats["Estimator 1"], stats["Estimator 2"]
    print(f"""
  Both estimators see the same data and make almost the same position
  error. They differ in what they claim. Estimator 1 reports about
  {np.sqrt(np.trace(e1['predicted'])):.0f} m and is actually wrong by {np.sqrt(np.trace(e1['empirical'])) / 1000:.1f} km: its 95 % ellipse contains
  the truth {e1['coverage']:.1f} % of the time and its NEES is {e1['nees'].mean():,.0f} instead of 2. It is
  overconfident by a factor of about {np.sqrt(np.trace(e1['empirical']) / np.trace(e1['predicted'])):.0f}.
  Estimator 2 reports {np.sqrt(np.trace(e2['predicted'])) / 1000:.1f} km and achieves {e2['coverage']:.1f} % coverage. It is not more
  accurate, it is honest.
  Note the residuals cannot reveal this: a fixed bias moves all twenty
  observations together, so Estimator 1's fits look perfectly healthy.""")

    print("=" * 78)
    print("Part 3 - stated confidence versus number of observations")
    print("=" * 78)
    counts = [1, 5, 20, 80]
    rows = []
    print(f"  {'N':>4}{'actual RMS':>13}{'Est. 1 reports':>17}{'Est. 1 coverage':>18}"
          f"{'Est. 2 reports':>17}{'Est. 2 coverage':>18}")
    for n in counts:
        data = run(rng, 200, n, systematic=True)
        row = {"n": n}
        for name, (errors, covariances) in data.items():
            nees = np.einsum("bi,bij,bj->b", errors, np.linalg.inv(covariances), errors)
            row[name] = (np.sqrt(np.mean(np.sum(errors**2, axis=1))),
                         np.sqrt(np.trace(covariances.mean(axis=0))),
                         100 * np.mean(nees < CHI2_95))
        rows.append(row)
        a, b = row["Estimator 1"], row["Estimator 2"]
        print(f"  {n:>4}{a[0] / 1000:>10.2f} km{a[1]:>15.0f} m{a[2]:>16.1f} %"
              f"{b[1] / 1000:>14.2f} km{b[2]:>16.1f} %")
    print("""
  The actual error does not improve with more observations, because the
  dominant errors are common to all of them. Estimator 1's stated
  uncertainty nevertheless falls as 1/sqrt(N), so the more data it
  processes, the more wrong its confidence becomes. Estimator 2's stated
  uncertainty stays at the level set by the shared errors, as it should.

  When the confidence can be trusted: independent, correctly modelled
  random noise, with fixed errors either absent or represented.
  When it cannot: any error shared across observations that the covariance
  treats as independent, or leaves out.
  What Estimator 2 still assumes: that the fixed errors really are drawn
  from the stated priors. A bias larger than the prior, or an unmodelled
  acceleration, would make it overconfident too.""")

    make_figure(stats_random, stats, rows)


def ellipse(ax, covariance, **style):
    """95 % ellipse of a (north, east) covariance on (east, north) axes."""
    swapped = covariance[::-1, ::-1]
    values, vectors = np.linalg.eigh(swapped)
    angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
    width, height = 2 * np.sqrt(CHI2_95 * values[1]), 2 * np.sqrt(CHI2_95 * values[0])
    ax.add_patch(Ellipse((0, 0), width, height, angle=angle, fill=False, **style))


def make_figure(stats_random, stats, rows):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 12))

    ax = axes[0, 0]
    e = stats_random["errors"]
    ax.scatter(e[:, 1], e[:, 0], s=6, alpha=0.4, color="tab:blue")
    ellipse(ax, stats_random["predicted"], color="black", linewidth=1.8,
            label="reported 95% ellipse (mean)")
    ellipse(ax, stats_random["empirical"], color="tab:red", linewidth=1.5,
            linestyle="--", label="empirical 95% ellipse")
    limit = np.abs(e).max() * 1.1
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_xlabel("East error [m]")
    ax.set_ylabel("North error [m]")
    ax.set_title(f"Random noise only: {stats_random['coverage']:.1f}% inside "
                 f"the reported ellipse")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    grid = np.linspace(0, 16, 200)
    ax.hist(stats_random["nees"], bins=np.linspace(0, 16, 41), density=True,
            alpha=0.6, color="tab:blue", label="random noise only")
    ax.hist(stats["Estimator 2"]["nees"], bins=np.linspace(0, 16, 41), density=True,
            histtype="step", linewidth=1.8, color="tab:green",
            label="fixed errors, Estimator 2")
    ax.plot(grid, chi2.pdf(grid, 2), "k-", label="chi-square, 2 degrees of freedom")
    ax.axvline(CHI2_95, color="gray", linestyle=":", label="95% limit (5.99)")
    ax.set_xlabel("NEES")
    ax.set_ylabel("Density")
    ax.set_title(f"NEES distribution (Estimator 1 with fixed errors: mean "
                 f"{stats['Estimator 1']['nees'].mean():,.0f}, off this scale)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    e = stats["Estimator 1"]["errors"] / 1000
    ax.scatter(e[:, 1], e[:, 0], s=6, alpha=0.4, color="tab:orange",
               label="position errors (both estimators)")
    ellipse(ax, stats["Estimator 2"]["predicted"] / 1e6, color="tab:green",
            linewidth=1.8, label="Estimator 2 reported 95% ellipse")
    ellipse(ax, stats["Estimator 1"]["predicted"] / 1e6, color="tab:red",
            linewidth=2.0, label="Estimator 1 reported 95% ellipse (the dot)")
    ax.plot(0, 0, "o", color="tab:red", markersize=3)
    limit = np.abs(e).max() * 1.1
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_xlabel("East error [km]")
    ax.set_ylabel("North error [km]")
    ax.set_title(f"Fixed instrument errors: coverage "
                 f"{stats['Estimator 1']['coverage']:.1f}% (Est. 1) versus "
                 f"{stats['Estimator 2']['coverage']:.1f}% (Est. 2)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    counts = [r["n"] for r in rows]
    ax.loglog(counts, [r["Estimator 1"][0] for r in rows], "ko-",
              label="actual RMS error")
    ax.loglog(counts, [r["Estimator 1"][1] for r in rows], "s--", color="tab:red",
              label="Estimator 1 reported RMS")
    ax.loglog(counts, [r["Estimator 2"][1] for r in rows], "^--", color="tab:green",
              label="Estimator 2 reported RMS")
    ax.set_xticks(counts)
    ax.set_xticklabels([str(c) for c in counts])
    ax.minorticks_off()
    ax.set_xlabel("Observations per batch")
    ax.set_ylabel("Horizontal position error [m]")
    ax.set_title("More data, more confidence, no more accuracy")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(HERE / "covariance_validation.png", dpi=130)
    print("\nSaved covariance_validation.png")


if __name__ == "__main__":
    main()
