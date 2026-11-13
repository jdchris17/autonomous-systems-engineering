"""
error_budget_experiment.py - Space Systems Engineering, Module 8, Experiment D

The complete navigation error budget and improvement ranking.

Step 1  baseline: Module 6/7 assumptions, now in a physically realistic
        world (refraction, aberration, deflection, imperfect optics)
Step 2  each new effect neglected, then corrected
Step 3  1,000-instrument Monte Carlo; repeatability versus accuracy
Step 4  candidate improvements ranked by the change they actually produce,
        then applied cumulatively; dimensionless sensitivities
Step 5  predicted uncertainty versus Monte Carlo behaviour

The Monte Carlo takes about 25 minutes in total, so each stage is cached in
error_budget_cache.json. Run with no arguments to compute whatever is
missing and print the report; `--stage NAME` computes one stage;
`--fresh` discards the cache.

Run:  python error_budget_experiment.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import navigation_error_budget as nb

HERE = Path(__file__).parent
CACHE = HERE / "error_budget_cache.json"
SEED = 5708
PHASE1_M = (11_100.0, 18_500.0)
PHASE2_M = (100.0, 200.0)

BASELINE = nb.BASELINE

CANDIDATES = {
    "Camera-IMU alignment 10x better (18\")": dict(misalignment_sigma_arcsec=18.0),
    "Diverse-attitude calibration (20 obs.)": dict(
        diverse_attitudes=True, n_observations=20, estimate_calibration=True),
    "Accelerometer bias 10x smaller": dict(accel_bias_sigma=0.0005),
    "Camera intrinsic calibration": dict(
        principal_point_sigma_px=0.05, focal_length_sigma_fraction=5e-5,
        model_distortion=True),
    "Atmospheric refraction correction": dict(model_refraction=True),
    "Apparent-place star model": dict(model_apparent_place=True),
    "Deflection correction (to 1\")": dict(deflection_model_residual_arcsec=1.0),
    "Absolute time 10x better (0.01 s)": dict(clock_sigma_s=0.01),
    "40 matched stars": dict(n_stars=40),
    "Narrower field (10 deg)": dict(hfov_deg=10.0),
    "Point at the zenith": dict(zenith_angle_deg=0.0),
}

# Order in which improvements are accumulated (largest individual gain first,
# then the corrections that only become visible afterwards).
CUMULATIVE_ORDER = [
    "Diverse-attitude calibration (20 obs.)",
    "Accelerometer bias 10x smaller",
    "Atmospheric refraction correction",
    "Camera intrinsic calibration",
    "Apparent-place star model",
    "Deflection correction (to 1\")",
    "Absolute time 10x better (0.01 s)",
    "40 matched stars",
]
FINAL_EXTRA = ("Accelerometer bias 1e-4 m/s^2 (10 micro-g)",
               dict(accel_bias_sigma=1e-4))

SWEEP_FIELDS = ["misalignment_sigma_arcsec", "accel_bias_sigma",
                "principal_point_sigma_px", "deflection_sigma_arcsec",
                "clock_sigma_s", "centroid_noise_px", "accel_noise"]


def improved_scenario(with_extra=True):
    changes = {}
    for name in CUMULATIVE_ORDER:
        changes.update(CANDIDATES[name])
    if with_extra:
        changes.update(FINAL_EXTRA[1])
    return BASELINE.with_changes(name="improved", **changes)


# ------------------------------------------------------------------------ stages
def pack(result):
    return {"rms": result.rms, "median": result.median, "p95": result.p95,
            "repeatability_rms": result.repeatability_rms,
            "reported_rms": result.reported_rms, "mean_nees": result.mean_nees,
            "coverage95": result.coverage95, "trials": result.trials,
            "rms_north": result.diagnostics["rms_north"],
            "rms_east": result.diagnostics["rms_east"],
            "withheld": result.diagnostics["covariance_withheld"],
            "contributions": result.contributions,
            "horizontal": [float(v) for v in result.horizontal]}


def stage_baseline():
    return pack(nb.run_error_budget(BASELINE, trials=1000, seed=SEED,
                                    contribution_trials=200))


def stage_reference():
    module7 = BASELINE.with_changes(**nb.MODULE7_WORLD)
    naive = BASELINE.with_changes(declare_uncertainties=False)
    return {
        "module7_world": pack(nb.run_error_budget(module7, trials=300, seed=SEED,
                                                  contributions=False)),
        "baseline_undeclared": pack(nb.run_error_budget(naive, trials=300, seed=SEED,
                                                        contributions=False)),
    }


def stage_corrections():
    """Each new physical effect alone: neglected, then corrected."""
    cases = {
        "Atmospheric refraction": dict(model_refraction=True),
        "Stellar apparent place": dict(model_apparent_place=True),
        "Deflection of the vertical": dict(deflection_model_residual_arcsec=1.0),
        "Camera intrinsics": dict(principal_point_sigma_px=0.05,
                                  focal_length_sigma_fraction=5e-5,
                                  model_distortion=True),
    }
    out = {}
    for source, fix in cases.items():
        neglected, *_ = nb._run(nb.isolate(BASELINE, source), 200, SEED)
        corrected, *_ = nb._run(nb.isolate(BASELINE.with_changes(**fix), source)
                                .with_changes(**{k: v for k, v in fix.items()
                                                 if k.startswith("model_")
                                                 or k.startswith("deflection_model")}),
                                200, SEED)
        out[source] = {"neglected": nb.summarize(neglected),
                       "corrected": nb.summarize(corrected)}
    return out


def resumable(key, items, evaluate):
    """
    Evaluate items one at a time, saving each result to the cache under
    `key` as soon as it is done, so an interrupted run resumes where it
    stopped instead of starting again.
    """
    done = load_cache().get(key, {})
    for name in items:
        if name in done:
            continue
        print(f"    {key}: {name}", flush=True)
        done[name] = evaluate(name)
        cache = load_cache()
        cache[key] = done
        CACHE.write_text(json.dumps(cache))
    return done


def stage_improvements():
    """Same procedure as nb.compare_improvements(), one candidate at a time."""
    def evaluate(name):
        changes = {} if name == "baseline" else CANDIDATES[name]
        errors, *_ = nb._run(BASELINE.with_changes(**changes), 300, SEED)
        return nb.summarize(errors)

    parts = resumable("_improvements", ["baseline"] + list(CANDIDATES), evaluate)
    base = parts["baseline"]
    rows = [{"name": name, **parts[name],
             "rms_reduction": base["rms"] - parts[name]["rms"],
             "p95_reduction": base["p95"] - parts[name]["p95"],
             "rms_ratio": parts[name]["rms"] / base["rms"]} for name in CANDIDATES]
    rows.sort(key=lambda row: -row["rms_reduction"])
    return [{"name": "baseline", **base, "rms_reduction": 0.0,
             "p95_reduction": 0.0, "rms_ratio": 1.0}] + rows


def stage_cumulative():
    steps = [(name, CANDIDATES[name]) for name in CUMULATIVE_ORDER] + [FINAL_EXTRA]
    names = [name for name, _ in steps]

    def evaluate(name):
        changes = {}
        for step_name, change in steps[:names.index(name) + 1]:
            changes.update(change)
        errors, *_ = nb._run(BASELINE.with_changes(**changes), 250, SEED)
        return nb.summarize(errors)

    parts = resumable("_cumulative", names, evaluate)
    return [{"name": name, **parts[name]} for name in names]


def stage_improved():
    return pack(nb.run_error_budget(improved_scenario(), trials=500, seed=SEED,
                                    contribution_trials=120))


def stage_sensitivity_baseline():
    def evaluate(name):
        return nb.sensitivity_analysis(BASELINE, {name: [0.5, 2.0]}, trials=150,
                                       seed=SEED)[name]

    return resumable("_sensitivity_baseline", SWEEP_FIELDS, evaluate)


def stage_sensitivity_improved():
    scenario = improved_scenario()
    fields = ["accel_bias_sigma", "deflection_model_residual_arcsec",
              "centroid_noise_px", "accel_noise", "clock_sigma_s",
              "principal_point_sigma_px"]
    return nb.sensitivity_analysis(scenario, {f: [0.5, 2.0] for f in fields},
                                   trials=100, seed=SEED)


STAGES = {
    "baseline": stage_baseline,
    "reference": stage_reference,
    "corrections": stage_corrections,
    "improvements": stage_improvements,
    "cumulative": stage_cumulative,
    "improved": stage_improved,
    "sensitivity_baseline": stage_sensitivity_baseline,
    "sensitivity_improved": stage_sensitivity_improved,
}


def load_cache():
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def compute(stage, cache):
    print(f"[computing stage '{stage}' ...]", flush=True)
    result = STAGES[stage]()
    cache = load_cache()          # re-read: other stages may have finished
    cache[stage] = result
    CACHE.write_text(json.dumps(cache))
    return cache


# ------------------------------------------------------------------------ report
def metres(value):
    return f"{value / 1000:.2f} km" if value >= 1000 else f"{value:.0f} m"


def budget_table(title, data, scenario):
    assumed = {
        "Star centroid noise": f"{scenario.centroid_noise_px} px, {scenario.n_stars} stars",
        "Accelerometer noise": f"{scenario.accel_noise} m/s^2 x {scenario.accel_samples}",
        "Accelerometer bias": f"{scenario.accel_bias_sigma:g} m/s^2 per axis",
        "Camera-IMU misalignment": (
            f"{scenario.misalignment_sigma_arcsec:g}\" per axis"
            + (", estimated" if scenario.estimate_calibration else "")),
        "Camera intrinsics": (
            f"pp {scenario.principal_point_sigma_px:g} px, f "
            f"{scenario.focal_length_sigma_fraction * 100:g} %, k1 {scenario.lens_k1:g}"
            + (" modelled" if scenario.model_distortion else " ignored")),
        "Clock": f"{scenario.clock_sigma_s:g} s",
        "Atmospheric refraction": ("modelled" if scenario.model_refraction
                                   else "not modelled") + ", 1013 hPa, 15 C",
        "Stellar apparent place": ("modelled" if scenario.model_apparent_place
                                   else "not modelled") + ", aberration + pm",
        "Deflection of the vertical": (
            f"{scenario.deflection_sigma_arcsec:g}\" per component"
            + ("" if scenario.deflection_model_residual_arcsec is None else
               f", corrected to {scenario.deflection_model_residual_arcsec:g}\"")),
        "Earth orientation data": f"UT1 {scenario.eop_ut1_sigma_s * 1000:.1f} ms, "
                                  f"pole {scenario.eop_pm_sigma_arcsec * 1000:.0f} mas",
    }
    parts = data["contributions"]
    total_square = sum(c["rms"] ** 2 for c in parts.values())
    print(f"\n{title}")
    print(f"  {'error source':<28}{'class':<15}{'assumed (illustrative)':<36}"
          f"{'RMS':>10}{'95th pct':>11}{'share':>8}")
    for name, c in sorted(parts.items(), key=lambda item: -item[1]["rms"]):
        print(f"  {name:<28}{c['category']:<15}{assumed[name]:<36}"
              f"{metres(c['rms']):>10}{metres(c['p95']):>11}"
              f"{100 * c['rms'] ** 2 / total_square:>7.1f}%")
    by_class = {}
    for c in parts.values():
        by_class[c["category"]] = by_class.get(c["category"], 0.0) + c["rms"] ** 2
    print(f"  {'root-sum-square of the sources':<79}{metres(np.sqrt(total_square)):>10}")
    print(f"  {'combined Monte Carlo (' + str(data['trials']) + ' instruments)':<79}"
          f"{metres(data['rms']):>10}{metres(data['p95']):>11}   median "
          f"{metres(data['median'])}")
    print("  by class: " + ", ".join(
        f"{k} {metres(np.sqrt(v))} ({100 * v / total_square:.1f}%)"
        for k, v in sorted(by_class.items(), key=lambda item: -item[1])))
    print(f"  population accuracy {metres(data['rms'])} RMS; within-instrument "
          f"repeatability {metres(data['repeatability_rms'])} RMS "
          f"(scatter of one instrument's own fixes).")


def report(cache):
    base, improved = cache["baseline"], cache["improved"]
    reference = cache["reference"]
    final = improved_scenario()

    print("=" * 78)
    print("Step 1 - baseline")
    print("=" * 78)
    print("  Module 6/7 estimator: 1920 x 1080 camera, 20 deg HFOV, 20 stars, "
          "0.2 px; accelerometer 0.005 m/s^2\n  noise and 0.005 m/s^2 bias; "
          "misalignment 0.05 deg; time 0.1 s; no atmospheric correction; "
          "WGS84-normal\n  vertical; Module 2 Earth orientation. 10 "
          "observations per instrument, boresight 30 deg from zenith.")
    print("  New in this module, as assumptions about the real world: "
          "standard atmosphere, stellar aberration and\n  proper motion, "
          "deflection of the vertical 5 arcsec per component, principal "
          "point 0.5 px, focal\n  length 0.05 %, lens distortion k1 = 0.02. "
          "All illustrative; none is a verified specification.")
    m7 = reference["module7_world"]
    print(f"\n  Same estimator in the idealized Module 6/7 world: RMS "
          f"{metres(m7['rms'])}, 95th percentile {metres(m7['p95'])}.")
    print(f"  In the physically perturbed world:                 RMS "
          f"{metres(base['rms'])}, 95th percentile {metres(base['p95'])}.")

    print("\n" + "=" * 78)
    print("Step 2 - each new effect, neglected and then corrected "
          "(that effect alone, 200 instruments)")
    print("=" * 78)
    print(f"  {'effect':<30}{'neglected: RMS':>16}{'95th pct':>11}"
          f"{'corrected: RMS':>18}{'95th pct':>11}")
    for name, row in cache["corrections"].items():
        print(f"  {name:<30}{metres(row['neglected']['rms']):>16}"
              f"{metres(row['neglected']['p95']):>11}"
              f"{metres(row['corrected']['rms']):>18}"
              f"{metres(row['corrected']['p95']):>11}")
    print("  Corrections applied: refraction model at the true conditions; "
          "apparent-place star model;\n  deflection known to 1 arcsec; "
          "intrinsics calibrated to 0.05 px and 0.005 % with distortion modelled.")

    print("\n" + "=" * 78)
    print("Step 3 - combined error budget")
    print("=" * 78)
    budget_table("Baseline", base, BASELINE)

    print("\n" + "=" * 78)
    print("Step 4 - candidate improvements")
    print("=" * 78)
    rows = cache["improvements"]
    print("  Each applied alone to the baseline (300 instruments, same random "
          "seed):")
    print(f"  {'candidate':<42}{'RMS':>10}{'95th pct':>11}{'RMS change':>13}"
          f"{'95th change':>13}{'RMS ratio':>11}")
    for row in rows:
        print(f"  {row['name']:<42}{metres(row['rms']):>10}{metres(row['p95']):>11}"
              f"{-row['rms_reduction']:>+11.0f} m{-row['p95_reduction']:>+11.0f} m"
              f"{row['rms_ratio']:>11.2f}")
    print("  Only improvements to the dominant terms move the total. Corrections "
          "worth hundreds of metres\n  are invisible behind nine kilometres of "
          "alignment and bias error: ranking by individual gain\n  from the "
          "baseline says nothing about them, so they are also applied in "
          "sequence below.")

    print("\n  Applied cumulatively:")
    print(f"  {'after adding':<46}{'median':>10}{'RMS':>10}{'95th pct':>11}")
    print(f"  {'(baseline)':<46}{metres(base['median']):>10}"
          f"{metres(base['rms']):>10}{metres(base['p95']):>11}")
    for row in cache["cumulative"]:
        print(f"  {'+ ' + row['name']:<46}{metres(row['median']):>10}"
              f"{metres(row['rms']):>10}{metres(row['p95']):>11}")

    budget_table("Improved scenario (all of the above)", improved, final)

    print("\n  Dimensionless sensitivity S = d ln(RMS error) / d ln(parameter), "
          "from x0.5 and x2 runs:")
    labels = {"misalignment_sigma_arcsec": "camera-IMU misalignment",
              "accel_bias_sigma": "accelerometer bias",
              "principal_point_sigma_px": "principal point",
              "deflection_sigma_arcsec": "deflection of the vertical",
              "deflection_model_residual_arcsec": "deflection model residual",
              "clock_sigma_s": "clock", "centroid_noise_px": "centroid noise",
              "accel_noise": "accelerometer noise"}
    sb, si = cache["sensitivity_baseline"], cache["sensitivity_improved"]
    print(f"  {'parameter':<30}{'baseline':>10}{'improved':>10}")
    for name in dict.fromkeys(list(sb) + list(si)):
        b = f"{sb[name]['sensitivity']:+.2f}" if name in sb else "-"
        i = f"{si[name]['sensitivity']:+.2f}" if name in si else "-"
        print(f"  {labels[name]:<30}{b:>10}{i:>10}")
    print("  The sensitivities of independent sources are their shares of the "
          "error variance, so they sum to\n  about one. In the baseline the "
          "leverage is all in alignment and bias. Once those are dealt with\n  "
          "it moves to whatever is then largest. Sensitivity shows where "
          "accuracy leverage is, not which\n  improvement is affordable.")

    print("\n" + "=" * 78)
    print("Step 5 - predicted uncertainty versus Monte Carlo")
    print("=" * 78)
    naive = reference["baseline_undeclared"]
    print(f"  {'scenario':<44}{'actual RMS':>12}{'reported RMS':>14}"
          f"{'mean NEES':>11}{'95% coverage':>14}")
    for label, d in (("Baseline, fixed errors not declared", naive),
                     ("Baseline, uncertainties declared", base),
                     ("Improved, uncertainties declared", improved)):
        print(f"  {label:<44}{metres(d['rms']):>12}{metres(d['reported_rms']):>14}"
              f"{d['mean_nees']:>11.1f}{d['coverage95']:>13.1f}%")
    print("""  Declared = accelerometer bias and calibration as shared errors (Module 7),
  plus clock and deflection added to the position covariance. NOT in any
  covariance: the biases from unmodelled refraction and aberration, and
  focal-length or distortion errors.
  In the baseline those undeclared biases (about 1 km) hide inside a 9 km
  declared uncertainty, so coverage looks healthy for the wrong reason.
  Clean residuals and a plausible ellipse do not show that the model is
  complete; they show that the remaining errors are small beside the ones
  that were declared.""")

    print("\n" + "=" * 78)
    print("Against the two objectives")
    print("=" * 78)
    for label, d in (("Baseline", base), ("Improved", improved)):
        h = np.array(d["horizontal"])
        print(f"  {label:<10} median {metres(d['median'])}, RMS {metres(d['rms'])}, "
              f"95th {metres(d['p95'])};  within 10 NM {100 * np.mean(h < PHASE1_M[1]):.0f}%, "
              f"6 NM {100 * np.mean(h < PHASE1_M[0]):.0f}%, 200 m "
              f"{100 * np.mean(h < PHASE2_M[1]):.0f}%, 100 m "
              f"{100 * np.mean(h < PHASE2_M[0]):.0f}%")
    top = sorted(improved["contributions"].items(), key=lambda kv: -kv[1]["rms"])[:3]
    print(f"""
  Dominant limitations, baseline: camera-IMU misalignment and accelerometer
  bias, which are fixed instrument errors that no amount of repeated
  observation removes.
  Dominant limitations, improved: {', '.join(f"{n.lower()} ({metres(c['rms'])})" for n, c in top)}.
  Remaining uncertainties this model does not cover: real star density and
  identification errors, real atmospheric conditions and their errors,
  measured deflection data, unmodelled acceleration (Module 5), thermal
  drift of the calibration, and whether the assumed error sizes are right.""")


def make_figures(cache):
    base, improved = cache["baseline"], cache["improved"]
    class_colors = {"random": "tab:blue", "fixed": "tab:red",
                    "environmental": "tab:green", "geographic": "tab:purple"}

    # 1. error budget
    fig, ax = plt.subplots(figsize=(13, 6.5))
    names = sorted(base["contributions"], key=lambda n: -base["contributions"][n]["rms"])
    x = np.arange(len(names) + 1)
    for offset, data, hatch, label in ((-0.2, base, None, "baseline"),
                                       (0.2, improved, "//", "improved")):
        values = [data["contributions"][n]["rms"] for n in names] + [data["rms"]]
        colors = [class_colors[data["contributions"][n]["category"]] for n in names] + ["black"]
        ax.bar(x + offset, np.maximum(values, 0.3), 0.4, color=colors, hatch=hatch,
               edgecolor="white", linewidth=0.5, alpha=0.9 if hatch is None else 0.55)
    for name, color in class_colors.items():
        ax.bar(0, 0, color=color, label=f"{name} error")
    ax.bar(0, 0, color="black", label="combined")
    ax.bar(0, 0, color="gray", hatch="//", alpha=0.55, label="hatched: improved scenario")
    ax.axhspan(*PHASE1_M, color="gold", alpha=0.25, label="6-10 NM objective")
    ax.axhspan(*PHASE2_M, color="limegreen", alpha=0.25, label="100-200 m objective")
    ax.set_yscale("log")
    ax.set_ylim(0.3, 5e4)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" of the ", " of\nthe ").replace(" ", "\n", 1)
                        for n in names] + ["Combined"], fontsize=8)
    ax.set_ylabel("RMS horizontal position error [m]")
    ax.set_title("Error budget: position-error contribution by source "
                 "(solid: baseline, hatched: improved)")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(HERE / "error_budget.png", dpi=130)
    plt.close(fig)

    # 2. improvement ranking
    rows = [r for r in cache["improvements"] if r["name"] != "baseline"]
    fig, ax = plt.subplots(figsize=(12, 6))
    y = np.arange(len(rows))[::-1]
    ax.barh(y + 0.2, [r["rms_reduction"] for r in rows], 0.4, color="tab:blue",
            label="reduction in RMS error")
    ax.barh(y - 0.2, [r["p95_reduction"] for r in rows], 0.4, color="tab:orange",
            label="reduction in 95th-percentile error")
    ax.set_yticks(y)
    ax.set_yticklabels([r["name"] for r in rows], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xscale("symlog", linthresh=50)
    ax.set_xlabel("Predicted reduction in horizontal position error [m] "
                  "(each change applied alone to the baseline)")
    ax.set_title(f"Improvement ranking (baseline RMS "
                 f"{metres(cache['improvements'][0]['rms'])})")
    ax.grid(True, axis="x", which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(HERE / "improvement_ranking.png", dpi=130)
    plt.close(fig)

    # 3. performance comparison
    steps = [{"name": "Baseline", "median": base["median"], "rms": base["rms"],
              "p95": base["p95"]}] + [dict(r, name="+ " + r["name"])
                                      for r in cache["cumulative"]]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(len(steps))
    ax.semilogy(x, [s["p95"] for s in steps], "v--", color="tab:red",
                label="95th percentile")
    ax.semilogy(x, [s["rms"] for s in steps], "o-", color="black", label="RMS")
    ax.semilogy(x, [s["median"] for s in steps], "s--", color="tab:blue",
                label="median")
    ax.axhspan(*PHASE1_M, color="gold", alpha=0.3, label="Phase 1: 6-10 NM")
    ax.axhspan(*PHASE2_M, color="limegreen", alpha=0.3, label="Phase 2: 100-200 m")
    ax.set_xticks(x)
    ax.set_xticklabels([s["name"] for s in steps], rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("Horizontal position error [m]")
    ax.set_title("Performance comparison: baseline and cumulative improvements "
                 "against the two objectives")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "performance_comparison.png", dpi=130)
    plt.close(fig)
    print("\nSaved error_budget.png, improvement_ranking.png and "
          "performance_comparison.png")


def main(arguments):
    if "--fresh" in arguments and CACHE.exists():
        CACHE.unlink()
    cache = load_cache()
    if "--stage" in arguments:
        stage = arguments[arguments.index("--stage") + 1]
        compute(stage, cache)
        return
    for stage in STAGES:
        if stage not in cache:
            cache = compute(stage, cache)
    cache = load_cache()
    report(cache)
    make_figures(cache)


if __name__ == "__main__":
    main(sys.argv[1:])
