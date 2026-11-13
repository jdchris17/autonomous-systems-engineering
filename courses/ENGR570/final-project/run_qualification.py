"""
run_qualification.py - Work Package D: the simulator's performance envelope.

Runs the complete engine (solve_navigation_fix, nothing else) against

  A-E   five representative operational scenarios
  F     twelve injected faults, to see what the integrity layer does
  G     a sweep across geography, pointing, star count, atmosphere,
        calibration quality and reference-data age, for two hardware classes

and writes qualification_report.txt and three figures.

The Monte Carlo takes about 20 minutes in total, so each stage is cached
in qualification_cache.json:

    python run_qualification.py                 report from the cache,
                                                computing any missing stage
    python run_qualification.py --stage B       recompute one stage
    python run_qualification.py --fresh         recompute everything

All error magnitudes are illustrative assumptions. Phase 1 (6-10 NM) and
Phase 2 (100-200 m) are engineering objectives, not device specifications.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from enduria.contracts import FIX_STATUSES, STATUSES  # noqa: E402
from enduria.qualification import (  # noqa: E402
    FAULTS,
    NAUTICAL_MILE,
    SCENARIOS,
    fault_scenario,
    run_scenario,
    run_trial,
    summarize,
)

HERE = Path(__file__).resolve().parent
CACHE = HERE / "qualification_cache.json"
REPORT = HERE / "qualification_report.txt"
SEED = 5710
TRIALS = {"A": 200, "B": 1000, "C": 500, "D": 500, "E": 500}
FAULT_TRIALS = 60
SWEEP_TRIALS = 400

# ---- the sweep (section 9) ---------------------------------------------------
STAR_COUNTS = [3, 4, 5, 6, 8, 10, 15, 20, 30, 40]
EOP_AGES = [0.0, 30.0, 180.0, 365.0, 730.0]
CALIBRATION_QUALITY = [0.3, 1.0, 3.0]            # multiplies the alignment sigma
# Illustrative frequencies of operational events, so that the probability of
# obtaining a fix is not trivially 100 %. They are assumptions, not data.
EVENTS = {"none": 0.84, "vibration_all": 0.05, "vibration_some": 0.05,
          "few_stars": 0.06}
SWEEP_BASE = {"C": "C", "D": "D"}                # hardware class -> scenario


def sweep(hardware, trials, seed):
    """Random configurations around one hardware class. Returns records."""
    rng = np.random.default_rng(seed)
    base = SCENARIOS[SWEEP_BASE[hardware]].with_changes(
        position="global", pressure_hpa=(1000.0, 30.0), temperature_c=(-30.0, 45.0))
    records = []
    for _ in range(trials):
        day = rng.uniform(0.0, 1040.0)               # 2024-01-01 .. 2026-11-06
        date = np.datetime64("2024-01-01T00:00:00") + np.timedelta64(
            int(day * 86400), "s")
        quality = float(rng.choice(CALIBRATION_QUALITY))
        factors = dict(
            n_stars=int(rng.choice(STAR_COUNTS)),
            max_zenith_deg=float(rng.uniform(25.0, 65.0)),
            eop_age_days=float(rng.choice(EOP_AGES)),
            misalignment_sigma_arcsec=base.misalignment_sigma_arcsec * quality,
            start_utc=str(date),
            fault=str(rng.choice(list(EVENTS), p=list(EVENTS.values()))))
        record = run_trial(base.with_changes(**factors), rng)
        record.update(calibration_quality=quality,
                      max_zenith=factors["max_zenith_deg"],
                      eop_age=factors["eop_age_days"], event=factors["fault"])
        records.append(record)
    return records


# ------------------------------------------------------------------------ caching
def _plain(value):
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def load_cache():
    return json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}


def save_cache(cache):
    CACHE.write_text(json.dumps(_plain(cache)), encoding="utf-8")


def compute(stage):
    started = time.time()
    if stage in TRIALS:
        summary, records = run_scenario(SCENARIOS[stage], TRIALS[stage],
                                        SEED + ord(stage))
        result = {"summary": summary, "records": records,
                  "description": SCENARIOS[stage].description}
    elif stage == "F":
        result = {}
        for index, fault in enumerate(FAULTS):
            summary, records = run_scenario(fault_scenario(fault), FAULT_TRIALS,
                                            SEED + 100 + index)
            result[fault] = {"summary": summary, "records": records}
    elif stage in ("G-C", "G-D"):
        hardware = stage[-1]
        result = {"records": sweep(hardware, SWEEP_TRIALS, SEED + 200 + ord(hardware))}
    else:
        raise SystemExit(f"unknown stage {stage!r}")
    result["seconds"] = time.time() - started
    return result


STAGES = ["A", "B", "C", "D", "E", "F", "G-C", "G-D"]


# ------------------------------------------------------------------------ report
def _m(value):
    if value is None or not np.isfinite(value):
        return "-"
    if value >= 10000:
        return f"{value / 1000:,.1f} km"
    if value >= 1000:
        return f"{value / 1000:,.2f} km"
    if value >= 1:
        return f"{value:,.0f} m"
    if value >= 1e-3:
        return f"{value * 1000:.3g} mm"
    return f"{value * 1e6:.2g} um"


def _pct(value):
    return "-" if value is None else f"{100 * value:5.1f}%"


def scenario_table(cache, out):
    out("Scenario                        trials  fix prob   median      RMS     95th"
        "   reported  NEES  95% cov  VALID DEGR  UNOB  REJ")
    for key in "ABCDE":
        s = cache[key]["summary"]
        c = s["status_counts"]
        out(f"{s['name']:<31} {s['trials']:>6} {_pct(s['fix_probability']):>9} "
            f"{_m(s.get('median_m')):>8} {_m(s.get('rms_m')):>8} "
            f"{_m(s.get('p95_m')):>8} {_m(s.get('reported_rms_m')):>9} "
            f"{s.get('mean_nees', float('nan')):5.2f} {_pct(s.get('coverage95')):>8} "
            f"{c['VALID']:>5} {c['DEGRADED']:>5} {c['UNOBSERVABLE']:>5} "
            f"{c['REJECTED']:>4}")
    out()
    out("  um = micrometres. Scenario A is DEGRADED by design: its perfect "
        "measurements fit better than the")
    out("  declared noise floor allows, and the engine reports that the stated "
        "uncertainties may be wrong.")
    out()
    out("Against the objectives (fraction of fixes within the stated radius):")
    out("Scenario                        within 10 NM  within 6 NM  within 200 m"
        "  within 100 m")
    for key in "BCDE":
        s = cache[key]["summary"]
        out(f"{s['name']:<31} {_pct(s['within_10nm']):>12} {_pct(s['within_6nm']):>12}"
            f" {_pct(s['within_200m']):>13} {_pct(s['within_100m']):>13}")


def fault_table(cache, out):
    out("Injected fault                                        VALID DEGR UNOB  REJ"
        "   fix error (median / max)   misleading")
    for fault, text in FAULTS.items():
        s = cache["F"][fault]["summary"]
        c = s["status_counts"]
        errors = (f"{_m(s['median_m'])} / {_m(s['max_m'])}" if s["fixes"] else "-")
        out(f"{text:<53} {c['VALID']:>5} {c['DEGRADED']:>4} {c['UNOBSERVABLE']:>4} "
            f"{c['REJECTED']:>4}   {errors:>24}   {s.get('misleading', 0):>5}")
    out()
    out("'misleading' counts fixes that were reported (VALID or DEGRADED) while "
        "the true error lay outside the")
    out("99.99 % region of the reported covariance. With 60 trials per fault the "
        "expected count for an honest")
    out("covariance is 0.006.")


def _bin_table(records, key, bins, labels, out, title):
    out(title)
    out(f"  {'':<22} configs  fix prob   median     95th      max   95% cov")
    for (low, high), label in zip(bins, labels):
        subset = [r for r in records if low <= r[key] < high]
        if not subset:
            continue
        s = summarize(subset)
        out(f"  {label:<22} {s['trials']:>7} {_pct(s['fix_probability']):>9} "
            f"{_m(s.get('median_m')):>8} {_m(s.get('p95_m')):>8} "
            f"{_m(s.get('max_m')):>8} {_pct(s.get('coverage95')):>8}")


def sweep_tables(cache, out):
    for hardware, label in (("C", "phone-class hardware, calibrated in the batch"),
                            ("D", "precision hardware")):
        records = cache[f"G-{hardware}"]["records"]
        s = summarize(records)
        errors = np.array([r["error_m"] for r in records
                           if r["status"] in FIX_STATUSES])
        out(f"--- {label}: {len(records)} random configurations ---")
        out(f"  Probability of obtaining a fix: {_pct(s['fix_probability'])}   "
            f"(VALID {s['status_counts']['VALID']}, DEGRADED "
            f"{s['status_counts']['DEGRADED']}, UNOBSERVABLE "
            f"{s['status_counts']['UNOBSERVABLE']}, REJECTED "
            f"{s['status_counts']['REJECTED']})")
        out("  Accuracy of a fix once obtained (percentiles of horizontal error):")
        out("     " + "   ".join(f"{p}%: {_m(np.percentile(errors, p))}"
                                 for p in (10, 50, 68, 90, 95, 99)))
        out(f"     best {_m(errors.min())}, worst {_m(errors.max())}, RMS "
            f"{_m(s['rms_m'])}; inside reported 95 % ellipse {_pct(s['coverage95'])}"
            f"; misleading {s['misleading']}")
        events = {}
        for r in records:
            events.setdefault(r["event"], []).append(r)
        out("  Fix probability by operational event: " + ", ".join(
            f"{name} {_pct(summarize(rs)['fix_probability']).strip()} "
            f"({len(rs)})" for name, rs in events.items()))
        clean = [r for r in records if r["event"] == "none"]
        absolute = [dict(r, latitude=abs(r["latitude"])) for r in clean]
        _bin_table(absolute, "latitude", [(0, 30), (30, 60), (60, 90)],
                   ["|lat| 0-30 deg", "|lat| 30-60 deg", "|lat| 60-85 deg"], out,
                   "  By latitude (configurations without an operational event):")
        _bin_table(clean, "max_zenith", [(25, 40), (40, 55), (55, 66)],
                   ["to 25-40 deg zenith", "to 40-55 deg", "to 55-65 deg"], out,
                   "  By how far from the zenith the sequence points:")
        _bin_table(clean, "n_stars", [(3, 6), (6, 11), (15, 41)],
                   ["3-5 stars", "6-10 stars", "15-40 stars"], out,
                   "  By matched stars per exposure:")
        _bin_table(clean, "calibration_quality", [(0.2, 0.5), (0.5, 2), (2, 4)],
                   ["alignment 0.3x sigma", "alignment 1x", "alignment 3x"], out,
                   "  By calibration quality (declared honestly in each case):")
        _bin_table(clean, "eop_age", [(0, 1), (1, 200), (200, 400), (400, 800)],
                   ["EOP current", "30-180 days old", "1 year old", "2 years old"],
                   out, "  By age of the Earth-orientation data:")
        out()


def write_report(cache):
    lines = []

    def out(text=""):
        lines.append(text)

    out("ENDURIA NAVIGATION ENGINE - PERFORMANCE QUALIFICATION (Work Package D)")
    out("=" * 72)
    out("Every number below is the output of solve_navigation_fix() on a "
        "simulated batch.")
    out("Error magnitudes are illustrative assumptions; Phase 1 (6-10 NM = "
        "11.1-18.5 km) and")
    out("Phase 2 (100-200 m) are objectives, not device specifications.")
    out(f"Base seed {SEED}. Accuracy statistics are over trials that produced a "
        "fix; availability is separate.")
    out()
    out("1. REPRESENTATIVE SCENARIOS")
    out("-" * 72)
    for key in "ABCDE":
        out(f"  {SCENARIOS[key].name}: {SCENARIOS[key].description}")
    out()
    scenario_table(cache, out)
    out()
    out("2. INTEGRITY RESPONSE (scenario F: phone-class hardware, one fault per "
        f"trial, {FAULT_TRIALS} trials each)")
    out("-" * 72)
    fault_table(cache, out)
    out()
    out("3. PERFORMANCE ACROSS GEOGRAPHY AND OBSERVATION GEOMETRY")
    out("-" * 72)
    out("Each configuration draws: a place anywhere within 85 deg of the equator, "
        "a date in 2024-2026,")
    out(f"matched stars per exposure from {STAR_COUNTS}, a pointing limit of 25-65 "
        "deg from the zenith,")
    out("pressure 1000 +/- 30 hPa, temperature -30 to +45 C (measured by the "
        "instrument), calibration")
    out(f"quality {CALIBRATION_QUALITY} x nominal, Earth-orientation data age from "
        f"{[int(a) for a in EOP_AGES]} days,")
    out("and an operational event with assumed probability "
        + ", ".join(f"{k} {100 * v:.0f}%" for k, v in EVENTS.items()) + ".")
    out()
    sweep_tables(cache, out)
    seconds = sum(cache[s].get("seconds", 0.0) for s in STAGES)
    out(f"Monte Carlo time for all stages: {seconds / 60:.1f} minutes.")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


# ------------------------------------------------------------------------ figures
COLORS = {"B": "tab:red", "C": "tab:orange", "D": "tab:green", "E": "tab:purple"}
STATUS_COLORS = {"VALID": "tab:green", "DEGRADED": "gold",
                 "UNOBSERVABLE": "tab:gray", "REJECTED": "tab:red"}


def _fix_errors(records):
    return np.array([r["error_m"] for r in records if r["status"] in FIX_STATUSES])


def figure_scenarios(cache):
    fig, (ax, bars) = plt.subplots(1, 2, figsize=(14, 5.5),
                                   gridspec_kw={"width_ratios": [1.6, 1]})
    for key in "BCDE":
        errors = np.sort(_fix_errors(cache[key]["records"]))
        ax.plot(errors, np.arange(1, len(errors) + 1) / len(errors),
                color=COLORS[key], lw=2, label=SCENARIOS[key].name)
    ax.axvspan(6 * NAUTICAL_MILE, 10 * NAUTICAL_MILE, color="tab:red", alpha=0.10)
    ax.axvspan(100, 200, color="tab:green", alpha=0.12)
    ax.text(6.2 * NAUTICAL_MILE, 0.03, "Phase 1\n6-10 NM", fontsize=9, va="bottom")
    ax.text(103, 0.03, "Phase 2\n100-200 m", fontsize=9, va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(5, 60000)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Horizontal position error [m]")
    ax.set_ylabel("Fraction of fixes with smaller error")
    ax.set_title("Error distribution of each scenario (complete engine)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    keys = list("BCDE")
    x = np.arange(len(keys))
    actual = [cache[k]["summary"]["rms_m"] for k in keys]
    reported = [cache[k]["summary"]["reported_rms_m"] for k in keys]
    bars.bar(x - 0.2, actual, 0.4, color=[COLORS[k] for k in keys],
             label="actual RMS error")
    bars.bar(x + 0.2, reported, 0.4, color="lightgray", edgecolor="k",
             label="RMS of reported 1-sigma")
    for i, k in enumerate(keys):
        s = cache[k]["summary"]
        bars.text(i, max(actual[i], reported[i]) * 1.25,
                  f"95% cover\n{100 * s['coverage95']:.1f}%", ha="center", fontsize=8)
    bars.set_yscale("log")
    bars.set_ylim(10, 60000)
    bars.set_xticks(x, keys)
    bars.set_ylabel("metres")
    bars.set_title("Does the reported uncertainty match?")
    bars.grid(True, axis="y", which="both", alpha=0.3)
    bars.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(HERE / "qualification_scenarios.png", dpi=150)
    plt.close(fig)


def figure_integrity(cache):
    faults = list(FAULTS)
    fig, ax = plt.subplots(figsize=(12, 6))
    left = np.zeros(len(faults))
    for status in STATUSES:
        counts = np.array([cache["F"][f]["summary"]["status_counts"][status]
                           for f in faults])
        ax.barh(np.arange(len(faults)), counts, left=left,
                color=STATUS_COLORS[status], edgecolor="white", label=status)
        left += counts
    for i, fault in enumerate(faults):
        s = cache["F"][fault]["summary"]
        note = ("no position reported" if not s["fixes"] else
                f"fix median {_m(s['median_m'])}, max {_m(s['max_m'])}"
                + (f"  [{s['misleading']} misleading]" if s.get("misleading") else ""))
        ax.text(FAULT_TRIALS + 1, i, note, va="center", fontsize=9)
    ax.set_yticks(np.arange(len(faults)), [FAULTS[f] for f in faults], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, FAULT_TRIALS * 1.75)
    ax.set_xlabel(f"Trials (of {FAULT_TRIALS})")
    ax.set_title("Integrity response to injected faults (scenario F, "
                 "phone-class hardware)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=4, fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "qualification_integrity.png", dpi=150)
    plt.close(fig)


def figure_sweep(cache):
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), layout="constrained")
    for row, (hardware, label) in enumerate((("C", "Phone-class, calibrated"),
                                             ("D", "Precision hardware"))):
        records = cache[f"G-{hardware}"]["records"]
        fixes = [r for r in records if r["status"] in FIX_STATUSES]
        lat = np.array([r["latitude"] for r in fixes])
        err = np.array([r["error_m"] for r in fixes])
        stars = np.array([r["n_stars"] for r in fixes])
        ax = axes[row, 0]
        scatter = ax.scatter(lat, err, c=np.log10(stars), cmap="viridis", s=14)
        no_fix = [r["latitude"] for r in records if r["status"] not in FIX_STATUSES]
        ax.plot(no_fix, np.full(len(no_fix), 0.03), "x", color="tab:red", ms=6,
                transform=ax.get_xaxis_transform(), label="no fix reported")
        ax.set_yscale("log")
        ax.set_xlabel("Latitude [deg]")
        ax.set_ylabel("Horizontal error of the fix [m]")
        ax.set_title(f"{label}: every configuration")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)
        colorbar = fig.colorbar(scatter, ax=ax, ticks=np.log10([3, 5, 10, 20, 40]))
        colorbar.ax.set_yticklabels(["3", "5", "10", "20", "40"])
        colorbar.set_label("matched stars per exposure")

        ax = axes[row, 1]
        groups = [err[stars == n] for n in STAR_COUNTS]
        ax.boxplot(groups, tick_labels=[str(n) for n in STAR_COUNTS], whis=(5, 95),
                   showfliers=False)
        ax.set_yscale("log")
        ax.set_xlabel("Matched stars per exposure")
        ax.set_ylabel("Horizontal error of the fix [m]")
        ax.set_title("Error against star count\n(box 25-75 %, whiskers 5-95 %)")
        ax.grid(True, axis="y", which="both", alpha=0.3)

        ax = axes[row, 2]
        errors = np.sort(err)
        ax.plot(errors, np.arange(1, len(errors) + 1) / len(errors), lw=2,
                color=COLORS[hardware])
        s = summarize(records)
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Horizontal error of the fix [m]")
        ax.set_ylabel("Fraction of fixes")
        ax.set_title(f"Fix obtained in {100 * s['fix_probability']:.1f}% of "
                     f"configurations\nmedian {_m(s['median_m'])}, 95th "
                     f"{_m(s['p95_m'])}, worst {_m(s['max_m'])}")
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("Performance across geography, star field, atmosphere, "
                 "calibration quality and reference-data age", fontsize=14)
    fig.savefig(HERE / "qualification_coverage.png", dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--no-compute", action="store_true",
                        help="report only what is cached")
    args = parser.parse_args()
    cache = {} if args.fresh else load_cache()
    todo = [args.stage] if args.stage else [s for s in STAGES if s not in cache]
    if args.no_compute:
        todo = []
    for stage in todo:
        print(f"computing stage {stage} ...", flush=True)
        result = compute(stage)
        cache = {} if (args.fresh and stage == todo[0]) else load_cache()
        cache[stage] = result                 # merge with what is on disk now
        save_cache(cache)
        print(f"  done in {cache[stage]['seconds']:.0f} s", flush=True)
    missing = [s for s in STAGES if s not in cache]
    if missing:
        print(f"stages not yet computed: {missing}")
        return 1
    write_report(cache)
    figure_scenarios(cache)
    figure_integrity(cache)
    figure_sweep(cache)
    print("\nSaved qualification_report.txt, qualification_scenarios.png, "
          "qualification_integrity.png, qualification_coverage.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
