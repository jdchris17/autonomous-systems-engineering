"""
navigation_observability.py - Space Systems Engineering, Module 7, Experiment C

Which parameters can the celestial-vertical measurements determine?

For a candidate state vector the whitened measurement Jacobian is formed,
each state is scaled to "one arcsecond of effect" (see
navigation_estimator.state_scales), and the singular values are inspected:

  well conditioned    every direction in state space is constrained
  poorly conditioned  full rank, but some direction is constrained to worse
                      than 1 degree: mathematically observable, useless
  rank deficient      some combination of states does not affect the
                      measurements at all

C1  x1 = [lat, lon]
C2  x2 = [lat, lon, clock offset], simplified and full Earth models
C3  x3 = [lat, lon, calibration error (3)], four attitude sequences,
    absolute time known

Run:  python navigation_observability.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import navigation_estimator as ne
import navigation_scenarios as ns

SEED = 707
HERE = Path(__file__).parent

ATTITUDE_CONFIGURATIONS = {
    "A: 20 identical attitudes": [ns.DEFAULT_ATTITUDE] * 20,
    "B: 4 observations, rolled 90° each": [(30.0, 40.0, 15.0 + 90.0 * k)
                                           for k in range(4)],
    "C: 8 with varied roll and tilt": [(z, az, r) for z, az, r in zip(
        [10, 25, 40, 55, 15, 35, 50, 30], [0, 50, 110, 170, 220, 260, 310, 80],
        [0, 45, 90, 135, 180, 225, 270, 315])],
    "D: 20 deliberately diverse": ns.diverse_attitudes(20),
}


def observability_report(measurements, position_deg, *, estimate_clock=False,
                         estimate_calibration=False, clock_model="simple"):
    """
    Singular-value analysis of the whitened, scaled measurement Jacobian at
    a position (clock offset and calibration error taken as zero).

    Returns a dict with the state names, singular values, classification,
    numerical rank, the weakest direction in scaled state space, and the
    1-sigma uncertainty along it in arcseconds of effect (infinite if rank
    deficient).
    """
    problem = ne._Problem(measurements, estimate_clock, estimate_calibration,
                          clock_model, None, None)
    x = np.zeros(len(problem.names))
    x[:2] = np.radians(position_deg)
    scales = ne.state_scales(x[0], problem.names)
    _, s, vt = np.linalg.svd(problem.jacobian(x) * scales, full_matrices=False)
    classification = ne.classify_observability(s)
    return {
        "names": list(problem.names),
        "singular_values": s,
        "classification": classification,
        "rank": int(np.sum(s > ne.RANK_TOLERANCE * s.max())),
        "condition_number": float(s.max() / max(s.min(), 1e-300)),
        "weakest_direction": dict(zip(problem.names, vt[-1])),
        "weakest_sigma_arcsec": (float("inf") if classification == "rank deficient"
                                 else float(1.0 / s.min())),
        "scales": scales,
    }


def describe(label, report):
    s = report["singular_values"]
    weak = report["weakest_direction"]
    parts = ", ".join(f"{name} {value:+.2f}" for name, value in weak.items()
                      if abs(value) > 0.15)
    sigma = report["weakest_sigma_arcsec"]
    sigma_text = "unbounded" if np.isinf(sigma) else f"{sigma:,.1f} arcsec"
    print(f"  {label}")
    print(f"    singular values: {'  '.join(f'{v:.2e}' for v in s)}")
    print(f"    rank {report['rank']} of {len(s)}, condition number "
          f"{report['condition_number']:.1e}  ->  {report['classification'].upper()}")
    print(f"    weakest direction: {parts};  1-sigma along it: {sigma_text}")


def main():
    rng = np.random.default_rng(SEED)
    cases = {}

    print("=" * 78)
    print("C1 - latitude and longitude only")
    print("=" * 78)
    fixed, _ = ns.simulate_batch(rng, 20, run_quality_check=False)
    position = ne.batch_position_estimate(fixed, ns.single_fix(fixed[0])).lat_lon
    cases["x1 = [lat, lon]\n20 observations"] = observability_report(fixed, position)
    describe("x1 = [lat, lon], 20 observations", cases["x1 = [lat, lon]\n20 observations"])
    print("    Both singular values are of order one: each is the number of "
          "sigmas the data moves per\n    arcsecond of state change. "
          "Latitude and longitude are both well determined.")
    print("\n  The same state at other latitudes (longitude is scaled by "
          "1/cos(lat), so this tests the\n  physical east-west constraint, "
          "not the coordinate):")
    for lat in (0.0, 60.0, 85.0, 89.9):
        batch, _ = ns.simulate_batch(rng, 20, position=(lat, -75.0),
                                     run_quality_check=False)
        report = observability_report(batch, (lat, -75.0))
        raw = ne._Problem(batch, False, False, "simple", None, None)
        unscaled = np.linalg.svd(raw.jacobian(np.radians([lat, -75.0]))
                                 * ne.ARCSEC, compute_uv=False)
        print(f"    latitude {lat:>5}: scaled {report['singular_values'].round(3)}, "
              f"{report['classification']};  per arcsec of raw longitude: "
              f"{unscaled.min():.4f}")
    print("    The physical constraint is the same everywhere. The raw "
          "longitude coordinate loses its\n    grip as cos(lat): the polar "
          "coordinate singularity of Module 6, not a loss of information.")

    print("\n" + "=" * 78)
    print("C2 - latitude, longitude and clock offset")
    print("=" * 78)
    simple = observability_report(fixed, position, estimate_clock=True)
    cases["x2 = [lat, lon, Δt]\nsimplified Earth model"] = simple
    describe("Simplified Earth rotation (constant rate about the z axis)", simple)
    null = simple["weakest_direction"]
    print(f"    The null direction is equal and opposite longitude and clock "
          f"(in scaled units {null['longitude']:+.3f}, {null['clock']:+.3f}):\n"
          f"    moving the observer east by X arcsec and the clock back by "
          f"X/15.04 s changes nothing.\n    Only longitude + rate x time is "
          f"measured. No covariance is reported for this state.")

    per_observation = np.sqrt(np.mean([np.trace(m.covariance) for m in fixed])) / ne.ARCSEC
    print(f"\n  Full Earth-orientation model (Module 2), measurement noise "
          f"{per_observation:.0f} arcsec per observation,\n  20 observations "
          f"spread over different time spans:")
    for span_label, spacing in (("19 seconds", 1.0), ("6 hours", 6 * 3600 / 19),
                                ("24 hours", 24 * 3600 / 19)):
        batch = fixed if spacing == 1.0 else ns.simulate_batch(
            rng, 20, spacing_s=spacing, run_quality_check=False)[0]
        full = observability_report(batch, position, estimate_clock=True,
                                    clock_model="full")
        cases[f"x2 = [lat, lon, Δt]\nfull model, {span_label}"] = full
        describe(f"span {span_label}", full)
        if not np.isinf(full["weakest_sigma_arcsec"]):
            sigma = full["weakest_sigma_arcsec"]
            clock_sigma = (sigma * abs(full["weakest_direction"]["clock"])
                           * ne.ARCSEC / ne.EARTH_ROTATION_RATE_UT1)
            print(f"    implied clock uncertainty about {clock_sigma:,.0f} s "
                  f"({clock_sigma / 86400:,.1f} days), i.e. longitude "
                  f"unconstrained around the whole globe")
    print("""
    In the full model time is not exactly the same as longitude: Earth
    turns about its rotation pole, 0.34 arcsec from the Earth-fixed z axis,
    and precession and nutation move slowly. But at any one instant a
    clock offset still displaces the vertical within the tangent plane,
    where a small change of latitude AND longitude can absorb it. The only
    leverage left is how slowly the pole moves during the batch, and the
    smallest singular value stays many orders of magnitude below one.
    Whether the label is "rank deficient" or "poorly conditioned", the
    conclusion is the same: a clock offset cannot be estimated from these
    measurements, and inverting the information matrix would produce a
    finite clock covariance that means nothing. The estimator withholds it.""")
    result = ne.batch_position_estimate(fixed, ns.single_fix(fixed[0]),
                                        estimate_clock=True, clock_model="full")
    print(f"    batch_position_estimate(estimate_clock=True, clock_model='full') "
          f"-> observability '{result.observability}', covariance "
          f"{result.covariance}")

    print("\n" + "=" * 78)
    print("C3 - latitude, longitude and camera-IMU calibration error")
    print("=" * 78)
    print("  Absolute time is known. No prior is placed on the calibration "
          "states.\n")
    for label, attitudes in ATTITUDE_CONFIGURATIONS.items():
        batch, _ = ns.simulate_batch(rng, len(attitudes), attitudes=attitudes,
                                     run_quality_check=False)
        start = ne.batch_position_estimate(batch, ns.single_fix(batch[0])).lat_lon
        report = observability_report(batch, start, estimate_calibration=True)
        cases[f"x3 = [lat, lon, δθ]\n{label}"] = report
        describe(label, report)
    print("""
  A: with one attitude a calibration error shifts every measurement by the
     same amount, exactly like a change of position, and rotation about
     the vertical does nothing at all. Five unknowns, two constrained. The
     other three singular values are not exactly zero only because Earth
     turns 5 arcminutes during the 20 s and each star solution differs
     slightly; they constrain nothing useful (tens of degrees).
  B: rolling about the boresight turns the error's effect around in the
     sky, which separates the two calibration components across the
     boresight from position. A calibration rotation about the roll axis
     itself produces the same celestial shift in every observation, so it
     is still indistinguishable from a change of position.
  C, D: changing the tilt as well moves the vertical within the camera
     frame, and all three calibration components become distinguishable
     from position and from each other. More and more varied attitudes
     raise the smallest singular value.
  Calibration is therefore something a deliberate manoeuvre can estimate
  and a static instrument cannot.""")

    make_figure(cases)
    return cases


def make_figure(cases):
    colors = {"well conditioned": "tab:green", "poorly conditioned": "tab:orange",
              "rank deficient": "tab:red"}
    floor = 1e-11
    fig, ax = plt.subplots(figsize=(21, 7.5))
    labels = []
    for index, (label, report) in enumerate(cases.items()):
        s = report["singular_values"]
        color = colors[report["classification"]]
        offsets = np.linspace(-0.28, 0.28, len(s)) if len(s) > 1 else [0.0]
        for offset, value in zip(offsets, s):
            plotted = max(value, floor)
            ax.plot([index + offset] * 2, [floor, plotted], color=color, linewidth=2.5)
            ax.plot(index + offset, plotted, "o" if value > floor else "v",
                    color=color, markersize=8)
        labels.append(label)
    weak = 1.0 / ne.WEAK_SIGMA_LIMIT_ARCSEC
    ax.axhline(weak, color="gray", linestyle="--", linewidth=1)
    ax.text(len(cases) - 0.55, weak * 1.4,
            "below this line: constrained to worse than 1°", fontsize=8,
            color="gray", ha="right")
    for name, color in colors.items():
        ax.plot([], [], "o-", color=color, label=name)
    ax.plot([], [], "v", color="tab:red", label="zero to machine precision "
            "(drawn at the floor)")
    ax.set_yscale("log")
    ax.set_ylim(floor / 3, 20)
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels([label.replace(', rolled', ',' + chr(10) + 'rolled').replace(' with varied', chr(10) + 'with varied').replace(' deliberately', chr(10) + 'deliberately') for label in labels], fontsize=8)
    ax.set_ylabel("Singular values of the whitened, scaled Jacobian\n"
                  "[sigmas per arcsecond of state change]")
    ax.set_title("Observability of each state vector and observing configuration")
    ax.grid(True, axis="y", which="major", alpha=0.3)
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4)
    fig.tight_layout()
    fig.savefig(HERE / "navigation_observability.png", dpi=140)
    print("\nSaved navigation_observability.png")


if __name__ == "__main__":
    main()
