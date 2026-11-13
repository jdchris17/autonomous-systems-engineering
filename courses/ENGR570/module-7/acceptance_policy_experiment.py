"""
acceptance_policy_experiment.py - Space Systems Engineering, Module 7, Experiment D

A navigation solution acceptance policy: report not just coordinates and a
covariance, but whether the assumptions behind them held, and say plainly
what cannot be checked.

Statuses (navigation_estimator.assess_navigation_solution)
  VALID         observable, converged, consistent with the declared
                measurement assumptions. Conditional on those assumptions.
  DEGRADED      usable, but uncertainty or reference data limit confidence.
  UNOBSERVABLE  the measurements do not uniquely constrain the state.
  REJECTED      a measurement or model validity check failed.

Cases
  1  well-conditioned stationary solution, correctly modelled noise
  2  one anomalous celestial-vertical measurement
  3  augmented state with an unobservable clock offset
  4  stale or uncertain Earth-orientation data
  5  biased or accelerating instrument whose measurements are self-consistent

Run:  python acceptance_policy_experiment.py
"""

import numpy as np

import navigation_estimator as ne
import navigation_scenarios as ns
from gravity_estimator import tangent_basis

SEED = 7007
cp = ne.cp


def metres(value):
    return f"{value / 1000:.2f} km" if value >= 1000 else f"{value:.0f} m"


def solve(measurements, **kwargs):
    return ne.batch_position_estimate(measurements, ns.single_fix(measurements[0]),
                                      **kwargs)


def show(title, result, assessment, note=None):
    error = cp.horizontal_position_error(result.lat_lon, ns.TRUE_POSITION)
    sigma = ("withheld" if result.position_covariance_ne is None
             else metres(np.sqrt(np.trace(result.position_covariance_ne))))
    metrics = assessment.metrics
    print(f"\n{title}")
    print(f"  status: {assessment.status}")
    for reason in assessment.reasons:
        print(f"    - {reason}")
    chi2_text = (f"chi2 {metrics['chi2']:.1f} / {metrics['dof']} dof"
                 if metrics["dof"] > 0 else "no redundancy")
    print(f"  reported: ({result.latitude_deg:.5f}, {result.longitude_deg:.5f}), "
          f"1-sigma {sigma}; {chi2_text}; {metrics['observability']}")
    print(f"  truth check (not available to the software): actual error "
          f"{metres(error)}")
    if note:
        print(f"  {note}")
    return error


def main():
    rng = np.random.default_rng(SEED)
    summary = []

    # ---- case 1 -----------------------------------------------------------------
    measurements, _ = ns.simulate_batch(rng)
    result = solve(measurements)
    assessment = ne.assess_batch_result(result, measurements)
    reference = assessment.metrics["reference"]
    error = show("Case 1 - stationary, correctly modelled random noise",
                 result, assessment,
                 note=f"Earth orientation: {reference['source']}, "
                      f"{reference['days_predicted']:.1f} days past the last "
                      f"measured value, UT1 sigma {reference['ut1_sigma_s'] * 1000:.1f} ms "
                      f"= {assessment.metrics['reference_east_error_m']:.1f} m east.")
    summary.append(("1 clean stationary", assessment.status, error))

    # ---- case 2 -----------------------------------------------------------------
    measurements, _ = ns.simulate_batch(rng)
    bad = measurements[11]
    kick = tangent_basis(bad.up_celestial)[0] * 300 * ne.ARCSEC
    bad.up_celestial = (bad.up_celestial + kick) / np.linalg.norm(bad.up_celestial + kick)
    result = solve(measurements)
    assessment = ne.assess_batch_result(result, measurements)
    worst = int(np.argmax(assessment.metrics["measurement_chi2"]))
    error = show("Case 2 - one anomalous measurement (observation 11 displaced "
                 "by 300 arcsec)", result, assessment,
                 note=f"Largest normalized residual: observation {worst}, "
                      f"{assessment.metrics['measurement_chi2'][worst]:.0f} "
                      f"(about 2 expected).")
    summary.append(("2 one anomalous measurement", assessment.status, error))
    cleaned = [m for i, m in enumerate(measurements) if i != worst]
    result = solve(cleaned)
    assessment = ne.assess_batch_result(result, cleaned)
    error = show("Case 2b - the same batch with that observation removed",
                 result, assessment)
    summary.append(("2b anomaly removed", assessment.status, error))

    # ---- case 3 -----------------------------------------------------------------
    measurements, _ = ns.simulate_batch(rng)
    result = solve(measurements, estimate_clock=True)
    assessment = ne.assess_batch_result(result, measurements)
    weakest = assessment.metrics["weakest_direction"]
    error = show("Case 3 - clock offset added to the state", result, assessment,
                 note=f"Null direction (scaled units): {weakest}. The optimizer "
                      f"converged and the residuals are fine;\n  the answer "
                      f"is one of infinitely many. Numerical convergence is "
                      f"not observability.")
    summary.append(("3 clock offset in state", assessment.status, error))

    # ---- case 4 -----------------------------------------------------------------
    measurements, _ = ns.simulate_batch(rng)
    result = solve(measurements)
    current = ne.reference_status(measurements[0].utc)
    stale = dict(current, days_predicted=730.0, is_prediction=True,
                 ut1_sigma_s=0.00025 * 730.0**0.75,
                 source="IERS-A prediction, table two years old")
    missing = {"available": False, "source": "no Earth-orientation table",
               "is_prediction": False, "ut1_sigma_s": 0.9}
    print("\nCase 4 - Earth-orientation data of decreasing quality (same "
          "measurements and position fit)")
    for label, reference in (("current table", current),
                             ("table two years stale", stale),
                             ("no table at all", missing)):
        assessment = ne.assess_batch_result(result, measurements, reference)
        print(f"  {label:<24} UT1 sigma {reference['ut1_sigma_s'] * 1000:>7.1f} ms"
              f" -> {assessment.metrics['reference_east_error_m']:>6.1f} m east,"
              f" against {assessment.metrics['east_sigma_m']:.0f} m reported"
              f" east sigma  =>  {assessment.status}")
        for reason in assessment.reasons:
            if "Earth-orientation" in reason:
                print(f"      - {reason}")
        summary.append((f"4 EOP: {label}", assessment.status,
                        cp.horizontal_position_error(result.lat_lon,
                                                     ns.TRUE_POSITION)))
    large, _ = ns.simulate_batch(rng, 200, run_quality_check=False)
    result_large = solve(large)
    assessment = ne.assess_batch_result(result_large, large, stale)
    print(f"  two years stale, 200 observations: reported east sigma "
          f"{assessment.metrics['east_sigma_m']:.0f} m  =>  {assessment.status}")
    print("  The rule is relative: reference-data error matters once it is "
          "no longer small beside the\n  uncertainty the solution claims. "
          "The better the measurement, the fresher the data must be.")
    summary.append(("4 EOP: two years stale, 200 obs", assessment.status,
                    cp.horizontal_position_error(result_large.lat_lon,
                                                 ns.TRUE_POSITION)))

    # ---- case 5 -----------------------------------------------------------------
    measurements, _ = ns.simulate_batch(rng, acceleration_enu=(0.05, 0.0, 0.0))
    result = solve(measurements)
    assessment = ne.assess_batch_result(result, measurements)
    verdicts = sorted({m.quality.verdict for m in measurements})
    error = show("Case 5a - instrument accelerating at 0.05 m/s^2 east "
                 "throughout", result, assessment,
                 note=f"Accelerometer quality verdicts: {verdicts}.")
    summary.append(("5a constant acceleration", assessment.status, error))

    measurements, _ = ns.simulate_batch(
        rng, accel_bias=[0.004, -0.006, 0.002],
        misalignment_rad=np.radians([0.05, -0.03, 0.04]))
    result = solve(measurements)
    assessment = ne.assess_batch_result(result, measurements)
    error = show("Case 5b - fixed accelerometer bias and camera-IMU "
                 "misalignment, not modelled", result, assessment)
    summary.append(("5b fixed bias + misalignment", assessment.status, error))

    aware = solve(measurements, shared_errors={
        "accelerometer_bias": ns.ACCEL_BIAS_SIGMA**2 * np.eye(3),
        "calibration": ns.MISALIGNMENT_SIGMA**2 * np.eye(3)})
    assessment = ne.assess_batch_result(aware, measurements)
    error = show("Case 5c - the same data, with calibration uncertainty "
                 "declared to the estimator", aware, assessment)
    summary.append(("5c calibration uncertainty declared", assessment.status, error))

    print(f"""
What case 5 shows
  5a and 5b are VALID and wrong by tens of kilometres and kilometres, with
  residuals indistinguishable from case 1. No check failed because there
  is nothing in the data to fail: a constant acceleration or a fixed
  calibration error moves every observation together.
  The policy therefore cannot certify stationarity or calibration. It says
  so in every VALID or DEGRADED result ("stationarity and calibration are
  assumed, not verified") and carries this caveat:
    {assessment.caveat}
  5c shows what the software CAN do: if the size of the calibration
  uncertainty is declared, the reported sigma grows to cover the error.
  That makes the uncertainty honest. It still depends on the declared
  prior being right, and it does nothing for an undeclared acceleration.
  A low residual is not proof of correctness.""")

    print("\nSummary")
    print(f"  {'case':<38}{'status':<14}{'actual error':>14}")
    for label, status, error in summary:
        print(f"  {label:<38}{status:<14}{metres(error):>14}")


if __name__ == "__main__":
    main()
