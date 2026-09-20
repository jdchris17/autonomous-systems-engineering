"""Monte Carlo validation: simulate the sensor and check the empirical
positive-predictive-value against the analytical Bayesian posterior."""

import numpy as np

from m02_bayes_sensor import bayes_update

N = 1_000_000
PRIOR = 0.01
DETECTION_PROBABILITY = 0.95
FALSE_ALARM_PROBABILITY = 0.05


def simulate(n, prior, detection_probability, false_alarm_probability, rng):
    """Run n independent trials of "target present? -> sensor report?".

    Returns two boolean arrays of length n: whether the target was actually
    present, and whether the sensor reported positive.
    """
    target_present = rng.random(n) < prior

    report_prob = np.where(target_present, detection_probability, false_alarm_probability)
    positive_report = rng.random(n) < report_prob

    return target_present, positive_report


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    target_present, positive_report = simulate(
        N, PRIOR, DETECTION_PROBABILITY, FALSE_ALARM_PROBABILITY, rng
    )

    all_positive_detections = np.count_nonzero(positive_report)
    true_positive_detections = np.count_nonzero(target_present & positive_report)
    empirical_posterior = true_positive_detections / all_positive_detections

    theoretical = bayes_update(PRIOR, DETECTION_PROBABILITY, FALSE_ALARM_PROBABILITY)

    print(f"Simulated observations:         {N:,}")
    print(f"Positive detections (total):    {all_positive_detections:,}")
    print(f"Positive detections (true):     {true_positive_detections:,}")
    print()
    print(f"Empirical  P(T | +) = {empirical_posterior:.4f}")
    print(f"Analytical P(T | +) = {theoretical['posterior']:.4f}")
    print(f"Difference           = {abs(empirical_posterior - theoretical['posterior']):.5f}")
