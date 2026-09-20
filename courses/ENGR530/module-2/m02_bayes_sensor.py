"""Bayesian sensor updater: turn a noisy detector's report into a posterior."""

import matplotlib.pyplot as plt
import numpy as np


def bayes_update(prior, detection_probability, false_alarm_probability):
    """Update a prior on "target present" given a positive sensor report.

    prior                  -- P(T),   probability the target is present before any report
    detection_probability  -- P(+|T), probability of a positive report given the target is present
    false_alarm_probability-- P(+|~T),probability of a positive report given no target

    Returns a dict with the four pieces of Bayes' rule:
        P(T|+) = P(+|T) P(T) / P(+)
    """
    likelihood = detection_probability
    evidence = likelihood * prior + false_alarm_probability * (1 - prior)
    posterior = likelihood * prior / evidence
    return {
        "prior": prior,
        "likelihood": likelihood,
        "evidence": evidence,
        "posterior": posterior,
    }


def print_report(result):
    print(f"Prior target probability:      {result['prior']:.4f}")
    print(f"Detection probability:         {result['likelihood']:.4f}")
    print(f"Probability of positive report: {result['evidence']:.4f}")
    print(f"Posterior target probability:  {result['posterior']:.4f}")


def plot_prior_sweep(detection_probability, false_alarm_probability, path):
    priors = np.linspace(0.001, 0.5, 500)
    posteriors = [
        bayes_update(p, detection_probability, false_alarm_probability)["posterior"]
        for p in priors
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(priors, posteriors, color="steelblue", linewidth=2)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="P(T|+) = P(T)")
    ax.set_xlabel("Prior  P(T)")
    ax.set_ylabel("Posterior  P(T | +)")
    ax.set_title(
        f"Posterior vs. prior  (detection={detection_probability:.2f}, "
        f"false alarm={false_alarm_probability:.2f})"
    )
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nSaved plot to {path}")


if __name__ == "__main__":
    result = bayes_update(prior=0.01, detection_probability=0.95, false_alarm_probability=0.05)
    print_report(result)

    plot_prior_sweep(
        detection_probability=0.95,
        false_alarm_probability=0.05,
        path="bayes_prior_sweep.png",
    )
