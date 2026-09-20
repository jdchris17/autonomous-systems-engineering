"""Sequential Bayesian updating: fold in one sensor report at a time, using
each posterior as the next prior, and watch belief evolve."""

import matplotlib.pyplot as plt
import numpy as np

PRIOR = 0.02
DETECTION_PROBABILITY = 0.95
FALSE_ALARM_PROBABILITY = 0.05
N_OBSERVATIONS = 40


def observation_likelihoods(observation, detection_probability, false_alarm_probability):
    """P(observation | target) and P(observation | no target) for one +/- report."""
    if observation == "+":
        return detection_probability, false_alarm_probability
    elif observation == "-":
        return 1 - detection_probability, 1 - false_alarm_probability
    else:
        raise ValueError(f"observation must be '+' or '-', got {observation!r}")


def sequential_update(prior, observation, detection_probability, false_alarm_probability):
    """One Bayes step for a single +/- observation, prior -> posterior."""
    p_obs_given_target, p_obs_given_no_target = observation_likelihoods(
        observation, detection_probability, false_alarm_probability
    )
    evidence = p_obs_given_target * prior + p_obs_given_no_target * (1 - prior)
    return p_obs_given_target * prior / evidence


def simulate_stream(n, target_present, detection_probability, false_alarm_probability, rng):
    """Generate a stream of +/- sensor reports for a fixed ground truth."""
    hit_prob = detection_probability if target_present else false_alarm_probability
    return ["+" if rng.random() < hit_prob else "-" for _ in range(n)]


def run_updates(prior, stream, detection_probability, false_alarm_probability):
    """Fold each observation into the belief, returning [prior, post-1, post-2, ...].

    Also returns the log-odds trajectory, accumulated directly as a running
    sum of log-likelihood-ratios rather than recovered from `beliefs` — once
    a probability rounds to exactly 1.0 in float64 (it does here, around
    k=15), log(p / (1 - p)) would divide by zero even though the evidence
    keeps accumulating.
    """
    beliefs = [prior]
    belief = prior
    log_odds = [np.log(prior / (1 - prior))]
    for observation in stream:
        belief = sequential_update(belief, observation, detection_probability, false_alarm_probability)
        beliefs.append(belief)

        p_target, p_no_target = observation_likelihoods(
            observation, detection_probability, false_alarm_probability
        )
        log_odds.append(log_odds[-1] + np.log(p_target / p_no_target))
    return beliefs, log_odds


def plot_belief(beliefs, log_odds, target_present, path):
    k = np.arange(len(beliefs))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

    ax1.plot(k, beliefs, marker="o", markersize=4, color="steelblue", label="P(T | E1, ..., Ek)")
    ax1.axhline(PRIOR, color="gray", linestyle=":", linewidth=1, label=f"initial prior ({PRIOR})")
    ax1.axhline(
        1.0 if target_present else 0.0,
        color="firebrick",
        linestyle="--",
        linewidth=1,
        label="ground truth",
    )
    ax1.set_ylabel("Belief  P(T | E1, ..., Ek)")
    ax1.set_title("Sequential Bayesian updating")
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Once P(T|...) saturates near 0 or 1 it is visually flat above; log-odds
    # stays additive (+log(d/f) per "+", +log((1-d)/(1-f)) per "-") so every
    # observation, including misses that briefly pull belief back down,
    # remains visible for the whole stream.
    ax2.plot(k, log_odds, marker="o", markersize=4, color="darkorange")
    ax2.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax2.set_xlabel("Measurement number k")
    ax2.set_ylabel("log-odds  ln[P(T|...) / (1 - P(T|...))]")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved plot to {path}")


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    target_present = True

    stream = simulate_stream(
        N_OBSERVATIONS, target_present, DETECTION_PROBABILITY, FALSE_ALARM_PROBABILITY, rng
    )
    beliefs, log_odds = run_updates(PRIOR, stream, DETECTION_PROBABILITY, FALSE_ALARM_PROBABILITY)

    print(f"Ground truth: target {'present' if target_present else 'absent'}")
    print(f"Observation stream ({N_OBSERVATIONS} reports):")
    print("  " + " ".join(stream))
    print()
    print(f"{'k':>3}  {'obs':>3}  {'P(T | E1..Ek)':>14}  {'log-odds':>10}")
    print(f"{0:>3}  {'':>3}  {beliefs[0]:>14.4f}  {log_odds[0]:>10.3f}")
    for k, (observation, belief, lo) in enumerate(zip(stream, beliefs[1:], log_odds[1:]), start=1):
        print(f"{k:>3}  {observation:>3}  {belief:>14.4f}  {lo:>10.3f}")

    plot_belief(beliefs, log_odds, target_present, "sequential_bayes_belief.png")
