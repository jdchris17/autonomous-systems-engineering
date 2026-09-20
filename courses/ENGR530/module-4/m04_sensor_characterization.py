"""Sensor characterization: estimate mean, bias, variance, std, and RMSE from
simulated measurements and compare against theory."""

import numpy as np

X_TRUE = 100.0
NOISE_MEAN = 2.0
NOISE_STD = 3.0
N_MEASUREMENTS = 100_000


def sample_mean(x):
    return sum(x) / len(x)


def sample_variance(x):
    """Unbiased sample variance: sum((x - mean)^2) / (n - 1)."""
    m = sample_mean(x)
    return sum((xi - m) ** 2 for xi in x) / (len(x) - 1)


def rmse(x, true_value):
    """Root mean squared error against the truth: sqrt(mean((x - truth)^2))."""
    return (sum((xi - true_value) ** 2 for xi in x) / len(x)) ** 0.5


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    noise = rng.normal(loc=NOISE_MEAN, scale=NOISE_STD, size=N_MEASUREMENTS)
    measurements = (X_TRUE + noise).tolist()

    est_mean = sample_mean(measurements)
    est_bias = est_mean - X_TRUE
    est_var = sample_variance(measurements)
    est_std = est_var**0.5
    est_rmse = rmse(measurements, X_TRUE)

    theory_mean = X_TRUE + NOISE_MEAN
    theory_bias = NOISE_MEAN
    theory_std = NOISE_STD
    theory_var = NOISE_STD**2
    theory_rmse = (NOISE_STD**2 + NOISE_MEAN**2) ** 0.5

    print(f"x_true = {X_TRUE}, noise ~ Normal(mean={NOISE_MEAN}, std={NOISE_STD}), N = {N_MEASUREMENTS:,}\n")
    print(f"{'quantity':<20}{'estimated':>12}{'theory':>12}")
    print(f"{'sample mean':<20}{est_mean:>12.4f}{theory_mean:>12.4f}")
    print(f"{'bias':<20}{est_bias:>12.4f}{theory_bias:>12.4f}")
    print(f"{'sample variance':<20}{est_var:>12.4f}{theory_var:>12.4f}")
    print(f"{'sample std dev':<20}{est_std:>12.4f}{theory_std:>12.4f}")
    print(f"{'RMSE':<20}{est_rmse:>12.4f}{theory_rmse:>12.4f}")

    print(f"\nCheck: RMSE^2 = bias^2 + variance -> {est_rmse**2:.4f} vs {est_bias**2 + est_var:.4f}")

    print(
        f"""
What this means
---------------
This sensor has two separate kinds of error, and they are different problems:
  - Bias (~{est_bias:.1f}): a systematic offset. Every reading is pushed the same
    direction, so averaging more readings does NOT remove it. It is only
    fixable by calibration (subtract the known offset).
  - Standard deviation (~{est_std:.1f}): random scatter from reading to reading.
    This one DOES shrink when you average: the mean of n readings has
    standard error std/sqrt(n) = {est_std / N_MEASUREMENTS**0.5:.4f} here.
RMSE (~{est_rmse:.2f}) combines both: RMSE^2 = bias^2 + variance ({theory_bias**2:.0f} + {theory_var:.0f} = {theory_bias**2 + theory_var:.0f}).
It is the typical size of a single reading's error against the truth, which is
why it is the honest single-number quality score. Note that std alone (3.0)
would make this sensor look better than it is; it ignores that the readings
are centered on 102, not 100."""
    )
