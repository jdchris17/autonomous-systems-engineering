"""A simple discrete sensor noise model: X = x_true + N, N a small discrete
noise term. Estimate how often a reading lands within 1 unit of truth."""

import matplotlib.pyplot as plt
import numpy as np

X_TRUE = 100
NOISE_VALUES = np.array([-2, -1, 0, 1, 2])
NOISE_PMF = np.array([0.05, 0.15, 0.60, 0.15, 0.05])
N_MEASUREMENTS = 100_000


def generate_measurements(x_true, noise_values, noise_pmf, n, rng):
    noise = rng.choice(noise_values, size=n, p=noise_pmf)
    return x_true + noise


def empirical_cdf(values):
    """Step-function CDF from raw samples: sorted values vs. their rank / n."""
    sorted_values = np.sort(values)
    n = len(sorted_values)
    cumulative_prob = np.arange(1, n + 1) / n
    return sorted_values, cumulative_prob


def plot_sensor_model(measurements, x_true, noise_values, noise_pmf, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax1 = axes[0]
    ax1.bar(noise_values, noise_pmf, color="steelblue", width=0.6)
    ax1.set_title("Noise PMF: P(N = n)")
    ax1.set_xlabel("n")
    ax1.set_ylabel("P(N = n)")
    ax1.set_xticks(noise_values)

    ax2 = axes[1]
    measurement_values = x_true + noise_values
    ax2.hist(
        measurements,
        bins=np.arange(measurement_values.min() - 0.5, measurement_values.max() + 1.5),
        density=True,
        color="steelblue",
        rwidth=0.6,
    )
    ax2.axvline(x_true, color="firebrick", linestyle="--", linewidth=1, label=f"x_true = {x_true}")
    ax2.set_title(f"Measurement histogram  ({N_MEASUREMENTS:,} readings)")
    ax2.set_xlabel("X = x_true + N")
    ax2.set_ylabel("density")
    ax2.legend()

    ax3 = axes[2]
    sorted_values, cumulative_prob = empirical_cdf(measurements)
    ax3.step(sorted_values, cumulative_prob, where="post", color="darkorange", linewidth=2)
    ax3.axvline(x_true, color="firebrick", linestyle="--", linewidth=1, label=f"x_true = {x_true}")
    ax3.set_title("Measurement empirical CDF")
    ax3.set_xlabel("X")
    ax3.set_ylabel("P(X <= x)")
    ax3.legend()
    ax3.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved plot to {path}")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    measurements = generate_measurements(X_TRUE, NOISE_VALUES, NOISE_PMF, N_MEASUREMENTS, rng)

    within_1 = np.abs(measurements - X_TRUE) <= 1
    empirical_p = np.mean(within_1)

    # X - x_true = N exactly, so the theoretical answer reads straight off
    # the noise PMF: P(|N| <= 1) = P(N in {-1, 0, 1}).
    theoretical_p = NOISE_PMF[np.abs(NOISE_VALUES) <= 1].sum()

    print(f"x_true = {X_TRUE}")
    print(f"Noise support: {NOISE_VALUES.tolist()}")
    print(f"Noise PMF:     {NOISE_PMF.tolist()}")
    print(f"Generated {N_MEASUREMENTS:,} measurements X = x_true + N\n")
    print(f"Empirical  P(|X - x_true| <= 1) = {empirical_p:.4f}")
    print(f"Theoretical P(|X - x_true| <= 1) = {theoretical_p:.4f}  (= P(N in {{-1, 0, 1}}))")

    plot_sensor_model(measurements, X_TRUE, NOISE_VALUES, NOISE_PMF, "sensor_noise_model.png")
