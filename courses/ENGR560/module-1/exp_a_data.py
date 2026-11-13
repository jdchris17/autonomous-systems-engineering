"""
Experiment A: generate the data and look at it.

Plots the 30 training observations as a scatterplot. The true line is NOT drawn here
on purpose: the question is what you can infer from the noisy points alone.
A second figure shows four other, independent 30-point datasets from the same sensor,
which addresses "would every collection of 30 observations look the same?".
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from learning_core import NOISE_STD, generate_data, starter_datasets

OUT = Path(__file__).parent / "outputs"


def main():
    OUT.mkdir(exist_ok=True)
    x_train, y_train, x_test, y_test = starter_datasets()

    print("Experiment A: data summary")
    print(f"  training set: N = {x_train.size},  x in [{x_train.min():.2f}, {x_train.max():.2f}],"
          f"  y in [{y_train.min():.2f}, {y_train.max():.2f}]")
    print(f"  test set:     N = {x_test.size}")
    # A crude "by eye" slope: rise over run between the averages of the left and right thirds.
    order = np.argsort(x_train)
    lo, hi = order[:10], order[-10:]
    eye_slope = (y_train[hi].mean() - y_train[lo].mean()) / (x_train[hi].mean() - x_train[lo].mean())
    print(f"  rough slope from left/right thirds of the scatter: {eye_slope:.2f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(x_train, y_train, s=28, color="tab:blue", edgecolor="k", linewidth=0.4,
               label="training observations (N = 30)")
    ax.set_xlabel("input x")
    ax.set_ylabel("sensor output y")
    ax.set_title(f"Experiment A: noisy sensor observations (sigma = {NOISE_STD})")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "exp_a_training_scatter.png", dpi=150)
    plt.close(fig)

    # Four more independent draws of 30 observations from the same sensor.
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharex=True, sharey=True)
    for i, ax in enumerate(axes):
        x, y = generate_data(30, np.random.default_rng(1000 + i))
        ax.scatter(x, y, s=14, color="tab:gray", edgecolor="k", linewidth=0.3)
        ax.set_title(f"another 30 observations (seed {1000 + i})", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.suptitle("Same sensor, different samples: the scatter is never quite the same", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "exp_a_other_samples.png", dpi=150)
    plt.close(fig)

    print(f"  saved {OUT / 'exp_a_training_scatter.png'}")
    print(f"  saved {OUT / 'exp_a_other_samples.png'}")


if __name__ == "__main__":
    main()
