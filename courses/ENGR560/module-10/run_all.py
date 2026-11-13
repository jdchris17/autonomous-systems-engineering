"""Run every Module 10 experiment in order (~3 minutes). Experiment C uses the checkpoint written by B."""

import exp_a_numpy_vs_torch
import exp_b_optimizers
import exp_c_inference

if __name__ == "__main__":
    for mod in (exp_a_numpy_vs_torch, exp_b_optimizers, exp_c_inference):
        print("=" * 110)
        mod.main()
        print()
