"""Run every Module 12 experiment in order (~8 minutes on CPU). Experiment A writes the checkpoint B and C use."""

import exp_a_train_cnn
import exp_b_filters
import exp_c_pipelines

if __name__ == "__main__":
    for mod in (exp_a_train_cnn, exp_b_filters, exp_c_pipelines):
        print("=" * 110)
        mod.main()
        print()
