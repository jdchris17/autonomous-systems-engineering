"""Run every Module 2 experiment in order. Figures land in ./outputs."""

import exp_a_engine_checks
import exp_b_gd_vs_lstsq
import exp_c_gradient_check
import exp_d_learning_rates
import exp_e_batch_sgd_minibatch

if __name__ == "__main__":
    for mod in (exp_a_engine_checks, exp_b_gd_vs_lstsq, exp_c_gradient_check, exp_d_learning_rates,
                exp_e_batch_sgd_minibatch):
        print("=" * 100)
        mod.main()
        print()
