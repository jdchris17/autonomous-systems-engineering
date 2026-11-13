"""Run every Module 3 experiment in order. Figures land in ./outputs (takes ~5 minutes)."""

import exp_a_regression_methods
import exp_b_weighted_least_squares
import exp_c_regularization
import exp_d_robust_regression
import exp_e_integrated_evaluation

if __name__ == "__main__":
    for mod in (exp_a_regression_methods, exp_b_weighted_least_squares, exp_c_regularization,
                exp_d_robust_regression, exp_e_integrated_evaluation):
        print("=" * 100)
        mod.main()
        print()
