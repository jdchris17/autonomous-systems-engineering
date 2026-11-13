"""Run every Module 5 experiment in order. Figures land in ./outputs."""

import exp_a_leakage
import exp_b_cross_validation
import exp_c_learning_curves
import exp_d_thresholds_calibration
import exp_e_shift_failure

if __name__ == "__main__":
    for mod in (exp_a_leakage, exp_b_cross_validation, exp_c_learning_curves,
                exp_d_thresholds_calibration, exp_e_shift_failure):
        print("=" * 110)
        mod.main()
        print()
