"""Run every Module 4 experiment in order, then build the final system card (~10 minutes)."""

import exp_a_features
import exp_b_perceptron_logistic
import exp_c_svm
import exp_d_knn_trees
import exp_e_thresholds_calibration
import exp_f_distribution_shift
import star_classifier

if __name__ == "__main__":
    for mod in (exp_a_features, exp_b_perceptron_logistic, exp_c_svm, exp_d_knn_trees,
                exp_e_thresholds_calibration, exp_f_distribution_shift, star_classifier):
        print("=" * 110)
        mod.main()
        print()
