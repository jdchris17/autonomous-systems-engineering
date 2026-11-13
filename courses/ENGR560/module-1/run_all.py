"""Run every Module 1 experiment in order. Figures and the CSV land in ./outputs."""

import exp_a_data
import exp_b_fit
import exp_c_generalization
import exp_d_sample_size_noise
import exp_e_model_mismatch

if __name__ == "__main__":
    for mod in (exp_a_data, exp_b_fit, exp_c_generalization, exp_d_sample_size_noise,
                exp_e_model_mismatch):
        print("=" * 100)
        mod.main()
        print()
