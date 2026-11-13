"""Run every Module 9 experiment in order. Figures land in ./outputs (~10 minutes)."""

import exp_a_bagging
import exp_b_random_forest
import exp_c_boosting
import exp_d_voting_stacking
import exp_e_final_system

if __name__ == "__main__":
    for mod in (exp_a_bagging, exp_b_random_forest, exp_c_boosting, exp_d_voting_stacking, exp_e_final_system):
        print("=" * 110)
        mod.main()
        print()
