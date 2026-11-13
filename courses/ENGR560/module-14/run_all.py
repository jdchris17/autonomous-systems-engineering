"""Run every Module 14 experiment in order (about 1 minute on CPU)."""

import exp_a_qlearning
import exp_b_design
import exp_c_robustness

if __name__ == "__main__":
    for mod in (exp_a_qlearning, exp_b_design, exp_c_robustness):
        print("=" * 110)
        mod.main()
        print()
