"""Run every Module 8 experiment in order. Figures land in ./outputs (~5 minutes)."""

import exp_a_xor
import exp_b_gradcheck
import exp_c_star

if __name__ == "__main__":
    for mod in (exp_a_xor, exp_b_gradcheck, exp_c_star):
        print("=" * 110)
        mod.main()
        print()
