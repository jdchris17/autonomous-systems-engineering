"""Run every Module 13 experiment in order (~6 minutes; Experiment C needs Astrometry.net in WSL)."""

import exp_a_wahba
import exp_b_identify
import exp_c_integrate

if __name__ == "__main__":
    for mod in (exp_a_wahba, exp_b_identify, exp_c_integrate):
        print("=" * 110)
        mod.main()
        print()
