"""Run every Module 11 experiment in order (~3 minutes)."""

import exp_a_detection
import exp_b_centroids
import exp_c_photutils

if __name__ == "__main__":
    for mod in (exp_a_detection, exp_b_centroids, exp_c_photutils):
        print("=" * 110)
        mod.main()
        print()
