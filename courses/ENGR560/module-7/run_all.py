"""Run every Module 7 experiment in order. Figures land in ./outputs (~10 minutes; Experiment C dominates)."""

import exp_a_kmeans
import exp_b_gmm_em
import exp_c_model_selection
import exp_d_hierarchical_dbscan
import exp_e_feature_discovery

if __name__ == "__main__":
    for mod in (exp_a_kmeans, exp_b_gmm_em, exp_c_model_selection, exp_d_hierarchical_dbscan,
                exp_e_feature_discovery):
        print("=" * 110)
        mod.main()
        print()
