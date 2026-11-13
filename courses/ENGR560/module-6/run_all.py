"""Run every Module 6 experiment in order. Figures land in ./outputs (~6 minutes)."""

import exp_a_patches
import exp_b_pca_scratch
import exp_c_reconstruction
import exp_d_pca_classification
import exp_e_lda_tsne

if __name__ == "__main__":
    for mod in (exp_a_patches, exp_b_pca_scratch, exp_c_reconstruction, exp_d_pca_classification,
                exp_e_lda_tsne):
        print("=" * 110)
        mod.main()
        print()
