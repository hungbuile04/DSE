"""
Unit tests for Stage 2 (`profile_self_correction`):
  1. Verifies that for training pairs (d, s), the vectorized Gram-matrix rank-1 downdate
     `compute_batch_self_corrected_profiles` matches recomputing `cosine_similarity` on
     `Y_train` after setting entry (d, s) = 0, with max absolute error < 1e-6 across
     all 4 profile similarity matrices: D9, D10, S3, S4.
  2. Verifies edge cases where a drug or side effect has only a single positive interaction
     in `Y_train` (so setting (d, s) = 0 makes the entire row/column zero -> den = 0 -> 0).
  3. Verifies that for all `val_select`, `val_calib`, and `test` pairs in both Split A and
     Split B, `Y_train[d, s] == 0` identically and `MSSFPlusModel._apply_profile_self_correction`
     leaves the input feature rows 100% unchanged (max diff == 0.0).
"""

import unittest
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

from MSSF_plus.protocol import load_fold_data, N_DRUGS, N_SIDES
from MSSF_plus.views import (
    compute_profile_views_from_train,
    compute_batch_self_corrected_profiles,
    build_fold_views,
)
from MSSF_plus.model import MSSFPlusModel
from MSSF_plus.configs.experiments import get_experiment_config


def _prepare_gram_tensors(y_np: np.ndarray, device: torch.device = torch.device('cpu')):
    y_f32 = np.asarray(y_np, dtype=np.float32)
    b_f32 = (y_f32 > 0.0).astype(np.float32)
    g_d9 = y_f32 @ y_f32.T
    g_d10 = b_f32 @ b_f32.T
    g_s3 = y_f32.T @ y_f32
    g_s4 = b_f32.T @ b_f32

    return (
        torch.from_numpy(y_f32).to(device),
        torch.from_numpy(b_f32).to(device),
        torch.from_numpy(g_d9).to(device),
        torch.from_numpy(np.diag(g_d9).copy()).to(device),
        torch.from_numpy(g_d10).to(device),
        torch.from_numpy(np.diag(g_d10).copy()).to(device),
        torch.from_numpy(g_s3).to(device),
        torch.from_numpy(np.diag(g_s3).copy()).to(device),
        torch.from_numpy(g_s4).to(device),
        torch.from_numpy(np.diag(g_s4).copy()).to(device),
    )


class TestProfileSelfCorrection(unittest.TestCase):

    def test_vectorized_gram_matches_recomputed_cosine_on_real_fold(self):
        """
        On real Fold 0 (both Split A and Split B), sample 40 random training pairs (d, s)
        and verify that D9[d, :], D10[d, :], S3[s, :], S4[s, :] match sklearn cosine_similarity
        recomputed on Y_train with Y_train[d, s] = 0, with max error < 1e-6.
        """
        rng = np.random.RandomState(42)
        for split_type in ['A', 'B']:
            fold_split = load_fold_data(0, split_type)
            prof = compute_profile_views_from_train(fold_split.train)
            y_train = prof['drug_side_train'].astype(np.float32)

            tensors = _prepare_gram_tensors(y_train)
            sample_indices = rng.choice(len(fold_split.train), size=40, replace=False)
            sampled_pairs = fold_split.train[sample_indices]

            d_idx = torch.from_numpy(sampled_pairs[:, 0].astype(np.int64))
            s_idx = torch.from_numpy(sampled_pairs[:, 1].astype(np.int64))

            d9_corr, d10_corr, s3_corr, s4_corr = compute_batch_self_corrected_profiles(
                d_idx, s_idx, *tensors
            )

            d9_np = d9_corr.numpy()
            d10_np = d10_corr.numpy()
            s3_np = s3_corr.numpy()
            s4_np = s4_corr.numpy()

            for b, (d, s, _) in enumerate(sampled_pairs):
                d = int(d)
                s = int(s)
                y_mod = y_train.copy()
                y_mod[d, s] = 0.0
                b_mod = (y_mod > 0.0).astype(np.float32)

                ref_d9 = cosine_similarity(y_mod[d:d + 1].astype(np.float64), y_mod.astype(np.float64))[0]
                ref_d10 = cosine_similarity(b_mod[d:d + 1].astype(np.float64), b_mod.astype(np.float64))[0]
                ref_s3 = cosine_similarity(y_mod[:, s:s + 1].T.astype(np.float64), y_mod.T.astype(np.float64))[0]
                ref_s4 = cosine_similarity(b_mod[:, s:s + 1].T.astype(np.float64), b_mod.T.astype(np.float64))[0]

                err_d9 = float(np.max(np.abs(d9_np[b] - ref_d9)))
                err_d10 = float(np.max(np.abs(d10_np[b] - ref_d10)))
                err_s3 = float(np.max(np.abs(s3_np[b] - ref_s3)))
                err_s4 = float(np.max(np.abs(s4_np[b] - ref_s4)))

                self.assertLess(err_d9, 1e-6, f"Split {split_type} pair ({d},{s}) D9 max err={err_d9}")
                self.assertLess(err_d10, 1e-6, f"Split {split_type} pair ({d},{s}) D10 max err={err_d10}")
                self.assertLess(err_s3, 1e-6, f"Split {split_type} pair ({d},{s}) S3 max err={err_s3}")
                self.assertLess(err_s4, 1e-6, f"Split {split_type} pair ({d},{s}) S4 max err={err_s4}")

    def test_single_entry_row_and_col_zero_denominator_edge_case(self):
        """
        Verify that when drug d or side effect s has ONLY ONE non-zero entry (d, s) in Y_train,
        removing (d, s) makes its norm 0 -> den = 0 -> entire row (including diagonal) is 0.0,
        matching sklearn cosine_similarity on a zero row.
        """
        y_syn = np.zeros((N_DRUGS, N_SIDES), dtype=np.float32)
        # Drug 0 has only 1 entry at SE 0; SE 0 has only 1 entry at Drug 0
        y_syn[0, 0] = 4.0
        # Drug 1 has 2 entries: SE 1 and SE 2; SE 1 has only 1 entry at Drug 1
        y_syn[1, 1] = 3.0
        y_syn[1, 2] = 5.0
        y_syn[2, 2] = 2.0

        tensors = _prepare_gram_tensors(y_syn)
        d_idx = torch.tensor([0, 1], dtype=torch.long)
        s_idx = torch.tensor([0, 1], dtype=torch.long)

        d9_corr, d10_corr, s3_corr, s4_corr = compute_batch_self_corrected_profiles(
            d_idx, s_idx, *tensors
        )

        # Pair (0, 0): both drug 0 and SE 0 become completely empty -> all zeros
        self.assertEqual(float(torch.max(torch.abs(d9_corr[0]))), 0.0)
        self.assertEqual(float(torch.max(torch.abs(d10_corr[0]))), 0.0)
        self.assertEqual(float(torch.max(torch.abs(s3_corr[0]))), 0.0)
        self.assertEqual(float(torch.max(torch.abs(s4_corr[0]))), 0.0)

        # Pair (1, 1): drug 1 still has entry at SE 2 (so D9[1, 1] == 1.0, D9[1, 2] == 1.0),
        # while SE 1 becomes empty (S3[1] == 0.0)
        y_mod = y_syn.copy()
        y_mod[1, 1] = 0.0
        b_mod = (y_mod > 0.0).astype(np.float32)
        ref_d9 = cosine_similarity(y_mod[1:2], y_mod)[0]
        ref_d10 = cosine_similarity(b_mod[1:2], b_mod)[0]
        ref_s3 = cosine_similarity(y_mod[:, 1:2].T, y_mod.T)[0]
        ref_s4 = cosine_similarity(b_mod[:, 1:2].T, b_mod.T)[0]

        self.assertLess(float(np.max(np.abs(d9_corr[1].numpy() - ref_d9))), 1e-6)
        self.assertLess(float(np.max(np.abs(d10_corr[1].numpy() - ref_d10))), 1e-6)
        self.assertLess(float(np.max(np.abs(s3_corr[1].numpy() - ref_s3))), 1e-6)
        self.assertLess(float(np.max(np.abs(s4_corr[1].numpy() - ref_s4))), 1e-6)

    def test_val_and_test_pairs_are_strictly_unchanged(self):
        """
        For all val_select, val_calib, and test pairs in both Split A and Split B:
          1. Y_train[d, s] == 0 identically.
          2. Calling MSSFPlusModel._apply_profile_self_correction leaves drug and SE features
             bit-for-bit identical (max diff == 0.0).
        """
        cfg = get_experiment_config('MSSF_v2', profile_self_correction=True)
        for split_type in ['A', 'B']:
            fold_split = load_fold_data(0, split_type)
            fold_views = build_fold_views(fold_split, cfg)
            y_train = fold_views.drug_side_train

            model = MSSFPlusModel(
                cfg=cfg,
                n_drug_views=len(fold_views.drug_view_names),
                n_se_views=len(fold_views.se_view_names),
                profile_drug_view_indices=fold_views.profile_drug_view_indices,
                profile_se_view_indices=fold_views.profile_se_view_indices,
                drug_side_train=y_train,
            )

            for split_name, arr in [
                ('valsel', fold_split.valsel),
                ('valcal', fold_split.valcal),
                ('test', fold_split.test),
            ]:
                d_arr = arr[:, 0].astype(np.int64)
                s_arr = arr[:, 1].astype(np.int64)
                # Assert Y_train[d, s] == 0 for every evaluation pair
                self.assertTrue(
                    np.all(y_train[d_arr, s_arr] == 0.0),
                    f"Split {split_type} {split_name} has non-zero entries in Y_train!",
                )

                # Test a batch of 64 evaluation pairs through _apply_profile_self_correction
                # in both eval() and train() mode
                d_b = torch.from_numpy(d_arr[:64])
                s_b = torch.from_numpy(s_arr[:64])
                drugs_in = torch.from_numpy(fold_views.drug_features_matrix[d_b]).clone()
                sides_in = torch.from_numpy(fold_views.se_features_matrix[s_b]).clone()
                d_avail_in = torch.from_numpy(fold_views.drug_avail[d_b]).clone()
                s_avail_in = torch.from_numpy(fold_views.se_avail[s_b]).clone()

                for training_mode in [False, True]:
                    model.train(training_mode)
                    d_out, s_out, da_out, sa_out = model._apply_profile_self_correction(
                        drugs_in, sides_in, d_avail_in, s_avail_in, d_b, s_b
                    )
                    self.assertTrue(torch.equal(d_out, drugs_in))
                    self.assertTrue(torch.equal(s_out, sides_in))
                    self.assertTrue(torch.equal(da_out, d_avail_in))
                    self.assertTrue(torch.equal(sa_out, s_avail_in))


if __name__ == '__main__':
    unittest.main()
