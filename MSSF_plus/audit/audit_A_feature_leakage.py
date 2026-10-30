"""Audit A — Feature leakage of profile-derived views in Split A (warm-start).

Question: are drug [9-10] and SE [3-4] computed from the full drug_side.pkl or only from
the train pairs of the fold?

Evidence:
  * Static: MSSF/mssf.py L96-98 zeroes test pairs before L101/L111/L128/L129 cosine sims.
    The notebook (cell 9) patches L98 to cast float indices to int (otherwise numpy raises).
  * Numeric (this script): for each fold, compare
      (full)    views from the full matrix,
      (mssf)    views from the matrix masked exactly as patched MSSF does,
      (train)   views from a matrix rebuilt from scratch using ONLY the train pairs.
    If mssf == train exactly -> no leakage in the original pipeline.
"""
import numpy as np
from common import (drug_side_full, load_splits, mssf_masking_as_in_notebook,
                    mssf_profile_views_from_masked, matrix_from_pairs, diff_stats, save_json)


def run(folds=range(10), detailed_fold=0):
    full = drug_side_full()
    v_full = mssf_profile_views_from_masked(full)
    report = {'per_fold': {}}
    for k in folds:
        tr, te = load_splits(k, 'A')
        masked, dtype = mssf_masking_as_in_notebook(full, te)
        train_only = matrix_from_pairs(tr)
        v_mssf = mssf_profile_views_from_masked(masked)
        v_train = mssf_profile_views_from_masked(train_only)
        fold_rep = {
            'data_test_dtype_inside_mssf': str(dtype),
            'masked_matrix_equals_train_only_matrix': bool(np.array_equal(masked, train_only)),
            'n_train_pairs': int(len(tr)), 'n_test_pairs': int(len(te)),
            'n_nonzero_masked': int((masked > 0).sum()),
        }
        for name in v_full:
            fold_rep[name] = {
                'mssf_vs_trainonly': diff_stats(v_mssf[name], v_train[name]),
                'full_vs_trainonly': diff_stats(v_full[name], v_train[name]),
            }
        # Warm-start drugs/SEs that have NO train pair at all in this fold (profile row empty)
        fold_rep['drugs_with_zero_train_pairs'] = np.where(train_only.sum(1) == 0)[0].tolist()
        fold_rep['ses_with_zero_train_pairs'] = np.where(train_only.sum(0) == 0)[0].tolist()
        report['per_fold'][k] = fold_rep

        if k == detailed_fold:
            print(f'=== Split A fold {k} ===')
            print(f'data_test dtype inside MSSF: {dtype}  (float -> int cast patch is required)')
            print(f'masked == train-only matrix: {fold_rep["masked_matrix_equals_train_only_matrix"]}')
            for name in v_full:
                a = fold_rep[name]['mssf_vs_trainonly']
                b = fold_rep[name]['full_vs_trainonly']
                print(f'{name:22s} | MSSF vs train-only max|d|={a["max_abs_diff"]:.2e} '
                      f'| FULL vs train-only max|d|={b["max_abs_diff"]:.4f} mean|d|={b["mean_abs_diff"]:.5f} '
                      f'changed={b["frac_entries_changed(>1e-12)"]:.3f}')
            # how different is a test drug's own row under full vs train-only?
            te_drugs = np.unique(te[:, 0].astype(int))
            d_full, d_tr = v_full['D9_profile_freq_cos'], v_train['D9_profile_freq_cos']
            row_corr = [np.corrcoef(d_full[d], d_tr[d])[0, 1] for d in te_drugs[:200]]
            print(f'D9 row corr(full, train-only) over test drugs: mean={np.nanmean(row_corr):.4f}')
            report['detailed_fold_D9_rowcorr_mean'] = float(np.nanmean(row_corr))

    allok = all(r['masked_matrix_equals_train_only_matrix'] and
                all(r[n]['mssf_vs_trainonly']['max_abs_diff'] < 1e-12 for n in v_full)
                for r in report['per_fold'].values())
    report['ALL_FOLDS_MSSF_FEATURES_TRAIN_ONLY'] = bool(allok)
    print(f'\nAll 10 folds: MSSF profile views == train-only views: {allok}')
    print('Folds with drugs/SEs lacking any train pair:',
          {k: (len(r['drugs_with_zero_train_pairs']), len(r['ses_with_zero_train_pairs']))
           for k, r in report['per_fold'].items()})
    save_json(report, 'audit_A_feature_leakage.json')


if __name__ == '__main__':
    run()
