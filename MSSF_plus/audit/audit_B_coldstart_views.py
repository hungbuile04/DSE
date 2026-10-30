"""Audit B — Which views are zero / intact for test drugs in Split B (drug cold-start).

Replicates MSSF read_raw_data() masking for each Split-B fold and measures, per view,
how many test-drug rows are empty (all-zero) or carry no off-diagonal information.
Also checks:
  * the 7 '750 vs 757 unmatched' drugs (prepare_all_data.py L127-129 fills them with zeros + diag 1),
  * MGPred Split B: Ten_Fold_test.py L214-216 fills STITCH/Mol2vec ONLY for drugs in u_adj (train),
  * SDPred Split B: notebook cell 12 samples train/test negatives from all zero pairs of all drugs.
"""
import numpy as np
from common import (drug_side_full, static_views, load_splits, load_test_drugs, load_pkl,
                    mssf_masking_as_in_notebook, mssf_profile_views_from_masked,
                    DRUG_VIEW_NAMES, SE_VIEW_NAMES, SHARED, save_json, N_DRUGS)


def row_stats(M, rows):
    sub = M[rows]
    offdiag = sub.copy()
    offdiag[np.arange(len(rows)), rows] = 0.0  # drop self-similarity
    return {
        'frac_rows_all_zero': float((np.abs(sub).sum(1) == 0).mean()),
        'frac_rows_no_offdiag_info': float((np.abs(offdiag).sum(1) == 0).mean()),
        'mean_row_abs_sum': float(np.abs(sub).sum(1).mean()),
        'mean_diag': float(M[rows, rows].mean()),
    }


def find_unmatched_drugs(dv):
    """Drugs with no off-diagonal STITCH info in all 5 STITCH views (filled by prepare_all_data)."""
    res = []
    for d in range(N_DRUGS):
        if all(np.abs(np.delete(dv[n][d], d)).sum() == 0 for n in DRUG_VIEW_NAMES[:5]):
            res.append(d)
    return res


def run():
    full = drug_side_full()
    dv, sv = static_views()
    unmatched = find_unmatched_drugs(dv)
    print(f'Drugs with empty STITCH rows (unmatched 750<->757): {unmatched}')
    unmatched_views = {}
    for d in unmatched:
        unmatched_views[d] = [n for n in dv if np.abs(np.delete(dv[n][d], d)).sum() == 0]
        print(f'  drug {d}: empty views = {unmatched_views[d]}  (#SE known = {(full[d] > 0).sum()})')

    report = {'unmatched_drugs': unmatched, 'unmatched_drug_empty_views': unmatched_views, 'per_fold': {}}
    for k in range(10):
        tr, te = load_splits(k, 'B')
        test_drugs = np.sort(load_test_drugs(k, 'B').astype(int))
        train_drugs = np.setdiff1d(np.arange(N_DRUGS), test_drugs)
        masked, _ = mssf_masking_as_in_notebook(full, te)
        pv = mssf_profile_views_from_masked(masked)
        allv = {**dv, **pv}
        fr = {'n_test_drugs': len(test_drugs),
              'test_rows_in_masked_matrix_all_zero': bool((masked[test_drugs].sum(1) == 0).all()),
              'unmatched_drugs_in_test': [int(d) for d in unmatched if d in set(test_drugs)],
              'drug_views_test': {}, 'drug_views_train': {}}
        for n in DRUG_VIEW_NAMES:
            fr['drug_views_test'][n] = row_stats(allv[n], test_drugs)
            fr['drug_views_train'][n] = row_stats(allv[n], train_drugs)
        # train-drug rows: entries pointing to test drugs inside D9/D10 are zero by construction
        fr['D9_train_rows_cols_testdrugs_all_zero'] = bool(
            np.abs(pv['D9_profile_freq_cos'][np.ix_(train_drugs, test_drugs)]).sum() == 0)
        # SE views: S3/S4 computed from train-drug rows only
        se_zero_train = np.where(masked.sum(0) == 0)[0]
        fr['ses_without_any_train_pair'] = se_zero_train.tolist()
        sev = {**sv, **{n: pv[n] for n in SE_VIEW_NAMES[2:]}}
        fr['se_views_all_ses'] = {n: row_stats(sev[n], np.arange(sev[n].shape[0])) for n in SE_VIEW_NAMES}
        test_ses = np.unique(te[:, 1].astype(int))
        fr['n_test_pairs_whose_SE_has_no_train_pair'] = int(np.isin(te[:, 1].astype(int), se_zero_train).sum())
        fr['n_test_ses'] = int(len(test_ses))

        # MGPred Split B emulation: data_train = (se, drug, f); u = drug; u_adj keys = train drugs
        u_keys = set(int(r[0]) for r in tr)
        fr['MGPred_test_drugs_missing_from_u_adj'] = int(sum(d not in u_keys for d in test_drugs))

        # SDPred Split B emulation (notebook cell 12)
        sd = np.array(load_pkl(f'{SHARED}/sdpred_750/drug_side.pkl'))
        zero_pairs = np.argwhere(sd == 0)
        np.random.seed(42 + k)
        perm = np.random.permutation(len(zero_pairs))
        tr_neg = zero_pairs[perm[:len(tr)]]
        te_neg = zero_pairs[perm[len(tr):len(tr) + len(te)]]
        tset = set(test_drugs.tolist())
        fr['SDPred_train_negatives_with_test_drug'] = int(sum(int(d) in tset for d in tr_neg[:, 0]))
        fr['SDPred_test_negatives_with_train_drug'] = int(sum(int(d) not in tset for d in te_neg[:, 0]))
        fr['SDPred_n_train_neg'] = int(len(tr_neg))
        fr['SDPred_n_test_neg'] = int(len(te_neg))
        report['per_fold'][k] = fr

        if k == 0:
            print(f'\n=== Split B fold 0: {len(test_drugs)} test drugs ===')
            print(f'test-drug rows of masked drug_side all zero: {fr["test_rows_in_masked_matrix_all_zero"]}')
            print(f'{"view":22s} {"test:all0":>9s} {"test:noOff":>10s} {"test:diag":>9s} | '
                  f'{"train:all0":>10s} {"train:noOff":>11s}')
            for n in DRUG_VIEW_NAMES:
                a, b = fr['drug_views_test'][n], fr['drug_views_train'][n]
                print(f'{n:22s} {a["frac_rows_all_zero"]:9.3f} {a["frac_rows_no_offdiag_info"]:10.3f} '
                      f'{a["mean_diag"]:9.3f} | {b["frac_rows_all_zero"]:10.3f} {b["frac_rows_no_offdiag_info"]:11.3f}')
            for n in SE_VIEW_NAMES:
                a = fr['se_views_all_ses'][n]
                print(f'{n:22s} SE rows all-zero frac={a["frac_rows_all_zero"]:.4f}')
            print(f'D9 train-rows x test-drug cols all zero: {fr["D9_train_rows_cols_testdrugs_all_zero"]}')

    print('\nPer fold summary:')
    print('fold | unmatched in test | SEs w/o train pair | test pairs on such SEs | '
          'MGPred test drugs w/o STITCH/Mol2vec | SDPred train-neg w/ test drug | SDPred test-neg w/ train drug')
    for k, fr in report['per_fold'].items():
        print(f'{k:4d} | {str(fr["unmatched_drugs_in_test"]):17s} | {len(fr["ses_without_any_train_pair"]):18d} | '
              f'{fr["n_test_pairs_whose_SE_has_no_train_pair"]:22d} | {fr["MGPred_test_drugs_missing_from_u_adj"]:2d}/'
              f'{fr["n_test_drugs"]} | {fr["SDPred_train_negatives_with_test_drug"]:5d}/{fr["SDPred_n_train_neg"]} | '
              f'{fr["SDPred_test_negatives_with_train_drug"]:4d}/{fr["SDPred_n_test_neg"]}')
    save_json(report, 'audit_B_coldstart_views.json')


if __name__ == '__main__':
    run()
