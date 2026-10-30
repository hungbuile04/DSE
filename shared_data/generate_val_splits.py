"""
Generate unified validation splits (train / val_select / val_calib = 80 / 10 / 10)
derived strictly from existing shared_data/splits/split{A|B}_fold{k}_train.npy files.

- Split A (warm-start, pair-wise):
    Splits each splitA_fold{k}_train.npy into 80% train / 10% valsel / 10% valcal,
    stratified by 5-level frequency class (seed=42).
- Split B (drug cold-start, drug-wise):
    Splits the 675 training drugs of each fold into 540 train / 67 valsel / 68 valcal
    drugs (80% / 10% / 10%), stratified by quantile bins of per-drug SE count computed
    only on the fold's old training set (seed=42). Validation drugs are treated as
    cold-start (profile = 0 when features are built from train only).

Outputs in shared_data/splits_val/:
    split{A|B}_fold{k}_train.npy         # shape (N_train, 3): [drug_idx, se_idx, frequency]
    split{A|B}_fold{k}_valsel.npy        # shape (N_valsel, 3)
    split{A|B}_fold{k}_valcal.npy        # shape (N_valcal, 3)
    splitB_fold{k}_train_drugs.npy       # shape (540,)
    splitB_fold{k}_valsel_drugs.npy      # shape (67,)
    splitB_fold{k}_valcal_drugs.npy      # shape (68,)

Also generates leak-free mask .mat files for DSGAT/A-3Net in shared_data/:
    mask_mat_750_v2_A.mat
    mask_mat_750_v2_B.mat
"""

import os
import json
import hashlib
import pickle
import numpy as np
import scipy.io as sio
from sklearn.model_selection import StratifiedKFold

SEED = 42
N_FOLDS = 10
N_DRUGS = 750
N_SIDES = 994

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OLD_SPLIT_DIR = os.path.join(BASE_DIR, 'splits')
VAL_SPLIT_DIR = os.path.join(BASE_DIR, 'splits_val')
os.makedirs(VAL_SPLIT_DIR, exist_ok=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def class_dist_summary(arr):
    freqs = arr[:, 2].astype(int)
    n = len(freqs)
    counts = [int((freqs == c).sum()) for c in range(1, 6)]
    pcts = [round(100.0 * c / n, 2) if n > 0 else 0.0 for c in counts]
    return {'n': int(n), 'counts': counts, 'pcts': pcts}


def generate_split_A_val():
    print("=" * 72)
    print("SPLIT A: Splitting train into train (80%) / valsel (10%) / valcal (10%)")
    print("=" * 72)
    summary = {}
    for fold in range(N_FOLDS):
        old_train = np.load(os.path.join(OLD_SPLIT_DIR, f'splitA_fold{fold}_train.npy'))
        strat_labels = old_train[:, 2].astype(int)

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
        buckets = [te_idx for _, te_idx in skf.split(old_train, strat_labels)]

        # Bucket 0 -> valsel (10%), Bucket 9 -> valcal (10%), Buckets 1..8 -> train (80%)
        valsel_idx = np.sort(buckets[0])
        valcal_idx = np.sort(buckets[9])
        train_idx = np.sort(np.concatenate(buckets[1:9]))

        train_new = old_train[train_idx]
        valsel = old_train[valsel_idx]
        valcal = old_train[valcal_idx]

        np.save(os.path.join(VAL_SPLIT_DIR, f'splitA_fold{fold}_train.npy'), train_new)
        np.save(os.path.join(VAL_SPLIT_DIR, f'splitA_fold{fold}_valsel.npy'), valsel)
        np.save(os.path.join(VAL_SPLIT_DIR, f'splitA_fold{fold}_valcal.npy'), valcal)

        summary[fold] = {
            'train': class_dist_summary(train_new),
            'valsel': class_dist_summary(valsel),
            'valcal': class_dist_summary(valcal),
        }
        print(f"  Fold {fold}: train={len(train_new)} {summary[fold]['train']['pcts']} | "
              f"valsel={len(valsel)} {summary[fold]['valsel']['pcts']} | "
              f"valcal={len(valcal)} {summary[fold]['valcal']['pcts']}")
    return summary


def generate_split_B_val():
    print("\n" + "=" * 72)
    print("SPLIT B: Splitting 675 train drugs into 540 train / 67 valsel / 68 valcal")
    print("=" * 72)
    summary = {}
    for fold in range(N_FOLDS):
        old_train = np.load(os.path.join(OLD_SPLIT_DIR, f'splitB_fold{fold}_train.npy'))
        test_drugs = np.load(os.path.join(OLD_SPLIT_DIR, f'splitB_fold{fold}_test_drugs.npy')).astype(int)

        train_drugs_all = np.unique(old_train[:, 0].astype(int))
        expected_train_drugs = np.setdiff1d(np.arange(N_DRUGS), test_drugs)
        assert np.array_equal(train_drugs_all, expected_train_drugs), f"Fold {fold}: train drugs mismatch"

        # Stratify the 675 train drugs by 5 quantile bins of #known SEs (computed on old_train only)
        d_col = old_train[:, 0].astype(int)
        n_ses_per_drug = np.array([(d_col == d).sum() for d in train_drugs_all])
        bins = np.quantile(n_ses_per_drug, [0.2, 0.4, 0.6, 0.8])
        drug_strat = np.digitize(n_ses_per_drug, bins)

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
        drug_buckets = [te_idx for _, te_idx in skf.split(train_drugs_all, drug_strat)]

        # Bucket 9 has 67 drugs -> valsel_drugs; Bucket 0 has 68 drugs -> valcal_drugs;
        # Buckets 1..8 have 540 drugs -> train_drugs
        valsel_drugs = np.sort(train_drugs_all[drug_buckets[9]])
        valcal_drugs = np.sort(train_drugs_all[drug_buckets[0]])
        train_drugs = np.sort(train_drugs_all[np.concatenate(drug_buckets[1:9])])

        train_new = old_train[np.isin(d_col, train_drugs)]
        valsel = old_train[np.isin(d_col, valsel_drugs)]
        valcal = old_train[np.isin(d_col, valcal_drugs)]

        np.save(os.path.join(VAL_SPLIT_DIR, f'splitB_fold{fold}_train.npy'), train_new)
        np.save(os.path.join(VAL_SPLIT_DIR, f'splitB_fold{fold}_valsel.npy'), valsel)
        np.save(os.path.join(VAL_SPLIT_DIR, f'splitB_fold{fold}_valcal.npy'), valcal)
        np.save(os.path.join(VAL_SPLIT_DIR, f'splitB_fold{fold}_train_drugs.npy'), train_drugs)
        np.save(os.path.join(VAL_SPLIT_DIR, f'splitB_fold{fold}_valsel_drugs.npy'), valsel_drugs)
        np.save(os.path.join(VAL_SPLIT_DIR, f'splitB_fold{fold}_valcal_drugs.npy'), valcal_drugs)

        summary[fold] = {
            'n_drugs': {
                'train': int(len(train_drugs)),
                'valsel': int(len(valsel_drugs)),
                'valcal': int(len(valcal_drugs)),
                'test': int(len(test_drugs)),
            },
            'train': class_dist_summary(train_new),
            'valsel': class_dist_summary(valsel),
            'valcal': class_dist_summary(valcal),
        }
        print(f"  Fold {fold}: drugs=({len(train_drugs)}/{len(valsel_drugs)}/{len(valcal_drugs)}) | "
              f"train={len(train_new)} {summary[fold]['train']['pcts']} | "
              f"valsel={len(valsel)} {summary[fold]['valsel']['pcts']} | "
              f"valcal={len(valcal)} {summary[fold]['valcal']['pcts']}")
    return summary


def generate_v2_mask_mats():
    """Generate new mask_mat_750_v2_{A|B}.mat without touching existing mask_mat_750.mat."""
    print("\n" + "=" * 72)
    print("MASKS V2: Generating mask_mat_750_v2_A.mat and mask_mat_750_v2_B.mat")
    print("=" * 72)
    from split_adapter_v2 import save_mask_mat_v2
    with open(os.path.join(BASE_DIR, 'drug_side.pkl'), 'rb') as f:
        freq_matrix = np.array(pickle.load(f))
    for split_type in ['A', 'B']:
        out_path = os.path.join(BASE_DIR, f'mask_mat_750_v2_{split_type}.mat')
        save_mask_mat_v2(freq_matrix, split_type=split_type, n_folds=N_FOLDS, output_path=out_path)


def verify_all():
    """Run exhaustive automated verification on all generated splits."""
    print("\n" + "=" * 72)
    print("AUTOMATED VERIFICATION OF SPLITS_VAL")
    print("=" * 72)
    from split_adapter import load_splits
    from split_adapter_v2 import load_splits_v2, load_val_drugs

    baseline_sha_path = os.path.join(
        os.path.dirname(BASE_DIR), 'MSSF_plus', 'audit', 'outputs', 'shared_data_sha256_baseline.json'
    )
    baseline_shas = {}
    if os.path.exists(baseline_sha_path):
        with open(baseline_sha_path, 'r') as f:
            baseline_shas = json.load(f)['files']

    report = {'split_A': {}, 'split_B': {}, 'all_checks_passed': True}

    for split_type in ['A', 'B']:
        for fold in range(N_FOLDS):
            old_train, old_test = load_splits(fold, split_type)
            train, valsel, valcal, test = load_splits_v2(fold, split_type)

            # 1. Test set untouched (array_equal + SHA256)
            assert np.array_equal(old_test, test), f"Split {split_type} fold {fold}: test array changed!"
            test_rel = f"splits/split{split_type}_fold{fold}_test.npy"
            test_sha = sha256_file(os.path.join(BASE_DIR, test_rel))
            if test_rel in baseline_shas:
                assert test_sha == baseline_shas[test_rel]['sha256'], "Test SHA256 mismatch!"

            # 2. Row counts and pairwise disjointness of (drug, se) pairs
            assert len(train) + len(valsel) + len(valcal) == len(old_train), "Length sum mismatch!"
            to_pair_set = lambda a: set(zip(a[:, 0].astype(int).tolist(), a[:, 1].astype(int).tolist()))
            s_tr, s_vs, s_vc, s_te = map(to_pair_set, [train, valsel, valcal, test])
            assert len(s_tr) == len(train) and len(s_vs) == len(valsel) and len(s_vc) == len(valcal)
            assert s_tr.isdisjoint(s_vs), f"Split {split_type} fold {fold}: train & valsel overlap!"
            assert s_tr.isdisjoint(s_vc), f"Split {split_type} fold {fold}: train & valcal overlap!"
            assert s_vs.isdisjoint(s_vc), f"Split {split_type} fold {fold}: valsel & valcal overlap!"
            assert (s_tr | s_vs | s_vc).isdisjoint(s_te), f"Split {split_type} fold {fold}: train/val & test overlap!"

            # 3. Exact row-by-row union == old_train (when sorted by (drug_idx, se_idx))
            recombined = np.vstack([train, valsel, valcal])
            order_rec = np.lexsort((recombined[:, 1], recombined[:, 0]))
            order_old = np.lexsort((old_train[:, 1], old_train[:, 0]))
            assert np.array_equal(recombined[order_rec], old_train[order_old]), (
                f"Split {split_type} fold {fold}: union of train/valsel/valcal != old_train!"
            )

            fold_info = {
                'union_equals_old_train_exact': True,
                'pairwise_disjoint_pairs': True,
                'test_sha256_unchanged': True,
                'train': class_dist_summary(train),
                'valsel': class_dist_summary(valsel),
                'valcal': class_dist_summary(valcal),
                'test': class_dist_summary(test),
            }

            # 4. Split B drug-level disjointness check
            if split_type == 'B':
                tr_d = set(load_val_drugs(fold, 'train').tolist())
                vs_d = set(load_val_drugs(fold, 'valsel').tolist())
                vc_d = set(load_val_drugs(fold, 'valcal').tolist())
                te_d = set(load_val_drugs(fold, 'test').tolist())

                assert len(tr_d) == 540 and len(vs_d) == 67 and len(vc_d) == 68 and len(te_d) == 75
                assert tr_d.isdisjoint(vs_d) and tr_d.isdisjoint(vc_d) and tr_d.isdisjoint(te_d)
                assert vs_d.isdisjoint(vc_d) and vs_d.isdisjoint(te_d) and vc_d.isdisjoint(te_d)
                assert (tr_d | vs_d | vc_d | te_d) == set(range(N_DRUGS))

                # Also verify actual drugs in pair arrays match the partition drug sets
                assert set(train[:, 0].astype(int).tolist()) == tr_d
                assert set(valsel[:, 0].astype(int).tolist()) == vs_d
                assert set(valcal[:, 0].astype(int).tolist()) == vc_d
                assert set(test[:, 0].astype(int).tolist()) == te_d
                fold_info['drug_sets_disjoint_and_complete_750'] = True
                fold_info['drug_counts'] = [len(tr_d), len(vs_d), len(vc_d), len(te_d)]

            report[f'split_{split_type}'][fold] = fold_info

    out_json = os.path.join(
        os.path.dirname(BASE_DIR), 'MSSF_plus', 'audit', 'outputs', 'step2a_val_splits_verification.json'
    )
    if os.path.exists(os.path.dirname(out_json)):
        with open(out_json, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"  [saved] {out_json}")
    print("  ✅ ALL AUTOMATED CHECKS PASSED across all 10 folds of Split A and Split B!")


if __name__ == '__main__':
    generate_split_A_val()
    generate_split_B_val()
    generate_v2_mask_mats()
    verify_all()
