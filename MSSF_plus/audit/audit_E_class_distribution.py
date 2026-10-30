"""Audit E — 5-class frequency distribution globally and per fold (Split A & Split B).

Labels:
  SIDER frequency in {1, 2, 3, 4, 5} <-> ordinal class index in {0, 1, 2, 3, 4}.
  (0 in drug_side.pkl means unobserved / negative pair.)
"""
import numpy as np
from common import drug_side_full, load_splits, load_test_drugs, save_json, N_DRUGS, N_SES

FREQ_NAMES = {
    1: 'class_0 (freq=1, very rare)',
    2: 'class_1 (freq=2, rare)',
    3: 'class_2 (freq=3, infrequent)',
    4: 'class_3 (freq=4, frequent)',
    5: 'class_4 (freq=5, very frequent)',
}


def class_counts_and_pcts(freqs):
    freqs = np.asarray(freqs, dtype=int)
    n = len(freqs)
    counts = {int(c - 1): int((freqs == c).sum()) for c in range(1, 6)}
    pcts = {int(c - 1): round(100.0 * counts[c - 1] / n, 3) if n > 0 else 0.0 for c in range(1, 6)}
    c_arr = np.array([counts[k] for k in range(5)], dtype=np.float64)
    imbalance_max_min = float(c_arr.max() / max(c_arr.min(), 1.0))
    # Inverse-frequency weights normalized to mean=1: w_k = (N / (5 * N_k))
    inv_weights = (n / (5.0 * np.maximum(c_arr, 1.0))).round(4).tolist()
    return {
        'total_positive_pairs': int(n),
        'counts_class_0_to_4': [counts[k] for k in range(5)],
        'pcts_class_0_to_4': [pcts[k] for k in range(5)],
        'max_to_min_ratio': round(imbalance_max_min, 3),
        'inv_freq_weights_norm': inv_weights,
    }


def run():
    full = drug_side_full()
    total_cells = N_DRUGS * N_SES
    n_zero = int((full == 0).sum())
    pos_freqs = full[full > 0].astype(int)
    global_dist = class_counts_and_pcts(pos_freqs)
    global_dist['matrix_shape'] = [N_DRUGS, N_SES]
    global_dist['total_cells'] = total_cells
    global_dist['unobserved_zero_cells'] = n_zero
    global_dist['positive_density_pct'] = round(100.0 * len(pos_freqs) / total_cells, 3)

    print('=== Global Distribution (750 x 994) ===')
    print(f'Total cells: {total_cells:,} | Unobserved (0): {n_zero:,} | Positive (>0): {len(pos_freqs):,} '
          f'({global_dist["positive_density_pct"]:.3f}%)')
    for k in range(5):
        print(f'  Class {k} (freq={k+1}): {global_dist["counts_class_0_to_4"][k]:6d} '
              f'({global_dist["pcts_class_0_to_4"][k]:6.2f}%)  inv_w={global_dist["inv_freq_weights_norm"][k]:.4f}')
    print(f'  Max/Min class ratio: {global_dist["max_to_min_ratio"]:.2f}x')

    report = {'global': global_dist, 'split_A': {}, 'split_B': {}}

    for split in ['A', 'B']:
        print(f'\n=== Split {split} Per-Fold Class Distribution ===')
        print('fold |   N_train  train_counts [c0, c1, c2, c3, c4]        |  N_test  test_pcts [c0, c1, c2, c3, c4]')
        for fold in range(10):
            tr, te = load_splits(fold, split)
            tr_stats = class_counts_and_pcts(tr[:, 2])
            te_stats = class_counts_and_pcts(te[:, 2])
            entry = {'train': tr_stats, 'test': te_stats}
            if split == 'B':
                td = load_test_drugs(fold, 'B')
                entry['n_test_drugs'] = int(len(td))
                entry['n_train_drugs'] = int(N_DRUGS - len(td))
            report[f'split_{split}'][fold] = entry
            print(f'{fold:4d} | {tr_stats["total_positive_pairs"]:8d}  {str(tr_stats["counts_class_0_to_4"]):40s} | '
                  f'{te_stats["total_positive_pairs"]:7d}  {te_stats["pcts_class_0_to_4"]}')

    # Measure cross-fold variation in test class percentages (Split A vs Split B)
    for split in ['A', 'B']:
        pct_mat = np.array([report[f'split_{split}'][k]['test']['pcts_class_0_to_4'] for k in range(10)])
        report[f'split_{split}_test_pct_mean'] = pct_mat.mean(axis=0).round(3).tolist()
        report[f'split_{split}_test_pct_std'] = pct_mat.std(axis=0).round(3).tolist()
        print(f'Split {split} test class % mean ± std across 10 folds: '
              + ', '.join(f'c{i}={m:.2f}±{s:.2f}%' for i, (m, s) in
                          enumerate(zip(report[f"split_{split}_test_pct_mean"], report[f"split_{split}_test_pct_std"]))))

    save_json(report, 'audit_E_class_distribution.json')


if __name__ == '__main__':
    run()
