"""Audit D — Side-effect (SE) ontology feature files, shapes, semantics, and 994-SE index alignment.

Files inspected:
  1. shared_data/side_effect_label_750.mat          (DSGAT / A3Net / HSTrans MedDRA 243-D vectors)
  2. DSGAT/data_WS/side_effect_label_750.mat        (verify identical to shared_data)
  3. shared_data/raw_frequency_750.mat              (verify column alignment with drug_side.pkl)
  4. MVSL-DSF/side_vector_level_123.pkl             (893 x 243 original MVSL-DSF MedDRA vectors)
  5. shared_data/mvsl_750_994/side_vector_level_123.pkl & col_mapping_893_to_994.npy
  6. shared_data/effect_side_semantic.pkl vs MSSF/Datas/side_effect_semantic.pkl (ADReCS 994 x 994)
"""
import os
import numpy as np
import scipy.io as sio
from sklearn.metrics.pairwise import cosine_similarity
from common import REPO, SHARED, MSSF750, drug_side_full, load_pkl, save_json


def run():
    ds_750 = drug_side_full()  # (750, 994) from shared_data/mssf_750/drug_side.pkl
    ds_shared = np.array(load_pkl(os.path.join(SHARED, 'drug_side.pkl')), dtype=np.float64)
    ds_mssf_orig = np.array(load_pkl(os.path.join(REPO, 'MSSF', 'Datas', 'drug_side.pkl')), dtype=np.float64)

    # 1. Check raw_frequency_750.mat alignment with drug_side.pkl
    raw_mat = sio.loadmat(os.path.join(SHARED, 'raw_frequency_750.mat'))
    raw_R = np.array(raw_mat['R'], dtype=np.float64)
    r_equals_ds = bool(np.array_equal(raw_R, ds_750))

    #Also check column frequency signatures between MSSF original (757x994) and unified (750x994)
    # Using the 748 unambiguously matched drugs from prepare_all_data
    sd757 = np.array(load_pkl(os.path.join(REPO, 'SDPred', 'data', 'drug_side.pkl')), dtype=np.float64)
    mssf757_equals_sd757 = bool(np.array_equal(ds_mssf_orig, sd757))
    keys757 = {}
    for j in range(sd757.shape[0]):
        keys757.setdefault(sd757[j].tobytes(), j)
    matched_pairs = [(i, keys757[ds_750[i].tobytes()]) for i in range(750) if ds_750[i].tobytes() in keys757]
    idx750 = [p[0] for p in matched_pairs]
    idx757 = [p[1] for p in matched_pairs]
    cols_aligned_750_vs_757 = bool(np.array_equal(ds_750[idx750, :], ds_mssf_orig[idx757, :]))

    # 2. Inspect side_effect_label_750.mat
    se_mat_shared = sio.loadmat(os.path.join(SHARED, 'side_effect_label_750.mat'))
    se_mat_dsgat = sio.loadmat(os.path.join(REPO, 'DSGAT', 'data_WS', 'side_effect_label_750.mat'))
    mat_keys = [k for k in se_mat_shared.keys() if not k.startswith('__')]
    node_label = np.array(se_mat_shared['node_label'], dtype=np.float64)
    dsgat_identical = bool(np.array_equal(node_label, se_mat_dsgat['node_label']))

    unique_vals_mat = np.unique(node_label).tolist()
    row_sums_mat = (node_label > 0).sum(axis=1)
    col_sums_mat = (node_label > 0).sum(axis=0)
    zero_rows_mat = np.where(np.abs(node_label).sum(axis=1) == 0)[0].tolist()
    zero_cols_mat = np.where(np.abs(node_label).sum(axis=0) == 0)[0].tolist()

    # Cosine similarity of side_effect_label_750.mat (for C2)
    se_ont_cos = cosine_similarity(node_label)
    off_diag = se_ont_cos[~np.eye(994, dtype=bool)]

    # 3. Inspect MVSL-DSF side_vector_level_123.pkl (893x243) and mvsl_750_994 (994x243)
    mvsl_orig_vec = np.array(load_pkl(os.path.join(REPO, 'MVSL-DSF', 'side_vector_level_123.pkl')), dtype=np.float64)
    mvsl_orig_freq = np.array(load_pkl(os.path.join(REPO, 'MVSL-DSF', 'drug_side_frequency_matrix.pkl')), dtype=np.float64)
    mvsl_orig_assoc = np.array(load_pkl(os.path.join(REPO, 'MVSL-DSF', 'drug_side_association_matrix.pkl')), dtype=np.float64)
    mvsl_exp_vec = np.array(load_pkl(os.path.join(SHARED, 'mvsl_750_994', 'side_vector_level_123.pkl')), dtype=np.float64)
    col_map = np.load(os.path.join(SHARED, 'mvsl_750_994', 'col_mapping_893_to_994.npy'))

    # Check how col_mapping_893_to_994 was built in prepare_all_data.py L197-203:
    #Wait: prepare_all_data compared freq_893[:, j893] with mg750[:, j994] using np.array_equal!
    # Let's check what max(freq_893) is and whether col_map has 893 elements or if there are duplicate SE columns!
    exact_matches_with_raw = 0
    exact_matches_with_div5 = 0
    exact_matches_assoc = 0
    col_map_via_assoc = []
    ambiguous_se_cols = []
    for j893 in range(mvsl_orig_freq.shape[1]):
        c_f = mvsl_orig_freq[:, j893]
        c_a = mvsl_orig_assoc[:, j893]
        m_raw = [j994 for j994 in range(994) if np.array_equal(c_f, ds_750[:, j994])]
        m_div5 = [j994 for j994 in range(994) if np.allclose(c_f * 5.0, ds_750[:, j994])]
        m_assoc = [j994 for j994 in range(994) if np.array_equal(c_a, (ds_750[:, j994] > 0).astype(c_a.dtype))]
        if m_raw:
            exact_matches_with_raw += 1
        if m_div5:
            exact_matches_with_div5 += 1
        if m_assoc:
            exact_matches_assoc += 1
            col_map_via_assoc.append(m_assoc[0])
        if len(m_raw) > 1 or len(m_div5) > 1:
            ambiguous_se_cols.append((j893, m_raw, m_div5))

    # Compare MVSL 893 vectors against side_effect_label_750.mat[col_map]
    if len(col_map) == 893:
        sub_node_label = node_label[col_map]
        mvsl_vs_mat_max_diff = float(np.abs(mvsl_orig_vec - sub_node_label).max())
        mvsl_vs_mat_equal_rows = int(np.all(mvsl_orig_vec == sub_node_label, axis=1).sum())
    else:
        mvsl_vs_mat_max_diff = -1.0
        mvsl_vs_mat_equal_rows = 0

    # Check how many rows of mvsl_orig_vec match ANY row of node_label
    mat_row_set = {node_label[i].tobytes(): i for i in range(994)}
    mvsl_rows_in_mat = sum(1 for j in range(mvsl_orig_vec.shape[0]) if mvsl_orig_vec[j].tobytes() in mat_row_set)

    # Check why prepare_all_data.py imputed 101 missing SEs with mean_vec in mvsl_750_994/side_vector_level_123.pkl
    # when side_effect_label_750.mat ALREADY has real 243-D MedDRA vectors for all 994 SEs!
    missing_994 = sorted(list(set(range(994)) - set(col_map.tolist())))
    imputed_in_mvsl_exp = bool(np.allclose(mvsl_exp_vec[missing_994], mvsl_orig_vec.mean(axis=0)))
    real_in_mat_for_missing = int((np.abs(node_label[missing_994]).sum(axis=1) > 0).sum())

    # 4. Compare ADReCS semantic similarity (S1) across folders and against MedDRA cosine sim
    s1_shared = np.array(load_pkl(os.path.join(SHARED, 'effect_side_semantic.pkl')), dtype=np.float64)
    s1_mssf750 = np.array(load_pkl(os.path.join(MSSF750, 'side_effect_semantic.pkl')), dtype=np.float64)
    s1_mssf_orig = np.array(load_pkl(os.path.join(REPO, 'MSSF', 'Datas', 'side_effect_semantic.pkl')), dtype=np.float64)
    s1_identical = bool(np.array_equal(s1_shared, s1_mssf750) and np.array_equal(s1_mssf750, s1_mssf_orig))
    s1_zero_rows = np.where(np.abs(s1_mssf750).sum(axis=1) == 0)[0].tolist()
    corr_s1_meddra = float(np.corrcoef(s1_mssf750[np.triu_indices(994, 1)], se_ont_cos[np.triu_indices(994, 1)])[0, 1])

    report = {
        'alignment_994_SE': {
            'drug_side_750_equals_shared_drug_side': bool(np.array_equal(ds_750, ds_shared)),
            'raw_frequency_750_mat_R_equals_drug_side_750': r_equals_ds,
            'MSSF_757_equals_SDPred_757': mssf757_equals_sd757,
            'columns_994_aligned_between_750_and_757_on_748_matched_drugs': cols_aligned_750_vs_757,
        },
        'side_effect_label_750_mat': {
            'keys': mat_keys,
            'shape': list(node_label.shape),
            'dtype': str(node_label.dtype),
            'shared_identical_to_DSGAT': dsgat_identical,
            'unique_values': unique_vals_mat,
            'nnz_per_SE_min_median_mean_max': [
                int(row_sums_mat.min()), float(np.median(row_sums_mat)),
                float(row_sums_mat.mean()), int(row_sums_mat.max())
            ],
            'n_zero_SE_rows': len(zero_rows_mat),
            'zero_SE_rows': zero_rows_mat,
            'n_zero_feature_cols': len(zero_cols_mat),
            'zero_feature_cols': zero_cols_mat,
            'cosine_sim_stats': {
                'shape': list(se_ont_cos.shape),
                'diag_min_max': [float( np.diag(se_ont_cos).min()), float(np.diag(se_ont_cos).max())],
                'offdiag_min_mean_std_max': [
                    float(off_diag.min()), float(off_diag.mean()),
                    float(off_diag.std()), float(off_diag.max())
                ],
                'frac_offdiag_positive': float((off_diag > 0).mean()),
                'pearson_corr_with_S1_adrecs_upper_tri': corr_s1_meddra,
            },
        },
        'mvsl_side_vector_level_123': {
            'orig_shape': list(mvsl_orig_vec.shape),
            'exp_shape': list(mvsl_exp_vec.shape),
            'mvsl_orig_freq_min_max_unique_count': [
                float(mvsl_orig_freq.min()), float(mvsl_orig_freq.max()), int(len(np.unique(mvsl_orig_freq)))
            ],
            'col_mapping_len': int(len(col_map)),
            'col_mapping_unique_len': int(len(np.unique(col_map))),
            'exact_matches_with_raw_drug_side': exact_matches_with_raw,
            'exact_matches_with_div5_drug_side': exact_matches_with_div5,
            'exact_matches_with_assoc': exact_matches_assoc,
            'ambiguous_se_cols_count': len(ambiguous_se_cols),
            'ambiguous_se_cols': ambiguous_se_cols,
            'mvsl_893_vs_side_effect_label_750_at_col_map_max_diff': mvsl_vs_mat_max_diff,
            'mvsl_893_vs_side_effect_label_750_equal_rows': mvsl_vs_mat_equal_rows,
            'mvsl_893_rows_found_anywhere_in_side_effect_label_750': mvsl_rows_in_mat,
            'n_missing_SEs_in_mvsl_893': len(missing_994),
            'mvsl_750_994_imputed_missing_101_with_mean': imputed_in_mvsl_exp,
            'side_effect_label_750_has_real_vectors_for_those_101': real_in_mat_for_missing,
        },
        'S1_adrecs_semantic': {
            'shape': list(s1_mssf750.shape),
            'identical_across_shared_mssf750_mssfOrig': s1_identical,
            'zero_rows_in_S1': s1_zero_rows,
        }
    }

    print('=== Audit D: Ontology & SE Feature Alignment ===')
    for k, v in report['alignment_994_SE'].items():
        print(f'  {k}: {v}')
    print('\n--- side_effect_label_750.mat ---')
    for k, v in report['side_effect_label_750_mat'].items():
        print(f'  {k}: {v}')
    print('\n--- MVSL side_vector_level_123.pkl ---')
    for k, v in report['mvsl_side_vector_level_123'].items():
        print(f'  {k}: {v}')
    print('\n--- S1 ADReCS semantic ---')
    for k, v in report['S1_adrecs_semantic'].items():
        print(f'  {k}: {v}')

    save_json(report, 'audit_D_ontology.json')


if __name__ == '__main__':
    run()
