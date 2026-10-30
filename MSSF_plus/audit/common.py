"""Shared helpers for the MSSF_plus audit scripts.

Read-only: never writes into baseline folders, shared_data/ or results/.
All outputs go to MSSF_plus/audit/outputs/.
"""
import os
import sys
import json
import pickle

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

AUDIT_DIR = os.path.dirname(os.path.abspath(__file__))
PLUS_DIR = os.path.dirname(AUDIT_DIR)
REPO = os.path.dirname(PLUS_DIR)
SHARED = os.path.join(REPO, 'shared_data')
MSSF750 = os.path.join(SHARED, 'mssf_750')
OUT_DIR = os.path.join(AUDIT_DIR, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, SHARED)
from split_adapter import load_splits, load_test_drugs  # noqa: E402

N_DRUGS, N_SES = 750, 994

# Order of the 11 drug / 4 SE matrices exactly as in MSSF/mssf.py read_raw_data() L114-135
DRUG_VIEW_NAMES = [
    'D1_stitch_one', 'D2_stitch_two', 'D3_stitch_three', 'D4_stitch_four', 'D5_stitch_five',
    'D6_mol2vec_cos', 'D7_target_cos', 'D8_fingerprint', 'D9_profile_freq_cos',
    'D10_profile_bin_cos', 'D11_pathway_enzyme',
]
SE_VIEW_NAMES = ['S1_adrecs_semantic', 'S2_glove_cos', 'S3_profile_freq_cos', 'S4_profile_bin_cos']
PROFILE_DRUG_VIEWS = [8, 9]   # 0-based -> D9, D10
PROFILE_SE_VIEWS = [2, 3]     # 0-based -> S3, S4


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def drug_side_full():
    """750x994 frequency matrix used by MSSF in the unified runs (copied from mssf_750/)."""
    return np.array(load_pkl(os.path.join(MSSF750, 'drug_side.pkl')), dtype=np.float64)


def static_views():
    """Non-profile views, computed exactly like MSSF read_raw_data()."""
    r = MSSF750
    d = {}
    for i, name in enumerate(['one', 'two', 'three', 'four', 'five']):
        d[DRUG_VIEW_NAMES[i]] = np.array(load_pkl(f'{r}/Text_similarity_{name}.pkl'), dtype=np.float64)
    d['D6_mol2vec_cos'] = cosine_similarity(np.array(load_pkl(f'{r}/drug_mol.pkl'), dtype=np.float64))
    d['D7_target_cos'] = cosine_similarity(np.array(load_pkl(f'{r}/drug_target.pkl'), dtype=np.float64))
    d['D8_fingerprint'] = np.array(load_pkl(f'{r}/fingerprint_similarity.pkl'), dtype=np.float64)
    d['D11_pathway_enzyme'] = np.array(load_pkl(f'{r}/drug_pathway_enzyme_similarity.pkl'), dtype=np.float64)
    s = {
        'S1_adrecs_semantic': np.array(load_pkl(f'{r}/side_effect_semantic.pkl'), dtype=np.float64),
        'S2_glove_cos': cosine_similarity(np.array(load_pkl(f'{r}/glove_wordEmbedding.pkl'), dtype=np.float64)),
    }
    return d, s


def mssf_profile_views_from_masked(drug_side_masked):
    """Replicates MSSF read_raw_data() L100-129 on an already-masked matrix."""
    drug_side_label = (drug_side_masked > 0).astype(np.float64)
    return {
        'D9_profile_freq_cos': cosine_similarity(drug_side_masked),
        'D10_profile_bin_cos': cosine_similarity(drug_side_label),
        'S3_profile_freq_cos': cosine_similarity(drug_side_masked.T),
        'S4_profile_bin_cos': cosine_similarity(drug_side_label.T),
    }


def mssf_masking_as_in_notebook(drug_side, test_pos):
    """Replicates exactly what the patched MSSF does with data_test (notebook cell 9 + mssf.py L96-98).

    notebook: data_test = [(int(r[0]), int(r[1]), r[2]) for r in test_pos]; mssf: np.array(data_test)
    -> float64 array; patched loop: drug_side[int(data_test[i,0]), int(data_test[i,1])] = 0
    """
    ds = drug_side.copy()
    data_test = np.array([(int(r[0]), int(r[1]), r[2]) for r in test_pos])
    for i in range(data_test.shape[0]):
        ds[int(data_test[i, 0]), int(data_test[i, 1])] = 0
    return ds, data_test.dtype


def matrix_from_pairs(pairs):
    """Build a 750x994 matrix directly from [drug, se, freq] rows (train-only reconstruction)."""
    m = np.zeros((N_DRUGS, N_SES), dtype=np.float64)
    p = np.asarray(pairs)
    m[p[:, 0].astype(int), p[:, 1].astype(int)] = p[:, 2]
    return m


def diff_stats(a, b):
    d = np.abs(a - b)
    return {
        'max_abs_diff': float(d.max()),
        'mean_abs_diff': float(d.mean()),
        'frac_entries_changed(>1e-12)': float((d > 1e-12).mean()),
    }


def save_json(obj, name):
    path = os.path.join(OUT_DIR, name)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print(f'[saved] {os.path.relpath(path, REPO)}')
