"""
Unified Split Adapter v2 — extends split_adapter.py with validation splits (train / valsel / valcal / test)
without modifying the original shared_data/split_adapter.py file (preserving its SHA256 checksum).

Usage:
    from split_adapter_v2 import (
        load_splits,          # re-exported unchanged from split_adapter
        load_test_drugs,      # re-exported unchanged from split_adapter
        load_splits_v2,       # NEW: returns (train, valsel, valcal, test)
        load_val_drugs,       # NEW: returns drug indices for 'train' | 'valsel' | 'valcal' | 'test' in Split B
        make_train_only_freq, # NEW: builds 750x994 frequency matrix from train (80%) pairs only
        make_split_masks_v2,  # NEW: returns train/valsel/valcal/test masks + 5-state partition matrix
        save_mask_mat_v2,     # NEW: saves mask_mat_750_v2_{A|B}.mat for DSGAT / A-3Net
    )
"""

import os
import numpy as np
import scipy.io as sio

from split_adapter import (  # noqa: F401
    SPLIT_DIR,
    load_splits,
    load_test_drugs,
    make_mask_matrix,
    make_masked_freq,
    save_mask_mat,
    make_negative_samples,
    get_train_test_for_pairwise_method,
)

VAL_SPLIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'splits_val')


def load_splits_v2(fold, split_type='A'):
    """
    Load unified 4-way split (train 80% / val_select 10% / val_calib 10% / test) for a given fold.

    Args:
        fold: fold index (0-9)
        split_type: 'A' (warm-start pair-wise) or 'B' (drug cold-start)

    Returns:
        train:  np.ndarray of shape (N_train, 3)  -> [drug_idx, se_idx, frequency]
        valsel: np.ndarray of shape (N_valsel, 3) -> [drug_idx, se_idx, frequency]
        valcal: np.ndarray of shape (N_valcal, 3) -> [drug_idx, se_idx, frequency]
        test:   np.ndarray of shape (N_test, 3)   -> [drug_idx, se_idx, frequency]
    """
    prefix = f'split{split_type}_fold{fold}'
    train = np.load(os.path.join(VAL_SPLIT_DIR, f'{prefix}_train.npy'))
    valsel = np.load(os.path.join(VAL_SPLIT_DIR, f'{prefix}_valsel.npy'))
    valcal = np.load(os.path.join(VAL_SPLIT_DIR, f'{prefix}_valcal.npy'))
    test = np.load(os.path.join(SPLIT_DIR, f'{prefix}_test.npy'))
    return train, valsel, valcal, test


def load_val_drugs(fold, part='valsel', split_type='B'):
    """
    Load drug indices for a given partition in Split B (drug cold-start).

    Args:
        fold: fold index (0-9)
        part: 'train' (540 drugs), 'valsel' (67 drugs), 'valcal' (68 drugs), or 'test' (75 drugs)
        split_type: must be 'B'

    Returns:
        np.ndarray of 1D integer drug indices
    """
    if split_type != 'B':
        raise ValueError("Drug-level partition indices are only defined for split_type='B'.")
    if part == 'test':
        return load_test_drugs(fold, 'B').astype(int)
    if part not in ('train', 'valsel', 'valcal'):
        raise ValueError(f"Unknown part={part!r}; expected 'train', 'valsel', 'valcal', or 'test'.")
    return np.load(os.path.join(VAL_SPLIT_DIR, f'splitB_fold{fold}_{part}_drugs.npy')).astype(int)


def make_train_only_freq(train_data, shape=(750, 994), dtype=np.float64):
    """
    Construct the drug-SE frequency matrix starting from zeros and filling ONLY the pairs
    in `train_data` (80% training split). Prevents any leakage from val_select, val_calib, or test.
    """
    mat = np.zeros(shape, dtype=dtype)
    if len(train_data) > 0:
        d = train_data[:, 0].astype(int)
        s = train_data[:, 1].astype(int)
        mat[d, s] = train_data[:, 2].astype(dtype)
    return mat


def make_split_masks_v2(freq_matrix, fold, split_type='A'):
    """
    Build masks for DSGAT / A-3Net under the 4-way split (train / valsel / valcal / test).

    Returns a dict with:
        'train_mask':     (750, 994) float64, 1 = visible during training, 0 = held-out (valsel/valcal/test).
                          In Split B, entire rows of valsel, valcal, and test drugs are 0.
        'valsel_mask':    (750, 994) float64, 1 only at positive val_select pairs.
        'valcal_mask':    (750, 994) float64, 1 only at positive val_calib pairs.
        'test_mask':      (750, 994) float64, 1 only at positive test pairs.
        'partition_code': (750, 994) int8:
                          0 = unobserved (freq==0), 1 = train_pos, 2 = valsel_pos, 3 = valcal_pos, 4 = test_pos.
    """
    train, valsel, valcal, test = load_splits_v2(fold, split_type)
    shape = freq_matrix.shape

    train_mask = np.ones(shape, dtype=np.float64)
    for part_arr in (valsel, valcal, test):
        train_mask[part_arr[:, 0].astype(int), part_arr[:, 1].astype(int)] = 0.0

    if split_type == 'B':
        for part in ('valsel', 'valcal', 'test'):
            heldout_drugs = load_val_drugs(fold, part, 'B')
            train_mask[heldout_drugs, :] = 0.0

    def pos_mask(arr):
        m = np.zeros(shape, dtype=np.float64)
        m[arr[:, 0].astype(int), arr[:, 1].astype(int)] = 1.0
        return m

    valsel_mask = pos_mask(valsel)
    valcal_mask = pos_mask(valcal)
    test_mask = pos_mask(test)

    partition_code = np.zeros(shape, dtype=np.int8)
    partition_code[train[:, 0].astype(int), train[:, 1].astype(int)] = 1
    partition_code[valsel[:, 0].astype(int), valsel[:, 1].astype(int)] = 2
    partition_code[valcal[:, 0].astype(int), valcal[:, 1].astype(int)] = 3
    partition_code[test[:, 0].astype(int), test[:, 1].astype(int)] = 4

    return {
        'train_mask': train_mask,
        'valsel_mask': valsel_mask,
        'valcal_mask': valcal_mask,
        'test_mask': test_mask,
        'partition_code': partition_code,
    }


def save_mask_mat_v2(freq_matrix, split_type='A', n_folds=10, output_path=None):
    """
    Save .mat file with leak-free training masks (`mask0..mask9`, compatible with DSGAT/A-3Net),
    plus `valsel_mask0..9`, `valcal_mask0..9`, `test_mask0..9`, and `partition0..9`.
    """
    mat_dict = {}
    for fold in range(n_folds):
        m = make_split_masks_v2(freq_matrix, fold, split_type)
        mat_dict[f'mask{fold}'] = m['train_mask']
        mat_dict[f'valsel_mask{fold}'] = m['valsel_mask']
        mat_dict[f'valcal_mask{fold}'] = m['valcal_mask']
        mat_dict[f'test_mask{fold}'] = m['test_mask']
        mat_dict[f'partition{fold}'] = m['partition_code']

    if output_path:
        sio.savemat(output_path, mat_dict, do_compression=True)
        print(f"  Saved v2 mask matrix ({split_type}) to {output_path}")
    return mat_dict
