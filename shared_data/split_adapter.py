"""
Unified Split Adapter for all DSE methods.

Provides functions to convert unified splits into the format each method expects:
- Pair-based methods (MGPred, SDPred, MSSF, HSTrans): data_train/data_test arrays
- Mask-based methods (DSGAT, A3Net): mask matrices
- Sample-based methods (MVSL-DSF): sample arrays with labels

Usage:
    from split_adapter import load_splits, make_mask_matrix, make_masked_freq
"""

import os
import numpy as np
import scipy.io as sio


SPLIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'splits')


def load_splits(fold, split_type='A'):
    """
    Load unified split for a given fold.
    
    Args:
        fold: fold index (0-9)
        split_type: 'A' (warm-start pair-wise) or 'B' (drug cold-start)
    
    Returns:
        train_data: np.array of shape (N_train, 3) -> [drug_idx, se_idx, frequency]
        test_data:  np.array of shape (N_test, 3)  -> [drug_idx, se_idx, frequency]
    """
    prefix = f'split{split_type}'
    train = np.load(os.path.join(SPLIT_DIR, f'{prefix}_fold{fold}_train.npy'))
    test = np.load(os.path.join(SPLIT_DIR, f'{prefix}_fold{fold}_test.npy'))
    return train, test


def load_test_drugs(fold, split_type='B'):
    """Load held-out drug indices for cold-start split."""
    return np.load(os.path.join(SPLIT_DIR, f'split{split_type}_fold{fold}_test_drugs.npy'))


def make_mask_matrix(freq_matrix, test_data):
    """
    Create a binary mask matrix for DSGAT/A3Net.
    mask[i,j] = 0 for test pairs, 1 for everything else.
    
    Args:
        freq_matrix: original frequency matrix (N_drugs, N_sides)
        test_data: array of [drug_idx, se_idx, frequency]
    
    Returns:
        mask: np.array of shape (N_drugs, N_sides), 0 = test, 1 = train
    """
    mask = np.ones(freq_matrix.shape, dtype=np.float64)
    for row in test_data:
        d, s = int(row[0]), int(row[1])
        mask[d, s] = 0
    return mask


def make_masked_freq(freq_matrix, test_data):
    """
    Create frequency matrix with test pairs zeroed out (for data leakage prevention).
    
    Args:
        freq_matrix: original frequency matrix (N_drugs, N_sides)
        test_data: array of [drug_idx, se_idx, frequency]
    
    Returns:
        masked_freq: copy of freq_matrix with test entries set to 0
    """
    masked_freq = freq_matrix.copy()
    for row in test_data:
        d, s = int(row[0]), int(row[1])
        masked_freq[d, s] = 0
    return masked_freq


def save_mask_mat(freq_matrix, split_type='A', n_folds=10, output_path=None):
    """
    Generate mask_mat.mat file compatible with DSGAT/A3Net from unified splits.
    Saves a .mat file with keys mask0, mask1, ..., mask9.
    
    Args:
        freq_matrix: original frequency matrix (N_drugs, N_sides)
        split_type: 'A' or 'B'
        n_folds: number of folds
        output_path: path to save the .mat file
    """
    masks = {}
    for fold in range(n_folds):
        _, test_data = load_splits(fold, split_type)
        mask = make_mask_matrix(freq_matrix, test_data)
        masks[f'mask{fold}'] = mask
    
    if output_path:
        sio.savemat(output_path, masks)
        print(f"Saved unified mask matrix to {output_path}")
    
    return masks


def make_negative_samples(freq_matrix, train_data, n_neg=None, seed=42):
    """
    Sample negative pairs (drug-SE pairs with freq=0) for training.
    Uses the MASKED frequency matrix (test pairs already zeroed).
    
    Args:
        freq_matrix: MASKED frequency matrix (test pairs = 0)
        train_data: positive training pairs
        n_neg: number of negatives (default: same as positives)
        seed: random seed
    
    Returns:
        neg_data: np.array of shape (n_neg, 3) -> [drug_idx, se_idx, 0]
    """
    if n_neg is None:
        n_neg = len(train_data)
    
    rng = np.random.RandomState(seed)
    zero_indices = np.argwhere(freq_matrix == 0)
    
    selected = rng.choice(len(zero_indices), size=min(n_neg, len(zero_indices)), replace=False)
    neg_pairs = zero_indices[selected]
    neg_data = np.column_stack([neg_pairs, np.zeros(len(neg_pairs))])
    
    return neg_data


def get_train_test_for_pairwise_method(freq_matrix, fold, split_type='A', 
                                        include_neg=True, neg_ratio=1.0, seed=42):
    """
    Get train/test data in the format used by MGPred/SDPred/MSSF/HSTrans.
    
    Returns:
        data_train_pos: positive training pairs [drug, se, freq]
        data_test_pos: positive test pairs [drug, se, freq]
        data_train_neg: negative training pairs [drug, se, 0] (if include_neg)
        data_test_neg: negative test pairs [drug, se, 0] (if include_neg)
        masked_freq: frequency matrix with test pairs zeroed
    """
    train_pos, test_pos = load_splits(fold, split_type)
    masked_freq = make_masked_freq(freq_matrix, test_pos)
    
    if not include_neg:
        return train_pos, test_pos, None, None, masked_freq
    
    n_neg_train = int(len(train_pos) * neg_ratio)
    n_neg_test = int(len(test_pos) * neg_ratio)
    
    rng = np.random.RandomState(seed + fold)
    
    # Train negatives: from masked matrix
    zero_indices = np.argwhere(masked_freq == 0)
    # Exclude test pair positions from negatives
    test_set = set(map(tuple, test_pos[:, :2].astype(int)))
    zero_indices = np.array([idx for idx in zero_indices if tuple(idx) not in test_set])
    
    sel = rng.choice(len(zero_indices), size=min(n_neg_train, len(zero_indices)), replace=False)
    train_neg = np.column_stack([zero_indices[sel], np.zeros(len(sel))])
    
    # Test negatives: sample from test drug/SE combinations with freq=0
    test_drugs = np.unique(test_pos[:, 0].astype(int))
    test_ses = np.unique(test_pos[:, 1].astype(int))
    test_neg_candidates = []
    for d in test_drugs:
        for s in test_ses:
            if freq_matrix[d, s] == 0 and (d, s) not in test_set:
                test_neg_candidates.append([d, s, 0])
    test_neg_candidates = np.array(test_neg_candidates) if test_neg_candidates else np.empty((0, 3))
    
    if len(test_neg_candidates) > n_neg_test:
        sel = rng.choice(len(test_neg_candidates), size=n_neg_test, replace=False)
        test_neg = test_neg_candidates[sel]
    else:
        test_neg = test_neg_candidates
    
    return train_pos, test_pos, train_neg, test_neg, masked_freq


if __name__ == '__main__':
    import pickle
    
    # Verify
    with open(os.path.join(os.path.dirname(SPLIT_DIR), 'drug_side.pkl'), 'rb') as f:
        freq = np.array(pickle.load(f))
    
    print(f"Frequency matrix: {freq.shape}")
    
    for split in ['A', 'B']:
        print(f"\n=== Split {split} ===")
        for fold in range(2):  # Just test first 2 folds
            train, test = load_splits(fold, split)
            mask = make_mask_matrix(freq, test)
            print(f"  Fold {fold}: train={len(train)}, test={len(test)}, "
                  f"mask_zeros={int((mask == 0).sum())}")
