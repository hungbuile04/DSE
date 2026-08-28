"""
Generate unified train/test splits for all DSE baselines.

Two scenarios:
  Split A — Warm-start (pair-wise): 10-fold on (drug, SE) pairs
  Split B — Drug cold-start (drug-level): 10-fold on drugs

Output:
  shared_data/splits/
    splitA_fold{k}_train.npy   # shape (N_train, 3): [drug_idx, se_idx, frequency]
    splitA_fold{k}_test.npy    # shape (N_test, 3):  [drug_idx, se_idx, frequency]
    splitB_fold{k}_train.npy
    splitB_fold{k}_test.npy
    splitB_fold{k}_test_drugs.npy  # indices of held-out drugs

Usage:
    python generate_splits.py
"""

import os
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold

SEED = 42
N_FOLDS = 10

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'splits')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load frequency matrix
with open(os.path.join(BASE_DIR, 'drug_side.pkl'), 'rb') as f:
    freq_matrix = np.array(pickle.load(f))  # (750, 994)

N_DRUGS, N_SIDES = freq_matrix.shape
print(f"Frequency matrix: {N_DRUGS} drugs × {N_SIDES} side effects")


# ============================================================
# SPLIT A: Warm-start (pair-wise)
# Randomly split positive (drug, SE) pairs into 10 folds
# Negative samples: all zero pairs (shared across folds)
# ============================================================
def generate_split_A():
    print("\n" + "=" * 60)
    print("SPLIT A: Warm-start (pair-wise)")
    print("=" * 60)
    
    # Extract positive pairs
    pos_indices = np.argwhere(freq_matrix > 0)  # (N_pos, 2)
    pos_freq = freq_matrix[pos_indices[:, 0], pos_indices[:, 1]]
    
    # For stratification: use frequency class
    strat_labels = pos_freq.astype(int)
    
    print(f"  Positive pairs: {len(pos_indices)}")
    print(f"  Stratified by frequency: {dict(zip(*np.unique(strat_labels, return_counts=True)))}")
    
    kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(pos_indices, strat_labels)):
        train_pairs = pos_indices[train_idx]
        test_pairs = pos_indices[test_idx]
        train_freq = pos_freq[train_idx]
        test_freq = pos_freq[test_idx]
        
        # Save: [drug_idx, se_idx, frequency]
        train_data = np.column_stack([train_pairs, train_freq])
        test_data = np.column_stack([test_pairs, test_freq])
        
        np.save(os.path.join(OUTPUT_DIR, f'splitA_fold{fold}_train.npy'), train_data)
        np.save(os.path.join(OUTPUT_DIR, f'splitA_fold{fold}_test.npy'), test_data)
        
        print(f"  Fold {fold}: train={len(train_data)}, test={len(test_data)}")
    
    print("  ✅ Split A complete!")


# ============================================================
# SPLIT B: Drug cold-start (drug-level)
# Hold out entire drugs — test drugs never seen during training
# ============================================================
def generate_split_B():
    print("\n" + "=" * 60)
    print("SPLIT B: Drug cold-start (drug-level)")
    print("=" * 60)
    
    drug_indices = np.arange(N_DRUGS)
    
    # Stratify by number of known SEs per drug (binned)
    n_ses_per_drug = (freq_matrix > 0).sum(axis=1)
    # Bin into 5 groups for stratification
    bins = np.quantile(n_ses_per_drug, [0.2, 0.4, 0.6, 0.8])
    drug_strat = np.digitize(n_ses_per_drug, bins)
    
    print(f"  Total drugs: {N_DRUGS}")
    print(f"  SEs per drug: min={n_ses_per_drug.min()}, max={n_ses_per_drug.max()}, mean={n_ses_per_drug.mean():.1f}")
    
    kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    for fold, (train_drug_idx, test_drug_idx) in enumerate(kfold.split(drug_indices, drug_strat)):
        train_drugs = drug_indices[train_drug_idx]
        test_drugs = drug_indices[test_drug_idx]
        
        # Training pairs: all known pairs from train drugs
        train_pairs = []
        for d in train_drugs:
            for s in range(N_SIDES):
                if freq_matrix[d, s] > 0:
                    train_pairs.append([d, s, freq_matrix[d, s]])
        
        # Test pairs: all known pairs from test drugs
        test_pairs = []
        for d in test_drugs:
            for s in range(N_SIDES):
                if freq_matrix[d, s] > 0:
                    test_pairs.append([d, s, freq_matrix[d, s]])
        
        train_data = np.array(train_pairs)
        test_data = np.array(test_pairs)
        
        np.save(os.path.join(OUTPUT_DIR, f'splitB_fold{fold}_train.npy'), train_data)
        np.save(os.path.join(OUTPUT_DIR, f'splitB_fold{fold}_test.npy'), test_data)
        np.save(os.path.join(OUTPUT_DIR, f'splitB_fold{fold}_test_drugs.npy'), test_drugs)
        
        print(f"  Fold {fold}: train_drugs={len(train_drugs)}, test_drugs={len(test_drugs)}, "
              f"train_pairs={len(train_data)}, test_pairs={len(test_data)}")
    
    print("  ✅ Split B complete!")


# ============================================================
# SPLIT COMPATIBILITY ANALYSIS
# ============================================================
def print_compatibility():
    print("\n" + "=" * 60)
    print("METHOD COMPATIBILITY")
    print("=" * 60)
    print("""
    Method      | Split A (Warm) | Split B (Cold) | Lý do
    ------------|----------------|----------------|---------------------------
    MGPred      |      ✅        |      ❌        | Drug features = freq-based similarity
    SDPred      |      ✅        |      ❌        | Drug features include freq cosine sim
    MSSF        |      ✅        |      ❌        | Same as SDPred
    DSGAT       |      ✅        |      ✅        | Drug = molecular graph (SMILES)
    A3Net       |      ✅        |      ✅        | Drug = molecular graph (SMILES)
    HSTrans     |      ✅        |      ✅        | Drug = SMILES substructures
    MVSL-DSF    |      ✅        |      ⚠️        | 3/7 views OK, 2 views use freq profile

    ⚠️ MVSL-DSF cold-start:
       Views 1-3 (graph, SMILES, fingerprint) → OK (drug-intrinsic)
       Views 4-5 (association/frequency profile) → LEAK if not masked
       Views 6-7 (SE features) → OK (SE-intrinsic)
       → CAN work if views 4-5 are zeroed for test drugs
    """)


if __name__ == '__main__':
    generate_split_A()
    generate_split_B()
    print_compatibility()
    
    print(f"\nAll splits saved to: {OUTPUT_DIR}/")
    print(f"Files: {sorted(os.listdir(OUTPUT_DIR))}")
