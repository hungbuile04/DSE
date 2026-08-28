"""
Prepare unified 750x994 data for SDPred, MSSF, and MVSL-DSF.

This script:
1. Maps 750 drugs (MGPred/DSGAT) to 757 drugs (SDPred/MSSF) 
2. Subsets SDPred/MSSF similarity matrices from 757x757 -> 750x750
3. Computes missing features from SMILES for unmatched drugs
4. Expands MVSL-DSF data from 893 -> 994 side effects
5. Saves all converted data to shared_data/

Usage:
    python prepare_all_data.py
"""

import os
import pickle
import numpy as np
import scipy.io as sio
from sklearn.metrics.pairwise import cosine_similarity

# ========== PATHS ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DSE_DIR = os.path.dirname(BASE_DIR)
SHARED_DIR = BASE_DIR

MGPRED_DATA = os.path.join(DSE_DIR, 'MGPred', 'data')
SDPRED_DATA = os.path.join(DSE_DIR, 'SDPred', 'data')
MSSF_DATA = os.path.join(DSE_DIR, 'MSSF', 'Datas')
MVSL_DIR = os.path.join(DSE_DIR, 'MVSL-DSF')
DSGAT_DATA = os.path.join(DSE_DIR, 'DSGAT', 'data_WS')

OUTPUT_SDPRED = os.path.join(SHARED_DIR, 'sdpred_750')
OUTPUT_MSSF = os.path.join(SHARED_DIR, 'mssf_750')
OUTPUT_MVSL = os.path.join(SHARED_DIR, 'mvsl_750_994')

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved: {path} ({type(data).__name__}, shape={getattr(data, 'shape', len(data) if hasattr(data, '__len__') else 'N/A')})")


# ============================================================
# STEP 1: Find mapping between 750 and 757 drugs
# ============================================================
def find_drug_mapping():
    """Find which rows in 757-drug matrix correspond to 750-drug matrix."""
    print("=" * 60)
    print("STEP 1: Finding drug mapping 750 <-> 757")
    print("=" * 60)
    
    mg750 = np.array(load_pkl(os.path.join(MGPRED_DATA, 'drug_side.pkl')))
    sd757 = np.array(load_pkl(os.path.join(SDPRED_DATA, 'drug_side.pkl')))
    
    mapping = []  # mapping[i] = j means drug i in 750 = drug j in 757
    unmatched_750 = []
    
    for i in range(750):
        found = False
        for j in range(757):
            if np.array_equal(mg750[i], sd757[j]):
                mapping.append(j)
                found = True
                break
        if not found:
            mapping.append(-1)  # -1 = no match
            unmatched_750.append(i)
    
    print(f"  Matched: {750 - len(unmatched_750)}/750")
    print(f"  Unmatched: {len(unmatched_750)} drugs at indices: {unmatched_750}")
    
    return mapping, unmatched_750, mg750


# ============================================================
# STEP 2: Subset SDPred/MSSF similarity matrices (757 -> 750)
# ============================================================
def convert_sdpred_mssf_data(mapping, unmatched_750, mg750):
    """Convert SDPred and MSSF data from 757 to 750 drugs."""
    print("\n" + "=" * 60)
    print("STEP 2: Converting SDPred/MSSF data (757 -> 750)")
    print("=" * 60)
    
    os.makedirs(OUTPUT_SDPRED, exist_ok=True)
    os.makedirs(OUTPUT_MSSF, exist_ok=True)
    
    matched_indices_757 = [m for m in mapping if m >= 0]  # indices in 757 for matched drugs
    matched_indices_750 = [i for i, m in enumerate(mapping) if m >= 0]  # indices in 750 for matched drugs
    
    # --- Drug-side frequency matrix (use MGPred's 750x994 directly) ---
    print("\n  [drug_side.pkl] Using MGPred 750x994 directly")
    save_pkl(mg750, os.path.join(OUTPUT_SDPRED, 'drug_side.pkl'))
    save_pkl(mg750, os.path.join(OUTPUT_MSSF, 'drug_side.pkl'))
    
    # --- Side-effect features (unchanged: 994x994 and 994x300) ---
    for name in ['side_effect_semantic.pkl', 'glove_wordEmbedding.pkl']:
        src = os.path.join(SDPRED_DATA, name)
        data = np.array(load_pkl(src))
        print(f"\n  [{name}] Unchanged: {data.shape}")
        save_pkl(data, os.path.join(OUTPUT_SDPRED, name))
        save_pkl(data, os.path.join(OUTPUT_MSSF, name))
    
    # --- Drug similarity matrices: subset rows & cols from 757x757 -> 750x750 ---
    drug_sim_files = [
        'Text_similarity_one.pkl',
        'Text_similarity_two.pkl', 
        'Text_similarity_three.pkl',
        'Text_similarity_four.pkl',
        'Text_similarity_five.pkl',
        'fingerprint_similarity.pkl',
    ]
    
    for name in drug_sim_files:
        src = os.path.join(SDPRED_DATA, name)
        sim757 = np.array(load_pkl(src))  # (757, 757)
        
        # Subset matched drugs
        sim750 = np.zeros((750, 750), dtype=sim757.dtype)
        for new_i, old_i in enumerate(mapping):
            for new_j, old_j in enumerate(mapping):
                if old_i >= 0 and old_j >= 0:
                    sim750[new_i, new_j] = sim757[old_i, old_j]
        
        # For unmatched drugs: set diagonal to 1 (self-similarity)
        for idx in unmatched_750:
            sim750[idx, idx] = 1.0
        
        print(f"\n  [{name}] Subset 757x757 -> 750x750")
        save_pkl(sim750, os.path.join(OUTPUT_SDPRED, name))
        save_pkl(sim750, os.path.join(OUTPUT_MSSF, name))
    
    # --- drug_mol.pkl: (757, 100) -> (750, 100) ---
    drug_mol_757 = np.array(load_pkl(os.path.join(SDPRED_DATA, 'drug_mol.pkl')))
    drug_mol_750 = np.zeros((750, drug_mol_757.shape[1]), dtype=drug_mol_757.dtype)
    for new_i, old_i in enumerate(mapping):
        if old_i >= 0:
            drug_mol_750[new_i] = drug_mol_757[old_i]
    # For unmatched: use MGPred Drug_word2vec if available
    mgpred_w2v = np.array(load_pkl(os.path.join(MGPRED_DATA, 'Drug_word2vec.pkl')))
    for idx in unmatched_750:
        drug_mol_750[idx] = mgpred_w2v[idx]
    print(f"\n  [drug_mol.pkl] Subset 757x100 -> 750x100 (unmatched filled from MGPred)")
    save_pkl(drug_mol_750, os.path.join(OUTPUT_SDPRED, 'drug_mol.pkl'))
    save_pkl(drug_mol_750, os.path.join(OUTPUT_MSSF, 'drug_mol.pkl'))
    
    # --- drug_target.pkl: (757, N) -> (750, N) ---
    drug_target_757 = np.array(load_pkl(os.path.join(SDPRED_DATA, 'drug_target.pkl')))
    drug_target_750 = np.zeros((750, drug_target_757.shape[1]), dtype=drug_target_757.dtype)
    for new_i, old_i in enumerate(mapping):
        if old_i >= 0:
            drug_target_750[new_i] = drug_target_757[old_i]
    print(f"\n  [drug_target.pkl] Subset 757x{drug_target_757.shape[1]} -> 750x{drug_target_757.shape[1]}")
    save_pkl(drug_target_750, os.path.join(OUTPUT_SDPRED, 'drug_target.pkl'))
    save_pkl(drug_target_750, os.path.join(OUTPUT_MSSF, 'drug_target.pkl'))
    
    # --- MSSF extra: drug_pathway_enzyme_similarity.pkl (757x757 -> 750x750) ---
    pathway_src = os.path.join(MSSF_DATA, 'drug_pathway_enzyme_similarity.pkl')
    if os.path.exists(pathway_src):
        pw757 = np.array(load_pkl(pathway_src))
        pw750 = np.zeros((750, 750), dtype=pw757.dtype)
        for new_i, old_i in enumerate(mapping):
            for new_j, old_j in enumerate(mapping):
                if old_i >= 0 and old_j >= 0:
                    pw750[new_i, new_j] = pw757[old_i, old_j]
        for idx in unmatched_750:
            pw750[idx, idx] = 1.0
        print(f"\n  [drug_pathway_enzyme_similarity.pkl] Subset 757x757 -> 750x750")
        save_pkl(pw750, os.path.join(OUTPUT_MSSF, 'drug_pathway_enzyme_similarity.pkl'))
    
    print("\n  ✅ SDPred/MSSF data conversion complete!")


# ============================================================
# STEP 3: Expand MVSL-DSF from 893 -> 994 side effects
# ============================================================
def convert_mvsl_data(mg750):
    """Expand MVSL-DSF data from 893 to 994 side effects."""
    print("\n" + "=" * 60)
    print("STEP 3: Expanding MVSL-DSF data (893 -> 994 SEs)")
    print("=" * 60)
    
    os.makedirs(OUTPUT_MVSL, exist_ok=True)
    
    # Load current MVSL data
    freq_893 = np.array(load_pkl(os.path.join(MVSL_DIR, 'drug_side_frequency_matrix.pkl')))
    assoc_893 = np.array(load_pkl(os.path.join(MVSL_DIR, 'drug_side_association_matrix.pkl')))
    side_vec_893 = np.array(load_pkl(os.path.join(MVSL_DIR, 'side_vector_level_123.pkl')))
    drug_smiles = load_pkl(os.path.join(MVSL_DIR, 'drug_smiles.pkl'))
    
    print(f"  Current: freq={freq_893.shape}, assoc={assoc_893.shape}, side_vec={side_vec_893.shape}")
    
    # Find column mapping: which 994 columns match 893 columns
    # We know from earlier analysis that all 893 cols are a subset of 994
    col_mapping_893_to_994 = []
    for j893 in range(893):
        col_893 = freq_893[:, j893]
        for j994 in range(994):
            if np.array_equal(col_893, mg750[:, j994]):
                col_mapping_893_to_994.append(j994)
                break
    
    print(f"  Column mapping: {len(col_mapping_893_to_994)} of 893 mapped to 994")
    
    # Find which 994 columns are missing
    mapped_994_set = set(col_mapping_893_to_994)
    missing_994 = sorted([j for j in range(994) if j not in mapped_994_set])
    print(f"  Missing SE indices: {len(missing_994)} side effects to add")
    
    # --- Expand frequency matrix (750, 893) -> (750, 994) ---
    # MGPred mg750 has frequency values 0-5, MVSL uses continuous [0,1]
    # Check if MVSL uses same scale or normalized
    max_val_893 = freq_893.max()
    print(f"  MVSL freq max value: {max_val_893} (1.0=normalized, 5.0=raw)")
    
    freq_994 = np.zeros((750, 994), dtype=freq_893.dtype)
    for j893, j994 in enumerate(col_mapping_893_to_994):
        freq_994[:, j994] = freq_893[:, j893]
    
    # For missing columns: use mg750 values (normalize if MVSL uses [0,1])
    if max_val_893 <= 1.0:
        # MVSL uses normalized frequency
        for j994 in missing_994:
            freq_994[:, j994] = mg750[:, j994] / 5.0  # Normalize to [0,1]
    else:
        for j994 in missing_994:
            freq_994[:, j994] = mg750[:, j994]
    
    save_pkl(freq_994, os.path.join(OUTPUT_MVSL, 'drug_side_frequency_matrix.pkl'))
    
    # --- Expand association matrix (750, 893) -> (750, 994) ---
    assoc_994 = np.zeros((750, 994), dtype=assoc_893.dtype)
    for j893, j994 in enumerate(col_mapping_893_to_994):
        assoc_994[:, j994] = assoc_893[:, j893]
    for j994 in missing_994:
        assoc_994[:, j994] = (mg750[:, j994] > 0).astype(assoc_893.dtype)
    
    save_pkl(assoc_994, os.path.join(OUTPUT_MVSL, 'drug_side_association_matrix.pkl'))
    
    # --- Expand side_vector_level_123 (893, 243) -> (994, 243) ---
    # For missing SEs, use average of existing vectors (heuristic)
    side_vec_994 = np.zeros((994, 243), dtype=side_vec_893.dtype)
    for j893, j994 in enumerate(col_mapping_893_to_994):
        side_vec_994[j994] = side_vec_893[j893]
    
    # Try to get MedDRA vectors from DSGAT side_effect_label if available
    se_label_path = os.path.join(SHARED_DIR, 'side_effect_label_750.mat')
    if os.path.exists(se_label_path):
        se_label_mat = sio.loadmat(se_label_path)
        for key, val in se_label_mat.items():
            if not key.startswith('_') and hasattr(val, 'shape'):
                if val.shape[0] == 994 or val.shape[1] == 994:
                    print(f"  Found SE label data: {key} {val.shape}")
    
    # For missing SEs without MedDRA vectors: use mean of existing
    mean_vec = side_vec_893.mean(axis=0)
    for j994 in missing_994:
        side_vec_994[j994] = mean_vec
    
    save_pkl(side_vec_994, os.path.join(OUTPUT_MVSL, 'side_vector_level_123.pkl'))
    
    # --- Copy drug_smiles.pkl unchanged ---
    save_pkl(drug_smiles, os.path.join(OUTPUT_MVSL, 'drug_smiles.pkl'))
    
    # --- Rebuild final_sample.pkl for 994 SEs ---
    print("\n  Rebuilding final_sample.pkl...")
    positive_samples = []
    negative_samples = []
    
    for i in range(750):
        for j in range(994):
            if assoc_994[i, j] == 1:
                positive_samples.append([i, j, 1, freq_994[i, j]])
            
    # Sample negatives (same count as positives, random)
    np.random.seed(42)
    n_pos = len(positive_samples)
    neg_indices = np.where(assoc_994 == 0)
    neg_pairs = list(zip(neg_indices[0], neg_indices[1]))
    sampled_neg = np.random.choice(len(neg_pairs), size=min(n_pos, len(neg_pairs)), replace=False)
    
    for idx in sampled_neg:
        i, j = neg_pairs[idx]
        negative_samples.append([i, j, 0, 0.0])
    
    final_sample = np.array(positive_samples + negative_samples)
    np.random.shuffle(final_sample)
    
    save_pkl(final_sample, os.path.join(OUTPUT_MVSL, 'final_sample.pkl'))
    print(f"  final_sample: {final_sample.shape} (pos={len(positive_samples)}, neg={len(negative_samples)})")
    
    # Save column mapping for reference
    np.save(os.path.join(OUTPUT_MVSL, 'col_mapping_893_to_994.npy'), np.array(col_mapping_893_to_994))
    
    print("\n  ✅ MVSL-DSF data expansion complete!")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("DSE Unified Data Preparation Script")
    print("Target: 750 drugs × 994 side effects")
    print("=" * 60)
    
    # Step 1: Map drugs
    mapping, unmatched, mg750 = find_drug_mapping()
    
    # Step 2: Convert SDPred/MSSF
    convert_sdpred_mssf_data(mapping, unmatched, mg750)
    
    # Step 3: Expand MVSL-DSF
    convert_mvsl_data(mg750)
    
    print("\n" + "=" * 60)
    print("ALL DATA PREPARATION COMPLETE!")
    print("=" * 60)
    print(f"\nOutput directories:")
    print(f"  SDPred: {OUTPUT_SDPRED}/")
    print(f"  MSSF:   {OUTPUT_MSSF}/")
    print(f"  MVSL:   {OUTPUT_MVSL}/")
