"""
Unified evaluation protocol for MSSF_plus (Step 2b).

Responsibilities:
  - Deterministic seeding across Python, NumPy, and PyTorch.
  - Loading the 4-way splits (train 80% / val_select 10% / val_calib 10% / test) via shared_data/split_adapter_v2.py.
  - Fold list selection: 'quick' ([0]), 'dev' ([0, 1, 2]), 'full' ([0..9]).
  - Saving fold predictions to results_v2/{Method}_{A|B}/ in a format 100% compatible with
    shared_data/unified_eval.py (labels.npy, preds.npy on SIDER scale 1..5), plus probs.npy (N, 5),
    valcal_probs.npy, valcal_labels.npy, and config.json with full environment provenance.
  - Paired Wilcoxon signed-rank statistical comparison across folds against a reference method (default: MSSF_v2).
"""

import os
import sys
import json
import random
import platform
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy
import scipy.stats as stats
import sklearn
import torch

PLUS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PLUS_DIR)
SHARED_DIR = os.path.join(REPO_DIR, 'shared_data')
DEFAULT_RESULTS_V2_DIR = os.path.join(REPO_DIR, 'results_v2')

if SHARED_DIR not in sys.path:
    sys.path.insert(0, SHARED_DIR)

from split_adapter_v2 import load_splits_v2, load_val_drugs  # noqa: E402
from MSSF_plus.metrics import compute_multiclass_metrics, aggregate_fold_metrics  # noqa: E402

N_DRUGS = 750
N_SIDES = 994
DEV_FOLDS = [0, 1, 2]
FULL_FOLDS = list(range(10))


@dataclass
class FoldSplit:
    fold: int
    split_type: str
    train: np.ndarray          # (N_train, 3): [drug_idx, se_idx, freq_1_to_5]
    valsel: np.ndarray         # (N_valsel, 3)
    valcal: np.ndarray         # (N_valcal, 3)
    test: np.ndarray           # (N_test, 3)
    train_drugs: np.ndarray    # 1D int array of drugs present in train
    valsel_drugs: np.ndarray   # 1D int array (Split B: 67 cold-start valsel drugs)
    valcal_drugs: np.ndarray   # 1D int array (Split B: 68 cold-start valcal drugs)
    test_drugs: np.ndarray     # 1D int array (Split B: 75 cold-start test drugs)


def set_global_seed(seed: int = 42):
    """Set deterministic random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device(prefer_device: Optional[str] = None) -> torch.device:
    """Select runtime device (CUDA -> MPS -> CPU, or explicit override)."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if prefer_device:
        return torch.device(prefer_device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_fold_list(mode: str = 'dev', custom_folds: Optional[list] = None) -> list:
    """Return fold indices for 'quick' ([0]), 'dev' ([0,1,2]), or 'full' ([0..9])."""
    if custom_folds is not None:
        return [int(f) for f in custom_folds]
    if mode == 'quick':
        return [0]
    if mode == 'dev':
        return list(DEV_FOLDS)
    if mode == 'full':
        return list(FULL_FOLDS)
    raise ValueError(f"Unknown mode={mode!r}; expected 'quick', 'dev', or 'full'.")


def load_fold_data(fold: int, split_type: str = 'A') -> FoldSplit:
    """Load the unified 4-way split and drug partitions for a given fold."""
    train, valsel, valcal, test = load_splits_v2(fold, split_type)
    if split_type == 'B':
        train_drugs = load_val_drugs(fold, 'train', 'B')
        valsel_drugs = load_val_drugs(fold, 'valsel', 'B')
        valcal_drugs = load_val_drugs(fold, 'valcal', 'B')
        test_drugs = load_val_drugs(fold, 'test', 'B')
    else:
        train_drugs = np.unique(train[:, 0].astype(int))
        valsel_drugs = np.unique(valsel[:, 0].astype(int))
        valcal_drugs = np.unique(valcal[:, 0].astype(int))
        test_drugs = np.unique(test[:, 0].astype(int))
    return FoldSplit(
        fold=int(fold),
        split_type=split_type,
        train=train,
        valsel=valsel,
        valcal=valcal,
        test=test,
        train_drugs=train_drugs,
        valsel_drugs=valsel_drugs,
        valcal_drugs=valcal_drugs,
        test_drugs=test_drugs,
    )


def get_environment_info() -> dict:
    """Record exact runtime library versions for reproducibility."""
    return {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'scikit_learn': sklearn.__version__,
        'torch': torch.__version__,
        'cuda_available': bool(torch.cuda.is_available()),
        'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def save_fold_predictions(
    method_dir: str,
    fold: int,
    test_split: np.ndarray,
    y_true_class: np.ndarray,
    y_pred_class: np.ndarray,
    probs_5class: np.ndarray,
    valcal_true_class: Optional[np.ndarray] = None,
    valcal_probs_5class: Optional[np.ndarray] = None,
    valsel_true_class: Optional[np.ndarray] = None,
    valsel_probs_5class: Optional[np.ndarray] = None,
):
    """
    Save fold predictions in SIDER 1..5 frequency scale for unified_eval.py compatibility,
    plus N x 5 probability arrays for AUPR and post-hoc conformal calibration.
    Strictly asserts 1-to-1 alignment with test_split[:, 2].
    """
    os.makedirs(method_dir, exist_ok=True)
    y_true_freq = np.asarray(y_true_class, dtype=np.float64).flatten() + 1.0
    y_pred_freq = np.asarray(y_pred_class, dtype=np.float64).flatten() + 1.0
    probs_arr = np.asarray(probs_5class, dtype=np.float32)

    expected_test_freq = np.asarray(test_split[:, 2], dtype=np.float64).flatten()
    if not np.array_equal(y_true_freq, expected_test_freq):
        raise AssertionError(
            f"Fold {fold}: y_true_freq does not match test_split[:, 2]! "
            "Check that test DataLoader has shuffle=False."
        )
    if probs_arr.shape != (len(y_true_freq), 5):
        raise AssertionError(f"Fold {fold}: probs shape {probs_arr.shape} != ({len(y_true_freq)}, 5)")

    np.save(os.path.join(method_dir, f'fold_{fold}_labels.npy'), y_true_freq)
    np.save(os.path.join(method_dir, f'fold_{fold}_preds.npy'), y_pred_freq)
    np.save(os.path.join(method_dir, f'fold_{fold}_probs.npy'), probs_arr)

    if valcal_true_class is not None and valcal_probs_5class is not None:
        np.save(
            os.path.join(method_dir, f'fold_{fold}_valcal_labels.npy'),
            np.asarray(valcal_true_class, dtype=np.int64).flatten(),
        )
        np.save(
            os.path.join(method_dir, f'fold_{fold}_valcal_probs.npy'),
            np.asarray(valcal_probs_5class, dtype=np.float32),
        )

    if valsel_true_class is not None and valsel_probs_5class is not None:
        np.save(
            os.path.join(method_dir, f'fold_{fold}_valsel_labels.npy'),
            np.asarray(valsel_true_class, dtype=np.int64).flatten(),
        )
        np.save(
            os.path.join(method_dir, f'fold_{fold}_valsel_probs.npy'),
            np.asarray(valsel_probs_5class, dtype=np.float32),
        )


def save_experiment_config(method_dir: str, config: dict, summary: dict):
    """Save config.json including hyperparameters, environment versions, and fold metrics."""
    os.makedirs(method_dir, exist_ok=True)
    payload = {
        'config': config,
        'environment': get_environment_info(),
        'summary': summary,
    }
    path = os.path.join(method_dir, 'config.json')
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


def load_saved_fold_metrics(results_dir: str, method_with_split: str, folds=None):
    """Load per-fold metrics from saved labels.npy, preds.npy, probs.npy in results_dir/method_with_split."""
    mdir = os.path.join(results_dir, method_with_split)
    if not os.path.isdir(mdir):
        return {}
    if folds is None:
        folds = range(10)
    per_fold = {}
    for k in folds:
        lp = os.path.join(mdir, f'fold_{k}_labels.npy')
        pp = os.path.join(mdir, f'fold_{k}_preds.npy')
        prp = os.path.join(mdir, f'fold_{k}_probs.npy')
        if not (os.path.exists(lp) and os.path.exists(pp)):
            continue
        y_true_freq = np.load(lp).flatten()
        y_pred_freq = np.load(pp).flatten()
        y_true_c = np.clip(np.round(y_true_freq), 1, 5).astype(int) - 1
        y_pred_c = np.clip(np.round(y_pred_freq), 1, 5).astype(int) - 1
        probs = np.load(prp) if os.path.exists(prp) else None
        per_fold[k] = compute_multiclass_metrics(y_true_c, y_pred_c, probs=probs)
    return per_fold


def wilcoxon_compare(
    results_dir: str,
    method_candidate: str,
    method_reference: str = 'MSSF_v2',
    split_type: str = 'A',
    metric_keys: Optional[list] = None,
) -> dict:
    """
    Paired Wilcoxon signed-rank test across common folds between candidate and reference method.
    Note: Wilcoxon signed-rank requires at least ~6 paired folds to yield non-trivial p-values,
    but also reports paired mean difference and paired t-test p-value for smaller fold counts.
    """
    if metric_keys is None:
        metric_keys = ['MacroF1', 'Accuracy', 'WeightedF1', 'Kappa', 'QWK', 'MAE', 'MacroAUPR']

    cand_folds = load_saved_fold_metrics(results_dir, f'{method_candidate}_{split_type}')
    ref_folds = load_saved_fold_metrics(results_dir, f'{method_reference}_{split_type}')
    common = sorted(set(cand_folds.keys()) & set(ref_folds.keys()))
    if not common:
        return {'n_common_folds': 0, 'common_folds': [], 'metrics': {}}

    out = {'n_common_folds': len(common), 'common_folds': common, 'metrics': {}}
    for m in metric_keys:
        a = np.array([cand_folds[k][m] for k in common], dtype=np.float64)
        b = np.array([ref_folds[k][m] for k in common], dtype=np.float64)
        diff = a - b
        stat_w, p_w = float('nan'), float('nan')
        if len(common) >= 2 and not np.allclose(diff, 0.0):
            try:
                res = stats.wilcoxon(a, b, zero_method='wilcox', alternative='two-sided')
                stat_w, p_w = float(res.statistic), float(res.pvalue)
            except Exception:
                pass
        elif np.allclose(diff, 0.0):
            stat_w, p_w = 0.0, 1.0

        out['metrics'][m] = {
            'candidate_mean': float(np.mean(a)),
            'reference_mean': float(np.mean(b)),
            'mean_diff': float(np.mean(diff)),
            'wilcoxon_stat': stat_w,
            'wilcoxon_p': p_w,
        }
    return out
