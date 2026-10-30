"""
Shared protocol helpers for leak-free baseline reruns under Protocol v2.

Invariants enforced across all baseline wrappers:
  1. Original baseline directories, `shared_data/`, and `results/` are NEVER modified.
  2. Every fold loads `(train [80%], valsel [10%], valcal [10%], test [10%])` via `load_fold_data`.
  3. Any feature/graph/enrichment derived from drug-SE interactions is built strictly from
     `make_train_only_freq(fold_split.train)`.
  4. Primary checkpoint selection maximizes 5-class `MacroF1` on `valsel` (`val_select`)
     (after `np.clip(np.round(pred), 1, 5)` for regression outputs).
  5. For regression baselines, a secondary checkpoint minimizing `RMSE` on `valsel` is ALSO
     tracked and saved to `results_v2/{Method}_rmseSel_{A|B}/` for paper appendix reporting.
  6. Test evaluation runs strictly ONCE per fold with `shuffle=False`, asserting exact 1-to-1
     sample order alignment with `fold_split.test[:, 2]`.
"""

import os
import sys
import json
import shutil
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from MSSF_plus.protocol import (
    REPO_DIR,
    SHARED_DIR,
    DEFAULT_RESULTS_V2_DIR,
    N_DRUGS,
    N_SIDES,
    FoldSplit,
    save_experiment_config,
)
from MSSF_plus.metrics import compute_multiclass_metrics, aggregate_fold_metrics

if SHARED_DIR not in sys.path:
    sys.path.insert(0, SHARED_DIR)

from split_adapter_v2 import make_train_only_freq  # noqa: E402


def eval_regression_predictions(
    y_true_freq_1_to_5: np.ndarray,
    y_pred_float: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate continuous frequency predictions against ground-truth SIDER frequencies (1..5):
      - Continuous regression metrics: RMSE, RegMAE, Pearson, Spearman, R2
      - Discretized 5-class metrics after `np.clip(np.round(y_pred_float), 1, 5)`:
        MacroF1, WeightedF1, Accuracy, Kappa, QWK, MAE (discrete class MAE), F1_c0..c4
    """
    yt = np.asarray(y_true_freq_1_to_5, dtype=np.float64).flatten()
    yp = np.asarray(y_pred_float, dtype=np.float64).flatten()
    assert len(yt) == len(yp), f"Length mismatch: y_true={len(yt)} vs y_pred={len(yp)}"

    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    reg_mae = float(mean_absolute_error(yt, yp))
    r2 = float(r2_score(yt, yp)) if len(yt) > 1 else float('nan')

    if len(yt) > 1 and np.std(yp) > 1e-12 and np.std(yt) > 1e-12:
        pearson = float(stats.pearsonr(yt, yp)[0])
        spearman = float(stats.spearmanr(yt, yp)[0])
    else:
        pearson = 0.0
        spearman = 0.0

    yt_class = np.clip(np.round(yt), 1, 5).astype(np.int64) - 1
    yp_class = np.clip(np.round(yp), 1, 5).astype(np.int64) - 1
    cls_metrics = compute_multiclass_metrics(yt_class, yp_class, probs=None)

    out = dict(cls_metrics)
    out.update({
        'RMSE': rmse,
        'RegMAE': reg_mae,
        'Pearson': pearson,
        'Spearman': spearman,
        'R2': r2,
    })
    return out


def sample_negative_pairs_v2(
    drug_side_train: np.ndarray,
    fold_split: FoldSplit,
    n_samples: int,
    part: str = 'train',
    seed: int = 42,
) -> np.ndarray:
    """
    Leak-free negative pair sampling returning `(N, 2)` int64 array of `(drug_idx, se_idx)`.

    Rules:
      - `part == 'train'`:
        * Split A: samples strictly from `(drug_side_train == 0)` WITHOUT peeking at
          `valsel`, `valcal`, or `test` ground truth (preserving warm-start realism).
        * Split B: restricts candidate drugs strictly to `fold_split.train_drugs` (540 drugs)
          where `drug_side_train[d, s] == 0` (zero cold-start drug leakage).
      - `part in {'valsel', 'test'}` (for optional binary association AUC/AUPR logging):
        * Split A: samples from zero entries excluding known positives in `train` and `part`.
        * Split B: restricts candidate drugs strictly to `fold_split.{part}_drugs` (cold-start
          drugs only) excluding positives in `fold_split.{part}`.
    """
    rng = np.random.default_rng(seed)

    if fold_split.split_type == 'B':
        if part == 'train':
            allowed_drugs = np.asarray(fold_split.train_drugs, dtype=np.int64)
            sub_mat = drug_side_train[allowed_drugs]
            rel_d, s_idx = np.where(sub_mat == 0)
            candidates = np.column_stack([allowed_drugs[rel_d], s_idx])
        elif part in ('valsel', 'valcal', 'test'):
            part_arr = getattr(fold_split, part)
            allowed_drugs = np.asarray(getattr(fold_split, f'{part}_drugs'), dtype=np.int64)
            mask = np.zeros((N_DRUGS, N_SIDES), dtype=bool)
            mask[allowed_drugs, :] = True
            if len(part_arr) > 0:
                mask[part_arr[:, 0].astype(int), part_arr[:, 1].astype(int)] = False
            d_idx, s_idx = np.where(mask)
            candidates = np.column_stack([d_idx, s_idx])
        else:
            raise ValueError(f"Unknown part={part!r}")
    else:
        if part == 'train':
            d_idx, s_idx = np.where(drug_side_train == 0)
            candidates = np.column_stack([d_idx, s_idx])
        elif part in ('valsel', 'valcal', 'test'):
            part_arr = getattr(fold_split, part)
            mask = (drug_side_train == 0)
            if len(part_arr) > 0:
                mask = mask.copy()
                mask[part_arr[:, 0].astype(int), part_arr[:, 1].astype(int)] = False
            d_idx, s_idx = np.where(mask)
            candidates = np.column_stack([d_idx, s_idx])
        else:
            raise ValueError(f"Unknown part={part!r}")

    replace = len(candidates) < n_samples
    chosen = rng.choice(len(candidates), size=n_samples, replace=replace)
    return candidates[chosen].astype(np.int64)


def sync_dir_to_drive(local_dir: str, drive_dir: Optional[str], folder_name: str):
    """Mirror all files in `local_dir` to `drive_dir/folder_name` if `drive_dir` is set."""
    if not drive_dir or not os.path.isdir(local_dir):
        return
    dst_dir = os.path.join(drive_dir, folder_name)
    os.makedirs(dst_dir, exist_ok=True)
    for fname in os.listdir(local_dir):
        s_path = os.path.join(local_dir, fname)
        d_path = os.path.join(dst_dir, fname)
        if os.path.isfile(s_path) and os.path.abspath(s_path) != os.path.abspath(d_path):
            shutil.copy2(s_path, d_path)


def try_restore_baseline_fold(
    method_name: str,
    split_type: str,
    fold: int,
    results_dir: str,
    drive_dir: Optional[str] = None,
    has_rmse_sel: bool = True,
) -> Optional[dict]:
    """
    Check if `fold` is already completed locally or in `drive_dir`. If found in `drive_dir`,
    restore files locally and return the loaded `fold_{k}_summary.json`.
    """
    primary_folder = f'{method_name}_{split_type}'
    primary_dir = os.path.join(results_dir, primary_folder)
    req_files = [
        f'fold_{fold}_labels.npy',
        f'fold_{fold}_preds.npy',
        f'fold_{fold}_summary.json',
    ]

    folders = [(primary_folder, primary_dir)]
    if has_rmse_sel:
        rmse_folder = f'{method_name}_rmseSel_{split_type}'
        rmse_dir = os.path.join(results_dir, rmse_folder)
        folders.append((rmse_folder, rmse_dir))

    if drive_dir:
        for folder_name, local_d in folders:
            drive_d = os.path.join(drive_dir, folder_name)
            if os.path.isdir(drive_d):
                os.makedirs(local_d, exist_ok=True)
                for fname in req_files:
                    lp = os.path.join(local_d, fname)
                    dp = os.path.join(drive_d, fname)
                    if not os.path.exists(lp) and os.path.exists(dp):
                        shutil.copy2(dp, lp)

    if all(os.path.exists(os.path.join(primary_dir, f)) for f in req_files):
        if has_rmse_sel:
            rmse_dir = os.path.join(results_dir, f'{method_name}_rmseSel_{split_type}')
            if not all(os.path.exists(os.path.join(rmse_dir, f)) for f in req_files[:2]):
                return None
        with open(os.path.join(primary_dir, f'fold_{fold}_summary.json'), 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def save_baseline_fold_outputs(
    method_name: str,
    split_type: str,
    fold: int,
    test_split: np.ndarray,
    test_preds_float: np.ndarray,
    fold_summary: dict,
    results_dir: str = DEFAULT_RESULTS_V2_DIR,
    drive_dir: Optional[str] = None,
    rmse_sel_preds_float: Optional[np.ndarray] = None,
) -> None:
    """
    Save primary (`MacroF1`-selected) and optional secondary (`RMSE`-selected) fold outputs,
    asserting exact 1-to-1 sample alignment with `test_split[:, 2]`, and sync to `drive_dir`.
    """
    expected_labels = np.asarray(test_split[:, 2], dtype=np.float64).flatten()
    preds_primary = np.asarray(test_preds_float, dtype=np.float32).flatten()
    if len(preds_primary) != len(expected_labels):
        raise AssertionError(
            f"{method_name} Split {split_type} Fold {fold}: pred length {len(preds_primary)} "
            f"!= test_split length {len(expected_labels)}"
        )

    primary_folder = f'{method_name}_{split_type}'
    primary_dir = os.path.join(results_dir, primary_folder)
    os.makedirs(primary_dir, exist_ok=True)
    np.save(os.path.join(primary_dir, f'fold_{fold}_labels.npy'), expected_labels)
    np.save(os.path.join(primary_dir, f'fold_{fold}_preds.npy'), preds_primary)
    with open(os.path.join(primary_dir, f'fold_{fold}_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(fold_summary, f, indent=2)
    sync_dir_to_drive(primary_dir, drive_dir, primary_folder)

    if rmse_sel_preds_float is not None:
        preds_rmse = np.asarray(rmse_sel_preds_float, dtype=np.float32).flatten()
        if len(preds_rmse) != len(expected_labels):
            raise AssertionError(
                f"{method_name}_rmseSel Split {split_type} Fold {fold}: pred length {len(preds_rmse)} "
                f"!= test_split length {len(expected_labels)}"
            )
        rmse_folder = f'{method_name}_rmseSel_{split_type}'
        rmse_dir = os.path.join(results_dir, rmse_folder)
        os.makedirs(rmse_dir, exist_ok=True)
        np.save(os.path.join(rmse_dir, f'fold_{fold}_labels.npy'), expected_labels)
        np.save(os.path.join(rmse_dir, f'fold_{fold}_preds.npy'), preds_rmse)
        with open(os.path.join(rmse_dir, f'fold_{fold}_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(fold_summary, f, indent=2)
        sync_dir_to_drive(rmse_dir, drive_dir, rmse_folder)


def finalize_baseline_experiment(
    method_name: str,
    split_type: str,
    mode: str,
    folds: List[int],
    cfg: dict,
    fold_summaries: List[dict],
    results_dir: str = DEFAULT_RESULTS_V2_DIR,
    drive_dir: Optional[str] = None,
    has_rmse_sel: bool = True,
    verbose: bool = True,
) -> dict:
    """Aggregate fold summaries, save `config.json` & `summary_{mode}.json`, and sync to Drive."""
    test_mean, test_std = aggregate_fold_metrics([s['test_metrics'] for s in fold_summaries])
    valsel_mean, valsel_std = aggregate_fold_metrics([s['valsel_metrics'] for s in fold_summaries])

    exp_summary = {
        'method_name': method_name,
        'split_type': split_type,
        'mode': mode,
        'folds': folds,
        'config': cfg,
        'valsel_summary': {'mean': valsel_mean, 'std': valsel_std},
        'test_summary': {'mean': test_mean, 'std': test_std},
        'fold_summaries': fold_summaries,
    }

    if has_rmse_sel and all('rmse_sel_test_metrics' in s for s in fold_summaries):
        rm_mean, rm_std = aggregate_fold_metrics([s['rmse_sel_test_metrics'] for s in fold_summaries])
        exp_summary['rmse_sel_test_summary'] = {'mean': rm_mean, 'std': rm_std}

        rmse_folder = f'{method_name}_rmseSel_{split_type}'
        rmse_dir = os.path.join(results_dir, rmse_folder)
        rmse_cfg = dict(cfg, selection_metric='RMSE')
        save_experiment_config(method_dir=rmse_dir, config=rmse_cfg, summary=exp_summary)
        with open(os.path.join(rmse_dir, f'summary_{mode}.json'), 'w', encoding='utf-8') as f:
            json.dump(exp_summary, f, indent=2)
        sync_dir_to_drive(rmse_dir, drive_dir, rmse_folder)

    primary_folder = f'{method_name}_{split_type}'
    primary_dir = os.path.join(results_dir, primary_folder)
    save_experiment_config(method_dir=primary_dir, config=cfg, summary=exp_summary)
    summary_path = os.path.join(primary_dir, f'summary_{mode}.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(exp_summary, f, indent=2)
    sync_dir_to_drive(primary_dir, drive_dir, primary_folder)

    if verbose:
        print(
            f"=== Summary {method_name} (Split {split_type}, {len(folds)} folds, MacroF1-selected) ===\n"
            f"  MacroF1    : {test_mean['MacroF1']:.4f} +/- {test_std['MacroF1']:.4f}\n"
            f"  WeightedF1 : {test_mean['WeightedF1']:.4f} +/- {test_std['WeightedF1']:.4f}\n"
            f"  Accuracy   : {test_mean['Accuracy']:.4f} +/- {test_std['Accuracy']:.4f}\n"
            f"  QWK        : {test_mean['QWK']:.4f} +/- {test_std['QWK']:.4f}\n"
            f"  RMSE       : {test_mean['RMSE']:.4f} +/- {test_std['RMSE']:.4f}\n"
            f"  RegMAE     : {test_mean['RegMAE']:.4f} +/- {test_std['RegMAE']:.4f}\n"
            f"  Saved to   : {summary_path}"
        )
    return exp_summary
