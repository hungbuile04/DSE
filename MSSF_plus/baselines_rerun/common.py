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


def fit_ordinal_thresholds_on_valsel(
    y_true_freq_1_to_5: np.ndarray,
    y_pred_float: np.ndarray,
) -> np.ndarray:
    """
    Select 4 strictly monotonic thresholds `t1 < t2 < t3 < t4` on `val_select` to maximize
    5-class `MacroF1` when discretizing continuous predictions via `np.digitize(pred, [t1, t2, t3, t4])`.
    Always includes the standard rounding thresholds `[1.5, 2.5, 3.5, 4.5]` in the candidate pool.
    """
    yt = np.clip(np.round(np.asarray(y_true_freq_1_to_5, dtype=np.float64).flatten()), 1, 5).astype(np.int64) - 1
    yp = np.asarray(y_pred_float, dtype=np.float64).flatten()
    n = len(yt)
    default_thr = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float64)
    if n < 10 or np.std(yp) < 1e-12:
        return default_thr

    order = np.argsort(yp, kind='mergesort')
    yp_s = yp[order]
    yt_s = yt[order]

    # Only allow cuts where yp_s[idx - 1] < yp_s[idx] so thresholds are strictly well-defined
    valid_cuts = np.where(yp_s[1:] > yp_s[:-1] + 1e-12)[0] + 1
    if len(valid_cuts) < 4:
        return default_thr

    pref = np.zeros((n + 1, 5), dtype=np.int32)
    for c in range(5):
        pref[1:, c] = np.cumsum(yt_s == c, dtype=np.int32)
    tot_true = pref[n, :].astype(np.float64)

    def _macro_f1_for_cuts(i1: int, i2: int, i3: int, i4: int) -> float:
        cuts = (0, i1, i2, i3, i4, n)
        f1_sum = 0.0
        for c in range(5):
            tp = float(pref[cuts[c + 1], c] - pref[cuts[c], c])
            p_cnt = float(cuts[c + 1] - cuts[c])
            denom = p_cnt + tot_true[c]
            if denom > 0.0:
                f1_sum += (2.0 * tp) / denom
        return f1_sum / 5.0

    # Candidate cuts: quantiles + empirical class proportions + default [1.5, 2.5, 3.5, 4.5]
    q_idx = np.linspace(0, len(valid_cuts) - 1, min(55, len(valid_cuts))).round().astype(int)
    cand_set = set(valid_cuts[q_idx].tolist())
    for t_def in default_thr:
        pos = int(np.searchsorted(yp_s, t_def, side='right'))
        nearest = valid_cuts[np.argmin(np.abs(valid_cuts - pos))]
        cand_set.add(int(nearest))
    cand = np.array(sorted(cand_set), dtype=np.int32)

    m = len(cand)
    best_cuts = None
    best_score = -1.0
    if m >= 4:
        from itertools import combinations
        combos = np.array(list(combinations(cand, 4)), dtype=np.int32)
        c1, c2, c3, c4 = combos[:, 0], combos[:, 1], combos[:, 2], combos[:, 3]
        tp0 = (pref[c1, 0]).astype(np.float64)
        tp1 = (pref[c2, 1] - pref[c1, 1]).astype(np.float64)
        tp2 = (pref[c3, 2] - pref[c2, 2]).astype(np.float64)
        tp3 = (pref[c4, 3] - pref[c3, 3]).astype(np.float64)
        tp4 = (pref[n, 4] - pref[c4, 4]).astype(np.float64)

        p0 = c1.astype(np.float64)
        p1 = (c2 - c1).astype(np.float64)
        p2 = (c3 - c2).astype(np.float64)
        p3 = (c4 - c3).astype(np.float64)
        p4 = (n - c4).astype(np.float64)

        f0 = np.where(p0 + tot_true[0] > 0, 2.0 * tp0 / np.maximum(p0 + tot_true[0], 1.0), 0.0)
        f1 = np.where(p1 + tot_true[1] > 0, 2.0 * tp1 / np.maximum(p1 + tot_true[1], 1.0), 0.0)
        f2 = np.where(p2 + tot_true[2] > 0, 2.0 * tp2 / np.maximum(p2 + tot_true[2], 1.0), 0.0)
        f3 = np.where(p3 + tot_true[3] > 0, 2.0 * tp3 / np.maximum(p3 + tot_true[3], 1.0), 0.0)
        f4 = np.where(p4 + tot_true[4] > 0, 2.0 * tp4 / np.maximum(p4 + tot_true[4], 1.0), 0.0)
        scores = (f0 + f1 + f2 + f3 + f4) / 5.0
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_cuts = [int(x) for x in combos[best_idx]]

        # Coordinate polish over all valid_cuts
        for _ in range(3):
            improved = False
            for k_pos in range(4):
                lo = 0 if k_pos == 0 else best_cuts[k_pos - 1]
                hi = n if k_pos == 3 else best_cuts[k_pos + 1]
                sub_valid = valid_cuts[(valid_cuts > lo) & (valid_cuts < hi)]
                for cand_i in sub_valid:
                    trial = list(best_cuts)
                    trial[k_pos] = int(cand_i)
                    sc = _macro_f1_for_cuts(trial[0], trial[1], trial[2], trial[3])
                    if sc > best_score + 1e-12:
                        best_score = sc
                        best_cuts = trial
                        improved = True
            if not improved:
                break

    if best_cuts is None:
        return default_thr

    thr = np.array(
        [0.5 * (yp_s[i - 1] + yp_s[i]) for i in best_cuts],
        dtype=np.float64,
    )
    # Verify monotonic and compare against default [1.5, 2.5, 3.5, 4.5]
    pred_def = np.clip(np.round(yp), 1, 5).astype(np.int64) - 1
    f1_def = float(compute_multiclass_metrics(yt, pred_def, probs=None)['MacroF1'])
    pred_thr = np.digitize(yp, thr).astype(np.int64)
    f1_thr = float(compute_multiclass_metrics(yt, pred_thr, probs=None)['MacroF1'])
    if f1_thr < f1_def or not np.all(np.diff(thr) > 0):
        return default_thr
    return thr


def apply_ordinal_thresholds_to_freq(
    y_pred_float: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """
    Map continuous predictions to discrete ordinal frequencies `{1.0, 2.0, 3.0, 4.0, 5.0}`
    using `thresholds = [t1, t2, t3, t4]` selected on `val_select`.
    """
    yp = np.asarray(y_pred_float, dtype=np.float64).flatten()
    thr = np.asarray(thresholds, dtype=np.float64).flatten()
    assert len(thr) == 4 and np.all(np.diff(thr) > 0), f"Thresholds must be strictly increasing: {thr}"
    return (np.digitize(yp, thr) + 1).astype(np.float32)


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
    eval_test: bool = False,
) -> Optional[dict]:
    """
    Check if `fold` is already completed locally or in `drive_dir`. If found in `drive_dir`,
    restore files locally and return the loaded `fold_{k}_summary.json`.
    """
    primary_folder = f'{method_name}_{split_type}'
    primary_dir = os.path.join(results_dir, primary_folder)
    req_files = [
        f'fold_{fold}_valsel_labels.npy',
        f'fold_{fold}_valsel_preds.npy',
        f'fold_{fold}_summary.json',
    ]
    if eval_test:
        req_files.extend([
            f'fold_{fold}_labels.npy',
            f'fold_{fold}_preds.npy',
        ])

    thr_folder = f'{method_name}_thr_{split_type}'
    thr_dir = os.path.join(results_dir, thr_folder)
    folders = [(primary_folder, primary_dir), (thr_folder, thr_dir)]
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
        sum_path = os.path.join(primary_dir, f'fold_{fold}_summary.json')
        with open(sum_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
        if 'best_train_macro_f1' in summary_data:
            # Ensure _thr folder and valsel_thr_metrics exist
            if not all(os.path.exists(os.path.join(thr_dir, f)) for f in req_files[:3]) or 'valsel_thr_metrics' not in summary_data:
                vs_labels = np.load(os.path.join(primary_dir, f'fold_{fold}_valsel_labels.npy')).flatten()
                vs_preds = np.load(os.path.join(primary_dir, f'fold_{fold}_valsel_preds.npy')).flatten()
                thr = fit_ordinal_thresholds_on_valsel(vs_labels, vs_preds)
                vs_preds_thr = apply_ordinal_thresholds_to_freq(vs_preds, thr)
                valsel_thr_metrics = eval_regression_predictions(vs_labels, vs_preds_thr)
                summary_data['valsel_thresholds'] = [float(x) for x in thr]
                summary_data['valsel_thr_metrics'] = valsel_thr_metrics
                os.makedirs(thr_dir, exist_ok=True)
                np.save(os.path.join(thr_dir, f'fold_{fold}_valsel_labels.npy'), vs_labels)
                np.save(os.path.join(thr_dir, f'fold_{fold}_valsel_preds.npy'), vs_preds_thr)
                thr_summary = dict(summary_data)
                thr_summary['method_name'] = f'{method_name}_thr'
                thr_summary['valsel_metrics'] = valsel_thr_metrics
                thr_summary['train_valsel_macro_f1_gap'] = float(
                    float(summary_data['best_train_macro_f1']) - float(valsel_thr_metrics['MacroF1'])
                )
                with open(sum_path, 'w', encoding='utf-8') as f:
                    json.dump(summary_data, f, indent=2)
                with open(os.path.join(thr_dir, f'fold_{fold}_summary.json'), 'w', encoding='utf-8') as f:
                    json.dump(thr_summary, f, indent=2)
                sync_dir_to_drive(primary_dir, drive_dir, primary_folder)
                sync_dir_to_drive(thr_dir, drive_dir, thr_folder)
            return summary_data
    return None


def save_baseline_fold_outputs(
    method_name: str,
    split_type: str,
    fold: int,
    test_split: np.ndarray,
    test_preds_float: Optional[np.ndarray],
    fold_summary: dict,
    results_dir: str = DEFAULT_RESULTS_V2_DIR,
    drive_dir: Optional[str] = None,
    rmse_sel_preds_float: Optional[np.ndarray] = None,
    valsel_split: Optional[np.ndarray] = None,
    valsel_preds_float: Optional[np.ndarray] = None,
    rmse_sel_valsel_preds_float: Optional[np.ndarray] = None,
    eval_test: bool = False,
) -> None:
    """
    Save primary (`MacroF1`-selected) and optional secondary (`RMSE`-selected) fold outputs.
    Always saves `fold_{k}_valsel_labels.npy` and `fold_{k}_valsel_preds.npy`.
    Only saves `fold_{k}_labels.npy` and `fold_{k}_preds.npy` when `eval_test=True` (Stage 7).
    """
    primary_folder = f'{method_name}_{split_type}'
    primary_dir = os.path.join(results_dir, primary_folder)
    os.makedirs(primary_dir, exist_ok=True)

    thr_folder = f'{method_name}_thr_{split_type}'
    thr_dir = os.path.join(results_dir, thr_folder)
    thr_summary = None

    if valsel_split is not None and valsel_preds_float is not None:
        vs_labels = np.asarray(valsel_split[:, 2], dtype=np.float64).flatten()
        vs_preds = np.asarray(valsel_preds_float, dtype=np.float32).flatten()
        assert len(vs_preds) == len(vs_labels), (
            f"{method_name} Split {split_type} Fold {fold}: valsel pred length {len(vs_preds)} != {len(vs_labels)}"
        )
        np.save(os.path.join(primary_dir, f'fold_{fold}_valsel_labels.npy'), vs_labels)
        np.save(os.path.join(primary_dir, f'fold_{fold}_valsel_preds.npy'), vs_preds)

        # Fit 4 monotonic thresholds t1 < t2 < t3 < t4 on val_select and apply unchanged to val_select & test
        thr = fit_ordinal_thresholds_on_valsel(vs_labels, vs_preds)
        vs_preds_thr = apply_ordinal_thresholds_to_freq(vs_preds, thr)
        valsel_thr_metrics = eval_regression_predictions(vs_labels, vs_preds_thr)
        fold_summary['valsel_thresholds'] = [float(x) for x in thr]
        fold_summary['valsel_thr_metrics'] = valsel_thr_metrics

        os.makedirs(thr_dir, exist_ok=True)
        np.save(os.path.join(thr_dir, f'fold_{fold}_valsel_labels.npy'), vs_labels)
        np.save(os.path.join(thr_dir, f'fold_{fold}_valsel_preds.npy'), vs_preds_thr)

        thr_summary = dict(fold_summary)
        thr_summary['method_name'] = f'{method_name}_thr'
        thr_summary['valsel_metrics'] = valsel_thr_metrics
        if 'best_train_macro_f1' in fold_summary:
            thr_summary['train_valsel_macro_f1_gap'] = float(
                float(fold_summary['best_train_macro_f1']) - float(valsel_thr_metrics['MacroF1'])
            )

    if eval_test and test_preds_float is not None:
        expected_labels = np.asarray(test_split[:, 2], dtype=np.float64).flatten()
        preds_primary = np.asarray(test_preds_float, dtype=np.float32).flatten()
        if len(preds_primary) != len(expected_labels):
            raise AssertionError(
                f"{method_name} Split {split_type} Fold {fold}: pred length {len(preds_primary)} "
                f"!= test_split length {len(expected_labels)}"
            )
        np.save(os.path.join(primary_dir, f'fold_{fold}_labels.npy'), expected_labels)
        np.save(os.path.join(primary_dir, f'fold_{fold}_preds.npy'), preds_primary)

        if thr_summary is not None and 'valsel_thresholds' in fold_summary:
            thr_arr = np.asarray(fold_summary['valsel_thresholds'], dtype=np.float64)
            preds_test_thr = apply_ordinal_thresholds_to_freq(preds_primary, thr_arr)
            np.save(os.path.join(thr_dir, f'fold_{fold}_labels.npy'), expected_labels)
            np.save(os.path.join(thr_dir, f'fold_{fold}_preds.npy'), preds_test_thr)
            test_thr_metrics = eval_regression_predictions(expected_labels, preds_test_thr)
            fold_summary['test_thr_metrics'] = test_thr_metrics
            thr_summary['test_metrics'] = test_thr_metrics

    with open(os.path.join(primary_dir, f'fold_{fold}_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(fold_summary, f, indent=2)
    sync_dir_to_drive(primary_dir, drive_dir, primary_folder)

    if thr_summary is not None:
        with open(os.path.join(thr_dir, f'fold_{fold}_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(thr_summary, f, indent=2)
        sync_dir_to_drive(thr_dir, drive_dir, thr_folder)

    if rmse_sel_preds_float is not None or rmse_sel_valsel_preds_float is not None:
        rmse_folder = f'{method_name}_rmseSel_{split_type}'
        rmse_dir = os.path.join(results_dir, rmse_folder)
        os.makedirs(rmse_dir, exist_ok=True)
        if valsel_split is not None and rmse_sel_valsel_preds_float is not None:
            vs_labels = np.asarray(valsel_split[:, 2], dtype=np.float64).flatten()
            vs_rmse = np.asarray(rmse_sel_valsel_preds_float, dtype=np.float32).flatten()
            np.save(os.path.join(rmse_dir, f'fold_{fold}_valsel_labels.npy'), vs_labels)
            np.save(os.path.join(rmse_dir, f'fold_{fold}_valsel_preds.npy'), vs_rmse)
        if eval_test and rmse_sel_preds_float is not None:
            expected_labels = np.asarray(test_split[:, 2], dtype=np.float64).flatten()
            preds_rmse = np.asarray(rmse_sel_preds_float, dtype=np.float32).flatten()
            if len(preds_rmse) != len(expected_labels):
                raise AssertionError(
                    f"{method_name}_rmseSel Split {split_type} Fold {fold}: pred length {len(preds_rmse)} "
                    f"!= test_split length {len(expected_labels)}"
                )
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
    valsel_mean, valsel_std = aggregate_fold_metrics([s['valsel_metrics'] for s in fold_summaries])
    has_test = all(s.get('test_metrics') is not None for s in fold_summaries)
    if has_test:
        test_mean, test_std = aggregate_fold_metrics([s['test_metrics'] for s in fold_summaries])
        test_summary = {'mean': test_mean, 'std': test_std}
    else:
        test_mean, test_std = None, None
        test_summary = None

    exp_summary = {
        'method_name': method_name,
        'split_type': split_type,
        'mode': mode,
        'folds': folds,
        'config': cfg,
        'valsel_summary': {'mean': valsel_mean, 'std': valsel_std},
        'test_summary': test_summary,
        'fold_summaries': fold_summaries,
    }

    if all(s.get('valsel_thr_metrics') is not None for s in fold_summaries):
        thr_vm, thr_vs = aggregate_fold_metrics([s['valsel_thr_metrics'] for s in fold_summaries])
        exp_summary['valsel_thr_summary'] = {'mean': thr_vm, 'std': thr_vs}
        thr_test_summary = None
        if all(s.get('test_thr_metrics') is not None for s in fold_summaries):
            thr_tm, thr_ts = aggregate_fold_metrics([s['test_thr_metrics'] for s in fold_summaries])
            thr_test_summary = {'mean': thr_tm, 'std': thr_ts}
            exp_summary['test_thr_summary'] = thr_test_summary

        thr_folder = f'{method_name}_thr_{split_type}'
        thr_dir = os.path.join(results_dir, thr_folder)
        os.makedirs(thr_dir, exist_ok=True)
        thr_cfg = dict(cfg, discretization='val_select_monotonic_thresholds')
        thr_exp_summary = {
            'method_name': f'{method_name}_thr',
            'split_type': split_type,
            'mode': mode,
            'folds': folds,
            'config': thr_cfg,
            'valsel_summary': {'mean': thr_vm, 'std': thr_vs},
            'test_summary': thr_test_summary,
            'fold_summaries': fold_summaries,
        }
        save_experiment_config(method_dir=thr_dir, config=thr_cfg, summary=thr_exp_summary)
        with open(os.path.join(thr_dir, f'summary_{mode}.json'), 'w', encoding='utf-8') as f:
            json.dump(thr_exp_summary, f, indent=2)
        sync_dir_to_drive(thr_dir, drive_dir, thr_folder)

    if has_rmse_sel:
        if all(s.get('rmse_sel_valsel_metrics') is not None for s in fold_summaries):
            rm_vm, rm_vs = aggregate_fold_metrics([s['rmse_sel_valsel_metrics'] for s in fold_summaries])
            exp_summary['rmse_sel_valsel_summary'] = {'mean': rm_vm, 'std': rm_vs}
        if all(s.get('rmse_sel_test_metrics') is not None for s in fold_summaries):
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
        rep_m = test_mean if test_mean is not None else valsel_mean
        rep_s = test_std if test_std is not None else valsel_std
        rep_tag = "Test" if test_mean is not None else "val_select (test locked)"
        print(
            f"=== Summary {method_name} (Split {split_type}, {len(folds)} folds, {rep_tag}, MacroF1-selected) ===\n"
            f"  MacroF1    : {rep_m['MacroF1']:.4f} +/- {rep_s['MacroF1']:.4f}\n"
            f"  WeightedF1 : {rep_m['WeightedF1']:.4f} +/- {rep_s['WeightedF1']:.4f}\n"
            f"  Accuracy   : {rep_m['Accuracy']:.4f} +/- {rep_s['Accuracy']:.4f}\n"
            f"  QWK        : {rep_m['QWK']:.4f} +/- {rep_s['QWK']:.4f}\n"
            f"  MAE        : {rep_m['MAE']:.4f} +/- {rep_s['MAE']:.4f}\n"
            f"  RMSE       : {rep_m['RMSE']:.4f} +/- {rep_s['RMSE']:.4f}\n"
            f"  Saved to   : {summary_path}"
        )
        if 'valsel_thr_summary' in exp_summary:
            tm_d = exp_summary.get('test_thr_summary') or exp_summary['valsel_thr_summary']
            tm_m, tm_s = tm_d['mean'], tm_d['std']
            print(
                f"=== Summary {method_name}_thr (Split {split_type}, {len(folds)} folds, {rep_tag}, 4-threshold calibrated) ===\n"
                f"  MacroF1    : {tm_m['MacroF1']:.4f} +/- {tm_s['MacroF1']:.4f}\n"
                f"  WeightedF1 : {tm_m['WeightedF1']:.4f} +/- {tm_s['WeightedF1']:.4f}\n"
                f"  Accuracy   : {tm_m['Accuracy']:.4f} +/- {tm_s['Accuracy']:.4f}\n"
                f"  QWK        : {tm_m['QWK']:.4f} +/- {tm_s['QWK']:.4f}\n"
                f"  MAE        : {tm_m['MAE']:.4f} +/- {tm_s['MAE']:.4f}"
            )
    return exp_summary
