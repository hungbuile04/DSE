"""
One-shot Stage 7 Held-Out `test` Evaluation & Paper Analysis Summarizer (`mssf_slim/summarize_stage7_test.py`).

Computes all publication tables directly from saved `.npy` and `.json` files inside `results_slim/`
without modifying any files outside `mssf_slim/` and `results_slim/`:
  0. Protocol Integrity & Deterministic `val_select` Parity Audit (comparing Stage 7 `val_select`
     against the locked Stage 6 `val_select` runs) + Parameter / Runtime / Peak GPU table.
  1. Main 10-Fold Held-Out `test` Benchmark Table (Split B & Split A, Folds `0..9`, K=10) with
     paired fold differences vs `slim_nodec`, Win/Tie/Loss counts, exact two-sided Wilcoxon
     signed-rank p-values, and Holm-Bonferroni step-down adjusted p-values (`p_Holm`).
     `HSTrans` / `HSTrans_thr` (Folds `0..2`, K=3) is strictly separated into a 3-fold footnote.
  2. 10-Fold Held-Out `test` Ablation & Leave-One-Out (LOO) Table (`slim_nodec`, `cand_A`,
     `slim_all_cnnshared`, `MSSF_v2_selfcorr`, and the 4 LOO configurations) + side-by-side
     `val_select` vs `test` consistency table.
  3. Held-Out `test` C4 Contiguous Ordinal Conformal Prediction (calibrated strictly on `val_calib`):
       3a. Main C4 table at pre-registered alpha = 0.10 (`LAC`, `APS`, `Ordinal_LAC`, `Ordinal_APS`).
       3b. Class-conditional coverage breakdown (`Cov_c0..Cov_c4`) at alpha = 0.10.
       3c. Post-hoc multi-alpha sensitivity curve (`alpha in {0.05, 0.10, 0.15, 0.20}`) for `slim_nodec`.
  4. Paper Interpretability, Ordinal Error Severity & Stratification Analyses on Held-Out `test`:
       4a. Exact-match, Adjacent-accuracy (`|y_hat - y| <= 1`), Severe-jump error rate (`|y_hat - y| >= 2`,
           `|y_hat - y| >= 3`), and 10-fold pooled row-normalized 5x5 confusion matrices.
       4b. Stratification by Side-Effect Training Support tertiles on Split B (Cold-Start Drugs) and
           by Drug Known-Interaction Degree tertiles on Split A (Warm-Start Pairs).
"""

import os
import sys
import glob
import json
import shutil
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from MSSF_plus.protocol import load_fold_data
from MSSF_plus.metrics import compute_multiclass_metrics, aggregate_fold_metrics
from MSSF_plus.conformal import evaluate_all_conformal_methods

from mssf_slim.configs import (
    ALL_REPORT_ORDER,
    DEFAULT_RESULTS_SLIM_DIR,
    FINAL_10FOLD_EXPERIMENTS,
    SLIM_EXPERIMENTS,
)
from mssf_slim.model import get_named_children_param_breakdown
from mssf_slim.summarize_results import (
    BASELINE_REPORT_ORDER,
    _instantiate_model_for_cfg_name,
    _is_valid_npy_file,
)

PRIMARY_10FOLD_THR_BASELINES = [
    'MGPred_selfcorr_thr',
    'SDPred_selfcorr_thr',
    'MVSL-DSF_selfcorr_thr',
    'DSGAT_thr',
    'A3Net_thr',
    # Fallbacks if non-selfcorr _thr was run instead
    'MGPred_thr',
    'SDPred_thr',
    'MVSL-DSF_thr',
]


def _is_valid_json_file(path: str) -> bool:
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False
    try:
        with open(path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        return isinstance(obj, dict)
    except Exception:
        return False


def _sync_valid_files_from_drive(
    local_root: str,
    drive_root: Optional[str],
    split_type: str,
    only_methods: Optional[List[str]] = None,
) -> None:
    """Safely mirror valid `.npy` and `.json` files from `drive_root/*_{split_type}` to `local_root`."""
    if not drive_root or not os.path.isdir(drive_root):
        return
    os.makedirs(local_root, exist_ok=True)
    pattern = os.path.join(drive_root, f'*_{split_type}')
    for src_mdir in sorted(glob.glob(pattern)):
        if not os.path.isdir(src_mdir):
            continue
        folder_name = os.path.basename(src_mdir)
        suffix = f'_{split_type}'
        if not folder_name.endswith(suffix):
            continue
        m_name = folder_name[:-len(suffix)]
        if m_name.endswith('_rmseSel'):
            continue
        if only_methods is not None and m_name not in only_methods:
            continue
        dst_mdir = os.path.join(local_root, folder_name)
        os.makedirs(dst_mdir, exist_ok=True)
        for fname in os.listdir(src_mdir):
            sp = os.path.join(src_mdir, fname)
            dp = os.path.join(dst_mdir, fname)
            if not os.path.isfile(sp):
                continue
            if fname.endswith('.npy') and not _is_valid_npy_file(sp):
                continue
            if fname.endswith('.json') and not _is_valid_json_file(sp):
                continue
            if os.path.exists(dp):
                if fname.endswith('.npy') and not _is_valid_npy_file(dp):
                    os.remove(dp)
                elif fname.endswith('.json') and not _is_valid_json_file(dp):
                    os.remove(dp)
            if not os.path.exists(dp):
                shutil.copy2(sp, dp)


def _paired_wilcoxon_p(diffs: List[float]) -> Optional[float]:
    arr = np.asarray(diffs, dtype=np.float64)
    if len(arr) < 6 or np.allclose(arr, 0.0):
        return None
    try:
        _, p_val = stats.wilcoxon(arr)
        return float(p_val)
    except Exception:
        return None


def _holm_bonferroni_adjust(raw_p_map: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    """
    Step-down Holm-Bonferroni multiple-testing correction over all non-None p-values in `raw_p_map`.
    """
    valid_items = sorted(
        [(k, float(v)) for k, v in raw_p_map.items() if v is not None],
        key=lambda kv: kv[1],
    )
    out: Dict[str, Optional[float]] = {k: None for k in raw_p_map}
    m = len(valid_items)
    if m == 0:
        return out
    running_max = 0.0
    for idx, (k, p_raw) in enumerate(valid_items):
        adj = (m - idx) * p_raw
        running_max = max(running_max, adj)
        out[k] = float(min(1.0, running_max))
    return out


def _win_tie_loss_str(cand_vals: List[float], ref_vals: List[float], higher_is_better: bool = True) -> str:
    """Count folds where `ref` (`slim_nodec`) wins / ties / loses against `cand`."""
    w, t, l = 0, 0, 0
    for c_v, r_v in zip(cand_vals, ref_vals):
        diff = (r_v - c_v) if higher_is_better else (c_v - r_v)
        if abs(diff) <= 1e-12:
            t += 1
        elif diff > 0:
            w += 1
        else:
            l += 1
    return f"{w}/{t}/{l}"


def collect_stage7_test_results(
    results_dir: str,
    split_type: str,
    only_folds: Optional[List[int]] = None,
    only_methods: Optional[List[str]] = None,
    drive_dir: Optional[str] = None,
    compute_conformal: bool = True,
    multi_alphas: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20),
) -> Dict[str, dict]:
    """
    Scan `results_dir/*_{split_type}` and evaluate held-out `test` (plus `val_select`) directly
    from saved `.npy` files, asserting 1-to-1 sample alignment with `load_fold_data(k, split_type).test`.
    """
    _sync_valid_files_from_drive(results_dir, drive_dir, split_type, only_methods=only_methods)
    pattern = os.path.join(results_dir, f'*_{split_type}')
    method_dirs = sorted(glob.glob(pattern))
    collected: Dict[str, dict] = {}

    for mdir in method_dirs:
        if not os.path.isdir(mdir):
            continue
        folder_name = os.path.basename(mdir)
        suffix = f'_{split_type}'
        if not folder_name.endswith(suffix):
            continue
        method_name = folder_name[:-len(suffix)]
        if method_name.endswith('_rmseSel'):
            continue
        if only_methods is not None and method_name not in only_methods:
            continue

        folds_found: List[int] = []
        fold_test_list: List[dict] = []
        fold_valsel_map: Dict[int, dict] = {}
        fold_y_true_map: Dict[int, np.ndarray] = {}
        fold_y_pred_map: Dict[int, np.ndarray] = {}
        fold_probs_map: Dict[int, np.ndarray] = {}
        fold_times: List[float] = []
        fold_peak_gpus: List[float] = []
        fold_gaps: List[float] = []
        total_params: Optional[int] = None

        c4_methods = ['LAC', 'APS', 'Ordinal_LAC', 'Ordinal_APS']
        fold_test_conformal: Dict[str, List[dict]] = {m: [] for m in c4_methods}
        fold_multi_alpha_conformal: Dict[float, Dict[str, List[dict]]] = {
            a: {m: [] for m in c4_methods} for a in multi_alphas
        }

        for k in range(10):
            if only_folds is not None and k not in only_folds:
                continue

            test_lbl_path = os.path.join(mdir, f'fold_{k}_labels.npy')
            test_prd_path = os.path.join(mdir, f'fold_{k}_preds.npy')
            test_prb_path = os.path.join(mdir, f'fold_{k}_probs.npy')
            vcal_lbl_path = os.path.join(mdir, f'fold_{k}_valcal_labels.npy')
            vcal_prb_path = os.path.join(mdir, f'fold_{k}_valcal_probs.npy')
            vsel_lbl_path = os.path.join(mdir, f'fold_{k}_valsel_labels.npy')
            vsel_prb_path = os.path.join(mdir, f'fold_{k}_valsel_probs.npy')
            vsel_prd_path = os.path.join(mdir, f'fold_{k}_valsel_preds.npy')
            sum_path = os.path.join(mdir, f'fold_{k}_summary.json')

            if not (_is_valid_npy_file(test_lbl_path) and _is_valid_npy_file(test_prd_path)):
                continue

            fold_split = load_fold_data(k, split_type)
            exp_test_freq = np.round(fold_split.test[:, 2]).astype(np.int64)

            y_true_raw = np.load(test_lbl_path).flatten()
            y_true_freq = np.round(y_true_raw).astype(np.int64) if y_true_raw.min() >= 1 else (y_true_raw.astype(np.int64) + 1)
            if not np.array_equal(y_true_freq, exp_test_freq):
                raise AssertionError(
                    f"CRITICAL: {method_name} Split {split_type} Fold {k} test labels do not match fold_split.test[:, 2]!"
                )
            y_true_c = np.clip(y_true_freq, 1, 5) - 1

            has_test_probs = _is_valid_npy_file(test_prb_path)
            if has_test_probs:
                probs = np.load(test_prb_path).astype(np.float32)
                y_pred_c = np.argmax(probs, axis=1).astype(np.int64)
            else:
                probs = None
                y_pred_raw = np.load(test_prd_path).flatten()
                y_pred_c = np.clip(np.round(y_pred_raw), 1, 5).astype(np.int64) - 1

            t_metrics = compute_multiclass_metrics(y_true_c, y_pred_c, probs=probs)
            folds_found.append(k)
            fold_test_list.append(t_metrics)
            fold_y_true_map[k] = y_true_c
            fold_y_pred_map[k] = y_pred_c
            if probs is not None:
                fold_probs_map[k] = probs

            # Optional val_select metrics for parity & val_select-vs-test consistency check
            s_json = None
            if _is_valid_json_file(sum_path):
                with open(sum_path, 'r', encoding='utf-8') as sf:
                    s_json = json.load(sf)

            vs_metrics_k = None
            if _is_valid_npy_file(vsel_lbl_path) and _is_valid_npy_file(vsel_prb_path):
                vs_raw = np.load(vsel_lbl_path).flatten()
                vs_c = np.clip(np.round(vs_raw), 1, 5).astype(int) - 1 if vs_raw.min() >= 1 else vs_raw.astype(int)
                vs_prb = np.load(vsel_prb_path)
                vs_metrics_k = compute_multiclass_metrics(vs_c, np.argmax(vs_prb, axis=1), probs=vs_prb)
            elif _is_valid_npy_file(vsel_lbl_path) and _is_valid_npy_file(vsel_prd_path):
                vs_raw = np.load(vsel_lbl_path).flatten()
                vs_c = np.clip(np.round(vs_raw), 1, 5).astype(int) - 1 if vs_raw.min() >= 1 else vs_raw.astype(int)
                vs_prd_raw = np.load(vsel_prd_path).flatten()
                vs_prd_c = np.clip(np.round(vs_prd_raw), 1, 5).astype(int) - 1
                vs_metrics_k = compute_multiclass_metrics(vs_c, vs_prd_c, probs=None)
            elif s_json is not None and isinstance(s_json.get('valsel_metrics'), dict):
                vs_metrics_k = s_json['valsel_metrics']

            if vs_metrics_k is not None:
                fold_valsel_map[k] = vs_metrics_k

            if s_json is not None:
                if s_json.get('best_train_macro_f1') is not None and vs_metrics_k is not None:
                    fold_gaps.append(float(s_json['best_train_macro_f1']) - float(vs_metrics_k['MacroF1']))
                elif s_json.get('train_valsel_macro_f1_gap') is not None:
                    fold_gaps.append(float(s_json['train_valsel_macro_f1_gap']))
                if s_json.get('elapsed_seconds') is not None:
                    fold_times.append(float(s_json['elapsed_seconds']))
                if s_json.get('peak_gpu_mb') is not None:
                    fold_peak_gpus.append(float(s_json['peak_gpu_mb']))
                if total_params is None and s_json.get('total_params') is not None:
                    total_params = int(s_json['total_params'])

            if (
                compute_conformal
                and probs is not None
                and _is_valid_npy_file(vcal_lbl_path)
                and _is_valid_npy_file(vcal_prb_path)
            ):
                vcal_c = np.load(vcal_lbl_path).flatten()
                vcal_prb = np.load(vcal_prb_path).astype(np.float32)
                vcal_freq = (vcal_c + 1) if vcal_c.min() == 0 else vcal_c
                test_drug_ids = fold_split.test[:, 0].astype(int)

                c4_main = evaluate_all_conformal_methods(
                    valcal_probs=vcal_prb,
                    valcal_labels_1_to_5=vcal_freq,
                    test_probs=probs,
                    test_labels_1_to_5=y_true_freq,
                    alpha=0.10,
                    seed=42 + k,
                    test_drug_ids=test_drug_ids,
                )
                for c4_name, c4_val in c4_main.items():
                    fold_test_conformal[c4_name].append(c4_val)

                if method_name == 'slim_nodec':
                    for a_val in multi_alphas:
                        c4_a = evaluate_all_conformal_methods(
                            valcal_probs=vcal_prb,
                            valcal_labels_1_to_5=vcal_freq,
                            test_probs=probs,
                            test_labels_1_to_5=y_true_freq,
                            alpha=float(a_val),
                            seed=42 + k,
                            test_drug_ids=test_drug_ids,
                        )
                        for c4_name, c4_val in c4_a.items():
                            fold_multi_alpha_conformal[a_val][c4_name].append(c4_val)

        if not fold_test_list:
            continue

        tm, ts = aggregate_fold_metrics(fold_test_list)
        test_agg = {'mean': tm, 'std': ts, 'per_fold': fold_test_list}

        valsel_agg = None
        if len(fold_valsel_map) == len(folds_found):
            vs_list = [fold_valsel_map[f] for f in folds_found]
            vm, vs = aggregate_fold_metrics(vs_list)
            valsel_agg = {'mean': vm, 'std': vs, 'per_fold': vs_list, 'by_fold': fold_valsel_map}
        else:
            valsel_agg = {'mean': None, 'std': None, 'per_fold': [], 'by_fold': fold_valsel_map}

        agg_test_c4 = {}
        for c4_name, c4_list in fold_test_conformal.items():
            if c4_list:
                cm, cs = aggregate_fold_metrics(c4_list)
                agg_test_c4[c4_name] = {'mean': cm, 'std': cs, 'per_fold': c4_list}

        agg_multi_alpha_c4: Dict[float, Dict[str, dict]] = {}
        if method_name == 'slim_nodec':
            for a_val, m_dict in fold_multi_alpha_conformal.items():
                agg_multi_alpha_c4[a_val] = {}
                for c4_name, c4_list in m_dict.items():
                    if c4_list:
                        cm, cs = aggregate_fold_metrics(c4_list)
                        agg_multi_alpha_c4[a_val][c4_name] = {'mean': cm, 'std': cs}

        if total_params is None and method_name in SLIM_EXPERIMENTS:
            m_inst = _instantiate_model_for_cfg_name(method_name)
            total_params = get_named_children_param_breakdown(m_inst)['TOTAL']

        collected[method_name] = {
            'folds': folds_found,
            'test': test_agg,
            'test_by_fold': {f: m for f, m in zip(folds_found, fold_test_list)},
            'valsel': valsel_agg,
            'y_true_by_fold': fold_y_true_map,
            'y_pred_by_fold': fold_y_pred_map,
            'probs_by_fold': fold_probs_map,
            'test_conformal': agg_test_c4,
            'multi_alpha_conformal': agg_multi_alpha_c4,
            'total_params': total_params or 0,
            'time_mean': float(np.mean(fold_times)) if fold_times else float('nan'),
            'time_std': float(np.std(fold_times)) if fold_times else float('nan'),
            'fold_times': fold_times,
            'peak_gpu_mean': float(np.mean(fold_peak_gpus)) if fold_peak_gpus else float('nan'),
            'peak_gpu_max': float(np.max(fold_peak_gpus)) if fold_peak_gpus else float('nan'),
            'gap_mean': float(np.mean(fold_gaps)) if fold_gaps else float('nan'),
            'gap_std': float(np.std(fold_gaps)) if fold_gaps else float('nan'),
        }

    return collected


def _load_locked_valsel_f1_map(
    locked_dir: Optional[str],
    locked_drive_dir: Optional[str],
    split_type: str,
    methods: List[str],
) -> Dict[str, Dict[int, float]]:
    """Load locked Stage 6 `val_select` per-fold Macro-F1 for deterministic parity audit."""
    if not locked_dir and not locked_drive_dir:
        return {}
    eff_dir = locked_dir or os.path.join(DEFAULT_RESULTS_SLIM_DIR, '_locked_valsel_cache')
    if locked_drive_dir and os.path.isdir(locked_drive_dir):
        _sync_valid_files_from_drive(eff_dir, locked_drive_dir, split_type, only_methods=methods)
    if not os.path.isdir(eff_dir):
        return {}

    out: Dict[str, Dict[int, float]] = {}
    for m_name in methods:
        mdir = os.path.join(eff_dir, f'{m_name}_{split_type}')
        if not os.path.isdir(mdir):
            continue
        fmap: Dict[int, float] = {}
        for k in range(10):
            vs_lbl = os.path.join(mdir, f'fold_{k}_valsel_labels.npy')
            vs_prb = os.path.join(mdir, f'fold_{k}_valsel_probs.npy')
            if _is_valid_npy_file(vs_lbl) and _is_valid_npy_file(vs_prb):
                raw_y = np.load(vs_lbl).flatten()
                yc = np.clip(np.round(raw_y), 1, 5).astype(int) - 1 if raw_y.min() >= 1 else raw_y.astype(int)
                prb = np.load(vs_prb)
                fmap[k] = float(compute_multiclass_metrics(yc, np.argmax(prb, axis=1), probs=prb)['MacroF1'])
        if fmap:
            out[m_name] = fmap
    return out


def format_section0_parity_audit(
    slim_by_split: Dict[str, Dict[str, dict]],
    locked_dir: Optional[str],
    locked_drive_dir: Optional[str],
) -> str:
    lines = ["## 0. Protocol Integrity & Deterministic `val_select` Re-Run Parity Audit\n"]
    any_rows = False
    lines.append(
        "| Split | Configuration | Matched Folds | Locked Stage 6 `val_select` F1 | Stage 7 Re-Run `val_select` F1 | Mean $|\\Delta|$ | Max $|\\Delta|$ | Status |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")

    for sp in ['B', 'A']:
        col = slim_by_split.get(sp, {})
        if not col:
            continue
        locked_map = _load_locked_valsel_f1_map(locked_dir, locked_drive_dir, sp, list(col.keys()))
        for m_name in ALL_REPORT_ORDER:
            if m_name not in col or m_name not in locked_map:
                continue
            cur_map = {
                int(f): float(v['MacroF1'])
                for f, v in col[m_name]['valsel']['by_fold'].items()
            }
            common = sorted(set(cur_map.keys()) & set(locked_map[m_name].keys()))
            if not common:
                continue
            any_rows = True
            l_vals = [locked_map[m_name][f] for f in common]
            c_vals = [cur_map[f] for f in common]
            abs_diffs = [abs(c - l) for c, l in zip(c_vals, l_vals)]
            mean_abs = float(np.mean(abs_diffs))
            max_abs = float(np.max(abs_diffs))
            status = "`PASS (Exact/GPU-Tol)`" if mean_abs <= 0.0030 else "`CHECK-DRIFT`"
            lines.append(
                f"| Split {sp} | `{m_name}` | `{len(common)}` | "
                f"{np.mean(l_vals):.4f} ± {np.std(l_vals):.4f} | "
                f"{np.mean(c_vals):.4f} ± {np.std(c_vals):.4f} | "
                f"`{mean_abs:.6f}` | `{max_abs:.6f}` | {status} |"
            )

    if not any_rows:
        return (
            "## 0. Protocol Integrity & Deterministic `val_select` Re-Run Parity Audit\n\n"
            "_Locked Stage 6 `val_select` directory not provided or same directory used; skipping cross-run diff._\n"
        )
    lines.append("")
    return "\n".join(lines)


def format_section1_main_benchmark(
    slim_by_split: Dict[str, Dict[str, dict]],
    base_by_split: Dict[str, Dict[str, dict]],
    target_folds: List[int],
) -> str:
    fold_tag = f"Folds `{target_folds[0]}–{target_folds[-1]}` ($K={len(target_folds)}$)"
    lines = [f"## 1. Main Benchmark Comparison on Held-Out `test` ({fold_tag})\n"]
    target_set = set(target_folds)

    for sp in ['B', 'A']:
        col_s = slim_by_split.get(sp, {})
        col_b = base_by_split.get(sp, {})
        if not col_s and not col_b:
            continue

        lines.append(f"### 1.{1 if sp == 'B' else 2}. Split {sp} — Held-Out `test` Main Comparison ({fold_tag})")
        lines.append(
            "| Group | Method | Folds | Macro-F1 | Weighted-F1 | Accuracy | QWK | MAE | "
            "Paired $\\Delta$ F1 vs `slim_nodec` | W/T/L (`slim_nodec`) | Wilcoxon $p$ | Holm $p_{\\text{Holm}}$ |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

        ref_entry = col_s.get('slim_nodec')
        ref_fmap = (
            {f: float(ref_entry['test_by_fold'][f]['MacroF1']) for f in target_folds}
            if (ref_entry and target_set.issubset(set(ref_entry['test_by_fold'].keys())))
            else {}
        )

        # Precompute raw Wilcoxon p-values across primary 10-fold calibrated baselines (`*_thr`) for Holm-Bonferroni
        ordered_bases = [b for b in BASELINE_REPORT_ORDER if b in col_b] + sorted(
            [b for b in col_b.keys() if b not in BASELINE_REPORT_ORDER and b not in col_s]
        )
        primary_thr_p_raw: Dict[str, Optional[float]] = {}
        all_base_p_raw: Dict[str, Optional[float]] = {}
        for b_name in ordered_bases:
            if b_name.startswith('HSTrans') and len(target_folds) > 3:
                continue
            b_entry = col_b[b_name]
            if not target_set.issubset(set(b_entry['test_by_fold'].keys())) or not ref_fmap:
                continue
            diffs = [float(b_entry['test_by_fold'][f]['MacroF1']) - ref_fmap[f] for f in target_folds]
            p_val = _paired_wilcoxon_p(diffs)
            all_base_p_raw[b_name] = p_val
            if b_name.endswith('_thr'):
                primary_thr_p_raw[b_name] = p_val

        holm_thr = _holm_bonferroni_adjust(primary_thr_p_raw)
        holm_all = _holm_bonferroni_adjust(all_base_p_raw)

        def _format_row(group_lbl: str, m_name: str, entry: dict, is_ext: bool) -> Optional[str]:
            if not target_set.issubset(set(entry['test_by_fold'].keys())):
                return None
            sel = [entry['test_by_fold'][f] for f in target_folds]
            mm, ms = aggregate_fold_metrics(sel)
            if ref_fmap:
                c_vals = [float(entry['test_by_fold'][f]['MacroF1']) for f in target_folds]
                r_vals = [ref_fmap[f] for f in target_folds]
                diffs = [c - r for c, r in zip(c_vals, r_vals)]
                d_str = f"{np.mean(diffs):+.4f} ± {np.std(diffs):.4f}"
                wtl = "—" if m_name == 'slim_nodec' else _win_tie_loss_str(c_vals, r_vals, higher_is_better=True)
                pw = _paired_wilcoxon_p(diffs)
                p_str = "`1.0000`" if m_name == 'slim_nodec' else (f"`{pw:.4f}`" if pw is not None else "N/A")
                if is_ext:
                    ph = holm_thr.get(m_name) if m_name in holm_thr else holm_all.get(m_name)
                    ph_str = f"`{ph:.4f}`" if ph is not None else "N/A"
                else:
                    ph_str = "—"
            else:
                d_str, wtl, p_str, ph_str = "N/A", "N/A", "N/A", "N/A"

            return (
                f"| {group_lbl} | `{m_name}` | `{len(target_folds)}` | "
                f"{mm['MacroF1']:.4f} ± {ms['MacroF1']:.4f} | "
                f"{mm['WeightedF1']:.4f} ± {ms['WeightedF1']:.4f} | "
                f"{mm['Accuracy']:.4f} ± {ms['Accuracy']:.4f} | "
                f"{mm['QWK']:.4f} ± {ms['QWK']:.4f} | "
                f"{mm['MAE']:.4f} ± {ms['MAE']:.4f} | "
                f"{d_str} | `{wtl}` | {p_str} | {ph_str} |"
            )

        for m_name in ['slim_nodec', 'slim_all_cnnshared', 'MSSF_v2_selfcorr']:
            if m_name in col_s:
                r = _format_row("Ours / Internal", m_name, col_s[m_name], is_ext=False)
                if r:
                    lines.append(r)

        hstrans_notes: List[Tuple[str, dict]] = []
        for b_name in ordered_bases:
            if b_name.startswith('HSTrans') and len(target_folds) > 3:
                hstrans_notes.append((b_name, col_b[b_name]))
                continue
            r = _format_row("External Baseline", b_name, col_b[b_name], is_ext=True)
            if r:
                lines.append(r)
            elif b_name.startswith('HSTrans'):
                hstrans_notes.append((b_name, col_b[b_name]))

        # Secondary statistical tests on QWK and MAE for External Baselines vs slim_nodec
        if ref_entry and target_set.issubset(set(ref_entry['test_by_fold'].keys())) and ordered_bases:
            lines.append(f"\n#### Split {sp} — Paired Ordinal Tests (`QWK` & `MAE`) and Per-Class F1 on Held-Out `test`")
            lines.append(
                "| Method | F1_c0 | F1_c1 | F1_c2 | F1_c3 | F1_c4 | "
                "Paired $\\Delta$ QWK vs `slim_nodec` (Wilcoxon $p$) | Paired $\\Delta$ MAE vs `slim_nodec` (Wilcoxon $p$) |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            ref_qwk = [float(ref_entry['test_by_fold'][f]['QWK']) for f in target_folds]
            ref_mae = [float(ref_entry['test_by_fold'][f]['MAE']) for f in target_folds]

            show_list: List[Tuple[str, dict]] = []
            for m_n in ['slim_nodec', 'slim_all_cnnshared', 'MSSF_v2_selfcorr']:
                if m_n in col_s and target_set.issubset(set(col_s[m_n]['test_by_fold'].keys())):
                    show_list.append((m_n, col_s[m_n]))
            for b_n in ordered_bases:
                if b_n.startswith('HSTrans') and len(target_folds) > 3:
                    continue
                if target_set.issubset(set(col_b[b_n]['test_by_fold'].keys())):
                    show_list.append((b_n, col_b[b_n]))

            for m_n, ent in show_list:
                sel = [ent['test_by_fold'][f] for f in target_folds]
                mm, _ = aggregate_fold_metrics(sel)
                q_diffs = [float(ent['test_by_fold'][f]['QWK']) - rq for f, rq in zip(target_folds, ref_qwk)]
                m_diffs = [float(ent['test_by_fold'][f]['MAE']) - rm for f, rm in zip(target_folds, ref_mae)]
                pq = _paired_wilcoxon_p(q_diffs)
                pm = _paired_wilcoxon_p(m_diffs)
                pq_s = "—" if m_n == 'slim_nodec' else (f"`{pq:.4f}`" if pq is not None else "N/A")
                pm_s = "—" if m_n == 'slim_nodec' else (f"`{pm:.4f}`" if pm is not None else "N/A")
                lines.append(
                    f"| `{m_n}` | {mm['F1_c0']:.4f} | {mm['F1_c1']:.4f} | {mm['F1_c2']:.4f} | "
                    f"{mm['F1_c3']:.4f} | {mm['F1_c4']:.4f} | "
                    f"{np.mean(q_diffs):+.4f} ({pq_s}) | {np.mean(m_diffs):+.4f} ({pm_s}) |"
                )

        if hstrans_notes:
            lines.append(
                "\n> **Chú thích riêng cho `HSTrans` trên `test` (3 fold `0–2` do chi phí `~7.900 giây/fold` — KHÔNG gộp vào cột 10 fold)**:"
            )
            for h_name, h_ent in hstrans_notes:
                hm = h_ent['test']['mean']
                hs = h_ent['test']['std']
                h_folds = h_ent['folds']
                matched_str = ""
                if ref_entry and set(h_folds).issubset(set(ref_entry['test_by_fold'].keys())):
                    r_sub = [float(ref_entry['test_by_fold'][f]['MacroF1']) for f in h_folds]
                    h_sub = [float(h_ent['test_by_fold'][f]['MacroF1']) for f in h_folds]
                    d_sub = [h - r for h, r in zip(h_sub, r_sub)]
                    matched_str = (
                        f" | Matched Folds `{h_folds}` `slim_nodec` = `{np.mean(r_sub):.4f} ± {np.std(r_sub):.4f}` "
                        f"($\\Delta = {np.mean(d_sub):+.4f}$)"
                    )
                lines.append(
                    f"> - `{h_name}` (Folds `{h_folds}`, $K={len(h_folds)}$): "
                    f"Test Macro-F1 = `{hm['MacroF1']:.4f} ± {hs['MacroF1']:.4f}`, "
                    f"QWK = `{hm['QWK']:.4f} ± {hs['QWK']:.4f}`, "
                    f"MAE = `{hm['MAE']:.4f} ± {hs['MAE']:.4f}`{matched_str}."
                )
        lines.append("")

    return "\n".join(lines)


def format_section2_ablation_and_loo(
    slim_by_split: Dict[str, Dict[str, dict]],
    target_folds: List[int],
) -> str:
    fold_tag = f"Folds `{target_folds[0]}–{target_folds[-1]}` ($K={len(target_folds)}$)"
    lines = [f"## 2. 10-Fold Ablation & Leave-One-Out (LOO) Results on Held-Out `test` ({fold_tag})\n"]
    target_set = set(target_folds)

    for sp in ['B', 'A']:
        col = slim_by_split.get(sp, {})
        if not col:
            continue
        lines.append(f"### 2.{1 if sp == 'B' else 2}. Split {sp} — Compression & LOO Ablation on Held-Out `test`")
        lines.append(
            "| Configuration | Params | `val_select` Macro-F1 | **`test` Macro-F1** | `test` QWK | `test` MAE | "
            "F1_c0 | F1_c4 | Paired $\\Delta_{\\text{test}}$ vs `slim_nodec` | Wilcoxon $p$ | Time/Fold (s) | Peak GPU (MB) |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

        ref_entry = col.get('slim_nodec')
        ref_fmap = (
            {f: float(ref_entry['test_by_fold'][f]['MacroF1']) for f in target_folds}
            if (ref_entry and target_set.issubset(set(ref_entry['test_by_fold'].keys())))
            else {}
        )

        for m_name in ALL_REPORT_ORDER:
            if m_name not in col:
                continue
            ent = col[m_name]
            if not target_set.issubset(set(ent['test_by_fold'].keys())):
                continue
            sel = [ent['test_by_fold'][f] for f in target_folds]
            mm, ms = aggregate_fold_metrics(sel)

            vs_by_f = ent['valsel']['by_fold']
            if target_set.issubset(set(vs_by_f.keys())):
                vs_vals = [float(vs_by_f[f]['MacroF1']) for f in target_folds]
                vs_str = f"{np.mean(vs_vals):.4f} ± {np.std(vs_vals):.4f}"
            else:
                vs_str = "N/A"

            if ref_fmap:
                diffs = [float(ent['test_by_fold'][f]['MacroF1']) - ref_fmap[f] for f in target_folds]
                d_str = f"{np.mean(diffs):+.4f} ± {np.std(diffs):.4f}"
                pw = _paired_wilcoxon_p(diffs)
                p_str = "`1.0000`" if m_name == 'slim_nodec' else (f"`{pw:.4f}`" if pw is not None else "N/A")
            else:
                d_str, p_str = "N/A", "N/A"

            t_str = f"{ent['time_mean']:.1f} ± {ent['time_std']:.1f}" if not np.isnan(ent['time_mean']) else "N/A"
            g_str = f"{ent['peak_gpu_mean']:.1f}" if not np.isnan(ent['peak_gpu_mean']) and ent['peak_gpu_mean'] > 0 else "N/A"
            lines.append(
                f"| `{m_name}` | {ent['total_params']:,} | {vs_str} | "
                f"**{mm['MacroF1']:.4f} ± {ms['MacroF1']:.4f}** | "
                f"{mm['QWK']:.4f} ± {ms['QWK']:.4f} | "
                f"{mm['MAE']:.4f} ± {ms['MAE']:.4f} | "
                f"{mm['F1_c0']:.4f} | {mm['F1_c4']:.4f} | "
                f"{d_str} | {p_str} | {t_str} | {g_str} |"
            )
        lines.append("")

    return "\n".join(lines)


def format_section3_conformal(
    slim_by_split: Dict[str, Dict[str, dict]],
) -> str:
    lines = ["## 3. C4 Contiguous Ordinal Conformal Prediction on Held-Out `test` (Calibrated on `val_calib`)\n"]
    c4_models = ['slim_nodec', 'cand_A', 'slim_all_cnnshared', 'MSSF_v2_selfcorr']
    c4_methods = ['LAC', 'APS', 'Ordinal_LAC', 'Ordinal_APS']

    for sp in ['B', 'A']:
        col = slim_by_split.get(sp, {})
        if not col:
            continue
        lines.append(f"### 3.{1 if sp == 'B' else 2}. Split {sp} — Held-Out `test` Conformal Prediction ($\\alpha = 0.10$, Target $90\\%$)")
        lines.append(
            "| Model | C4 Method | Pair Coverage | Drug Cov (Mean ± Std) | Drug Cov $<80\\%$ | "
            "Avg Set Size | Singleton Rate | Contiguity |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for m_name in c4_models:
            if m_name not in col or not col[m_name]['test_conformal']:
                continue
            tc = col[m_name]['test_conformal']
            for c4_m in c4_methods:
                if c4_m not in tc:
                    continue
                cm, cs = tc[c4_m]['mean'], tc[c4_m]['std']
                lines.append(
                    f"| `{m_name}` | `{c4_m}` | "
                    f"{cm['empirical_coverage']:.4f} ± {cs['empirical_coverage']:.4f} | "
                    f"{cm['drug_coverage_mean']:.4f} ± {cm['drug_coverage_std']:.4f} | "
                    f"{cm['drug_coverage_lt80_rate']:.4f} ± {cs['drug_coverage_lt80_rate']:.4f} | "
                    f"{cm['avg_set_size']:.4f} ± {cs['avg_set_size']:.4f} | "
                    f"{cm['singleton_rate']:.4f} ± {cs['singleton_rate']:.4f} | "
                    f"{cm['contiguity_rate']:.4f} |"
                )

        # 3b. Class-conditional coverage breakdown (Cov_c0..c4)
        lines.append(f"\n#### Split {sp} — Class-Conditional Coverage (`Cov_c0..Cov_c4`) on Held-Out `test` ($\\alpha = 0.10$)")
        lines.append("| Model | C4 Method | Cov_c0 (Very Rare) | Cov_c1 (Rare) | Cov_c2 (Infreq) | Cov_c3 (Freq) | Cov_c4 (Very Freq) |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for m_name in ['slim_nodec', 'MSSF_v2_selfcorr']:
            if m_name not in col or not col[m_name]['test_conformal']:
                continue
            tc = col[m_name]['test_conformal']
            for c4_m in c4_methods:
                if c4_m not in tc:
                    continue
                cm, cs = tc[c4_m]['mean'], tc[c4_m]['std']
                lines.append(
                    f"| `{m_name}` | `{c4_m}` | "
                    f"{cm['coverage_c0']:.4f} ± {cs['coverage_c0']:.4f} | "
                    f"{cm['coverage_c1']:.4f} ± {cs['coverage_c1']:.4f} | "
                    f"{cm['coverage_c2']:.4f} ± {cs['coverage_c2']:.4f} | "
                    f"{cm['coverage_c3']:.4f} ± {cs['coverage_c3']:.4f} | "
                    f"{cm['coverage_c4']:.4f} ± {cs['coverage_c4']:.4f} |"
                )

        # 3c. Multi-alpha sensitivity curve for slim_nodec
        if 'slim_nodec' in col and col['slim_nodec']['multi_alpha_conformal']:
            lines.append(f"\n#### Split {sp} — Multi-$\\alpha$ Sensitivity Curve on Held-Out `test` (`slim_nodec`)")
            lines.append("| Target $1-\\alpha$ ($\\alpha$) | C4 Method | Pair Coverage | Drug Cov Mean | Avg Set Size | Singleton Rate | Contiguity |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            ma = col['slim_nodec']['multi_alpha_conformal']
            for a_val in sorted(ma.keys()):
                for c4_m in ['Ordinal_LAC', 'Ordinal_APS', 'APS', 'LAC']:
                    if c4_m not in ma[a_val]:
                        continue
                    cm, cs = ma[a_val][c4_m]['mean'], ma[a_val][c4_m]['std']
                    lines.append(
                        f"| `{1.0 - a_val:.0%}` (`{a_val:.2f}`) | `{c4_m}` | "
                        f"{cm['empirical_coverage']:.4f} ± {cs['empirical_coverage']:.4f} | "
                        f"{cm['drug_coverage_mean']:.4f} | "
                        f"{cm['avg_set_size']:.4f} ± {cs['avg_set_size']:.4f} | "
                        f"{cm['singleton_rate']:.4f} | "
                        f"{cm['contiguity_rate']:.4f} |"
                    )
        lines.append("")

    return "\n".join(lines)


def format_section4_paper_analyses(
    slim_by_split: Dict[str, Dict[str, dict]],
    base_by_split: Dict[str, Dict[str, dict]],
    target_folds: List[int],
) -> str:
    lines = ["## 4. Paper Error Severity, Confusion Matrices & Stratification Analyses on Held-Out `test`\n"]
    target_set = set(target_folds)

    for sp in ['B', 'A']:
        col_s = slim_by_split.get(sp, {})
        col_b = base_by_split.get(sp, {})
        merged: Dict[str, dict] = {}
        for k, v in col_s.items():
            merged[k] = v
        for k, v in col_b.items():
            if k not in merged:
                merged[k] = v
        if not merged:
            continue

        cand_models = ['slim_nodec', 'cand_A', 'slim_all_cnnshared', 'MSSF_v2_selfcorr']
        for b_name in PRIMARY_10FOLD_THR_BASELINES:
            if b_name in col_b and b_name not in cand_models:
                cand_models.append(b_name)

        # 4a. Ordinal jump error severity
        lines.append(f"### 4.{1 if sp == 'B' else 2}a. Split {sp} — Ordinal Jump Error Severity Across Folds on Held-Out `test`")
        lines.append(
            "| Method | Exact Match ($|\\hat{y}-y|=0$) | Adjacent Acc ($|\\hat{y}-y|\\le 1$) | "
            "Large Jump ($|\\hat{y}-y|\\ge 2$) | Extreme Jump ($|\\hat{y}-y|\\ge 3$) | MAE |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")

        for m_name in cand_models:
            if m_name not in merged:
                continue
            ent = merged[m_name]
            if not target_set.issubset(set(ent['y_true_by_fold'].keys())):
                continue
            ex_list, adj_list, lg2_list, lg3_list, mae_list = [], [], [], [], []
            for f in target_folds:
                yt = ent['y_true_by_fold'][f]
                yp = ent['y_pred_by_fold'][f]
                ad = np.abs(yp - yt)
                ex_list.append(float(np.mean(ad == 0)))
                adj_list.append(float(np.mean(ad <= 1)))
                lg2_list.append(float(np.mean(ad >= 2)))
                lg3_list.append(float(np.mean(ad >= 3)))
                mae_list.append(float(np.mean(ad)))
            lines.append(
                f"| `{m_name}` | "
                f"{np.mean(ex_list):.4f} ± {np.std(ex_list):.4f} | "
                f"{np.mean(adj_list):.4f} ± {np.std(adj_list):.4f} | "
                f"{np.mean(lg2_list):.4f} ± {np.std(lg2_list):.4f} | "
                f"{np.mean(lg3_list):.4f} ± {np.std(lg3_list):.4f} | "
                f"{np.mean(mae_list):.4f} ± {np.std(mae_list):.4f} |"
            )

        # Pooled row-normalized 5x5 confusion matrices for slim_nodec, MSSF_v2_selfcorr, and top external baseline
        top_ext = None
        best_ext_f1 = -1.0
        for b_name in PRIMARY_10FOLD_THR_BASELINES:
            if b_name in col_b and target_set.issubset(set(col_b[b_name]['test_by_fold'].keys())):
                f1_m = float(np.mean([col_b[b_name]['test_by_fold'][f]['MacroF1'] for f in target_folds]))
                if f1_m > best_ext_f1:
                    best_ext_f1 = f1_m
                    top_ext = b_name

        cm_targets = [m for m in ['slim_nodec', 'MSSF_v2_selfcorr', top_ext] if m and m in merged]
        for m_name in cm_targets:
            ent = merged[m_name]
            if not target_set.issubset(set(ent['y_true_by_fold'].keys())):
                continue
            conf = np.zeros((5, 5), dtype=np.int64)
            for f in target_folds:
                yt = ent['y_true_by_fold'][f]
                yp = ent['y_pred_by_fold'][f]
                for c_t in range(5):
                    for c_p in range(5):
                        conf[c_t, c_p] += int(np.sum((yt == c_t) & (yp == c_p)))
            row_sums = np.maximum(conf.sum(axis=1, keepdims=True), 1)
            norm_conf = conf.astype(np.float64) / row_sums
            cls_names = ["c0 (Very Rare)", "c1 (Rare)", "c2 (Infreq)", "c3 (Freq)", "c4 (Very Freq)"]
            lines.append(f"\n#### Split {sp} — Row-Normalized Confusion Matrix on `test`: `{m_name}`")
            lines.append("| True \\ Pred | Pred `c0` | Pred `c1` | Pred `c2` | Pred `c3` | Pred `c4` | Row Count |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for r_i in range(5):
                lines.append(
                    f"| **{cls_names[r_i]}** | "
                    f"{norm_conf[r_i, 0]:.4f} | {norm_conf[r_i, 1]:.4f} | {norm_conf[r_i, 2]:.4f} | "
                    f"{norm_conf[r_i, 3]:.4f} | {norm_conf[r_i, 4]:.4f} | {int(conf[r_i].sum()):,} |"
                )

        # 4b. Stratification analysis (SE support tertiles on Split B; Drug interaction degree tertiles on Split A)
        strat_title = (
            "Side-Effect Training Support Tertiles (Cold-Start Drugs)"
            if sp == 'B'
            else "Drug Known-Interaction Degree Tertiles in `train` (Warm-Start Pairs)"
        )
        lines.append(f"\n### 4.{1 if sp == 'B' else 2}b. Split {sp} — Stratification by {strat_title} on Held-Out `test`")
        lines.append(
            "| Method | Low-Support / Sparse (Macro-F1 / MAE) | Medium-Support / Moderate (Macro-F1 / MAE) | "
            "High-Support / Rich (Macro-F1 / MAE) |"
        )
        lines.append("|---|---:|---:|---:|")

        # Precompute per-fold tertile masks strictly from `fold_split.train`
        fold_bucket_masks: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for f in target_folds:
            fs = load_fold_data(f, sp)
            if sp == 'B':
                # Count how many training drugs exhibit each side effect s in fold_split.train
                se_counts = np.bincount(fs.train[:, 1].astype(int), minlength=994)
                pair_support = se_counts[fs.test[:, 1].astype(int)]
            else:
                # Count how many known training side effects each drug d has in fold_split.train
                d_counts = np.bincount(fs.train[:, 0].astype(int), minlength=750)
                pair_support = d_counts[fs.test[:, 0].astype(int)]
            q1, q2 = np.quantile(pair_support, [1.0 / 3.0, 2.0 / 3.0])
            m_low = pair_support <= q1
            m_med = (pair_support > q1) & (pair_support <= q2)
            m_high = pair_support > q2
            fold_bucket_masks[f] = (m_low, m_med, m_high)

        for m_name in cand_models:
            if m_name not in merged:
                continue
            ent = merged[m_name]
            if not target_set.issubset(set(ent['y_true_by_fold'].keys())):
                continue
            b_stats = [[], [], []]
            for f in target_folds:
                yt = ent['y_true_by_fold'][f]
                yp = ent['y_pred_by_fold'][f]
                masks = fold_bucket_masks[f]
                for b_i in range(3):
                    msk = masks[b_i]
                    if np.any(msk):
                        bm = compute_multiclass_metrics(yt[msk], yp[msk], probs=None)
                        b_stats[b_i].append((float(bm['MacroF1']), float(bm['MAE'])))
            cells = []
            for b_i in range(3):
                if b_stats[b_i]:
                    f1_arr = [x[0] for x in b_stats[b_i]]
                    mae_arr = [x[1] for x in b_stats[b_i]]
                    cells.append(f"`{np.mean(f1_arr):.4f}` / `{np.mean(mae_arr):.4f}`")
                else:
                    cells.append("N/A")
            lines.append(f"| `{m_name}` | {cells[0]} | {cells[1]} | {cells[2]} |")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Summarize one-shot Stage 7 held-out test results, paired Wilcoxon/Holm tests, C4 conformal, and paper analyses.'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default=os.path.join(DEFAULT_RESULTS_SLIM_DIR, 'test_final'),
        help='Directory containing Stage 7 eval_test=True outputs for mssf_slim models.',
    )
    parser.add_argument(
        '--drive_dir',
        type=str,
        default='/content/drive/MyDrive/DSE_results_slim_test',
        help='Google Drive directory for Stage 7 mssf_slim test outputs (auto-ignored if absent).',
    )
    parser.add_argument(
        '--baseline_results_dir',
        type=str,
        default=os.path.join(DEFAULT_RESULTS_SLIM_DIR, 'test_final_baselines'),
        help='Local directory inside results_slim/ for Stage 7 external baseline test outputs.',
    )
    parser.add_argument(
        '--baseline_drive_dir',
        type=str,
        default='/content/drive/MyDrive/DSE_results_v2_test',
        help='Google Drive directory for Stage 7 external baseline test outputs (auto-ignored if absent).',
    )
    parser.add_argument(
        '--locked_valsel_dir',
        type=str,
        default=DEFAULT_RESULTS_SLIM_DIR,
        help='Locked Stage 6 val_select local directory for deterministic parity audit.',
    )
    parser.add_argument(
        '--locked_valsel_drive_dir',
        type=str,
        default='/content/drive/MyDrive/DSE_results_slim',
        help='Locked Stage 6 val_select Google Drive directory for deterministic parity audit.',
    )
    parser.add_argument('--splits', type=str, nargs='+', default=['B', 'A'], choices=['A', 'B'])
    parser.add_argument('--folds', type=int, nargs='*', default=None)
    parser.add_argument('--methods', type=str, nargs='*', default=None)
    parser.add_argument('--no_conformal', action='store_true')
    parser.add_argument('--out_md', type=str, default=None)
    parser.add_argument(
        '--out_raw',
        type=str,
        default=os.path.join(DEFAULT_RESULTS_SLIM_DIR, 'stage7_test_summarize_results.txt'),
    )
    args = parser.parse_args()

    target_folds = sorted(args.folds) if args.folds else list(range(10))
    methods_filter = list(args.methods) if args.methods else list(FINAL_10FOLD_EXPERIMENTS)

    cmd_str = "python -m mssf_slim.summarize_stage7_test " + " ".join(sys.argv[1:])
    sections = [
        "\n".join([
            "SCRIPT: mssf_slim/summarize_stage7_test.py",
            f"COMMAND: {cmd_str}",
            "=" * 110,
            "# Stage 7 One-Shot Held-Out `test` Evaluation & Paper Analysis Report (`mssf_slim`)\n",
        ])
    ]

    slim_by_split: Dict[str, Dict[str, dict]] = {}
    base_by_split: Dict[str, Dict[str, dict]] = {}
    for sp in args.splits:
        slim_by_split[sp] = collect_stage7_test_results(
            results_dir=args.results_dir,
            split_type=sp,
            only_folds=target_folds,
            only_methods=methods_filter,
            drive_dir=args.drive_dir if (args.drive_dir and os.path.isdir(args.drive_dir)) else None,
            compute_conformal=not args.no_conformal,
        )
        b_drive = args.baseline_drive_dir if (args.baseline_drive_dir and os.path.isdir(args.baseline_drive_dir)) else None
        if b_drive or (args.baseline_results_dir and os.path.isdir(args.baseline_results_dir)):
            base_by_split[sp] = collect_stage7_test_results(
                results_dir=args.baseline_results_dir,
                split_type=sp,
                only_folds=None,  # Keep 3-fold HSTrans available for its dedicated footnote
                only_methods=None,
                drive_dir=b_drive,
                compute_conformal=False,
            )

    sections.append(
        format_section0_parity_audit(
            slim_by_split=slim_by_split,
            locked_dir=args.locked_valsel_dir,
            locked_drive_dir=args.locked_valsel_drive_dir,
        )
    )
    sections.append(format_section1_main_benchmark(slim_by_split, base_by_split, target_folds))
    sections.append(format_section2_ablation_and_loo(slim_by_split, target_folds))
    if not args.no_conformal:
        sections.append(format_section3_conformal(slim_by_split))
    sections.append(format_section4_paper_analyses(slim_by_split, base_by_split, target_folds))

    full_report = "\n".join(sections)
    print(full_report)

    if args.out_md:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_md)), exist_ok=True)
        with open(args.out_md, 'w', encoding='utf-8') as f:
            f.write(full_report + "\n")
        print(f"Saved Markdown report to {args.out_md}")

    if args.out_raw:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_raw)), exist_ok=True)
        with open(args.out_raw, 'w', encoding='utf-8') as f:
            f.write(full_report + "\n")
        print(f"Saved verbatim Stage 7 test report to {args.out_raw}")


if __name__ == '__main__':
    main()
