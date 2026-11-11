"""
Summarize Stage 5 & 10-fold (`mssf_slim`) `val_select` results directly from saved `.npy` and `.json`
files in `results_slim/` (and optionally fold-matched external baselines from `results_v2/`).

Generates the complete verbatim Markdown report:
  0. Step 0 Parity Check status + Parameter counts by `named_children()` across all registered
     configurations + verbatim `print(model)` of `slim_all`.
  1. Overall & Ordinal Classification Metrics (`Macro-F1`, `Weighted-F1`, `Accuracy`, `QWK`,
     `MAE`, `Gap (tr - val)`, `Time/Fold (s)`, `Peak GPU (MB)`).
  2. Per-Class F1 Breakdown (`F1_c0..c4`) and Per-Fold Breakdown.
  2e. Paired Fold Comparison vs `cand_A` and vs `slim_nodec` (with paired Wilcoxon signed-rank
      p-value when common folds >= 6).
  2f. Learned `EN-add` view weights `w_v` (mean +/- std across folds) for interpretability.
  3. C4 Ordinal Conformal Prediction metrics (`val_select`).
  4. Joint Cross-Split Selection Table & Approved Final Model Rationale (`slim_nodec` final model;
     `slim_all_cnnshared` maximum-compression ablation variant).
  5. Optional Fold-Matched Baseline Comparison Table (`--baseline_results_dir` / `--baseline_drive_dir`),
     enforcing identical fold sets across compared models and keeping 3-fold `HSTrans` explicitly
     separated with footnote ("3 fold do chi phí ~7.900 giây/fold").
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
    SLIM_EXPERIMENTS,
    SLIM_ORDER,
    get_slim_config,
)
from mssf_slim.model import (
    MSSFSlimModel,
    get_named_children_param_breakdown,
)


BASELINE_REPORT_ORDER = [
    'SDPred_selfcorr_thr',
    'SDPred_thr',
    'SDPred_selfcorr',
    'SDPred',
    'MGPred_selfcorr_thr',
    'MGPred_thr',
    'MGPred_selfcorr',
    'MGPred',
    'MVSL-DSF_selfcorr_thr',
    'MVSL-DSF_thr',
    'MVSL-DSF_selfcorr',
    'MVSL-DSF',
    'DSGAT_thr',
    'DSGAT',
    'A3Net_thr',
    'A3Net',
    'HSTrans_thr',
    'HSTrans',
]


def _is_valid_npy_file(path: str) -> bool:
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False
    try:
        arr = np.load(path, allow_pickle=False)
        return arr.size > 0
    except Exception:
        return False


def _sync_from_drive_if_available(
    results_dir: str,
    drive_dir: Optional[str],
    split_type: str,
    only_methods: Optional[List[str]] = None,
) -> None:
    """Copy valid completed fold files from drive_dir/*_{split_type} into results_dir if missing locally."""
    if not drive_dir or not os.path.isdir(drive_dir):
        return
    src_s0 = os.path.join(drive_dir, 'step0_parity_check.json')
    dst_s0 = os.path.join(results_dir, 'step0_parity_check.json')
    if os.path.isfile(src_s0) and not os.path.exists(dst_s0):
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy2(src_s0, dst_s0)

    pattern = os.path.join(drive_dir, f'*_{split_type}')
    for src_mdir in sorted(glob.glob(pattern)):
        if not os.path.isdir(src_mdir):
            continue
        folder_name = os.path.basename(src_mdir)
        suffix = f'_{split_type}'
        if not folder_name.endswith(suffix):
            continue
        m_name = folder_name[:-len(suffix)]
        if only_methods is not None and m_name not in only_methods:
            continue
        dst_mdir = os.path.join(results_dir, folder_name)
        os.makedirs(dst_mdir, exist_ok=True)
        for fname in os.listdir(src_mdir):
            sp = os.path.join(src_mdir, fname)
            dp = os.path.join(dst_mdir, fname)
            if not os.path.isfile(sp):
                continue
            if fname.endswith('.npy') and not _is_valid_npy_file(sp):
                print(f"[WARN] Skipping corrupted .npy file on Drive: {sp}", file=sys.stderr)
                continue
            if os.path.exists(dp) and fname.endswith('.npy') and not _is_valid_npy_file(dp):
                os.remove(dp)
            if not os.path.exists(dp):
                shutil.copy2(sp, dp)


def _instantiate_model_for_cfg_name(exp_name: str) -> MSSFSlimModel:
    cfg = get_slim_config(exp_name)
    se_mode = str(cfg.get('c2_se_mode', 'plus_ontology'))
    n_se = 5 if se_mode == 'plus_ontology' else (3 if se_mode == 'no_profile' else 4)
    prof_se = [2, 3] if n_se >= 4 else []
    return MSSFSlimModel(
        cfg=cfg,
        n_drug_views=11,
        n_se_views=n_se,
        profile_drug_view_indices=[8, 9],
        profile_se_view_indices=prof_se,
    )


def format_architecture_and_params_section(
    results_dir: str,
    only_methods: Optional[List[str]] = None,
) -> str:
    """
    Build Section 0:
      - Step 0 Parity Check summary (if `step0_parity_check.json` exists)
      - Parameter counts by `model.named_children()` across configurations
      - Verbatim `print(model)` of `slim_all`
    """
    lines = ["## 0. Step 0 Parity Verification & Parameter Breakdown (`named_children()`)\n"]

    s0_path = os.path.join(results_dir, 'step0_parity_check.json')
    if os.path.exists(s0_path):
        with open(s0_path, 'r', encoding='utf-8') as f:
            s0 = json.load(f)
        lines.append("### 0a. Step 0 Mandatory Parity Check (`cand_A`, Fold 0, Split B, `seed=42`)")
        lines.append("| Reference (`MSSF_plus` Fold 0 Split B) | Re-run in `mssf_slim` (Fold 0 Split B) | Absolute Diff $|\\Delta|$ | Threshold | Status | Peak GPU (MB) |")
        lines.append("|---:|---:|---:|---:|---|---:|")
        ref_f1 = float(s0.get('reference_macro_f1', 0.4260))
        got_f1 = float(s0.get('slim_cand_a_fold0_macro_f1', float('nan')))
        diff_f1 = float(s0.get('abs_diff', abs(got_f1 - ref_f1)))
        tol = float(s0.get('tolerance', 0.005))
        status = "PASS" if s0.get('passed', False) else "FAIL"
        pgpu = float(s0.get('peak_gpu_mb', 0.0))
        lines.append(
            f"| `{ref_f1:.4f}` | `{got_f1:.4f}` | `{diff_f1:.6f}` | `$\\le {tol:.4f}$` | **`{status}`** | `{pgpu:.1f}` |\n"
        )

    param_names = [m for m in ALL_REPORT_ORDER if (only_methods is None or m in only_methods or m == 'cand_A')]
    if not param_names:
        param_names = list(SLIM_ORDER)

    breakdowns: Dict[str, Dict[str, int]] = {}
    for exp_name in param_names:
        if exp_name in SLIM_EXPERIMENTS:
            m = _instantiate_model_for_cfg_name(exp_name)
            breakdowns[exp_name] = get_named_children_param_breakdown(m)

    slim_all_repr = str(_instantiate_model_for_cfg_name('slim_all'))

    has_bvi_col = any(bd.get('gaussian_parametrizer', 0) > 0 for bd in breakdowns.values())
    child_cols = [
        'sharedViewEmbed',
        'encoderConnection',
        'encoderAddition',
        'preprocess',
        'crossProduction',
        'attention',
        'det_projector',
    ]
    if has_bvi_col:
        child_cols.append('gaussian_parametrizer')
    child_cols.append('classifier')

    base_total = breakdowns.get('cand_A', get_named_children_param_breakdown(_instantiate_model_for_cfg_name('cand_A')))['TOTAL']

    lines.append("### 0b. Parameter Counts by `model.named_children()` Across Configurations")
    header = "| Configuration | " + " | ".join([f"`{c}`" for c in child_cols]) + " | **`TOTAL`** | **vs `cand_A`** |"
    align = "|---|" + "|".join(["---:" for _ in child_cols]) + "|---:|---:|"
    lines.append(header)
    lines.append(align)

    for exp_name in param_names:
        if exp_name not in breakdowns:
            continue
        bd = breakdowns[exp_name]
        tot = bd['TOTAL']
        delta_pct = (float(tot) / float(base_total) - 1.0) * 100.0
        row_cells = [f"{bd.get(c, 0):,}" for c in child_cols]
        lines.append(
            f"| `{exp_name}` | " + " | ".join(row_cells) + f" | **{tot:,}** | `{delta_pct:+.2f}%` ({tot / base_total:.3f}x) |"
        )

    lines.append("\n### 0c. Verbatim `print(model)` of `slim_all`")
    lines.append("```text")
    lines.append(slim_all_repr)
    lines.append("```\n")
    return "\n".join(lines)


def collect_method_results(
    results_dir: str,
    split_type: str,
    only_folds: Optional[List[int]] = None,
    only_methods: Optional[List[str]] = None,
    drive_dir: Optional[str] = None,
    compute_conformal: bool = True,
) -> Dict[str, dict]:
    """Scan `results_dir/*_{split_type}` and evaluate all available folds directly from saved `.npy` files."""
    _sync_from_drive_if_available(results_dir, drive_dir, split_type, only_methods=only_methods)
    pattern = os.path.join(results_dir, f'*_{split_type}')
    method_dirs = sorted(glob.glob(pattern))
    collected = {}

    for mdir in method_dirs:
        if not os.path.isdir(mdir):
            continue
        folder_name = os.path.basename(mdir)
        suffix = f'_{split_type}'
        if not folder_name.endswith(suffix):
            continue
        method_name = folder_name[:-len(suffix)]
        if only_methods is not None and method_name not in only_methods:
            continue

        fold_valsel_list = []
        fold_gaps = []
        fold_times = []
        fold_peak_gpus = []
        fold_add_weights = []
        fold_c5_list = []
        fold_valsel_conformal_list = {k: [] for k in ['LAC', 'APS', 'Ordinal_LAC', 'Ordinal_APS']}
        folds_found = []
        total_params = None
        named_children_params = None

        for k in range(10):
            if only_folds is not None and k not in only_folds:
                continue

            vcal_lbl_path = os.path.join(mdir, f'fold_{k}_valcal_labels.npy')
            vcal_prb_path = os.path.join(mdir, f'fold_{k}_valcal_probs.npy')
            vsel_lbl_path = os.path.join(mdir, f'fold_{k}_valsel_labels.npy')
            vsel_prb_path = os.path.join(mdir, f'fold_{k}_valsel_probs.npy')
            sum_path = os.path.join(mdir, f'fold_{k}_summary.json')

            has_valsel_npy = _is_valid_npy_file(vsel_lbl_path) and _is_valid_npy_file(vsel_prb_path)
            has_sum_json = os.path.isfile(sum_path) and os.path.getsize(sum_path) > 0
            if not (has_valsel_npy or has_sum_json):
                continue

            s_json = None
            if has_sum_json:
                try:
                    with open(sum_path, 'r', encoding='utf-8') as sf:
                        s_json = json.load(sf)
                except Exception:
                    s_json = None

            vs_metrics_k = None
            if has_valsel_npy:
                vs_raw = np.load(vsel_lbl_path).flatten()
                vs_c = np.clip(np.round(vs_raw), 1, 5).astype(int) - 1 if vs_raw.min() >= 1 else vs_raw.astype(int)
                vs_prb = np.load(vsel_prb_path)
                vs_prd = np.argmax(vs_prb, axis=1)
                vs_metrics_k = compute_multiclass_metrics(vs_c, vs_prd, probs=vs_prb)
            elif s_json is not None and s_json.get('valsel_metrics') is not None:
                vs_metrics_k = s_json['valsel_metrics']

            if vs_metrics_k is None:
                continue

            folds_found.append(k)
            fold_valsel_list.append(vs_metrics_k)

            if s_json is not None:
                if s_json.get('best_train_macro_f1') is not None:
                    fold_gaps.append(float(s_json['best_train_macro_f1']) - float(vs_metrics_k['MacroF1']))
                elif s_json.get('train_valsel_macro_f1_gap') is not None:
                    fold_gaps.append(float(s_json['train_valsel_macro_f1_gap']))
                if 'elapsed_seconds' in s_json and s_json['elapsed_seconds'] is not None:
                    fold_times.append(float(s_json['elapsed_seconds']))
                if 'peak_gpu_mb' in s_json and s_json['peak_gpu_mb'] is not None:
                    fold_peak_gpus.append(float(s_json['peak_gpu_mb']))
                if total_params is None and 'total_params' in s_json and s_json['total_params'] is not None:
                    total_params = int(s_json['total_params'])
                if named_children_params is None and 'named_children_params' in s_json:
                    named_children_params = s_json['named_children_params']
                if s_json.get('add_weights') is not None:
                    aw = dict(s_json['add_weights'])
                    aw['fold'] = k
                    fold_add_weights.append(aw)
                if isinstance(s_json.get('c5_diagnostics'), dict):
                    c5d = dict(s_json['c5_diagnostics'])
                    c5d['fold'] = k
                    fold_c5_list.append(c5d)

            if (
                compute_conformal
                and has_valsel_npy
                and _is_valid_npy_file(vcal_lbl_path)
                and _is_valid_npy_file(vcal_prb_path)
            ):
                vcal_c = np.load(vcal_lbl_path).flatten()
                vcal_prb = np.load(vcal_prb_path)
                vcal_freq = (vcal_c + 1) if vcal_c.min() == 0 else vcal_c
                fold_split = load_fold_data(k, split_type)
                vs_c = np.load(vsel_lbl_path).flatten()
                vs_freq = (vs_c + 1) if vs_c.min() == 0 else vs_c
                vs_prb = np.load(vsel_prb_path)
                c4_vs = evaluate_all_conformal_methods(
                    valcal_probs=vcal_prb,
                    valcal_labels_1_to_5=vcal_freq,
                    test_probs=vs_prb,
                    test_labels_1_to_5=vs_freq,
                    alpha=0.10,
                    seed=42 + k,
                    test_drug_ids=fold_split.valsel[:, 0].astype(int),
                )
                for c4_name, c4_val in c4_vs.items():
                    fold_valsel_conformal_list[c4_name].append(c4_val)

        if not fold_valsel_list:
            continue

        vm, vs = aggregate_fold_metrics(fold_valsel_list)
        valsel_agg = {'mean': vm, 'std': vs, 'per_fold': fold_valsel_list}

        agg_valsel_c4 = {}
        for c4_name, c4_list in fold_valsel_conformal_list.items():
            if c4_list:
                cm, cs = aggregate_fold_metrics(c4_list)
                agg_valsel_c4[c4_name] = {'mean': cm, 'std': cs}

        if total_params is None and method_name in SLIM_EXPERIMENTS:
            m = _instantiate_model_for_cfg_name(method_name)
            bd = get_named_children_param_breakdown(m)
            total_params = bd['TOTAL']
            named_children_params = bd

        collected[method_name] = {
            'folds': folds_found,
            'valsel': valsel_agg,
            'valsel_conformal': agg_valsel_c4,
            'c5_per_fold': fold_c5_list,
            'add_weights_per_fold': fold_add_weights,
            'total_params': total_params or 0,
            'named_children_params': named_children_params or {},
            'gap_mean': float(np.mean(fold_gaps)) if fold_gaps else float('nan'),
            'gap_std': float(np.std(fold_gaps)) if fold_gaps else float('nan'),
            'fold_gaps': fold_gaps,
            'time_mean': float(np.mean(fold_times)) if fold_times else float('nan'),
            'time_std': float(np.std(fold_times)) if fold_times else float('nan'),
            'fold_times': fold_times,
            'peak_gpu_mean': float(np.mean(fold_peak_gpus)) if fold_peak_gpus else float('nan'),
            'peak_gpu_max': float(np.max(fold_peak_gpus)) if fold_peak_gpus else float('nan'),
            'fold_peak_gpus': fold_peak_gpus,
        }

    return collected


def _ordered_method_names(collected: Dict[str, dict]) -> List[str]:
    ordered = [m for m in ALL_REPORT_ORDER if m in collected]
    extras = sorted([m for m in collected.keys() if m not in ALL_REPORT_ORDER])
    return ordered + extras


def _paired_wilcoxon_p(diffs: List[float]) -> Optional[float]:
    arr = np.asarray(diffs, dtype=np.float64)
    if len(arr) < 6 or np.allclose(arr, 0.0):
        return None
    try:
        _, p_val = stats.wilcoxon(arr)
        return float(p_val)
    except Exception:
        return None


def format_markdown_tables(
    collected: Dict[str, dict],
    split_type: str,
) -> str:
    target_label = "Validation Selection (`val_select`)"
    lines = [f"## Split {split_type} — {target_label} Results (`results_slim`)\n"]
    if not collected:
        lines.append("_No completed folds found._\n")
        return "\n".join(lines)

    method_names = _ordered_method_names(collected)

    # 1. Main classification table
    lines.append(f"### 1. Overall & Ordinal Classification Metrics ({target_label})")
    lines.append(
        "| Method | Folds | Total Params | Macro-F1 | Weighted-F1 | Accuracy | QWK | MAE | Gap (tr - val) | Time/Fold (s) | Peak GPU (MB) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m_name in method_names:
        data = collected[m_name]
        if data['valsel'] is None:
            continue
        folds = data['folds']
        mean = data['valsel']['mean']
        std = data['valsel']['std']
        g_m, g_s = data['gap_mean'], data['gap_std']
        g_str = f"{g_m:+.4f} ± {g_s:.4f}" if not np.isnan(g_m) else "N/A"
        t_m, t_s = data['time_mean'], data['time_std']
        t_str = f"{t_m:.1f} ± {t_s:.1f}" if not np.isnan(t_m) else "N/A"
        p_m, p_max = data['peak_gpu_mean'], data['peak_gpu_max']
        p_str = f"{p_m:.1f} (max {p_max:.1f})" if not np.isnan(p_m) and p_m > 0 else "N/A"
        t_params = data.get('total_params', 0)
        p_cnt_str = f"{t_params:,}" if t_params > 0 else "N/A"
        lines.append(
            f"| `{m_name}` | {len(folds)} | {p_cnt_str} | "
            f"{mean['MacroF1']:.4f} ± {std['MacroF1']:.4f} | "
            f"{mean['WeightedF1']:.4f} ± {std['WeightedF1']:.4f} | "
            f"{mean['Accuracy']:.4f} ± {std['Accuracy']:.4f} | "
            f"{mean['QWK']:.4f} ± {std['QWK']:.4f} | "
            f"{mean['MAE']:.4f} ± {std['MAE']:.4f} | "
            f"{g_str} | {t_str} | {p_str} |"
        )

    # 2. Per-class F1 table
    lines.append(f"\n### 2. Per-Class F1 Breakdown (`c0..c4`, {target_label})")
    lines.append("| Method | F1 | F1_c0 (Very Rare) | F1_c1 (Rare) | F1_c2 (Infrequent) | F1_c3 (Frequent) | F1_c4 (Very Freq) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for m_name in method_names:
        data = collected[m_name]
        src = data['valsel']
        if src is None:
            continue
        mean = src['mean']
        std = src['std']
        lines.append(
            f"| `{m_name}` | {mean['MacroF1']:.4f} | "
            f"{mean['F1_c0']:.4f} ± {std['F1_c0']:.4f} | "
            f"{mean['F1_c1']:.4f} ± {std['F1_c1']:.4f} | "
            f"{mean['F1_c2']:.4f} ± {std['F1_c2']:.4f} | "
            f"{mean['F1_c3']:.4f} ± {std['F1_c3']:.4f} | "
            f"{mean['F1_c4']:.4f} ± {std['F1_c4']:.4f} |"
        )

    # 2b. Per-fold breakdown table
    lines.append(f"\n### 2b. Per-Fold Breakdown ({target_label})")
    lines.append(
        "| Method | Fold | Macro-F1 | Weighted-F1 | Accuracy | QWK | MAE | F1_c0 | F1_c1 | F1_c2 | F1_c3 | F1_c4 | Gap(eval) | Time(s) | Peak GPU (MB) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m_name in method_names:
        data = collected[m_name]
        src = data['valsel']
        if src is None or 'per_fold' not in src:
            continue
        f_gaps = data.get('fold_gaps', [])
        f_times = data.get('fold_times', [])
        f_gpus = data.get('fold_peak_gpus', [])
        for idx, (f_idx, fm) in enumerate(zip(data['folds'], src['per_fold'])):
            gap_v = f"{f_gaps[idx]:+.4f}" if idx < len(f_gaps) else "N/A"
            tim_v = f"{f_times[idx]:.1f}" if idx < len(f_times) else "N/A"
            gpu_v = f"{f_gpus[idx]:.1f}" if idx < len(f_gpus) and f_gpus[idx] > 0 else "N/A"
            lines.append(
                f"| `{m_name}` | {f_idx} | "
                f"{fm['MacroF1']:.4f} | {fm['WeightedF1']:.4f} | {fm['Accuracy']:.4f} | "
                f"{fm['QWK']:.4f} | {fm['MAE']:.4f} | "
                f"{fm['F1_c0']:.4f} | {fm['F1_c1']:.4f} | {fm['F1_c2']:.4f} | "
                f"{fm['F1_c3']:.4f} | {fm['F1_c4']:.4f} | {gap_v} | {tim_v} | {gpu_v} |"
            )

    # 2e-1. Paired fold comparison vs slim_nodec (final model)
    if 'slim_nodec' in collected and len(method_names) >= 2:
        ref_map = {
            int(f_i): float(fm['MacroF1'])
            for f_i, fm in zip(collected['slim_nodec']['folds'], collected['slim_nodec']['valsel']['per_fold'])
        }
        lines.append(
            f"\n### 2d. Paired Fold Comparison vs Final Model `slim_nodec` (Split {split_type}, {target_label})"
        )
        lines.append(
            "| Configuration | Common Folds | Total Params | Config Macro-F1 | `slim_nodec` (Matched Folds) | Paired $\\Delta$ (`Cfg - slim_nodec`) | Wilcoxon $p$ |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for m_name in method_names:
            m_data = collected[m_name]
            f_map = {
                int(f_i): float(fm['MacroF1'])
                for f_i, fm in zip(m_data['folds'], m_data['valsel']['per_fold'])
            }
            common = sorted(set(f_map.keys()) & set(ref_map.keys()))
            if not common:
                continue
            c_vals = [f_map[f] for f in common]
            r_vals = [ref_map[f] for f in common]
            diffs = [f_map[f] - ref_map[f] for f in common]
            p_w = _paired_wilcoxon_p(diffs)
            p_str = f"`{p_w:.4f}`" if p_w is not None else ("`1.0000`" if m_name == 'slim_nodec' else "N/A (<6 folds)")
            lines.append(
                f"| `{m_name}` | {len(common)} (`{common[0]}..{common[-1]}`) | {m_data['total_params']:,} | "
                f"{np.mean(c_vals):.4f} ± {np.std(c_vals):.4f} | "
                f"{np.mean(r_vals):.4f} ± {np.std(r_vals):.4f} | "
                f"{np.mean(diffs):+.4f} ± {np.std(diffs):.4f} | {p_str} |"
            )

    # 2e-2. Paired fold comparison vs cand_A (control with decoders)
    if 'cand_A' in collected and len(method_names) >= 2:
        ref_map_ca = {
            int(f_i): float(fm['MacroF1'])
            for f_i, fm in zip(collected['cand_A']['folds'], collected['cand_A']['valsel']['per_fold'])
        }
        lines.append(
            f"\n### 2e. Paired Fold Comparison vs `cand_A` & 1-Fold-Std Drop Check (Split {split_type}, {target_label})"
        )
        lines.append(
            "| Configuration | Common Folds | Total Params | Config Macro-F1 | Paired $\\Delta$ vs `cand_A` (Mean ± Std) | Wilcoxon $p$ | Drop $> 1\\,\\text{Std}_{\\text{cand\\_A}}$? |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for m_name in method_names:
            m_data = collected[m_name]
            f_map = {
                int(f_i): float(fm['MacroF1'])
                for f_i, fm in zip(m_data['folds'], m_data['valsel']['per_fold'])
            }
            common = sorted(set(f_map.keys()) & set(ref_map_ca.keys()))
            if not common:
                continue
            c_vals = [f_map[f] for f in common]
            r_vals = [ref_map_ca[f] for f in common]
            ref_std = float(np.std(r_vals))
            diffs = [f_map[f] - ref_map_ca[f] for f in common]
            d_mean = float(np.mean(diffs))
            d_std = float(np.std(diffs))
            p_w = _paired_wilcoxon_p(diffs)
            p_str = f"`{p_w:.4f}`" if p_w is not None else ("`1.0000`" if m_name == 'cand_A' else "N/A (<6 folds)")
            drops_1std = d_mean < -ref_std
            flag_str = (
                f"**CẢNH BÁO: TỤT > 1 Std** ($\\Delta={d_mean:+.4f} < -{ref_std:.4f}$)"
                if drops_1std
                else f"KHÔNG ($\\Delta={d_mean:+.4f} \\ge -{ref_std:.4f}$)"
            )
            lines.append(
                f"| `{m_name}` | {len(common)} | {m_data['total_params']:,} | "
                f"{np.mean(c_vals):.4f} ± {np.std(c_vals):.4f} | "
                f"{d_mean:+.4f} ± {d_std:.4f} | {p_str} | {flag_str} |"
            )

    # 2f. Learned EN-add View Weights w_v (mean across folds)
    methods_with_weights = [m for m in method_names if collected[m].get('add_weights_per_fold')]
    if methods_with_weights:
        lines.append(
            f"\n### 2f. Learned `EN-add` View Weights $w_v = \\text{{softmax}}(\\text{{logit}}_v)$ (Split {split_type}, Mean ± Std Across Folds)"
        )
        for m_name in methods_with_weights:
            aw_list = collected[m_name]['add_weights_per_fold']
            d_names = aw_list[0].get('drug_view_names', [f'D{i+1}' for i in range(11)])
            s_names = aw_list[0].get('se_view_names', [f'S{i+1}' for i in range(5)])

            d_logits_arr = np.asarray([aw['drug_logits'] for aw in aw_list], dtype=np.float64)
            d_full_arr = np.asarray([aw['drug_weights_full'] for aw in aw_list], dtype=np.float64)
            d_cold_arr = np.asarray([aw['drug_weights_coldstart'] for aw in aw_list], dtype=np.float64)
            has_emp_d = all('valsel_empirical_drug_weights' in aw for aw in aw_list)
            d_emp_arr = (
                np.asarray([aw['valsel_empirical_drug_weights'] for aw in aw_list], dtype=np.float64)
                if has_emp_d
                else None
            )

            s_logits_arr = np.asarray([aw['se_logits'] for aw in aw_list], dtype=np.float64)
            s_full_arr = np.asarray([aw['se_weights_full'] for aw in aw_list], dtype=np.float64)
            has_emp_s = all('valsel_empirical_se_weights' in aw for aw in aw_list)
            s_emp_arr = (
                np.asarray([aw['valsel_empirical_se_weights'] for aw in aw_list], dtype=np.float64)
                if has_emp_s
                else None
            )

            lines.append(f"\n#### `{m_name}` (Split {split_type}, {len(aw_list)} folds) — Drug Views (`D1..D11`, Uniform = `1/11 = 0.0909`, Cold-Start Uniform = `1/9 = 0.1111`)")
            lines.append(
                "| Drug View | Learned Logit (Mean ± Std) | Full-View $w_v^{\\text{full}}$ (Mean ± Std) | Cold-Start $w_v^{\\text{cold}}$ (`D9,D10=0`) | Empirical `val_select` $\\bar{w}_v$ |"
            )
            lines.append("|---|---:|---:|---:|---:|")
            for idx, v_name in enumerate(d_names):
                lg_m, lg_s = float(np.mean(d_logits_arr[:, idx])), float(np.std(d_logits_arr[:, idx]))
                wf_m, wf_s = float(np.mean(d_full_arr[:, idx])), float(np.std(d_full_arr[:, idx]))
                wc_m, wc_s = float(np.mean(d_cold_arr[:, idx])), float(np.std(d_cold_arr[:, idx]))
                if d_emp_arr is not None:
                    we_m, we_s = float(np.mean(d_emp_arr[:, idx])), float(np.std(d_emp_arr[:, idx]))
                    we_str = f"{we_m:.4f} ± {we_s:.4f}"
                else:
                    we_str = "N/A"
                lines.append(
                    f"| `{v_name}` | {lg_m:+.4f} ± {lg_s:.4f} | {wf_m:.4f} ± {wf_s:.4f} | {wc_m:.4f} ± {wc_s:.4f} | {we_str} |"
                )

            lines.append(f"\n#### `{m_name}` (Split {split_type}, {len(aw_list)} folds) — Side-Effect Views (`S1..S{len(s_names)}`)")
            lines.append(
                "| SE View | Learned Logit (Mean ± Std) | Full-View $w_u^{\\text{full}}$ (Mean ± Std) | Empirical `val_select` $\\bar{w}_u$ |"
            )
            lines.append("|---|---:|---:|---:|")
            for idx, u_name in enumerate(s_names):
                lg_m, lg_s = float(np.mean(s_logits_arr[:, idx])), float(np.std(s_logits_arr[:, idx]))
                wf_m, wf_s = float(np.mean(s_full_arr[:, idx])), float(np.std(s_full_arr[:, idx]))
                if s_emp_arr is not None:
                    we_m, we_s = float(np.mean(s_emp_arr[:, idx])), float(np.std(s_emp_arr[:, idx]))
                    we_str = f"{we_m:.4f} ± {we_s:.4f}"
                else:
                    we_str = "N/A"
                lines.append(
                    f"| `{u_name}` | {lg_m:+.4f} ± {lg_s:.4f} | {wf_m:.4f} ± {wf_s:.4f} | {we_str} |"
                )

    # 3. C4 Conformal table
    has_any_c4 = any(bool(collected[m].get('valsel_conformal')) for m in method_names)
    if has_any_c4:
        lines.append(f"\n### 3. C4 Ordinal Conformal Prediction ($\\alpha = 0.10$, Target Coverage = 90%, {target_label})")
        lines.append(
            "| Method | Folds | C4 Variant | Pair Coverage | Drug Cov (Mean ± Std) | Drug Cov < 80% | Avg Set Size | Singleton Rate | Contiguity | Cov_c0 | Cov_c1 | Cov_c2 | Cov_c3 | Cov_c4 |"
        )
        lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for m_name in method_names:
            c4_dict = collected[m_name].get('valsel_conformal', {})
            n_f = len(collected[m_name]['folds'])
            for c4_name in ['LAC', 'APS', 'Ordinal_LAC', 'Ordinal_APS']:
                if c4_name not in c4_dict:
                    continue
                cm = c4_dict[c4_name]['mean']
                cs = c4_dict[c4_name]['std']
                dc_m = cm.get('drug_coverage_mean', float('nan'))
                dc_s = cm.get('drug_coverage_std', float('nan'))
                dc_lt80 = cm.get('drug_coverage_lt80_rate', float('nan'))
                dc_cell = f"{dc_m:.4f} ± {dc_s:.4f}" if not np.isnan(dc_m) else "N/A"
                lt80_cell = f"{dc_lt80:.4f} ± {cs.get('drug_coverage_lt80_rate', 0.0):.4f}" if not np.isnan(dc_lt80) else "N/A"
                lines.append(
                    f"| `{m_name}` | {n_f} | `{c4_name}` | "
                    f"{cm['empirical_coverage']:.4f} ± {cs['empirical_coverage']:.4f} | "
                    f"{dc_cell} | {lt80_cell} | "
                    f"{cm['avg_set_size']:.3f} ± {cs['avg_set_size']:.3f} | "
                    f"{cm['singleton_rate']:.4f} ± {cs['singleton_rate']:.4f} | "
                    f"{cm['contiguity_rate']:.4f} | "
                    f"{cm['coverage_c0']:.4f} | {cm['coverage_c1']:.4f} | "
                    f"{cm['coverage_c2']:.4f} | {cm['coverage_c3']:.4f} | "
                    f"{cm['coverage_c4']:.4f} |"
                )

    lines.append("")
    return "\n".join(lines)


def format_joint_selection_section(
    collected_by_split: Dict[str, Dict[str, dict]],
) -> str:
    """
    Build the joint Split B & Split A comparison table across identical common folds, together with
    the approved Stage 5 selection decision (`slim_nodec` final model; `slim_all_cnnshared` max-compression variant).
    """
    if 'B' not in collected_by_split or 'A' not in collected_by_split:
        return ""
    col_b = collected_by_split['B']
    col_a = collected_by_split['A']
    if 'cand_A' not in col_b or 'cand_A' not in col_a:
        return ""

    lines = [
        "## Joint Cross-Split Summary (Split B & Split A, Fold-Matched `val_select`)\n",
        "| Configuration | Common Folds (B / A) | Total Params | vs `cand_A` | Split B Macro-F1 | Split B Paired $\\Delta$ vs `cand_A` | Split A Macro-F1 | Split A Paired $\\Delta$ vs `cand_A` | Role / Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    ref_b_fmap = {int(f): float(m['MacroF1']) for f, m in zip(col_b['cand_A']['folds'], col_b['cand_A']['valsel']['per_fold'])}
    ref_a_fmap = {int(f): float(m['MacroF1']) for f, m in zip(col_a['cand_A']['folds'], col_a['cand_A']['valsel']['per_fold'])}
    base_params = int(col_b['cand_A']['total_params'])

    role_labels = {
        'slim_nodec': '**FINAL MODEL** (`-36.51%`, no decoders)',
        'cand_A': 'Control (`+decoders`, isolates nodec effect)',
        'slim_nodec_addw': 'Stage 5 ablation (`+add_weights`, 3 folds)',
        'slim_all': 'Stage 5 ablation (`+shared_emb EN-con`, 3 folds)',
        'slim_all_cnnshared': '**Max-Compression Variant** (`-79.12%`)',
        'slim_nodec_loo_no_vdrop': 'LOO on `slim_nodec`: drop `vdrop`',
        'slim_nodec_loo_no_C2': 'LOO on `slim_nodec`: drop `C2`',
        'slim_nodec_loo_no_C5': 'LOO on `slim_nodec`: drop `C5`',
        'slim_nodec_loo_no_all': 'LOO on `slim_nodec`: drop `vdrop+C2+C5`',
        'MSSF_v2_selfcorr': 'Base MSSF (`Protocol v2 + selfcorr`)',
    }

    for m_name in ALL_REPORT_ORDER:
        if m_name not in col_b or m_name not in col_a:
            continue
        tot_p = int(col_b[m_name]['total_params'])
        delta_pct = (float(tot_p) / float(base_params) - 1.0) * 100.0

        b_fmap = {int(f): float(m['MacroF1']) for f, m in zip(col_b[m_name]['folds'], col_b[m_name]['valsel']['per_fold'])}
        b_common = sorted(set(b_fmap.keys()) & set(ref_b_fmap.keys()))
        b_vals = [b_fmap[f] for f in b_common]
        b_diffs = [b_fmap[f] - ref_b_fmap[f] for f in b_common]

        a_fmap = {int(f): float(m['MacroF1']) for f, m in zip(col_a[m_name]['folds'], col_a[m_name]['valsel']['per_fold'])}
        a_common = sorted(set(a_fmap.keys()) & set(ref_a_fmap.keys()))
        a_vals = [a_fmap[f] for f in a_common]
        a_diffs = [a_fmap[f] - ref_a_fmap[f] for f in a_common]

        if not b_common or not a_common:
            continue

        lines.append(
            f"| `{m_name}` | `{len(b_common)} / {len(a_common)}` | {tot_p:,} | `{delta_pct:+.2f}%` | "
            f"{np.mean(b_vals):.4f} ± {np.std(b_vals):.4f} | {np.mean(b_diffs):+.4f} ± {np.std(b_diffs):.4f} | "
            f"{np.mean(a_vals):.4f} ± {np.std(a_vals):.4f} | {np.mean(a_diffs):+.4f} ± {np.std(a_diffs):.4f} | "
            f"{role_labels.get(m_name, 'Ablation')} |"
        )

    lines.append(
        "\n> **Quyết định chọn mô hình cuối Giai đoạn 5 (Đã duyệt)**:\n"
        "> - **Mô hình cuối (`Final Model`)**: **`slim_nodec`** (**`6,803,109`** tham số, **`-36.51%`** so với `cand_A`). "
        "Quy tắc tiền định chọn cấu hình gọn không tụt quá $1\\,\\text{Std}$ giữa các fold; `slim_nodec` vừa gọn hơn `cand_A` "
        "vừa **không tụt mà còn tăng** ở cả hai split trên Stage 5 (`0.4223` vs `0.4123` ở Split B; `0.6475` vs `0.6435` ở Split A), "
        "nên trội hơn `slim_all_cnnshared` theo cả tiêu chí độ chính xác lẫn tính đơn giản kiến trúc.\n"
        "> - **Biến thể nén tối đa (`Max-Compression Ablation`)**: **`slim_all_cnnshared`** (**`2,237,621`** tham số, **`-79.12%`** so với `cand_A`) "
        "được giữ trong bảng ablation để minh họa giới hạn nén tham số cực đại.\n"
        "> - **Ghi chú diễn giải `slim_all` vs `slim_all_cnnshared`**: Chênh lệch `0.0155` Macro-F1 trên 3 fold ở Split B (`0.3832` vs `0.3987`) "
        "nằm trong biên nhiễu giữa các fold (3 fold), nên được mô tả là *biến thiên không đơn điệu theo số tham số, nhiều khả năng do nhiễu mẫu nhỏ*, không diễn giải là hiệu ứng hồi phục mạnh.\n"
    )
    return "\n".join(lines)


def format_fold_matched_baseline_section(
    slim_by_split: Dict[str, Dict[str, dict]],
    base_by_split: Dict[str, Dict[str, dict]],
    target_folds: List[int],
) -> str:
    """
    Build a strictly fold-matched comparison table on `target_folds` (e.g., `[0, 1, 2]` or `[0..9]`),
    ensuring every row in the main table is evaluated on the exact same `target_folds`.
    `HSTrans` / `HSTrans_thr` (3 folds `0..2`) is reported in the main table ONLY when `target_folds == [0, 1, 2]`,
    and is separated into a dedicated 3-fold footnote block whenever `len(target_folds) > 3`.
    """
    fold_tag = f"Folds `{target_folds[0]}–{target_folds[-1]}` ($K={len(target_folds)}$)"
    lines = [f"## Fold-Matched Comparison vs External Baselines ({fold_tag}, `val_select`)\n"]
    target_set = set(target_folds)

    for split_type in ['B', 'A']:
        if split_type not in slim_by_split and split_type not in base_by_split:
            continue
        col_s = slim_by_split.get(split_type, {})
        col_b = base_by_split.get(split_type, {})

        lines.append(f"### Split {split_type} — Fold-Matched Table ({fold_tag}, `val_select`)")
        lines.append(
            "| Group | Method | Exact Folds | Macro-F1 | Weighted-F1 | Accuracy | QWK | MAE | Gap (tr - val) | Time/Fold (s) | Paired $\\Delta$ vs `slim_nodec` | Wilcoxon $p$ |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

        ref_fmap: Dict[int, float] = {}
        if 'slim_nodec' in col_s:
            s_nd = col_s['slim_nodec']
            f_all = {int(f): float(m['MacroF1']) for f, m in zip(s_nd['folds'], s_nd['valsel']['per_fold'])}
            if target_set.issubset(set(f_all.keys())):
                ref_fmap = {f: f_all[f] for f in target_folds}

        def _emit_row(group_lbl: str, m_name: str, m_entry: dict) -> Optional[str]:
            f_idx_map = {int(f): idx for idx, f in enumerate(m_entry['folds'])}
            if not target_set.issubset(set(f_idx_map.keys())):
                return None
            sel_metrics = [m_entry['valsel']['per_fold'][f_idx_map[f]] for f in target_folds]
            mm, ms = aggregate_fold_metrics(sel_metrics)

            f_gaps = m_entry.get('fold_gaps', [])
            if len(f_gaps) == len(m_entry['folds']):
                sel_gaps = [f_gaps[f_idx_map[f]] for f in target_folds]
                g_str = f"{np.mean(sel_gaps):+.4f} ± {np.std(sel_gaps):.4f}"
            else:
                g_str = "N/A"

            f_times = m_entry.get('fold_times', [])
            if len(f_times) == len(m_entry['folds']):
                sel_times = [f_times[f_idx_map[f]] for f in target_folds]
                t_str = f"{np.mean(sel_times):.1f} ± {np.std(sel_times):.1f}"
            else:
                t_str = "N/A"

            if ref_fmap:
                diffs = [float(sel_metrics[i]['MacroF1']) - ref_fmap[f] for i, f in enumerate(target_folds)]
                d_str = f"{np.mean(diffs):+.4f} ± {np.std(diffs):.4f}"
                pw = _paired_wilcoxon_p(diffs)
                p_str = f"`{pw:.4f}`" if pw is not None else ("`1.0000`" if m_name == 'slim_nodec' else "N/A (<6 folds)")
            else:
                d_str = "N/A"
                p_str = "N/A"

            return (
                f"| {group_lbl} | `{m_name}` | `{target_folds[0]}..{target_folds[-1]}` ({len(target_folds)}) | "
                f"{mm['MacroF1']:.4f} ± {ms['MacroF1']:.4f} | "
                f"{mm['WeightedF1']:.4f} ± {ms['WeightedF1']:.4f} | "
                f"{mm['Accuracy']:.4f} ± {ms['Accuracy']:.4f} | "
                f"{mm['QWK']:.4f} ± {ms['QWK']:.4f} | "
                f"{mm['MAE']:.4f} ± {ms['MAE']:.4f} | "
                f"{g_str} | {t_str} | {d_str} | {p_str} |"
            )

        for m_name in ALL_REPORT_ORDER:
            if m_name in col_s:
                row = _emit_row("Proposed / Ablation", m_name, col_s[m_name])
                if row:
                    lines.append(row)

        ordered_bases = [b for b in BASELINE_REPORT_ORDER if b in col_b] + sorted(
            [b for b in col_b.keys() if b not in BASELINE_REPORT_ORDER and b not in col_s]
        )
        hstrans_partial_notes: List[Tuple[str, dict]] = []
        for b_name in ordered_bases:
            if b_name.startswith('HSTrans') and len(target_folds) > 3:
                hstrans_partial_notes.append((b_name, col_b[b_name]))
                continue
            row = _emit_row("External Baseline", b_name, col_b[b_name])
            if row:
                lines.append(row)
            elif b_name.startswith('HSTrans'):
                hstrans_partial_notes.append((b_name, col_b[b_name]))

        if hstrans_partial_notes:
            lines.append(
                "\n> **Chú thích riêng cho `HSTrans` (3 fold `0–2` do chi phí `~7.900 giây/fold` — KHÔNG gộp chung vào cột 10 fold)**:"
            )
            for h_name, h_entry in hstrans_partial_notes:
                hm = h_entry['valsel']['mean']
                hs = h_entry['valsel']['std']
                hg = h_entry['gap_mean']
                ht = h_entry['time_mean']
                lines.append(
                    f"> - `{h_name}` (Folds `{h_entry['folds']}`, $K={len(h_entry['folds'])}$): "
                    f"Macro-F1 = `{hm['MacroF1']:.4f} ± {hs['MacroF1']:.4f}`, "
                    f"QWK = `{hm['QWK']:.4f} ± {hs['QWK']:.4f}`, "
                    f"**MAE = `{hm['MAE']:.4f} ± {hs['MAE']:.4f}`**, "
                    f"**Gap(eval) = `{hg:+.4f}`**, Time = `{ht:.1f}s/fold`."
                )
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Summarize Stage 5 & 10-fold (mssf_slim) results directly from saved .npy files.')
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_SLIM_DIR)
    parser.add_argument('--drive_dir', type=str, default=None)
    parser.add_argument('--baseline_results_dir', type=str, default=None)
    parser.add_argument('--baseline_drive_dir', type=str, default=None)
    parser.add_argument('--splits', type=str, nargs='+', default=['B', 'A'], choices=['A', 'B'])
    parser.add_argument('--folds', type=int, nargs='*', default=None)
    parser.add_argument('--methods', type=str, nargs='*', default=None)
    parser.add_argument('--no_conformal', action='store_true')
    parser.add_argument('--out_md', type=str, default=None)
    parser.add_argument('--out_raw', type=str, default=None)
    args = parser.parse_args()

    cmd_str = "python -m mssf_slim.summarize_results " + " ".join(sys.argv[1:])
    header_lines = [
        f"SCRIPT: mssf_slim/summarize_results.py",
        f"COMMAND: {cmd_str}",
        "=" * 110,
        "# `mssf_slim` Unified Evaluation Summary (Computed Directly from `.npy`)\n",
    ]
    sections = ["\n".join(header_lines)]

    if args.drive_dir:
        for sp in args.splits:
            _sync_from_drive_if_available(args.results_dir, args.drive_dir, sp, only_methods=args.methods)

    sections.append(format_architecture_and_params_section(args.results_dir, only_methods=args.methods))

    collected_by_split: Dict[str, Dict[str, dict]] = {}
    for split_type in args.splits:
        collected = collect_method_results(
            args.results_dir,
            split_type,
            only_folds=args.folds,
            only_methods=args.methods,
            drive_dir=args.drive_dir,
            compute_conformal=not args.no_conformal,
        )
        collected_by_split[split_type] = collected
        sections.append(format_markdown_tables(collected, split_type))

    joint_sec = format_joint_selection_section(collected_by_split)
    if joint_sec:
        sections.append(joint_sec)

    if args.baseline_results_dir or args.baseline_drive_dir:
        b_local = args.baseline_results_dir or os.path.join(os.path.dirname(args.results_dir), 'results_v2_cache')
        base_by_split: Dict[str, Dict[str, dict]] = {}
        for split_type in args.splits:
            base_by_split[split_type] = collect_method_results(
                b_local,
                split_type,
                only_folds=args.folds,
                only_methods=None,
                drive_dir=args.baseline_drive_dir,
                compute_conformal=False,
            )
        target_folds = sorted(args.folds) if args.folds else list(range(10))
        sections.append(format_fold_matched_baseline_section(collected_by_split, base_by_split, target_folds))

    full_report = "\n".join(sections)
    print(full_report)

    if args.out_md:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_md)), exist_ok=True)
        with open(args.out_md, 'w', encoding='utf-8') as f:
            f.write(full_report)
        print(f"Saved Markdown report to {args.out_md}")

    if args.out_raw:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_raw)), exist_ok=True)
        with open(args.out_raw, 'w', encoding='utf-8') as f:
            f.write(full_report + "\n")
        print(f"Saved verbatim raw output to {args.out_raw}")


if __name__ == '__main__':
    main()
