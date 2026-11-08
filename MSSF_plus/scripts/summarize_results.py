"""
Aggregate results from `results_v2/`, compute mean +/- std across available folds,
perform paired Wilcoxon signed-rank tests against `MSSF_v2`, and generate Markdown
tables for both classification metrics (`test` or `valsel`) and C4 ordinal conformal
prediction metrics (including per-drug coverage mean, std, and <80% rate in Split B).
"""

import os
import sys
import glob
import json
import shutil
import argparse
from typing import Dict, List, Optional

import numpy as np

from MSSF_plus.protocol import DEFAULT_RESULTS_V2_DIR, load_fold_data, wilcoxon_compare
from MSSF_plus.metrics import compute_multiclass_metrics, aggregate_fold_metrics
from MSSF_plus.conformal import evaluate_all_conformal_methods


def _sync_from_drive_if_available(
    results_dir: str,
    drive_dir: Optional[str],
    split_type: str,
    only_methods: Optional[List[str]] = None,
) -> None:
    """Copy completed fold files from drive_dir/*_{split_type} into results_dir if missing locally."""
    if not drive_dir or not os.path.isdir(drive_dir):
        return
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
            if os.path.isfile(sp) and not os.path.exists(dp):
                shutil.copy2(sp, dp)


def collect_method_results(
    results_dir: str,
    split_type: str,
    only_folds: list = None,
    only_methods: Optional[List[str]] = None,
    drive_dir: Optional[str] = None,
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
        # Exclude any deprecated softquant folders if present on disk
        if 'softquant' in method_name.lower():
            continue
        if only_methods is not None and method_name not in only_methods:
            continue

        fold_metrics_list = []
        fold_valsel_list = []
        fold_gaps = []
        fold_times = []
        fold_c5_list = []
        fold_bvi_list = []
        fold_conformal_list = {k: [] for k in ['LAC', 'APS', 'Ordinal_LAC', 'Ordinal_APS']}
        fold_valsel_conformal_list = {k: [] for k in ['LAC', 'APS', 'Ordinal_LAC', 'Ordinal_APS']}
        folds_found = []

        for k in range(10):
            if only_folds is not None and k not in only_folds:
                continue

            lbl_path = os.path.join(mdir, f'fold_{k}_labels.npy')
            prd_path = os.path.join(mdir, f'fold_{k}_preds.npy')
            prb_path = os.path.join(mdir, f'fold_{k}_probs.npy')
            vcal_lbl_path = os.path.join(mdir, f'fold_{k}_valcal_labels.npy')
            vcal_prb_path = os.path.join(mdir, f'fold_{k}_valcal_probs.npy')
            vsel_lbl_path = os.path.join(mdir, f'fold_{k}_valsel_labels.npy')
            vsel_prb_path = os.path.join(mdir, f'fold_{k}_valsel_probs.npy')
            vsel_prd_path = os.path.join(mdir, f'fold_{k}_valsel_preds.npy')
            sigma_path = os.path.join(mdir, f'fold_{k}_sigma.npy')
            tr_sigma_path = os.path.join(mdir, f'fold_{k}_train_sigma.npy')
            no_d9d10_sigma_path = os.path.join(mdir, f'fold_{k}_sigma_no_d9d10.npy')
            sum_path = os.path.join(mdir, f'fold_{k}_summary.json')

            has_test = os.path.exists(lbl_path) and os.path.exists(prd_path)
            has_valsel_npy = os.path.exists(vsel_lbl_path) and (
                os.path.exists(vsel_prb_path) or os.path.exists(vsel_prd_path)
            )
            has_valsel = has_valsel_npy or os.path.exists(sum_path)
            if not (has_test or has_valsel):
                continue

            folds_found.append(k)

            if has_test:
                y_true_freq = np.load(lbl_path).flatten()
                y_pred_freq = np.load(prd_path).flatten()
                y_true_c = np.clip(np.round(y_true_freq), 1, 5).astype(int) - 1
                y_pred_c = np.clip(np.round(y_pred_freq), 1, 5).astype(int) - 1
                y_probs = np.load(prb_path) if os.path.exists(prb_path) else None

                m = compute_multiclass_metrics(y_true_c, y_pred_c, probs=y_probs)
                fold_metrics_list.append(m)
            else:
                y_true_freq = None
                y_probs = None

            # Primary source of truth for val_select metrics: saved .npy files on disk
            vs_metrics_k = None
            if has_valsel_npy:
                vs_raw = np.load(vsel_lbl_path).flatten()
                vs_c = np.clip(np.round(vs_raw), 1, 5).astype(int) - 1 if vs_raw.min() >= 1 else vs_raw.astype(int)
                if os.path.exists(vsel_prb_path):
                    vs_prb = np.load(vsel_prb_path)
                    vs_prd = np.argmax(vs_prb, axis=1)
                    vs_metrics_k = compute_multiclass_metrics(vs_c, vs_prd, probs=vs_prb)
                elif os.path.exists(vsel_prd_path):
                    vs_pred_raw = np.load(vsel_prd_path).flatten()
                    vs_prd = np.clip(np.round(vs_pred_raw), 1, 5).astype(int) - 1
                    vs_metrics_k = compute_multiclass_metrics(vs_c, vs_prd, probs=None)
                fold_valsel_list.append(vs_metrics_k)

            bvi_k = None
            if os.path.exists(sum_path):
                with open(sum_path, 'r', encoding='utf-8') as sf:
                    s_json = json.load(sf)
                if vs_metrics_k is None and 'valsel_metrics' in s_json and s_json['valsel_metrics'] is not None:
                    vs_metrics_k = s_json['valsel_metrics']
                    fold_valsel_list.append(vs_metrics_k)
                if 'best_train_macro_f1' in s_json and s_json['best_train_macro_f1'] is not None and vs_metrics_k is not None:
                    fold_gaps.append(float(s_json['best_train_macro_f1']) - float(vs_metrics_k['MacroF1']))
                elif 'train_valsel_macro_f1_gap' in s_json and s_json['train_valsel_macro_f1_gap'] is not None:
                    fold_gaps.append(float(s_json['train_valsel_macro_f1_gap']))
                if 'elapsed_seconds' in s_json:
                    fold_times.append(float(s_json['elapsed_seconds']))
                if 'c5_diagnostics' in s_json and isinstance(s_json['c5_diagnostics'], dict):
                    c5d = dict(s_json['c5_diagnostics'])
                    c5d['fold'] = k
                    fold_c5_list.append(c5d)
                if 'bvi_diagnostics' in s_json and isinstance(s_json['bvi_diagnostics'], dict):
                    bvi_k = dict(s_json['bvi_diagnostics'])
                    bvi_k['fold'] = k

            if os.path.exists(sigma_path):
                sig_arr = np.load(sigma_path).flatten()
                if bvi_k is None:
                    bvi_k = {'fold': k}
                bvi_k['valsel_sigma_mean'] = float(np.mean(sig_arr))
                bvi_k['valsel_sigma_std'] = float(np.std(sig_arr))
                if os.path.exists(tr_sigma_path):
                    tr_sig_arr = np.load(tr_sigma_path).flatten()
                    bvi_k['train_sigma_mean'] = float(np.mean(tr_sig_arr))
                    bvi_k['train_sigma_std'] = float(np.std(tr_sig_arr))
                    if split_type == 'B':
                        bvi_k['sigma_has_d9d10_mean'] = float(np.mean(tr_sig_arr))
                        bvi_k['sigma_missing_d9d10_mean'] = float(np.mean(sig_arr))
                if split_type == 'A':
                    bvi_k['sigma_has_d9d10_mean'] = float(np.mean(sig_arr))
                    if os.path.exists(no_d9d10_sigma_path):
                        no_d_sig = np.load(no_d9d10_sigma_path).flatten()
                        bvi_k['sigma_missing_d9d10_mean'] = float(np.mean(no_d_sig))
            if bvi_k is not None:
                fold_bvi_list.append(bvi_k)

            if os.path.exists(vcal_lbl_path) and os.path.exists(vcal_prb_path):
                vcal_c = np.load(vcal_lbl_path).flatten()
                vcal_prb = np.load(vcal_prb_path)
                vcal_freq = (vcal_c + 1) if vcal_c.min() == 0 else vcal_c
                fold_split = load_fold_data(k, split_type)

                if os.path.exists(vsel_lbl_path) and os.path.exists(vsel_prb_path):
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

                if y_probs is not None and y_true_freq is not None:
                    test_drug_ids = fold_split.test[:, 0].astype(int)
                    c4 = evaluate_all_conformal_methods(
                        valcal_probs=vcal_prb,
                        valcal_labels_1_to_5=vcal_freq,
                        test_probs=y_probs,
                        test_labels_1_to_5=y_true_freq,
                        alpha=0.10,
                        seed=42 + k,
                        test_drug_ids=test_drug_ids,
                    )
                    for c4_name, c4_val in c4.items():
                        fold_conformal_list[c4_name].append(c4_val)

        if not (fold_metrics_list or fold_valsel_list):
            continue

        cls_agg = None
        if fold_metrics_list:
            cls_mean, cls_std = aggregate_fold_metrics(fold_metrics_list)
            cls_agg = {'mean': cls_mean, 'std': cls_std, 'per_fold': fold_metrics_list}

        valsel_agg = None
        if fold_valsel_list:
            vm, vs = aggregate_fold_metrics(fold_valsel_list)
            valsel_agg = {'mean': vm, 'std': vs, 'per_fold': fold_valsel_list}

        agg_c4 = {}
        for c4_name, c4_list in fold_conformal_list.items():
            if c4_list:
                cm, cs = aggregate_fold_metrics(c4_list)
                agg_c4[c4_name] = {'mean': cm, 'std': cs}

        agg_valsel_c4 = {}
        for c4_name, c4_list in fold_valsel_conformal_list.items():
            if c4_list:
                cm, cs = aggregate_fold_metrics(c4_list)
                agg_valsel_c4[c4_name] = {'mean': cm, 'std': cs}

        collected[method_name] = {
            'folds': folds_found,
            'cls': cls_agg,
            'valsel': valsel_agg,
            'conformal': agg_c4,
            'valsel_conformal': agg_valsel_c4,
            'c5_per_fold': fold_c5_list,
            'bvi_per_fold': fold_bvi_list,
            'gap_mean': float(np.mean(fold_gaps)) if fold_gaps else float('nan'),
            'gap_std': float(np.std(fold_gaps)) if fold_gaps else float('nan'),
            'fold_gaps': fold_gaps,
            'time_mean': float(np.mean(fold_times)) if fold_times else float('nan'),
            'time_std': float(np.std(fold_times)) if fold_times else float('nan'),
        }

    return collected


def format_markdown_tables(
    results_dir: str,
    collected: Dict[str, dict],
    split_type: str,
    ref_method: str = 'MSSF_v2',
    eval_target: str = 'test',
) -> str:
    target_label = "Validation Selection (`val_select`)" if eval_target == 'valsel' else "Test (`test`)"
    lines = [f"## Split {split_type} — {target_label} Results (`results_v2`)\n"]
    if not collected:
        lines.append("_No completed folds found._\n")
        return "\n".join(lines)

    # 1. Main classification table
    lines.append(f"### 1. Overall & Ordinal Classification Metrics ({target_label})")
    if eval_target == 'valsel':
        lines.append(
            "| Method | Folds | Macro-F1 | Weighted-F1 | Accuracy | QWK | MAE | Gap (tr - val) | Time/Fold (s) |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for m_name, data in sorted(collected.items()):
            if data['valsel'] is None:
                continue
            folds = data['folds']
            mean = data['valsel']['mean']
            std = data['valsel']['std']
            g_m, g_s = data['gap_mean'], data['gap_std']
            g_str = f"{g_m:+.4f} ± {g_s:.4f}" if not np.isnan(g_m) else "N/A"
            t_m, t_s = data['time_mean'], data['time_std']
            t_str = f"{t_m:.1f} ± {t_s:.1f}" if not np.isnan(t_m) else "N/A"
            lines.append(
                f"| `{m_name}` | {len(folds)} | "
                f"{mean['MacroF1']:.4f} ± {std['MacroF1']:.4f} | "
                f"{mean['WeightedF1']:.4f} ± {std['WeightedF1']:.4f} | "
                f"{mean['Accuracy']:.4f} ± {std['Accuracy']:.4f} | "
                f"{mean['QWK']:.4f} ± {std['QWK']:.4f} | "
                f"{mean['MAE']:.4f} ± {std['MAE']:.4f} | "
                f"{g_str} | {t_str} |"
            )
    else:
        lines.append(
            "| Method | Folds | Macro-F1 | Weighted-F1 | Accuracy | QWK | MAE | Macro-AUPR | p-val (Macro-F1) |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

        for m_name, data in sorted(collected.items()):
            if data['cls'] is None:
                continue
            folds = data['folds']
            mean = data['cls']['mean']
            std = data['cls']['std']

            p_str = "ref" if m_name == ref_method else "N/A"
            if m_name != ref_method and ref_method in collected:
                w = wilcoxon_compare(
                    results_dir=results_dir,
                    method_candidate=m_name,
                    method_reference=ref_method,
                    split_type=split_type,
                    metric_keys=['MacroF1'],
                )
                mf1_w = w.get('metrics', {}).get('MacroF1', {})
                p_val = mf1_w.get('wilcoxon_p', float('nan'))
                if not np.isnan(p_val):
                    p_str = f"{p_val:.4f}"

            aupr_m = mean.get('MacroAUPR', float('nan'))
            aupr_s = std.get('MacroAUPR', float('nan'))
            aupr_cell = f"{aupr_m:.4f} ± {aupr_s:.4f}" if not np.isnan(aupr_m) else "N/A"

            lines.append(
                f"| `{m_name}` | {len(folds)} | "
                f"{mean['MacroF1']:.4f} ± {std['MacroF1']:.4f} | "
                f"{mean['WeightedF1']:.4f} ± {std['WeightedF1']:.4f} | "
                f"{mean['Accuracy']:.4f} ± {std['Accuracy']:.4f} | "
                f"{mean['QWK']:.4f} ± {std['QWK']:.4f} | "
                f"{mean['MAE']:.4f} ± {std['MAE']:.4f} | "
                f"{aupr_cell} | {p_str} |"
            )

    # 2. Per-class F1 table
    lines.append(f"\n### 2. Per-Class F1 Breakdown (`c0..c4`, {target_label})")
    lines.append("| Method | F1_c0 (Very Rare) | F1_c1 (Rare) | F1_c2 (Infrequent) | F1_c3 (Frequent) | F1_c4 (Very Freq) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for m_name, data in sorted(collected.items()):
        src = data['valsel'] if eval_target == 'valsel' else data['cls']
        if src is None:
            continue
        mean = src['mean']
        std = src['std']
        lines.append(
            f"| `{m_name}` | "
            f"{mean['F1_c0']:.4f} ± {std['F1_c0']:.4f} | "
            f"{mean['F1_c1']:.4f} ± {std['F1_c1']:.4f} | "
            f"{mean['F1_c2']:.4f} ± {std['F1_c2']:.4f} | "
            f"{mean['F1_c3']:.4f} ± {std['F1_c3']:.4f} | "
            f"{mean['F1_c4']:.4f} ± {std['F1_c4']:.4f} |"
        )

    # 2b. Per-fold breakdown table
    lines.append(f"\n### 2b. Per-Fold Breakdown ({target_label})")
    lines.append("| Method | Fold | Macro-F1 | Weighted-F1 | Accuracy | QWK | MAE | F1_c0 | F1_c1 | F1_c2 | F1_c3 | F1_c4 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m_name, data in sorted(collected.items()):
        src = data['valsel'] if eval_target == 'valsel' else data['cls']
        if src is None or 'per_fold' not in src:
            continue
        for f_idx, fm in zip(data['folds'], src['per_fold']):
            lines.append(
                f"| `{m_name}` | {f_idx} | "
                f"{fm['MacroF1']:.4f} | {fm['WeightedF1']:.4f} | {fm['Accuracy']:.4f} | "
                f"{fm['QWK']:.4f} | {fm['MAE']:.4f} | "
                f"{fm['F1_c0']:.4f} | {fm['F1_c1']:.4f} | {fm['F1_c2']:.4f} | "
                f"{fm['F1_c3']:.4f} | {fm['F1_c4']:.4f} |"
            )

    # 2c. Paired fold comparison for final model candidates (cand_A, cand_B, cand_C, cand_D)
    cand_order = [
        ('cand_A', 1, 'vdrop_only + C2 + C5 + CE, no BVI, no impute, no mod'),
        ('cand_B', 2, 'cand_A + modulator (no impute)'),
        ('cand_C', 2, 'cand_A + ridge impute (no mod)'),
        ('cand_D', 3, 'full_no_bvi (cand_A + modulator + ridge impute)'),
    ]
    present_cands = [c for c in cand_order if c[0] in collected]
    if len(present_cands) >= 2:
        lines.append(
            f"\n### 2c. Paired Fold Comparison & Simplicity-Aware Candidate Selection (`cand_A..cand_D`, {target_label})"
        )
        cand_fold_maps: Dict[str, Dict[int, float]] = {}
        cand_means: Dict[str, float] = {}
        cand_stds: Dict[str, float] = {}
        for c_name, _, _ in present_cands:
            c_data = collected[c_name]
            src = c_data['valsel'] if eval_target == 'valsel' else c_data['cls']
            if src is not None and 'per_fold' in src:
                f_map = {int(f_i): float(fm['MacroF1']) for f_i, fm in zip(c_data['folds'], src['per_fold'])}
                cand_fold_maps[c_name] = f_map
                cand_means[c_name] = float(src['mean']['MacroF1'])
                cand_stds[c_name] = float(src['std']['MacroF1'])

        if cand_means:
            best_cand = max(cand_means.keys(), key=lambda k: cand_means[k])
            best_mean = cand_means[best_cand]
            best_std = cand_stds[best_cand]
            ref_cand = 'cand_A' if 'cand_A' in cand_fold_maps else present_cands[0][0]
            common_folds = sorted(
                set.intersection(*(set(m.keys()) for m in cand_fold_maps.values()))
            )
            fold_cols = " | ".join([f"Fold {f}" for f in common_folds])
            fold_align = "|".join(["---:" for _ in common_folds])
            lines.append(
                f"| Candidate | Complexity Rank | Description | {fold_cols} | Mean ± Std | Paired $\\Delta$ vs `{ref_cand}` (Per-Fold) | Paired $\\Delta$ vs `{ref_cand}` (Mean ± Std) | $|\\Delta\\text{{ vs best}}| < 1\\,\\text{{Std}}$ |"
            )
            lines.append(f"|---|---:|---|{fold_align}|---:|---|---:|---|")
            selected_cand = None
            for c_name, c_rank, c_desc in present_cands:
                if c_name not in cand_fold_maps:
                    continue
                f_map = cand_fold_maps[c_name]
                f_vals_str = " | ".join([f"{f_map[f]:.4f}" for f in common_folds])
                diffs_ref = [f_map[f] - cand_fold_maps[ref_cand][f] for f in common_folds]
                diffs_ref_str = ", ".join([f"{d:+.4f}" for d in diffs_ref])
                d_mean = float(np.mean(diffs_ref))
                d_std = float(np.std(diffs_ref))
                gap_to_best = abs(best_mean - cand_means[c_name])
                within_1std = gap_to_best <= max(best_std, cand_stds[c_name])
                if within_1std and selected_cand is None:
                    selected_cand = c_name
                within_str = f"YES ($|\\Delta|={gap_to_best:.4f} \\le {max(best_std, cand_stds[c_name]):.4f}$)" if within_1std else f"NO ($|\\Delta|={gap_to_best:.4f} > {max(best_std, cand_stds[c_name]):.4f}$)"
                lines.append(
                    f"| `{c_name}` | {c_rank} | {c_desc} | {f_vals_str} | "
                    f"{cand_means[c_name]:.4f} ± {cand_stds[c_name]:.4f} | "
                    f"`[{diffs_ref_str}]` | {d_mean:+.4f} ± {d_std:.4f} | {within_str} |"
                )
            if selected_cand is not None:
                lines.append(
                    f"\n> **Quy tắc chọn ứng viên (Split {split_type})**: Đỉnh trung bình là `{best_cand}` "
                    f"(`{best_mean:.4f} ± {best_std:.4f}`). Theo quy tắc ưu tiên cấu hình đơn giản hơn khi chênh lệch "
                    f"nhỏ hơn 1 độ lệch chuẩn giữa fold, ứng viên được chọn trên Split {split_type} là **`{selected_cand}`** "
                    f"(`{cand_means[selected_cand]:.4f} ± {cand_stds[selected_cand]:.4f}`)."
                )

    # 3. C4 Conformal table (with per-drug coverage statistics per Change 4)
    c4_key = 'valsel_conformal' if eval_target == 'valsel' else 'conformal'
    has_any_c4 = any(bool(data.get(c4_key)) for data in collected.values())
    if has_any_c4:
        lines.append(f"\n### 3. C4 Ordinal Conformal Prediction ($\\alpha = 0.10$, Target Coverage = 90%, {target_label})")
        if split_type == 'B':
            lines.append(
                "> **Ghi chú (Split B)**: Ở Split B, `val_calib` được chia theo thuốc nên đảm bảo độ phủ "
                "(coverage) mang tính trung bình trên thuốc; các cặp tương tác trong cùng một thuốc có tương quan. "
                "Đồng thời, độ phủ của lớp `Very Rare` (`Cov_c0`) ở Split B chỉ đạt khoảng `0.39–0.53` dù độ phủ tổng thể "
                "(`Pair Coverage`) đạt `~0.90` — conformal tiêu chuẩn và thứ tự chỉ đảm bảo độ phủ trung bình (marginal coverage), "
                "không đảm bảo độ phủ có điều kiện theo từng lớp (class-conditional coverage).\n"
            )
        lines.append(
            "| Method | C4 Variant | Pair Coverage | Drug Cov (Mean ± Std) | Drug Cov < 80% | Avg Set Size | Singleton Rate | Contiguity | Cov_c0 | Cov_c1 | Cov_c2 | Cov_c3 | Cov_c4 |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for m_name, data in sorted(collected.items()):
            c4_dict = data.get(c4_key, {})
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
                    f"| `{m_name}` | `{c4_name}` | "
                    f"{cm['empirical_coverage']:.4f} ± {cs['empirical_coverage']:.4f} | "
                    f"{dc_cell} | {lt80_cell} | "
                    f"{cm['avg_set_size']:.3f} ± {cs['avg_set_size']:.3f} | "
                    f"{cm['singleton_rate']:.4f} ± {cs['singleton_rate']:.4f} | "
                    f"{cm['contiguity_rate']:.4f} | "
                    f"{cm['coverage_c0']:.4f} | {cm['coverage_c1']:.4f} | "
                    f"{cm['coverage_c2']:.4f} | {cm['coverage_c3']:.4f} | "
                    f"{cm['coverage_c4']:.4f} |"
                )

    # 4. C5 Soft-Label Diagnostics table (K*, softened pair ratio, entropy of q)
    has_any_c5 = any(
        any((d.get('n_clusters', 0) > 0 or d.get('frac_smoothed', 0.0) > 0.0) for d in data.get('c5_per_fold', []))
        for data in collected.values()
    )
    if has_any_c5:
        lines.append("\n### 4. C5 Soft-Label Diagnostics (`K*`, Softened Pair Ratio, and Entropy of `q`)")
        lines.append("| Method | Fold | K_range | K* (`n_clusters`) | Silhouette | Eigengap | Softened Pair Ratio (`frac_smoothed`) | Entropy `H(q)` (all) | Entropy `H(q)` (softened) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for m_name, data in sorted(collected.items()):
            c5_list = data.get('c5_per_fold', [])
            if not any((d.get('n_clusters', 0) > 0 or d.get('frac_smoothed', 0.0) > 0.0) for d in c5_list):
                continue
            for d in c5_list:
                f_idx = d.get('fold', -1)
                k_min = d.get('k_min', 0)
                k_max = d.get('k_max', 0)
                fix_k = int(d.get('fixed_k', 0) or 0)
                if fix_k > 0:
                    k_rng = f"fixed={fix_k}"
                elif k_max > 0:
                    k_rng = f"[{k_min}, {k_max}]"
                else:
                    k_rng = "N/A"
                k_star = int(d.get('n_clusters', 0))
                sil = float(d.get('silhouette', 0.0))
                eg = float(d.get('eigengap', 0.0))
                frac = float(d.get('frac_smoothed', d.get('softened_pair_ratio', 0.0)))
                ent_all = d.get('entropy_q', None)
                ent_sm = d.get('entropy_q_smoothed', None)
                ent_all_s = f"{float(ent_all):.4f}" if ent_all is not None else "N/A"
                ent_sm_s = f"{float(ent_sm):.4f}" if ent_sm is not None else "N/A"
                lines.append(
                    f"| `{m_name}` | {f_idx} | {k_rng} | {k_star} | {sil:.4f} | {eg:.4f} | {frac:.4f} ({frac*100:.2f}%) | {ent_all_s} | {ent_sm_s} |"
                )

    # 5. BVI Posterior Uncertainty Diagnostics table (fold_{k}_sigma.npy)
    has_any_bvi = any(bool(data.get('bvi_per_fold')) for data in collected.values())
    if has_any_bvi:
        lines.append(
            "\n### 5. BVI Posterior Uncertainty Diagnostics (`fold_{k}_sigma.npy`: Full Views vs. Missing `D9, D10` & Correlation with Conformal Set Size)"
        )
        lines.append(
            "| Method | Fold | $\\bar{\\sigma}$ (`val_select`) | $\\bar{\\sigma}$ (Has `D9,D10`) | $\\bar{\\sigma}$ (Missing `D9,D10`) | $\\bar{\\sigma}$ (Full Static Views) | $\\bar{\\sigma}$ (Missing `D7/D11`) | Corr($\\sigma$, `|C|` Ord-APS) $r / \\rho$ | Corr($\\sigma$, `|C|` Ord-LAC) $r / \\rho$ |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

        def _fmt_f(v) -> str:
            return f"{float(v):.4f}" if v is not None and not np.isnan(float(v)) else "N/A"

        for m_name, data in sorted(collected.items()):
            bvi_list = data.get('bvi_per_fold', [])
            if not bvi_list:
                continue
            for bd in bvi_list:
                f_idx = bd.get('fold', -1)
                vs_m = bd.get('valsel_sigma_mean')
                vs_s = bd.get('valsel_sigma_std')
                vs_str = f"{float(vs_m):.4f} ± {float(vs_s):.4f}" if vs_m is not None else "N/A"
                has_d = _fmt_f(bd.get('sigma_has_d9d10_mean'))
                mis_d = _fmt_f(bd.get('sigma_missing_d9d10_mean'))
                ful_st = _fmt_f(bd.get('sigma_full_static_mean'))
                mis_st = _fmt_f(bd.get('sigma_missing_static_mean'))
                r_oa = bd.get('corr_sigma_ord_aps_pearson')
                rho_oa = bd.get('corr_sigma_ord_aps_spearman')
                oa_str = f"{float(r_oa):+.4f} / {float(rho_oa):+.4f}" if r_oa is not None else "N/A"
                r_ol = bd.get('corr_sigma_ord_lac_pearson')
                rho_ol = bd.get('corr_sigma_ord_lac_spearman')
                ol_str = f"{float(r_ol):+.4f} / {float(rho_ol):+.4f}" if r_ol is not None else "N/A"
                lines.append(
                    f"| `{m_name}` | {f_idx} | {vs_str} | {has_d} | {mis_d} | {ful_st} | {mis_st} | {oa_str} | {ol_str} |"
                )

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Summarize results_v2 experiments directly from saved .npy files.')
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_V2_DIR)
    parser.add_argument('--drive_dir', type=str, default=None, help='Optional Google Drive directory to sync from')
    parser.add_argument('--ref_method', type=str, default='MSSF_v2')
    parser.add_argument('--eval_target', type=str, default='test', choices=['test', 'valsel'])
    parser.add_argument('--splits', type=str, nargs='+', default=['A', 'B'], choices=['A', 'B'])
    parser.add_argument('--folds', type=int, nargs='*', default=None, help='Optional subset of folds for paired comparison')
    parser.add_argument('--methods', type=str, nargs='*', default=None, help='Optional list of method names to include')
    parser.add_argument('--out_md', type=str, default=None)
    parser.add_argument('--out_raw', type=str, default=None, help='Save verbatim script output to reports/raw/*.txt')
    args = parser.parse_args()

    cmd_str = "python -m MSSF_plus.scripts.summarize_results " + " ".join(sys.argv[1:])
    header_lines = [
        f"SCRIPT: MSSF_plus/scripts/summarize_results.py",
        f"COMMAND: {cmd_str}",
        "=" * 110,
        "# MSSF_plus Unified Evaluation Summary (Computed from `.npy`)\n",
    ]
    sections = ["\n".join(header_lines)]
    for split_type in args.splits:
        collected = collect_method_results(
            args.results_dir,
            split_type,
            only_folds=args.folds,
            only_methods=args.methods,
            drive_dir=args.drive_dir,
        )
        sections.append(
            format_markdown_tables(
                args.results_dir,
                collected,
                split_type,
                ref_method=args.ref_method,
                eval_target=args.eval_target,
            )
        )

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

