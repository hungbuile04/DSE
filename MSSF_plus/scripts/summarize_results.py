"""
Aggregate results from `results_v2/`, compute mean +/- std across available folds,
perform paired Wilcoxon signed-rank tests against `MSSF_v2`, and generate Markdown
tables for both classification metrics (`test` or `valsel`) and C4 ordinal conformal
prediction metrics (including per-drug coverage mean, std, and <80% rate in Split B).
"""

import os
import glob
import json
import argparse
from typing import Dict

import numpy as np

from MSSF_plus.protocol import DEFAULT_RESULTS_V2_DIR, load_fold_data, wilcoxon_compare
from MSSF_plus.metrics import compute_multiclass_metrics, aggregate_fold_metrics
from MSSF_plus.conformal import evaluate_all_conformal_methods


def collect_method_results(
    results_dir: str,
    split_type: str,
    only_folds: list = None,
) -> Dict[str, dict]:
    """Scan `results_dir/*_{split_type}` and evaluate all available folds directly from saved `.npy` files."""
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

        fold_metrics_list = []
        fold_valsel_list = []
        fold_gaps = []
        fold_times = []
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

    # 3. C4 Conformal table (with per-drug coverage statistics per Change 4)
    c4_key = 'valsel_conformal' if eval_target == 'valsel' else 'conformal'
    has_any_c4 = any(bool(data.get(c4_key)) for data in collected.values())
    if not has_any_c4:
        lines.append("")
        return "\n".join(lines)

    lines.append(f"\n### 3. C4 Ordinal Conformal Prediction ($\\alpha = 0.10$, Target Coverage = 90%, {target_label})")
    if split_type == 'B':
        lines.append(
            "> **Ghi chú (Split B)**: Ở Split B, `val_calib` được chia theo thuốc nên đảm bảo độ phủ "
            "(coverage) mang tính trung bình trên thuốc; các cặp tương tác trong cùng một thuốc có tương quan.\n"
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

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Summarize results_v2 experiments directly from saved .npy files.')
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_V2_DIR)
    parser.add_argument('--ref_method', type=str, default='MSSF_v2')
    parser.add_argument('--eval_target', type=str, default='test', choices=['test', 'valsel'])
    parser.add_argument('--folds', type=int, nargs='*', default=None, help='Optional subset of folds for paired comparison')
    parser.add_argument('--out_md', type=str, default=None)
    args = parser.parse_args()

    sections = ["# MSSF_plus Unified Evaluation Summary (Computed from `.npy`)\n"]
    for split_type in ['A', 'B']:
        collected = collect_method_results(args.results_dir, split_type, only_folds=args.folds)
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


if __name__ == '__main__':
    main()

