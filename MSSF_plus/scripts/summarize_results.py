"""
Aggregate results from `results_v2/`, compute mean +/- std across available folds,
perform paired Wilcoxon signed-rank tests against `MSSF_v2`, and generate Markdown
tables for both classification metrics and C4 ordinal conformal prediction metrics.
"""

import os
import glob
import argparse
from typing import Dict

import numpy as np

from MSSF_plus.protocol import DEFAULT_RESULTS_V2_DIR, wilcoxon_compare
from MSSF_plus.metrics import compute_multiclass_metrics, aggregate_fold_metrics
from MSSF_plus.conformal import evaluate_all_conformal_methods


def collect_method_results(results_dir: str, split_type: str) -> Dict[str, dict]:
    """Scan `results_dir/*_{split_type}` and evaluate all available folds."""
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

        fold_metrics_list = []
        fold_conformal_list = {k: [] for k in ['LAC', 'APS', 'Ordinal_LAC', 'Ordinal_APS']}
        folds_found = []

        for k in range(10):
            lbl_path = os.path.join(mdir, f'fold_{k}_labels.npy')
            prd_path = os.path.join(mdir, f'fold_{k}_preds.npy')
            prb_path = os.path.join(mdir, f'fold_{k}_probs.npy')
            vcal_lbl_path = os.path.join(mdir, f'fold_{k}_valcal_labels.npy')
            vcal_prb_path = os.path.join(mdir, f'fold_{k}_valcal_probs.npy')

            if not (os.path.exists(lbl_path) and os.path.exists(prd_path)):
                continue

            y_true_freq = np.load(lbl_path).flatten()
            y_pred_freq = np.load(prd_path).flatten()
            y_true_c = np.clip(np.round(y_true_freq), 1, 5).astype(int) - 1
            y_pred_c = np.clip(np.round(y_pred_freq), 1, 5).astype(int) - 1
            y_probs = np.load(prb_path) if os.path.exists(prb_path) else None

            m = compute_multiclass_metrics(y_true_c, y_pred_c, probs=y_probs)
            fold_metrics_list.append(m)
            folds_found.append(k)

            if y_probs is not None and os.path.exists(vcal_lbl_path) and os.path.exists(vcal_prb_path):
                vcal_c = np.load(vcal_lbl_path).flatten()
                vcal_prb = np.load(vcal_prb_path)
                # vcal_c saved by save_fold_predictions is 0..4 class index
                vcal_freq = (vcal_c + 1) if vcal_c.min() == 0 else vcal_c
                c4 = evaluate_all_conformal_methods(
                    valcal_probs=vcal_prb,
                    valcal_labels_1_to_5=vcal_freq,
                    test_probs=y_probs,
                    test_labels_1_to_5=y_true_freq,
                    alpha=0.10,
                    seed=42 + k,
                )
                for c4_name, c4_val in c4.items():
                    fold_conformal_list[c4_name].append(c4_val)

        if not fold_metrics_list:
            continue

        cls_mean, cls_std = aggregate_fold_metrics(fold_metrics_list)
        agg_c4 = {}
        for c4_name, c4_list in fold_conformal_list.items():
            if c4_list:
                cm, cs = aggregate_fold_metrics(c4_list)
                agg_c4[c4_name] = {'mean': cm, 'std': cs}

        collected[method_name] = {
            'folds': folds_found,
            'cls': {'mean': cls_mean, 'std': cls_std},
            'conformal': agg_c4,
        }

    return collected


def format_markdown_tables(
    results_dir: str,
    collected: Dict[str, dict],
    split_type: str,
    ref_method: str = 'MSSF_v2',
) -> str:
    lines = [f"## Split {split_type} Results (`results_v2`)\n"]
    if not collected:
        lines.append("_No completed folds found._\n")
        return "\n".join(lines)

    # 1. Main classification table
    lines.append("### 1. Overall & Ordinal Classification Metrics")
    lines.append(
        "| Method | Folds | Macro-F1 | Weighted-F1 | Accuracy | QWK | MAE | Macro-AUPR | p-val (Macro-F1) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for m_name, data in sorted(collected.items()):
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
    lines.append("\n### 2. Per-Class F1 Breakdown (`c0..c4`)")
    lines.append("| Method | F1_c0 (Very Rare) | F1_c1 (Rare) | F1_c2 (Infrequent) | F1_c3 (Frequent) | F1_c4 (Very Freq) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for m_name, data in sorted(collected.items()):
        mean = data['cls']['mean']
        std = data['cls']['std']
        lines.append(
            f"| `{m_name}` | "
            f"{mean['F1_c0']:.4f} ± {std['F1_c0']:.4f} | "
            f"{mean['F1_c1']:.4f} ± {std['F1_c1']:.4f} | "
            f"{mean['F1_c2']:.4f} ± {std['F1_c2']:.4f} | "
            f"{mean['F1_c3']:.4f} ± {std['F1_c3']:.4f} | "
            f"{mean['F1_c4']:.4f} ± {std['F1_c4']:.4f} |"
        )

    # 3. C4 Conformal table
    lines.append("\n### 3. C4 Ordinal Conformal Prediction ($\\alpha = 0.10$, Target Coverage = 90%)")
    lines.append(
        "| Method | C4 Variant | Coverage | Avg Set Size | Singleton Rate | Contiguity Rate | Cov_c0 (Very Rare) | Cov_c4 (Very Freq) |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for m_name, data in sorted(collected.items()):
        c4_dict = data['conformal']
        for c4_name in ['LAC', 'APS', 'Ordinal_LAC', 'Ordinal_APS']:
            if c4_name not in c4_dict:
                continue
            cm = c4_dict[c4_name]['mean']
            cs = c4_dict[c4_name]['std']
            lines.append(
                f"| `{m_name}` | `{c4_name}` | "
                f"{cm['empirical_coverage']:.4f} ± {cs['empirical_coverage']:.4f} | "
                f"{cm['avg_set_size']:.3f} ± {cs['avg_set_size']:.3f} | "
                f"{cm['singleton_rate']:.4f} ± {cs['singleton_rate']:.4f} | "
                f"{cm['contiguity_rate']:.4f} | "
                f"{cm['coverage_c0']:.4f} | "
                f"{cm['coverage_c4']:.4f} |"
            )

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Summarize results_v2 experiments.')
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_V2_DIR)
    parser.add_argument('--ref_method', type=str, default='MSSF_v2')
    parser.add_argument('--out_md', type=str, default=None)
    args = parser.parse_args()

    sections = ["# MSSF_plus Unified Evaluation Summary\n"]
    for split_type in ['A', 'B']:
        collected = collect_method_results(args.results_dir, split_type)
        sections.append(
            format_markdown_tables(args.results_dir, collected, split_type, ref_method=args.ref_method)
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
