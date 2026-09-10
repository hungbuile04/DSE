"""
Unified evaluation metrics for all DSE baselines.
Comprehensive metric suite — run ONCE, get ALL metrics.

Handles different prediction scales from each model:
- MGPred, SDPred, DSGAT, A3Net, HSTrans: frequency scale (0-5)
- MSSF: discrete classes {1,2,3,4,5} (only known pairs)
- MVSL-DSF: frequency scale (1-5) (only known pairs, already ×5)

Usage:
    python unified_eval.py --results_dir /path/to/results/
    python unified_eval.py --results_dir /path/to/results/ --csv results.csv

Expected file format per method:
    results/<method_name>_<A|B>/
        fold_0_labels.npy    # Ground truth
        fold_0_preds.npy     # Predicted scores
"""

import numpy as np
import os
import sys
import argparse
from math import sqrt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    precision_recall_curve,
    ndcg_score,
)
from scipy.stats import pearsonr, spearmanr


# ============================================================
# Core metric functions
# ============================================================

def safe_pearson(y_true, y_pred):
    if len(y_true) < 3 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    r, _ = pearsonr(y_true, y_pred)
    return r if not np.isnan(r) else 0.0

def safe_spearman(y_true, y_pred):
    if len(y_true) < 3 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    r, _ = spearmanr(y_true, y_pred)
    return r if not np.isnan(r) else 0.0

def mean_auc_per_drug(y_true_freq, y_pred_scores, drug_indices):
    """Macro-averaged AUC computed per drug (drugAUC)."""
    drugs = np.unique(drug_indices)
    aucs = []
    for d in drugs:
        mask = drug_indices == d
        yt = (y_true_freq[mask] > 0).astype(int)
        yp = y_pred_scores[mask]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
    return np.mean(aucs) if aucs else 0.0

def mean_aupr_per_drug(y_true_freq, y_pred_scores, drug_indices):
    """Macro-averaged AUPR computed per drug (drugAUPR)."""
    drugs = np.unique(drug_indices)
    auprs = []
    for d in drugs:
        mask = drug_indices == d
        yt = (y_true_freq[mask] > 0).astype(int)
        yp = y_pred_scores[mask]
        if len(np.unique(yt)) < 2:
            continue
        auprs.append(average_precision_score(yt, yp))
    return np.mean(auprs) if auprs else 0.0

def precision_at_k(y_true_binary, y_pred_scores, k):
    """Precision@K: fraction of top-K predictions that are positive."""
    if len(y_pred_scores) < k:
        k = len(y_pred_scores)
    top_k_idx = np.argsort(y_pred_scores)[-k:]
    return np.mean(y_true_binary[top_k_idx])

def recall_at_k(y_true_binary, y_pred_scores, k):
    """Recall@K: fraction of positives found in top-K."""
    if len(y_pred_scores) < k:
        k = len(y_pred_scores)
    n_pos = np.sum(y_true_binary)
    if n_pos == 0:
        return 0.0
    top_k_idx = np.argsort(y_pred_scores)[-k:]
    return np.sum(y_true_binary[top_k_idx]) / n_pos

def mean_average_precision(y_true_binary, y_pred_scores, drug_indices):
    """Mean Average Precision (MAP) across drugs."""
    drugs = np.unique(drug_indices)
    aps = []
    for d in drugs:
        mask = drug_indices == d
        yt = y_true_binary[mask]
        yp = y_pred_scores[mask]
        if np.sum(yt) == 0:
            continue
        aps.append(average_precision_score(yt, yp))
    return np.mean(aps) if aps else 0.0

def compute_ndcg(y_true_freq, y_pred_scores, drug_indices, k=10):
    """Mean nDCG@K across drugs."""
    drugs = np.unique(drug_indices)
    ndcgs = []
    for d in drugs:
        mask = drug_indices == d
        yt = y_true_freq[mask]
        yp = y_pred_scores[mask]
        if np.sum(yt) == 0 or len(yt) < 2:
            continue
        try:
            ndcgs.append(ndcg_score([yt], [yp], k=min(k, len(yt))))
        except:
            continue
    return np.mean(ndcgs) if ndcgs else 0.0


# ============================================================
# Main compute function
# ============================================================

def compute_all_metrics(y_true, y_pred, drug_indices=None):
    """
    Compute comprehensive metrics from ground truth and predictions.

    Args:
        y_true: array of true frequency values
                - For full-matrix models: 0=negative, 1-5=frequency
                - For known-pair-only models: 1-5 frequency
        y_pred: array of predicted scores (same scale as y_true)
        drug_indices: optional array of drug indices for per-drug metrics

    Returns:
        dict of all metrics
    """
    results = {}
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    has_negatives = np.any(y_true == 0)

    # ============ REGRESSION METRICS (on known pairs: freq > 0) ============
    if has_negatives:
        known_mask = y_true > 0
    else:
        known_mask = np.ones(len(y_true), dtype=bool)

    n_known = known_mask.sum()

    if n_known > 0:
        yt_k = y_true[known_mask]
        yp_k = y_pred[known_mask]

        results['RMSE'] = sqrt(mean_squared_error(yt_k, yp_k))
        results['MAE'] = mean_absolute_error(yt_k, yp_k)
        results['MSE'] = mean_squared_error(yt_k, yp_k)
        results['Pearson'] = safe_pearson(yt_k, yp_k)
        results['Spearman'] = safe_spearman(yt_k, yp_k)

        # R-squared
        ss_res = np.sum((yt_k - yp_k) ** 2)
        ss_tot = np.sum((yt_k - np.mean(yt_k)) ** 2)
        results['R2'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    else:
        for m in ['RMSE', 'MAE', 'MSE', 'Pearson', 'Spearman', 'R2']:
            results[m] = 0.0

    # ============ BINARY CLASSIFICATION METRICS ============
    if has_negatives:
        y_binary = (y_true > 0).astype(int)

        if len(np.unique(y_binary)) == 2:
            results['AUC'] = roc_auc_score(y_binary, y_pred)
            results['AUPR'] = average_precision_score(y_binary, y_pred)
        else:
            results['AUC'] = 0.0
            results['AUPR'] = 0.0

        # Optimal threshold by F1
        if len(np.unique(y_binary)) == 2:
            prec_curve, rec_curve, thresholds = precision_recall_curve(y_binary, y_pred)
            f1_curve = 2 * prec_curve * rec_curve / (prec_curve + rec_curve + 1e-10)
            best_idx = np.argmax(f1_curve)
            best_threshold = thresholds[min(best_idx, len(thresholds)-1)]
        else:
            best_threshold = 0.5

        y_pred_binary = (y_pred > best_threshold).astype(int)

        results['Accuracy'] = accuracy_score(y_binary, y_pred_binary)
        results['Precision'] = precision_score(y_binary, y_pred_binary, zero_division=0)
        results['Recall'] = recall_score(y_binary, y_pred_binary, zero_division=0)
        results['F1'] = f1_score(y_binary, y_pred_binary, zero_division=0)
        results['Specificity'] = recall_score(1 - y_binary, 1 - y_pred_binary, zero_division=0)
        results['MCC'] = matthews_corrcoef(y_binary, y_pred_binary)
        results['Kappa'] = cohen_kappa_score(y_binary, y_pred_binary)

        # Top-K metrics
        for k in [5, 10, 15, 20]:
            results[f'P@{k}'] = precision_at_k(y_binary, y_pred, k)
            results[f'R@{k}'] = recall_at_k(y_binary, y_pred, k)

        # Per-drug metrics (if drug indices available)
        if drug_indices is not None:
            results['drugAUC'] = mean_auc_per_drug(y_true, y_pred, drug_indices)
            results['drugAUPR'] = mean_aupr_per_drug(y_true, y_pred, drug_indices)
            results['MAP'] = mean_average_precision(y_binary, y_pred, drug_indices)
            results['nDCG@10'] = compute_ndcg(y_true, y_pred, drug_indices, k=10)
        else:
            results['drugAUC'] = 0.0
            results['drugAUPR'] = 0.0
            results['MAP'] = 0.0
            results['nDCG@10'] = 0.0
    else:
        # Models that only save known pairs (MSSF, MVSL-DSF)
        for m in ['AUC', 'AUPR', 'Accuracy', 'Precision', 'Recall', 'F1',
                   'Specificity', 'MCC', 'Kappa', 'drugAUC', 'drugAUPR', 'MAP', 'nDCG@10']:
            results[m] = np.nan
        for k in [5, 10, 15, 20]:
            results[f'P@{k}'] = np.nan
            results[f'R@{k}'] = np.nan

    # ============ MULTI-CLASS METRICS (for 5-level frequency) ============
    if n_known > 0:
        yt_k = y_true[known_mask]
        yp_k = y_pred[known_mask]

        # Round predictions to nearest integer for class-level eval
        yp_class = np.clip(np.round(yp_k), 1, 5).astype(int)
        yt_class = np.clip(np.round(yt_k), 1, 5).astype(int)

        results['ClassAcc'] = accuracy_score(yt_class, yp_class)
        results['WeightedF1'] = f1_score(yt_class, yp_class, average='weighted', zero_division=0)
        results['MacroF1'] = f1_score(yt_class, yp_class, average='macro', zero_division=0)
        results['ClassKappa'] = cohen_kappa_score(yt_class, yp_class)
    else:
        for m in ['ClassAcc', 'WeightedF1', 'MacroF1', 'ClassKappa']:
            results[m] = 0.0

    return results


# ============================================================
# Evaluate all folds for a method
# ============================================================

def evaluate_method(results_dir, method_name):
    """Evaluate all folds for a method and return averaged metrics."""
    method_dir = os.path.join(results_dir, method_name)
    if not os.path.isdir(method_dir):
        return None

    all_metrics = []
    fold = 0
    while True:
        labels_path = os.path.join(method_dir, f'fold_{fold}_labels.npy')
        preds_path = os.path.join(method_dir, f'fold_{fold}_preds.npy')

        if not os.path.exists(labels_path):
            break

        y_true = np.load(labels_path, allow_pickle=True).flatten().astype(float)
        y_pred = np.load(preds_path, allow_pickle=True).flatten().astype(float)

        # Handle NaN/Inf
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[valid]
        y_pred = y_pred[valid]

        if len(y_true) == 0:
            fold += 1
            continue

        metrics = compute_all_metrics(y_true, y_pred)
        all_metrics.append(metrics)
        fold += 1

    if not all_metrics:
        return None

    # Average across folds
    avg_metrics = {}
    std_metrics = {}
    for key in all_metrics[0]:
        values = [m[key] for m in all_metrics if not np.isnan(m[key])]
        if values:
            avg_metrics[key] = np.mean(values)
            std_metrics[key] = np.std(values)
        else:
            avg_metrics[key] = np.nan
            std_metrics[key] = np.nan

    return avg_metrics, std_metrics, len(all_metrics)


# ============================================================
# Output formatting
# ============================================================

# Metric groups for organized display
METRIC_GROUPS = {
    'Classification (Binary)': ['AUC', 'AUPR', 'F1', 'Accuracy', 'Precision', 'Recall',
                                 'Specificity', 'MCC', 'Kappa'],
    'Regression (Frequency)':  ['RMSE', 'MAE', 'MSE', 'Pearson', 'Spearman', 'R2'],
    'Multi-class (5-level)':   ['ClassAcc', 'WeightedF1', 'MacroF1', 'ClassKappa'],
    'Ranking':                 ['drugAUC', 'drugAUPR', 'MAP', 'nDCG@10'],
    'Top-K':                   ['P@5', 'P@10', 'P@15', 'P@20', 'R@5', 'R@10', 'R@15', 'R@20'],
}


def print_comparison_table(all_results, metrics_to_show=None):
    """Print formatted comparison tables by metric group."""

    methods = list(all_results.keys())

    for group_name, group_metrics in METRIC_GROUPS.items():
        if metrics_to_show:
            group_metrics = [m for m in group_metrics if m in metrics_to_show]
        if not group_metrics:
            continue

        print(f"\n{'='*80}")
        print(f"  {group_name}")
        print(f"{'='*80}")

        # Header
        header = f"{'Method':<15}" + "".join(f"{m:>14}" for m in group_metrics)
        print(header)
        print("-" * len(header))

        for method in methods:
            if all_results[method] is None:
                print(f"{method:<15}  (no results)")
                continue

            avg, std, n_folds = all_results[method]
            row = f"{method:<15}"
            for m in group_metrics:
                val = avg.get(m, np.nan)
                s = std.get(m, np.nan)
                if np.isnan(val):
                    row += f"{'N/A':>14}"
                else:
                    row += f"  {val:.4f}±{s:.3f}"
            print(row)

        print()


def save_csv(all_results, csv_path):
    """Save all results to CSV for easy import into papers/spreadsheets."""
    all_metrics_keys = []
    for group_metrics in METRIC_GROUPS.values():
        all_metrics_keys.extend(group_metrics)

    with open(csv_path, 'w') as f:
        # Header
        f.write("Method,Split,Folds," + ",".join(
            f"{m}_mean,{m}_std" for m in all_metrics_keys
        ) + "\n")

        for method_name, result in all_results.items():
            if result is None:
                continue

            avg, std, n_folds = result
            # Parse split from method name
            parts = method_name.rsplit('_', 1)
            model = parts[0]
            split = parts[1] if len(parts) > 1 else 'A'

            values = []
            for m in all_metrics_keys:
                val = avg.get(m, np.nan)
                s = std.get(m, np.nan)
                values.append(f"{val:.6f}" if not np.isnan(val) else "")
                values.append(f"{s:.6f}" if not np.isnan(s) else "")

            f.write(f"{model},{split},{n_folds}," + ",".join(values) + "\n")

    print(f"\n✅ CSV saved: {csv_path}")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified DSE Evaluation — Comprehensive Metrics')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory containing method result subdirectories')
    parser.add_argument('--csv', type=str, default=None,
                        help='Save results to CSV file')
    args = parser.parse_args()

    # Auto-detect methods from directory listing
    METHODS = []
    if os.path.isdir(args.results_dir):
        for d in sorted(os.listdir(args.results_dir)):
            if os.path.isdir(os.path.join(args.results_dir, d)):
                METHODS.append(d)

    if not METHODS:
        # Fallback: default method names
        METHODS = [
            'MGPred_A', 'MGPred_B',
            'SDPred_A', 'SDPred_B',
            'MSSF_A', 'MSSF_B',
            'DSGAT_A', 'DSGAT_B',
            'A3Net_A', 'A3Net_B',
            'HSTrans_A', 'HSTrans_B',
            'MVSL-DSF_A', 'MVSL-DSF_B',
        ]

    all_results = {}
    for method in METHODS:
        result = evaluate_method(args.results_dir, method)
        if result:
            avg, std, n_folds = result
            print(f"✅ {method}: {n_folds} folds loaded")
        else:
            print(f"⚠️  {method}: no results found")
        all_results[method] = result

    print_comparison_table(all_results)

    if args.csv:
        save_csv(all_results, args.csv)
    else:
        csv_path = os.path.join(args.results_dir, 'comparison_results.csv')
        save_csv(all_results, csv_path)

    print("\n📊 Total metrics per method: 31")
    print("   Classification: AUC, AUPR, F1, Accuracy, Precision, Recall, Specificity, MCC, Kappa")
    print("   Regression:     RMSE, MAE, MSE, Pearson, Spearman, R²")
    print("   Multi-class:    ClassAcc, WeightedF1, MacroF1, ClassKappa")
    print("   Ranking:        drugAUC, drugAUPR, MAP, nDCG@10")
    print("   Top-K:          P@5/10/15/20, R@5/10/15/20")
