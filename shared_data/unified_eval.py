"""
Unified evaluation metrics for all DSE baselines.
Each method saves predictions as .npy files, then this script computes
the same metrics for all methods for fair comparison.

Usage:
    python unified_eval.py --results_dir /path/to/results/
    
Expected file format per method:
    results/<method_name>/
        fold_0_labels.npy    # Ground truth frequency (0-5)
        fold_0_preds.npy     # Predicted scores
        fold_1_labels.npy
        fold_1_preds.npy
        ...
"""

import numpy as np
import os
import argparse
from math import sqrt
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score,
    mean_absolute_error,
    precision_score, 
    recall_score, 
    f1_score,
    accuracy_score
)
from scipy.stats import pearsonr, spearmanr


def compute_all_metrics(y_true, y_pred):
    """
    Compute unified metrics from ground truth frequencies and predictions.
    
    Args:
        y_true: array of true frequency values (0-5)
        y_pred: array of predicted scores
        
    Returns:
        dict of all metrics
    """
    results = {}
    
    # ===== REGRESSION METRICS (on known pairs only, freq > 0) =====
    known_mask = y_true > 0
    if known_mask.sum() > 0:
        y_true_known = y_true[known_mask]
        y_pred_known = y_pred[known_mask]
        
        results['RMSE'] = sqrt(np.mean((y_true_known - y_pred_known) ** 2))
        results['MAE'] = mean_absolute_error(y_true_known, y_pred_known)
        
        if len(y_true_known) > 2:
            results['Pearson'], _ = pearsonr(y_true_known, y_pred_known)
            results['Spearman'], _ = spearmanr(y_true_known, y_pred_known)
        else:
            results['Pearson'] = 0.0
            results['Spearman'] = 0.0
    else:
        results['RMSE'] = 0.0
        results['MAE'] = 0.0
        results['Pearson'] = 0.0
        results['Spearman'] = 0.0
    
    # ===== BINARY CLASSIFICATION METRICS =====
    # Binary: freq > 0 → positive, freq == 0 → negative
    y_binary = (y_true > 0).astype(int)
    
    if len(np.unique(y_binary)) == 2:
        results['AUC'] = roc_auc_score(y_binary, y_pred)
        results['AUPR'] = average_precision_score(y_binary, y_pred)
    else:
        results['AUC'] = 0.0
        results['AUPR'] = 0.0
    
    # Binary with threshold = 0.5
    if y_pred.max() <= 5:
        # If predictions are in frequency scale (0-5), threshold at 0.5
        y_pred_binary = (y_pred > 0.5).astype(int)
    else:
        y_pred_binary = (y_pred > 0.5).astype(int)
    
    if len(np.unique(y_binary)) == 2 and len(np.unique(y_pred_binary)) >= 1:
        results['Accuracy'] = accuracy_score(y_binary, y_pred_binary)
        results['Precision'] = precision_score(y_binary, y_pred_binary, zero_division=0)
        results['Recall'] = recall_score(y_binary, y_pred_binary, zero_division=0)
        results['F1'] = f1_score(y_binary, y_pred_binary, zero_division=0)
    else:
        results['Accuracy'] = 0.0
        results['Precision'] = 0.0
        results['Recall'] = 0.0
        results['F1'] = 0.0
    
    return results


def evaluate_method(results_dir, method_name):
    """Evaluate all folds for a method and return averaged metrics."""
    method_dir = os.path.join(results_dir, method_name)
    if not os.path.isdir(method_dir):
        print(f"  ⚠️ Directory not found: {method_dir}")
        return None
    
    all_metrics = []
    fold = 0
    while True:
        labels_path = os.path.join(method_dir, f'fold_{fold}_labels.npy')
        preds_path = os.path.join(method_dir, f'fold_{fold}_preds.npy')
        
        if not os.path.exists(labels_path):
            break
        
        y_true = np.load(labels_path)
        y_pred = np.load(preds_path)
        
        metrics = compute_all_metrics(y_true, y_pred)
        all_metrics.append(metrics)
        fold += 1
    
    if not all_metrics:
        print(f"  ⚠️ No fold results found in {method_dir}")
        return None
    
    # Average across folds
    avg_metrics = {}
    std_metrics = {}
    for key in all_metrics[0]:
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = np.mean(values)
        std_metrics[key] = np.std(values)
    
    return avg_metrics, std_metrics, len(all_metrics)


def print_comparison_table(all_results):
    """Print a formatted comparison table."""
    
    methods = list(all_results.keys())
    metrics = ['AUC', 'AUPR', 'RMSE', 'MAE', 'Pearson', 'Spearman', 'F1', 'Accuracy']
    
    # Header
    header = f"{'Method':<12}" + "".join(f"{m:>14}" for m in metrics)
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    
    for method in methods:
        if all_results[method] is None:
            print(f"{method:<12}  (no results)")
            continue
        
        avg, std, n_folds = all_results[method]
        row = f"{method:<12}"
        for m in metrics:
            val = avg.get(m, 0)
            s = std.get(m, 0)
            row += f"  {val:.4f}±{s:.3f}"
        print(row)
    
    print("=" * len(header))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified DSE Evaluation')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory containing method result subdirectories')
    args = parser.parse_args()
    
    METHODS = ['MGPred', 'SDPred', 'MSSF', 'DSGAT', 'A3Net', 'HSTrans', 'MVSL-DSF']
    
    all_results = {}
    for method in METHODS:
        print(f"\nEvaluating {method}...")
        result = evaluate_method(args.results_dir, method)
        all_results[method] = result
    
    print("\n")
    print_comparison_table(all_results)
