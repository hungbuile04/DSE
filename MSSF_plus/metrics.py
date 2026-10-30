"""
Comprehensive 5-class ordinal evaluation metrics for MSSF_plus.

Labels convention:
  - Internal model class index: {0, 1, 2, 3, 4}
  - SIDER frequency scale (saved to labels.npy / preds.npy for unified_eval.py): {1, 2, 3, 4, 5}
"""

from math import sqrt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize

CLASSES_0_TO_4 = [0, 1, 2, 3, 4]


def compute_multiclass_metrics(y_true_class, y_pred_class, probs=None):
    """
    Compute the full suite of 5-class classification and ordinal metrics.

    Args:
        y_true_class: 1D array-like of integer classes in {0, 1, 2, 3, 4}
        y_pred_class: 1D array-like of predicted integer classes in {0, 1, 2, 3, 4}
        probs: optional 2D array of shape (N, 5) with class probabilities summing to 1

    Returns:
        dict of float metrics:
            Accuracy, WeightedF1, MacroF1,
            Kappa (Cohen's unweighted), QWK (Quadratic Weighted Kappa),
            MAE (on discrete class index), ExpectedMAE (if probs provided), RMSE,
            F1_c0, F1_c1, F1_c2, F1_c3, F1_c4,
            MacroPrecision, MacroRecall, MCC, MacroAUPR
    """
    y_true = np.asarray(y_true_class, dtype=int).flatten()
    y_pred = np.asarray(y_pred_class, dtype=int).flatten()

    acc = float(accuracy_score(y_true, y_pred))
    wf1 = float(f1_score(y_true, y_pred, labels=CLASSES_0_TO_4, average='weighted', zero_division=0))
    mf1 = float(f1_score(y_true, y_pred, labels=CLASSES_0_TO_4, average='macro', zero_division=0))
    per_class_f1 = f1_score(y_true, y_pred, labels=CLASSES_0_TO_4, average=None, zero_division=0)

    kappa = float(cohen_kappa_score(y_true, y_pred, labels=CLASSES_0_TO_4))
    qwk = float(cohen_kappa_score(y_true, y_pred, labels=CLASSES_0_TO_4, weights='quadratic'))

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(sqrt(mean_squared_error(y_true, y_pred)))

    m_prec = float(precision_score(y_true, y_pred, labels=CLASSES_0_TO_4, average='macro', zero_division=0))
    m_rec = float(recall_score(y_true, y_pred, labels=CLASSES_0_TO_4, average='macro', zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    if probs is not None:
        probs_arr = np.asarray(probs, dtype=np.float64)
        y_onehot = label_binarize(y_true, classes=CLASSES_0_TO_4)
        macro_aupr = float(average_precision_score(y_onehot, probs_arr, average='macro'))
        expected_class = (probs_arr * np.arange(5, dtype=np.float64)[None, :]).sum(axis=1)
        expected_mae = float(mean_absolute_error(y_true.astype(np.float64), expected_class))
    else:
        macro_aupr = float('nan')
        expected_mae = mae

    out = {
        'Accuracy': acc,
        'WeightedF1': wf1,
        'MacroF1': mf1,
        'Kappa': kappa,
        'QWK': qwk,
        'MAE': mae,
        'ExpectedMAE': expected_mae,
        'RMSE': rmse,
        'F1_c0': float(per_class_f1[0]),
        'F1_c1': float(per_class_f1[1]),
        'F1_c2': float(per_class_f1[2]),
        'F1_c3': float(per_class_f1[3]),
        'F1_c4': float(per_class_f1[4]),
        'MacroPrecision': m_prec,
        'MacroRecall': m_rec,
        'MCC': mcc,
        'MacroAUPR': macro_aupr,
    }
    return out


def aggregate_fold_metrics(fold_metrics_list):
    """Compute mean and std across folds for a list of metric dicts."""
    if not fold_metrics_list:
        return {}, {}
    keys = list(fold_metrics_list[0].keys())
    mean_dict, std_dict = {}, {}
    for k in keys:
        vals = [m[k] for m in fold_metrics_list if m.get(k) is not None and not np.isnan(m[k])]
        if vals:
            mean_dict[k] = float(np.mean(vals))
            std_dict[k] = float(np.std(vals))
        else:
            mean_dict[k] = float('nan')
            std_dict[k] = float('nan')
    return mean_dict, std_dict
