"""
C4 — Post-hoc Ordinal Conformal Prediction calibrated on `val_calib` (10%) and evaluated on `test`.

Implements 4 conformal prediction set methods at miscoverage level alpha (default 0.10 -> 90% coverage):
  1. `LAC`: Standard Least Ambiguous set-valued Classifiers (s(x, y) = 1 - p(y|x)).
  2. `APS`: Standard Adaptive Prediction Sets sorting classes by descending probability.
  3. `Ordinal_LAC`: Contiguous ordinal interval expansion from y_hat = argmax p(k|x),
     thresholding boundary class probability while strictly preserving nested contiguity.
  4. `Ordinal_APS`: Contiguous ordinal interval expansion from y_hat = argmax p(k|x),
     accumulating probability mass along the greedy contiguous path until reaching q_hat.

All 4 methods are strictly post-hoc: they never alter point predictions (argmax p)
and guarantee finite-sample marginal coverage >= 1 - alpha under exchangeability.
"""

from typing import Dict, Optional, Tuple
import math
import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float = 0.10) -> float:
    """
    Finite-sample conformal quantile: ceil((n + 1) * (1 - alpha)) / n empirical quantile.
    """
    n = len(scores)
    if n == 0:
        return 1.0
    level = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / float(n))
    return float(np.quantile(scores, level, method='higher'))


def is_contiguous_set(mask_row: np.ndarray) -> bool:
    """Check whether a boolean mask of length 5 forms a contiguous ordinal interval [l, u]."""
    idx = np.flatnonzero(mask_row)
    if len(idx) <= 1:
        return True
    return bool((idx[-1] - idx[0] + 1) == len(idx))


def _greedy_contiguous_orders(probs: np.ndarray) -> np.ndarray:
    """
    For each sample i with probability vector p = probs[i] (shape (5,)),
    construct a sequence of classes (c_0, c_1, c_2, c_3, c_4) such that for every m in {1..5},
    the prefix {c_0, ..., c_{m-1}} is a contiguous ordinal interval [l, u] subset of {0..4},
    starting at c_0 = argmax_k p_k and greedily expanding to the adjacent class (l-1 or u+1)
    with larger probability.

    Returns:
        orders: (N, 5) int64 array where orders[i, r] is the class added at step r (0..4).
    """
    n, k_classes = probs.shape
    orders = np.zeros((n, k_classes), dtype=np.int64)
    center = np.argmax(probs, axis=1)

    for i in range(n):
        c0 = int(center[i])
        orders[i, 0] = c0
        left = c0
        right = c0
        for step in range(1, k_classes):
            can_left = left > 0
            can_right = right < (k_classes - 1)
            if can_left and can_right:
                if probs[i, left - 1] >= probs[i, right + 1]:
                    left -= 1
                    orders[i, step] = left
                else:
                    right += 1
                    orders[i, step] = right
            elif can_left:
                left -= 1
                orders[i, step] = left
            else:
                right += 1
                orders[i, step] = right
    return orders


def _evaluate_prediction_sets(
    pred_sets: np.ndarray,
    y_true_0_to_4: np.ndarray,
    q_hat: float,
    test_drug_ids: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute conformal metrics given boolean prediction set matrix (N, 5), true classes (N,) in {0..4},
    and optional `test_drug_ids` (N,) for per-drug coverage statistics (Change 4).
    """
    n = len(y_true_0_to_4)
    covered = pred_sets[np.arange(n), y_true_0_to_4]
    set_sizes = pred_sets.sum(axis=1)
    contiguous = np.array([is_contiguous_set(pred_sets[i]) for i in range(n)], dtype=bool)

    res = {
        'q_hat': float(q_hat),
        'empirical_coverage': float(covered.mean()),
        'avg_set_size': float(set_sizes.mean()),
        'singleton_rate': float((set_sizes == 1).mean()),
        'contiguity_rate': float(contiguous.mean()),
    }
    for c in range(5):
        mask_c = y_true_0_to_4 == c
        if np.any(mask_c):
            res[f'coverage_c{c}'] = float(covered[mask_c].mean())
        else:
            res[f'coverage_c{c}'] = float('nan')

    if test_drug_ids is not None and len(test_drug_ids) == n:
        d_ids = np.asarray(test_drug_ids, dtype=int).flatten()
        uniq_drugs = np.unique(d_ids)
        if len(uniq_drugs) > 0:
            per_drug_cov = np.array(
                [float(covered[d_ids == d].mean()) for d in uniq_drugs], dtype=np.float64
            )
            res['drug_coverage_mean'] = float(per_drug_cov.mean())
            res['drug_coverage_std'] = float(per_drug_cov.std(ddof=0))
            res['drug_coverage_lt80_rate'] = float((per_drug_cov < 0.80).mean())
            res['n_test_drugs'] = float(len(uniq_drugs))
        else:
            res['drug_coverage_mean'] = float('nan')
            res['drug_coverage_std'] = float('nan')
            res['drug_coverage_lt80_rate'] = float('nan')
            res['n_test_drugs'] = 0.0
    else:
        res['drug_coverage_mean'] = float('nan')
        res['drug_coverage_std'] = float('nan')
        res['drug_coverage_lt80_rate'] = float('nan')
        res['n_test_drugs'] = float('nan')

    return res


def run_lac(
    cal_probs: np.ndarray,
    cal_y: np.ndarray,
    test_probs: np.ndarray,
    test_y: np.ndarray,
    alpha: float = 0.10,
    test_drug_ids: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Standard LAC: s(x, y) = 1 - p(y|x)."""
    n_cal = len(cal_y)
    cal_scores = 1.0 - cal_probs[np.arange(n_cal), cal_y]
    q_hat = conformal_quantile(cal_scores, alpha=alpha)

    test_scores = 1.0 - test_probs
    pred_sets = test_scores <= q_hat
    # Guarantee non-empty set by always including argmax
    top1 = np.argmax(test_probs, axis=1)
    pred_sets[np.arange(len(test_probs)), top1] = True

    return _evaluate_prediction_sets(pred_sets, test_y, q_hat, test_drug_ids=test_drug_ids), pred_sets


def run_aps(
    cal_probs: np.ndarray,
    cal_y: np.ndarray,
    test_probs: np.ndarray,
    test_y: np.ndarray,
    alpha: float = 0.10,
    seed: int = 42,
    test_drug_ids: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Standard randomized APS sorting classes by descending probability."""
    rng = np.random.RandomState(seed)

    # Calibration scores
    n_cal = len(cal_y)
    cal_order = np.argsort(-cal_probs, axis=1)
    cal_sorted_p = np.take_along_axis(cal_probs, cal_order, axis=1)
    cal_cum_before = np.cumsum(cal_sorted_p, axis=1) - cal_sorted_p

    # Find rank of true class in sorted order
    cal_ranks = np.argmax(cal_order == cal_y[:, None], axis=1)
    u_cal = rng.uniform(0.0, 1.0, size=n_cal)
    cal_scores = (
        cal_cum_before[np.arange(n_cal), cal_ranks]
        + u_cal * cal_sorted_p[np.arange(n_cal), cal_ranks]
    )
    q_hat = conformal_quantile(cal_scores, alpha=alpha)

    # Test prediction sets
    n_test = len(test_y)
    test_order = np.argsort(-test_probs, axis=1)
    test_sorted_p = np.take_along_axis(test_probs, test_order, axis=1)
    test_cum_before = np.cumsum(test_sorted_p, axis=1) - test_sorted_p
    u_test = rng.uniform(0.0, 1.0, size=(n_test, 1))
    test_cand_scores = test_cum_before + u_test * test_sorted_p
    include_sorted = test_cand_scores <= q_hat
    include_sorted[:, 0] = True  # Always include top-1 class

    pred_sets = np.zeros_like(test_probs, dtype=bool)
    np.put_along_axis(pred_sets, test_order, include_sorted, axis=1)

    return _evaluate_prediction_sets(pred_sets, test_y, q_hat, test_drug_ids=test_drug_ids), pred_sets


def run_ordinal_lac(
    cal_probs: np.ndarray,
    cal_y: np.ndarray,
    test_probs: np.ndarray,
    test_y: np.ndarray,
    alpha: float = 0.10,
    test_drug_ids: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Ordinal LAC: expands a contiguous interval greedily from argmax p(k|x).
    Along the nested interval sequence I_1 subset I_2 subset ... subset I_5,
    defines non-conformity at step r as s_r(x) = max_{j <= r} (1 - p(c_j | x)),
    which is monotonically non-decreasing in r and guarantees both exact coverage >= 1 - alpha
    and 100% contiguous ordinal intervals [l, u].
    """
    n_cal = len(cal_y)
    cal_orders = _greedy_contiguous_orders(cal_probs)
    cal_ordered_p = np.take_along_axis(cal_probs, cal_orders, axis=1)
    cal_step_scores = np.maximum.accumulate(1.0 - cal_ordered_p, axis=1)
    cal_step_scores[:, 0] = 0.0  # Top-1 center is always included

    cal_ranks = np.argmax(cal_orders == cal_y[:, None], axis=1)
    cal_scores = cal_step_scores[np.arange(n_cal), cal_ranks]
    q_hat = conformal_quantile(cal_scores, alpha=alpha)

    test_orders = _greedy_contiguous_orders(test_probs)
    test_ordered_p = np.take_along_axis(test_probs, test_orders, axis=1)
    test_step_scores = np.maximum.accumulate(1.0 - test_ordered_p, axis=1)
    test_step_scores[:, 0] = 0.0
    include_ordered = test_step_scores <= q_hat

    pred_sets = np.zeros_like(test_probs, dtype=bool)
    np.put_along_axis(pred_sets, test_orders, include_ordered, axis=1)

    return _evaluate_prediction_sets(pred_sets, test_y, q_hat, test_drug_ids=test_drug_ids), pred_sets


def run_ordinal_aps(
    cal_probs: np.ndarray,
    cal_y: np.ndarray,
    test_probs: np.ndarray,
    test_y: np.ndarray,
    alpha: float = 0.10,
    seed: int = 42,
    test_drug_ids: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Ordinal APS: accumulates probability mass along the greedy contiguous interval path
    starting from argmax p(k|x). Guarantees both finite-sample coverage >= 1 - alpha
    and 100% contiguous ordinal intervals [l, u].
    """
    rng = np.random.RandomState(seed)

    n_cal = len(cal_y)
    cal_orders = _greedy_contiguous_orders(cal_probs)
    cal_ordered_p = np.take_along_axis(cal_probs, cal_orders, axis=1)
    cal_cum_before = np.cumsum(cal_ordered_p, axis=1) - cal_ordered_p

    cal_ranks = np.argmax(cal_orders == cal_y[:, None], axis=1)
    u_cal = rng.uniform(0.0, 1.0, size=n_cal)
    cal_scores = (
        cal_cum_before[np.arange(n_cal), cal_ranks]
        + u_cal * cal_ordered_p[np.arange(n_cal), cal_ranks]
    )
    q_hat = conformal_quantile(cal_scores, alpha=alpha)

    n_test = len(test_y)
    test_orders = _greedy_contiguous_orders(test_probs)
    test_ordered_p = np.take_along_axis(test_probs, test_orders, axis=1)
    test_cum_before = np.cumsum(test_ordered_p, axis=1) - test_ordered_p
    u_test = rng.uniform(0.0, 1.0, size=(n_test, 1))
    test_cand_scores = test_cum_before + u_test * test_ordered_p
    include_ordered = test_cand_scores <= q_hat
    include_ordered[:, 0] = True

    pred_sets = np.zeros_like(test_probs, dtype=bool)
    np.put_along_axis(pred_sets, test_orders, include_ordered, axis=1)

    return _evaluate_prediction_sets(pred_sets, test_y, q_hat, test_drug_ids=test_drug_ids), pred_sets


def evaluate_all_conformal_methods(
    valcal_probs: np.ndarray,
    valcal_labels_1_to_5: np.ndarray,
    test_probs: np.ndarray,
    test_labels_1_to_5: np.ndarray,
    alpha: float = 0.10,
    seed: int = 42,
    test_drug_ids: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Calibrate and evaluate all 4 C4 conformal methods (`LAC`, `APS`, `Ordinal_LAC`, `Ordinal_APS`)
    on a single fold using `val_calib` and `test`.
     Note (Split B): `val_calib` is partitioned by drug, so coverage guarantees hold at the
    drug level on average while pairs within the same drug are correlated.
    """
    cal_y = np.clip(np.round(valcal_labels_1_to_5).astype(int) - 1, 0, 4)
    test_y = np.clip(np.round(test_labels_1_to_5).astype(int) - 1, 0, 4)

    lac_metrics, _ = run_lac(
        valcal_probs, cal_y, test_probs, test_y, alpha=alpha, test_drug_ids=test_drug_ids
    )
    aps_metrics, _ = run_aps(
        valcal_probs, cal_y, test_probs, test_y, alpha=alpha, seed=seed, test_drug_ids=test_drug_ids
    )
    ord_lac_metrics, _ = run_ordinal_lac(
        valcal_probs, cal_y, test_probs, test_y, alpha=alpha, test_drug_ids=test_drug_ids
    )
    ord_aps_metrics, _ = run_ordinal_aps(
        valcal_probs, cal_y, test_probs, test_y, alpha=alpha, seed=seed, test_drug_ids=test_drug_ids
    )

    return {
        'LAC': lac_metrics,
        'APS': aps_metrics,
        'Ordinal_LAC': ord_lac_metrics,
        'Ordinal_APS': ord_aps_metrics,
    }
