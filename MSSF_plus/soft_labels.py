"""
C5 — Soft label generation via Multi-View Spectral Clustering (plus Single-View and kNN baselines).

Key invariants (Section 8 / C5):
  1. Clustering is fit ONLY on training drugs (`fold_split.train_drugs`), using non-profile drug
     similarities plus train-only profile similarities (D9, D10).
  2. Number of clusters K is selected automatically on training drugs via normalized Laplacian
     eigengap + spectral silhouette score, and logged in the fold summary.
  3. For every training pair (d, s) with d in cluster c and true label y in {0..4}, the empirical
     5-class distribution P_{c,s}^{(-d)} is computed over training pairs (d', s) with d' in c and
     d' != d (strict leave-one-out exclusion of d itself).
  4. If the number of leave-one-out neighbors n_{c,s}^{(-d)} < c5_min_samples, keeps onehot(y).
  5. Evaluation/test labels are NEVER modified. Cold-start drugs do not need cluster IDs at inference.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from MSSF_plus.protocol import FoldSplit, N_DRUGS, N_SIDES
from MSSF_plus.views import FoldViews


@dataclass
class SoftLabelResult:
    mode: str
    soft_targets: np.ndarray          # (N_train, 5) float32 probability distributions summing to 1
    drug_cluster_ids: np.ndarray      # (N_train,) int64 cluster ID for each training pair (-1 if mode=='none'/'knn')
    n_clusters: int                   # selected K (0 if mode in {'none', 'knn'})
    diagnostics: Dict[str, float]     # K, silhouette, eigengap, frac_smoothed


def _normalize_affinity(sim_sub: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Symmetrize, zero diagonal, clip non-negative, and apply symmetric degree normalization."""
    w = 0.5 * (sim_sub + sim_sub.T)
    w = np.maximum(w, 0.0)
    np.fill_diagonal(w, 0.0)
    deg = w.sum(axis=1)
    inv_sqrt_deg = np.where(deg > eps, 1.0 / np.sqrt(np.maximum(deg, eps)), 0.0)
    w_norm = (inv_sqrt_deg[:, None] * w) * inv_sqrt_deg[None, :]
    return 0.5 * (w_norm + w_norm.T)


def _build_fused_affinity_on_train_drugs(
    fold_views: FoldViews,
    train_drugs: np.ndarray,
    mode: str = 'multi_view',
    single_view_name: str = 'D5_stitch_five',
    weighting: str = 'uniform',
) -> np.ndarray:
    """Build normalized affinity matrix (N_tr_drugs, N_tr_drugs) on training drugs."""
    idx = np.asarray(train_drugs, dtype=int)
    if mode == 'single_view':
        if single_view_name in fold_views.drug_view_names:
            v_idx = fold_views.drug_view_names.index(single_view_name)
        else:
            v_idx = 0
        sub = fold_views.drug_views_list[v_idx][np.ix_(idx, idx)]
        return _normalize_affinity(sub)

    normed_views = []
    weights = []
    for name, mat in zip(fold_views.drug_view_names, fold_views.drug_views_list):
        sub = mat[np.ix_(idx, idx)]
        w_norm = _normalize_affinity(sub)
        off_mass = float((w_norm > 1e-8).mean())
        if off_mass < 1e-4:
            continue
        normed_views.append(w_norm)
        if weighting == 'quality':
            # Weight by effective non-empty row coverage on train drugs
            row_cov = float((w_norm.sum(axis=1) > 1e-6).mean())
            weights.append(max(row_cov, 0.05))
        else:
            weights.append(1.0)

    if not normed_views:
        return np.eye(len(idx), dtype=np.float64)

    w_arr = np.array(weights, dtype=np.float64)
    w_arr = w_arr / w_arr.sum()
    fused = sum(w * m for w, m in zip(w_arr, normed_views))
    return _normalize_affinity(fused)


def _select_k_and_cluster(
    affinity: np.ndarray,
    k_min: int = 5,
    k_max: int = 15,
    fixed_k: Optional[int] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, int, float, float]:
    """
    Spectral clustering on normalized affinity matrix with automatic K selection via
    Laplacian eigengap + spectral embedding Silhouette score.
    """
    n = affinity.shape[0]
    deg = affinity.sum(axis=1)
    inv_sqrt = np.where(deg > 1e-8, 1.0 / np.sqrt(np.maximum(deg, 1e-8)), 0.0)
    lap = np.eye(n) - (inv_sqrt[:, None] * affinity) * inv_sqrt[None, :]
    lap = 0.5 * (lap + lap.T)

    max_eig = min(k_max + 2, n - 1)
    evals, evecs = eigh(lap, subset_by_index=(0, max_eig - 1))
    evals = np.clip(evals, 0.0, 2.0)

    if fixed_k is not None and fixed_k >= 2:
        cand_ks = [int(fixed_k)]
    else:
        cand_ks = list(range(max(2, k_min), min(k_max, n // 5) + 1))

    best_score = -float('inf')
    best_k = cand_ks[0]
    best_labels = np.zeros(n, dtype=np.int64)
    best_sil = 0.0
    best_gap = 0.0

    # Compute eigengaps: gap(k) = evals[k] - evals[k - 1]
    gaps = {k: float(evals[k] - evals[k - 1]) for k in cand_ks}
    max_gap = max(gaps.values()) if gaps else 1.0

    for k in cand_ks:
        u = evecs[:, :k].copy()
        row_norms = np.linalg.norm(u, axis=1, keepdims=True)
        u_norm = np.where(row_norms > 1e-8, u / np.maximum(row_norms, 1e-8), 0.0)

        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(u_norm)
        if len(np.unique(labels)) < 2:
            sil = -1.0
        else:
            sil = float(silhouette_score(u_norm, labels))

        norm_gap = gaps[k] / max(max_gap, 1e-8)
        combined = 0.6 * sil + 0.4 * norm_gap
        if combined > best_score:
            best_score = combined
            best_k = int(k)
            best_labels = labels.astype(np.int64)
            best_sil = sil
            best_gap = gaps[k]

    return best_labels, best_k, best_sil, best_gap


def build_soft_targets(
    fold_split: FoldSplit,
    fold_views: FoldViews,
    cfg: dict,
) -> SoftLabelResult:
    """
    Build training soft targets (N_train, 5) according to C5 config:
        c5_mode: 'none' | 'knn' | 'single_view' | 'multi_view' (default: 'none')
        c5_lambda: float in [0, 1] (default: 0.2)
        c5_min_samples: int minimum leave-one-out samples per (cluster, SE) (default: 3)
        c5_k_min: int (default: 5)
        c5_k_max: int (default: 15)
        c5_fixed_k: optional int (default: None)
        c5_knn_k: int (default: 10)
        c5_single_view_name: str (default: 'D5_stitch_five')
        c5_view_weighting: 'uniform' | 'quality' (default: 'quality')
    """
    mode = cfg.get('c5_mode', 'none')
    lam = float(cfg.get('c5_lambda', cfg.get('c5_alpha', 0.2)))
    min_samples = int(cfg.get('c5_min_samples', 3))
    seed = int(cfg.get('seed', 42)) + int(fold_split.fold)

    train_arr = fold_split.train
    n_train = len(train_arr)
    d_col = train_arr[:, 0].astype(int)
    s_col = train_arr[:, 1].astype(int)
    y_col = np.clip(np.round(train_arr[:, 2]).astype(int) - 1, 0, 4)

    onehot = np.eye(5, dtype=np.float32)[y_col]

    if mode == 'none' or lam <= 0.0:
        return SoftLabelResult(
            mode='none',
            soft_targets=onehot,
            drug_cluster_ids=np.full(n_train, -1, dtype=np.int64),
            n_clusters=0,
            diagnostics={'n_clusters': 0, 'silhouette': 0.0, 'eigengap': 0.0, 'frac_smoothed': 0.0},
        )

    train_drugs = np.asarray(fold_split.train_drugs, dtype=int)
    drug_to_local = {int(d): i for i, d in enumerate(train_drugs)}

    if mode == 'knn':
        knn_k = int(cfg.get('c5_knn_k', 10))
        affinity = _build_fused_affinity_on_train_drugs(
            fold_views, train_drugs, mode='multi_view', weighting=cfg.get('c5_view_weighting', 'quality')
        )
        # Group training pairs by SE s: se -> list of (local_drug_idx, class_y)
        se_to_entries: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for s_id in np.unique(s_col):
            mask = (s_col == s_id)
            local_ds = np.array([drug_to_local[int(d)] for d in d_col[mask]], dtype=int)
            ys = y_col[mask]
            se_to_entries[int(s_id)] = (local_ds, ys)

        soft_targets = onehot.copy()
        n_smoothed = 0
        for i in range(n_train):
            loc_d = drug_to_local[int(d_col[i])]
            s_id = int(s_col[i])
            loc_ds, ys = se_to_entries[s_id]
            # Strict leave-one-out: exclude loc_d itself
            other_mask = (loc_ds != loc_d)
            if other_mask.sum() < min_samples:
                continue
            cand_ds = loc_ds[other_mask]
            cand_ys = ys[other_mask]
            sims = affinity[loc_d, cand_ds]
            pos_mask = sims > 1e-8
            if pos_mask.sum() < min_samples:
                continue
            cand_sims = sims[pos_mask]
            cand_ys = cand_ys[pos_mask]
            if len(cand_sims) > knn_k:
                top_idx = np.argsort(cand_sims)[-knn_k:]
                cand_sims = cand_sims[top_idx]
                cand_ys = cand_ys[top_idx]
            w_sum = cand_sims.sum()
            if w_sum <= 1e-8:
                continue
            dist = np.zeros(5, dtype=np.float32)
            for c_val in range(5):
                dist[c_val] = float(cand_sims[cand_ys == c_val].sum() / w_sum)
            soft_targets[i] = (1.0 - lam) * onehot[i] + lam * dist
            n_smoothed += 1

        frac_sm = float(n_smoothed / max(n_train, 1))
        return SoftLabelResult(
            mode='knn',
            soft_targets=soft_targets,
            drug_cluster_ids=np.full(n_train, -1, dtype=np.int64),
            n_clusters=0,
            diagnostics={'n_clusters': 0, 'silhouette': 0.0, 'eigengap': 0.0, 'frac_smoothed': frac_sm},
        )

    if mode not in ('single_view', 'multi_view'):
        raise ValueError(f"Unknown c5_mode={mode!r}; expected 'none', 'knn', 'single_view', or 'multi_view'.")

    affinity = _build_fused_affinity_on_train_drugs(
        fold_views,
        train_drugs,
        mode=mode,
        single_view_name=cfg.get('c5_single_view_name', 'D5_stitch_five'),
        weighting=cfg.get('c5_view_weighting', 'quality'),
    )
    cluster_labels_local, best_k, best_sil, best_gap = _select_k_and_cluster(
        affinity,
        k_min=int(cfg.get('c5_k_min', 5)),
        k_max=int(cfg.get('c5_k_max', 15)),
        fixed_k=cfg.get('c5_fixed_k', None),
        seed=seed,
    )

    # Map global drug_idx -> cluster_id in {0 .. best_k - 1}
    drug_to_cluster = np.full(N_DRUGS, -1, dtype=np.int64)
    drug_to_cluster[train_drugs] = cluster_labels_local

    pair_clusters = drug_to_cluster[d_col]

    # Build cluster x SE x 5 class count tensor on training pairs
    counts = np.zeros((best_k, N_SIDES, 5), dtype=np.float32)
    np.add.at(counts, (pair_clusters, s_col, y_col), 1.0)

    soft_targets = onehot.copy()
    n_smoothed = 0
    for i in range(n_train):
        c_id = int(pair_clusters[i])
        s_id = int(s_col[i])
        y_id = int(y_col[i])

        # Strict leave-one-out: subtract the current pair (d, s, y) itself
        loo_hist = counts[c_id, s_id].copy()
        loo_hist[y_id] = max(0.0, loo_hist[y_id] - 1.0)
        loo_total = float(loo_hist.sum())

        if loo_total >= min_samples:
            p_cs = loo_hist / loo_total
            soft_targets[i] = (1.0 - lam) * onehot[i] + lam * p_cs
            n_smoothed += 1

    frac_sm = float(n_smoothed / max(n_train, 1))
    return SoftLabelResult(
        mode=mode,
        soft_targets=soft_targets,
        drug_cluster_ids=pair_clusters,
        n_clusters=int(best_k),
        diagnostics={
            'n_clusters': int(best_k),
            'silhouette': float(best_sil),
            'eigengap': float(best_gap),
            'frac_smoothed': frac_sm,
        },
    )
