"""
Dump verbatim C5 diagnostics (K*, Silhouette, Eigengap, Softened Pair Ratio, and Entropy of q)
for both K ranges [4, 12] and [4, 24] across Folds 0-2 on Split B and Split A,
plus the full K=4..24 spectral score profile on Split B Folds 0-2,
and save verbatim output to reports/raw/c5_k_ranges_fold012.txt.
"""

import os
import io
import contextlib
import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from MSSF_plus.protocol import load_fold_data, set_global_seed
from MSSF_plus.views import build_fold_views
from MSSF_plus.soft_labels import build_soft_targets, _build_fused_affinity_on_train_drugs
from MSSF_plus.configs.experiments import get_experiment_config


def _profile_all_k_scores(affinity: np.ndarray, k_min: int, k_max: int, seed: int):
    n = affinity.shape[0]
    deg = affinity.sum(axis=1)
    inv_sqrt = np.where(deg > 1e-8, 1.0 / np.sqrt(np.maximum(deg, 1e-8)), 0.0)
    lap = np.eye(n) - (inv_sqrt[:, None] * affinity) * inv_sqrt[None, :]
    lap = 0.5 * (lap + lap.T)
    max_eig = min(k_max + 2, n - 1)
    evals, evecs = eigh(lap, subset_by_index=(0, max_eig - 1))
    evals = np.clip(evals, 0.0, 2.0)
    cand_ks = list(range(k_min, k_max + 1))
    gaps = {k: float(evals[k] - evals[k - 1]) for k in cand_ks}
    max_gap = max(gaps.values()) if gaps else 1.0
    rows = []
    for k in cand_ks:
        u = evecs[:, :k].copy()
        rn = np.linalg.norm(u, axis=1, keepdims=True)
        u_norm = np.where(rn > 1e-8, u / np.maximum(rn, 1e-8), 0.0)
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(u_norm)
        sil = float(silhouette_score(u_norm, labels)) if len(np.unique(labels)) >= 2 else -1.0
        ng = gaps[k] / max(max_gap, 1e-8)
        score = 0.6 * sil + 0.4 * ng
        rows.append((k, sil, gaps[k], ng, score))
    return rows


def main():
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'reports', 'raw')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'c5_k_ranges_fold012.txt')

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print("SCRIPT: MSSF_plus/scripts/dump_c5_k_ranges.py")
        print("COMMAND: MSSF_plus/.venv/bin/python -m MSSF_plus.scripts.dump_c5_k_ranges")
        print("=" * 115)
        print("=== 1. C5 SPECTRAL CLUSTERING DIAGNOSTICS: K IN [4, 12] VS K IN [4, 24] (FOLDS 0-2, SPLIT B & A) ===")
        print("=" * 115)
        print(
            f"{'Split':<5} | {'Fold':>4} | {'C5 Config':<22} | {'c5_mode':<12} | {'K_range':<8} | "
            f"{'K*':>3} | {'Silhouette':>10} | {'Eigengap':>8} | {'Softened Pairs (%)':>18} | "
            f"{'H(q) all':>10} | {'H(q) soft':>10}"
        )
        print("-" * 115)

        c5_configs = [
            'C5_knn',
            'C5_single_view',
            'C5_single_view_k24',
            'C5_multi_view',
            'C5_multi_view_k24',
        ]

        for split_type in ['B', 'A']:
            for fold in [0, 1, 2]:
                fold_seed = 42 + fold
                set_global_seed(fold_seed)
                fs = load_fold_data(fold, split_type)
                base_cfg = get_experiment_config('MSSF_v2_selfcorr')
                fv = build_fold_views(fs, base_cfg, device='cpu')

                for exp_name in c5_configs:
                    cfg = get_experiment_config(exp_name)
                    sr = build_soft_targets(fs, fv, cfg)
                    d = sr.diagnostics
                    k_rng = f"[{d['k_min']}, {d['k_max']}]" if d['k_max'] > 0 else "N/A(kNN)"
                    frac = d['frac_smoothed']
                    print(
                        f"{split_type:<5} | {fold:>4d} | {exp_name:<22} | {sr.mode:<12} | {k_rng:<8} | "
                        f"{d['n_clusters']:>3d} | {d['silhouette']:>10.4f} | {d['eigengap']:>8.4f} | "
                        f"{frac:>8.4f} ({frac*100:>6.2f}%) | {d['entropy_q']:>10.4f} | {d['entropy_q_smoothed']:>10.4f}"
                    )
            print("-" * 115)

        print("\n" + "=" * 115)
        print("=== 2. MULTI-VIEW SPECTRAL SCORE PROFILE FOR K = 4..24 ON SPLIT B & A (FOLDS 0-2) ===")
        print("=" * 115)
        for split_type in ['B', 'A']:
            for fold in [0, 1, 2]:
                fold_seed = 42 + fold
                set_global_seed(fold_seed)
                fs = load_fold_data(fold, split_type)
                fv = build_fold_views(fs, get_experiment_config('MSSF_v2_selfcorr'), device='cpu')
                aff = _build_fused_affinity_on_train_drugs(fv, fs.train_drugs, mode='multi_view', weighting='uniform')
                rows_12 = _profile_all_k_scores(aff, 4, 12, fold_seed)
                rows_24 = _profile_all_k_scores(aff, 4, 24, fold_seed)
                best_12 = max(rows_12, key=lambda r: r[4])
                best_24 = max(rows_24, key=lambda r: r[4])
                top5_24 = sorted(rows_24, key=lambda r: r[4], reverse=True)[:5]
                top5_str = ", ".join([f"K={r[0]}(score={r[4]:.4f}, sil={r[1]:.4f}, gap={r[2]:.4f})" for r in top5_24])
                print(
                    f"Split {split_type} Fold {fold} | Best in [4, 12]: K*={best_12[0]:2d} (score={best_12[4]:.4f}) | "
                    f"Best in [4, 24]: K*={best_24[0]:2d} (score={best_24[4]:.4f})"
                )
                print(f"  Top-5 K in [4, 24]: {top5_str}")

    text = buf.getvalue()
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(text, end="")


if __name__ == '__main__':
    main()
