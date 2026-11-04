"""
Feature view construction, per-fold leak-free profile similarity computation,
C1 missing-view handling (availability indicator, zero/drop/mean/regress/diffusion imputation),
and C2 SE MedDRA ontology view construction.
"""

import os
import math
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.nn.functional as F

from MSSF_plus.protocol import REPO_DIR, SHARED_DIR, FoldSplit, N_DRUGS, N_SIDES
from split_adapter_v2 import make_train_only_freq

MSSF_750_DIR = os.path.join(SHARED_DIR, 'mssf_750')

STATIC_DRUG_VIEW_NAMES = [
    'D1_stitch_one',
    'D2_stitch_two',
    'D3_stitch_three',
    'D4_stitch_four',
    'D5_stitch_five',
    'D6_mol2vec_cos',
    'D7_target_cos',
    'D8_fingerprint',
    'D11_pathway_enzyme',
]

ALL_11_DRUG_VIEW_NAMES = [
    'D1_stitch_one',
    'D2_stitch_two',
    'D3_stitch_three',
    'D4_stitch_four',
    'D5_stitch_five',
    'D6_mol2vec_cos',
    'D7_target_cos',
    'D8_fingerprint',
    'D9_profile_freq_cos',
    'D10_profile_bin_cos',
    'D11_pathway_enzyme',
]

_STATIC_CACHE: Optional[Dict[str, np.ndarray]] = None


def _load_pkl(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_static_raw_views() -> Dict[str, np.ndarray]:
    """Load and cache fold-independent drug and SE similarity matrices."""
    global _STATIC_CACHE
    if _STATIC_CACHE is not None:
        return {k: v.copy() for k, v in _STATIC_CACHE.items()}

    d = {}
    for idx, suffix in enumerate(['one', 'two', 'three', 'four', 'five']):
        mat = np.array(_load_pkl(os.path.join(MSSF_750_DIR, f'Text_similarity_{suffix}.pkl')), dtype=np.float64)
        d[f'D{idx+1}_stitch_{suffix}'] = mat

    drug_mol = np.array(_load_pkl(os.path.join(MSSF_750_DIR, 'drug_mol.pkl')), dtype=np.float64)
    d['D6_mol2vec_cos'] = cosine_similarity(drug_mol)

    drug_target = np.array(_load_pkl(os.path.join(MSSF_750_DIR, 'drug_target.pkl')), dtype=np.float64)
    d['D7_target_cos'] = cosine_similarity(drug_target)

    d['D8_fingerprint'] = np.array(
        _load_pkl(os.path.join(MSSF_750_DIR, 'fingerprint_similarity.pkl')), dtype=np.float64
    )
    d['D11_pathway_enzyme'] = np.array(
        _load_pkl(os.path.join(MSSF_750_DIR, 'drug_pathway_enzyme_similarity.pkl')), dtype=np.float64
    )

    d['S1_adrecs_semantic'] = np.array(
        _load_pkl(os.path.join(MSSF_750_DIR, 'side_effect_semantic.pkl')), dtype=np.float64
    )
    glove = np.array(_load_pkl(os.path.join(MSSF_750_DIR, 'glove_wordEmbedding.pkl')), dtype=np.float64)
    d['S2_glove_cos'] = cosine_similarity(glove)

    # C2: MedDRA 243-D ontology features from shared_data/side_effect_label_750.mat['node_label']
    se_mat = sio.loadmat(os.path.join(SHARED_DIR, 'side_effect_label_750.mat'))
    node_label = np.array(se_mat['node_label'], dtype=np.float64)  # (994, 243)
    d['S5_meddra_ontology_cos'] = cosine_similarity(node_label)

    _STATIC_CACHE = d
    return {k: v.copy() for k, v in _STATIC_CACHE.items()}


def compute_view_availability(sim_matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Compute binary availability indicator per row:
        a[i] = 1.0 if sum_{j != i} |S[i, j]| > eps else 0.0
    """
    off = sim_matrix.copy()
    if off.shape[0] == off.shape[1]:
        np.fill_diagonal(off, 0.0)
    return (np.abs(off).sum(axis=1) > eps).astype(np.float32)


def compute_profile_views_from_train(train_pairs: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Construct the 750x994 drug-SE frequency matrix strictly from `train_pairs` (80% split)
    and compute D9, D10 (750x750) and S3, S4 (994x994).
    """
    drug_side_train = make_train_only_freq(train_pairs, shape=(N_DRUGS, N_SIDES), dtype=np.float64)
    drug_side_bin = (drug_side_train > 0).astype(np.float64)

    return {
        'drug_side_train': drug_side_train,
        'D9_profile_freq_cos': cosine_similarity(drug_side_train),
        'D10_profile_bin_cos': cosine_similarity(drug_side_bin),
        'S3_profile_freq_cos': cosine_similarity(drug_side_train.T),
        'S4_profile_bin_cos': cosine_similarity(drug_side_bin.T),
    }


# ============================================================
# C1 Imputation Models: MLP Regressor & Conditional Diffusion
# ============================================================

class ProfileViewRegressor(nn.Module):
    """MLP predicting missing [D9, D10] rows (2 * 750 = 1500-D) from static drug views + availability."""

    def __init__(self, in_dim: int, out_dim: int = 2 * N_DRUGS, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid(),  # Cosine similarities of non-negative profiles lie in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_drug_crossfit_splits(
    train_drugs: np.ndarray,
    n_splits: int = 5,
    val_frac: float = 0.10,
    seed: int = 42,
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray, np.ndarray, int]]]:
    """
    Deterministically partition `train_drugs` by drug into `n_splits` disjoint groups.
    For each group k:
      - `holdout`: group k (receives strictly out-of-fold predictions)
      - `pool`: remaining K-1 groups, split into (1 - val_frac) `fit_sub` and val_frac `val_sub`
                for internal early stopping.
    """
    tr_idx = np.sort(np.unique(np.asarray(train_drugs, dtype=int)))
    if len(tr_idx) < 2:
        raise ValueError(f"Need at least 2 training drugs for cross-fitting, got {len(tr_idx)}.")
    eff_splits = min(int(n_splits), len(tr_idx))

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(tr_idx))
    groups = np.array_split(tr_idx[perm], eff_splits)

    folds = []
    for k in range(eff_splits):
        fold_seed = int(seed) + 100 * (k + 1)
        holdout = groups[k]
        pool = np.concatenate([groups[m] for m in range(eff_splits) if m != k])
        rng_k = np.random.RandomState(fold_seed)
        perm_k = rng_k.permutation(len(pool))
        n_val = min(max(10, int(round(val_frac * len(pool)))), max(1, len(pool) - 1))
        val_sub = pool[perm_k[:n_val]]
        fit_sub = pool[perm_k[n_val:]]
        folds.append((fit_sub, val_sub, holdout, fold_seed))

    return tr_idx, folds


def train_and_impute_regress(
    static_concat: np.ndarray,
    static_avail: np.ndarray,
    d9_raw: np.ndarray,
    d10_raw: np.ndarray,
    train_drugs: np.ndarray,
    device: torch.device,
    epochs: int = 80,
    lr: float = 1e-3,
    seed: int = 42,
    n_splits: int = 5,
    return_ensemble: bool = False,
):
    """
    K-fold cross-fitting (`n_splits=5`) of ProfileViewRegressor on `train_drugs`:
      - Splits `train_drugs` into K=5 disjoint drug groups (fixed seed).
      - For each group k, trains one regressor on the remaining K-1 groups (with 10% of those
        K-1 groups held out for early stopping) and predicts out-of-fold for group k.
      - For non-train drugs (`val_select`, `val_calib`, `test`), uses the average prediction
        of all K regressors.
    """
    x_all = np.hstack([static_concat, static_avail]).astype(np.float32)
    y_all = np.hstack([d9_raw, d10_raw]).astype(np.float32)

    x_t = torch.from_numpy(x_all).to(device)
    y_t = torch.from_numpy(y_all).to(device)

    tr_idx, cf_folds = _make_drug_crossfit_splits(train_drugs, n_splits=n_splits, val_frac=0.10, seed=seed)
    eff_splits = len(cf_folds)
    non_train_mask = ~np.isin(np.arange(N_DRUGS), tr_idx)

    oof_pred = np.zeros_like(y_all, dtype=np.float64)
    ens_pred = np.zeros_like(y_all, dtype=np.float64)
    batch_size = 64

    for fit_sub, val_sub, holdout, fold_seed in cf_folds:
        torch.manual_seed(fold_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(fold_seed)

        model = ProfileViewRegressor(in_dim=x_all.shape[1], out_dim=y_all.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        best_val_loss = float('inf')
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        fit_x, fit_y = x_t[fit_sub], y_t[fit_sub]
        val_x, val_y = x_t[val_sub], y_t[val_sub]

        for _ in range(epochs):
            model.train()
            order = torch.randperm(len(fit_sub), device=device)
            for start in range(0, len(fit_sub), batch_size):
                b = order[start:start + batch_size]
                opt.zero_grad()
                pred = model(fit_x[b])
                mse = F.mse_loss(pred, fit_y[b])
                cos = 1.0 - F.cosine_similarity(pred, fit_y[b], dim=-1).mean()
                loss = mse + 0.1 * cos
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                v_pred = model(val_x)
                v_loss = float(F.mse_loss(v_pred, val_y).item())
                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pred_k = model(x_t).cpu().numpy().astype(np.float64)

        oof_pred[holdout] = pred_k[holdout]
        ens_pred += pred_k / float(eff_splits)

    final_pred = np.zeros_like(oof_pred)
    final_pred[tr_idx] = oof_pred[tr_idx]
    final_pred[non_train_mask] = ens_pred[non_train_mask]

    d9_pred = final_pred[:, :N_DRUGS].copy()
    d10_pred = final_pred[:, N_DRUGS:].copy()
    np.fill_diagonal(d9_pred, 1.0)
    np.fill_diagonal(d10_pred, 1.0)

    if return_ensemble:
        d9_ens = ens_pred[:, :N_DRUGS].copy()
        d10_ens = ens_pred[:, N_DRUGS:].copy()
        np.fill_diagonal(d9_ens, 1.0)
        np.fill_diagonal(d10_ens, 1.0)
        return d9_pred, d10_pred, d9_ens, d10_ens
    return d9_pred, d10_pred


def train_and_impute_ridge(
    static_concat: np.ndarray,
    static_avail: np.ndarray,
    d9_raw: np.ndarray,
    d10_raw: np.ndarray,
    train_drugs: np.ndarray,
    seed: int = 42,
    n_splits: int = 5,
    alphas: Tuple[float, ...] = (0.1, 1.0, 10.0, 50.0, 100.0, 200.0, 500.0, 1000.0),
    return_ensemble: bool = False,
):
    """
    Cheap C1 imputation variant (`regress_ridge`): K-fold cross-fitting (`n_splits=5`) of
    multi-output Ridge regression on `train_drugs`.
      - Uses the exact same K=5 cross-fit splits (`_make_drug_crossfit_splits`) as `regress`.
      - In each cross-fit fold k, selects regularization `alpha` in `alphas` minimizing MSE
        on `val_sub` (10% of the K-1 fit folds), then refits on `fit_sub U val_sub` (all K-1
        fit folds) to predict out-of-fold for `holdout` and averages across the K models for
        non-train drugs (`val_select`, `val_calib`, `test`).
    """
    x_all = np.hstack([static_concat, static_avail]).astype(np.float64)
    y_all = np.hstack([d9_raw, d10_raw]).astype(np.float64)

    tr_idx = np.unique(np.asarray(train_drugs, dtype=int))
    eff_splits = max(2, min(int(n_splits), len(tr_idx)))
    cf_splits = _make_drug_crossfit_splits(tr_idx, n_splits=eff_splits, val_frac=0.10, seed=seed)

    oof_pred = np.zeros_like(y_all, dtype=np.float64)
    ens_pred = np.zeros_like(y_all, dtype=np.float64)
    non_train_mask = np.ones(N_DRUGS, dtype=bool)
    non_train_mask[tr_idx] = False

    def _fit_predict_dual_ridge(x_tr: np.ndarray, y_tr: np.ndarray, x_ev: np.ndarray, alpha: float) -> np.ndarray:
        x_mean = x_tr.mean(axis=0, keepdims=True)
        y_mean = y_tr.mean(axis=0, keepdims=True)
        xc = x_tr - x_mean
        yc = y_tr - y_mean
        gram = xc @ xc.T
        gram_reg = gram.copy()
        gram_reg[np.diag_indices_from(gram_reg)] += float(alpha)
        dual_coef = np.linalg.solve(gram_reg, yc)
        pred = ((x_ev - x_mean) @ xc.T) @ dual_coef + y_mean
        return np.clip(pred, 0.0, 1.0)

    for fit_sub, val_sub, holdout in cf_splits:
        x_fit = x_all[fit_sub]
        y_fit = y_all[fit_sub]
        x_val = x_all[val_sub]
        y_val = y_all[val_sub]

        x_mean = x_fit.mean(axis=0, keepdims=True)
        y_mean = y_fit.mean(axis=0, keepdims=True)
        xc = x_fit - x_mean
        yc = y_fit - y_mean
        gram = xc @ xc.T
        k_val = (x_val - x_mean) @ xc.T

        best_alpha = float(alphas[0])
        best_val_mse = float('inf')
        for a in alphas:
            gram_reg = gram.copy()
            gram_reg[np.diag_indices_from(gram_reg)] += float(a)
            dual_coef = np.linalg.solve(gram_reg, yc)
            pred_val = np.clip(k_val @ dual_coef + y_mean, 0.0, 1.0)
            mse_val = float(np.mean((pred_val - y_val) ** 2))
            if mse_val < best_val_mse:
                best_val_mse = mse_val
                best_alpha = float(a)

        fit_full = np.concatenate([fit_sub, val_sub])
        pred_k = _fit_predict_dual_ridge(x_all[fit_full], y_all[fit_full], x_all, best_alpha)
        oof_pred[holdout] = pred_k[holdout]
        ens_pred += pred_k / float(eff_splits)

    final_pred = np.zeros_like(oof_pred)
    final_pred[tr_idx] = oof_pred[tr_idx]
    final_pred[non_train_mask] = ens_pred[non_train_mask]

    d9_pred = final_pred[:, :N_DRUGS].copy()
    d10_pred = final_pred[:, N_DRUGS:].copy()
    np.fill_diagonal(d9_pred, 1.0)
    np.fill_diagonal(d10_pred, 1.0)

    if return_ensemble:
        d9_ens = ens_pred[:, :N_DRUGS].copy()
        d10_ens = ens_pred[:, N_DRUGS:].copy()
        np.fill_diagonal(d9_ens, 1.0)
        np.fill_diagonal(d10_ens, 1.0)
        return d9_pred, d10_pred, d9_ens, d10_ens
    return d9_pred, d10_pred


def train_and_impute_knn(
    static_drug_mats: List[np.ndarray],
    static_avail: np.ndarray,
    d9_raw: np.ndarray,
    d10_raw: np.ndarray,
    train_drugs: np.ndarray,
    seed: int = 42,
    n_splits: int = 5,
    ks: Tuple[int, ...] = (5, 10, 15, 20, 30, 50),
    return_ensemble: bool = False,
):
    """
    Cheap C1 imputation variant (`regress_knn`): K-fold cross-fitting (`n_splits=5`) of
    static-view k-nearest-neighbor row averaging on `train_drugs`.
      - Computes the availability-weighted average static similarity across the 9 static drug
        views (`D1..D8, D11`).
      - Uses the exact same K=5 cross-fit splits (`_make_drug_crossfit_splits`) as `regress`.
      - In each cross-fit fold, selects neighbor count `k` in `ks` minimizing MSE on `val_sub`,
        then averages the `D9, D10` rows of the top-`k` nearest neighbors in `fit_sub U val_sub`
        (excluding self) to predict out-of-fold for `holdout` and averages across the K folds
        for non-train drugs (`val_select`, `val_calib`, `test`).
    """
    y_all = np.hstack([d9_raw, d10_raw]).astype(np.float64)

    # Availability-weighted average static similarity across the 9 static drug views
    sim_sum = np.zeros((N_DRUGS, N_DRUGS), dtype=np.float64)
    sim_cnt = np.zeros((N_DRUGS, N_DRUGS), dtype=np.float64)
    for v_idx, mat in enumerate(static_drug_mats):
        av = (static_avail[:, v_idx] > 0.5).astype(np.float64)
        pair_av = np.outer(av, av)
        sim_sum += np.asarray(mat, dtype=np.float64) * pair_av
        sim_cnt += pair_av
    s_static = np.where(sim_cnt > 0.0, sim_sum / np.maximum(sim_cnt, 1.0), 0.0)

    tr_idx = np.unique(np.asarray(train_drugs, dtype=int))
    eff_splits = max(2, min(int(n_splits), len(tr_idx)))
    cf_splits = _make_drug_crossfit_splits(tr_idx, n_splits=eff_splits, val_frac=0.10, seed=seed)

    oof_pred = np.zeros_like(y_all, dtype=np.float64)
    ens_pred = np.zeros_like(y_all, dtype=np.float64)
    non_train_mask = np.ones(N_DRUGS, dtype=bool)
    non_train_mask[tr_idx] = False

    def _knn_avg_rows(query_idx: np.ndarray, ref_idx: np.ndarray, k_val: int) -> np.ndarray:
        k_eff = min(int(k_val), len(ref_idx))
        sim_sub = s_static[np.ix_(query_idx, ref_idx)].copy()
        for qi, q_drug in enumerate(query_idx):
            self_pos = np.where(ref_idx == q_drug)[0]
            if len(self_pos) > 0:
                sim_sub[qi, self_pos] = -1e9
        top_pos = np.argsort(-sim_sub, axis=1)[:, :k_eff]
        y_ref = y_all[ref_idx]
        out = np.zeros((len(query_idx), y_all.shape[1]), dtype=np.float64)
        for qi in range(len(query_idx)):
            out[qi] = y_ref[top_pos[qi]].mean(axis=0)
        return np.clip(out, 0.0, 1.0)

    all_drug_idx = np.arange(N_DRUGS, dtype=int)
    for fit_sub, val_sub, holdout in cf_splits:
        best_k = int(ks[0])
        best_val_mse = float('inf')
        y_val = y_all[val_sub]
        for k_cand in ks:
            pred_val = _knn_avg_rows(val_sub, fit_sub, int(k_cand))
            mse_val = float(np.mean((pred_val - y_val) ** 2))
            if mse_val < best_val_mse:
                best_val_mse = mse_val
                best_k = int(k_cand)

        fit_full = np.concatenate([fit_sub, val_sub])
        pred_k = _knn_avg_rows(all_drug_idx, fit_full, best_k)
        oof_pred[holdout] = pred_k[holdout]
        ens_pred += pred_k / float(eff_splits)

    final_pred = np.zeros_like(oof_pred)
    final_pred[tr_idx] = oof_pred[tr_idx]
    final_pred[non_train_mask] = ens_pred[non_train_mask]

    d9_pred = final_pred[:, :N_DRUGS].copy()
    d10_pred = final_pred[:, N_DRUGS:].copy()
    np.fill_diagonal(d9_pred, 1.0)
    np.fill_diagonal(d10_pred, 1.0)

    if return_ensemble:
        d9_ens = ens_pred[:, :N_DRUGS].copy()
        d10_ens = ens_pred[:, N_DRUGS:].copy()
        np.fill_diagonal(d9_ens, 1.0)
        np.fill_diagonal(d10_ens, 1.0)
        return d9_pred, d10_pred, d9_ens, d10_ens
    return d9_pred, d10_pred


class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ConditionalDenoiserMLP(nn.Module):
    """Conditional denoiser predicting clean profile views x_0 from (x_t, t, cond)."""

    def __init__(self, cond_in_dim: int, target_dim: int = 2 * N_DRUGS, hidden_dim: int = 256):
        super().__init__()
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmb(64),
            nn.Linear(64, hidden_dim),
            nn.SiLU(),
        )
        self.x_proj = nn.Linear(target_dim, hidden_dim)
        self.blocks = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, target_dim),
            nn.Sigmoid(),  # Direct x_0 prediction bounded in [0, 1]
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.x_proj(x_t) + self.cond_proj(cond) + self.time_mlp(t)
        return self.blocks(h)


def train_and_impute_diffusion(
    static_concat: np.ndarray,
    static_avail: np.ndarray,
    d9_raw: np.ndarray,
    d10_raw: np.ndarray,
    train_drugs: np.ndarray,
    device: torch.device,
    n_timesteps: int = 25,
    n_samples: int = 10,
    epochs: int = 100,
    lr: float = 1e-3,
    seed: int = 42,
    n_splits: int = 5,
    return_ensemble: bool = False,
):
    """
    K-fold cross-fitting (`n_splits=5`) of conditional DDPM (x_0-prediction) on `train_drugs`:
      - Splits `train_drugs` into K=5 disjoint drug groups (fixed seed).
      - For each group k, trains one denoiser on the remaining K-1 groups (with 10% of those
        K-1 groups held out for deterministic multi-timestep denoising early stopping) and
        predicts out-of-fold for group k by averaging `n_samples` reverse trajectories.
      - For non-train drugs (`val_select`, `val_calib`, `test`), averages predictions across
        all K denoisers.
    """
    cond_all = np.hstack([static_concat, static_avail]).astype(np.float32)
    x0_all = np.hstack([d9_raw, d10_raw]).astype(np.float32)

    cond_t = torch.from_numpy(cond_all).to(device)
    x0_t = torch.from_numpy(x0_all).to(device)

    betas = torch.linspace(1e-4, 0.2, n_timesteps, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    tr_idx, cf_folds = _make_drug_crossfit_splits(train_drugs, n_splits=n_splits, val_frac=0.10, seed=seed)
    eff_splits = len(cf_folds)
    non_train_idx = np.where(~np.isin(np.arange(N_DRUGS), tr_idx))[0]
    non_train_mask = ~np.isin(np.arange(N_DRUGS), tr_idx)

    target_dim = x0_all.shape[1]
    oof_pred = np.zeros_like(x0_all, dtype=np.float64)
    ens_pred = np.zeros_like(x0_all, dtype=np.float64)
    batch_size = 64

    val_grid_np = np.unique(np.linspace(0, n_timesteps - 1, min(5, n_timesteps), dtype=int))

    for fit_sub, val_sub, holdout, fold_seed in cf_folds:
        torch.manual_seed(fold_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(fold_seed)

        model = ConditionalDenoiserMLP(cond_in_dim=cond_all.shape[1], target_dim=target_dim).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        fit_cond = cond_t[fit_sub]
        fit_x0 = x0_t[fit_sub]

        # Pre-generate deterministic validation (x_noisy, t, cond -> x0) pairs for noise-free early stopping
        val_rng = np.random.RandomState(fold_seed + 777)
        n_rep = len(val_grid_np)
        val_cond_rep = cond_t[val_sub].repeat(n_rep, 1)
        val_x0_rep = x0_t[val_sub].repeat(n_rep, 1)
        val_t_rep = torch.as_tensor(
            np.repeat(val_grid_np, len(val_sub)), dtype=torch.long, device=device
        )
        val_noise = torch.from_numpy(
            val_rng.randn(val_x0_rep.shape[0], target_dim).astype(np.float32)
        ).to(device)
        val_ab = alpha_bar[val_t_rep].unsqueeze(1)
        val_x_noisy = torch.sqrt(val_ab) * val_x0_rep + torch.sqrt(1.0 - val_ab) * val_noise

        best_val_loss = float('inf')
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        for _ in range(epochs):
            model.train()
            order = torch.randperm(len(fit_sub), device=device)
            for start in range(0, len(fit_sub), batch_size):
                b = order[start:start + batch_size]
                x0_b = fit_x0[b]
                c_b = fit_cond[b]
                t = torch.randint(0, n_timesteps, (len(b),), device=device)
                noise = torch.randn_like(x0_b)
                ab = alpha_bar[t].unsqueeze(1)
                x_noisy = torch.sqrt(ab) * x0_b + torch.sqrt(1.0 - ab) * noise

                opt.zero_grad()
                x0_hat = model(x_noisy, t, c_b)
                loss = F.mse_loss(x0_hat, x0_b)
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                v_hat = model(val_x_noisy, val_t_rep, val_cond_rep)
                v_loss = float(F.mse_loss(v_hat, val_x0_rep).item())
                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        model.eval()

        if return_ensemble:
            eval_idx = np.arange(N_DRUGS, dtype=int)
        else:
            eval_idx = np.concatenate([holdout, non_train_idx]).astype(int)

        eval_cond = cond_t[eval_idx]
        n_eval = len(eval_idx)
        acc_samples = torch.zeros((n_eval, target_dim), dtype=torch.float32, device=device)

        eval_idx_t = torch.as_tensor(eval_idx, dtype=torch.long, device=device)
        with torch.no_grad():
            for s_idx in range(n_samples):
                torch.manual_seed(fold_seed + 1000 + s_idx)
                x_cur = torch.randn((N_DRUGS, target_dim), device=device)[eval_idx_t]
                for step in reversed(range(n_timesteps)):
                    t_vec = torch.full((n_eval,), step, dtype=torch.long, device=device)
                    x0_hat = model(x_cur, t_vec, eval_cond)
                    if step > 0:
                        ab_t = alpha_bar[step]
                        ab_prev = alpha_bar[step - 1]
                        a_t = alphas[step]
                        b_t = betas[step]
                        eps_hat = (x_cur - torch.sqrt(ab_t) * x0_hat) / torch.sqrt(1.0 - ab_t + 1e-8)
                        mean = (1.0 / torch.sqrt(a_t)) * (x_cur - (b_t / torch.sqrt(1.0 - ab_t + 1e-8)) * eps_hat)
                        var = b_t * (1.0 - ab_prev) / (1.0 - ab_t + 1e-8)
                        z_step = torch.randn((N_DRUGS, target_dim), device=device)[eval_idx_t]
                        x_cur = mean + torch.sqrt(torch.clamp(var, min=1e-8)) * z_step
                    else:
                        x_cur = x0_hat
                acc_samples += torch.clamp(x_cur, 0.0, 1.0)

        pred_eval = (acc_samples / float(n_samples)).cpu().numpy().astype(np.float64)
        if return_ensemble:
            oof_pred[holdout] = pred_eval[holdout]
            ens_pred += pred_eval / float(eff_splits)
        else:
            n_h = len(holdout)
            oof_pred[holdout] = pred_eval[:n_h]
            if len(non_train_idx) > 0:
                ens_pred[non_train_idx] += pred_eval[n_h:] / float(eff_splits)

    final_pred = np.zeros_like(oof_pred)
    final_pred[tr_idx] = oof_pred[tr_idx]
    final_pred[non_train_mask] = ens_pred[non_train_mask]

    d9_diff = final_pred[:, :N_DRUGS].copy()
    d10_diff = final_pred[:, N_DRUGS:].copy()
    np.fill_diagonal(d9_diff, 1.0)
    np.fill_diagonal(d10_diff, 1.0)

    if return_ensemble:
        d9_ens = ens_pred[:, :N_DRUGS].copy()
        d10_ens = ens_pred[:, N_DRUGS:].copy()
        np.fill_diagonal(d9_ens, 1.0)
        np.fill_diagonal(d10_ens, 1.0)
        return d9_diff, d10_diff, d9_ens, d10_ens
    return d9_diff, d10_diff


def compute_batch_self_corrected_profiles(
    drug_idx: torch.Tensor,
    se_idx: torch.Tensor,
    y_train: torch.Tensor,
    b_train: torch.Tensor,
    g_d9: torch.Tensor,
    norm2_d9: torch.Tensor,
    g_d10: torch.Tensor,
    norm2_d10: torch.Tensor,
    g_s3: torch.Tensor,
    norm2_s3: torch.Tensor,
    g_s4: torch.Tensor,
    norm2_s4: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized GPU/device computation of leave-one-pair-out profile similarity rows
    (`D9[d, :]`, `D10[d, :]`, `S3[s, :]`, `S4[s, :]`) for a batch of pairs `(drug_idx, se_idx)`.

    When only the single cell `(d, s)` of `Y_train` is set to 0 (`Y_mod[d, s] = 0`):
      - Numerator: `num_j = G[d, j] - Y[d, s] * Y[j, s]`.
      - Denominator:
        * For `j == d`, row `d` loses `Y[d, s]^2`, so its squared norm becomes `||x_d||^2 - Y[d, s]^2`,
          and the diagonal `D9[d, d]` is `1.0` if `||x_d||^2 - Y[d, s]^2 > 0` else `0.0`.
        * For `j != d`, cell `(d, s)` does not belong to row `j`, so `Y_mod[j, :] == Y_train[j, :]`
          and its squared norm in `Y_mod` is `||x_j||^2 = norm2_d9[j]`.
      - Thus `den_j = sqrt(||x_d||^2 - Y[d, s]^2) * sqrt(||x_j||^2)` for `j != d` (with `j == d`
        handled by the diagonal rule) matches `cosine_similarity(Y_mod)` with cell `(d, s) = 0`
        up to `float32` precision (`< 1e-6`).
    """
    bsz = drug_idx.shape[0]
    batch_arange = torch.arange(bsz, device=drug_idx.device)

    y_ds = y_train[drug_idx, se_idx]  # (B,)
    b_ds = b_train[drug_idx, se_idx]  # (B,)

    # 1. D9 (750-D frequency cosine similarity per drug when Y_train[d, s] = 0)
    y_j_at_s = y_train[:, se_idx].T  # (B, 750)
    num_d9 = g_d9[drug_idx] - y_ds.unsqueeze(1) * y_j_at_s
    n2_d_d9 = torch.clamp(norm2_d9[drug_idx] - y_ds * y_ds, min=0.0)  # (B,)
    n2_j_d9 = norm2_d9.unsqueeze(0).expand(bsz, -1)  # (B, 750) — rows j != d are unchanged when only (d,s) is zeroed
    den_d9 = torch.sqrt(n2_d_d9.unsqueeze(1) * n2_j_d9)
    d9_corr = torch.where(den_d9 > 0.0, num_d9 / torch.clamp(den_d9, min=1e-12), torch.zeros_like(num_d9))
    d9_corr[batch_arange, drug_idx] = (n2_d_d9 > 0.0).to(d9_corr.dtype)

    # 2. D10 (750-D binary cosine similarity per drug when B_train[d, s] = 0)
    b_j_at_s = b_train[:, se_idx].T  # (B, 750)
    num_d10 = g_d10[drug_idx] - b_ds.unsqueeze(1) * b_j_at_s
    n2_d_d10 = torch.clamp(norm2_d10[drug_idx] - b_ds * b_ds, min=0.0)  # (B,)
    n2_j_d10 = norm2_d10.unsqueeze(0).expand(bsz, -1)  # (B, 750)
    den_d10 = torch.sqrt(n2_d_d10.unsqueeze(1) * n2_j_d10)
    d10_corr = torch.where(den_d10 > 0.0, num_d10 / torch.clamp(den_d10, min=1e-12), torch.zeros_like(num_d10))
    d10_corr[batch_arange, drug_idx] = (n2_d_d10 > 0.0).to(d10_corr.dtype)

    # 3. S3 (994-D frequency cosine similarity per side effect when Y_train[d, s] = 0)
    y_k_at_d = y_train[drug_idx, :]  # (B, 994)
    num_s3 = g_s3[se_idx] - y_ds.unsqueeze(1) * y_k_at_d
    n2_s_s3 = torch.clamp(norm2_s3[se_idx] - y_ds * y_ds, min=0.0)  # (B,)
    n2_k_s3 = norm2_s3.unsqueeze(0).expand(bsz, -1)  # (B, 994) — cols k != s are unchanged when only (d,s) is zeroed
    den_s3 = torch.sqrt(n2_s_s3.unsqueeze(1) * n2_k_s3)
    s3_corr = torch.where(den_s3 > 0.0, num_s3 / torch.clamp(den_s3, min=1e-12), torch.zeros_like(num_s3))
    s3_corr[batch_arange, se_idx] = (n2_s_s3 > 0.0).to(s3_corr.dtype)

    # 4. S4 (994-D binary cosine similarity per side effect when B_train[d, s] = 0)
    b_k_at_d = b_train[drug_idx, :]  # (B, 994)
    num_s4 = g_s4[se_idx] - b_ds.unsqueeze(1) * b_k_at_d
    n2_s_s4 = torch.clamp(norm2_s4[se_idx] - b_ds * b_ds, min=0.0)  # (B,)
    n2_k_s4 = norm2_s4.unsqueeze(0).expand(bsz, -1)  # (B, 994)
    den_s4 = torch.sqrt(n2_s_s4.unsqueeze(1) * n2_k_s4)
    s4_corr = torch.where(den_s4 > 0.0, num_s4 / torch.clamp(den_s4, min=1e-12), torch.zeros_like(num_s4))
    s4_corr[batch_arange, se_idx] = (n2_s_s4 > 0.0).to(s4_corr.dtype)

    return d9_corr, d10_corr, s3_corr, s4_corr


@dataclass
class FoldViews:
    drug_side_train: np.ndarray            # (750, 994) built strictly from train (80%)
    drug_view_names: List[str]             # length V_d (11 or 9 when c1_impute=='drop')
    se_view_names: List[str]               # length V_s (4, 5, or 3)
    drug_views_list: List[np.ndarray]      # list of V_d arrays of shape (750, 750)
    se_views_list: List[np.ndarray]        # list of V_s arrays of shape (994, 994)
    drug_features_matrix: np.ndarray       # (750, V_d * 750) float32
    se_features_matrix: np.ndarray         # (994, V_s * 994) float32
    drug_avail: np.ndarray                 # (750, V_d) float32 pre-imputation availability indicator
    se_avail: np.ndarray                   # (994, V_s) float32 availability indicator
    profile_drug_view_indices: List[int]   # indices of D9, D10 inside drug_views_list ([] if dropped)
    fallback_d9: np.ndarray                # (750, 750) fallback row used when view_dropout masks D9
    fallback_d10: np.ndarray               # (750, 750) fallback row used when view_dropout masks D10
    profile_se_view_indices: Optional[List[int]] = None  # indices of S3, S4 inside se_views_list ([] if no_profile)


def build_fold_views(
    fold_split: FoldSplit,
    cfg: dict,
    device: Optional[torch.device] = None,
) -> FoldViews:
    """
    Build all drug and SE similarity views for a given fold under C1 and C2 configurations.

    Config keys used:
        c1_impute: 'zero' | 'drop' | 'mean' | 'regress' | 'diffusion' (default: 'zero')
        c1_crossfit_folds: int (default: 5)
        c1_diffusion_steps: int (default: 25)
        c1_diffusion_samples: int (default: 10)
        c1_imputer_epochs: int (default: 80 for regress)
        c1_diffusion_epochs: int (default: 100 for diffusion)
        c2_se_mode: 'original' | 'plus_ontology' | 'no_profile' (default: 'original')
        seed: int (default: 42)
    """
    if device is None:
        device = torch.device('cpu')

    c1_impute = cfg.get('c1_impute', 'zero')
    c2_se_mode = cfg.get('c2_se_mode', 'original')
    n_crossfit = int(cfg.get('c1_crossfit_folds', 5))
    seed = int(cfg.get('seed', 42)) + int(fold_split.fold)

    static = get_static_raw_views()
    prof = compute_profile_views_from_train(fold_split.train)

    d9_raw = prof['D9_profile_freq_cos']
    d10_raw = prof['D10_profile_bin_cos']
    d9_avail = compute_view_availability(d9_raw)
    d10_avail = compute_view_availability(d10_raw)

    # Static drug views for regression/diffusion conditioning
    static_drug_mats = [static[name] for name in STATIC_DRUG_VIEW_NAMES]
    static_drug_avail = np.column_stack([compute_view_availability(m) for m in static_drug_mats])
    static_drug_concat = np.hstack(static_drug_mats)

    train_drugs = fold_split.train_drugs
    missing_profile_mask = (d9_avail < 0.5)  # True for cold-start drugs (valsel, valcal, test in Split B)

    # Determine imputed/fallback D9 and D10 matrices
    if c1_impute == 'zero':
        d9_final = d9_raw.copy()
        d10_final = d10_raw.copy()
        fallback_d9 = np.zeros_like(d9_raw, dtype=np.float32)
        fallback_d10 = np.zeros_like(d10_raw, dtype=np.float32)
    elif c1_impute == 'drop':
        d9_final = d9_raw.copy()
        d10_final = d10_raw.copy()
        fallback_d9 = np.zeros_like(d9_raw, dtype=np.float32)
        fallback_d10 = np.zeros_like(d10_raw, dtype=np.float32)
    elif c1_impute == 'mean':
        mean_d9 = d9_raw[train_drugs].mean(axis=0)
        mean_d10 = d10_raw[train_drugs].mean(axis=0)
        fallback_d9 = np.tile(mean_d9, (N_DRUGS, 1)).astype(np.float32)
        fallback_d10 = np.tile(mean_d10, (N_DRUGS, 1)).astype(np.float32)
        np.fill_diagonal(fallback_d9, 1.0)
        np.fill_diagonal(fallback_d10, 1.0)

        d9_final = d9_raw.copy()
        d10_final = d10_raw.copy()
        d9_final[missing_profile_mask] = fallback_d9[missing_profile_mask]
        d10_final[missing_profile_mask] = fallback_d10[missing_profile_mask]
    elif c1_impute == 'regress':
        imp_epochs = int(cfg.get('c1_imputer_epochs', cfg.get('c1_impute_epochs', 80)))
        d9_pred, d10_pred = train_and_impute_regress(
            static_drug_concat, static_drug_avail, d9_raw, d10_raw,
            train_drugs=train_drugs, device=device, epochs=imp_epochs,
            seed=seed, n_splits=n_crossfit,
        )
        fallback_d9 = d9_pred.astype(np.float32)
        fallback_d10 = d10_pred.astype(np.float32)
        d9_final = d9_raw.copy()
        d10_final = d10_raw.copy()
        d9_final[missing_profile_mask] = d9_pred[missing_profile_mask]
        d10_final[missing_profile_mask] = d10_pred[missing_profile_mask]
    elif c1_impute == 'regress_ridge':
        d9_pred, d10_pred = train_and_impute_ridge(
            static_drug_concat, static_drug_avail, d9_raw, d10_raw,
            train_drugs=train_drugs, seed=seed, n_splits=n_crossfit,
        )
        fallback_d9 = d9_pred.astype(np.float32)
        fallback_d10 = d10_pred.astype(np.float32)
        d9_final = d9_raw.copy()
        d10_final = d10_raw.copy()
        d9_final[missing_profile_mask] = d9_pred[missing_profile_mask]
        d10_final[missing_profile_mask] = d10_pred[missing_profile_mask]
    elif c1_impute == 'regress_knn':
        d9_pred, d10_pred = train_and_impute_knn(
            static_drug_mats, static_drug_avail, d9_raw, d10_raw,
            train_drugs=train_drugs, seed=seed, n_splits=n_crossfit,
        )
        fallback_d9 = d9_pred.astype(np.float32)
        fallback_d10 = d10_pred.astype(np.float32)
        d9_final = d9_raw.copy()
        d10_final = d10_raw.copy()
        d9_final[missing_profile_mask] = d9_pred[missing_profile_mask]
        d10_final[missing_profile_mask] = d10_pred[missing_profile_mask]
    elif c1_impute == 'diffusion':
        imp_epochs = int(cfg.get('c1_diffusion_epochs', cfg.get('c1_impute_epochs', 100)))
        n_steps = int(cfg.get('c1_diffusion_steps', 25))
        n_samples = int(cfg.get('c1_diffusion_samples', 10))
        d9_diff, d10_diff = train_and_impute_diffusion(
            static_drug_concat, static_drug_avail, d9_raw, d10_raw,
            train_drugs=train_drugs, device=device,
            n_timesteps=n_steps, n_samples=n_samples, epochs=imp_epochs,
            seed=seed, n_splits=n_crossfit,
        )
        fallback_d9 = d9_diff.astype(np.float32)
        fallback_d10 = d10_diff.astype(np.float32)
        d9_final = d9_raw.copy()
        d10_final = d10_raw.copy()
        d9_final[missing_profile_mask] = d9_diff[missing_profile_mask]
        d10_final[missing_profile_mask] = d10_diff[missing_profile_mask]
    else:
        raise ValueError(f"Unknown c1_impute={c1_impute!r}")

    # Assemble drug views list & pre-imputation availability indicator
    if c1_impute == 'drop':
        drug_view_names = list(STATIC_DRUG_VIEW_NAMES)
        drug_views_list = [static[name] for name in drug_view_names]
        drug_avail = static_drug_avail.copy()
        profile_drug_indices = []
    else:
        drug_view_names = list(ALL_11_DRUG_VIEW_NAMES)
        all_raw_for_avail = [
            static['D1_stitch_one'],
            static['D2_stitch_two'],
            static['D3_stitch_three'],
            static['D4_stitch_four'],
            static['D5_stitch_five'],
            static['D6_mol2vec_cos'],
            static['D7_target_cos'],
            static['D8_fingerprint'],
            d9_raw,   # Pre-imputation availability: 0 for cold-start drugs
            d10_raw,  # Pre-imputation availability: 0 for cold-start drugs
            static['D11_pathway_enzyme'],
        ]
        drug_avail = np.column_stack([compute_view_availability(m) for m in all_raw_for_avail])
        drug_views_list = [
            static['D1_stitch_one'],
            static['D2_stitch_two'],
            static['D3_stitch_three'],
            static['D4_stitch_four'],
            static['D5_stitch_five'],
            static['D6_mol2vec_cos'],
            static['D7_target_cos'],
            static['D8_fingerprint'],
            d9_final,
            d10_final,
            static['D11_pathway_enzyme'],
        ]
        profile_drug_indices = [8, 9]

    # Assemble SE views under C2 configuration
    if c2_se_mode == 'original':
        se_view_names = ['S1_adrecs_semantic', 'S2_glove_cos', 'S3_profile_freq_cos', 'S4_profile_bin_cos']
        se_views_list = [
            static['S1_adrecs_semantic'],
            static['S2_glove_cos'],
            prof['S3_profile_freq_cos'],
            prof['S4_profile_bin_cos'],
        ]
        profile_se_indices = [2, 3]
    elif c2_se_mode == 'plus_ontology':
        se_view_names = [
            'S1_adrecs_semantic',
            'S2_glove_cos',
            'S3_profile_freq_cos',
            'S4_profile_bin_cos',
            'S5_meddra_ontology_cos',
        ]
        se_views_list = [
            static['S1_adrecs_semantic'],
            static['S2_glove_cos'],
            prof['S3_profile_freq_cos'],
            prof['S4_profile_bin_cos'],
            static['S5_meddra_ontology_cos'],
        ]
        profile_se_indices = [2, 3]
    elif c2_se_mode == 'no_profile':
        se_view_names = ['S1_adrecs_semantic', 'S2_glove_cos', 'S5_meddra_ontology_cos']
        se_views_list = [
            static['S1_adrecs_semantic'],
            static['S2_glove_cos'],
            static['S5_meddra_ontology_cos'],
        ]
        profile_se_indices = []
    else:
        raise ValueError(f"Unknown c2_se_mode={c2_se_mode!r}")

    se_avail = np.column_stack([compute_view_availability(m) for m in se_views_list])

    drug_features_matrix = np.hstack(drug_views_list).astype(np.float32)
    se_features_matrix = np.hstack(se_views_list).astype(np.float32)

    return FoldViews(
        drug_side_train=prof['drug_side_train'],
        drug_view_names=drug_view_names,
        se_view_names=se_view_names,
        drug_views_list=drug_views_list,
        se_views_list=se_views_list,
        drug_features_matrix=drug_features_matrix,
        se_features_matrix=se_features_matrix,
        drug_avail=drug_avail.astype(np.float32),
        se_avail=se_avail.astype(np.float32),
        profile_drug_view_indices=profile_drug_indices,
        fallback_d9=fallback_d9,
        fallback_d10=fallback_d10,
        profile_se_view_indices=profile_se_indices,
    )
