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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train ProfileViewRegressor on train_drugs and return predicted (D9_pred, D10_pred) for all 750 drugs.
    For train_drugs (used when view_dropout triggers), uses out-of-fold / regularized predictions.
    """
    torch.manual_seed(seed)
    x_all = np.hstack([static_concat, static_avail]).astype(np.float32)
    y_all = np.hstack([d9_raw, d10_raw]).astype(np.float32)

    x_t = torch.from_numpy(x_all).to(device)
    y_t = torch.from_numpy(y_all).to(device)

    tr_idx = np.asarray(train_drugs, dtype=int)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(tr_idx))
    n_val = max(10, int(0.1 * len(tr_idx)))
    val_sub = tr_idx[perm[:n_val]]
    fit_sub = tr_idx[perm[n_val:]]

    model = ProfileViewRegressor(in_dim=x_all.shape[1], out_dim=y_all.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float('inf')
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    fit_x, fit_y = x_t[fit_sub], y_t[fit_sub]
    val_x, val_y = x_t[val_sub], y_t[val_sub]

    batch_size = 64
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
        pred_all = model(x_t).cpu().numpy().astype(np.float64)

    d9_pred = pred_all[:, :N_DRUGS].copy()
    d10_pred = pred_all[:, N_DRUGS:].copy()
    np.fill_diagonal(d9_pred, 1.0)
    np.fill_diagonal(d10_pred, 1.0)
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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train a conditional DDPM (x_0-prediction formulation) on train_drugs and generate B=n_samples
    imputations per drug, averaging them to obtain (D9_diff, D10_diff).
    """
    torch.manual_seed(seed)
    cond_all = np.hstack([static_concat, static_avail]).astype(np.float32)
    x0_all = np.hstack([d9_raw, d10_raw]).astype(np.float32)

    cond_t = torch.from_numpy(cond_all).to(device)
    x0_t = torch.from_numpy(x0_all).to(device)
    tr_idx = torch.as_tensor(train_drugs, dtype=torch.long, device=device)

    betas = torch.linspace(1e-4, 0.2, n_timesteps, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    model = ConditionalDenoiserMLP(cond_in_dim=cond_all.shape[1], target_dim=x0_all.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    fit_cond = cond_t[tr_idx]
    fit_x0 = x0_t[tr_idx]
    batch_size = 64

    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(tr_idx), device=device)
        for start in range(0, len(tr_idx), batch_size):
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

    # Reverse sampling: average over B=n_samples trajectories for all 750 drugs
    model.eval()
    n_all = cond_t.shape[0]
    target_dim = x0_all.shape[1]
    acc_samples = torch.zeros((n_all, target_dim), dtype=torch.float32, device=device)

    with torch.no_grad():
        for s_idx in range(n_samples):
            torch.manual_seed(seed + 1000 + s_idx)
            x_cur = torch.randn((n_all, target_dim), device=device)
            for step in reversed(range(n_timesteps)):
                t_vec = torch.full((n_all,), step, dtype=torch.long, device=device)
                x0_hat = model(x_cur, t_vec, cond_t)
                if step > 0:
                    ab_t = alpha_bar[step]
                    ab_prev = alpha_bar[step - 1]
                    a_t = alphas[step]
                    b_t = betas[step]
                    eps_hat = (x_cur - torch.sqrt(ab_t) * x0_hat) / torch.sqrt(1.0 - ab_t + 1e-8)
                    mean = (1.0 / torch.sqrt(a_t)) * (x_cur - (b_t / torch.sqrt(1.0 - ab_t + 1e-8)) * eps_hat)
                    var = b_t * (1.0 - ab_prev) / (1.0 - ab_t + 1e-8)
                    x_cur = mean + torch.sqrt(torch.clamp(var, min=1e-8)) * torch.randn_like(x_cur)
                else:
                    x_cur = x0_hat
            acc_samples += torch.clamp(x_cur, 0.0, 1.0)

    mean_pred = (acc_samples / float(n_samples)).cpu().numpy().astype(np.float64)
    d9_diff = mean_pred[:, :N_DRUGS].copy()
    d10_diff = mean_pred[:, N_DRUGS:].copy()
    np.fill_diagonal(d9_diff, 1.0)
    np.fill_diagonal(d10_diff, 1.0)
    return d9_diff, d10_diff


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


def build_fold_views(
    fold_split: FoldSplit,
    cfg: dict,
    device: Optional[torch.device] = None,
) -> FoldViews:
    """
    Build all drug and SE similarity views for a given fold under C1 and C2 configurations.

    Config keys used:
        c1_impute: 'zero' | 'drop' | 'mean' | 'regress' | 'diffusion' (default: 'zero')
        c1_diffusion_steps: int (default: 25)
        c1_diffusion_samples: int (default: 10)
        c1_impute_epochs: int (default: 80)
        c2_se_mode: 'original' | 'plus_ontology' | 'no_profile' (default: 'original')
        seed: int (default: 42)
    """
    if device is None:
        device = torch.device('cpu')

    c1_impute = cfg.get('c1_impute', 'zero')
    c2_se_mode = cfg.get('c2_se_mode', 'original')
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
        imp_epochs = int(cfg.get('c1_impute_epochs', 80))
        d9_pred, d10_pred = train_and_impute_regress(
            static_drug_concat, static_drug_avail, d9_raw, d10_raw,
            train_drugs=train_drugs, device=device, epochs=imp_epochs, seed=seed
        )
        fallback_d9 = d9_pred.astype(np.float32)
        fallback_d10 = d10_pred.astype(np.float32)
        d9_final = d9_raw.copy()
        d10_final = d10_raw.copy()
        d9_final[missing_profile_mask] = d9_pred[missing_profile_mask]
        d10_final[missing_profile_mask] = d10_pred[missing_profile_mask]
    elif c1_impute == 'diffusion':
        imp_epochs = int(cfg.get('c1_impute_epochs', 100))
        n_steps = int(cfg.get('c1_diffusion_steps', 25))
        n_samples = int(cfg.get('c1_diffusion_samples', 10))
        d9_diff, d10_diff = train_and_impute_diffusion(
            static_drug_concat, static_drug_avail, d9_raw, d10_raw,
            train_drugs=train_drugs, device=device,
            n_timesteps=n_steps, n_samples=n_samples, epochs=imp_epochs, seed=seed
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
    elif c2_se_mode == 'no_profile':
        se_view_names = ['S1_adrecs_semantic', 'S2_glove_cos', 'S5_meddra_ontology_cos']
        se_views_list = [
            static['S1_adrecs_semantic'],
            static['S2_glove_cos'],
            static['S5_meddra_ontology_cos'],
        ]
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
    )
