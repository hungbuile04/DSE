"""
Leak-free Protocol v2 wrapper for SDPred (`SDPred/up_ten_fold.py` & `SDPred/network.py`).

Fixes applied without modifying `SDPred/`:
  1. Uses unified 4-way split `(train, valsel, valcal, test)` via `load_fold_data`.
  2. Computes all 4 interaction-profile cosine similarities (`drug_side_sim`, `drug_side_label_sim`,
     `side_drug_sim`, `side_drug_label_sim`) strictly from `make_train_only_freq(fold_split.train)`
     (eliminating Split A validation/test leakage and zeroing all cold-start rows in Split B).
  3. Uses leak-free 1:1 negative sampling via `sample_negative_pairs_v2` (restricting Split B
     training negatives strictly to `fold_split.train_drugs`).
  4. Enforces identical hyperparameters across Split A and Split B (`epochs=100, batch_size=512,
     lr=0.0001, weight_decay=1e-5, embed_dim=128, dropout=0.5`), removes `retain_graph=True`
     and redundant per-epoch train evaluation, and keeps feature tables on device (~38 MB).
  5. Selects primary checkpoint on `valsel` maximizing 5-class `MacroF1` (and secondary
     checkpoint minimizing `RMSE` saved to `SDPred_rmseSel_{A|B}`).
"""

import os
import sys
import copy
import time
import pickle
import argparse
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity

from MSSF_plus.protocol import (
    REPO_DIR,
    SHARED_DIR,
    DEFAULT_RESULTS_V2_DIR,
    N_DRUGS,
    N_SIDES,
    set_global_seed,
    get_device,
    get_fold_list,
    load_fold_data,
)
from MSSF_plus.baselines_rerun.common import (
    make_train_only_freq,
    sample_negative_pairs_v2,
    eval_regression_predictions,
    try_restore_baseline_fold,
    save_baseline_fold_outputs,
    finalize_baseline_experiment,
)

SDPRED_DIR = os.path.join(REPO_DIR, 'SDPred')
if SDPRED_DIR not in sys.path:
    sys.path.insert(0, SDPRED_DIR)

from network import ConvNCF  # noqa: E402


def _load_sdpred_static_views() -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load the 8 static drug similarity matrices and 2 static SE similarity matrices once."""
    sd_data = os.path.join(SHARED_DIR, 'sdpred_750')

    def _pkl(fname: str) -> np.ndarray:
        with open(os.path.join(sd_data, fname), 'rb') as f:
            return np.asarray(pickle.load(f), dtype=np.float32)

    drug_static = [
        _pkl('Text_similarity_one.pkl'),
        _pkl('Text_similarity_two.pkl'),
        _pkl('Text_similarity_three.pkl'),
        _pkl('Text_similarity_four.pkl'),
        _pkl('Text_similarity_five.pkl'),
        cosine_similarity(_pkl('drug_mol.pkl')).astype(np.float32),
        cosine_similarity(_pkl('drug_target.pkl')).astype(np.float32),
        _pkl('fingerprint_similarity.pkl'),
    ]
    se_static = [
        _pkl('side_effect_semantic.pkl'),
        cosine_similarity(_pkl('glove_wordEmbedding.pkl')).astype(np.float32),
    ]
    return drug_static, se_static


def _build_sdpred_fold_matrices(
    train_pairs: np.ndarray,
    drug_static: List[np.ndarray],
    se_static: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build `(750, 7500)` drug matrix and `(994, 3976)` SE matrix strictly from `train_pairs`."""
    drug_side_train = make_train_only_freq(train_pairs).astype(np.float32)
    drug_side_label = (drug_side_train > 0).astype(np.float32)

    d_sim = cosine_similarity(drug_side_train).astype(np.float32)
    d_lbl_sim = cosine_similarity(drug_side_label).astype(np.float32)
    s_sim = cosine_similarity(drug_side_train.T).astype(np.float32)
    s_lbl_sim = cosine_similarity(drug_side_label.T).astype(np.float32)

    drug_mat = np.hstack(drug_static + [d_sim, d_lbl_sim]).astype(np.float32)
    side_mat = np.hstack(se_static + [s_sim, s_lbl_sim]).astype(np.float32)
    assert drug_mat.shape == (N_DRUGS, 7500), f"Unexpected drug_mat shape {drug_mat.shape}"
    assert side_mat.shape == (N_SIDES, 3976), f"Unexpected side_mat shape {side_mat.shape}"
    return drug_mat, side_mat, drug_side_train


class _FastIndexLoader:
    """Vectorized tensor batch loader avoiding 59,666 per-sample Python __getitem__ calls per epoch."""

    def __init__(
        self,
        pairs_with_rating: np.ndarray,
        batch_size: int,
        shuffle: bool,
        device: Optional[torch.device] = None,
    ):
        self.n = int(len(pairs_with_rating))
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(shuffle and (self.n > self.batch_size) and (self.n % self.batch_size == 1))
        self.device = device
        d_np = pairs_with_rating[:, 0].astype(np.int64)
        s_np = pairs_with_rating[:, 1].astype(np.int64)
        r_np = pairs_with_rating[:, 2].astype(np.float32)
        self.r_cpu = r_np
        if device is not None and device.type != 'cpu':
            self.d = torch.tensor(d_np, dtype=torch.long, device=device)
            self.s = torch.tensor(s_np, dtype=torch.long, device=device)
            self.r = torch.tensor(r_np, dtype=torch.float32, device=device)
        else:
            self.d = torch.tensor(d_np, dtype=torch.long)
            self.s = torch.tensor(s_np, dtype=torch.long)
            self.r = torch.tensor(r_np, dtype=torch.float32)

    def __len__(self) -> int:
        if self.drop_last:
            return self.n // self.batch_size
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            perm_cpu = torch.randperm(self.n)
            perm = perm_cpu.to(self.d.device) if self.d.device.type != 'cpu' else perm_cpu
            d_ep = self.d[perm]
            s_ep = self.s[perm]
            r_ep = self.r[perm]
        else:
            d_ep = self.d
            s_ep = self.s
            r_ep = self.r

        limit = (self.n // self.batch_size) * self.batch_size if self.drop_last else self.n
        for start in range(0, limit, self.batch_size):
            end = min(start + self.batch_size, limit)
            yield d_ep[start:end], s_ep[start:end], r_ep[start:end]


def _make_index_loader(
    pairs_with_rating: np.ndarray,
    batch_size: int,
    shuffle: bool,
    device: Optional[torch.device] = None,
) -> _FastIndexLoader:
    return _FastIndexLoader(pairs_with_rating, batch_size=batch_size, shuffle=shuffle, device=device)


def _attach_fast_convncf_forward(model: ConvNCF) -> None:
    """
    Replace the 40-iteration `torch.cat` loop + `cnn_interaction[0]` (`Conv2d(40, 32, k=2, s=2)`)
    in `ConvNCF.forward` with its exact factored `einsum` equivalent (`< 1e-6` numerical difference),
    eliminating 39 intermediate `(B, k, 128, 128)` tensor allocations per batch without modifying `SDPred/network.py`.
    """
    import types
    import torch.nn.functional as F

    def _fast_forward(self, drug_features, side_features, device):
        x_drugs = F.relu(self.drugs_bn(self.drugs_layer(drug_features)), inplace=True)
        x_drugs = F.dropout(x_drugs, training=self.training, p=self.dropout1)
        x_drugs = self.drugs_layer_1(x_drugs)

        x_sides = F.relu(self.sides_bn(self.sides_layer(side_features)), inplace=True)
        x_sides = F.dropout(x_sides, training=self.training, p=self.dropout1)
        x_sides = self.sides_layer_1(x_sides)

        drug1, drug2, drug3, drug4, drug5, drug6, drug7, drug8, drug9, drug10 = drug_features.chunk(10, 1)
        side1, side2, side3, side4 = side_features.chunk(4, 1)

        drugs = []
        for idx_d, d_in in enumerate([drug1, drug2, drug3, drug4, drug5, drug6, drug7, drug8, drug9, drug10], start=1):
            l1 = getattr(self, f'drug_layer{idx_d}')
            bn = getattr(self, f'drug{idx_d}_bn')
            l2 = getattr(self, f'drug_layer{idx_d}_1')
            xd = F.relu(bn(l1(d_in)), inplace=True)
            xd = F.dropout(xd, training=self.training, p=self.dropout1)
            drugs.append(l2(xd))

        sides = []
        for idx_s, s_in in enumerate([side1, side2, side3, side4], start=1):
            l1 = getattr(self, f'side_layer{idx_s}')
            bn = getattr(self, f'side{idx_s}_bn')
            l2 = getattr(self, f'side_layer{idx_s}_1')
            xs = F.relu(bn(l1(s_in)), inplace=True)
            xs = F.dropout(xs, training=self.training, p=self.dropout1)
            sides.append(l2(xs))

        bsz = x_drugs.shape[0]
        half_dim = self.embed_dim // 2
        d_stack = torch.stack(drugs, dim=1).view(bsz, 10, half_dim, 2)
        s_stack = torch.stack(sides, dim=1).view(bsz, 4, half_dim, 2)

        conv0 = self.cnn_interaction[0]
        w = conv0.weight.view(self.channel_size, 10, 4, 2, 2)
        wd = torch.einsum('biux,oijxy->bojuy', d_stack, w)
        conv0_out = torch.einsum('bojuy,bjvy->bouv', wd, s_stack)
        if conv0.bias is not None:
            conv0_out = conv0_out + conv0.bias.view(1, -1, 1, 1)

        feature_map = self.cnn_interaction[1:](conv0_out)
        h = feature_map.reshape(-1, 32 * 4)
        total = torch.cat((x_drugs, h, x_sides), dim=1)
        total = F.relu(self.total_layer(total), inplace=True)
        total = F.dropout(total, training=self.training, p=self.dropout2)
        classification = self.classifier(total)
        regression = self.con_layer(total)
        return classification.squeeze(), regression.squeeze()

    model.forward = types.MethodType(_fast_forward, model)


@torch.no_grad()
def _predict_sdpred(
    model: ConvNCF,
    loader: _FastIndexLoader,
    drug_mat_t: torch.Tensor,
    side_mat_t: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `(y_true_ratings, pred_binary_scores, pred_freq_scores)`."""
    model.eval()
    cls_t, reg_t = [], []
    for batch_d, batch_s, _ in loader:
        b_d = batch_d.to(device) if batch_d.device != device else batch_d
        b_s = batch_s.to(device) if batch_s.device != device else batch_s
        d_feat = drug_mat_t[b_d]
        s_feat = side_mat_t[b_s]
        scores_cls, scores_reg = model(d_feat, s_feat, device)
        cls_t.append(scores_cls.view(-1))
        reg_t.append(scores_reg.view(-1))
    cls_np = torch.cat(cls_t, dim=0).detach().cpu().numpy().astype(np.float32)
    reg_np = torch.cat(reg_t, dim=0).detach().cpu().numpy().astype(np.float32)
    return loader.r_cpu.copy(), cls_np, reg_np


def run_sdpred_fold(
    fold: int,
    split_type: str,
    cfg: dict,
    static_views: Tuple[List[np.ndarray], List[np.ndarray]],
    device: torch.device,
    results_dir: str = DEFAULT_RESULTS_V2_DIR,
    drive_dir: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    t0 = time.time()
    fold_seed = int(cfg['seed']) + int(fold)
    set_global_seed(fold_seed)
    fold_split = load_fold_data(fold, split_type)

    drug_mat, side_mat, drug_side_train = _build_sdpred_fold_matrices(
        fold_split.train, static_views[0], static_views[1]
    )
    drug_mat_t = torch.tensor(drug_mat, dtype=torch.float32, device=device)
    side_mat_t = torch.tensor(side_mat, dtype=torch.float32, device=device)

    use_self_corr = bool(cfg.get('profile_self_correction', False))
    eval_test = bool(cfg.get('eval_test', False))
    method_name = 'SDPred_selfcorr' if use_self_corr else 'SDPred'

    if use_self_corr:
        from MSSF_plus.views import compute_batch_self_corrected_profiles
        y_tr_t = torch.tensor(drug_side_train, dtype=torch.float32, device=device)
        b_tr_t = (y_tr_t > 0.0).to(torch.float32)
        g_d9_t = y_tr_t @ y_tr_t.t()
        n2_d9_t = (y_tr_t * y_tr_t).sum(dim=1)
        g_d10_t = b_tr_t @ b_tr_t.t()
        n2_d10_t = (b_tr_t * b_tr_t).sum(dim=1)
        g_s3_t = y_tr_t.t() @ y_tr_t
        n2_s3_t = (y_tr_t * y_tr_t).sum(dim=0)
        g_s4_t = b_tr_t.t() @ b_tr_t
        n2_s4_t = (b_tr_t * b_tr_t).sum(dim=0)

    # 1:1 leak-free negative sampling for training (matching SDPred multi-task loss)
    neg_train_xy = sample_negative_pairs_v2(
        drug_side_train, fold_split, n_samples=len(fold_split.train), part='train', seed=fold_seed
    )
    neg_train = np.column_stack([neg_train_xy, np.zeros(len(neg_train_xy), dtype=np.float32)])
    train_combined = np.vstack([fold_split.train.astype(np.float32), neg_train])

    # Optional leak-free test negatives solely for logging binary AUC/AUPR in summary.json
    neg_test_xy = sample_negative_pairs_v2(
        drug_side_train, fold_split, n_samples=len(fold_split.test), part='test', seed=fold_seed + 100
    )
    neg_test = np.column_stack([neg_test_xy, np.zeros(len(neg_test_xy), dtype=np.float32)])

    batch_size = int(cfg['batch_size'])
    eval_bs = int(cfg['eval_batch_size'])
    epochs = int(cfg['epochs'])

    train_loader = _make_index_loader(train_combined, batch_size=batch_size, shuffle=True, device=device)
    valsel_loader = _make_index_loader(fold_split.valsel, batch_size=eval_bs, shuffle=False, device=device)
    test_pos_loader = _make_index_loader(fold_split.test, batch_size=eval_bs, shuffle=False, device=device)
    test_neg_loader = _make_index_loader(neg_test, batch_size=eval_bs, shuffle=False, device=device)

    drop = float(cfg['dropout'])
    model = ConvNCF(
        7500, 3976, int(cfg['embed_dim']), batch_size, dropout1=drop, dropout2=drop
    ).to(device)
    _attach_fast_convncf_forward(model)

    cls_criterion = nn.MSELoss()
    reg_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(cfg['lr']), weight_decay=float(cfg['weight_decay'])
    )

    best_f1 = -float('inf')
    best_f1_epoch = 1
    best_train_macro_f1 = 0.0
    best_f1_state = copy.deepcopy(model.state_dict())

    best_rmse = float('inf')
    best_rmse_epoch = 1
    best_rmse_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss_t = torch.zeros((), dtype=torch.float32, device=device)
        n_batches = 0
        pos_tr_true_t, pos_tr_pred_t = [], []

        for b_d_dev, b_s_dev, batch_r_dev in train_loader:
            batch_lbl_dev = (batch_r_dev > 0).float()
            d_feat = drug_mat_t[b_d_dev]
            s_feat = side_mat_t[b_s_dev]

            if use_self_corr:
                d_feat = d_feat.clone()
                s_feat = s_feat.clone()
                d9_c, d10_c, s3_c, s4_c = compute_batch_self_corrected_profiles(
                    drug_idx=b_d_dev,
                    se_idx=b_s_dev,
                    y_train=y_tr_t,
                    b_train=b_tr_t,
                    g_d9=g_d9_t,
                    norm2_d9=n2_d9_t,
                    g_d10=g_d10_t,
                    norm2_d10=n2_d10_t,
                    g_s3=g_s3_t,
                    norm2_s3=n2_s3_t,
                    g_s4=g_s4_t,
                    norm2_s4=n2_s4_t,
                )
                d_feat[:, 8 * N_DRUGS:9 * N_DRUGS] = d9_c
                d_feat[:, 9 * N_DRUGS:10 * N_DRUGS] = d10_c
                s_feat[:, 2 * N_SIDES:3 * N_SIDES] = s3_c
                s_feat[:, 3 * N_SIDES:4 * N_SIDES] = s4_c

            optimizer.zero_grad(set_to_none=True)
            logits, pred_reg = model(d_feat, s_feat, device)
            logits = logits.view(-1)
            pred_reg = pred_reg.view(-1)

            loss1 = cls_criterion(logits, batch_lbl_dev)
            pos_mask = batch_r_dev > 0
            if pos_mask.any():
                loss2 = reg_criterion(pred_reg[pos_mask], batch_r_dev[pos_mask])
                total_loss = loss1 * loss2
                pos_tr_true_t.append(batch_r_dev[pos_mask].detach())
                pos_tr_pred_t.append(pred_reg[pos_mask].detach())
            else:
                total_loss = loss1

            total_loss.backward()
            optimizer.step()

            running_loss_t += total_loss.detach()
            n_batches += 1

        running_loss = float(running_loss_t.item())
        tr_ep_metrics = eval_regression_predictions(
            torch.cat(pos_tr_true_t, dim=0).cpu().numpy(),
            torch.cat(pos_tr_pred_t, dim=0).cpu().numpy(),
        )
        tr_ep_f1 = float(tr_ep_metrics['MacroF1'])

        val_y, _, val_p = _predict_sdpred(model, valsel_loader, drug_mat_t, side_mat_t, device)
        val_metrics = eval_regression_predictions(val_y, val_p)

        if val_metrics['MacroF1'] > best_f1:
            best_f1 = val_metrics['MacroF1']
            best_f1_epoch = epoch
            best_train_macro_f1 = tr_ep_f1
            best_f1_state = copy.deepcopy(model.state_dict())

        if val_metrics['RMSE'] < best_rmse:
            best_rmse = val_metrics['RMSE']
            best_rmse_epoch = epoch
            best_rmse_state = copy.deepcopy(model.state_dict())

        if verbose and (epoch == 1 or epoch % 20 == 0 or epoch == epochs):
            print(
                f"  [{method_name} Split {split_type} Fold {fold}] Ep {epoch:03d}/{epochs:03d} | "
                f"Loss={running_loss / max(1, n_batches):.4f} | "
                f"train MacroF1={tr_ep_f1:.4f} | "
                f"valsel MacroF1={val_metrics['MacroF1']:.4f} (best={best_f1:.4f}@{best_f1_epoch}) | "
                f"RMSE={val_metrics['RMSE']:.4f} (best={best_rmse:.4f}@{best_rmse_epoch})"
            )

    # 1. Primary evaluation (MacroF1-selected checkpoint)
    model.load_state_dict(best_f1_state)
    val_y, _, val_p_f1 = _predict_sdpred(model, valsel_loader, drug_mat_t, side_mat_t, device)
    assert np.array_equal(np.round(val_y), np.round(fold_split.valsel[:, 2]))
    valsel_final = eval_regression_predictions(val_y, val_p_f1)

    if best_rmse_epoch == best_f1_epoch:
        val_p_rmse = val_p_f1.copy()
        valsel_metrics_rmse = dict(valsel_final)
    else:
        model.load_state_dict(best_rmse_state)
        _, _, val_p_rmse = _predict_sdpred(model, valsel_loader, drug_mat_t, side_mat_t, device)
        valsel_metrics_rmse = eval_regression_predictions(val_y, val_p_rmse)

    if eval_test:
        model.load_state_dict(best_f1_state)
        test_y, test_cls_pos, test_p_f1 = _predict_sdpred(
            model, test_pos_loader, drug_mat_t, side_mat_t, device
        )
        _, test_cls_neg, _ = _predict_sdpred(model, test_neg_loader, drug_mat_t, side_mat_t, device)
        assert np.array_equal(np.round(test_y), np.round(fold_split.test[:, 2]))

        bin_y = np.concatenate([np.ones(len(test_cls_pos)), np.zeros(len(test_cls_neg))])
        bin_s = np.concatenate([test_cls_pos, test_cls_neg])
        bin_auc = float(roc_auc_score(bin_y, bin_s))
        bin_aupr = float(average_precision_score(bin_y, bin_s))

        test_metrics_f1 = eval_regression_predictions(test_y, test_p_f1)
        test_metrics_f1['BinaryAUC'] = bin_auc
        test_metrics_f1['BinaryAUPR'] = bin_aupr

        if best_rmse_epoch == best_f1_epoch:
            test_p_rmse = test_p_f1.copy()
            test_metrics_rmse = dict(test_metrics_f1)
        else:
            model.load_state_dict(best_rmse_state)
            _, _, test_p_rmse = _predict_sdpred(model, test_pos_loader, drug_mat_t, side_mat_t, device)
            test_metrics_rmse = eval_regression_predictions(test_y, test_p_rmse)
    else:
        test_p_f1 = None
        test_p_rmse = None
        test_metrics_f1 = None
        test_metrics_rmse = None

    elapsed = time.time() - t0
    train_valsel_gap = float(best_train_macro_f1 - float(valsel_final['MacroF1']))
    fold_summary = {
        'method_name': method_name,
        'split_type': split_type,
        'fold': int(fold),
        'profile_self_correction': bool(use_self_corr),
        'best_epoch': int(best_f1_epoch),
        'best_valsel_MacroF1': float(best_f1),
        'best_train_macro_f1': float(best_train_macro_f1),
        'train_valsel_macro_f1_gap': float(train_valsel_gap),
        'best_rmse_epoch': int(best_rmse_epoch),
        'best_valsel_RMSE': float(best_rmse),
        'valsel_metrics': valsel_final,
        'rmse_sel_valsel_metrics': valsel_metrics_rmse,
        'test_metrics': test_metrics_f1,
        'rmse_sel_test_metrics': test_metrics_rmse,
        'elapsed_seconds': float(elapsed),
    }

    save_baseline_fold_outputs(
        method_name=method_name,
        split_type=split_type,
        fold=fold,
        test_split=fold_split.test,
        test_preds_float=test_p_f1,
        fold_summary=fold_summary,
        results_dir=results_dir,
        drive_dir=drive_dir,
        rmse_sel_preds_float=test_p_rmse,
        valsel_split=fold_split.valsel,
        valsel_preds_float=val_p_f1,
        rmse_sel_valsel_preds_float=val_p_rmse,
        eval_test=eval_test,
    )

    if verbose:
        print(
            f"  --> {method_name} Fold {fold} Done ({elapsed:.1f}s) | Best Ep={best_f1_epoch} | "
            f"valsel MacroF1={valsel_final['MacroF1']:.4f}, QWK={valsel_final['QWK']:.4f}, "
            f"MAE={valsel_final['MAE']:.4f}, RMSE={valsel_final['RMSE']:.4f} | "
            f"Gap(tr-val)={train_valsel_gap:+.4f}"
        )
    return fold_summary


def run_sdpred_experiment(
    split_type: str = 'B',
    mode: str = 'full',
    custom_folds: Optional[List[int]] = None,
    results_dir: str = DEFAULT_RESULTS_V2_DIR,
    drive_dir: Optional[str] = None,
    resume: bool = False,
    overrides: Optional[dict] = None,
    verbose: bool = True,
) -> dict:
    cfg = {
        'seed': 42,
        'epochs': 100,
        'batch_size': 512,
        'eval_batch_size': 256,
        'lr': 0.0001,
        'weight_decay': 1e-5,
        'embed_dim': 128,
        'dropout': 0.5,
        'selection_metric': 'MacroF1',
        'profile_self_correction': False,
        'eval_test': False,
        'device': None,
    }
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                cfg[k] = v

    use_self_corr = bool(cfg.get('profile_self_correction', False))
    eval_test = bool(cfg.get('eval_test', False))
    method_name = 'SDPred_selfcorr' if use_self_corr else 'SDPred'

    folds = get_fold_list(mode=mode, custom_folds=custom_folds)
    device = get_device(cfg.get('device'))
    static_views = _load_sdpred_static_views()

    if verbose:
        print(
            f"=== Running {method_name} | Split {split_type} | Mode={mode} (folds={folds}) | "
            f"Device={device} | Epochs={cfg['epochs']} | SelfCorr={use_self_corr} | Resume={resume} ==="
        )

    fold_summaries = []
    for fold in folds:
        if resume:
            cached = try_restore_baseline_fold(
                method_name, split_type, fold, results_dir,
                drive_dir=drive_dir, has_rmse_sel=True, eval_test=eval_test,
            )
            if cached is not None:
                if verbose:
                    vm = cached['valsel_metrics']
                    print(
                        f"  [RESUME] Skipping {method_name}_{split_type} Fold {fold} "
                        f"(valsel MacroF1={vm['MacroF1']:.4f}, RMSE={vm['RMSE']:.4f})"
                    )
                fold_summaries.append(cached)
                continue

        s = run_sdpred_fold(
            fold=fold,
            split_type=split_type,
            cfg=cfg,
            static_views=static_views,
            device=device,
            results_dir=results_dir,
            drive_dir=drive_dir,
            verbose=verbose,
        )
        fold_summaries.append(s)

    return finalize_baseline_experiment(
        method_name=method_name,
        split_type=split_type,
        mode=mode,
        folds=folds,
        cfg=cfg,
        fold_summaries=fold_summaries,
        results_dir=results_dir,
        drive_dir=drive_dir,
        has_rmse_sel=True,
        verbose=verbose,
    )


def main():
    parser = argparse.ArgumentParser(description='Run leak-free SDPred under Protocol v2.')
    parser.add_argument('--split', type=str, default='B', choices=['A', 'B'])
    parser.add_argument('--mode', type=str, default='full', choices=['quick', 'dev', 'full'])
    parser.add_argument('--folds', type=int, nargs='*', default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--profile_self_correction', type=int, default=None, choices=[0, 1])
    parser.add_argument('--eval_test', action='store_true')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_V2_DIR)
    parser.add_argument('--drive_dir', type=str, default=None)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    run_sdpred_experiment(
        split_type=args.split,
        mode=args.mode,
        custom_folds=args.folds,
        results_dir=args.results_dir,
        drive_dir=args.drive_dir,
        resume=args.resume,
        overrides={
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'profile_self_correction': (
                bool(args.profile_self_correction) if args.profile_self_correction is not None else None
            ),
            'eval_test': bool(args.eval_test),
            'device': args.device,
        },
        verbose=True,
    )


if __name__ == '__main__':
    main()
