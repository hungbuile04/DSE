"""
Leak-free Protocol v2 wrapper for MGPred (`MGPred/Ten_Fold_test.py`).

Fixes applied without modifying `MGPred/`:
  1. Uses unified 4-way split `(train, valsel, valcal, test)` via `load_fold_data`.
  2. Fixes Split B static feature bug & NumPy >= 1.24 inhomogeneous shape crash:
     populates static STITCH (`750x750`) & Mol2vec (`750x100`) features for ALL 750 drugs
     (and SE semantic `994x994` & GloVe `994x300` for ALL 994 SEs), while building interaction
     profile `ufea1`/`ifea1` and neighbor dicts `u_adj`/`i_adj` strictly from `fold_split.train`.
     During `model.eval()`, cold-start drugs absent from `u_adj` naturally use MGPred's built-in
     `aggregator.py` L53-62 self-feature branch.
  3. Fixes `MGPred/Ten_Fold_test.py` L60 & L114 EMA bias-correction bug: `load_ema_params()`
     divides `avg_param` by `(1 - beta_ema ** steps_ema)` (Louizos et al. 2018), which requires
     zero-initialized `avg_param` rather than `deepcopy(p.data)` (eliminating `~1e13` early-epoch blowup).
  4. Selects primary checkpoint on `valsel` maximizing 5-class `MacroF1` (and secondary
     checkpoint minimizing `RMSE` saved to `MGPred_rmseSel_{A|B}`).
  5. Evaluates `test` once with `shuffle=False`, asserting exact alignment with `fold_split.test`.
"""

import os
import sys
import copy
import time
import pickle
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from MSSF_plus.protocol import (
    REPO_DIR,
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
    eval_regression_predictions,
    try_restore_baseline_fold,
    save_baseline_fold_outputs,
    finalize_baseline_experiment,
)

MGPRED_DIR = os.path.join(REPO_DIR, 'MGPred')
if MGPRED_DIR not in sys.path:
    sys.path.insert(0, MGPRED_DIR)

from Ten_Fold_test import MGPred  # noqa: E402
from utils.aggregator import aggregator, a_res  # noqa: E402
from utils.encoder import encoder  # noqa: E402


class _SelfCorrectingAggregator(aggregator):
    """
    Extension of `MGPred`'s `aggregator` supporting `profile_self_correction`:
      - When `profile_self_correction=True` and a batch sample `(d = nodes_u[b], s = nodes_i[b])`
        belongs to `Y_train` (`y_train_mat[d, s] > 0`):
        1. Target edge `(d, s)` is removed from `u_adj[d]` (for drug aggregators) and `(s, d)`
           is removed from `i_adj[s]` (for SE aggregators).
        2. If `is_interaction_view=True` (`u_agg1` / `i_agg1`, where initial features are
           rows of `Y_train` and `Y_train.T`), position `s` in drug `d`'s initial feature vector
           `u2e1[d, :]` (and position `d` in SE `s`'s initial feature vector `i2e1[s, :]`)
           is set to `0.0`.
      - For `val_select`, `val_calib`, and `test` pairs (`y_train_mat[d, s] == 0`), features and
        adjacency lists are 100% unchanged.
    """

    def __init__(
        self,
        drug_feature: nn.Embedding,
        side_feature: nn.Embedding,
        adj: Dict[int, list],
        embed_dim: int,
        weight_decay: float = 0.0005,
        droprate: float = 0.5,
        cuda="cpu",
        is_drug_part: bool = True,
        profile_self_correction: bool = False,
        is_interaction_view: bool = False,
        y_train_mat: Optional[np.ndarray] = None,
    ):
        super().__init__(
            drug_feature, side_feature, adj, embed_dim,
            weight_decay=weight_decay, droprate=droprate, cuda=cuda, is_drug_part=is_drug_part,
        )
        self.profile_self_correction = bool(profile_self_correction)
        self.is_interaction_view = bool(is_interaction_view)
        self.y_train_mat = y_train_mat

    def forward_pair_aware(
        self,
        nodes_u: torch.Tensor,
        nodes_i: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = int(nodes_u.shape[0])
        u_np = nodes_u.detach().cpu().numpy().astype(int)
        i_np = nodes_i.detach().cpu().numpy().astype(int)
        threshold = 24

        if self.profile_self_correction and self.y_train_mat is not None:
            is_train_pair = (self.y_train_mat[u_np, i_np] > 0.0)
        else:
            is_train_pair = np.zeros(bsz, dtype=bool)

        if self.is_drug:
            u_raw = self.drugfeature.weight[nodes_u]
            if self.is_interaction_view and np.any(is_train_pair):
                u_raw = u_raw.clone()
                b_pos = torch.as_tensor(np.where(is_train_pair)[0], dtype=torch.long, device=self.device)
                u_raw[b_pos, nodes_i[b_pos]] = 0.0
            nodes_fea = self.u_layer(u_raw).to(self.device)
        else:
            i_raw = self.sidefeature.weight[nodes_i]
            if self.is_interaction_view and np.any(is_train_pair):
                i_raw = i_raw.clone()
                b_pos = torch.as_tensor(np.where(is_train_pair)[0], dtype=torch.long, device=self.device)
                i_raw[b_pos, nodes_u[b_pos]] = 0.0
            nodes_fea = self.i_layer(i_raw).to(self.device)

        neigh_embed = torch.empty((bsz, self.embed_dim), dtype=torch.float32, device=self.device)

        for b in range(bsz):
            u_id = int(u_np[b])
            i_id = int(i_np[b])
            sc_active = bool(is_train_pair[b])

            if self.is_drug:
                raw_single = self.drugfeature.weight[nodes_u[b:b + 1]]
                if self.is_interaction_view and sc_active:
                    raw_single = raw_single.clone()
                    raw_single[0, i_id] = 0.0
                node_feature = self.u_layer(raw_single).to(self.device)
                interactions = self.adj.get(u_id, [])
                if sc_active and interactions:
                    interactions = [item for item in interactions if item[0] != i_id]
            else:
                raw_single = self.sidefeature.weight[nodes_i[b:b + 1]]
                if self.is_interaction_view and sc_active:
                    raw_single = raw_single.clone()
                    raw_single[0, u_id] = 0.0
                node_feature = self.i_layer(raw_single).to(self.device)
                interactions = self.adj.get(i_id, [])
                if sc_active and interactions:
                    interactions = [item for item in interactions if item[0] != u_id]

            n = len(interactions)
            if n == 0:
                neigh_embed[b] = node_feature[0]
                continue

            if n < threshold:
                sel_ids = [item[0] for item in interactions]
                times = n
            else:
                sel_ids = a_res(interactions, threshold)
                times = threshold

            sel_t = torch.as_tensor(sel_ids, dtype=torch.long, device=self.device)
            if self.is_drug:
                neighs_feature = self.i_layer(self.sidefeature.weight[sel_t]).to(self.device)
            else:
                neighs_feature = self.u_layer(self.drugfeature.weight[sel_t]).to(self.device)

            att_w = self.att(neighs_feature, node_feature, times)
            embedding = torch.mm(neighs_feature.t(), att_w)
            neigh_embed[b] = embedding.t()[0]

        return nodes_fea, neigh_embed


class _SelfCorrectingEncoder(encoder):
    def __init__(self, embedding_dim: int, agg: _SelfCorrectingAggregator, cuda="cpu", is_user_part: bool = True):
        super().__init__(embedding_dim, agg, cuda=cuda, is_user_part=is_user_part)

    def forward(self, nodes_u: torch.Tensor, nodes_i: torch.Tensor) -> torch.Tensor:
        if getattr(self.aggregator, 'profile_self_correction', False):
            nodes_fea, neigh_embed = self.aggregator.forward_pair_aware(nodes_u, nodes_i)
            combined = torch.cat((nodes_fea, neigh_embed), dim=1)
            return self.layer(combined).to(self.device)
        return super().forward(nodes_u, nodes_i)


def _load_mgpred_static_features() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the 4 static feature matrices used by MGPred for all 750 drugs and 994 SEs."""
    mg_data = os.path.join(MGPRED_DIR, 'data')
    sd_data = os.path.join(REPO_DIR, 'shared_data', 'sdpred_750')

    def _read_pkl(primary_path: str, fallback_path: str) -> np.ndarray:
        path = primary_path if os.path.exists(primary_path) else fallback_path
        with open(path, 'rb') as f:
            return np.asarray(pickle.load(f), dtype=np.float32)

    drug_stitch5 = _read_pkl(
        os.path.join(mg_data, 'Text_similarity_five.pkl'),
        os.path.join(sd_data, 'Text_similarity_five.pkl'),
    )
    drug_w2v = _read_pkl(
        os.path.join(mg_data, 'Drug_word2vec.pkl'),
        os.path.join(sd_data, 'drug_mol.pkl'),
    )
    se_semantic = _read_pkl(
        os.path.join(mg_data, 'effect_side_semantic.pkl'),
        os.path.join(sd_data, 'side_effect_semantic.pkl'),
    )
    se_glove = _read_pkl(
        os.path.join(mg_data, 'glove_wordEmbedding.pkl'),
        os.path.join(sd_data, 'glove_wordEmbedding.pkl'),
    )

    assert drug_stitch5.shape == (N_DRUGS, N_DRUGS), f"Unexpected STITCH shape {drug_stitch5.shape}"
    assert drug_w2v.shape[0] == N_DRUGS, f"Unexpected Mol2vec shape {drug_w2v.shape}"
    assert se_semantic.shape == (N_SIDES, N_SIDES), f"Unexpected SE semantic shape {se_semantic.shape}"
    assert se_glove.shape[0] == N_SIDES, f"Unexpected GloVe shape {se_glove.shape}"
    return drug_stitch5, drug_w2v, se_semantic, se_glove


def _make_frozen_or_trainable_embedding(mat: np.ndarray, device: torch.device) -> nn.Embedding:
    """Create `nn.Embedding` initialized from `mat`, matching `Ten_Fold_test.py` L238-274."""
    emb = nn.Embedding(mat.shape[0], mat.shape[1])
    emb.weight = nn.Parameter(torch.tensor(mat, dtype=torch.float32))
    return emb.to(device)


def _build_mgpred_model_for_fold(
    train_pairs: np.ndarray,
    static_feats: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    cfg: dict,
    device: torch.device,
) -> MGPred:
    """Construct leak-free `MGPred` instance using only `train_pairs` for `ufea1/ifea1` and `adj`."""
    drug_stitch5, drug_w2v, se_semantic, se_glove = static_feats
    drug_side_train = make_train_only_freq(train_pairs).astype(np.float32)
    self_corr = bool(cfg.get('profile_self_correction', False))

    u_adj: Dict[int, list] = {}
    i_adj: Dict[int, list] = {}
    for row in train_pairs:
        u = int(row[0])
        i = int(row[1])
        r = int(round(float(row[2])))
        if r > 0:
            u_adj.setdefault(u, []).append((i, r))
            i_adj.setdefault(i, []).append((u, r))

    u2e1 = _make_frozen_or_trainable_embedding(drug_side_train, device)
    i2e1 = _make_frozen_or_trainable_embedding(drug_side_train.T, device)
    u2e2 = _make_frozen_or_trainable_embedding(drug_stitch5, device)
    i2e2 = _make_frozen_or_trainable_embedding(se_semantic, device)
    u2e3 = _make_frozen_or_trainable_embedding(drug_w2v, device)
    i2e3 = _make_frozen_or_trainable_embedding(se_glove, device)

    embed_dim = int(cfg['embed_dim'])
    wd = float(cfg['weight_decay'])
    drop = float(cfg['droprate'])

    u_agg1 = _SelfCorrectingAggregator(
        u2e1, i2e1, u_adj, embed_dim, cuda=device, weight_decay=wd, droprate=drop,
        is_drug_part=True, profile_self_correction=self_corr, is_interaction_view=True,
        y_train_mat=drug_side_train,
    )
    u_enc1 = _SelfCorrectingEncoder(embed_dim, u_agg1, cuda=device, is_user_part=True)
    u_agg2 = _SelfCorrectingAggregator(
        u2e2, i2e2, u_adj, embed_dim, cuda=device, weight_decay=wd, droprate=drop,
        is_drug_part=True, profile_self_correction=self_corr, is_interaction_view=False,
        y_train_mat=drug_side_train,
    )
    u_enc2 = _SelfCorrectingEncoder(embed_dim, u_agg2, cuda=device, is_user_part=True)
    u_agg3 = _SelfCorrectingAggregator(
        u2e3, i2e3, u_adj, embed_dim, cuda=device, weight_decay=wd, droprate=drop,
        is_drug_part=True, profile_self_correction=self_corr, is_interaction_view=False,
        y_train_mat=drug_side_train,
    )
    u_enc3 = _SelfCorrectingEncoder(embed_dim, u_agg3, cuda=device, is_user_part=True)

    i_agg1 = _SelfCorrectingAggregator(
        u2e1, i2e1, i_adj, embed_dim, cuda=device, weight_decay=wd, droprate=drop,
        is_drug_part=False, profile_self_correction=self_corr, is_interaction_view=True,
        y_train_mat=drug_side_train,
    )
    i_enc1 = _SelfCorrectingEncoder(embed_dim, i_agg1, cuda=device, is_user_part=False)
    i_agg2 = _SelfCorrectingAggregator(
        u2e2, i2e2, i_adj, embed_dim, cuda=device, weight_decay=wd, droprate=drop,
        is_drug_part=False, profile_self_correction=self_corr, is_interaction_view=False,
        y_train_mat=drug_side_train,
    )
    i_enc2 = _SelfCorrectingEncoder(embed_dim, i_agg2, cuda=device, is_user_part=False)
    i_agg3 = _SelfCorrectingAggregator(
        u2e3, i2e3, i_adj, embed_dim, cuda=device, weight_decay=wd, droprate=drop,
        is_drug_part=False, profile_self_correction=self_corr, is_interaction_view=False,
        y_train_mat=drug_side_train,
    )
    i_enc3 = _SelfCorrectingEncoder(embed_dim, i_agg3, cuda=device, is_user_part=False)

    model = MGPred(
        u_enc1, u_enc2, u_enc3,
        i_enc1, i_enc2, i_enc3,
        embed_dim=embed_dim,
        N=int(cfg['N']),
        droprate=drop,
        beta_ema=float(cfg['beta_ema']),
    ).to(device)

    if model.beta_ema > 0.0:
        # MGPred.load_ema_params() divides avg_param by (1 - beta_ema ** steps_ema)
        # (Louizos et al. 2018 bias correction), which mathematically requires
        # zero-initialized avg_param so EMA weights sum to 1.0 from step 1.
        model.avg_param = [torch.zeros_like(p.data, device=device) for p in model.parameters()]
        model.steps_ema = 0.0

    for layer in model.layers:
        layer.floatTensor = (
            torch.cuda.FloatTensor if device.type == 'cuda' else torch.FloatTensor
        )

    return model


def _make_pair_loader(pairs: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    u = torch.tensor(pairs[:, 0].astype(np.int64), dtype=torch.long)
    i = torch.tensor(pairs[:, 1].astype(np.int64), dtype=torch.long)
    r = torch.tensor(pairs[:, 2].astype(np.float32), dtype=torch.float32)
    ds = TensorDataset(u, i, r)
    drop_last = bool(shuffle and (len(ds) > batch_size) and (len(ds) % batch_size == 1))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


@torch.no_grad()
def _predict_mgpred(
    model: MGPred,
    loader: DataLoader,
    device: torch.device,
    use_ema: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[dict]]:
    """Run deterministic inference (optionally loading & snapshotting EMA weights)."""
    model.eval()
    old_params = None
    if use_ema and model.beta_ema > 0.0 and model.steps_ema > 0:
        old_params = model.get_params()
        model.load_ema_params()

    preds_list = []
    labels_list = []
    for batch_u, batch_i, batch_r in loader:
        scores = model(batch_u.to(device), batch_i.to(device))
        preds_list.append(np.atleast_1d(scores.detach().cpu().numpy()).astype(np.float32))
        labels_list.append(np.atleast_1d(batch_r.numpy()).astype(np.float32))

    ema_state = copy.deepcopy(model.state_dict())

    if old_params is not None:
        model.load_params(old_params)

    return np.concatenate(labels_list), np.concatenate(preds_list), ema_state


def run_mgpred_fold(
    fold: int,
    split_type: str,
    cfg: dict,
    static_feats: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    device: torch.device,
    results_dir: str = DEFAULT_RESULTS_V2_DIR,
    drive_dir: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    t0 = time.time()
    set_global_seed(int(cfg['seed']) + int(fold))
    fold_split = load_fold_data(fold, split_type)
    self_corr = bool(cfg.get('profile_self_correction', False))
    method_name = 'MGPred_selfcorr' if self_corr else 'MGPred'
    eval_test = bool(cfg.get('eval_test', False))

    model = _build_mgpred_model_for_fold(fold_split.train, static_feats, cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg['lr']))

    batch_size = int(cfg['batch_size'])
    eval_bs = int(cfg['eval_batch_size'])
    epochs = int(cfg['epochs'])
    eval_every = max(1, int(cfg['eval_every']))

    train_loader = _make_pair_loader(fold_split.train, batch_size=batch_size, shuffle=True)
    train_eval_loader = _make_pair_loader(fold_split.train, batch_size=eval_bs, shuffle=False)
    valsel_loader = _make_pair_loader(fold_split.valsel, batch_size=eval_bs, shuffle=False)
    test_loader = _make_pair_loader(fold_split.test, batch_size=eval_bs, shuffle=False)

    best_f1 = -float('inf')
    best_f1_epoch = 1
    best_f1_state = copy.deepcopy(model.state_dict())

    best_rmse = float('inf')
    best_rmse_epoch = 1
    best_rmse_state = copy.deepcopy(model.state_dict())
    epoch_valsel_curve = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch_u, batch_i, batch_r in train_loader:
            optimizer.zero_grad()
            loss = model.loss(batch_u.to(device), batch_i.to(device), batch_r.to(device))
            loss.backward()
            optimizer.step()

            for layer in model.layers:
                layer.constrain_parameters()
            if model.beta_ema > 0.0:
                model.update_ema()

            running_loss += float(loss.item())
            n_batches += 1

        if epoch == 1 or epoch % eval_every == 0 or epoch == epochs:
            val_y, val_p, ema_state = _predict_mgpred(model, valsel_loader, device, use_ema=True)
            val_metrics = eval_regression_predictions(val_y, val_p)
            epoch_valsel_curve.append(float(val_metrics['MacroF1']))

            if val_metrics['MacroF1'] > best_f1:
                best_f1 = val_metrics['MacroF1']
                best_f1_epoch = epoch
                best_f1_state = ema_state

            if val_metrics['RMSE'] < best_rmse:
                best_rmse = val_metrics['RMSE']
                best_rmse_epoch = epoch
                best_rmse_state = ema_state

            if verbose and (epoch == 1 or epoch % max(eval_every, 5) == 0 or epoch == epochs):
                print(
                    f"  [{method_name} Split {split_type} Fold {fold}] Ep {epoch:02d}/{epochs:02d} | "
                    f"Loss={running_loss / max(1, n_batches):.4f} | "
                    f"valsel MacroF1={val_metrics['MacroF1']:.4f} (best={best_f1:.4f}@{best_f1_epoch}) | "
                    f"RMSE={val_metrics['RMSE']:.4f} (best={best_rmse:.4f}@{best_rmse_epoch})"
                )

    # 1. Primary evaluation (MacroF1-selected checkpoint)
    model.load_state_dict(best_f1_state)
    tr_y_eval, tr_p_eval, _ = _predict_mgpred(model, train_eval_loader, device, use_ema=False)
    train_eval_final = eval_regression_predictions(tr_y_eval, tr_p_eval)
    best_train_macro_f1 = float(train_eval_final['MacroF1'])

    val_y, val_p_f1, _ = _predict_mgpred(model, valsel_loader, device, use_ema=False)
    valsel_final = eval_regression_predictions(val_y, val_p_f1)

    if eval_test:
        test_y, test_p_f1, _ = _predict_mgpred(model, test_loader, device, use_ema=False)
        assert np.array_equal(np.round(test_y), np.round(fold_split.test[:, 2]))
        test_metrics_f1 = eval_regression_predictions(test_y, test_p_f1)
    else:
        test_p_f1 = None
        test_metrics_f1 = None

    # 2. Secondary appendix evaluation (RMSE-selected checkpoint)
    model.load_state_dict(best_rmse_state)
    _, val_p_rmse, _ = _predict_mgpred(model, valsel_loader, device, use_ema=False)
    valsel_rmse_final = eval_regression_predictions(val_y, val_p_rmse)
    if eval_test:
        _, test_p_rmse, _ = _predict_mgpred(model, test_loader, device, use_ema=False)
        test_metrics_rmse = eval_regression_predictions(test_y, test_p_rmse)
    else:
        test_p_rmse = None
        test_metrics_rmse = None

    elapsed = time.time() - t0
    train_valsel_gap = float(best_train_macro_f1 - float(valsel_final['MacroF1']))
    fold_summary = {
        'method_name': method_name,
        'split_type': split_type,
        'fold': int(fold),
        'profile_self_correction': bool(self_corr),
        'best_epoch': int(best_f1_epoch),
        'best_valsel_MacroF1': float(best_f1),
        'best_train_macro_f1': float(best_train_macro_f1),
        'train_eval_mode': True,
        'train_valsel_macro_f1_gap': float(train_valsel_gap),
        'epoch_valsel_macro_f1_curve': epoch_valsel_curve,
        'best_rmse_epoch': int(best_rmse_epoch),
        'best_valsel_RMSE': float(best_rmse),
        'valsel_metrics': valsel_final,
        'rmse_sel_valsel_metrics': valsel_rmse_final,
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
        rep_m = test_metrics_f1 if test_metrics_f1 is not None else valsel_final
        rep_tag = "Test" if test_metrics_f1 is not None else "valsel"
        print(
            f"  --> {method_name} Fold {fold} Done ({elapsed:.1f}s) | Best Ep={best_f1_epoch} | "
            f"{rep_tag} MacroF1={rep_m['MacroF1']:.4f}, QWK={rep_m['QWK']:.4f}, "
            f"MAE={rep_m['MAE']:.4f}, RMSE={rep_m['RMSE']:.4f} | Gap(tr-val)={train_valsel_gap:+.4f}"
        )
    return fold_summary


def run_mgpred_experiment(
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
        'eval_every': 5,
        'batch_size': 256,
        'eval_batch_size': 256,
        'lr': 0.001,
        'embed_dim': 64,
        'weight_decay': 0.0005,
        'droprate': 0.5,
        'N': 30000,
        'beta_ema': 0.999,
        'selection_metric': 'MacroF1',
        'profile_self_correction': False,
        'eval_test': False,
        'device': None,
    }
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                cfg[k] = v

    self_corr = bool(cfg.get('profile_self_correction', False))
    method_name = 'MGPred_selfcorr' if self_corr else 'MGPred'
    eval_test = bool(cfg.get('eval_test', False))
    folds = get_fold_list(mode=mode, custom_folds=custom_folds)
    device = get_device(cfg.get('device'))
    static_feats = _load_mgpred_static_features()

    if verbose:
        print(
            f"=== Running {method_name} | Split {split_type} | Mode={mode} (folds={folds}) | "
            f"Device={device} | Epochs={cfg['epochs']} | SelfCorr={self_corr} | Resume={resume} ==="
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

        s = run_mgpred_fold(
            fold=fold,
            split_type=split_type,
            cfg=cfg,
            static_feats=static_feats,
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
    parser = argparse.ArgumentParser(description='Run leak-free MGPred under Protocol v2.')
    parser.add_argument('--split', type=str, default='B', choices=['A', 'B'])
    parser.add_argument('--mode', type=str, default='full', choices=['quick', 'dev', 'full'])
    parser.add_argument('--folds', type=int, nargs='*', default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--eval_every', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--profile_self_correction', type=int, choices=[0, 1], default=None)
    parser.add_argument('--eval_test', action='store_true')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_V2_DIR)
    parser.add_argument('--drive_dir', type=str, default=None)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    run_mgpred_experiment(
        split_type=args.split,
        mode=args.mode,
        custom_folds=args.folds,
        results_dir=args.results_dir,
        drive_dir=args.drive_dir,
        resume=args.resume,
        overrides={
            'epochs': args.epochs,
            'eval_every': args.eval_every,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'profile_self_correction': (
                bool(args.profile_self_correction) if args.profile_self_correction is not None else None
            ),
            'eval_test': True if args.eval_test else None,
            'device': args.device,
        },
        verbose=True,
    )


if __name__ == '__main__':
    main()
