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
    GPU-vectorized extension of `MGPred`'s `aggregator` supporting `profile_self_correction`:
      - Replaces the 128-iteration Python `for` loop and CPU `heapq` reservoir sampling per batch
        with batched GPU Efraimidis-Spirakis sampling (`k = u ** (1/w)` + `torch.topk`) and
        batched attention over `(B, <=24, embed_dim)`.
      - When `profile_self_correction=True` and a batch sample `(d = nodes_u[b], s = nodes_i[b])`
        belongs to `Y_train` (`y_train_mat[d, s] > 0`):
        1. Target edge `(d, s)` is masked out of `u_adj[d]` (for drug aggregators) and `(s, d)`
           is masked out of `i_adj[s]` (for SE aggregators).
        2. If `is_interaction_view=True` (`u_agg1` / `i_agg1`), entry `(d, s)` in `u2e1[d, :]`
           (and `(s, d)` in `i2e1[s, :]`) is masked to `0.0`.
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

        n_nodes = int(self.drugfeature.num_embeddings if self.is_drug else self.sidefeature.num_embeddings)
        max_deg = max((len(v) for v in adj.values()), default=1)
        adj_ids = np.zeros((n_nodes, max_deg), dtype=np.int64)
        adj_inv_w = np.zeros((n_nodes, max_deg), dtype=np.float32)
        adj_valid = np.zeros((n_nodes, max_deg), dtype=bool)
        for node_id, nbr_list in adj.items():
            for j, (nbr_id, w) in enumerate(nbr_list):
                adj_ids[int(node_id), j] = int(nbr_id)
                adj_inv_w[int(node_id), j] = 1.0 / float(max(w, 1e-6))
                adj_valid[int(node_id), j] = True

        self._adj_ids_t = torch.tensor(adj_ids, dtype=torch.long, device=self.device)
        self._adj_inv_w_t = torch.tensor(adj_inv_w, dtype=torch.float32, device=self.device)
        self._adj_valid_t = torch.tensor(adj_valid, dtype=torch.bool, device=self.device)
        if y_train_mat is not None:
            self._y_train_bool_t = torch.tensor(y_train_mat > 0.0, dtype=torch.bool, device=self.device)
        else:
            self._y_train_bool_t = None

    def forward_pair_aware(
        self,
        nodes_u: torch.Tensor,
        nodes_i: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        import torch.nn.functional as F

        nodes_u = nodes_u.to(self.device) if nodes_u.device != self.device else nodes_u
        nodes_i = nodes_i.to(self.device) if nodes_i.device != self.device else nodes_i

        if self.profile_self_correction and self._y_train_bool_t is not None:
            sc_mask = self._y_train_bool_t[nodes_u, nodes_i]
            has_sc = bool(sc_mask.any().item())
        else:
            sc_mask = None
            has_sc = False

        if self.is_drug:
            u_raw = self.drugfeature.weight[nodes_u]
            if self.is_interaction_view and has_sc:
                u_raw = u_raw.clone()
                b_pos = torch.nonzero(sc_mask, as_tuple=False).squeeze(1)
                u_raw[b_pos, nodes_i[b_pos]] = 0.0
            nodes_fea = self.u_layer(u_raw)
            center_ids = nodes_u
            partner_ids = nodes_i
        else:
            i_raw = self.sidefeature.weight[nodes_i]
            if self.is_interaction_view and has_sc:
                i_raw = i_raw.clone()
                b_pos = torch.nonzero(sc_mask, as_tuple=False).squeeze(1)
                i_raw[b_pos, nodes_u[b_pos]] = 0.0
            nodes_fea = self.i_layer(i_raw)
            center_ids = nodes_i
            partner_ids = nodes_u

        nbr_ids = self._adj_ids_t[center_ids]
        inv_w = self._adj_inv_w_t[center_ids]
        valid = self._adj_valid_t[center_ids]
        if has_sc:
            valid = valid & (~(sc_mask.unsqueeze(1) & (nbr_ids == partner_ids.unsqueeze(1))))

        k_sel = min(24, int(nbr_ids.size(1)))
        u_rand = torch.empty_like(inv_w).uniform_(1e-6, 1.0 - 1e-6)
        keys = torch.where(valid, u_rand.pow(inv_w), torch.full_like(inv_w, -1.0))
        top_keys, top_pos = torch.topk(keys, k=k_sel, dim=1)
        top_valid = top_keys > 0.0
        top_nbr_ids = torch.gather(nbr_ids, 1, top_pos)

        if self.is_drug:
            all_nbr_emb = self.i_layer(self.sidefeature.weight)
        else:
            all_nbr_emb = self.u_layer(self.drugfeature.weight)
        neighs_feature = all_nbr_emb[top_nbr_ids]

        node_rep = nodes_fea.unsqueeze(1).expand(-1, k_sel, -1)
        att_in = torch.cat((neighs_feature, node_rep), dim=-1)
        att_h = F.relu(self.att.att1(att_in), inplace=True)
        att_h = F.dropout(att_h, training=self.training, p=self.droprate)
        att_logits = self.att.att2(att_h).squeeze(-1)
        att_logits = att_logits.masked_fill(~top_valid, -1e9)
        att_w = F.softmax(att_logits, dim=1).unsqueeze(-1)
        att_w = torch.where(top_valid.unsqueeze(-1), att_w, torch.zeros_like(att_w))
        neigh_embed = (neighs_feature * att_w).sum(dim=1)

        has_any_nbr = top_valid[:, 0].unsqueeze(-1)
        neigh_embed = torch.where(has_any_nbr, neigh_embed, nodes_fea)
        return nodes_fea, neigh_embed


class _SelfCorrectingEncoder(encoder):
    def __init__(self, embedding_dim: int, agg: _SelfCorrectingAggregator, cuda="cpu", is_user_part: bool = True):
        super().__init__(embedding_dim, agg, cuda=cuda, is_user_part=is_user_part)

    def forward(self, nodes_u: torch.Tensor, nodes_i: torch.Tensor) -> torch.Tensor:
        nodes_fea, neigh_embed = self.aggregator.forward_pair_aware(nodes_u, nodes_i)
        combined = torch.cat((nodes_fea, neigh_embed), dim=1)
        return self.layer(combined)


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


class _FastPairLoader:
    """Vectorized tensor batch loader avoiding per-sample Python __getitem__ overhead."""

    def __init__(
        self,
        pairs: np.ndarray,
        batch_size: int,
        shuffle: bool,
        device: Optional[torch.device] = None,
    ):
        self.n = int(len(pairs))
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(shuffle and (self.n > self.batch_size) and (self.n % self.batch_size == 1))
        self.device = device
        u_np = pairs[:, 0].astype(np.int64)
        i_np = pairs[:, 1].astype(np.int64)
        r_np = pairs[:, 2].astype(np.float32)
        self.r_cpu = r_np
        if device is not None and device.type != 'cpu':
            self.u = torch.tensor(u_np, dtype=torch.long, device=device)
            self.i = torch.tensor(i_np, dtype=torch.long, device=device)
            self.r = torch.tensor(r_np, dtype=torch.float32, device=device)
        else:
            self.u = torch.tensor(u_np, dtype=torch.long)
            self.i = torch.tensor(i_np, dtype=torch.long)
            self.r = torch.tensor(r_np, dtype=torch.float32)

    def __len__(self) -> int:
        if self.drop_last:
            return self.n // self.batch_size
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            perm_cpu = torch.randperm(self.n)
            perm = perm_cpu.to(self.u.device) if self.u.device.type != 'cpu' else perm_cpu
            u_ep = self.u[perm]
            i_ep = self.i[perm]
            r_ep = self.r[perm]
        else:
            u_ep = self.u
            i_ep = self.i
            r_ep = self.r

        limit = (self.n // self.batch_size) * self.batch_size if self.drop_last else self.n
        for start in range(0, limit, self.batch_size):
            end = min(start + self.batch_size, limit)
            yield u_ep[start:end], i_ep[start:end], r_ep[start:end]


def _make_pair_loader(
    pairs: np.ndarray,
    batch_size: int,
    shuffle: bool,
    device: Optional[torch.device] = None,
) -> _FastPairLoader:
    return _FastPairLoader(pairs, batch_size=batch_size, shuffle=shuffle, device=device)


@torch.no_grad()
def _predict_mgpred(
    model: MGPred,
    loader: _FastPairLoader,
    device: torch.device,
    use_ema: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[dict]]:
    """Run deterministic inference (optionally loading & snapshotting EMA weights)."""
    model.eval()
    old_params = None
    if use_ema and model.beta_ema > 0.0 and model.steps_ema > 0:
        old_params = model.get_params()
        model.load_ema_params()

    preds_t = []
    for batch_u, batch_i, _ in loader:
        b_u = batch_u.to(device) if batch_u.device != device else batch_u
        b_i = batch_i.to(device) if batch_i.device != device else batch_i
        scores = model(b_u, b_i).view(-1)
        preds_t.append(scores)

    preds_np = torch.cat(preds_t, dim=0).detach().cpu().numpy().astype(np.float32)
    ema_state = copy.deepcopy(model.state_dict())

    if old_params is not None:
        model.load_params(old_params)

    return loader.r_cpu.copy(), preds_np, ema_state


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
    eval_bs = max(int(cfg['eval_batch_size']), 1024)
    epochs = int(cfg['epochs'])
    eval_every = max(1, int(cfg['eval_every']))

    train_loader = _make_pair_loader(fold_split.train, batch_size=batch_size, shuffle=True, device=device)
    train_eval_loader = _make_pair_loader(fold_split.train, batch_size=eval_bs, shuffle=False, device=device)
    valsel_loader = _make_pair_loader(fold_split.valsel, batch_size=eval_bs, shuffle=False, device=device)
    test_loader = _make_pair_loader(fold_split.test, batch_size=eval_bs, shuffle=False, device=device)

    best_f1 = -float('inf')
    best_f1_epoch = 1
    best_f1_state = copy.deepcopy(model.state_dict())

    best_rmse = float('inf')
    best_rmse_epoch = 1
    best_rmse_state = copy.deepcopy(model.state_dict())
    epoch_valsel_curve = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss_t = torch.zeros((), dtype=torch.float32, device=device)
        n_batches = 0

        for batch_u, batch_i, batch_r in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(batch_u, batch_i, batch_r)
            loss.backward()
            optimizer.step()

            for layer in model.layers:
                layer.constrain_parameters()
            if model.beta_ema > 0.0:
                model.update_ema()

            running_loss_t += loss.detach()
            n_batches += 1

        if epoch == 1 or epoch % eval_every == 0 or epoch == epochs:
            running_loss = float(running_loss_t.item())
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
