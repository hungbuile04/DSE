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


def _make_index_loader(pairs_with_rating: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    d = torch.tensor(pairs_with_rating[:, 0].astype(np.int64), dtype=torch.long)
    s = torch.tensor(pairs_with_rating[:, 1].astype(np.int64), dtype=torch.long)
    r = torch.tensor(pairs_with_rating[:, 2].astype(np.float32), dtype=torch.float32)
    ds = TensorDataset(d, s, r)
    drop_last = bool(shuffle and (len(ds) > batch_size) and (len(ds) % batch_size == 1))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


@torch.no_grad()
def _predict_sdpred(
    model: ConvNCF,
    loader: DataLoader,
    drug_mat_t: torch.Tensor,
    side_mat_t: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `(y_true_ratings, pred_binary_scores, pred_freq_scores)`."""
    model.eval()
    ratings_all, cls_all, reg_all = [], [], []
    for batch_d, batch_s, batch_r in loader:
        d_feat = drug_mat_t[batch_d.to(device)]
        s_feat = side_mat_t[batch_s.to(device)]
        scores_cls, scores_reg = model(d_feat, s_feat, device)
        cls_all.append(np.atleast_1d(scores_cls.detach().cpu().numpy()).astype(np.float32))
        reg_all.append(np.atleast_1d(scores_reg.detach().cpu().numpy()).astype(np.float32))
        ratings_all.append(np.atleast_1d(batch_r.numpy()).astype(np.float32))
    return np.concatenate(ratings_all), np.concatenate(cls_all), np.concatenate(reg_all)


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

    train_loader = _make_index_loader(train_combined, batch_size=batch_size, shuffle=True)
    valsel_loader = _make_index_loader(fold_split.valsel, batch_size=eval_bs, shuffle=False)
    test_pos_loader = _make_index_loader(fold_split.test, batch_size=eval_bs, shuffle=False)
    test_neg_loader = _make_index_loader(neg_test, batch_size=eval_bs, shuffle=False)

    drop = float(cfg['dropout'])
    model = ConvNCF(
        7500, 3976, int(cfg['embed_dim']), batch_size, dropout1=drop, dropout2=drop
    ).to(device)

    cls_criterion = nn.MSELoss()
    reg_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(cfg['lr']), weight_decay=float(cfg['weight_decay'])
    )

    best_f1 = -float('inf')
    best_f1_epoch = 1
    best_f1_state = copy.deepcopy(model.state_dict())

    best_rmse = float('inf')
    best_rmse_epoch = 1
    best_rmse_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch_d, batch_s, batch_r in train_loader:
            batch_r_dev = batch_r.to(device)
            batch_lbl_dev = (batch_r_dev > 0).float()
            d_feat = drug_mat_t[batch_d.to(device)]
            s_feat = side_mat_t[batch_s.to(device)]

            optimizer.zero_grad()
            logits, pred_reg = model(d_feat, s_feat, device)
            logits = logits.view(-1)
            pred_reg = pred_reg.view(-1)

            loss1 = cls_criterion(logits, batch_lbl_dev)
            pos_mask = batch_r_dev > 0
            if pos_mask.any():
                loss2 = reg_criterion(pred_reg[pos_mask], batch_r_dev[pos_mask])
                total_loss = loss1 * loss2
            else:
                total_loss = loss1

            total_loss.backward()
            optimizer.step()

            running_loss += float(total_loss.item())
            n_batches += 1

        val_y, _, val_p = _predict_sdpred(model, valsel_loader, drug_mat_t, side_mat_t, device)
        val_metrics = eval_regression_predictions(val_y, val_p)

        if val_metrics['MacroF1'] > best_f1:
            best_f1 = val_metrics['MacroF1']
            best_f1_epoch = epoch
            best_f1_state = copy.deepcopy(model.state_dict())

        if val_metrics['RMSE'] < best_rmse:
            best_rmse = val_metrics['RMSE']
            best_rmse_epoch = epoch
            best_rmse_state = copy.deepcopy(model.state_dict())

        if verbose and (epoch == 1 or epoch % 20 == 0 or epoch == epochs):
            print(
                f"  [SDPred Split {split_type} Fold {fold}] Ep {epoch:03d}/{epochs:03d} | "
                f"Loss={running_loss / max(1, n_batches):.4f} | "
                f"valsel MacroF1={val_metrics['MacroF1']:.4f} (best={best_f1:.4f}@{best_f1_epoch}) | "
                f"RMSE={val_metrics['RMSE']:.4f} (best={best_rmse:.4f}@{best_rmse_epoch})"
            )

    # 1. Primary evaluation (MacroF1-selected checkpoint)
    model.load_state_dict(best_f1_state)
    val_y, _, val_p_f1 = _predict_sdpred(model, valsel_loader, drug_mat_t, side_mat_t, device)
    test_y, test_cls_pos, test_p_f1 = _predict_sdpred(
        model, test_pos_loader, drug_mat_t, side_mat_t, device
    )
    _, test_cls_neg, _ = _predict_sdpred(model, test_neg_loader, drug_mat_t, side_mat_t, device)
    assert np.array_equal(np.round(test_y), np.round(fold_split.test[:, 2]))

    bin_y = np.concatenate([np.ones(len(test_cls_pos)), np.zeros(len(test_cls_neg))])
    bin_s = np.concatenate([test_cls_pos, test_cls_neg])
    bin_auc = float(roc_auc_score(bin_y, bin_s))
    bin_aupr = float(average_precision_score(bin_y, bin_s))

    valsel_final = eval_regression_predictions(val_y, val_p_f1)
    test_metrics_f1 = eval_regression_predictions(test_y, test_p_f1)
    test_metrics_f1['BinaryAUC'] = bin_auc
    test_metrics_f1['BinaryAUPR'] = bin_aupr

    # 2. Secondary appendix evaluation (RMSE-selected checkpoint)
    if best_rmse_epoch == best_f1_epoch:
        test_p_rmse = test_p_f1.copy()
        test_metrics_rmse = dict(test_metrics_f1)
    else:
        model.load_state_dict(best_rmse_state)
        _, _, test_p_rmse = _predict_sdpred(model, test_pos_loader, drug_mat_t, side_mat_t, device)
        test_metrics_rmse = eval_regression_predictions(test_y, test_p_rmse)

    elapsed = time.time() - t0
    fold_summary = {
        'method_name': 'SDPred',
        'split_type': split_type,
        'fold': int(fold),
        'best_epoch': int(best_f1_epoch),
        'best_valsel_MacroF1': float(best_f1),
        'best_rmse_epoch': int(best_rmse_epoch),
        'best_valsel_RMSE': float(best_rmse),
        'valsel_metrics': valsel_final,
        'test_metrics': test_metrics_f1,
        'rmse_sel_test_metrics': test_metrics_rmse,
        'elapsed_seconds': float(elapsed),
    }

    save_baseline_fold_outputs(
        method_name='SDPred',
        split_type=split_type,
        fold=fold,
        test_split=fold_split.test,
        test_preds_float=test_p_f1,
        fold_summary=fold_summary,
        results_dir=results_dir,
        drive_dir=drive_dir,
        rmse_sel_preds_float=test_p_rmse,
    )

    if verbose:
        print(
            f"  --> SDPred Fold {fold} Done ({elapsed:.1f}s) | "
            f"Test MacroF1={test_metrics_f1['MacroF1']:.4f}, QWK={test_metrics_f1['QWK']:.4f}, "
            f"RMSE={test_metrics_f1['RMSE']:.4f}, BinAUC={bin_auc:.4f}"
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
        'device': None,
    }
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                cfg[k] = v

    folds = get_fold_list(mode=mode, custom_folds=custom_folds)
    device = get_device(cfg.get('device'))
    static_views = _load_sdpred_static_views()

    if verbose:
        print(
            f"=== Running SDPred | Split {split_type} | Mode={mode} (folds={folds}) | "
            f"Device={device} | Epochs={cfg['epochs']} | Resume={resume} ==="
        )

    fold_summaries = []
    for fold in folds:
        if resume:
            cached = try_restore_baseline_fold(
                'SDPred', split_type, fold, results_dir, drive_dir=drive_dir, has_rmse_sel=True
            )
            if cached is not None:
                if verbose:
                    tm = cached['test_metrics']
                    print(
                        f"  [RESUME] Skipping SDPred_{split_type} Fold {fold} "
                        f"(MacroF1={tm['MacroF1']:.4f}, RMSE={tm['RMSE']:.4f})"
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
        method_name='SDPred',
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
            'device': args.device,
        },
        verbose=True,
    )


if __name__ == '__main__':
    main()
