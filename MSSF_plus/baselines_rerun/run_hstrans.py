"""
Leak-free Protocol v2 wrapper for HSTrans (`HSTrans/HSTrans_original/HSTrans/`).

Fixes applied without modifying `HSTrans/`:
  1. Uses unified 4-way split `(train, valsel, valcal, test)` via `load_fold_data`.
  2. Fixes critical data leakage in `identify_sub`: recomputes the top-50 enriched BPE
     chemical substructures per SE (`SE_sub_index_50` and `SE_sub_mask_50`) PER FOLD
     strictly from `fold_split.train` (80% training positives only), vectorized in NumPy.
  3. Fixes 50x I/O bottleneck: pre-tokenizes all 750 drug SMILES once in memory instead
     of calling `pd.read_csv()` inside `__getitem__` on every sample and epoch.
  4. Uses leak-free 1:1 negative sampling via `sample_negative_pairs_v2` (restricting
     Split B training negatives strictly to `fold_split.train_drugs`).
  5. Sets `model.device = device` directly in Python (zero `sed` patching), and selects
     primary checkpoint on `valsel` maximizing 5-class `MacroF1` (plus secondary `RMSE`
     checkpoint saved to `HSTrans_rmseSel_{A|B}`).
"""

import os
import sys
import csv
import copy
import time
import codecs
import argparse
import importlib.util
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score

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

HSTRANS_DIR = os.path.join(REPO_DIR, 'HSTrans', 'HSTrans_original', 'HSTrans')


def _import_hstrans_model():
    if HSTRANS_DIR not in sys.path:
        sys.path.insert(0, HSTRANS_DIR)
    spec = importlib.util.spec_from_file_location(
        'hstrans_net_mod', os.path.join(HSTRANS_DIR, 'Net.py')
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load HSTrans Net.py from {HSTRANS_DIR}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Trans


def _pretokenize_all_750_drugs(max_d: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pre-tokenize all 750 SMILES in `shared_data/drug_SMILES_750.csv` once using HSTrans's
    ChEMBL BPE vocabulary (`2586` subword units), returning `(750, 50)` token IDs and masks.
    """
    from subword_nmt.apply_bpe import BPE

    smiles_csv = os.path.join(SHARED_DIR, 'drug_SMILES_750.csv')
    drug_smiles: List[str] = []
    with open(smiles_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                drug_smiles.append(row[1])
    assert len(drug_smiles) == N_DRUGS, f"Expected {N_DRUGS} SMILES, got {len(drug_smiles)}"

    vocab_path = os.path.join(HSTRANS_DIR, 'data', 'drug_codes_chembl_freq_1500.txt')
    sub_csv_path = os.path.join(HSTRANS_DIR, 'data', 'subword_units_map_chembl_freq_1500.csv')
    with open(sub_csv_path, 'r', encoding='utf-8') as f:
        idx2word = [row['index'] for row in csv.DictReader(f)]
    words2idx = dict(zip(idx2word, range(len(idx2word))))

    with codecs.open(vocab_path, encoding='utf-8') as bpe_codes:
        dbpe = BPE(bpe_codes, merges=-1, separator='')

    tokens_mat = np.zeros((N_DRUGS, max_d), dtype=np.int64)
    masks_mat = np.zeros((N_DRUGS, max_d), dtype=np.int64)

    for d_idx, smi in enumerate(drug_smiles):
        pieces = dbpe.process_line(smi).split()
        try:
            ids = np.asarray([words2idx[p] for p in pieces], dtype=np.int64)
        except KeyError:
            ids = np.array([0], dtype=np.int64)

        L = len(ids)
        if L < max_d:
            tokens_mat[d_idx, :L] = ids
            masks_mat[d_idx, :L] = 1
        else:
            tokens_mat[d_idx, :] = ids[:max_d]
            masks_mat[d_idx, :] = 1

    return tokens_mat, masks_mat


def _identify_sub_train_only(
    train_pairs: np.ndarray,
    drug_tokens: np.ndarray,
    n_sub_cols: int = 2686,
    top_k: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact vectorized equivalent of `HSTrans/main.py::identify_sub`, computed strictly
    from `train_pairs` of the current fold (zero test/validation leakage).
    """
    se_sub = np.zeros((N_SIDES, n_sub_cols), dtype=np.float64)

    for row in train_pairs:
        d = int(row[0])
        s = int(row[1])
        lbl = float(row[2])
        if lbl <= 0:
            continue
        toks = drug_tokens[d]
        nz_toks = toks[toks != 0]
        if len(nz_toks) > 0:
            np.add.at(se_sub[s], nz_toks, lbl)

    total_n = float(np.sum(se_sub))
    if total_n <= 0:
        return np.zeros((N_SIDES, top_k), dtype=np.int64), np.zeros((N_SIDES, top_k), dtype=np.int64)

    se_p = np.sum(se_sub, axis=1, keepdims=True) / total_n
    sub_p = np.sum(se_sub, axis=0, keepdims=True) / total_n
    se_sub_p = se_sub / total_n

    with np.errstate(divide='ignore', invalid='ignore'):
        denom = np.sqrt((se_p * sub_p / total_n) * (1.0 - se_p) * (1.0 - sub_p))
        freq = ((se_sub_p - se_p * sub_p) / denom) + 1e-5

    non_nan = freq[~np.isnan(freq)]
    pct_95 = float(np.percentile(non_nan, 95)) if len(non_nan) > 0 else 0.0

    se_sub_index = np.zeros((N_SIDES, top_k), dtype=np.int64)
    for i in range(N_SIDES):
        row_f = freq[i]
        valid_cols = np.where((~np.isnan(row_f)) & (row_f > pct_95))[0]
        if len(valid_cols) > 0:
            order = valid_cols[np.argsort(row_f[valid_cols])[::-1]]
            chosen = order[:top_k]
            se_sub_index[i, :len(chosen)] = chosen

    se_sub_mask = (se_sub_index > 0).astype(np.int64)
    return se_sub_index, se_sub_mask


def _make_pair_loader(pairs_with_rating: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    d = torch.tensor(pairs_with_rating[:, 0].astype(np.int64), dtype=torch.long)
    s = torch.tensor(pairs_with_rating[:, 1].astype(np.int64), dtype=torch.long)
    r = torch.tensor(pairs_with_rating[:, 2].astype(np.float32), dtype=torch.float32)
    ds = TensorDataset(d, s, r)
    drop_last = bool(shuffle and (len(ds) > batch_size) and (len(ds) % batch_size == 1))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


@torch.no_grad()
def _predict_hstrans(
    model: torch.nn.Module,
    loader: DataLoader,
    drug_tokens_t: torch.Tensor,
    drug_masks_t: torch.Tensor,
    se_tokens_t: torch.Tensor,
    se_masks_t: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels_all, preds_all = [], []
    for batch_d, batch_s, batch_r in loader:
        d_tok = drug_tokens_t[batch_d.to(device)]
        d_msk = drug_masks_t[batch_d.to(device)]
        s_tok = se_tokens_t[batch_s.to(device)]
        s_msk = se_masks_t[batch_s.to(device)]
        out, _, _ = model(d_tok, s_tok, d_msk, s_msk)
        preds_all.append(np.atleast_1d(out.view(-1).detach().cpu().numpy()).astype(np.float32))
        labels_all.append(np.atleast_1d(batch_r.numpy()).astype(np.float32))
    return np.concatenate(labels_all), np.concatenate(preds_all)


def run_hstrans_fold(
    fold: int,
    split_type: str,
    cfg: dict,
    TransCls,
    drug_tok_bundle: Tuple[np.ndarray, np.ndarray],
    device: torch.device,
    results_dir: str = DEFAULT_RESULTS_V2_DIR,
    drive_dir: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    t0 = time.time()
    fold_seed = int(cfg['seed']) + int(fold)
    set_global_seed(fold_seed)
    fold_split = load_fold_data(fold, split_type)

    drug_tokens, drug_masks = drug_tok_bundle
    drug_side_train = make_train_only_freq(fold_split.train).astype(np.float32)

    # Leak-free SE substructure extraction strictly from fold_split.train
    se_sub_index, se_sub_mask = _identify_sub_train_only(fold_split.train, drug_tokens)

    drug_tokens_t = torch.tensor(drug_tokens, dtype=torch.long, device=device)
    drug_masks_t = torch.tensor(drug_masks, dtype=torch.long, device=device)
    se_tokens_t = torch.tensor(se_sub_index, dtype=torch.long, device=device)
    se_masks_t = torch.tensor(se_sub_mask, dtype=torch.long, device=device)

    # 1:1 leak-free negative sampling for training
    neg_train_xy = sample_negative_pairs_v2(
        drug_side_train, fold_split, n_samples=len(fold_split.train), part='train', seed=fold_seed
    )
    neg_train = np.column_stack([neg_train_xy, np.zeros(len(neg_train_xy), dtype=np.float32)])
    train_combined = np.vstack([fold_split.train.astype(np.float32), neg_train])

    neg_test_xy = sample_negative_pairs_v2(
        drug_side_train, fold_split, n_samples=len(fold_split.test), part='test', seed=fold_seed + 100
    )
    neg_test = np.column_stack([neg_test_xy, np.zeros(len(neg_test_xy), dtype=np.float32)])

    batch_size = int(cfg['batch_size'])
    eval_bs = int(cfg['eval_batch_size'])
    epochs = int(cfg['epochs'])
    eval_every = max(1, int(cfg['eval_every']))

    train_loader = _make_pair_loader(train_combined, batch_size=batch_size, shuffle=True)
    valsel_loader = _make_pair_loader(fold_split.valsel, batch_size=eval_bs, shuffle=False)
    test_pos_loader = _make_pair_loader(fold_split.test, batch_size=eval_bs, shuffle=False)
    test_neg_loader = _make_pair_loader(neg_test, batch_size=eval_bs, shuffle=False)

    model = TransCls().to(device)
    model.device = device
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

    eval_test = bool(cfg.get('eval_test', False))
    train_pos_loader = _make_pair_loader(fold_split.train, batch_size=eval_bs, shuffle=False)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch_d, batch_s, batch_r in train_loader:
            d_tok = drug_tokens_t[batch_d.to(device)]
            d_msk = drug_masks_t[batch_d.to(device)]
            s_tok = se_tokens_t[batch_s.to(device)]
            s_msk = se_masks_t[batch_s.to(device)]
            target = batch_r.to(device)

            optimizer.zero_grad()
            out, _, _ = model(d_tok, s_tok, d_msk, s_msk)
            loss = torch.sum((out.view(-1) - target) ** 2)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        if epoch == 1 or epoch % eval_every == 0 or epoch == epochs:
            val_y, val_p = _predict_hstrans(
                model, valsel_loader, drug_tokens_t, drug_masks_t, se_tokens_t, se_masks_t, device
            )
            val_metrics = eval_regression_predictions(val_y, val_p)

            if val_metrics['MacroF1'] > best_f1:
                best_f1 = val_metrics['MacroF1']
                best_f1_epoch = epoch
                best_f1_state = copy.deepcopy(model.state_dict())

            if val_metrics['RMSE'] < best_rmse:
                best_rmse = val_metrics['RMSE']
                best_rmse_epoch = epoch
                best_rmse_state = copy.deepcopy(model.state_dict())

            if verbose and (epoch == 1 or epoch % max(eval_every, 20) == 0 or epoch == epochs):
                print(
                    f"  [HSTrans Split {split_type} Fold {fold}] Ep {epoch:03d}/{epochs:03d} | "
                    f"Loss={running_loss / max(1, n_batches):.2f} | "
                    f"valsel MacroF1={val_metrics['MacroF1']:.4f} (best={best_f1:.4f}@{best_f1_epoch}) | "
                    f"RMSE={val_metrics['RMSE']:.4f} (best={best_rmse:.4f}@{best_rmse_epoch})"
                )

    # 1. Primary evaluation (MacroF1-selected checkpoint)
    model.load_state_dict(best_f1_state)
    tr_y, tr_p_f1 = _predict_hstrans(
        model, train_pos_loader, drug_tokens_t, drug_masks_t, se_tokens_t, se_masks_t, device
    )
    train_eval_metrics = eval_regression_predictions(tr_y, tr_p_f1)
    best_train_macro_f1 = float(train_eval_metrics['MacroF1'])

    val_y, val_p_f1 = _predict_hstrans(
        model, valsel_loader, drug_tokens_t, drug_masks_t, se_tokens_t, se_masks_t, device
    )
    assert np.array_equal(np.round(val_y), np.round(fold_split.valsel[:, 2]))
    valsel_final = eval_regression_predictions(val_y, val_p_f1)

    if best_rmse_epoch == best_f1_epoch:
        val_p_rmse = val_p_f1.copy()
        valsel_metrics_rmse = dict(valsel_final)
    else:
        model.load_state_dict(best_rmse_state)
        _, val_p_rmse = _predict_hstrans(
            model, valsel_loader, drug_tokens_t, drug_masks_t, se_tokens_t, se_masks_t, device
        )
        valsel_metrics_rmse = eval_regression_predictions(val_y, val_p_rmse)

    if eval_test:
        model.load_state_dict(best_f1_state)
        test_y, test_p_f1 = _predict_hstrans(
            model, test_pos_loader, drug_tokens_t, drug_masks_t, se_tokens_t, se_masks_t, device
        )
        _, test_p_neg = _predict_hstrans(
            model, test_neg_loader, drug_tokens_t, drug_masks_t, se_tokens_t, se_masks_t, device
        )
        assert np.array_equal(np.round(test_y), np.round(fold_split.test[:, 2]))

        bin_y = np.concatenate([np.ones(len(test_p_f1)), np.zeros(len(test_p_neg))])
        bin_s = np.concatenate([test_p_f1, test_p_neg])
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
            _, test_p_rmse = _predict_hstrans(
                model, test_pos_loader, drug_tokens_t, drug_masks_t, se_tokens_t, se_masks_t, device
            )
            test_metrics_rmse = eval_regression_predictions(test_y, test_p_rmse)
    else:
        test_p_f1 = None
        test_p_rmse = None
        test_metrics_f1 = None
        test_metrics_rmse = None

    elapsed = time.time() - t0
    train_valsel_gap = float(best_train_macro_f1 - float(valsel_final['MacroF1']))
    fold_summary = {
        'method_name': 'HSTrans',
        'split_type': split_type,
        'fold': int(fold),
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
        method_name='HSTrans',
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
        thr_m = fold_summary.get('valsel_thr_metrics', {})
        thr_f1_str = f", thr_MacroF1={thr_m['MacroF1']:.4f}" if 'MacroF1' in thr_m else ""
        print(
            f"  --> HSTrans Fold {fold} Done ({elapsed:.1f}s) | Best Ep={best_f1_epoch} | "
            f"valsel MacroF1={valsel_final['MacroF1']:.4f}{thr_f1_str}, QWK={valsel_final['QWK']:.4f}, "
            f"MAE={valsel_final['MAE']:.4f}, RMSE={valsel_final['RMSE']:.4f} | "
            f"Gap(tr-val)={train_valsel_gap:+.4f}"
        )
    return fold_summary


def run_hstrans_experiment(
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
        'epochs': 200,
        'eval_every': 5,
        'batch_size': 128,
        'eval_batch_size': 256,
        'lr': 1e-4,
        'weight_decay': 0.01,
        'selection_metric': 'MacroF1',
        'eval_test': False,
        'device': None,
    }
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                cfg[k] = v

    eval_test = bool(cfg.get('eval_test', False))
    folds = get_fold_list(mode=mode, custom_folds=custom_folds)
    device = get_device(cfg.get('device'))
    TransCls = _import_hstrans_model()
    drug_tok_bundle = _pretokenize_all_750_drugs()

    if verbose:
        print(
            f"=== Running HSTrans | Split {split_type} | Mode={mode} (folds={folds}) | "
            f"Device={device} | Epochs={cfg['epochs']} | Resume={resume} ==="
        )

    fold_summaries = []
    for fold in folds:
        if resume:
            cached = try_restore_baseline_fold(
                'HSTrans', split_type, fold, results_dir,
                drive_dir=drive_dir, has_rmse_sel=True, eval_test=eval_test,
            )
            if cached is not None:
                if verbose:
                    vm = cached['valsel_metrics']
                    print(
                        f"  [RESUME] Skipping HSTrans_{split_type} Fold {fold} "
                        f"(valsel MacroF1={vm['MacroF1']:.4f}, RMSE={vm['RMSE']:.4f})"
                    )
                fold_summaries.append(cached)
                continue

        s = run_hstrans_fold(
            fold=fold,
            split_type=split_type,
            cfg=cfg,
            TransCls=TransCls,
            drug_tok_bundle=drug_tok_bundle,
            device=device,
            results_dir=results_dir,
            drive_dir=drive_dir,
            verbose=verbose,
        )
        fold_summaries.append(s)

    return finalize_baseline_experiment(
        method_name='HSTrans',
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
    parser = argparse.ArgumentParser(description='Run leak-free HSTrans under Protocol v2.')
    parser.add_argument('--split', type=str, default='B', choices=['A', 'B'])
    parser.add_argument('--mode', type=str, default='full', choices=['quick', 'dev', 'full'])
    parser.add_argument('--folds', type=int, nargs='*', default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--eval_every', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--wd', type=float, default=None)
    parser.add_argument('--eval_test', action='store_true')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_V2_DIR)
    parser.add_argument('--drive_dir', type=str, default=None)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    run_hstrans_experiment(
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
            'weight_decay': args.wd,
            'eval_test': bool(args.eval_test),
            'device': args.device,
        },
        verbose=True,
    )


if __name__ == '__main__':
    main()
