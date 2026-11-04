"""
Leak-free Protocol v2 wrapper for MVSL-DSF (`MVSL-DSF/Network.py`).

Fixes applied without modifying `MVSL-DSF/`:
  1. Uses unified 4-way split `(train, valsel, valcal, test)` via `load_fold_data`.
  2. Constructs Views 4-6 (`assoc_train`, `freq_train = raw / 5.0`, `assoc_train.T`) strictly
     from `make_train_only_freq(fold_split.train)` so all `valsel`, `valcal`, and `test` pairs
     (and all cold-start drug rows/columns in Split B) are strictly zeroed out.
  3. Uses complete 994 x 243 MedDRA ontology vectors from `shared_data/side_effect_label_750.mat`
     by default (`--se_source mat`, avoiding the 110 mean-imputed SEs in `mvsl_750_994/side_vector_level_123.pkl`,
     with `--se_source mvsl_pkl` available for ablation/reproduction).
  4. Uses leak-free 1:1 negative sampling via `sample_negative_pairs_v2` (restricting Split B
     training negatives strictly to `fold_split.train_drugs`).
  5. Selects primary checkpoint on `valsel` maximizing 5-class `MacroF1` (and secondary
     checkpoint minimizing `RMSE` saved to `MVSL-DSF_rmseSel_{A|B}`).
"""

import os
import copy
import time
import pickle
import argparse
import importlib.util
from typing import List, Optional, Tuple

import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
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

SMILES_CHARS = [
    '?', '#', '%', ')', '(', '+', '-', '.', '1', '0', '3', '2', '5', '4',
    '7', '6', '9', '8', '=', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I',
    'H', 'K', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V',
    'Y', '[', 'Z', ']', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
    'h', 'm', 'l', 'o', 'n', 's', 'r', 'u', 't', 'y',
]
CHAR_TO_IDX = {c: idx for idx, c in enumerate(SMILES_CHARS)}
MAX_SEQ = 100
CHAR_DIM = 63

BASE_CONFIG_TEXTCNN = {
    'dropout_rate': 0.3,
    'embedding_size': 64,
    'feature_size': 64,
    'max_text_len': MAX_SEQ,
    'window_sizes': [1],
}


def _import_mvsl_network():
    from MSSF_plus.baselines_rerun.dgl_shim import install_dgl_shim
    install_dgl_shim()
    net_path = os.path.join(REPO_DIR, 'MVSL-DSF', 'Network.py')
    spec = importlib.util.spec_from_file_location('mvsl_network_mod', net_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load MVSL-DSF Network.py from {net_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ConvNCF


def _precompute_mvsl_static_features(se_source: str = 'mat') -> Tuple[np.ndarray, np.ndarray, list, np.ndarray]:
    """Precompute 750 SMILES one-hots, 2048-bit Morgan FPs, DGL AttentiveFP graphs, and 994x243 SE vectors."""
    from MSSF_plus.baselines_rerun.dgl_shim import install_dgl_shim
    install_dgl_shim()
    try:
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
    except Exception:
        pass
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import dgl
    from dgllife.utils import smiles_to_bigraph, AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer

    mvsl_data = os.path.join(SHARED_DIR, 'mvsl_750_994')
    with open(os.path.join(mvsl_data, 'drug_smiles.pkl'), 'rb') as f:
        drug_smiles_data = pickle.load(f)

    atom_featurizer = AttentiveFPAtomFeaturizer()
    bond_featurizer = AttentiveFPBondFeaturizer(self_loop=True)

    onehot_list, fp_list, graph_list = [], [], []
    for item in sorted(drug_smiles_data, key=lambda x: x[0]):
        smi = item[1]
        padded = smi[:MAX_SEQ].ljust(MAX_SEQ, '?')
        oh = np.zeros((MAX_SEQ, CHAR_DIM), dtype=np.float32)
        for pos, ch in enumerate(padded):
            c_idx = CHAR_TO_IDX.get(ch, -1)
            if c_idx >= 0:
                oh[pos, c_idx] = 1.0
        onehot_list.append(oh)

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fp = np.zeros(2048, dtype=np.float32)
        else:
            fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), dtype=np.float32)
        fp_list.append(fp)

        g = smiles_to_bigraph(
            smi, add_self_loop=True,
            node_featurizer=atom_featurizer,
            edge_featurizer=bond_featurizer,
        )
        if g is None:
            g = dgl.graph(([0], [0]))
            g.ndata['h'] = torch.zeros(1, atom_featurizer.feat_size())
            g.edata['e'] = torch.zeros(1, bond_featurizer.feat_size())
        graph_list.append(g)

    if se_source == 'mat':
        se_mat_path = os.path.join(SHARED_DIR, 'side_effect_label_750.mat')
        side_vec = np.asarray(scipy.io.loadmat(se_mat_path)['node_label'], dtype=np.float32)
    else:
        with open(os.path.join(mvsl_data, 'side_vector_level_123.pkl'), 'rb') as f:
            side_vec = np.asarray(pickle.load(f), dtype=np.float32)

    assert len(graph_list) == N_DRUGS
    assert side_vec.shape == (N_SIDES, 243)
    return np.asarray(onehot_list, dtype=np.float32), np.asarray(fp_list, dtype=np.float32), graph_list, side_vec


class _MVSLPairDataset(Dataset):
    def __init__(self, pairs_with_rating: np.ndarray):
        self.d = pairs_with_rating[:, 0].astype(np.int64)
        self.s = pairs_with_rating[:, 1].astype(np.int64)
        self.r = pairs_with_rating[:, 2].astype(np.float32)

    def __len__(self) -> int:
        return len(self.d)

    def __getitem__(self, idx: int):
        r_val = float(self.r[idx])
        return int(self.d[idx]), int(self.s[idx]), (1 if r_val > 0 else 0), r_val


def _make_mvsl_loader(
    pairs_with_rating: np.ndarray,
    batch_size: int,
    shuffle: bool,
    static_bundle: Tuple[np.ndarray, np.ndarray, list, np.ndarray],
    assoc_train: np.ndarray,
    freq_train: np.ndarray,
    self_corr_in_collate: bool = False,
) -> DataLoader:
    onehots_arr, fps_arr, graphs_list, side_vec_arr = static_bundle
    ds = _MVSLPairDataset(pairs_with_rating)
    drop_last = bool(shuffle and (len(ds) > batch_size) and (len(ds) % batch_size == 1))

    def _collate(batch):
        d_idx = [b[0] for b in batch]
        s_idx = [b[1] for b in batch]
        a_lbl = torch.tensor([b[2] for b in batch], dtype=torch.long)
        f_lbl = torch.tensor([b[3] / 5.0 for b in batch], dtype=torch.float32)
        f_raw = np.array([b[3] for b in batch], dtype=np.float32)

        assoc_np = assoc_train[d_idx, :].copy()
        freq_np = freq_train[d_idx, :].copy()
        assoc_t_np = assoc_train[:, s_idx].T.copy()

        if self_corr_in_collate:
            row_idx = np.arange(len(batch))
            assoc_np[row_idx, s_idx] = 0.0
            freq_np[row_idx, s_idx] = 0.0
            assoc_t_np[row_idx, d_idx] = 0.0

        assoc_prof = torch.tensor(assoc_np, dtype=torch.float32)
        freq_prof = torch.tensor(freq_np, dtype=torch.float32)
        assoc_prof_t = torch.tensor(assoc_t_np, dtype=torch.float32)

        # Clone graphs because MVSL-DSF's DGL_AttentiveFP.forward pops 'h' and 'e' from bg.ndata/edata!
        graphs = [graphs_list[i].clone() for i in d_idx]
        onehots = torch.tensor(onehots_arr[d_idx], dtype=torch.float32)
        fps = torch.tensor(fps_arr[d_idx], dtype=torch.float32)
        side_vec = torch.tensor(side_vec_arr[s_idx], dtype=torch.float32)
        return a_lbl, f_lbl, f_raw, assoc_prof, freq_prof, assoc_prof_t, graphs, onehots, fps, side_vec

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=_collate)


@torch.no_grad()
def _predict_mvsl(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `(y_true_raw_freq, pred_binary_prob, pred_freq_1_to_5)`."""
    model.eval()
    raw_all, cls_all, freq_all = [], [], []
    for batch in loader:
        _, _, f_raw, assoc_prof, freq_prof, assoc_prof_t, graphs, onehots, fps, side_vec = batch
        cls_out, freq_out = model(
            graphs, onehots, fps, None, side_vec, None,
            [assoc_prof, freq_prof], [assoc_prof_t], device,
        )
        p_cls = torch.softmax(cls_out, dim=1)[:, 1].detach().cpu().numpy().astype(np.float32)
        p_freq = (torch.sigmoid(freq_out).view(-1).detach().cpu().numpy() * 5.0).astype(np.float32)
        raw_all.append(f_raw)
        cls_all.append(np.atleast_1d(p_cls))
        freq_all.append(np.atleast_1d(p_freq))
    return np.concatenate(raw_all), np.concatenate(cls_all), np.concatenate(freq_all)


def run_mvsl_fold(
    fold: int,
    split_type: str,
    cfg: dict,
    MVSLConvNCF,
    static_bundle: Tuple[np.ndarray, np.ndarray, list, np.ndarray],
    device: torch.device,
    results_dir: str = DEFAULT_RESULTS_V2_DIR,
    drive_dir: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    t0 = time.time()
    fold_seed = int(cfg['seed']) + int(fold)
    set_global_seed(fold_seed)
    fold_split = load_fold_data(fold, split_type)

    use_self_corr = bool(cfg.get('profile_self_correction', False))
    eval_test = bool(cfg.get('eval_test', False))
    method_name = 'MVSL-DSF_selfcorr' if use_self_corr else 'MVSL-DSF'

    raw_train_freq = make_train_only_freq(fold_split.train).astype(np.float32)
    freq_train = (raw_train_freq / 5.0).astype(np.float32)
    assoc_train = (raw_train_freq > 0).astype(np.float32)

    neg_train_xy = sample_negative_pairs_v2(
        raw_train_freq, fold_split, n_samples=len(fold_split.train), part='train', seed=fold_seed
    )
    neg_train = np.column_stack([neg_train_xy, np.zeros(len(neg_train_xy), dtype=np.float32)])
    train_combined = np.vstack([fold_split.train.astype(np.float32), neg_train])

    neg_test_xy = sample_negative_pairs_v2(
        raw_train_freq, fold_split, n_samples=len(fold_split.test), part='test', seed=fold_seed + 100
    )
    neg_test = np.column_stack([neg_test_xy, np.zeros(len(neg_test_xy), dtype=np.float32)])

    batch_size = int(cfg['batch_size'])
    eval_bs = int(cfg['eval_batch_size'])
    epochs = int(cfg['epochs'])

    train_loader = _make_mvsl_loader(
        train_combined, batch_size, True, static_bundle, assoc_train, freq_train,
        self_corr_in_collate=use_self_corr,
    )
    valsel_loader = _make_mvsl_loader(fold_split.valsel, eval_bs, False, static_bundle, assoc_train, freq_train)
    test_pos_loader = _make_mvsl_loader(fold_split.test, eval_bs, False, static_bundle, assoc_train, freq_train)
    test_neg_loader = _make_mvsl_loader(neg_test, eval_bs, False, static_bundle, assoc_train, freq_train)

    model = MVSLConvNCF(
        ad_dim=[N_SIDES, N_DRUGS],
        drug_embed_dim=2048,
        side_embed_dim=[243],
        hid_embed_dim=int(cfg['hid_dim']),
        dropout=float(cfg['dropout']),
        **BASE_CONFIG_TEXTCNN,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg['lr']))
    criterion_cls = nn.CrossEntropyLoss()
    criterion_freq = nn.BCEWithLogitsLoss()

    best_f1 = -float('inf')
    best_f1_epoch = 1
    best_train_macro_f1 = 0.0
    best_f1_state = copy.deepcopy(model.state_dict())

    best_rmse = float('inf')
    best_rmse_epoch = 1
    best_rmse_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        pos_tr_true, pos_tr_pred = [], []

        for batch in train_loader:
            a_lbl, f_lbl, f_raw, assoc_prof, freq_prof, assoc_prof_t, graphs, onehots, fps, side_vec = batch
            a_lbl_dev = a_lbl.to(device)
            f_lbl_dev = f_lbl.to(device)

            optimizer.zero_grad()
            cls_out, freq_out = model(
                graphs, onehots, fps, None, side_vec, None,
                [assoc_prof, freq_prof], [assoc_prof_t], device,
            )
            freq_out = freq_out.view(-1)
            loss_cls = criterion_cls(cls_out, a_lbl_dev)
            pos_mask = a_lbl_dev == 1
            if pos_mask.any():
                loss_freq = criterion_freq(freq_out[pos_mask], f_lbl_dev[pos_mask])
                pos_mask_np = pos_mask.detach().cpu().numpy()
                p_freq_pos = (torch.sigmoid(freq_out[pos_mask]).detach().cpu().numpy() * 5.0).astype(np.float32)
                pos_tr_true.append(f_raw[pos_mask_np])
                pos_tr_pred.append(p_freq_pos)
            else:
                loss_freq = freq_out.sum() * 0.0
            loss = loss_cls + loss_freq + model.consistency_loss
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        tr_ep_metrics = eval_regression_predictions(
            np.concatenate(pos_tr_true), np.concatenate(pos_tr_pred)
        )
        tr_ep_f1 = float(tr_ep_metrics['MacroF1'])

        val_y, _, val_p = _predict_mvsl(model, valsel_loader, device)
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

        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            print(
                f"  [{method_name} Split {split_type} Fold {fold}] Ep {epoch:02d}/{epochs:02d} | "
                f"Loss={running_loss / max(1, n_batches):.4f} | "
                f"train MacroF1={tr_ep_f1:.4f} | "
                f"valsel MacroF1={val_metrics['MacroF1']:.4f} (best={best_f1:.4f}@{best_f1_epoch}) | "
                f"RMSE={val_metrics['RMSE']:.4f} (best={best_rmse:.4f}@{best_rmse_epoch})"
            )

    # 1. Primary evaluation (MacroF1-selected checkpoint)
    model.load_state_dict(best_f1_state)
    val_y, _, val_p_f1 = _predict_mvsl(model, valsel_loader, device)
    assert np.array_equal(np.round(val_y), np.round(fold_split.valsel[:, 2]))
    valsel_final = eval_regression_predictions(val_y, val_p_f1)

    if best_rmse_epoch == best_f1_epoch:
        val_p_rmse = val_p_f1.copy()
        valsel_metrics_rmse = dict(valsel_final)
    else:
        model.load_state_dict(best_rmse_state)
        _, _, val_p_rmse = _predict_mvsl(model, valsel_loader, device)
        valsel_metrics_rmse = eval_regression_predictions(val_y, val_p_rmse)

    if eval_test:
        model.load_state_dict(best_f1_state)
        test_y, test_cls_pos, test_p_f1 = _predict_mvsl(model, test_pos_loader, device)
        _, test_cls_neg, _ = _predict_mvsl(model, test_neg_loader, device)
        assert np.array_equal(np.round(test_y), np.round(fold_split.test[:, 2]))

        bin_y = np.concatenate([np.ones(len(test_cls_pos)), np.zeros(len(test_cls_neg))])
        bin_s = np.concatenate([test_cls_pos, test_cls_neg])
        bin_auc = float(roc_auc_score(bin_y, bin_s))
        bin_aupr = float(average_precision_score(bin_y, bin_s))

        test_metrics_f1 = eval_regression_predictions(test_y, test_p_f1)
        test_metrics_f1['BinaryAUC'] = bin_auc
        test_metrics_f1['BinaryAUPR'] = bin_aupr

        model.load_state_dict(best_rmse_state)
        _, _, test_p_rmse = _predict_mvsl(model, test_pos_loader, device)
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
        'se_source': str(cfg['se_source']),
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


def run_mvsl_experiment(
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
        'epochs': 50,
        'batch_size': 128,
        'eval_batch_size': 256,
        'lr': 1e-4,
        'dropout': 0.3,
        'hid_dim': 64,
        'se_source': 'mat',  # 'mat' = clean 994x243 MedDRA ontology; 'mvsl_pkl' = 893->994 mapped
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
    method_name = 'MVSL-DSF_selfcorr' if use_self_corr else 'MVSL-DSF'

    folds = get_fold_list(mode=mode, custom_folds=custom_folds)
    device = get_device(cfg.get('device'))
    MVSLConvNCF = _import_mvsl_network()
    static_bundle = _precompute_mvsl_static_features(se_source=str(cfg['se_source']))

    if verbose:
        print(
            f"=== Running {method_name} | Split {split_type} | Mode={mode} (folds={folds}) | "
            f"Device={device} | Epochs={cfg['epochs']} | SE={cfg['se_source']} | "
            f"SelfCorr={use_self_corr} | Resume={resume} ==="
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

        s = run_mvsl_fold(
            fold=fold,
            split_type=split_type,
            cfg=cfg,
            MVSLConvNCF=MVSLConvNCF,
            static_bundle=static_bundle,
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
    parser = argparse.ArgumentParser(description='Run leak-free MVSL-DSF under Protocol v2.')
    parser.add_argument('--split', type=str, default='B', choices=['A', 'B'])
    parser.add_argument('--mode', type=str, default='full', choices=['quick', 'dev', 'full'])
    parser.add_argument('--folds', type=int, nargs='*', default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--se_source', type=str, default=None, choices=['mat', 'mvsl_pkl'])
    parser.add_argument('--profile_self_correction', type=int, default=None, choices=[0, 1])
    parser.add_argument('--eval_test', action='store_true')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_V2_DIR)
    parser.add_argument('--drive_dir', type=str, default=None)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    run_mvsl_experiment(
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
            'se_source': args.se_source,
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
