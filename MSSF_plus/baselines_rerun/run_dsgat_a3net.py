"""
Leak-free Protocol v2 wrapper for PyTorch Geometric baselines:
  - DSGAT (`DSGAT/Net.py` -> `GAT3`)
  - A3Net (`A-3Net-master/Net.py` -> `A3_Net`)

Fixes applied without modifying `DSGAT/` or `A-3Net-master/`:
  1. Uses unified 4-way split `(train, valsel, valcal, test)` via `load_fold_data`.
  2. Fixes critical Split B cold-start bug from old `notebook2_pyg.ipynb`:
     In Split B, `train_loader` contains ONLY the 540 warm-start `fold_split.train_drugs`
     (matching `DSGAT/ICS.py` L84 and `A-3Net-master/cold-scence.py` L84), instead of training
     masked cold-start drugs toward background `eps = 0.5`.
  3. Builds the SE k-NN graph per fold strictly from `drug_side_train[active_train_drugs].T`
     and extracts undirected bidirectional edges directly from `scipy.sparse` without calling
     deprecated `networkx.from_numpy_matrix` (compatible with NetworkX 3.x on modern Colab).
  4. Precomputes the 750 RDKit molecular graphs once in memory (with safe `(2, 0)` empty
     `edge_index` tensors for single-atom ions under PyG >= 2.3) instead of writing/deleting
     fragile `processed/*.pt` disk cache files.
  5. Selects primary checkpoint on `valsel` maximizing 5-class `MacroF1` (and secondary
     checkpoint minimizing `RMSE` saved to `{DSGAT|A3Net}_rmseSel_{A|B}`).
"""

import os
import copy
import time
import warnings
import argparse
import importlib.util
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io
from sklearn.neighbors import kneighbors_graph
import torch

warnings.filterwarnings(
    'ignore',
    message=".*scatter.*torch-scatter.*",
    category=UserWarning,
)

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
    eval_regression_predictions,
    try_restore_baseline_fold,
    save_baseline_fold_outputs,
    finalize_baseline_experiment,
)


def _load_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _import_pyg_and_models():
    try:
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
    except Exception:
        pass
    from torch_geometric.data import Data
    try:
        from torch_geometric.loader import DataLoader as PyGDataLoader
    except ImportError:
        from torch_geometric.data import DataLoader as PyGDataLoader

    dsgat_net = _load_module_from_path('dsgat_net_mod', os.path.join(REPO_DIR, 'DSGAT', 'Net.py'))
    a3net_net = _load_module_from_path('a3net_net_mod', os.path.join(REPO_DIR, 'A-3Net-master', 'Net.py'))
    s2v = _load_module_from_path('dsgat_s2v_mod', os.path.join(REPO_DIR, 'DSGAT', 'smiles2vector.py'))
    return Data, PyGDataLoader, dsgat_net.GAT3, a3net_net.A3_Net, s2v


def _precompute_molecular_graphs(s2v_mod) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Convert all 750 drug SMILES in `shared_data/drug_SMILES_750.csv` to 109-D atom features
    and `(2, E)` directed bond indices once in memory.
    """
    smiles_csv = os.path.join(SHARED_DIR, 'drug_SMILES_750.csv')
    _, drug_smiles = s2v_mod.load_drug_smile(smiles_csv)
    assert len(drug_smiles) == N_DRUGS, f"Expected {N_DRUGS} SMILES, got {len(drug_smiles)}"

    feat_list: List[np.ndarray] = []
    edge_list: List[np.ndarray] = []
    for smi in drug_smiles:
        _, features, edge_index, _ = s2v_mod.smile_to_graph(smi)
        feat_arr = np.asarray(features, dtype=np.float32)
        if len(edge_index) == 0:
            edge_arr = np.empty((2, 0), dtype=np.int64)
        else:
            edge_arr = np.asarray(edge_index, dtype=np.int64)
            if edge_arr.ndim != 2 or edge_arr.shape[0] != 2:
                edge_arr = edge_arr.reshape(2, -1)
        feat_list.append(feat_arr)
        edge_list.append(edge_arr)
    return feat_list, edge_list


def _build_se_knn_graph(
    DataCls,
    drug_side_train: np.ndarray,
    active_train_drugs: np.ndarray,
    se_node_label: np.ndarray,
    knn: int,
    metric: str = 'cosine',
):
    """
    Build the 994-node SE k-NN graph strictly from `drug_side_train[active_train_drugs].T`
    without `networkx.from_numpy_matrix`.
    """
    se_profiles = drug_side_train[active_train_drugs, :].T  # (994, N_active_drugs)
    A = kneighbors_graph(se_profiles, knn, mode='connectivity', metric=metric, include_self=False)
    A_sym = ((A + A.T) > 0).tocoo()
    edges = np.vstack([A_sym.row, A_sym.col]).astype(np.int64)
    return DataCls(
        x=torch.tensor(se_node_label, dtype=torch.float32),
        edge_index=torch.tensor(edges, dtype=torch.long),
    )


def _build_drug_pyg_dataset(
    DataCls,
    drug_indices: np.ndarray,
    feat_list: List[np.ndarray],
    edge_list: List[np.ndarray],
    target_mat: np.ndarray,
) -> list:
    data_list = []
    for d in drug_indices:
        d_int = int(d)
        g = DataCls(
            x=torch.tensor(feat_list[d_int], dtype=torch.float32),
            edge_index=torch.tensor(edge_list[d_int], dtype=torch.long),
            y=torch.tensor(target_mat[d_int:d_int + 1, :], dtype=torch.float32),
        )
        g.drug_id = int(d_int)
        data_list.append(g)
    return data_list


def _dsgat_loss(pred: torch.Tensor, target: torch.Tensor, lamb: float, eps: float) -> torch.Tensor:
    pos = target != 0
    zero = ~pos
    loss_pos = torch.sum((pred[pos] - target[pos]) ** 2) if pos.any() else pred.sum() * 0.0
    loss_zero = torch.sum((pred[zero] - eps) ** 2) if zero.any() else pred.sum() * 0.0
    return loss_pos + lamb * loss_zero


@torch.no_grad()
def _predict_pairs_pyg(
    model: torch.nn.Module,
    PyGDataLoaderCls,
    DataCls,
    pairs: np.ndarray,
    feat_list: List[np.ndarray],
    edge_list: List[np.ndarray],
    se_graph_dev,
    device: torch.device,
    df_flag: bool = False,
    not_fc_flag: bool = False,
    eval_batch: int = 32,
) -> np.ndarray:
    """Predict scores for specific `(drug_idx, se_idx)` pairs in exact `pairs` row order."""
    model.eval()
    unique_drugs = np.unique(pairs[:, 0].astype(np.int64))
    dummy_targets = np.zeros((N_DRUGS, N_SIDES), dtype=np.float32)
    eval_graphs = _build_drug_pyg_dataset(DataCls, unique_drugs, feat_list, edge_list, dummy_targets)
    loader = PyGDataLoaderCls(eval_graphs, batch_size=eval_batch, shuffle=False)

    pred_rows: Dict[int, np.ndarray] = {}
    for batch_data in loader:
        batch_data = batch_data.to(device)
        out, _, _ = model(batch_data, se_graph_dev, df_flag, not_fc_flag)
        out_np = out.detach().cpu().numpy().astype(np.float32)
        d_ids = batch_data.drug_id
        if torch.is_tensor(d_ids):
            d_ids_list = d_ids.detach().cpu().numpy().tolist()
        else:
            d_ids_list = [int(x) for x in d_ids]
        for row_idx, d_id in enumerate(d_ids_list):
            pred_rows[int(d_id)] = out_np[row_idx]

    preds = np.empty(len(pairs), dtype=np.float32)
    for k, row in enumerate(pairs):
        preds[k] = pred_rows[int(row[0])][int(row[1])]
    return preds


def run_pyg_baseline_fold(
    model_name: str,
    fold: int,
    split_type: str,
    cfg: dict,
    pyg_bundle: tuple,
    mol_graphs: Tuple[List[np.ndarray], List[np.ndarray]],
    se_node_label: np.ndarray,
    device: torch.device,
    results_dir: str = DEFAULT_RESULTS_V2_DIR,
    drive_dir: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    t0 = time.time()
    DataCls, PyGDataLoaderCls, GAT3Cls, A3NetCls, _ = pyg_bundle
    feat_list, edge_list = mol_graphs

    set_global_seed(int(cfg['seed']) + int(fold))
    fold_split = load_fold_data(fold, split_type)
    drug_side_train = make_train_only_freq(fold_split.train).astype(np.float32)

    if split_type == 'B':
        active_train_drugs = np.asarray(fold_split.train_drugs, dtype=np.int64)
    else:
        active_train_drugs = np.arange(N_DRUGS, dtype=np.int64)

    se_graph_dev = _build_se_knn_graph(
        DataCls,
        drug_side_train=drug_side_train,
        active_train_drugs=active_train_drugs,
        se_node_label=se_node_label,
        knn=int(cfg['knn']),
        metric=str(cfg['metric']),
    ).to(device)

    train_graphs = _build_drug_pyg_dataset(
        DataCls, active_train_drugs, feat_list, edge_list, drug_side_train
    )
    train_loader = PyGDataLoaderCls(
        train_graphs, batch_size=int(cfg['train_batch']), shuffle=True
    )

    ModelCls = GAT3Cls if model_name == 'DSGAT' else A3NetCls
    model = ModelCls(input_dim=109, output_dim=int(cfg['output_dim'])).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(cfg['lr']), weight_decay=float(cfg['weight_decay'])
    )

    epochs = int(cfg['epochs'])
    eval_every = max(1, int(cfg['eval_every']))
    lamb = float(cfg['lamb'])
    eps = float(cfg['eps'])
    df_flag = bool(cfg['DF'])
    not_fc_flag = bool(cfg['not_FC'])

    best_f1 = -float('inf')
    best_f1_epoch = 1
    best_train_macro_f1 = 0.0
    best_f1_state = copy.deepcopy(model.state_dict())

    best_rmse = float('inf')
    best_rmse_epoch = 1
    best_rmse_state = copy.deepcopy(model.state_dict())

    eval_test = bool(cfg.get('eval_test', False))
    train_true = fold_split.train[:, 2].astype(np.float32)
    val_true = fold_split.valsel[:, 2].astype(np.float32)
    test_true = fold_split.test[:, 2].astype(np.float32)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            out, _, _ = model(batch_data, se_graph_dev, df_flag, not_fc_flag)
            loss = _dsgat_loss(out.view(-1), batch_data.y.view(-1), lamb=lamb, eps=eps)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            n_batches += 1

        if epoch == 1 or epoch % eval_every == 0 or epoch == epochs:
            val_preds = _predict_pairs_pyg(
                model, PyGDataLoaderCls, DataCls, fold_split.valsel,
                feat_list, edge_list, se_graph_dev, device,
                df_flag=df_flag, not_fc_flag=not_fc_flag,
            )
            val_metrics = eval_regression_predictions(val_true, val_preds)

            if val_metrics['MacroF1'] > best_f1:
                best_f1 = val_metrics['MacroF1']
                best_f1_epoch = epoch
                best_f1_state = copy.deepcopy(model.state_dict())

            if val_metrics['RMSE'] < best_rmse:
                best_rmse = val_metrics['RMSE']
                best_rmse_epoch = epoch
                best_rmse_state = copy.deepcopy(model.state_dict())

            if verbose and (epoch == 1 or epoch % max(eval_every, 25) == 0 or epoch == epochs):
                print(
                    f"  [{model_name} Split {split_type} Fold {fold}] Ep {epoch:03d}/{epochs:03d} | "
                    f"Loss={running_loss / max(1, n_batches):.2f} | "
                    f"valsel MacroF1={val_metrics['MacroF1']:.4f} (best={best_f1:.4f}@{best_f1_epoch}) | "
                    f"RMSE={val_metrics['RMSE']:.4f} (best={best_rmse:.4f}@{best_rmse_epoch})"
                )

    # 1. Primary evaluation (MacroF1-selected checkpoint)
    model.load_state_dict(best_f1_state)
    train_p_f1 = _predict_pairs_pyg(
        model, PyGDataLoaderCls, DataCls, fold_split.train,
        feat_list, edge_list, se_graph_dev, device,
        df_flag=df_flag, not_fc_flag=not_fc_flag,
    )
    train_eval_metrics = eval_regression_predictions(train_true, train_p_f1)
    best_train_macro_f1 = float(train_eval_metrics['MacroF1'])

    val_p_f1 = _predict_pairs_pyg(
        model, PyGDataLoaderCls, DataCls, fold_split.valsel,
        feat_list, edge_list, se_graph_dev, device,
        df_flag=df_flag, not_fc_flag=not_fc_flag,
    )
    valsel_final = eval_regression_predictions(val_true, val_p_f1)

    if best_rmse_epoch == best_f1_epoch:
        val_p_rmse = val_p_f1.copy()
        valsel_metrics_rmse = dict(valsel_final)
    else:
        model.load_state_dict(best_rmse_state)
        val_p_rmse = _predict_pairs_pyg(
            model, PyGDataLoaderCls, DataCls, fold_split.valsel,
            feat_list, edge_list, se_graph_dev, device,
            df_flag=df_flag, not_fc_flag=not_fc_flag,
        )
        valsel_metrics_rmse = eval_regression_predictions(val_true, val_p_rmse)

    if eval_test:
        model.load_state_dict(best_f1_state)
        test_p_f1 = _predict_pairs_pyg(
            model, PyGDataLoaderCls, DataCls, fold_split.test,
            feat_list, edge_list, se_graph_dev, device,
            df_flag=df_flag, not_fc_flag=not_fc_flag,
        )
        test_metrics_f1 = eval_regression_predictions(test_true, test_p_f1)

        if best_rmse_epoch == best_f1_epoch:
            test_p_rmse = test_p_f1.copy()
            test_metrics_rmse = dict(test_metrics_f1)
        else:
            model.load_state_dict(best_rmse_state)
            test_p_rmse = _predict_pairs_pyg(
                model, PyGDataLoaderCls, DataCls, fold_split.test,
                feat_list, edge_list, se_graph_dev, device,
                df_flag=df_flag, not_fc_flag=not_fc_flag,
            )
            test_metrics_rmse = eval_regression_predictions(test_true, test_p_rmse)
    else:
        test_p_f1 = None
        test_p_rmse = None
        test_metrics_f1 = None
        test_metrics_rmse = None

    elapsed = time.time() - t0
    train_valsel_gap = float(best_train_macro_f1 - float(valsel_final['MacroF1']))
    fold_summary = {
        'method_name': model_name,
        'split_type': split_type,
        'fold': int(fold),
        'n_active_train_drugs': int(len(active_train_drugs)),
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
        method_name=model_name,
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
            f"  --> {model_name} Fold {fold} Done ({elapsed:.1f}s) | Best Ep={best_f1_epoch} | "
            f"valsel MacroF1={valsel_final['MacroF1']:.4f}{thr_f1_str}, QWK={valsel_final['QWK']:.4f}, "
            f"MAE={valsel_final['MAE']:.4f}, RMSE={valsel_final['RMSE']:.4f} | "
            f"Gap(tr-val)={train_valsel_gap:+.4f}"
        )
    return fold_summary


def run_pyg_baseline_experiment(
    model_name: str = 'DSGAT',
    split_type: str = 'B',
    mode: str = 'full',
    custom_folds: Optional[List[int]] = None,
    results_dir: str = DEFAULT_RESULTS_V2_DIR,
    drive_dir: Optional[str] = None,
    resume: bool = False,
    overrides: Optional[dict] = None,
    verbose: bool = True,
) -> dict:
    if model_name not in ('DSGAT', 'A3Net'):
        raise ValueError(f"Expected model_name in ('DSGAT', 'A3Net'), got {model_name!r}")

    default_knn = 5 if split_type == 'B' else 10
    default_wd = 0.001

    cfg = {
        'seed': 42,
        'epochs': 200,
        'eval_every': 5,
        'train_batch': 10,
        'lr': 1e-4,
        'weight_decay': default_wd,
        'lamb': 0.03,
        'eps': 0.5,
        'knn': default_knn,
        'metric': 'cosine',
        'output_dim': 200,
        'DF': False,
        'not_FC': False,
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

    pyg_bundle = _import_pyg_and_models()
    mol_graphs = _precompute_molecular_graphs(pyg_bundle[4])
    se_mat_path = os.path.join(SHARED_DIR, 'side_effect_label_750.mat')
    se_node_label = np.asarray(scipy.io.loadmat(se_mat_path)['node_label'], dtype=np.float32)

    if verbose:
        print(
            f"=== Running {model_name} | Split {split_type} | Mode={mode} (folds={folds}) | "
            f"Device={device} | Epochs={cfg['epochs']} | KNN={cfg['knn']} | Resume={resume} ==="
        )

    fold_summaries = []
    for fold in folds:
        if resume:
            cached = try_restore_baseline_fold(
                model_name, split_type, fold, results_dir,
                drive_dir=drive_dir, has_rmse_sel=True, eval_test=eval_test,
            )
            if cached is not None:
                if verbose:
                    vm = cached['valsel_metrics']
                    print(
                        f"  [RESUME] Skipping {model_name}_{split_type} Fold {fold} "
                        f"(valsel MacroF1={vm['MacroF1']:.4f}, RMSE={vm['RMSE']:.4f})"
                    )
                fold_summaries.append(cached)
                continue

        s = run_pyg_baseline_fold(
            model_name=model_name,
            fold=fold,
            split_type=split_type,
            cfg=cfg,
            pyg_bundle=pyg_bundle,
            mol_graphs=mol_graphs,
            se_node_label=se_node_label,
            device=device,
            results_dir=results_dir,
            drive_dir=drive_dir,
            verbose=verbose,
        )
        fold_summaries.append(s)

    return finalize_baseline_experiment(
        method_name=model_name,
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
    parser = argparse.ArgumentParser(description='Run leak-free DSGAT or A3Net under Protocol v2.')
    parser.add_argument('--model', type=str, required=True, choices=['DSGAT', 'A3Net'])
    parser.add_argument('--split', type=str, default='B', choices=['A', 'B'])
    parser.add_argument('--mode', type=str, default='full', choices=['quick', 'dev', 'full'])
    parser.add_argument('--folds', type=int, nargs='*', default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--eval_every', type=int, default=None)
    parser.add_argument('--train_batch', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--wd', type=float, default=None)
    parser.add_argument('--knn', type=int, default=None)
    parser.add_argument('--eval_test', action='store_true')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_V2_DIR)
    parser.add_argument('--drive_dir', type=str, default=None)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    run_pyg_baseline_experiment(
        model_name=args.model,
        split_type=args.split,
        mode=args.mode,
        custom_folds=args.folds,
        results_dir=args.results_dir,
        drive_dir=args.drive_dir,
        resume=args.resume,
        overrides={
            'epochs': args.epochs,
            'eval_every': args.eval_every,
            'train_batch': args.train_batch,
            'lr': args.lr,
            'weight_decay': args.wd,
            'knn': args.knn,
            'eval_test': bool(args.eval_test),
            'device': args.device,
        },
        verbose=True,
    )


if __name__ == '__main__':
    main()
