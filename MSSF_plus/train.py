"""
Unified training, validation checkpoint selection (`val_select`), post-hoc C4 conformal
calibration (`val_calib`), and final test evaluation (`test`) for MSSF_plus.

Strict Protocol Invariants:
  1. Test set is NEVER evaluated during the training loop; it is evaluated exactly once
     after restoring the checkpoint that maximized `MacroF1` on `val_select`.
  2. All profile similarity views (D9, D10, S3, S4) and C1/C5 auxiliary statistics are
     computed strictly from the 80% training split of the current fold.
  3. All evaluation DataLoaders enforce `shuffle=False` and assert exact 1-to-1 label
     order alignment with the ground-truth split array.
"""

import os
import time
import copy
import json
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from MSSF_plus.protocol import (
    DEFAULT_RESULTS_V2_DIR,
    set_global_seed,
    get_device,
    get_fold_list,
    load_fold_data,
    save_fold_predictions,
    save_experiment_config,
)
from MSSF_plus.metrics import compute_multiclass_metrics, aggregate_fold_metrics
from MSSF_plus.views import build_fold_views
from MSSF_plus.soft_labels import build_soft_targets
from MSSF_plus.data import build_dataloaders
from MSSF_plus.model import MSSFPlusModel
from MSSF_plus.losses import (
    compute_class_weights_from_train,
    logits_to_probs,
    MSSFPlusCriterion,
)
from MSSF_plus.conformal import evaluate_all_conformal_methods
from MSSF_plus.configs.experiments import get_experiment_config, EXPERIMENTS


@torch.no_grad()
def evaluate_loader(
    model: MSSFPlusModel,
    loader: torch.utils.data.DataLoader,
    loss_type: str,
    device: torch.device,
    expected_pairs: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on a DataLoader (`shuffle=False`) and return:
      (metrics_dict, y_true_class_0_to_4, y_pred_class_0_to_4, probs_5class)
    """
    model.eval()
    all_probs = []
    all_labels = []
    all_freqs = []

    for batch in loader:
        drugs = batch['drug_feat'].to(device)
        sides = batch['se_feat'].to(device)
        d_avail = batch['drug_avail'].to(device)
        s_avail = batch['se_avail'].to(device)
        fb_d9 = batch['fallback_d9'].to(device)
        fb_d10 = batch['fallback_d10'].to(device)

        logits, _, _, _, _, _, _, _ = model(
            drugs, sides, drug_avail=d_avail, se_avail=s_avail,
            fallback_d9=fb_d9, fallback_d10=fb_d10
        )
        probs = logits_to_probs(logits, loss_type=loss_type)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(batch['label'].numpy())
        all_freqs.append(batch['freq'].numpy())

    probs_arr = np.vstack(all_probs).astype(np.float32)
    y_true_class = np.concatenate(all_labels).astype(np.int64)
    freqs_arr = np.concatenate(all_freqs).astype(np.float32)
    y_pred_class = np.argmax(probs_arr, axis=1).astype(np.int64)

    if expected_pairs is not None:
        exp_y = np.round(expected_pairs[:, 2]).astype(np.int64)
        got_y = np.round(freqs_arr).astype(np.int64)
        assert np.array_equal(got_y, exp_y), (
            "CRITICAL: Evaluation DataLoader sample order does not match ground-truth split!"
        )

    metrics = compute_multiclass_metrics(y_true_class, y_pred_class, probs=probs_arr)
    return metrics, y_true_class, y_pred_class, probs_arr


def run_single_fold(
    fold: int,
    split_type: str,
    cfg: dict,
    device: torch.device,
    results_dir: str = DEFAULT_RESULTS_V2_DIR,
    verbose: bool = True,
) -> dict:
    """
    Train, select best checkpoint on `val_select`, calibrate C4 on `val_calib`,
    and evaluate once on `test` for a single fold.
    """
    t0 = time.time()
    base_seed = int(cfg.get('seed', 42))
    fold_seed = base_seed + int(fold)
    set_global_seed(fold_seed)

    exp_name = str(cfg.get('exp_name', 'MSSF_plus'))
    loss_type = str(cfg.get('c3_loss_type', 'ce'))
    c5_mode = str(cfg.get('c5_mode', 'none'))
    selection_metric = str(cfg.get('selection_metric', 'MacroF1'))
    epochs = int(cfg.get('epochs', 50))

    # 1. Load 4-way split
    fold_split = load_fold_data(fold, split_type)

    # 2. Construct leak-free views (including C1 imputation & C2 ontology views)
    fold_views = build_fold_views(fold_split, cfg, device=device)

    # 3. Construct C5 soft labels on training pairs only
    soft_res = build_soft_targets(fold_split, fold_views, cfg)

    # 4. Build DataLoaders
    train_loader, valsel_loader, valcal_loader, test_loader = build_dataloaders(
        fold_split, fold_views, soft_res, cfg
    )

    # 5. Initialize model, loss criterion, and optimizer
    n_drug_views = len(fold_views.drug_view_names)
    n_se_views = len(fold_views.se_view_names)

    model = MSSFPlusModel(
        cfg=cfg,
        n_drug_views=n_drug_views,
        n_se_views=n_se_views,
        profile_drug_view_indices=fold_views.profile_drug_view_indices,
        n_clusters=soft_res.n_clusters,
    ).to(device)

    class_weights = compute_class_weights_from_train(fold_split.train)
    criterion = MSSFPlusCriterion(
        cfg=cfg,
        n_drug_views=n_drug_views,
        n_se_views=n_se_views,
        class_weights=class_weights,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.get('lr', 0.001)),
        weight_decay=float(cfg.get('weight_decay', 1e-6)),
    )

    best_epoch = 1
    best_valsel_score = -float('inf')
    best_state_dict = copy.deepcopy(model.state_dict())

    use_soft_targets = c5_mode != 'none'

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            drugs = batch['drug_feat'].to(device)
            sides = batch['se_feat'].to(device)
            d_avail = batch['drug_avail'].to(device)
            s_avail = batch['se_avail'].to(device)
            fb_d9 = batch['fallback_d9'].to(device)
            fb_d10 = batch['fallback_d10'].to(device)
            hard_labels = batch['label'].to(device)
            soft_targets = batch['soft_target'].to(device) if use_soft_targets else None
            cluster_ids = batch['cluster_id'].to(device)

            optimizer.zero_grad()
            logits, rec_con, rec_add, mu, logvar, mod_drugs, mod_sides, cluster_logits = model(
                drugs, sides,
                drug_avail=d_avail, se_avail=s_avail,
                fallback_d9=fb_d9, fallback_d10=fb_d10,
            )

            loss = criterion(
                logits=logits,
                rec_con=rec_con,
                rec_add=rec_add,
                mu=mu,
                logvar=logvar,
                batch_drug=mod_drugs,
                batch_side=mod_sides,
                hard_labels=hard_labels,
                soft_targets=soft_targets,
                cluster_logits=cluster_logits,
                cluster_ids=cluster_ids,
                epoch=epoch,
            )
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        # Evaluate ONLY on val_select (10%) for checkpoint selection
        valsel_metrics, _, _, _ = evaluate_loader(
            model, valsel_loader, loss_type=loss_type, device=device,
            expected_pairs=fold_split.valsel
        )
        score = float(valsel_metrics[selection_metric])
        if score > best_valsel_score:
            best_valsel_score = score
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            avg_loss = running_loss / max(1, n_batches)
            print(
                f"  [{exp_name} Split {split_type} Fold {fold}] "
                f"Epoch {epoch:02d}/{epochs:02d} | Loss={avg_loss:.4f} | "
                f"valsel {selection_metric}={score:.4f} (best={best_valsel_score:.4f} @ ep {best_epoch})"
            )

    # Restore best checkpoint selected on val_select
    model.load_state_dict(best_state_dict)

    # Final evaluation on val_select, val_calib, and test (once!)
    valsel_final, valsel_true_c, _, valsel_probs = evaluate_loader(
        model, valsel_loader, loss_type=loss_type, device=device,
        expected_pairs=fold_split.valsel
    )
    _, valcal_true_c, _, valcal_probs = evaluate_loader(
        model, valcal_loader, loss_type=loss_type, device=device,
        expected_pairs=fold_split.valcal
    )
    test_metrics, test_true_c, test_pred_c, test_probs = evaluate_loader(
        model, test_loader, loss_type=loss_type, device=device,
        expected_pairs=fold_split.test
    )

    # C4: Post-hoc ordinal conformal prediction calibrated on val_calib -> evaluated on test
    c4_alpha = float(cfg.get('c4_alpha', 0.10))
    conformal_res = evaluate_all_conformal_methods(
        valcal_probs=valcal_probs,
        valcal_labels_1_to_5=valcal_true_c + 1,
        test_probs=test_probs,
        test_labels_1_to_5=test_true_c + 1,
        alpha=c4_alpha,
        seed=fold_seed,
    )

    # Save fold predictions in unified_eval.py format
    method_dir = os.path.join(results_dir, f'{exp_name}_{split_type}')
    save_fold_predictions(
        method_dir=method_dir,
        fold=fold,
        test_split=fold_split.test,
        y_true_class=test_true_c,
        y_pred_class=test_pred_c,
        probs_5class=test_probs,
        valcal_true_class=valcal_true_c,
        valcal_probs_5class=valcal_probs,
        valsel_true_class=valsel_true_c,
        valsel_probs_5class=valsel_probs,
    )

    elapsed = time.time() - t0
    fold_summary = {
        'exp_name': exp_name,
        'split_type': split_type,
        'fold': int(fold),
        'best_epoch': int(best_epoch),
        'best_valsel_score': float(best_valsel_score),
        'valsel_metrics': valsel_final,
        'test_metrics': test_metrics,
        'conformal': conformal_res,
        'c5_diagnostics': soft_res.diagnostics,
        'n_drug_views': int(n_drug_views),
        'n_se_views': int(n_se_views),
        'elapsed_seconds': float(elapsed),
    }

    with open(os.path.join(method_dir, f'fold_{fold}_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(fold_summary, f, indent=2)

    if verbose:
        ord_aps = conformal_res['Ordinal_APS']
        print(
            f"  --> Fold {fold} Done ({elapsed:.1f}s) | Best Epoch={best_epoch} | "
            f"Test MacroF1={test_metrics['MacroF1']:.4f}, QWK={test_metrics['QWK']:.4f}, "
            f"MAE={test_metrics['MAE']:.4f}, Acc={test_metrics['Accuracy']:.4f} | "
            f"Ord-APS Cov={ord_aps['empirical_coverage']:.3f}, Size={ord_aps['avg_set_size']:.2f}"
        )

    return fold_summary


def _sync_method_dir(src_dir: str, drive_dir: Optional[str], method_folder: str):
    """Mirror files from local method_dir to Google Drive if drive_dir is configured."""
    if not drive_dir:
        return
    import shutil
    dst_dir = os.path.join(drive_dir, method_folder)
    os.makedirs(dst_dir, exist_ok=True)
    if not os.path.isdir(src_dir):
        return
    for fname in os.listdir(src_dir):
        s_path = os.path.join(src_dir, fname)
        d_path = os.path.join(dst_dir, fname)
        if os.path.isfile(s_path) and os.path.abspath(s_path) != os.path.abspath(d_path):
            shutil.copy2(s_path, d_path)


def _try_restore_completed_fold(
    method_dir: str,
    drive_dir: Optional[str],
    method_folder: str,
    fold: int,
) -> Optional[dict]:
    """
    If fold outputs (`labels.npy`, `preds.npy`, `probs.npy`, `summary.json`) already exist
    locally or in `drive_dir`, restore them locally and return the loaded fold_summary.
    """
    import shutil
    req_files = [
        f'fold_{fold}_labels.npy',
        f'fold_{fold}_preds.npy',
        f'fold_{fold}_probs.npy',
        f'fold_{fold}_summary.json',
    ]
    opt_files = [
        f'fold_{fold}_valcal_labels.npy',
        f'fold_{fold}_valcal_probs.npy',
        f'fold_{fold}_valsel_labels.npy',
        f'fold_{fold}_valsel_probs.npy',
    ]
    if drive_dir:
        drive_method_dir = os.path.join(drive_dir, method_folder)
        if os.path.isdir(drive_method_dir):
            os.makedirs(method_dir, exist_ok=True)
            for fname in req_files + opt_files:
                local_p = os.path.join(method_dir, fname)
                drive_p = os.path.join(drive_method_dir, fname)
                if not os.path.exists(local_p) and os.path.exists(drive_p):
                    shutil.copy2(drive_p, local_p)

    if all(os.path.exists(os.path.join(method_dir, f)) for f in req_files):
        with open(os.path.join(method_dir, f'fold_{fold}_summary.json'), 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def run_experiment(
    exp_name: str,
    split_type: str = 'B',
    mode: str = 'dev',
    custom_folds: Optional[List[int]] = None,
    results_dir: str = DEFAULT_RESULTS_V2_DIR,
    drive_dir: Optional[str] = None,
    resume: bool = False,
    cli_overrides: Optional[dict] = None,
    verbose: bool = True,
) -> dict:
    """
    Run an experiment across the requested folds ('quick': [0], 'dev': [0,1,2], 'full': [0..9]),
    aggregate classification and conformal metrics, and save provenance files.
    """
    overrides = cli_overrides or {}
    cfg = get_experiment_config(exp_name, **overrides)
    folds = get_fold_list(mode=mode, custom_folds=custom_folds)
    device = get_device(cfg.get('device', None))
    method_folder = f'{exp_name}_{split_type}'
    method_dir = os.path.join(results_dir, method_folder)

    if verbose:
        print(
            f"=== Running {exp_name} | Split {split_type} | Mode={mode} (folds={folds}) | "
            f"Device={device} | Epochs={cfg['epochs']} | Resume={resume} ==="
        )

    fold_summaries = []
    for fold in folds:
        if resume:
            cached = _try_restore_completed_fold(method_dir, drive_dir, method_folder, fold)
            if cached is not None:
                if verbose:
                    tm = cached['test_metrics']
                    print(
                        f"  [RESUME] Skipping {method_folder} Fold {fold} (already completed: "
                        f"MacroF1={tm['MacroF1']:.4f}, QWK={tm['QWK']:.4f})"
                    )
                fold_summaries.append(cached)
                _sync_method_dir(method_dir, drive_dir, method_folder)
                continue

        summary = run_single_fold(
            fold=fold,
            split_type=split_type,
            cfg=cfg,
            device=device,
            results_dir=results_dir,
            verbose=verbose,
        )
        fold_summaries.append(summary)
        _sync_method_dir(method_dir, drive_dir, method_folder)

    # Aggregate test and val_select classification metrics across folds
    test_mean, test_std = aggregate_fold_metrics([s['test_metrics'] for s in fold_summaries])
    valsel_mean, valsel_std = aggregate_fold_metrics([s['valsel_metrics'] for s in fold_summaries])

    # Aggregate C4 conformal metrics across folds for each of the 4 methods
    agg_conformal = {}
    for c4_method in ['LAC', 'APS', 'Ordinal_LAC', 'Ordinal_APS']:
        c_mean, c_std = aggregate_fold_metrics(
            [s['conformal'][c4_method] for s in fold_summaries]
        )
        agg_conformal[c4_method] = {'mean': c_mean, 'std': c_std}

    exp_summary = {
        'exp_name': exp_name,
        'split_type': split_type,
        'mode': mode,
        'folds': folds,
        'config': cfg,
        'valsel_summary': {'mean': valsel_mean, 'std': valsel_std},
        'test_summary': {'mean': test_mean, 'std': test_std},
        'conformal_summary': agg_conformal,
        'fold_summaries': fold_summaries,
    }

    save_experiment_config(method_dir=method_dir, config=cfg, summary=exp_summary)

    summary_path = os.path.join(method_dir, f'summary_{mode}.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(exp_summary, f, indent=2)

    _sync_method_dir(method_dir, drive_dir, method_folder)

    if verbose:
        print(
            f"=== Summary {exp_name} (Split {split_type}, {len(folds)} folds) ===\n"
            f"  MacroF1    : {test_mean['MacroF1']:.4f} +/- {test_std['MacroF1']:.4f}\n"
            f"  WeightedF1 : {test_mean['WeightedF1']:.4f} +/- {test_std['WeightedF1']:.4f}\n"
            f"  Accuracy   : {test_mean['Accuracy']:.4f} +/- {test_std['Accuracy']:.4f}\n"
            f"  QWK        : {test_mean['QWK']:.4f} +/- {test_std['QWK']:.4f}\n"
            f"  MAE        : {test_mean['MAE']:.4f} +/- {test_std['MAE']:.4f}\n"
            f"  Saved to   : {summary_path}"
        )

    return exp_summary


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate MSSF_plus experiments.')
    parser.add_argument(
        '--exp',
        type=str,
        default='MSSF_v2',
        choices=sorted(EXPERIMENTS.keys()),
        help='Experiment configuration name from MSSF_plus/configs/experiments.py',
    )
    parser.add_argument('--split', type=str, default='B', choices=['A', 'B'])
    parser.add_argument('--mode', type=str, default='dev', choices=['quick', 'dev', 'full'])
    parser.add_argument('--folds', type=int, nargs='*', default=None, help='Explicit fold list override')
    parser.add_argument('--epochs', type=int, default=None, help='Override number of training epochs')
    parser.add_argument('--imputer_epochs', type=int, default=None, help='Override C1 imputer training epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Override training batch size')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    parser.add_argument('--device', type=str, default=None, help='Explicit torch device (cuda/cpu/mps)')
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_V2_DIR)
    parser.add_argument('--drive_dir', type=str, default=None, help='Optional Google Drive backup directory')
    parser.add_argument('--resume', action='store_true', help='Skip already completed folds')
    args = parser.parse_args()

    overrides = {}
    if args.epochs is not None:
        overrides['epochs'] = args.epochs
    if args.imputer_epochs is not None:
        overrides['c1_imputer_epochs'] = args.imputer_epochs
    if args.batch_size is not None:
        overrides['batch_size'] = args.batch_size
    if args.lr is not None:
        overrides['lr'] = args.lr
    if args.device is not None:
        overrides['device'] = args.device

    run_experiment(
        exp_name=args.exp,
        split_type=args.split,
        mode=args.mode,
        custom_folds=args.folds,
        results_dir=args.results_dir,
        drive_dir=args.drive_dir,
        resume=args.resume,
        cli_overrides=overrides,
        verbose=True,
    )


if __name__ == '__main__':
    main()
