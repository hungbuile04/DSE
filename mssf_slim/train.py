"""
Copied and isolated Stage 5 (`mssf_slim`) training, validation checkpoint selection (`val_select`),
post-hoc C4 conformal calibration (`val_calib`), peak GPU memory tracking, and learned `EN-add`
view weight extraction.

All outputs are written strictly to `results_slim/` (`DEFAULT_RESULTS_SLIM_DIR`).
Existing read-only utilities in `MSSF_plus` are imported without modifying any files outside
`mssf_slim/` and `results_slim/`.
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
from MSSF_plus.conformal import evaluate_all_conformal_methods

from mssf_slim.configs import (
    DEFAULT_RESULTS_SLIM_DIR,
    SLIM_EXPERIMENTS,
    get_slim_config,
)
from mssf_slim.model import (
    MSSFSlimModel,
    get_named_children_param_breakdown,
)
from mssf_slim.losses import (
    compute_class_weights_from_train,
    logits_to_probs,
    MSSFSlimCriterion,
)


@torch.no_grad()
def evaluate_loader(
    model: MSSFSlimModel,
    loader: torch.utils.data.DataLoader,
    loss_type: str,
    device: torch.device,
    expected_pairs: Optional[np.ndarray] = None,
    allow_train_pairs: bool = False,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on a DataLoader (`shuffle=False`) in `model.eval()` mode
    (no view_dropout, no module dropout, deterministic latent projection, hard labels, and
    `profile_self_correction` applied if enabled on training pairs) and return:
      (metrics_dict, y_true_class_0_to_4, y_pred_class_0_to_4, probs_5class)
    """
    model.eval()
    all_probs_t = []
    all_labels = []
    all_freqs = []

    for batch in loader:
        drugs = batch['drug_feat'].to(device)
        sides = batch['se_feat'].to(device)
        d_avail = batch['drug_avail'].to(device)
        s_avail = batch['se_avail'].to(device)
        fb_d9 = batch['fallback_d9'].to(device)
        fb_d10 = batch['fallback_d10'].to(device)
        d_idx = batch['drug_idx'].to(device)
        s_idx = batch['se_idx'].to(device)

        if not allow_train_pairs and model._sc_y_train is not None:
            y_ds_check = model._sc_y_train[d_idx, s_idx]
            assert torch.all(y_ds_check == 0.0), (
                "CRITICAL: Evaluation pair found with non-zero entry in Y_train!"
            )

        logits, _, _, _, _, _, _, _ = model(
            drugs, sides, drug_avail=d_avail, se_avail=s_avail,
            fallback_d9=fb_d9, fallback_d10=fb_d10,
            drug_idx=d_idx, se_idx=s_idx,
        )
        probs = logits_to_probs(logits, loss_type=loss_type)
        all_probs_t.append(probs)
        lbl_cpu = batch.get('label_cpu')
        all_labels.append((lbl_cpu if lbl_cpu is not None else batch['label'].cpu()).numpy())
        all_freqs.append(batch['freq'].numpy())

    probs_arr = torch.cat(all_probs_t, dim=0).cpu().numpy().astype(np.float32)
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


@torch.no_grad()
def compute_empirical_add_weights_on_loader(
    model: MSSFSlimModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Optional[Dict[str, List[float]]]:
    """
    Compute the empirical mean effective view weights w_v on a DataLoader (e.g., `val_select`)
    when `model.encoderAddition.use_add_weights` is True, accounting for per-pair view availability.
    """
    enc_add = model.encoderAddition
    if not enc_add.use_add_weights or enc_add.drug_view_logits is None or enc_add.se_view_logits is None:
        return None
    model.eval()
    d_weights_batches = []
    s_weights_batches = []
    for batch in loader:
        d_avail = batch['drug_avail'].to(device)
        s_avail = batch['se_avail'].to(device)
        w_d = enc_add._masked_softmax(enc_add.drug_view_logits, d_avail)
        w_s = enc_add._masked_softmax(enc_add.se_view_logits, s_avail)
        d_weights_batches.append(w_d.detach().cpu())
        s_weights_batches.append(w_s.detach().cpu())

    w_d_mean = torch.cat(d_weights_batches, dim=0).mean(dim=0).numpy().astype(np.float64)
    w_s_mean = torch.cat(s_weights_batches, dim=0).mean(dim=0).numpy().astype(np.float64)
    return {
        'valsel_empirical_drug_weights': [float(x) for x in w_d_mean],
        'valsel_empirical_se_weights': [float(x) for x in w_s_mean],
    }


_FOLD_PREP_CACHE: Dict[tuple, tuple] = {}


def _get_or_build_fold_prep(fold: int, split_type: str, cfg: dict, fold_seed: int):
    cache_key = (
        str(split_type),
        int(fold),
        int(fold_seed),
        str(cfg.get('c1_impute', 'zero')),
        int(cfg.get('c1_imputer_epochs', cfg.get('c1_impute_epochs', 80))),
        str(cfg.get('c2_se_mode', 'plus_ontology')),
        str(cfg.get('c5_mode', 'multi_view')),
        float(cfg.get('c5_alpha', 0.15)),
        int(cfg.get('c5_k_min', 4)),
        int(cfg.get('c5_k_max', 12)),
        int(cfg.get('c5_fixed_k') or 0),
        int(cfg.get('c5_min_samples', 3)),
    )
    if cache_key in _FOLD_PREP_CACHE:
        return _FOLD_PREP_CACHE[cache_key]

    t_prep0 = time.time()
    set_global_seed(fold_seed)
    fold_split = load_fold_data(fold, split_type)
    fold_views = build_fold_views(fold_split, cfg, device='cpu')
    soft_res = build_soft_targets(fold_split, fold_views, cfg)
    prep_elapsed = time.time() - t_prep0

    if len(_FOLD_PREP_CACHE) >= 6:
        oldest_key = next(iter(_FOLD_PREP_CACHE))
        del _FOLD_PREP_CACHE[oldest_key]
    _FOLD_PREP_CACHE[cache_key] = (fold_split, fold_views, soft_res, prep_elapsed)
    return fold_split, fold_views, soft_res, prep_elapsed


def run_single_fold(
    fold: int,
    split_type: str,
    cfg: dict,
    device: torch.device,
    results_dir: str = DEFAULT_RESULTS_SLIM_DIR,
    verbose: bool = True,
) -> dict:
    """
    Train, select best checkpoint on `val_select`, calibrate C4 on `val_calib`,
    measure peak GPU memory, extract `named_children()` parameter counts and `EN-add` learned
    view weights, and save outputs to `results_slim/{exp_name}_{split_type}/`.
    """
    base_seed = int(cfg.get('seed', 42))
    fold_seed = base_seed + int(fold)

    exp_name = str(cfg.get('exp_name', 'cand_A'))
    loss_type = str(cfg.get('c3_loss_type', 'ce'))
    c5_mode = str(cfg.get('c5_mode', 'multi_view'))
    selection_metric = str(cfg.get('selection_metric', 'MacroF1'))
    epochs = int(cfg.get('epochs', 50))
    eval_test = bool(cfg.get('eval_test', False))

    # 1-3. Load 4-way split, construct leak-free views (C1/C2) and C5 soft labels
    fold_split, fold_views, soft_res, prep_elapsed = _get_or_build_fold_prep(
        fold, split_type, cfg, fold_seed
    )

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    # Reset seed right before DataLoader & Model init so initial weights and batch shuffling
    # are strictly invariant to RNG calls inside C1/C5 preprocessing
    t_train0 = time.time()
    set_global_seed(fold_seed)

    # 4. Build DataLoaders (staging compact entity lookup tables on device)
    train_loader, valsel_loader, valcal_loader, test_loader = build_dataloaders(
        fold_split, fold_views, soft_res, cfg, device=device
    )

    # 5. Initialize model, loss criterion, and optimizer
    n_drug_views = len(fold_views.drug_view_names)
    n_se_views = len(fold_views.se_view_names)

    model = MSSFSlimModel(
        cfg=cfg,
        n_drug_views=n_drug_views,
        n_se_views=n_se_views,
        profile_drug_view_indices=fold_views.profile_drug_view_indices,
        profile_se_view_indices=fold_views.profile_se_view_indices,
        drug_side_train=fold_views.drug_side_train,
        n_clusters=soft_res.n_clusters,
    ).to(device)

    param_breakdown = get_named_children_param_breakdown(model)
    total_params = int(param_breakdown.get('TOTAL', 0))
    model_repr = str(model)

    class_weights = compute_class_weights_from_train(fold_split.train)
    criterion = MSSFSlimCriterion(
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
    best_train_macro_f1 = 0.0
    best_state_dict = copy.deepcopy(model.state_dict())

    use_soft_targets = c5_mode != 'none'
    epoch_valsel_curve = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss_t = torch.zeros((), dtype=torch.float32, device=device)
        n_batches = 0
        epoch_train_preds_t = []
        epoch_train_labels = []

        for batch in train_loader:
            drugs = batch['drug_feat'].to(device)
            sides = batch['se_feat'].to(device)
            d_avail = batch['drug_avail'].to(device)
            s_avail = batch['se_avail'].to(device)
            fb_d9 = batch['fallback_d9'].to(device)
            fb_d10 = batch['fallback_d10'].to(device)
            d_idx = batch['drug_idx'].to(device)
            s_idx = batch['se_idx'].to(device)
            hard_labels = batch['label'].to(device)
            soft_targets = batch['soft_target'].to(device) if use_soft_targets else None
            cluster_ids = batch['cluster_id'].to(device)

            optimizer.zero_grad()
            logits, rec_con, rec_add, mu, logvar, mod_drugs, mod_sides, cluster_logits = model(
                drugs, sides,
                drug_avail=d_avail, se_avail=s_avail,
                fallback_d9=fb_d9, fallback_d10=fb_d10,
                drug_idx=d_idx, se_idx=s_idx,
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

            with torch.no_grad():
                b_probs = logits_to_probs(logits.detach(), loss_type=loss_type)
                epoch_train_preds_t.append(torch.argmax(b_probs, dim=-1))
                lbl_cpu = batch.get('label_cpu')
                epoch_train_labels.append((lbl_cpu if lbl_cpu is not None else batch['label'].cpu()).numpy())
                running_loss_t += loss.detach()

            n_batches += 1

        running_loss = float(running_loss_t.item())
        tr_y_true = np.concatenate(epoch_train_labels).astype(np.int64)
        tr_y_pred = torch.cat(epoch_train_preds_t, dim=0).cpu().numpy().astype(np.int64)
        train_ep_metrics = compute_multiclass_metrics(tr_y_true, tr_y_pred)
        train_ep_macro_f1 = float(train_ep_metrics['MacroF1'])

        # Evaluate ONLY on val_select for checkpoint selection
        valsel_metrics, _, _, _ = evaluate_loader(
            model, valsel_loader, loss_type=loss_type, device=device,
            expected_pairs=fold_split.valsel
        )
        score = float(valsel_metrics[selection_metric])
        epoch_valsel_curve.append(float(valsel_metrics['MacroF1']))
        if score > best_valsel_score:
            best_valsel_score = score
            best_epoch = epoch
            best_train_macro_f1 = train_ep_macro_f1
            best_state_dict = copy.deepcopy(model.state_dict())

        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            avg_loss = running_loss / max(1, n_batches)
            print(
                f"  [{exp_name} Split {split_type} Fold {fold}] "
                f"Epoch {epoch:02d}/{epochs:02d} | Loss={avg_loss:.4f} | "
                f"train MacroF1={train_ep_macro_f1:.4f} | "
                f"valsel {selection_metric}={score:.4f} (best={best_valsel_score:.4f} @ ep {best_epoch})"
            )

    # Restore best checkpoint selected on val_select
    model.load_state_dict(best_state_dict)

    # Evaluate training set in model.eval() mode
    train_eval_loader = torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=int(cfg.get('eval_batch_size', 512)),
        shuffle=False,
        drop_last=False,
        collate_fn=train_loader.collate_fn,
    )
    train_eval_final, _, _, _ = evaluate_loader(
        model, train_eval_loader, loss_type=loss_type, device=device,
        expected_pairs=fold_split.train, allow_train_pairs=True,
    )
    best_train_loop_macro_f1 = float(best_train_macro_f1)
    best_train_macro_f1 = float(train_eval_final['MacroF1'])

    # Final evaluation on val_select and val_calib
    valsel_final, valsel_true_c, _, valsel_probs = evaluate_loader(
        model, valsel_loader, loss_type=loss_type, device=device,
        expected_pairs=fold_split.valsel
    )
    _, valcal_true_c, _, valcal_probs = evaluate_loader(
        model, valcal_loader, loss_type=loss_type, device=device,
        expected_pairs=fold_split.valcal
    )

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        peak_gpu_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    else:
        peak_gpu_mb = 0.0

    # Extract learned EN-add weights w_v if enabled
    add_weights_info = model.encoderAddition.get_learned_weights_summary(
        profile_drug_indices=fold_views.profile_drug_view_indices
    )
    if add_weights_info is not None:
        emp_w = compute_empirical_add_weights_on_loader(model, valsel_loader, device=device)
        if emp_w is not None:
            add_weights_info.update(emp_w)
        add_weights_info['drug_view_names'] = list(fold_views.drug_view_names)
        add_weights_info['se_view_names'] = list(fold_views.se_view_names)

    # C4 calibrated on val_calib -> evaluated on val_select
    c4_alpha = float(cfg.get('c4_alpha', 0.10))
    valsel_conformal = evaluate_all_conformal_methods(
        valcal_probs=valcal_probs,
        valcal_labels_1_to_5=valcal_true_c + 1,
        test_probs=valsel_probs,
        test_labels_1_to_5=valsel_true_c + 1,
        alpha=c4_alpha,
        seed=fold_seed,
        test_drug_ids=fold_split.valsel[:, 0].astype(int),
    )

    method_dir = os.path.join(results_dir, f'{exp_name}_{split_type}')
    os.makedirs(method_dir, exist_ok=True)

    np.save(os.path.join(method_dir, f'fold_{fold}_valcal_labels.npy'), valcal_true_c.astype(np.int64))
    np.save(os.path.join(method_dir, f'fold_{fold}_valcal_probs.npy'), valcal_probs.astype(np.float32))
    np.save(os.path.join(method_dir, f'fold_{fold}_valsel_labels.npy'), valsel_true_c.astype(np.int64))
    np.save(os.path.join(method_dir, f'fold_{fold}_valsel_probs.npy'), valsel_probs.astype(np.float32))

    if add_weights_info is not None:
        np.save(
            os.path.join(method_dir, f'fold_{fold}_add_weights_drug.npy'),
            np.asarray(add_weights_info['drug_weights_full'], dtype=np.float32),
        )
        np.save(
            os.path.join(method_dir, f'fold_{fold}_add_weights_se.npy'),
            np.asarray(add_weights_info['se_weights_full'], dtype=np.float32),
        )

    if eval_test:
        test_metrics, test_true_c, test_pred_c, test_probs = evaluate_loader(
            model, test_loader, loss_type=loss_type, device=device,
            expected_pairs=fold_split.test
        )
        conformal_res = evaluate_all_conformal_methods(
            valcal_probs=valcal_probs,
            valcal_labels_1_to_5=valcal_true_c + 1,
            test_probs=test_probs,
            test_labels_1_to_5=test_true_c + 1,
            alpha=c4_alpha,
            seed=fold_seed,
            test_drug_ids=fold_split.test[:, 0].astype(int),
        )
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
    else:
        test_metrics = None
        conformal_res = valsel_conformal

    train_eval_elapsed = time.time() - t_train0
    elapsed = float(prep_elapsed + train_eval_elapsed)
    train_valsel_gap = float(best_train_macro_f1 - float(valsel_final['MacroF1']))

    fold_summary = {
        'exp_name': exp_name,
        'split_type': split_type,
        'fold': int(fold),
        'seed': int(base_seed),
        'best_epoch': int(best_epoch),
        'best_valsel_score': float(best_valsel_score),
        'best_train_macro_f1': float(best_train_macro_f1),
        'best_train_loop_macro_f1': float(best_train_loop_macro_f1),
        'train_eval_mode': True,
        'train_eval_metrics': train_eval_final,
        'train_valsel_macro_f1_gap': float(train_valsel_gap),
        'epoch_valsel_macro_f1_curve': epoch_valsel_curve,
        'valsel_metrics': valsel_final,
        'valsel_conformal': valsel_conformal,
        'test_metrics': test_metrics,
        'conformal': conformal_res,
        'c5_diagnostics': soft_res.diagnostics,
        'bvi_diagnostics': None,
        'n_drug_views': int(n_drug_views),
        'n_se_views': int(n_se_views),
        'drug_view_names': list(fold_views.drug_view_names),
        'se_view_names': list(fold_views.se_view_names),
        'total_params': int(total_params),
        'named_children_params': param_breakdown,
        'model_repr': model_repr,
        'add_weights': add_weights_info,
        'peak_gpu_mb': float(peak_gpu_mb),
        'prep_elapsed_seconds': float(prep_elapsed),
        'train_eval_elapsed_seconds': float(train_eval_elapsed),
        'elapsed_seconds': float(elapsed),
    }

    with open(os.path.join(method_dir, f'fold_{fold}_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(fold_summary, f, indent=2)

    if verbose:
        ord_aps = valsel_conformal['Ordinal_APS']
        print(
            f"  --> Fold {fold} Done ({elapsed:.1f}s | PeakGPU={peak_gpu_mb:.1f}MB | Params={total_params:,}) | "
            f"Best Epoch={best_epoch} | train(eval) MacroF1={best_train_macro_f1:.4f} | "
            f"valsel MacroF1={valsel_final['MacroF1']:.4f}, QWK={valsel_final['QWK']:.4f}, "
            f"MAE={valsel_final['MAE']:.4f} | Gap(tr-val)={train_valsel_gap:+.4f} | "
            f"valsel Ord-APS Cov={ord_aps['empirical_coverage']:.3f}, Size={ord_aps['avg_set_size']:.2f}"
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
    eval_test: bool = False,
) -> Optional[dict]:
    """
    If fold outputs already exist locally in `results_slim/` or in `drive_dir`, restore them
    locally and return the loaded fold_summary. Requires `'peak_gpu_mb'` and `'named_children_params'`
    to be present in `fold_{fold}_summary.json` so Stage 5 diagnostics are complete.
    """
    import shutil
    base_req_files = [
        f'fold_{fold}_valsel_labels.npy',
        f'fold_{fold}_valsel_probs.npy',
        f'fold_{fold}_valcal_labels.npy',
        f'fold_{fold}_valcal_probs.npy',
        f'fold_{fold}_summary.json',
    ]
    test_req_files = [
        f'fold_{fold}_labels.npy',
        f'fold_{fold}_preds.npy',
        f'fold_{fold}_probs.npy',
    ]
    opt_files = [
        f'fold_{fold}_add_weights_drug.npy',
        f'fold_{fold}_add_weights_se.npy',
    ]
    req_files = base_req_files + (test_req_files if eval_test else [])

    if drive_dir:
        drive_method_dir = os.path.join(drive_dir, method_folder)
        if os.path.isdir(drive_method_dir):
            os.makedirs(method_dir, exist_ok=True)
            for fname in set(req_files + opt_files):
                local_p = os.path.join(method_dir, fname)
                drive_p = os.path.join(drive_method_dir, fname)
                if not os.path.exists(local_p) and os.path.exists(drive_p):
                    shutil.copy2(drive_p, local_p)

    if all(os.path.exists(os.path.join(method_dir, f)) for f in req_files):
        sum_file = os.path.join(method_dir, f'fold_{fold}_summary.json')
        with open(sum_file, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
        if (
            summary_data.get('train_eval_mode', False) is True
            and 'best_train_macro_f1' in summary_data
            and 'named_children_params' in summary_data
            and 'peak_gpu_mb' in summary_data
        ):
            return summary_data
    return None


def run_experiment(
    exp_name: str,
    split_type: str = 'B',
    mode: str = 'dev',
    custom_folds: Optional[List[int]] = None,
    results_dir: str = DEFAULT_RESULTS_SLIM_DIR,
    drive_dir: Optional[str] = None,
    resume: bool = False,
    cli_overrides: Optional[dict] = None,
    verbose: bool = True,
) -> dict:
    """
    Run a Stage 5 experiment across the requested folds ('quick': [0], 'dev': [0,1,2], 'full': [0..9]),
    aggregate classification, conformal, peak GPU memory, parameter counts, and learned EN-add weights,
    and save provenance files to `results_slim/`.
    """
    overrides = cli_overrides or {}
    cfg = get_slim_config(exp_name, **overrides)
    eval_test = bool(cfg.get('eval_test', False))
    folds = get_fold_list(mode=mode, custom_folds=custom_folds)
    device = get_device(cfg.get('device', None))
    method_folder = f'{exp_name}_{split_type}'
    method_dir = os.path.join(results_dir, method_folder)

    if verbose:
        print(
            f"=== Running [mssf_slim] {exp_name} | Split {split_type} | Mode={mode} (folds={folds}) | "
            f"Device={device} | Epochs={cfg['epochs']} | SelfCorr={cfg.get('profile_self_correction', True)} | "
            f"Resume={resume} ==="
        )

    fold_summaries = []
    for fold in folds:
        if resume:
            cached = _try_restore_completed_fold(
                method_dir, drive_dir, method_folder, fold, eval_test=eval_test
            )
            if cached is not None:
                if verbose:
                    vm = cached['valsel_metrics']
                    pgpu = float(cached.get('peak_gpu_mb', 0.0))
                    print(
                        f"  [RESUME] Skipping {method_folder} Fold {fold} (already completed: "
                        f"valsel MacroF1={vm['MacroF1']:.4f}, QWK={vm['QWK']:.4f}, PeakGPU={pgpu:.1f}MB)"
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

    # Aggregate val_select classification metrics across folds
    valsel_mean, valsel_std = aggregate_fold_metrics([s['valsel_metrics'] for s in fold_summaries])
    gaps = [float(s.get('train_valsel_macro_f1_gap', float('nan'))) for s in fold_summaries]
    gap_mean = float(np.nanmean(gaps)) if gaps else float('nan')
    gap_std = float(np.nanstd(gaps)) if gaps else float('nan')

    peak_gpus = [float(s.get('peak_gpu_mb', 0.0)) for s in fold_summaries]
    peak_gpu_mean = float(np.nanmean(peak_gpus)) if peak_gpus else 0.0
    peak_gpu_max = float(np.nanmax(peak_gpus)) if peak_gpus else 0.0

    test_list = [s['test_metrics'] for s in fold_summaries if s.get('test_metrics') is not None]
    if test_list and len(test_list) == len(fold_summaries):
        test_mean, test_std = aggregate_fold_metrics(test_list)
        test_summary = {'mean': test_mean, 'std': test_std}
    else:
        test_summary = None

    agg_conformal = {}
    for c4_method in ['LAC', 'APS', 'Ordinal_LAC', 'Ordinal_APS']:
        c_mean, c_std = aggregate_fold_metrics(
            [s.get('valsel_conformal', s['conformal'])[c4_method] for s in fold_summaries]
        )
        agg_conformal[c4_method] = {'mean': c_mean, 'std': c_std}

    first_fold = fold_summaries[0] if fold_summaries else {}
    exp_summary = {
        'exp_name': exp_name,
        'split_type': split_type,
        'mode': mode,
        'folds': folds,
        'config': cfg,
        'total_params': first_fold.get('total_params', 0),
        'named_children_params': first_fold.get('named_children_params', {}),
        'model_repr': first_fold.get('model_repr', ''),
        'peak_gpu_mb_mean': peak_gpu_mean,
        'peak_gpu_mb_max': peak_gpu_max,
        'valsel_summary': {'mean': valsel_mean, 'std': valsel_std},
        'train_valsel_gap_summary': {'mean': gap_mean, 'std': gap_std},
        'test_summary': test_summary,
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
            f"=== Summary [mssf_slim] {exp_name} (Split {split_type}, {len(folds)} folds, val_select) ===\n"
            f"  Total Params      : {first_fold.get('total_params', 0):,}\n"
            f"  Peak GPU (MB)     : {peak_gpu_mean:.1f} (max={peak_gpu_max:.1f})\n"
            f"  valsel MacroF1    : {valsel_mean['MacroF1']:.4f} +/- {valsel_std['MacroF1']:.4f}\n"
            f"  valsel WeightedF1 : {valsel_mean['WeightedF1']:.4f} +/- {valsel_std['WeightedF1']:.4f}\n"
            f"  valsel Accuracy   : {valsel_mean['Accuracy']:.4f} +/- {valsel_std['Accuracy']:.4f}\n"
            f"  valsel QWK        : {valsel_mean['QWK']:.4f} +/- {valsel_std['QWK']:.4f}\n"
            f"  valsel MAE        : {valsel_mean['MAE']:.4f} +/- {valsel_std['MAE']:.4f}\n"
            f"  Gap (train-valsel): {gap_mean:+.4f} +/- {gap_std:.4f}\n"
            f"  Saved to          : {summary_path}"
        )

    return exp_summary


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate Stage 5 (mssf_slim) experiments.')
    parser.add_argument(
        '--exp',
        type=str,
        default='cand_A',
        choices=sorted(SLIM_EXPERIMENTS.keys()),
        help='Experiment configuration name from mssf_slim/configs.py',
    )
    parser.add_argument('--split', type=str, default='B', choices=['A', 'B'])
    parser.add_argument('--mode', type=str, default='dev', choices=['quick', 'dev', 'full'])
    parser.add_argument('--folds', type=int, nargs='*', default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_SLIM_DIR)
    parser.add_argument('--drive_dir', type=str, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--eval_test', action='store_true')
    args = parser.parse_args()

    overrides = {}
    if args.seed is not None:
        overrides['seed'] = int(args.seed)
    if args.epochs is not None:
        overrides['epochs'] = args.epochs
    if args.batch_size is not None:
        overrides['batch_size'] = args.batch_size
    if args.lr is not None:
        overrides['lr'] = args.lr
    if args.device is not None:
        overrides['device'] = args.device
    if args.eval_test:
        overrides['eval_test'] = True

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
