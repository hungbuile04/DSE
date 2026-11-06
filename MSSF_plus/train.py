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
from MSSF_plus.conformal import (
    evaluate_all_conformal_methods,
    run_lac,
    run_aps,
    run_ordinal_lac,
    run_ordinal_aps,
)
from MSSF_plus.configs.experiments import get_experiment_config, EXPERIMENTS


@torch.no_grad()
def evaluate_loader(
    model: MSSFPlusModel,
    loader: torch.utils.data.DataLoader,
    loss_type: str,
    device: torch.device,
    expected_pairs: Optional[np.ndarray] = None,
    allow_train_pairs: bool = False,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on a DataLoader (`shuffle=False`) in `model.eval()` mode
    (no view_dropout, no module dropout, deterministic mu, hard labels, and
    `profile_self_correction` applied if enabled on training pairs) and return:
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


@torch.no_grad()
def compute_loader_sigma(
    model: MSSFPlusModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    mask_d9d10: bool = False,
) -> Optional[np.ndarray]:
    """
    Compute per-pair mean posterior standard deviation sigma_i = mean_j exp(0.5 * logvar_{i,j})
    in model.eval() mode when model.use_bvi is True.
    If mask_d9d10=True, masks D9 and D10 to their C1 fallback vectors with availability=0
    to measure posterior uncertainty on pairs missing D9, D10.
    """
    if not getattr(model, 'use_bvi', True):
        return None
    model.eval()
    all_sigmas = []
    for batch in loader:
        drugs = batch['drug_feat'].to(device)
        sides = batch['se_feat'].to(device)
        d_avail = batch['drug_avail'].to(device)
        s_avail = batch['se_avail'].to(device)
        fb_d9 = batch['fallback_d9'].to(device)
        fb_d10 = batch['fallback_d10'].to(device)
        d_idx = batch['drug_idx'].to(device)
        s_idx = batch['se_idx'].to(device)

        if mask_d9d10:
            drugs = drugs.clone()
            d_avail = d_avail.clone()
            p_idx = getattr(model, 'profile_drug_view_indices', []) or []
            if len(p_idx) >= 2:
                i9, i10 = int(p_idx[0]), int(p_idx[1])
                d_dim = fb_d9.shape[1]
                drugs[:, i9 * d_dim:(i9 + 1) * d_dim] = fb_d9
                drugs[:, i10 * d_dim:(i10 + 1) * d_dim] = fb_d10
                d_avail[:, i9] = 0.0
                d_avail[:, i10] = 0.0
            d_idx_Arg = None
            s_idx_Arg = None
        else:
            d_idx_Arg = d_idx
            s_idx_Arg = s_idx

        _, _, _, _, logvar, _, _, _ = model(
            drugs, sides, drug_avail=d_avail, se_avail=s_avail,
            fallback_d9=fb_d9, fallback_d10=fb_d10,
            drug_idx=d_idx_Arg, se_idx=s_idx_Arg,
        )
        if logvar is None:
            return None
        sigma_b = torch.exp(0.5 * logvar).mean(dim=1).detach().cpu().numpy().astype(np.float32)
        all_sigmas.append(sigma_b)
    return np.concatenate(all_sigmas).astype(np.float32) if all_sigmas else None


_FOLD_PREP_CACHE: Dict[tuple, tuple] = {}


def _get_or_build_fold_prep(fold: int, split_type: str, cfg: dict, fold_seed: int):
    cache_key = (
        str(split_type),
        int(fold),
        int(fold_seed),
        str(cfg.get('c1_impute', 'zero')),
        int(cfg.get('c1_imputer_epochs', cfg.get('c1_impute_epochs', 80))),
        int(cfg.get('c1_diffusion_epochs', 100)),
        int(cfg.get('c1_diffusion_steps', 25)),
        int(cfg.get('c1_diffusion_samples', 10)),
        str(cfg.get('c2_se_mode', 'original')),
        str(cfg.get('c5_mode', 'none')),
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
    # Run small entity-level view imputer (N <= 600 drugs) on CPU for minimal launch overhead
    fold_views = build_fold_views(fold_split, cfg, device='cpu')
    soft_res = build_soft_targets(fold_split, fold_views, cfg)
    prep_elapsed = time.time() - t_prep0

    # Keep up to 6 fold entries (~300 MB max) so dev folds [0, 1, 2] stay cached across C3 runs
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
    results_dir: str = DEFAULT_RESULTS_V2_DIR,
    verbose: bool = True,
) -> dict:
    """
    Train, select best checkpoint on `val_select`, calibrate C4 on `val_calib`,
    and evaluate on `val_select` (and optionally `test` only when `eval_test=True` in Stage 7).
    """
    base_seed = int(cfg.get('seed', 42))
    fold_seed = base_seed + int(fold)

    exp_name = str(cfg.get('exp_name', 'MSSF_plus'))
    loss_type = str(cfg.get('c3_loss_type', 'ce'))
    c5_mode = str(cfg.get('c5_mode', 'none'))
    selection_metric = str(cfg.get('selection_metric', 'MacroF1'))
    epochs = int(cfg.get('epochs', 50))
    eval_test = bool(cfg.get('eval_test', False))

    # 1-3. Load 4-way split, construct leak-free views (C1/C2) and C5 soft labels
    fold_split, fold_views, soft_res, prep_elapsed = _get_or_build_fold_prep(
        fold, split_type, cfg, fold_seed
    )

    # Reset seed right before DataLoader & Model init so initial weights and batch shuffling
    # are strictly invariant to RNG calls inside C1/C5 preprocessing
    t_train0 = time.time()
    set_global_seed(fold_seed)

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
        profile_se_view_indices=fold_views.profile_se_view_indices,
        drug_side_train=fold_views.drug_side_train,
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
    best_train_macro_f1 = 0.0
    best_state_dict = copy.deepcopy(model.state_dict())

    use_soft_targets = c5_mode != 'none'
    epoch_valsel_curve = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        epoch_train_preds = []
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
                epoch_train_preds.append(torch.argmax(b_probs, dim=-1).cpu().numpy())
                epoch_train_labels.append(batch['label'].numpy())

            running_loss += float(loss.item())
            n_batches += 1

        tr_y_true = np.concatenate(epoch_train_labels).astype(np.int64)
        tr_y_pred = np.concatenate(epoch_train_preds).astype(np.int64)
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

    # Evaluate training set in model.eval() mode (no view_dropout, no module dropout,
    # deterministic mu, hard labels, and profile_self_correction applied if enabled)
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

    # C4 calibrated on val_calib -> evaluated on val_select (for Stages 1-6 before test is opened)
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

    # Save valsel and valcal arrays always; only evaluate & save test when eval_test=True (Stage 7)
    np.save(os.path.join(method_dir, f'fold_{fold}_valcal_labels.npy'), valcal_true_c.astype(np.int64))
    np.save(os.path.join(method_dir, f'fold_{fold}_valcal_probs.npy'), valcal_probs.astype(np.float32))
    np.save(os.path.join(method_dir, f'fold_{fold}_valsel_labels.npy'), valsel_true_c.astype(np.int64))
    np.save(os.path.join(method_dir, f'fold_{fold}_valsel_probs.npy'), valsel_probs.astype(np.float32))

    # Stage 4 BVI posterior uncertainty diagnostics: save fold_{k}_sigma.npy and compute group stats + C4 correlation
    bvi_diagnostics = None
    if getattr(model, 'use_bvi', True):
        import scipy.stats as sp_stats
        valsel_sigma = compute_loader_sigma(model, valsel_loader, device=device, mask_d9d10=False)
        train_sigma = compute_loader_sigma(model, train_eval_loader, device=device, mask_d9d10=False)
        if valsel_sigma is not None:
            np.save(os.path.join(method_dir, f'fold_{fold}_sigma.npy'), valsel_sigma.astype(np.float32))
            if train_sigma is not None:
                np.save(os.path.join(method_dir, f'fold_{fold}_train_sigma.npy'), train_sigma.astype(np.float32))

            if split_type == 'A':
                valsel_sigma_no_d9d10 = compute_loader_sigma(
                    model, valsel_loader, device=device, mask_d9d10=True
                )
                np.save(
                    os.path.join(method_dir, f'fold_{fold}_sigma_no_d9d10.npy'),
                    valsel_sigma_no_d9d10.astype(np.float32),
                )
                sigma_has_d9d10_mean = float(np.mean(valsel_sigma))
                sigma_has_d9d10_std = float(np.std(valsel_sigma))
                sigma_missing_d9d10_mean = float(np.mean(valsel_sigma_no_d9d10))
                sigma_missing_d9d10_std = float(np.std(valsel_sigma_no_d9d10))
            else:
                sigma_has_d9d10_mean = float(np.mean(train_sigma)) if train_sigma is not None else float('nan')
                sigma_has_d9d10_std = float(np.std(train_sigma)) if train_sigma is not None else float('nan')
                sigma_missing_d9d10_mean = float(np.mean(valsel_sigma))
                sigma_missing_d9d10_std = float(np.std(valsel_sigma))

            valsel_d_ids = fold_split.valsel[:, 0].astype(int)
            prof_set = set(fold_views.profile_drug_view_indices or [])
            static_indices = [i for i in range(fold_views.drug_avail.shape[1]) if i not in prof_set]
            static_avail = fold_views.drug_avail[valsel_d_ids][:, static_indices]
            full_static_mask = np.all(static_avail > 0.5, axis=1)
            sigma_full_static_mean = (
                float(np.mean(valsel_sigma[full_static_mask])) if np.any(full_static_mask) else float('nan')
            )
            sigma_missing_static_mean = (
                float(np.mean(valsel_sigma[~full_static_mask])) if np.any(~full_static_mask) else float('nan')
            )

            _, sets_lac = run_lac(valcal_probs, valcal_true_c, valsel_probs, valsel_true_c, alpha=c4_alpha)
            _, sets_aps = run_aps(
                valcal_probs, valcal_true_c, valsel_probs, valsel_true_c, alpha=c4_alpha, seed=fold_seed
            )
            _, sets_olac = run_ordinal_lac(
                valcal_probs, valcal_true_c, valsel_probs, valsel_true_c, alpha=c4_alpha
            )
            _, sets_oaps = run_ordinal_aps(
                valcal_probs, valcal_true_c, valsel_probs, valsel_true_c, alpha=c4_alpha, seed=fold_seed
            )

            def _corr_pair(sig: np.ndarray, psets: np.ndarray) -> Tuple[float, float]:
                sz = psets.sum(axis=1).astype(np.float64)
                s64 = sig.astype(np.float64)
                if len(s64) < 3 or np.std(s64) < 1e-12 or np.std(sz) < 1e-12:
                    return 0.0, 0.0
                pr = float(sp_stats.pearsonr(s64, sz)[0])
                sr = float(sp_stats.spearmanr(s64, sz)[0])
                return pr, sr

            r_oaps, rho_oaps = _corr_pair(valsel_sigma, sets_oaps)
            r_olac, rho_olac = _corr_pair(valsel_sigma, sets_olac)
            r_aps, rho_aps = _corr_pair(valsel_sigma, sets_aps)
            r_lac, rho_lac = _corr_pair(valsel_sigma, sets_lac)

            bvi_diagnostics = {
                'valsel_sigma_mean': float(np.mean(valsel_sigma)),
                'valsel_sigma_std': float(np.std(valsel_sigma)),
                'train_sigma_mean': float(np.mean(train_sigma)) if train_sigma is not None else float('nan'),
                'train_sigma_std': float(np.std(train_sigma)) if train_sigma is not None else float('nan'),
                'sigma_has_d9d10_mean': sigma_has_d9d10_mean,
                'sigma_has_d9d10_std': sigma_has_d9d10_std,
                'sigma_missing_d9d10_mean': sigma_missing_d9d10_mean,
                'sigma_missing_d9d10_std': sigma_missing_d9d10_std,
                'sigma_full_static_mean': sigma_full_static_mean,
                'sigma_missing_static_mean': sigma_missing_static_mean,
                'n_valsel_full_static': int(np.sum(full_static_mask)),
                'n_valsel_missing_static': int(np.sum(~full_static_mask)),
                'corr_sigma_ord_aps_pearson': r_oaps,
                'corr_sigma_ord_aps_spearman': rho_oaps,
                'corr_sigma_ord_lac_pearson': r_olac,
                'corr_sigma_ord_lac_spearman': rho_olac,
                'corr_sigma_aps_pearson': r_aps,
                'corr_sigma_aps_spearman': rho_aps,
                'corr_sigma_lac_pearson': r_lac,
                'corr_sigma_lac_spearman': rho_lac,
            }

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
        'bvi_diagnostics': bvi_diagnostics,
        'n_drug_views': int(n_drug_views),
        'n_se_views': int(n_se_views),
        'prep_elapsed_seconds': float(prep_elapsed),
        'train_eval_elapsed_seconds': float(train_eval_elapsed),
        'elapsed_seconds': float(elapsed),
    }

    with open(os.path.join(method_dir, f'fold_{fold}_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(fold_summary, f, indent=2)

    if verbose:
        ord_aps = valsel_conformal['Ordinal_APS']
        print(
            f"  --> Fold {fold} Done ({elapsed:.1f}s) | Best Epoch={best_epoch} | "
            f"train(eval) MacroF1={best_train_macro_f1:.4f} | "
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
    cfg: Optional[dict] = None,
    split_type: Optional[str] = None,
) -> Optional[dict]:
    """
    If fold outputs already exist locally or in `drive_dir`, restore them locally
    and return the loaded fold_summary.
    """
    import shutil
    base_req_files = [
        f'fold_{fold}_valsel_labels.npy',
        f'fold_{fold}_valsel_probs.npy',
        f'fold_{fold}_valcal_labels.npy',
        f'fold_{fold}_valcal_probs.npy',
        f'fold_{fold}_summary.json',
    ]
    opt_bvi_files = [
        f'fold_{fold}_sigma.npy',
        f'fold_{fold}_train_sigma.npy',
        f'fold_{fold}_sigma_no_d9d10.npy',
    ]
    test_req_files = [
        f'fold_{fold}_labels.npy',
        f'fold_{fold}_preds.npy',
        f'fold_{fold}_probs.npy',
    ]
    req_files = base_req_files + (test_req_files if eval_test else [])
    if cfg is not None and bool(cfg.get('use_bvi', True)) and cfg.get('exp_name') in (
        'C3_ce_unweighted_ridge_selfcorr',
        'bvi_keep_no_c5',
    ):
        req_files = req_files + [f'fold_{fold}_sigma.npy']

    if drive_dir:
        drive_method_dir = os.path.join(drive_dir, method_folder)
        if os.path.isdir(drive_method_dir):
            os.makedirs(method_dir, exist_ok=True)
            for fname in set(req_files + opt_bvi_files):
                local_p = os.path.join(method_dir, fname)
                drive_p = os.path.join(drive_method_dir, fname)
                if not os.path.exists(local_p) and os.path.exists(drive_p):
                    shutil.copy2(drive_p, local_p)

    if all(os.path.exists(os.path.join(method_dir, f)) for f in req_files):
        sum_file = os.path.join(method_dir, f'fold_{fold}_summary.json')
        with open(sum_file, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
        if summary_data.get('train_eval_mode', False) is True and 'best_train_macro_f1' in summary_data:
            if (
                cfg is not None
                and bool(cfg.get('use_bvi', True))
                and cfg.get('exp_name') in ('C3_ce_unweighted_ridge_selfcorr', 'bvi_keep_no_c5')
                and not summary_data.get('bvi_diagnostics')
            ):
                return None
            c5d = summary_data.get('c5_diagnostics', {})
            if cfg is not None and split_type is not None and ('entropy_q' not in c5d or 'k_max' not in c5d):
                fold_seed = int(cfg.get('seed', 42)) + int(fold)
                set_global_seed(fold_seed)
                fs = load_fold_data(fold, split_type)
                cfg_fast = dict(cfg)
                cfg_fast['c1_impute'] = 'zero'
                fv = build_fold_views(fs, cfg_fast, device='cpu')
                sr = build_soft_targets(fs, fv, cfg)
                summary_data['c5_diagnostics'] = sr.diagnostics
                with open(sum_file, 'w', encoding='utf-8') as f:
                    json.dump(summary_data, f, indent=2)
            return summary_data
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
    eval_test = bool(cfg.get('eval_test', False))
    folds = get_fold_list(mode=mode, custom_folds=custom_folds)
    device = get_device(cfg.get('device', None))
    method_folder = f'{exp_name}_{split_type}'
    method_dir = os.path.join(results_dir, method_folder)

    if verbose:
        print(
            f"=== Running {exp_name} | Split {split_type} | Mode={mode} (folds={folds}) | "
            f"Device={device} | Epochs={cfg['epochs']} | SelfCorr={cfg.get('profile_self_correction', False)} | "
            f"Resume={resume} ==="
        )

    fold_summaries = []
    for fold in folds:
        if resume:
            cached = _try_restore_completed_fold(
                method_dir, drive_dir, method_folder, fold, eval_test=eval_test, cfg=cfg, split_type=split_type
            )
            if cached is not None:
                if verbose:
                    vm = cached['valsel_metrics']
                    print(
                        f"  [RESUME] Skipping {method_folder} Fold {fold} (already completed: "
                        f"valsel MacroF1={vm['MacroF1']:.4f}, QWK={vm['QWK']:.4f})"
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

    test_list = [s['test_metrics'] for s in fold_summaries if s.get('test_metrics') is not None]
    if test_list and len(test_list) == len(fold_summaries):
        test_mean, test_std = aggregate_fold_metrics(test_list)
        test_summary = {'mean': test_mean, 'std': test_std}
    else:
        test_summary = None

    # Aggregate C4 conformal metrics across folds for each of the 4 methods
    agg_conformal = {}
    for c4_method in ['LAC', 'APS', 'Ordinal_LAC', 'Ordinal_APS']:
        c_mean, c_std = aggregate_fold_metrics(
            [s.get('valsel_conformal', s['conformal'])[c4_method] for s in fold_summaries]
        )
        agg_conformal[c4_method] = {'mean': c_mean, 'std': c_std}

    exp_summary = {
        'exp_name': exp_name,
        'split_type': split_type,
        'mode': mode,
        'folds': folds,
        'config': cfg,
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
            f"=== Summary {exp_name} (Split {split_type}, {len(folds)} folds, val_select) ===\n"
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
    parser.add_argument('--seed', type=int, default=None, help='Override random seed (default: 42)')
    parser.add_argument('--profile_self_correction', type=int, choices=[0, 1], default=None)
    parser.add_argument('--eval_test', action='store_true', help='Only enable in Stage 7 after final approval')
    args = parser.parse_args()

    overrides = {}
    if args.seed is not None:
        overrides['seed'] = int(args.seed)
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
    if args.profile_self_correction is not None:
        overrides['profile_self_correction'] = bool(args.profile_self_correction)
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
