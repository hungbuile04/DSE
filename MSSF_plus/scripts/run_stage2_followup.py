"""
Unified execution and reporting script for Stage 2 Follow-up (Items 2, 3, and 4):

  - Item 2: Evaluate cheap C1 cross-fitted (K=5) imputation variants (`regress_ridge` and
    `regress_knn`) against `mean`, `diffusion`, and `regress` (MLP) on Fold 0, Split B:
      (a) Imputation quality on `val_select` (67 cold-start drugs): MSE, Spearman rank
          correlation, and Top-10 neighbor overlap on D9, D10, and Avg(D9, D10).
      (b) Downstream `MSSF_plus` (`C3_ce_unweighted`) Macro-F1 on Fold 0, Split B for both
          `profile_self_correction=0` and `profile_self_correction=1`.
  - Item 3: Redefine and recompute `Gap (train - val_select)` for all Stage 2 runs:
    training Macro-F1 computed in `model.eval()` mode, no view dropout, hard labels, and
    features constructed identically to validation evaluation (with `profile_self_correction`
    applied if enabled).
  - Item 4: Estimate run-to-run seed noise on Fold 0, Split B across 3 seeds (`seed=42, 43, 44`)
    for `MSSF_v2` and `C3_ce_unweighted` under both `profile_self_correction` settings (`0` and `1`).
"""

import os
import time
import json
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.stats as stats
from sklearn.metrics.pairwise import cosine_similarity
import torch

from MSSF_plus.protocol import (
    DEFAULT_RESULTS_V2_DIR,
    N_DRUGS,
    N_SIDES,
    set_global_seed,
    get_device,
    load_fold_data,
)
from MSSF_plus.views import (
    STATIC_DRUG_VIEW_NAMES,
    get_static_raw_views,
    compute_view_availability,
    compute_profile_views_from_train,
    train_and_impute_regress,
    train_and_impute_ridge,
    train_and_impute_knn,
    train_and_impute_diffusion,
)
from MSSF_plus.train import (
    run_experiment,
    evaluate_loader,
    _get_or_build_fold_prep,
    _sync_method_dir,
)
from MSSF_plus.data import build_dataloaders
from MSSF_plus.model import MSSFPlusModel
from MSSF_plus.losses import compute_class_weights_from_train, MSSFPlusCriterion
from MSSF_plus.metrics import aggregate_fold_metrics
from MSSF_plus.configs.experiments import get_experiment_config


def _eval_imputation_matrix(
    pred_mat: np.ndarray,
    true_mat: np.ndarray,
    eval_drugs: np.ndarray,
    train_drugs: np.ndarray,
    top_k: int = 10,
) -> Dict[str, float]:
    """
    Evaluate imputed similarity rows `pred_mat[eval_drugs][:, train_drugs]` against
    ground-truth profile similarity rows `true_mat[eval_drugs][:, train_drugs]` on:
      - MSE
      - Row-wise Spearman rank correlation
      - Top-K neighbor overlap (%) among `train_drugs`
    """
    p_sub = np.asarray(pred_mat[np.ix_(eval_drugs, train_drugs)], dtype=np.float64)
    t_sub = np.asarray(true_mat[np.ix_(eval_drugs, train_drugs)], dtype=np.float64)

    mse = float(np.mean((p_sub - t_sub) ** 2))
    sp_list = []
    ov_list = []
    for i in range(len(eval_drugs)):
        pr = p_sub[i]
        tr = t_sub[i]
        if np.std(pr) > 1e-12 and np.std(tr) > 1e-12:
            sp = float(stats.spearmanr(pr, tr)[0])
            if not np.isnan(sp):
                sp_list.append(sp)
        top_p = set(np.argsort(-pr)[:top_k].tolist())
        top_t = set(np.argsort(-tr)[:top_k].tolist())
        ov_list.append(len(top_p.intersection(top_t)) / float(top_k))

    return {
        'MSE': mse,
        'Spearman': float(np.mean(sp_list)) if sp_list else 0.0,
        'Top10_Overlap_Pct': 100.0 * (float(np.mean(ov_list)) if ov_list else 0.0),
    }


def evaluate_c1_imputation_quality_fold0_split_b(
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compare C1 imputation quality (`mean`, `regress_knn`, `diffusion`, `regress` MLP,
    and `regress_ridge`, all with cross-fit K=5) on Fold 0, Split B (`val_select` 67 cold-start drugs).
    """
    if device is None:
        device = torch.device('cpu')
    fold_split = load_fold_data(0, 'B')
    train_drugs = np.unique(fold_split.train_drugs)
    valsel_drugs = np.unique(fold_split.valsel_drugs)

    # True profile views of val_select drugs against train_drugs (strictly for oracle quality evaluation)
    y_oracle = np.zeros((N_DRUGS, N_SIDES), dtype=np.float64)
    for arr in (fold_split.train, fold_split.valsel):
        d = arr[:, 0].astype(int)
        s = arr[:, 1].astype(int)
        y_oracle[d, s] = arr[:, 2]
    b_oracle = (y_oracle > 0.0).astype(np.float64)
    d9_true = cosine_similarity(y_oracle)
    d10_true = cosine_similarity(b_oracle)

    static = get_static_raw_views()
    prof = compute_profile_views_from_train(fold_split.train)
    d9_raw = prof['D9_profile_freq_cos']
    d10_raw = prof['D10_profile_bin_cos']

    static_drug_mats = [static[name] for name in STATIC_DRUG_VIEW_NAMES]
    static_drug_avail = np.column_stack([compute_view_availability(m) for m in static_drug_mats])
    static_drug_concat = np.hstack(static_drug_mats)

    results = {}

    # 1. mean
    t0 = time.time()
    d9_mean = np.tile(d9_raw[train_drugs].mean(axis=0), (N_DRUGS, 1))
    d10_mean = np.tile(d10_raw[train_drugs].mean(axis=0), (N_DRUGS, 1))
    t_mean = time.time() - t0

    # 2. regress_knn (K=5)
    t0 = time.time()
    d9_knn, d10_knn = train_and_impute_knn(
        static_drug_mats, static_drug_avail, d9_raw, d10_raw,
        train_drugs=train_drugs, seed=seed, n_splits=5,
    )
    t_knn = time.time() - t0

    # 3. regress_ridge (K=5)
    t0 = time.time()
    d9_ridge, d10_ridge = train_and_impute_ridge(
        static_drug_concat, static_drug_avail, d9_raw, d10_raw,
        train_drugs=train_drugs, seed=seed, n_splits=5,
    )
    t_ridge = time.time() - t0

    # 4. regress MLP (K=5, epochs=80)
    t0 = time.time()
    set_global_seed(seed)
    d9_mlp, d10_mlp = train_and_impute_regress(
        static_drug_concat, static_drug_avail, d9_raw, d10_raw,
        train_drugs=train_drugs, device=device, epochs=80,
        seed=seed, n_splits=5,
    )
    t_mlp = time.time() - t0

    # 5. diffusion (K=5, epochs=100, T=25, B=10)
    t0 = time.time()
    set_global_seed(seed)
    d9_diff, d10_diff = train_and_impute_diffusion(
        static_drug_concat, static_drug_avail, d9_raw, d10_raw,
        train_drugs=train_drugs, device=device,
        n_timesteps=25, n_samples=10, epochs=100,
        seed=seed, n_splits=5,
    )
    t_diff = time.time() - t0

    for name, (m9, m10, elapsed_s) in [
        ('mean', (d9_mean, d10_mean, t_mean)),
        ('regress_knn', (d9_knn, d10_knn, t_knn)),
        ('diffusion', (d9_diff, d10_diff, t_diff)),
        ('regress (MLP)', (d9_mlp, d10_mlp, t_mlp)),
        ('regress_ridge', (d9_ridge, d10_ridge, t_ridge)),
    ]:
        r9 = _eval_imputation_matrix(m9, d9_true, valsel_drugs, train_drugs, top_k=10)
        r10 = _eval_imputation_matrix(m10, d10_true, valsel_drugs, train_drugs, top_k=10)
        results[name] = {
            'D9_MSE': r9['MSE'],
            'D9_Spearman': r9['Spearman'],
            'D9_Top10_Pct': r9['Top10_Overlap_Pct'],
            'D10_MSE': r10['MSE'],
            'D10_Spearman': r10['Spearman'],
            'D10_Top10_Pct': r10['Top10_Overlap_Pct'],
            'Avg_MSE': 0.5 * (r9['MSE'] + r10['MSE']),
            'Avg_Spearman': 0.5 * (r9['Spearman'] + r10['Spearman']),
            'Avg_Top10_Pct': 0.5 * (r9['Top10_Overlap_Pct'] + r10['Top10_Overlap_Pct']),
            'Time_s': float(elapsed_s),
        }

    print("\n=== Item 2A: C1 Imputation Quality on Fold 0, Split B (val_select, 67 cold-start drugs, K=5) ===")
    print(
        f"{'Method':<16} | {'Avg MSE':>9} | {'Avg Spearman':>12} | {'Avg Top10 (%)':>13} | "
        f"{'D9 MSE':>9} | {'D9 Sp':>7} | {'D9 Top10':>8} | {'D10 MSE':>9} | {'D10 Sp':>7} | {'D10 Top10':>9} | {'Time(s)':>7}"
    )
    print("-" * 132)
    for name, m in results.items():
        print(
            f"{name:<16} | {m['Avg_MSE']:9.6f} | {m['Avg_Spearman']:12.4f} | {m['Avg_Top10_Pct']:12.2f}% | "
            f"{m['D9_MSE']:9.6f} | {m['D9_Spearman']:7.4f} | {m['D9_Top10_Pct']:7.2f}% | "
            f"{m['D10_MSE']:9.6f} | {m['D10_Spearman']:7.4f} | {m['D10_Top10_Pct']:8.2f}% | {m['Time_s']:7.2f}"
        )
    return results


def recompute_stage2_eval_gap_fast(
    exp_name: str,
    split_type: str,
    folds: List[int],
    results_dir: str,
    drive_dir: Optional[str],
    device: torch.device,
) -> dict:
    """
    For any fold of `exp_name` on `split_type` that already completed in Stage 2 (with
    `best_epoch` recorded in `fold_{k}_summary.json` and `fold_{k}_valsel_probs.npy` saved),
    if `train_eval_mode` is not yet `True`, deterministically replay training up to `best_epoch`,
    evaluate `train_eval_loader` in `model.eval()` mode (no view dropout, hard labels,
    with `profile_self_correction` if enabled), and update `fold_{k}_summary.json` and
    `summary_dev.json`. If the fold has not been run yet, run it via `run_experiment`.
    """
    cfg = get_experiment_config(exp_name)
    method_folder = f'{exp_name}_{split_type}'
    method_dir = os.path.join(results_dir, method_folder)

    # First sync any existing files from drive_dir
    if drive_dir:
        drive_m_dir = os.path.join(drive_dir, method_folder)
        if os.path.isdir(drive_m_dir):
            os.makedirs(method_dir, exist_ok=True)
            import shutil
            for fn in os.listdir(drive_m_dir):
                sp = os.path.join(drive_m_dir, fn)
                dp = os.path.join(method_dir, fn)
                if os.path.isfile(sp) and not os.path.exists(dp):
                    shutil.copy2(sp, dp)

    need_full_run = []
    for fold in folds:
        s_path = os.path.join(method_dir, f'fold_{fold}_summary.json')
        p_path = os.path.join(method_dir, f'fold_{fold}_valsel_probs.npy')
        if not (os.path.exists(s_path) and os.path.exists(p_path)):
            need_full_run.append(fold)

    if need_full_run:
        run_experiment(
            exp_name=exp_name,
            split_type=split_type,
            mode='dev',
            custom_folds=need_full_run,
            results_dir=results_dir,
            drive_dir=drive_dir,
            resume=True,
            cli_overrides={'device': str(device)},
            verbose=True,
        )

    base_seed = int(cfg.get('seed', 42))
    loss_type = str(cfg.get('c3_loss_type', 'ce'))
    c5_mode = str(cfg.get('c5_mode', 'none'))
    use_soft_targets = c5_mode != 'none'

    fold_summaries = []
    for fold in folds:
        s_path = os.path.join(method_dir, f'fold_{fold}_summary.json')
        with open(s_path, 'r', encoding='utf-8') as f:
            s_data = json.load(f)

        if s_data.get('train_eval_mode', False) is True and 'best_train_macro_f1' in s_data:
            print(
                f"  [Item 3 Cached] {exp_name} Split {split_type} Fold {fold}: "
                f"best_ep={s_data['best_epoch']} | train(eval) MacroF1={s_data['best_train_macro_f1']:.4f} | "
                f"valsel MacroF1={s_data['valsel_metrics']['MacroF1']:.4f} | "
                f"Gap(tr-val)={s_data['train_valsel_macro_f1_gap']:+.4f}"
            )
            fold_summaries.append(s_data)
            continue

        target_epoch = int(s_data['best_epoch'])
        fold_seed = base_seed + int(fold)
        t0 = time.time()

        fold_split, fold_views, soft_res, _ = _get_or_build_fold_prep(fold, split_type, cfg, fold_seed)
        set_global_seed(fold_seed)
        train_loader, valsel_loader, _, _ = build_dataloaders(fold_split, fold_views, soft_res, cfg)

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
            lr=float(cfg.get('lr', 0.0001)),
            weight_decay=float(cfg.get('weight_decay', 1e-5)),
        )

        for epoch in range(1, target_epoch + 1):
            model.train()
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
                    logits=logits, rec_con=rec_con, rec_add=rec_add, mu=mu, logvar=logvar,
                    batch_drug=mod_drugs, batch_side=mod_sides, hard_labels=hard_labels,
                    soft_targets=soft_targets, cluster_logits=cluster_logits,
                    cluster_ids=cluster_ids, epoch=epoch,
                )
                loss.backward()
                optimizer.step()

            # Keep RNG progression identical to train.py by evaluating valsel_loader each epoch
            evaluate_loader(model, valsel_loader, loss_type=loss_type, device=device, expected_pairs=fold_split.valsel)

        # Evaluate train_eval_loader in model.eval() mode at target_epoch
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
        valsel_check, _, _, _ = evaluate_loader(
            model, valsel_loader, loss_type=loss_type, device=device,
            expected_pairs=fold_split.valsel,
        )

        old_train_loop_f1 = float(s_data.get('best_train_macro_f1', 0.0))
        new_train_eval_f1 = float(train_eval_final['MacroF1'])
        valsel_f1 = float(s_data['valsel_metrics']['MacroF1'])
        new_gap = float(new_train_eval_f1 - valsel_f1)

        s_data['best_train_loop_macro_f1'] = old_train_loop_f1
        s_data['best_train_macro_f1'] = new_train_eval_f1
        s_data['train_eval_mode'] = True
        s_data['train_eval_metrics'] = train_eval_final
        s_data['train_valsel_macro_f1_gap'] = new_gap
        s_data['replayed_valsel_macro_f1'] = float(valsel_check['MacroF1'])

        with open(s_path, 'w', encoding='utf-8') as f:
            json.dump(s_data, f, indent=2)

        print(
            f"  [Item 3 Recomputed] {exp_name} Split {split_type} Fold {fold} ({time.time() - t0:.1f}s): "
            f"best_ep={target_epoch} | train(loop)={old_train_loop_f1:.4f} -> train(eval)={new_train_eval_f1:.4f} | "
            f"valsel={valsel_f1:.4f} (replay={valsel_check['MacroF1']:.4f}) | Gap(tr-val)={new_gap:+.4f}"
        )
        fold_summaries.append(s_data)

    # Update summary_dev.json
    valsel_mean, valsel_std = aggregate_fold_metrics([s['valsel_metrics'] for s in fold_summaries])
    gaps = [float(s['train_valsel_macro_f1_gap']) for s in fold_summaries]
    tr_f1s = [float(s['best_train_macro_f1']) for s in fold_summaries]
    gap_mean, gap_std = float(np.mean(gaps)), float(np.std(gaps))
    tr_mean, tr_std = float(np.mean(tr_f1s)), float(np.std(tr_f1s))

    summary_dev_path = os.path.join(method_dir, 'summary_dev.json')
    if os.path.exists(summary_dev_path):
        with open(summary_dev_path, 'r', encoding='utf-8') as f:
            exp_sum = json.load(f)
    else:
        exp_sum = {'exp_name': exp_name, 'split_type': split_type, 'mode': 'dev', 'folds': folds, 'config': cfg}

    exp_sum['valsel_summary'] = {'mean': valsel_mean, 'std': valsel_std}
    exp_sum['train_eval_macro_f1_summary'] = {'mean': tr_mean, 'std': tr_std}
    exp_sum['train_valsel_gap_summary'] = {'mean': gap_mean, 'std': gap_std}
    exp_sum['fold_summaries'] = fold_summaries

    with open(summary_dev_path, 'w', encoding='utf-8') as f:
        json.dump(exp_sum, f, indent=2)
    _sync_method_dir(method_dir, drive_dir, method_folder)
    return exp_sum


def main():
    parser = argparse.ArgumentParser(description='Run Stage 2 Follow-up (Items 2, 3, and 4).')
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_V2_DIR)
    parser.add_argument('--drive_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--resume', action='store_true', default=True)
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"=== Running Stage 2 Follow-up (Items 2, 3, 4) on Device={device} ===")

    # -------------------------------------------------------------------------
    # ITEM 2A: C1 Imputation Quality on Fold 0, Split B (67 cold-start drugs)
    # -------------------------------------------------------------------------
    imp_quality = evaluate_c1_imputation_quality_fold0_split_b(seed=42)

    # -------------------------------------------------------------------------
    # ITEM 3: Recompute model.eval() Train Macro-F1 & Gap for Stage 2 Runs
    # -------------------------------------------------------------------------
    print("\n=== Item 3: Recomputing model.eval() Train Macro-F1 & Gap for Stage 2 Runs (Folds 0-2) ===")
    stage2_gap_results = {}
    for split_type in ['B', 'A']:
        for exp_name in ['MSSF_v2', 'MSSF_v2_selfcorr', 'C3_ce_unweighted', 'C3_ce_unweighted_selfcorr']:
            exp_sum = recompute_stage2_eval_gap_fast(
                exp_name=exp_name,
                split_type=split_type,
                folds=[0, 1, 2],
                results_dir=args.results_dir,
                drive_dir=args.drive_dir,
                device=device,
            )
            stage2_gap_results[f'{exp_name}_{split_type}'] = exp_sum

    # -------------------------------------------------------------------------
    # ITEM 2B: Downstream MSSF_plus Macro-F1 with regress_ridge & regress_knn (Fold 0, Split B)
    # -------------------------------------------------------------------------
    print("\n=== Item 2B: Running Downstream MSSF_plus with regress_ridge & regress_knn (Fold 0, Split B) ===")
    item2_downstream = {}
    for exp_name in [
        'C3_ce_unweighted_ridge',
        'C3_ce_unweighted_ridge_selfcorr',
        'C3_ce_unweighted_knn',
        'C3_ce_unweighted_knn_selfcorr',
    ]:
        res = run_experiment(
            exp_name=exp_name,
            split_type='B',
            mode='quick',
            custom_folds=[0],
            results_dir=args.results_dir,
            drive_dir=args.drive_dir,
            resume=args.resume,
            cli_overrides={'device': str(device)},
            verbose=True,
        )
        item2_downstream[exp_name] = res['fold_summaries'][0]

    # Include Fold 0 from C3_ce_unweighted and C3_ce_unweighted_selfcorr (regress MLP)
    item2_downstream['C3_ce_unweighted'] = stage2_gap_results['C3_ce_unweighted_B']['fold_summaries'][0]
    item2_downstream['C3_ce_unweighted_selfcorr'] = stage2_gap_results['C3_ce_unweighted_selfcorr_B']['fold_summaries'][0]

    # -------------------------------------------------------------------------
    # ITEM 4: Multi-Seed Run-to-Run Noise Estimation (Fold 0, Split B, 3 seeds: 42, 43, 44)
    # -------------------------------------------------------------------------
    print("\n=== Item 4: Running Multi-Seed Noise Estimation (Seeds 43, 44 on Fold 0, Split B) ===")
    seed_runs = {}
    for exp_name in [
        'MSSF_v2_s43',
        'MSSF_v2_s44',
        'MSSF_v2_selfcorr_s43',
        'MSSF_v2_selfcorr_s44',
        'C3_ce_unweighted_s43',
        'C3_ce_unweighted_s44',
        'C3_ce_unweighted_selfcorr_s43',
        'C3_ce_unweighted_selfcorr_s44',
    ]:
        res = run_experiment(
            exp_name=exp_name,
            split_type='B',
            mode='quick',
            custom_folds=[0],
            results_dir=args.results_dir,
            drive_dir=args.drive_dir,
            resume=args.resume,
            cli_overrides={'device': str(device)},
            verbose=True,
        )
        seed_runs[exp_name] = res['fold_summaries'][0]

    seed_groups = {
        'MSSF_v2 (SelfCorr=0)': [
            stage2_gap_results['MSSF_v2_B']['fold_summaries'][0],
            seed_runs['MSSF_v2_s43'],
            seed_runs['MSSF_v2_s44'],
        ],
        'MSSF_v2_selfcorr (SelfCorr=1)': [
            stage2_gap_results['MSSF_v2_selfcorr_B']['fold_summaries'][0],
            seed_runs['MSSF_v2_selfcorr_s43'],
            seed_runs['MSSF_v2_selfcorr_s44'],
        ],
        'C3_ce_unweighted (SelfCorr=0)': [
            stage2_gap_results['C3_ce_unweighted_B']['fold_summaries'][0],
            seed_runs['C3_ce_unweighted_s43'],
            seed_runs['C3_ce_unweighted_s44'],
        ],
        'C3_ce_unweighted_selfcorr (SelfCorr=1)': [
            stage2_gap_results['C3_ce_unweighted_selfcorr_B']['fold_summaries'][0],
            seed_runs['C3_ce_unweighted_selfcorr_s43'],
            seed_runs['C3_ce_unweighted_selfcorr_s44'],
        ],
    }

    # Print formatted summary tables for Items 2, 3, and 4
    print("\n" + "=" * 100)
    print("=== SUMMARY REPORT FOR ITEMS 2, 3, AND 4 ===")
    print("=" * 100)

    print("\n[Item 2B] Downstream MSSF_plus on Fold 0, Split B (Comparing C1 Imputers, K=5):")
    print(f"{'Config':<32} | {'SelfCorr':>8} | {'Imputer':<14} | {'Macro-F1':>8} | {'WF1':>8} | {'Acc':>8} | {'QWK':>8} | {'MAE':>8} | {'Gap(eval)':>9}")
    print("-" * 120)
    for k_name, imp_label in [
        ('C3_ce_unweighted_selfcorr', 'regress (MLP)'),
        ('C3_ce_unweighted_ridge_selfcorr', 'regress_ridge'),
        ('C3_ce_unweighted_knn_selfcorr', 'regress_knn'),
        ('C3_ce_unweighted', 'regress (MLP)'),
        ('C3_ce_unweighted_ridge', 'regress_ridge'),
        ('C3_ce_unweighted_knn', 'regress_knn'),
    ]:
        s = item2_downstream[k_name]
        vm = s['valsel_metrics']
        sc_flag = 1 if 'selfcorr' in k_name else 0
        print(
            f"{k_name:<32} | {sc_flag:>8d} | {imp_label:<14} | {vm['MacroF1']:8.4f} | "
            f"{vm['WeightedF1']:8.4f} | {vm['Accuracy']:8.4f} | {vm['QWK']:8.4f} | "
            f"{vm['MAE']:8.4f} | {s['train_valsel_macro_f1_gap']:+9.4f}"
        )

    print("\n[Item 3] Redefined Gap (train Macro-F1 in model.eval(), no view_dropout, hard labels) across Stage 2 Folds 0-2:")
    print(f"{'Method':<28} | {'Split':>5} | {'train(eval) F1':>17} | {'val_select F1':>17} | {'Gap (eval tr - val)':>20} | {'Old Gap (train loop)':>20}")
    print("-" * 120)
    for split_type in ['B', 'A']:
        for exp_name in ['C3_ce_unweighted_selfcorr', 'C3_ce_unweighted', 'MSSF_v2_selfcorr', 'MSSF_v2']:
            es = stage2_gap_results[f'{exp_name}_{split_type}']
            fs_list = es['fold_summaries']
            tr_eval = [float(x['best_train_macro_f1']) for x in fs_list]
            tr_loop = [float(x.get('best_train_loop_macro_f1', x['best_train_macro_f1'])) for x in fs_list]
            val_f1 = [float(x['valsel_metrics']['MacroF1']) for x in fs_list]
            new_gaps = [float(x['train_valsel_macro_f1_gap']) for x in fs_list]
            old_gaps = [tr_loop[i] - val_f1[i] for i in range(len(fs_list))]
            print(
                f"{exp_name:<28} | {split_type:>5} | "
                f"{np.mean(tr_eval):.4f} +/- {np.std(tr_eval):.4f} | "
                f"{np.mean(val_f1):.4f} +/- {np.std(val_f1):.4f} | "
                f"{np.mean(new_gaps):+.4f} +/- {np.std(new_gaps):.4f} | "
                f"{np.mean(old_gaps):+.4f} +/- {np.std(old_gaps):.4f}"
            )

    print("\n[Item 4] Multi-Seed Run-to-Run Noise on Fold 0, Split B (3 Seeds: 42, 43, 44):")
    print(f"{'Model Group':<38} | {'Seed 42':>8} | {'Seed 43':>8} | {'Seed 44':>8} | {'Mean +/- Std (Macro-F1)':>24} | {'QWK (Mean +/- Std)':>20}")
    print("-" * 120)
    for grp_name, s_list in seed_groups.items():
        f1s = [float(x['valsel_metrics']['MacroF1']) for x in s_list]
        qwks = [float(x['valsel_metrics']['QWK']) for x in s_list]
        print(
            f"{grp_name:<38} | {f1s[0]:8.4f} | {f1s[1]:8.4f} | {f1s[2]:8.4f} | "
            f"{np.mean(f1s):10.4f} +/- {np.std(f1s):.4f}   | {np.mean(qwks):8.4f} +/- {np.std(qwks):.4f}"
        )

    out_json_path = os.path.join(args.results_dir, 'stage2_followup_summary.json')
    os.makedirs(args.results_dir, exist_ok=True)
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'item2_imputation_quality_fold0_B': imp_quality,
            'item2_downstream_fold0_B': item2_downstream,
            'item3_stage2_gap_summaries': stage2_gap_results,
            'item4_seed_groups_fold0_B': seed_groups,
        }, f, indent=2)
    if args.drive_dir:
        os.makedirs(args.drive_dir, exist_ok=True)
        import shutil
        shutil.copy2(out_json_path, os.path.join(args.drive_dir, 'stage2_followup_summary.json'))
    print(f"\nSaved complete JSON report to {out_json_path}")


if __name__ == '__main__':
    main()
