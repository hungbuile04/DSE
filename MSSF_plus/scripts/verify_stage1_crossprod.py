"""
Stage 1 Equivalence Verification for `CrossProduction` (Original vs Factored):
  - Runs `MSSF_v2` on Fold 0, Split B, with identical seed (`seed=42`) in `float32`:
      1. Measures maximum single-batch output difference `max |out_orig - out_factored|`
         and `max |logits_orig - logits_factored|` at initialization on the first training batch.
      2. Trains both versions (`use_factored_crossprod=False` vs `use_factored_crossprod=True`),
         recording the `val_select` Macro-F1 curve across epochs, `best_epoch`, final `val_select`
         Macro-F1, and wall-clock time per epoch.
      3. Verifies acceptance criterion: `|MacroF1_orig - MacroF1_factored| <= 0.005`.
"""

import os
import json
import time
import copy
import argparse
import numpy as np
import torch

from MSSF_plus.protocol import (
    DEFAULT_RESULTS_V2_DIR,
    set_global_seed,
    get_device,
    load_fold_data,
)
from MSSF_plus.views import build_fold_views
from MSSF_plus.soft_labels import build_soft_targets
from MSSF_plus.data import build_dataloaders
from MSSF_plus.model import MSSFPlusModel
from MSSF_plus.losses import (
    compute_class_weights_from_train,
    logits_to_probs,
    MSSFPlusCriterion,
)
from MSSF_plus.train import evaluate_loader
from MSSF_plus.configs.experiments import get_experiment_config


def run_stage1_verification(
    fold: int = 0,
    split_type: str = 'B',
    epochs: int = 50,
    device_str: str = None,
    out_json: str = None,
) -> dict:
    device = get_device(device_str)
    base_cfg = get_experiment_config('MSSF_v2', epochs=epochs)
    fold_seed = int(base_cfg['seed']) + int(fold)

    set_global_seed(fold_seed)
    fold_split = load_fold_data(fold, split_type)
    fold_views = build_fold_views(fold_split, base_cfg, device='cpu')
    soft_res = build_soft_targets(fold_split, fold_views, base_cfg)

    n_drug_views = len(fold_views.drug_view_names)
    n_se_views = len(fold_views.se_view_names)

    # 1. Measure single-batch output difference at identical weights
    set_global_seed(fold_seed)
    train_loader, valsel_loader, _, _ = build_dataloaders(
        fold_split, fold_views, soft_res, base_cfg
    )
    first_batch = next(iter(train_loader))
    drugs_b = first_batch['drug_feat'].to(device, dtype=torch.float32)
    sides_b = first_batch['se_feat'].to(device, dtype=torch.float32)

    cfg_orig = copy.deepcopy(base_cfg)
    cfg_orig['use_factored_crossprod'] = False
    cfg_fact = copy.deepcopy(base_cfg)
    cfg_fact['use_factored_crossprod'] = True

    set_global_seed(fold_seed)
    model_orig = MSSFPlusModel(
        cfg=cfg_orig, n_drug_views=n_drug_views, n_se_views=n_se_views
    ).to(device)
    model_fact = MSSFPlusModel(
        cfg=cfg_fact, n_drug_views=n_drug_views, n_se_views=n_se_views
    ).to(device)
    model_fact.load_state_dict(copy.deepcopy(model_orig.state_dict()))

    model_orig.eval()
    model_fact.eval()
    with torch.no_grad():
        prep_d, prep_s = model_orig.preprocess(drugs_b, sides_b)
        cp_orig = model_orig.crossProduction(prep_d, prep_s)
        cp_fact = model_fact.crossProduction(prep_d, prep_s)
        max_cp_diff_f32 = float(torch.max(torch.abs(cp_orig - cp_fact)).item())

        logits_orig, *_ = model_orig(drugs_b, sides_b)
        logits_fact, *_ = model_fact(drugs_b, sides_b)
        max_logits_diff_f32 = float(torch.max(torch.abs(logits_orig - logits_fact)).item())

    print(
        f"[Stage 1 Single-Batch Check (float32, B={drugs_b.shape[0]}, device={device})]\n"
        f"  max |CrossProduction_orig - CrossProduction_factored| = {max_cp_diff_f32:.3e}\n"
        f"  max |logits_orig - logits_factored|                   = {max_logits_diff_f32:.3e}"
    )

    # 2. Train both versions for `epochs` with identical seed
    results = {
        'fold': int(fold),
        'split_type': split_type,
        'seed': int(fold_seed),
        'epochs': int(epochs),
        'device': str(device),
        'single_batch_max_crossprod_diff_float32': max_cp_diff_f32,
        'single_batch_max_logits_diff_float32': max_logits_diff_f32,
        'runs': {},
    }

    class_weights = compute_class_weights_from_train(fold_split.train)

    for label, cfg_run in [('original', cfg_orig), ('factored', cfg_fact)]:
        set_global_seed(fold_seed)
        tr_loader, vs_loader, _, _ = build_dataloaders(
            fold_split, fold_views, soft_res, cfg_run
        )
        model = MSSFPlusModel(
            cfg=cfg_run,
            n_drug_views=n_drug_views,
            n_se_views=n_se_views,
            profile_drug_view_indices=fold_views.profile_drug_view_indices,
            profile_se_view_indices=fold_views.profile_se_view_indices,
            drug_side_train=fold_views.drug_side_train,
            n_clusters=soft_res.n_clusters,
        ).to(device)
        criterion = MSSFPlusCriterion(
            cfg=cfg_run,
            n_drug_views=n_drug_views,
            n_se_views=n_se_views,
            class_weights=class_weights,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(cfg_run['lr']),
            weight_decay=float(cfg_run['weight_decay']),
        )

        best_epoch = 1
        best_valsel_f1 = -float('inf')
        best_valsel_metrics = None
        curve_f1 = []
        epoch_times = []

        print(f"\n--- Running MSSF_v2 ({label} CrossProduction) for {epochs} epochs ---")
        for ep in range(1, epochs + 1):
            t0 = time.time()
            model.train()
            for batch in tr_loader:
                drugs = batch['drug_feat'].to(device)
                sides = batch['se_feat'].to(device)
                d_avail = batch['drug_avail'].to(device)
                s_avail = batch['se_avail'].to(device)
                fb_d9 = batch['fallback_d9'].to(device)
                fb_d10 = batch['fallback_d10'].to(device)
                hard_labels = batch['label'].to(device)
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
                    soft_targets=None,
                    cluster_logits=cluster_logits,
                    cluster_ids=cluster_ids,
                    epoch=ep,
                )
                loss.backward()
                optimizer.step()

            vs_metrics, _, _, _ = evaluate_loader(
                model, vs_loader, loss_type='ce', device=device,
                expected_pairs=fold_split.valsel,
            )
            ep_time = time.time() - t0
            epoch_times.append(float(ep_time))
            f1 = float(vs_metrics['MacroF1'])
            curve_f1.append(f1)

            if f1 > best_valsel_f1:
                best_valsel_f1 = f1
                best_epoch = ep
                best_valsel_metrics = vs_metrics

            if ep == 1 or ep % 10 == 0 or ep == epochs:
                print(
                    f"  [{label}] Ep {ep:02d}/{epochs:02d} | valsel MacroF1={f1:.4f} "
                    f"(best={best_valsel_f1:.4f} @ ep {best_epoch}) | {ep_time:.2f}s/ep"
                )

        results['runs'][label] = {
            'best_epoch': int(best_epoch),
            'best_valsel_macro_f1': float(best_valsel_f1),
            'best_valsel_metrics': best_valsel_metrics,
            'valsel_macro_f1_curve': curve_f1,
            'mean_epoch_seconds': float(np.mean(epoch_times)),
            'std_epoch_seconds': float(np.std(epoch_times)),
            'total_seconds': float(np.sum(epoch_times)),
        }

    f1_orig = results['runs']['original']['best_valsel_macro_f1']
    f1_fact = results['runs']['factored']['best_valsel_macro_f1']
    f1_diff = abs(f1_orig - f1_fact)
    curve_orig = np.array(results['runs']['original']['valsel_macro_f1_curve'])
    curve_fact = np.array(results['runs']['factored']['valsel_macro_f1_curve'])
    max_curve_diff = float(np.max(np.abs(curve_orig - curve_fact)))

    results['abs_best_macro_f1_diff'] = float(f1_diff)
    results['max_epoch_curve_macro_f1_diff'] = max_curve_diff
    results['accepted_le_0_005'] = bool(f1_diff <= 0.005)

    print(
        f"\n=== Stage 1 Equivalence Summary ===\n"
        f"  Single-batch CrossProduction max diff (float32): {max_cp_diff_f32:.3e}\n"
        f"  Original best valsel MacroF1 : {f1_orig:.6f} @ ep {results['runs']['original']['best_epoch']} "
        f"({results['runs']['original']['mean_epoch_seconds']:.2f}s/ep)\n"
        f"  Factored best valsel MacroF1 : {f1_fact:.6f} @ ep {results['runs']['factored']['best_epoch']} "
        f"({results['runs']['factored']['mean_epoch_seconds']:.2f}s/ep)\n"
        f"  |MacroF1_orig - MacroF1_fact|: {f1_diff:.6f} (<= 0.005: {results['accepted_le_0_005']})\n"
        f"  Max curve diff across epochs : {max_curve_diff:.6f}"
    )

    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"Saved Stage 1 verification JSON to {out_json}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Stage 1 CrossProduction Equivalence Check')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--split', type=str, default='B', choices=['A', 'B'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument(
        '--out_json',
        type=str,
        default=os.path.join(DEFAULT_RESULTS_V2_DIR, 'stage1_crossprod_equivalence.json'),
    )
    args = parser.parse_args()

    run_stage1_verification(
        fold=args.fold,
        split_type=args.split,
        epochs=args.epochs,
        device_str=args.device,
        out_json=args.out_json,
    )


if __name__ == '__main__':
    main()
