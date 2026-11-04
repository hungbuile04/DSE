"""
Verification script for Items 1 and 2:
  1. Item 1: Instantiate `MSSF_v2` and `C3_ce_unweighted` (plus `C3_ce_unweighted_ridge_selfcorr`),
     print the exact list of drug views and SE views (name, matrix shape, input vector dimension),
     and print exact submodule parameter shapes (`drug_modulator`, `se_modulator`, `encoderConnection`,
     `encoderAddition`, `preprocess`, `crossProduction`) directly from the live PyTorch model objects.
     Also print exact cross-fit K=5 drug counts on Split B and Split A.
  2. Item 2: Run `C3_ce_unweighted_ridge_selfcorr` (`c1_impute='regress_ridge'`, `profile_self_correction=1`)
     on Split B Folds 0–2 (`resume=True`, so Fold 0 is reused and Folds 1–2 are trained/saved),
     and compare `Macro-F1`, `QWK`, `MAE`, `Weighted-F1`, `Accuracy`, and `Gap(eval)` across Folds 0–2
     against `C3_ce_unweighted_selfcorr` (`regress` MLP) and `MSSF_v2_selfcorr`.
"""

import os
import json
import argparse
from typing import Dict, List

import numpy as np
import torch

from MSSF_plus.protocol import (
    DEFAULT_RESULTS_V2_DIR,
    N_DRUGS,
    N_SIDES,
    get_device,
    load_fold_data,
)
from MSSF_plus.views import (
    build_fold_views,
    _make_drug_crossfit_splits,
)
from MSSF_plus.model import MSSFPlusModel
from MSSF_plus.train import run_experiment, _sync_method_dir
from MSSF_plus.metrics import compute_multiclass_metrics, aggregate_fold_metrics
from MSSF_plus.configs.experiments import get_experiment_config


def verify_instantiated_views_and_modules() -> None:
    print("=" * 110)
    print("=== ITEM 1: VERIFY VIEWS & MODULE DIMENSIONS FROM INSTANTIATED PYTORCH MODELS ===")
    print("=" * 110)

    fold0_b = load_fold_data(0, 'B')
    fold0_a = load_fold_data(0, 'A')

    for sp_name, sp_obj in [('B (Cold-Start Drug)', fold0_b), ('A (Warm-Start Pair)', fold0_a)]:
        tr_idx, cf_folds = _make_drug_crossfit_splits(sp_obj.train_drugs, n_splits=5, val_frac=0.10, seed=42)
        cf_tuples = [(len(f), len(v), len(h)) for f, v, h, _ in cf_folds]
        print(
            f"[Cross-Fit K=5 Drug Counts | Split {sp_name}] "
            f"total_train_drugs={len(tr_idx)} / {N_DRUGS} | "
            f"(fit_sub, val_sub, holdout) per split = {cf_tuples}"
        )
    print()

    for exp_name in ['MSSF_v2', 'C3_ce_unweighted', 'C3_ce_unweighted_ridge_selfcorr']:
        cfg = get_experiment_config(exp_name)
        # Use regress_ridge during view shape inspection so we don't spend 30s training MLP just for view inspection
        cfg_inspect = dict(cfg)
        if cfg_inspect.get('c1_impute') == 'regress':
            cfg_inspect['c1_impute'] = 'regress_ridge'

        fv = build_fold_views(fold0_b, cfg_inspect, device=torch.device('cpu'))
        model = MSSFPlusModel(
            cfg=cfg,
            n_drug_views=len(fv.drug_view_names),
            n_se_views=len(fv.se_view_names),
            profile_drug_view_indices=fv.profile_drug_view_indices,
            profile_se_view_indices=fv.profile_se_view_indices,
            drug_side_train=fv.drug_side_train,
            n_clusters=0,
        )

        tot_params = sum(p.numel() for p in model.parameters())
        print("-" * 110)
        print(
            f"Model: {exp_name} | Total Params: {tot_params:,} | "
            f"n_drug_views={model.n_drug_views} | n_se_views={model.n_se_views} | "
            f"c1_impute={cfg.get('c1_impute')} | c2_se_mode={cfg.get('c2_se_mode')}"
        )
        print(f"  Concatenated drug feature vector dim per drug : {fv.drug_features_matrix.shape[1]} "
              f"(matrix shape={fv.drug_features_matrix.shape})")
        print(f"  Concatenated SE feature vector dim per SE     : {fv.se_features_matrix.shape[1]} "
              f"(matrix shape={fv.se_features_matrix.shape})")

        print(f"  Drug Views ({len(fv.drug_view_names)} views, {len(model.preprocess.drug_layers)} Preprocess branches):")
        for i, (vname, vmat, branch) in enumerate(
            zip(fv.drug_view_names, fv.drug_views_list, model.preprocess.drug_layers)
        ):
            lin = branch[0]
            is_prof = i in fv.profile_drug_view_indices
            tag = " [profile view]" if is_prof else " [static view]"
            print(
                f"    DrugView[{i:02d}]: {vname:<24} | view_mat={str(vmat.shape):<12} | "
                f"input_vec_dim={vmat.shape[1]} | Preprocess Linear({lin.in_features} -> {lin.out_features}){tag}"
            )

        print(f"  SE Views ({len(fv.se_view_names)} views, {len(model.preprocess.side_layers)} Preprocess branches):")
        for j, (vname, vmat, branch) in enumerate(
            zip(fv.se_view_names, fv.se_views_list, model.preprocess.side_layers)
        ):
            lin = branch[0]
            is_prof = (fv.profile_se_view_indices is not None) and (j in fv.profile_se_view_indices)
            tag = " [profile view]" if is_prof else " [static/ontology view]"
            print(
                f"    SEView  [{j:02d}]: {vname:<24} | view_mat={str(vmat.shape):<12} | "
                f"input_vec_dim={vmat.shape[1]} | Preprocess Linear({lin.in_features} -> {lin.out_features}){tag}"
            )

        print("  Instantiated Submodule Parameter & Layer Verification:")
        for name, child in model.named_children():
            n_p = sum(p.numel() for p in child.parameters())
            print(f"    - {name:<22} ({child.__class__.__name__:<22}): {n_p:>10,} params")
        if getattr(model, 'drug_modulator', None) is not None:
            dm = model.drug_modulator
            print(
                f"      * drug_modulator details: missing_token={tuple(dm.missing_token.shape)} "
                f"({dm.missing_token.numel()} params), gate_fc1={dm.gate_fc1}, gate_fc2={dm.gate_fc2}"
            )
        if getattr(model, 'se_modulator', None) is not None:
            sm = model.se_modulator
            print(
                f"      * se_modulator details  : missing_token={tuple(sm.missing_token.shape)} "
                f"({sm.missing_token.numel()} params), gate_fc1={sm.gate_fc1}, gate_fc2={sm.gate_fc2}"
            )
        print(
            f"      * encoderConnection     : l1[0]={model.encoderConnection.l1[0]}, "
            f"l3[3]={model.encoderConnection.l3[3]}"
        )
        print(
            f"      * encoderAddition       : l1[0]={model.encoderAddition.l1[0]}, "
            f"l3[3]={model.encoderAddition.l3[3]}"
        )
        print(
            f"      * preprocess            : {len(model.preprocess.drug_layers)} drug_layers + "
            f"{len(model.preprocess.side_layers)} side_layers = "
            f"{len(model.preprocess.drug_layers) + len(model.preprocess.side_layers)} total branches"
        )
        print(f"      * crossProduction       : cnn[0]={model.crossProduction.cnn[0]}")
    print("=" * 110)
    print()


def _load_dev_fold_summary(results_dir: str, exp_name: str, split_type: str = 'B', folds: List[int] = [0, 1, 2]) -> Dict:
    mdir = os.path.join(results_dir, f"{exp_name}_{split_type}")
    cfg_path = os.path.join(mdir, 'config.json')
    cfg_json = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r') as f:
            cfg_json = json.load(f)
    fold_records = cfg_json.get('summary', {}).get('fold_records', {})

    per_fold = {}
    for k in folds:
        v_lbl_p = os.path.join(mdir, f"fold_{k}_valsel_labels.npy")
        v_prb_p = os.path.join(mdir, f"fold_{k}_valsel_probs.npy")
        if not (os.path.exists(v_lbl_p) and os.path.exists(v_prb_p)):
            continue
        y_true = np.load(v_lbl_p).astype(int).flatten()
        probs = np.load(v_prb_p).astype(np.float32)
        y_pred = np.argmax(probs, axis=1)
        m = compute_multiclass_metrics(y_true, y_pred, probs=probs)
        rec = fold_records.get(str(k), fold_records.get(k, {}))
        m['best_epoch'] = rec.get('best_epoch', -1)
        m['best_train_macro_f1'] = rec.get('best_train_macro_f1', float('nan'))
        m['gap_eval'] = (
            float(m['best_train_macro_f1']) - float(m['MacroF1'])
            if not np.isnan(m.get('best_train_macro_f1', float('nan')))
            else float(rec.get('train_valsel_macro_f1_gap', float('nan')))
        )
        m['elapsed_sec'] = rec.get('elapsed_sec', float('nan'))
        per_fold[k] = m
    return per_fold


def run_item2_ridge_vs_mlp_folds012(
    results_dir: str,
    drive_dir: str,
    device_str: str,
) -> None:
    print("=" * 110)
    print("=== ITEM 2: RUN C3_ce_unweighted_ridge_selfcorr ON FOLDS 0-2, SPLIT B & COMPARE VS MLP ===")
    print("=" * 110)

    for exp_n in ['C3_ce_unweighted_ridge_selfcorr', 'C3_ce_unweighted_selfcorr', 'MSSF_v2_selfcorr']:
        _sync_method_dir(results_dir, drive_dir, f"{exp_n}_B", pull_from_drive=True)

    run_experiment(
        exp_name='C3_ce_unweighted_ridge_selfcorr',
        split_type='B',
        mode='dev',
        custom_folds=[0, 1, 2],
        results_dir=results_dir,
        drive_dir=drive_dir,
        prefer_device=device_str,
        eval_test=False,
        resume=True,
    )

    compare_models = [
        ('C3_ce_unweighted_ridge_selfcorr', 'regress_ridge (K=5)'),
        ('C3_ce_unweighted_selfcorr', 'regress MLP (K=5)'),
        ('MSSF_v2_selfcorr', 'none (zero)'),
    ]

    print("\n=== Item 2A: Per-Fold Comparison on Split B (Folds 0, 1, 2 | val_select | SelfCorr=1) ===")
    header = (
        f"{'Model':<33} | {'Imputer':<20} | {'Fold':>4} | {'Ep':>3} | "
        f"{'MacroF1':>8} | {'QWK':>8} | {'MAE':>8} | {'WeightF1':>8} | {'Acc':>8} | {'Gap(eval)':>9} | "
        f"{'F1_c0':>6} {'F1_c1':>6} {'F1_c2':>6} {'F1_c3':>6} {'F1_c4':>6}"
    )
    print(header)
    print("-" * len(header))

    loaded_all = {}
    for exp_n, imp_label in compare_models:
        pf = _load_dev_fold_summary(results_dir, exp_n, split_type='B', folds=[0, 1, 2])
        loaded_all[exp_n] = (imp_label, pf)
        for k in sorted(pf.keys()):
            m = pf[k]
            f1s = m['PerClassF1']
            print(
                f"{exp_n:<33} | {imp_label:<20} | {k:>4d} | {int(m['best_epoch']):>3d} | "
                f"{m['MacroF1']:>8.4f} | {m['QWK']:>8.4f} | {m['MAE']:>8.4f} | "
                f"{m['WeightedF1']:>8.4f} | {m['Accuracy']:>8.4f} | {m['gap_eval']:>+9.4f} | "
                f"{f1s[0]:>6.4f} {f1s[1]:>6.4f} {f1s[2]:>6.4f} {f1s[3]:>6.4f} {f1s[4]:>6.4f}"
            )

    print("\n=== Item 2B: Summary Mean +/- Std Across Folds 0-2, Split B (val_select | SelfCorr=1) ===")
    sum_header = (
        f"{'Model':<33} | {'Imputer':<20} | {'Folds':>5} | "
        f"{'Macro-F1 (Mean+/-Std)':>21} | {'QWK (Mean+/-Std)':>18} | {'MAE (Mean+/-Std)':>18} | "
        f"{'Weighted-F1':>18} | {'Gap(eval)':>18}"
    )
    print(sum_header)
    print("-" * len(sum_header))
    for exp_n, (imp_label, pf) in loaded_all.items():
        ks = sorted(pf.keys())
        if not ks:
            continue
        mf1 = np.array([pf[k]['MacroF1'] for k in ks])
        qwk = np.array([pf[k]['QWK'] for k in ks])
        mae = np.array([pf[k]['MAE'] for k in ks])
        wf1 = np.array([pf[k]['WeightedF1'] for k in ks])
        gap = np.array([pf[k]['gap_eval'] for k in ks])
        print(
            f"{exp_n:<33} | {imp_label:<20} | {len(ks):>5d} | "
            f"{mf1.mean():>8.4f} +/- {mf1.std():<8.4f} | "
            f"{qwk.mean():>6.4f} +/- {qwk.std():<6.4f} | "
            f"{mae.mean():>6.4f} +/- {mae.std():<6.4f} | "
            f"{wf1.mean():>6.4f} +/- {wf1.std():<6.4f} | "
            f"{gap.mean():>+6.4f} +/- {gap.std():<6.4f}"
        )
    print("=" * 110)


def main():
    parser = argparse.ArgumentParser(description="Verify views (Item 1) and run regress_ridge Folds 0-2 on Split B (Item 2)")
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_V2_DIR)
    parser.add_argument('--drive_dir', type=str, default='/content/drive/MyDrive/DSE_results_v2')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    dev = get_device(args.device)
    verify_instantiated_views_and_modules()
    run_item2_ridge_vs_mlp_folds012(
        results_dir=args.results_dir,
        drive_dir=args.drive_dir,
        device_str=str(dev),
    )


if __name__ == '__main__':
    main()
