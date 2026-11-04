"""
Experiment configurations for MSSF_plus covering:
  - MSSF_v2 (leak-free baseline under unified protocol)
  - C1 ablations (availability indicator, view dropout, zero/drop/mean/regress/diffusion imputation,
    and `C1_regress_no_modulator` without AvailabilityModulator)
  - C2 ablations (original 4 SE views, +MedDRA ontology 5 SE views, no-profile 3 SE views)
  - C3 ordinal objective selection (CE, EMD, CORN, each with/without inverse-frequency class weighting,
    with C1/C2/C5 held as in MSSF_plus_full)
  - C5 ablations (none, kNN, single-view spectral, multi-view spectral, +auxiliary cluster head)
  - MSSF_plus_full and Leave-One-Out (LOO) ablations (`loo_no_C1`, `loo_no_C2`, `loo_no_C3`, `loo_no_C5`)
"""

import copy
from typing import Dict

BASE_CONFIG: Dict[str, object] = {
    'exp_name': 'MSSF_v2',
    'seed': 42,
    # Optimization & architecture defaults (exact match with MSSF/mssf.py & notebook1)
    'epochs': 50,
    'batch_size': 512,
    'eval_batch_size': 512,
    'lr': 0.0001,
    'weight_decay': 1e-5,
    'dropout': 0.4,
    'gp': 64,
    'kl_weight': 0.001,
    'rec_con_weight': 0.0001,
    'rec_add_weight': 0.0001,
    'selection_metric': 'MacroF1',
    # C1: Missing view handling & K-fold cross-fitted imputation
    'c1_use_availability': False,
    'c1_view_dropout_prob': 0.0,
    'c1_impute': 'zero',            # 'zero' | 'drop' | 'mean' | 'regress' | 'diffusion'
    'c1_crossfit_folds': 5,
    'c1_imputer_epochs': 80,
    'c1_diffusion_epochs': 100,
    'c1_diffusion_steps': 25,
    'c1_diffusion_samples': 10,
    # C2: SE ontology view
    'c2_se_mode': 'original',       # 'original' | 'plus_ontology' | 'no_profile'
    # C3: Ordinal loss & class weighting ('ce' | 'emd' | 'corn')
    'c3_loss_type': 'ce',
    'c3_class_weighting': False,
    # C4: Ordinal conformal prediction (post-hoc on valcal -> test)
    'c4_alpha': 0.10,
    # C5: Multi-view spectral soft labels (auxiliary cluster head off by default)
    'c5_mode': 'none',              # 'none' | 'knn' | 'single_view' | 'multi_view'
    'c5_alpha': 0.15,
    'c5_min_samples': 5,
    'c5_knn_k': 15,
    'c5_k_min': 4,
    'c5_k_max': 12,
    'c5_single_view_name': 'D5_stitch_five',
    'c5_view_weighting': 'uniform', # 'uniform' | 'quality'
    'c5_aux_cluster_weight': 0.0,
    # Stage 1, 2 & 4 flags (defaults preserve exact legacy behavior)
    'use_factored_crossprod': True,
    'profile_self_correction': False,
    'use_bvi': True,
}


def _make_cfg(name: str, **overrides) -> Dict[str, object]:
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg['exp_name'] = name
    cfg.update(overrides)
    return cfg


# Common C1 + C2 + C5 settings shared by MSSF_plus_full and the 6 C3 loss-selection configurations
_FULL_NON_C3_KWARGS: Dict[str, object] = {
    'c1_use_availability': True,
    'c1_view_dropout_prob': 0.25,
    'c1_impute': 'regress',
    'c2_se_mode': 'plus_ontology',
    'c5_mode': 'multi_view',
    'c5_alpha': 0.15,
    'c5_aux_cluster_weight': 0.0,
}


EXPERIMENTS: Dict[str, Dict[str, object]] = {
    # 0. Leak-free baseline
    'MSSF_v2': _make_cfg('MSSF_v2'),

    # 1. C1 — Missing View Handling & Cross-Fitted Cold-Start Imputation
    'C1_avail_only': _make_cfg(
        'C1_avail_only',
        c1_use_availability=True,
        c1_view_dropout_prob=0.0,
        c1_impute='zero',
    ),
    'C1_vdrop_only': _make_cfg(
        'C1_vdrop_only',
        c1_use_availability=False,
        c1_view_dropout_prob=0.25,
        c1_impute='zero',
    ),
    'C1_avail_vdrop_zero': _make_cfg(
        'C1_avail_vdrop_zero',
        c1_use_availability=True,
        c1_view_dropout_prob=0.25,
        c1_impute='zero',
    ),
    'C1_avail_vdrop_drop': _make_cfg(
        'C1_avail_vdrop_drop',
        c1_use_availability=True,
        c1_view_dropout_prob=0.0,
        c1_impute='drop',
    ),
    'C1_avail_vdrop_mean': _make_cfg(
        'C1_avail_vdrop_mean',
        c1_use_availability=True,
        c1_view_dropout_prob=0.25,
        c1_impute='mean',
    ),
    'C1_avail_vdrop_regress': _make_cfg(
        'C1_avail_vdrop_regress',
        c1_use_availability=True,
        c1_view_dropout_prob=0.25,
        c1_impute='regress',
    ),
    'C1_avail_vdrop_ridge': _make_cfg(
        'C1_avail_vdrop_ridge',
        c1_use_availability=True,
        c1_view_dropout_prob=0.25,
        c1_impute='regress_ridge',
    ),
    'C1_avail_vdrop_knn': _make_cfg(
        'C1_avail_vdrop_knn',
        c1_use_availability=True,
        c1_view_dropout_prob=0.25,
        c1_impute='regress_knn',
    ),
    'C1_regress_no_modulator': _make_cfg(
        'C1_regress_no_modulator',
        c1_use_availability=False,
        c1_view_dropout_prob=0.25,
        c1_impute='regress',
    ),
    # Kept strictly as a single ablation line on dev folds (excluded from MSSF_plus_full, tuning, and 10-fold run)
    'C1_avail_vdrop_diffusion': _make_cfg(
        'C1_avail_vdrop_diffusion',
        c1_use_availability=True,
        c1_view_dropout_prob=0.25,
        c1_impute='diffusion',
    ),

    # 2. C2 — SE MedDRA Ontology View
    'C2_plus_ontology': _make_cfg(
        'C2_plus_ontology',
        c2_se_mode='plus_ontology',
    ),
    'C2_no_profile': _make_cfg(
        'C2_no_profile',
        c2_se_mode='no_profile',
    ),

    # 3. C3 — Ordinal Objective Selection ({ce, emd, corn} x {unweighted, weighted})
    # Other components held as in MSSF_plus_full per Change 2
    'C3_ce_unweighted': _make_cfg(
        'C3_ce_unweighted',
        **_FULL_NON_C3_KWARGS,
        c3_loss_type='ce',
        c3_class_weighting=False,
    ),
    'C3_ce_weighted': _make_cfg(
        'C3_ce_weighted',
        **_FULL_NON_C3_KWARGS,
        c3_loss_type='ce',
        c3_class_weighting=True,
    ),
    'C3_emd_unweighted': _make_cfg(
        'C3_emd_unweighted',
        **_FULL_NON_C3_KWARGS,
        c3_loss_type='emd',
        c3_class_weighting=False,
    ),
    'C3_emd_weighted': _make_cfg(
        'C3_emd_weighted',
        **_FULL_NON_C3_KWARGS,
        c3_loss_type='emd',
        c3_class_weighting=True,
    ),
    'C3_corn_unweighted': _make_cfg(
        'C3_corn_unweighted',
        **_FULL_NON_C3_KWARGS,
        c3_loss_type='corn',
        c3_class_weighting=False,
    ),
    'C3_corn_weighted': _make_cfg(
        'C3_corn_weighted',
        **_FULL_NON_C3_KWARGS,
        c3_loss_type='corn',
        c3_class_weighting=True,
    ),

    # 4. C5 — Soft Label Generation
    'C5_knn': _make_cfg(
        'C5_knn',
        c5_mode='knn',
        c5_alpha=0.15,
    ),
    'C5_single_view': _make_cfg(
        'C5_single_view',
        c5_mode='single_view',
        c5_alpha=0.15,
    ),
    'C5_multi_view': _make_cfg(
        'C5_multi_view',
        c5_mode='multi_view',
        c5_alpha=0.15,
        c5_aux_cluster_weight=0.0,
    ),
    'C5_multi_view_aux': _make_cfg(
        'C5_multi_view_aux',
        c5_mode='multi_view',
        c5_alpha=0.15,
        c5_aux_cluster_weight=0.05,
    ),

    # 5. Full MSSF_plus Model & Leave-One-Out (LOO) Ablations
    # (Cluster head off by default; C3 loss finalized from fold 0-2 val_select comparison)
    'MSSF_plus_full': _make_cfg(
        'MSSF_plus_full',
        c1_use_availability=True,
        c1_view_dropout_prob=0.25,
        c1_impute='regress',
        c2_se_mode='plus_ontology',
        c3_loss_type='ce',
        c3_class_weighting=True,
        c5_mode='multi_view',
        c5_alpha=0.15,
        c5_aux_cluster_weight=0.0,
    ),
    'loo_no_C1': _make_cfg(
        'loo_no_C1',
        c1_use_availability=False,
        c1_view_dropout_prob=0.0,
        c1_impute='zero',
        c2_se_mode='plus_ontology',
        c3_loss_type='ce',
        c3_class_weighting=True,
        c5_mode='multi_view',
        c5_alpha=0.15,
        c5_aux_cluster_weight=0.0,
    ),
    'loo_no_C2': _make_cfg(
        'loo_no_C2',
        c1_use_availability=True,
        c1_view_dropout_prob=0.25,
        c1_impute='regress',
        c2_se_mode='original',
        c3_loss_type='ce',
        c3_class_weighting=True,
        c5_mode='multi_view',
        c5_alpha=0.15,
        c5_aux_cluster_weight=0.0,
    ),
    'loo_no_C3': _make_cfg(
        'loo_no_C3',
        c1_use_availability=True,
        c1_view_dropout_prob=0.25,
        c1_impute='regress',
        c2_se_mode='plus_ontology',
        c3_loss_type='ce',
        c3_class_weighting=False,
        c5_mode='multi_view',
        c5_alpha=0.15,
        c5_aux_cluster_weight=0.0,
    ),
    'loo_no_C5': _make_cfg(
        'loo_no_C5',
        c1_use_availability=True,
        c1_view_dropout_prob=0.25,
        c1_impute='regress',
        c2_se_mode='plus_ontology',
        c3_loss_type='ce',
        c3_class_weighting=True,
        c5_mode='none',
        c5_aux_cluster_weight=0.0,
    ),
    # Stage 4: BVI ablation (replaces GaussianParametrizer with Linear(384 -> 64) + Dropout, drops KL)
    'full_no_bvi': _make_cfg(
        'full_no_bvi',
        **_FULL_NON_C3_KWARGS,
        c3_loss_type='ce',
        c3_class_weighting=False,
        use_bvi=False,
    ),
    # Stage 2: Profile self-correction comparison configurations
    'MSSF_v2_selfcorr': _make_cfg(
        'MSSF_v2_selfcorr',
        profile_self_correction=True,
    ),
    'C3_ce_unweighted_selfcorr': _make_cfg(
        'C3_ce_unweighted_selfcorr',
        **_FULL_NON_C3_KWARGS,
        c3_loss_type='ce',
        c3_class_weighting=False,
        profile_self_correction=True,
    ),
    # Cheap C1 imputation variants on MSSF_plus (`C3_ce_unweighted` with `regress_ridge` / `regress_knn`)
    'C3_ce_unweighted_ridge': _make_cfg(
        'C3_ce_unweighted_ridge',
        **{**_FULL_NON_C3_KWARGS, 'c1_impute': 'regress_ridge'},
        c3_loss_type='ce',
        c3_class_weighting=False,
        profile_self_correction=False,
    ),
    'C3_ce_unweighted_ridge_selfcorr': _make_cfg(
        'C3_ce_unweighted_ridge_selfcorr',
        **{**_FULL_NON_C3_KWARGS, 'c1_impute': 'regress_ridge'},
        c3_loss_type='ce',
        c3_class_weighting=False,
        profile_self_correction=True,
    ),
    'C3_ce_unweighted_knn': _make_cfg(
        'C3_ce_unweighted_knn',
        **{**_FULL_NON_C3_KWARGS, 'c1_impute': 'regress_knn'},
        c3_loss_type='ce',
        c3_class_weighting=False,
        profile_self_correction=False,
    ),
    'C3_ce_unweighted_knn_selfcorr': _make_cfg(
        'C3_ce_unweighted_knn_selfcorr',
        **{**_FULL_NON_C3_KWARGS, 'c1_impute': 'regress_knn'},
        c3_loss_type='ce',
        c3_class_weighting=False,
        profile_self_correction=True,
    ),
    # Multi-seed run-to-run variance estimation configs (seed=43, 44; seed=42 is the default config)
    'MSSF_v2_s43': _make_cfg(
        'MSSF_v2_s43',
        seed=43,
        profile_self_correction=False,
    ),
    'MSSF_v2_s44': _make_cfg(
        'MSSF_v2_s44',
        seed=44,
        profile_self_correction=False,
    ),
    'MSSF_v2_selfcorr_s43': _make_cfg(
        'MSSF_v2_selfcorr_s43',
        seed=43,
        profile_self_correction=True,
    ),
    'MSSF_v2_selfcorr_s44': _make_cfg(
        'MSSF_v2_selfcorr_s44',
        seed=44,
        profile_self_correction=True,
    ),
    'C3_ce_unweighted_s43': _make_cfg(
        'C3_ce_unweighted_s43',
        **_FULL_NON_C3_KWARGS,
        seed=43,
        c3_loss_type='ce',
        c3_class_weighting=False,
        profile_self_correction=False,
    ),
    'C3_ce_unweighted_s44': _make_cfg(
        'C3_ce_unweighted_s44',
        **_FULL_NON_C3_KWARGS,
        seed=44,
        c3_loss_type='ce',
        c3_class_weighting=False,
        profile_self_correction=False,
    ),
    'C3_ce_unweighted_selfcorr_s43': _make_cfg(
        'C3_ce_unweighted_selfcorr_s43',
        **_FULL_NON_C3_KWARGS,
        seed=43,
        c3_loss_type='ce',
        c3_class_weighting=False,
        profile_self_correction=True,
    ),
    'C3_ce_unweighted_selfcorr_s44': _make_cfg(
        'C3_ce_unweighted_selfcorr_s44',
        **_FULL_NON_C3_KWARGS,
        seed=44,
        c3_loss_type='ce',
        c3_class_weighting=False,
        profile_self_correction=True,
    ),
}


def get_experiment_config(exp_name: str, **cli_overrides) -> Dict[str, object]:
    """Return a deep copy of a named experiment configuration with optional overrides."""
    if exp_name not in EXPERIMENTS:
        available = ', '.join(sorted(EXPERIMENTS.keys()))
        raise KeyError(f"Unknown experiment {exp_name!r}. Available: {available}")
    cfg = copy.deepcopy(EXPERIMENTS[exp_name])
    for k, v in cli_overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg

