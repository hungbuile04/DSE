"""
Experiment configurations for MSSF_plus covering:
  - MSSF_v2 (leak-free baseline under unified protocol)
  - C1 ablations (availability indicator, view dropout, zero/drop/mean/regress/diffusion imputation)
  - C2 ablations (original 4 SE views, +MedDRA ontology 5 SE views, no-profile 3 SE views)
  - C3 ablations (CE, EMD, CORN, CE+SoftQuant, each with/without inverse-frequency class weighting)
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
    # C1: Missing view handling
    'c1_use_availability': False,
    'c1_view_dropout_prob': 0.0,
    'c1_impute': 'zero',            # 'zero' | 'drop' | 'mean' | 'regress' | 'diffusion'
    'c1_imputer_epochs': 80,
    'c1_diffusion_steps': 25,
    'c1_diffusion_samples': 10,
    # C2: SE ontology view
    'c2_se_mode': 'original',       # 'original' | 'plus_ontology' | 'no_profile'
    # C3: Ordinal loss & class weighting
    'c3_loss_type': 'ce',           # 'ce' | 'emd' | 'corn' | 'ce_softquant'
    'c3_class_weighting': False,
    'c3_softquant_weight': 0.10,
    'c3_gamma_min': 1.0,
    'c3_gamma_max': 10.0,
    # C4: Ordinal conformal prediction (post-hoc on valcal -> test)
    'c4_alpha': 0.10,
    # C5: Multi-view spectral soft labels
    'c5_mode': 'none',              # 'none' | 'knn' | 'single_view' | 'multi_view'
    'c5_alpha': 0.15,
    'c5_min_samples': 5,
    'c5_knn_k': 15,
    'c5_k_min': 4,
    'c5_k_max': 12,
    'c5_single_view_name': 'D5_stitch_five',
    'c5_view_weighting': 'uniform', # 'uniform' | 'quality'
    'c5_aux_cluster_weight': 0.0,
}


def _make_cfg(name: str, **overrides) -> Dict[str, object]:
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg['exp_name'] = name
    cfg.update(overrides)
    return cfg


EXPERIMENTS: Dict[str, Dict[str, object]] = {
    # 0. Leak-free baseline
    'MSSF_v2': _make_cfg('MSSF_v2'),

    # 1. C1 — Missing View Handling & Cold-Start Imputation
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

    # 3. C3 — Ordinal Losses x Independent Class Weighting
    'C3_ce_unweighted': _make_cfg(
        'C3_ce_unweighted',
        c3_loss_type='ce',
        c3_class_weighting=False,
    ),
    'C3_ce_weighted': _make_cfg(
        'C3_ce_weighted',
        c3_loss_type='ce',
        c3_class_weighting=True,
    ),
    'C3_emd_unweighted': _make_cfg(
        'C3_emd_unweighted',
        c3_loss_type='emd',
        c3_class_weighting=False,
    ),
    'C3_emd_weighted': _make_cfg(
        'C3_emd_weighted',
        c3_loss_type='emd',
        c3_class_weighting=True,
    ),
    'C3_corn_unweighted': _make_cfg(
        'C3_corn_unweighted',
        c3_loss_type='corn',
        c3_class_weighting=False,
    ),
    'C3_corn_weighted': _make_cfg(
        'C3_corn_weighted',
        c3_loss_type='corn',
        c3_class_weighting=True,
    ),
    'C3_softquant_unweighted': _make_cfg(
        'C3_softquant_unweighted',
        c3_loss_type='ce_softquant',
        c3_class_weighting=False,
    ),
    'C3_softquant_weighted': _make_cfg(
        'C3_softquant_weighted',
        c3_loss_type='ce_softquant',
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
    ),
    'C5_multi_view_aux': _make_cfg(
        'C5_multi_view_aux',
        c5_mode='multi_view',
        c5_alpha=0.15,
        c5_aux_cluster_weight=0.05,
    ),

    # 5. Full MSSF_plus Model & Leave-One-Out (LOO) Ablations
    'MSSF_plus_full': _make_cfg(
        'MSSF_plus_full',
        c1_use_availability=True,
        c1_view_dropout_prob=0.25,
        c1_impute='regress',
        c2_se_mode='plus_ontology',
        c3_loss_type='ce_softquant',
        c3_class_weighting=True,
        c5_mode='multi_view',
        c5_alpha=0.15,
        c5_aux_cluster_weight=0.05,
    ),
    'loo_no_C1': _make_cfg(
        'loo_no_C1',
        c1_use_availability=False,
        c1_view_dropout_prob=0.0,
        c1_impute='zero',
        c2_se_mode='plus_ontology',
        c3_loss_type='ce_softquant',
        c3_class_weighting=True,
        c5_mode='multi_view',
        c5_alpha=0.15,
        c5_aux_cluster_weight=0.05,
    ),
    'loo_no_C2': _make_cfg(
        'loo_no_C2',
        c1_use_availability=True,
        c1_view_dropout_prob=0.25,
        c1_impute='regress',
        c2_se_mode='original',
        c3_loss_type='ce_softquant',
        c3_class_weighting=True,
        c5_mode='multi_view',
        c5_alpha=0.15,
        c5_aux_cluster_weight=0.05,
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
        c5_aux_cluster_weight=0.05,
    ),
    'loo_no_C5': _make_cfg(
        'loo_no_C5',
        c1_use_availability=True,
        c1_view_dropout_prob=0.25,
        c1_impute='regress',
        c2_se_mode='plus_ontology',
        c3_loss_type='ce_softquant',
        c3_class_weighting=True,
        c5_mode='none',
        c5_aux_cluster_weight=0.0,
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
