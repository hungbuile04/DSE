"""
Configuration registry for Stage 5 (`mssf_slim`) progressive compression experiments on `cand_A`.

Configurations (each adding one change on top of `cand_A`):
  0. `cand_A`:
       `vdrop_only` (p_cold=0.25) + C2 (`plus_ontology`) + C5 (`multi_view`, alpha=0.15) + CE,
       no BVI (`use_bvi=False`), no imputation (`c1_impute='zero'`), no modulator (`c1_use_availability=False`),
       `profile_self_correction=True`.
  1. `slim_nodec`:
       Drop decoders of `EN-con` and `EN-add`, and drop both reconstruction losses (`rec_con_weight=0.0`, `rec_add_weight=0.0`).
  2. `slim_nodec_addw`:
       `slim_nodec` + `EN-add` uses masked learned weighted sum `w = softmax(learned_logits)` strictly over present views
       (missing views = 0, sum = 1); downstream encoder unchanged.
  3. `slim_all`:
       `slim_nodec_addw` + shared entity embedding tables `E_drug (750x64)`, `E_se (994x64)` with
       `e_v = LayerNorm(gamma_v * (x^(v) @ E) + beta_v)`; `EN-con` concatenates `[16x64; a_d; a_s] = 1040` dims
       -> `Linear(1040, 256) -> BN -> LeakyReLU -> Dropout(0.4) -> Attention(256) -> Linear(256, 128)`.
  4. `slim_all_cnnshared`:
       `slim_all` + `Preprocess` uses `e_v + Linear(64, 128)` per view (replacing the 16 `Linear(750/994, 128)` branches);
       `CrossProduction` and its `Conv2d` layers remain unchanged.
"""

import os
from copy import deepcopy
from typing import Dict

from MSSF_plus.protocol import REPO_DIR

DEFAULT_RESULTS_SLIM_DIR = os.path.join(REPO_DIR, 'results_slim')

CAND_A_BASE_CONFIG: Dict[str, object] = {
    'exp_name': 'cand_A',
    'description': (
        'Final model candidate A: vdrop_only (p_cold=0.25) + C2 + C5 + CE, '
        'no BVI, no imputation, no modulator (selfcorr=1)'
    ),
    # Training & optimization hyperparameters (inherited unchanged from MSSF / cand_A)
    'seed': 42,
    'epochs': 50,
    'batch_size': 512,
    'eval_batch_size': 512,
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'dropout': 0.4,
    'gp': 64,
    'kl_weight': 0.0,
    'rec_con_weight': 1e-4,
    'rec_add_weight': 1e-4,
    'selection_metric': 'MacroF1',
    # Protocol v2 & C1-C5 settings of cand_A
    'profile_self_correction': True,
    'use_factored_crossprod': True,
    'use_bvi': False,
    'c1_impute': 'zero',
    'c1_use_availability': False,
    'c1_view_dropout_prob': 0.25,
    'c2_se_mode': 'plus_ontology',
    'c3_loss_type': 'ce',
    'c3_class_weighting': False,
    'c4_alpha': 0.10,
    'c5_mode': 'multi_view',
    'c5_alpha': 0.15,
    'c5_k_min': 4,
    'c5_k_max': 12,
    'c5_fixed_k': None,
    'c5_min_samples': 5,
    'c5_view_weighting': 'uniform',
    'c5_aux_cluster_weight': 0.0,
    'eval_test': False,
    # Stage 5 Slim architectural flags
    'slim_use_decoder': True,
    'slim_use_add_weights': False,
    'slim_use_shared_emb': False,
    'slim_shared_emb_dim': 64,
    'slim_use_cnn_shared': False,
}


def _make_slim_cfg(name: str, desc: str, **overrides) -> Dict[str, object]:
    cfg = deepcopy(CAND_A_BASE_CONFIG)
    cfg['exp_name'] = name
    cfg['description'] = desc
    cfg.update(overrides)
    return cfg


SLIM_EXPERIMENTS: Dict[str, Dict[str, object]] = {
    'cand_A': _make_slim_cfg(
        'cand_A',
        'cand_A baseline in mssf_slim (vdrop_only p_cold=0.25 + C2 + C5 + CE, no BVI, no impute, no mod)',
        slim_use_decoder=True,
        rec_con_weight=1e-4,
        rec_add_weight=1e-4,
        slim_use_add_weights=False,
        slim_use_shared_emb=False,
        slim_use_cnn_shared=False,
    ),
    'slim_nodec': _make_slim_cfg(
        'slim_nodec',
        'Drop decoders of EN-con and EN-add and drop both reconstruction losses',
        slim_use_decoder=False,
        rec_con_weight=0.0,
        rec_add_weight=0.0,
        slim_use_add_weights=False,
        slim_use_shared_emb=False,
        slim_use_cnn_shared=False,
    ),
    'slim_nodec_addw': _make_slim_cfg(
        'slim_nodec_addw',
        'slim_nodec + EN-add masked learned weighted sum w = softmax(logits) over present views',
        slim_use_decoder=False,
        rec_con_weight=0.0,
        rec_add_weight=0.0,
        slim_use_add_weights=True,
        slim_use_shared_emb=False,
        slim_use_cnn_shared=False,
    ),
    'slim_all': _make_slim_cfg(
        'slim_all',
        'slim_nodec_addw + shared E_drug(750x64), E_se(994x64) & EN-con [16x64; a_d; a_s]=1040 -> 256 -> Attn -> 128',
        slim_use_decoder=False,
        rec_con_weight=0.0,
        rec_add_weight=0.0,
        slim_use_add_weights=True,
        slim_use_shared_emb=True,
        slim_shared_emb_dim=64,
        slim_use_cnn_shared=False,
    ),
    'slim_all_cnnshared': _make_slim_cfg(
        'slim_all_cnnshared',
        'slim_all + Preprocess using shared e_v + Linear(64, 128) per view instead of 16 Linear(750/994, 128)',
        slim_use_decoder=False,
        rec_con_weight=0.0,
        rec_add_weight=0.0,
        slim_use_add_weights=True,
        slim_use_shared_emb=True,
        slim_shared_emb_dim=64,
        slim_use_cnn_shared=True,
    ),
    # Base MSSF under leak-free Protocol v2 with self-correction (BVI=True, decoders=True, 11x4 original views, hard CE)
    'MSSF_v2_selfcorr': _make_slim_cfg(
        'MSSF_v2_selfcorr',
        'Leak-free MSSF_v2 with profile self-correction (11 drug + 4 SE views, BVI KL=1e-3, decoders=1e-4, no vdrop/C2/C5)',
        use_bvi=True,
        kl_weight=1e-3,
        slim_use_decoder=True,
        rec_con_weight=1e-4,
        rec_add_weight=1e-4,
        c1_use_availability=False,
        c1_view_dropout_prob=0.0,
        c1_impute='zero',
        c2_se_mode='original',
        c3_loss_type='ce',
        c3_class_weighting=False,
        c5_mode='none',
        slim_use_add_weights=False,
        slim_use_shared_emb=False,
        slim_use_cnn_shared=False,
    ),
    # 4 Leave-One-Out (LOO) ablations anchored on the approved final model `slim_nodec`
    'slim_nodec_loo_no_vdrop': _make_slim_cfg(
        'slim_nodec_loo_no_vdrop',
        'LOO on slim_nodec: drop training-time profile view dropout (p_cold=0.0); keep C2 + C5 + nodec',
        slim_use_decoder=False,
        rec_con_weight=0.0,
        rec_add_weight=0.0,
        c1_view_dropout_prob=0.0,
        c2_se_mode='plus_ontology',
        c5_mode='multi_view',
        c5_alpha=0.15,
    ),
    'slim_nodec_loo_no_C2': _make_slim_cfg(
        'slim_nodec_loo_no_C2',
        'LOO on slim_nodec: drop C2 MedDRA ontology view (c2_se_mode="original", 4 SE views); keep vdrop + C5 + nodec',
        slim_use_decoder=False,
        rec_con_weight=0.0,
        rec_add_weight=0.0,
        c1_view_dropout_prob=0.25,
        c2_se_mode='original',
        c5_mode='multi_view',
        c5_alpha=0.15,
    ),
    'slim_nodec_loo_no_C5': _make_slim_cfg(
        'slim_nodec_loo_no_C5',
        'LOO on slim_nodec: drop C5 multi-view spectral soft labels (c5_mode="none"); keep vdrop + C2 + nodec',
        slim_use_decoder=False,
        rec_con_weight=0.0,
        rec_add_weight=0.0,
        c1_view_dropout_prob=0.25,
        c2_se_mode='plus_ontology',
        c5_mode='none',
    ),
    'slim_nodec_loo_no_all': _make_slim_cfg(
        'slim_nodec_loo_no_all',
        'LOO on slim_nodec: drop all three (vdrop=0.0, C2="original", C5="none") on deterministic nodec backbone',
        slim_use_decoder=False,
        rec_con_weight=0.0,
        rec_add_weight=0.0,
        c1_view_dropout_prob=0.0,
        c2_se_mode='original',
        c5_mode='none',
    ),
}

SLIM_ORDER = [
    'cand_A',
    'slim_nodec',
    'slim_nodec_addw',
    'slim_all',
    'slim_all_cnnshared',
]

FINAL_10FOLD_EXPERIMENTS = [
    'slim_nodec',
    'cand_A',
    'slim_all_cnnshared',
    'MSSF_v2_selfcorr',
    'slim_nodec_loo_no_vdrop',
    'slim_nodec_loo_no_C2',
    'slim_nodec_loo_no_C5',
    'slim_nodec_loo_no_all',
]

ALL_REPORT_ORDER = [
    'slim_nodec',
    'cand_A',
    'slim_nodec_addw',
    'slim_all',
    'slim_all_cnnshared',
    'slim_nodec_loo_no_vdrop',
    'slim_nodec_loo_no_C2',
    'slim_nodec_loo_no_C5',
    'slim_nodec_loo_no_all',
    'MSSF_v2_selfcorr',
]


def get_slim_config(exp_name: str, **cli_overrides) -> Dict[str, object]:
    if exp_name not in SLIM_EXPERIMENTS:
        raise KeyError(
            f"Unknown slim experiment {exp_name!r}. Valid options: {list(SLIM_EXPERIMENTS.keys())}"
        )
    cfg = deepcopy(SLIM_EXPERIMENTS[exp_name])
    for k, v in cli_overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg

