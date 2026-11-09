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
}

SLIM_ORDER = [
    'cand_A',
    'slim_nodec',
    'slim_nodec_addw',
    'slim_all',
    'slim_all_cnnshared',
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
