"""
Generate verbatim raw verification outputs into `reports/raw/` for:
  1. `reports/raw/item1_models_and_named_children.txt`:
     `print(model)` and parameter counts per `named_children()` for `MSSF_v2` and
     `C3_ce_unweighted_ridge_selfcorr`, using the exact initialization path in `MSSF_plus/train.py`.
  2. `reports/raw/item2_views_and_mssf_comparison.txt`:
     Every view in `build_fold_views`: view name, source file path, source file SHA256,
     raw shape, final view matrix shape, and exact comparison against `MSSF/mssf.py` (`read_raw_data`).
  3. `reports/raw/item4_classifier_head_code.txt`:
     Exact code lines and instantiated layer inspection of the final classification head in
     `MSSF_plus/model.py` and `MSSF/model.py`.

Command:
    MSSF_plus/.venv/bin/python -m MSSF_plus.scripts.dump_raw_verification
"""

import hashlib
import os
import pickle
import sys
from typing import Dict, List, Tuple

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn

from MSSF_plus.configs.experiments import get_experiment_config
from MSSF_plus.model import MSSFPlusModel
from MSSF_plus.protocol import (
    N_DRUGS,
    N_SIDES,
    REPO_DIR,
    SHARED_DIR,
    load_fold_data,
    set_global_seed,
)
from MSSF_plus.soft_labels import build_soft_targets
from MSSF_plus.views import MSSF_750_DIR, build_fold_views

RAW_OUT_DIR = os.path.join(REPO_DIR, 'reports', 'raw')


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def instantiate_via_train_py_path(exp_name: str, fold: int = 0, split_type: str = 'B'):
    """
    Exact initialization sequence from `MSSF_plus/train.py` (`run_single_fold`, L160-L197).
    """
    cfg = get_experiment_config(exp_name)
    base_seed = int(cfg.get('seed', 42))
    fold_seed = base_seed + int(fold)

    set_global_seed(fold_seed)
    fold_split = load_fold_data(fold, split_type)
    fold_views = build_fold_views(fold_split, cfg, device='cpu')
    soft_res = build_soft_targets(fold_split, fold_views, cfg)

    set_global_seed(fold_seed)
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
    )
    return cfg, fold_split, fold_views, soft_res, model


def dump_item1() -> str:
    lines: List[str] = []
    lines.append("# Command: MSSF_plus/.venv/bin/python -m MSSF_plus.scripts.dump_raw_verification")
    lines.append("# Script : MSSF_plus/scripts/dump_raw_verification.py")
    lines.append("# Item 1 : print(model) and parameter count per named_children() for MSSF_v2 and C3_ce_unweighted_ridge_selfcorr")
    lines.append("=" * 110)

    for exp_name in ['MSSF_v2', 'C3_ce_unweighted_ridge_selfcorr']:
        cfg, fold_split, fold_views, soft_res, model = instantiate_via_train_py_path(
            exp_name=exp_name, fold=0, split_type='B'
        )
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        lines.append("")
        lines.append("=" * 110)
        lines.append(f"MODEL: {exp_name}")
        lines.append(
            f"Initialized via train.py path on Fold={fold_split.fold}, Split={fold_split.split_type} | "
            f"n_drug_views={model.n_drug_views} | n_se_views={model.n_se_views} | "
            f"n_clusters={model.n_clusters} | profile_self_correction={model.profile_self_correction}"
        )
        lines.append(f"Total Parameters    : {total_params:,}")
        lines.append(f"Trainable Parameters: {trainable_params:,}")
        lines.append("-" * 110)
        lines.append(f"{'Index':>5} | {'named_children() Name':<25} | {'Child Class':<25} | {'Total Params':>14} | {'Trainable':>14} | {'Share (%)':>9}")
        lines.append("-" * 110)

        sum_children = 0
        sum_trainable = 0
        for idx, (name, child) in enumerate(model.named_children()):
            c_tot = sum(p.numel() for p in child.parameters())
            c_trn = sum(p.numel() for p in child.parameters() if p.requires_grad)
            pct = (100.0 * c_tot / total_params) if total_params > 0 else 0.0
            sum_children += c_tot
            sum_trainable += c_trn
            lines.append(
                f"{idx:>5d} | {name:<25} | {child.__class__.__name__:<25} | "
                f"{c_tot:>14,d} | {c_trn:>14,d} | {pct:>8.2f}%"
            )

        lines.append("-" * 110)
        lines.append(
            f"{'SUM':>5} | {'TOTAL (all named_children)':<25} | {'MSSFPlusModel':<25} | "
            f"{sum_children:>14,d} | {sum_trainable:>14,d} | {100.00:>8.2f}%"
        )
        assert sum_children == total_params, f"Mismatch: {sum_children} != {total_params}"

        lines.append("")
        lines.append(f"--- Verbatim print(model) for {exp_name} ---")
        lines.append(str(model))
        lines.append("=" * 110)

    out_path = os.path.join(RAW_OUT_DIR, 'item1_models_and_named_children.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    return out_path


class _DummyPandasObj:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.state = None

    def __setstate__(self, state):
        self.state = state


class _PandasFallbackUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('pandas'):
            return _DummyPandasObj
        return super().find_class(module, name)


def _extract_ndarray_shapes(obj, found: List[Tuple[int, ...]]) -> None:
    if isinstance(obj, np.ndarray):
        found.append(obj.shape)
    elif isinstance(obj, _DummyPandasObj):
        _extract_ndarray_shapes(getattr(obj, 'args', None), found)
        _extract_ndarray_shapes(getattr(obj, 'state', None), found)
        _extract_ndarray_shapes(getattr(obj, '__dict__', None), found)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _extract_ndarray_shapes(item, found)
    elif isinstance(obj, dict):
        for v in obj.values():
            _extract_ndarray_shapes(v, found)


def _raw_object_shape(path: str) -> str:
    if path.endswith('.pkl'):
        try:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            arr = np.asarray(obj)
            return str(arr.shape)
        except ModuleNotFoundError:
            with open(path, 'rb') as f:
                obj = _PandasFallbackUnpickler(f).load()
            shapes: List[Tuple[int, ...]] = []
            _extract_ndarray_shapes(obj, shapes)
            twod = [s for s in shapes if len(s) == 2]
            if twod:
                return f"{twod[0]} (DataFrame)"
            return str(shapes)
    elif path.endswith('.mat'):
        mat = sio.loadmat(path)
        if 'node_label' in mat:
            return f"node_label:{np.asarray(mat['node_label']).shape}"
        keys = [k for k in mat.keys() if not k.startswith('__')]
        return ",".join(f"{k}:{np.asarray(mat[k]).shape}" for k in keys)
    elif path.endswith('.npy'):
        arr = np.load(path)
        return str(arr.shape)
    elif path.endswith('.npz'):
        z = np.load(path)
        return ",".join(f"{k}:{z[k].shape}" for k in z.files)
    return "unknown"


def dump_item2() -> str:
    lines: List[str] = []
    lines.append("# Command: MSSF_plus/.venv/bin/python -m MSSF_plus.scripts.dump_raw_verification")
    lines.append("# Script : MSSF_plus/scripts/dump_raw_verification.py")
    lines.append("# Item 2 : Views in build_fold_views (name, source file path, SHA256, shape) vs original MSSF/mssf.py")
    lines.append("=" * 130)

    fold_split = load_fold_data(0, 'B')
    split_file_rel = os.path.join('shared_data', 'splits_val', f'split{fold_split.split_type}_fold{fold_split.fold}_train.npy')

    # Mapping from MSSF_plus view name to (relative source file path, how view is constructed, original MSSF/mssf.py counterpart)
    view_source_map = {
        'D1_stitch_one': (
            os.path.join('shared_data', 'mssf_750', 'Text_similarity_one.pkl'),
            'direct pickle load (750, 750)',
            "MSSF/mssf.py L34-36, L115: Text_similarity_one.pkl -> drug_Tfeature_one",
            os.path.join('MSSF', 'Datas', 'Text_similarity_one.pkl'),
        ),
        'D2_stitch_two': (
            os.path.join('shared_data', 'mssf_750', 'Text_similarity_two.pkl'),
            'direct pickle load (750, 750)',
            "MSSF/mssf.py L39-41, L116: Text_similarity_two.pkl -> drug_Tfeature_two",
            os.path.join('MSSF', 'Datas', 'Text_similarity_two.pkl'),
        ),
        'D3_stitch_three': (
            os.path.join('shared_data', 'mssf_750', 'Text_similarity_three.pkl'),
            'direct pickle load (750, 750)',
            "MSSF/mssf.py L44-46, L117: Text_similarity_three.pkl -> drug_Tfeature_three",
            os.path.join('MSSF', 'Datas', 'Text_similarity_three.pkl'),
        ),
        'D4_stitch_four': (
            os.path.join('shared_data', 'mssf_750', 'Text_similarity_four.pkl'),
            'direct pickle load (750, 750)',
            "MSSF/mssf.py L49-51, L118: Text_similarity_four.pkl -> drug_Tfeature_four",
            os.path.join('MSSF', 'Datas', 'Text_similarity_four.pkl'),
        ),
        'D5_stitch_five': (
            os.path.join('shared_data', 'mssf_750', 'Text_similarity_five.pkl'),
            'direct pickle load (750, 750)',
            "MSSF/mssf.py L54-56, L119: Text_similarity_five.pkl -> drug_Tfeature_five",
            os.path.join('MSSF', 'Datas', 'Text_similarity_five.pkl'),
        ),
        'D6_mol2vec_cos': (
            os.path.join('shared_data', 'mssf_750', 'drug_mol.pkl'),
            'cosine_similarity(drug_mol) from (750, 100) -> (750, 750)',
            "MSSF/mssf.py L64-67, L120: drug_mol.pkl -> cosine_similarity(Drug_word2vec)",
            os.path.join('MSSF', 'Datas', 'drug_mol.pkl'),
        ),
        'D7_target_cos': (
            os.path.join('shared_data', 'mssf_750', 'drug_target.pkl'),
            'cosine_similarity(drug_target) from (750, 847) -> (750, 750)',
            "MSSF/mssf.py L76-79, L121: drug_target.pkl -> cosine_similarity(drug_target)",
            os.path.join('MSSF', 'Datas', 'drug_target.pkl'),
        ),
        'D8_fingerprint': (
            os.path.join('shared_data', 'mssf_750', 'fingerprint_similarity.pkl'),
            'direct pickle load (750, 750)',
            "MSSF/mssf.py L82-84, L122: fingerprint_similarity.pkl -> drug_f_sim",
            os.path.join('MSSF', 'Datas', 'fingerprint_similarity.pkl'),
        ),
        'D9_profile_freq_cos': (
            split_file_rel,
            'cosine_similarity(drug_side_train) from train_pairs (750, 994) (+ C1 impute on missing rows)',
            "MSSF/mssf.py L87-101, L123: drug_side.pkl (zeroing test) -> cosine_similarity(drug_side)",
            os.path.join('MSSF', 'Datas', 'drug_side.pkl'),
        ),
        'D10_profile_bin_cos': (
            split_file_rel,
            'cosine_similarity(drug_side_train > 0) from train_pairs (750, 994) (+ C1 impute on missing rows)',
            "MSSF/mssf.py L104-111, L124: (drug_side > 0) -> cosine_similarity(drug_side_label)",
            os.path.join('MSSF', 'Datas', 'drug_side.pkl'),
        ),
        'D11_pathway_enzyme': (
            os.path.join('shared_data', 'mssf_750', 'drug_pathway_enzyme_similarity.pkl'),
            'direct pickle load (750, 750)',
            "MSSF/mssf.py L92-94, L125: drug_pathway_enzyme_similarity.pkl -> drug_p_e_sim",
            os.path.join('MSSF', 'Datas', 'drug_pathway_enzyme_similarity.pkl'),
        ),
        'S1_adrecs_semantic': (
            os.path.join('shared_data', 'mssf_750', 'side_effect_semantic.pkl'),
            'direct pickle load (994, 994)',
            "MSSF/mssf.py L59-61, L132: side_effect_semantic.pkl -> effect_side_semantic",
            os.path.join('MSSF', 'Datas', 'side_effect_semantic.pkl'),
        ),
        'S2_glove_cos': (
            os.path.join('shared_data', 'mssf_750', 'glove_wordEmbedding.pkl'),
            'cosine_similarity(glove) from (994, 300) -> (994, 994)',
            "MSSF/mssf.py L70-73, L133: glove_wordEmbedding.pkl -> cosine_similarity(glove_word)",
            os.path.join('MSSF', 'Datas', 'glove_wordEmbedding.pkl'),
        ),
        'S3_profile_freq_cos': (
            split_file_rel,
            'cosine_similarity(drug_side_train.T) from train_pairs (994, 750) -> (994, 994)',
            "MSSF/mssf.py L128, L134: cosine_similarity(drug_side.T) -> side_drug_sim",
            os.path.join('MSSF', 'Datas', 'drug_side.pkl'),
        ),
        'S4_profile_bin_cos': (
            split_file_rel,
            'cosine_similarity((drug_side_train > 0).T) from train_pairs (994, 750) -> (994, 994)',
            "MSSF/mssf.py L129, L135: cosine_similarity(drug_side_label.T) -> side_drug_label_sim",
            os.path.join('MSSF', 'Datas', 'drug_side.pkl'),
        ),
        'S5_meddra_ontology_cos': (
            os.path.join('shared_data', 'side_effect_label_750.mat'),
            "cosine_similarity(node_label) from mat['node_label'] (994, 243) -> (994, 994)",
            "NOT IN ORIGINAL MSSF (Added by C2 plus_ontology from shared_data/side_effect_label_750.mat)",
            None,
        ),
    }

    for exp_name in ['MSSF_v2', 'C3_ce_unweighted_ridge_selfcorr']:
        cfg = get_experiment_config(exp_name)
        fv = build_fold_views(fold_split, cfg, device=torch.device('cpu'))

        lines.append("")
        lines.append("=" * 130)
        lines.append(
            f"CONFIG: {exp_name} | Fold={fold_split.fold}, Split={fold_split.split_type} | "
            f"c1_impute={cfg.get('c1_impute')} | c2_se_mode={cfg.get('c2_se_mode')}"
        )
        lines.append(
            f"drug_features_matrix shape={fv.drug_features_matrix.shape} (dtype={fv.drug_features_matrix.dtype}) | "
            f"se_features_matrix shape={fv.se_features_matrix.shape} (dtype={fv.se_features_matrix.dtype})"
        )
        lines.append("-" * 130)
        lines.append("DRUG VIEWS IN build_fold_views:")
        lines.append(
            f"{'Idx':>3} | {'View Name':<22} | {'View Shape':<12} | {'Raw Shape':<14} | "
            f"{'Source File Path (relative to repo)':<48} | {'SHA256 of Source File':<64}"
        )
        lines.append("-" * 130)
        for i, (vname, vmat) in enumerate(zip(fv.drug_view_names, fv.drug_views_list)):
            rel_path, _, _, _ = view_source_map[vname]
            abs_path = os.path.join(REPO_DIR, rel_path)
            sha = sha256_file(abs_path)
            raw_sh = _raw_object_shape(abs_path)
            lines.append(
                f"{i:>3d} | {vname:<22} | {str(vmat.shape):<12} | {raw_sh:<14} | "
                f"{rel_path:<48} | {sha:<64}"
            )

        lines.append("-" * 130)
        lines.append("SE VIEWS IN build_fold_views:")
        lines.append(
            f"{'Idx':>3} | {'View Name':<22} | {'View Shape':<12} | {'Raw Shape':<14} | "
            f"{'Source File Path (relative to repo)':<48} | {'SHA256 of Source File':<64}"
        )
        lines.append("-" * 130)
        for j, (vname, vmat) in enumerate(zip(fv.se_view_names, fv.se_views_list)):
            rel_path, _, _, _ = view_source_map[vname]
            abs_path = os.path.join(REPO_DIR, rel_path)
            sha = sha256_file(abs_path)
            raw_sh = _raw_object_shape(abs_path)
            lines.append(
                f"{j:>3d} | {vname:<22} | {str(vmat.shape):<12} | {raw_sh:<14} | "
                f"{rel_path:<48} | {sha:<64}"
            )

    lines.append("")
    lines.append("=" * 130)
    lines.append("COMPARISON WITH ORIGINAL MSSF/mssf.py (read_raw_data, L31-138):")
    lines.append("=" * 130)
    lines.append(
        "Note 1: Original MSSF (`MSSF/Datas/`) has 757 drugs x 994 side effects, whereas the unified benchmark "
        "`shared_data/mssf_750/` filters to the 750 common drugs x 994 side effects (via `shared_data/mssf_750/drug_keep_indices_757_to_750.npy`)."
    )
    ds_750_rel = os.path.join('shared_data', 'mssf_750', 'drug_side.pkl')
    ds_750_abs = os.path.join(REPO_DIR, ds_750_rel)
    lines.append(
        f"Note 2: `shared_data/mssf_750/drug_side.pkl` on disk has shape={_raw_object_shape(ds_750_abs)} and "
        f"SHA256={sha256_file(ds_750_abs)}. Instead of loading the full matrix and masking test pairs like "
        f"`MSSF/mssf.py` L87-99, `MSSF_plus` builds `drug_side_train` (750, 994) strictly from the 80% training split "
        f"(`{split_file_rel}`) via `make_train_only_freq(fold_split.train)` so `val_select`, `val_calib`, and `test` are 100% excluded."
    )
    lines.append("-" * 130)
    lines.append(
        f"{'MSSF_plus View':<24} | {'MSSF_plus Source File (750 drugs)':<46} | "
        f"{'Original MSSF/mssf.py Source File (757 drugs)':<46} | {'MSSF/Datas Raw Shape':<20} | {'MSSF/Datas SHA256'}"
    )
    lines.append("-" * 130)
    for vname, (rel_750, desc, mssf_code_ref, rel_757) in view_source_map.items():
        if rel_757 is not None:
            abs_757 = os.path.join(REPO_DIR, rel_757)
            if os.path.exists(abs_757):
                sha_757 = sha256_file(abs_757)
                sh_757 = _raw_object_shape(abs_757)
            else:
                sha_757 = "MISSING"
                sh_757 = "MISSING"
            lines.append(
                f"{vname:<24} | {rel_750:<46} | {rel_757:<46} | {sh_757:<20} | {sha_757}"
            )
            lines.append(f"   -> Construction: {desc}")
            lines.append(f"   -> Original ref: {mssf_code_ref}")
        else:
            lines.append(
                f"{vname:<24} | {rel_750:<46} | {'(None - new in C2 plus_ontology)':<46} | {'N/A':<20} | N/A"
            )
            lines.append(f"   -> Construction: {desc}")
            lines.append(f"   -> Original ref: {mssf_code_ref}")

    lines.append("")
    lines.append("--- Verbatim excerpt of MSSF/mssf.py lines 30-139 (read_raw_data) ---")
    mssf_py_path = os.path.join(REPO_DIR, 'MSSF', 'mssf.py')
    with open(mssf_py_path, 'r', encoding='utf-8') as f:
        mssf_lines = f.readlines()
    for ln_no in range(30, 139):
        lines.append(f"{ln_no:>4d}: {mssf_lines[ln_no - 1].rstrip()}")

    out_path = os.path.join(RAW_OUT_DIR, 'item2_views_and_mssf_comparison.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def dump_item4() -> str:
    lines: List[str] = []
    lines.append("# Command: MSSF_plus/.venv/bin/python -m MSSF_plus.scripts.dump_raw_verification")
    lines.append("# Script : MSSF_plus/scripts/dump_raw_verification.py")
    lines.append("# Item 4 : Classification head inspection (5-class vs 1-D output) and exact source code lines")
    lines.append("=" * 110)

    for exp_name in ['MSSF_v2', 'C3_ce_unweighted_ridge_selfcorr']:
        cfg, _, _, _, model = instantiate_via_train_py_path(exp_name, fold=0, split_type='B')
        last_linear = model.classifier.classifier[-1]
        lines.append(f"Model: {exp_name}")
        lines.append(f"  cfg['c3_loss_type']           : {cfg.get('c3_loss_type')!r}")
        lines.append(f"  model.classifier.classes      : {model.classifier.classes}")
        lines.append(f"  print(model.classifier)       :\n{model.classifier}")
        lines.append(
            f"  Final layer (`model.classifier.classifier[2]`): {last_linear} | "
            f"in_features={last_linear.in_features}, out_features={last_linear.out_features}, "
            f"weight.shape={tuple(last_linear.weight.shape)}, bias.shape={tuple(last_linear.bias.shape)}, "
            f"params={sum(p.numel() for p in last_linear.parameters())}"
        )
        lines.append(
            f"  Total `model.classifier` params: {sum(p.numel() for p in model.classifier.parameters())} "
            f"(Linear(64, 32)={64*32+32} + Linear(32, {last_linear.out_features})={32*last_linear.out_features + last_linear.out_features})"
        )
        lines.append("-" * 110)

    model_py_path = os.path.join(REPO_DIR, 'MSSF_plus', 'model.py')
    with open(model_py_path, 'r', encoding='utf-8') as f:
        m_lines = f.readlines()

    lines.append("")
    lines.append("--- Verbatim code from MSSF_plus/model.py lines 290-304 (class Classifier) ---")
    for ln_no in range(290, 305):
        lines.append(f"{ln_no:>4d}: {m_lines[ln_no - 1].rstrip()}")

    lines.append("")
    lines.append("--- Verbatim code from MSSF_plus/model.py lines 362-366 & 445-446 (out_classes & self.classifier) ---")
    for ln_no in range(362, 367):
        lines.append(f"{ln_no:>4d}: {m_lines[ln_no - 1].rstrip()}")
    lines.append("...")
    for ln_no in range(445, 447):
        lines.append(f"{ln_no:>4d}: {m_lines[ln_no - 1].rstrip()}")

    lines.append("")
    lines.append("--- Verbatim code from MSSF_plus/model.py lines 612-620 (forward output) ---")
    for ln_no in range(612, 621):
        lines.append(f"{ln_no:>4d}: {m_lines[ln_no - 1].rstrip()}")

    orig_model_py_path = os.path.join(REPO_DIR, 'MSSF', 'model.py')
    with open(orig_model_py_path, 'r', encoding='utf-8') as f:
        om_lines = f.readlines()

    lines.append("")
    lines.append("--- Verbatim code from original MSSF/model.py lines 350-393 (Classifier & Mulmodel.__init__) ---")
    for ln_no in range(350, 394):
        lines.append(f"{ln_no:>4d}: {om_lines[ln_no - 1].rstrip()}")

    out_path = os.path.join(RAW_OUT_DIR, 'item4_classifier_head_code.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def main():
    os.makedirs(RAW_OUT_DIR, exist_ok=True)
    p1 = dump_item1()
    print(f"Saved: {p1}")
    p2 = dump_item2()
    print(f"Saved: {p2}")
    p4 = dump_item4()
    print(f"Saved: {p4}")


if __name__ == '__main__':
    main()
