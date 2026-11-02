"""
Generate MSSF_plus/ARCHITECTURE_DUMP.md directly from instantiated PyTorch modules,
ExperimentConfig objects, and function signatures in the repository.

Single source of truth for all architecture descriptions in MSSF_plus.

Usage:
    python -m MSSF_plus.scripts.dump_architecture
"""

import argparse
import inspect
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from MSSF_plus.configs.experiments import EXPERIMENTS, get_experiment_config
from MSSF_plus.model import (
    AvailabilityModulator,
    CrossProduction,
    MSSFPlusModel,
)
from MSSF_plus.protocol import N_DRUGS, N_SIDES, REPO_DIR
from MSSF_plus.views import (
    ALL_11_DRUG_VIEW_NAMES,
    STATIC_DRUG_VIEW_NAMES,
    ConditionalDenoiserMLP,
    ProfileViewRegressor,
    train_and_impute_diffusion,
    train_and_impute_regress,
)

DEFAULT_OUT_PATH = os.path.join(REPO_DIR, 'MSSF_plus', 'ARCHITECTURE_DUMP.md')
TARGET_CONFIGS = [
    'MSSF_v2',
    'MSSF_v2_selfcorr',
    'C3_ce_unweighted',
    'C3_ce_unweighted_selfcorr',
]


def _resolve_view_metadata(cfg: dict) -> Tuple[List[str], List[str], List[int], List[int]]:
    c1_impute = cfg.get('c1_impute', 'zero')
    c2_se_mode = cfg.get('c2_se_mode', 'original')

    if c1_impute == 'drop':
        drug_view_names = list(STATIC_DRUG_VIEW_NAMES)
        profile_drug_indices = []
    else:
        drug_view_names = list(ALL_11_DRUG_VIEW_NAMES)
        profile_drug_indices = [8, 9]

    if c2_se_mode == 'original':
        se_view_names = [
            'S1_adrecs_semantic',
            'S2_glove_cos',
            'S3_profile_freq_cos',
            'S4_profile_bin_cos',
        ]
        profile_se_indices = [2, 3]
    elif c2_se_mode == 'plus_ontology':
        se_view_names = [
            'S1_adrecs_semantic',
            'S2_glove_cos',
            'S3_profile_freq_cos',
            'S4_profile_bin_cos',
            'S5_meddra_ontology_cos',
        ]
        profile_se_indices = [2, 3]
    elif c2_se_mode == 'no_profile':
        se_view_names = [
            'S1_adrecs_semantic',
            'S2_glove_cos',
            'S5_meddra_ontology_cos',
        ]
        profile_se_indices = []
    else:
        raise ValueError(f"Unknown c2_se_mode={c2_se_mode!r}")

    return drug_view_names, se_view_names, profile_drug_indices, profile_se_indices


def _count_params(module: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def _module_param_table(model: nn.Module) -> List[Tuple[str, str, int, int, float]]:
    total_all, _ = _count_params(model)
    rows = []
    for name, child in model.named_children():
        tot, trn = _count_params(child)
        pct = (100.0 * tot / total_all) if total_all > 0 else 0.0
        rows.append((name, child.__class__.__name__, tot, trn, pct))
    return rows


def _submodule_param_breakdown(model: nn.Module) -> List[Tuple[str, str, int, List[str]]]:
    rows = []
    for name, mod in model.named_modules():
        if name == '':
            continue
        direct_params = list(mod.named_parameters(recurse=False))
        if direct_params:
            n_p = sum(p.numel() for _, p in direct_params)
            shapes = [f"{pn}:{tuple(p.shape)}" for pn, p in direct_params]
            rows.append((name, mod.__class__.__name__, n_p, shapes))
    return rows


def _extract_sequential_layer_specs(seq: nn.Sequential) -> List[Dict[str, object]]:
    specs = []
    for idx, layer in enumerate(seq):
        info: Dict[str, object] = {
            'index': idx,
            'type': layer.__class__.__name__,
            'repr': repr(layer),
            'params': sum(p.numel() for p in layer.parameters()),
        }
        if isinstance(layer, nn.Linear):
            info['in_features'] = layer.in_features
            info['out_features'] = layer.out_features
            info['bias'] = layer.bias is not None
        elif isinstance(layer, nn.LayerNorm):
            info['normalized_shape'] = tuple(layer.normalized_shape)
            info['eps'] = layer.eps
        elif isinstance(layer, nn.Dropout):
            info['p'] = layer.p
        specs.append(info)
    return specs


def generate_architecture_dump(out_path: str = DEFAULT_OUT_PATH) -> str:
    lines: List[str] = []
    lines.append("# MSSF_plus Architecture Dump (Single Source of Truth)")
    lines.append("")
    lines.append(
        "> File này được sinh tự động bởi `python -m MSSF_plus.scripts.dump_architecture` "
        "trực tiếp từ các đối tượng PyTorch và `ExperimentConfig` trong repo. "
        "Mọi mô tả kiến trúc trong tài liệu phải trích từ file này hoặc trực tiếp từ dòng code."
    )
    lines.append("")

    # 1. Repository File Map
    lines.append("## 1. Cấu trúc file thực tế trong `MSSF_plus/`")
    lines.append("")
    mssf_plus_dir = os.path.join(REPO_DIR, 'MSSF_plus')
    py_files = []
    for root, dirs, files in os.walk(mssf_plus_dir):
        dirs[:] = [d for d in sorted(dirs) if not d.startswith('.') and d != '__pycache__']
        for fn in sorted(files):
            if fn.endswith('.py'):
                rel = os.path.relpath(os.path.join(root, fn), REPO_DIR)
                py_files.append(rel)
    lines.append("```text")
    for p in py_files:
        lines.append(p)
    lines.append("```")
    lines.append("")

    # 2. C1 Imputation Modules (ProfileViewRegressor & ConditionalDenoiserMLP)
    lines.append("## 2. Cấu hình thực tế của C1 Imputation Modules (`MSSF_plus/views.py`)")
    lines.append("")

    cond_in_dim = len(STATIC_DRUG_VIEW_NAMES) * N_DRUGS + len(STATIC_DRUG_VIEW_NAMES)  # 9 * 750 + 9 = 6759
    target_dim = 2 * N_DRUGS  # 2 * 750 = 1500

    # 2.1 ProfileViewRegressor
    reg_model = ProfileViewRegressor(in_dim=cond_in_dim, out_dim=target_dim)
    reg_tot, reg_trn = _count_params(reg_model)
    reg_sig = inspect.signature(train_and_impute_regress)
    reg_defaults = {k: v.default for k, v in reg_sig.parameters.items() if v.default is not inspect.Parameter.empty}

    lines.append("### 2.1 `ProfileViewRegressor` (`MSSF_plus/views.py`)")
    lines.append("")
    lines.append(f"- **File định nghĩa:** `MSSF_plus/views.py` (lớp `ProfileViewRegressor`, hàm `train_and_impute_regress`)")
    lines.append(f"- **Chiều đầu vào (`in_dim`):** `{cond_in_dim}` (`9 * 750 = 6750` từ 9 view tĩnh `STATIC_DRUG_VIEW_NAMES` nối với `9` cờ `static_drug_avail`)")
    lines.append(f"- **Chiều đầu ra (`out_dim`):** `{target_dim}` (`[D9, D10]` = `2 * 750`)")
    lines.append(f"- **Tổng số tham số:** `{reg_tot:,}` (trainable: `{reg_trn:,}`)")
    lines.append("")
    lines.append("#### `print(ProfileViewRegressor(in_dim=6759, out_dim=1500))`")
    lines.append("```text")
    lines.append(str(reg_model))
    lines.append("```")
    lines.append("")
    lines.append("#### Chi tiết từng lớp từ đối tượng `ProfileViewRegressor.net`")
    lines.append("```json")
    lines.append(json.dumps(_extract_sequential_layer_specs(reg_model.net), indent=2))
    lines.append("```")
    lines.append("")
    lines.append("#### Tham số huấn luyện mặc định trong `train_and_impute_regress` (đọc qua `inspect.signature` & code)")
    lines.append("```json")
    reg_train_meta = {
        'function_signature_defaults': reg_defaults,
        'optimizer': 'torch.optim.AdamW(lr=1e-3, weight_decay=1e-4)',
        'batch_size': 64,
        'crossfit_val_frac': 0.10,
        'training_loss_formula': 'mse + 0.1 * cos, where mse = F.mse_loss(pred, fit_y[b]) and cos = 1.0 - F.cosine_similarity(pred, fit_y[b], dim=-1).mean()',
        'early_stopping_metric': 'F.mse_loss(v_pred, val_y)',
        'diagonal_postprocessing': 'np.fill_diagonal(d9_pred, 1.0); np.fill_diagonal(d10_pred, 1.0)',
    }
    lines.append(json.dumps(reg_train_meta, indent=2))
    lines.append("```")
    lines.append("")

    # 2.2 ConditionalDenoiserMLP
    diff_model = ConditionalDenoiserMLP(cond_in_dim=cond_in_dim, target_dim=target_dim)
    diff_tot, diff_trn = _count_params(diff_model)
    diff_sig = inspect.signature(train_and_impute_diffusion)
    diff_defaults = {k: v.default for k, v in diff_sig.parameters.items() if v.default is not inspect.Parameter.empty}

    lines.append("### 2.2 `ConditionalDenoiserMLP` (`MSSF_plus/views.py`)")
    lines.append("")
    lines.append(f"- **File định nghĩa:** `MSSF_plus/views.py` (lớp `SinusoidalTimeEmb`, `ConditionalDenoiserMLP`, hàm `train_and_impute_diffusion`)")
    lines.append(f"- **Chiều điều kiện (`cond_in_dim`):** `{cond_in_dim}` (`[static_concat (6750), static_avail (9)]`)")
    lines.append(f"- **Chiều mục tiêu (`target_dim`):** `{target_dim}` (`[D9, D10]` = `2 * 750`)")
    lines.append(f"- **Tổng số tham số:** `{diff_tot:,}` (trainable: `{diff_trn:,}`)")
    lines.append("")
    lines.append("#### `print(ConditionalDenoiserMLP(cond_in_dim=6759, target_dim=1500))`")
    lines.append("```text")
    lines.append(str(diff_model))
    lines.append("```")
    lines.append("")
    lines.append("#### Hàm `ConditionalDenoiserMLP.forward` thực tế")
    lines.append("```python")
    lines.append(inspect.getsource(ConditionalDenoiserMLP.forward).rstrip())
    lines.append("```")
    lines.append("")
    lines.append("#### Tham số huấn luyện & lấy mẫu mặc định trong `train_and_impute_diffusion`")
    lines.append("```json")
    diff_train_meta = {
        'function_signature_defaults': diff_defaults,
        'time_embedding_dim': diff_model.time_mlp[0].dim,
        'hidden_dim': diff_model.x_proj.out_features,
        'has_dropout': any(isinstance(m, nn.Dropout) for m in diff_model.modules()),
        'beta_schedule': 'torch.linspace(1e-4, 0.2, n_timesteps=25)',
        'optimizer': 'torch.optim.AdamW(lr=1e-3, weight_decay=1e-4)',
        'batch_size': 64,
        'crossfit_val_frac': 0.10,
        'training_loss_formula': 'F.mse_loss(x0_hat, x0_b) (direct x_0 prediction)',
        'early_stopping_metric': 'F.mse_loss(v_hat, val_x0_rep) on deterministic grid of 5 timesteps',
        'reverse_sampling': 'Averages n_samples=10 trajectories over reversed(range(n_timesteps=25)), clamped to [0, 1]',
        'diagonal_postprocessing': 'np.fill_diagonal(d9_diff, 1.0); np.fill_diagonal(d10_diff, 1.0)',
    }
    lines.append(json.dumps(diff_train_meta, indent=2))
    lines.append("```")
    lines.append("")

    # 3. CrossProduction & AvailabilityModulator exact implementations
    lines.append("## 3. `CrossProduction` và `AvailabilityModulator` (`MSSF_plus/model.py`)")
    lines.append("")
    lines.append("### 3.1 Mã nguồn `CrossProduction` (`MSSF_plus/model.py`)")
    lines.append("```python")
    lines.append(inspect.getsource(CrossProduction).rstrip())
    lines.append("```")
    lines.append("")
    lines.append("### 3.2 Mã nguồn `AvailabilityModulator` (`MSSF_plus/model.py`)")
    lines.append("```python")
    lines.append(inspect.getsource(AvailabilityModulator).rstrip())
    lines.append("```")
    lines.append("")

    # 4. Target Configurations Dump
    lines.append("## 4. Chi tiết từng cấu hình (`MSSF_v2`, `MSSF_v2_selfcorr`, `C3_ce_unweighted`, `C3_ce_unweighted_selfcorr`)")
    lines.append("")

    dummy_y_train = np.zeros((N_DRUGS, N_SIDES), dtype=np.float32)

    for exp_name in TARGET_CONFIGS:
        exp_cfg = get_experiment_config(exp_name)
        cfg_dict = exp_cfg if isinstance(exp_cfg, dict) else exp_cfg.to_dict()
        drug_views, se_views, prof_d_idx, prof_s_idx = _resolve_view_metadata(cfg_dict)

        model = MSSFPlusModel(
            cfg=cfg_dict,
            n_drug_views=len(drug_views),
            n_se_views=len(se_views),
            profile_drug_view_indices=prof_d_idx,
            profile_se_view_indices=prof_s_idx,
            drug_side_train=dummy_y_train,
            n_clusters=0,
        )
        tot_params, trn_params = _count_params(model)

        lines.append(f"### 4.{TARGET_CONFIGS.index(exp_name) + 1} Cấu hình `{exp_name}`")
        lines.append("")
        lines.append(f"- **Số view thuốc (`n_drug_views`):** `{len(drug_views)}` (`{drug_views}`) — tổng chiều nối: `{len(drug_views) * N_DRUGS}`")
        lines.append(f"- **Số view phản ứng phụ (`n_se_views`):** `{len(se_views)}` (`{se_views}`) — tổng chiều nối: `{len(se_views) * N_SIDES}`")
        lines.append(f"- **Số kênh `CrossProduction` (`input_channel = V_d * V_s`):** `{len(drug_views) * len(se_views)}`")
        lines.append(f"- **Tổng số tham số (`MSSFPlusModel`):** `{tot_params:,}` (trainable: `{trn_params:,}`)")
        if cfg_dict.get('c1_impute') == 'regress':
            lines.append(
                f"- **Tham số bộ điền khuyết tiền xử lý (`ProfileViewRegressor` x `K={cfg_dict.get('c1_crossfit_folds', 5)}` fold cross-fit):** "
                f"`{reg_tot:,}` tham số / mô hình (`{reg_tot * int(cfg_dict.get('c1_crossfit_folds', 5)):,}` trên 5 fold cross-fit, huấn luyện tách biệt trước khi train `MSSFPlusModel`)"
            )
        lines.append("")

        lines.append(f"#### a) Toàn bộ `ExperimentConfig` (`{exp_name}`) dạng JSON")
        lines.append("```json")
        lines.append(json.dumps(cfg_dict, indent=2))
        lines.append("```")
        lines.append("")

        lines.append(f"#### b) Bảng số tham số theo từng module con (`{exp_name}`)")
        lines.append("")
        lines.append("| Module | Lớp (Class) | Tổng tham số | Trainable | Tỷ trọng (%) |")
        lines.append("| :--- | :--- | ---: | ---: | ---: |")
        for m_name, m_cls, m_tot, m_trn, m_pct in _module_param_table(model):
            lines.append(f"| `{m_name}` | `{m_cls}` | {m_tot:,} | {m_trn:,} | {m_pct:.2f}% |")
        lines.append(f"| **Tổng cộng (`MSSFPlusModel`)** | `MSSFPlusModel` | **{tot_params:,}** | **{trn_params:,}** | **100.00%** |")
        lines.append("")

        buffers = list(model.named_buffers())
        if buffers:
            lines.append(f"#### c) Các `register_buffer` không học (`persistent=False`) trong `{exp_name}`")
            lines.append("")
            lines.append("| Tên buffer | Shape | Dtype | Số phần tử |")
            lines.append("| :--- | :--- | :--- | ---: |")
            for b_name, b_tensor in buffers:
                lines.append(f"| `{b_name}` | `{tuple(b_tensor.shape)}` | `{b_tensor.dtype}` | {b_tensor.numel():,} |")
            lines.append("")

        lines.append(f"#### d) Chi tiết tham số từng lớp lá (`{exp_name}`)")
        lines.append("")
        lines.append("| Đường dẫn module | Lớp | Số tham số | Tensors |")
        lines.append("| :--- | :--- | ---: | :--- |")
        for sub_name, sub_cls, sub_cnt, sub_shapes in _submodule_param_breakdown(model):
            lines.append(f"| `{sub_name}` | `{sub_cls}` | {sub_cnt:,} | `{', '.join(sub_shapes)}` |")
        lines.append("")

        lines.append(f"#### e) `print(model)` (`{exp_name}`)")
        lines.append("```text")
        lines.append(str(model))
        lines.append("```")
        lines.append("")

    content = "\n".join(lines)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Dump MSSF_plus architecture to ARCHITECTURE_DUMP.md")
    parser.add_argument('--out', type=str, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()
    out_path = generate_architecture_dump(args.out)
    print(f"Wrote architecture dump to: {out_path}")


if __name__ == '__main__':
    main()
