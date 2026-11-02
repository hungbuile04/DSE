# MSSF_plus Architecture Dump (Single Source of Truth)

> File này được sinh tự động bởi `python -m MSSF_plus.scripts.dump_architecture` trực tiếp từ các đối tượng PyTorch và `ExperimentConfig` trong repo. Mọi mô tả kiến trúc trong tài liệu phải trích từ file này hoặc trực tiếp từ dòng code.

## 1. Cấu trúc file thực tế trong `MSSF_plus/`

```text
MSSF_plus/__init__.py
MSSF_plus/conformal.py
MSSF_plus/data.py
MSSF_plus/losses.py
MSSF_plus/metrics.py
MSSF_plus/model.py
MSSF_plus/protocol.py
MSSF_plus/soft_labels.py
MSSF_plus/train.py
MSSF_plus/views.py
MSSF_plus/audit/audit_A_feature_leakage.py
MSSF_plus/audit/audit_B2_static_view_sparsity.py
MSSF_plus/audit/audit_B_coldstart_views.py
MSSF_plus/audit/audit_C_selection.py
MSSF_plus/audit/audit_D_ontology.py
MSSF_plus/audit/audit_E_class_distribution.py
MSSF_plus/audit/common.py
MSSF_plus/audit/verify_checksums.py
MSSF_plus/baselines_rerun/__init__.py
MSSF_plus/baselines_rerun/common.py
MSSF_plus/baselines_rerun/run_all.py
MSSF_plus/baselines_rerun/run_dsgat_a3net.py
MSSF_plus/baselines_rerun/run_hstrans.py
MSSF_plus/baselines_rerun/run_mgpred.py
MSSF_plus/baselines_rerun/run_mvsl.py
MSSF_plus/baselines_rerun/run_sdpred.py
MSSF_plus/configs/__init__.py
MSSF_plus/configs/experiments.py
MSSF_plus/scripts/dump_architecture.py
MSSF_plus/scripts/run_ablation.py
MSSF_plus/scripts/summarize_results.py
MSSF_plus/scripts/verify_stage1_crossprod.py
MSSF_plus/tests/test_profile_self_correction.py
```

## 2. Cấu hình thực tế của C1 Imputation Modules (`MSSF_plus/views.py`)

### 2.1 `ProfileViewRegressor` (`MSSF_plus/views.py`)

- **File định nghĩa:** `MSSF_plus/views.py` (lớp `ProfileViewRegressor`, hàm `train_and_impute_regress`)
- **Chiều đầu vào (`in_dim`):** `6759` (`9 * 750 = 6750` từ 9 view tĩnh `STATIC_DRUG_VIEW_NAMES` nối với `9` cờ `static_drug_avail`)
- **Chiều đầu ra (`out_dim`):** `1500` (`[D9, D10]` = `2 * 750`)
- **Tổng số tham số:** `2,182,876` (trainable: `2,182,876`)

#### `print(ProfileViewRegressor(in_dim=6759, out_dim=1500))`
```text
ProfileViewRegressor(
  (net): Sequential(
    (0): Linear(in_features=6759, out_features=256, bias=True)
    (1): LayerNorm((256,), eps=1e-05, elementwise_affine=True, bias=True)
    (2): SiLU()
    (3): Dropout(p=0.2, inplace=False)
    (4): Linear(in_features=256, out_features=256, bias=True)
    (5): LayerNorm((256,), eps=1e-05, elementwise_affine=True, bias=True)
    (6): SiLU()
    (7): Dropout(p=0.2, inplace=False)
    (8): Linear(in_features=256, out_features=1500, bias=True)
    (9): Sigmoid()
  )
)
```

#### Chi tiết từng lớp từ đối tượng `ProfileViewRegressor.net`
```json
[
  {
    "index": 0,
    "type": "Linear",
    "repr": "Linear(in_features=6759, out_features=256, bias=True)",
    "params": 1730560,
    "in_features": 6759,
    "out_features": 256,
    "bias": true
  },
  {
    "index": 1,
    "type": "LayerNorm",
    "repr": "LayerNorm((256,), eps=1e-05, elementwise_affine=True, bias=True)",
    "params": 512,
    "normalized_shape": [
      256
    ],
    "eps": 1e-05
  },
  {
    "index": 2,
    "type": "SiLU",
    "repr": "SiLU()",
    "params": 0
  },
  {
    "index": 3,
    "type": "Dropout",
    "repr": "Dropout(p=0.2, inplace=False)",
    "params": 0,
    "p": 0.2
  },
  {
    "index": 4,
    "type": "Linear",
    "repr": "Linear(in_features=256, out_features=256, bias=True)",
    "params": 65792,
    "in_features": 256,
    "out_features": 256,
    "bias": true
  },
  {
    "index": 5,
    "type": "LayerNorm",
    "repr": "LayerNorm((256,), eps=1e-05, elementwise_affine=True, bias=True)",
    "params": 512,
    "normalized_shape": [
      256
    ],
    "eps": 1e-05
  },
  {
    "index": 6,
    "type": "SiLU",
    "repr": "SiLU()",
    "params": 0
  },
  {
    "index": 7,
    "type": "Dropout",
    "repr": "Dropout(p=0.2, inplace=False)",
    "params": 0,
    "p": 0.2
  },
  {
    "index": 8,
    "type": "Linear",
    "repr": "Linear(in_features=256, out_features=1500, bias=True)",
    "params": 385500,
    "in_features": 256,
    "out_features": 1500,
    "bias": true
  },
  {
    "index": 9,
    "type": "Sigmoid",
    "repr": "Sigmoid()",
    "params": 0
  }
]
```

#### Tham số huấn luyện mặc định trong `train_and_impute_regress` (đọc qua `inspect.signature` & code)
```json
{
  "function_signature_defaults": {
    "epochs": 80,
    "lr": 0.001,
    "seed": 42,
    "n_splits": 5,
    "return_ensemble": false
  },
  "optimizer": "torch.optim.AdamW(lr=1e-3, weight_decay=1e-4)",
  "batch_size": 64,
  "crossfit_val_frac": 0.1,
  "training_loss_formula": "mse + 0.1 * cos, where mse = F.mse_loss(pred, fit_y[b]) and cos = 1.0 - F.cosine_similarity(pred, fit_y[b], dim=-1).mean()",
  "early_stopping_metric": "F.mse_loss(v_pred, val_y)",
  "diagonal_postprocessing": "np.fill_diagonal(d9_pred, 1.0); np.fill_diagonal(d10_pred, 1.0)"
}
```

### 2.2 `ConditionalDenoiserMLP` (`MSSF_plus/views.py`)

- **File định nghĩa:** `MSSF_plus/views.py` (lớp `SinusoidalTimeEmb`, `ConditionalDenoiserMLP`, hàm `train_and_impute_diffusion`)
- **Chiều điều kiện (`cond_in_dim`):** `6759` (`[static_concat (6750), static_avail (9)]`)
- **Chiều mục tiêu (`target_dim`):** `1500` (`[D9, D10]` = `2 * 750`)
- **Tổng số tham số:** `2,650,076` (trainable: `2,650,076`)

#### `print(ConditionalDenoiserMLP(cond_in_dim=6759, target_dim=1500))`
```text
ConditionalDenoiserMLP(
  (cond_proj): Sequential(
    (0): Linear(in_features=6759, out_features=256, bias=True)
    (1): LayerNorm((256,), eps=1e-05, elementwise_affine=True, bias=True)
    (2): SiLU()
  )
  (time_mlp): Sequential(
    (0): SinusoidalTimeEmb()
    (1): Linear(in_features=64, out_features=256, bias=True)
    (2): SiLU()
  )
  (x_proj): Linear(in_features=1500, out_features=256, bias=True)
  (blocks): Sequential(
    (0): Linear(in_features=256, out_features=256, bias=True)
    (1): LayerNorm((256,), eps=1e-05, elementwise_affine=True, bias=True)
    (2): SiLU()
    (3): Linear(in_features=256, out_features=256, bias=True)
    (4): LayerNorm((256,), eps=1e-05, elementwise_affine=True, bias=True)
    (5): SiLU()
    (6): Linear(in_features=256, out_features=1500, bias=True)
    (7): Sigmoid()
  )
)
```

#### Hàm `ConditionalDenoiserMLP.forward` thực tế
```python
    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.x_proj(x_t) + self.cond_proj(cond) + self.time_mlp(t)
        return self.blocks(h)
```

#### Tham số huấn luyện & lấy mẫu mặc định trong `train_and_impute_diffusion`
```json
{
  "function_signature_defaults": {
    "n_timesteps": 25,
    "n_samples": 10,
    "epochs": 100,
    "lr": 0.001,
    "seed": 42,
    "n_splits": 5,
    "return_ensemble": false
  },
  "time_embedding_dim": 64,
  "hidden_dim": 256,
  "has_dropout": false,
  "beta_schedule": "torch.linspace(1e-4, 0.2, n_timesteps=25)",
  "optimizer": "torch.optim.AdamW(lr=1e-3, weight_decay=1e-4)",
  "batch_size": 64,
  "crossfit_val_frac": 0.1,
  "training_loss_formula": "F.mse_loss(x0_hat, x0_b) (direct x_0 prediction)",
  "early_stopping_metric": "F.mse_loss(v_hat, val_x0_rep) on deterministic grid of 5 timesteps",
  "reverse_sampling": "Averages n_samples=10 trajectories over reversed(range(n_timesteps=25)), clamped to [0, 1]",
  "diagonal_postprocessing": "np.fill_diagonal(d9_diff, 1.0); np.fill_diagonal(d10_diff, 1.0)"
}
```

## 3. `CrossProduction` và `AvailabilityModulator` (`MSSF_plus/model.py`)

### 3.1 Mã nguồn `CrossProduction` (`MSSF_plus/model.py`)
```python
class CrossProduction(nn.Module):
    def __init__(self, cross_dim: int, feature_dim: int, input_channel: int, use_factored: bool = True):
        super().__init__()
        self.cross_dim = cross_dim
        self.feature_dim = feature_dim
        self.kernel_size = 4
        self.strides = 4
        self.latent_channel = 32
        self.input_channel = input_channel
        self.use_factored = bool(use_factored)

        self.cnn = nn.Sequential(
            nn.Conv2d(self.input_channel, self.latent_channel, kernel_size=self.kernel_size, stride=self.strides),
            nn.BatchNorm2d(self.latent_channel),
            nn.ReLU(),
            nn.Conv2d(self.latent_channel, self.latent_channel, kernel_size=self.kernel_size, stride=self.strides),
            nn.BatchNorm2d(self.latent_channel),
            nn.ReLU(),
            nn.Conv2d(self.latent_channel, self.latent_channel, kernel_size=self.kernel_size, stride=self.strides),
            nn.BatchNorm2d(self.latent_channel),
            nn.ReLU(),
        )

    def forward(self, drugs: List[torch.Tensor], sides: List[torch.Tensor]) -> torch.Tensor:
        if (
            self.use_factored
            and self.kernel_size == self.strides
            and self.cross_dim % self.kernel_size == 0
            and len(drugs) * len(sides) == self.input_channel
        ):
            # Exact algebraic identity for Conv2d(k=4, s=4) applied to outer products d_v (x) s_u:
            # Avoids materializing the (B, V_d * V_s, 128, 128) tensor (~1.84 GB/batch)
            b = drugs[0].shape[0]
            vd = len(drugs)
            vs = len(sides)
            k = self.kernel_size
            g = self.cross_dim // k
            c_out = self.latent_channel

            d_flat = (
                torch.stack(drugs, dim=1)
                .view(b, vd, g, k)
                .permute(0, 2, 1, 3)
                .reshape(b * g, vd * k)
            )
            w_flat = (
                self.cnn[0]
                .weight.view(c_out, vd, vs, k, k)
                .permute(1, 3, 0, 2, 4)
                .reshape(vd * k, c_out * vs * k)
            )
            m = torch.mm(d_flat, w_flat).view(b, g * c_out, vs * k)
            s_flat = (
                torch.stack(sides, dim=1)
                .view(b, vs, g, k)
                .permute(0, 1, 3, 2)
                .reshape(b, vs * k, g)
            )
            conv0_out = (
                torch.bmm(m, s_flat)
                .view(b, g, c_out, g)
                .permute(0, 2, 1, 3)
                .contiguous()
            )
            if self.cnn[0].bias is not None:
                conv0_out = conv0_out + self.cnn[0].bias.view(1, c_out, 1, 1)
            x = self.cnn[1:](conv0_out).view(-1, self.feature_dim)
            return x

        crosspro = []
        for d_emb in drugs:
            for s_emb in sides:
                crosspro.append(torch.bmm(d_emb.unsqueeze(2), s_emb.unsqueeze(1)))
        crosspro2d = torch.stack(crosspro, dim=1)
        x = self.cnn(crosspro2d).view(-1, self.feature_dim)
        return x
```

### 3.2 Mã nguồn `AvailabilityModulator` (`MSSF_plus/model.py`)
```python
class AvailabilityModulator(nn.Module):
    """
    C1 Input-level availability modulation:
      - Adds a learnable per-view missingness token when `avail_{i,v} == 0`.
      - Computes a per-view residual scale gate conditioned on the full availability vector `a_i`,
        allowing available views to re-scale when profile or sparse static views are missing.
      - Zero-initialized so at epoch 0 it acts as an exact identity mapping.
    """

    def __init__(self, n_views: int, view_dim: int):
        super().__init__()
        self.n_views = n_views
        self.view_dim = view_dim
        self.missing_token = nn.Parameter(torch.zeros(1, n_views, view_dim))
        self.gate_fc1 = nn.Linear(n_views, 2 * n_views)
        self.gate_act = nn.SiLU()
        self.gate_fc2 = nn.Linear(2 * n_views, n_views)
        nn.init.zeros_(self.gate_fc2.weight)
        nn.init.zeros_(self.gate_fc2.bias)

    def forward(self, flat_features: torch.Tensor, avail: torch.Tensor) -> torch.Tensor:
        b = flat_features.shape[0]
        views = flat_features.view(b, self.n_views, self.view_dim)
        gate_delta = torch.tanh(self.gate_fc2(self.gate_act(self.gate_fc1(avail))))  # (b, V)
        scale = 1.0 + 0.2 * gate_delta.unsqueeze(-1)                                # (b, V, 1)
        missing_mask = (1.0 - avail).unsqueeze(-1)                                  # (b, V, 1)
        modulated = views * scale + missing_mask * self.missing_token
        return modulated.reshape(b, self.n_views * self.view_dim)
```

## 4. Chi tiết từng cấu hình (`MSSF_v2`, `MSSF_v2_selfcorr`, `C3_ce_unweighted`, `C3_ce_unweighted_selfcorr`)

### 4.1 Cấu hình `MSSF_v2`

- **Số view thuốc (`n_drug_views`):** `11` (`['D1_stitch_one', 'D2_stitch_two', 'D3_stitch_three', 'D4_stitch_four', 'D5_stitch_five', 'D6_mol2vec_cos', 'D7_target_cos', 'D8_fingerprint', 'D9_profile_freq_cos', 'D10_profile_bin_cos', 'D11_pathway_enzyme']`) — tổng chiều nối: `8250`
- **Số view phản ứng phụ (`n_se_views`):** `4` (`['S1_adrecs_semantic', 'S2_glove_cos', 'S3_profile_freq_cos', 'S4_profile_bin_cos']`) — tổng chiều nối: `3976`
- **Số kênh `CrossProduction` (`input_channel = V_d * V_s`):** `44`
- **Tổng số tham số (`MSSFPlusModel`):** `10,097,399` (trainable: `10,097,399`)

#### a) Toàn bộ `ExperimentConfig` (`MSSF_v2`) dạng JSON
```json
{
  "exp_name": "MSSF_v2",
  "seed": 42,
  "epochs": 50,
  "batch_size": 512,
  "eval_batch_size": 512,
  "lr": 0.0001,
  "weight_decay": 1e-05,
  "dropout": 0.4,
  "gp": 64,
  "kl_weight": 0.001,
  "rec_con_weight": 0.0001,
  "rec_add_weight": 0.0001,
  "selection_metric": "MacroF1",
  "c1_use_availability": false,
  "c1_view_dropout_prob": 0.0,
  "c1_impute": "zero",
  "c1_crossfit_folds": 5,
  "c1_imputer_epochs": 80,
  "c1_diffusion_epochs": 100,
  "c1_diffusion_steps": 25,
  "c1_diffusion_samples": 10,
  "c2_se_mode": "original",
  "c3_loss_type": "ce",
  "c3_class_weighting": false,
  "c4_alpha": 0.1,
  "c5_mode": "none",
  "c5_alpha": 0.15,
  "c5_min_samples": 5,
  "c5_knn_k": 15,
  "c5_k_min": 4,
  "c5_k_max": 12,
  "c5_single_view_name": "D5_stitch_five",
  "c5_view_weighting": "uniform",
  "c5_aux_cluster_weight": 0.0,
  "use_factored_crossprod": true,
  "profile_self_correction": false,
  "use_bvi": true
}
```

#### b) Bảng số tham số theo từng module con (`MSSF_v2`)

| Module | Lớp (Class) | Tổng tham số | Trainable | Tỷ trọng (%) |
| :--- | :--- | ---: | ---: | ---: |
| `encoderConnection` | `EncoderConnection` | 6,602,562 | 6,602,562 | 65.39% |
| `encoderAddition` | `EncoderAddition` | 1,225,296 | 1,225,296 | 12.13% |
| `preprocess` | `Preprocess` | 1,570,688 | 1,570,688 | 15.56% |
| `crossProduction` | `CrossProduction` | 55,584 | 55,584 | 0.55% |
| `attention` | `Attention` | 591,744 | 591,744 | 5.86% |
| `gaussian_parametrizer` | `GaussianParametrizer` | 49,280 | 49,280 | 0.49% |
| `classifier` | `Classifier` | 2,245 | 2,245 | 0.02% |
| **Tổng cộng (`MSSFPlusModel`)** | `MSSFPlusModel` | **10,097,399** | **10,097,399** | **100.00%** |

#### c) Các `register_buffer` không học (`persistent=False`) trong `MSSF_v2`

| Tên buffer | Shape | Dtype | Số phần tử |
| :--- | :--- | :--- | ---: |
| `encoderConnection.l1.1.running_mean` | `(256,)` | `torch.float32` | 256 |
| `encoderConnection.l1.1.running_var` | `(256,)` | `torch.float32` | 256 |
| `encoderConnection.l1.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `encoderConnection.l3.1.running_mean` | `(256,)` | `torch.float32` | 256 |
| `encoderConnection.l3.1.running_var` | `(256,)` | `torch.float32` | 256 |
| `encoderConnection.l3.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `encoderAddition.l1.1.running_mean` | `(256,)` | `torch.float32` | 256 |
| `encoderAddition.l1.1.running_var` | `(256,)` | `torch.float32` | 256 |
| `encoderAddition.l1.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `encoderAddition.l3.1.running_mean` | `(256,)` | `torch.float32` | 256 |
| `encoderAddition.l3.1.running_var` | `(256,)` | `torch.float32` | 256 |
| `encoderAddition.l3.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.0.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.0.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.0.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.1.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.1.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.1.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.2.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.2.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.2.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.3.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.3.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.3.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.4.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.4.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.4.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.5.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.5.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.5.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.6.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.6.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.6.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.7.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.7.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.7.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.8.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.8.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.8.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.9.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.9.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.9.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.10.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.10.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.10.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.side_layers.0.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.side_layers.0.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.side_layers.0.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.side_layers.1.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.side_layers.1.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.side_layers.1.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.side_layers.2.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.side_layers.2.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.side_layers.2.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.side_layers.3.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.side_layers.3.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.side_layers.3.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `crossProduction.cnn.1.running_mean` | `(32,)` | `torch.float32` | 32 |
| `crossProduction.cnn.1.running_var` | `(32,)` | `torch.float32` | 32 |
| `crossProduction.cnn.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `crossProduction.cnn.4.running_mean` | `(32,)` | `torch.float32` | 32 |
| `crossProduction.cnn.4.running_var` | `(32,)` | `torch.float32` | 32 |
| `crossProduction.cnn.4.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `crossProduction.cnn.7.running_mean` | `(32,)` | `torch.float32` | 32 |
| `crossProduction.cnn.7.running_var` | `(32,)` | `torch.float32` | 32 |
| `crossProduction.cnn.7.num_batches_tracked` | `()` | `torch.int64` | 1 |

#### d) Chi tiết tham số từng lớp lá (`MSSF_v2`)

| Đường dẫn module | Lớp | Số tham số | Tensors |
| :--- | :--- | ---: | :--- |
| `encoderConnection.l1.0` | `Linear` | 3,130,112 | `weight:(256, 12226), bias:(256,)` |
| `encoderConnection.l1.1` | `BatchNorm1d` | 512 | `weight:(256,), bias:(256,)` |
| `encoderConnection.attention.WQ` | `Linear` | 65,536 | `weight:(256, 256)` |
| `encoderConnection.attention.WK` | `Linear` | 65,536 | `weight:(256, 256)` |
| `encoderConnection.attention.WV` | `Linear` | 65,536 | `weight:(256, 256)` |
| `encoderConnection.attention.LN1` | `LayerNorm` | 512 | `weight:(256,), bias:(256,)` |
| `encoderConnection.attention.l1` | `Linear` | 65,792 | `weight:(256, 256), bias:(256,)` |
| `encoderConnection.attention.LN2` | `LayerNorm` | 512 | `weight:(256,), bias:(256,)` |
| `encoderConnection.l2` | `Linear` | 32,896 | `weight:(128, 256), bias:(128,)` |
| `encoderConnection.l3.0` | `Linear` | 33,024 | `weight:(256, 128), bias:(256,)` |
| `encoderConnection.l3.1` | `BatchNorm1d` | 512 | `weight:(256,), bias:(256,)` |
| `encoderConnection.l3.3` | `Linear` | 3,142,082 | `weight:(12226, 256), bias:(12226,)` |
| `encoderAddition.l1.0` | `Linear` | 446,720 | `weight:(256, 1744), bias:(256,)` |
| `encoderAddition.l1.1` | `BatchNorm1d` | 512 | `weight:(256,), bias:(256,)` |
| `encoderAddition.attention.WQ` | `Linear` | 65,536 | `weight:(256, 256)` |
| `encoderAddition.attention.WK` | `Linear` | 65,536 | `weight:(256, 256)` |
| `encoderAddition.attention.WV` | `Linear` | 65,536 | `weight:(256, 256)` |
| `encoderAddition.attention.LN1` | `LayerNorm` | 512 | `weight:(256,), bias:(256,)` |
| `encoderAddition.attention.l1` | `Linear` | 65,792 | `weight:(256, 256), bias:(256,)` |
| `encoderAddition.attention.LN2` | `LayerNorm` | 512 | `weight:(256,), bias:(256,)` |
| `encoderAddition.l2` | `Linear` | 32,896 | `weight:(128, 256), bias:(128,)` |
| `encoderAddition.l3.0` | `Linear` | 33,024 | `weight:(256, 128), bias:(256,)` |
| `encoderAddition.l3.1` | `BatchNorm1d` | 512 | `weight:(256,), bias:(256,)` |
| `encoderAddition.l3.3` | `Linear` | 448,208 | `weight:(1744, 256), bias:(1744,)` |
| `preprocess.drug_layers.0.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.0.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.1.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.1.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.2.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.2.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.3.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.3.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.4.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.4.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.5.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.5.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.6.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.6.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.7.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.7.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.8.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.8.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.9.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.9.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.10.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.10.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.side_layers.0.0` | `Linear` | 127,360 | `weight:(128, 994), bias:(128,)` |
| `preprocess.side_layers.0.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.side_layers.1.0` | `Linear` | 127,360 | `weight:(128, 994), bias:(128,)` |
| `preprocess.side_layers.1.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.side_layers.2.0` | `Linear` | 127,360 | `weight:(128, 994), bias:(128,)` |
| `preprocess.side_layers.2.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.side_layers.3.0` | `Linear` | 127,360 | `weight:(128, 994), bias:(128,)` |
| `preprocess.side_layers.3.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `crossProduction.cnn.0` | `Conv2d` | 22,560 | `weight:(32, 44, 4, 4), bias:(32,)` |
| `crossProduction.cnn.1` | `BatchNorm2d` | 64 | `weight:(32,), bias:(32,)` |
| `crossProduction.cnn.3` | `Conv2d` | 16,416 | `weight:(32, 32, 4, 4), bias:(32,)` |
| `crossProduction.cnn.4` | `BatchNorm2d` | 64 | `weight:(32,), bias:(32,)` |
| `crossProduction.cnn.6` | `Conv2d` | 16,416 | `weight:(32, 32, 4, 4), bias:(32,)` |
| `crossProduction.cnn.7` | `BatchNorm2d` | 64 | `weight:(32,), bias:(32,)` |
| `attention.WQ` | `Linear` | 147,456 | `weight:(384, 384)` |
| `attention.WK` | `Linear` | 147,456 | `weight:(384, 384)` |
| `attention.WV` | `Linear` | 147,456 | `weight:(384, 384)` |
| `attention.LN1` | `LayerNorm` | 768 | `weight:(384,), bias:(384,)` |
| `attention.l1` | `Linear` | 147,840 | `weight:(384, 384), bias:(384,)` |
| `attention.LN2` | `LayerNorm` | 768 | `weight:(384,), bias:(384,)` |
| `gaussian_parametrizer.h1` | `Linear` | 24,640 | `weight:(64, 384), bias:(64,)` |
| `gaussian_parametrizer.h2` | `Linear` | 24,640 | `weight:(64, 384), bias:(64,)` |
| `classifier.classifier.0` | `Linear` | 2,080 | `weight:(32, 64), bias:(32,)` |
| `classifier.classifier.2` | `Linear` | 165 | `weight:(5, 32), bias:(5,)` |

#### e) `print(model)` (`MSSF_v2`)
```text
MSSFPlusModel(
  (encoderConnection): EncoderConnection(
    (reluDrop): Sequential(
      (0): LeakyReLU(negative_slope=0.01)
      (1): Dropout(p=0.4, inplace=False)
    )
    (l1): Sequential(
      (0): Linear(in_features=12226, out_features=256, bias=True)
      (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      (2): Sequential(
        (0): LeakyReLU(negative_slope=0.01)
        (1): Dropout(p=0.4, inplace=False)
      )
    )
    (attention): Attention(
      (WQ): Linear(in_features=256, out_features=256, bias=False)
      (WK): Linear(in_features=256, out_features=256, bias=False)
      (WV): Linear(in_features=256, out_features=256, bias=False)
      (LN1): LayerNorm((256,), eps=1e-05, elementwise_affine=True, bias=True)
      (l1): Linear(in_features=256, out_features=256, bias=True)
      (LN2): LayerNorm((256,), eps=1e-05, elementwise_affine=True, bias=True)
    )
    (l2): Linear(in_features=256, out_features=128, bias=True)
    (l3): Sequential(
      (0): Linear(in_features=128, out_features=256, bias=True)
      (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      (2): Sequential(
        (0): LeakyReLU(negative_slope=0.01)
        (1): Dropout(p=0.4, inplace=False)
      )
      (3): Linear(in_features=256, out_features=12226, bias=True)
    )
  )
  (encoderAddition): EncoderAddition(
    (reluDrop): Sequential(
      (0): LeakyReLU(negative_slope=0.01)
      (1): Dropout(p=0.4, inplace=False)
    )
    (l1): Sequential(
      (0): Linear(in_features=1744, out_features=256, bias=True)
      (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      (2): Sequential(
        (0): LeakyReLU(negative_slope=0.01)
        (1): Dropout(p=0.4, inplace=False)
      )
    )
    (attention): Attention(
      (WQ): Linear(in_features=256, out_features=256, bias=False)
      (WK): Linear(in_features=256, out_features=256, bias=False)
      (WV): Linear(in_features=256, out_features=256, bias=False)
      (LN1): LayerNorm((256,), eps=1e-05, elementwise_affine=True, bias=True)
      (l1): Linear(in_features=256, out_features=256, bias=True)
      (LN2): LayerNorm((256,), eps=1e-05, elementwise_affine=True, bias=True)
    )
    (l2): Linear(in_features=256, out_features=128, bias=True)
    (l3): Sequential(
      (0): Linear(in_features=128, out_features=256, bias=True)
      (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      (2): Sequential(
        (0): LeakyReLU(negative_slope=0.01)
        (1): Dropout(p=0.4, inplace=False)
      )
      (3): Linear(in_features=256, out_features=1744, bias=True)
    )
  )
  (preprocess): Preprocess(
    (drug_layers): ModuleList(
      (0-10): 11 x Sequential(
        (0): Linear(in_features=750, out_features=128, bias=True)
        (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        (2): LeakyReLU(negative_slope=0.01)
        (3): Dropout(p=0.4, inplace=False)
      )
    )
    (side_layers): ModuleList(
      (0-3): 4 x Sequential(
        (0): Linear(in_features=994, out_features=128, bias=True)
        (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        (2): LeakyReLU(negative_slope=0.01)
        (3): Dropout(p=0.4, inplace=False)
      )
    )
  )
  (crossProduction): CrossProduction(
    (cnn): Sequential(
      (0): Conv2d(44, 32, kernel_size=(4, 4), stride=(4, 4))
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      (2): ReLU()
      (3): Conv2d(32, 32, kernel_size=(4, 4), stride=(4, 4))
      (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      (5): ReLU()
      (6): Conv2d(32, 32, kernel_size=(4, 4), stride=(4, 4))
      (7): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      (8): ReLU()
    )
  )
  (attention): Attention(
    (WQ): Linear(in_features=384, out_features=384, bias=False)
    (WK): Linear(in_features=384, out_features=384, bias=False)
    (WV): Linear(in_features=384, out_features=384, bias=False)
    (LN1): LayerNorm((384,), eps=1e-05, elementwise_affine=True, bias=True)
    (l1): Linear(in_features=384, out_features=384, bias=True)
    (LN2): LayerNorm((384,), eps=1e-05, elementwise_affine=True, bias=True)
  )
  (gaussian_parametrizer): GaussianParametrizer(
    (h1): Linear(in_features=384, out_features=64, bias=True)
    (h2): Linear(in_features=384, out_features=64, bias=True)
  )
  (classifier): Classifier(
    (reluDrop): Sequential(
      (0): LeakyReLU(negative_slope=0.01)
      (1): Dropout(p=0.4, inplace=False)
    )
    (classifier): Sequential(
      (0): Linear(in_features=64, out_features=32, bias=True)
      (1): Sequential(
        (0): LeakyReLU(negative_slope=0.01)
        (1): Dropout(p=0.4, inplace=False)
      )
      (2): Linear(in_features=32, out_features=5, bias=True)
    )
  )
)
```

### 4.2 Cấu hình `MSSF_v2_selfcorr`

- **Số view thuốc (`n_drug_views`):** `11` (`['D1_stitch_one', 'D2_stitch_two', 'D3_stitch_three', 'D4_stitch_four', 'D5_stitch_five', 'D6_mol2vec_cos', 'D7_target_cos', 'D8_fingerprint', 'D9_profile_freq_cos', 'D10_profile_bin_cos', 'D11_pathway_enzyme']`) — tổng chiều nối: `8250`
- **Số view phản ứng phụ (`n_se_views`):** `4` (`['S1_adrecs_semantic', 'S2_glove_cos', 'S3_profile_freq_cos', 'S4_profile_bin_cos']`) — tổng chiều nối: `3976`
- **Số kênh `CrossProduction` (`input_channel = V_d * V_s`):** `44`
- **Tổng số tham số (`MSSFPlusModel`):** `10,097,399` (trainable: `10,097,399`)

#### a) Toàn bộ `ExperimentConfig` (`MSSF_v2_selfcorr`) dạng JSON
```json
{
  "exp_name": "MSSF_v2_selfcorr",
  "seed": 42,
  "epochs": 50,
  "batch_size": 512,
  "eval_batch_size": 512,
  "lr": 0.0001,
  "weight_decay": 1e-05,
  "dropout": 0.4,
  "gp": 64,
  "kl_weight": 0.001,
  "rec_con_weight": 0.0001,
  "rec_add_weight": 0.0001,
  "selection_metric": "MacroF1",
  "c1_use_availability": false,
  "c1_view_dropout_prob": 0.0,
  "c1_impute": "zero",
  "c1_crossfit_folds": 5,
  "c1_imputer_epochs": 80,
  "c1_diffusion_epochs": 100,
  "c1_diffusion_steps": 25,
  "c1_diffusion_samples": 10,
  "c2_se_mode": "original",
  "c3_loss_type": "ce",
  "c3_class_weighting": false,
  "c4_alpha": 0.1,
  "c5_mode": "none",
  "c5_alpha": 0.15,
  "c5_min_samples": 5,
  "c5_knn_k": 15,
  "c5_k_min": 4,
  "c5_k_max": 12,
  "c5_single_view_name": "D5_stitch_five",
  "c5_view_weighting": "uniform",
  "c5_aux_cluster_weight": 0.0,
  "use_factored_crossprod": true,
  "profile_self_correction": true,
  "use_bvi": true
}
```

#### b) Bảng số tham số theo từng module con (`MSSF_v2_selfcorr`)

| Module | Lớp (Class) | Tổng tham số | Trainable | Tỷ trọng (%) |
| :--- | :--- | ---: | ---: | ---: |
| `encoderConnection` | `EncoderConnection` | 6,602,562 | 6,602,562 | 65.39% |
| `encoderAddition` | `EncoderAddition` | 1,225,296 | 1,225,296 | 12.13% |
| `preprocess` | `Preprocess` | 1,570,688 | 1,570,688 | 15.56% |
| `crossProduction` | `CrossProduction` | 55,584 | 55,584 | 0.55% |
| `attention` | `Attention` | 591,744 | 591,744 | 5.86% |
| `gaussian_parametrizer` | `GaussianParametrizer` | 49,280 | 49,280 | 0.49% |
| `classifier` | `Classifier` | 2,245 | 2,245 | 0.02% |
| **Tổng cộng (`MSSFPlusModel`)** | `MSSFPlusModel` | **10,097,399** | **10,097,399** | **100.00%** |

#### c) Các `register_buffer` không học (`persistent=False`) trong `MSSF_v2_selfcorr`

| Tên buffer | Shape | Dtype | Số phần tử |
| :--- | :--- | :--- | ---: |
| `_sc_y_train` | `(750, 994)` | `torch.float32` | 745,500 |
| `_sc_b_train` | `(750, 994)` | `torch.float32` | 745,500 |
| `_sc_g_d9` | `(750, 750)` | `torch.float32` | 562,500 |
| `_sc_norm2_d9` | `(750,)` | `torch.float32` | 750 |
| `_sc_g_d10` | `(750, 750)` | `torch.float32` | 562,500 |
| `_sc_norm2_d10` | `(750,)` | `torch.float32` | 750 |
| `_sc_g_s3` | `(994, 994)` | `torch.float32` | 988,036 |
| `_sc_norm2_s3` | `(994,)` | `torch.float32` | 994 |
| `_sc_g_s4` | `(994, 994)` | `torch.float32` | 988,036 |
| `_sc_norm2_s4` | `(994,)` | `torch.float32` | 994 |
| `encoderConnection.l1.1.running_mean` | `(256,)` | `torch.float32` | 256 |
| `encoderConnection.l1.1.running_var` | `(256,)` | `torch.float32` | 256 |
| `encoderConnection.l1.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `encoderConnection.l3.1.running_mean` | `(256,)` | `torch.float32` | 256 |
| `encoderConnection.l3.1.running_var` | `(256,)` | `torch.float32` | 256 |
| `encoderConnection.l3.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `encoderAddition.l1.1.running_mean` | `(256,)` | `torch.float32` | 256 |
| `encoderAddition.l1.1.running_var` | `(256,)` | `torch.float32` | 256 |
| `encoderAddition.l1.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `encoderAddition.l3.1.running_mean` | `(256,)` | `torch.float32` | 256 |
| `encoderAddition.l3.1.running_var` | `(256,)` | `torch.float32` | 256 |
| `encoderAddition.l3.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.0.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.0.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.0.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.1.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.1.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.1.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.2.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.2.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.2.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.3.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.3.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.3.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.4.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.4.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.4.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.5.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.5.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.5.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.6.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.6.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.6.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.7.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.7.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.7.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.8.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.8.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.8.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.9.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.9.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.9.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.10.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.10.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.10.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.side_layers.0.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.side_layers.0.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.side_layers.0.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.side_layers.1.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.side_layers.1.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.side_layers.1.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.side_layers.2.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.side_layers.2.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.side_layers.2.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.side_layers.3.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.side_layers.3.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.side_layers.3.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `crossProduction.cnn.1.running_mean` | `(32,)` | `torch.float32` | 32 |
| `crossProduction.cnn.1.running_var` | `(32,)` | `torch.float32` | 32 |
| `crossProduction.cnn.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `crossProduction.cnn.4.running_mean` | `(32,)` | `torch.float32` | 32 |
| `crossProduction.cnn.4.running_var` | `(32,)` | `torch.float32` | 32 |
| `crossProduction.cnn.4.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `crossProduction.cnn.7.running_mean` | `(32,)` | `torch.float32` | 32 |
| `crossProduction.cnn.7.running_var` | `(32,)` | `torch.float32` | 32 |
| `crossProduction.cnn.7.num_batches_tracked` | `()` | `torch.int64` | 1 |

#### d) Chi tiết tham số từng lớp lá (`MSSF_v2_selfcorr`)

| Đường dẫn module | Lớp | Số tham số | Tensors |
| :--- | :--- | ---: | :--- |
| `encoderConnection.l1.0` | `Linear` | 3,130,112 | `weight:(256, 12226), bias:(256,)` |
| `encoderConnection.l1.1` | `BatchNorm1d` | 512 | `weight:(256,), bias:(256,)` |
| `encoderConnection.attention.WQ` | `Linear` | 65,536 | `weight:(256, 256)` |
| `encoderConnection.attention.WK` | `Linear` | 65,536 | `weight:(256, 256)` |
| `encoderConnection.attention.WV` | `Linear` | 65,536 | `weight:(256, 256)` |
| `encoderConnection.attention.LN1` | `LayerNorm` | 512 | `weight:(256,), bias:(256,)` |
| `encoderConnection.attention.l1` | `Linear` | 65,792 | `weight:(256, 256), bias:(256,)` |
| `encoderConnection.attention.LN2` | `LayerNorm` | 512 | `weight:(256,), bias:(256,)` |
| `encoderConnection.l2` | `Linear` | 32,896 | `weight:(128, 256), bias:(128,)` |
| `encoderConnection.l3.0` | `Linear` | 33,024 | `weight:(256, 128), bias:(256,)` |
| `encoderConnection.l3.1` | `BatchNorm1d` | 512 | `weight:(256,), bias:(256,)` |
| `encoderConnection.l3.3` | `Linear` | 3,142,082 | `weight:(12226, 256), bias:(12226,)` |
| `encoderAddition.l1.0` | `Linear` | 446,720 | `weight:(256, 1744), bias:(256,)` |
| `encoderAddition.l1.1` | `BatchNorm1d` | 512 | `weight:(256,), bias:(256,)` |
| `encoderAddition.attention.WQ` | `Linear` | 65,536 | `weight:(256, 256)` |
| `encoderAddition.attention.WK` | `Linear` | 65,536 | `weight:(256, 256)` |
| `encoderAddition.attention.WV` | `Linear` | 65,536 | `weight:(256, 256)` |
| `encoderAddition.attention.LN1` | `LayerNorm` | 512 | `weight:(256,), bias:(256,)` |
| `encoderAddition.attention.l1` | `Linear` | 65,792 | `weight:(256, 256), bias:(256,)` |
| `encoderAddition.attention.LN2` | `LayerNorm` | 512 | `weight:(256,), bias:(256,)` |
| `encoderAddition.l2` | `Linear` | 32,896 | `weight:(128, 256), bias:(128,)` |
| `encoderAddition.l3.0` | `Linear` | 33,024 | `weight:(256, 128), bias:(256,)` |
| `encoderAddition.l3.1` | `BatchNorm1d` | 512 | `weight:(256,), bias:(256,)` |
| `encoderAddition.l3.3` | `Linear` | 448,208 | `weight:(1744, 256), bias:(1744,)` |
| `preprocess.drug_layers.0.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.0.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.1.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.1.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.2.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.2.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.3.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.3.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.4.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.4.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.5.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.5.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.6.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.6.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.7.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.7.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.8.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.8.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.9.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.9.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.10.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.10.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.side_layers.0.0` | `Linear` | 127,360 | `weight:(128, 994), bias:(128,)` |
| `preprocess.side_layers.0.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.side_layers.1.0` | `Linear` | 127,360 | `weight:(128, 994), bias:(128,)` |
| `preprocess.side_layers.1.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.side_layers.2.0` | `Linear` | 127,360 | `weight:(128, 994), bias:(128,)` |
| `preprocess.side_layers.2.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.side_layers.3.0` | `Linear` | 127,360 | `weight:(128, 994), bias:(128,)` |
| `preprocess.side_layers.3.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `crossProduction.cnn.0` | `Conv2d` | 22,560 | `weight:(32, 44, 4, 4), bias:(32,)` |
| `crossProduction.cnn.1` | `BatchNorm2d` | 64 | `weight:(32,), bias:(32,)` |
| `crossProduction.cnn.3` | `Conv2d` | 16,416 | `weight:(32, 32, 4, 4), bias:(32,)` |
| `crossProduction.cnn.4` | `BatchNorm2d` | 64 | `weight:(32,), bias:(32,)` |
| `crossProduction.cnn.6` | `Conv2d` | 16,416 | `weight:(32, 32, 4, 4), bias:(32,)` |
| `crossProduction.cnn.7` | `BatchNorm2d` | 64 | `weight:(32,), bias:(32,)` |
| `attention.WQ` | `Linear` | 147,456 | `weight:(384, 384)` |
| `attention.WK` | `Linear` | 147,456 | `weight:(384, 384)` |
| `attention.WV` | `Linear` | 147,456 | `weight:(384, 384)` |
| `attention.LN1` | `LayerNorm` | 768 | `weight:(384,), bias:(384,)` |
| `attention.l1` | `Linear` | 147,840 | `weight:(384, 384), bias:(384,)` |
| `attention.LN2` | `LayerNorm` | 768 | `weight:(384,), bias:(384,)` |
| `gaussian_parametrizer.h1` | `Linear` | 24,640 | `weight:(64, 384), bias:(64,)` |
| `gaussian_parametrizer.h2` | `Linear` | 24,640 | `weight:(64, 384), bias:(64,)` |
| `classifier.classifier.0` | `Linear` | 2,080 | `weight:(32, 64), bias:(32,)` |
| `classifier.classifier.2` | `Linear` | 165 | `weight:(5, 32), bias:(5,)` |

#### e) `print(model)` (`MSSF_v2_selfcorr`)
```text
MSSFPlusModel(
  (encoderConnection): EncoderConnection(
    (reluDrop): Sequential(
      (0): LeakyReLU(negative_slope=0.01)
      (1): Dropout(p=0.4, inplace=False)
    )
    (l1): Sequential(
      (0): Linear(in_features=12226, out_features=256, bias=True)
      (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      (2): Sequential(
        (0): LeakyReLU(negative_slope=0.01)
        (1): Dropout(p=0.4, inplace=False)
      )
    )
    (attention): Attention(
      (WQ): Linear(in_features=256, out_features=256, bias=False)
      (WK): Linear(in_features=256, out_features=256, bias=False)
      (WV): Linear(in_features=256, out_features=256, bias=False)
      (LN1): LayerNorm((256,), eps=1e-05, elementwise_affine=True, bias=True)
      (l1): Linear(in_features=256, out_features=256, bias=True)
      (LN2): LayerNorm((256,), eps=1e-05, elementwise_affine=True, bias=True)
    )
    (l2): Linear(in_features=256, out_features=128, bias=True)
    (l3): Sequential(
      (0): Linear(in_features=128, out_features=256, bias=True)
      (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      (2): Sequential(
        (0): LeakyReLU(negative_slope=0.01)
        (1): Dropout(p=0.4, inplace=False)
      )
      (3): Linear(in_features=256, out_features=12226, bias=True)
    )
  )
  (encoderAddition): EncoderAddition(
    (reluDrop): Sequential(
      (0): LeakyReLU(negative_slope=0.01)
      (1): Dropout(p=0.4, inplace=False)
    )
    (l1): Sequential(
      (0): Linear(in_features=1744, out_features=256, bias=True)
      (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      (2): Sequential(
        (0): LeakyReLU(negative_slope=0.01)
        (1): Dropout(p=0.4, inplace=False)
      )
    )
    (attention): Attention(
      (WQ): Linear(in_features=256, out_features=256, bias=False)
      (WK): Linear(in_features=256, out_features=256, bias=False)
      (WV): Linear(in_features=256, out_features=256, bias=False)
      (LN1): LayerNorm((256,), eps=1e-05, elementwise_affine=True, bias=True)
      (l1): Linear(in_features=256, out_features=256, bias=True)
      (LN2): LayerNorm((256,), eps=1e-05, elementwise_affine=True, bias=True)
    )
    (l2): Linear(in_features=256, out_features=128, bias=True)
    (l3): Sequential(
      (0): Linear(in_features=128, out_features=256, bias=True)
      (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      (2): Sequential(
        (0): LeakyReLU(negative_slope=0.01)
        (1): Dropout(p=0.4, inplace=False)
      )
      (3): Linear(in_features=256, out_features=1744, bias=True)
    )
  )
  (preprocess): Preprocess(
    (drug_layers): ModuleList(
      (0-10): 11 x Sequential(
        (0): Linear(in_features=750, out_features=128, bias=True)
        (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        (2): LeakyReLU(negative_slope=0.01)
        (3): Dropout(p=0.4, inplace=False)
      )
    )
    (side_layers): ModuleList(
      (0-3): 4 x Sequential(
        (0): Linear(in_features=994, out_features=128, bias=True)
        (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        (2): LeakyReLU(negative_slope=0.01)
        (3): Dropout(p=0.4, inplace=False)
      )
    )
  )
  (crossProduction): CrossProduction(
    (cnn): Sequential(
      (0): Conv2d(44, 32, kernel_size=(4, 4), stride=(4, 4))
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      (2): ReLU()
      (3): Conv2d(32, 32, kernel_size=(4, 4), stride=(4, 4))
      (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      (5): ReLU()
      (6): Conv2d(32, 32, kernel_size=(4, 4), stride=(4, 4))
      (7): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      (8): ReLU()
    )
  )
  (attention): Attention(
    (WQ): Linear(in_features=384, out_features=384, bias=False)
    (WK): Linear(in_features=384, out_features=384, bias=False)
    (WV): Linear(in_features=384, out_features=384, bias=False)
    (LN1): LayerNorm((384,), eps=1e-05, elementwise_affine=True, bias=True)
    (l1): Linear(in_features=384, out_features=384, bias=True)
    (LN2): LayerNorm((384,), eps=1e-05, elementwise_affine=True, bias=True)
  )
  (gaussian_parametrizer): GaussianParametrizer(
    (h1): Linear(in_features=384, out_features=64, bias=True)
    (h2): Linear(in_features=384, out_features=64, bias=True)
  )
  (classifier): Classifier(
    (reluDrop): Sequential(
      (0): LeakyReLU(negative_slope=0.01)
      (1): Dropout(p=0.4, inplace=False)
    )
    (classifier): Sequential(
      (0): Linear(in_features=64, out_features=32, bias=True)
      (1): Sequential(
        (0): LeakyReLU(negative_slope=0.01)
        (1): Dropout(p=0.4, inplace=False)
      )
      (2): Linear(in_features=32, out_features=5, bias=True)
    )
  )
)
```

### 4.3 Cấu hình `C3_ce_unweighted`

- **Số view thuốc (`n_drug_views`):** `11` (`['D1_stitch_one', 'D2_stitch_two', 'D3_stitch_three', 'D4_stitch_four', 'D5_stitch_five', 'D6_mol2vec_cos', 'D7_target_cos', 'D8_fingerprint', 'D9_profile_freq_cos', 'D10_profile_bin_cos', 'D11_pathway_enzyme']`) — tổng chiều nối: `8250`
- **Số view phản ứng phụ (`n_se_views`):** `5` (`['S1_adrecs_semantic', 'S2_glove_cos', 'S3_profile_freq_cos', 'S4_profile_bin_cos', 'S5_meddra_ontology_cos']`) — tổng chiều nối: `4970`
- **Số kênh `CrossProduction` (`input_channel = V_d * V_s`):** `55`
- **Tổng số tham số (`MSSFPlusModel`):** `10,754,421` (trainable: `10,754,421`)
- **Tham số bộ điền khuyết tiền xử lý (`ProfileViewRegressor` x `K=5` fold cross-fit):** `2,182,876` tham số / mô hình (`10,914,380` trên 5 fold cross-fit, huấn luyện tách biệt trước khi train `MSSFPlusModel`)

#### a) Toàn bộ `ExperimentConfig` (`C3_ce_unweighted`) dạng JSON
```json
{
  "exp_name": "C3_ce_unweighted",
  "seed": 42,
  "epochs": 50,
  "batch_size": 512,
  "eval_batch_size": 512,
  "lr": 0.0001,
  "weight_decay": 1e-05,
  "dropout": 0.4,
  "gp": 64,
  "kl_weight": 0.001,
  "rec_con_weight": 0.0001,
  "rec_add_weight": 0.0001,
  "selection_metric": "MacroF1",
  "c1_use_availability": true,
  "c1_view_dropout_prob": 0.25,
  "c1_impute": "regress",
  "c1_crossfit_folds": 5,
  "c1_imputer_epochs": 80,
  "c1_diffusion_epochs": 100,
  "c1_diffusion_steps": 25,
  "c1_diffusion_samples": 10,
  "c2_se_mode": "plus_ontology",
  "c3_loss_type": "ce",
  "c3_class_weighting": false,
  "c4_alpha": 0.1,
  "c5_mode": "multi_view",
  "c5_alpha": 0.15,
  "c5_min_samples": 5,
  "c5_knn_k": 15,
  "c5_k_min": 4,
  "c5_k_max": 12,
  "c5_single_view_name": "D5_stitch_five",
  "c5_view_weighting": "uniform",
  "c5_aux_cluster_weight": 0.0,
  "use_factored_crossprod": true,
  "profile_self_correction": false,
  "use_bvi": true
}
```

#### b) Bảng số tham số theo từng module con (`C3_ce_unweighted`)

| Module | Lớp (Class) | Tổng tham số | Trainable | Tỷ trọng (%) |
| :--- | :--- | ---: | ---: | ---: |
| `drug_modulator` | `AvailabilityModulator` | 8,767 | 8,767 | 0.08% |
| `se_modulator` | `AvailabilityModulator` | 5,085 | 5,085 | 0.05% |
| `encoderConnection` | `EncoderConnection` | 7,112,484 | 7,112,484 | 66.14% |
| `encoderAddition` | `EncoderAddition` | 1,225,296 | 1,225,296 | 11.39% |
| `preprocess` | `Preprocess` | 1,698,304 | 1,698,304 | 15.79% |
| `crossProduction` | `CrossProduction` | 61,216 | 61,216 | 0.57% |
| `attention` | `Attention` | 591,744 | 591,744 | 5.50% |
| `gaussian_parametrizer` | `GaussianParametrizer` | 49,280 | 49,280 | 0.46% |
| `classifier` | `Classifier` | 2,245 | 2,245 | 0.02% |
| **Tổng cộng (`MSSFPlusModel`)** | `MSSFPlusModel` | **10,754,421** | **10,754,421** | **100.00%** |

#### c) Các `register_buffer` không học (`persistent=False`) trong `C3_ce_unweighted`

| Tên buffer | Shape | Dtype | Số phần tử |
| :--- | :--- | :--- | ---: |
| `encoderConnection.l1.1.running_mean` | `(256,)` | `torch.float32` | 256 |
| `encoderConnection.l1.1.running_var` | `(256,)` | `torch.float32` | 256 |
| `encoderConnection.l1.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `encoderConnection.l3.1.running_mean` | `(256,)` | `torch.float32` | 256 |
| `encoderConnection.l3.1.running_var` | `(256,)` | `torch.float32` | 256 |
| `encoderConnection.l3.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `encoderAddition.l1.1.running_mean` | `(256,)` | `torch.float32` | 256 |
| `encoderAddition.l1.1.running_var` | `(256,)` | `torch.float32` | 256 |
| `encoderAddition.l1.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `encoderAddition.l3.1.running_mean` | `(256,)` | `torch.float32` | 256 |
| `encoderAddition.l3.1.running_var` | `(256,)` | `torch.float32` | 256 |
| `encoderAddition.l3.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.0.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.0.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.0.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.1.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.1.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.1.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.2.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.2.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.2.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.3.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.3.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.3.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.4.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.4.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.4.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.5.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.5.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.5.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.6.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.6.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.6.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.7.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.7.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.7.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.8.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.8.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.8.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.9.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.9.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.9.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.10.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.10.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.10.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.side_layers.0.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.side_layers.0.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.side_layers.0.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.side_layers.1.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.side_layers.1.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.side_layers.1.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.side_layers.2.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.side_layers.2.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.side_layers.2.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.side_layers.3.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.side_layers.3.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.side_layers.3.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.side_layers.4.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.side_layers.4.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.side_layers.4.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `crossProduction.cnn.1.running_mean` | `(32,)` | `torch.float32` | 32 |
| `crossProduction.cnn.1.running_var` | `(32,)` | `torch.float32` | 32 |
| `crossProduction.cnn.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `crossProduction.cnn.4.running_mean` | `(32,)` | `torch.float32` | 32 |
| `crossProduction.cnn.4.running_var` | `(32,)` | `torch.float32` | 32 |
| `crossProduction.cnn.4.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `crossProduction.cnn.7.running_mean` | `(32,)` | `torch.float32` | 32 |
| `crossProduction.cnn.7.running_var` | `(32,)` | `torch.float32` | 32 |
| `crossProduction.cnn.7.num_batches_tracked` | `()` | `torch.int64` | 1 |

#### d) Chi tiết tham số từng lớp lá (`C3_ce_unweighted`)

| Đường dẫn module | Lớp | Số tham số | Tensors |
| :--- | :--- | ---: | :--- |
| `drug_modulator` | `AvailabilityModulator` | 8,250 | `missing_token:(1, 11, 750)` |
| `drug_modulator.gate_fc1` | `Linear` | 264 | `weight:(22, 11), bias:(22,)` |
| `drug_modulator.gate_fc2` | `Linear` | 253 | `weight:(11, 22), bias:(11,)` |
| `se_modulator` | `AvailabilityModulator` | 4,970 | `missing_token:(1, 5, 994)` |
| `se_modulator.gate_fc1` | `Linear` | 60 | `weight:(10, 5), bias:(10,)` |
| `se_modulator.gate_fc2` | `Linear` | 55 | `weight:(5, 10), bias:(5,)` |
| `encoderConnection.l1.0` | `Linear` | 3,384,576 | `weight:(256, 13220), bias:(256,)` |
| `encoderConnection.l1.1` | `BatchNorm1d` | 512 | `weight:(256,), bias:(256,)` |
| `encoderConnection.attention.WQ` | `Linear` | 65,536 | `weight:(256, 256)` |
| `encoderConnection.attention.WK` | `Linear` | 65,536 | `weight:(256, 256)` |
| `encoderConnection.attention.WV` | `Linear` | 65,536 | `weight:(256, 256)` |
| `encoderConnection.attention.LN1` | `LayerNorm` | 512 | `weight:(256,), bias:(256,)` |
| `encoderConnection.attention.l1` | `Linear` | 65,792 | `weight:(256, 256), bias:(256,)` |
| `encoderConnection.attention.LN2` | `LayerNorm` | 512 | `weight:(256,), bias:(256,)` |
| `encoderConnection.l2` | `Linear` | 32,896 | `weight:(128, 256), bias:(128,)` |
| `encoderConnection.l3.0` | `Linear` | 33,024 | `weight:(256, 128), bias:(256,)` |
| `encoderConnection.l3.1` | `BatchNorm1d` | 512 | `weight:(256,), bias:(256,)` |
| `encoderConnection.l3.3` | `Linear` | 3,397,540 | `weight:(13220, 256), bias:(13220,)` |
| `encoderAddition.l1.0` | `Linear` | 446,720 | `weight:(256, 1744), bias:(256,)` |
| `encoderAddition.l1.1` | `BatchNorm1d` | 512 | `weight:(256,), bias:(256,)` |
| `encoderAddition.attention.WQ` | `Linear` | 65,536 | `weight:(256, 256)` |
| `encoderAddition.attention.WK` | `Linear` | 65,536 | `weight:(256, 256)` |
| `encoderAddition.attention.WV` | `Linear` | 65,536 | `weight:(256, 256)` |
| `encoderAddition.attention.LN1` | `LayerNorm` | 512 | `weight:(256,), bias:(256,)` |
| `encoderAddition.attention.l1` | `Linear` | 65,792 | `weight:(256, 256), bias:(256,)` |
| `encoderAddition.attention.LN2` | `LayerNorm` | 512 | `weight:(256,), bias:(256,)` |
| `encoderAddition.l2` | `Linear` | 32,896 | `weight:(128, 256), bias:(128,)` |
| `encoderAddition.l3.0` | `Linear` | 33,024 | `weight:(256, 128), bias:(256,)` |
| `encoderAddition.l3.1` | `BatchNorm1d` | 512 | `weight:(256,), bias:(256,)` |
| `encoderAddition.l3.3` | `Linear` | 448,208 | `weight:(1744, 256), bias:(1744,)` |
| `preprocess.drug_layers.0.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.0.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.1.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.1.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.2.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.2.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.3.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.3.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.4.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.4.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.5.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.5.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.6.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.6.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.7.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.7.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.8.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.8.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.9.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.9.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.10.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.10.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.side_layers.0.0` | `Linear` | 127,360 | `weight:(128, 994), bias:(128,)` |
| `preprocess.side_layers.0.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.side_layers.1.0` | `Linear` | 127,360 | `weight:(128, 994), bias:(128,)` |
| `preprocess.side_layers.1.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.side_layers.2.0` | `Linear` | 127,360 | `weight:(128, 994), bias:(128,)` |
| `preprocess.side_layers.2.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.side_layers.3.0` | `Linear` | 127,360 | `weight:(128, 994), bias:(128,)` |
| `preprocess.side_layers.3.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.side_layers.4.0` | `Linear` | 127,360 | `weight:(128, 994), bias:(128,)` |
| `preprocess.side_layers.4.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `crossProduction.cnn.0` | `Conv2d` | 28,192 | `weight:(32, 55, 4, 4), bias:(32,)` |
| `crossProduction.cnn.1` | `BatchNorm2d` | 64 | `weight:(32,), bias:(32,)` |
| `crossProduction.cnn.3` | `Conv2d` | 16,416 | `weight:(32, 32, 4, 4), bias:(32,)` |
| `crossProduction.cnn.4` | `BatchNorm2d` | 64 | `weight:(32,), bias:(32,)` |
| `crossProduction.cnn.6` | `Conv2d` | 16,416 | `weight:(32, 32, 4, 4), bias:(32,)` |
| `crossProduction.cnn.7` | `BatchNorm2d` | 64 | `weight:(32,), bias:(32,)` |
| `attention.WQ` | `Linear` | 147,456 | `weight:(384, 384)` |
| `attention.WK` | `Linear` | 147,456 | `weight:(384, 384)` |
| `attention.WV` | `Linear` | 147,456 | `weight:(384, 384)` |
| `attention.LN1` | `LayerNorm` | 768 | `weight:(384,), bias:(384,)` |
| `attention.l1` | `Linear` | 147,840 | `weight:(384, 384), bias:(384,)` |
| `attention.LN2` | `LayerNorm` | 768 | `weight:(384,), bias:(384,)` |
| `gaussian_parametrizer.h1` | `Linear` | 24,640 | `weight:(64, 384), bias:(64,)` |
| `gaussian_parametrizer.h2` | `Linear` | 24,640 | `weight:(64, 384), bias:(64,)` |
| `classifier.classifier.0` | `Linear` | 2,080 | `weight:(32, 64), bias:(32,)` |
| `classifier.classifier.2` | `Linear` | 165 | `weight:(5, 32), bias:(5,)` |

#### e) `print(model)` (`C3_ce_unweighted`)
```text
MSSFPlusModel(
  (drug_modulator): AvailabilityModulator(
    (gate_fc1): Linear(in_features=11, out_features=22, bias=True)
    (gate_act): SiLU()
    (gate_fc2): Linear(in_features=22, out_features=11, bias=True)
  )
  (se_modulator): AvailabilityModulator(
    (gate_fc1): Linear(in_features=5, out_features=10, bias=True)
    (gate_act): SiLU()
    (gate_fc2): Linear(in_features=10, out_features=5, bias=True)
  )
  (encoderConnection): EncoderConnection(
    (reluDrop): Sequential(
      (0): LeakyReLU(negative_slope=0.01)
      (1): Dropout(p=0.4, inplace=False)
    )
    (l1): Sequential(
      (0): Linear(in_features=13220, out_features=256, bias=True)
      (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      (2): Sequential(
        (0): LeakyReLU(negative_slope=0.01)
        (1): Dropout(p=0.4, inplace=False)
      )
    )
    (attention): Attention(
      (WQ): Linear(in_features=256, out_features=256, bias=False)
      (WK): Linear(in_features=256, out_features=256, bias=False)
      (WV): Linear(in_features=256, out_features=256, bias=False)
      (LN1): LayerNorm((256,), eps=1e-05, elementwise_affine=True, bias=True)
      (l1): Linear(in_features=256, out_features=256, bias=True)
      (LN2): LayerNorm((256,), eps=1e-05, elementwise_affine=True, bias=True)
    )
    (l2): Linear(in_features=256, out_features=128, bias=True)
    (l3): Sequential(
      (0): Linear(in_features=128, out_features=256, bias=True)
      (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      (2): Sequential(
        (0): LeakyReLU(negative_slope=0.01)
        (1): Dropout(p=0.4, inplace=False)
      )
      (3): Linear(in_features=256, out_features=13220, bias=True)
    )
  )
  (encoderAddition): EncoderAddition(
    (reluDrop): Sequential(
      (0): LeakyReLU(negative_slope=0.01)
      (1): Dropout(p=0.4, inplace=False)
    )
    (l1): Sequential(
      (0): Linear(in_features=1744, out_features=256, bias=True)
      (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      (2): Sequential(
        (0): LeakyReLU(negative_slope=0.01)
        (1): Dropout(p=0.4, inplace=False)
      )
    )
    (attention): Attention(
      (WQ): Linear(in_features=256, out_features=256, bias=False)
      (WK): Linear(in_features=256, out_features=256, bias=False)
      (WV): Linear(in_features=256, out_features=256, bias=False)
      (LN1): LayerNorm((256,), eps=1e-05, elementwise_affine=True, bias=True)
      (l1): Linear(in_features=256, out_features=256, bias=True)
      (LN2): LayerNorm((256,), eps=1e-05, elementwise_affine=True, bias=True)
    )
    (l2): Linear(in_features=256, out_features=128, bias=True)
    (l3): Sequential(
      (0): Linear(in_features=128, out_features=256, bias=True)
      (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      (2): Sequential(
        (0): LeakyReLU(negative_slope=0.01)
        (1): Dropout(p=0.4, inplace=False)
      )
      (3): Linear(in_features=256, out_features=1744, bias=True)
    )
  )
  (preprocess): Preprocess(
    (drug_layers): ModuleList(
      (0-10): 11 x Sequential(
        (0): Linear(in_features=750, out_features=128, bias=True)
        (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        (2): LeakyReLU(negative_slope=0.01)
        (3): Dropout(p=0.4, inplace=False)
      )
    )
    (side_layers): ModuleList(
      (0-4): 5 x Sequential(
        (0): Linear(in_features=994, out_features=128, bias=True)
        (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        (2): LeakyReLU(negative_slope=0.01)
        (3): Dropout(p=0.4, inplace=False)
      )
    )
  )
  (crossProduction): CrossProduction(
    (cnn): Sequential(
      (0): Conv2d(55, 32, kernel_size=(4, 4), stride=(4, 4))
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      (2): ReLU()
      (3): Conv2d(32, 32, kernel_size=(4, 4), stride=(4, 4))
      (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      (5): ReLU()
      (6): Conv2d(32, 32, kernel_size=(4, 4), stride=(4, 4))
      (7): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      (8): ReLU()
    )
  )
  (attention): Attention(
    (WQ): Linear(in_features=384, out_features=384, bias=False)
    (WK): Linear(in_features=384, out_features=384, bias=False)
    (WV): Linear(in_features=384, out_features=384, bias=False)
    (LN1): LayerNorm((384,), eps=1e-05, elementwise_affine=True, bias=True)
    (l1): Linear(in_features=384, out_features=384, bias=True)
    (LN2): LayerNorm((384,), eps=1e-05, elementwise_affine=True, bias=True)
  )
  (gaussian_parametrizer): GaussianParametrizer(
    (h1): Linear(in_features=384, out_features=64, bias=True)
    (h2): Linear(in_features=384, out_features=64, bias=True)
  )
  (classifier): Classifier(
    (reluDrop): Sequential(
      (0): LeakyReLU(negative_slope=0.01)
      (1): Dropout(p=0.4, inplace=False)
    )
    (classifier): Sequential(
      (0): Linear(in_features=64, out_features=32, bias=True)
      (1): Sequential(
        (0): LeakyReLU(negative_slope=0.01)
        (1): Dropout(p=0.4, inplace=False)
      )
      (2): Linear(in_features=32, out_features=5, bias=True)
    )
  )
)
```

### 4.4 Cấu hình `C3_ce_unweighted_selfcorr`

- **Số view thuốc (`n_drug_views`):** `11` (`['D1_stitch_one', 'D2_stitch_two', 'D3_stitch_three', 'D4_stitch_four', 'D5_stitch_five', 'D6_mol2vec_cos', 'D7_target_cos', 'D8_fingerprint', 'D9_profile_freq_cos', 'D10_profile_bin_cos', 'D11_pathway_enzyme']`) — tổng chiều nối: `8250`
- **Số view phản ứng phụ (`n_se_views`):** `5` (`['S1_adrecs_semantic', 'S2_glove_cos', 'S3_profile_freq_cos', 'S4_profile_bin_cos', 'S5_meddra_ontology_cos']`) — tổng chiều nối: `4970`
- **Số kênh `CrossProduction` (`input_channel = V_d * V_s`):** `55`
- **Tổng số tham số (`MSSFPlusModel`):** `10,754,421` (trainable: `10,754,421`)
- **Tham số bộ điền khuyết tiền xử lý (`ProfileViewRegressor` x `K=5` fold cross-fit):** `2,182,876` tham số / mô hình (`10,914,380` trên 5 fold cross-fit, huấn luyện tách biệt trước khi train `MSSFPlusModel`)

#### a) Toàn bộ `ExperimentConfig` (`C3_ce_unweighted_selfcorr`) dạng JSON
```json
{
  "exp_name": "C3_ce_unweighted_selfcorr",
  "seed": 42,
  "epochs": 50,
  "batch_size": 512,
  "eval_batch_size": 512,
  "lr": 0.0001,
  "weight_decay": 1e-05,
  "dropout": 0.4,
  "gp": 64,
  "kl_weight": 0.001,
  "rec_con_weight": 0.0001,
  "rec_add_weight": 0.0001,
  "selection_metric": "MacroF1",
  "c1_use_availability": true,
  "c1_view_dropout_prob": 0.25,
  "c1_impute": "regress",
  "c1_crossfit_folds": 5,
  "c1_imputer_epochs": 80,
  "c1_diffusion_epochs": 100,
  "c1_diffusion_steps": 25,
  "c1_diffusion_samples": 10,
  "c2_se_mode": "plus_ontology",
  "c3_loss_type": "ce",
  "c3_class_weighting": false,
  "c4_alpha": 0.1,
  "c5_mode": "multi_view",
  "c5_alpha": 0.15,
  "c5_min_samples": 5,
  "c5_knn_k": 15,
  "c5_k_min": 4,
  "c5_k_max": 12,
  "c5_single_view_name": "D5_stitch_five",
  "c5_view_weighting": "uniform",
  "c5_aux_cluster_weight": 0.0,
  "use_factored_crossprod": true,
  "profile_self_correction": true,
  "use_bvi": true
}
```

#### b) Bảng số tham số theo từng module con (`C3_ce_unweighted_selfcorr`)

| Module | Lớp (Class) | Tổng tham số | Trainable | Tỷ trọng (%) |
| :--- | :--- | ---: | ---: | ---: |
| `drug_modulator` | `AvailabilityModulator` | 8,767 | 8,767 | 0.08% |
| `se_modulator` | `AvailabilityModulator` | 5,085 | 5,085 | 0.05% |
| `encoderConnection` | `EncoderConnection` | 7,112,484 | 7,112,484 | 66.14% |
| `encoderAddition` | `EncoderAddition` | 1,225,296 | 1,225,296 | 11.39% |
| `preprocess` | `Preprocess` | 1,698,304 | 1,698,304 | 15.79% |
| `crossProduction` | `CrossProduction` | 61,216 | 61,216 | 0.57% |
| `attention` | `Attention` | 591,744 | 591,744 | 5.50% |
| `gaussian_parametrizer` | `GaussianParametrizer` | 49,280 | 49,280 | 0.46% |
| `classifier` | `Classifier` | 2,245 | 2,245 | 0.02% |
| **Tổng cộng (`MSSFPlusModel`)** | `MSSFPlusModel` | **10,754,421** | **10,754,421** | **100.00%** |

#### c) Các `register_buffer` không học (`persistent=False`) trong `C3_ce_unweighted_selfcorr`

| Tên buffer | Shape | Dtype | Số phần tử |
| :--- | :--- | :--- | ---: |
| `_sc_y_train` | `(750, 994)` | `torch.float32` | 745,500 |
| `_sc_b_train` | `(750, 994)` | `torch.float32` | 745,500 |
| `_sc_g_d9` | `(750, 750)` | `torch.float32` | 562,500 |
| `_sc_norm2_d9` | `(750,)` | `torch.float32` | 750 |
| `_sc_g_d10` | `(750, 750)` | `torch.float32` | 562,500 |
| `_sc_norm2_d10` | `(750,)` | `torch.float32` | 750 |
| `_sc_g_s3` | `(994, 994)` | `torch.float32` | 988,036 |
| `_sc_norm2_s3` | `(994,)` | `torch.float32` | 994 |
| `_sc_g_s4` | `(994, 994)` | `torch.float32` | 988,036 |
| `_sc_norm2_s4` | `(994,)` | `torch.float32` | 994 |
| `encoderConnection.l1.1.running_mean` | `(256,)` | `torch.float32` | 256 |
| `encoderConnection.l1.1.running_var` | `(256,)` | `torch.float32` | 256 |
| `encoderConnection.l1.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `encoderConnection.l3.1.running_mean` | `(256,)` | `torch.float32` | 256 |
| `encoderConnection.l3.1.running_var` | `(256,)` | `torch.float32` | 256 |
| `encoderConnection.l3.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `encoderAddition.l1.1.running_mean` | `(256,)` | `torch.float32` | 256 |
| `encoderAddition.l1.1.running_var` | `(256,)` | `torch.float32` | 256 |
| `encoderAddition.l1.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `encoderAddition.l3.1.running_mean` | `(256,)` | `torch.float32` | 256 |
| `encoderAddition.l3.1.running_var` | `(256,)` | `torch.float32` | 256 |
| `encoderAddition.l3.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.0.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.0.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.0.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.1.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.1.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.1.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.2.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.2.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.2.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.3.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.3.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.3.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.4.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.4.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.4.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.5.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.5.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.5.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.6.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.6.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.6.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.7.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.7.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.7.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.8.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.8.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.8.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.9.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.9.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.9.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.drug_layers.10.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.10.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.drug_layers.10.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.side_layers.0.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.side_layers.0.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.side_layers.0.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.side_layers.1.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.side_layers.1.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.side_layers.1.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.side_layers.2.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.side_layers.2.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.side_layers.2.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.side_layers.3.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.side_layers.3.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.side_layers.3.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `preprocess.side_layers.4.1.running_mean` | `(128,)` | `torch.float32` | 128 |
| `preprocess.side_layers.4.1.running_var` | `(128,)` | `torch.float32` | 128 |
| `preprocess.side_layers.4.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `crossProduction.cnn.1.running_mean` | `(32,)` | `torch.float32` | 32 |
| `crossProduction.cnn.1.running_var` | `(32,)` | `torch.float32` | 32 |
| `crossProduction.cnn.1.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `crossProduction.cnn.4.running_mean` | `(32,)` | `torch.float32` | 32 |
| `crossProduction.cnn.4.running_var` | `(32,)` | `torch.float32` | 32 |
| `crossProduction.cnn.4.num_batches_tracked` | `()` | `torch.int64` | 1 |
| `crossProduction.cnn.7.running_mean` | `(32,)` | `torch.float32` | 32 |
| `crossProduction.cnn.7.running_var` | `(32,)` | `torch.float32` | 32 |
| `crossProduction.cnn.7.num_batches_tracked` | `()` | `torch.int64` | 1 |

#### d) Chi tiết tham số từng lớp lá (`C3_ce_unweighted_selfcorr`)

| Đường dẫn module | Lớp | Số tham số | Tensors |
| :--- | :--- | ---: | :--- |
| `drug_modulator` | `AvailabilityModulator` | 8,250 | `missing_token:(1, 11, 750)` |
| `drug_modulator.gate_fc1` | `Linear` | 264 | `weight:(22, 11), bias:(22,)` |
| `drug_modulator.gate_fc2` | `Linear` | 253 | `weight:(11, 22), bias:(11,)` |
| `se_modulator` | `AvailabilityModulator` | 4,970 | `missing_token:(1, 5, 994)` |
| `se_modulator.gate_fc1` | `Linear` | 60 | `weight:(10, 5), bias:(10,)` |
| `se_modulator.gate_fc2` | `Linear` | 55 | `weight:(5, 10), bias:(5,)` |
| `encoderConnection.l1.0` | `Linear` | 3,384,576 | `weight:(256, 13220), bias:(256,)` |
| `encoderConnection.l1.1` | `BatchNorm1d` | 512 | `weight:(256,), bias:(256,)` |
| `encoderConnection.attention.WQ` | `Linear` | 65,536 | `weight:(256, 256)` |
| `encoderConnection.attention.WK` | `Linear` | 65,536 | `weight:(256, 256)` |
| `encoderConnection.attention.WV` | `Linear` | 65,536 | `weight:(256, 256)` |
| `encoderConnection.attention.LN1` | `LayerNorm` | 512 | `weight:(256,), bias:(256,)` |
| `encoderConnection.attention.l1` | `Linear` | 65,792 | `weight:(256, 256), bias:(256,)` |
| `encoderConnection.attention.LN2` | `LayerNorm` | 512 | `weight:(256,), bias:(256,)` |
| `encoderConnection.l2` | `Linear` | 32,896 | `weight:(128, 256), bias:(128,)` |
| `encoderConnection.l3.0` | `Linear` | 33,024 | `weight:(256, 128), bias:(256,)` |
| `encoderConnection.l3.1` | `BatchNorm1d` | 512 | `weight:(256,), bias:(256,)` |
| `encoderConnection.l3.3` | `Linear` | 3,397,540 | `weight:(13220, 256), bias:(13220,)` |
| `encoderAddition.l1.0` | `Linear` | 446,720 | `weight:(256, 1744), bias:(256,)` |
| `encoderAddition.l1.1` | `BatchNorm1d` | 512 | `weight:(256,), bias:(256,)` |
| `encoderAddition.attention.WQ` | `Linear` | 65,536 | `weight:(256, 256)` |
| `encoderAddition.attention.WK` | `Linear` | 65,536 | `weight:(256, 256)` |
| `encoderAddition.attention.WV` | `Linear` | 65,536 | `weight:(256, 256)` |
| `encoderAddition.attention.LN1` | `LayerNorm` | 512 | `weight:(256,), bias:(256,)` |
| `encoderAddition.attention.l1` | `Linear` | 65,792 | `weight:(256, 256), bias:(256,)` |
| `encoderAddition.attention.LN2` | `LayerNorm` | 512 | `weight:(256,), bias:(256,)` |
| `encoderAddition.l2` | `Linear` | 32,896 | `weight:(128, 256), bias:(128,)` |
| `encoderAddition.l3.0` | `Linear` | 33,024 | `weight:(256, 128), bias:(256,)` |
| `encoderAddition.l3.1` | `BatchNorm1d` | 512 | `weight:(256,), bias:(256,)` |
| `encoderAddition.l3.3` | `Linear` | 448,208 | `weight:(1744, 256), bias:(1744,)` |
| `preprocess.drug_layers.0.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.0.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.1.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.1.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.2.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.2.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.3.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.3.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.4.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.4.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.5.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.5.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.6.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.6.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.7.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.7.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.8.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.8.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.9.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.9.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.drug_layers.10.0` | `Linear` | 96,128 | `weight:(128, 750), bias:(128,)` |
| `preprocess.drug_layers.10.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.side_layers.0.0` | `Linear` | 127,360 | `weight:(128, 994), bias:(128,)` |
| `preprocess.side_layers.0.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.side_layers.1.0` | `Linear` | 127,360 | `weight:(128, 994), bias:(128,)` |
| `preprocess.side_layers.1.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.side_layers.2.0` | `Linear` | 127,360 | `weight:(128, 994), bias:(128,)` |
| `preprocess.side_layers.2.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.side_layers.3.0` | `Linear` | 127,360 | `weight:(128, 994), bias:(128,)` |
| `preprocess.side_layers.3.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `preprocess.side_layers.4.0` | `Linear` | 127,360 | `weight:(128, 994), bias:(128,)` |
| `preprocess.side_layers.4.1` | `BatchNorm1d` | 256 | `weight:(128,), bias:(128,)` |
| `crossProduction.cnn.0` | `Conv2d` | 28,192 | `weight:(32, 55, 4, 4), bias:(32,)` |
| `crossProduction.cnn.1` | `BatchNorm2d` | 64 | `weight:(32,), bias:(32,)` |
| `crossProduction.cnn.3` | `Conv2d` | 16,416 | `weight:(32, 32, 4, 4), bias:(32,)` |
| `crossProduction.cnn.4` | `BatchNorm2d` | 64 | `weight:(32,), bias:(32,)` |
| `crossProduction.cnn.6` | `Conv2d` | 16,416 | `weight:(32, 32, 4, 4), bias:(32,)` |
| `crossProduction.cnn.7` | `BatchNorm2d` | 64 | `weight:(32,), bias:(32,)` |
| `attention.WQ` | `Linear` | 147,456 | `weight:(384, 384)` |
| `attention.WK` | `Linear` | 147,456 | `weight:(384, 384)` |
| `attention.WV` | `Linear` | 147,456 | `weight:(384, 384)` |
| `attention.LN1` | `LayerNorm` | 768 | `weight:(384,), bias:(384,)` |
| `attention.l1` | `Linear` | 147,840 | `weight:(384, 384), bias:(384,)` |
| `attention.LN2` | `LayerNorm` | 768 | `weight:(384,), bias:(384,)` |
| `gaussian_parametrizer.h1` | `Linear` | 24,640 | `weight:(64, 384), bias:(64,)` |
| `gaussian_parametrizer.h2` | `Linear` | 24,640 | `weight:(64, 384), bias:(64,)` |
| `classifier.classifier.0` | `Linear` | 2,080 | `weight:(32, 64), bias:(32,)` |
| `classifier.classifier.2` | `Linear` | 165 | `weight:(5, 32), bias:(5,)` |

#### e) `print(model)` (`C3_ce_unweighted_selfcorr`)
```text
MSSFPlusModel(
  (drug_modulator): AvailabilityModulator(
    (gate_fc1): Linear(in_features=11, out_features=22, bias=True)
    (gate_act): SiLU()
    (gate_fc2): Linear(in_features=22, out_features=11, bias=True)
  )
  (se_modulator): AvailabilityModulator(
    (gate_fc1): Linear(in_features=5, out_features=10, bias=True)
    (gate_act): SiLU()
    (gate_fc2): Linear(in_features=10, out_features=5, bias=True)
  )
  (encoderConnection): EncoderConnection(
    (reluDrop): Sequential(
      (0): LeakyReLU(negative_slope=0.01)
      (1): Dropout(p=0.4, inplace=False)
    )
    (l1): Sequential(
      (0): Linear(in_features=13220, out_features=256, bias=True)
      (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      (2): Sequential(
        (0): LeakyReLU(negative_slope=0.01)
        (1): Dropout(p=0.4, inplace=False)
      )
    )
    (attention): Attention(
      (WQ): Linear(in_features=256, out_features=256, bias=False)
      (WK): Linear(in_features=256, out_features=256, bias=False)
      (WV): Linear(in_features=256, out_features=256, bias=False)
      (LN1): LayerNorm((256,), eps=1e-05, elementwise_affine=True, bias=True)
      (l1): Linear(in_features=256, out_features=256, bias=True)
      (LN2): LayerNorm((256,), eps=1e-05, elementwise_affine=True, bias=True)
    )
    (l2): Linear(in_features=256, out_features=128, bias=True)
    (l3): Sequential(
      (0): Linear(in_features=128, out_features=256, bias=True)
      (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      (2): Sequential(
        (0): LeakyReLU(negative_slope=0.01)
        (1): Dropout(p=0.4, inplace=False)
      )
      (3): Linear(in_features=256, out_features=13220, bias=True)
    )
  )
  (encoderAddition): EncoderAddition(
    (reluDrop): Sequential(
      (0): LeakyReLU(negative_slope=0.01)
      (1): Dropout(p=0.4, inplace=False)
    )
    (l1): Sequential(
      (0): Linear(in_features=1744, out_features=256, bias=True)
      (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      (2): Sequential(
        (0): LeakyReLU(negative_slope=0.01)
        (1): Dropout(p=0.4, inplace=False)
      )
    )
    (attention): Attention(
      (WQ): Linear(in_features=256, out_features=256, bias=False)
      (WK): Linear(in_features=256, out_features=256, bias=False)
      (WV): Linear(in_features=256, out_features=256, bias=False)
      (LN1): LayerNorm((256,), eps=1e-05, elementwise_affine=True, bias=True)
      (l1): Linear(in_features=256, out_features=256, bias=True)
      (LN2): LayerNorm((256,), eps=1e-05, elementwise_affine=True, bias=True)
    )
    (l2): Linear(in_features=256, out_features=128, bias=True)
    (l3): Sequential(
      (0): Linear(in_features=128, out_features=256, bias=True)
      (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      (2): Sequential(
        (0): LeakyReLU(negative_slope=0.01)
        (1): Dropout(p=0.4, inplace=False)
      )
      (3): Linear(in_features=256, out_features=1744, bias=True)
    )
  )
  (preprocess): Preprocess(
    (drug_layers): ModuleList(
      (0-10): 11 x Sequential(
        (0): Linear(in_features=750, out_features=128, bias=True)
        (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        (2): LeakyReLU(negative_slope=0.01)
        (3): Dropout(p=0.4, inplace=False)
      )
    )
    (side_layers): ModuleList(
      (0-4): 5 x Sequential(
        (0): Linear(in_features=994, out_features=128, bias=True)
        (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
        (2): LeakyReLU(negative_slope=0.01)
        (3): Dropout(p=0.4, inplace=False)
      )
    )
  )
  (crossProduction): CrossProduction(
    (cnn): Sequential(
      (0): Conv2d(55, 32, kernel_size=(4, 4), stride=(4, 4))
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      (2): ReLU()
      (3): Conv2d(32, 32, kernel_size=(4, 4), stride=(4, 4))
      (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      (5): ReLU()
      (6): Conv2d(32, 32, kernel_size=(4, 4), stride=(4, 4))
      (7): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
      (8): ReLU()
    )
  )
  (attention): Attention(
    (WQ): Linear(in_features=384, out_features=384, bias=False)
    (WK): Linear(in_features=384, out_features=384, bias=False)
    (WV): Linear(in_features=384, out_features=384, bias=False)
    (LN1): LayerNorm((384,), eps=1e-05, elementwise_affine=True, bias=True)
    (l1): Linear(in_features=384, out_features=384, bias=True)
    (LN2): LayerNorm((384,), eps=1e-05, elementwise_affine=True, bias=True)
  )
  (gaussian_parametrizer): GaussianParametrizer(
    (h1): Linear(in_features=384, out_features=64, bias=True)
    (h2): Linear(in_features=384, out_features=64, bias=True)
  )
  (classifier): Classifier(
    (reluDrop): Sequential(
      (0): LeakyReLU(negative_slope=0.01)
      (1): Dropout(p=0.4, inplace=False)
    )
    (classifier): Sequential(
      (0): Linear(in_features=64, out_features=32, bias=True)
      (1): Sequential(
        (0): LeakyReLU(negative_slope=0.01)
        (1): Dropout(p=0.4, inplace=False)
      )
      (2): Linear(in_features=32, out_features=5, bias=True)
    )
  )
)
```
