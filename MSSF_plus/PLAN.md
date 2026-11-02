# Kế hoạch Triển khai & Thiết kế Kỹ thuật — `MSSF_plus`

## 1. Cấu trúc thư mục `MSSF_plus/`

```text
MSSF_plus/
├── PLAN.md                         # Tài liệu thiết kế kiến trúc, interface & kế hoạch ablation
├── requirements.txt                # 4 thư viện chuẩn: numpy, scipy, scikit-learn, torch
├── requirements-lock.txt           # Phiên bản khoá chính xác của môi trường chạy
├── __init__.py
├── protocol.py                     # Load splits_v2, quản lý seed, lưu results_v2, kiểm định Wilcoxon
├── metrics.py                      # Bộ metric 5 lớp đầy đủ (Acc, WF1, MacroF1, Kappa, QWK, MAE, F1_c0..4, AUPR)
├── views.py                        # Load 11 drug + 4 SE views, tính lại profile từ train 80%, C1 (mask/impute/dropout), C2 (ontology)
├── data.py                         # Dataset & DataLoader (train shuffle=True; valsel/valcal/test shuffle=False + assert labels)
├── model.py                        # Mulmodel mở rộng: AvailabilityModulator (C1) + 3 Encoder + Self-Attn + BVI + Classifier + AuxClusterHead (C5)
├── losses.py                       # Loss phụ MSSF (KL, RecCon, RecAdd) + C3 (ce, emd, corn, ce_softquant) + class weighting độc lập
├── soft_labels.py                  # C5: Multi-view / single-view spectral clustering & kNN smoothing (strict leave-one-out d' != d)
├── conformal.py                    # C4: Post-hoc Conformal Prediction trên val_calib (LAC, APS, Ordinal-LAC, Ordinal-APS ở mức 90%)
├── train.py                        # Entry-point huấn luyện, chọn epoch theo Macro-F1 trên val_select, đánh giá test & conformal
├── configs/
│   └── experiments.py              # Định nghĩa đầy đủ cấu hình cho Baseline MSSF_v2, C1–C5 ablations, Full model & Leave-One-Out
├── scripts/
│   ├── run_ablation.py             # Chạy từng cấu hình hoặc nhóm ablation (--quick, --dev 3 folds, --full 10 folds)
│   └── summarize_results.py        # Tổng hợp bảng kết quả Markdown/CSV + p-value Wilcoxon signed-rank so với MSSF_v2
└── audit/                          # Các script kiểm toán Bước 1 & kiểm tra bất biến SHA256 shared_data/
```

---

## 2. Interface từng Module

### 2.1. `protocol.py`
- `set_global_seed(seed: int = 42) -> None`: cố định `random`, `numpy`, `torch` (CPU/CUDA/MPS).
- `get_fold_list(mode: str = 'dev') -> list[int]`:
  - `'quick'` $\to$ `[0]`
  - `'dev'` $\to$ `[0, 1, 2]` (3 fold phát triển & chọn siêu tham số)
  - `'full'` $\to$ `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` (10 fold báo cáo cuối)
- `load_fold_data(fold: int, split_type: str) -> FoldSplit`:
  - Trả về dataclass `FoldSplit(fold, split_type, train, valsel, valcal, test, train_drugs, valsel_drugs, valcal_drugs, test_drugs)` đọc từ `shared_data/split_adapter_v2.py`.
- `save_fold_predictions(output_dir, fold, test_split, y_true_freq, y_pred_freq, probs_5class, extra_arrays=None) -> None`:
  - Kiểm tra bất biến `assert np.array_equal(y_true_freq, test_split[:, 2])`.
  - Lưu `fold_{k}_labels.npy` (thang `1..5`), `fold_{k}_preds.npy` (thang `1..5`, tương thích 100% với `shared_data/unified_eval.py`), `fold_{k}_probs.npy` (`N × 5`), và `fold_{k}_valcal_{labels,probs}.npy` phục vụ C4.
- `save_experiment_config(output_dir, config_dict, fold_summaries) -> None`:
  - Ghi `config.json` kèm thông tin phiên bản `python`, `numpy`, `scipy`, `scikit-learn`, `torch`.
- `wilcoxon_compare(results_dir, method_a, method_b, split_type='A', metric_keys=None) -> dict`:
  - Kiểm định Wilcoxon signed-rank theo cặp fold giữa `method_a` và `method_b`.

### 2.2. `metrics.py`
- `compute_multiclass_metrics(y_true_class, y_pred_class, probs=None) -> dict`:
  - Đầu vào: `y_true_class`, `y_pred_class` $\in \{0, 1, 2, 3, 4\}$; `probs` shape `(N, 5)`.
  - Trả về đầy đủ:
    - `Accuracy`, `WeightedF1`, `MacroF1` (metric chọn checkpoint mặc định trên `val_select`)
    - `Kappa` (Cohen's Kappa), `QWK` (Quadratic Weighted Kappa)
    - `MAE` (trên chỉ số lớp `|y_true - y_pred|`), `ExpectedMAE` ($\frac{1}{N}\sum |\hat{y}_i - y_i|$ với $\hat{y}_i = \sum_{k=0}^4 k p_{i,k}$), `RMSE`
    - `F1_c0`, `F1_c1`, `F1_c2`, `F1_c3`, `F1_c4` (F1 từng lớp)
    - `MacroPrecision`, `MacroRecall`, `MCC`, `MacroAUPR`.

### 2.3. `views.py` (C1 & C2)
- `build_fold_views(fold_split: FoldSplit, cfg: dict, device='cpu') -> FoldViews`:
  1. **Dựng ma trận `drug_side_train` (`750 × 994`)**: khởi tạo `np.zeros((750, 994))` và chỉ điền các cặp thuộc `fold_split.train` (80%).
  2. **Tính 4 view profile**: `D9_profile_freq_cos`, `D10_profile_bin_cos` (`750 × 750`) và `S3_profile_freq_cos`, `S4_profile_bin_cos` (`994 × 994`) từ `drug_side_train`.
  3. **C2 — Chế độ SE views (`c2_se_mode`)**:
     - `'original'`: `[S1_adrecs, S2_glove, S3_freq, S4_bin]` (4 views, `3976-D`).
     - `'plus_ontology'`: `[S1_adrecs, S2_glove, S3_freq, S4_bin, S5_meddra_cos]` (5 views, `4970-D`), với `S5_meddra_cos = cosine_similarity(side_effect_label_750.mat['node_label'])`.
     - `'no_profile'`: `[S1_adrecs, S2_glove, S5_meddra_cos]` (3 views, `2982-D`, bỏ hoàn toàn `S3, S4`).
  4. **C1 — Availability Indicator**:
     - Với mỗi thuốc $d$ và drug view $v$: `drug_avail[d, v] = 1.0` nếu $\sum_{j \neq d} |S^{(v)}_{d,j}| > 0$, ngược lại `0.0`.
     - Với mỗi SE $s$ và SE view $u$: `se_avail[s, u] = 1.0` nếu $\sum_{j \neq s} |S^{(u)}_{s,j}| > 0$, ngược lại `0.0`.
  5. **C1 — Chiến lược xử lý view profile thiếu (`c1_impute`)**:
     - `'zero'`: giữ nguyên vector `0` ở `D9, D10` cho các thuốc không có cặp train.
     - `'drop'`: loại bỏ `D9, D10` khỏi danh sách drug views (chỉ dùng 9 drug views tĩnh `[D1..D8, D11]`, `6750-D`).
     - `'mean'`: điền hàng `D9[d], D10[d]` của các thuốc thiếu profile ($a_{d,\text{D9}}=0$) bằng vector trung bình của các thuốc thuộc `train_drugs`.
     - `'regress'`: huấn luyện $K=5$ mạng MLP `ProfileViewRegressor` theo cơ chế **Cross-Fitting theo thuốc** trên `train_drugs` (chia `train_drugs` thành $K=5$ nhóm, mỗi regressor train trên 4 nhóm với 10% early-stopping nội bộ và dự đoán out-of-fold cho nhóm thứ 5 làm `fallback_d9, fallback_d10`), sau đó dùng trung bình cộng của $K=5$ regressor để điền cho mọi thuốc cold-start có $a_{d,\text{D9}} = 0$ (`valsel`, `valcal`, `test` ở Split B).
     - `'diffusion'`: áp dụng cùng cơ chế Cross-Fitting $K=5$ cho `ConditionalDenoiserMLP`; chỉ giữ làm 1 dòng đối chứng trong bảng Ablation C1 (trên dev folds) do MSE trên `val_select` kém hơn `mean` và `regress`.

### 2.4. `model.py` (Giữ nguyên 3 Encoder + Self-Attention + BVI, mở rộng C1 & C5)
- `MSSFPlusModel(nn.Module)`:
  - Hỗ trợ số lượng drug views $V_d \in \{9, 11\}$ và SE views $V_s \in \{3, 4, 5\}$ tự động trong `EncoderConnection`, `EncoderAddition`, `Preprocess`, và `CrossProduction` ($V_d \times V_s$ kênh).
  - Khi `c1_use_availability=False`, `c1_impute='zero'`, `c2_se_mode='original'`, `c3_loss_type='ce'`, `c5_mode='none'`, kiến trúc và số tham số **trùng khớp 100% với `Mulmodel` gốc của MSSF**.
  - Khi `c1_use_availability=True`: module `AvailabilityModulator` xử lý trước khi vào 3 encoder:
    - Với mỗi view $v$, học bộ hệ số điều biến nhẹ $\gamma_v(a_d) \in \mathbb{R}^{750}$ và vector bù khuyết $b_v(a_d) \in \mathbb{R}^{750}$ (khởi tạo $\gamma_v = 1, b_v = 0$):
      $$\tilde{x}^{(v)}_d = \gamma_v(a_d) \odot x^{(v)}_d + (1 - a_{d,v}) \cdot b_v(a_d)$$
    - Hỗ trợ **View Dropout khi train (`c1_view_dropout_prob`)**: trong `model.train()`, với xác suất $p_{\text{drop}}$, ngẫu nhiên thay view profile `D9, D10` của thuốc train bằng vector dự phòng out-of-fold (`fallback_d9, fallback_d10`) và hạ bit $a_{d,\text{D9}} = a_{d,\text{D10}} = 0$.
  - Đầu ra phân lớp: `5` logits cho `ce` / `emd`, hoặc `4` logits cho `corn` (kèm hàm `predict_proba(logits)` quy đổi thống nhất về xác suất 5 lớp `(batch, 5)`). Nhánh phụ `aux_cluster_head` mặc định **TẮT** (`c5_aux_cluster_weight = 0.0`) trong `MSSF_plus_full`.

### 2.5. `losses.py` (C3 — Lựa chọn hàm mục tiêu thứ tự)
- `MSSFPlusCriterion(nn.Module)`:
  - Loss phụ của MSSF gốc (giữ nguyên trọng số, không coi là thành phần mới): `0.001 * KL + 0.0001 * RecCon + 0.0001 * RecAdd`.
  - Loss chính (`c3_loss_type` — đã gỡ bỏ `ce_softquant` khỏi `MSSF_plus`):
    - `'ce'`: Cross-entropy (hỗ trợ cả nhãn cứng `0..4` và nhãn mềm `soft_target` `(batch, 5)` từ C5).
    - `'emd'`: Squared Earth Mover's Distance trên CDF của `softmax(logits)` và CDF của target.
    - `'corn'`: CORN binary cross-entropy trên 4 ngưỡng $y > k$ ($k=0..3$); với nhãn mềm $q$, chuyển trực tiếp sang target tích lũy $P(y > k) = \sum_{j > k} q_j$.
  - Cờ độc lập `c3_class_weighting: bool`: nhân trọng số nghịch tần suất chuẩn hoá $w_k = \frac{N_{\text{train}}}{5 N_{\text{train}, k}}$.

### 2.6. `soft_labels.py` (C5)
- `build_soft_targets(fold_split, fold_views, cfg) -> SoftLabelResult`:
  - `c5_mode in {'none', 'knn', 'single_view', 'multi_view'}`:
    - Chuẩn hoá từng ma trận tương đồng thuốc trên `train_drugs`, hợp nhất đa view, chạy Spectral Clustering trên đồ thị tương đồng của `train_drugs`.
    - Tự động chọn số cụm $K \in [c5\_k\_min, c5\_k\_max]$ bằng kết hợp eigengap và Silhouette score trên `train_drugs` (ghi $K$ vào log/config).
    - Với mỗi cặp train $(d, s)$ có $d \in c$: tính phân phối 5 lớp $P_{c,s}^{(-d)}$ chỉ trên các cặp train $(d', s)$ với $d' \in c, d' \neq d$ (**bắt buộc loại chính $d$**). Nếu số mẫu $< \text{min\_samples}$ (mặc định `3`) thì giữ nguyên `onehot(y)`, ngược lại:
      $$q_{d,s} = (1 - \lambda)\,\text{onehot}(y) + \lambda\, P_{c,s}^{(-d)}$$

### 2.7. `conformal.py` (C4)
- `evaluate_all_conformal_methods(valcal_probs, valcal_labels_class, test_probs, test_labels_class, alpha=0.10, test_drug_ids=None) -> dict`:
  - Hiệu chỉnh ngưỡng $\hat{q}_{1-\alpha}$ trên `val_calib` với quy tắc lượng tử hữu hạn mẫu $\lceil (n + 1)(1 - \alpha) \rceil / n$.
  - 4 phương pháp: `LAC`, `APS`, `Ordinal_LAC`, và `Ordinal_APS`.
  - Báo cáo ở mức $90\%$: `coverage`, `coverage_c0..c4`, `avg_set_size`, `singleton_rate`, `contiguity_rate`, và thống kê độ phủ theo từng thuốc test (`drug_coverage_mean`, `drug_coverage_std`, `drug_coverage_lt80_rate`).

---

## 3. Bảng Kế hoạch Ablation Bắt buộc (`configs/experiments.py`)

| Nhóm | Tên cấu hình (`exp_id`) | C1 (`impute` / `avail` / `vdrop`) | C2 (`se_mode`) | C3 (`loss` / `cw`) | C5 (`soft_label` / `aux`) | C4 (`conformal`) | Split chạy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0. Baseline** | `MSSF_v2` | `zero` / `False` / `0.0` | `original` | `ce` / `False` | `none` / `0.0` | Post-hoc | A, B |
| **1. C1 (Views)** | `C1_avail_only` | `zero` / `True` / `0.0` | `original` | `ce` / `False` | `none` / `0.0` | — | A, B |
| | `C1_vdrop_only` | `zero` / `False` / `0.25` | `original` | `ce` / `False` | `none` / `0.0` | — | A, B |
| | `C1_avail_vdrop_zero`, `C1_avail_vdrop_drop` | `zero, drop` / `True` / `0.25` | `original` | `ce` / `False` | `none` / `0.0` | — | A, B |
| | `C1_avail_vdrop_mean`, `C1_avail_vdrop_regress` | `mean, regress` / `True` / `0.25` | `original` | `ce` / `False` | `none` / `0.0` | — | A, B |
| | `C1_regress_no_modulator` *(Mới)* | `regress` / `False` / `0.25` | `original` | `ce` / `False` | `none` / `0.0` | — | A, B |
| | `C1_avail_vdrop_diffusion` *(Chỉ chạy dev folds)* | `diffusion` / `True` / `0.25` | `original` | `ce` / `False` | `none` / `0.0` | — | A, B (dev) |
| **2. C2 (SE)** | `C2_plus_ontology`, `C2_no_profile` | `zero` / `False` / `0.0` | `plus_ontology, no_profile` | `ce` / `False` | `none` / `0.0` | — | A, B |
| **3. C3 (Loss)** | `C3_{ce,emd,corn}_{unweighted,weighted}` (6 tổ hợp) | `regress` / `True` / `0.25` | `plus_ontology` | `{ce,emd,corn}` / `{F,T}` | `multi_view` / `0.0` | — | A, B |
| **4. C5 (Soft)** | `C5_knn`, `C5_single_view`, `C5_multi_view` | `zero` / `False` / `0.0` | `original` | `ce` / `False` | `{knn,single,multi}` / `0.0` | — | A, B |
| | `C5_multi_view_aux` | `zero` / `False` / `0.0` | `original` | `ce` / `False` | `multi_view` / `0.05` | — | A, B |
| **5. Full & LOO** | `MSSF_plus_full` | `regress` / `True` / `0.25` | `plus_ontology` | Best C3 | `multi_view` / `0.0` | `LAC, APS, Ord` | A, B |
| | `loo_no_C1`, `loo_no_C2`, `loo_no_C3`, `loo_no_C5` | Bỏ lần lượt từng thành phần khỏi `MSSF_plus_full` | | | | Post-hoc | A, B |

---

## 4. Thứ tự Triển khai & Ước lượng Thời gian GPU (Colab T4 / L4)

1. **Bước 1**: `protocol.py` + `metrics.py` + tái tạo `MSSF_v2` (mọi cờ tắt).
2. **Bước 2**: `losses.py` (C3: `ce`, `emd`, `corn`, `ce_softquant` × `cw`).
3. **Bước 3**: `soft_labels.py` (C5: `knn`, `single_view`, `multi_view` + optional aux cluster head).
4. **Bước 4**: `views.py` + `model.py` cho C1 (`zero`, `drop`, `mean`, `regress`, availability modulator, view dropout).
5. **Bước 5**: `views.py` cho C2 (`original`, `plus_ontology`, `no_profile`).
6. **Bước 6**: `conformal.py` cho C4 (post-hoc trên `val_calib`, 0 phút GPU huấn luyện lại).
7. **Bước 7**: `views.py` cho C1 `diffusion` (`ConditionalViewDiffusion`).
8. **Bước 8**: `configs/experiments.py` + `scripts/run_ablation.py` + `scripts/summarize_results.py`.

**Ước lượng thời gian trên Colab GPU (sau khi loại bỏ đánh giá dư thừa trên `_train` mỗi epoch và bỏ `retain_graph=True`)**:
- 1 fold (50 epochs, `batch_size=512`): **~2.5 – 4.0 phút/fold** trên T4/L4 GPU (giảm ~8× so với ~40 phút/fold của notebook cũ do notebook cũ chạy `test(model, _train)` bằng `batch_size=512` không `no_grad` ở mỗi epoch!).
- Chạy Dev (3 folds: 0, 1, 2): **~8 – 12 phút / cấu hình**.
- Chạy Full (10 folds): **~25 – 40 phút / cấu hình**.
- Huấn luyện module phụ C1 (`regress` hoặc `diffusion` trên 540 thuốc train × 750 chiều): **< 5 giây / fold**.
- Hiệu chỉnh & đánh giá C4 Conformal Prediction: **< 1 giây / fold** (post-hoc).
