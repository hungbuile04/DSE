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
  2. **Tính 4 view profile (`profile_self_correction=True`)**: `D9_profile_freq_cos`, `D10_profile_bin_cos` (`750 × 750`) và `S3_profile_freq_cos`, `S4_profile_bin_cos` (`994 × 994`) từ `drug_side_train` theo đúng công thức loại tự tương tác của `MSSF/similarity_load.py`.
  3. **C2 — Chế độ SE views (`c2_se_mode`)**:
     - `'original'`: `[S1_adrecs, S2_glove, S3_freq, S4_bin]` (4 views, `3976-D`).
     - `'plus_ontology'` (**chốt ở mô hình cuối `cand_A`**): `[S1_adrecs, S2_glove, S3_freq, S4_bin, S5_meddra_cos]` (5 views, `4970-D`), với `S5_meddra_cos = cosine_similarity(side_effect_label_750.mat['node_label'])`.
     - `'no_profile'`: `[S1_adrecs, S2_glove, S5_meddra_cos]` (3 views, `2982-D`, bỏ hoàn toàn `S3, S4`).
  4. **C1 — Rút gọn ở mô hình cuối `cand_A` còn "Mô phỏng Cold-Start khi Huấn luyện" (`c1_impute='zero'`, `c1_use_availability=False`, `use_bvi=False`)**:
     - Giữ nguyên vector `0` ở `D9, D10` cho các thuốc cold-start (`c1_impute='zero'`); trong lúc train (`model.train()`), ngẫu nhiên đặt `D9, D10` của thuốc warm-start về `0` với xác suất $p_{\text{cold}}$ (`c1_view_dropout_prob`).
     - Các biến thể bù khuyết (`regress_ridge`, `regress_knn`, `regress` MLP, `mean`, `drop`) và `AvailabilityModulator` chỉ giữ trong bảng Ablation Giai đoạn 4 (trên Folds 0–2) để chứng minh đóng góp thực sự của C1 đến từ View Dropout.

### 2.4. `model.py` (3 Encoder + Self-Attention + Chiếu Tất định `use_bvi=False` ở `cand_A`)
- `MSSFPlusModel(nn.Module)`:
  - Hỗ trợ số lượng drug views $V_d \in \{9, 11\}$ và SE views $V_s \in \{3, 4, 5\}$ tự động trong `EncoderConnection`, `EncoderAddition`, `Preprocess`, và `CrossProduction` phân rã đại số (`use_factored_crossprod=True`).
  - Ở mô hình cuối **`cand_A`**: `c1_use_availability=False`, `c1_impute='zero'`, `c1_view_dropout_prob` $\in \{0.15, 0.25\}$ (chọn trên `val_select`), `c2_se_mode='plus_ontology'`, `c3_loss_type='ce'`, `c3_class_weighting=False`, `c5_mode='multi_view'`, `use_bvi=False`, `profile_self_correction=True`.
  - Khi `use_bvi=False` (`cand_A`), nhánh biến phân `GaussianParametrizer` $(\mu, \log\sigma^2)$ được thay bằng phép chiếu tuyến tính tất định `det_proj: Linear(384, 64)`. Kết quả độ bất định hậu nghiệm $\sigma$ từ cấu hình chẩn đoán giữ BVI (`C1_avail_only`, `loo_no_C1` vs `C3_ce_unweighted_ridge_selfcorr`) được giữ làm **phân tích phụ (supplementary analysis)** cho paper.

### 2.5. `losses.py` (C3 — Hàm mục tiêu của `cand_A`)
- `MSSFPlusCriterion(nn.Module)`:
  - Ở mô hình cuối `cand_A` (`c3_loss_type='ce'`, `c3_class_weighting=False`, `use_bvi=False`), hàm mất mát huấn luyện là:
    $$\mathcal{L}_{\text{cand\_A}} = \mathcal{L}_{\text{CE}}(\text{logits}, q) + 10^{-4}\mathcal{L}_{\text{rec,con}} + 10^{-4}\mathcal{L}_{\text{rec,add}}$$
  - Trong Ablation Giai đoạn 4, 6 tổ hợp `{ce, emd, corn} × {unweighted, weighted}` đã được đối chiếu trên `val_select` (Folds 0–2, cả hai split) để chọn `ce` không trọng số lớp.

### 2.6. `soft_labels.py` (C5)
- `build_soft_targets(fold_split, fold_views, cfg) -> SoftLabelResult`:
  - `c5_mode in {'none', 'knn', 'single_view', 'multi_view'}`:
    - Chuẩn hoá từng ma trận tương đồng thuốc trên `train_drugs`, hợp nhất đa view, chạy Spectral Clustering trên đồ thị tương đồng của `train_drugs`.
    - Tự động chọn số cụm $K \in [c5\_k\_min, c5\_k\_max]$ (`[4, 12]`) bằng kết hợp eigengap và Silhouette score trên `train_drugs` (trung bình Macro-F1 ngang với `fixedK6` nhưng giảm phương sai giữa các fold từ `± 0.0268` xuống `± 0.0091`).
    - Với mỗi cặp train $(d, s)$ có $d \in c$: tính phân phối 5 lớp $P_{c,s}^{(-d)}$ chỉ trên các cặp train $(d', s)$ với $d' \in c, d' \neq d$ (**bắt buộc loại chính $d$**). Nếu số mẫu $< \text{min\_samples}$ (`5`) thì giữ nguyên `onehot(y)`, ngược lại:
      $$q_{d,s} = (1 - \alpha_{\text{c5}})\,\text{onehot}(y) + \alpha_{\text{c5}}\, P_{c,s}^{(-d)}, \qquad \alpha_{\text{c5}} = 0.15$$

### 2.7. `conformal.py` (C4)
- `evaluate_all_conformal_methods(valcal_probs, valcal_labels_class, test_probs, test_labels_class, alpha=0.10, test_drug_ids=None) -> dict`:
  - Hiệu chỉnh ngưỡng $\hat{q}_{1-\alpha}$ trên `val_calib` với quy tắc lượng tử hữu hạn mẫu $\lceil (n + 1)(1 - \alpha) \rceil / n$.
  - 4 phương pháp: `LAC`, `APS`, `Ordinal_LAC`, và `Ordinal_APS`.
  - Báo cáo ở mức $90\%$: `coverage`, `coverage_c0..c4`, `avg_set_size`, `singleton_rate`, `contiguity_rate`, và thống kê độ phủ theo từng thuốc (`drug_coverage_mean`, `drug_coverage_std`, `drug_coverage_lt80_rate`).

---

## 3. Bảng Cấu hình Thực nghiệm Chính & Leave-One-Out trên Mô hình Cuối `cand_A` (`configs/experiments.py`)

| Nhóm | Tên cấu hình (`exp_id`) | C1 (`vdrop` / `impute` / `avail`) | C2 (`se_mode`) | C3 (`loss` / `cw`) | C5 (`soft_label`) | BVI (`use_bvi`) | Vai trò / Phạm vi chạy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **0. Baseline Nội bộ** | `MSSF_v2_selfcorr` | `0.0` / `zero` / `False` | `original` | `ce` / `False` | `none` | `True` | 10 folds (Split A & B) |
| **1. Mô hình Cuối** | `cand_A` | $p_{\text{cold}}$ / `zero` / `False` | `plus_ontology` | `ce` / `False` | `multi_view` | `False` | **Mô hình cuối** — 10 folds (Split A & B) |
| **2. 4 Cấu hình LOO trên `cand_A`** | `cand_A_loo_no_vdrop` | `0.0` / `zero` / `False` | `plus_ontology` | `ce` / `False` | `multi_view` | `False` | Bỏ `vdrop` — 10 folds (Split A & B) |
| | `cand_A_loo_no_C2` | $p_{\text{cold}}$ / `zero` / `False` | `original` | `ce` / `False` | `multi_view` | `False` | Bỏ `C2` — 10 folds (Split A & B) |
| | `cand_A_loo_no_C5` | $p_{\text{cold}}$ / `zero` / `False` | `plus_ontology` | `ce` / `False` | `none` | `False` | Bỏ `C5` — 10 folds (Split A & B) |
| | `cand_A_loo_no_all` | `0.0` / `zero` / `False` | `original` | `ce` / `False` | `none` | `False` | Bỏ cả ba (`vdrop, C2, C5`) — 10 folds (Split A & B) |
| **3. Phân tích phụ $\sigma$** | `C1_avail_only`, `loo_no_C1`, `C3_ce_unweighted_ridge_selfcorr` | — | — | — | — | `True` | Chẩn đoán $\sigma$ (Folds 0–2, Split A & B) |

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
