# Minh bạch Siêu tham số & Quy trình Lựa chọn (`MSSF_plus/HPARAM_SELECTION.md`)

> **Nguyên tắc bất biến**:
> 1. **Tuyệt đối không dùng tập `test`** để chọn epoch, siêu tham số, hàm mất mát hay bật/tắt thành phần kiến trúc.
> 2. Mọi quyết định lựa chọn mô hình, thành phần ablation và siêu tham số đều chỉ dựa trên **`val_select` (10%)** ở các fold phát triển **`fold 0, 1, 2`** (hoặc early-stopping nội bộ trên `train_drugs` đối với mô hình bù khuyết `C1`).
> 3. **Tổng ngân sách tinh chỉnh (tuning budget)** cho các siêu tham số liên tục của `MSSF_plus` là **$\le 6$ cấu hình số**, bằng đúng ngân sách tối đa của mỗi mô hình baseline trong Protocol v2.

---

## 1. Bảng Kiểm kê Toàn bộ Siêu tham số của Hệ thống

### 1.1. Siêu tham số Backbone & Tối ưu hóa (Kế thừa cố định từ `MSSF` gốc)

| Siêu tham số | Khóa trong `cfg` / vị trí code | Giá trị hiện tại | Cách đã chọn / Nguồn gốc | Có dùng `test`? | Trạng thái |
| :--- | :--- | :---: | :--- | :---: | :--- |
| Số epoch huấn luyện | `epochs` (`experiments.py`) | `50` | Kế thừa nguyên bản từ `MSSF/mssf.py` & `notebook1` | **Không** | Cố định |
| Kích thước mini-batch | `batch_size` / `eval_batch_size` | `512` / `512` | Kế thừa từ cấu hình chạy chuẩn của `MSSF` | **Không** | Cố định |
| Tốc độ học & L2 decay | `lr` / `weight_decay` | `1e-4` / `1e-5` | Kế thừa nguyên bản từ `MSSF/mssf.py` (Adam) | **Không** | Cố định |
| Tỷ lệ Dropout backbone | `dropout` (`experiments.py`, `model.py`) | `0.4` | Kế thừa nguyên bản từ `MSSF/mssf.py` (`Dropout(p=0.4, inplace=False)` trong `print(model)`) | **Không** | Cố định |
| Chiều không gian chiếu đặc trưng hợp nhất | `gp` (`model.py`) | `64` (`use_bvi=False` ở mô hình cuối `cand_A`) | Kế thừa chiều `64` từ `MSSF/mssf.py`; mô hình cuối `cand_A` **bỏ BVI** (`use_bvi=False`, dùng chiếu tất định `Linear(384, 64)` sau khi so sánh 2×2 BVI trên `val_select`) | **Không** | Cố định (`use_bvi=False` ở `cand_A`) |
| Số đầu Self-Attention | `heads` (`model.py`) | `4` | Kế thừa nguyên bản từ `MSSF/model.py` | **Không** | Cố định |
| Trọng số loss phụ MSSF gốc | `kl_weight`, `rec_con_weight`, `rec_add_weight` | `0.0` (bỏ KL khi `use_bvi=False`), `1e-4`, `1e-4` | Kế thừa nguyên bản từ `MSSF/mssf.py` L306; ở `cand_A` (`use_bvi=False`) không dùng $\mathcal{L}_{\text{KL}}$ | **Không** | Cố định |
| Tiêu chí chọn Best Epoch | `selection_metric` (`train.py`) | `'MacroF1'` | Chốt từ Audit Report, đánh giá trên `val_select` mỗi epoch | **Không** | Cố định |

---

### 1.2. Siêu tham số Thành phần `C1` — Rút gọn còn "Mô phỏng Cold-Start khi Huấn luyện" ở Mô hình Cuối (`cand_A`)

> **Kết luận từ Giai đoạn 4 & Chọn Ứng viên Mô hình Cuối (`cand_A..cand_D`) trên `val_select`**:
> - Đóng góp của `C1` đến từ **Profile View Dropout khi huấn luyện** (`C1_vdrop_only` đạt `0.4176 ± 0.0258` trên Split B, ngang hoặc vượt mọi biến thể có bù khuyết `0.4022–0.4121`, trong khi `C1_avail_only` không có View Dropout chỉ đạt `0.3413 ± 0.0149`).
> - So sánh 4 ứng viên `cand_A..cand_D` trên Folds 0–2 (`val_select`, cả hai split) theo quy tắc ưu tiên cấu hình đơn giản hơn khi $|\Delta\text{ vs best}| \le 1\text{ Std}$ giữa các fold đã chọn **`cand_A`** (`vdrop_only + C2 + C5 + CE`, **bỏ BVI**, **KHÔNG bù khuyết** `c1_impute='zero'`, **KHÔNG modulator** `c1_use_availability=False`).
> - Do đó, ở mô hình cuối `cand_A`, thành phần `C1` chỉ còn duy nhất 1 siêu tham số là **xác suất ẩn view hồ sơ tương tác khi train** (`c1_view_dropout_prob`, ký hiệu $p_{\text{cold}}$).

| Siêu tham số | Khóa trong `cfg` / vị trí code | Giá trị ở `cand_A` | Cách đã chọn / Nguồn gốc | Có dùng `test`? | Trạng thái |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **Xác suất mô phỏng Cold-Start khi train ($p_{\text{cold}}$)** | `c1_view_dropout_prob` (`experiments.py`) | So sánh $p_{\text{cold}} \in \{0.15, 0.25\}$ trên `val_select` | Trong lần chạy 4 ứng viên `cand_A..cand_D` (`experiments.py` L460), `cand_A` dùng giá trị khởi tạo heuristic $p_{\text{cold}} = 0.25$ (ghi chú `p = 0.15` trong bản tóm tắt văn bản trước đó là nhầm với `c5_alpha = 0.15`). Để chọn minh bạch trên `val_select`, chạy trực tiếp `cand_A` với $p_{\text{cold}} \in \{0.15, 0.25\}$ (`cand_A_p015` vs `cand_A`) trên Fold 0–2, Split B qua `MSSF_plus/scripts/run_p_cold_selection.py` | **Không** | **Chọn trực tiếp trên `val_select` Fold 0–2, Split B (`{0.15, 0.25}`)** |
| Chiến lược bù khuyết | `c1_impute` (`views.py`) | `'zero'` (Không bù khuyết ở `cand_A`) | Đã kiểm chứng qua Ablation Giai đoạn 4 và `cand_A..cand_D` trên `val_select`: bỏ bù khuyết ở mô hình cuối `cand_A` (`c1_impute='zero'`). Các cấu hình có bù khuyết (`regress_ridge`, `regress_knn`, `regress` MLP, `mean`) chỉ giữ trong bảng Ablation Giai đoạn 4 và phân tích phụ $\sigma$ | **Không** | **Đã loại bỏ khỏi mô hình cuối `cand_A`** |
| Bộ điều biến hiện diện `AvailabilityModulator` | `c1_use_availability` (`model.py`) | `False` (Tắt ở `cand_A`) | Đã kiểm chứng qua `cand_A` vs `cand_B` trên `val_select` Fold 0–2 ($|\Delta| < 1\text{ Std}$): bỏ `AvailabilityModulator` ở mô hình cuối `cand_A` | **Không** | **Đã loại bỏ khỏi mô hình cuối `cand_A`** |

---

### 1.3. Siêu tham số Thành phần `C2` (View Ontology MedDRA cho Tác dụng phụ)

| Siêu tham số | Khóa trong `cfg` / vị trí code | Giá trị ở `cand_A` | Cách đã chọn / Nguồn gốc | Có dùng `test`? | Trạng thái |
| :--- | :--- | :---: | :--- | :---: | :--- |
| Chế độ SE Views | `c2_se_mode` (`views.py`) | `'plus_ontology'` | Dùng ma trận chuẩn `side_effect_label_750.mat['node_label']` ($994 \times 243$, chốt từ Audit Report); kiểm chứng bật/tắt qua ablation `C2` & `cand_A_loo_no_C2` | **Không** | Không có siêu tham số số; cố định ở `cand_A` |

---

### 1.4. Siêu tham số Thành phần `C3` (Lựa chọn Hàm Mục tiêu Ordinal & Cân bằng Lớp)

| Siêu tham số | Khóa trong `cfg` / vị trí code | Giá trị ở `cand_A` | Cách đã chọn / Nguồn gốc | Có dùng `test`? | Trạng thái |
| :--- | :--- | :---: | :--- | :---: | :--- |
| Loại hàm mục tiêu & Cân bằng lớp | `c3_loss_type='ce'`, `c3_class_weighting=False` | `'ce'` (Unweighted) | Đã chạy đủ 6 tổ hợp `{ce, emd, corn} × {unweighted, weighted}` trên `fold 0–2` (Split B & A, `reports/raw/stage4_summarize_results.txt`): `C3_ce_unweighted_ridge_selfcorr` đạt đỉnh trung bình (`0.4179 ± 0.0230` trên Split B; `0.6509 ± 0.0074` trên Split A) $\to$ chốt `ce` không trọng số lớp cho `cand_A` | **Không** | **Đã chọn trên `val_select` fold 0–2 (Giai đoạn 4)** |

---

### 1.5. Siêu tham số Thành phần `C4` (Tiên đoán Bảo toàn Thứ tự Liên thông Hậu kiểm)

| Siêu tham số | Khóa trong `cfg` / vị trí code | Giá trị ở `cand_A` | Cách đã chọn / Nguồn gốc | Có dùng `test`? | Trạng thái |
| :--- | :--- | :---: | :--- | :---: | :--- |
| Mức sai số mục tiêu $\alpha$ | `c4_alpha` (`conformal.py`) | `0.10` (Độ phủ mục tiêu $90\%$) | Mức tin cậy thống kê tiêu chuẩn đặt trước ($1 - \alpha = 90\%$); ngưỡng $\hat{q}_{1-\alpha}$ tính hoàn toàn trên `val_calib` (10%) | **Không** | Cố định theo định nghĩa bài toán |

---

### 1.6. Siêu tham số Thành phần `C5` (Phân cụm Phổ Đa khung nhìn & Nhãn mềm Leave-One-Out)

| Siêu tham số | Khóa trong `cfg` / vị trí code | Giá trị ở `cand_A` | Cách đã chọn / Nguồn gốc | Có dùng `test`? | Trạng thái |
| :--- | :--- | :---: | :--- | :---: | :--- |
| Khoảng duyệt số cụm $[K_{\min}, K_{\max}]$ | `c5_k_min`, `c5_k_max` (`soft_labels.py`) | `[4, 12]` | Đã kiểm chứng trên Fold 0–2 (`reports/raw/c5_k_ranges_fold012.txt`): khoảng `[4, 12]` và `[4, 24]` cho cùng $K^* = (12, 5, 4)$ ở Split B và $(6, 6, 12)$ ở Split A; tự chọn $K^*$ có trung bình Macro-F1 ngang `fixedK6` (`0.3701` vs `0.3693`) nhưng giảm phương sai giữa các fold từ `± 0.0268` xuống `± 0.0091` | **Không** | Cố định `[4, 12]` |
| Trọng số tiêu chí chọn $K^*$ | `0.6 * Silhouette + 0.4 * NormEigengap` (`soft_labels.py` L131) | `(0.6, 0.4)` | Chỉ dùng ma trận tương đồng trên `train_drugs` (hoàn toàn không dùng nhãn `val` hay `test`) | **Không** | Cố định |
| Trọng số ghép các view khi phân cụm | `c5_view_weighting` (`soft_labels.py`) | `'uniform'` | Trung bình đều các Laplacian chuẩn hóa đối xứng trên `train_drugs` | **Không** | Cố định |
| Hệ số trộn nhãn mềm $\alpha_{\text{c5}}$ | `c5_alpha` (`experiments.py`) | `0.15` | Giá trị chuẩn cố định xuyên suốt toàn bộ thực nghiệm C5 (`0.15`) | **Không** | Cố định (`0.15`) |
| Ngưỡng số láng giềng tối thiểu $m_{\min}$ | `c5_min_samples` (`experiments.py`) | `5` | Giá trị chuẩn cố định xuyên suốt toàn bộ thực nghiệm C5 (`5`) | **Không** | Cố định (`5`) |
| Trọng số nhánh phụ `aux_cluster_head` | `c5_aux_cluster_weight` (`experiments.py`) | `0.0` (**TẮT** ở `cand_A`) | Đã kiểm chứng trên `val_select` Fold 0–2 (`C5_multi_view` `0.3701 ± 0.0091` $\ge$ `C5_multi_view_aux` `0.3665 ± 0.0115`) $\to$ tắt hoàn toàn ở `cand_A` | **Không** | Đã tắt (`0.0`) |

---

## 2. Giải trình & Quy trình Chọn Minh bạch $p_{\text{cold}} \in \{0.15, 0.25\}$ cho `cand_A` trên `val_select` (Fold 0–2, Split B)

1. **Giải trình về con số `0.15` vs `0.25` trước đó**:
   - Trong mã nguồn [`MSSF_plus/configs/experiments.py`](file:///Users/buihung/project_DL/DSE/MSSF_plus/configs/experiments.py#L457-L470), cấu hình `cand_A` đã chạy trên Folds 0–2 thực tế đặt **`c1_view_dropout_prob = 0.25`** và **`c5_alpha = 0.15`**.
   - Việc ghi `vdrop_only (p = 0.15)` trong văn bản tóm tắt trước đó là do nhầm giá trị của `c5_alpha = 0.15` sang `c1_view_dropout_prob`; thực tế tại thời điểm đó $p_{\text{cold}}$ vẫn đang giữ ở mức khởi tạo heuristic `0.25` và **chưa được so sánh trực tiếp với `0.15` trên `val_select`**.
2. **So sánh thực nghiệm minh bạch $p_{\text{cold}} \in \{0.15, 0.25\}$ trên `val_select` (Fold 0–2, Split B)**:
   - Hai cấu hình đối chiếu trong `MSSF_plus/configs/experiments.py`:
     - **`cand_A_p015`**: `cand_A` với `c1_view_dropout_prob = 0.15`
     - **`cand_A`**: `cand_A` với `c1_view_dropout_prob = 0.25` *(đã chạy xong trên Fold 0–2, Split B: Macro-F1 = `0.4107 ± 0.0259`, từng fold `[0.4260, 0.3742, 0.4319]`)*
   - Lệnh chạy và xuất bảng nguyên văn từ `.npy`:
     ```bash
     python3 -m MSSF_plus.scripts.run_p_cold_selection \
         --splits B --mode dev --device cuda \
         --results_dir /content/DSE/results_v2 \
         --drive_dir /content/drive/MyDrive/DSE_results_v2
     ```
   - **Quy tắc chọn**: So sánh theo cặp fold trên `val_select` (Fold 0–2, Split B); nếu chênh lệch trung bình nhỏ hơn $1\text{ Std}$ giữa các fold thì giữ giá trị mặc định `0.25`, ngược lại chọn giá trị có `val_select` Macro-F1 cao hơn. Kết quả nguyên văn được lưu tại `reports/raw/p_cold_selection_summarize_results.txt`.

