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
| Tỷ lệ Dropout backbone | `dropout` (`model.py`) | `0.4` | Kế thừa nguyên bản từ `MSSF/mssf.py` | **Không** | Cố định |
| Chiều không gian BVI | `gp` (`model.py`) | `64` | Kế thừa nguyên bản từ `MSSF/mssf.py` | **Không** | Cố định |
| Số đầu Self-Attention | `heads` (`model.py`) | `4` | Kế thừa nguyên bản từ `MSSF/model.py` | **Không** | Cố định |
| Trọng số loss phụ MSSF gốc | `kl_weight`, `rec_con_weight`, `rec_add_weight` | `1e-3`, `1e-4`, `1e-4` | Kế thừa nguyên bản từ `MSSF/mssf.py` L306 (không coi là thành phần mới) | **Không** | Cố định |
| Tiêu chí chọn Best Epoch | `selection_metric` (`train.py`) | `'MacroF1'` | Chốt từ Audit Report, đánh giá trên `val_select` mỗi epoch | **Không** | Cố định |

---

### 1.2. Siêu tham số Thành phần `C1` (Xử lý View Thiếu & Bù khuyết Cross-Fitting)

| Siêu tham số | Khóa trong `cfg` / vị trí code | Giá trị hiện tại | Cách đã chọn / Nguồn gốc | Có dùng `test`? | Trạng thái |
| :--- | :--- | :---: | :--- | :---: | :--- |
| Chiến lược bù khuyết | `c1_impute` (`views.py`) | `'regress'` | Chọn qua kiểm chứng MSE/Spearman/Top-10 trên `val_select` Fold 0 và đối chiếu `Macro-F1` trên `val_select` Fold 0–2 (`mean` vs `regress`) | **Không** | Kiểm chứng trên `val_select` |
| Số nhóm cross-fitting $K$ | `c1_crossfit_folds` (`views.py`) | `5` | Chuẩn 5-fold cross-fitting theo thuốc trên `train_drugs` | **Không** | Cố định theo thiết kế |
| Tỷ lệ tách early-stop nội bộ | `val_frac` (`_make_drug_crossfit_splits`) | `0.10` (10%) | Tách nội bộ 10% từ 4 nhóm `train_drugs` để chọn best epoch cho từng regressor | **Không** | Cố định theo thiết kế |
| Số epoch tối đa của `ProfileViewRegressor` | `c1_imputer_epochs` (`views.py`) | `80` | Có early-stopping nội bộ trên 10% `val_sub` của `train_drugs` | **Không** | Tự chọn qua `val_sub` |
| Kiến trúc & tối ưu `ProfileViewRegressor` | `hidden_dim=256`, `dropout=0.2`, `lr=1e-3`, `wd=1e-4`, `batch_size=64`, `cos_weight=0.1` | Cố định trong `views.py` L133–236 | Thiết kế MLP chuẩn; chọn epoch bằng MSE trên `val_sub` (10% `train_drugs`) | **Không** | Cố định |
| Siêu tham số `diffusion` (`T`, `B`, `epochs`, $\beta$) | `c1_diffusion_steps=25`, `c1_diffusion_samples=10`, `c1_diffusion_epochs=100`, $\beta \in [10^{-4}, 0.2]$ | Cố định trong `views.py` L343–408 | Do `diffusion` có MSE trên `val_select` Split B cao hơn `mean` (`0.010614 > 0.010361`), đã **loại khỏi `MSSF_plus_full`, loại khỏi ngân sách tune và loại khỏi chạy 10 fold** (chỉ giữ 1 dòng ở bảng ablation C1) | **Không** | Đã đóng băng (chỉ dùng cho 1 dòng ablation) |
| Biên độ cổng điều biến `AvailabilityModulator` | Hệ số `0.2` trong $1 + 0.2\tanh(\cdot)$ (`model.py` L75) | `0.2` | Thiết kế kiến trúc khởi tạo đồng nhất ($W_2=0, b_2=0, m_v=0 \implies \tilde{X}=X$ tại $t=0$); kiểm chứng qua ablation `C1_regress_no_modulator` trên `val_select` | **Không** | Kiểm chứng qua ablation trên `val_select` |
| **Xác suất Profile View Dropout khi train** | `c1_view_dropout_prob` (`experiments.py`) | `0.25` | Khởi tạo heuristic ban đầu ($0.25$), **chưa chọn trên `val_select`** | **Không** | **Sẽ tune trên `val_select` fold 0–2 (trong ngân sách $\le 6$ cấu hình)** |

---

### 1.3. Siêu tham số Thành phần `C2` (View Ontology MedDRA cho Tác dụng phụ)

| Siêu tham số | Khóa trong `cfg` / vị trí code | Giá trị hiện tại | Cách đã chọn / Nguồn gốc | Có dùng `test`? | Trạng thái |
| :--- | :--- | :---: | :--- | :---: | :--- |
| Chế độ SE Views | `c2_se_mode` (`views.py`) | `'plus_ontology'` | Dùng ma trận chuẩn `side_effect_label_750.mat['node_label']` ($994 \times 243$, chốt từ Audit Report); kiểm chứng bật/tắt qua ablation `C2` & `loo_no_C2` trên `val_select` | **Không** | Không có siêu tham số số; kiểm chứng qua ablation |

---

### 1.4. Siêu tham số Thành phần `C3` (Lựa chọn Hàm Mục tiêu Ordinal & Cân bằng Lớp)

| Siêu tham số | Khóa trong `cfg` / vị trí code | Giá trị hiện tại | Cách đã chọn / Nguồn gốc | Có dùng `test`? | Trạng thái |
| :--- | :--- | :---: | :--- | :---: | :--- |
| Loại hàm mục tiêu & Cân bằng lớp | `c3_loss_type` $\in \{\text{'ce'}, \text{'emd'}, \text{'corn'}\}$, `c3_class_weighting` $\in \{\text{False}, \text{True}\}$ | Đang chạy chọn trên fold 0–2 | Chạy đủ 6 tổ hợp trên `fold 0–2` (Split A & B), chọn theo `Macro-F1` trung bình trên `val_select` (ưu tiên `ce > emd > corn` nếu chênh lệch $< 1\text{ std}$) | **Không** | **Chọn trực tiếp trên `val_select` fold 0–2 (Bước 3)** |
| Trọng số lớp $w_k$ | `compute_class_weights_from_train` (`losses.py`) | $w_k = \frac{N_{\text{train}}}{5\,N_{\text{train},k}}$ | Công thức nghịch đảo tần suất xác định hoàn toàn từ tập `train` (80%) của từng fold | **Không** | Tất định từ `train` |

---

### 1.5. Siêu tham số Thành phần `C4` (Tiên đoán Bảo toàn Thứ tự Liên thông Hậu kiểm)

| Siêu tham số | Khóa trong `cfg` / vị trí code | Giá trị hiện tại | Cách đã chọn / Nguồn gốc | Có dùng `test`? | Trạng thái |
| :--- | :--- | :---: | :--- | :---: | :--- |
| Mức sai số mục tiêu $\alpha$ | `c4_alpha` (`conformal.py`) | `0.10` (Độ phủ mục tiêu $90\%$) | Mức tin cậy thống kê tiêu chuẩn đặt trước ($1 - \alpha = 90\%$); ngưỡng $\hat{q}_{1-\alpha}$ tính hoàn toàn trên `val_calib` (10%) | **Không** | Cố định theo định nghĩa bài toán |

---

### 1.6. Siêu tham số Thành phần `C5` (Phân cụm Phổ Đa khung nhìn & Nhãn mềm Leave-One-Out)

| Siêu tham số | Khóa trong `cfg` / vị trí code | Giá trị hiện tại | Cách đã chọn / Nguồn gốc | Có dùng `test`? | Trạng thái |
| :--- | :--- | :---: | :--- | :---: | :--- |
| Khoảng duyệt số cụm $[K_{\min}, K_{\max}]$ | `c5_k_min`, `c5_k_max` (`soft_labels.py`) | `[4, 12]` | Miền cụm tự nhiên cho $540\text{–}750$ thuốc; $K^*$ tự động chọn không giám sát trên `train_drugs` | **Không** | Cố định |
| Trọng số tiêu chí chọn $K^*$ | `0.6 * Silhouette + 0.4 * NormEigengap` (`soft_labels.py` L131) | `(0.6, 0.4)` | Chỉ dùng ma trận tương đồng trên `train_drugs` (hoàn toàn không dùng nhãn `val` hay `test`) | **Không** | Cố định |
| Trọng số ghép các view khi phân cụm | `c5_view_weighting` (`soft_labels.py`) | `'uniform'` | Trung bình đều các Laplacian chuẩn hóa đối xứng trên `train_drugs` | **Không** | Cố định |
| **Hệ số trộn nhãn mềm $\alpha_{\text{c5}}$** | `c5_alpha` (`experiments.py`) | `0.15` | Khởi tạo heuristic ban đầu ($0.15$), **chưa chọn trên `val_select`** | **Không** | **Sẽ tune trên `val_select` fold 0–2 (trong ngân sách $\le 6$ cấu hình)** |
| **Ngưỡng số láng giềng tối thiểu $m_{\min}$** | `c5_min_samples` (`experiments.py`) | `5` | Khởi tạo heuristic ban đầu ($5$), **chưa chọn trên `val_select`** | **Không** | **Sẽ tune trên `val_select` fold 0–2 (trong ngân sách $\le 6$ cấu hình)** |
| Trọng số nhánh phụ `aux_cluster_head` | `c5_aux_cluster_weight` (`experiments.py`) | `0.0` (Mặc định **TẮT** trong `MSSF_plus_full`) | Đã tắt mặc định theo Thay đổi 2; chỉ kiểm tra ở dòng ablation `C5_multi_view_aux` (`0.05`) trên `val_select` | **Không** | Mặc định TẮT |
| Siêu tham số đối chứng `C5_knn` / `C5_single_view` | `c5_knn_k=15`, `c5_single_view_name='D5_stitch_five'` | `15`, `'D5_stitch_five'` | Chỉ dùng cho 2 dòng đối chứng trong bảng ablation `C5` | **Không** | Cố định |

---

## 2. Kế hoạch Sử dụng Ngân sách Tune $\le 6$ Cấu hình Số (Bước 4 — Sau khi chốt C3)

Sau khi chọn xong hàm mục tiêu `C3` tốt nhất từ thực nghiệm fold `0–2` trên `val_select` (Bước 3), toàn bộ ngân sách tinh chỉnh $\le 6$ cấu hình số của `MSSF_plus` sẽ được dùng để chọn bộ ba siêu tham số liên tục chưa được chọn trên `val_select` gồm `(c1_view_dropout_prob, c5_alpha, c5_min_samples)` trên `fold 0–2` (cả Split A và Split B):

| Thứ tự cấu hình | `c1_view_dropout_prob` | `c5_alpha` | `c5_min_samples` | Mục đích kiểm tra trên `val_select` (fold 0–2) |
| :---: | :---: | :---: | :---: | :--- |
| **Config 1** *(Mặc định hiện tại)* | `0.25` | `0.15` | `5` | Điểm neo tham chiếu (đã chạy sẵn trong bước chọn C3, tốn 0 lượt chạy mới) |
| **Config 2** | `0.15` | `0.15` | `5` | Giảm tỷ lệ ẩn profile view khi train ($p_{\text{drop}} = 0.15$) |
| **Config 3** | `0.35` | `0.15` | `5` | Tăng tỷ lệ giả lập cold-start khi train ($p_{\text{drop}} = 0.35$) |
| **Config 4** | *(Tốt nhất từ 1–3)* | `0.10` | `5` | Làm trơn nhãn mềm bảo thủ hơn ($\alpha_{\text{c5}} = 0.10$) |
| **Config 5** | *(Tốt nhất từ 1–3)* | `0.25` | `5` | Tăng đóng góp phân phối cụm phổ ($\alpha_{\text{c5}} = 0.25$) |
| **Config 6** | *(Tốt nhất từ 1–5)* | *(Tốt nhất từ 1–5)* | `3` (hoặc `8`) | Kiểm tra độ nhạy ngưỡng số láng giềng tối thiểu trong cụm $m_{\min}$ |

- **Tổng số cấu hình số được thử**: đúng **6 cấu hình** (trong đó Config 1 dùng lại trực tiếp từ Bước 3, chỉ cần chạy thêm 5 cấu hình mới trên fold 0–2).
- **Quy tắc chọn**: Chọn cấu hình có trung bình `Macro-F1` trên `val_select` (fold 0–2, xét đồng thời Split B và Split A) cao nhất; nếu chênh lệch nhỏ hơn độ lệch chuẩn giữa các fold, giữ giá trị mặc định đơn giản hơn.
