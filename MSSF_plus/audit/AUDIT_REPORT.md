# Báo cáo Đọc hiểu (Bước 0) & Kiểm toán Thực nghiệm (Bước 1) — Dự án `MSSF_plus`

> **Mục tiêu tài liệu**: Ghi nhận toàn bộ kết quả đọc hiểu kiến trúc/dữ liệu gốc (Bước 0) và bằng chứng kiểm toán định lượng bằng script (Bước 1) trước khi triển khai `MSSF_plus/protocol.py` (Bước 2) và `MSSF_plus/PLAN.md` (Bước 3).

---

## 0. Môi trường & Danh mục File Kiểm toán đã tạo

- **Môi trường Python riêng**: `MSSF_plus/.venv` (Python 3.12.9)
  - File khai báo: [`MSSF_plus/requirements.txt`](../requirements.txt) (chỉ gồm 4 thư viện được phép: `numpy`, `scipy`, `scikit-learn`, `torch`).
  - File khoá phiên bản: [`MSSF_plus/requirements-lock.txt`](../requirements-lock.txt) (`numpy==2.5.4`, `scipy==1.18.1`, `scikit-learn==1.9.1`, `torch==2.14.1`).
- **Các script kiểm toán & kết quả JSON** (hoạt động ở chế độ **read-only**, tuân thủ Nguyên tắc bất biến #1 — không sửa bất kỳ file nào ngoài `MSSF_plus/`):
  - [`MSSF_plus/audit/common.py`](common.py) — hàm dùng chung tái lập chính xác bước đọc & tính feature của MSSF.
  - [`MSSF_plus/audit/audit_A_feature_leakage.py`](audit_A_feature_leakage.py) → [`outputs/audit_A_feature_leakage.json`](outputs/audit_A_feature_leakage.json)
  - [`MSSF_plus/audit/audit_B_coldstart_views.py`](audit_B_coldstart_views.py) → [`outputs/audit_B_coldstart_views.json`](outputs/audit_B_coldstart_views.json)
  - [`MSSF_plus/audit/audit_B2_static_view_sparsity.py`](audit_B2_static_view_sparsity.py) → [`outputs/audit_B2_static_view_sparsity.json`](outputs/audit_B2_static_view_sparsity.json)
  - [`MSSF_plus/audit/audit_C_selection.py`](audit_C_selection.py) → [`outputs/audit_C_selection.json`](outputs/audit_C_selection.json)
  - [`MSSF_plus/audit/audit_D_ontology.py`](audit_D_ontology.py) → [`outputs/audit_D_ontology.json`](outputs/audit_D_ontology.json)
  - [`MSSF_plus/audit/audit_E_class_distribution.py`](audit_E_class_distribution.py) → [`outputs/audit_E_class_distribution.json`](outputs/audit_E_class_distribution.json)

---

## 1. Bước 0 — Tóm tắt đọc hiểu code gốc

### 1.1. Cách MSSF load và ghép 11 drug matrix + 4 SE matrix
Trong [`MSSF/mssf.py` (`read_raw_data`, dòng 31–138)](../../MSSF/mssf.py#L31-L138) và [`fold_files` (dòng 142–183)](../../MSSF/mssf.py#L142-L183):

- **11 ma trận tương đồng thuốc (Drug views)**, mỗi ma trận kích thước `750 × 750` (trong bộ dữ liệu chuẩn hoá `shared_data/mssf_750/`; bản gốc của MSSF là `757 × 757`), được ghép ngang bằng `np.hstack` (dòng 158–159) tạo thành vector đặc trưng **8,250 chiều** cho mỗi thuốc theo đúng thứ tự:

| Thứ tự (1-based) | Chỉ số (0-based) | Tên view trong audit | Nguồn dữ liệu & cách tính trong `read_raw_data()` | Phụ thuộc `drug_side.pkl`? |
| :---: | :---: | :--- | :--- | :---: |
| **Drug [1]** | `0` | `D1_stitch_one` | `Text_similarity_one.pkl` (STITCH subchannel 1) | Không |
| **Drug [2]** | `1` | `D2_stitch_two` | `Text_similarity_two.pkl` (STITCH experimental) | Không |
| **Drug [3]** | `2` | `D3_stitch_three` | `Text_similarity_three.pkl` (STITCH database) | Không |
| **Drug [4]** | `3` | `D4_stitch_four` | `Text_similarity_four.pkl` (STITCH text mining) | Không |
| **Drug [5]** | `4` | `D5_stitch_five` | `Text_similarity_five.pkl` (STITCH combined) | Không |
| **Drug [6]** | `5` | `D6_mol2vec_cos` | `cosine_similarity(drug_mol.pkl)` (`750 × 100` → `750 × 750`, dòng 64–67) | Không |
| **Drug [7]** | `6` | `D7_target_cos` | `cosine_similarity(drug_target.pkl)` (`750 × 847` → `750 × 750`, dòng 76–79) | Không |
| **Drug [8]** | `7` | `D8_fingerprint` | `fingerprint_similarity.pkl` (Morgan FP Jaccard) | Không |
| **Drug [9]** | `8` | `D9_profile_freq_cos` | `cosine_similarity(drug_side)` — tương đồng cosine trên ma trận tần suất sau khi xoá cặp test (dòng 101) | **CÓ (Profile)** |
| **Drug [10]** | `9` | `D10_profile_bin_cos` | `cosine_similarity(drug_side_label)` với `drug_side_label = (drug_side > 0)` (dòng 104–111) | **CÓ (Profile)** |
| **Drug [11]** | `10` | `D11_pathway_enzyme` | `drug_pathway_enzyme_similarity.pkl` (DrugBank pathway + enzyme, dòng 92–94) | Không |

- **4 ma trận tương đồng tác dụng phụ (SE views)**, mỗi ma trận kích thước `994 × 994`, ghép ngang bằng `np.hstack` (dòng 165–166) tạo thành vector đặc trưng **3,976 chiều** cho mỗi SE:

| Thứ tự (1-based) | Chỉ số (0-based) | Tên view trong audit | Nguồn dữ liệu & cách tính trong `read_raw_data()` | Phụ thuộc `drug_side.pkl`? |
| :---: | :---: | :--- | :--- | :---: |
| **SE [1]** | `0` | `S1_adrecs_semantic` | `side_effect_semantic.pkl` (ADReCS DAG semantic similarity, dòng 59–61) | Không |
| **SE [2]** | `1` | `S2_glove_cos` | `cosine_similarity(glove_wordEmbedding.pkl)` (`994 × 300` → `994 × 994`, dòng 70–73) | Không |
| **SE [3]** | `2` | `S3_profile_freq_cos` | `cosine_similarity(drug_side.T)` — tương đồng cosine trên cột tần suất (dòng 128) | **CÓ (Profile)** |
| **SE [4]** | `3` | `S4_profile_bin_cos` | `cosine_similarity(drug_side_label.T)` — tương đồng cosine trên cột nhị phân (dòng 129) | **CÓ (Profile)** |

---

### 1.2. Self-attention của MSSF fuse những gì?
Trong [`MSSF/model.py`](../../MSSF/model.py):
1. Ba nhánh encoder độc lập trích xuất đặc trưng mức cặp `(drug, SE)`:
   - **`EncoderConnection` (EN-con, dòng 72–115)**: nối `[drugs (8250), sides (3976)]` → `Linear(12226 → 256) + BN + LeakyReLU + Dropout` → `Attention(256, heads=4)` → `Linear(256 → 128)` tạo `feature1` (`batch × 128`), đồng thời có nhánh decoder tái tạo vector nối `recCon` (`batch × 12226`).
   - **`EncoderAddition` (EN-add, dòng 119–173)**: tách 11 drug views (`chunk(11, dim=1)`) cộng từng phần tử thành vector `750-D`, tách 4 SE views (`chunk(4, dim=1)`) cộng từng phần tử thành vector `994-D`, nối thành `add_features` (`1744-D`) → `Linear(1744 → 256) + BN + LeakyReLU + Dropout` → `Attention(256, heads=4)` → `Linear(256 → 128)` tạo `feature2` (`batch × 128`) và bản tái tạo `recAdd` (`batch × 1744`).
   - **`Preprocess` + `CrossProduction` (CNN-im, dòng 178–348)**: chiếu riêng từng drug view (`750 → 128`) và từng SE view (`994 → 128`), tính `11 × 4 = 44` tích ngoài `torch.bmm(drug_i.unsqueeze(2), side_j.unsqueeze(1))` tạo tensor 4D `(batch, 44, 128, 128)` → qua 3 tầng `Conv2d(kernel_size=4, stride=4) + BN + ReLU` (`128 → 32 → 8 → 2`, kênh `44 → 32 → 32 → 32`) → làm phẳng thành `feature3` (`batch × 128`, vì `32 × 2 × 2 = 128`).
2. **Self-attention fusion ([`Mulmodel.forward`, dòng 416–418](../../MSSF/model.py#L416-L418))**:
   - Ghép 3 đầu ra encoder: `features = torch.cat((feature1, feature2, feature3), dim=1)` có kích thước `(batch, 384)`.
   - Đưa qua `self.attention = Attention(inputdim=384, heads=4)` (dòng 27–68).
   - **Chi tiết kỹ thuật trong [`Attention.forward` (dòng 49–67)](../../MSSF/model.py#L49-L67)**:
     ```python
     Q = self.WQ(x).view(-1, self.heads, self.dq).transpose(0, 1)  # shape: (heads, batch, dq)
     K = self.WK(x).view(-1, self.heads, self.dk).transpose(0, 1)
     V = self.WV(x).view(-1, self.heads, self.dv).transpose(0, 1)
     QK = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.dk)  # shape: (heads, batch, batch)
     ```
     Do `x` là tensor 2D `(batch, 384)`, phép `.view(-1, heads, dq).transpose(0, 1)` tạo tensor `(heads=4, batch, dq=96)`, nên `QK` thực hiện attention giữa các mẫu trong mini-batch trên không gian đặc trưng hợp nhất 384 chiều, sau đó qua 2 lớp residual `LayerNorm` + feed-forward `Linear(384, 384)`.
3. Đầu ra `features` (`batch × 384`) đi qua **`GaussianParametrizer` (BVI, dòng 6–25, 390–401)** sinh `mu, logvar` (`batch × 64`), lấy mẫu `latent_features` bằng reparameterization khi `self.training=True` (và dùng trực tiếp `mu` khi `eval()`), rồi qua **`Classifier` (dòng 351–374)**: `Linear(64 → 32) + LeakyReLU + Dropout → Linear(32 → 5)` xuất logits 5 lớp.

---

### 1.3. Format trả về của `load_splits()`
Trong [`shared_data/split_adapter.py` (dòng 21–41)](../../shared_data/split_adapter.py#L21-L41):
- `load_splits(fold, split_type='A')` đọc `shared_data/splits/split{A|B}_fold{fold}_{train|test}.npy` và trả về tuple 2 phần tử `(train_data, test_data)`.
- Mỗi phần tử là `np.ndarray` (`dtype=float64`, shape `(N, 3)`) **chỉ chứa các cặp dương** (`frequency > 0`), với thứ tự 3 cột:
  $$\text{row} = [\text{drug\_idx}, \text{se\_idx}, \text{frequency}]$$
  trong đó `drug_idx` $\in \{0, \dots, 749\}$, `se_idx` $\in \{0, \dots, 993\}$, `frequency` $\in \{1.0, 2.0, 3.0, 4.0, 5.0\}$.
- Với Split B, hàm `load_test_drugs(fold, split_type='B')` đọc `splitB_fold{fold}_test_drugs.npy` trả về mảng 1D shape `(75,)` chứa chỉ số của 75 thuốc cold-start bị giữ lại làm test.

---

### 1.4. Format file mà `unified_eval.py` đọc
Trong [`shared_data/unified_eval.py` (`evaluate_method`, dòng 260–306)](../../shared_data/unified_eval.py#L260-L306):
- Kỳ vọng thư mục con `results/<Method>_<A|B>/` chứa các cặp file:
  - `fold_{k}_labels.npy`: nhãn thực, được đọc bằng `np.load(...).flatten().astype(float)`.
  - `fold_{k}_preds.npy`: giá trị dự đoán, cùng kích thước và thứ tự phần tử với `fold_{k}_labels.npy`.
- Trong [`compute_all_metrics` (dòng 135–253)](../../shared_data/unified_eval.py#L135-L253):
  - Kiểm tra `has_negatives = np.any(y_true == 0)`.
  - Với các mô hình chỉ đánh giá trên cặp dương như MSSF: notebook cộng `+1` vào nhãn/dự đoán lớp `{0..4}` trước khi lưu để về thang `{1, 2, 3, 4, 5}` (`has_negatives = False`). Khi đó:
    - Các metric nhị phân/ranking (`AUC`, `AUPR`, `F1`, `P@K`, `drugAUC`, ...) được gán `np.nan`.
    - Các metric Regression (`RMSE`, `MAE`, `MSE`, `Pearson`, `Spearman`, `R2`) và Multi-class 5 mức (`ClassAcc`, `WeightedF1`, `MacroF1`, `ClassKappa`, sau khi `np.clip(np.round(yp_k), 1, 5).astype(int)`) được tính trên toàn bộ các cặp dương trong file.

---

## 2. Bước 1 — Báo cáo Kiểm toán Thực nghiệm (Audit A – E)

### Audit A — Kiểm tra rò rỉ feature ở Split A
- **Script kiểm chứng**: [`MSSF_plus/audit/audit_A_feature_leakage.py`](audit_A_feature_leakage.py)
- **File kết quả số**: [`MSSF_plus/audit/outputs/audit_A_feature_leakage.json`](outputs/audit_A_feature_leakage.json)

#### Bằng chứng trong code gốc
Tại [`MSSF/mssf.py` dòng 96–98](../../MSSF/mssf.py#L96-L98), ngay sau khi đọc `drug_side.pkl` và trước khi tính 4 ma trận tương đồng `drug_side_sim` (D9), `drug_side_label_sim` (D10), `side_drug_sim` (S3), `side_drug_label_sim` (S4) tại dòng 101–129, code thực hiện xoá toàn bộ các cặp trong `data_test`:
```python
# Remove test set information from the training matrix (MSSF/mssf.py L96-98)
for i in range(data_test.shape[0]):
    drug_side[int(data_test[i, 0]), int(data_test[i, 1])] = 0
```
*(Ghi chú: Trong `mssf.py` gốc viết `drug_side[data_test[i, 0], data_test[i, 1]] = 0`; vì `data_test` được ép sang `np.array` từ danh sách tuple `(int, int, float)` nên có `dtype=float64`, và `notebooks/notebook1_pytorch.ipynb` cell 9 đã patch thêm `int(...)` bằng `sed`).*

#### Kiểm chứng bằng số (Fold 0 và toàn bộ 10 fold của Split A)
Ta so sánh 3 cách tính bộ 4 ma trận `{D9, D10, S3, S4}`:
1. **`full`**: Tính trực tiếp trên toàn bộ `drug_side.pkl` (37,071 cặp dương — giả định bị rò rỉ hoàn toàn).
2. **`mssf`**: Tính theo đúng quy trình `read_raw_data()` của MSSF (xoá `test_pos` khỏi `drug_side.pkl`).
3. **`train-only`**: Khởi tạo ma trận `np.zeros((750, 994))` và chỉ điền các cặp thuộc `train_pos`.

| View dẫn xuất từ `drug_side.pkl` | `mssf` vs. `train-only` `max\|diff\|` (trên cả 10 fold) | `full` vs. `train-only` `max\|diff\|` (Fold 0) | `full` vs. `train-only` `mean\|diff\|` (Fold 0) | Tỷ lệ phần tử khác biệt (`>1e-12`) giữa `full` và `train-only` |
| :--- | :---: | :---: | :---: | :---: |
| **`D9_profile_freq_cos`** (`750×750`) | **`0.00e+00`** | `0.4629` | `0.02985` | `90.50%` |
| **`D10_profile_bin_cos`** (`750×750`) | **`0.00e+00`** | `0.4629` | `0.02827` | `90.50%` |
| **`S3_profile_freq_cos`** (`994×994`) | **`0.00e+00`** | `0.7053` | `0.01893` | `69.80%` |
| **`S4_profile_bin_cos`** (`994×994`) | **`0.00e+00`** | `0.5477` | `0.02170` | `69.80%` |

- Ma trận `masked` của MSSF **trùng khớp tuyệt đối** (`np.array_equal == True`) với ma trận `train-only` trên cả 10/10 fold.
- Ngoài ra, ở Split A, trên cả 10 fold đều có **0 thuốc** và **0 SE** nào bị rơi vào trạng thái không có cặp train nào (`drugs_with_zero_train_pairs = 0`, `ses_with_zero_train_pairs = 0`).
- **Kết luận & cảnh báo khi chuyển sang Bước 2**:
  - Với bộ chia 2 tập `(train, test)` cũ, MSSF và SDPred **không bị rò rỉ test vào 4 view profile**.
  - Tuy nhiên, khi sang Bước 2 (chia `train_pos` thành `train 80% / val_select 10% / val_calib 10%`), nếu vẫn giữ hàm `read_raw_data(rawdata_dir, data_test)` chỉ xoá `data_test` thì **toàn bộ 20% cặp của `val_select` và `val_calib` sẽ bị rò rỉ vào ma trận `drug_side`**! Do đó, trong `MSSF_plus/views.py` và các wrapper baseline, ta bắt buộc phải dựng ma trận `drug_side_train` **từ ma trận toàn 0 và chỉ điền các cặp thuộc tập `train` (80%)**.

---

### Audit B — Kiểm tra các view ở Split B (Drug Cold-start) & độ thưa nội tại
- **Script kiểm chứng**: [`MSSF_plus/audit/audit_B_coldstart_views.py`](audit_B_coldstart_views.py) và [`MSSF_plus/audit/audit_B2_static_view_sparsity.py`](audit_B2_static_view_sparsity.py)
- **File kết quả số**: [`outputs/audit_B_coldstart_views.json`](outputs/audit_B_coldstart_views.json) và [`outputs/audit_B2_static_view_sparsity.json`](outputs/audit_B2_static_view_sparsity.json)

#### B.1. View nào bị zero cho thuốc test, view nào còn nguyên ở Split B?
Ở mỗi fold của Split B, toàn bộ các cặp dương của 75 thuốc test nằm trong `data_test`. Sau vòng lặp xoá `data_test`, cả 75 hàng tương ứng với thuốc test trong `drug_side` đều bằng `0` (`test_rows_in_masked_matrix_all_zero = True` trên cả 10 fold).

Bảng thống kê trên **Fold 0** (75 thuốc test, 675 thuốc train, 994 SE) và tổng quát 750 thuốc:

| View | Thuốc test: tỷ lệ hàng toàn `0` | Thuốc test: tỷ lệ hàng không có thông tin ngoài đường chéo | Trung bình đường chéo (test) | Toàn bộ 750 thuốc: số thuốc có hàng toàn `0` | Toàn bộ 750 thuốc: số thuốc không có thông tin ngoài đường chéo | Kết luận trạng thái ở Split B |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `D1_stitch_one` | `8.0%` | `57.3%` | `0.867` | `62 / 750` | `416 / 750` | Còn nguyên (nhưng vốn thưa) |
| `D2_stitch_two` | `10.7%` | `49.3%` | `0.867` | `61 / 750` | `296 / 750` | Còn nguyên (nhưng vốn thưa) |
| `D3_stitch_three` | `12.0%` | `92.0%` | `0.867` | `93 / 750` | `723 / 750` | Còn nguyên (rất thưa: 96.4% thuốc rỗng) |
| `D4_stitch_four` | `1.3%` | `16.0%` | `0.867` | `13 / 750` | `124 / 750` | Còn nguyên |
| `D5_stitch_five` | `1.3%` | `16.0%` | `0.867` | `12 / 750` | `111 / 750` | Còn nguyên |
| `D6_mol2vec_cos` | **`0.0%`** | **`0.0%`** | `1.000` | `0 / 750` | `0 / 750` | **Còn nguyên 100% (đặc)** |
| `D7_target_cos` | `16.0%` | `21.3%` | `0.840` | `69 / 750` | `134 / 750` | Còn nguyên (134 thuốc không có target) |
| `D8_fingerprint` | **`0.0%`** | `1.3%` | `1.000` | `0 / 750` | `2 / 750` | **Còn nguyên** (trừ 2 thuốc `[617, 628]`) |
| **`D9_profile_freq_cos`** | **`100.0%`** | **`100.0%`** | **`0.000`** | — | — | **BỊ ZERO 100% cho mọi thuốc cold-start** |
| **`D10_profile_bin_cos`** | **`100.0%`** | **`100.0%`** | **`0.000`** | — | — | **BỊ ZERO 100% cho mọi thuốc cold-start** |
| `D11_pathway_enzyme` | `81.3%` | `84.0%` | `0.180` | `567 / 750` | `585 / 750` | Còn nguyên (nhưng 78.0% thuốc vốn rỗng) |
| `S1_adrecs_semantic` | `0.50%` (SE) | `0.50%` (SE) | `0.995` | `5 / 994` SE | `5 / 994` SE | Còn nguyên (5 SE vốn bằng 0) |
| `S2_glove_cos` | **`0.0%`** | **`0.0%`** | `1.000` | `0 / 994` SE | `0 / 994` SE | **Còn nguyên 100%** |
| **`S3_profile_freq_cos`** | **`0.0%`** | **`0.0%`** | `1.000` | `0 / 994` SE | `0 / 994` SE | **Còn nguyên 100% (tính từ 675 thuốc train)** |
| **`S4_profile_bin_cos`** | **`0.0%`** | **`0.0%`** | `1.000` | `0 / 994` SE | `0 / 994` SE | **Còn nguyên 100% (tính từ 675 thuốc train)** |

- **Xác nhận kỳ vọng về SE [3–4]**: Trên cả 10/10 fold của Split B, mọi SE trong 994 SE đều xuất hiện ở ít nhất một thuốc train (`ses_without_any_train_pair = 0` trên cả 10 fold). Do đó, **`S3_profile_freq_cos` và `S4_profile_bin_cos` luôn tính được đầy đủ cho cả 994 SE chỉ từ các thuốc train**.
- **Nhận xét quan trọng cho C1 (Xử lý view thiếu ở mức input)**:
  1. Ở `D9` và `D10`, không chỉ **hàng** `D9[d_test, :] = 0` đối với thuốc cold-start, mà **cột** `D9[d_train, d_test]` trong vector đặc trưng của các thuốc train cũng bằng `0` (`D9_train_rows_cols_testdrugs_all_zero = True`).
  2. Ngay cả ở các view không dẫn xuất từ profile (`D1–D5`, `D7`, `D8`, `D11`), tình trạng khuyết view diễn ra phổ biến ở cả Split A và Split B: trong 9 view tĩnh của thuốc, phân bố số view thực sự có thông tin ngoài đường chéo trên 750 thuốc là `{1 view: 2 thuốc, 2: 24, 3: 59, 4: 57, 5: 118, 6: 206, 7: 201, 8: 76, 9: 7}` (chỉ có 7/750 thuốc có đủ cả 9 view tĩnh!). Đặc biệt, các thuốc khuyết STITCH có trung bình `38.6` SE so với `51.3` SE ở nhóm có STITCH. Do đó, vector nhị phân **Availability Indicator** ở C1 (đánh dấu view $v$ có mặt khi $\sum_{j \neq d} |S^{(v)}_{d,j}| > 0$) có ý nghĩa thực chất ở cả hai kịch bản Warm-start và Cold-start.

#### B.2. Ba vấn đề phát hiện thêm trong Audit B đối với baseline & dữ liệu
1. **Lỗi nghiêm trọng của MGPred ở Split B (`MGPred/Ten_Fold_test.py` dòng 196–216)**:
   - Trong [`fold_files` của MGPred (dòng 196–216)](../../MGPred/Ten_Fold_test.py#L196-L216):
     ```python
     for i in range(n_users):
         ufeature1[i] = [0 for _ in range(n_items)]
         ufeature2[i] = [0 for _ in range(n_users)]
         ufeature3[i] = [0 for _ in range(n_users)]
     ...
     for key in u_adj.keys():
         ufeature2[key] = drug_Tfeature_five[key].tolist()
         ufeature3[key] = drug_feature_fingerprint[key].tolist()
     ```
     Vì `u_adj` chỉ được xây dựng từ `u_train` (dòng 180–187), ở Split B **cả 75/75 thuốc test đều không nằm trong `u_adj.keys()`**. Hệ quả là không chỉ `ufeature1` (interaction profile) bằng 0, mà **`ufeature2` (STITCH) và `ufeature3` (Mol2vec) của toàn bộ 75 thuốc test trong MGPred cũng bị gán toàn số 0**!
2. **Lỗi lấy mẫu âm (negative sampling) của SDPred ở Split B (`notebook1_pytorch.ipynb` cell 12)**:
   - Cell 12 lấy `zero_pairs = np.argwhere(drug_side_sd == 0)` trên toàn bộ 750 thuốc rồi trộn ngẫu nhiên chia cho `train_neg` và `test_neg`.
   - Kiểm chứng số trên 10 fold: ở mỗi fold, có **`3,211–3,482` cặp âm của thuốc test lọt vào `train_neg`** (`~10%` tập âm huấn luyện), và **`2,923–3,758` trên tổng số `3,266–4,165` cặp âm của `test_neg` (`~90%`) thực chất là cặp của thuốc train (warm-start)**!
3. **Hiện tượng trùng dòng khi ánh xạ 750 ↔ 757 thuốc trong [`shared_data/prepare_all_data.py` (dòng 58–70)](../../shared_data/prepare_all_data.py#L58-L70)**:
   - Hàm `find_drug_mapping()` dò dòng trùng giữa `MGPred/data/drug_side.pkl` (`750×994`) và `SDPred/data/drug_side.pkl` (`757×994`) bằng `np.array_equal` và `break` ở vị trí đầu tiên.
   - Có 2 cặp thuốc trong 750 có vector tần suất SE giống hệt nhau: cặp thuốc `(15, 17)` cùng khớp với dòng `[549, 739]` của 757 (cả hai đều bị gán dòng `549`), và cặp thuốc `(26, 458)` cùng khớp với dòng `[65, 544]` (cả hai đều bị gán dòng `65`). Ngoài ra 2 thuốc `[617, 628]` không có dòng khớp nên các ma trận similarity của chúng được điền toàn `0` kèm đường chéo `= 1.0` (riêng `drug_mol.pkl` lấy từ `MGPred/data/Drug_word2vec.pkl`). Vì ta không sửa `shared_data/` theo Nguyên tắc #1, hiện tượng này được giữ cố định cho mọi model để đảm bảo công bằng tuyệt đối, và được C1 xử lý tự nhiên qua availability indicator.

---

### Audit C — Quy tắc chọn Epoch / Checkpoint của các Baseline
- **Script kiểm chứng**: [`MSSF_plus/audit/audit_C_selection.py`](audit_C_selection.py)
- **File kết quả số**: [`MSSF_plus/audit/outputs/audit_C_selection.json`](outputs/audit_C_selection.json)

#### C.1. MSSF — Xác nhận chọn epoch & lưu dự đoán theo `<Best Test>`
- **Bằng chứng code**:
  - Trong [`MSSF/mssf.py` dòng 227–245](../../MSSF/mssf.py#L227-L245), ở mỗi epoch (từ 1 đến `args.epochs`), mô hình chạy `test(model, _test, device)` và nếu `acc_te > acc_tested` thì cập nhật toàn bộ 8 chỉ số theo epoch đó.
  - Trong `notebooks/notebook1_pytorch.ipynb` cell 9, câu lệnh `sed` patch vào `mssf.py`:
    ```bash
    !sed -i '/acc_tested = 0/a\    rating_te_best = None\n    pred_te_best = None' mssf.py
    !sed -i '/acc_tested = acc_te/a\            rating_te_best = rating_te\n            pred_te_best = pred_te' mssf.py
    ```
    Như vậy, **mảng dự đoán lưu trong `results/MSSF_{A|B}/fold_{k}_preds.npy` chính là dự đoán tại epoch đạt Accuracy cao nhất trên tập Test của fold đó**.
- **Định lượng mức độ lạc quan (optimism) từ log 50 epochs × 10 folds trong `notebook1_pytorch.ipynb`**:

##### Bảng C1 — MSSF Split A (10 folds, 50 epochs/fold)
| Fold | Epoch đạt `<Best Test>` Acc | Acc (`Best Test`) | Acc (`Last` - Ep 50) | Acc (`Mean Last 5`) | Macro-F1 (`@Best Acc`) | Macro-F1 (`Last` - Ep 50) | Macro-F1 (`Mean Last 5`) | Macro-F1 (`Oracle Max`) | Độ lệch chuẩn Acc 10 epoch cuối |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 48 | 0.6990 | 0.6788 | 0.6837 | 0.6470 | 0.6360 | 0.6417 | 0.6504 | 0.0199 |
| 2 | 48 | 0.7111 | 0.6747 | 0.6701 | 0.6532 | 0.6382 | 0.6260 | 0.6532 | 0.0368 |
| 3 | 37 | 0.6998 | 0.6698 | 0.6612 | 0.6454 | 0.6113 | 0.6242 | 0.6454 | 0.0247 |
| 4 | 48 | 0.6906 | 0.6504 | 0.6704 | 0.6410 | 0.6152 | 0.6244 | 0.6410 | 0.0288 |
| 5 | 44 | 0.6895 | 0.6550 | 0.6486 | 0.6400 | 0.5451 | 0.5927 | 0.6400 | 0.0368 |
| 6 | 45 | 0.7014 | 0.6984 | 0.6086 | 0.6650 | 0.6497 | 0.5686 | 0.6650 | 0.0650 |
| 7 | 49 | 0.6909 | 0.6817 | 0.6651 | 0.6154 | 0.6342 | 0.6109 | 0.6419 | 0.0296 |
| 8 | 34 | 0.7084 | 0.6925 | 0.6442 | 0.6359 | 0.6529 | 0.6106 | 0.6529 | 0.0541 |
| 9 | 32 | 0.6922 | 0.6731 | 0.6672 | 0.6383 | 0.6310 | 0.5982 | 0.6395 | 0.0270 |
| 10 | 39 | 0.6935 | 0.6437 | 0.6660 | 0.6528 | 0.6262 | 0.6321 | 0.6528 | 0.0195 |
| **TB** | **42.4** | **0.6976** | **0.6718** | **0.6585** | **0.6434** | **0.6240** | **0.6129** | **0.6482** | **0.0342** |

  - **Độ phóng đại do chọn trên Test ở Split A**:
    - `Acc`: `+0.0258` (`+2.58%`) so với epoch cuối; `+0.0391` (`+3.91%`) so với trung bình 5 epoch cuối.
    - `Macro-F1`: `+0.0194` (`+1.94%`) so với epoch cuối; `+0.0305` (`+3.05%`) so với trung bình 5 epoch cuối.

##### Bảng C2 — MSSF Split B (10 folds, 50 epochs/fold)
| Fold | Epoch đạt `<Best Test>` Acc | Acc (`Best Test`) | Acc (`Last` - Ep 50) | Acc (`Mean Last 5`) | Macro-F1 (`@Best Acc`) | Macro-F1 (`Last` - Ep 50) | Macro-F1 (`Mean Last 5`) | Macro-F1 (`Oracle Max`) | Độ lệch chuẩn Acc 10 epoch cuối |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **3** | 0.5062 | 0.4875 | 0.4428 | 0.3325 | 0.2933 | 0.2767 | 0.3457 | 0.0540 |
| 2 | 20 | 0.5361 | 0.5088 | 0.4672 | 0.2722 | 0.2895 | 0.2457 | 0.3501 | 0.0501 |
| 3 | 46 | 0.5283 | 0.5006 | 0.4737 | 0.2331 | 0.2843 | 0.2724 | 0.3535 | 0.0323 |
| 4 | 12 | 0.5412 | 0.4609 | 0.4920 | 0.2891 | 0.3487 | 0.3014 | 0.3949 | 0.0271 |
| 5 | 34 | 0.5345 | 0.5030 | 0.5013 | 0.3449 | 0.3412 | 0.3350 | 0.3849 | 0.0175 |
| 6 | **9** | 0.4915 | 0.4216 | 0.4411 | 0.3191 | 0.2841 | 0.2641 | 0.3386 | 0.0218 |
| 7 | 19 | 0.5327 | 0.4954 | 0.4833 | 0.3227 | 0.3166 | 0.2956 | 0.3556 | 0.0383 |
| 8 | **3** | 0.5498 | 0.3540 | 0.4421 | **0.1844** | 0.2845 | 0.2852 | 0.3488 | 0.0574 |
| 9 | 22 | 0.5024 | 0.4761 | 0.4502 | 0.2791 | 0.1701 | 0.2182 | 0.3696 | 0.0524 |
| 10 | **10** | 0.5562 | 0.5210 | 0.5261 | 0.3244 | 0.2795 | 0.3314 | 0.4049 | 0.0237 |
| **TB** | **17.8** | **0.5279** | **0.4729** | **0.4720** | **0.2902** | **0.2892** | **0.2826** | **0.3647** | **0.0375** |

  - **Nhận xét quan trọng ở Split B**:
    1. Chọn epoch theo `Best Test Acc` phóng đại Accuracy thêm **`+5.50%`** so với epoch cuối (`0.5279` vs `0.4729`).
    2. Vì lớp đa số (Class 3) chiếm `~48%` mẫu, ở những epoch rất sớm (như Fold 1 và Fold 8 tại **epoch 3**), mô hình dự đoán hầu hết thành Class 3 nên đạt đỉnh `Acc = 0.5498` nhưng `Macro-F1` sụp xuống chỉ còn **`0.1844`** (trong khi `Oracle Max Macro-F1` trung bình đạt `0.3647`). Điều này chứng minh việc chọn checkpoint bằng **Macro-F1 trên `val_select`** ở Bước 2 là hoàn toàn đúng đắn.

#### C.2. MGPred — Xác nhận chọn checkpoint & dừng sớm trên Test set
- Trong [`MGPred/Ten_Fold_test.py` dòng 340–364](../../MGPred/Ten_Fold_test.py#L340-L364): mỗi epoch gọi `test(model, _test, _test, device)`, khi `rmse_mn > rmse` (tức **test RMSE giảm**) thì ghi đè `final_p{fold}.p` và đặt lại `endure_count = 0`; dừng sớm khi `endure_count > 30`.
- Trong `notebooks/notebook1_pytorch.ipynb` cell 5: patch thành `epochs = 15`, đánh giá `_test` mỗi 3 epoch (`if epoch % 3 == 0 or epoch >= args.epochs - 1:`), lưu `final_p{fold}.p` khi test RMSE thấp nhất và dừng sớm khi `endure_count > 2` (tức 3 lần kiểm tra test liên tiếp không giảm RMSE).

#### C.3. SDPred — Không chọn trên Test nhưng bất nhất cấu hình giữa Split A và Split B
- Trong [`SDPred/up_ten_fold.py` dòng 165–190](../../SDPred/up_ten_fold.py#L165-L190): mỗi epoch đánh giá trên `_train` (`test(model, _train, _train, device)`), đặt lại `endure_count = 0` khi **đồng thời** `AUC_mn < t_i_auc and AUPR_mn < t_iPR_auc` trên tập train, và dừng sớm khi `endure_count > 10`. Tập `_test` chỉ được đánh giá 1 lần ở dòng 190 sau khi thoát vòng lặp (lấy trọng số của epoch cuối cùng, không lưu checkpoint tốt nhất).
- Trong `notebooks/notebook1_pytorch.ipynb`:
  - **Cell 7 (Split A)**: chạy `sed -i '170,189s/^/# /' up_ten_fold.py` để comment toàn bộ khối đánh giá `_train` và early stopping → luôn chạy đủ `100` epochs với `batch_size = 512`, đánh giá test ở epoch 100.
  - **Cell 12 (Split B)**: viết lại khối `sed` nhưng **không comment dòng 170–189**, đổi `endure_count > 10` thành `endure_count > 5`, và đổi `batch_size = 2048` (gấp 4 lần Split A).

#### C.4. Lỗi xáo trộn thứ tự mẫu Test (`shuffle=True`) ở cả 3 baseline
- Trong [`MSSF/mssf.py` dòng 199](../../MSSF/mssf.py#L199), [`MGPred/Ten_Fold_test.py` dòng 289](../../MGPred/Ten_Fold_test.py#L289), và [`SDPred/up_ten_fold.py` dòng 141](../../SDPred/up_ten_fold.py#L141), `_test = DataLoader(testset, ..., shuffle=True)`.
- Dù mỗi lần gọi `test()` đều thu thập cả `labels` và `preds` theo cùng thứ tự batch đã xáo trộn (nên metric tính gộp trên 1 fold không đổi), nhưng thứ tự phần tử trong `fold_{k}_labels.npy` / `fold_{k}_preds.npy` bị xáo trộn ngẫu nhiên theo từng epoch và không khớp với thứ tự dòng của `load_splits(fold, split)`. Trong `MSSF_plus/`, mọi DataLoader đánh giá (`val_select`, `val_calib`, `test`) bắt buộc đặt **`shuffle=False`**.

---

### Audit D — Các file feature SE dạng Ontology & ánh xạ về 994 SE của MSSF
- **Script kiểm chứng**: [`MSSF_plus/audit/audit_D_ontology.py`](audit_D_ontology.py)
- **File kết quả số**: [`MSSF_plus/audit/outputs/audit_D_ontology.json`](outputs/audit_D_ontology.json)

#### D.1. Tổng hợp các file ontology có trong repo

| File | Kích thước | Kiểu / Giá trị | Ý nghĩa & Nguồn | Khớp thứ tự 994 SE của MSSF? |
| :--- | :---: | :---: | :--- | :---: |
| [`shared_data/side_effect_label_750.mat`](../../shared_data/side_effect_label_750.mat) (khóa `'node_label'`) | **`(994, 243)`** | `float64`, nhị phân `{0.0, 1.0}` | Vector phân cấp MedDRA 3 mức (SOC / HLGT / HLT) của DSGAT & A-3Net; kèm khóa `'side_effect'` `(994, 1)` lưu tên tiếng Anh của 994 SE | **KHỚP 100% (1–1)** |
| [`shared_data/effect_side_semantic.pkl`](../../shared_data/effect_side_semantic.pkl) (`= MSSF/Datas/side_effect_semantic.pkl`) | **`(994, 994)`** | `float64`, `[0.0, 1.0]` | Ma trận tương đồng ngữ nghĩa DAG trên ADReCS (chính là view `S1` hiện tại của MSSF) | **KHỚP 100% (1–1)** |
| [`MVSL-DSF/side_vector_level_123.pkl`](../../MVSL-DSF/side_vector_level_123.pkl) | `(893, 243)` | `float64`, nhị phân `{0.0, 1.0}` | Tập con 893 vector MedDRA 243-D dùng trong MVSL-DSF gốc (cả 893 hàng đều nằm trong `side_effect_label_750.mat`) | Chỉ có 893 SE |
| [`shared_data/mvsl_750_994/side_vector_level_123.pkl`](../../shared_data/mvsl_750_994/side_vector_level_123.pkl) | `(994, 243)` | `float64`, `{0.0, 1.0}` + trung bình | Bản mở rộng 893 → 994 do `prepare_all_data.py` sinh ra; **bị lỗi va chạm ánh xạ cột nên 110 SE bị điền bằng vector trung bình** | **KHÔNG DÙNG** (đã có bản gốc đầy đủ ở `.mat`) |

#### D.2. Bằng chứng kiểm chứng sự khớp thứ tự 994 SE và chất lượng `side_effect_label_750.mat`
1. **Kiểm chứng thứ tự cột 994 SE**:
   - Ma trận `'R'` trong `shared_data/raw_frequency_750.mat` (đi kèm `side_effect_label_750.mat` của DSGAT/A-3Net) **trùng khớp tuyệt đối từng phần tử** (`np.array_equal == True`) với `shared_data/mssf_750/drug_side.pkl` (`750 × 994`).
   - So sánh 994 cột của `MSSF/Datas/drug_side.pkl` (`757 × 994`) và `shared_data/mssf_750/drug_side.pkl` (`750 × 994`) trên toàn bộ các thuốc khớp: **`columns_994_aligned = True`**.
   - Do đó, dòng thứ `s` (`0..993`) của `side_effect_label_750.mat['node_label']` chính là SE thứ `s` của MSSF, không cần bất kỳ phép hoán vị nào.
2. **Thống kê trên `side_effect_label_750.mat['node_label']` (`994 × 243`)**:
   - **Số dòng SE toàn 0**: **`0 / 994`** (mọi SE đều có từ `2` đến `8` bit bằng `1`, trung bình `2.93` bit/SE, trung vị `2.0`).
   - **Số cột đặc trưng toàn 0**: **`0 / 243`**.
   - Khi tính `S5_meddra_ontology = cosine_similarity(node_label)` (`994 × 994`) cho thành phần **C2**:
     - Đường chéo chính: toàn bộ bằng `1.0`.
     - Các phần tử ngoài đường chéo: thuộc `[0.0, 1.0]`, trung bình `0.0476 ± 0.1467`, có `11.57%` cặp SE có độ tương đồng `> 0`.
     - Tương quan Pearson với `S1_adrecs_semantic` (trên tam giác trên): `r = 0.8725`. Đặc biệt, trong `S1_adrecs_semantic` có **5 SE bị toàn số 0** (`[154, 349, 495, 533, 727]`), trong khi ở `S5_meddra_ontology` cả 5 SE này đều có vector MedDRA đầy đủ!
3. **Phát hiện lỗi trong `shared_data/mvsl_750_994/side_vector_level_123.pkl`**:
   - Trong [`prepare_all_data.py` dòng 197–204](../../shared_data/prepare_all_data.py#L197-L204), script tìm vị trí của 893 SE trong 994 SE bằng cách so sánh cột tần suất `np.array_equal(col_893, mg750[:, j994])` và `break` ở cột `j994` đầu tiên khớp.
   - Tuy nhiên, có **18 cột SE** trong 893 cột có vector tần suất trên 750 thuốc trùng nhau từng cặp/bộ ba (ví dụ 3 cột `[50, 51, 78]` giống hệt nhau nên đều bị gán vào `j994 = 50`). Hệ quả là `col_mapping_893_to_994.npy` có độ dài 893 nhưng chỉ chứa **884 chỉ số duy nhất**, làm **110 SE** (chứ không phải 101 SE) trong `mvsl_750_994/side_vector_level_123.pkl` bị điền bằng vector trung bình `mean_vec`.
   - Vì `shared_data/side_effect_label_750.mat['node_label']` đã chứa sẵn 100% vector MedDRA thật cho toàn bộ 994 SE, trong `MSSF_plus/views.py` ta chỉ sử dụng `side_effect_label_750.mat['node_label']`.

---

### Audit E — Phân bố 5 lớp trên toàn bộ dữ liệu và trong từng fold
- **Script kiểm chứng**: [`MSSF_plus/audit/audit_E_class_distribution.py`](audit_E_class_distribution.py)
- **File kết quả số**: [`MSSF_plus/audit/outputs/audit_E_class_distribution.json`](outputs/audit_E_class_distribution.json)

#### E.1. Phân bố toàn cục trên ma trận `750 × 994` (`745,500` cặp)
- **Cặp chưa quan sát (`frequency = 0`)**: `708,429` (`95.027%`).
- **Cặp dương đã biết (`frequency ∈ {1, 2, 3, 4, 5}` ↔ `class ∈ {0, 1, 2, 3, 4}`)**: **`37,071`** (`4.973%`).

| Lớp (0-based) | Tần suất SIDER | Nhãn ngữ nghĩa | Số lượng cặp | Tỷ lệ (%) | Trọng số nghịch tần suất chuẩn hoá $w_k = \frac{N}{5 N_k}$ |
| :---: | :---: | :--- | :---: | :---: | :---: |
| **Class 0** | `1` | Very rare (Rất hiếm) | `1,190` | **`3.210%`** | **`6.2304`** |
| **Class 1** | `2` | Rare (Hiếm) | `4,186` | **`11.292%`** | **`1.7712`** |
| **Class 2** | `3` | Infrequent (Không thường xuyên) | `9,978` | **`26.916%`** | **`0.7431`** |
| **Class 3** | `4` | Frequent (Thường xuyên) | `17,594` | **`47.460%`** | **`0.4214`** |
| **Class 4** | `5` | Very frequent (Rất thường xuyên) | `4,123` | **`11.122%`** | **`1.7983`** |
| **Tổng / Độ lệch** | — | — | **`37,071`** | **`100.000%`** | **Tỷ lệ $\max / \min = 14.785\times$** (Class 3 / Class 0) |

#### E.2. Phân bố 5 lớp theo từng fold ở Split A (Warm-start, StratifiedKFold theo cặp)

| Fold | Tổng Train | Train `[c0, c1, c2, c3, c4]` | Tổng Test | Test `[c0, c1, c2, c3, c4]` | Test `% [c0, c1, c2, c3, c4]` |
| :---: | :---: | :--- | :---: | :--- | :--- |
| **0** | 33,363 | `[1071, 3767, 8980, 15834, 3711]` | 3,708 | `[119, 419, 998, 1760, 412]` | `[3.21, 11.30, 26.92, 47.47, 11.11]` |
| **1** | 33,364 | `[1071, 3768, 8980, 15834, 3711]` | 3,707 | `[119, 418, 998, 1760, 412]` | `[3.21, 11.28, 26.92, 47.48, 11.11]` |
| **2** | 33,364 | `[1071, 3768, 8980, 15834, 3711]` | 3,707 | `[119, 418, 998, 1760, 412]` | `[3.21, 11.28, 26.92, 47.48, 11.11]` |
| **3** | 33,364 | `[1071, 3768, 8980, 15834, 3711]` | 3,707 | `[119, 418, 998, 1760, 412]` | `[3.21, 11.28, 26.92, 47.48, 11.11]` |
| **4** | 33,364 | `[1071, 3768, 8980, 15835, 3710]` | 3,707 | `[119, 418, 998, 1759, 413]` | `[3.21, 11.28, 26.92, 47.45, 11.14]` |
| **5** | 33,364 | `[1071, 3767, 8981, 15835, 3710]` | 3,707 | `[119, 419, 997, 1759, 413]` | `[3.21, 11.30, 26.90, 47.45, 11.14]` |
| **6** | 33,364 | `[1071, 3767, 8981, 15835, 3710]` | 3,707 | `[119, 419, 997, 1759, 413]` | `[3.21, 11.30, 26.90, 47.45, 11.14]` |
| **7** | 33,364 | `[1071, 3767, 8980, 15835, 3711]` | 3,707 | `[119, 419, 998, 1759, 412]` | `[3.21, 11.30, 26.92, 47.45, 11.11]` |
| **8** | 33,364 | `[1071, 3767, 8980, 15835, 3711]` | 3,707 | `[119, 419, 998, 1759, 412]` | `[3.21, 11.30, 26.92, 47.45, 11.11]` |
| **9** | 33,364 | `[1071, 3767, 8980, 15835, 3711]` | 3,707 | `[119, 419, 998, 1759, 412]` | `[3.21, 11.30, 26.92, 47.45, 11.11]` |
| **TB ± SD** | — | — | — | — | **`[3.21±0.00, 11.29±0.01, 26.92±0.01, 47.46±0.01, 11.12±0.01]`** |

#### E.3. Phân bố 5 lớp theo từng fold ở Split B (Drug cold-start, 675 thuốc train / 75 thuốc test)

| Fold | Tổng Train | Train `[c0, c1, c2, c3, c4]` | Tổng Test | Test `[c0, c1, c2, c3, c4]` | Test `% [c0, c1, c2, c3, c4]` |
| :---: | :---: | :--- | :---: | :--- | :--- |
| **0** | 33,602 | `[1092, 3944, 8949, 15911, 3706]` | 3,469 | `[98, 242, 1029, 1683, 417]` | `[2.83, 6.98, 29.66, 48.52, 12.02]` |
| **1** | 33,221 | `[1070, 3842, 8896, 15638, 3775]` | 3,850 | `[120, 344, 1082, 1956, 348]` | `[3.12, 8.94, 28.10, 50.81, 9.04]` |
| **2** | 33,569 | `[1088, 3901, 9059, 15807, 3714]` | 3,502 | `[102, 285, 919, 1787, 409]` | `[2.91, 8.14, 26.24, 51.03, 11.68]` |
| **3** | 33,361 | `[1052, 3796, 9011, 15810, 3692]` | 3,710 | `[138, 390, 967, 1784, 431]` | `[3.72, 10.51, 26.07, 48.09, 11.62]` |
| **4** | 33,290 | `[1051, 3588, 9012, 15882, 3757]` | 3,781 | `[139, 598, 966, 1712, 366]` | `[3.68, 15.82, 25.55, 45.28, 9.68]` |
| **5** | 32,906 | `[1013, 3501, 8856, 15880, 3656]` | 4,165 | `[177, 685, 1122, 1714, 467]` | `[4.25, 16.45, 26.94, 41.15, 11.21]` |
| **6** | 33,046 | `[1069, 3951, 8753, 15687, 3586]` | 4,025 | `[121, 235, 1225, 1907, 537]` | `[3.01, 5.84, 30.44, 47.38, 13.34]` |
| **7** | 33,797 | `[1084, 3936, 9174, 15826, 3777]` | 3,274 | `[106, 250, 804, 1768, 346]` | `[3.24, 7.64, 24.56, 54.00, 10.57]` |
| **8** | 33,805 | `[1135, 3674, 9174, 16073, 3749]` | 3,266 | `[55, 512, 804, 1521, 374]` | `[1.68, 15.68, 24.62, 46.57, 11.45]` |
| **9** | 33,042 | `[1056, 3541, 8918, 15832, 3695]` | 4,029 | `[134, 645, 1060, 1762, 428]` | `[3.33, 16.01, 26.31, 43.73, 10.62]` |
| **TB ± SD** | — | — | — | — | **`[3.18±0.65, 11.20±4.08, 26.85±1.89, 47.66±3.56, 11.12±1.16]`** |

---

## 3. Đề xuất Cách sửa & Kế hoạch Chạy lại Baseline

| Mã | Vấn đề phát hiện | Giải pháp kỹ thuật trong `MSSF_plus/` | Baseline phải chạy lại (qua `MSSF_plus/baselines_rerun/`) |
| :-: | :--- | :--- | :--- |
| **P1** | **Chọn epoch theo Test set** ở MSSF (`<Best Test>` theo `acc_te`) và MGPred (`rmse_mn` + early stopping trên `_test`). | Tạo bộ chia thống nhất `train (80%) / val_select (10%) / val_calib (10%)` trong `MSSF_plus/protocol.py`. Chọn epoch tốt nhất chỉ theo **`val_select`** (`Macro-F1` cho MSSF/MSSF_plus; `RMSE` trên `val_select` cho MGPred/SDPred). Đánh giá `test` đúng 1 lần duy nhất từ checkpoint tốt nhất trên `val_select`. | **MSSF (Split A & B)**, **MGPred (Split A & B)**, **SDPred (Split A & B)** |
| **P2** | **Nguy cơ rò rỉ `val_select` và `val_calib` vào 4 view dẫn xuất từ `drug_side.pkl`** nếu chỉ xoá `test` khỏi ma trận đầy đủ. | Trong `MSSF_plus/views.py` và các wrapper ở `baselines_rerun/`, dựng ma trận `drug_side_train` (`750 × 994`) bắt đầu từ `np.zeros((750, 994))` và **chỉ điền các cặp thuộc tập `train` (80%)**. Như vậy ở Split B, cả thuốc `val_select`, `val_calib` và `test` đều có dòng profile bằng `0` (đúng điều kiện drug cold-start). | **MSSF (Split A & B)**, **SDPred (Split A & B)**, **MGPred (Split A & B)** |
| **P3** | **DataLoader `_test` đặt `shuffle=True`** ở MSSF, SDPred, MGPred làm xáo trộn thứ tự mẫu trong `fold_{k}_preds.npy`. | Đặt `shuffle=False` cho mọi DataLoader của `val_select`, `val_calib`, và `test`; thêm `assert np.array_equal(saved_labels, test_data[:, 2])`. | Khắc phục đồng thời khi chạy lại **MSSF, SDPred, MGPred** |
| **P4** | **MGPred ở Split B gán vector 0 cho cả STITCH và Mol2vec của toàn bộ 75 thuốc test** (do chỉ điền `ufeature2, ufeature3` cho `key in u_adj.keys()`). | Trong wrapper `MSSF_plus/baselines_rerun/run_mgpred.py`, điền `ufeature2` (STITCH) và `ufeature3` (Mol2vec) cho toàn bộ `range(n_users)` (750 thuốc), chỉ giữ `ufeature1` (interaction profile) theo `u_adj` của tập `train` 80%. | **MGPred Split B** |
| **P5** | **SDPred bất nhất giữa Split A & B** (`batch_size=512` không early-stop vs. `batch_size=2048` + train early-stop) và **lấy mẫu âm ở Split B trộn lẫn thuốc train/test**. | Trong wrapper `MSSF_plus/baselines_rerun/run_sdpred.py`: đồng nhất siêu tham số, chọn checkpoint theo `val_select`, và ở Split B chỉ lấy mẫu âm cho `train`/`val_select`/`val_calib`/`test` từ các thuốc thuộc đúng phân vùng đó. | **SDPred (Split A & B)** |
| **P6** | **Ontology SE cho C2**: `mvsl_750_994/side_vector_level_123.pkl` bị va chạm ánh xạ cột nên 110 SE bị điền trung bình. | Dùng trực tiếp `shared_data/side_effect_label_750.mat['node_label']` (`994 × 243`) — đầy đủ 100% vector MedDRA thật và khớp 1–1 với 994 SE của MSSF. | Áp dụng trực tiếp trong `MSSF_plus/views.py` (C2) |

### Ước lượng thời gian chạy trên Colab GPU
- **Tối ưu tốc độ hợp lệ (không thay đổi toán học mô hình)**:
  - Trong `MSSF/mssf.py` gốc (dòng 232), mỗi epoch đều gọi `test(model, _train, device)` trên toàn bộ `~33,364` cặp train chỉ để in log, không dùng `with torch.no_grad():`, và gọi `loss.backward(retain_graph=True)`.
  - Khi bỏ đánh giá trên `_train` mỗi epoch (chỉ đánh giá trên `val_select` với `torch.no_grad()`, kích thước bằng $1/8$ tập train) và bỏ `retain_graph=True`, thời gian chạy MSSF giảm từ `~40 phút/fold` xuống còn **`~15–20 phút/fold`** (`~2.5–3.3 giờ / 10 folds`, hoặc `~45–60 phút` cho 3 folds phát triển).
- **MGPred (10 folds)**: `~15–20 phút` / split.
- **SDPred (10 folds)**: `~35–45 phút` / split (khi bỏ đánh giá trên `_train` mỗi epoch và chỉ đánh giá trên `val_select`).
