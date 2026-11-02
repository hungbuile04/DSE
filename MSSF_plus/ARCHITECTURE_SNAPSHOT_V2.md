# TÀI LIỆU ĐÔNG KẾT KIẾN TRÚC & CHUẨN THỰC NGHIỆM
# Phiên bản: MSSF_plus (CONCORD-DSE v2.0 Snapshot)
> **Mục đích tài liệu**: Đóng băng (*freeze*) toàn bộ thông số kỹ thuật, cấu trúc tensor, công thức toán học, giao thức đánh giá không rò rỉ (Protocol v2) và các kết quả thực nghiệm mốc cơ sở của phiên bản hiện tại (`MSSF_plus`). Tài liệu này được thiết kế để làm **hệ quy chiếu chuẩn (ground truth baseline)** phục vụ việc đối chiếu, so sánh và đánh giá khi tiến hành **cải tiến kiến trúc lớn** ở giai đoạn tiếp theo.
>
> **Thời điểm chốt mốc**: 31/10/2026 — Hoàn tất Bước 2 (Khử rò rỉ, $K=5$ Cross-Fitting, Đơn giản hóa Loss, MedDRA Ontology, Conformal theo thuốc, Tối ưu hóa đại số).

---

## 1. Thông tin Phiên bản & Toàn vẹn Dữ liệu

- **Tên kỹ thuật trong mã nguồn**: `MSSF_plus/`
- **Tên đề xuất cho bài báo**: **CONCORD-DSE** (*Cold-start Multi-view Ordinal Network with Conformal Prediction for Drug Side-Effect Frequency*)
- **Trạng thái kiểm toán mã băm SHA256 (`verify_checksums.py`)**:
  - `90/90` tệp dữ liệu gốc và split trong `shared_data/`: **Bit-for-bit unchanged (PASS)**.
  - Tuyệt đối không chạm vào các thư mục baseline cũ (`HSTrans/HSTrans_original`, `baseline/`, `notebooks/notebook1_pytorch.ipynb`).
- **Nguyên tắc bất biến của hệ thống**:
  1. Tập `test` (10%) chỉ được nạp để đánh giá **duy nhất một lần** sau khi đã chốt checkpoint tốt nhất; không bao giờ được dùng để chọn siêu tham số hay rẽ nhánh thuật toán.
  2. Mọi lựa chọn thành phần, siêu tham số và hàm loss đều dựa trên **`val_select`** (10%).
  3. Mọi ma trận tương tác hồ sơ (profile) và nhãn mềm phân cụm chỉ được xây dựng từ tập **`train` (80%)** của từng fold.

---

## 2. Định nghĩa Bài toán & Giao thức Đánh giá (Protocol v2)

### 2.1. Phân loại thứ tự tần suất tác dụng phụ
- Cho $N_d = 750$ thuốc và $N_s = 994$ tác dụng phụ (Side Effects — SE) với $37{,}071$ cặp tương tác dương trong SIDER.
- Mỗi cặp $(d, s)$ mang một nhãn tần suất thứ tự 5 mức:
  $$y_{d,s} \in \{0, 1, 2, 3, 4\} \quad (\text{tương ứng nhãn gốc SIDER } y^{\text{freq}} \in \{1, 2, 3, 4, 5\})$$
  - Lớp $c_0$ (*Very Rare*, $<0.01\%$): $3.06\%$
  - Lớp $c_1$ (*Rare*, $0.01\% - 0.1\%$): $11.49\%$
  - Lớp $c_2$ (*Infrequent*, $0.1\% - 1.0\%$): $26.85\%$
  - Lớp $c_3$ (*Frequent*, $1.0\% - 10.0\%$): $47.41\%$ (lớp chiếm đa số)
  - Lớp $c_4$ (*Very Frequent*, $\ge 10.0\%$): $11.19\%$

### 2.2. Hai kịch bản đánh giá 10-Fold CV
- **Split A (Warm-Start / Pair-level CV)**: Mỗi fold chia $37{,}071$ cặp tương tác dương thành 4 tập rời nhau: `train` ($26{,}690$ cặp), `val_select` ($3{,}337$ cặp), `val_calib` ($3{,}337$ cặp), và `test` ($3{,}707$ cặp) $\approx 72\% / 9\% / 9\% / 10\%$ tổng số cặp (trong đó tỷ lệ $80/10/10$ là tỷ lệ chia bên trong $90\%$ phần không phải `test`). Đánh giá khả năng dự đoán tác dụng phụ mới trên các thuốc đã biết một phần hồ sơ tương tác.
- **Split B (Cold-Start Drug / Drug-level CV — Trọng tâm chính)**: Chia $750$ thuốc thành 10 phần rời nhau. Mỗi fold có $75$ thuốc `test` hoàn toàn mới (toàn bộ hàng tương tác bằng $0$ lúc train).
  - Phân chia 4 tập mỗi fold: `train` ($540$ thuốc), `val_select` ($67$ thuốc), `val_calib` ($68$ thuốc), và `test` ($75$ thuốc) $\approx 72\% / 9\% / 9\% / 10\%$ tổng số thuốc (tỷ lệ $80/10/10$ là tỷ lệ chia bên trong $675$ thuốc không thuộc tập `test`).

---

## 3. Bản đồ Khung nhìn Đa nguồn (Multi-View System)

Mô hình vận hành trên hệ thống gồm **11 Drug Views** và **5 SE Views**:

| Phân loại | Tên view | Kích thước | Nguồn gốc / Phương pháp tính | Đặc tính rò rỉ |
| :--- | :--- | :---: | :--- | :--- |
| **Drug (Tĩnh)** | `D1_stitch_one` .. `D5_stitch_five` | $5 \times (750 \times 750)$ | 5 kênh tương tác hóa sinh STITCH | Tĩnh (Cố định) |
| | `D6_mol2vec_cos` | $750 \times 750$ | Cosine similarity trên Mol2vec (100-D) | Tĩnh |
| | `D7_target_cos` | $750 \times 750$ | Cosine similarity trên profile Protein Target | Tĩnh (Có hàng rỗng $=0$) |
| | `D8_fingerprint` | $750 \times 750$ | Morgan Fingerprint Jaccard similarity | Tĩnh |
| | `D11_pathway_enzyme` | $750 \times 750$ | Pathway & Enzyme similarity | Tĩnh (Có hàng rỗng $=0$) |
| **Drug (Profile)**| `D9_profile_freq_cos` | $750 \times 750$ | Cosine trên hàng ma trận tần suất `drug_side_train` | **Chỉ tính từ Train 80%** |
| | `D10_profile_bin_cos` | $750 \times 750$ | Cosine trên hàng ma trận nhị phân `(drug_side_train > 0)` | **Chỉ tính từ Train 80%** |
| **SE (Tĩnh)** | `S1_adrecs_semantic` | $994 \times 994$ | ADReCS DAG semantic similarity | Tĩnh |
| | `S2_glove_cos` | $994 \times 994$ | GloVe word embedding cosine similarity (300-D) | Tĩnh |
| | `S5_meddra_ontology` (**C2**) | $994 \times 994$ | Cosine trên ma trận MedDRA hierarchy (994 × 243) | **Tĩnh (Mới, phủ đủ 100%)** |
| **SE (Profile)** | `S3_profile_freq_cos` | $994 \times 994$ | Cosine trên cột `drug_side_train.T` | **Chỉ tính từ Train 80%** |
| | `S4_profile_bin_cos` | $994 \times 994$ | Cosine trên cột `(drug_side_train > 0).T` | **Chỉ tính từ Train 80%** |

- Kích thước đầu vào biểu diễn phẳng:
  - Thuốc ($V_d = 11$): $x_d \in \mathbb{R}^{11 \times 750} = \mathbb{R}^{8250}$, vector hiện diện $a_d \in \{0, 1\}^{11}$.
  - Tác dụng phụ ($V_s = 5$): $x_s \in \mathbb{R}^{5 \times 994} = \mathbb{R}^{4970}$, vector hiện diện $a_s \in \{0, 1\}^{5}$.

---

## 4. Chi tiết Toán học 5 Trụ cột Cải tiến (C1–C5)

### 4.1. C1 — Xử lý Thiếu Khung nhìn & Bù khuyết Cold-Start (Đóng góp chính)
1. **Chỉ báo hiện diện trước bù khuyết ($a_{i,v}$)**:
   $$a_{i,v} = \mathbb{I}\!\left(\sum_{j \neq i} \bigl|S^{(v)}_{i,j}\bigr| > 10^{-12}\right) \in \{0, 1\}$$
   Tự động nhận diện các thuốc thiếu profile ở Split B ($a_{d,\text{D9}} = a_{d,\text{D10}} = 0$) cũng như các thuốc thiếu target tĩnh ở `D7, D11`.
2. **Ba chiến lược bù khuyết view tương tác (`c1_impute`) & Cơ chế Cross-Fitting $K=5$ theo thuốc (`MSSF_plus/views.py`)**:
   - **`mean` (Vector trung bình các thuốc train)**: Tính trung bình cộng các hàng của `train_drugs` trên `D9` và `D10` (đặt đường chéo $= 1.0$) để điền cho các thuốc thiếu hồ sơ ($a_{d,\text{D9}} = 0$).
   - **`regress` (`ProfileViewRegressor` trong `MSSF_plus/views.py`, Cross-Fitting $K=5$ — Đề xuất)**:
     - Chia tập `train_drugs` thành $K=5$ nhóm thuốc độc lập (`val_frac = 0.10` tách nội bộ từ 4 nhóm train để early stopping theo MSE).
     - Kiến trúc thực tế từ [`ARCHITECTURE_DUMP.md`](file:///Users/buihung/project_DL/DSE/MSSF_plus/ARCHITECTURE_DUMP.md) (`2,182,876` tham số / mô hình):
       `Linear(6759, 256) -> LayerNorm((256,)) -> SiLU() -> Dropout(p=0.2) -> Linear(256, 256) -> LayerNorm((256,)) -> SiLU() -> Dropout(p=0.2) -> Linear(256, 1500) -> Sigmoid()`.
     - Huấn luyện (`train_and_impute_regress`): `epochs = 80`, `batch_size = 64`, `AdamW(lr=1e-3, weight_decay=1e-4)`, hàm mất mát:
       $$\mathcal{L}_{\text{imp}} = \text{MSE}(\hat{y}, y) + 0.1\,\bigl(1 - \cos(\hat{y}, y)\bigr)$$
     - Dự đoán **out-of-fold** cho nhóm bị giữ lại làm `fallback_d9, fallback_d10`; điền cho thuốc `val_select`, `val_calib`, `test` bằng **trung bình cộng (ensemble) của 5 mạng `ProfileViewRegressor`** (đặt đường chéo $= 1.0$).
   - **`diffusion` (`ConditionalDenoiserMLP` trong `MSSF_plus/views.py`, Cross-Fitting $K=5$ — Đối chứng ablation)**:
     - Kiến trúc thực tế từ [`ARCHITECTURE_DUMP.md`](file:///Users/buihung/project_DL/DSE/MSSF_plus/ARCHITECTURE_DUMP.md) (`2,650,076` tham số / mô hình, không có Dropout):
       - `cond_proj`: `Linear(6759, 256) -> LayerNorm((256,)) -> SiLU()`
       - `time_mlp`: `SinusoidalTimeEmb(64) -> Linear(64, 256) -> SiLU()`
       - `x_proj`: `Linear(1500, 256)`
       - Cộng gộp: `h = x_proj(x_t) + cond_proj(cond) + time_mlp(t)`
       - `blocks`: `Linear(256, 256) -> LayerNorm((256,)) -> SiLU() -> Linear(256, 256) -> LayerNorm((256,)) -> SiLU() -> Linear(256, 1500) -> Sigmoid()`
     - Huấn luyện & lấy mẫu (`train_and_impute_diffusion`): dự đoán trực tiếp $x_0 \in [0, 1]^{1500}$ với loss `F.mse_loss(x0_hat, x0_b)`, `epochs = 100`, `batch_size = 64`, `AdamW(lr=1e-3, weight_decay=1e-4)`, số bước khuếch tán `n_timesteps (T) = 25` ($\beta \in [10^{-4}, 0.2]$), lấy mẫu trung bình `n_samples (B) = 10` quỹ đạo ngược theo cơ chế Cross-Fitting $K=5$.
3. **Training Profile View Dropout**:
   Trong quá trình huấn luyện, với xác suất $p_{\text{drop}} = 0.25$, ngẫu nhiên che `D9, D10` của thuốc train bằng `fallback_d9, fallback_d10` và hạ $a_{d,\text{D9}} = a_{d,\text{D10}} = 0$ để mô hình học cách thích ứng với thuốc mới.
4. **Bộ Điều biến Hiện diện (`AvailabilityModulator` trong `MSSF_plus/model.py`)**:
   Đặt trước các encoder. Với tham số học được `gate_fc1: Linear(V, 2V)`, `gate_act: SiLU()`, `gate_fc2: Linear(2V, V)` và token thiếu khả học `missing_token` $\in \mathbb{R}^{1 \times V \times D_v}$ (`8,767` tham số cho thuốc và `5,085` tham số cho SE):
   $$\tilde{X}_{b,v,:} = X_{b,v,:} \cdot \Bigl(1 + 0.2\,\tanh\!\bigl(W_2\,\text{SiLU}(W_1 a + b_1) + b_2\bigr)_{b,v}\Bigr) + (1 - a_{b,v})\,m_v$$
   Khởi tạo $W_2=0, b_2=0, m_v=0$ bảo đảm tại epoch 0 module là **ánh xạ đồng nhất tuyệt đối**.

### 4.2. C2 — Bổ sung Tri thức Y khoa MedDRA Ontology (`S5`)
- Trích xuất từ phân cấp chuẩn MedDRA (`side_effect_label_750.mat['node_label']`, ma trận nhị phân $994 \times 243$).
- Tính Cosine similarity tạo view tĩnh $S_5 \in [0, 1]^{994 \times 994}$ (`S5_meddra_ontology_cos`), cung cấp cấu trúc hệ cơ quan giải phẫu cho 100% 994 tác dụng phụ, bù đắp cho việc thiếu hồ sơ tương tác ở các SE hiếm.

### 4.3. C3 — Đơn giản hóa Hàm Mất mát Thứ tự Cân bằng Lớp (`MSSF_plus/losses.py`)
- Cấu trúc loss chuẩn của `MSSF_plus_full`:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{C3}}(\text{logits}, y, q) + 10^{-3}\mathcal{L}_{\text{KL}} + 10^{-4}\mathcal{L}_{\text{rec,con}} + 10^{-4}\mathcal{L}_{\text{rec,add}}$$
- Đã loại bỏ hoàn toàn số hạng `ce_softquant`.
- Hỗ trợ 3 họ hàm mục tiêu ordinal kết hợp trọng số lớp nghịch tần suất $w_k = \frac{N_{\text{train}}}{5 N_{\text{train}, k}}$:
  1. **`ce`**: $-\sum_{k=0}^4 q_k \log p_k$ (nhân $w_y$ khi bật weighting).
  2. **`emd`**: $\sum_{k=0}^3 \bigl(\sum_{j \le k} p_j - \sum_{j \le k} q_j\bigr)^2$ (Squared Earth Mover's Distance).
  3. **`corn`**: Conditional Ordinal Regression trên 4 bài toán nhị phân. Khi kết hợp với nhãn mềm $q$ của C5, phân phối được ánh xạ trực tiếp sang **xác suất tích lũy**:
     $$T_k = P(y > k) = \sum_{j = k+1}^4 q_j \quad (k = 0, 1, 2, 3)$$

### 4.4. C4 — Tiên đoán Bảo toàn Thứ tự Liên thông & Độ phủ theo Thuốc (`MSSF_plus/conformal.py`)
- Mở rộng khoảng thứ tự tham lam từ lớp cực đại $c^{(0)} = \arg\max_k p_k$, bảo đảm tập dự đoán luôn liên thông 100%: $\mathcal{C}(x) = [l, u] \subseteq \{0..4\}$.
- Hai phương pháp đề xuất: `Ordinal_LAC` và `Ordinal_APS` ở mức ý nghĩa $\alpha = 0.10$ (mục tiêu độ phủ $90\%$).
- Đánh giá thêm **độ phủ theo từng thuốc** ở Split B: `drug_coverage_mean`, `drug_coverage_std`, và tỷ lệ thuốc có độ phủ $< 80\%$ (`drug_coverage_lt80_rate`).

### 4.5. C5 — Nhãn mềm Cụm phổ Đa khung nhìn Không rò rỉ (`MSSF_plus/soft_labels.py`)
- Xây dựng đồ thị tương đồng đồng thuận từ các drug view chỉ trên `train_drugs`.
- Tự động chọn số cụm tối ưu $K^* \in [4, 12]$ cực đại hóa $\text{Score}(K) = 0.6\,\text{Silhouette} + 0.4\,\text{Eigengap}$.
- Sinh nhãn mềm Leave-One-Out nghiêm ngặt ($d' \neq d$):
  $$q_{d,s} = (1 - \alpha_{\text{c5}})\,\mathbf{e}_y + \alpha_{\text{c5}}\,P_{c,s}^{(-d)}, \qquad \alpha_{\text{c5}} = 0.15$$
- Nhánh phụ `aux_cluster_head` mặc định **TẮT** (`c5_aux_cluster_weight = 0.0`) trong mô hình chính.

---

## 5. Cấu trúc Mạng Thần kinh, Tham số Từng Module & Tối ưu hóa Đại số (`MSSF_plus/model.py`)

Mạng `MSSFPlusModel` kế thừa cấu trúc 3 nhánh mã hóa nhưng được tối ưu hóa toán học:

```
[x_d (8250-D), x_s (4970-D)] + [a_d, a_s] + [fallback_d9, fallback_d10]
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   AvailabilityModulator             AvailabilityModulator
     (Drug: 11 x 750)                   (SE: 5 x 994)
            │                                 │
     [mod_drugs]                       [mod_sides]
            ├────────────────┬────────────────┤
            ▼                ▼                ▼
     EncoderConnection  EncoderAddition   Preprocess
     (Concat 13220-D)   (Sum-pool)      (16 Linear proj)
            │                │                │
            │                │                ▼
            │                │        CrossProduction
            │                │    (Bilinear Factored Conv2d)
            │                │                │
            ▼                ▼                ▼
        feature1         feature2         feature3
        (128-D)          (128-D)          (128-D)
            └────────────────┼────────────────┘
                             ▼
              Concat [f1, f2, f3] (384-D)
                             │
                             ▼
              Multi-Head Attention (heads = 4)
                             │
                             ▼
             GaussianParametrizer (BVI: 64-D)
               mu (64-D), logvar (64-D)
                 z = mu + eps * std
                             │
                             ▼
                     Classifier MLP
                  (64 -> 32 -> 5 logits)
```

### Bảng Số Tham số Từng Module (Trích trực tiếp từ [`ARCHITECTURE_DUMP.md`](file:///Users/buihung/project_DL/DSE/MSSF_plus/ARCHITECTURE_DUMP.md))

| Module | Lớp (`MSSF_plus/model.py`) | `MSSF_v2` / `MSSF_v2_selfcorr` ($V_d=11, V_s=4$) | `C3_ce_unweighted` / `_selfcorr` ($V_d=11, V_s=5$) | Tỷ trọng (`C3_ce_unweighted`) |
| :--- | :--- | ---: | ---: | ---: |
| `drug_modulator` | `AvailabilityModulator` | `None` (0) | 8,767 | 0.08% |
| `se_modulator` | `AvailabilityModulator` | `None` (0) | 5,085 | 0.05% |
| `encoderConnection` | `EncoderConnection` | 6,602,562 | 7,112,484 | 66.14% |
| `encoderAddition` | `EncoderAddition` | 1,225,296 | 1,225,296 | 11.39% |
| `preprocess` | `Preprocess` | 1,570,688 | 1,698,304 | 15.79% |
| `crossProduction` | `CrossProduction` | 55,584 | 61,216 | 0.57% |
| `attention` | `Attention` | 591,744 | 591,744 | 5.50% |
| `gaussian_parametrizer` | `GaussianParametrizer` | 49,280 | 49,280 | 0.46% |
| `classifier` | `Classifier` | 2,245 | 2,245 | 0.02% |
| **Tổng (`MSSFPlusModel`)** | `MSSFPlusModel` | **10,097,399** | **10,754,421** | **100.00%** |

*(Lưu ý: ở giai đoạn tiền xử lý C1 của `C3_ce_unweighted`, mỗi mạng `ProfileViewRegressor` có `2,182,876` tham số, được huấn luyện độc lập trước khi khởi tạo `MSSFPlusModel`).*

### Công thức Tối ưu hóa Đại số cho Khối `CrossProduction` (`MSSF_plus/model.py` L236–279)
- **Vấn đề bản cũ**: Sau khi khối `Preprocess` chiếu từng drug view (`750 -> 128`) và từng SE view (`994 -> 128`), bản gốc tính tích ngoài từng cặp $e_d^{(v)} \otimes e_s^{(u)}$ sinh ra tensor 4 chiều kích thước $(B,\; V_d \times V_s,\; 128,\; 128)$ — tức $(B, 44, 128, 128)$ ở `MSSF_v2` ($V_d=11, V_s=4$) hoặc $(B, 55, 128, 128)$ ở `MSSF_plus` ($V_d=11, V_s=5$), ngốn $\sim 1.84\text{ GB}$ VRAM mỗi batch ($B=512$) và làm nghẽn cổ chai bộ nhớ GPU/CPU.
- **Giải pháp phân rã toán học tương đương chính xác**: Do tầng tích chập đầu tiên (`cnn[0]`) có kích thước cửa sổ bằng đúng bước nhảy ($\text{kernel\_size} = \text{stride} = 4$) với trọng số reshape thành 5 chiều $W^{(0)} \in \mathbb{R}^{32 \times V_d \times V_s \times 4 \times 4}$, phần tử tại vị trí $(i, j) \in \{0..31\}^2$ của kênh đầu ra $c \in \{0..31\}$ được biểu diễn dưới dạng tích vô hướng của các ma trận khối $D \in \mathbb{R}^{B \times V_d \times 32 \times 4}$ và $S \in \mathbb{R}^{B \times V_s \times 32 \times 4}$:
  $$\text{conv0\_out}_{b, c, i, j} = \sum_{v=1}^{V_d} \sum_{u=1}^{V_s} \sum_{p=0}^3 \sum_{q=0}^3 W_{c, v, u, p, q}^{(0)} \cdot D_{b, v, i, p} \cdot S_{b, u, j, q} + b_c^{(0)}$$
- Thực thi thông qua 2 phép nhân ma trận gọn (`torch.mm` và `torch.bmm`):
  $$M = \text{torch.mm}(D_{\text{flat}}, W_{\text{flat}}) \in \mathbb{R}^{(B \cdot 32) \times (32 \cdot V_s \cdot 4)}$$
  $$\text{conv0\_out} = \text{torch.bmm}(M, S_{\text{flat}}) \in \mathbb{R}^{B \times 32 \times 32 \times 32}$$
- **Kiểm chứng độ chính xác**: Độ sai lệch cực đại trên kiểu dữ liệu `float64` giữa bản gốc và bản phân rã là **$6.44 \times 10^{-15}$** (`1.49 \times 10^{-8}` ở `float32`), hoàn toàn tương đương về mặt toán học và giữ nguyên 100% trọng số checkpoint.

---

## 6. Bảng Tham số Cố định Của Mô hình Cơ sở (`ExperimentConfig` trong `MSSF_plus/configs/experiments.py`)

Bảng giá trị đọc trực tiếp từ [`ARCHITECTURE_DUMP.md`](file:///Users/buihung/project_DL/DSE/MSSF_plus/ARCHITECTURE_DUMP.md) và [`HPARAM_SELECTION.md`](file:///Users/buihung/project_DL/DSE/MSSF_plus/HPARAM_SELECTION.md):

| Tham số | Giá trị trong `ExperimentConfig` | Ý nghĩa / Vai trò |
| :--- | :---: | :--- |
| `epochs` | `50` | Số epoch huấn luyện tối đa (chọn checkpoint theo best `val_select` Macro-F1) |
| `batch_size` | `512` | Kích thước batch huấn luyện (khớp `MSSF` gốc) |
| `eval_batch_size` | `512` | Kích thước batch khi suy luận / đánh giá |
| `lr` | `0.0001` ($10^{-4}$) | Tốc độ học (`Adam` optimizer, khớp `MSSF` gốc) |
| `weight_decay` | `1e-05` ($10^{-5}$) | Hệ số suy giảm trọng số L2 (khớp `MSSF` gốc) |
| `dropout` | `0.4` | Dropout trong các khối `EncoderConnection`, `EncoderAddition`, `Preprocess`, `Classifier` |
| `gp` (latent dim) | `64` | Chiều không gian tiềm ẩn BVI ($z \sim \mathcal{N}(\mu, \sigma^2)$) |
| `kl_weight`, `rec_con_weight`, `rec_add_weight` | `0.001`, `0.0001`, `0.0001` | Trọng số 3 loss phụ kế thừa từ `MSSF/mssf.py` |
| `c1_view_dropout_prob` | `0.25` | Xác suất che view profile khi huấn luyện |
| `c1_impute` | `'regress'` | Phương pháp bù khuyết chính thức (`ProfileViewRegressor`, Cross-fitting $K=5$) |
| `c1_imputer_epochs` | `80` | Số epoch huấn luyện `ProfileViewRegressor` |
| `c2_se_mode` | `'plus_ontology'` | Kích hoạt view MedDRA $S_5$ ($V_s = 5$) |
| `c4_alpha` | `0.1` | Mức ý nghĩa Conformal Prediction (độ phủ mục tiêu $90\%$) |
| `c5_mode` | `'multi_view'` | Phân cụm phổ đa khung nhìn sinh nhãn mềm |
| `c5_alpha` | `0.15` | Hệ số trộn nhãn mềm cụm ($q = 0.85\,\mathbf{e}_y + 0.15\,P_{\text{cluster}}^{(-d)}$) |
| `c5_aux_cluster_weight`| `0.0` | Mặc định tắt cluster head phụ trong mô hình chính |
| `use_factored_crossprod` | `True` | Bật phân rã đại số chính xác cho `CrossProduction` |
| `profile_self_correction` | `False` (hoặc `True` ở bản `_selfcorr`) | Cờ khử đóng góp của chính cặp đang train trên `D9, D10, S3, S4` |
| `use_bvi` | `True` | Bật `GaussianParametrizer` (`False` ở biến thể ablation `full_no_bvi`) |

---

## 7. Kết quả Thực nghiệm Mốc Cơ sở (Benchmark Baseline Results)

### 7.1. Hiệu quả Bù khuyết C1 trên Tập thuốc Cold-Start (`val_select` Split B Fold 0)

| Chiến lược bù khuyết | MSE (Càng thấp càng tốt) | Tương quan hạng Spearman | Độ trùng Top-10 Láng giềng |
| :--- | :---: | :---: | :---: |
| `mean` (Vector trung bình các thuốc `train`) | $0.010361$ | $0.6364$ | $7.46\%$ |
| `diffusion` (`ConditionalDenoiserMLP`, $K=5$ Cross-fitting) | $0.010614$ | $0.6693$ | $14.03\%$ |
| **`regress` (`ProfileViewRegressor` $6759 \to 256 \to 256 \to 1500$, $K=5$ Cross-fitting)** | **$0.009274$** | **$0.6843$** | **$14.78\%$** (riêng D9: $16.57\%$) |

### 7.2. So sánh Phân loại trên `val_select` (Xuất trực tiếp từ `summarize_results.py --eval_target valsel`)

| Kịch bản | Mô hình | Số fold | Macro-F1 | Weighted-F1 | Accuracy | QWK | MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Split B** *(Cold-start Thuốc mới)* | `MSSF_v2` (Gốc khử rò rỉ) | 3 | **0.3630 ± 0.0148** | 0.4668 ± 0.0140 | 0.4877 ± 0.0356 | 0.4428 ± 0.0248 | 0.6447 ± 0.0574 |
| **Split B** *(Cold-start Thuốc mới)* | `C3_ce_unweighted` | 2 | **0.4117 ± 0.0153** | **0.5183 ± 0.0131** | **0.5350 ± 0.0052** | **0.4877 ± 0.0317** | **0.5688 ± 0.0015** |
| **Split A** *(Warm-start)* | `C3_ce_unweighted` | 2 | **0.6572 ± 0.0010** | **0.6948 ± 0.0044** | **0.6969 ± 0.0046** | **0.7297 ± 0.0030** | **0.3554 ± 0.0042** |

- **Mức cải thiện cốt lõi**: Trên tập thuốc hoàn toàn mới (Split B), mô hình cải tiến tăng **+0.0487 Macro-F1** (tăng tương đối **+13.4%** so với 3 fold, hoặc **+0.0592 Macro-F1** khi so cùng cặp Fold 0–1), tăng **+4.73% Accuracy**, và giảm sai số cấp bậc MAE từ **0.6447** xuống **0.5688**.

---

## 8. Danh mục Tệp Mã Nguồn của Phiên bản Đông kết

```text
MSSF_plus/
├── ARCHITECTURE_DUMP.md                  # [Nguồn sự thật duy nhất] Dump tự động từ đối tượng PyTorch & ExperimentConfig
├── ARCHITECTURE_SNAPSHOT_V2.md           # [Tài liệu này] Báo cáo đông kết kiến trúc mốc tham chiếu
├── ARCHITECTURE_REPORT.md                # Báo cáo phương pháp luận chi tiết phục vụ viết bài báo
├── BASELINE_FIDELITY.md                  # Đối chiếu thiết kế gốc tác giả vs Protocol v2 & cờ profile_self_correction
├── STEP2_COMPLETION_REPORT.md            # Báo cáo kiểm chứng số Bước 2 & hướng dẫn Colab
├── HPARAM_SELECTION.md                   # Minh bạch toàn bộ siêu tham số C1–C5
├── CHANGE1_CROSSFIT_REPORT.md            # Báo cáo kiểm chứng số K=5 Cross-fitting
├── figures/
│   ├── architecture_diagram.svg          # Sơ đồ vector kiến trúc tổng thể
│   └── architecture_diagram.png          # Sơ đồ độ phân giải cao nhúng bài báo / PDF
├── protocol.py                           # Seed, nạp split 4 tập (72/9/9/10; 80/10/10 trong non-test), lưu .npy chuẩn unified_eval
├── metrics.py                            # Bộ chỉ số phân lớp 5 mức & thứ tự (MacroF1, QWK, MAE, ...)
├── views.py                              # Dựng 11 Drug + 5 SE views, K=5 Cross-fitting ProfileViewRegressor & ConditionalDenoiserMLP
├── soft_labels.py                        # C5: Multi-view spectral clustering sinh nhãn mềm LOO
├── data.py                               # MSSFPlusDataset (vectorized __getitems__) & DataLoaders
├── model.py                              # MSSFPlusModel (AvailabilityModulator, Factored CrossProduction)
├── losses.py                             # C3: MSSFPlusCriterion ({ce, emd, corn} x class weighting)
├── conformal.py                          # C4: Ordinal-LAC, Ordinal-APS, thống kê per-drug coverage
├── train.py                              # Vòng lặp huấn luyện, chọn epoch trên val_select, cache view
├── configs/experiments.py                # 32 cấu hình thực nghiệm (bao gồm _selfcorr và full_no_bvi)
├── scripts/dump_architecture.py          # Sinh tự động ARCHITECTURE_DUMP.md từ đối tượng model & config
├── scripts/verify_stage1_crossprod.py    # Kiểm chứng tương đương CrossProduction gốc vs phân rã (Giai đoạn 1)
├── scripts/run_ablation.py               # CLI chạy thực nghiệm theo nhóm hoặc danh sách cấu hình
├── scripts/summarize_results.py          # Script duy nhất xuất mọi bảng kết quả từ các file .npy
├── tests/test_profile_self_correction.py # Unit test kiểm chứng công thức Gram vector hóa (sai số < 1e-6)
└── notebooks_v2/
    └── colab_0_proposed_and_ablations.ipynb  # Notebook Colab tích hợp sẵn ô chạy Giai đoạn 1–5
```

---

## 9. Tiêu chí Đánh giá để So sánh Khi Thay đổi Kiến trúc Lớn

Khi triển khai kiến trúc mới ở giai đoạn tới, cần đối chiếu trực tiếp với các mốc chuẩn trong tài liệu này theo checklist sau:

1. **Hiệu năng trên tập thuốc mới (Split B Cold-start)**:
   - Metric tiên quyết: `val_select Macro-F1` có vượt qua mốc **$0.4117$** hay không?
   - Tính trật tự: `QWK` có vượt qua **$0.4877$** và `MAE` có thấp hơn **$0.5688$** không?
2. **Khả năng tổng quát trên thuốc đã biết (Split A Warm-start)**:
   - `Macro-F1` có duy trì $\ge \mathbf{0.6572}$ và `QWK` $\ge \mathbf{0.7297}$ hay không?
3. **Chất lượng định lượng bất định (C4 Conformal)**:
   - Độ phủ thực nghiệm có đạt mức cam kết $90\%$ ($\alpha = 0.10$)? Kích thước tập dự đoán trung bình (`avg_set_size`) có thu hẹp hơn mức hiện tại ($2.50 - 2.85$) hay không?
   - Tỷ lệ thuốc test có độ phủ $< 80\%$ ở Split B có giảm xuống không?
4. **Độ phức tạp tính toán & Tốc độ huấn luyện**:
   - Kiến trúc mới có vượt quá ngưỡng $30 - 40\text{ giây / fold}$ trên GPU chuẩn (Colab T4) hay không?
