# Báo cáo Ngắn — Thay đổi 1: Cross-fitting $K=5$ cho Bù khuyết View Profile (`C1`)

## 1. Vấn đề & Giải pháp đã triển khai
- **Vấn đề cũ**: `ProfileViewRegressor` (`regress`) và `ConditionalDenoiserMLP` (`diffusion`) được huấn luyện trực tiếp trên `train_drugs` rồi dự đoán lại cho chính `train_drugs` để làm vector dự phòng (`fallback_d9`, `fallback_d10`). Do đó, khi `view_dropout` kích hoạt lúc train, mô hình nhận dự đoán **in-sample** (tốt giả tạo hơn nhiều so với dự đoán cold-start thật ở test).
- **Giải pháp đã sửa trong [`MSSF_plus/views.py`](file:///Users/buihung/project_DL/DSE/MSSF_plus/views.py)**:
  1. Hàm [`_make_drug_crossfit_splits`](file:///Users/buihung/project_DL/DSE/MSSF_plus/views.py#L152-L186): Chia `train_drugs` theo thuốc (seed cố định) thành $K = 5$ nhóm rời nhau ($108\text{ thuốc/nhóm}$ ở Split B; $150\text{ thuốc/nhóm}$ ở Split A). Với mỗi nhóm $k$, gộp 4 nhóm còn lại rồi tách 10% làm `val_sub` (early-stopping nội bộ) và 90% làm `fit_sub`.
  2. Hàm [`train_and_impute_regress`](file:///Users/buihung/project_DL/DSE/MSSF_plus/views.py#L189-L287) & [`train_and_impute_diffusion`](file:///Users/buihung/project_DL/DSE/MSSF_plus/views.py#L343-L490):
     - **Thuốc `train`**: Nhận dự đoán **out-of-fold** từ mô hình thứ $k$ không nhìn thấy thuốc đó $\rightarrow$ gán vào `fallback_d9`, `fallback_d10`.
     - **Thuốc `val_select`, `val_calib`, `test`**: Nhận **trung bình dự đoán của cả $K = 5$ mô hình**.
     - Chiến lược `mean` giữ nguyên 100%.

---

## 2. Xác nhận Toàn vẹn Dữ liệu (`shared_data/`)
- Đã kiểm tra chữ ký SHA256 bằng [`verify_checksums.py --check`](file:///Users/buihung/project_DL/DSE/MSSF_plus/audit/verify_checksums.py) trước và sau khi sửa:
  - **`90/90` file dữ liệu, split và code gốc trong `shared_data/` giữ nguyên bit-for-bit (`[PASS]`).**

---

## 3. Kết quả Kiểm chứng bằng Số (Fold 0 — Split B & Split A)

> *Ghi chú đo lường*: MSE và Cosine được tính trên các cột tham chiếu `train_drugs` ngoài đường chéo ($j \neq d$) để loại bỏ đường chéo tự tương đồng cứng `1.0` và các cột `0` cố định. Ở Split B, tương đồng thật (oracle) của 67 thuốc cold-start `val_select` được tính giữa hồ sơ thật của chúng trong `fold_split.valsel` với `drug_side_train`.

### 3.1. Split B — Fold 0 (`540` thuốc `train` rời hoàn toàn với `67` thuốc cold-start `val_select`)

| Chiến lược | Trường hợp đánh giá | Số thuốc | `D9` MSE | `D9` Cos | `D10` MSE | `D10` Cos | **Avg MSE** | **Concat Cos** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `mean` *(mốc)* | Dự đoán trên `val_select` | 67 | `0.010922` | `0.9222` | `0.009799` | `0.9195` | `0.010361` | `0.9206` |
| **`regress`** | **(a) Fallback in-sample cũ trên `train`** *(90% `fit_sub`: `0.008534`)* | 540 | `0.009025` | `0.9289` | `0.008236` | `0.9274` | **`0.008631`** | **`0.9280`** |
| **`regress`** | **(b) Fallback out-of-fold mới ($K=5$) trên `train`** | 540 | `0.010601` | `0.9259` | `0.009499` | `0.9241` | **`0.010050`** | **`0.9248`** |
| **`regress`** | **(c) Trung bình $K=5$ regressor trên `val_select`** *(1 model cũ: `0.009519`)* | 67 | `0.009696` | `0.9293` | `0.008852` | `0.9267` | **`0.009274`** | **`0.9278`** |
| **`diffusion`** | **(a) Fallback in-sample cũ trên `train`** *(không early-stop)* | 540 | `0.002711` | `0.9578` | `0.002555` | `0.9572` | **`0.002633`** | **`0.9573`** |
| **`diffusion`** | **(b) Fallback out-of-fold mới ($K=5$) trên `train`** | 540 | `0.012308` | `0.9236` | `0.011012` | `0.9219` | **`0.011660`** | **`0.9226`** |
| **`diffusion`** | **(c) Trung bình $K=5$ denoiser trên `val_select`** *(1 model cũ: `0.011075`)* | 67 | `0.011042` | `0.9276` | `0.010187` | `0.9250` | **`0.010614`** | **`0.9261`** |

### 3.2. Split A — Fold 0 (`750` thuốc `train`; `621` thuốc có cặp trong `val_select`)

| Chiến lược | Trường hợp đánh giá | Số thuốc | `D9` MSE | `D9` Cos | `D10` MSE | `D10` Cos | **Avg MSE** | **Concat Cos** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `mean` *(mốc)* | Trên `train` (750) / trên 621 thuốc `val_select` | 750 / 621 | `0.008991` / `0.008385` | `0.8794` / `0.8944` | `0.007967` / `0.007402` | `0.8773` / `0.8923` | `0.008479` / `0.007894` | `0.8781` / `0.8931` |
| **`regress`** | **(a) Fallback in-sample cũ trên `train`** *(90% `fit_sub`: `0.006381`)* | 750 | `0.006782` | `0.8852` | `0.006161` | `0.8832` | **`0.006471`** | **`0.8840`** |
| **`regress`** | **(b) Fallback out-of-fold mới ($K=5$) trên `train`** | 750 | `0.008860` | `0.8797` | `0.007902` | `0.8776` | **`0.008381`** | **`0.8784`** |
| **`regress`** | **(c) Fallback OOF trên 621 thuốc có mặt ở `val_select`** | 621 | `0.008355` | `0.8944` | `0.007423` | `0.8924` | **`0.007889`** | **`0.8932`** |
| **`diffusion`** | **(a) Fallback in-sample cũ trên `train`** | 750 | `0.002745` | `0.9341` | `0.002592` | `0.9323` | **`0.002669`** | **`0.9331`** |
| **`diffusion`** | **(b) Fallback out-of-fold mới ($K=5$) trên `train`** | 750 | `0.010111` | `0.8797` | `0.008967` | `0.8780` | **`0.009539`** | **`0.8786`** |
| **`diffusion`** | **(c) Fallback OOF trên 621 thuốc có mặt ở `val_select`** | 621 | `0.009607` | `0.8946` | `0.008498` | `0.8930` | **`0.009052`** | **`0.8936`** |

### 3.3. Tương quan thứ hạng Spearman & Độ trùng Top-10 láng giềng trên `val_select` (Split B, Fold 0 — 67 thuốc cold-start)

| Chiến lược bù khuyết | `D9` Spearman | `D9` Top-10 Overlap | `D10` Spearman | `D10` Top-10 Overlap | **Avg Spearman** | **Avg Top-10 Overlap** | **Avg MSE** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `mean` | `0.6456` | `7.91%` | `0.6273` | `7.01%` | `0.6364` | `7.46%` | `0.010361` |
| **`regress` ($K=5$)** | **`0.6929`** | **`16.57%`** | **`0.6758`** | **`12.99%`** | **`0.6843`** | **`14.78%`** | **`0.009274`** |
| `diffusion` ($K=5$) | `0.6788` | `15.22%` | `0.6598` | `12.84%` | `0.6693` | `14.03%` | `0.010614` |

> **Nhận xét**: Chiến lược `mean` gán cùng một vector trung bình tĩnh cho mọi thuốc cold-start nên độ trùng Top-10 láng giềng chỉ đạt `7.46%`. Chiến lược `regress` cá nhân hóa dự đoán từ 9 view tĩnh của từng thuốc, giúp **tăng gấp đôi độ trùng Top-10 láng giềng** (`14.78%` vs `7.46%`, riêng `D9` đạt `16.57%`), tăng Spearman từ `0.6364` lên `0.6843` và đạt MSE thấp nhất (`0.009274`). Ngược lại, `diffusion` có MSE cao hơn `mean` (`0.010614` vs `0.010361`) nên chỉ giữ làm một dòng đối chứng trong bảng ablation `C1` trên dev folds, loại khỏi `MSSF_plus_full`, khỏi ngân sách tune và khỏi lượt chạy 10 fold.

---

## 4. Kết luận & Các Vấn đề Phát hiện

1. **Kỳ vọng `(b) ≈ (c)` và `(a)` tốt giả tạo hơn hẳn được xác nhận hoàn toàn**:
   - Ở `regress`: `(a)` cũ có MSE thấp giả tạo hơn `(b)` tới **14.1%** (Split B: `0.008631` vs `0.010050`) và **22.8%** (Split A: `0.006471` vs `0.008381`), trong khi `(b)` và `(c)` nằm sát nhau (`0.010050` vs `0.009274` ở Split B; `0.008381` vs `0.007889` ở Split A).
   - Ở `diffusion`: `(a)` cũ có MSE thấp giả tạo gấp **4.4 lần** ở Split B (`0.002633` vs `0.011660`) và **3.6 lần** ở Split A (`0.002669` vs `0.009539`) do bản cũ train trên 100% `train_drugs` không có early-stopping; sau khi sửa, `(b) ≈ (c)` (`0.011660` vs `0.010614`).
2. **Đặc thù của Split A (chia theo cặp)**:
   - Cả `750/750` thuốc đều thuộc `train_drugs` (`d9_avail == 1.0`), nên `621` thuốc xuất hiện trong `val_select` là tập con của `train_drugs`. Khi đánh giá `val_select`/`test` ở Split A, mô hình dùng trực tiếp `d9_raw` từ `train`; còn khi `view_dropout` lúc train, việc khóa `final_pred[tr_idx] = oof_pred[tr_idx]` giúp ngăn chặn hoàn toàn rò rỉ $4/5$ in-sample (nếu lấy trung bình 5 mô hình cho 621 thuốc trùng này thì MSE sẽ tụt giả tạo xuống `0.005358`).
3. **Đồng bộ khóa cấu hình số epoch bù khuyết**:
   - Phát hiện [`experiments.py`](file:///Users/buihung/project_DL/DSE/MSSF_plus/configs/experiments.py#L33) đặt tên `'c1_imputer_epochs': 80` trong khi [`views.py`](file:///Users/buihung/project_DL/DSE/MSSF_plus/views.py#L568-L580) cũ đọc `'c1_impute_epochs'`. Đã sửa để nhận cả hai khóa và tách riêng `'c1_diffusion_epochs'` (mặc định `100`).
