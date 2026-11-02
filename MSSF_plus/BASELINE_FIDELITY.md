# TÀI LIỆU ĐỐI CHIẾU TÍNH TRUNG THỰC CỦA BASELINE & GIAO THỨC V2 (`BASELINE_FIDELITY.md`)

> **Mục đích tài liệu**: Ghi nhận minh bạch thiết kế gốc của 7 mô hình đối sánh (baselines), các điểm rò rỉ dữ liệu phát hiện trong mã nguồn tác giả gốc, các điều chỉnh chuẩn hóa khi đưa về **Protocol v2**, và cơ chế cờ `--profile_self_correction {0, 1}` để vừa báo cáo trung thực theo thiết kế gốc của tác giả, vừa đánh giá đối chứng khi khử rò rỉ nhãn của chính cặp đang huấn luyện.

---

## 1. Xác minh Thiết kế Gốc của `MSSF` và Bản Chuẩn hóa `MSSF_v2`

- **Trong mã nguồn gốc của tác giả ([`MSSF/mssf.py`](file:///Users/buihung/project_DL/DSE/MSSF/mssf.py#L268-L310))**:
  - `MSSF` là mô hình **phân lớp đa lớp 5 mức độ tần suất** ($\{0, 1, 2, 3, 4\}$ tương ứng với nhãn tần suất $\{1, 2, 3, 4, 5\}$), **không phải mô hình hồi quy MSE**.
  - Trích mã nguồn gốc [`MSSF/mssf.py` dòng 271–308](file:///Users/buihung/project_DL/DSE/MSSF/mssf.py#L271-L308):
    ```python
    kl_div = kl_func(mu, logvar).mean()
    loss_func = nn.CrossEntropyLoss()
    multi_labels = (batch_ratings.long()-1).to(device)
    ...
    multi_loss = loss_func(multi_pred, multi_labels)
    reconst_loss = nn.MSELoss(reduction='none')
    rec_loss1 = reconst_loss(recCon, batch_vec.to(device)).sum(dim=-1).mean()
    rec_loss2 = reconst_loss(recAdd, add_features.to(device)).sum(dim=-1).mean()
    Loss = multi_loss + 0.001 * kl_div + 0.0001 * rec_loss1 + 0.0001 * rec_loss2
    ```
- **Trong mã nguồn `MSSF_v2` hiện tại ([`MSSF_plus/configs/experiments.py`](file:///Users/buihung/project_DL/DSE/MSSF_plus/configs/experiments.py#L16-L61) & [`MSSF_plus/losses.py`](file:///Users/buihung/project_DL/DSE/MSSF_plus/losses.py#L120-L237))**:
  - Cấu hình `MSSF_v2` kế thừa 100% kiến trúc và hàm mất mát của `MSSF/mssf.py`:
    - `'c3_loss_type': 'ce'`, `'c3_class_weighting': False`, `'c5_mode': 'none'`, `'kl_weight': 0.001`, `'rec_con_weight': 0.0001`, `'rec_add_weight': 0.0001`, `'epochs': 50`, `'batch_size': 512`, `'lr': 0.0001`, `'weight_decay': 1e-5`, `'dropout': 0.4`.
    - Tại [`MSSF_plus/losses.py` dòng 120–122](file:///Users/buihung/project_DL/DSE/MSSF_plus/losses.py#L120-L122): khi `soft_targets is None` và `use_class_weights=False`, hàm `_ce_with_soft_targets` gọi trực tiếp `F.cross_entropy(logits, hard_labels, weight=None)` cộng với `0.001 * kl_div + 0.0001 * rec_loss1 + 0.0001 * rec_loss2` (dòng 232–237).
  - Như vậy, `MSSF_v2` trong codebase đã đúng hoàn toàn 100% với thiết kế phân lớp 5 lớp (`CE + 0.001*KL + 0.0001*rec_con + 0.0001*rec_add`) của tác giả gốc; chỉ loại bỏ lỗi rò rỉ tập `val`/`test` khi dựng ma trận tương đồng hồ sơ và lỗi chọn checkpoint trực tiếp trên tập `test`.

---

## 2. Bảng Đối chiếu: Protocol Gốc của Tác giả vs. Protocol v2 vs. Cờ `--profile_self_correction`

| Mô hình | Bản chất đầu ra & Hàm Loss gốc | Vấn đề trong Protocol gốc của tác giả | Chuẩn hóa trong Protocol v2 (`baselines_rerun/`, mặc định `--profile_self_correction 0`) | Khi bật `--profile_self_correction 1` (lưu vào `*_selfcorr_{A\|B}`) |
| :--- | :--- | :--- | :--- | :--- |
| **MSSF (`MSSF_v2`)** | Phân lớp 5 mức (`CrossEntropyLoss` + `0.001 KL` + `0.0001 MSE_con` + `0.0001 MSE_add`) | Chọn epoch tốt nhất theo `test` Accuracy (`mssf.py` L236); ở Split B chỉ xóa các cặp dương trong `data_test`. | Chia 4 tập `72% / 9% / 9% / 10%` (`train / val_select / val_calib / test`); chỉ dựng `D9, D10, S3, S4` từ `train`; chọn checkpoint theo `val_select` Macro-F1; khóa `test`. | Với mỗi cặp train $(d, s)$, trừ đóng góp của ô $Y_{\text{train}}[d, s]$ khỏi hàng `D9[d,:]`, `D10[d,:]`, `S3[s,:]`, `S4[s,:]` bằng công thức Gram vector hóa trên GPU. |
| **SDPred** | Đa nhiệm: Nhị phân (`MSE`) $\times$ Hồi quy tần suất $1..5$ (`MSE`) trên `ConvNCF` | Chọn epoch trên `test`; dùng cấu hình siêu tham số khác nhau giữa Split A và B; tính `d_sim, s_sim` có chứa cặp `val`. | Dựng 4 view hồ sơ `D9, D10, S3, S4` chỉ từ `train`; lấy mẫu âm 1:1 không rò rỉ thuốc cold-start; chọn checkpoint chính theo `val_select` Macro-F1 (sau khi `round & clip 1..5`) và lưu thêm checkpoint theo `RMSE`. | Tương tự `MSSF_v2`: khử đóng góp của chính cặp $(d, s)$ trên 4 kênh tương đồng hồ sơ `[D9, D10, S3, S4]` trong mỗi batch huấn luyện. |
| **MVSL-DSF** | Đa nhiệm: Nhị phân (`CE`) + Tần suất chuẩn hóa $y/5 \in [0, 1]$ (`BCEWithLogitsLoss`) + `consistency_loss` | Nhánh `ConvNCF` nhận trực tiếp vector hàng tần suất `freq_prof = freq_train[d, :]` (chứa nguyên nhãn đích $y(d, s)/5$ tại vị trí $s$). | Xây dựng `assoc_train` và `freq_train` chỉ từ `train` (zero toàn bộ cặp `val_select`, `val_calib`, `test`); dùng ontology MedDRA đầy đủ $994 \times 243$. | Trong mỗi batch huấn luyện cặp $(d, s)$, đặt trực tiếp `assoc_prof[b, s] = 0`, `freq_prof[b, s] = 0`, và `assoc_prof_t[b, d] = 0` trước khi đưa vào `ConvNCF`. |
| **MGPred** | Hồi quy tần suất $1..5$ (`MSE`) trên mạng đồ thị đa quan hệ + EMA | `u_adj[d]` và `u2e1(d)` chứa cạnh đích $(s, r)$; lỗi khởi tạo `avg_param = deepcopy(p.data)` chia cho $(1 - \beta^t)$ gây nổ số ở các epoch đầu. | Xây dựng `u_adj, i_adj, u2e1, i2e1` chỉ từ `train`; sửa khởi tạo EMA về `0` đúng công thức Louizos et al. (2018); chọn checkpoint trên `val_select`. | Che cạnh đích $(d, s)$ khỏi danh sách láng giềng và đặt `u2e1(d)[s] = 0, i2e1(s)[d] = 0` khi tính biểu diễn cặp huấn luyện $(d, s)$. |
| **DSGAT** | Hồi quy ma trận $750 \times 994$ (`MSE` trên cặp dương + $\lambda \cdot \text{MSE}(0.5)$ trên ô $0$) | Chọn checkpoint trên `test`; đồ thị k-NN của SE dựng trên toàn bộ dữ liệu. | Ở Split B chỉ đưa 540 thuốc `train_drugs` vào `train_loader`; dựng đồ thị k-NN của SE chỉ từ `drug_side_train[active_train_drugs].T`. | Không bị ảnh hưởng bởi tự rò rỉ cặp $(d, s)$ vì đầu vào của thuốc chỉ là đồ thị phân tử SMILES. |
| **A3Net** | Hồi quy ma trận $750 \times 994$ (`MSE` có trọng số âm) | Tương tự `DSGAT`: chọn checkpoint trên `test`. | Chuẩn hóa giống `DSGAT` trên 4 tập của Protocol v2; chọn checkpoint trên `val_select`. | Không bị ảnh hưởng bởi tự rò rỉ cặp $(d, s)$ ở đầu vào. |
| **HSTrans** | Hồi quy tần suất $1..5$ (`MSE`) trên Transformer cấu trúc con BPE | Hàm `identify_sub` trích xuất top-50 cấu trúc con BPE làm giàu cho từng SE trên toàn bộ ma trận `drug_side` (rò rỉ tập `test`). | Tính lại `_identify_sub_train_only` riêng cho từng fold chỉ từ các cặp dương của tập `train` (80%). | Không dùng ma trận hồ sơ tương tác `D9/D10/S3/S4` làm đầu vào. |

---

## 3. Nguyên tắc Báo cáo Số liệu trong Bài báo

1. **Nguyên tắc nhất quán trong cùng một bảng**: Mọi mô hình xuất hiện trong cùng một bảng so sánh đều phải dùng **cùng một thiết lập `profile_self_correction`** (hoặc cùng tắt `False`, hoặc cùng bật `True`).
2. **Một nguồn sinh bảng duy nhất**: Mọi bảng kết quả phân lớp và Conformal Prediction đều được sinh tự động từ các tệp `.npy` đã lưu trên đĩa thông qua script [`MSSF_plus/scripts/summarize_results.py`](file:///Users/buihung/project_DL/DSE/MSSF_plus/scripts/summarize_results.py).
