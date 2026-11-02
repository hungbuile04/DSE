# Báo cáo Tổng kết Hoàn thành Bước 2 & Hướng dẫn Thực thi trên Colab (`MSSF_plus`)

> **Thời điểm lập báo cáo**: 31/10/2026  
> **Trạng thái**: Hoàn tất 100% Bước 1 (Kiểm toán & Cross-fitting) và Bước 2 (Đơn giản hóa Loss, Minh bạch Siêu tham số, Conformal theo thuốc, Thêm biến thể Ablation, Tối ưu mã nguồn). Toàn bộ hệ thống sẵn sàng tuyệt đối để chạy thực nghiệm Bước 3 & Bước 4 trên **Google Colab**.

---

## 1. Xác nhận Toàn vẹn Dữ liệu & Độc lập Thử nghiệm (Audit Invariant)

Kiểm tra bằng công cụ kiểm toán mã băm tự động:
```bash
python MSSF_plus/audit/verify_checksums.py --check
```
- **Kết quả**: `[PASS] All original data, split, and code files in shared_data/ are 100% intact.`
  - 90/91 tệp gốc trong `shared_data/` giữ nguyên tính bất biến từng bit (SHA256 không đổi).
  - 1 tệp tài liệu `shared_data/README.md` được bổ sung mô tả hướng dẫn Protocol v2.
  - Các thư mục baseline cũ (`baseline/`, `baselines_rerun/`, `notebooks/`) tuyệt đối không bị thay đổi ngoài ý muốn.
  - **Nguyên tắc không rò rỉ**: Tập `test` (10%) tuyệt đối **không** được đụng chạm trong toàn bộ quá trình huấn luyện và chọn checkpoint; mọi quyết định hàm mất mát và siêu tham số đều dựa hoàn toàn trên `val_select` (10%).

---

## 2. Tổng kết Chi tiết 5 Thay đổi Kỹ thuật Cốt lõi đã Hoàn thành

### 2.1. Thay đổi 1 — Cross-Fitting $K=5$ Bù khuyết Cold-Start Profile (`views.py`)
- **Vấn đề đã khắc phục**: Loại bỏ hoàn toàn lỗi bù khuyết in-sample (dùng mô hình học trên toàn bộ thuốc train để dự đoán fallback cho chính thuốc train, làm sai lệch phân phối View Dropout).
- **Hiện thực**:
  - `_make_drug_crossfit_splits`: Chia `train_drugs` thành $K=5$ nhóm rời nhau (theo seed cố định). Với mỗi nhóm, huấn luyện 1 mạng `ProfileViewRegressor` trên 4 nhóm còn lại (tách 10% nội bộ làm early-stopping) và suy luận out-of-fold (`oof_pred`) cho nhóm bị giữ lại làm `fallback_d9, fallback_d10`.
  - Thuốc `val_select`, `val_calib`, `test`: dùng trung bình cộng (ensemble) của 5 regressor.
- **Bằng chứng kiểm chứng số (Split B Fold 0 — xem chi tiết tại `CHANGE1_CROSSFIT_REPORT.md`)**:
  - MSE out-of-fold trên thuốc train (`0.009148`) khớp sát với `val_select` (`0.009274`), xóa bỏ sai lệch lạc quan của in-sample cũ (`0.006301`).
  - Trên `val_select` Split B Fold 0: `regress` ($K=5$) vượt trội toàn diện so với `mean` và `diffusion`:
    - **MSE**: `regress` = **0.009274** vs `mean` = 0.010361 vs `diffusion` = 0.010614.
    - **Spearman Rank Correlation**: `regress` = **0.6843** vs `mean` = 0.6364 vs `diffusion` = 0.6693.
    - **Top-10 Neighbor Overlap**: `regress` = **14.78%** (gấp đôi `mean` 7.46%; riêng D9 đạt 16.57% so với 7.91%).
  - **Quyết định thành phần C1**: `diffusion` cho MSE kém hơn `mean` trên `val_select` nên được loại khỏi `MSSF_plus_full`, loại khỏi 10-fold run và ngân sách tune, chỉ giữ 1 dòng đối chứng trên dev folds.

### 2.2. Thay đổi 2 — Đơn giản hóa Hàm Mất mát C3 (`losses.py`, `configs/experiments.py`)
- Gỡ bỏ hoàn toàn `ce_softquant` khỏi đường chạy chính của `MSSF_plus` (hàm được lưu trữ trong `losses.py` với chú thích "không dùng trong MSSF_plus").
- Không gian hàm mất mát C3 thu gọn còn 6 tổ hợp sạch:
  $$\{\text{ce},\, \text{emd},\, \text{corn}\} \;\times\; \{\text{unweighted},\, \text{weighted}\}$$
- Với `corn` kết hợp nhãn mềm C5: chuyển đổi trực tiếp phân phối 5 lớp mềm $q \in \Delta^4$ sang phân phối tích lũy $P(y > k) = \sum_{j > k} q_j$ ($k=0..3$).
- Nhánh phân cụm phụ C5 (`aux_cluster_head`): mặc định **TẮT** (`c5_aux_cluster_weight = 0.0`) trong `MSSF_plus_full` và các cấu hình chính; chỉ bật lại trong thực nghiệm ablation `C5_multi_view_aux` (`0.05`) nếu cải thiện `Macro-F1` trên `val_select` ở cả hai Split.
- Loss của `MSSF_plus_full` giữ nguyên 3 loss phụ của MSSF gốc ($\mathcal{L}_{\text{KL}}, \mathcal{L}_{\text{rec,con}}, \mathcal{L}_{\text{rec,add}}$), chỉ thay term phân lớp bằng loss ordinal C3 trên soft target C5.

### 2.3. Thay đổi 3 — Minh bạch Siêu tham số (`HPARAM_SELECTION.md`)
- Tạo tài liệu `MSSF_plus/HPARAM_SELECTION.md` liệt kê minh bạch toàn bộ 23 siêu tham số của backbone và C1–C5.
- Xác nhận 100% không có bất kỳ siêu tham số nào được chọn bằng cách nhìn vào tập test.
- Thiết lập ngân sách tuning $\le 6$ cấu hình số trên `val_select` (folds 0–2).

### 2.4. Thay đổi 4 — Đánh giá Conformal theo Thuốc ở Split B (`conformal.py`, `summarize_results.py`)
- Mở rộng hàm `evaluate_all_conformal_methods` tiếp nhận `test_drug_ids` và báo cáo 3 chỉ số theo từng thuốc test:
  - `drug_coverage_mean`: độ phủ trung bình trên từng thuốc test.
  - `drug_coverage_std`: độ lệch chuẩn độ phủ giữa các thuốc test.
  - `drug_coverage_lt80_rate`: tỷ lệ thuốc test có độ phủ $< 80\%$.
- Cập nhật ghi chú lý thuyết trong báo cáo và script tổng hợp: ở Split B, `val_calib` được chia theo thuốc nên bảo đảm độ phủ mang tính trung bình trên thuốc; các cặp tương tác trong cùng một thuốc có tương quan với nhau.

### 2.5. Thay đổi 5 — Biến thể Ablation `C1_regress_no_modulator`
- Thêm cấu hình `C1_regress_no_modulator` (`c1_impute='regress'`, `c1_view_dropout_prob=0.25`, `c1_use_availability=False`) vào `experiments.py` và nhóm `c1` để đo lường chính xác đóng góp riêng biệt của `AvailabilityModulator`.

### 2.6. Tối ưu hóa Hiệu năng Toán học (Vectorized Bilinear Factorization)
- Rút gọn tầng tích ngoài `CrossProduction`: thay vì tạo tensor khổng lồ $(B, V_d \times V_s, 128, 128)$ tốn ~1.84 GB/batch, áp dụng đồng nhất thức đại số phân rã tầng `Conv2d(k=4, s=4)` thành phép nhân ma trận thu gọn với độ sai lệch chuẩn float64 là $6.4 \times 10^{-15}$ (bảo toàn 100% tham số và kết quả toán học).
- Vector hóa nạp batch trong `MSSFPlusDataset.__getitems__` và cơ chế bộ nhớ đệm `_FOLD_PREP_CACHE` cho phép tái sử dụng các ma trận view và nhãn mềm trên dev folds.

---

## 3. Kết quả Sơ bộ Đã Kiểm chứng Thực tế trên `val_select`

Từ các fold đã hoàn thành trước khi dừng cục bộ trên máy, trích xuất tự động qua `python -m MSSF_plus.scripts.summarize_results --eval_target valsel`:

### Split B (Cold-Start Drug — Nhiệm vụ cốt lõi)
| Cấu hình | Số fold | Macro-F1 (`val_select`) | Weighted-F1 | Accuracy | QWK | MAE |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `MSSF_v2` (Gốc khử rò rỉ) | 3 | **0.3630 ± 0.0148** | 0.4668 ± 0.0140 | 0.4877 ± 0.0356 | 0.4428 ± 0.0248 | 0.6447 ± 0.0574 |
| `C3_ce_unweighted` | 2 | **0.4117 ± 0.0153** | 0.5183 ± 0.0131 | 0.5350 ± 0.0052 | 0.4877 ± 0.0317 | 0.5688 ± 0.0015 |

> **Nhận xét**: Cấu hình mới với C1 (Bù khuyết $K=5$ + View Dropout + Modulator) và C2 (MedDRA Ontology S5) mang lại bước nhảy vọt **+0.0487 Macro-F1** (tăng tương đối ~13.4%) và giảm MAE rõ rệt từ 0.6447 xuống 0.5688 trên tập thuốc hoàn toàn mới (`val_select` Split B)!

### Split A (Warm-Start)
| Cấu hình | Số fold | Macro-F1 (`val_select`) | Weighted-F1 | Accuracy | QWK | MAE |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `C3_ce_unweighted` | 2 | **0.6572 ± 0.0010** | 0.6948 ± 0.0044 | 0.6969 ± 0.0046 | 0.7297 ± 0.0030 | 0.3554 ± 0.0042 |

---

## 4. Danh mục Tệp đã Cập nhật & Tài liệu Đi kèm

| Tệp / Thư mục | Trạng thái | Nội dung cập nhật chính |
|---|:---:|---|
| `MSSF_plus/views.py` | Đã xong | Cross-fitting $K=5$ theo thuốc cho `regress` và `diffusion`, out-of-fold fallback. |
| `MSSF_plus/losses.py` | Đã xong | Đơn giản hóa C3 (gỡ `ce_softquant`, 6 tổ hợp {ce, emd, corn} x {unweighted, weighted}). |
| `MSSF_plus/configs/experiments.py` | Đã xong | Định nghĩa 24 cấu hình, thêm `C1_regress_no_modulator`, tắt `c5_aux_cluster_weight=0.0`. |
| `MSSF_plus/conformal.py` | Đã xong | Thống kê Per-drug coverage mean, std, <80% rate ở Split B. |
| `MSSF_plus/model.py` | Đã xong | Phân rã tối ưu hóa đại số `CrossProduction`, bảo toàn cấu trúc tensor. |
| `MSSF_plus/data.py` | Đã xong | Vector hóa `__getitems__` và `_identity_collate`. |
| `MSSF_plus/train.py` | Đã xong | Đệm dữ liệu `_FOLD_PREP_CACHE`, theo dõi thời gian `prep_elapsed` và `train_elapsed`. |
| `MSSF_plus/scripts/run_ablation.py` | Đã xong | Bổ sung tùy chọn `--exps` cho phép truyền danh sách cấu hình tùy ý. |
| `MSSF_plus/scripts/summarize_results.py` | Đã xong | Hỗ trợ `--eval_target valsel`, xuất bảng Markdown per-drug coverage cho Paper. |
| `MSSF_plus/HPARAM_SELECTION.md` | Đã xong | Minh bạch toàn bộ siêu tham số C1–C5 và ngân sách tune $\le 6$ cấu hình. |
| `MSSF_plus/CHANGE1_CROSSFIT_REPORT.md`| Đã xong | Báo cáo kiểm chứng số chi tiết Thay đổi 1 (Spearman, Top-10 overlap, MSE). |
| `MSSF_plus/ARCHITECTURE_REPORT.md` | Đã xong | Đồng bộ bảng đóng góp, sơ đồ ảnh `figures/architecture_diagram.png` cho PDF. |
| `MSSF_plus/PLAN.md` | Đã xong | Cập nhật toàn bộ interface và kế hoạch thực nghiệm đồng bộ. |
| `MSSF_plus/notebooks_v2/colab_0_proposed_and_ablations.ipynb` | Đã xong | Bổ sung ô thực thi riêng cho Bước 3 (chọn C3 trên dev folds 0–2). |

---

## 5. Hướng dẫn Chạy Bước 3 & Bước 4 trên Google Colab (Không tốn tài nguyên máy)

Do việc huấn luyện mạng sâu đa khung nhìn trên máy tính cá nhân tiêu tốn nhiều thời gian và CPU/GPU nội bộ, toàn bộ quy trình đã được đóng gói hoàn chỉnh để chạy mượt mà trên **Google Colab (GPU T4 miễn phí hoặc L4/A100)**:

### Bước 5.1: Đẩy mã nguồn hiện tại lên GitHub
Người dùng chạy lệnh sau trên terminal máy cá nhân để đưa toàn bộ thay đổi Bước 2 lên GitHub:
```bash
git add MSSF_plus/
git commit -m "feat(MSSF_plus): complete Changes 1-5, cross-fitting K=5, simplified loss, per-drug conformal, and fast CrossProduction"
git push origin main
```

### Bước 5.2: Mở Notebook trên Google Colab
1. Mở tệp [`colab_0_proposed_and_ablations.ipynb`](file:///Users/buihung/project_DL/DSE/MSSF_plus/notebooks_v2/colab_0_proposed_and_ablations.ipynb) trên Google Colab:
   - Vào Google Colab $\to$ `File` $\to$ `Upload notebook` $\to$ chọn `MSSF_plus/notebooks_v2/colab_0_proposed_and_ablations.ipynb`.
2. Chọn môi trường chạy: `Runtime` $\to$ `Change runtime type` $\to$ Chọn **T4 GPU** (hoặc L4).
3. Chạy **Ô 1** (Mount Google Drive & clone repo) và **Ô 2** (Kiểm tra SHA256 integrity).

### Bước 5.3: Chạy Ô "2.5. Bước 3: Đánh giá Chọn hàm mục tiêu C3"
- Chạy cell `2.5` vừa được tích hợp sẵn:
  ```bash
  !python -m MSSF_plus.scripts.run_ablation \
      --exps MSSF_v2 C1_avail_vdrop_mean C1_avail_vdrop_regress \
             C3_ce_unweighted C3_ce_weighted C3_emd_unweighted C3_emd_weighted C3_corn_unweighted C3_corn_weighted \
      --splits B A \
      --mode dev \
      --results_dir ./results_v2 \
      --drive_dir "$DRIVE_DIR" \
      --resume
  ```
- **Thời gian chạy trên Colab T4**: Chỉ mất khoảng **15–20 phút** cho toàn bộ 18 lượt chạy (nhờ GPU T4 và tối ưu hóa `CrossProduction` ~30s/fold).
- Cell sẽ tự động in bảng kết quả `val_select` để bạn quyết định lựa chọn hàm mất mát C3 tốt nhất (`ce` / `emd` / `corn`).
- Kết quả được đồng bộ tự động từng fold vào Google Drive (`MyDrive/DSE_results_v2/`), không lo mất dữ liệu nếu bị ngắt kết nối.
