# Quyết định Duyệt Giai đoạn 5 (`mssf_slim`), Bảng Đối chiếu Cùng Tập Fold `0–2`, và Kế hoạch 10-Fold `val_select`

> **Trạng thái giao thức**: Tập `test` (10%) giữ **ĐÓNG TUYỆT ĐỐI (`eval_test = False`)** cho đến khi hoàn tất và được duyệt toàn bộ kết quả 10 fold trên `val_select`.

---

## 1. Quyết định Chọn Mô hình Cuối Giai đoạn 5 (Đã Duyệt)

### 1.1. Mô hình cuối được chọn: **`slim_nodec`**
- **Quy mô tham số**: **`6,803,109`** tham số — **giảm `-36.51%` (`3.91 triệu` tham số)** so với `cand_A` (`10,715,929` tham số), giảm bộ nhớ GPU đỉnh từ `881.3 MB` xuống `758.1 MB` (`-14.0%`), và rút ngắn thời gian huấn luyện từ `~85.5s/fold` xuống `~74.6s/fold`.
- **Hiệu năng trên `val_select` (Folds `0–2`)**:
  - **Split B (Cold-start theo thuốc)**: `Macro-F1 = 0.4223 ± 0.0291` ($\Delta = +0.0100 \pm 0.0056$ so với `cand_A`, tăng đồng loạt trên cả 3 fold: `+0.0034, +0.0094, +0.0172`), `QWK = 0.4941 ± 0.0468`, `MAE = 0.5782 ± 0.0286`.
  - **Split A (Warm-start theo cặp)**: `Macro-F1 = 0.6475 ± 0.0082` ($\Delta = +0.0040 \pm 0.0050$ so với `cand_A`), `QWK = 0.7301 ± 0.0045`, `MAE = 0.3639 ± 0.0080`.
- **Lý do ghi nhận chính thức**:
  Quy tắc tiền định yêu cầu chọn cấu hình gọn nhất không tụt quá $1\,\text{Std}$ giữa các fold ở cả hai split. **`slim_nodec` vừa gọn hơn `cand_A` (`-36.51%` tham số), vừa KHÔNG tụt mà còn tăng Macro-F1 ở cả hai split**, nên trội hơn `slim_all_cnnshared` theo cả tiêu chí độ chính xác dự đoán lẫn tính tối giản kiến trúc (chỉ bỏ hai nhánh decoder tự mã hóa dư thừa mà không cần thêm cơ chế phụ).

### 1.2. Vai trò của `slim_all_cnnshared` và Ghi chú Diễn giải trong Bảng Ablation
- **`slim_all_cnnshared` (`2,237,621` tham số, `-79.12%` so với `cand_A`)**: Được giữ lại trong bảng ablation chính thức với vai trò **biến thể "nén tối đa" (`maximum-compression variant`)**, minh họa khả năng giảm gần 80% tham số mà vẫn vượt xa các baseline ngoại trên cả hai split (`0.3987 ± 0.0140` ở Split B; `0.6456 ± 0.0079` ở Split A).
- **Ghi chú diễn giải giữa `slim_all` và `slim_all_cnnshared`**: Trên Split B (Folds `0–2`), `slim_all` (`3.80M` tham số) đạt `0.3832 ± 0.0303` và `slim_all_cnnshared` (`2.24M` tham số) đạt `0.3987 ± 0.0140`. Khoảng chênh lệch `0.0155` nằm trong biên dao động giữa các fold với cỡ mẫu 3 fold, do đó trong bài báo hiện tượng này được mô tả thận trọng là **"biến thiên không đơn điệu theo số tham số, nhiều khả năng do nhiễu giữa các fold"**, không diễn giải là hiệu ứng "hồi phục mạnh".

---

## 2. Thiết kế Khối Chạy 10 Fold trên `val_select` (`eval_test = False`)

Chạy đủ **10 fold (`0..9`)** trên cả **Split B** và **Split A** (`mssf_slim/run_10fold.py`) cho **8 cấu hình**:

| STT | Tên cấu hình (`exp_name`) | Số view (`Drug × SE`) | Số tham số (`TOTAL`) | Vai trò Khoa học trong Bài báo |
|---:|---|:---:|---:|---|
| 1 | **`slim_nodec`** | `11 × 5` | **`6,803,109`** | **Mô hình cuối đề xuất (`Final Model`)**: `vdrop (0.25) + C2 + C5 + CE`, bỏ BVI, bỏ 2 decoder (`EN-con`, `EN-add`). |
| 2 | `cand_A` | `11 × 5` | `10,715,929` | **Đối chứng có decoder**: giữ nguyên mọi thành phần của `slim_nodec` nhưng bật 2 decoder (`1e-4`), tách biệt chính xác tác dụng 10-fold của việc bỏ decoder. |
| 3 | `slim_all_cnnshared` | `11 × 5` | `2,237,621` | **Biến thể nén tối đa (`-79.12%`)** trong bảng ablation nén tham số. |
| 4 | `MSSF_v2_selfcorr` | `11 × 4` | `10,097,399` | **Baseline nội bộ gốc** dưới Protocol v2 chống rò rỉ (`BVI=True`, `decoders=True`, không `vdrop`/`C2`/`C5`). |
| 5 | `slim_nodec_loo_no_vdrop` | `11 × 5` | `6,803,109` | **LOO trên `slim_nodec`**: bỏ `vdrop` (`p_cold = 0.0`), giữ `C2 + C5 + nodec`. |
| 6 | `slim_nodec_loo_no_C2` | `11 × 4` | `6,415,397` | **LOO trên `slim_nodec`**: bỏ view MedDRA `C2` (`c2_se_mode = 'original'`), giữ `vdrop + C5 + nodec`. |
| 7 | `slim_nodec_loo_no_C5` | `11 × 5` | `6,803,109` | **LOO trên `slim_nodec`**: bỏ nhãn mềm phổ đa view `C5` (`c5_mode = 'none'`), giữ `vdrop + C2 + nodec`. |
| 8 | `slim_nodec_loo_no_all` | `11 × 4` | `6,415,397` | **LOO trên `slim_nodec`**: bỏ đồng thời cả ba (`vdrop=0.0`, `C2='original'`, `C5='none'`) trên nền tất định không decoder. |

---

## 3. Quy tắc Báo cáo `HSTrans` và Thảo luận trong Bài báo

1. **Giữ `HSTrans` / `HSTrans_thr` ở 3 fold (`0–2`)**:
   - Trong các bảng chính 10 fold, **tuyệt đối không gộp số liệu 3 fold của `HSTrans` vào chung cột 10 fold**.
   - Báo cáo `HSTrans` ở dòng/khối tách riêng kèm chú thích rõ: *"3 fold (`0–2`) do chi phí huấn luyện ~7.900 giây/fold (`7,933.2 ± 362.3 s/fold`)"*.
2. **Điểm thảo luận bắt buộc trong phần Discussion**:
   - Trên Split B (`val_select`, Folds `0–2`), `HSTrans` bộc lộ hiện tượng quá khớp nghiêm trọng và sai số thứ tự cao bất thường so với mọi mô hình khác: **`Gap(eval) = +0.6743 ± 0.0225`** (`+0.5945 ± 0.0153` với `HSTrans_thr`) và **`MAE = 1.3943 ± 0.0648`** (trong khi toàn bộ các baseline khác có `MAE` khoảng `0.58–0.95` và `slim_nodec` đạt `MAE = 0.5782`).

---

## 4. Bảng Đối chiếu Cùng Tập Fold `0–2` (`val_select`, Trong Khi Chờ 10 Fold Mô hình Chính)

> **Quy tắc nhất quán tập fold**: Mọi phép đối chiếu dưới đây được tính **chính xác trên cùng tập Folds `0, 1, 2` ($K = 3$)** từ file `.npy` đã lưu của cả mô hình đề xuất và 6 baseline ngoại.

### 4.1. Split B (Drug Cold-Start) — Cùng Tập Fold `0–2` (`val_select`)

| Nhóm | Mô hình | Tập Fold | Macro-F1 | Weighted-F1 | Accuracy | QWK | MAE | Gap (`tr - val`) | Time/Fold (s) | Paired $\Delta\text{Macro-F1}$ vs `slim_nodec` |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Mô hình cuối** | **`slim_nodec`** | `0–2` (3) | **`0.4223 ± 0.0291`** | `0.5116 ± 0.0138` | `0.5177 ± 0.0104` | **`0.4941 ± 0.0468`** | `0.5782 ± 0.0286` | `+0.3284 ± 0.1139` | **`74.2 ± 0.9`** | `+0.0000 ± 0.0000` |
| Đối chứng decoder | `cand_A` | `0–2` (3) | `0.4123 ± 0.0273` | **`0.5135 ± 0.0181`** | `0.5155 ± 0.0247` | `0.4834 ± 0.0587` | `0.5894 ± 0.0548` | `+0.2695 ± 0.0494` | `85.2 ± 0.5` | `-0.0100 ± 0.0056` |
| Nén tối đa (`-79%`) | `slim_all_cnnshared` | `0–2` (3) | `0.3987 ± 0.0140` | `0.5061 ± 0.0144` | **`0.5306 ± 0.0071`** | `0.4659 ± 0.0474` | **`0.5600 ± 0.0211`** | `+0.2971 ± 0.0323` | `84.4 ± 0.8` | `-0.0236 ± 0.0214` |
| Baseline nội bộ | `MSSF_v2_selfcorr` | `0–2` (3) | `0.3593 ± 0.0095` | `0.4757 ± 0.0088` | `0.5034 ± 0.0321` | `0.4110 ± 0.0054` | `0.6213 ± 0.0764` | `+0.3041 ± 0.0396` | `101.1 ± 0.4` | `-0.0630 ± 0.0221` |
| Baseline ngoại (`_thr`) | `MGPred_thr` | `0–2` (3) | `0.3382 ± 0.0192` | `0.4120 ± 0.0004` | `0.4064 ± 0.0010` | `0.3552 ± 0.0203` | `0.7562 ± 0.0156` | — | `402.4 ± 12.6` | `-0.0841 ± 0.0231` |
| Baseline ngoại (`_thr`) | `MVSL-DSF_selfcorr_thr` | `0–2` (3) | `0.3298 ± 0.0291` | `0.4200 ± 0.0084` | `0.4122 ± 0.0116` | `0.3938 ± 0.0300` | `0.7385 ± 0.0337` | — | `2061.7 ± 38.8` | `-0.0925 ± 0.0107` |
| Baseline ngoại (`_thr`) | `SDPred_selfcorr_thr` | `0–2` (3) | `0.3254 ± 0.0070` | `0.4441 ± 0.0107` | `0.4460 ± 0.0140` | `0.3996 ± 0.0595` | `0.6745 ± 0.0405` | — | `1700.3 ± 1669.1` | `-0.0969 ± 0.0222` |
| Baseline ngoại (`_thr`) | `MVSL-DSF_thr` | `0–2` (3) | `0.3242 ± 0.0232` | `0.4328 ± 0.0133` | `0.4310 ± 0.0199` | `0.3858 ± 0.0345` | `0.7269 ± 0.0608` | — | `2078.3 ± 36.7` | `-0.0981 ± 0.0094` |
| Baseline ngoại (`_thr`) | `A3Net_thr` | `0–2` (3) | `0.3218 ± 0.0098` | `0.4469 ± 0.0212` | `0.4481 ± 0.0276` | `0.3361 ± 0.0343` | `0.7226 ± 0.0465` | — | `~737.7` | `-0.1005 ± 0.0352` |
| Baseline ngoại (`_thr`) | `SDPred_thr` | `0–2` (3) | `0.3194 ± 0.0209` | `0.4464 ± 0.0139` | `0.4509 ± 0.0143` | `0.3980 ± 0.0728` | `0.6767 ± 0.0400` | — | `1688.7 ± 1662.0` | `-0.1029 ± 0.0113` |
| Baseline ngoại (`_thr`) | `MGPred_selfcorr_thr` | `0–2` (3) | `0.3146 ± 0.0524` | `0.4147 ± 0.0133` | `0.4268 ± 0.0160` | `0.2579 ± 0.1374` | `0.7339 ± 0.0238` | — | `397.9 ± 2.7` | `-0.1077 ± 0.0308` |
| Baseline ngoại (`_thr`) | `DSGAT_thr` | `0–2` (3) | `0.3086 ± 0.0128` | `0.4315 ± 0.0237` | `0.4262 ± 0.0261` | `0.3258 ± 0.0054` | `0.7702 ± 0.0330` | — | `240.6 ± 0.4` | `-0.1137 ± 0.0312` |
| Baseline ngoại (`_thr`) | `HSTrans_thr` *(3 fold)* | `0–2` (3) | `0.3027 ± 0.0079` | `0.4334 ± 0.0178` | `0.4312 ± 0.0192` | `0.2628 ± 0.0192` | `0.7928 ± 0.0098` | `+0.5945 ± 0.0153` | `7933.2 ± 362.3` | `-0.1196 ± 0.0301` |
| Baseline ngoại (gốc) | `HSTrans` *(3 fold)* | `0–2` (3) | `0.2229 ± 0.0269` | `0.3455 ± 0.0311` | `0.2950 ± 0.0250` | `0.1807 ± 0.0076` | **`1.3943 ± 0.0648`** | **`+0.6743 ± 0.0225`** | `7933.2 ± 362.3` | `-0.1994 ± 0.0268` |

### 4.2. Split A (Pair Warm-Start) — Cùng Tập Fold `0–2` (`val_select`)

| Nhóm | Mô hình | Tập Fold | Macro-F1 | Weighted-F1 | Accuracy | QWK | MAE | Paired $\Delta\text{Macro-F1}$ vs `slim_nodec` |
|---|---|:---:|---:|---:|---:|---:|---:|---:|
| **Mô hình cuối** | **`slim_nodec`** | `0–2` (3) | **`0.6475 ± 0.0082`** | **`0.6890 ± 0.0078`** | **`0.6895 ± 0.0072`** | `0.7301 ± 0.0045` | **`0.3639 ± 0.0080`** | `+0.0000 ± 0.0000` |
| Nén tối đa (`-79%`) | `slim_all_cnnshared` | `0–2` (3) | `0.6456 ± 0.0079` | `0.6830 ± 0.0109` | `0.6834 ± 0.0119` | `0.7239 ± 0.0032` | `0.3688 ± 0.0086` | `-0.0020 ± 0.0044` |
| Đối chứng decoder | `cand_A` | `0–2` (3) | `0.6435 ± 0.0050` | `0.6800 ± 0.0055` | `0.6782 ± 0.0071` | `0.7200 ± 0.0060` | `0.3787 ± 0.0071` | `-0.0040 ± 0.0050` |
| Baseline nội bộ | `MSSF_v2_selfcorr` | `0–2` (3) | `0.6421 ± 0.0054` | `0.6804 ± 0.0064` | `0.6818 ± 0.0079` | `0.7151 ± 0.0067` | `0.3768 ± 0.0173` | `-0.0054 ± 0.0030` |
| Baseline ngoại (`_thr`) | `SDPred_selfcorr_thr` | `0–2` (3) | `0.5886 ± 0.0168` | `0.6446 ± 0.0105` | `0.6449 ± 0.0119` | `0.7353 ± 0.0044` | `0.3872 ± 0.0099` | `-0.0590 ± 0.0088` |
| Baseline ngoại (`_thr`) | `SDPred_thr` | `0–2` (3) | `0.5784 ± 0.0185` | `0.6409 ± 0.0110` | `0.6406 ± 0.0111` | **`0.7371 ± 0.0029`** | `0.3901 ± 0.0101` | `-0.0691 ± 0.0105` |
| Baseline ngoại (`_thr`) | `MVSL-DSF_thr` | `0–2` (3) | `0.5695 ± 0.0124` | `0.6204 ± 0.0085` | `0.6188 ± 0.0089` | `0.7124 ± 0.0053` | `0.4180 ± 0.0086` | `-0.0780 ± 0.0044` |
| Baseline ngoại (`_thr`) | `MVSL-DSF_selfcorr_thr` | `0–2` (3) | `0.5541 ± 0.0123` | `0.6162 ± 0.0084` | `0.6163 ± 0.0089` | `0.7022 ± 0.0042` | `0.4200 ± 0.0067` | `-0.0934 ± 0.0081` |
| Baseline ngoại (`_thr`) | `MGPred_selfcorr_thr` | `0–2` (3) | `0.4746 ± 0.0366` | `0.5600 ± 0.0184` | `0.5684 ± 0.0131` | `0.6227 ± 0.0324` | `0.4777 ± 0.0142` | `-0.1729 ± 0.0382` |
| Baseline ngoại (`_thr`) | `MGPred_thr` | `0–2` (3) | `0.4621 ± 0.0370` | `0.5454 ± 0.0204` | `0.5453 ± 0.0176` | `0.6232 ± 0.0311` | `0.5106 ± 0.0298` | `-0.1854 ± 0.0291` |
| Baseline ngoại (`_thr`) | `A3Net_thr` | `0–2` (3) | `0.4112 ± 0.0057` | `0.4871 ± 0.0102` | `0.4761 ± 0.0104` | `0.4673 ± 0.0082` | `0.6855 ± 0.0141` | `-0.2363 ± 0.0127` |
| Baseline ngoại (`_thr`) | `DSGAT_thr` | `0–2` (3) | `0.3824 ± 0.0042` | `0.4685 ± 0.0035` | `0.4570 ± 0.0034` | `0.4067 ± 0.0155` | `0.7319 ± 0.0210` | `-0.2651 ± 0.0040` |

---

## 5. Bảng Tham chiếu Độc lập 10 Fold (`0–9`) của 5 Baseline Ngoại Đã Hoàn tất (`val_select`)
*(Sẽ ghép đối chiếu trực tiếp 1-đối-1 ngay khi hoàn tất 10 fold của `slim_nodec` và các biến thể; `HSTrans` tách riêng ở chú thích 3 fold)*

| Baseline Ngoại | Tập Fold | Split B Macro-F1 (10 fold) | Split B QWK | Split B MAE | Split A Macro-F1 (10 fold) | Split A QWK | Split A MAE |
|---|:---:|---:|---:|---:|---:|---:|---:|
| `MGPred_selfcorr_thr` | `0–9` (10) | **`0.3525 ± 0.0515`** | `0.3823 ± 0.1328` | `0.6500 ± 0.0827` | `0.4747 ± 0.0276` | `0.6203 ± 0.0259` | `0.4898 ± 0.0266` |
| `SDPred_selfcorr_thr` | `0–9` (10) | `0.3434 ± 0.0238` | **`0.4551 ± 0.0619`** | `0.6229 ± 0.0531` | **`0.5849 ± 0.0115`** | **`0.7322 ± 0.0087`** | **`0.3859 ± 0.0077`** |
| `MVSL-DSF_selfcorr_thr` | `0–9` (10) | `0.3420 ± 0.0273` | `0.4272 ± 0.0484` | `0.6768 ± 0.0835` | `0.5572 ± 0.0160` | `0.7046 ± 0.0088` | `0.4199 ± 0.0125` |
| `MVSL-DSF_thr` | `0–9` (10) | `0.3400 ± 0.0279` | `0.4268 ± 0.0510` | `0.6786 ± 0.0859` | `0.5637 ± 0.0160` | `0.7108 ± 0.0110` | `0.4168 ± 0.0108` |
| `SDPred_thr` | `0–9` (10) | `0.3373 ± 0.0273` | `0.4533 ± 0.0658` | **`0.6207 ± 0.0548`** | `0.5741 ± 0.0112` | `0.7316 ± 0.0068` | `0.3900 ± 0.0090` |
| `MGPred_thr` | `0–9` (10) | `0.3358 ± 0.0262` | `0.3627 ± 0.0437` | `0.6927 ± 0.0566` | `0.4613 ± 0.0267` | `0.6150 ± 0.0242` | `0.5154 ± 0.0296` |
| `DSGAT_thr` | `0–9` (10) | `0.3354 ± 0.0300` | `0.3866 ± 0.0555` | `0.7023 ± 0.0634` | `0.3811 ± 0.0052` | `0.4134 ± 0.0189` | `0.7232 ± 0.0204` |
| `A3Net_thr` | `0–9` (10) | `0.3311 ± 0.0195` | `0.3724 ± 0.0593` | `0.6966 ± 0.0822` | `0.4031 ± 0.0083` | `0.4583 ± 0.0113` | `0.6944 ± 0.0182` |

> **Chú thích riêng `HSTrans` (KHÔNG gộp vào cột 10 fold)**: `HSTrans` / `HSTrans_thr` chỉ chạy **3 fold (`0–2`)** trên Split B do chi phí tính toán `~7.900 giây/fold` (`7,933.2 ± 362.3 s/fold`), đạt `0.3027 ± 0.0079` Macro-F1 (`HSTrans_thr`) và `0.2229 ± 0.0269` (`HSTrans` gốc, `Gap(eval) = +0.6743`, `MAE = 1.3943`).
