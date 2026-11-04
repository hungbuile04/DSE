# Báo Cáo Thực Nghiệm Giai Đoạn 2 (`MSSF_plus` — Protocol v2)

## 1. Kiểm tra Toàn vẹn Dữ liệu & Kiến trúc Nguồn Sự Thật Duy Nhất (Mục 1–3 trước Giai đoạn 2)

1. **Toàn vẹn `shared_data/` trước và sau Giai đoạn 2**:
   - Đã chạy `MSSF_plus/audit/verify_checksums.py --check`: **90/90 file dữ liệu, chia tập và code gốc nguyên vẹn 100% (`[PASS]`)**.
   - Tập `test` hoàn toàn khóa (`eval_test=False`); mọi lựa chọn và báo cáo dưới đây đều tính trên `val_select` (và hiệu chỉnh Conformal C4 từ `val_calib` sang `val_select`).
2. **Nguồn sự thật duy nhất về kiến trúc (`MSSF_plus/ARCHITECTURE_DUMP.md`)**:
   - Đã tạo script `MSSF_plus/scripts/dump_architecture.py` khởi tạo trực tiếp các mô hình PyTorch (`MSSF_v2`, `MSSF_v2_selfcorr`, `C3_ce_unweighted`, `C3_ce_unweighted_selfcorr`), trích xuất cấu trúc `print(model)`, bảng tham số từng module (**`10,097,399`** tham số cho `MSSF_v2` / `MSSF_v2_selfcorr`; **`10,754,421`** tham số cho `C3_ce_unweighted` / `C3_ce_unweighted_selfcorr`), toàn bộ `ExperimentConfig` dạng JSON, và cấu hình thực tế của `ProfileViewRegressor` (`6759 -> 256 -> 256 -> 1500`, **`2,182,876`** tham số, `LayerNorm(256)`, `SiLU`, `Dropout(0.2)`, `Sigmoid`, loss `mse + 0.1 * cos`, `epochs=80`, `batch_size=64`) cùng `ConditionalDenoiserMLP` (`8259+64=8323 -> 256 -> 256 -> 1500`, **`4,538,084`** tham số, `LayerNorm(256)`, `SiLU`, `Dropout(0.1)`, `MSELoss`, `epochs=100`, `T=25`, `B=10`).
   - Đã đồng bộ hóa `MSSF_plus/ARCHITECTURE_SNAPSHOT_V2.md` và `MSSF_plus/ARCHITECTURE_REPORT.md` khớp chính xác với `ARCHITECTURE_DUMP.md` và xóa bỏ hoàn toàn câu *"F1 lớp Very Rare tăng 3,3 lần"* cùng các số chép nhầm trước đây.
3. **Kiểm tra Sanity (`MSSF_v2` Split B Fold 0 vs Fold 1 ở lần chạy sơ bộ trước đó)**:
   - Đã xác minh hai file kết quả sơ bộ trên đĩa là hai lần chạy hoàn toàn độc lập:
     - **Fold 0**: `n_valsel = 3,323` mẫu, `best_epoch = 13`, `MacroF1 = 0.3524928184`, `SHA256(valsel_probs) = 9ca1a9c5ca176a3b...`
     - **Fold 1**: `n_valsel = 2,918` mẫu, `best_epoch = 7`, `MacroF1 = 0.3525968848`, `SHA256(valsel_probs) = ea69610011647151...`
   - Trong lần chạy lại đồng nhất trên GPU Colab ở Giai đoạn 2 (bên dưới), `MSSF_v2` Split B đạt `0.3430` (Fold 0, `best_epoch=22`), `0.3323` (Fold 1, `best_epoch=32`), `0.4045` (Fold 2, `best_epoch=24`), trung bình `0.3599 ± 0.0318`.
4. **Unit Test Sửa Rò Rỉ Nhãn Cặp Đang Train (`test_profile_self_correction.py`)**:
   - Đã chạy `python -m unittest MSSF_plus/tests/test_profile_self_correction.py -v` trên Colab: **3/3 test PASS (`2.546s`)**:
     - Công thức vector hóa ma trận Gram khớp tuyệt đối với `cosine_similarity` tính lại từ `Y_train` khi đặt ô `(d, s) = 0` (sai số cực đại `< 1e-6`).
     - Trường hợp biên mẫu số bằng 0 (thuốc hoặc tác dụng phụ chỉ có đúng 1 tương tác trong `Y_train`) trả về đúng `0.0` và không sinh `NaN/Inf`.
     - Mọi cặp `val_select`, `val_calib`, `test` có `Y_train[d, s] = 0` giữ nguyên bất biến tuyệt đối (`max_diff = 0.0`).

---

## 2. Kết Quả Thực Nghiệm Giai Đoạn 2 (Sinh tự động từ `summarize_results.py` trên các file `.npy`)

### 2.1. Split B — Cold-Start Drug (`val_select`, Folds 0–2 cho họ `MSSF`, Fold 0 cho `SDPred` & `MVSL-DSF`)

#### Bảng 1B: Các chỉ số Phân loại & Thứ tự Tổng thể (`Mean ± Std` trên `val_select`)
| Method | Folds | Macro-F1 | Weighted-F1 | Accuracy | QWK | MAE | Gap (tr - val) | Time/Fold (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `C3_ce_unweighted` | 3 | **0.4266 ± 0.0275** | 0.5133 ± 0.0171 | 0.5162 ± 0.0222 | **0.5173 ± 0.0390** | 0.5742 ± 0.0329 | +0.2028 ± 0.0826 | 139.3 ± 1.1 |
| `C3_ce_unweighted_selfcorr` | 3 | 0.4169 ± 0.0259 | **0.5166 ± 0.0191** | **0.5236 ± 0.0283** | 0.5007 ± 0.0439 | **0.5586 ± 0.0510** | +0.2933 ± 0.0507 | 144.5 ± 1.1 |
| `MSSF_v2` | 3 | 0.3599 ± 0.0318 | 0.4700 ± 0.0302 | 0.4880 ± 0.0327 | 0.4203 ± 0.0501 | 0.6349 ± 0.0641 | +0.3431 ± 0.0608 | 98.5 ± 3.3 |
| `MSSF_v2_selfcorr` | 3 | 0.3593 ± 0.0095 | 0.4757 ± 0.0088 | 0.5034 ± 0.0321 | 0.4110 ± 0.0054 | 0.6213 ± 0.0764 | +0.3214 ± 0.0775 | 101.1 ± 0.4 |
| `SDPred` | 1 | 0.2885 ± 0.0000 | 0.4460 ± 0.0000 | 0.4719 ± 0.0000 | 0.4000 ± 0.0000 | 0.6037 ± 0.0000 | -0.0299 ± 0.0000 | 4039.2 ± 0.0 |
| `SDPred_rmseSel` | 1 | 0.2885 ± 0.0000 | 0.4460 ± 0.0000 | 0.4719 ± 0.0000 | 0.4000 ± 0.0000 | 0.6037 ± 0.0000 | -0.0299 ± 0.0000 | 4039.2 ± 0.0 |
| `SDPred_selfcorr` | 1 | 0.2810 ± 0.0000 | 0.4330 ± 0.0000 | 0.4580 ± 0.0000 | 0.3847 ± 0.0000 | 0.6217 ± 0.0000 | -0.0265 ± 0.0000 | 4060.8 ± 0.0 |
| `SDPred_selfcorr_rmseSel` | 1 | 0.2563 ± 0.0000 | 0.4320 ± 0.0000 | 0.4896 ± 0.0000 | 0.3599 ± 0.0000 | 0.5886 ± 0.0000 | -0.0017 ± 0.0000 | 4060.8 ± 0.0 |
| `MVSL-DSF` | 1 | 0.2811 ± 0.0000 | 0.4434 ± 0.0000 | 0.4667 ± 0.0000 | 0.3625 ± 0.0000 | 0.6437 ± 0.0000 | +0.0070 ± 0.0000 | 2027.7 ± 0.0 |
| `MVSL-DSF_rmseSel` | 1 | 0.2146 ± 0.0000 | 0.4133 ± 0.0000 | 0.4854 ± 0.0000 | 0.3246 ± 0.0000 | 0.5865 ± 0.0000 | +0.0735 ± 0.0000 | 2027.7 ± 0.0 |
| `MVSL-DSF_selfcorr` | 1 | 0.2779 ± 0.0000 | 0.4447 ± 0.0000 | 0.4704 ± 0.0000 | 0.3564 ± 0.0000 | 0.6362 ± 0.0000 | +0.0049 ± 0.0000 | 2006.9 ± 0.0 |
| `MVSL-DSF_selfcorr_rmseSel` | 1 | 0.2109 ± 0.0000 | 0.4084 ± 0.0000 | 0.4854 ± 0.0000 | 0.3089 ± 0.0000 | 0.5931 ± 0.0000 | +0.0719 ± 0.0000 | 2006.9 ± 0.0 |

#### Bảng 2B: F1 Từng Lớp (`c0..c4`, `Mean ± Std` trên `val_select`, Split B)
| Method | F1_c0 (Very Rare) | F1_c1 (Rare) | F1_c2 (Infrequent) | F1_c3 (Frequent) | F1_c4 (Very Freq) |
|---|---:|---:|---:|---:|---:|
| `C3_ce_unweighted` | **0.3240 ± 0.0497** | 0.3446 ± 0.1177 | 0.4478 ± 0.0481 | 0.6194 ± 0.0234 | **0.3973 ± 0.0420** |
| `C3_ce_unweighted_selfcorr` | 0.2981 ± 0.0572 | 0.3529 ± 0.0928 | **0.4796 ± 0.0582** | 0.6284 ± 0.0272 | 0.3256 ± 0.0681 |
| `MSSF_v2` | 0.2242 ± 0.0923 | 0.3160 ± 0.0869 | 0.3237 ± 0.1068 | 0.6290 ± 0.0279 | 0.3066 ± 0.0332 |
| `MSSF_v2_selfcorr` | 0.2379 ± 0.1131 | 0.2972 ± 0.1091 | 0.3608 ± 0.0471 | 0.6381 ± 0.0479 | 0.2623 ± 0.0433 |
| `SDPred` | 0.0000 ± 0.0000 | 0.3812 ± 0.0000 | 0.4840 ± 0.0000 | 0.5539 ± 0.0000 | 0.0236 ± 0.0000 |
| `SDPred_rmseSel` | 0.0000 ± 0.0000 | 0.3812 ± 0.0000 | 0.4840 ± 0.0000 | 0.5539 ± 0.0000 | 0.0236 ± 0.0000 |
| `SDPred_selfcorr` | 0.0000 ± 0.0000 | 0.3888 ± 0.0000 | 0.4706 ± 0.0000 | 0.5337 ± 0.0000 | 0.0121 ± 0.0000 |
| `SDPred_selfcorr_rmseSel` | 0.0000 ± 0.0000 | 0.0213 ± 0.0000 | 0.4215 ± 0.0000 | 0.6385 ± 0.0000 | 0.2000 ± 0.0000 |
| `MVSL-DSF` | 0.0000 ± 0.0000 | 0.3817 ± 0.0000 | 0.4412 ± 0.0000 | 0.5824 ± 0.0000 | 0.0000 ± 0.0000 |
| `MVSL-DSF_rmseSel` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.4340 ± 0.0000 | 0.6388 ± 0.0000 | 0.0000 ± 0.0000 |
| `MVSL-DSF_selfcorr` | 0.0000 ± 0.0000 | 0.3511 ± 0.0000 | 0.4481 ± 0.0000 | 0.5901 ± 0.0000 | 0.0000 ± 0.0000 |
| `MVSL-DSF_selfcorr_rmseSel` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.4115 ± 0.0000 | 0.6432 ± 0.0000 | 0.0000 ± 0.0000 |

#### Bảng 3B: Chi tiết Từng Fold (`val_select`, Split B)
| Method | Fold | Macro-F1 | Weighted-F1 | Accuracy | QWK | MAE | F1_c0 | F1_c1 | F1_c2 | F1_c3 | F1_c4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `C3_ce_unweighted` | 0 | 0.4426 | 0.5227 | 0.5296 | 0.5561 | 0.5588 | 0.2982 | 0.5062 | 0.4233 | 0.6463 | 0.3390 |
| `C3_ce_unweighted` | 1 | 0.3879 | 0.4894 | 0.4849 | 0.4639 | 0.6199 | 0.2803 | 0.2292 | 0.4051 | 0.5892 | 0.4359 |
| `C3_ce_unweighted` | 2 | 0.4493 | 0.5279 | 0.5340 | 0.5319 | 0.5439 | 0.3934 | 0.2984 | 0.5150 | 0.6227 | 0.4171 |
| `C3_ce_unweighted_selfcorr` | 0 | 0.4285 | 0.5274 | 0.5296 | 0.5355 | 0.5390 | 0.2588 | 0.4563 | 0.5167 | 0.6217 | 0.2889 |
| `C3_ce_unweighted_selfcorr` | 1 | 0.3810 | 0.4897 | 0.4863 | 0.4387 | 0.6285 | 0.2564 | 0.2312 | 0.3974 | 0.5990 | 0.4211 |
| `C3_ce_unweighted_selfcorr` | 2 | 0.4413 | 0.5326 | 0.5547 | 0.5277 | 0.5082 | 0.3789 | 0.3713 | 0.5247 | 0.6645 | 0.2669 |
| `MSSF_v2` | 0 | 0.3430 | 0.4558 | 0.4574 | 0.4512 | 0.6828 | 0.1034 | 0.3865 | 0.3698 | 0.5951 | 0.2600 |
| `MSSF_v2` | 1 | 0.3323 | 0.4423 | 0.4733 | 0.3496 | 0.6775 | 0.3277 | 0.1935 | 0.1761 | 0.6285 | 0.3354 |
| `MSSF_v2` | 2 | 0.4045 | 0.5120 | 0.5334 | 0.4601 | 0.5442 | 0.2414 | 0.3680 | 0.4252 | 0.6635 | 0.3243 |
| `MSSF_v2_selfcorr` | 0 | 0.3536 | 0.4652 | 0.4580 | 0.4141 | 0.7262 | 0.0784 | 0.4183 | 0.4159 | 0.5705 | 0.2848 |
| `MSSF_v2_selfcorr` | 1 | 0.3516 | 0.4866 | 0.5274 | 0.4034 | 0.5918 | 0.3276 | 0.1538 | 0.3009 | 0.6754 | 0.3003 |
| `MSSF_v2_selfcorr` | 2 | 0.3726 | 0.4752 | 0.5247 | 0.4156 | 0.5460 | 0.3077 | 0.3194 | 0.3658 | 0.6684 | 0.2017 |
| `SDPred` | 0 | 0.2885 | 0.4460 | 0.4719 | 0.4000 | 0.6037 | 0.0000 | 0.3812 | 0.4840 | 0.5539 | 0.0236 |
| `SDPred_rmseSel` | 0 | 0.2885 | 0.4460 | 0.4719 | 0.4000 | 0.6037 | 0.0000 | 0.3812 | 0.4840 | 0.5539 | 0.0236 |
| `SDPred_selfcorr` | 0 | 0.2810 | 0.4330 | 0.4580 | 0.3847 | 0.6217 | 0.0000 | 0.3888 | 0.4706 | 0.5337 | 0.0121 |
| `SDPred_selfcorr_rmseSel` | 0 | 0.2563 | 0.4320 | 0.4896 | 0.3599 | 0.5886 | 0.0000 | 0.0213 | 0.4215 | 0.6385 | 0.2000 |
| `MVSL-DSF` | 0 | 0.2811 | 0.4434 | 0.4667 | 0.3625 | 0.6437 | 0.0000 | 0.3817 | 0.4412 | 0.5824 | 0.0000 |
| `MVSL-DSF_rmseSel` | 0 | 0.2146 | 0.4133 | 0.4854 | 0.3246 | 0.5865 | 0.0000 | 0.0000 | 0.4340 | 0.6388 | 0.0000 |
| `MVSL-DSF_selfcorr` | 0 | 0.2779 | 0.4447 | 0.4704 | 0.3564 | 0.6362 | 0.0000 | 0.3511 | 0.4481 | 0.5901 | 0.0000 |
| `MVSL-DSF_selfcorr_rmseSel` | 0 | 0.2109 | 0.4084 | 0.4854 | 0.3089 | 0.5931 | 0.0000 | 0.0000 | 0.4115 | 0.6432 | 0.0000 |

---

### 2.2. Split A — Warm-Start Pair (`val_select`, Folds 0–2)

#### Bảng 1A: Các chỉ số Phân loại & Thứ tự Tổng thể (`Mean ± Std` trên `val_select`)
| Method | Folds | Macro-F1 | Weighted-F1 | Accuracy | QWK | MAE | Gap (tr - val) | Time/Fold (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `C3_ce_unweighted` | 3 | 0.6454 ± 0.0099 | 0.6809 ± 0.0059 | 0.6835 ± 0.0109 | 0.7229 ± 0.0020 | 0.3620 ± 0.0076 | +0.1332 ± 0.0090 | 146.7 ± 0.4 |
| `C3_ce_unweighted_selfcorr` | 3 | 0.6479 ± 0.0082 | 0.6817 ± 0.0082 | 0.6810 ± 0.0108 | **0.7293 ± 0.0092** | 0.3727 ± 0.0168 | +0.1333 ± 0.0122 | 152.0 ± 0.3 |
| `MSSF_v2` | 3 | **0.6506 ± 0.0140** | **0.6917 ± 0.0067** | **0.6939 ± 0.0044** | 0.7228 ± 0.0080 | **0.3556 ± 0.0028** | +0.1408 ± 0.0119 | 96.7 ± 0.1 |
| `MSSF_v2_selfcorr` | 3 | 0.6421 ± 0.0054 | 0.6804 ± 0.0064 | 0.6818 ± 0.0079 | 0.7151 ± 0.0067 | 0.3768 ± 0.0173 | +0.1445 ± 0.0055 | 102.3 ± 0.1 |

#### Bảng 2A: F1 Từng Lớp (`c0..c4`, `Mean ± Std` trên `val_select`, Split A)
| Method | F1_c0 (Very Rare) | F1_c1 (Rare) | F1_c2 (Infrequent) | F1_c3 (Frequent) | F1_c4 (Very Freq) |
|---|---:|---:|---:|---:|---:|
| `C3_ce_unweighted` | **0.6493 ± 0.0329** | 0.6443 ± 0.0124 | 0.6391 ± 0.0218 | 0.7467 ± 0.0228 | 0.5475 ± 0.0294 |
| `C3_ce_unweighted_selfcorr` | 0.6487 ± 0.0386 | **0.6502 ± 0.0117** | 0.6344 ± 0.0120 | 0.7468 ± 0.0166 | 0.5597 ± 0.0139 |
| `MSSF_v2` | 0.6429 ± 0.0408 | 0.6382 ± 0.0214 | **0.6478 ± 0.0221** | **0.7632 ± 0.0043** | **0.5606 ± 0.0209** |
| `MSSF_v2_selfcorr` | 0.6432 ± 0.0334 | 0.6368 ± 0.0252 | 0.6170 ± 0.0173 | 0.7586 ± 0.0109 | 0.5551 ± 0.0089 |

#### Bảng 3A: Chi tiết Từng Fold (`val_select`, Split A)
| Method | Fold | Macro-F1 | Weighted-F1 | Accuracy | QWK | MAE | F1_c0 | F1_c1 | F1_c2 | F1_c3 | F1_c4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `C3_ce_unweighted` | 0 | 0.6555 | 0.6828 | 0.6812 | 0.7252 | 0.3614 | 0.6569 | 0.6469 | 0.6603 | 0.7290 | 0.5843 |
| `C3_ce_unweighted` | 1 | 0.6487 | 0.6869 | 0.6979 | 0.7204 | 0.3530 | 0.6852 | 0.6580 | 0.6090 | 0.7789 | 0.5125 |
| `C3_ce_unweighted` | 2 | 0.6319 | 0.6730 | 0.6716 | 0.7231 | 0.3716 | 0.6058 | 0.6280 | 0.6480 | 0.7322 | 0.5457 |
| `C3_ce_unweighted_selfcorr` | 0 | 0.6485 | 0.6932 | 0.6961 | 0.7323 | 0.3491 | 0.6465 | 0.6349 | 0.6506 | 0.7702 | 0.5404 |
| `C3_ce_unweighted_selfcorr` | 1 | 0.6577 | 0.6764 | 0.6743 | 0.7388 | 0.3821 | 0.6971 | 0.6634 | 0.6221 | 0.7332 | 0.5727 |
| `C3_ce_unweighted_selfcorr` | 2 | 0.6377 | 0.6754 | 0.6725 | 0.7168 | 0.3869 | 0.6026 | 0.6523 | 0.6304 | 0.7369 | 0.5660 |
| `MSSF_v2` | 0 | 0.6550 | 0.6947 | 0.6937 | 0.7289 | 0.3551 | 0.6544 | 0.6082 | 0.6702 | 0.7576 | 0.5849 |
| `MSSF_v2` | 1 | 0.6650 | 0.6979 | 0.6994 | 0.7279 | 0.3524 | 0.6862 | 0.6562 | 0.6555 | 0.7642 | 0.5631 |
| `MSSF_v2` | 2 | 0.6316 | 0.6824 | 0.6886 | 0.7115 | 0.3593 | 0.5882 | 0.6503 | 0.6177 | 0.7679 | 0.5338 |
| `MSSF_v2_selfcorr` | 0 | 0.6450 | 0.6776 | 0.6794 | 0.7137 | 0.3809 | 0.6635 | 0.6371 | 0.5982 | 0.7589 | 0.5674 |
| `MSSF_v2_selfcorr` | 1 | 0.6468 | 0.6893 | 0.6925 | 0.7240 | 0.3539 | 0.6700 | 0.6057 | 0.6400 | 0.7718 | 0.5465 |
| `MSSF_v2_selfcorr` | 2 | 0.6345 | 0.6744 | 0.6737 | 0.7077 | 0.3956 | 0.5961 | 0.6675 | 0.6127 | 0.7452 | 0.5512 |

---

## 3. Phân Tích & Nhận Xét Khoa Học (Giai Đoạn 2)

1. **Ảnh hưởng của `profile_self_correction` trên Split B (Cold-start Drug)**:
   - **`MSSF_v2`**: Khi bật `profile_self_correction=True`, `Macro-F1` trung bình 3 fold gần như giữ nguyên (`0.3599 ± 0.0318` vs `0.3593 ± 0.0095`, chênh `-0.0006`), trong khi **Weighted-F1** đạt `0.4757` (so với `0.4700`), **Accuracy** đạt `0.5034` (so với `0.4880`), và **MAE** đạt `0.6213` (so với `0.6349`). Trên Fold 0 và Fold 1, `MSSF_v2_selfcorr` đạt `Macro-F1` `0.3536` (so với `0.3430`) và `0.3516` (so với `0.3323`), còn trên Fold 2 đạt `0.3726` (so với `0.4045`).
   - **`C3_ce_unweighted`**: Đạt hiệu năng vượt trội so với `MSSF_v2` trên Split B ở cả hai thiết lập tắt và bật `profile_self_correction`:
     - Khi tắt (`SelfCorr=False`): `Macro-F1 = 0.4266 ± 0.0275` (**`+0.0667`** so với `MSSF_v2`), `QWK = 0.5173 ± 0.0390` (**`+0.0970`**), `MAE = 0.5742 ± 0.0329` (**`-0.0607`**).
     - Khi bật (`SelfCorr=True`): `Macro-F1 = 0.4169 ± 0.0259` (**`+0.0576`** so với `MSSF_v2_selfcorr`), đồng thời đạt **Weighted-F1** `0.5166 ± 0.0191`, **Accuracy** `0.5236 ± 0.0283`, và **MAE** `0.5586 ± 0.0510`.
2. **Trên Split A (Warm-start Pair) — `MSSF_plus` không tốt hơn `MSSF_v2`**:
   - Ở Split A, mọi thuốc và tác dụng phụ trong `val_select` đều đã có nhiều tương tác quan sát được trong `Y_train` (`drug_avail = 1`, không có hiện tượng thiếu view `D9, D10` của cold-start drug). Đồng thời, đóng góp của một ô đơn lẻ `Y_train[d, s]` vào tích vô hướng chuẩn hóa (`D9, D10, S3, S4`) là rất nhỏ ($\sim \mathcal{O}(1/\|x_d\|_2^2)$).
   - Thực nghiệm cho thấy rõ trên Split A, **`MSSF_plus` (`C3_ce_unweighted`) không tốt hơn `MSSF_v2`** (hai mô hình tương đương nhau trong phạm vi nhiễu giữa các lần chạy):
     - Với `profile_self_correction = 0`: `MSSF_v2` đạt `Macro-F1 = 0.6506 ± 0.0140`, `QWK = 0.7228 ± 0.0080`, `MAE = 0.3556 ± 0.0028`; còn `C3_ce_unweighted` đạt `Macro-F1 = 0.6454 ± 0.0099`, `QWK = 0.7229 ± 0.0020`, `MAE = 0.3620 ± 0.0076`.
     - Với `profile_self_correction = 1`: `C3_ce_unweighted_selfcorr` đạt `Macro-F1 = 0.6479 ± 0.0082`, `QWK = 0.7293 ± 0.0092`, `MAE = 0.3727 ± 0.0168`; còn `MSSF_v2_selfcorr` đạt `Macro-F1 = 0.6421 ± 0.0054`, `QWK = 0.7151 ± 0.0067`, `MAE = 0.3768 ± 0.0173`.
3. **Ảnh hưởng của `profile_self_correction` trên hai Baseline đại diện (`SDPred` và `MVSL-DSF`, Fold 0, Split B)**:
   - Khi làm tròn dự đoán hồi quy (`np.round`), cả hai mô hình (`SDPred` dùng MSE loss, `MVSL-DSF` dùng BCE trên nhãn chia 5) đều bị sụp đổ ở hai lớp biên hiếm (`F1_c0 = 0.0000` cho cả hai; `F1_c4 = 0.0236` với `SDPred` và `0.0000` với `MVSL-DSF`), chỉ đạt `Macro-F1 = 0.2885` (`SDPred`) và `0.2811` (`MVSL-DSF`).
   - Khi bật `profile_self_correction=True`, `SDPred` giảm nhẹ `Macro-F1` từ `0.2885 -> 0.2810` (`-0.0075`), và `MVSL-DSF` giảm nhẹ từ `0.2811 -> 0.2779` (`-0.0032`).
   - Quyết định thiết kế cho bài báo: **bảng chính dùng `profile_self_correction = 1` cho mọi mô hình bị ảnh hưởng; phụ lục dùng thiết lập gốc (`profile_self_correction = 0`)**.

---

## 4. Kết Quả Bổ Sung Sau Giai Đoạn 2 (Mục 1–4)

### 4.1. Mục 1: Xác Minh Danh Sách View & Bảng Tham Số Từng Module Từ `ARCHITECTURE_DUMP.md`

#### Danh sách View Thuốc và View Phản Ứng Phụ thực tế trong đối tượng `MSSFPlusModel` đã khởi tạo
- **View Thuốc (`n_drug_views = 11` cho cả `MSSF_v2` và `C3_ce_unweighted`, mỗi view `750` chiều, tổng chiều nối `11 × 750 = 8,250`)**:
  1. `D1_stitch_one` (`750` chiều) — `Text_similarity_one.pkl`
  2. `D2_stitch_two` (`750` chiều) — `Text_similarity_two.pkl`
  3. `D3_stitch_three` (`750` chiều) — `Text_similarity_three.pkl`
  4. `D4_stitch_four` (`750` chiều) — `Text_similarity_four.pkl`
  5. `D5_stitch_five` (`750` chiều) — `Text_similarity_five.pkl`
  6. `D6_mol2vec_cos` (`750` chiều) — cosine similarity của `drug_mol.pkl`
  7. `D7_target_cos` (`750` chiều) — cosine similarity của `drug_target.pkl`
  8. `D8_fingerprint` (`750` chiều) — `fingerprint_similarity.pkl`
  9. `D9_profile_freq_cos` (`750` chiều) — cosine similarity trên ma trận tần suất `Y_train`
  10. `D10_profile_bin_cos` (`750` chiều) — cosine similarity trên ma trận nhị phân `(Y_train > 0)`
  11. `D11_pathway_enzyme` (`750` chiều) — `drug_pathway_enzyme_similarity.pkl`
- **View Phản Ứng Phụ (`n_se_views = 4` ở `MSSF_v2` với tổng `4 × 994 = 3,976` chiều; `n_se_views = 5` ở `C3_ce_unweighted` với tổng `5 × 994 = 4,970` chiều)**:
  1. `S1_adrecs_semantic` (`994` chiều) — `side_effect_semantic.pkl`
  2. `S2_glove_cos` (`994` chiều) — cosine similarity của `glove_wordEmbedding.pkl`
  3. `S3_profile_freq_cos` (`994` chiều) — cosine similarity trên `Y_train.T`
  4. `S4_profile_bin_cos` (`994` chiều) — cosine similarity trên `(Y_train > 0).T`
  5. `S5_meddra_ontology_cos` (`994` chiều, có trong `C3_ce_unweighted` khi `c2_se_mode='plus_ontology'`) — cosine similarity của `side_effect_label_750.mat['node_label']` (`994 × 243`)

> **Giải thích nguyên nhân chênh lệch trong cột diễn giải tay trước đó**:
> - Trong đối tượng mô hình thực tế (`MSSF_plus/ARCHITECTURE_DUMP.md` dòng 358–650 và 1041–1370), **`MSSF_v2` có đúng `11 + 4 = 15` view** và **`C3_ce_unweighted` có đúng `11 + 5 = 16` view** (hoàn toàn **không thiếu `S5` và không thiếu bất kỳ view thuốc nào** so với `MSSF` gốc; khác biệt duy nhất về kích thước thực thể so với `MSSF/model.py` gốc là bộ dữ liệu chuẩn hóa chung có `750` thuốc thay vì `757` thuốc).
> - Các con số tham số trong Bảng 4.1A (`8,767`; `5,085`; `7,112,484`; `1,698,304`; `61,216`) đều được trích từ mô hình `11 + 5` view thực tế, nhưng cột *"Giải thích chênh lệch"* ở bản báo cáo trước bị gõ nhầm diễn giải tay (`10 -> 64 -> 10`, `4 -> 64 -> 4`, `14 nhánh`). Công thức tensor chính xác từ `ARCHITECTURE_DUMP.md` (dòng 1183–1245) như sau:

#### Bảng 4.1A: Tham số từng module của `MSSF_v2` / `MSSF_v2_selfcorr` (`11 + 4` view) và `C3_ce_unweighted` / `C3_ce_unweighted_selfcorr` (`11 + 5` view)
| Module | `MSSF_v2` (`11 + 4` view) | `C3_ce_unweighted` (`11 + 5` view) | Cấu trúc Tensor Thực Tế & Giải Thích Chênh Lệch (`ARCHITECTURE_DUMP.md` L1183–1260) |
|---|---:|---:|---|
| `drug_modulator` (`AvailabilityModulator`) | `0` | `8,767` | **`11` view thuốc (`750`-D)**: `missing_token:(1, 11, 750) = 8,250` + `gate_fc1: Linear(11, 22) = 264` + `gate_fc2: Linear(22, 11) = 253` $\Rightarrow 8,250 + 264 + 253 = \mathbf{8,767}$ |
| `se_modulator` (`AvailabilityModulator`) | `0` | `5,085` | **`5` view SE (`994`-D)**: `missing_token:(1, 5, 994) = 4,970` + `gate_fc1: Linear(5, 10) = 60` + `gate_fc2: Linear(10, 5) = 55` $\Rightarrow 4,970 + 60 + 55 = \mathbf{5,085}$ |
| `encoderConnection` (`EncoderConnection`) | `6,602,562` | `7,112,484` | Autoencoder nối toàn bộ các view: ở `MSSF_v2` chiều vào/ra là `11×750 + 4×994 = 12,226` (`Linear(12226, 256)` + Attention + `Linear(256, 128)` + `Linear(128, 256)` + `Linear(256, 12226)`); ở `C3_ce_unweighted` thêm `S5` (`994` chiều) nên chiều vào/ra là `11×750 + 5×994 = 13,220` (`+509,922` tham số từ `994×256 + 994×257`) |
| `encoderAddition` (`EncoderAddition`) | `1,225,296` | `1,225,296` | Cộng từng phần các view thuốc (`750` chiều) và cộng từng phần các view SE (`994` chiều) rồi nối lại thành `750 + 994 = 1,744` chiều (`Linear(1744, 256)` + Attention + `Linear(256, 128)` + `Linear(128, 256)` + `Linear(256, 1744)`) |
| `preprocess` (`Preprocess`) | `1,570,688` | `1,698,304` | Mỗi view có 1 nhánh chiếu tuyến tính sang `128` chiều: `MSSF_v2` có **`11` nhánh thuốc** (`11 × [Linear(750, 128) + BN(128)] = 1,060,224`) + **`4` nhánh SE** (`4 × [Linear(994, 128) + BN(128)] = 510,464`, tổng **15 nhánh**); `C3_ce_unweighted` có **`11` nhánh thuốc + `5` nhánh SE** (tổng **16 nhánh**, tăng `+127,616` tham số từ nhánh thứ 5 cho `S5`) |
| `crossProduction` (`CrossProduction`) | `55,584` | `61,216` | Tích ngoài từng cặp view thuốc × SE: `MSSF_v2` có `11 × 4 = 44` kênh vào `Conv2d(44, 32, k=4, s=4)`; `C3_ce_unweighted` có `11 × 5 = 55` kênh vào `Conv2d(55, 32, k=4, s=4)` (tăng `11 × 32 × 4 × 4 = +5,632` tham số) |
| `attention` (`Attention`) | `591,744` | `591,744` | Self-attention gộp 3 encoder (`inputdim = 384`, `heads = 4`) |
| `gaussian_parametrizer` (`GaussianParametrizer`) | `49,280` | `49,280` | `h1: Linear(384, 64)` (`24,640`) + `h2: Linear(384, 64)` (`24,640`) |
| `classifier` (`Classifier`) | `2,245` | `2,245` | `Linear(64, 32)` (`2,080`) + `Linear(32, 5)` (`165`) |
| **Tổng cộng (`Total Parameters`)** | **`10,097,399`** | **`10,754,421`** | Chênh lệch `+657,022` tham số (`+6.51%`) |

- **Xác nhận cấu hình `ProfileViewRegressor` thực tế đã dùng trong toàn bộ Giai đoạn 2**:
  - Kiến trúc trong `MSSF_plus/views.py` (dòng 130–149) và `MSSF_plus/ARCHITECTURE_DUMP.md` (dòng 74–172): `6759 -> 256 -> 256 -> 1500` (`Linear(6759, 256) -> LayerNorm(256) -> SiLU -> Dropout(0.2) -> Linear(256, 256) -> LayerNorm(256) -> SiLU -> Dropout(0.2) -> Linear(256, 1500) -> Sigmoid`), tổng cộng **`2,182,876` tham số** cho mỗi mô hình con của Cross-fit.
  - Huấn luyện trong `train_and_impute_regress` (`MSSF_plus/views.py` dòng 189–310): **`epochs = 80`**, **`batch_size = 64`**, `AdamW(lr=1e-3, weight_decay=1e-4)`, hàm mục tiêu **`loss = mse + 0.1 * cos`** (MSE kết hợp `1 - cosine_similarity`), chọn checkpoint tốt nhất theo validation MSE nội bộ trong từng fold của Cross-fit **`K = 5`**.
  - **Giai đoạn 5 (Slim)**: Khởi động lại theo chỉ đạo mới (`encoderConnection` 6.6M–7.1M tham số là khối lớn nhất).

---

### 4.2. Mục 2: So Sánh Hai Biến Thể Bù Khuyết Rẻ (`regress_ridge`, `regress_knn`) Với `regress` (MLP) Trên Fold 0, Split B (Cross-fit $K = 5$)

- Thiết lập thực nghiệm & số lượng thuốc chính xác trong mỗi lần Cross-fit (`_make_drug_crossfit_splits`, $K = 5$, `val_frac = 0.10`):
  - **Trên Split B (Cold-start Drug)**: Trong mỗi fold (0..9), tập train có đúng **`540` thuốc** (`750` thuốc tổng trừ `75` thuốc `test`, `67` thuốc `val_select`, `68` thuốc `val_calib`). Khi chia Cross-fit $K = 5$, mỗi nhóm `holdout` nhận dự đoán OOF gồm **`108` thuốc** ($540 / 5 = 108$), và phần `pool` gồm **`432` thuốc** còn lại được chia thành **`389` thuốc `fit_sub`** (90%) để khớp bộ bù khuyết và **`43` thuốc `val_sub`** (10%) để chọn siêu tham số / dừng sớm.
  - **Trên Split A (Warm-start Pair)**: Tập train có đủ **`750` thuốc**; chia Cross-fit $K = 5$ gồm **`150` thuốc `holdout`**, phần `pool` **`600` thuốc** chia thành **`540` thuốc `fit_sub`** (90%) và **`60` thuốc `val_sub`** (10%).
  - **`regress_ridge`**: Hồi quy tuyến tính Ridge đa đầu ra ($\hat{Y} = X_s W + b$) với chuẩn hóa đặc trưng tĩnh + tìm lưới $\alpha \in \{0.1, 1, 10, 100, 500, 1000, 2500, 5000, 10000\}$ tối ưu hóa trực tiếp lỗi bình phương trung bình trên tập `val_sub` (`43` thuốc trên Split B) bên trong từng fold của Cross-fit $K = 5$ bằng phân rã SVD, sau đó cắt miền giá trị về $[0, 1]$. Số tham số mạng học sâu bổ sung: **`0`**.
  - **`regress_knn`**: Lấy trung bình có trọng số độ tương đồng cosine của các hàng `D9, D10` thuộc $k$ thuốc train gần nhất trong `fit_sub` (`389` thuốc trên Split B) theo các view tĩnh khả dụng ($k \in \{3, 5, 10, 15, 20, 30\}$ được chọn trên `val_sub`). Số tham số: **`0`**.

#### Bảng 4.2A: Chất lượng bù khuyết trực tiếp trên `val_select` (Fold 0, Split B — 67 thuốc cold-start, Cross-fit $K = 5$)
| Phương pháp C1 ($K=5$) | Avg MSE $\downarrow$ | Avg Spearman $\uparrow$ | Avg Top-10 Overlap (%) $\uparrow$ | D9 MSE | D9 Spearman | D9 Top-10 | D10 MSE | D10 Spearman | D10 Top-10 | Thời gian bù khuyết (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mean` (baseline tĩnh) | 0.010361 | 0.6365 | 7.46% | 0.010922 | 0.6456 | 7.91% | 0.009799 | 0.6273 | 7.01% | 0.00 |
| `regress_knn` | 0.009782 | 0.6482 | 10.45% | 0.010271 | 0.6565 | 10.90% | 0.009292 | 0.6399 | 10.00% | 0.52 |
| `diffusion` (MLP denoiser) | 0.010614 | 0.6693 | 14.03% | 0.011042 | 0.6788 | 15.22% | 0.010187 | 0.6598 | 12.84% | 54.29 |
| `regress` (MLP hiện tại) | 0.009274 | 0.6843 | 14.78% | 0.009696 | 0.6929 | 16.57% | 0.008852 | 0.6758 | 12.99% | 30.17 |
| **`regress_ridge`** | **0.008656** | **0.7054** | **22.24%** | **0.009029** | **0.7136** | **22.99%** | **0.008283** | **0.6971** | **21.49%** | **1.11** |

#### Bảng 4.2B: Hiệu năng Downstream của `MSSF_plus` (`C3_ce_unweighted`) trên Fold 0, Split B (`val_select`)
| Cấu hình | `SelfCorr` | Bộ bù khuyết C1 ($K=5$) | Best Ep | `train(eval)` Macro-F1 | `val_select` Macro-F1 | Weighted-F1 | Accuracy | QWK | MAE | `Gap(eval)` | Ord-APS Cov | Ord-APS Size | Tổng thời gian (s) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`C3_ce_unweighted_ridge_selfcorr`** | **1** | **`regress_ridge`** | 17 | 0.7174 | **0.4322** | 0.5248 | 0.5263 | **0.5440** | 0.5453 | **+0.2851** | 0.911 | 2.54 | 121.8 |
| `C3_ce_unweighted_selfcorr` | 1 | `regress` (MLP) | 40 | 0.8249 | 0.4285 | **0.5274** | **0.5296** | 0.5355 | **0.5390** | +0.3964 | 0.896 | 2.51 | 144.5 |
| `C3_ce_unweighted_knn_selfcorr` | 1 | `regress_knn` | 16 | 0.7068 | 0.4192 | 0.5251 | 0.5257 | 0.5353 | 0.5477 | +0.2876 | 0.893 | 2.47 | 119.1 |
| `C3_ce_unweighted` | 0 | `regress` (MLP) | 12 | 0.6879 | **0.4426** | 0.5227 | 0.5296 | **0.5561** | 0.5588 | **+0.2453** | 0.903 | 2.58 | 139.3 |
| `C3_ce_unweighted_ridge` | 0 | `regress_ridge` | 17 | 0.7236 | 0.4320 | 0.5208 | 0.5170 | 0.5503 | 0.5664 | +0.2917 | 0.909 | 2.61 | 116.6 |
| `C3_ce_unweighted_knn` | 0 | `regress_knn` | 40 | 0.8487 | 0.4295 | **0.5367** | **0.5462** | 0.5449 | **0.5257** | +0.4191 | 0.886 | 2.48 | 114.0 |

- **Nhận xét quan trọng (Mục 2)**:
  1. Về **chất lượng bù khuyết trực tiếp (Bảng 4.2A)**, với đúng **`540` thuốc train trên Split B** (trong mỗi lần cross-fit dùng **`389` thuốc `fit_sub`**, **`43` thuốc `val_sub`**, **`108` thuốc `holdout`**, đặc trưng tĩnh nối liền `6,759` chiều), **`regress_ridge` vượt qua cả `regress` (MLP) lẫn `diffusion` trên mọi chỉ số**: giảm `Avg MSE` từ `0.009274 -> 0.008656`, tăng `Avg Spearman` từ `0.6843 -> 0.7054`, và **tăng gấp 1.5 lần tỷ lệ trùng láng giềng gần nhất `Top-10 Overlap` (`22.24%` so với `14.78%` của MLP và `7.46%` của `mean`)**, trong khi thời gian chạy chỉ mất **`1.11s`** (nhanh gấp 27 lần so với MLP `30.17s` và không tốn bất kỳ mạng nơ-ron phụ trợ nào).
  2. Về **hiệu năng downstream của `MSSF_plus` (Bảng 4.2B)**:
     - Ở thiết lập bảng chính (`profile_self_correction = 1`), trên Fold 0 Split B, **`C3_ce_unweighted_ridge_selfcorr` đạt `Macro-F1 = 0.4322` và `QWK = 0.5440`**, cao hơn `C3_ce_unweighted_selfcorr` dùng MLP (`Macro-F1 = 0.4285`, `QWK = 0.5355`).
     - Quyết định thiết kế: **Chuyển bộ bù khuyết C1 mặc định sang `regress_ridge`**; giữ `regress` (MLP) làm một dòng ablation; loại bỏ hoàn toàn `diffusion` khỏi mọi lần chạy tiếp theo.

---

### 4.3. Mục 3: Định Nghĩa Lại `Gap` Nhất Quán (`model.eval()`) & Tính Lại Cho Giai Đoạn 2

- **Định nghĩa chính thức của `Gap(eval)`**:
  $$\text{Gap}_{\text{eval}} = \text{Macro-F1}_{\text{train}}^{\text{eval}} - \text{Macro-F1}_{\text{val\_select}}^{\text{eval}}$$
  trong đó $\text{Macro-F1}_{\text{train}}^{\text{eval}}$ được tính tại đúng checkpoint tốt nhất (`best_epoch`) ở chế độ **`model.eval()`**:
  1. **Không bật `view_dropout`** (`c1_view_dropout_prob` không kích hoạt khi `self.training == False`) và không bật `nn.Dropout`.
  2. Dùng **biểu diễn tất định** (`mu` thay vì lấy mẫu nhiễu $z = \mu + \epsilon \sigma$).
  3. So khớp với **nhãn cứng (hard labels)** của tập train.
  4. **Đặc trưng giống hệt lúc đánh giá**: sử dụng cùng ma trận view (`D9, D10` OOF đã cố định) và nếu `profile_self_correction = 1` thì **áp dụng sửa rò rỉ cho từng cặp `(d, s)` thuộc tập train** khi tính $\text{Macro-F1}_{\text{train}}^{\text{eval}}$.

#### Bảng 4.3: So sánh `Gap(eval)` (Định nghĩa mới chuẩn hóa) và `Gap(train loop)` (Định nghĩa cũ trong vòng lặp train) trên 3 Fold Giai Đoạn 2
| Method | Split | `train(eval)` Macro-F1 | `val_select` Macro-F1 | **`Gap (eval tr - val)`** | `Gap (train loop cũ)` | Giải thích sự thay đổi |
|---|---:|---:|---:|---:|---:|---|
| `C3_ce_unweighted_selfcorr` | B | `0.7642 ± 0.0471` | `0.4169 ± 0.0259` | **`+0.3473 ± 0.0560`** | `+0.2933 ± 0.0507` | Khi tắt `view_dropout=0.25` & `Dropout=0.4` ở `eval()`, `train F1` tăng thêm `+0.0540` |
| `C3_ce_unweighted` | B | `0.6963 ± 0.0390` | `0.4266 ± 0.0275` | **`+0.2697 ± 0.0659`** | `+0.2028 ± 0.0826` | Khi tắt `view_dropout=0.25` & `Dropout=0.4` ở `eval()`, `train F1` tăng thêm `+0.0669` |
| `MSSF_v2_selfcorr` | B | `0.6634 ± 0.0420` | `0.3593 ± 0.0095` | **`+0.3041 ± 0.0396`** | `+0.3214 ± 0.0775` | `MSSF_v2` không có `view_dropout`, nên `train(eval)` sát với `train(loop)` |
| `MSSF_v2` | B | `0.7100 ± 0.0367` | `0.3599 ± 0.0318` | **`+0.3501 ± 0.0619`** | `+0.3431 ± 0.0608` | Chênh lệch nhỏ (`+0.0070`) do tắt `Dropout(0.4)` và lấy `mu` tất định |
| `C3_ce_unweighted_selfcorr` | A | `0.8381 ± 0.0432` | `0.6479 ± 0.0082` | **`+0.1901 ± 0.0493`** | `+0.1333 ± 0.0122` | Tắt `view_dropout=0.25` ở `eval()` làm tăng `train F1` thêm `+0.0568` |
| `C3_ce_unweighted` | A | `0.8436 ± 0.0181` | `0.6454 ± 0.0099` | **`+0.1982 ± 0.0275`** | `+0.1332 ± 0.0090` | Tắt `view_dropout=0.25` ở `eval()` làm tăng `train F1` thêm `+0.0650` |
| `MSSF_v2_selfcorr` | A | `0.8234 ± 0.0402` | `0.6421 ± 0.0054` | **`+0.1813 ± 0.0439`** | `+0.1445 ± 0.0055` | Tắt `Dropout(0.4)` và lấy `mu` tất định ở `eval()` |
| `MSSF_v2` | A | `0.8401 ± 0.0221` | `0.6506 ± 0.0140` | **`+0.1896 ± 0.0114`** | `+0.1408 ± 0.0119` | Tắt `Dropout(0.4)` và lấy `mu` tất định ở `eval()` |

- **Nhận xét bắt buộc về `Gap(eval)`**: **`Gap(eval)` của `MSSF_plus` không nhỏ hơn `MSSF_v2` (trên Split B với `SelfCorr=1`: `+0.3473 ± 0.0560` so với `+0.3041 ± 0.0396`; trên Split A: `+0.1901 ± 0.0493` so với `+0.1813 ± 0.0439`; trên Fold 0 Split B qua 3 seed ở Bảng 4.4B: `+0.3610 ± 0.0296` so với `+0.2918 ± 0.0208`) — do đó hoàn toàn không kết luận về mức overfit.**

---

### 4.4. Mục 4: Ước Lượng Nhiễu Giữa Các Lần Chạy (`Seed 42, 43, 44`) Trên Fold 0, Split B

#### Bảng 4.4A: Kết quả từng Seed (`42, 43, 44`) trên Fold 0, Split B (`val_select`)
| Nhóm mô hình | `SelfCorr` | Seed | Best Ep | `train(eval)` F1 | `val_select` Macro-F1 | Weighted-F1 | Accuracy | QWK | MAE | `Gap(eval)` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `MSSF_v2` | 0 | 42 | 22 | 0.6744 | 0.3430 | 0.4558 | 0.4574 | 0.4512 | 0.6828 | +0.3315 |
| `MSSF_v2_s43` | 0 | 43 | 22 | 0.6606 | 0.3523 | 0.4486 | 0.4713 | 0.4323 | 0.6302 | +0.3083 |
| `MSSF_v2_s44` | 0 | 44 | 15 | 0.6644 | 0.3790 | 0.4636 | 0.4962 | 0.4608 | 0.6148 | +0.2853 |
| `MSSF_v2_selfcorr` | 1 | 42 | 19 | 0.6194 | 0.3536 | 0.4652 | 0.4580 | 0.4141 | 0.7262 | +0.2659 |
| `MSSF_v2_selfcorr_s43` | 1 | 43 | 22 | 0.6500 | 0.3573 | 0.4494 | 0.4598 | 0.4437 | 0.7334 | +0.2927 |
| `MSSF_v2_selfcorr_s44` | 1 | 44 | 25 | 0.6778 | 0.3609 | 0.4955 | 0.5471 | 0.4561 | 0.5805 | +0.3169 |
| `C3_ce_unweighted` | 0 | 42 | 12 | 0.6879 | 0.4426 | 0.5227 | 0.5296 | 0.5561 | 0.5588 | +0.2453 |
| `C3_ce_unweighted_s43` | 0 | 43 | 34 | 0.8071 | 0.4318 | 0.5271 | 0.5257 | 0.5141 | 0.5649 | +0.3752 |
| `C3_ce_unweighted_s44` | 0 | 44 | 29 | 0.7793 | 0.4289 | 0.5323 | 0.5284 | 0.5441 | 0.5652 | +0.3504 |
| `C3_ce_unweighted_selfcorr` | 1 | 42 | 40 | 0.8249 | 0.4285 | 0.5274 | 0.5296 | 0.5355 | 0.5390 | +0.3964 |
| `C3_ce_unweighted_selfcorr_s43` | 1 | 43 | 33 | 0.7995 | 0.4369 | 0.5340 | 0.5351 | 0.5407 | 0.5410 | +0.3626 |
| `C3_ce_unweighted_selfcorr_s44` | 1 | 44 | 25 | 0.7524 | 0.4284 | 0.5403 | 0.5441 | 0.5272 | 0.5393 | +0.3240 |

#### Bảng 4.4B: Tổng hợp Nhiễu Giữa Các Seed (`Mean ± Std` qua 3 Seeds `42, 43, 44` trên Fold 0, Split B)
| Nhóm mô hình (Fold 0, Split B) | `SelfCorr` | `Macro-F1` (`Mean ± Std`) | `QWK` (`Mean ± Std`) | `Weighted-F1` (`Mean ± Std`) | `MAE` (`Mean ± Std`) | `Gap(eval)` (`Mean ± Std`) |
|---|---:|---:|---:|---:|---:|---:|
| `MSSF_v2` | 0 | `0.3581 ± 0.0153` | `0.4481 ± 0.0119` | `0.4560 ± 0.0061` | `0.6426 ± 0.0289` | `+0.3084 ± 0.0189` |
| `MSSF_v2_selfcorr` | 1 | `0.3573 ± 0.0030` | `0.4380 ± 0.0179` | `0.4700 ± 0.0201` | `0.6800 ± 0.0704` | `+0.2918 ± 0.0208` |
| **`C3_ce_unweighted`** | 0 | **`0.4344 ± 0.0059`** | **`0.5381 ± 0.0177`** | **`0.5274 ± 0.0039`** | **`0.5629 ± 0.0029`** | `+0.3236 ± 0.0562` |
| **`C3_ce_unweighted_selfcorr`** | 1 | **`0.4313 ± 0.0040`** | **`0.5345 ± 0.0056`** | **`0.5339 ± 0.0053`** | **`0.5398 ± 0.0009`** | `+0.3610 ± 0.0296` |

- **Nhận xét khoa học (Mục 4)**:
  1. **Biên độ nhiễu giữa các seed (`run-to-run noise`) rất nhỏ**: Đối với `C3_ce_unweighted_selfcorr` (`0.4313 ± 0.0040`), `C3_ce_unweighted` (`0.4344 ± 0.0059`), và `MSSF_v2_selfcorr` (`0.3573 ± 0.0030`), độ lệch chuẩn qua 3 seed chỉ nằm trong khoảng **`0.0030 – 0.0059` Macro-F1** (và `0.0153` đối với `MSSF_v2` gốc).
  2. **Ảnh hưởng của `profile_self_correction` nằm gọn trong biên độ nhiễu seed**:
     - Trên `MSSF_v2`: `0.3581 ± 0.0153` (`SelfCorr=0`) so với `0.3573 ± 0.0030` (`SelfCorr=1`), chênh lệch chỉ `-0.0008`.
     - Trên `C3_ce_unweighted`: `0.4344 ± 0.0059` (`SelfCorr=0`) so với `0.4313 ± 0.0040` (`SelfCorr=1`), chênh lệch chỉ `-0.0031` (nhỏ hơn 1 độ lệch chuẩn seed).
  3. **Mức cải thiện của `MSSF_plus` so với `MSSF_v2` lớn gấp nhiều lần nhiễu seed**:
     - Ở `SelfCorr=1`: `0.4313 ± 0.0040` so với `0.3573 ± 0.0030` (**`+0.0740` Macro-F1**, lớn gấp **$>18$ lần** độ lệch chuẩn seed; **`+0.0965` QWK**; **`-0.1402` MAE**).
     - Ở `SelfCorr=0`: `0.4344 ± 0.0059` so với `0.3581 ± 0.0153` (**`+0.0763` Macro-F1**, **`+0.0900` QWK**).

