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
