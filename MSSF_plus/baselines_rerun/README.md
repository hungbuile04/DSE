# Protocol v2 Leak-Free Baseline Reruns & Colab Notebooks (`baselines_rerun/` & `notebooks_v2/`)

Thư mục này cung cấp bộ wrapper chạy lại toàn bộ baseline dưới **Protocol đánh giá thống nhất v2** (`shared_data/split_adapter_v2.py` → `load_splits_v2`: `train` 80% / `val_select` 10% / `val_calib` 10% / `test` 10%), **tuyệt đối không chỉnh sửa** bất kỳ file nào trong thư mục gốc của các baseline, `shared_data/`, hay `results/`.

---

## 1. Bảng tổng hợp các lỗi rò rỉ / sai lệch đã được khắc phục trong Protocol v2

| Mô hình | Vấn đề ở code gốc / Notebook v1 | Cách khắc phục trong `MSSF_plus/baselines_rerun/` |
|---|---|---|
| **MSSF** (`MSSF_v2`) | (1) Chọn epoch trực tiếp trên `test` (`<Best Test>`); (2) Views profile `D9, D10, S3, S4` ở Split A chứa cặp validation. | Chạy qua `MSSF_plus/train.py --exp MSSF_v2`: dựng ma trận tần suất từ ma trận `0` chỉ điền `train` (80%); chọn checkpoint theo `MacroF1` trên `val_select`. |
| **MGPred** (`run_mgpred.py`) | (1) Chọn checkpoint theo `test` RMSE; (2) Ở Split B, thuốc cold-start không có trong `u_adj.keys()` bị gán vector `0` cho cả đặc trưng tĩnh STITCH & Mol2vec và gây lỗi `ValueError: inhomogeneous shape` (`750` vs `100` chiều) trên NumPy ≥ 1.24; (3) `Ten_Fold_test.py` (L60 & L114) khởi tạo `avg_param = deepcopy(p.data)` thay vì `0` nhưng lại chia bias-correction `(1 - beta_ema ** steps_ema)`, làm bùng nổ dự đoán `~10^13` ở các epoch đầu. | Nạp đầy đủ đặc trưng tĩnh STITCH (`750×750`) & Mol2vec (`750×100`) cho toàn bộ 750 thuốc; dựng `ufea1/ifea1` và `u_adj/i_adj` chỉ từ `train` (80%); khởi tạo chuẩn `avg_param = zeros_like(p)` khớp với công thức bias-correction của Louizos et al. (2018); thuốc cold-start đi qua nhánh self-feature ở `MGPred/utils/aggregator.py` (L53–62). |
| **SDPred** (`run_sdpred.py`) | (1) Bất nhất siêu tham số giữa Split A (`batch_size=512`, không early-stop) và Split B (`batch_size=2048`, early-stop trên train); (2) Negative sampling ở Split B lấy nhầm cặp âm từ thuốc test vào tập train và 90% cặp âm test từ thuốc warm-start; (3) `retain_graph=True` gây chậm >2×. | Đồng nhất cấu hình giữa Split A & B (`epochs=100, batch_size=512, lr=1e-4, dropout=0.5`); tính 4 ma trận cosine profile chỉ từ `train` (80%); lấy mẫu âm leak-free qua `sample_negative_pairs_v2` (ở Split B khóa chặt trong `train_drugs`). |
| **DSGAT** & **A3Net** (`run_dsgat_a3net.py`) | (1) Notebook v2 dùng script warm-start (`WS_v4.py` / `warm-scence.py`) cho Split B nên 75 thuốc cold-start vẫn nằm trong `train_loader` với hàng = `0`, bị hàm loss ép dự đoán về `eps = 0.5`; (2) `nx.from_numpy_matrix` crash trên NetworkX ≥ 3.0; (3) Chọn epoch cuối không qua validation. | Ở Split B chỉ đưa 540 thuốc `train_drugs` vào `train_loader` (đúng thiết kế `ICS.py` / `cold-scence.py` của tác giả); dựng đồ thị k-NN của SE trực tiếp từ `scipy.sparse` trên `drug_side_train[active_train_drugs].T`; chọn checkpoint trên `val_select`. |
| **HSTrans** (`run_hstrans.py`) | (1) Rò rỉ nhãn test trong `identify_sub`: trích xuất top-50 BPE substructures cho mỗi SE trên toàn bộ 37.071 cặp dương (gồm cả test); (2) Đọc lại CSV bằng `pd.read_csv()` trong `__getitem__` ở mỗi mẫu làm chậm ~50×. | Vector hóa `identify_sub` bằng NumPy và tính lại **theo từng fold chỉ từ `fold_split.train`** (<0.05s/fold); tiền mã hóa BPE cho 750 thuốc 1 lần trên RAM; gán `model.device = device` trực tiếp không cần `sed`. |
| **MVSL-DSF** (`run_mvsl.py`) | (1) Không chọn checkpoint qua `val_select`; (2) `mvsl_750_994/side_vector_level_123.pkl` có 110 SE bị điền vector trung bình do map từ 893→994 SE. | Dựng `freq_train` và `assoc_train` chỉ từ `fold_split.train` (zero toàn bộ thuốc cold-start ở Split B); mặc định dùng ontology MedDRA đầy đủ `994×243` từ `shared_data/side_effect_label_750.mat['node_label']`; chọn checkpoint trên `val_select`. |

---

## 2. Quy tắc chọn Checkpoint & Đầu ra `results_v2/`

- **Checkpoint chính (báo cáo bảng chính trong bài báo)**: chọn epoch đạt **5-class `MacroF1`** cao nhất trên `val_select` (với mô hình hồi quy: sau khi làm tròn `np.clip(np.round(pred), 1, 5)`). Lưu tại:
  - `results_v2/{Method}_{A|B}/fold_{k}_labels.npy` (thang SIDER `1..5`, khớp 1-1 với `fold_split.test[:, 2]`)
  - `results_v2/{Method}_{A|B}/fold_{k}_preds.npy`
- **Checkpoint phụ cho mô hình hồi quy (báo cáo Phụ lục / Appendix)**: đồng thời lưu dự đoán của epoch đạt **`RMSE` thấp nhất trên `val_select`** tại:
  - `results_v2/{Method}_rmseSel_{A|B}/fold_{k}_labels.npy`
  - `results_v2/{Method}_rmseSel_{A|B}/fold_{k}_preds.npy`
- Mọi thư mục trong `results_v2/` đều tương thích 100% với cả hai công cụ đánh giá:
  - `python -m MSSF_plus.scripts.summarize_results --results_dir ./results_v2` (5-class metrics, Per-class F1 `c0..c4`, Conformal C4, kiểm định cặp Wilcoxon signed-rank).
  - `python shared_data/unified_eval.py --results_dir ./results_v2` (cả Regression metrics `RMSE, MAE, Pearson, Spearman` lẫn 5-class metrics).

---

## 3. Thứ tự chạy 4 Google Colab Notebooks v2 (`MSSF_plus/notebooks_v2/`)

Tất cả 4 notebook đều hỗ trợ **tự động đồng bộ từng fold lên Google Drive** (`--drive_dir /content/drive/MyDrive/DSE_results_v2`) và **tự động chạy tiếp từ fold dở dang nếu Colab ngắt kết nối** (`--resume`):

1. `colab_0_proposed_and_ablations.ipynb`: Chạy mô hình đề xuất (`MSSF_plus_full`), `MSSF_v2`, chuỗi ablation Leave-One-Out (`loo_no_C1..C5`) & `C1..C5`, kèm đánh giá Conformal C4.
2. `colab_1_pytorch_baselines_v2.ipynb`: Chạy nhóm PyTorch thuần (`MSSF_v2`, `SDPred`, `MGPred`) cho Split A & Split B.
3. `colab_2_pyg_hstrans_baselines_v2.ipynb`: Cài `torch_geometric`, `rdkit`, `subword-nmt` và chạy `DSGAT`, `A3Net`, `HSTrans` cho Split A & Split B.
4. `colab_3_dgl_mvsl_baseline_v2.ipynb`: Cài `dgl`, `dgllife`, `rdkit`, chạy `MVSL-DSF` cho Split A & Split B, và tổng hợp bảng kết quả cuối cùng của toàn bộ các mô hình từ Google Drive.
