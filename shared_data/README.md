# Shared Data — Unified Benchmark for DSE Baselines

## Tổng quan

Thư mục này chứa **dữ liệu thống nhất** (750 drugs × 994 side effects) dùng để đánh giá công bằng tất cả baseline models trên cùng một bộ train/test splits.

### Vấn đề cần giải quyết

Các bài báo gốc sử dụng kích thước dữ liệu khác nhau:

| Model | Drugs gốc | SEs gốc | Lý do khác |
|-------|:---------:|:-------:|------------|
| MGPred, DSGAT, A3Net, HSTrans | 750 | 994 | Loại 7 drugs thiếu STITCH |
| SDPred, MSSF | 757 | 994 | Giữ đầy đủ SIDER |
| MVSL-DSF | 750 | 893 | Chỉ dùng subset MedDRA |

→ Không thể so sánh trực tiếp kết quả giữa các bài!

**Giải pháp**: Chuẩn hóa tất cả về **750 drugs × 994 side effects** với cùng 10-fold splits (seed=42).

---

## Cấu trúc thư mục

```
shared_data/
├── drug_side.pkl                  # Ma trận tần suất gốc (750×994)
├── drug_SMILES_750.csv            # SMILES cho 750 drugs
├── raw_frequency_750.mat          # Frequency matrix (MAT format)
├── mask_mat_750.mat               # 10-fold masks cho DSGAT/A3Net
├── side_effect_label_750.mat      # MedDRA features (994×243)
├── effect_side_semantic.pkl       # SE semantic similarity (994×994)
├── glove_wordEmbedding.pkl        # GloVe 300-D cho 994 SEs
│
├── generate_splits.py             # Script tạo splits
├── split_adapter.py               # API load splits cho notebooks
├── unified_eval.py                # Script đánh giá thống nhất
├── prepare_all_data.py            # Pipeline chuyển đổi data gốc
│
├── splits/                        # 50 files split đã tạo sẵn
│   ├── splitA_fold{0-9}_train.npy # Split A: Warm-start (pair-wise)
│   ├── splitA_fold{0-9}_test.npy
│   ├── splitB_fold{0-9}_train.npy # Split B: Drug cold-start
│   ├── splitB_fold{0-9}_test.npy
│   └── splitB_fold{0-9}_test_drugs.npy
│
├── sdpred_750/                    # SDPred data adapted 757→750
├── mssf_750/                      # MSSF data adapted 757→750
└── mvsl_750_994/                  # MVSL-DSF data adapted 893→994 SEs
```

---

## Mô tả các file dữ liệu gốc

### `drug_side.pkl` — Ma trận tần suất
- **Shape**: `(750, 994)`, dtype `float64`
- **Giá trị**: Số nguyên {0, 1, 2, 3, 4, 5}
  - `0` = không có tác dụng phụ (unknown/negative)
  - `1` = very rare (rất hiếm)
  - `2` = rare (hiếm)
  - `3` = infrequent (không thường xuyên)
  - `4` = frequent (thường xuyên)
  - `5` = very frequent (rất thường xuyên)
- **Nguồn**: SIDER database (Side Effect Resource)
- **Positive pairs**: 37,071 cặp drug-SE có tần suất > 0

### `drug_SMILES_750.csv` — Cấu trúc phân tử
- **Format**: CSV 2 cột (không header): `drug_name, SMILES_string`
- **750 dòng**, mỗi dòng là 1 drug
- **SMILES** (Simplified Molecular Input Line Entry System): chuỗi ký tự biểu diễn cấu trúc 2D phân tử
- Ví dụ: `aspirin,CC(=O)Oc1ccccc1C(=O)O`

### `raw_frequency_750.mat` — Frequency matrix (MATLAB format)
- **Key**: `'R'`, shape `(750, 994)`
- Nội dung giống `drug_side.pkl` nhưng ở format `.mat`
- Dùng cho DSGAT và A3Net (code gốc đọc `.mat`)

### `mask_mat_750.mat` — 10-fold binary masks
- **Keys**: `'mask0'` đến `'mask9'`, mỗi mask shape `(750, 994)`
- Giá trị: `1` = training pair, `0` = test pair (bị mask)
- Dùng cho DSGAT và A3Net: `train_freq = R × mask[fold]`

### `side_effect_label_750.mat` — MedDRA features
- **Key**: `'node_label'`, shape `(994, 243)`
- 243-D vector cho mỗi SE dựa trên phân loại MedDRA:
  - Level 1: System Organ Class (SOC) — vd: "Cardiac disorders"
  - Level 2: High-Level Group Term (HLGT)
  - Level 3: High-Level Term (HLT)
- Dùng làm node features trong SE graph (DSGAT, A3Net)

### `effect_side_semantic.pkl` — SE semantic similarity
- **Shape**: `(994, 994)`, dtype `float64`
- Ma trận similarity giữa 994 SEs dựa trên ADReCS ontology (DAG-based)
- Giá trị trong [0, 1]: `sim[i][j]` = mức tương đồng ngữ nghĩa giữa SE i và SE j
- Dùng trong MGPred (SE graph view 2), SDPred/MSSF (SE feature 1)

### `glove_wordEmbedding.pkl` — GloVe word embeddings
- **Shape**: `(994, 300)`, dtype `float64`
- 300-D GloVe vector cho tên y khoa của mỗi SE (pre-trained trên Wikipedia)
- Dùng trong MGPred (SE graph view 3), SDPred/MSSF (→ cosine sim)

---

## Splits — Chia dữ liệu thống nhất

### Split A: Warm-start (Pair-wise 10-fold CV)
- Chia 37,071 positive pairs thành 10 folds (StratifiedKFold, seed=42)
- Mỗi fold: ~33,364 train pairs + ~3,707 test pairs
- File format: `(N, 3)` numpy array: `[drug_idx, se_idx, frequency]`
- **Tất cả drugs đều xuất hiện trong cả train lẫn test** → model đã "thấy" mọi drug

### Split B: Drug cold-start (Drug-level 10-fold CV)
- Chia 750 drugs thành 10 groups (~75 drugs/group)
- Test: tất cả positive pairs của 75 drugs held-out
- Train: tất cả positive pairs của 675 drugs còn lại
- `test_drugs.npy`: danh sách 75 drug indices bị held-out
- **Test drugs hoàn toàn mới** → model chưa từng "thấy" drug này khi train

### Cách load trong code:
```python
from split_adapter import load_splits, load_test_drugs

train_data, test_data = load_splits(fold=0, split_type='A')
# train_data shape: (N_train, 3) = [drug_idx, se_idx, freq]

test_drugs = load_test_drugs(fold=0)  # chỉ cho Split B
```

---

## Thư mục con — Data adapted cho từng model

### `sdpred_750/` — SDPred (757→750 drugs)

SDPred gốc dùng 757 drugs. Thư mục này chứa data đã loại 7 drugs thừa:

| File | Shape | Nội dung |
|------|-------|---------|
| `Text_similarity_{one..five}.pkl` | `(750, 750)` | 5 ma trận STITCH drug similarity |
| `fingerprint_similarity.pkl` | `(750, 750)` | Morgan fingerprint Jaccard similarity |
| `drug_mol.pkl` | `(750, 100)` | Mol2vec molecular embeddings |
| `drug_target.pkl` | `(750, ~847)` | Drug-target interaction profiles (DrugBank) |
| `drug_side.pkl` | `(750, 994)` | Frequency matrix |
| `glove_wordEmbedding.pkl` | `(994, 300)` | GloVe SE embeddings |
| `side_effect_semantic.pkl` | `(994, 994)` | SE semantic similarity |

SDPred concat 10 drug similarities = **7500-D** input, 4 SE similarities = **3976-D** input.

### `mssf_750/` — MSSF (757→750 drugs)

Giống `sdpred_750/` + thêm 1 file:

| File | Shape | Nội dung |
|------|-------|---------|
| `drug_pathway_enzyme_similarity.pkl` | `(750, 750)` | Pathway + enzyme similarity (DrugBank) |

MSSF concat 11 drug similarities = **8250-D** input, 4 SE similarities = **3976-D** input.

### `mvsl_750_994/` — MVSL-DSF (893→994 SEs)

MVSL-DSF gốc chỉ dùng 893 SEs. Thư mục này mở rộng lên 994:

| File | Shape | Nội dung |
|------|-------|---------|
| `drug_side_association_matrix.pkl` | `(750, 994)` | Binary association matrix {0,1} |
| `drug_side_frequency_matrix.pkl` | `(750, 994)` | Normalized frequency [0, 1] (= freq_gốc / 5) |
| `drug_smiles.pkl` | 750 items | List of `[drug_idx, SMILES_string]` |
| `side_vector_level_123.pkl` | `(994, 243)` | MedDRA vectors (101 SEs mới = mean imputation) |
| `final_sample.pkl` | `(N, 4)` | `[drug_idx, se_idx, assoc_label, freq_value]` |
| `col_mapping_893_to_994.npy` | `(893,)` | Ánh xạ cột: vị trí 893 SE gốc trong 994 SEs |

**Lưu ý**: `drug_side_frequency_matrix.pkl` đã normalize /5.0 (giá trị [0,1]), khác với `drug_side.pkl` gốc (giá trị {0-5}).

---

## Scripts

### `generate_splits.py`
Tạo tất cả 50 split files. Chạy 1 lần duy nhất:
```bash
python generate_splits.py
```

### `split_adapter.py`
API trung gian giúp notebooks load splits đúng format cho từng model:
- `load_splits(fold, split_type)` → `(train_data, test_data)`
- `load_test_drugs(fold)` → test drug indices (Split B)
- `save_mask_mat(freq, split_type, n_folds, path)` → tạo mask .mat cho DSGAT/A3Net
- `make_masked_freq(freq, fold, split_type)` → zero-out test pairs trong frequency matrix

### `unified_eval.py`
Đánh giá kết quả thống nhất:
```bash
python unified_eval.py --results_dir ./results
```
Tính: AUC, AUPR, RMSE, MAE trên `.npy` predictions từ tất cả models.

### `prepare_all_data.py`
Pipeline chuyển đổi data gốc (757→750, 893→994):
```bash
python prepare_all_data.py
```
Chỉ cần chạy 1 lần để tạo `sdpred_750/`, `mssf_750/`, `mvsl_750_994/`.
