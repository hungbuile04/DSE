# Data Documentation — Từng Baseline Model

Tài liệu giải thích dữ liệu đầu vào của từng model, cách dữ liệu được load và biến đổi trong code.

---

## 1. MGPred — Multi-Graph Prediction

**Thư mục gốc**: `MGPred/data/`  
**Kích thước**: 750 drugs × 994 side effects  
**Kiến trúc**: Bipartite multi-graph với 3 views cho mỗi bên (drug/SE)

### Input Data

```
Drug (750 nodes):
  View 1: drug_side.pkl (750×994)     → Drug-SE interaction adjacency
  View 2: Text_similarity_five.pkl    → STITCH combined drug similarity (750×750)
  View 3: Drug_word2vec.pkl           → Mol2vec molecular embedding (750×100)

Side Effect (994 nodes):
  View 1: drug_side.pkl.T (994×750)   → SE-Drug interaction adjacency (transpose)
  View 2: effect_side_semantic.pkl    → Semantic similarity DAG (994×994)
  View 3: glove_wordEmbedding.pkl     → GloVe word embedding (994×300)
```

### Cách hoạt động
- Xây dựng 3 bipartite graphs (1 graph per view)
- Mỗi graph: drug nodes ↔ SE nodes, edges theo similarity/interaction
- Graph Neural Network aggregate thông tin từ 3 views
- Output: predicted frequency cho mỗi cặp (drug, SE)

### Đặc điểm
- **Chỉ dùng 750 drugs** (loại 7 drugs thiếu STITCH data)
- Loss = MSE + L0 regularization
- Frequency là **regression** (dự đoán giá trị liên tục)

---

## 2. SDPred — Similarity-based Deep Prediction

**Thư mục gốc**: `SDPred/data/`  
**Kích thước gốc**: 757 drugs × 994 SEs (unified: 750×994)  
**Kiến trúc**: Outer-product CNN trên interaction maps

### Input Data

```
Drug Features (10 matrices, concat → 7570-D per drug):
  [1-5]  Text_similarity_{one..five}.pkl  → 5 kênh STITCH similarity (757×757 each)
  [6]    drug_mol.pkl → cosine_sim        → Mol2vec cosine similarity (757×757)
  [7]    drug_target.pkl → cosine_sim     → Drug-target cosine similarity (757×757)
  [8]    fingerprint_similarity.pkl       → Morgan FP Jaccard similarity (757×757)
  [9]    drug_side.pkl → cosine_sim       → Drug-SE frequency cosine sim (757×757) ⚠️
  [10]   drug_side.pkl → binary cosine    → Drug-SE binary cosine sim (757×757) ⚠️

Side Effect Features (4 matrices, concat → 3976-D per SE):
  [1]    side_effect_semantic.pkl          → ADReCS semantic similarity (994×994)
  [2]    glove_wordEmbedding.pkl → cosine → GloVe cosine similarity (994×994)
  [3]    drug_side.pkl.T → cosine_sim     → SE-Drug freq cosine sim (994×994) ⚠️
  [4]    drug_side.pkl.T → binary cosine  → SE-Drug binary cosine sim (994×994) ⚠️
```

### Cách hoạt động
1. Embed drug features: `(batch, 7570) → (batch, r)` (r=32)
2. Embed SE features: `(batch, 3976) → (batch, r)`
3. Outer product: `(batch, r, r)` → tạo interaction map
4. Tạo 10×4 = **40 interaction maps** (1 per drug-SE feature pair)
5. 2D CNN trên 40 maps → frequency + association prediction

### ⚠️ Data Leakage Warning
- Features [9-10] (drug) và [3-4] (SE) được tính từ `drug_side.pkl`
- Trong **cold-start**: test drugs chưa có association → features này bằng 0
- Paper SDPred xử lý bằng cách **loại bỏ 4 features** này trong "de novo" test

---

## 3. MSSF — Multi-Source Similarity Fusion

**Thư mục gốc**: `MSSF/Datas/`  
**Kích thước gốc**: 757 drugs × 994 SEs (unified: 750×994)  
**Kiến trúc**: Triple encoder (Concat + Addition + CNN) → Self-attention → BVI → 5-class

### Input Data

Giống SDPred nhưng thêm 1 drug feature:

```
Drug Features (11 matrices, concat → 8327-D):
  [1-10] Giống SDPred
  [11]   drug_pathway_enzyme_similarity.pkl → DrugBank pathway+enzyme sim (757×757)

Side Effect Features (4 matrices, concat → 3976-D):
  [1-4]  Giống SDPred
```

### Cách hoạt động
1. **EN-con** (Concatenation Encoder): concat 11+4 features → VAE encoder → latent
2. **EN-add** (Addition Encoder): element-wise sum 11 drug + 4 SE → VAE encoder → latent
3. **CNN-im** (Interaction Map): 11×4 = 44 outer products → 3-layer 2D CNN → features
4. **Self-attention fusion**: [EN-con, EN-add, CNN-im] → multi-head attention → fused
5. **BVI** (Bayesian Variational Inference): fused → μ, σ → reparameterize → latent (d=64)
6. **5-class classifier**: latent → softmax over {class 0,1,2,3,4} → frequency level 1-5

### Đặc điểm
- **5-class classification** (không phải regression!)
- Class 0-4 map tới frequency 1-5
- Loss = CrossEntropy + KL divergence + MSE reconstruction + L2 regularization

---

## 4. DSGAT — Drug-SE Graph Attention Network

**Thư mục gốc**: `DSGAT/data_WS/`  
**Kích thước**: 750 drugs × 994 SEs  
**Kiến trúc**: Dual GAT (drug graph + SE graph) → bipartite prediction

### Input Data

```
Drug representation:
  drug_SMILES_750.csv → RDKit → 2D molecular graph
    - Node features: 109-D atom features (atom type, degree, charge, etc.)
    - Edges: chemical bonds
    - Processed by PyG → data_WS/processed/data{0-9}.pt

Side Effect representation:
  side_effect_label_750.mat → node_label (994×243)
    - 243-D MedDRA hierarchical features per SE
    - k-NN graph (K=10, cosine metric) xây từ 994 SE nodes

Training labels:
  raw_frequency_750.mat → R (750×994)
    - Frequency matrix dùng làm ground truth
  mask_mat_750.mat → mask0..mask9 (750×994 each)
    - Binary masks: train_freq = R × mask[fold]
```

### Cách hoạt động
1. **Drug branch**: Molecular graph → 3-layer GAT (heads=10) → global_max_pool → drug embedding (200-D)
2. **SE branch**: k-NN graph → 3-layer GAT (heads=10) → SE embedding (200-D)
3. **Decoder**: `pred[d][s] = f(drug_emb[d], se_emb[s])` → predicted frequency
4. **Loss**: MSE cho observed pairs + ε-insensitive loss (ε=0.5, α=0.03) cho unobserved pairs

### ⚠️ PyG Cache
- `data_WS/processed/*.pt` là PyG cached graphs — **phải xóa** khi đổi mask!
- Nếu không xóa, PyG load cache cũ → bỏ qua unified masks

---

## 5. A3Net — Attention-based 3-branch Network

**Thư mục gốc**: `A-3Net-master/data/`  
**Kích thước**: 750 drugs × 994 SEs  
**Kiến trúc**: Dual GAT + **6-layer Transformer Cross-Attention**

### Input Data

**100% giống DSGAT** — cùng bộ files:
- `raw_frequency_750.mat`, `mask_mat_750.mat`
- `drug_SMILES_750.csv` → 109-D atom features
- `side_effect_label_750.mat` → 243-D MedDRA features

### Khác biệt so với DSGAT
- Thêm **Transformer Encoder** (6 layers, 8 heads) cross-attention giữa drug + SE embeddings
- Feature fusion: `x_final = (1-γ)·x_GAT + γ·x_Transformer` (γ=0.15)
- Cùng loss function (ε-insensitive)

---

## 6. HSTrans — Hierarchical Substructure Transformer

**Thư mục gốc**: `HSTrans/HSTrans_original/HSTrans/data/`  
**Kích thước**: 750 drugs × 994 SEs  
**Kiến trúc**: BPE Tokenizer → Dual Transformer → Outer Product CNN

### Input Data

```
Drug representation (BPE subword approach):
  drug_SMILES_750.csv → SMILES string
    ↓
  drug_codes_chembl_freq_1500.txt → BPE merge rules (ChEMBL)
  subword_units_map_chembl_freq_1500.csv → 2587 subword tokens
    ↓
  Tokenize SMILES → 50 subword indices + attention mask
    ↓
  Transformer Encoder (8 layers, 8 heads, hidden=512) → drug embedding

Side Effect representation (enriched substructures):
  drug_side.pkl → Extract_positive_negative_samples()
    ↓
  identify_sub() → tìm top-50 BPE substructures enriched cho mỗi SE
    ↓
  SE_sub_index_50.npy (994×50) → substructure indices
  SE_sub_mask_50.npy  (994×50) → attention masks
    ↓
  Transformer Encoder → SE embedding
```

### Cách hoạt động
1. Drug SMILES → BPE tokenization → 50 subword tokens → Transformer → drug embedding (200-D)
2. SE → Top-50 enriched substructures → Transformer → SE embedding (200-D)
3. Outer product: drug_emb × SE_emb → (50, 50, 200) → sum → (1, 50, 50) interaction map
4. CNN (3 kernels, size 3) → MLP → frequency prediction

### Đặc điểm
- **Không dùng RDKit molecular graph** — dùng BPE subword tokenization thay thế
- Negative sampling: 1:1 balanced (37,071 positives + 37,071 negatives)
- Paper claim 5-fold nhưng code thực tế chạy 10-fold

---

## 7. MVSL-DSF — Multi-View Subspace Learning with Dual-task Fusion

**Thư mục gốc**: `MVSL-DSF/`  
**Kích thước gốc**: 750 drugs × 893 SEs (unified: 750×994)  
**Kiến trúc**: 7 modalities → Multimodal Attention → Dual-task heads

### Input Data & 7 Modalities

```
drug_smiles.pkl → 3 drug modalities:
  [1] Molecular Graph  → DGL smiles_to_bigraph() → AttentiveFP GNN (3 layers)
  [2] SMILES Sequence   → One-hot (100×63)        → TextCNN (1D convolutions)
  [3] Fingerprint       → Morgan FP (2048-D)      → FingerprintCNN

drug_side_association_matrix.pkl (750×893):
  [4] Drug Assoc Profile → Row A[d,:] (893-D)     → Linear projection
  [6] SE Assoc Profile   → Col A[:,s] (750-D)     → Linear projection

drug_side_frequency_matrix.pkl (750×893):
  [5] Drug Freq Profile  → Row F[d,:] (893-D)     → Linear projection

side_vector_level_123.pkl (893×243):
  [7] SE Semantic        → Vec V[s,:] (243-D)     → Linear projection
```

### Cách hoạt động
1. Project 7 modalities → cùng hidden dim (64-D each)
2. L2 normalize + compute cosine similarity giữa các modalities
3. EMA-weighted importance scores + consistency loss (pairwise MSE)
4. **Multimodal Self-Attention** (8 heads) fuse 7×64 = 448-D
5. **Dual heads**:
   - Association: `Linear → ReLU → Linear → 2-class softmax` (CrossEntropyLoss)
   - Frequency: `Linear → ReLU → Linear → 1-D sigmoid` (BCEWithLogitsLoss)

### ⚠️ Cold-start Protocol
- Modalities [4-5] dùng drug profiles → phải **zero-out test drug rows** trong Split B
- Modalities [1-3] (molecular) và [6-7] (SE) vẫn hoạt động bình thường

### ⚠️ Frequency Normalization
- `drug_side_frequency_matrix.pkl` đã chia 5 → giá trị [0, 1]
- Khi train với BCEWithLogitsLoss: target phải trong [0, 1] ✓
- Khi eval: nhân sigmoid output × 5 để về scale [1, 5]

---

## 8. AGRL-DSE — Adaptive Graph Representation Learning

**Thư mục gốc**: `AGRL-DSE/`  
**Framework**: TensorFlow 1.12  
**Kiến trúc**: Heterogeneous GNN (GCN → GraphSAGE → GAT) → Adaptive Layer Attention

### Input Data

AGRL-DSE **không có data files riêng** — đọc từ thư mục `../data/`:

```
drug-similarity.csv    → Drug similarity matrix (N_d × N_d)
side-similarity.csv    → SE similarity matrix (N_s × N_s)
drug_side.csv          → Binary association matrix (N_d × N_s)
```

### Cách hoạt động
1. Xây dựng **heterogeneous graph**:
   ```
   A_H = | S_drug × μ    A        |  ∈ R^(N_d+N_s × N_d+N_s)
         | A^T            S_se × μ |
   ```
   (μ = similarity weight, default 6, paper nói 4)

2. 3-layer GNN stack:
   - Layer 1: **GCN** (spectral graph convolution) → h1
   - Layer 2: **GraphSAGE** (mean aggregation) → h2
   - Layer 3: **GAT** (attention mechanism) → h3

3. **Adaptive Layer Attention**: `emb = 0.5·h1 + 0.33·h2 + 0.25·h3` (learnable weights)

4. **Decoder**: `pred = sigmoid(E_drug · E_se^T)` (inner product, NO learnable matrix W)

5. **Loss**: Weighted BCE with `pos_weight = #negatives / #positives`

### ⚠️ Known Issues
- Paper nói bilinear decoder `σ(E·W·E^T)` nhưng code chỉ là inner product `σ(E·E^T)`
- Decoder output đã qua sigmoid nhưng lại feed vào `weighted_cross_entropy_with_logits` (expects logits) → double sigmoid
