"""
Dataset and DataLoader construction for MSSF_plus.

Stores the compact entity-level view matrices (750 drugs, 994 side effects) in memory
and indexes them by (drug_idx, se_idx) on the fly, avoiding multi-gigabyte pair-level copies.
Enforces `shuffle=False` on all evaluation loaders (valsel, valcal, test).
"""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from MSSF_plus.protocol import FoldSplit
from MSSF_plus.views import FoldViews
from MSSF_plus.soft_labels import SoftLabelResult


class MSSFPlusDataset(Dataset):
    """
    Pair-level dataset for drug-side effect frequency prediction.
    """

    def __init__(
        self,
        pairs: np.ndarray,
        fold_views: FoldViews,
        soft_targets: Optional[np.ndarray] = None,
        cluster_ids: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = torch.device(device) if device is not None else torch.device('cpu')
        self.pairs = np.asarray(pairs)
        self.drug_indices = self.pairs[:, 0].astype(np.int64)
        self.se_indices = self.pairs[:, 1].astype(np.int64)
        self.freqs = self.pairs[:, 2].astype(np.float32)
        self.labels = np.clip(np.round(self.freqs).astype(np.int64) - 1, 0, 4)

        self.drug_features = torch.from_numpy(fold_views.drug_features_matrix.astype(np.float32)).to(self.device)
        self.se_features = torch.from_numpy(fold_views.se_features_matrix.astype(np.float32)).to(self.device)
        self.drug_avail = torch.from_numpy(fold_views.drug_avail.astype(np.float32)).to(self.device)
        self.se_avail = torch.from_numpy(fold_views.se_avail.astype(np.float32)).to(self.device)
        self.fallback_d9 = torch.from_numpy(fold_views.fallback_d9.astype(np.float32)).to(self.device)
        self.fallback_d10 = torch.from_numpy(fold_views.fallback_d10.astype(np.float32)).to(self.device)

        n = len(self.pairs)
        if soft_targets is not None:
            assert len(soft_targets) == n, f"soft_targets length {len(soft_targets)} != {n}"
            self.soft_targets = torch.from_numpy(soft_targets.astype(np.float32)).to(self.device)
        else:
            onehot = np.zeros((n, 5), dtype=np.float32)
            onehot[np.arange(n), self.labels] = 1.0
            self.soft_targets = torch.from_numpy(onehot).to(self.device)

        if cluster_ids is not None:
            assert len(cluster_ids) == n, f"cluster_ids length {len(cluster_ids)} != {n}"
            self.cluster_ids = torch.from_numpy(cluster_ids.astype(np.int64)).to(self.device)
        else:
            self.cluster_ids = torch.full((n,), -1, dtype=torch.long, device=self.device)

        self._drug_indices_t = torch.from_numpy(self.drug_indices).to(self.device)
        self._se_indices_t = torch.from_numpy(self.se_indices).to(self.device)
        self._labels_t = torch.from_numpy(self.labels).to(self.device)
        self._labels_cpu_t = torch.from_numpy(self.labels)
        self._freqs_t = torch.from_numpy(self.freqs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        d = int(self.drug_indices[idx])
        s = int(self.se_indices[idx])
        return {
            'drug_idx': torch.tensor(d, dtype=torch.long, device=self.device),
            'se_idx': torch.tensor(s, dtype=torch.long, device=self.device),
            'drug_feat': self.drug_features[d],
            'se_feat': self.se_features[s],
            'drug_avail': self.drug_avail[d],
            'se_avail': self.se_avail[s],
            'fallback_d9': self.fallback_d9[d],
            'fallback_d10': self.fallback_d10[d],
            'label': torch.tensor(int(self.labels[idx]), dtype=torch.long, device=self.device),
            'label_cpu': torch.tensor(int(self.labels[idx]), dtype=torch.long),
            'freq': torch.tensor(float(self.freqs[idx]), dtype=torch.float32),
            'soft_target': self.soft_targets[idx],
            'cluster_id': self.cluster_ids[idx],
        }

    def __getitems__(self, indices) -> Dict[str, torch.Tensor]:
        idx_cpu = torch.as_tensor(indices, dtype=torch.long)
        idx = idx_cpu.to(self.device) if self.device.type != 'cpu' else idx_cpu
        d = self._drug_indices_t[idx]
        s = self._se_indices_t[idx]
        return {
            'drug_idx': d,
            'se_idx': s,
            'drug_feat': self.drug_features[d],
            'se_feat': self.se_features[s],
            'drug_avail': self.drug_avail[d],
            'se_avail': self.se_avail[s],
            'fallback_d9': self.fallback_d9[d],
            'fallback_d10': self.fallback_d10[d],
            'label': self._labels_t[idx],
            'label_cpu': self._labels_cpu_t[idx_cpu],
            'freq': self._freqs_t[idx_cpu],
            'soft_target': self.soft_targets[idx],
            'cluster_id': self.cluster_ids[idx],
        }


def _identity_collate(batch):
    return batch


def build_dataloaders(
    fold_split: FoldSplit,
    fold_views: FoldViews,
    soft_label_res: SoftLabelResult,
    cfg: dict,
    device: Optional[torch.device] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Create train, valsel, valcal, and test DataLoaders.
    All evaluation DataLoaders enforce `shuffle=False`.
    """
    batch_size = int(cfg.get('batch_size', 128))
    eval_batch_size = int(cfg.get('eval_batch_size', 512))
    seed = int(cfg.get('seed', 42)) + int(fold_split.fold)

    train_ds = MSSFPlusDataset(
        pairs=fold_split.train,
        fold_views=fold_views,
        soft_targets=soft_label_res.soft_targets,
        cluster_ids=soft_label_res.drug_cluster_ids,
        device=device,
    )
    valsel_ds = MSSFPlusDataset(pairs=fold_split.valsel, fold_views=fold_views, device=device)
    valcal_ds = MSSFPlusDataset(pairs=fold_split.valcal, fold_views=fold_views, device=device)
    test_ds = MSSFPlusDataset(pairs=fold_split.test, fold_views=fold_views, device=device)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        drop_last=False,
        collate_fn=_identity_collate,
    )
    valsel_loader = DataLoader(
        valsel_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=_identity_collate,
    )
    valcal_loader = DataLoader(
        valcal_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=_identity_collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=_identity_collate,
    )

    return train_loader, valsel_loader, valcal_loader, test_loader
