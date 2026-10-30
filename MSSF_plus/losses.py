"""
Loss functions for MSSF_plus:
  - C3 Ordinal classification losses: 'ce' | 'emd' | 'corn' | 'ce_softquant'
  - Independent inverse-frequency class weighting flag (`c3_class_weighting`)
  - Compatibility with C5 soft label distributions (including cumulative conversion for CORN)
  - Original MSSF variational KL divergence and autoencoder reconstruction losses
  - Optional C5 auxiliary cluster classification loss from BVI latent space
"""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights_from_train(train_pairs: np.ndarray) -> torch.Tensor:
    """
    Compute normalized inverse-frequency class weights w_k = N / (5 * N_k)
    strictly from the 80% training split (Audit E).
    """
    y_train = np.clip(np.round(train_pairs[:, 2]).astype(int) - 1, 0, 4)
    n_total = float(len(y_train))
    counts = np.array([max(1, int((y_train == k).sum())) for k in range(5)], dtype=np.float32)
    weights = n_total / (5.0 * counts)
    return torch.from_numpy(weights)


def logits_to_probs(logits: torch.Tensor, loss_type: str = 'ce') -> torch.Tensor:
    """
    Convert model output logits to 5-class probability distribution (batch, 5).
      - For 'ce', 'emd', 'ce_softquant': logits has shape (batch, 5) -> softmax(logits, dim=-1).
      - For 'corn': logits has shape (batch, 4) representing conditional logits P(y > k | y >= k)
        for k = 0, 1, 2, 3 -> cumulative products P(y > k) -> 5 class probabilities.
    """
    if loss_type == 'corn':
        cond_probs = torch.sigmoid(logits)               # (batch, 4): P(y > k | y >= k)
        cum_gt = torch.cumprod(cond_probs, dim=-1)       # (batch, 4): P(y > k) for k=0,1,2,3
        p0 = 1.0 - cum_gt[:, 0:1]
        p1 = cum_gt[:, 0:1] - cum_gt[:, 1:2]
        p2 = cum_gt[:, 1:2] - cum_gt[:, 2:3]
        p3 = cum_gt[:, 2:3] - cum_gt[:, 3:4]
        p4 = cum_gt[:, 3:4]
        probs = torch.cat([p0, p1, p2, p3, p4], dim=-1)
        probs = torch.clamp(probs, min=1e-7, max=1.0)
        return probs / probs.sum(dim=-1, keepdim=True)
    return F.softmax(logits, dim=-1)


def kl_divergence_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Exact MSSF KL divergence formula (MSSF/mssf.py L264-265)."""
    return (-0.5 * (1.0 + logvar - mu.pow(2) - torch.exp(logvar)).sum(dim=1)).mean()


class MSSFPlusCriterion(nn.Module):
    """
    Combined loss function for MSSF_plus supporting C3 ordinal losses, C5 soft targets,
    independent class weighting, MSSF reconstruction/KL losses, and optional C5 cluster head.
    """

    def __init__(
        self,
        cfg: dict,
        n_drug_views: int = 11,
        n_se_views: int = 4,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.loss_type = cfg.get('c3_loss_type', 'ce')
        self.use_class_weights = bool(cfg.get('c3_class_weighting', False))
        self.softquant_weight = float(cfg.get('c3_softquant_weight', 0.1))
        self.gamma_min = float(cfg.get('c3_gamma_min', 1.0))
        self.gamma_max = float(cfg.get('c3_gamma_max', 10.0))
        self.total_epochs = max(1, int(cfg.get('epochs', 50)))

        self.kl_weight = float(cfg.get('kl_weight', 0.001))
        self.rec_con_weight = float(cfg.get('rec_con_weight', 0.0001))
        self.rec_add_weight = float(cfg.get('rec_add_weight', 0.0001))
        self.aux_cluster_weight = float(cfg.get('c5_aux_cluster_weight', 0.0))

        self.n_drug_views = int(n_drug_views)
        self.n_se_views = int(n_se_views)

        if class_weights is not None:
            self.register_buffer('class_weights', class_weights.float())
        else:
            self.register_buffer('class_weights', torch.ones(5, dtype=torch.float32))

        self.register_buffer('class_indices', torch.arange(5, dtype=torch.float32))

    def _compute_gamma(self, epoch: int) -> float:
        if self.total_epochs <= 1:
            return self.gamma_max
        frac = min(1.0, max(0.0, float(epoch - 1) / float(self.total_epochs - 1)))
        return self.gamma_min + frac * (self.gamma_max - self.gamma_min)

    def _ce_with_soft_targets(
        self,
        logits: torch.Tensor,
        hard_labels: torch.Tensor,
        soft_targets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        w = self.class_weights if self.use_class_weights else None

        # Exact match with original MSSF nn.CrossEntropyLoss when targets are hard one-hot
        if soft_targets is None:
            return F.cross_entropy(logits, hard_labels, weight=w)

        log_probs = F.log_softmax(logits, dim=-1)
        if w is not None:
            weighted_targets = soft_targets * w.unsqueeze(0)
            per_sample = -(weighted_targets * log_probs).sum(dim=-1)
            norm = weighted_targets.sum(dim=-1).mean().clamp(min=1e-8)
            return per_sample.mean() / norm
        return -(soft_targets * log_probs).sum(dim=-1).mean()

    def _emd_loss(
        self,
        logits: torch.Tensor,
        hard_labels: torch.Tensor,
        soft_targets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        if soft_targets is None:
            targets = F.one_hot(hard_labels, num_classes=5).float()
        else:
            targets = soft_targets

        cdf_p = torch.cumsum(probs, dim=-1)[:, :4]
        cdf_q = torch.cumsum(targets, dim=-1)[:, :4]
        per_sample_emd = ((cdf_p - cdf_q) ** 2).sum(dim=-1)

        if self.use_class_weights:
            sample_w = (targets * self.class_weights.unsqueeze(0)).sum(dim=-1)
            return (per_sample_emd * sample_w).sum() / sample_w.sum().clamp(min=1e-8)
        return per_sample_emd.mean()

    def _corn_loss(
        self,
        logits: torch.Tensor,
        hard_labels: torch.Tensor,
        soft_targets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        CORN loss over K-1 = 4 conditional tasks k in {0, 1, 2, 3}:
            subset mask M_k = P_target(y >= k)
            conditional target T_k = P_target(y > k) / P_target(y >= k)
        Reduces identically to standard hard CORN when soft_targets is one-hot.
        """
        if soft_targets is None:
            targets = F.one_hot(hard_labels, num_classes=5).float()
        else:
            targets = soft_targets

        if self.use_class_weights:
            sample_w = (targets * self.class_weights.unsqueeze(0)).sum(dim=-1)
        else:
            sample_w = torch.ones(logits.shape[0], dtype=torch.float32, device=logits.device)

        total_corn = logits.new_tensor(0.0)
        total_weight = logits.new_tensor(0.0)

        for k in range(4):
            # Probability mass on {y >= k} and {y > k}
            prob_ge_k = targets[:, k:].sum(dim=-1)
            prob_gt_k = targets[:, k + 1:].sum(dim=-1)
            valid = prob_ge_k > 1e-6
            if not torch.any(valid):
                continue
            cond_target = (prob_gt_k[valid] / prob_ge_k[valid].clamp(min=1e-6)).clamp(0.0, 1.0)
            bce = F.binary_cross_entropy_with_logits(logits[valid, k], cond_target, reduction='none')
            w_k = prob_ge_k[valid] * sample_w[valid]
            total_corn = total_corn + (bce * w_k).sum()
            total_weight = total_weight + w_k.sum()

        return total_corn / total_weight.clamp(min=1e-8)

    def _ce_softquant_loss(
        self,
        logits: torch.Tensor,
        hard_labels: torch.Tensor,
        soft_targets: Optional[torch.Tensor],
        epoch: int,
    ) -> torch.Tensor:
        ce = self._ce_with_soft_targets(logits, hard_labels, soft_targets)
        probs = F.softmax(logits, dim=-1)
        # Expected class value y_hat = sum_{k=0..4} k * p_k
        y_hat = (probs * self.class_indices.unsqueeze(0)).sum(dim=-1, keepdim=True)  # (batch, 1)
        d_k = torch.abs(y_hat - self.class_indices.unsqueeze(0))                     # (batch, 5)
        gamma = self._compute_gamma(epoch)
        w_k = F.softmax(-gamma * d_k, dim=-1)                                        # softmin(gamma * d)
        l_q_per_sample = (w_k * d_k).sum(dim=-1)

        if self.use_class_weights:
            if soft_targets is None:
                sample_w = self.class_weights[hard_labels]
            else:
                sample_w = (soft_targets * self.class_weights.unsqueeze(0)).sum(dim=-1)
            l_q = (l_q_per_sample * sample_w).sum() / sample_w.sum().clamp(min=1e-8)
        else:
            l_q = l_q_per_sample.mean()

        return ce + self.softquant_weight * l_q

    def forward(
        self,
        logits: torch.Tensor,
        rec_con: torch.Tensor,
        rec_add: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        batch_drug: torch.Tensor,
        batch_side: torch.Tensor,
        hard_labels: torch.Tensor,
        soft_targets: Optional[torch.Tensor] = None,
        cluster_logits: Optional[torch.Tensor] = None,
        cluster_ids: Optional[torch.Tensor] = None,
        epoch: int = 1,
    ) -> torch.Tensor:
        if self.loss_type == 'ce':
            main_loss = self._ce_with_soft_targets(logits, hard_labels, soft_targets)
        elif self.loss_type == 'emd':
            main_loss = self._emd_loss(logits, hard_labels, soft_targets)
        elif self.loss_type == 'corn':
            main_loss = self._corn_loss(logits, hard_labels, soft_targets)
        elif self.loss_type == 'ce_softquant':
            main_loss = self._ce_softquant_loss(logits, hard_labels, soft_targets, epoch=epoch)
        else:
            raise ValueError(f"Unknown c3_loss_type={self.loss_type!r}")

        # Auxiliary KL divergence
        kl_div = kl_divergence_loss(mu, logvar)

        # Auxiliary reconstruction losses (exact match with MSSF/mssf.py L280-306)
        batch_vec = torch.cat((batch_drug, batch_side), dim=1)
        drug_chunks = batch_drug.chunk(self.n_drug_views, dim=1)
        side_chunks = batch_side.chunk(self.n_se_views, dim=1)
        drugs_sum = sum(drug_chunks)
        sides_sum = sum(side_chunks)
        add_features = torch.cat((drugs_sum, sides_sum), dim=1)

        rec_loss1 = F.mse_loss(rec_con, batch_vec, reduction='none').sum(dim=-1).mean()
        rec_loss2 = F.mse_loss(rec_add, add_features, reduction='none').sum(dim=-1).mean()

        total = (
            main_loss
            + self.kl_weight * kl_div
            + self.rec_con_weight * rec_loss1
            + self.rec_add_weight * rec_loss2
        )

        # Optional C5 auxiliary cluster classification loss
        if (
            self.aux_cluster_weight > 0.0
            and cluster_logits is not None
            and cluster_ids is not None
        ):
            valid_c = cluster_ids >= 0
            if torch.any(valid_c):
                c_loss = F.cross_entropy(cluster_logits[valid_c], cluster_ids[valid_c])
                total = total + self.aux_cluster_weight * c_loss

        return total
