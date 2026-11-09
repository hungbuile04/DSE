"""
Loss criterion for Stage 5 (`mssf_slim`), supporting both `cand_A` (with reconstruction losses)
and `slim_*` (where `rec_con is None`, `rec_add is None`, and reconstruction losses are dropped).
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from MSSF_plus.losses import compute_class_weights_from_train, logits_to_probs


class MSSFSlimCriterion(nn.Module):
    def __init__(
        self,
        cfg: dict,
        n_drug_views: int = 11,
        n_se_views: int = 5,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.loss_type = cfg.get('c3_loss_type', 'ce')
        self.use_class_weights = bool(cfg.get('c3_class_weighting', False))
        self.rec_con_weight = float(cfg.get('rec_con_weight', 0.0))
        self.rec_add_weight = float(cfg.get('rec_add_weight', 0.0))
        self.n_drug_views = int(n_drug_views)
        self.n_se_views = int(n_se_views)

        if class_weights is not None:
            self.register_buffer('class_weights', class_weights.float())
        else:
            self.register_buffer('class_weights', torch.ones(5, dtype=torch.float32))

    def _ce_with_soft_targets(
        self,
        logits: torch.Tensor,
        hard_labels: torch.Tensor,
        soft_targets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        w = self.class_weights if self.use_class_weights else None
        if soft_targets is None:
            return F.cross_entropy(logits, hard_labels, weight=w)
        log_probs = F.log_softmax(logits, dim=-1)
        if w is not None:
            weighted_targets = soft_targets * w.unsqueeze(0)
            per_sample = -(weighted_targets * log_probs).sum(dim=-1)
            norm = weighted_targets.sum(dim=-1).mean().clamp(min=1e-8)
            return per_sample.mean() / norm
        return -(soft_targets * log_probs).sum(dim=-1).mean()

    def forward(
        self,
        logits: torch.Tensor,
        rec_con: Optional[torch.Tensor],
        rec_add: Optional[torch.Tensor],
        batch_drug: torch.Tensor,
        batch_side: torch.Tensor,
        hard_labels: torch.Tensor,
        soft_targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        main_loss = self._ce_with_soft_targets(logits, hard_labels, soft_targets)
        total = main_loss

        if self.rec_con_weight > 0.0 and rec_con is not None:
            batch_vec = torch.cat((batch_drug, batch_side), dim=1)
            rec_loss1 = F.mse_loss(rec_con, batch_vec, reduction='none').sum(dim=-1).mean()
            total = total + self.rec_con_weight * rec_loss1

        if self.rec_add_weight > 0.0 and rec_add is not None:
            drug_chunks = batch_drug.chunk(self.n_drug_views, dim=1)
            side_chunks = batch_side.chunk(self.n_se_views, dim=1)
            drugs_sum = sum(drug_chunks)
            sides_sum = sum(side_chunks)
            add_features = torch.cat((drugs_sum, sides_sum), dim=1)
            rec_loss2 = F.mse_loss(rec_add, add_features, reduction='none').sum(dim=-1).mean()
            total = total + self.rec_add_weight * rec_loss2

        return total
