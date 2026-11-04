"""
MSSF_plus neural architecture extending MSSF/model.py (750 drugs x 994 side effects):
  - Exact preservation of MSSF's 3-encoder structure (EncoderConnection, EncoderAddition,
    Preprocess + CrossProduction), self-attention fusion, Bayesian Variational Inference
    (GaussianParametrizer), and MLP Classifier when run in baseline `MSSF_v2` mode.
  - Dynamic support for varying numbers of drug views (11 or 9 under C1 `drop`) and
    SE views (4, 5 under C2 `plus_ontology`, or 3 under C2 `no_profile`).
  - C1 input-level view dropout during training (replacing D9, D10 with fallback views
    and setting their availability indicators to 0 with probability `p_drop`).
  - C1 input-level availability modulation (`AvailabilityModulator`) before the 3 encoders.
  - C3 CORN support (4 conditional logits vs 5 class logits).
  - Optional C5 auxiliary cluster classification head on the BVI latent representation.
"""

from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from MSSF_plus.protocol import N_DRUGS, N_SIDES
from MSSF_plus.views import compute_batch_self_corrected_profiles


class GaussianParametrizer(nn.Module):
    def __init__(self, feature_dim: int, latent_dim: int):
        super().__init__()
        self.h1 = nn.Linear(feature_dim, latent_dim)
        self.h2 = nn.Linear(feature_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.h1(x), self.h2(x)


class Attention(nn.Module):
    """Exact MSSF Attention block (MSSF/model.py L27-68)."""

    def __init__(self, inputdim: int, heads: int):
        super().__init__()
        self.inputdim = inputdim
        self.heads = heads
        self.dq = self.dk = self.dv = inputdim // heads

        self.WQ = nn.Linear(self.inputdim, self.dq * self.heads, bias=False)
        self.WK = nn.Linear(self.inputdim, self.dk * self.heads, bias=False)
        self.WV = nn.Linear(self.inputdim, self.dv * self.heads, bias=False)

        self.LN1 = nn.LayerNorm(inputdim)
        self.l1 = nn.Linear(inputdim, inputdim)
        self.LN2 = nn.LayerNorm(inputdim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.WQ(x).view(-1, self.heads, self.dq).transpose(0, 1)
        K = self.WK(x).view(-1, self.heads, self.dk).transpose(0, 1)
        V = self.WV(x).view(-1, self.heads, self.dv).transpose(0, 1)

        QK = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.dk)
        QK = torch.softmax(QK, dim=-1)
        att = torch.matmul(QK, V)

        att = att.transpose(1, 2).reshape(-1, self.heads * self.dv)

        x = self.LN1(att + x)
        output = self.l1(x)
        x = self.LN2(output + x)
        return x


class EncoderConnection(nn.Module):
    def __init__(
        self,
        drugs_inputdim: int,
        sides_inputdim: int,
        latent_dim: int,
        feature_dim: int,
        heads: int,
        dropout: float,
    ):
        super().__init__()
        self.drugs_inputdim = drugs_inputdim
        self.sides_inputdim = sides_inputdim
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.heads = heads

        self.reluDrop = nn.Sequential(nn.LeakyReLU(0.01), nn.Dropout(dropout))

        self.l1 = nn.Sequential(
            nn.Linear(self.drugs_inputdim + self.sides_inputdim, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            self.reluDrop,
        )
        self.attention = Attention(inputdim=self.latent_dim, heads=self.heads)
        self.l2 = nn.Linear(self.latent_dim, self.feature_dim)

        self.l3 = nn.Sequential(
            nn.Linear(self.feature_dim, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            self.reluDrop,
            nn.Linear(self.latent_dim, self.drugs_inputdim + self.sides_inputdim),
        )

    def forward(self, drugs: torch.Tensor, sides: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat((drugs, sides), dim=1)
        x = self.l1(x)
        x = self.attention(x)
        x = self.l2(x)
        rec_conn = self.l3(x)
        return x, rec_conn


class EncoderAddition(nn.Module):
    def __init__(
        self,
        drugs_inputdim: int,
        sides_inputdim: int,
        n_drug_views: int,
        n_se_views: int,
        latent_dim: int,
        feature_dim: int,
        heads: int,
        dropout: float,
    ):
        super().__init__()
        self.drugs_inputdim = drugs_inputdim
        self.sides_inputdim = sides_inputdim
        self.n_drug_views = n_drug_views
        self.n_se_views = n_se_views
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.heads = heads

        self.reluDrop = nn.Sequential(nn.LeakyReLU(0.01), nn.Dropout(dropout))

        self.l1 = nn.Sequential(
            nn.Linear(self.drugs_inputdim + self.sides_inputdim, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            self.reluDrop,
        )
        self.attention = Attention(inputdim=self.latent_dim, heads=self.heads)
        self.l2 = nn.Linear(self.latent_dim, self.feature_dim)

        self.l3 = nn.Sequential(
            nn.Linear(self.feature_dim, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            self.reluDrop,
            nn.Linear(self.latent_dim, self.drugs_inputdim + self.sides_inputdim),
        )

    def forward(
        self, drug_features: torch.Tensor, side_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        drug_chunks = drug_features.chunk(self.n_drug_views, dim=1)
        side_chunks = side_features.chunk(self.n_se_views, dim=1)
        drugs = sum(drug_chunks)
        sides = sum(side_chunks)

        add_features = torch.cat((drugs, sides), dim=1)
        x = self.l1(add_features)
        x = self.attention(x)
        x = self.l2(x)
        rec_add = self.l3(x)
        return x, rec_add


class Preprocess(nn.Module):
    def __init__(
        self,
        drug_inputdim: int,
        side_inputdim: int,
        n_drug_views: int,
        n_se_views: int,
        embeddim: int,
        dropout: float,
    ):
        super().__init__()
        self.drug_inputdim = drug_inputdim
        self.side_inputdim = side_inputdim
        self.n_drug_views = n_drug_views
        self.n_se_views = n_se_views
        self.embeddim = embeddim

        self.drug_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.drug_inputdim, self.embeddim),
                nn.BatchNorm1d(self.embeddim),
                nn.LeakyReLU(0.01),
                nn.Dropout(dropout),
            )
            for _ in range(n_drug_views)
        ])
        self.side_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.side_inputdim, self.embeddim),
                nn.BatchNorm1d(self.embeddim),
                nn.LeakyReLU(0.01),
                nn.Dropout(dropout),
            )
            for _ in range(n_se_views)
        ])

    def forward(
        self, drug_features: torch.Tensor, side_features: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        drug_chunks = drug_features.chunk(self.n_drug_views, dim=1)
        side_chunks = side_features.chunk(self.n_se_views, dim=1)

        drugs = [layer(chunk) for layer, chunk in zip(self.drug_layers, drug_chunks)]
        sides = [layer(chunk) for layer, chunk in zip(self.side_layers, side_chunks)]
        return drugs, sides


class CrossProduction(nn.Module):
    def __init__(self, cross_dim: int, feature_dim: int, input_channel: int, use_factored: bool = True):
        super().__init__()
        self.cross_dim = cross_dim
        self.feature_dim = feature_dim
        self.kernel_size = 4
        self.strides = 4
        self.latent_channel = 32
        self.input_channel = input_channel
        self.use_factored = bool(use_factored)

        self.cnn = nn.Sequential(
            nn.Conv2d(self.input_channel, self.latent_channel, kernel_size=self.kernel_size, stride=self.strides),
            nn.BatchNorm2d(self.latent_channel),
            nn.ReLU(),
            nn.Conv2d(self.latent_channel, self.latent_channel, kernel_size=self.kernel_size, stride=self.strides),
            nn.BatchNorm2d(self.latent_channel),
            nn.ReLU(),
            nn.Conv2d(self.latent_channel, self.latent_channel, kernel_size=self.kernel_size, stride=self.strides),
            nn.BatchNorm2d(self.latent_channel),
            nn.ReLU(),
        )

    def forward(self, drugs: List[torch.Tensor], sides: List[torch.Tensor]) -> torch.Tensor:
        if (
            self.use_factored
            and self.kernel_size == self.strides
            and self.cross_dim % self.kernel_size == 0
            and len(drugs) * len(sides) == self.input_channel
        ):
            # Exact algebraic identity for Conv2d(k=4, s=4) applied to outer products d_v (x) s_u:
            # Avoids materializing the (B, V_d * V_s, 128, 128) tensor (~1.84 GB/batch)
            b = drugs[0].shape[0]
            vd = len(drugs)
            vs = len(sides)
            k = self.kernel_size
            g = self.cross_dim // k
            c_out = self.latent_channel

            d_flat = (
                torch.stack(drugs, dim=1)
                .view(b, vd, g, k)
                .permute(0, 2, 1, 3)
                .reshape(b * g, vd * k)
            )
            w_flat = (
                self.cnn[0]
                .weight.view(c_out, vd, vs, k, k)
                .permute(1, 3, 0, 2, 4)
                .reshape(vd * k, c_out * vs * k)
            )
            m = torch.mm(d_flat, w_flat).view(b, g * c_out, vs * k)
            s_flat = (
                torch.stack(sides, dim=1)
                .view(b, vs, g, k)
                .permute(0, 1, 3, 2)
                .reshape(b, vs * k, g)
            )
            conv0_out = (
                torch.bmm(m, s_flat)
                .view(b, g, c_out, g)
                .permute(0, 2, 1, 3)
                .contiguous()
            )
            if self.cnn[0].bias is not None:
                conv0_out = conv0_out + self.cnn[0].bias.view(1, c_out, 1, 1)
            x = self.cnn[1:](conv0_out).view(-1, self.feature_dim)
            return x

        crosspro = []
        for d_emb in drugs:
            for s_emb in sides:
                crosspro.append(torch.bmm(d_emb.unsqueeze(2), s_emb.unsqueeze(1)))
        crosspro2d = torch.stack(crosspro, dim=1)
        x = self.cnn(crosspro2d).view(-1, self.feature_dim)
        return x


class Classifier(nn.Module):
    def __init__(self, latent_dim: int, classes: int, dropout: float):
        super().__init__()
        self.latent_dim = latent_dim
        self.classes = classes
        self.reluDrop = nn.Sequential(nn.LeakyReLU(0.01), nn.Dropout(dropout))
        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim // 2),
            self.reluDrop,
            nn.Linear(self.latent_dim // 2, self.classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class AvailabilityModulator(nn.Module):
    """
    C1 Input-level availability modulation:
      - Adds a learnable per-view missingness token when `avail_{i,v} == 0`.
      - Computes a per-view residual scale gate conditioned on the full availability vector `a_i`,
        allowing available views to re-scale when profile or sparse static views are missing.
      - Zero-initialized so at epoch 0 it acts as an exact identity mapping.
    """

    def __init__(self, n_views: int, view_dim: int):
        super().__init__()
        self.n_views = n_views
        self.view_dim = view_dim
        self.missing_token = nn.Parameter(torch.zeros(1, n_views, view_dim))
        self.gate_fc1 = nn.Linear(n_views, 2 * n_views)
        self.gate_act = nn.SiLU()
        self.gate_fc2 = nn.Linear(2 * n_views, n_views)
        nn.init.zeros_(self.gate_fc2.weight)
        nn.init.zeros_(self.gate_fc2.bias)

    def forward(self, flat_features: torch.Tensor, avail: torch.Tensor) -> torch.Tensor:
        b = flat_features.shape[0]
        views = flat_features.view(b, self.n_views, self.view_dim)
        gate_delta = torch.tanh(self.gate_fc2(self.gate_act(self.gate_fc1(avail))))  # (b, V)
        scale = 1.0 + 0.2 * gate_delta.unsqueeze(-1)                                # (b, V, 1)
        missing_mask = (1.0 - avail).unsqueeze(-1)                                  # (b, V, 1)
        modulated = views * scale + missing_mask * self.missing_token
        return modulated.reshape(b, self.n_views * self.view_dim)


class MSSFPlusModel(nn.Module):
    """
    Full MSSF_plus model supporting C1, C2, C3, C5, profile_self_correction, and BVI ablation.
    """

    def __init__(
        self,
        cfg: dict,
        n_drug_views: int = 11,
        n_se_views: int = 4,
        profile_drug_view_indices: Optional[List[int]] = None,
        profile_se_view_indices: Optional[List[int]] = None,
        drug_side_train: Optional[np.ndarray] = None,
        n_clusters: int = 0,
    ):
        super().__init__()
        self.cfg = cfg
        self.n_drug_views = int(n_drug_views)
        self.n_se_views = int(n_se_views)
        self.profile_drug_view_indices = list(profile_drug_view_indices or [])
        if profile_se_view_indices is not None:
            self.profile_se_view_indices = list(profile_se_view_indices)
        else:
            self.profile_se_view_indices = [2, 3] if self.n_se_views >= 4 else []
        self.n_clusters = int(n_clusters)

        dropout = float(cfg.get('dropout', 0.3))
        gp_dim = int(cfg.get('gp', 64))
        loss_type = cfg.get('c3_loss_type', 'ce')
        out_classes = 4 if loss_type == 'corn' else 5

        self.use_availability = bool(cfg.get('c1_use_availability', False))
        self.view_dropout_prob = float(cfg.get('c1_view_dropout_prob', 0.0))
        self.aux_cluster_weight = float(cfg.get('c5_aux_cluster_weight', 0.0))
        self.profile_self_correction = bool(cfg.get('profile_self_correction', False))
        self.use_factored_crossprod = bool(cfg.get('use_factored_crossprod', True))
        self.use_bvi = bool(cfg.get('use_bvi', True))

        if self.profile_self_correction and drug_side_train is not None:
            y_np = np.asarray(drug_side_train, dtype=np.float32)
            b_np = (y_np > 0.0).astype(np.float32)
            g_d9_np = y_np @ y_np.T
            g_d10_np = b_np @ b_np.T
            g_s3_np = y_np.T @ y_np
            g_s4_np = b_np.T @ b_np

            self.register_buffer('_sc_y_train', torch.from_numpy(y_np), persistent=False)
            self.register_buffer('_sc_b_train', torch.from_numpy(b_np), persistent=False)
            self.register_buffer('_sc_g_d9', torch.from_numpy(g_d9_np), persistent=False)
            self.register_buffer('_sc_norm2_d9', torch.from_numpy(np.diag(g_d9_np).copy()), persistent=False)
            self.register_buffer('_sc_g_d10', torch.from_numpy(g_d10_np), persistent=False)
            self.register_buffer('_sc_norm2_d10', torch.from_numpy(np.diag(g_d10_np).copy()), persistent=False)
            self.register_buffer('_sc_g_s3', torch.from_numpy(g_s3_np), persistent=False)
            self.register_buffer('_sc_norm2_s3', torch.from_numpy(np.diag(g_s3_np).copy()), persistent=False)
            self.register_buffer('_sc_g_s4', torch.from_numpy(g_s4_np), persistent=False)
            self.register_buffer('_sc_norm2_s4', torch.from_numpy(np.diag(g_s4_np).copy()), persistent=False)
        else:
            self._sc_y_train = None

        if self.use_availability:
            self.drug_modulator = AvailabilityModulator(self.n_drug_views, N_DRUGS)
            self.se_modulator = AvailabilityModulator(self.n_se_views, N_SIDES)
        else:
            self.drug_modulator = None
            self.se_modulator = None

        self.feature_nums = self.n_drug_views * self.n_se_views
        self.encoderConnection = EncoderConnection(
            drugs_inputdim=N_DRUGS * self.n_drug_views,
            sides_inputdim=N_SIDES * self.n_se_views,
            latent_dim=256,
            feature_dim=128,
            heads=4,
            dropout=dropout,
        )
        self.encoderAddition = EncoderAddition(
            drugs_inputdim=N_DRUGS,
            sides_inputdim=N_SIDES,
            n_drug_views=self.n_drug_views,
            n_se_views=self.n_se_views,
            latent_dim=256,
            feature_dim=128,
            heads=4,
            dropout=dropout,
        )
        self.preprocess = Preprocess(
            drug_inputdim=N_DRUGS,
            side_inputdim=N_SIDES,
            n_drug_views=self.n_drug_views,
            n_se_views=self.n_se_views,
            embeddim=128,
            dropout=dropout,
        )
        self.crossProduction = CrossProduction(
            cross_dim=128,
            feature_dim=128,
            input_channel=self.feature_nums,
            use_factored=self.use_factored_crossprod,
        )
        self.attention = Attention(inputdim=128 * 3, heads=4)
        if self.use_bvi:
            self.gaussian_parametrizer = GaussianParametrizer(feature_dim=128 * 3, latent_dim=gp_dim)
            self.det_projector = None
        else:
            self.gaussian_parametrizer = None
            self.det_projector = nn.Sequential(
                nn.Linear(128 * 3, gp_dim),
                nn.Dropout(dropout),
            )
        self.classifier = Classifier(latent_dim=gp_dim, classes=out_classes, dropout=dropout)

        if self.aux_cluster_weight > 0.0 and self.n_clusters >= 2:
            self.aux_cluster_head = nn.Linear(gp_dim, self.n_clusters)
        else:
            self.aux_cluster_head = None

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        return mu

    def _apply_profile_self_correction(
        self,
        drugs: torch.Tensor,
        sides: torch.Tensor,
        drug_avail: Optional[torch.Tensor],
        se_avail: Optional[torch.Tensor],
        drug_idx: Optional[torch.Tensor],
        se_idx: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if (
            not self.profile_self_correction
            or self._sc_y_train is None
            or drug_idx is None
            or se_idx is None
        ):
            return drugs, sides, drug_avail, se_avail

        y_ds = self._sc_y_train[drug_idx, se_idx]
        if not torch.any(y_ds > 0.0):
            return drugs, sides, drug_avail, se_avail

        d9_corr, d10_corr, s3_corr, s4_corr = compute_batch_self_corrected_profiles(
            drug_idx=drug_idx,
            se_idx=se_idx,
            y_train=self._sc_y_train,
            b_train=self._sc_b_train,
            g_d9=self._sc_g_d9,
            norm2_d9=self._sc_norm2_d9,
            g_d10=self._sc_g_d10,
            norm2_d10=self._sc_norm2_d10,
            g_s3=self._sc_g_s3,
            norm2_s3=self._sc_norm2_s3,
            g_s4=self._sc_g_s4,
            norm2_s4=self._sc_norm2_s4,
        )

        b = drugs.shape[0]
        if len(self.profile_drug_view_indices) == 2:
            idx9, idx10 = self.profile_drug_view_indices[0], self.profile_drug_view_indices[1]
            drugs_3d = drugs.view(b, self.n_drug_views, N_DRUGS).clone()
            drugs_3d[:, idx9, :] = d9_corr
            drugs_3d[:, idx10, :] = d10_corr
            drugs = drugs_3d.reshape(b, self.n_drug_views * N_DRUGS)
            if drug_avail is not None:
                # If a drug had only this single pair in Y_train, its leave-one-pair-out profile is empty
                rem_norm2 = self._sc_norm2_d9[drug_idx] - y_ds * y_ds
                still_avail = (rem_norm2 > 1e-12).to(drug_avail.dtype)
                drug_avail = drug_avail.clone()
                drug_avail[:, idx9] = drug_avail[:, idx9] * still_avail
                drug_avail[:, idx10] = drug_avail[:, idx10] * still_avail

        if len(self.profile_se_view_indices) == 2:
            s_idx3, s_idx4 = self.profile_se_view_indices[0], self.profile_se_view_indices[1]
            sides_3d = sides.view(b, self.n_se_views, N_SIDES).clone()
            sides_3d[:, s_idx3, :] = s3_corr
            sides_3d[:, s_idx4, :] = s4_corr
            sides = sides_3d.reshape(b, self.n_se_views * N_SIDES)
            if se_avail is not None:
                rem_se_norm2 = self._sc_norm2_s3[se_idx] - y_ds * y_ds
                se_still_avail = (rem_se_norm2 > 1e-12).to(se_avail.dtype)
                se_avail = se_avail.clone()
                se_avail[:, s_idx3] = se_avail[:, s_idx3] * se_still_avail
                se_avail[:, s_idx4] = se_avail[:, s_idx4] * se_still_avail

        return drugs, sides, drug_avail, se_avail

    def _apply_c1_input_processing(
        self,
        drugs: torch.Tensor,
        sides: torch.Tensor,
        drug_avail: Optional[torch.Tensor],
        se_avail: Optional[torch.Tensor],
        fallback_d9: Optional[torch.Tensor],
        fallback_d10: Optional[torch.Tensor],
        drug_idx: Optional[torch.Tensor] = None,
        se_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        drugs, sides, drug_avail, se_avail = self._apply_profile_self_correction(
            drugs, sides, drug_avail, se_avail, drug_idx, se_idx
        )
        b = drugs.shape[0]
        if drug_avail is not None:
            d_avail = drug_avail.clone()
        else:
            d_avail = drugs.new_ones((b, self.n_drug_views))

        if se_avail is not None:
            s_avail = se_avail.clone()
        else:
            s_avail = sides.new_ones((b, self.n_se_views))

        # 1. C1 Training-time view dropout on profile views (D9, D10)
        if (
            self.training
            and self.view_dropout_prob > 0.0
            and len(self.profile_drug_view_indices) == 2
        ):
            drop_mask = torch.rand(b, device=drugs.device) < self.view_dropout_prob
            if torch.any(drop_mask):
                idx9, idx10 = self.profile_drug_view_indices[0], self.profile_drug_view_indices[1]
                drugs_3d = drugs.view(b, self.n_drug_views, N_DRUGS).clone()
                if fallback_d9 is not None:
                    drugs_3d[drop_mask, idx9, :] = fallback_d9[drop_mask]
                else:
                    drugs_3d[drop_mask, idx9, :] = 0.0
                if fallback_d10 is not None:
                    drugs_3d[drop_mask, idx10, :] = fallback_d10[drop_mask]
                else:
                    drugs_3d[drop_mask, idx10, :] = 0.0
                drugs = drugs_3d.reshape(b, self.n_drug_views * N_DRUGS)
                d_avail[drop_mask, idx9] = 0.0
                d_avail[drop_mask, idx10] = 0.0

        # 2. C1 Input-level availability modulation
        if self.drug_modulator is not None:
            drugs = self.drug_modulator(drugs, d_avail)
        if self.se_modulator is not None:
            sides = self.se_modulator(sides, s_avail)

        return drugs, sides

    def forward(
        self,
        drugs: torch.Tensor,
        sides: torch.Tensor,
        drug_avail: Optional[torch.Tensor] = None,
        se_avail: Optional[torch.Tensor] = None,
        fallback_d9: Optional[torch.Tensor] = None,
        fallback_d10: Optional[torch.Tensor] = None,
        drug_idx: Optional[torch.Tensor] = None,
        se_idx: Optional[torch.Tensor] = None,
    ):
        mod_drugs, mod_sides = self._apply_c1_input_processing(
            drugs, sides, drug_avail, se_avail, fallback_d9, fallback_d10,
            drug_idx=drug_idx, se_idx=se_idx,
        )

        feature1, rec_con = self.encoderConnection(mod_drugs, mod_sides)
        feature2, rec_add = self.encoderAddition(mod_drugs, mod_sides)
        prep_drugs, prep_sides = self.preprocess(mod_drugs, mod_sides)
        feature3 = self.crossProduction(prep_drugs, prep_sides)

        features = torch.cat((feature1, feature2, feature3), dim=1)
        features = self.attention(features)

        if self.use_bvi:
            mu, logvar = self.gaussian_parametrizer(features)
            latent_features = self.reparameterize(mu, logvar)
        else:
            latent_features = self.det_projector(features)
            mu = latent_features
            logvar = torch.zeros_like(latent_features)

        logits = self.classifier(latent_features)

        cluster_logits = (
            self.aux_cluster_head(latent_features)
            if self.aux_cluster_head is not None
            else None
        )

        return logits, rec_con, rec_add, mu, logvar, mod_drugs, mod_sides, cluster_logits
