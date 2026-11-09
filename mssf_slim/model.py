"""
Stage 5 (`mssf_slim`) neural architecture implementing `cand_A` and the 4 progressive Slim variants:
  - `cand_A`: exact parity with `MSSF_plus/model.py` under `cand_A`
  - `slim_nodec`: drop decoders of `EN-con` and `EN-add`
  - `slim_nodec_addw`: `slim_nodec` + masked learned view weights `w = softmax(logits)` over present views in `EN-add`
  - `slim_all`: `slim_nodec_addw` + shared entity embeddings `E_drug (750x64)`, `E_se (994x64)` with
                `e_v = LayerNorm(gamma_v * (x^(v) @ E) + beta_v)` and compressed `EN-con` on `[16x64; a_d; a_s] = 1040`
  - `slim_all_cnnshared`: `slim_all` + `Preprocess` using `e_v + Linear(64, 128)` per view instead of 16 `Linear(750/994, 128)`
"""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from MSSF_plus.protocol import N_DRUGS, N_SIDES
from MSSF_plus.views import compute_batch_self_corrected_profiles


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


class SharedViewEmbedding(nn.Module):
    """
    Shared entity embedding tables E_drug (750 x 64) and E_se (994 x 64) with per-view
    affine parameters (gamma_v, beta_v) and LayerNorm:
        e_v = LayerNorm(gamma_v * (x^(v) @ E) + beta_v)
    """

    def __init__(
        self,
        n_drug_views: int = 11,
        n_se_views: int = 5,
        drug_dim: int = N_DRUGS,
        se_dim: int = N_SIDES,
        embed_dim: int = 64,
    ):
        super().__init__()
        self.n_drug_views = int(n_drug_views)
        self.n_se_views = int(n_se_views)
        self.drug_dim = int(drug_dim)
        self.se_dim = int(se_dim)
        self.embed_dim = int(embed_dim)

        self.E_drug = nn.Parameter(torch.empty(self.drug_dim, self.embed_dim))
        self.E_se = nn.Parameter(torch.empty(self.se_dim, self.embed_dim))
        nn.init.kaiming_uniform_(self.E_drug, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.E_se, a=math.sqrt(5))

        self.gamma_drug = nn.Parameter(torch.ones(self.n_drug_views, self.embed_dim))
        self.beta_drug = nn.Parameter(torch.zeros(self.n_drug_views, self.embed_dim))
        self.gamma_se = nn.Parameter(torch.ones(self.n_se_views, self.embed_dim))
        self.beta_se = nn.Parameter(torch.zeros(self.n_se_views, self.embed_dim))

        self.layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False)

    def forward(
        self, drugs: torch.Tensor, sides: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b = drugs.shape[0]
        drugs_3d = drugs.view(b, self.n_drug_views, self.drug_dim)
        sides_3d = sides.view(b, self.n_se_views, self.se_dim)

        # (B, V_d, 750) @ (750, 64) -> (B, V_d, 64)
        proj_d = torch.matmul(drugs_3d, self.E_drug)
        aff_d = proj_d * self.gamma_drug.unsqueeze(0) + self.beta_drug.unsqueeze(0)
        e_d = self.layer_norm(aff_d)

        # (B, V_s, 994) @ (994, 64) -> (B, V_s, 64)
        proj_s = torch.matmul(sides_3d, self.E_se)
        aff_s = proj_s * self.gamma_se.unsqueeze(0) + self.beta_se.unsqueeze(0)
        e_s = self.layer_norm(aff_s)

        return e_d, e_s


class EncoderConnection(nn.Module):
    """
    EN-con module supporting:
      - Standard input (`drugs_inputdim + sides_inputdim = 13220`) or compressed input (`[16x64; a_d; a_s] = 1040`)
      - Optional decoder (`use_decoder=True` in `cand_A`, `False` in `slim_*`)
    """

    def __init__(
        self,
        drugs_inputdim: int,
        sides_inputdim: int,
        latent_dim: int,
        feature_dim: int,
        heads: int,
        dropout: float,
        use_decoder: bool = True,
        compressed_inputdim: Optional[int] = None,
    ):
        super().__init__()
        self.drugs_inputdim = drugs_inputdim
        self.sides_inputdim = sides_inputdim
        self.in_dim = (
            int(compressed_inputdim)
            if compressed_inputdim is not None
            else (self.drugs_inputdim + self.sides_inputdim)
        )
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.heads = heads
        self.use_decoder = bool(use_decoder)
        self.is_compressed = compressed_inputdim is not None

        self.reluDrop = nn.Sequential(nn.LeakyReLU(0.01), nn.Dropout(dropout))

        self.l1 = nn.Sequential(
            nn.Linear(self.in_dim, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            self.reluDrop,
        )
        self.attention = Attention(inputdim=self.latent_dim, heads=self.heads)
        self.l2 = nn.Linear(self.latent_dim, self.feature_dim)

        if self.use_decoder:
            self.l3 = nn.Sequential(
                nn.Linear(self.feature_dim, self.latent_dim),
                nn.BatchNorm1d(self.latent_dim),
                self.reluDrop,
                nn.Linear(self.latent_dim, self.in_dim),
            )
        else:
            self.l3 = None

    def forward(
        self,
        drugs: torch.Tensor,
        sides: torch.Tensor,
        e_d: Optional[torch.Tensor] = None,
        e_s: Optional[torch.Tensor] = None,
        d_avail: Optional[torch.Tensor] = None,
        s_avail: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.is_compressed:
            b = e_d.shape[0]
            x = torch.cat(
                (
                    e_d.reshape(b, -1),
                    e_s.reshape(b, -1),
                    d_avail,
                    s_avail,
                ),
                dim=1,
            )
        else:
            x = torch.cat((drugs, sides), dim=1)

        x = self.l1(x)
        x = self.attention(x)
        x = self.l2(x)
        rec_conn = self.l3(x) if self.l3 is not None else None
        return x, rec_conn


class EncoderAddition(nn.Module):
    """
    EN-add module supporting:
      - Unweighted view sum (`use_add_weights=False` in `cand_A` and `slim_nodec`)
      - Masked learned weighted sum `w = softmax(logits)` over present views (`use_add_weights=True`
        in `slim_nodec_addw`, `slim_all`, `slim_all_cnnshared`: missing views = 0, sum = 1)
      - Optional decoder (`use_decoder=True` in `cand_A`, `False` in `slim_*`)
    """

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
        use_decoder: bool = True,
        use_add_weights: bool = False,
    ):
        super().__init__()
        self.drugs_inputdim = drugs_inputdim
        self.sides_inputdim = sides_inputdim
        self.n_drug_views = n_drug_views
        self.n_se_views = n_se_views
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.heads = heads
        self.use_decoder = bool(use_decoder)
        self.use_add_weights = bool(use_add_weights)

        self.reluDrop = nn.Sequential(nn.LeakyReLU(0.01), nn.Dropout(dropout))

        self.l1 = nn.Sequential(
            nn.Linear(self.drugs_inputdim + self.sides_inputdim, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            self.reluDrop,
        )
        self.attention = Attention(inputdim=self.latent_dim, heads=self.heads)
        self.l2 = nn.Linear(self.latent_dim, self.feature_dim)

        if self.use_decoder:
            self.l3 = nn.Sequential(
                nn.Linear(self.feature_dim, self.latent_dim),
                nn.BatchNorm1d(self.latent_dim),
                self.reluDrop,
                nn.Linear(self.latent_dim, self.drugs_inputdim + self.sides_inputdim),
            )
        else:
            self.l3 = None

        if self.use_add_weights:
            self.drug_view_logits = nn.Parameter(torch.zeros(self.n_drug_views))
            self.se_view_logits = nn.Parameter(torch.zeros(self.n_se_views))
        else:
            self.register_parameter('drug_view_logits', None)
            self.register_parameter('se_view_logits', None)

    @staticmethod
    def _masked_softmax(logits_1d: torch.Tensor, avail_mask_2d: torch.Tensor) -> torch.Tensor:
        """
        Compute w = softmax(logits) strictly over present views (avail > 0.5).
        Missing views receive weight 0.0, and present views sum to 1.0 per sample.
        """
        present = avail_mask_2d > 0.5
        b = avail_mask_2d.shape[0]
        expanded = logits_1d.unsqueeze(0).expand(b, -1)
        masked_logits = expanded.masked_fill(~present, -1e9)
        w = torch.softmax(masked_logits, dim=1) * present.to(logits_1d.dtype)
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-12)
        return w

    @torch.no_grad()
    def get_learned_weights_summary(self, profile_drug_indices: Optional[List[int]] = None) -> Optional[Dict[str, List[float]]]:
        if not self.use_add_weights or self.drug_view_logits is None or self.se_view_logits is None:
            return None
        d_logits = self.drug_view_logits.detach().cpu()
        s_logits = self.se_view_logits.detach().cpu()

        w_d_full = torch.softmax(d_logits, dim=0).numpy().astype(np.float64)
        w_s_full = torch.softmax(s_logits, dim=0).numpy().astype(np.float64)

        # Also compute cold-start drug weights (when D9, D10 are absent = 0)
        cold_mask = torch.ones_like(d_logits, dtype=torch.bool)
        for idx in (profile_drug_indices or [8, 9]):
            if 0 <= idx < len(cold_mask):
                cold_mask[idx] = False
        d_cold_masked = d_logits.masked_fill(~cold_mask, -1e9)
        w_d_cold = (torch.softmax(d_cold_masked, dim=0) * cold_mask.float())
        w_d_cold = (w_d_cold / w_d_cold.sum().clamp(min=1e-12)).numpy().astype(np.float64)

        return {
            'drug_logits': [float(x) for x in d_logits.numpy()],
            'se_logits': [float(x) for x in s_logits.numpy()],
            'drug_weights_full': [float(x) for x in w_d_full],
            'drug_weights_coldstart': [float(x) for x in w_d_cold],
            'se_weights_full': [float(x) for x in w_s_full],
        }

    def forward(
        self,
        drug_features: torch.Tensor,
        side_features: torch.Tensor,
        d_avail: Optional[torch.Tensor] = None,
        s_avail: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.use_add_weights:
            drug_chunks = drug_features.chunk(self.n_drug_views, dim=1)
            side_chunks = side_features.chunk(self.n_se_views, dim=1)
            drugs = sum(drug_chunks)
            sides = sum(side_chunks)
        else:
            b = drug_features.shape[0]
            drugs_3d = drug_features.view(b, self.n_drug_views, self.drugs_inputdim)
            sides_3d = side_features.view(b, self.n_se_views, self.sides_inputdim)

            if d_avail is None:
                d_avail = (drugs_3d.abs().sum(dim=-1) > 1e-12).to(drug_features.dtype)
            if s_avail is None:
                s_avail = (sides_3d.abs().sum(dim=-1) > 1e-12).to(side_features.dtype)

            w_d = self._masked_softmax(self.drug_view_logits, d_avail)  # (B, V_d)
            w_s = self._masked_softmax(self.se_view_logits, s_avail)    # (B, V_s)

            drugs = (drugs_3d * w_d.unsqueeze(-1)).sum(dim=1)           # (B, 750)
            sides = (sides_3d * w_s.unsqueeze(-1)).sum(dim=1)           # (B, 994)

        add_features = torch.cat((drugs, sides), dim=1)
        x = self.l1(add_features)
        x = self.attention(x)
        x = self.l2(x)
        rec_add = self.l3(x) if self.l3 is not None else None
        return x, rec_add


class Preprocess(nn.Module):
    """
    Preprocess module before CrossProduction supporting:
      - Standard 16 raw branches: `Linear(750/994, 128) -> BN -> LeakyReLU -> Dropout` (`use_cnn_shared=False`)
      - Shared-embedding branches (`slim_all_cnnshared`, `use_cnn_shared=True`):
        takes `e_v` (64-D) and applies `Linear(64, 128) -> BN -> LeakyReLU -> Dropout` per view.
    """

    def __init__(
        self,
        drug_inputdim: int,
        side_inputdim: int,
        n_drug_views: int,
        n_se_views: int,
        embeddim: int,
        dropout: float,
        use_cnn_shared: bool = False,
        shared_emb_dim: int = 64,
    ):
        super().__init__()
        self.use_cnn_shared = bool(use_cnn_shared)
        self.drug_inputdim = int(shared_emb_dim) if self.use_cnn_shared else int(drug_inputdim)
        self.side_inputdim = int(shared_emb_dim) if self.use_cnn_shared else int(side_inputdim)
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
        self,
        drug_features: torch.Tensor,
        side_features: torch.Tensor,
        e_d: Optional[torch.Tensor] = None,
        e_s: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if self.use_cnn_shared:
            assert e_d is not None and e_s is not None, "e_d and e_s required when use_cnn_shared=True"
            drugs = [self.drug_layers[v](e_d[:, v, :]) for v in range(self.n_drug_views)]
            sides = [self.side_layers[u](e_s[:, u, :]) for u in range(self.n_se_views)]
            return drugs, sides

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


class MSSFSlimModel(nn.Module):
    """
    Stage 5 model class supporting `cand_A`, `slim_nodec`, `slim_nodec_addw`, `slim_all`, and `slim_all_cnnshared`.
    When configured with `cand_A` flags, submodule instantiation order and RNG calls are 100% identical to
    `MSSFPlusModel` under `cand_A`.
    """

    def __init__(
        self,
        cfg: dict,
        n_drug_views: int = 11,
        n_se_views: int = 5,
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

        dropout = float(cfg.get('dropout', 0.4))
        gp_dim = int(cfg.get('gp', 64))
        loss_type = cfg.get('c3_loss_type', 'ce')
        out_classes = 4 if loss_type == 'corn' else 5

        self.view_dropout_prob = float(cfg.get('c1_view_dropout_prob', 0.25))
        self.profile_self_correction = bool(cfg.get('profile_self_correction', True))
        self.use_factored_crossprod = bool(cfg.get('use_factored_crossprod', True))
        self.use_bvi = False

        # Slim flags
        self.slim_use_decoder = bool(cfg.get('slim_use_decoder', True))
        self.slim_use_add_weights = bool(cfg.get('slim_use_add_weights', False))
        self.slim_use_shared_emb = bool(cfg.get('slim_use_shared_emb', False))
        self.slim_shared_emb_dim = int(cfg.get('slim_shared_emb_dim', 64))
        self.slim_use_cnn_shared = bool(cfg.get('slim_use_cnn_shared', False))

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

        self.feature_nums = self.n_drug_views * self.n_se_views
        total_views = self.n_drug_views + self.n_se_views
        compressed_con_dim = (
            total_views * self.slim_shared_emb_dim + total_views
            if self.slim_use_shared_emb
            else None
        )

        # Preserve exact module creation order of MSSFPlusModel for cand_A parity
        if self.slim_use_shared_emb:
            self.sharedViewEmbed = SharedViewEmbedding(
                n_drug_views=self.n_drug_views,
                n_se_views=self.n_se_views,
                drug_dim=N_DRUGS,
                se_dim=N_SIDES,
                embed_dim=self.slim_shared_emb_dim,
            )
        else:
            self.sharedViewEmbed = None

        self.encoderConnection = EncoderConnection(
            drugs_inputdim=N_DRUGS * self.n_drug_views,
            sides_inputdim=N_SIDES * self.n_se_views,
            latent_dim=256,
            feature_dim=128,
            heads=4,
            dropout=dropout,
            use_decoder=self.slim_use_decoder,
            compressed_inputdim=compressed_con_dim,
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
            use_decoder=self.slim_use_decoder,
            use_add_weights=self.slim_use_add_weights,
        )
        self.preprocess = Preprocess(
            drug_inputdim=N_DRUGS,
            side_inputdim=N_SIDES,
            n_drug_views=self.n_drug_views,
            n_se_views=self.n_se_views,
            embeddim=128,
            dropout=dropout,
            use_cnn_shared=self.slim_use_cnn_shared,
            shared_emb_dim=self.slim_shared_emb_dim,
        )
        self.crossProduction = CrossProduction(
            cross_dim=128,
            feature_dim=128,
            input_channel=self.feature_nums,
            use_factored=self.use_factored_crossprod,
        )
        self.attention = Attention(inputdim=128 * 3, heads=4)
        self.det_projector = nn.Sequential(
            nn.Linear(128 * 3, gp_dim),
            nn.Dropout(dropout),
        )
        self.classifier = Classifier(latent_dim=gp_dim, classes=out_classes, dropout=dropout)

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        # C1 Training-time view dropout on profile views (D9, D10)
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

        return drugs, sides, d_avail, s_avail

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
        mod_drugs, mod_sides, d_avail, s_avail = self._apply_c1_input_processing(
            drugs, sides, drug_avail, se_avail, fallback_d9, fallback_d10,
            drug_idx=drug_idx, se_idx=se_idx,
        )

        if self.sharedViewEmbed is not None:
            e_d, e_s = self.sharedViewEmbed(mod_drugs, mod_sides)
        else:
            e_d, e_s = None, None

        feature1, rec_con = self.encoderConnection(
            mod_drugs, mod_sides, e_d=e_d, e_s=e_s, d_avail=d_avail, s_avail=s_avail
        )
        feature2, rec_add = self.encoderAddition(
            mod_drugs, mod_sides, d_avail=d_avail, s_avail=s_avail
        )
        prep_drugs, prep_sides = self.preprocess(
            mod_drugs, mod_sides, e_d=e_d, e_s=e_s
        )
        feature3 = self.crossProduction(prep_drugs, prep_sides)

        features = torch.cat((feature1, feature2, feature3), dim=1)
        features = self.attention(features)

        latent_features = self.det_projector(features)
        mu = latent_features
        logvar = torch.zeros_like(latent_features)

        logits = self.classifier(latent_features)
        return logits, rec_con, rec_add, mu, logvar, mod_drugs, mod_sides, None


def get_named_children_param_breakdown(model: nn.Module) -> Dict[str, int]:
    """
    Compute parameter counts for each direct child in `model.named_children()` plus `TOTAL`.
    """
    breakdown: Dict[str, int] = {}
    for name, child in model.named_children():
        breakdown[name] = sum(p.numel() for p in child.parameters())
    breakdown['TOTAL'] = sum(p.numel() for p in model.parameters())
    return breakdown
