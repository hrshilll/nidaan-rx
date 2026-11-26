"""
Nidaan-Rx v2.0 — Phase 3: Custom PyTorch Multimodal Architecture & Mahalanobis OOD Gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

AblationStage = Literal[
    "stage1_linear_baseline",
    "stage2_attention_mil",
    "stage3_adapter_hierarchical_focal",
    "stage4_multimodal_fusion",
]


class ResidualBottleneckAdapter(nn.Module):
    def __init__(self, embed_dim: int = 768, bottleneck_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.down = nn.Linear(embed_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, embed_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.up(self.drop(self.act(self.down(self.norm(x)))))
        return F.normalize(x + residual, p=2, dim=-1)


class GatedAttentionMIL(nn.Module):
    def __init__(self, embed_dim: int = 768, attn_dim: int = 192, dropout: float = 0.2) -> None:
        super().__init__()
        self.attn_v = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Tanh())
        self.attn_u = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Sigmoid())
        self.drop = nn.Dropout(dropout)
        self.attn_w = nn.Linear(attn_dim, 1)

    def forward(self, views: torch.Tensor, view_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gated = self.drop(self.attn_v(views) * self.attn_u(views))
        scores = self.attn_w(gated).squeeze(-1)
        scores = scores.masked_fill(~view_mask, -1e9)
        weights = torch.softmax(scores, dim=-1) * view_mask.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        pooled = torch.sum(views * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class GatedCrossAttentionFusion(nn.Module):
    """
    Flamingo-style tanh(alpha) zero-gated cross-attention late fusion.
    Masks out missing tabular rows (has_tabular=False) so image-only cases
    retain the exact Stage-3 visual representation while multimodal cases
    receive calibrated clinical-context refinement.
    """

    def __init__(
        self,
        img_dim: int = 768,
        tab_in_dim: int = 32,
        tab_hidden_dim: int = 128,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.tab_encoder = nn.Sequential(
            nn.Linear(tab_in_dim, tab_hidden_dim),
            nn.LayerNorm(tab_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(tab_hidden_dim, tab_hidden_dim),
            nn.LayerNorm(tab_hidden_dim),
            nn.GELU(),
        )
        self.img_q = nn.Linear(img_dim, tab_hidden_dim)
        self.tab_k = nn.Linear(tab_hidden_dim, tab_hidden_dim)
        self.tab_v = nn.Linear(tab_hidden_dim, tab_hidden_dim)

        self.tab_q = nn.Linear(tab_hidden_dim, tab_hidden_dim)
        self.img_k = nn.Linear(img_dim, tab_hidden_dim)
        self.img_v = nn.Linear(img_dim, tab_hidden_dim)

        self.gate_net = nn.Sequential(
            nn.Linear(img_dim + tab_hidden_dim, tab_hidden_dim),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(tab_hidden_dim * 2, img_dim // 2),
            nn.LayerNorm(img_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(img_dim // 2, img_dim),
        )
        self.res_alpha = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        v_img: torch.Tensor,
        x_tab: torch.Tensor,
        has_tabular: torch.Tensor | None = None,
    ) -> torch.Tensor:
        v_tab = self.tab_encoder(x_tab)
        scale = v_tab.shape[-1] ** -0.5

        w_it = torch.sigmoid((self.img_q(v_img) * self.tab_k(v_tab)).sum(dim=-1, keepdim=True) * scale)
        ctx_from_tab = w_it * self.tab_v(v_tab)

        w_ti = torch.sigmoid((self.tab_q(v_tab) * self.img_k(v_img)).sum(dim=-1, keepdim=True) * scale)
        ctx_from_img = w_ti * self.img_v(v_img)

        gate = self.gate_net(torch.cat([v_img, v_tab], dim=-1))
        gated_tab = gate * (v_tab + ctx_from_tab)
        delta = self.out_proj(torch.cat([gated_tab, ctx_from_img], dim=-1))

        if has_tabular is not None:
            delta = delta * has_tabular.float().unsqueeze(-1)

        return F.normalize(v_img + torch.tanh(self.res_alpha) * delta, p=2, dim=-1)


class NidaanMultimodalNet(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        tabular_dim: int = 32,
        num_tier1: int = 10,
        num_tier2: int = 54,
        num_canonical: int = 15,
        stage: AblationStage = "stage4_multimodal_fusion",
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.stage = stage
        self.embed_dim = embed_dim

        self.use_adapter = stage in {"stage3_adapter_hierarchical_focal", "stage4_multimodal_fusion"}
        self.use_mil = stage in {
            "stage2_attention_mil",
            "stage3_adapter_hierarchical_focal",
            "stage4_multimodal_fusion",
        }
        self.use_fusion = stage == "stage4_multimodal_fusion"

        self.adapter = ResidualBottleneckAdapter(embed_dim, bottleneck_dim=256, dropout=dropout) if self.use_adapter else nn.Identity()
        self.mil = GatedAttentionMIL(embed_dim, attn_dim=192, dropout=dropout) if self.use_mil else None
        self.fusion = GatedCrossAttentionFusion(embed_dim, tabular_dim, 128, dropout=0.35) if self.use_fusion else None

        self.head_tier1 = nn.Linear(embed_dim, num_tier1)
        self.head_tier2 = nn.Linear(embed_dim, num_tier2)
        self.head_canonical = nn.Linear(embed_dim, num_canonical)

        self.register_buffer("temperature", torch.tensor(1.0, dtype=torch.float32))

    def encode_representation(
        self,
        views: torch.Tensor,
        view_mask: torch.Tensor,
        tabular: torch.Tensor | None = None,
        has_tabular: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.adapter(views) if self.use_adapter else views

        if self.use_mil and self.mil is not None:
            v_img, attn_weights = self.mil(h, view_mask)
        else:
            mask_f = view_mask.float()
            denom = mask_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
            attn_weights = mask_f / denom
            v_img = torch.sum(h * attn_weights.unsqueeze(-1), dim=1)

        v_img = F.normalize(v_img, p=2, dim=-1)

        if self.use_fusion and self.fusion is not None:
            if tabular is None:
                tabular = torch.zeros((v_img.shape[0], 32), device=v_img.device, dtype=v_img.dtype)
                if has_tabular is None:
                    has_tabular = torch.zeros((v_img.shape[0],), device=v_img.device, dtype=torch.bool)
            rep = self.fusion(v_img, tabular, has_tabular=has_tabular)
        else:
            rep = v_img

        return rep, attn_weights

    def forward(
        self,
        views: torch.Tensor,
        view_mask: torch.Tensor,
        tabular: torch.Tensor | None = None,
        has_tabular: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        rep, attn_weights = self.encode_representation(views, view_mask, tabular, has_tabular)
        temp = self.temperature.clamp(min=0.05, max=10.0)
        return {
            "rep": rep,
            "attn_weights": attn_weights,
            "logits_tier1": self.head_tier1(rep) / temp,
            "logits_tier2": self.head_tier2(rep) / temp,
            "logits_canonical": self.head_canonical(rep) / temp,
        }


class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None, label_smoothing: float = 0.03) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


class JointHierarchicalLoss(nn.Module):
    def __init__(
        self,
        use_focal: bool = True,
        tier1_weights: torch.Tensor | None = None,
        canonical_weights: torch.Tensor | None = None,
        w_tier1: float = 0.35,
        w_tier2: float = 0.25,
        w_canonical: float = 0.40,
    ) -> None:
        super().__init__()
        self.w_tier1 = w_tier1
        self.w_tier2 = w_tier2
        self.w_canonical = w_canonical
        if use_focal:
            self.loss_t1 = FocalCrossEntropyLoss(gamma=2.0, weight=tier1_weights)
            self.loss_t2 = FocalCrossEntropyLoss(gamma=2.0)
            self.loss_can = FocalCrossEntropyLoss(gamma=2.0, weight=canonical_weights)
        else:
            self.loss_t1 = nn.CrossEntropyLoss(weight=tier1_weights)
            self.loss_t2 = nn.CrossEntropyLoss()
            self.loss_can = nn.CrossEntropyLoss(weight=canonical_weights)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        tier1_y: torch.Tensor,
        tier2_y: torch.Tensor,
        canonical_y: torch.Tensor,
    ) -> torch.Tensor:
        l1 = self.loss_t1(outputs["logits_tier1"], tier1_y)
        l2 = self.loss_t2(outputs["logits_tier2"], tier2_y)
        lc = self.loss_can(outputs["logits_canonical"], canonical_y)
        return self.w_tier1 * l1 + self.w_tier2 * l2 + self.w_canonical * lc


@dataclass
class MahalanobisOODDetector:
    centroids: np.ndarray
    precision: np.ndarray
    threshold: float

    @classmethod
    def fit(
        cls,
        train_reps: np.ndarray,
        train_labels: np.ndarray,
        val_reps: np.ndarray,
        percentile: float = 95.0,
        shrinkage: float = 0.15,
    ) -> "MahalanobisOODDetector":
        classes = np.unique(train_labels)
        d = train_reps.shape[1]
        centroids = np.zeros((int(classes.max()) + 1, d), dtype=np.float32)
        centered_parts = []
        for c in classes:
            mask = train_labels == c
            mu = train_reps[mask].mean(axis=0)
            centroids[int(c)] = mu
            centered_parts.append(train_reps[mask] - mu)

        centered = np.concatenate(centered_parts, axis=0)
        cov = np.cov(centered, rowvar=False).astype(np.float32)
        diag_mean = float(np.trace(cov) / max(1, d))
        cov_reg = (1.0 - shrinkage) * cov + shrinkage * diag_mean * np.eye(d, dtype=np.float32)
        precision = np.linalg.pinv(cov_reg).astype(np.float32)

        val_scores = cls._batch_min_dist(val_reps, centroids, precision)
        thresh = float(np.percentile(val_scores, percentile))
        return cls(centroids=centroids, precision=precision, threshold=thresh)

    @staticmethod
    def _batch_min_dist(reps: np.ndarray, centroids: np.ndarray, precision: np.ndarray) -> np.ndarray:
        diff = reps[:, None, :] - centroids[None, :, :]
        left = np.einsum("ncd,dk->nck", diff, precision)
        sq_dists = np.sum(left * diff, axis=-1)
        return np.sqrt(np.maximum(sq_dists.min(axis=-1), 0.0))

    def score(self, reps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dists = self._batch_min_dist(reps, self.centroids, self.precision)
        return dists, dists > self.threshold

    def state_dict(self) -> dict[str, np.ndarray | float]:
        return {
            "centroids": self.centroids,
            "precision": self.precision,
            "threshold": self.threshold,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "MahalanobisOODDetector":
        return cls(
            centroids=np.asarray(state["centroids"], dtype=np.float32),
            precision=np.asarray(state["precision"], dtype=np.float32),
            threshold=float(state["threshold"]),
        )
