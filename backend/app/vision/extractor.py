"""
Nidaan-Rx v2.0 — Phase 2/4 Vision Backbone Extractor (Local M2 MPS / CUDA / CPU).

Supports bit-exact local inference matching the Colab Phase 2B extraction pipeline:
  - panderm_base : Monash CAEv2/BEiT ViT-B/16 loaded directly from HF safetensors
                   (redlessone/DermLIP_PanDerm-base-w-PubMed-256) — 768-dim L2-normalized.
  - dinov2_small : Apache-2.0 commercial hedge via timm — 384-dim L2-normalized.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps

BackboneName = Literal["panderm_base", "dinov2_small", "dinov2_base"]


def resolve_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _PanDermAttn(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.v_bias = nn.Parameter(torch.zeros(dim))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias), dim=0)
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        qkv = qkv.reshape(bsz, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(bsz, seq_len, dim)
        return self.proj(out)


class _PanDermMLP(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _PanDermBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_hidden: int, has_gamma: bool) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _PanDermAttn(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _PanDermMLP(dim, mlp_hidden)
        self.gamma_1 = nn.Parameter(torch.ones(dim)) if has_gamma else None
        self.gamma_2 = nn.Parameter(torch.ones(dim)) if has_gamma else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gamma_1 is not None:
            x = x + self.gamma_1 * self.attn(self.norm1(x))
            x = x + self.gamma_2 * self.mlp(self.norm2(x))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class _PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class PanDermViT(nn.Module):
    def __init__(self, vis_state: dict[str, torch.Tensor]) -> None:
        super().__init__()
        w_patch = vis_state["patch_embed.proj.weight"]
        embed_dim = int(w_patch.shape[0])
        patch_size = int(w_patch.shape[2])
        depth = max(int(k.split(".")[1]) for k in vis_state if k.startswith("blocks.")) + 1
        mlp_hidden = int(vis_state["blocks.0.mlp.fc1.weight"].shape[0])
        has_gamma = "blocks.0.gamma_1" in vis_state
        num_heads = embed_dim // 64
        num_pos = int(vis_state["pos_embed"].shape[1]) if "pos_embed" in vis_state else 197
        grid = int(round(math.sqrt(num_pos - 1 if num_pos % 2 == 1 else num_pos)))

        self.img_size = grid * patch_size
        self.embed_dim = embed_dim
        self.patch_embed = _PatchEmbed(patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if "cls_token" in vis_state else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_pos, embed_dim)) if "pos_embed" in vis_state else None
        self.blocks = nn.ModuleList([_PanDermBlock(embed_dim, num_heads, mlp_hidden, has_gamma) for _ in range(depth)])
        self.fc_norm = nn.LayerNorm(embed_dim, eps=1e-6) if "fc_norm.weight" in vis_state else None
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6) if "norm.weight" in vis_state else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        if self.pos_embed is not None:
            if x.shape[1] == self.pos_embed.shape[1]:
                x = x + self.pos_embed
            elif self.cls_token is not None and x.shape[1] - 1 == self.pos_embed.shape[1]:
                x[:, 1:, :] = x[:, 1:, :] + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        if self.fc_norm is not None:
            return self.fc_norm(x[:, 1:, :].mean(dim=1) if self.cls_token is not None else x.mean(dim=1))
        if self.norm is not None:
            x = self.norm(x)
        return x[:, 0] if self.cls_token is not None else x.mean(dim=1)


def preprocess_pil(
    im: Image.Image,
    size: int = 224,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    im = ImageOps.exif_transpose(im).convert("RGB")
    w, h = im.size
    short = min(w, h)
    im = im.crop(((w - short) // 2, (h - short) // 2, (w + short) // 2, (h + short) // 2)).resize(
        (size, size), Image.Resampling.BICUBIC
    )
    arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = (arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return torch.from_numpy(arr.transpose(2, 0, 1))


class FrozenFeatureExtractor:
    """Unified local frozen vision encoder for live FastAPI inference."""

    PANDERM_REPO_ID = "redlessone/DermLIP_PanDerm-base-w-PubMed-256"

    def __init__(
        self,
        backbone: BackboneName = "panderm_base",
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        self.backbone = backbone
        self.device = resolve_device(device)
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.img_size = 224

        if backbone == "panderm_base":
            self.model = self._load_panderm(checkpoint_path)
            self.embed_dim = self.model.embed_dim
            self.img_size = self.model.img_size
        elif backbone in {"dinov2_small", "dinov2_base"}:
            self.model = self._load_dinov2(backbone)
            self.embed_dim = 384 if backbone == "dinov2_small" else 768
            self.img_size = 224
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.model = self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _load_panderm(self, checkpoint_path: str | Path | None) -> PanDermViT:
        from safetensors.torch import load_file as load_safetensors

        if checkpoint_path and Path(checkpoint_path).exists():
            w_path = Path(checkpoint_path)
            cfg_path = w_path.parent / "open_clip_config.json"
        else:
            from huggingface_hub import hf_hub_download

            cfg_path = Path(hf_hub_download(repo_id=self.PANDERM_REPO_ID, filename="open_clip_config.json"))
            w_path = Path(hf_hub_download(repo_id=self.PANDERM_REPO_ID, filename="open_clip_model.safetensors"))

        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            pp = cfg.get("preprocess_cfg", {})
            self.mean = tuple(pp.get("mean", self.mean))
            self.std = tuple(pp.get("std", self.std))

        full_sd = load_safetensors(str(w_path))
        vis_sd = {k[len("visual."):]: v for k, v in full_sd.items() if k.startswith("visual.")}
        if not vis_sd:
            vis_sd = full_sd
        backbone_sd = {k: v for k, v in vis_sd.items() if not k.startswith("head.") and k != "proj"}

        model = PanDermViT(backbone_sd)
        missing, _ = model.load_state_dict(backbone_sd, strict=False)
        if missing:
            raise RuntimeError(f"PanDerm-Base missing weights: {missing[:5]}")
        return model

    def _load_dinov2(self, backbone: BackboneName) -> nn.Module:
        import timm

        timm_name = (
            "vit_small_patch14_dinov2.lvd142m"
            if backbone == "dinov2_small"
            else "vit_base_patch14_dinov2.lvd142m"
        )
        return timm.create_model(timm_name, pretrained=True, num_classes=0, img_size=224)

    def preprocess(self, im: Image.Image) -> torch.Tensor:
        return preprocess_pil(im, size=self.img_size, mean=self.mean, std=self.std)

    @torch.inference_mode()
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        """Returns L2-normalized float32 embeddings of shape (N, embed_dim) on self.device."""
        if not images:
            return torch.empty((0, self.embed_dim), device=self.device, dtype=torch.float32)
        batch = torch.stack([self.preprocess(im) for im in images], dim=0).to(self.device)
        feats = self.model(batch)
        if feats.ndim == 3:
            feats = feats[:, 0, :]
        return F.normalize(feats.float(), p=2, dim=-1)


from dataclasses import asdict as _asdict, dataclass as _dataclass


@_dataclass(frozen=True)
class BackboneSpec:
    name: str
    embed_dim: int
    img_size: int
    input_size: int
    license: str

    @property
    def license_id(self) -> str:
        return self.license
    commercial_safe: bool
    source: str

    def __getitem__(self, key: str):
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def keys(self):
        return _asdict(self).keys()


BACKBONE_SPECS: dict[str, BackboneSpec] = {
    "panderm_base": BackboneSpec(
        name="panderm_base",
        embed_dim=768,
        img_size=224,
        input_size=224,
        license="CC-BY-NC-ND-4.0",
        commercial_safe=False,
        source="redlessone/DermLIP_PanDerm-base-w-PubMed-256",
    ),
    "dinov2_small": BackboneSpec(
        name="dinov2_small",
        embed_dim=384,
        img_size=224,
        input_size=224,
        license="Apache-2.0",
        commercial_safe=True,
        source="vit_small_patch14_dinov2.lvd142m",
    ),
    "dinov2_base": BackboneSpec(
        name="dinov2_base",
        embed_dim=768,
        img_size=224,
        input_size=224,
        license="Apache-2.0",
        commercial_safe=True,
        source="vit_base_patch14_dinov2.lvd142m",
    ),
}
