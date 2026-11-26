"""
Nidaan-Rx v2.0 — Phase 4B: End-to-End Multimodal Inference & Clinical Triage Orchestrator.

Pipeline:
  1. Image Sanitization & Tone-Calibrated Quality Gate (1-3 views)
  2. Frozen Vision Foundation Encoder (PanDerm-Base primary / DINOv2-Small hedge)
  3. Stage-4 Residual Adapter + Gated Attention-MIL + Gated Cross-Attention Fusion
  4. Calibrated Probabilities + Mahalanobis OOD Rejection Gate
  5. Dynamic India-Specific Trigger Bucket Selection (Stage-1 Triage)
  6. 12-Gate Clinical Safety Stack + PMBJP Jan Aushadhi Generic Equivalence (Stage-2 Assess)
"""

from __future__ import annotations

import inspect
import io
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from app.core.atomic_triggers import expand_trigger_bucket
from app.core.knowledge_graph import MedicineCatalog, select_dynamic_trigger_buckets
from app.core.safety_gates import evaluate_clinical_case
from app.core.schemas import (
    CaseContext,
    ClinicalAssessmentResponse,
    ModelInferenceOutput,
    PatientProfile,
    TriggerBucket,
)
from app.models.fusion_net import MahalanobisOODDetector, NidaanMultimodalNet
from app.vision.extractor import BackboneName, FrozenFeatureExtractor, resolve_device
from app.vision.quality_gate import evaluate_image_quality, sanitize_and_compress_image

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_medicine_catalog() -> MedicineCatalog:
    """Load processed PMBJP Jan Aushadhi generics and 1mg commercial/DDI catalogs."""
    pmbjp_candidates = [
        PROJECT_ROOT / "data" / "processed" / "pmbjp_dermatology_parsed.csv",
        PROJECT_ROOT / "backend" / "data" / "processed" / "pmbjp_dermatology_parsed.csv",
    ]
    onemg_candidates = [
        PROJECT_ROOT / "data" / "processed" / "1mg_dermatology_parsed.csv",
        PROJECT_ROOT / "backend" / "data" / "processed" / "1mg_dermatology_parsed.csv",
    ]
    pmbjp_csv = next((p for p in pmbjp_candidates if p.exists()), None)
    onemg_csv = next((p for p in onemg_candidates if p.exists()), None)
    return MedicineCatalog(pmbjp_csv=pmbjp_csv, onemg_csv=onemg_csv)

def encode_clinical_tabular(profile: PatientProfile, case: CaseContext, dim: int = 32) -> tuple[np.ndarray, bool]:
    """
    Encodes PatientProfile + CaseContext into the 32-dim normalized float32 clinical vector
    expected by Stage-4 GatedCrossAttentionFusion.
    """
    vec = np.zeros(dim, dtype=np.float32)

    age = float(getattr(profile, "age", 30) or 30)
    vec[0] = np.clip(age / 100.0, 0.0, 1.0)

    sex = str(getattr(profile, "biological_sex", "") or "").lower()
    vec[1] = 1.0 if sex.startswith("f") else 0.0
    vec[2] = 1.0 if sex.startswith("m") else 0.0

    fst = getattr(profile, "fitzpatrick_tone", None)
    if fst is not None:
        try:
            vec[3] = np.clip((float(fst) - 1.0) / 5.0, 0.0, 1.0)
        except (TypeError, ValueError):
            vec[3] = 0.6
    else:
        vec[3] = 0.6

    mst = getattr(profile, "monk_skin_tone", None)
    if mst is not None:
        try:
            vec[4] = np.clip((float(mst) - 1.0) / 9.0, 0.0, 1.0)
        except (TypeError, ValueError):
            vec[4] = 0.55
    else:
        vec[4] = 0.55

    dur = float(getattr(case, "duration_days", 7) or 7)
    vec[5] = np.clip(np.log1p(max(dur, 0.0)) / np.log1p(365.0), 0.0, 1.0)
    vec[6] = np.clip(float(getattr(case, "itch_score", 0) or 0) / 10.0, 0.0, 1.0)
    vec[7] = np.clip(float(getattr(case, "pain_burning_score", 0) or 0) / 10.0, 0.0, 1.0)
    vec[8] = 1.0 if getattr(case, "has_scaling", False) else 0.0
    vec[9] = 1.0 if getattr(case, "compromised_barrier", False) else 0.0

    rf = getattr(case, "red_flags", None)
    if rf is not None:
        vec[10] = 1.0 if getattr(rf, "fever", False) else 0.0
        vec[11] = 1.0 if getattr(rf, "rapidly_spreading_redness_warmth", False) else 0.0
        vec[12] = 1.0 if getattr(rf, "pus_oozing_bleeding", False) else 0.0
        vec[13] = 1.0 if getattr(rf, "severe_pain", False) else 0.0
        vec[14] = 1.0 if getattr(rf, "changing_bleeding_dark_mole", False) else 0.0

    site_val = str(getattr(case, "anatomical_site", "") or "").lower()
    site_buckets = [
        ("face", 15),
        ("scalp", 16),
        ("trunk", 17),
        ("groin", 18),
        ("fold", 18),
        ("arm", 19),
        ("hand", 20),
        ("leg", 21),
        ("foot", 22),
        ("neck", 23),
        ("nail", 24),
    ]
    for token, idx in site_buckets:
        if token in site_val and idx < dim:
            vec[idx] = 1.0

    climate = [str(t).lower() for t in (getattr(profile, "climate_tags", None) or [])]
    vec[25] = 1.0 if any("humid" in c or "monsoon" in c for c in climate) else 0.0
    vec[26] = 1.0 if any("hot" in c or "sun" in c or "summer" in c for c in climate) else 0.0
    vec[27] = 1.0 if any("dry" in c or "winter" in c for c in climate) else 0.0

    stress = str(getattr(profile, "stress_level", "") or "").lower()
    vec[28] = 1.0 if "high" in stress else (0.5 if "mod" in stress else 0.0)
    vec[29] = 1.0 if getattr(profile, "is_pregnant", False) else 0.0
    vec[30] = 1.0 if getattr(profile, "is_lactating", False) else 0.0
    vec[31] = 1.0 if len(getattr(case, "active_triggers", None) or []) > 0 else 0.0

    has_signal = bool(np.any(vec[5:] > 0.0))
    return vec, has_signal


@dataclass
class VisionPipelineResult:
    inference_output: ModelInferenceOutput
    view_attention_weights: list[float]
    quality_reports: list[dict[str, Any]]
    all_views_passed_quality: bool
    backbone: str
    latency_ms: float


class NidaanInferenceService:
    """Lazy-loaded multimodal inference engine for FastAPI."""

    def __init__(
        self,
        backbone: BackboneName | None = None,
        artifacts_dir: Path | str = ARTIFACTS_DIR,
        device: str | None = None,
        extractor: FrozenFeatureExtractor | None = None,
    ) -> None:
        self.backbone: BackboneName = backbone or os.getenv("NIDAAN_BACKBONE", "panderm_base")  # type: ignore[assignment]
        self.artifacts_dir = Path(artifacts_dir)
        self.device = resolve_device(device)
        self._extractor: FrozenFeatureExtractor | None = extractor
        self._catalog: MedicineCatalog = _load_medicine_catalog()

        ckpt_path = self.artifacts_dir / f"nidaan_v2_{self.backbone}_stage4.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Stage-4 checkpoint not found at {ckpt_path}. Run Phase 3 train_ablation.py first."
            )

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.embed_dim: int = int(ckpt["embed_dim"])
        self.tabular_dim: int = int(ckpt["tabular_dim"])
        self.tier1_classes: list[str] = list(ckpt["tier1_classes"])
        self.tier2_classes: list[str] = list(ckpt["tier2_classes"])
        self.canonical_classes: list[str] = list(ckpt["canonical_classes"])
        self.test_metrics: dict[str, Any] = dict(ckpt.get("test_metrics", {}))

        self.fusion_model = NidaanMultimodalNet(
            embed_dim=self.embed_dim,
            tabular_dim=self.tabular_dim,
            num_tier1=len(self.tier1_classes),
            num_tier2=len(self.tier2_classes),
            num_canonical=len(self.canonical_classes),
            stage=ckpt.get("stage", "stage4_multimodal_fusion"),
        )
        self.fusion_model.load_state_dict(ckpt["model_state_dict"])
        self.fusion_model = self.fusion_model.to(self.device).eval()

        self.ood_detector = MahalanobisOODDetector.from_state_dict(ckpt["ood_detector"])

    @property
    def extractor(self) -> FrozenFeatureExtractor:
        if self._extractor is None:
            self._extractor = FrozenFeatureExtractor(backbone=self.backbone, device=str(self.device))
        return self._extractor

    def _prepare_images(self, raw_images: list[bytes]) -> tuple[list[Image.Image], list[dict[str, Any]], bool]:
        if not raw_images or len(raw_images) > 3:
            raise ValueError("Provide between 1 and 3 skin photographs.")

        pil_images: list[Image.Image] = []
        quality_reports: list[dict[str, Any]] = []
        all_passed = True

        for idx, raw in enumerate(raw_images):
            sanitized = sanitize_and_compress_image(raw)
            clean_bytes = sanitized[0] if isinstance(sanitized, tuple) else sanitized
            im = Image.open(io.BytesIO(clean_bytes)).convert("RGB")
            qg = evaluate_image_quality(im)
            rep = qg.to_dict()
            rep["view_index"] = idx
            quality_reports.append(rep)
            if not qg.passed:
                all_passed = False
            pil_images.append(im)

        return pil_images, quality_reports, all_passed

    @torch.inference_mode()
    def run_vision_stage(
        self,
        raw_images: list[bytes],
        profile: PatientProfile,
        case: CaseContext,
        enforce_quality_gate: bool = True,
    ) -> VisionPipelineResult:
        t0 = time.perf_counter()
        pil_images, quality_reports, all_passed = self._prepare_images(raw_images)

        if enforce_quality_gate and not all_passed:
            failed = [r for r in quality_reports if not r.get("passed", False)]
            cues = []
            for r in failed:
                cues.extend(r.get("retake_cues") or r.get("issues") or [])
            raise ValueError(
                f"Image quality gate failed: {'; '.join(dict.fromkeys(cues)) or 'Retake photo in better lighting'}"
            )

        feats = self.extractor.encode_images(pil_images)
        n_views = feats.shape[0]
        max_views = 3

        padded = torch.zeros((1, max_views, self.embed_dim), device=self.device, dtype=torch.float32)
        padded[0, :n_views, :] = feats
        view_mask = torch.zeros((1, max_views), device=self.device, dtype=torch.bool)
        view_mask[0, :n_views] = True

        tab_np, has_tab_flag = encode_clinical_tabular(profile, case, dim=self.tabular_dim)
        tab_t = torch.from_numpy(tab_np).unsqueeze(0).to(self.device)
        has_tab_t = torch.tensor([has_tab_flag], device=self.device, dtype=torch.bool)

        out = self.fusion_model(padded, view_mask, tab_t, has_tabular=has_tab_t)
        can_probs = torch.softmax(out["logits_canonical"], dim=-1)[0].cpu().numpy()
        fam_probs = torch.softmax(out["logits_tier1"], dim=-1)[0].cpu().numpy()
        attn_w = out["attn_weights"][0, :n_views].cpu().numpy().tolist()
        rep_np = out["rep"].cpu().numpy()

        ood_dists, ood_flags = self.ood_detector.score(rep_np)
        ood_score = float(ood_dists[0])
        is_ood = bool(ood_flags[0])

        condition_probs = {
            cid: round(float(p), 6) for cid, p in zip(self.canonical_classes, can_probs.tolist())
        }
        family_probs = {
            fid: round(float(p), 6) for fid, p in zip(self.tier1_classes, fam_probs.tolist())
        }

        stage1_out = ModelInferenceOutput(
            condition_probs=condition_probs,
            family_probs=family_probs,
            is_ood=is_ood,
            ood_score=round(ood_score, 4),
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        return VisionPipelineResult(
            inference_output=stage1_out,
            view_attention_weights=[round(float(w), 4) for w in attn_w],
            quality_reports=quality_reports,
            all_views_passed_quality=all_passed,
            backbone=self.backbone,
            latency_ms=elapsed_ms,
        )

    def get_dynamic_triggers(
        self,
        stage1_output: ModelInferenceOutput,
        profile: PatientProfile,
        top_k_conditions: int = 3,
        n_cards: int = 4,
    ) -> list[TriggerBucket]:
        raw_buckets = select_dynamic_trigger_buckets(
            stage1_output=stage1_output,
            profile=profile,
            top_k_conditions=top_k_conditions,
            n_cards=n_cards + 3,
        )
        raw_sex = getattr(profile, "biological_sex", "")
        if hasattr(raw_sex, "value"):
            raw_sex = raw_sex.value
        sex = str(raw_sex or "").strip().lower()
        age = int(getattr(profile, "age", 25) or 25)
        is_male = sex == "male" or sex.endswith(".male")

        female_only_re = re.compile(
            r"(pre-?menstrual|menstrual|period|pcos|pcod|pregnan|postpartum|lactat|breastfeed|menopaus|ovarian)",
            re.IGNORECASE,
        )
        preg_re = re.compile(r"(pregnan|postpartum|lactat|breastfeed)", re.IGNORECASE)
        adolescent_re = re.compile(r"(adolescent|teen|puberty|14.?19|12.?19|13.?19)", re.IGNORECASE)

        filtered: list[TriggerBucket] = []
        for bucket in (expand_trigger_bucket(b) for b in raw_buckets):
            kept = []
            for tr in bucket.triggers:
                blob = f"{tr.trigger_id} {tr.label} {tr.description}"
                if is_male and female_only_re.search(blob):
                    continue
                if not is_male and (age < 13 or age > 55) and preg_re.search(blob):
                    continue
                if (age < 11 or age > 21) and adolescent_re.search(blob):
                    continue
                kept.append(tr)
            if kept:
                filtered.append(bucket.model_copy(update={"triggers": kept}))
        return filtered[:n_cards]

    def run_clinical_assessment(
        self,
        profile: PatientProfile,
        case: CaseContext,
        stage1_output: ModelInferenceOutput,
    ) -> ClinicalAssessmentResponse:
        return evaluate_clinical_case(
            profile=profile,
            case=case,
            stage1_output=stage1_output,
            catalog=self._catalog,
        )
