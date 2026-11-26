"""
End-to-end FastAPI integration tests for Nidaan-Rx v2.0 Phase 4B.
Exercises the real Stage-4 PanDerm-Base checkpoint, real 12-Gate Safety Stack, real PMBJP catalog,
and SQLite persistence with a deterministic stub ViT encoder so pytest finishes in <2 seconds.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from app import main as api_main
from app.core.schemas import AnatomicalSite
from app.db import CaseRepository
from app.services.inference import NidaanInferenceService


class _StubVisionExtractor:
    def __init__(self, embed_dim: int = 768, device: torch.device | None = None) -> None:
        self.embed_dim = embed_dim
        self.device = device or torch.device("cpu")

    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        rng = np.random.default_rng(7)
        arr = rng.normal(size=(len(images), self.embed_dim)).astype(np.float32)
        arr /= np.linalg.norm(arr, axis=-1, keepdims=True)
        return torch.from_numpy(arr).to(self.device)


def _make_skin_like_jpeg_bytes() -> bytes:
    rng = np.random.default_rng(42)
    h, w = 320, 320
    base = np.zeros((h, w, 3), dtype=np.float32)
    base[..., 0] = 168.0
    base[..., 1] = 118.0
    base[..., 2] = 88.0
    noise = rng.normal(0.0, 14.0, size=(h, w, 1))
    yy, xx = np.ogrid[:h, :w]
    ring = ((xx - 160) ** 2 + (yy - 160) ** 2 < 65**2) & ((xx - 160) ** 2 + (yy - 160) ** 2 > 42**2)
    img_arr = np.clip(base + noise, 0, 255)
    img_arr[ring, 0] = np.clip(img_arr[ring, 0] + 35, 0, 255)
    img_arr[ring, 1] = np.clip(img_arr[ring, 1] - 25, 0, 255)
    img_arr[ring, 2] = np.clip(img_arr[ring, 2] - 20, 0, 255)
    im = Image.fromarray(img_arr.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    svc = NidaanInferenceService(
        backbone="panderm_base",
        device="cpu",
        extractor=_StubVisionExtractor(embed_dim=768, device=torch.device("cpu")),  # type: ignore[arg-type]
    )
    repo = CaseRepository(db_path=tmp_path / "test_cases.db")
    monkeypatch.setattr(api_main, "get_inference_service", lambda: svc)
    monkeypatch.setattr(api_main, "get_case_repo", lambda: repo)
    return TestClient(api_main.app)


def test_health_and_metrics(client: TestClient) -> None:
    r = client.get("/api/v2/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["active_backbone"] == "panderm_base"
    assert body["canonical_classes"] == 15

    m = client.get("/api/v2/metrics")
    assert m.status_code == 200
    assert "panderm_base" in m.json()["ablation_reports"]


def test_end_to_end_triage_assess_and_persistence(client: TestClient) -> None:
    _reg = client.post("/api/v2/auth/register", json={"email": "e2e@example.com", "password": "Password123", "full_name": "E2E User"})
    assert _reg.status_code == 200
    _hdr = {"Authorization": f"Bearer {_reg.json()['access_token']}"}

    img1 = _make_skin_like_jpeg_bytes()
    img2 = _make_skin_like_jpeg_bytes()

    default_site = list(AnatomicalSite)[0].value
    profile = {
        "age": 26,
        "biological_sex": "male",
        "weight_kg": 64.0,
        "fitzpatrick_tone": 4,
        "monk_skin_tone": 6,
    }
    case_ctx = {
        "anatomical_site": default_site,
        "duration_days": 14,
        "itch_score": 8,
        "pain_burning_score": 2,
        "has_scaling": True,
        "compromised_barrier": False,
        "red_flags": {},
        "active_triggers": [],
    }

    t_resp = client.post(
        "/api/v2/triage",
        headers=_hdr,
        files=[
            ("images", ("view1.jpg", img1, "image/jpeg")),
            ("images", ("view2.jpg", img2, "image/jpeg")),
        ],
        data={
            "profile_json": json.dumps(profile),
            "case_json": json.dumps(case_ctx),
            "enforce_quality_gate": "false",
        },
    )
    assert t_resp.status_code == 200, t_resp.text
    t_data = t_resp.json()
    assert len(t_data["view_attention_weights"]) == 2
    assert abs(sum(t_data["view_attention_weights"]) - 1.0) < 1e-3
    assert len(t_data["stage1_output"]["condition_probs"]) == 15

    a_resp = client.post(
        "/api/v2/assess",
        headers=_hdr,
        files=[("images", ("view1.jpg", img1, "image/jpeg"))],
        data={
            "profile_json": json.dumps(profile),
            "case_json": json.dumps(case_ctx),
            "enforce_quality_gate": "false",
            "save_case": "true",
        },
    )
    assert a_resp.status_code == 200, a_resp.text
    a_data = a_resp.json()
    case_id = a_data["case_id"]
    assert case_id and case_id.startswith("ndx_")
    assert "disclaimer_banner" in a_data["assessment"]
    assert len(a_data["assessment"]["top_matches"]) >= 1

    list_resp = client.get("/api/v2/cases", headers=_hdr)
    assert list_resp.status_code == 200
    assert any(c["case_id"] == case_id for c in list_resp.json()["cases"])

    detail_resp = client.get(f"/api/v2/cases/{case_id}", headers=_hdr)
    assert detail_resp.status_code == 200
    assert detail_resp.json()["case_id"] == case_id
