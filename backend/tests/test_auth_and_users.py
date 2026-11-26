from __future__ import annotations

import io
import os
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app import main as api_main
from app.core.knowledge_graph import apply_bayesian_modifier
from app.core.schemas import ModelInferenceOutput


def _fresh_client(tmp_path: Path) -> TestClient:
    os.environ["NIDAAN_DB_PATH"] = str(tmp_path / "users_strict_test.db")
    api_main.get_case_repo.cache_clear()
    return TestClient(api_main.app)


def test_strict_auth_logout_revocation_and_pdf_export(tmp_path: Path) -> None:
    client = _fresh_client(tmp_path)

    # Unauthenticated uploads, preview, and cases are strictly blocked (401)
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (180, 140, 120)).save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    assert client.post("/api/v2/preview-image", files={"image": ("a.jpg", img_bytes, "image/jpeg")}).status_code == 401
    assert client.get("/api/v2/cases").status_code == 401

    # Register User A
    r_reg = client.post(
        "/api/v2/auth/register",
        json={"email": "alice@example.com", "password": "Password123", "full_name": "Alice Shah"},
    )
    assert r_reg.status_code == 200
    tok_a = r_reg.json()["access_token"]
    hdr_a = {"Authorization": f"Bearer {tok_a}"}

    # Authenticated preview succeeds
    r_prev = client.post("/api/v2/preview-image", headers=hdr_a, files={"image": ("a.jpg", img_bytes, "image/jpeg")})
    assert r_prev.status_code == 200
    assert r_prev.headers["content-type"] == "image/jpeg"

    # Save a case and download its multi-section PDF report
    prof_payload = {
        "age": 24,
        "biological_sex": "female",
        "weight_kg": 56.0,
        "fitzpatrick_tone": 4,
        "monk_skin_tone": 6,
        "drug_allergies": ["penicillin"],
        "ongoing_oral_meds": [],
        "systemic_conditions": [],
        "is_pregnant": False,
        "is_lactating": False,
        "sleep_hours": 5.5,
        "sleep_pattern": "irregular",
        "stress_level": "moderate",
        "climate_tags": ["humid"],
        "current_routine_actives": ["adapalene"],
    }
    r_assess = client.post(
        "/api/v2/assess-from-triage",
        headers=hdr_a,
        json={
            "profile": prof_payload,
            "case_context": {
                "anatomical_site": "face",
                "duration_days": 14,
                "itch_score": 2,
                "pain_burning_score": 1,
                "has_scaling": False,
                "compromised_barrier": False,
                "red_flags": {},
                "active_triggers": ["t04_poor_irregular_sleep__short_sleep", "t04_poor_irregular_sleep__irregular_schedule"],
                "other_trigger_note": "Testing PDF summary export",
            },
            "stage1_output": {
                "condition_probs": {"acne_vulgaris": 0.45, "atopic_dermatitis_eczema": 0.40, "tinea_corporis_cruris": 0.15},
                "family_probs": {"inflammatory_acneiform": 0.45, "eczematous_allergic": 0.40, "fungal_infections": 0.15},
                "is_ood": False,
                "ood_score": 1.1,
            },
            "save_case": True,
            "view_count": 1,
        },
    )
    assert r_assess.status_code == 200
    cid = r_assess.json()["case_id"]

    # Download real PDF summary
    r_pdf = client.get(f"/api/v2/cases/{cid}/report.pdf", headers=hdr_a)
    assert r_pdf.status_code == 200
    assert r_pdf.headers["content-type"] == "application/pdf"
    assert r_pdf.content.startswith(b"%PDF-")
    assert len(r_pdf.content) > 1500

    # User B cannot access User A's case or PDF
    r_reg_b = client.post(
        "/api/v2/auth/register",
        json={"email": "bob@example.com", "password": "Password123", "full_name": "Bob"},
    )
    hdr_b = {"Authorization": f"Bearer {r_reg_b.json()['access_token']}"}
    assert client.get(f"/api/v2/cases/{cid}/report.pdf", headers=hdr_b).status_code == 404

    # Server-side logout revokes User A's token immediately
    r_out = client.post("/api/v2/auth/logout", headers=hdr_a)
    assert r_out.status_code == 200
    assert client.get("/api/v2/auth/me", headers=hdr_a).status_code == 401
    assert client.get("/api/v2/cases", headers=hdr_a).status_code == 401


def test_sleep_triggers_do_not_bias_towards_acne() -> None:
    stage1 = ModelInferenceOutput(
        condition_probs={
            "tinea_corporis_cruris": 0.35,
            "acne_vulgaris": 0.35,
            "atopic_dermatitis_eczema": 0.30,
        },
        family_probs={"fungal_infections": 0.35, "inflammatory_acneiform": 0.35, "eczematous_allergic": 0.30},
        is_ood=False,
        ood_score=1.0,
    )
    preds = apply_bayesian_modifier(
        stage1,
        active_triggers=[
            "t04_poor_irregular_sleep__short_sleep",
            "t04_poor_irregular_sleep__irregular_schedule",
        ],
    )
    by_id = {p.condition_id: p.posterior_prob for p in preds}
    # Sleep should shift acne by < 2 percentage points and NEVER hijack diagnosis
    assert abs(by_id["acne_vulgaris"] - 0.35) < 0.02
    assert abs(by_id["atopic_dermatitis_eczema"] - 0.30) < 0.02


def test_eyelid_ocular_herpes_hard_stop_and_non_ophthalmic_cream_block() -> None:
    from app.core.safety_gates import evaluate_clinical_case
    from app.core.schemas import AnatomicalSite, CaseContext, PatientProfile

    profile = PatientProfile(
        age=21,
        biological_sex="male",
        weight_kg=63.0,
        fitzpatrick_tone=4,
        monk_skin_tone=5,
        drug_allergies=["clotrimazole"],
    )
    case_eyelid = CaseContext(
        anatomical_site=AnatomicalSite.EYELIDS,
        duration_days=7,
        itch_score=0,
        pain_burning_score=0,
        has_scaling=False,
        compromised_barrier=False,
        active_triggers=["t09_damp_clothes_wet_hair__damp_clothes"],
    )

    # 1) Eyelid with 13%+ Herpes -> MUST trigger Gate 4 Hard Stop (ophthalmic risk)
    stage1_viral = ModelInferenceOutput(
        condition_probs={"tinea_corporis_cruris": 0.60, "herpes_viral_eruption": 0.20, "atopic_dermatitis_eczema": 0.20},
        family_probs={"fungal_infections": 0.60, "bacterial_viral_parasitic": 0.20, "eczematous_allergic": 0.20},
        is_ood=False,
        ood_score=17.0,
    )
    res_viral = evaluate_clinical_case(profile, case_eyelid, stage1_viral)
    assert res_viral.escalation_hard_stop is True
    assert "ophthalmology" in (res_viral.escalation_reason or "").lower()
    assert res_viral.tier1_otc_care == []

    # 2) Eyelid with 0% Herpes (pure Tinea) -> MUST block Terbinafine & Luliconazole creams under Gate 10
    stage1_fungal = ModelInferenceOutput(
        condition_probs={"tinea_corporis_cruris": 0.92, "atopic_dermatitis_eczema": 0.08},
        family_probs={"fungal_infections": 0.92, "eczematous_allergic": 0.08},
        is_ood=False,
        ood_score=17.0,
    )
    res_fungal = evaluate_clinical_case(profile, case_eyelid, stage1_fungal)
    assert res_fungal.escalation_hard_stop is False
    otc_names = [c.option.display_name.lower() for c in res_fungal.tier1_otc_care]
    rx_names = [c.option.display_name.lower() for c in res_fungal.tier2_dermatologist_prep]
    assert not any("terbinafine" in n for n in otc_names)
    assert not any("luliconazole" in n for n in rx_names)
    assert any(g.gate_id == "G10_SITE_VEHICLE" and g.action == "BLOCK" for g in res_fungal.safety_audit_log)
    assert not any("cotton clothing" in s.lower() for s in res_fungal.lifestyle_action_plan)


def test_change_password_rotates_token_and_revokes_old_session(tmp_path: Path) -> None:
    client = _fresh_client(tmp_path)
    reg = client.post(
        "/api/v2/auth/register",
        json={"email": "pwuser@example.com", "password": "OldPassword1", "full_name": "PW User"},
    )
    assert reg.status_code == 200
    old_tok = reg.json()["access_token"]
    old_hdr = {"Authorization": f"Bearer {old_tok}"}

    # Wrong current password rejected
    bad = client.post(
        "/api/v2/auth/change-password",
        headers=old_hdr,
        json={"current_password": "WrongPassword", "new_password": "NewPassword2"},
    )
    assert bad.status_code == 400

    # Same password rejected
    same = client.post(
        "/api/v2/auth/change-password",
        headers=old_hdr,
        json={"current_password": "OldPassword1", "new_password": "OldPassword1"},
    )
    assert same.status_code == 422

    # Valid change rotates token & revokes old token
    ok = client.post(
        "/api/v2/auth/change-password",
        headers=old_hdr,
        json={"current_password": "OldPassword1", "new_password": "NewPassword2"},
    )
    assert ok.status_code == 200, ok.text
    new_tok = ok.json()["access_token"]
    assert new_tok != old_tok

    assert client.get("/api/v2/auth/me", headers=old_hdr).status_code == 401
    assert client.get("/api/v2/auth/me", headers={"Authorization": f"Bearer {new_tok}"}).status_code == 200

    # Old password fails login, new password succeeds
    assert client.post("/api/v2/auth/login", json={"email": "pwuser@example.com", "password": "OldPassword1"}).status_code == 401
    assert client.post("/api/v2/auth/login", json={"email": "pwuser@example.com", "password": "NewPassword2"}).status_code == 200
