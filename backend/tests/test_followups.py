from __future__ import annotations

import os
from pathlib import Path

from fastapi.testclient import TestClient

from app import main as api_main


def _fresh_client(tmp_path: Path) -> TestClient:
    os.environ["NIDAAN_DB_PATH"] = str(tmp_path / "followups_test.db")
    api_main.get_case_repo.cache_clear()
    return TestClient(api_main.app)


def test_longitudinal_followup_gate8_escalation_and_isolation(tmp_path: Path) -> None:
    client = _fresh_client(tmp_path)

    reg_a = client.post(
        "/api/v2/auth/register",
        json={"email": "fu_a@example.com", "password": "Password123", "full_name": "FollowUp Alice"},
    )
    assert reg_a.status_code == 200
    hdr_a = {"Authorization": f"Bearer {reg_a.json()['access_token']}"}

    reg_b = client.post(
        "/api/v2/auth/register",
        json={"email": "fu_b@example.com", "password": "Password123", "full_name": "FollowUp Bob"},
    )
    hdr_b = {"Authorization": f"Bearer {reg_b.json()['access_token']}"}

    # Create baseline case
    assess = client.post(
        "/api/v2/assess-from-triage",
        headers=hdr_a,
        json={
            "profile": {
                "age": 24,
                "biological_sex": "male",
                "weight_kg": 63.0,
                "fitzpatrick_tone": 4,
                "monk_skin_tone": 5,
            },
            "case_context": {
                "anatomical_site": "trunk",
                "duration_days": 14,
                "itch_score": 6,
                "pain_burning_score": 0,
                "has_scaling": True,
                "compromised_barrier": False,
                "red_flags": {},
                "active_triggers": [],
                "other_trigger_note": "",
            },
            "stage1_output": {
                "condition_probs": {"tinea_corporis_cruris": 0.82, "atopic_dermatitis_eczema": 0.18},
                "family_probs": {"fungal_infections": 0.82, "eczematous_allergic": 0.18},
                "is_ood": False,
                "ood_score": 16.5,
            },
            "save_case": True,
            "view_count": 1,
        },
    )
    assert assess.status_code == 200, assess.text
    cid = assess.json()["case_id"]

    # Unauthenticated blocked (401) & other user blocked (404)
    assert client.post(
        f"/api/v2/cases/{cid}/followups",
        json={"followup_day": 5, "followup_trend": "improving"},
    ).status_code == 401
    assert client.post(
        f"/api/v2/cases/{cid}/followups",
        headers=hdr_b,
        json={"followup_day": 5, "followup_trend": "improving"},
    ).status_code == 404

    # Check-in 1: Day 5 Improving -> NO Gate 8 escalation
    r1 = client.post(
        f"/api/v2/cases/{cid}/followups",
        headers=hdr_a,
        json={
            "followup_day": 5,
            "followup_trend": "improving",
            "itch_score": 3,
            "pain_burning_score": 0,
            "notes": "Redness fading along border",
        },
    )
    assert r1.status_code == 200, r1.text
    d1 = r1.json()
    assert d1["followup"]["escalated"] is False
    assert d1["assessment"]["escalation_reason"] is None
    assert len(d1["followups"]) == 1

    # Check-in 2: Day 7 Worsening -> MUST trigger Gate 8 Longitudinal Escalation
    r2 = client.post(
        f"/api/v2/cases/{cid}/followups",
        headers=hdr_a,
        json={
            "followup_day": 7,
            "followup_trend": "worsening",
            "itch_score": 8,
            "pain_burning_score": 2,
            "notes": "Ring expanded outward despite OTC cream",
        },
    )
    assert r2.status_code == 200, r2.text
    d2 = r2.json()
    assert d2["followup"]["escalated"] is True
    assert "Longitudinal Escalation" in (d2["assessment"]["escalation_reason"] or "")
    assert any(
        g["gate_id"] == "G8_LONGITUDINAL_ESCALATION" and g["action"] == "WARN"
        for g in d2["assessment"]["safety_audit_log"]
    )
    assert len(d2["followups"]) == 2

    # Check-in 3: Day 14 Unchanged -> MUST also trigger Gate 8
    r3 = client.post(
        f"/api/v2/cases/{cid}/followups",
        headers=hdr_a,
        json={
            "followup_day": 14,
            "followup_trend": "unchanged",
            "itch_score": 6,
            "pain_burning_score": 0,
            "notes": "No clearance after 2 weeks",
        },
    )
    assert r3.status_code == 200, r3.text
    assert r3.json()["followup"]["escalated"] is True

    # Cases list exposes follow-up stats & longitudinal_escalated badge
    cases_list = client.get("/api/v2/cases", headers=hdr_a).json()["cases"]
    assert len(cases_list) == 1
    assert cases_list[0]["followup_count"] == 3
    assert cases_list[0]["latest_followup_day"] == 14
    assert cases_list[0]["latest_followup_trend"] == "unchanged"
    assert cases_list[0]["longitudinal_escalated"] is True

    # PDF report still generates cleanly with the follow-up timeline embedded
    r_pdf = client.get(f"/api/v2/cases/{cid}/report.pdf", headers=hdr_a)
    assert r_pdf.status_code == 200
    assert r_pdf.content.startswith(b"%PDF-")

    # Deleting case cleans up followups too
    assert client.delete(f"/api/v2/cases/{cid}", headers=hdr_a).status_code == 200
    assert client.get(f"/api/v2/cases/{cid}", headers=hdr_a).status_code == 404


def test_followup_rejects_duplicate_or_past_days_and_supports_delete_rollback(tmp_path: Path) -> None:
    client = _fresh_client(tmp_path)
    reg = client.post(
        "/api/v2/auth/register",
        json={"email": "mono@example.com", "password": "Password123", "full_name": "Mono User"},
    )
    hdr = {"Authorization": f"Bearer {reg.json()['access_token']}"}

    assess = client.post(
        "/api/v2/assess-from-triage",
        headers=hdr,
        json={
            "profile": {"age": 24, "biological_sex": "male", "weight_kg": 63.0, "fitzpatrick_tone": 4, "monk_skin_tone": 5},
            "case_context": {
                "anatomical_site": "trunk",
                "duration_days": 14,
                "itch_score": 8,
                "pain_burning_score": 1,
                "has_scaling": True,
                "compromised_barrier": False,
                "red_flags": {},
                "active_triggers": [],
                "other_trigger_note": "",
            },
            "stage1_output": {
                "condition_probs": {"tinea_corporis_cruris": 0.85, "atopic_dermatitis_eczema": 0.15},
                "family_probs": {"fungal_infections": 0.85, "eczematous_allergic": 0.15},
                "is_ood": False,
                "ood_score": 16.0,
            },
            "save_case": True,
            "view_count": 1,
        },
    )
    cid = assess.json()["case_id"]

    # Log Day 7, 14, 21, 28
    f7 = client.post(f"/api/v2/cases/{cid}/followups", headers=hdr, json={"followup_day": 7, "followup_trend": "improving", "itch_score": 5}).json()["followup"]["followup_id"]
    client.post(f"/api/v2/cases/{cid}/followups", headers=hdr, json={"followup_day": 14, "followup_trend": "improving", "itch_score": 3})
    client.post(f"/api/v2/cases/{cid}/followups", headers=hdr, json={"followup_day": 21, "followup_trend": "unchanged", "itch_score": 3})
    r28 = client.post(f"/api/v2/cases/{cid}/followups", headers=hdr, json={"followup_day": 28, "followup_trend": "worsening", "itch_score": 7})
    assert r28.status_code == 200
    f28 = r28.json()["followup"]["followup_id"]

    # Day 0 baseline itch_score MUST remain 8 (not overwritten by Day 28's 7)
    detail = client.get(f"/api/v2/cases/{cid}", headers=hdr).json()
    assert detail["case_context"]["itch_score"] == 8

    # Attempting to log Day 14 or Day 28 again MUST return 422
    assert client.post(f"/api/v2/cases/{cid}/followups", headers=hdr, json={"followup_day": 14, "followup_trend": "improving"}).status_code == 422
    assert client.post(f"/api/v2/cases/{cid}/followups", headers=hdr, json={"followup_day": 28, "followup_trend": "improving"}).status_code == 422

    # Deleting Day 28 rolls latest_followup_day back cleanly to Day 21
    d_resp = client.delete(f"/api/v2/cases/{cid}/followups/{f28}", headers=hdr)
    assert d_resp.status_code == 200
    assert len(d_resp.json()["followups"]) == 3
    cases_list = client.get("/api/v2/cases", headers=hdr).json()["cases"]
    assert cases_list[0]["latest_followup_day"] == 21
    assert f7 != f28
