from __future__ import annotations

import os
from pathlib import Path

from fastapi.testclient import TestClient

from app import main as api_main
from app.core.schemas import PatientProfile
from app.services.rx_scanner import PrescriptionScannerService


def _fresh_client(tmp_path: Path) -> TestClient:
    os.environ["NIDAAN_DB_PATH"] = str(tmp_path / "rx_test.db")
    api_main.get_case_repo.cache_clear()
    return TestClient(api_main.app)


def test_rx_search_pmbjp_equivalence_allergy_and_cross_ddi(tmp_path: Path) -> None:
    client = _fresh_client(tmp_path)

    # Unauthenticated blocked
    assert client.post("/api/v2/rx/search", json={"query": "Clotrimazole 1% Cream"}).status_code == 401

    reg = client.post(
        "/api/v2/auth/register",
        json={"email": "rxuser@example.com", "password": "Password123", "full_name": "Rx Tester"},
    )
    assert reg.status_code == 200
    hdr = {"Authorization": f"Bearer {reg.json()['access_token']}"}

    # Save profile with clotrimazole allergy + atorvastatin oral med
    prof = {
        "age": 24,
        "biological_sex": "male",
        "weight_kg": 63.0,
        "fitzpatrick_tone": 4,
        "monk_skin_tone": 5,
        "drug_allergies": ["clotrimazole"],
        "ongoing_oral_meds": ["atorvastatin"],
        "systemic_conditions": [],
        "is_pregnant": False,
        "is_lactating": False,
        "sleep_hours": 7.0,
        "sleep_pattern": "regular",
        "stress_level": "low",
        "climate_tags": [],
        "current_routine_actives": [],
    }
    assert client.put("/api/v2/profile", json=prof, headers=hdr).status_code == 200

    # Search multiple drugs at once: Clotrimazole (allergy), Terbinafine (safe + PMBJP), Itraconazole (DDI w/ atorvastatin)
    resp = client.post(
        "/api/v2/rx/search",
        headers=hdr,
        json={"query": "Clotrimazole 1% Cream\nTerbinafine 1% Cream\nItraconazole 100mg Capsule"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["is_prescription"] is True
    assert data["summary"]["detected_medicines"] == 3
    assert data["summary"]["pmbjp_matched_medicines"] >= 3
    assert data["summary"]["total_savings_pct"] > 50

    all_alerts = [a["gate_id"] for it in data["items"] for a in it["safety_alerts"]]
    assert "G7_ALLERGY" in all_alerts
    assert "G11_DDI" in all_alerts


def test_rx_hard_negative_metadata_returns_zero_hallucinations() -> None:
    svc = PrescriptionScannerService()
    res = svc.analyze_lines(
        [
            {"text": "Dr. A. Sharma MBBS MD Dermatology", "confidence": 0.99},
            {"text": "Patient Name: Rahul Age: 25/M Date: 26/11/2026", "confidence": 0.98},
            {"text": "Ahmedabad Gujarat 380009", "confidence": 0.97},
            {"text": "Follow up after 7 days", "confidence": 0.95},
        ],
        profile=PatientProfile(age=25, biological_sex="male", weight_kg=65.0),
    )
    assert res["is_prescription"] is False
    assert res["items"] == []


def test_rx_cross_prescription_ddi_doxycycline_isotretinoin() -> None:
    svc = PrescriptionScannerService()
    res = svc.analyze_lines(
        [
            {"text": "Doxycycline 100mg Capsule", "confidence": 0.99},
            {"text": "Isotretinoin 20mg Capsule", "confidence": 0.99},
        ],
        profile=PatientProfile(age=22, biological_sex="male", weight_kg=62.0),
    )
    assert len(res["cross_rx_ddi_alerts"]) >= 1
    assert any("intracranial hypertension" in a["message"].lower() for a in res["cross_rx_ddi_alerts"])


def test_rx_deduplicates_brand_and_composition_subtitle() -> None:
    svc = PrescriptionScannerService()
    res = svc.analyze_lines(
        [
            {"text": "1) TABLET FORCAN 150", "confidence": 0.99},
            {"text": "FLUCONAZOLE 150MG", "confidence": 0.98},
            {"text": "2) OINTMENT DEWDERM", "confidence": 0.99},
            {"text": "BECLOMETHASONE 0.025%, CLOTRIMAZOLE 1%, GENTAMICIN 0.1%", "confidence": 0.97},
            {"text": "3) CREAM TURGODERM", "confidence": 0.99},
        ],
        profile=PatientProfile(age=17, biological_sex="male", weight_kg=60.0),
    )
    assert res["summary"]["detected_medicines"] == 3
    names = [it["matched_name"].lower() for it in res["items"]]
    assert any("forcan" in n for n in names)
    assert not any(n == "fluconazole 150mg" for n in names)
    forcan_item = next(it for it in res["items"] if "forcan" in it["matched_name"].lower())
    assert "FLUCONAZOLE 150MG" in forcan_item["ocr_line"]
