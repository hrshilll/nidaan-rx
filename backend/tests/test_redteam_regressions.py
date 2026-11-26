"""Regression tests for Red-Team Batteries 1-3 findings."""
from __future__ import annotations

import numpy as np
from PIL import Image

from app.core.knowledge_graph import apply_bayesian_modifier
from app.core.safety_gates import evaluate_clinical_case
from app.core.schemas import AnatomicalSite, CaseContext, ModelInferenceOutput, PatientProfile
from app.vision.quality_gate import evaluate_image_quality


def test_non_skin_red_fabric_flags_low_skin_coverage() -> None:
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    arr[:, :, 0] = 60   # B
    arr[:, :, 1] = 20   # G
    arr[:, :, 2] = 210  # R (bright magenta-red fabric)
    res = evaluate_image_quality(Image.fromarray(arr))
    assert res.metrics["skin_pixel_ratio"] < 0.08
    assert any("low_skin_pixel_coverage" in item or "skin-tone" in item.lower() for item in (res.reasons + res.warnings))


def test_trunk_site_penalizes_scalp_seborrheic_and_blocks_scalp_shampoo() -> None:
    profile = PatientProfile(age=21, sex="male", weight_kg=63.0, fitzpatrick_tone=4, monk_skin_tone=5)
    case = CaseContext(
        anatomical_site=AnatomicalSite.TRUNK,
        duration_days=7,
        itch_score=5,
        pain_burning_score=0,
        has_scaling=True,
        raw_cracked_barrier=False,
    )
    probs = {
        "seborrheic_dermatitis_scalp": 0.87,
        "tinea_corporis_cruris": 0.08,
        "psoriasis_vulgaris": 0.05,
    }
    refined = apply_bayesian_modifier(probs, profile, case)
    top_cid = max(refined.items(), key=lambda kv: kv[1])[0]
    assert top_cid != "seborrheic_dermatitis_scalp"

    inf = ModelInferenceOutput(
        backbone="panderm_base",
        ood_score=14.0,
        is_ood=False,
        condition_probs=refined,
    )
    card_res = evaluate_clinical_case(profile, case, inf)
    all_ids = {c.option.option_id for c in (card_res.tier1_otc_care + card_res.tier2_dermatologist_prep)}
    assert "seb_otc_ketoconazole_shampoo" not in all_ids
    assert "seb_otc_selenium_sulfide_shampoo" not in all_ids
    assert "seb_rx_scalp_steroid_lotion" not in all_ids


def test_asymptomatic_scabies_suppresses_permethrin_and_ivermectin() -> None:
    profile = PatientProfile(age=21, sex="male", weight_kg=63.0, fitzpatrick_tone=4, monk_skin_tone=5)
    case = CaseContext(
        anatomical_site=AnatomicalSite.ARMS_LEGS,
        duration_days=30,
        itch_score=0,
        pain_burning_score=0,
        has_scaling=False,
        raw_cracked_barrier=False,
    )
    inf = ModelInferenceOutput(
        backbone="panderm_base",
        ood_score=13.5,
        is_ood=False,
        condition_probs={
            "scabies": 0.84,
            "benign_growth_nevus": 0.10,
            "urticaria": 0.06,
        },
    )
    card_res = evaluate_clinical_case(profile, case, inf)
    all_ids = {c.option.option_id for c in (card_res.tier1_otc_care + card_res.tier2_dermatologist_prep)}
    assert not any("permethrin" in oid or "ivermectin" in oid for oid in all_ids)
    assert any(g.gate_id == "G4C_SYMPTOM_DISCORDANCE" and g.triggered for g in card_res.safety_audit_log)
