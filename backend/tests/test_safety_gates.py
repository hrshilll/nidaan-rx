"""
Nidaan-Rx v2.0 — Phase 1 Deterministic Verification Suite.
Verifies Knowledge Graph integrity, Stage 2 Bayesian Modifier, CDSCO Two-Tier
separation, and all 12 Deterministic Clinical Safety Gates.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from app.core.knowledge_graph import (
    CONDITION_GRAPH,
    TRIGGER_BUCKETS,
    MedicineCatalog,
    apply_bayesian_modifier,
    select_dynamic_trigger_buckets,
)
from app.core.safety_gates import evaluate_clinical_case
from app.core.schemas import (
    AnatomicalSite,
    CaseContext,
    FollowUpTrend,
    ModelInferenceOutput,
    PatientProfile,
    RedFlagChecklist,
    SaltEntity,
    Tier1Family,
)


def make_adult_profile(**overrides) -> PatientProfile:
    base = dict(
        age=25,
        biological_sex="male",
        weight_kg=68.0,
        weight_updated_at=date.today(),
        skin_type="combination",
        fitzpatrick_tone=4,
        monk_skin_tone=6,
    )
    base.update(overrides)
    return PatientProfile(**base)


def make_mock_catalog() -> MedicineCatalog:
    cat = MedicineCatalog()
    cat.pmbjp_rows = [
        {
            "product_name": "Clotrimazole Cream 1% w/w (PMBJP)",
            "generic_code": "PMBJP-0101",
            "mrp_inr": 18.0,
            "vehicle": "CREAM",
            "salts": [SaltEntity(salt="clotrimazole", strength="1%")],
        },
        {
            "product_name": "Clotrimazole Cream 2% w/w (PMBJP)",
            "generic_code": "PMBJP-0102",
            "mrp_inr": 20.0,
            "vehicle": "CREAM",
            "salts": [SaltEntity(salt="clotrimazole", strength="2%")],
        },
        {
            "product_name": "Clotrimazole Ointment 1% (PMBJP)",
            "generic_code": "PMBJP-0103",
            "mrp_inr": 17.0,
            "vehicle": "OINTMENT",
            "salts": [SaltEntity(salt="clotrimazole", strength="1%")],
        },
        {
            "product_name": "Clotrimazole 1% + Beclomethasone 0.025% Cream",
            "generic_code": "PMBJP-0104",
            "mrp_inr": 25.0,
            "vehicle": "CREAM",
            "salts": [
                SaltEntity(salt="clotrimazole", strength="1%"),
                SaltEntity(salt="beclomethasone", strength="0.025%"),
            ],
        },
    ]
    cat.commercial_avg_price_by_salt["clotrimazole"] = 90.0
    return cat


def test_taxonomy_and_14_trigger_buckets_complete():
    assert len(Tier1Family) == 10
    covered_families = {ck.tier1_family for ck in CONDITION_GRAPH.values()}
    assert covered_families == set(Tier1Family)
    assert len(TRIGGER_BUCKETS) == 14
    assert [b.bucket_id for b in TRIGGER_BUCKETS] == list(range(1, 15))
    assert TRIGGER_BUCKETS[0].india_critical is True


def test_dynamic_trigger_selector_excludes_bucket1_and_prioritizes_context():
    profile = make_adult_profile(sleep_hours=5.0, sleep_pattern="late_sleeper", climate_tags=["hard_water"])
    stage1 = ModelInferenceOutput(condition_probs={"seborrheic_dermatitis_scalp": 0.75, "acne_vulgaris": 0.25})
    cards = select_dynamic_trigger_buckets(stage1, profile, n_cards=3)
    bucket_ids = [b.bucket_id for b in cards]
    assert len(bucket_ids) == 3
    assert 1 not in bucket_ids
    assert 6 in bucket_ids


def test_bayesian_modifier_shifts_posterior_with_bucket_dampening():
    stage1 = ModelInferenceOutput(
        condition_probs={"atopic_dermatitis_eczema": 0.55, "tinea_incognito": 0.45}
    )
    preds = apply_bayesian_modifier(stage1, ["t01_otc_steroid_cream", "t01_rebound_flare"])
    assert preds[0].condition_id == "tinea_incognito"
    assert preds[0].posterior_prob > preds[0].prior_prob
    assert preds[0].posterior_prob < 0.85


def test_gates_1_to_3_pmbjp_equivalence_strict():
    profile = make_adult_profile()
    case = CaseContext(anatomical_site=AnatomicalSite.TRUNK, itch_score=5, has_scaling=True)
    stage1 = ModelInferenceOutput(condition_probs={"tinea_corporis_cruris": 0.90, "atopic_dermatitis_eczema": 0.10})
    res = evaluate_clinical_case(profile, case, stage1, catalog=make_mock_catalog())

    clotri_card = next(c for c in res.tier1_otc_care if c.option.option_id == "tinea_otc_clotrimazole_cream")
    assert len(clotri_card.pmbjp_matches) == 1
    assert clotri_card.pmbjp_matches[0].generic_code == "PMBJP-0101"
    assert clotri_card.pmbjp_matches[0].estimated_savings_pct == 80.0

    gate_ids = {e.gate_id for e in res.safety_audit_log}
    assert "G2_DOSAGE_ALIGN" in gate_ids
    assert "G3_VEHICLE_MATCH" in gate_ids


def test_gate4_red_flag_and_ood_hard_stop():
    profile = make_adult_profile()
    case = CaseContext(red_flags=RedFlagChecklist(eye_lip_throat_swelling=True))
    stage1 = ModelInferenceOutput(condition_probs={"urticaria": 0.95})
    res = evaluate_clinical_case(profile, case, stage1)

    assert res.escalation_hard_stop is True
    assert res.tier1_otc_care == []
    assert res.tier2_dermatologist_prep == []
    assert any(e.gate_id == "G4_RED_FLAG_HARD_STOP" for e in res.safety_audit_log)

    ood_res = evaluate_clinical_case(
        profile, CaseContext(), ModelInferenceOutput(condition_probs={"acne_vulgaris": 0.6}, is_ood=True)
    )
    assert ood_res.escalation_hard_stop is True
    assert ood_res.tier1_otc_care == []


def test_gate5_steroid_on_infection_block():
    profile = make_adult_profile()
    case = CaseContext(anatomical_site=AnatomicalSite.TRUNK, itch_score=5, has_scaling=True)
    stage1 = ModelInferenceOutput(condition_probs={"atopic_dermatitis_eczema": 0.60, "tinea_corporis_cruris": 0.40})
    res = evaluate_clinical_case(profile, case, stage1)

    all_ids = {c.option.option_id for c in res.tier1_otc_care + res.tier2_dermatologist_prep}
    assert "eczema_otc_hydrocortisone" not in all_ids
    assert "eczema_rx_mometasone" not in all_ids
    assert "eczema_otc_ceramide" in all_ids
    assert any(e.gate_id == "G5_STEROID_ON_INFECTION" for e in res.safety_audit_log)


def test_gate6_pregnancy_systemic_and_minor_notice():
    preg_profile = make_adult_profile(
        age=26, biological_sex="female", is_pregnant=True, systemic_conditions=["liver_disease"]
    )
    stage1 = ModelInferenceOutput(condition_probs={"acne_vulgaris": 0.95})
    res = evaluate_clinical_case(preg_profile, CaseContext(anatomical_site=AnatomicalSite.FACE), stage1)

    rx_ids = {c.option.option_id for c in res.tier2_dermatologist_prep}
    assert "acne_rx_adapalene" not in rx_ids
    assert "acne_rx_doxycycline" not in rx_ids
    assert any(e.gate_id == "G6_AGE_PREG_SYSTEMIC" for e in res.safety_audit_log)

    teen_profile = make_adult_profile(age=16, weight_kg=52.0)
    teen_res = evaluate_clinical_case(teen_profile, CaseContext(), stage1)
    assert teen_res.minor_guardian_notice is not None


def test_gate7_allergy_block_and_routine_active_conflict():
    profile = make_adult_profile(
        drug_allergies=["benzoyl peroxide"],
        current_routine_actives=["glycolic acid"],
    )
    stage1 = ModelInferenceOutput(condition_probs={"acne_vulgaris": 0.95})
    res = evaluate_clinical_case(profile, CaseContext(anatomical_site=AnatomicalSite.FACE), stage1)

    otc_ids = {c.option.option_id for c in res.tier1_otc_care}
    assert "acne_otc_bpo" not in otc_ids
    sal_card = next(c for c in res.tier1_otc_care if c.option.option_id == "acne_otc_salicylic")
    assert any("Active Conflict" in b for b in sal_card.inline_badges)


def test_gate8_longitudinal_non_response_escalation():
    profile = make_adult_profile()
    case = CaseContext(followup_day=14, followup_trend=FollowUpTrend.WORSENING)
    stage1 = ModelInferenceOutput(condition_probs={"acne_vulgaris": 0.92})
    res = evaluate_clinical_case(profile, case, stage1)

    assert res.escalation_reason is not None and "Day 14" in res.escalation_reason
    assert res.tier1_otc_care == []
    assert len(res.tier2_dermatologist_prep) > 0


def test_gate9_weight_based_pediatric_ceiling_and_90day_refresh():
    stale_date = date.today() - timedelta(days=95)
    light_teen = make_adult_profile(age=14, weight_kg=32.0, weight_updated_at=stale_date)
    stage1 = ModelInferenceOutput(condition_probs={"urticaria": 0.95})
    res = evaluate_clinical_case(light_teen, CaseContext(anatomical_site=AnatomicalSite.TRUNK), stage1)

    assert res.weight_refresh_prompt is True
    otc_ids = {c.option.option_id for c in res.tier1_otc_care}
    assert "urt_otc_cetirizine" not in otc_ids
    assert "urt_otc_calamine" in otc_ids
    assert any(e.gate_id == "G9_WEIGHT_PEDIATRIC_CEILING" for e in res.safety_audit_log)


def test_gate10_anatomical_site_steroid_and_scalp_vehicle():
    profile = make_adult_profile()
    res_face = evaluate_clinical_case(
        profile,
        CaseContext(anatomical_site=AnatomicalSite.FACE),
        ModelInferenceOutput(condition_probs={"atopic_dermatitis_eczema": 0.95}),
    )
    assert "eczema_rx_mometasone" not in {c.option.option_id for c in res_face.tier2_dermatologist_prep}
    assert "eczema_rx_desonide" in {c.option.option_id for c in res_face.tier2_dermatologist_prep}

    res_scalp = evaluate_clinical_case(
        profile,
        CaseContext(anatomical_site=AnatomicalSite.SCALP),
        ModelInferenceOutput(condition_probs={"seborrheic_dermatitis_scalp": 0.95}),
    )
    assert "scalp_otc_keto2" in {c.option.option_id for c in res_scalp.tier1_otc_care}

    res_scalp_eczema = evaluate_clinical_case(
        profile,
        CaseContext(anatomical_site=AnatomicalSite.SCALP),
        ModelInferenceOutput(condition_probs={"atopic_dermatitis_eczema": 0.95}),
    )
    assert "eczema_otc_ceramide" not in {c.option.option_id for c in res_scalp_eczema.tier1_otc_care}
    assert "eczema_rx_desonide" in {c.option.option_id for c in res_scalp_eczema.tier2_dermatologist_prep}


def test_gate11_ddi_blocks_serious_interaction():
    profile = make_adult_profile(ongoing_oral_meds=["atorvastatin"])
    stage1 = ModelInferenceOutput(condition_probs={"tinea_corporis_cruris": 0.95})
    res = evaluate_clinical_case(profile, CaseContext(anatomical_site=AnatomicalSite.TRUNK), stage1)

    rx_ids = {c.option.option_id for c in res.tier2_dermatologist_prep}
    assert "tinea_rx_itraconazole" not in rx_ids
    assert "tinea_rx_luliconazole" in rx_ids
    assert any(e.gate_id == "G11_DDI_CHECK" and e.action == "BLOCK" for e in res.safety_audit_log)


def test_gate12_photosensitivity_and_compromised_barrier():
    profile = make_adult_profile()
    stage1 = ModelInferenceOutput(condition_probs={"acne_vulgaris": 0.95})

    res_ok = evaluate_clinical_case(profile, CaseContext(anatomical_site=AnatomicalSite.FACE), stage1)
    bpo_card = next(c for c in res_ok.tier1_otc_care if c.option.option_id == "acne_otc_bpo")
    assert any("SPF 30+" in b for b in bpo_card.inline_badges)

    res_barrier = evaluate_clinical_case(
        profile,
        CaseContext(anatomical_site=AnatomicalSite.FACE, compromised_barrier=True),
        stage1,
    )
    otc_ids = {c.option.option_id for c in res_barrier.tier1_otc_care}
    assert "acne_otc_bpo" not in otc_ids
    assert "acne_otc_salicylic" not in otc_ids
    assert "acne_otc_niacinamide" in otc_ids


def test_cdsco_two_tier_labels_and_disclaimer_always_present():
    profile = make_adult_profile()
    res = evaluate_clinical_case(
        profile,
        CaseContext(anatomical_site=AnatomicalSite.FACE),
        ModelInferenceOutput(condition_probs={"acne_vulgaris": 0.95}),
    )
    assert "does not provide a medical diagnosis" in res.disclaimer_banner
    assert all("Tier 1" in c.regulatory_label for c in res.tier1_otc_care)
    assert all("Dermatologist Discussion Only" in c.regulatory_label for c in res.tier2_dermatologist_prep)


def test_onemg_csv_loader_integration_if_present():
    csv_path = Path("data/processed/1mg_dermatology_parsed.csv")
    if csv_path.exists():
        cat = MedicineCatalog(onemg_csv=csv_path)
        assert len(cat.commercial_avg_price_by_salt) > 20
        assert len(cat.ddi_index) > 0