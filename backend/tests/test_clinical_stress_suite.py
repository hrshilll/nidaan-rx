"""
Milestone 3 — Permanent 10-Scenario Multi-Condition Clinical & Safety Stress Suite.
Covers Acne, Scalp Seborrheic Dermatitis, Facial/Pediatric Eczema, Steroid-on-Infection,
Scalp vs. Trunk Psoriasis, Pregnant Melasma, Tinea + SERIOUS DDI, Pediatric Urticaria,
Neoplastic Red-Flag Hard-Stop, and PMBJP MRP sanity (> 0).
"""
from __future__ import annotations

from app.core.knowledge_graph import CONDITION_GRAPH
from app.core.safety_gates import evaluate_clinical_case
from app.core.schemas import (
    AnatomicalSite,
    CaseContext,
    ModelInferenceOutput,
    PatientProfile,
)
from app.services.inference import NidaanInferenceService

_SVC = NidaanInferenceService(device="cpu")
_CAT = _SVC._catalog


def _s1(primary: str, p1: float = 0.88, secondary: str | None = None) -> ModelInferenceOutput:
    ck1 = CONDITION_GRAPH[primary]
    probs = {primary: p1}
    fams = {ck1.tier1_family.value: p1}
    if secondary:
        ck2 = CONDITION_GRAPH[secondary]
        rem = round(1.0 - p1, 4)
        probs[secondary] = rem
        fams[ck2.tier1_family.value] = fams.get(ck2.tier1_family.value, 0.0) + rem
    return ModelInferenceOutput(condition_probs=probs, family_probs=fams, is_ood=False, ood_score=16.0)


def test_stress_01_pregnant_acne_pauses_existing_tretinoin_and_harsh_actives() -> None:
    res = evaluate_clinical_case(
        PatientProfile(
            age=25,
            biological_sex="female",
            weight_kg=55.0,
            is_pregnant=True,
            current_routine_actives=["tretinoin"],
        ),
        CaseContext(anatomical_site=AnatomicalSite.FACE, duration_days=30, compromised_barrier=True),
        _s1("acne_vulgaris"),
        catalog=_CAT,
    )
    assert not res.escalation_hard_stop
    assert [c.option.option_id for c in res.tier1_otc_care] == ["acne_otc_niacinamide"]
    assert res.tier2_dermatologist_prep == []
    assert any("tretinoin" in w.lower() and "pregnancy" in w.lower() for w in res.do_not_apply_warnings)


def test_stress_02_acne_isotretinoin_blocks_doxycycline_serious_ddi() -> None:
    res = evaluate_clinical_case(
        PatientProfile(
            age=21,
            biological_sex="male",
            weight_kg=62.0,
            ongoing_oral_meds=["isotretinoin"],
            current_routine_actives=["adapalene"],
        ),
        CaseContext(anatomical_site=AnatomicalSite.FACE, duration_days=60),
        _s1("acne_vulgaris"),
        catalog=_CAT,
    )
    rx_ids = {c.option.option_id for c in res.tier2_dermatologist_prep}
    assert "acne_rx_doxycycline" not in rx_ids
    assert any(g.gate_id == "G11_DDI_CHECK" and g.action == "BLOCK" for g in res.safety_audit_log)


def test_stress_03_scalp_seborrheic_dermatitis_matches_pmbjp_keto_shampoo() -> None:
    res = evaluate_clinical_case(
        PatientProfile(age=23, biological_sex="male", weight_kg=64.0),
        CaseContext(anatomical_site=AnatomicalSite.SCALP, duration_days=45, itch_score=6, has_scaling=True),
        _s1("seborrheic_dermatitis_scalp"),
        catalog=_CAT,
    )
    keto = next(c for c in res.tier1_otc_care if c.option.option_id == "scalp_otc_keto2")
    assert len(keto.pmbjp_matches) >= 1
    assert keto.pmbjp_matches[0].mrp_inr is not None and keto.pmbjp_matches[0].mrp_inr > 0


def test_stress_04_facial_eczema_minor_blocks_potent_steroid() -> None:
    res = evaluate_clinical_case(
        PatientProfile(age=14, biological_sex="female", weight_kg=40.0),
        CaseContext(anatomical_site=AnatomicalSite.FACE, duration_days=21, itch_score=7, has_scaling=True),
        _s1("atopic_dermatitis_eczema"),
        catalog=_CAT,
    )
    assert res.minor_guardian_notice is not None
    rx_ids = {c.option.option_id for c in res.tier2_dermatologist_prep}
    assert "eczema_rx_mometasone" not in rx_ids
    assert any(g.gate_id == "G10_ANATOMICAL_SITE" and g.action == "BLOCK" for g in res.safety_audit_log)


def test_stress_05_eczema_with_bacterial_infection_blocks_all_steroids() -> None:
    res = evaluate_clinical_case(
        PatientProfile(age=28, biological_sex="male", weight_kg=70.0),
        CaseContext(anatomical_site=AnatomicalSite.ARMS_LEGS, duration_days=14, itch_score=6, has_scaling=True),
        _s1("atopic_dermatitis_eczema", p1=0.70, secondary="bacterial_folliculitis_impetigo"),
        catalog=_CAT,
    )
    assert [c.option.option_id for c in res.tier1_otc_care] == ["eczema_otc_ceramide"]
    assert res.tier2_dermatologist_prep == []
    assert sum(1 for g in res.safety_audit_log if g.gate_id == "G5_STEROID_ON_INFECTION" and g.action == "BLOCK") == 3


def test_stress_06_scalp_vs_trunk_psoriasis_vehicle_routing() -> None:
    prof = PatientProfile(age=34, biological_sex="male", weight_kg=74.0)
    res_scalp = evaluate_clinical_case(
        prof,
        CaseContext(anatomical_site=AnatomicalSite.SCALP, duration_days=90, itch_score=5, has_scaling=True),
        _s1("psoriasis_vulgaris"),
        catalog=_CAT,
    )
    scalp_t1 = [c.option.option_id for c in res_scalp.tier1_otc_care]
    scalp_t2 = [c.option.option_id for c in res_scalp.tier2_dermatologist_prep]
    assert "psor_otc_coal_tar_shampoo" in scalp_t1
    assert "psor_rx_clobetasol_scalp_lotion" in scalp_t2
    assert "psor_otc_emollient" not in scalp_t1

    res_trunk = evaluate_clinical_case(
        prof,
        CaseContext(anatomical_site=AnatomicalSite.TRUNK, duration_days=90, itch_score=5, has_scaling=True),
        _s1("psoriasis_vulgaris"),
        catalog=_CAT,
    )
    trunk_t1 = [c.option.option_id for c in res_trunk.tier1_otc_care]
    assert "psor_otc_coal_tar_shampoo" not in trunk_t1
    assert "psor_otc_emollient" in trunk_t1


def test_stress_07_pregnant_melasma_blocks_hydroquinone_and_never_emits_zero_mrp() -> None:
    res = evaluate_clinical_case(
        PatientProfile(age=30, biological_sex="female", weight_kg=58.0, is_pregnant=True),
        CaseContext(anatomical_site=AnatomicalSite.FACE, duration_days=120),
        _s1("melasma_pih"),
        catalog=_CAT,
    )
    rx_ids = [c.option.option_id for c in res.tier2_dermatologist_prep]
    assert "pigment_rx_hydroquinone" not in rx_ids
    assert "pigment_rx_azelaic20" in rx_ids
    for card in res.tier1_otc_care + res.tier2_dermatologist_prep:
        for gm in card.pmbjp_matches:
            if gm.mrp_inr is not None:
                assert gm.mrp_inr > 0, f"Invalid zero/negative MRP on {gm.product_name}"


def test_stress_08_tinea_with_atorvastatin_blocks_oral_itraconazole_ddi() -> None:
    res = evaluate_clinical_case(
        PatientProfile(age=55, biological_sex="male", weight_kg=76.0, ongoing_oral_meds=["atorvastatin", "warfarin"]),
        CaseContext(anatomical_site=AnatomicalSite.TRUNK, duration_days=21, itch_score=7, has_scaling=True),
        _s1("tinea_corporis_cruris"),
        catalog=_CAT,
    )
    rx_ids = {c.option.option_id for c in res.tier2_dermatologist_prep}
    assert "tinea_rx_itraconazole" not in rx_ids
    assert "tinea_rx_luliconazole" in rx_ids
    assert any(g.gate_id == "G11_DDI_CHECK" and g.action == "BLOCK" for g in res.safety_audit_log)


def test_stress_09_pediatric_urticaria_enforces_age_ceiling() -> None:
    res = evaluate_clinical_case(
        PatientProfile(age=10, biological_sex="male", weight_kg=28.0),
        CaseContext(anatomical_site=AnatomicalSite.TRUNK, duration_days=2, itch_score=8),
        _s1("urticaria"),
        catalog=_CAT,
    )
    assert res.minor_guardian_notice is not None
    assert [c.option.option_id for c in res.tier1_otc_care] == ["urt_otc_calamine"]
    assert res.tier2_dermatologist_prep == []


def test_stress_10_neoplastic_secondary_risk_triggers_hard_stop() -> None:
    res = evaluate_clinical_case(
        PatientProfile(age=48, biological_sex="male", weight_kg=72.0),
        CaseContext(anatomical_site=AnatomicalSite.TRUNK, duration_days=60),
        _s1("melasma_pih", p1=0.82, secondary="suspected_neoplastic_lesion"),
        catalog=_CAT,
    )
    assert res.escalation_hard_stop is True
    assert res.tier1_otc_care == []
    assert res.tier2_dermatologist_prep == []
