"""
Nidaan-Rx v2.0 — 12-Gate Deterministic Clinical Safety Stack (Pillar 4).

Executes all 12 deterministic safety rules silently in the backend and emits
structured audit events, inline badges, amber "Do NOT Apply" warnings, or
hard-stop emergency escalation cards while enforcing strict CDSCO Two-Tier
separation (Tier 1 OTC vs. Tier 2 Schedule H/H1 Dermatologist Prep).

v1.0 PMBJP Chemical Equivalence Gates (Gates 1-3):
  G1: Salt Subset Verification
  G2: Dosage Alignment
  G3: Vehicle Match

v2.0 Patient-Aware Clinical Safety Gates (Gates 4-12):
  G4  (v2.0 #1): Red-Flag & Neoplastic / OOD Hard Stop
  G5  (v2.0 #2): Steroid-on-Infection Block
  G6  (v2.0 #3): Age / Pregnancy / Systemic Contraindication Gate
  G7  (v2.0 #4): Allergy & Active-Conflict Gate
  G8  (v2.0 #5): Longitudinal Non-Response Escalation
  G9  (v2.0 #6): Weight-Based Dosage & Pediatric Ceiling Gate
  G10 (v2.0 #7): Anatomical Site Sensitivity & Scalp-Vehicle Gate
  G11 (v2.0 #8): Drug-Drug Interaction (DDI) Gate
  G12 (v2.0 #9): Photosensitivity & Compromised-Barrier Gate
"""

from __future__ import annotations

import re

from .knowledge_graph import (
    CONDITION_GRAPH,
    TRIGGER_INDEX,
    MedicineCatalog,
    apply_bayesian_modifier,
)
from .schemas import (
    AnatomicalSite,
    CaseContext,
    ClinicalAssessmentResponse,
    ConditionPrediction,
    EvaluatedTreatmentCard,
    FollowUpTrend,
    GateEvent,
    ModelInferenceOutput,
    PatientProfile,
    PMBJPGenericMatch,
    RegulatoryTier,
    SaltEntity,
    SteroidPotency,
    Tier1Family,
    TreatmentOption,
)

MANDATORY_DISCLAIMER_BANNER = (
    "Nidaan-Rx is an academic research prototype and clinical decision-support tool. "
    "It does not provide a medical diagnosis or replace a licensed dermatologist."
)

# High-risk clinical dermatology Drug-Drug Interactions (bidirectional normalized pairs)
CURATED_DDI_PAIRS: dict[tuple[str, str], tuple[str, str]] = {
    ("itraconazole", "atorvastatin"): ("SERIOUS", "Risk of severe myopathy/rhabdomyolysis via CYP3A4 inhibition."),
    ("itraconazole", "simvastatin"): ("SERIOUS", "Contraindicated: high risk of rhabdomyolysis via CYP3A4 inhibition."),
    ("itraconazole", "warfarin"): ("SERIOUS", "Markedly increases bleeding risk (elevated INR)."),
    ("fluconazole", "warfarin"): ("SERIOUS", "Increases anticoagulation effect and bleeding risk."),
    ("fluconazole", "atorvastatin"): ("MODERATE", "Elevated statin exposure; monitor for muscle pain."),
    ("doxycycline", "isotretinoin"): ("SERIOUS", "Contraindicated: risk of intracranial hypertension (pseudotumor cerebri)."),
    ("doxycycline", "warfarin"): ("MODERATE", "May potentiate warfarin effect; monitor INR."),
    ("terbinafine", "tamoxifen"): ("SERIOUS", "CYP2D6 inhibition reduces active tamoxifen levels."),
    ("terbinafine", "amitriptyline"): ("MODERATE", "CYP2D6 inhibition elevates tricyclic levels."),
    ("cetirizine", "alcohol"): ("MODERATE", "Additive central nervous system sedation."),
    ("acyclovir", "methotrexate"): ("MODERATE", "Potential additive nephrotoxicity / delayed clearance."),
}

# Active topical conflict pairs (candidate salt vs current_routine_actives)
ACTIVE_CONFLICT_RULES: list[tuple[set[str], set[str], str]] = [
    (
        {"benzoyl peroxide"},
        {"tretinoin", "retinol"},
        "Active Conflict: Avoid layering Benzoyl Peroxide at the same time as Tretinoin/Retinol (causes inactivation & irritation; separate AM/PM).",
    ),
    (
        {"adapalene", "tretinoin"},
        {"adapalene", "tretinoin", "retinol"},
        "Duplicate Retinoid: Avoid stacking multiple retinoids simultaneously (high barrier irritation risk).",
    ),
    (
        {"salicylic acid", "coal tar", "azelaic acid"},
        {"glycolic acid", "lactic acid", "salicylic acid", "tretinoin", "adapalene"},
        "Exfoliant Stacking: Alternate nights with existing acids/retinoids to avoid chemical barrier burn.",
    ),
]

DELICATE_SITES = {
    AnatomicalSite.EYELIDS,
    AnatomicalSite.LIPS_MUCOSA,
    AnatomicalSite.FACE,
    AnatomicalSite.GROIN,
    AnatomicalSite.UNDERARMS,
}
ULTRA_DELICATE_SITES = {AnatomicalSite.EYELIDS, AnatomicalSite.LIPS_MUCOSA}
HAIR_BEARING_SITES = {AnatomicalSite.SCALP, AnatomicalSite.BEARD}
SCALP_ONLY_OPTION_IDS = {
    "psor_otc_coal_tar_shampoo",
    "psor_rx_clobetasol_scalp_lotion",
    "scalp_otc_keto2",
    "scalp_otc_selenium",
    "scalp_rx_clobetasol_lotion",
    "seb_otc_ketoconazole_shampoo",
    "seb_otc_selenium_sulfide_shampoo",
    "seb_rx_scalp_steroid_lotion",
}
HAIR_COMPATIBLE_VEHICLES = {"SHAMPOO", "LOTION", "FOAM", "SERUM", "GEL", "SOLUTION", "TABLET", "CAPSULE", "PROCEDURE"}
BARRIER_TRIGGER_IDS = {"t05_cold_dry_windburn", "t09_overwashing_barrier_strip", "t01_home_remedy_irritant"}


# ==============================================================================
# v1.0 GATES 1-3: PMBJP CHEMICAL EQUIVALENCE VERIFICATION
# ==============================================================================
def _norm_strength(s: str) -> str:
    txt = re.sub(r"\s+", "", str(s).lower().strip())
    txt = txt.replace("w/w", "").replace("w/v", "").replace("v/v", "")
    return "" if txt in {"n/a", "na", "none", ""} else txt


def _norm_vehicle(v: str) -> str:
    return str(v or "").strip().upper()


def _vehicles_compatible(req_vehicle: str, pmbjp_vehicle: str) -> bool:
    rv = _norm_vehicle(req_vehicle)
    pv = _norm_vehicle(pmbjp_vehicle)
    if rv == pv:
        return True
    if rv in {"TABLET", "CAPSULE"} and pv == "TABLET/CAPSULE":
        return True
    if rv == "LOTION" and pv in {"SOLUTION", "SUSPENSION"}:
        return True
    if rv == "SHAMPOO" and pv in {"LOTION", "SUSPENSION"}:
        return False
    return False


def match_pmbjp_with_equivalence_gates(
    option: TreatmentOption,
    catalog: MedicineCatalog | None,
    audit_log: list[GateEvent],
) -> list[PMBJPGenericMatch]:
    """
    Execute v1.0 Gates 1-3 against loaded PMBJP rows:
      G1: Salt Subset Verification (exact salt set equality)
      G2: Dosage Alignment (strict numeric strength equality)
      G3: Vehicle Match (dosage form compatibility)
    """
    if not catalog or not catalog.pmbjp_rows or not option.salts:
        return []

    req_salts_map = {s.salt.lower().strip(): _norm_strength(s.strength) for s in option.salts}
    req_salt_set = set(req_salts_map.keys())
    salt_key = "+".join(sorted(req_salt_set))
    comm_avg = catalog.commercial_avg_price_by_salt.get(salt_key)

    matches: list[PMBJPGenericMatch] = []
    rejected_dosage = 0
    rejected_vehicle = 0

    for row in catalog.pmbjp_rows:
        row_salts_map = {s.salt.lower().strip(): _norm_strength(s.strength) for s in row["salts"]}
        row_salt_set = set(row_salts_map.keys())

        # Gate 1: Exact Salt Subset / Set Verification
        if row_salt_set != req_salt_set:
            continue

        # Gate 2: Strict Dosage Alignment
        dosage_ok = True
        for salt_name, req_st in req_salts_map.items():
            row_st = row_salts_map.get(salt_name, "")
            if req_st and row_st and req_st != row_st:
                dosage_ok = False
                break
            if req_st and not row_st:
                dosage_ok = False
                break
        if not dosage_ok:
            rejected_dosage += 1
            continue

        # Gate 3: Vehicle Compatibility Match
        if not _vehicles_compatible(option.vehicle, row["vehicle"]):
            rejected_vehicle += 1
            continue

        raw_mrp = row.get("mrp_inr")
        mrp = float(raw_mrp) if (raw_mrp is not None and float(raw_mrp) > 0) else None
        savings_pct = None
        if mrp is not None and comm_avg and comm_avg > 0 and mrp < comm_avg:
            savings_pct = round((1.0 - mrp / comm_avg) * 100.0, 1)

        matches.append(
            PMBJPGenericMatch(
                product_name=row["product_name"],
                generic_code=row.get("generic_code"),
                mrp_inr=mrp,
                vehicle=row["vehicle"],
                salts=row["salts"],
                equivalence_verified=True,
                commercial_avg_inr=comm_avg,
                estimated_savings_pct=savings_pct,
            )
        )

    matches.sort(key=lambda m: (m.mrp_inr is None, m.mrp_inr or 999999.0))

    if rejected_dosage > 0:
        audit_log.append(GateEvent(
            gate_id="G2_DOSAGE_ALIGN",
            gate_name="Dosage Alignment (v1.0)",
            triggered=True,
            action="BLOCK",
            message=f"Rejected {rejected_dosage} PMBJP SKU(s) for '{option.display_name}' due to strength mismatch.",
            affected_option_id=option.option_id,
        ))
    if rejected_vehicle > 0:
        audit_log.append(GateEvent(
            gate_id="G3_VEHICLE_MATCH",
            gate_name="Vehicle Match (v1.0)",
            triggered=True,
            action="BLOCK",
            message=f"Rejected {rejected_vehicle} PMBJP SKU(s) for '{option.display_name}' due to incompatible dosage form.",
            affected_option_id=option.option_id,
        ))
    if matches:
        audit_log.append(GateEvent(
            gate_id="G1_G3_PMBJP_VERIFIED",
            gate_name="PMBJP Chemical Equivalence (Gates 1-3)",
            triggered=False,
            action="PASS",
            message=f"Verified {len(matches)} exact PMBJP generic match(es) for '{option.display_name}'.",
            affected_option_id=option.option_id,
        ))

    return matches[:3]


# ==============================================================================
# v2.0 GATES 4-12: PATIENT-AWARE CLINICAL SAFETY ENGINE
# ==============================================================================
def _check_ddi(
    option: TreatmentOption,
    ongoing_meds: list[str],
    catalog: MedicineCatalog | None,
) -> tuple[str | None, str | None]:
    """Return (severity, detail) for highest-severity DDI found, or (None, None)."""
    if not ongoing_meds or not option.salts:
        return None, None

    meds_norm = [m.strip().lower() for m in ongoing_meds if m.strip()]
    cand_salts = [s.salt.strip().lower() for s in option.salts]

    highest_sev: str | None = None
    highest_msg: str | None = None
    sev_rank = {"SERIOUS": 3, "MAJOR": 3, "MODERATE": 2, "MINOR": 1}

    for cs in cand_salts:
        for om in meds_norm:
            for pair in ((cs, om), (om, cs)):
                if pair in CURATED_DDI_PAIRS:
                    sev, reason = CURATED_DDI_PAIRS[pair]
                    if sev_rank.get(sev, 0) > sev_rank.get(highest_sev or "", 0):
                        highest_sev = sev
                        highest_msg = f"DDI ({sev}): {cs.title()} + {om.title()} — {reason}"

            if catalog and cs in catalog.ddi_index:
                for other_drug, eff in catalog.ddi_index[cs].items():
                    if other_drug == om or other_drug in om or om in other_drug:
                        if sev_rank.get(eff, 0) > sev_rank.get(highest_sev or "", 0):
                            highest_sev = eff
                            highest_msg = f"DDI ({eff}): {cs.title()} interacts with ongoing medication '{om.title()}'."

    return highest_sev, highest_msg


def _parse_single_dose_mg(salts: list[SaltEntity]) -> float | None:
    """Extract numeric mg strength from an oral tablet/capsule salt entity if present."""
    for s in salts:
        m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*mg\b", s.strength.strip().lower())
        if m:
            return float(m.group(1))
    return None


def evaluate_clinical_case(
    profile: PatientProfile,
    case: CaseContext,
    stage1_output: ModelInferenceOutput,
    catalog: MedicineCatalog | None = None,
) -> ClinicalAssessmentResponse:
    """
    Run Stage 2 Bayesian Modifier + all 12 Deterministic Safety Gates and return
    a CDSCO-compliant Two-Tier ClinicalAssessmentResponse.
    """
    audit_log: list[GateEvent] = []
    do_not_apply: list[str] = []

    top_matches: list[ConditionPrediction] = apply_bayesian_modifier(
        stage1_output, case.active_triggers, top_n=5, case=case
    )

    identified_trigger_labels = [
        TRIGGER_INDEX[tid].label
        for tid in dict.fromkeys(case.active_triggers)
        if tid in TRIGGER_INDEX
    ]
    if case.other_trigger_note.strip():
        identified_trigger_labels.append(f"Other reported factor: {case.other_trigger_note.strip()}")

    minor_notice = (
        "Minor Notice (Ages 14-17): Please review any new skincare or medication routine "
        "with a parent/guardian and a licensed dermatologist."
        if profile.age < 18 else None
    )
    weight_refresh = profile.needs_weight_refresh()

    # ------------------------------------------------------------------
    # GATE 4 (v2.0 #1): RED-FLAG & NEOPLASTIC / OOD HARD STOP
    # ------------------------------------------------------------------
    active_red_flags = case.red_flags.active_flags()
    neoplastic_prob = sum(
        p.posterior_prob for p in top_matches if p.tier1_family == Tier1Family.NEOPLASTIC_RED_FLAG
    )
    top_is_neoplastic = bool(
        top_matches and CONDITION_GRAPH[top_matches[0].condition_id].is_neoplastic_or_red_flag
    )
    herpes_prob = sum(
        p.posterior_prob for p in top_matches if p.condition_id == "herpes_viral_eruption"
    )
    ocular_viral_risk = (
        case.anatomical_site == AnatomicalSite.EYELIDS and herpes_prob >= 0.10
    )

    if active_red_flags or stage1_output.is_ood or top_is_neoplastic or neoplastic_prob >= 0.15 or ocular_viral_risk:
        reasons = []
        if active_red_flags:
            reasons.append(f"Active clinical red flags reported: {', '.join(active_red_flags)}")
        if stage1_output.is_ood:
            reasons.append("Lesion lies outside validated training distribution (OOD uncertainty gate)")
        if top_is_neoplastic or neoplastic_prob >= 0.15:
            reasons.append("Possible high-risk / neoplastic lesion requiring direct dermoscopy")
        if ocular_viral_risk:
            reasons.append(
                f"Eyelid lesion with possible viral vesicular eruption ({herpes_prob*100:.1f}% — ocular Herpes Simplex/Zoster risk requires urgent ophthalmology or dermatology evaluation)"
            )

        reason_str = (
            "; ".join(reasons)
            + ". All treatment recommendations have been suppressed—seek immediate in-person dermatological or emergency evaluation."
        )
        audit_log.append(GateEvent(
            gate_id="G4_RED_FLAG_HARD_STOP",
            gate_name="Red-Flag & Neoplastic/OOD Hard Stop (v2.0 #1)",
            triggered=True,
            action="HARD_STOP",
            message=reason_str,
        ))
        if profile.drug_allergies:
            audit_log.append(GateEvent(
                gate_id="G7_ALLERGY_HANDOFF",
                gate_name="Allergy Handoff Notice (v2.0 #4)",
                triggered=True,
                action="WARN",
                message=f"Alert treating physician to documented drug allergies: {', '.join(profile.drug_allergies)}.",
            ))
        if profile.ongoing_oral_meds:
            audit_log.append(GateEvent(
                gate_id="G11_MED_HANDOFF",
                gate_name="Ongoing Medication Handoff (v2.0 #8)",
                triggered=True,
                action="WARN",
                message=f"Alert treating physician to ongoing oral medications: {', '.join(profile.ongoing_oral_meds)}.",
            ))
        return ClinicalAssessmentResponse(
            disclaimer_banner=MANDATORY_DISCLAIMER_BANNER,
            escalation_hard_stop=True,
            escalation_reason=reason_str,
            minor_guardian_notice=minor_notice,
            weight_refresh_prompt=weight_refresh,
            top_matches=top_matches,
            identified_triggers=identified_trigger_labels,
            do_not_apply_warnings=[
                "Do NOT apply any OTC cream, steroid, home remedy, or unprescribed medication.",
                "Seek prompt in-person medical or emergency evaluation.",
            ],
            tier1_otc_care=[],
            tier2_dermatologist_prep=[],
            lifestyle_action_plan=[
                "Seek urgent in-person evaluation by a dermatologist or emergency clinic.",
                "If experiencing eye/lip/throat swelling or breathing trouble, go to the nearest ER immediately.",
            ],
            safety_audit_log=audit_log,
        )

    if not top_matches:
        return ClinicalAssessmentResponse(
            disclaimer_banner=MANDATORY_DISCLAIMER_BANNER,
            escalation_hard_stop=True,
            escalation_reason="Insufficient model confidence to produce safe triage guidance. Please consult a dermatologist.",
            minor_guardian_notice=minor_notice,
            weight_refresh_prompt=weight_refresh,
            safety_audit_log=audit_log,
        )

    primary_pred = top_matches[0]
    primary_ck = CONDITION_GRAPH[primary_pred.condition_id]
    do_not_apply.extend(primary_ck.do_not_apply_defaults)

    # Close-Differential Cross-Family Uncertainty Check
    close_diff_note: str | None = None
    if len(top_matches) >= 2:
        second_pred = top_matches[1]
        margin = primary_pred.posterior_prob - second_pred.posterior_prob
        cross_family = primary_pred.tier1_family != second_pred.tier1_family
        if cross_family and (margin < 0.15 or primary_pred.posterior_prob < 0.50):
            second_ck = CONDITION_GRAPH[second_pred.condition_id]
            if margin < 0.15:
                close_diff_note = (
                    f"Close cross-family differential between {primary_pred.display_name} "
                    f"({primary_pred.posterior_prob*100:.1f}%) and {second_pred.display_name} "
                    f"({second_pred.posterior_prob*100:.1f}%). Confirm in person with a dermatologist before starting targeted medication."
                )
            else:
                close_diff_note = (
                    f"Moderate overall confidence for {primary_pred.display_name} "
                    f"({primary_pred.posterior_prob*100:.1f}%; next alternative {second_pred.display_name} at {second_pred.posterior_prob*100:.1f}%). "
                    f"Confirm in person with a dermatologist if symptoms persist or worsen."
                )
            audit_log.append(GateEvent(
                gate_id="G4B_CLOSE_DIFFERENTIAL",
                gate_name="Close Cross-Family Differential Uncertainty Check",
                triggered=True,
                action="WARN",
                message=close_diff_note,
            ))
            for warn_item in second_ck.do_not_apply_defaults:
                if warn_item not in do_not_apply:
                    do_not_apply.append(warn_item)

    # ------------------------------------------------------------------
    # GATE 8 (v2.0 #5): LONGITUDINAL NON-RESPONSE ESCALATION
    # ------------------------------------------------------------------
    longitudinal_escalate = (
        (case.followup_day >= 14 and case.followup_trend in {FollowUpTrend.UNCHANGED, FollowUpTrend.WORSENING})
        or (case.followup_day >= 7 and case.followup_trend == FollowUpTrend.WORSENING)
    )
    escalation_reason: str | None = None
    if longitudinal_escalate:
        escalation_reason = (
            f"Longitudinal Escalation (Day {case.followup_day}, trend: {case.followup_trend.value}): "
            "No improvement or worsening observed on follow-up. Stop self-care escalation and book an in-person dermatologist consultation."
        )
        audit_log.append(GateEvent(
            gate_id="G8_LONGITUDINAL_ESCALATION",
            gate_name="Longitudinal Non-Response Escalation (v2.0 #5)",
            triggered=True,
            action="WARN",
            message=escalation_reason,
        ))

    infectious_prob = sum(
        p.posterior_prob
        for p in top_matches
        if p.tier1_family in {Tier1Family.FUNGAL_INFECTIONS, Tier1Family.BACTERIAL_VIRAL_PARASITIC}
        or CONDITION_GRAPH[p.condition_id].is_infectious
    )
    prior_steroid_misuse = any(
        t in {"t01_otc_steroid_cream", "t01_rebound_flare"} for t in case.active_triggers
    )
    barrier_compromised = case.compromised_barrier or any(
        t in BARRIER_TRIGGER_IDS for t in case.active_triggers
    )

    if prior_steroid_misuse and infectious_prob >= 0.25:
        msg = "STOP any OTC steroid or triple-combination cream immediately—steroids worsen fungal/bacterial infections (Tinea Incognito risk)."
        if msg not in do_not_apply:
            do_not_apply.insert(0, msg)

    if profile.is_pregnant and profile.current_routine_actives:
        preg_unsafe_actives = {"tretinoin", "retinol", "adapalene", "tazarotene", "isotretinoin", "hydroquinone"}
        flagged_actives = [
            act for act in profile.current_routine_actives
            if any(bad in act.strip().lower() for bad in preg_unsafe_actives)
        ]
        if flagged_actives:
            joined = ", ".join(a.title() for a in flagged_actives)
            preg_msg = (
                f"Pregnancy Routine Alert: PAUSE your existing skincare active(s) ({joined}) during pregnancy "
                "and confirm pregnancy-safe alternatives with your OB-GYN or dermatologist."
            )
            audit_log.append(GateEvent(
                gate_id="G6_AGE_PREG_SYSTEMIC",
                gate_name="Age / Pregnancy / Systemic Contraindication Gate (v2.0 #3)",
                triggered=True,
                action="WARN",
                message=preg_msg,
            ))
            if preg_msg not in do_not_apply:
                do_not_apply.insert(0, preg_msg)

    # ------------------------------------------------------------------
    # EVALUATE CANDIDATE OPTIONS THROUGH GATES 5-7, 9-12 & 1-3
    # ------------------------------------------------------------------
    tier1_cards: list[EvaluatedTreatmentCard] = []
    tier2_cards: list[EvaluatedTreatmentCard] = []

    all_candidates = list(primary_ck.otc_options) + list(primary_ck.rx_discussion_options)

    asymptomatic_profile = (
        case.itch_score == 0
        and case.pain_burning_score == 0
        and not case.has_scaling
        and not case.compromised_barrier
    )
    if primary_ck.condition_id == "scabies" and asymptomatic_profile:
        msg = (
            "Symptom discordance: scabies typically presents with prominent pruritus (itch=0, pain=0, no scaling). "
            "Targeted scabicides (permethrin, ivermectin) are suppressed pending in-person clinical confirmation."
        )
        audit_log.append(GateEvent(
            gate_id="G4C_SYMPTOM_DISCORDANCE",
            gate_name="Symptom Discordance Gate",
            triggered=True,
            action="BLOCK",
            message=msg,
        ))
        if msg not in do_not_apply:
            do_not_apply.append(msg)

    for opt in all_candidates:
        if (
            (opt.option_id in SCALP_ONLY_OPTION_IDS or _norm_vehicle(opt.vehicle) == "SHAMPOO")
            and case.anatomical_site not in HAIR_BEARING_SITES
        ):
            continue
        opt_salts_lower = {s.salt.lower().strip() for s in opt.salts}
        if primary_ck.condition_id == "scabies" and asymptomatic_profile and (
            opt_salts_lower & {"permethrin", "ivermectin"}
            or "permethrin" in opt.option_id
            or "ivermectin" in opt.option_id
        ):
            continue
        blocked = False
        badges: list[str] = []
        dosage_note: str | None = None

        if longitudinal_escalate and case.followup_trend == FollowUpTrend.WORSENING and opt.regulatory_tier == RegulatoryTier.TIER1_OTC:
            audit_log.append(GateEvent(
                gate_id="G8_LONGITUDINAL_ESCALATION",
                gate_name="Longitudinal Non-Response Escalation (v2.0 #5)",
                triggered=True,
                action="BLOCK",
                message=f"Blocked OTC option '{opt.display_name}' due to worsening follow-up at Day {case.followup_day}.",
                affected_option_id=opt.option_id,
            ))
            continue

        # GATE 5 (v2.0 #2): Steroid-on-Infection Block
        if opt.steroid_potency != SteroidPotency.NONE and (
            primary_ck.is_infectious or infectious_prob >= 0.25 or prior_steroid_misuse
        ):
            audit_log.append(GateEvent(
                gate_id="G5_STEROID_ON_INFECTION",
                gate_name="Steroid-on-Infection Block (v2.0 #2)",
                triggered=True,
                action="BLOCK",
                message=f"Blocked steroid '{opt.display_name}' (infectious_prob={infectious_prob:.2f}, prior_steroid_misuse={prior_steroid_misuse}).",
                affected_option_id=opt.option_id,
            ))
            warn = "Do NOT apply topical corticosteroids while fungal, bacterial, viral, or parasitic infection is suspected."
            if warn not in do_not_apply:
                do_not_apply.append(warn)
            continue

        # GATE 6 (v2.0 #3): Age / Pregnancy / Systemic Contraindication Gate
        if profile.age < opt.min_age_years:
            audit_log.append(GateEvent(
                gate_id="G6_AGE_PREG_SYSTEMIC",
                gate_name="Age / Pregnancy / Systemic Contraindication (v2.0 #3)",
                triggered=True,
                action="BLOCK",
                message=f"Blocked '{opt.display_name}': minimum age {opt.min_age_years}yr (patient is {profile.age}yr).",
                affected_option_id=opt.option_id,
            ))
            blocked = True
        elif profile.is_pregnant and not opt.pregnancy_safe:
            audit_log.append(GateEvent(
                gate_id="G6_AGE_PREG_SYSTEMIC",
                gate_name="Age / Pregnancy / Systemic Contraindication (v2.0 #3)",
                triggered=True,
                action="BLOCK",
                message=f"Blocked '{opt.display_name}': contraindicated during pregnancy.",
                affected_option_id=opt.option_id,
            ))
            blocked = True
        elif profile.is_lactating and not opt.lactation_safe:
            audit_log.append(GateEvent(
                gate_id="G6_AGE_PREG_SYSTEMIC",
                gate_name="Age / Pregnancy / Systemic Contraindication (v2.0 #3)",
                triggered=True,
                action="BLOCK",
                message=f"Blocked '{opt.display_name}': contraindicated during lactation.",
                affected_option_id=opt.option_id,
            ))
            blocked = True
        else:
            sys_norm = {c.strip().lower() for c in profile.systemic_conditions}
            conflicts = sys_norm & {c.lower() for c in opt.contraindicated_systemic}
            if conflicts:
                audit_log.append(GateEvent(
                    gate_id="G6_AGE_PREG_SYSTEMIC",
                    gate_name="Age / Pregnancy / Systemic Contraindication (v2.0 #3)",
                    triggered=True,
                    action="BLOCK",
                    message=f"Blocked '{opt.display_name}': systemic contraindication ({', '.join(sorted(conflicts))}).",
                    affected_option_id=opt.option_id,
                ))
                blocked = True
        if blocked:
            continue

        # GATE 7 (v2.0 #4): Allergy & Active-Conflict Gate
        allergies_norm = [a.strip().lower() for a in profile.drug_allergies if a.strip()]
        allergy_hit = [
            s for s in opt_salts_lower
            if any(a == s or a in s or s in a for a in allergies_norm)
        ]
        if allergy_hit:
            audit_log.append(GateEvent(
                gate_id="G7_ALLERGY_ACTIVE_CONFLICT",
                gate_name="Allergy & Active-Conflict Gate (v2.0 #4)",
                triggered=True,
                action="BLOCK",
                message=f"Blocked '{opt.display_name}': matches patient drug allergy ({', '.join(allergy_hit)}).",
                affected_option_id=opt.option_id,
            ))
            continue

        routine_norm = {r.strip().lower() for r in profile.current_routine_actives if r.strip()}
        for cand_set, routine_set, conflict_msg in ACTIVE_CONFLICT_RULES:
            if (opt_salts_lower & cand_set) and (routine_norm & routine_set):
                badges.append("Routine Active Conflict — Separate AM/PM or Alternate Nights")
                audit_log.append(GateEvent(
                    gate_id="G7_ALLERGY_ACTIVE_CONFLICT",
                    gate_name="Allergy & Active-Conflict Gate (v2.0 #4)",
                    triggered=True,
                    action="WARN",
                    message=conflict_msg,
                    affected_option_id=opt.option_id,
                ))
                break

        # GATE 9 (v2.0 #6): Weight-Based Dosage & Pediatric Ceiling Gate
        if opt.is_systemic:
            single_mg = _parse_single_dose_mg(opt.salts)
            ceiling_mg = (
                round(profile.weight_kg * opt.max_mg_per_kg_day, 1)
                if opt.max_mg_per_kg_day is not None
                else opt.max_adult_daily_mg
            )
            underweight_or_minor = profile.weight_kg < 45.0 or profile.age < 18
            exceeds_ceiling = bool(single_mg and ceiling_mg and single_mg > ceiling_mg)

            if exceeds_ceiling or underweight_or_minor:
                if opt.regulatory_tier == RegulatoryTier.TIER1_OTC and (exceeds_ceiling or profile.weight_kg < 45.0):
                    audit_log.append(GateEvent(
                        gate_id="G9_WEIGHT_PEDIATRIC_CEILING",
                        gate_name="Weight-Based Dosage & Pediatric Ceiling (v2.0 #6)",
                        triggered=True,
                        action="BLOCK",
                        message=(
                            f"Blocked adult OTC systemic '{opt.display_name}' for {profile.weight_kg} kg / {profile.age}yr "
                            f"(weight-based ceiling: {ceiling_mg} mg/day)."
                        ),
                        affected_option_id=opt.option_id,
                    ))
                    continue
                else:
                    dosage_note = (
                        f"Weight-Based Ceiling Alert ({profile.weight_kg} kg): Max safe reference ~{ceiling_mg} mg/day. "
                        "Pediatric/low-weight dose adjustment by a licensed doctor is mandatory."
                    )
                    badges.append(f"Weight-Adjusted Dosing Required ({profile.weight_kg} kg)")
                    audit_log.append(GateEvent(
                        gate_id="G9_WEIGHT_PEDIATRIC_CEILING",
                        gate_name="Weight-Based Dosage & Pediatric Ceiling (v2.0 #6)",
                        triggered=True,
                        action="MODIFY",
                        message=dosage_note,
                        affected_option_id=opt.option_id,
                    ))

        # GATE 10 (v2.0 #7): Anatomical Site Sensitivity & Scalp-Vehicle Gate
        site = case.anatomical_site
        if site in DELICATE_SITES and opt.steroid_potency in {
            SteroidPotency.MODERATE, SteroidPotency.POTENT, SteroidPotency.SUPER_POTENT
        }:
            audit_log.append(GateEvent(
                gate_id="G10_ANATOMICAL_SITE",
                gate_name="Anatomical Site Sensitivity Gate (v2.0 #7)",
                triggered=True,
                action="BLOCK",
                message=f"Blocked {opt.steroid_potency.value} steroid '{opt.display_name}' on delicate site '{site.value}'.",
                affected_option_id=opt.option_id,
            ))
            warn = f"Do NOT apply potent topical steroids on delicate skin ({site.value}) due to high skin-thinning and absorption risk."
            if warn not in do_not_apply:
                do_not_apply.append(warn)
            continue

        if site in ULTRA_DELICATE_SITES and (opt.steroid_potency != SteroidPotency.NONE or opt.is_barrier_harsh):
            audit_log.append(GateEvent(
                gate_id="G10_ANATOMICAL_SITE",
                gate_name="Anatomical Site Sensitivity Gate (v2.0 #7)",
                triggered=True,
                action="BLOCK",
                message=f"Blocked '{opt.display_name}' on ultra-delicate site '{site.value}'.",
                affected_option_id=opt.option_id,
            ))
            continue

        if site in {AnatomicalSite.GROIN, AnatomicalSite.UNDERARMS} and opt.is_barrier_harsh:
            audit_log.append(GateEvent(
                gate_id="G10_ANATOMICAL_SITE",
                gate_name="Anatomical Site Sensitivity Gate (v2.0 #7)",
                triggered=True,
                action="BLOCK",
                message=f"Blocked harsh topical '{opt.display_name}' in occluded skin fold '{site.value}'.",
                affected_option_id=opt.option_id,
            ))
            continue

        if site in HAIR_BEARING_SITES and _norm_vehicle(opt.vehicle) not in HAIR_COMPATIBLE_VEHICLES:
            audit_log.append(GateEvent(
                gate_id="G10_ANATOMICAL_SITE",
                gate_name="Anatomical Site Sensitivity Gate (v2.0 #7)",
                triggered=True,
                action="BLOCK",
                message=f"Blocked non-hair-compatible vehicle '{opt.vehicle}' ('{opt.display_name}') for {site.value}.",
                affected_option_id=opt.option_id,
            ))
            continue

        # GATE 11 (v2.0 #8): Drug-Drug Interaction (DDI) Gate
        ddi_sev, ddi_msg = _check_ddi(opt, profile.ongoing_oral_meds, catalog)
        if ddi_sev in {"SERIOUS", "MAJOR"}:
            audit_log.append(GateEvent(
                gate_id="G11_DDI_CHECK",
                gate_name="Drug-Drug Interaction Gate (v2.0 #8)",
                triggered=True,
                action="BLOCK",
                message=f"Blocked '{opt.display_name}': {ddi_msg}",
                affected_option_id=opt.option_id,
            ))
            continue
        elif ddi_sev == "MODERATE" and ddi_msg:
            badges.append("Moderate Drug Interaction — Verify With Doctor")
            audit_log.append(GateEvent(
                gate_id="G11_DDI_CHECK",
                gate_name="Drug-Drug Interaction Gate (v2.0 #8)",
                triggered=True,
                action="WARN",
                message=ddi_msg,
                affected_option_id=opt.option_id,
            ))

        # GATE 12 (v2.0 #9): Photosensitivity & Compromised-Barrier Gate
        if barrier_compromised and opt.is_barrier_harsh:
            audit_log.append(GateEvent(
                gate_id="G12_PHOTO_BARRIER",
                gate_name="Photosensitivity & Compromised-Barrier Gate (v2.0 #9)",
                triggered=True,
                action="BLOCK",
                message=f"Paused harsh active '{opt.display_name}' while skin barrier is acutely compromised.",
                affected_option_id=opt.option_id,
            ))
            warn = "Skin barrier appears acutely irritated/compromised: PAUSE harsh actives (retinoids, acids, benzoyl peroxide) and use only gentle fragrance-free moisturizer until stinging subsides."
            if warn not in do_not_apply:
                do_not_apply.append(warn)
            continue

        if opt.is_photosensitizing:
            badges.append("Night-Only Application + Daily SPF 30+ Mandatory")
            audit_log.append(GateEvent(
                gate_id="G12_PHOTO_BARRIER",
                gate_name="Photosensitivity & Compromised-Barrier Gate (v2.0 #9)",
                triggered=True,
                action="MODIFY",
                message=f"Enforced Night-Only + SPF 30+ photoprotection on '{opt.display_name}'.",
                affected_option_id=opt.option_id,
            ))

        if (
            case.anatomical_site in ULTRA_DELICATE_SITES
            and not opt.is_systemic
            and opt.vehicle.upper() in {"CREAM", "OINTMENT", "GEL", "LOTION", "POWDER", "SHAMPOO", "SOAP", "FOAM", "SERUM"}
        ):
            opt_salt_names = {s.salt.lower().strip() for s in opt.salts}
            bland_safe = bool(opt_salt_names) and opt_salt_names.issubset({"white soft paraffin", "sodium chloride"})
            if not bland_safe:
                site_label = case.anatomical_site.value.replace("_", " ")
                audit_log.append(GateEvent(
                    gate_id="G10_SITE_VEHICLE",
                    gate_name="Anatomical Site Sensitivity & Scalp-Vehicle Gate (v2.0 #7)",
                    triggered=True,
                    action="BLOCK",
                    message=(
                        f"Blocked '{opt.display_name}': non-ophthalmic topical {opt.vehicle.lower()} "
                        f"is unsafe for direct application on '{site_label}' (ocular/mucosal irritation risk)."
                    ),
                    affected_option_id=opt.option_id,
                ))
                eyelid_warn = (
                    "Do NOT apply non-ophthalmic OTC skin creams, antifungals, gels, or powders directly on the "
                    "eyelids or lip mucosa (risk of eye/corneal chemical irritation). Consult a dermatologist or ophthalmologist."
                )
                if eyelid_warn not in do_not_apply:
                    do_not_apply.insert(0, eyelid_warn)
                continue

        if opt.option_id == "viral_rx_acyclovir" and case.duration_days > 3:
            badges.append(f"Rash duration is {case.duration_days} days (>72h optimal antiviral window — confirm diagnosis with doctor)")

        # GATES 1-3 (v1.0): PMBJP Chemical Equivalence Matching
        pmbjp_matches = match_pmbjp_with_equivalence_gates(opt, catalog, audit_log)

        # CDSCO Two-Tier Regulatory Packaging
        if opt.regulatory_tier == RegulatoryTier.TIER1_OTC:
            tier1_cards.append(
                EvaluatedTreatmentCard(
                    option=opt,
                    regulatory_label="Tier 1 — OTC / Supportive Care",
                    inline_badges=badges,
                    pmbjp_matches=pmbjp_matches,
                    dosage_note=dosage_note,
                )
            )
        else:
            sched_tag = "Schedule H1" if opt.regulatory_tier == RegulatoryTier.TIER2_SCHEDULE_H1 else "Schedule H"
            tier2_cards.append(
                EvaluatedTreatmentCard(
                    option=opt,
                    regulatory_label=f"Tier 2 ({sched_tag}) — For Dermatologist Discussion Only (Prescription Required)",
                    inline_badges=badges,
                    pmbjp_matches=pmbjp_matches,
                    dosage_note=dosage_note,
                )
            )

    existing_gate_prefixes = {g.gate_id.split("_")[0] for g in audit_log}
    baseline_pass_gates = [
        ("G4", "G4_RED_FLAG_HARD_STOP", "Red-Flag & Neoplastic/OOD Screen (v2.0 #1)", "No urgent red flags, neoplastic risk, or OOD anomaly detected."),
        ("G5", "G5_STEROID_INFECTION", "Steroid-on-Infection Gate (v2.0 #2)", "Verified no inappropriate steroid recommended on infectious lesions."),
        ("G6", "G6_AGE_PREG_SYSTEMIC", "Age / Pregnancy / Systemic Gate (v2.0 #3)", "All displayed options cleared age, pregnancy/lactation, and systemic checks."),
        ("G7", "G7_ALLERGY_ACTIVE", "Allergy & Routine-Active Gate (v2.0 #4)", "No drug-allergy or active-ingredient conflicts remaining on displayed cards."),
        ("G8", "G8_LONGITUDINAL_ESCALATION", "Longitudinal Non-Response Escalation (v2.0 #5)", f"Follow-up Day {case.followup_day} (trend: {case.followup_trend.value}) — no longitudinal escalation triggered."),
        ("G9", "G9_WEIGHT_DOSING", "Weight-Based Dosage & Pediatric Ceiling (v2.0 #6)", f"Evaluated dosing ceilings for patient weight ({profile.weight_kg:.1f} kg)."),
        ("G10", "G10_SITE_VEHICLE", "Anatomical Site & Vehicle Gate (v2.0 #7)", f"All displayed formulations cleared for anatomical site '{case.anatomical_site.value.replace('_', ' ')}'."),
        ("G11", "G11_DDI_SCREEN", "Drug-Drug Interaction (DDI) Gate (v2.0 #8)", "No serious drug-drug interactions detected against reported oral medications."),
        ("G12", "G12_PHOTO_BARRIER", "Photosensitivity & Barrier Gate (v2.0 #9)", "Evaluated skin barrier integrity and photosensitivity precautions."),
    ]
    for prefix, gid, gname, gmsg in baseline_pass_gates:
        if prefix not in existing_gate_prefixes:
            audit_log.append(GateEvent(
                gate_id=gid,
                gate_name=gname,
                triggered=False,
                action="PASS",
                message=gmsg,
            ))

    lifestyle_out = list(primary_ck.lifestyle_plan)
    if case.anatomical_site in ULTRA_DELICATE_SITES:
        lifestyle_out = [
            item for item in lifestyle_out
            if not any(tok in item.lower() for tok in ("skin folds", "cotton clothing", "undergarments"))
        ]
        lifestyle_out.insert(
            0,
            "For eyelid or periorbital/mucosal lesions, cleanse gently with plain lukewarm water only, avoid rubbing, and seek in-person dermatologist or ophthalmologist evaluation.",
        )
    if close_diff_note:
        lifestyle_out.insert(0, close_diff_note)

    return ClinicalAssessmentResponse(
        disclaimer_banner=MANDATORY_DISCLAIMER_BANNER,
        escalation_hard_stop=False,
        escalation_reason=escalation_reason,
        minor_guardian_notice=minor_notice,
        weight_refresh_prompt=weight_refresh,
        top_matches=top_matches,
        identified_triggers=identified_trigger_labels,
        do_not_apply_warnings=do_not_apply,
        tier1_otc_care=tier1_cards,
        tier2_dermatologist_prep=tier2_cards,
        lifestyle_action_plan=lifestyle_out,
        safety_audit_log=audit_log,
    )