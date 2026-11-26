"""
Nidaan-Rx v2.0 — Phase 2B: Multi-Dataset Taxonomy Harmonizer & Shared Clinical Tabular Encoder.

Harmonizes raw dermatology labels across:
  - Google SCIN (multi-angle smartphone cases + real patient symptoms/body-site/MST metadata)
  - Fitzpatrick17k-C (Fitzpatrick17k cleaned via CleanPatrick near-duplicate & off-topic filters)
  - DermaCon-IN / South Asian clinical datasets
  - DermNet / SD-198

Maps every raw label into:
  1. Tier 1 Coarse Family (10 locked clinical families, IDs 0..9)
  2. Canonical Condition ID (links directly to CONDITION_GRAPH & 12 Safety Gates)
  3. Normalized Tier 2 Fine-Grained Class string

Also provides a unified 32-dim tabular clinical feature encoder shared identically between:
  - Offline SCIN training rows
  - Live Pillar 2 (PatientProfile) + Pillar 3 (CaseContext) API payloads
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from app.core.schemas import AnatomicalSite, CaseContext, PatientProfile, Tier1Family

TIER1_LIST: list[Tier1Family] = list(Tier1Family)
TIER1_TO_IDX: dict[Tier1Family, int] = {fam: i for i, fam in enumerate(TIER1_LIST)}
IDX_TO_TIER1: dict[int, Tier1Family] = {i: fam for i, fam in enumerate(TIER1_LIST)}

# Ordered from highest clinical specificity (neoplastic / scalp / tinea incognito) to broader families
_LABEL_RULES: list[tuple[re.Pattern[str], Tier1Family, str]] = [
    (
        re.compile(
            r"melanoma|basal cell|squamous cell|\bbcc\b|\bscc\b|sccis|bowen|kaposi|"
            r"merkel|mycosis fungoides|cutaneous t[- ]cell|lymphoma|malignant|carcinoma|"
            r"actinic keratosis|dysplastic nevus|atypical nevus|keratoacanthoma|paget",
            re.IGNORECASE,
        ),
        Tier1Family.NEOPLASTIC_RED_FLAG,
        "suspected_neoplastic_lesion",
    ),
    (
        re.compile(r"tinea capitis|scalp ringworm|kerion", re.IGNORECASE),
        Tier1Family.SCALP_FOLLICULAR,
        "tinea_capitis",
    ),
    (
        re.compile(
            r"seborrheic dermatitis|seborrhoeic|dandruff|pityriasis capitis|alopecia|"
            r"telogen effluvium|trichotillomania|lichen planopilaris|scalp psoriasis|"
            r"pseudofolliculitis|folliculitis decalvans",
            re.IGNORECASE,
        ),
        Tier1Family.SCALP_FOLLICULAR,
        "seborrheic_dermatitis_scalp",
    ),
    (
        re.compile(r"tinea incognito|steroid[- ]modified", re.IGNORECASE),
        Tier1Family.FUNGAL_INFECTIONS,
        "tinea_incognito",
    ),
    (
        re.compile(r"candid|intertrigo|moniliasis|thrush", re.IGNORECASE),
        Tier1Family.FUNGAL_INFECTIONS,
        "cutaneous_candidiasis",
    ),
    (
        re.compile(
            r"tinea|ringworm|dermatophyt|onychomycosis|pityriasis versicolor|"
            r"tinea versicolor|fungal|mycosis|malassezia",
            re.IGNORECASE,
        ),
        Tier1Family.FUNGAL_INFECTIONS,
        "tinea_corporis_cruris",
    ),
    (
        re.compile(r"scabies|pediculosis|lice|creeping eruption|larva migrans|mite", re.IGNORECASE),
        Tier1Family.BACTERIAL_VIRAL_PARASITIC,
        "scabies",
    ),
    (
        re.compile(
            r"herpes|shingles|zoster|varicella|chickenpox|molluscum|wart|verruca|"
            r"hpv|viral exanthem|hand[- ]foot[- ]mouth|pityriasis rosea",
            re.IGNORECASE,
        ),
        Tier1Family.BACTERIAL_VIRAL_PARASITIC,
        "herpes_viral_eruption",
    ),
    (
        re.compile(
            r"impetigo|folliculitis|furuncle|carbuncle|boil|abscess|cellulitis|"
            r"erysipelas|ecthyma|pyoderma|bacterial|staph|erythrasma|paronychia|leprosy|hansen",
            re.IGNORECASE,
        ),
        Tier1Family.BACTERIAL_VIRAL_PARASITIC,
        "bacterial_folliculitis_impetigo",
    ),
    (
        re.compile(
            r"acne|rosacea|perioral dermatitis|hidradenitis|milia|comedone|miliaria",
            re.IGNORECASE,
        ),
        Tier1Family.INFLAMMATORY_ACNEIFORM,
        "acne_vulgaris",
    ),
    (
        re.compile(
            r"urticaria|hive|wheal|angioedema|drug eruption|erythema multiforme|"
            r"erythema nodosum|arthropod|insect bite|bug bite|prurigo|vasculitis|"
            r"stevens[- ]johnson|fixed drug|sweet syndrome|dermographism",
            re.IGNORECASE,
        ),
        Tier1Family.URTICARIAL_REACTIVE,
        "urticaria",
    ),
    (
        re.compile(
            r"psoriasis|lichen planus|lichen nitidus|lichen striatus|ichthyosis|"
            r"pityriasis rubra|parapsoriasis|darier|keratosis pilaris",
            re.IGNORECASE,
        ),
        Tier1Family.PAPULOSQUAMOUS,
        "psoriasis_vulgaris",
    ),
    (
        re.compile(r"contact dermatitis|allergic contact|irritant contact|poison ivy|photodermatitis", re.IGNORECASE),
        Tier1Family.ECZEMATOUS_ALLERGIC,
        "contact_dermatitis",
    ),
    (
        re.compile(
            r"eczema|atopic|dermatitis|dyshidrotic|pompholyx|nummular|stasis|"
            r"lichen simplex|neurodermatitis|xerosis|asteatotic|pruritus|diaper",
            re.IGNORECASE,
        ),
        Tier1Family.ECZEMATOUS_ALLERGIC,
        "atopic_dermatitis_eczema",
    ),
    (
        re.compile(
            r"melasma|chloasma|hyperpigmentation|\bpih\b|vitiligo|leukoderma|"
            r"lentigo|freckle|ephelid|acanthosis nigricans|post[- ]inflammatory|"
            r"ashy dermatosis|lichen planus pigmentosus|hypopigmentation|pityriasis alba|cafe au lait",
            re.IGNORECASE,
        ),
        Tier1Family.PIGMENTARY,
        "melasma_pih",
    ),
    (
        re.compile(
            r"nevus|naevus|mole|seborrheic keratosis|seborrhoeic keratosis|skin tag|"
            r"acrochordon|dermatofibroma|lipoma|cyst|epidermoid|pilar cyst|keloid|"
            r"hypertrophic scar|hemangioma|angioma|cherry angioma|pyogenic granuloma|"
            r"xanthelasma|syringoma|corn|callus|fibroma|neurofibroma|benign",
            re.IGNORECASE,
        ),
        Tier1Family.BENIGN_GROWTHS,
        "benign_growth_nevus",
    ),
]


def normalize_tier2_label(raw_label: str) -> str:
    txt = raw_label.strip().lower()
    txt = re.sub(r"\([^)]*\)", " ", txt)
    txt = re.sub(r"[^a-z0-9]+", "_", txt).strip("_")
    return txt or "unknown"


def map_raw_label(raw_label: str) -> tuple[Tier1Family, str, str] | None:
    if not raw_label or not str(raw_label).strip():
        return None
    text = str(raw_label).strip()
    norm_t2 = normalize_tier2_label(text)
    for pattern, family, canonical_id in _LABEL_RULES:
        if pattern.search(text):
            return family, canonical_id, norm_t2
    return None


# ==============================================================================
# SHARED 32-DIM CLINICAL TABULAR FEATURE ENCODER (TRAINING <-> LIVE INFERENCE)
# ==============================================================================
TABULAR_DIM = 32
SITE_LIST: list[AnatomicalSite] = list(AnatomicalSite)


def encode_live_tabular_features(profile: PatientProfile, case: CaseContext) -> np.ndarray:
    """
    Encode live Pillar 2 (PatientProfile) + Pillar 3 (CaseContext) into a fixed
    float32 vector of shape (32,) for Stage-1 Gated Cross-Attention Late Fusion.
    """
    vec = np.zeros(TABULAR_DIM, dtype=np.float32)
    vec[0] = np.clip(profile.age / 100.0, 0.0, 1.0)
    vec[1] = 1.0 if profile.biological_sex == "male" else 0.0
    vec[2] = 1.0 if profile.biological_sex == "female" else 0.0
    vec[3] = np.clip((profile.fitzpatrick_tone - 1.0) / 5.0, 0.0, 1.0)
    vec[4] = np.clip((profile.monk_skin_tone - 1.0) / 9.0, 0.0, 1.0)
    vec[5] = float(np.clip(np.log1p(case.duration_days) / np.log1p(365.0), 0.0, 1.0))
    vec[6] = np.clip(case.itch_score / 10.0, 0.0, 1.0)
    vec[7] = np.clip(case.pain_burning_score / 10.0, 0.0, 1.0)
    vec[8] = 1.0 if case.has_scaling else 0.0
    vec[9] = 1.0 if (case.red_flags.fever or case.red_flags.rapidly_spreading_redness_warmth) else 0.0
    vec[10] = 1.0 if case.red_flags.pus_oozing_bleeding else 0.0

    trigs = set(case.active_triggers)
    vec[11] = 1.0 if (trigs & {"t01_otc_steroid_cream", "t01_rebound_flare", "t01_fairness_cream"}) else 0.0
    vec[12] = 1.0 if (trigs & {"t05_heat_humidity_monsoon", "t09_gym_sweat_unwashed", "t09_damp_clothes_wet_hair"}) else 0.0
    vec[13] = 1.0 if (trigs & {"t07_new_skincare_makeup", "t07_hair_dye_ppd", "t07_perfume_bindi_detergent", "t14_nickel_metal_latex", "t14_wet_work_detergents"}) else 0.0
    vec[14] = 1.0 if (trigs & {"t04_poor_irregular_sleep", "t04_high_stress"} or profile.sleep_hours < 6.0 or profile.stress_level == "high") else 0.0
    vec[15] = 1.0 if (trigs & {"t03_high_glycemic_sugar", "t03_high_dairy", "t02_whey_gym_supp", "t11_premenstrual_pcos", "t11_teen_androgen_surge"}) else 0.0
    vec[16] = 1.0 if (trigs & {"t10_household_rash_itch", "t10_pet_contact", "t09_shared_towels_combs"}) else 0.0
    vec[17] = 1.0 if (trigs & {"t05_intense_uv_sun", "t05_cold_dry_windburn"}) else 0.0
    vec[18] = 1.0 if (trigs & {"t06_hard_borewell_water", "t07_heavy_hair_oil"}) else 0.0
    vec[19] = 1.0 if case.compromised_barrier else 0.0

    site_idx = SITE_LIST.index(case.anatomical_site)
    vec[20 + site_idx] = 1.0
    return vec


_SCIN_AGE_MIDPOINTS = {
    "AGE_18_TO_29": 24.0,
    "AGE_30_TO_39": 35.0,
    "AGE_40_TO_49": 45.0,
    "AGE_50_TO_59": 55.0,
    "AGE_60_TO_69": 65.0,
    "AGE_70_TO_79": 75.0,
    "AGE_80_PLUS": 84.0,
}

_SCIN_DURATION_DAYS = {
    "ONE_DAY": 1.0,
    "LESS_THAN_ONE_WEEK": 4.0,
    "ONE_TO_FOUR_WEEKS": 14.0,
    "ONE_TO_THREE_MONTHS": 60.0,
    "THREE_TO_TWELVE_MONTHS": 180.0,
    "MORE_THAN_ONE_YEAR": 365.0,
    "MORE_THAN_FIVE_YEARS": 365.0,
}

_SCIN_BODY_PART_MAP: dict[str, AnatomicalSite] = {
    "body_parts_head_or_neck": AnatomicalSite.FACE,
    "body_parts_arm": AnatomicalSite.ARMS_LEGS,
    "body_parts_palm": AnatomicalSite.HANDS_FEET,
    "body_parts_back_of_hand": AnatomicalSite.HANDS_FEET,
    "body_parts_torso_front": AnatomicalSite.TRUNK,
    "body_parts_torso_back": AnatomicalSite.TRUNK,
    "body_parts_genitalia_or_groin": AnatomicalSite.GROIN,
    "body_parts_buttocks": AnatomicalSite.GROIN,
    "body_parts_leg": AnatomicalSite.ARMS_LEGS,
    "body_parts_foot_top_or_side": AnatomicalSite.HANDS_FEET,
    "body_parts_foot_sole": AnatomicalSite.HANDS_FEET,
}


def _truthy(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    return str(val).strip().upper() in {"YES", "TRUE", "1", "1.0"}


def encode_scin_tabular_features(row: dict[str, Any]) -> np.ndarray:
    """Encode a `google/scin` dataset row into the exact same 32-dim vector as live payloads."""
    vec = np.zeros(TABULAR_DIM, dtype=np.float32)

    age_val = _SCIN_AGE_MIDPOINTS.get(str(row.get("age_group") or "").upper(), 32.0)
    vec[0] = age_val / 100.0

    sex = str(row.get("sex_at_birth") or "").upper()
    vec[1] = 1.0 if sex == "MALE" else 0.0
    vec[2] = 1.0 if sex == "FEMALE" else 0.0

    fst_raw = str(row.get("fitzpatrick_skin_type") or row.get("dermatologist_fitzpatrick_skin_type_label_1") or "")
    m_fst = re.search(r"([1-6])", fst_raw)
    fst = float(m_fst.group(1)) if m_fst else 3.5
    vec[3] = np.clip((fst - 1.0) / 5.0, 0.0, 1.0)

    mst_raw = str(row.get("monk_skin_tone_label_india") or row.get("monk_skin_tone_label_us") or "")
    m_mst = re.search(r"(10|[1-9])", mst_raw)
    mst = float(m_mst.group(1)) if m_mst else 5.0
    vec[4] = np.clip((mst - 1.0) / 9.0, 0.0, 1.0)

    dur_days = _SCIN_DURATION_DAYS.get(str(row.get("condition_duration") or "").upper(), 14.0)
    vec[5] = float(np.clip(np.log1p(dur_days) / np.log1p(365.0), 0.0, 1.0))

    vec[6] = 0.8 if _truthy(row.get("condition_symptoms_itching")) else (0.5 if _truthy(row.get("condition_symptoms_bothersome_appearance")) else 0.0)
    vec[7] = 0.8 if (_truthy(row.get("condition_symptoms_pain")) or _truthy(row.get("condition_symptoms_burning"))) else 0.0
    vec[8] = 1.0 if _truthy(row.get("textures_rough_or_flaky")) else 0.0
    vec[9] = 1.0 if (_truthy(row.get("other_symptoms_fever")) or _truthy(row.get("other_symptoms_chills"))) else 0.0
    vec[10] = 1.0 if (_truthy(row.get("condition_symptoms_bleeding")) or _truthy(row.get("condition_symptoms_oozing"))) else 0.0

    for col, site in _SCIN_BODY_PART_MAP.items():
        if _truthy(row.get(col)):
            vec[20 + SITE_LIST.index(site)] = 1.0

    return vec