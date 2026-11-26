"""
Nidaan-Rx v2.0 — Clinical Etiology Knowledge Graph, Bayesian Modifier & PMBJP Catalog.

Implements:
  - Pillar 1: 10 Tier-1 Coarse Clinical Families + canonical Tier-2 dermatology & scalp taxonomy
              (with alias mapping hook for 150-250+ fine-grained dataset classes in Phase 2)
  - Pillar 3: Complete 14 Clinical Etiology & Trigger Buckets + dynamic Card 2-4 bucket selector
  - Pillar 4: Stage 2 Transparent Bayesian Clinical Modifier (log-odds with per-bucket dampening)
  - PMBJP (Jan Aushadhi) + 1mg reference catalog loader for chemical equivalence & DDI lookups
"""

from __future__ import annotations
from app.core.atomic_triggers import base_trigger_id, expand_trigger_item

import ast
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .schemas import (
    ConditionKnowledge,
    ConditionPrediction,
    ModelInferenceOutput,
    PatientProfile,
    RegulatoryTier,
    SaltEntity,
    SteroidPotency,
    Tier1Family,
    TreatmentOption,
    TriggerBucket,
    TriggerItem,
)


# ==============================================================================
# 1. PILLAR 3 — COMPLETE 14 CLINICAL ETIOLOGY & TRIGGER BUCKETS
# ==============================================================================
TRIGGER_BUCKETS: list[TriggerBucket] = [
    TriggerBucket(
        bucket_id=1,
        slug="prior_otc_steroid",
        title="Prior OTC / Steroid Creams (India Critical)",
        india_critical=True,
        triggers=[
            TriggerItem(trigger_id="t01_otc_steroid_cream", bucket_id=1, label="Used unprescribed steroid / triple-combo cream (e.g., Betnovate, Panderm, Quadriderm)"),
            TriggerItem(trigger_id="t01_fairness_cream", bucket_id=1, label="Used fairness / skin-lightening cream or unknown salon cream"),
            TriggerItem(trigger_id="t01_rebound_flare", bucket_id=1, label="Rash improved briefly with cream then flared back worse/larger"),
            TriggerItem(trigger_id="t01_home_remedy_irritant", bucket_id=1, label="Applied lemon, garlic, baking soda, toothpaste, or harsh home remedy"),
        ],
    ),
    TriggerBucket(
        bucket_id=2,
        slug="oral_meds_supplements",
        title="Oral Medications & Supplements",
        triggers=[
            TriggerItem(trigger_id="t02_recent_antibiotic_nsaid", bucket_id=2, label="Started new oral antibiotic, painkiller (NSAID), or anticonvulsant in last 1-6 weeks"),
            TriggerItem(trigger_id="t02_oral_steroids", bucket_id=2, label="Taking oral steroids or systemic immunosuppressants"),
            TriggerItem(trigger_id="t02_whey_gym_supp", bucket_id=2, label="Taking whey protein, high-dose biotin/B12, or gym supplements"),
            TriggerItem(trigger_id="t02_ayurvedic_herbal", bucket_id=2, label="Started Ayurvedic, Unani, or unlabelled herbal powders/tablets"),
        ],
    ),
    TriggerBucket(
        bucket_id=3,
        slug="diet_gut_hydration",
        title="Diet, Gut & Hydration",
        triggers=[
            TriggerItem(trigger_id="t03_high_glycemic_sugar", bucket_id=3, label="Frequent sweets, sugary drinks, white bread, or high-glycemic junk food"),
            TriggerItem(trigger_id="t03_high_dairy", bucket_id=3, label="Heavy daily dairy intake (milk, paneer, ice cream, shakes)"),
            TriggerItem(trigger_id="t03_food_allergen_trigger", bucket_id=3, label="Flare soon after seafood, nuts, eggs, or spicy/fermented foods"),
            TriggerItem(trigger_id="t03_low_hydration_alcohol", bucket_id=3, label="Low daily water intake, heavy alcohol, or smoking"),
        ],
    ),
    TriggerBucket(
        bucket_id=4,
        slug="sleep_stress_habits",
        title="Sleep, Stress & Habits",
        triggers=[
            TriggerItem(trigger_id="t04_poor_irregular_sleep", bucket_id=4, label="Sleeping <6 hrs, late nights, or irregular/night-shift cycle"),
            TriggerItem(trigger_id="t04_high_stress", bucket_id=4, label="Recent exam, workplace, or major emotional stress"),
            TriggerItem(trigger_id="t04_picking_scratching", bucket_id=4, label="Frequent face-touching, pimple-popping, or nighttime scratching"),
        ],
    ),
    TriggerBucket(
        bucket_id=5,
        slug="climate_weather_thermal",
        title="Climate, Weather & Thermal",
        triggers=[
            TriggerItem(trigger_id="t05_heat_humidity_monsoon", bucket_id=5, label="Hot, humid, or monsoon damp weather exposure"),
            TriggerItem(trigger_id="t05_cold_dry_windburn", bucket_id=5, label="Extreme cold, dry winter air, or windburn"),
            TriggerItem(trigger_id="t05_intense_uv_sun", bucket_id=5, label="Heavy outdoor sun/UV exposure or recent sunburn without SPF"),
        ],
    ),
    TriggerBucket(
        bucket_id=6,
        slug="water_air_indoor",
        title="Water, Air & Indoor Environment",
        triggers=[
            TriggerItem(trigger_id="t06_hard_borewell_water", bucket_id=6, label="Washing hair/face regularly with hard or borewell water"),
            TriggerItem(trigger_id="t06_pool_chlorine", bucket_id=6, label="Frequent swimming pool chlorine exposure"),
            TriggerItem(trigger_id="t06_dust_damp_bedding", bucket_id=6, label="High dust/pollution, damp room, or unwashed pillowcases/bedding"),
        ],
    ),
    TriggerBucket(
        bucket_id=7,
        slug="cosmetics_topicals_cultural",
        title="Cosmetics, Topicals & Cultural Products",
        triggers=[
            TriggerItem(trigger_id="t07_new_skincare_makeup", bucket_id=7, label="Started new cleanser, moisturizer, sunscreen, serum, or makeup"),
            TriggerItem(trigger_id="t07_heavy_hair_oil", bucket_id=7, label="Frequent heavy hair oiling or occlusive pomades"),
            TriggerItem(trigger_id="t07_hair_dye_ppd", bucket_id=7, label="Recent hair dye (PPD), henna/mehndi, or chemical hair treatment"),
            TriggerItem(trigger_id="t07_perfume_bindi_detergent", bucket_id=7, label="New perfume, laundry detergent, or bindi/kumkum adhesive"),
        ],
    ),
    TriggerBucket(
        bucket_id=8,
        slug="mechanical_friction_trauma",
        title="Mechanical Friction & Physical Trauma",
        triggers=[
            TriggerItem(trigger_id="t08_tight_synthetic_clothes", bucket_id=8, label="Tight jeans, synthetic undergarments, or chafing fabric"),
            TriggerItem(trigger_id="t08_helmet_cap_strap", bucket_id=8, label="Daily motorcycle helmet, tight cap, watch strap, or eyewear friction"),
            TriggerItem(trigger_id="t08_shaving_waxing_threading", bucket_id=8, label="Recent shaving, waxing, threading, or harsh scrubbing"),
            TriggerItem(trigger_id="t08_cut_scratch_koebner", bucket_id=8, label="Recent cut, scrape, or skin injury at the lesion site"),
        ],
    ),
    TriggerBucket(
        bucket_id=9,
        slug="sweat_occlusion_hygiene",
        title="Sweat, Occlusion & Hygiene",
        triggers=[
            TriggerItem(trigger_id="t09_gym_sweat_unwashed", bucket_id=9, label="Heavy gym/outdoor sweating without prompt wash"),
            TriggerItem(trigger_id="t09_damp_clothes_wet_hair", bucket_id=9, label="Staying in damp clothes/socks or sleeping with wet hair"),
            TriggerItem(trigger_id="t09_overwashing_barrier_strip", bucket_id=9, label="Overwashing with harsh soaps or daily physical exfoliation"),
            TriggerItem(trigger_id="t09_shared_towels_combs", bucket_id=9, label="Shared towels, combs, razors, helmets, or footwear"),
        ],
    ),
    TriggerBucket(
        bucket_id=10,
        slug="contagion_contacts_bites",
        title="Contagion, Contacts & Bites",
        triggers=[
            TriggerItem(trigger_id="t10_household_rash_itch", bucket_id=10, label="Family, hostel, or gym contacts have similar itchy rash"),
            TriggerItem(trigger_id="t10_travel_hotel_bedding", bucket_id=10, label="Recent travel or stay in hotel/hostel bedding"),
            TriggerItem(trigger_id="t10_pet_contact", bucket_id=10, label="Close contact with pets (dogs/cats with patchy hair loss or fleas)"),
            TriggerItem(trigger_id="t10_insect_bites", bucket_id=10, label="Recent mosquito, bedbug, or insect bites"),
        ],
    ),
    TriggerBucket(
        bucket_id=11,
        slug="hormonal_endocrine",
        title="Hormonal & Endocrine Shifts",
        triggers=[
            TriggerItem(trigger_id="t11_premenstrual_pcos", bucket_id=11, label="Pre-menstrual flares, irregular periods, or PCOS"),
            TriggerItem(trigger_id="t11_teen_androgen_surge", bucket_id=11, label="Adolescent oiliness / hormonal surge (ages 14-19)"),
            TriggerItem(trigger_id="t11_pregnancy_postpartum", bucket_id=11, label="Current pregnancy or postpartum hormonal shift"),
            TriggerItem(trigger_id="t11_thyroid_insulin_shift", bucket_id=11, label="Known thyroid imbalance or rapid weight/insulin changes"),
        ],
    ),
    TriggerBucket(
        bucket_id=12,
        slug="recent_illness_post_infectious",
        title="Recent Illness & Post-Infectious",
        triggers=[
            TriggerItem(trigger_id="t12_recent_viral_fever", bucket_id=12, label="Recent viral fever, cold/flu, dengue, or COVID (last 1-3 months)"),
            TriggerItem(trigger_id="t12_sore_throat_strep", bucket_id=12, label="Recent sore throat / tonsillitis (last 2-4 weeks)"),
            TriggerItem(trigger_id="t12_recent_vaccination", bucket_id=12, label="Recent vaccination or immune trigger"),
        ],
    ),
    TriggerBucket(
        bucket_id=13,
        slug="nutritional_deficiency",
        title="Nutritional & Deficiency Clues",
        triggers=[
            TriggerItem(trigger_id="t13_crash_diet_low_protein", bucket_id=13, label="Recent crash dieting or very low protein intake"),
            TriggerItem(trigger_id="t13_vitamin_mineral_def", bucket_id=13, label="Known/suspected Vitamin D3, B12, Iron/Ferritin, or Zinc deficiency"),
        ],
    ),
    TriggerBucket(
        bucket_id=14,
        slug="occupational_material",
        title="Occupational & Material Exposure",
        triggers=[
            TriggerItem(trigger_id="t14_wet_work_detergents", bucket_id=14, label="Frequent wet hands, dishwashing soaps, or chemical cleaners"),
            TriggerItem(trigger_id="t14_nickel_metal_latex", bucket_id=14, label="Contact with nickel/imitation jewelry, rubber, or latex gloves"),
            TriggerItem(trigger_id="t14_cement_solvents_plants", bucket_id=14, label="Exposure to cement, industrial solvents, lab chemicals, or plants"),
        ],
    ),
]

TRIGGER_INDEX: dict[str, TriggerItem] = {
    t.trigger_id: t for b in TRIGGER_BUCKETS for t in b.triggers
}
BUCKET_BY_ID: dict[int, TriggerBucket] = {b.bucket_id: b for b in TRIGGER_BUCKETS}


def _register_atomic_triggers() -> None:
    for _bucket in TRIGGER_BUCKETS:
        for _orig in _bucket.triggers:
            for _atom in expand_trigger_item(_orig):
                if _atom.trigger_id not in TRIGGER_INDEX:
                    TRIGGER_INDEX[_atom.trigger_id] = _atom


_register_atomic_triggers()

FAMILY_PRIORITY_BUCKETS: dict[Tier1Family, list[int]] = {
    Tier1Family.INFLAMMATORY_ACNEIFORM: [11, 3, 4, 7, 2, 9],
    Tier1Family.FUNGAL_INFECTIONS: [9, 5, 10, 8, 11],
    Tier1Family.BACTERIAL_VIRAL_PARASITIC: [10, 8, 9, 12, 6],
    Tier1Family.ECZEMATOUS_ALLERGIC: [7, 14, 5, 6, 9, 4],
    Tier1Family.PAPULOSQUAMOUS: [4, 12, 8, 5, 2],
    Tier1Family.PIGMENTARY: [5, 11, 7, 13, 4],
    Tier1Family.SCALP_FOLLICULAR: [6, 7, 9, 8, 4, 13],
    Tier1Family.URTICARIAL_REACTIVE: [2, 3, 5, 12, 10, 4],
    Tier1Family.BENIGN_GROWTHS: [5, 8, 11, 4],
    Tier1Family.NEOPLASTIC_RED_FLAG: [5, 8, 12],
}


# ==============================================================================
# 2. PILLAR 1 & 4 — CANONICAL DERMATOLOGY CONDITION KNOWLEDGE GRAPH
# ==============================================================================
CONDITION_GRAPH: dict[str, ConditionKnowledge] = {
    "acne_vulgaris": ConditionKnowledge(
        condition_id="acne_vulgaris",
        display_name="Acne Vulgaris",
        tier1_family=Tier1Family.INFLAMMATORY_ACNEIFORM,
        summary="Inflammatory pilosebaceous disorder presenting with comedones, papules, or pustules.",
        otc_options=[
            TreatmentOption(
                option_id="acne_otc_bpo",
                display_name="Benzoyl Peroxide 2.5% Gel / Wash",
                salts=[SaltEntity(salt="benzoyl peroxide", strength="2.5%")],
                vehicle="GEL",
                regulatory_tier=RegulatoryTier.TIER1_OTC,
                posology="Apply a thin pea-sized layer once daily at night (or short-contact wash for 2 min).",
                is_photosensitizing=True,
                is_barrier_harsh=True,
                min_age_years=12,
                pregnancy_safe=True,
                lactation_safe=True,
            ),
            TreatmentOption(
                option_id="acne_otc_salicylic",
                display_name="Salicylic Acid 2% Cleanser / Gel",
                salts=[SaltEntity(salt="salicylic acid", strength="2%")],
                vehicle="GEL",
                regulatory_tier=RegulatoryTier.TIER1_OTC,
                posology="Use gently once daily on affected oily/comedonal zones; follow with non-comedogenic moisturizer.",
                is_photosensitizing=True,
                is_barrier_harsh=True,
                min_age_years=12,
                pregnancy_safe=True,
                lactation_safe=True,
            ),
            TreatmentOption(
                option_id="acne_otc_niacinamide",
                display_name="Niacinamide 4% Gel (Barrier-Safe Supportive Care)",
                salts=[SaltEntity(salt="nicotinamide", strength="4%")],
                vehicle="GEL",
                regulatory_tier=RegulatoryTier.TIER1_OTC,
                posology="Apply thinly twice daily after gentle cleansing.",
                is_photosensitizing=False,
                is_barrier_harsh=False,
                min_age_years=10,
                pregnancy_safe=True,
                lactation_safe=True,
            ),
        ],
        rx_discussion_options=[
            TreatmentOption(
                option_id="acne_rx_adapalene",
                display_name="Adapalene 0.1% Gel",
                salts=[SaltEntity(salt="adapalene", strength="0.1%")],
                vehicle="GEL",
                regulatory_tier=RegulatoryTier.TIER2_SCHEDULE_H,
                posology="Dermatologist-guided pea-sized application at night, starting alternate days.",
                is_photosensitizing=True,
                is_barrier_harsh=True,
                min_age_years=12,
                pregnancy_safe=False,
                lactation_safe=True,
            ),
            TreatmentOption(
                option_id="acne_rx_clinda_adapalene",
                display_name="Clindamycin 1% + Adapalene 0.1% Gel",
                salts=[
                    SaltEntity(salt="clindamycin", strength="1%"),
                    SaltEntity(salt="adapalene", strength="0.1%"),
                ],
                vehicle="GEL",
                regulatory_tier=RegulatoryTier.TIER2_SCHEDULE_H,
                posology="Prescription combination evaluated for moderate inflammatory acne (short course to limit antibiotic resistance).",
                is_photosensitizing=True,
                is_barrier_harsh=True,
                min_age_years=12,
                pregnancy_safe=False,
                lactation_safe=True,
            ),
            TreatmentOption(
                option_id="acne_rx_doxycycline",
                display_name="Oral Doxycycline 100mg (Moderate-Severe Acne Discussion)",
                salts=[SaltEntity(salt="doxycycline", strength="100mg")],
                vehicle="CAPSULE",
                regulatory_tier=RegulatoryTier.TIER2_SCHEDULE_H1,
                posology="Systemic antibiotic evaluated by dermatologist for nodulocystic/widespread acne.",
                is_photosensitizing=True,
                is_systemic=True,
                min_age_years=12,
                pregnancy_safe=False,
                lactation_safe=False,
                contraindicated_systemic=["liver_disease"],
                max_adult_daily_mg=200.0,
                max_mg_per_kg_day=4.4,
            ),
        ],
        do_not_apply_defaults=[
            "Do NOT pop, squeeze, or scrub active pimples (increases scarring and post-inflammatory hyperpigmentation).",
            "Do NOT apply over-the-counter topical steroid or fairness creams on acne.",
        ],
        lifestyle_plan=[
            "Wash face gently after heavy sweating/gym sessions using a non-comedogenic cleanser.",
            "Reduce high-glycemic snacks and excess whey/dairy spikes; prioritize 7-8 hours of regular sleep.",
            "Change pillowcases twice weekly and clean helmet straps/phone screens.",
        ],
        trigger_lrs={
            "t01_otc_steroid_cream": 1.6,
            "t01_fairness_cream": 1.8,
            "t02_whey_gym_supp": 2.4,
            "t02_oral_steroids": 2.0,
            "t03_high_glycemic_sugar": 1.9,
            "t03_high_dairy": 1.7,
            "t04_poor_irregular_sleep": 1.08,
            "t04_high_stress": 1.7,
            "t04_picking_scratching": 1.8,
            "t07_new_skincare_makeup": 1.6,
            "t07_heavy_hair_oil": 1.8,
            "t08_helmet_cap_strap": 1.7,
            "t09_gym_sweat_unwashed": 1.8,
            "t11_premenstrual_pcos": 2.5,
            "t11_teen_androgen_surge": 2.4,
        },
    ),
    "tinea_corporis_cruris": ConditionKnowledge(
        condition_id="tinea_corporis_cruris",
        display_name="Tinea Corporis / Tinea Cruris (Ringworm / Dermatophytosis)",
        tier1_family=Tier1Family.FUNGAL_INFECTIONS,
        is_infectious=True,
        summary="Superficial dermatophyte infection presenting as itchy annular scaly plaques with active borders.",
        otc_options=[
            TreatmentOption(
                option_id="tinea_otc_clotrimazole_cream",
                display_name="Clotrimazole 1% Cream",
                salts=[SaltEntity(salt="clotrimazole", strength="1%")],
                vehicle="CREAM",
                regulatory_tier=RegulatoryTier.TIER1_OTC,
                posology="Apply thinly twice daily extending 2 cm beyond the rash edge for 2-4 weeks; continue 1 week after clearing.",
                min_age_years=2,
                pregnancy_safe=True,
                lactation_safe=True,
            ),
            TreatmentOption(
                option_id="tinea_otc_clotrimazole_powder",
                display_name="Clotrimazole 1% Dusting Powder",
                salts=[SaltEntity(salt="clotrimazole", strength="1%")],
                vehicle="POWDER",
                regulatory_tier=RegulatoryTier.TIER1_OTC,
                posology="Dust lightly in dry skin folds (groin/underarms) after thorough drying to reduce moisture.",
                min_age_years=2,
                pregnancy_safe=True,
                lactation_safe=True,
            ),
            TreatmentOption(
                option_id="tinea_otc_terbinafine",
                display_name="Terbinafine 1% Cream",
                salts=[SaltEntity(salt="terbinafine", strength="1%")],
                vehicle="CREAM",
                regulatory_tier=RegulatoryTier.TIER1_OTC,
                posology="Apply thinly once or twice daily on clean, completely dry skin.",
                min_age_years=12,
                pregnancy_safe=True,
                lactation_safe=True,
            ),
        ],
        rx_discussion_options=[
            TreatmentOption(
                option_id="tinea_rx_luliconazole",
                display_name="Luliconazole 1% Cream / Lotion",
                salts=[SaltEntity(salt="luliconazole", strength="1%")],
                vehicle="CREAM",
                regulatory_tier=RegulatoryTier.TIER2_SCHEDULE_H,
                posology="Once-daily prescription azole often evaluated for resistant Indian dermatophytosis.",
                min_age_years=12,
                pregnancy_safe=False,
                lactation_safe=False,
            ),
            TreatmentOption(
                option_id="tinea_rx_itraconazole",
                display_name="Oral Itraconazole 100mg / 200mg Capsule",
                salts=[SaltEntity(salt="itraconazole", strength="100mg")],
                vehicle="CAPSULE",
                regulatory_tier=RegulatoryTier.TIER2_SCHEDULE_H,
                posology="Systemic antifungal evaluated by a dermatologist for extensive, recurrent, or steroid-modified tinea.",
                is_systemic=True,
                min_age_years=12,
                pregnancy_safe=False,
                lactation_safe=False,
                contraindicated_systemic=["liver_disease"],
                max_adult_daily_mg=200.0,
                max_mg_per_kg_day=5.0,
            ),
        ],
        do_not_apply_defaults=[
            "Do NOT apply steroid-combination creams (e.g., Betnovate, Panderm, Quadriderm, Fourderm)—steroids mask redness briefly and worsen fungal spread (Tinea Incognito).",
            "Do NOT share towels, bedsheets, undergarments, or gym gear.",
        ],
        lifestyle_plan=[
            "Keep skin folds completely dry; switch to loose, breathable cotton clothing.",
            "Wash clothes and towels separately in hot water and dry in direct sunlight.",
            "Screen household members and pets for similar itchy patches to prevent reinfection.",
        ],
        trigger_lrs={
            "t01_otc_steroid_cream": 2.4,
            "t01_rebound_flare": 3.0,
            "t05_heat_humidity_monsoon": 2.3,
            "t08_tight_synthetic_clothes": 2.2,
            "t09_gym_sweat_unwashed": 2.1,
            "t09_damp_clothes_wet_hair": 2.4,
            "t09_shared_towels_combs": 2.5,
            "t10_household_rash_itch": 2.4,
            "t10_pet_contact": 2.2,
        },
    ),
    "tinea_incognito": ConditionKnowledge(
        condition_id="tinea_incognito",
        display_name="Steroid-Modified Dermatophytosis (Tinea Incognito)",
        tier1_family=Tier1Family.FUNGAL_INFECTIONS,
        is_infectious=True,
        summary="Fungal skin infection altered and worsened by unprescribed topical corticosteroid creams.",
        otc_options=[
            TreatmentOption(
                option_id="tinea_incog_otc_clotrimazole",
                display_name="Plain Clotrimazole 1% Cream (Steroid-Free Bridge Care)",
                salts=[SaltEntity(salt="clotrimazole", strength="1%")],
                vehicle="CREAM",
                regulatory_tier=RegulatoryTier.TIER1_OTC,
                posology="Use strictly plain steroid-free antifungal cream while arranging dermatologist review.",
                min_age_years=2,
                pregnancy_safe=True,
                lactation_safe=True,
            ),
        ],
        rx_discussion_options=[
            TreatmentOption(
                option_id="tinea_incog_rx_itraconazole",
                display_name="Oral Itraconazole + Prescription Topical Azole",
                salts=[SaltEntity(salt="itraconazole", strength="100mg")],
                vehicle="CAPSULE",
                regulatory_tier=RegulatoryTier.TIER2_SCHEDULE_H,
                posology="Dermatologist evaluation strongly advised for systemic therapy and steroid-withdrawal management.",
                is_systemic=True,
                min_age_years=12,
                pregnancy_safe=False,
                lactation_safe=False,
                contraindicated_systemic=["liver_disease"],
                max_adult_daily_mg=200.0,
                max_mg_per_kg_day=5.0,
            ),
        ],
        do_not_apply_defaults=[
            "STOP all OTC steroid and triple-combination creams immediately; expect a temporary rebound flare and consult a dermatologist.",
        ],
        lifestyle_plan=[
            "Avoid occlusive clothing and sweat retention; launder clothes/towels in hot water.",
            "Bring photos of all previously applied cream tubes to your dermatologist appointment.",
        ],
        trigger_lrs={
            "t01_otc_steroid_cream": 4.2,
            "t01_rebound_flare": 4.5,
            "t05_heat_humidity_monsoon": 1.9,
            "t09_damp_clothes_wet_hair": 1.8,
            "t10_household_rash_itch": 1.9,
        },
    ),
    "cutaneous_candidiasis": ConditionKnowledge(
        condition_id="cutaneous_candidiasis",
        display_name="Cutaneous Candidiasis (Intertrigo)",
        tier1_family=Tier1Family.FUNGAL_INFECTIONS,
        is_infectious=True,
        summary="Moist, red skin-fold yeast infection often with satellite pustules.",
        otc_options=[
            TreatmentOption(
                option_id="candida_otc_clotrimazole",
                display_name="Clotrimazole 1% Cream + Absorptive Powder",
                salts=[SaltEntity(salt="clotrimazole", strength="1%")],
                vehicle="CREAM",
                regulatory_tier=RegulatoryTier.TIER1_OTC,
                posology="Apply thinly twice daily to clean, air-dried folds.",
                min_age_years=2,
                pregnancy_safe=True,
                lactation_safe=True,
            ),
        ],
        rx_discussion_options=[
            TreatmentOption(
                option_id="candida_rx_fluconazole",
                display_name="Oral Fluconazole 150mg Tablet",
                salts=[SaltEntity(salt="fluconazole", strength="150mg")],
                vehicle="TABLET",
                regulatory_tier=RegulatoryTier.TIER2_SCHEDULE_H,
                posology="Evaluated by physician if recurrent or associated with uncontrolled blood sugar.",
                is_systemic=True,
                min_age_years=12,
                pregnancy_safe=False,
                lactation_safe=True,
                contraindicated_systemic=["liver_disease"],
                max_adult_daily_mg=150.0,
                max_mg_per_kg_day=6.0,
            ),
        ],
        do_not_apply_defaults=[
            "Do NOT apply steroid creams or thick petroleum jelly in damp skin folds.",
        ],
        lifestyle_plan=[
            "Pat folds dry thoroughly after bathing and keep area ventilated.",
            "Check fasting blood sugar / HbA1c if fold yeast infections recur frequently.",
        ],
        trigger_lrs={
            "t05_heat_humidity_monsoon": 2.2,
            "t09_damp_clothes_wet_hair": 2.5,
            "t08_tight_synthetic_clothes": 2.0,
            "t02_recent_antibiotic_nsaid": 2.1,
            "t11_thyroid_insulin_shift": 2.0,
        },
    ),
    "bacterial_folliculitis_impetigo": ConditionKnowledge(
        condition_id="bacterial_folliculitis_impetigo",
        display_name="Bacterial Folliculitis / Superficial Impetigo",
        tier1_family=Tier1Family.BACTERIAL_VIRAL_PARASITIC,
        is_infectious=True,
        summary="Superficial bacterial infection of hair follicles or epidermis with pustules or honey-colored crusts.",
        otc_options=[
            TreatmentOption(
                option_id="bact_otc_povidone",
                display_name="Povidone-Iodine 5% / Gentle Antiseptic Wash",
                salts=[SaltEntity(salt="povidone iodine", strength="5%")],
                vehicle="LOTION",
                regulatory_tier=RegulatoryTier.TIER1_OTC,
                posology="Gently cleanse affected area once daily while awaiting clinical review.",
                min_age_years=6,
                pregnancy_safe=False,
                lactation_safe=False,
                contraindicated_systemic=["thyroid"],
            ),
        ],
        rx_discussion_options=[
            TreatmentOption(
                option_id="bact_rx_mupirocin",
                display_name="Mupirocin 2% Ointment",
                salts=[SaltEntity(salt="mupirocin", strength="2%")],
                vehicle="OINTMENT",
                regulatory_tier=RegulatoryTier.TIER2_SCHEDULE_H,
                posology="Prescription topical antibiotic commonly evaluated for localized impetigo/folliculitis.",
                min_age_years=2,
                pregnancy_safe=True,
                lactation_safe=True,
            ),
            TreatmentOption(
                option_id="bact_rx_fusidic",
                display_name="Fusidic Acid 2% Cream",
                salts=[SaltEntity(salt="fusidic acid", strength="2%")],
                vehicle="CREAM",
                regulatory_tier=RegulatoryTier.TIER2_SCHEDULE_H,
                posology="Prescription topical antibiotic option for localized superficial pyoderma.",
                min_age_years=2,
                pregnancy_safe=True,
                lactation_safe=True,
            ),
        ],
        do_not_apply_defaults=[
            "Do NOT squeeze pustules or apply topical steroid creams over bacterial infection.",
            "Avoid shaving or waxing directly over infected follicles.",
        ],
        lifestyle_plan=[
            "Use a clean, dedicated towel daily and disinfect razors/trimmers after each use.",
            "Seek immediate medical care if redness spreads rapidly, becomes hot/tender, or fever develops.",
        ],
        trigger_lrs={
            "t08_shaving_waxing_threading": 2.8,
            "t08_cut_scratch_koebner": 2.4,
            "t09_gym_sweat_unwashed": 2.0,
            "t09_shared_towels_combs": 2.1,
            "t04_picking_scratching": 2.0,
            "t01_otc_steroid_cream": 1.7,
        },
    ),
    "scabies": ConditionKnowledge(
        condition_id="scabies",
        display_name="Scabies (Sarcoptes Infestation)",
        tier1_family=Tier1Family.BACTERIAL_VIRAL_PARASITIC,
        is_infectious=True,
        summary="Intensely pruritic parasitic infestation worse at night, frequently affecting finger webs, wrists, waist, and household contacts.",
        otc_options=[
            TreatmentOption(
                option_id="scabies_otc_calamine",
                display_name="Calamine Lotion (Temporary Itch Relief Only)",
                salts=[SaltEntity(salt="calamine", strength="8%")],
                vehicle="LOTION",
                regulatory_tier=RegulatoryTier.TIER1_OTC,
                posology="Soothing relief only; definitive prescription scabicide is required for patient and close contacts.",
                min_age_years=2,
                pregnancy_safe=True,
                lactation_safe=True,
            ),
        ],
        rx_discussion_options=[
            TreatmentOption(
                option_id="scabies_rx_permethrin",
                display_name="Permethrin 5% Cream / Lotion",
                salts=[SaltEntity(salt="permethrin", strength="5%")],
                vehicle="CREAM",
                regulatory_tier=RegulatoryTier.TIER2_SCHEDULE_H,
                posology="Doctor-directed neck-to-toe overnight application with repeat dose after 7 days; simultaneous treatment of close contacts.",
                min_age_years=2,
                pregnancy_safe=True,
                lactation_safe=True,
            ),
            TreatmentOption(
                option_id="scabies_rx_ivermectin",
                display_name="Oral Ivermectin (Weight-Based Dosing)",
                salts=[SaltEntity(salt="ivermectin", strength="6mg")],
                vehicle="TABLET",
                regulatory_tier=RegulatoryTier.TIER2_SCHEDULE_H,
                posology="Weight-calculated prescription option (approx. 200 mcg/kg) evaluated by doctor.",
                is_systemic=True,
                min_age_years=12,
                pregnancy_safe=False,
                lactation_safe=False,
                max_adult_daily_mg=12.0,
                max_mg_per_kg_day=0.2,
            ),
        ],
        do_not_apply_defaults=[
            "Do NOT rely on steroid creams—steroids temporarily dull itching while mites multiply.",
        ],
        lifestyle_plan=[
            "Wash all worn clothing, bedsheets, and towels in hot water (>50°C) and seal non-washables for 72 hours.",
            "All close household/hostel contacts should be evaluated simultaneously.",
        ],
        trigger_lrs={
            "t10_household_rash_itch": 4.2,
            "t10_travel_hotel_bedding": 2.6,
            "t09_shared_towels_combs": 2.4,
            "t04_picking_scratching": 1.9,
        },
    ),
    "herpes_viral_eruption": ConditionKnowledge(
        condition_id="herpes_viral_eruption",
        display_name="Viral Vesicular Eruption (Herpes Simplex / Zoster)",
        tier1_family=Tier1Family.BACTERIAL_VIRAL_PARASITIC,
        is_infectious=True,
        summary="Grouped painful or tingling blisters on a red base, often unilateral in zoster.",
        otc_options=[
            TreatmentOption(
                option_id="viral_otc_cool_compress",
                display_name="Cool Saline Compress (Supportive Comfort Only)",
                salts=[SaltEntity(salt="sodium chloride", strength="0.9%")],
                vehicle="LOTION",
                regulatory_tier=RegulatoryTier.TIER1_OTC,
                posology="Keep lesions clean and dry; prompt doctor visit within 72 hours is critical for prescription antivirals.",
                min_age_years=2,
                pregnancy_safe=True,
                lactation_safe=True,
            ),
        ],
        rx_discussion_options=[
            TreatmentOption(
                option_id="viral_rx_acyclovir",
                display_name="Oral Acyclovir / Valacyclovir (Prescription Antiviral)",
                salts=[SaltEntity(salt="acyclovir", strength="400mg")],
                vehicle="TABLET",
                regulatory_tier=RegulatoryTier.TIER2_SCHEDULE_H,
                posology="Doctor-prescribed early antiviral therapy (ideally started within 72 hours of rash onset).",
                is_systemic=True,
                min_age_years=12,
                pregnancy_safe=True,
                lactation_safe=True,
                contraindicated_systemic=["kidney_disease"],
                max_adult_daily_mg=4000.0,
                max_mg_per_kg_day=80.0,
            ),
        ],
        do_not_apply_defaults=[
            "Do NOT apply topical steroid creams on viral blisters (can trigger severe viral spread).",
            "Seek urgent medical care immediately if blisters appear near the eye or tip of the nose.",
        ],
        lifestyle_plan=[
            "Keep blisters covered and avoid skin-to-skin contact with infants, pregnant individuals, or immunocompromised people.",
        ],
        trigger_lrs={
            "t12_recent_viral_fever": 2.6,
            "t04_high_stress": 2.1,
            "t04_poor_irregular_sleep": 1.12,
            "t05_intense_uv_sun": 1.7,
        },
    ),
    "atopic_dermatitis_eczema": ConditionKnowledge(
        condition_id="atopic_dermatitis_eczema",
        display_name="Atopic Dermatitis / Eczema",
        tier1_family=Tier1Family.ECZEMATOUS_ALLERGIC,
        summary="Pruritic, dry, inflamed skin barrier disorder with flexural or patchy erythema and scaling.",
        otc_options=[
            TreatmentOption(
                option_id="eczema_otc_ceramide",
                display_name="Fragrance-Free Ceramide / Colloidal Oatmeal Emollient Cream",
                salts=[SaltEntity(salt="white soft paraffin", strength="10%")],
                vehicle="CREAM",
                regulatory_tier=RegulatoryTier.TIER1_OTC,
                posology="Apply generously 2-3 times daily, especially within 3 minutes after a short lukewarm shower.",
                min_age_years=1,
                pregnancy_safe=True,
                lactation_safe=True,
            ),
            TreatmentOption(
                option_id="eczema_otc_hydrocortisone",
                display_name="Hydrocortisone 1% Mild Cream (Short-Term Non-Infected Flare)",
                salts=[SaltEntity(salt="hydrocortisone", strength="1%")],
                vehicle="CREAM",
                regulatory_tier=RegulatoryTier.TIER1_OTC,
                posology="Thin layer once or twice daily for up to 5-7 days on non-infected body patches only.",
                steroid_potency=SteroidPotency.MILD,
                min_age_years=12,
                pregnancy_safe=True,
                lactation_safe=True,
            ),
        ],
        rx_discussion_options=[
            TreatmentOption(
                option_id="eczema_rx_desonide",
                display_name="Desonide 0.05% Lotion / Cream",
                salts=[SaltEntity(salt="desonide", strength="0.05%")],
                vehicle="LOTION",
                regulatory_tier=RegulatoryTier.TIER2_SCHEDULE_H,
                posology="Low-potency prescription steroid evaluated by dermatologists for sensitive-site flares.",
                steroid_potency=SteroidPotency.MILD,
                min_age_years=2,
                pregnancy_safe=True,
                lactation_safe=True,
            ),
            TreatmentOption(
                option_id="eczema_rx_mometasone",
                display_name="Mometasone Furoate 0.1% Cream",
                salts=[SaltEntity(salt="mometasone", strength="0.1%")],
                vehicle="CREAM",
                regulatory_tier=RegulatoryTier.TIER2_SCHEDULE_H,
                posology="Potent prescription steroid discussed for thick trunk/limb eczema plaques (avoid face/groin).",
                steroid_potency=SteroidPotency.POTENT,
                min_age_years=12,
                pregnancy_safe=False,
                lactation_safe=True,
            ),
        ],
        do_not_apply_defaults=[
            "Avoid scented lotions, harsh alkaline soaps, and scalding hot showers.",
        ],
        lifestyle_plan=[
            "Use lukewarm water (<5-7 min showers) and a soap-free syndet cleanser.",
            "Seal moisture immediately after bathing with a thick fragrance-free emollient.",
            "Wear soft cotton fabrics and trim nails to minimize nighttime scratch damage.",
        ],
        trigger_lrs={
            "t05_cold_dry_windburn": 2.6,
            "t06_hard_borewell_water": 1.8,
            "t09_overwashing_barrier_strip": 2.5,
            "t07_new_skincare_makeup": 1.9,
            "t04_high_stress": 1.8,
            "t04_picking_scratching": 2.0,
            "t04_poor_irregular_sleep": 1.08,
            "t14_wet_work_detergents": 2.1,
        },
    ),
    "contact_dermatitis": ConditionKnowledge(
        condition_id="contact_dermatitis",
        display_name="Allergic / Irritant Contact Dermatitis",
        tier1_family=Tier1Family.ECZEMATOUS_ALLERGIC,
        summary="Localized itchy, red, or scaly reaction triggered by direct contact with an allergen or irritant.",
        otc_options=[
            TreatmentOption(
                option_id="cd_otc_barrier_cream",
                display_name="Fragrance-Free Barrier Repair Emollient",
                salts=[SaltEntity(salt="white soft paraffin", strength="10%")],
                vehicle="CREAM",
                regulatory_tier=RegulatoryTier.TIER1_OTC,
                posology="Apply 2-3 times daily after eliminating the suspected trigger product/material.",
                min_age_years=1,
                pregnancy_safe=True,
                lactation_safe=True,
            ),
            TreatmentOption(
                option_id="cd_otc_cetirizine",
                display_name="Cetirizine 10mg Tablet (Oral Itch Relief)",
                salts=[SaltEntity(salt="cetirizine", strength="10mg")],
                vehicle="TABLET",
                regulatory_tier=RegulatoryTier.TIER1_OTC,
                posology="1 tablet at bedtime for itch relief (may cause mild drowsiness).",
                is_systemic=True,
                min_age_years=12,
                pregnancy_safe=True,
                lactation_safe=True,
                contraindicated_systemic=["kidney_disease"],
                max_adult_daily_mg=10.0,
                max_mg_per_kg_day=0.25,
            ),
        ],
        rx_discussion_options=[
            TreatmentOption(
                option_id="cd_rx_mometasone",
                display_name="Mometasone 0.1% Cream (Short Pulse on Trunk/Limbs)",
                salts=[SaltEntity(salt="mometasone", strength="0.1%")],
                vehicle="CREAM",
                regulatory_tier=RegulatoryTier.TIER2_SCHEDULE_H,
                posology="Dermatologist-directed short course plus patch testing if allergen is unclear.",
                steroid_potency=SteroidPotency.POTENT,
                min_age_years=12,
                pregnancy_safe=False,
                lactation_safe=True,
            ),
        ],
        do_not_apply_defaults=[
            "Stop all newly introduced cosmetics, dyes, fragrances, or suspect adhesives immediately.",
        ],
        lifestyle_plan=[
            "Switch to fragrance-free detergent and wear cotton-lined gloves during wet chores/cleaning.",
            "Keep a simple 3-step routine (mild cleanser, plain moisturizer, mineral SPF) until skin calms.",
        ],
        trigger_lrs={
            "t07_new_skincare_makeup": 2.9,
            "t07_hair_dye_ppd": 3.5,
            "t07_perfume_bindi_detergent": 3.2,
            "t01_home_remedy_irritant": 2.8,
            "t14_wet_work_detergents": 3.0,
            "t14_nickel_metal_latex": 3.4,
            "t14_cement_solvents_plants": 3.2,
            "t08_helmet_cap_strap": 1.8,
        },
    ),
    "psoriasis_vulgaris": ConditionKnowledge(
        condition_id="psoriasis_vulgaris",
        display_name="Plaque Psoriasis",
        tier1_family=Tier1Family.PAPULOSQUAMOUS,
        summary="Chronic immune-mediated well-demarcated erythematous plaques with silvery-white scales.",
        otc_options=[
            TreatmentOption(
                option_id="psor_otc_coal_tar_shampoo",
                display_name="Coal Tar 1% + Salicylic Acid 3% Medicated Scalp Shampoo",
                regulatory_tier=RegulatoryTier.TIER1_OTC,
                vehicle="SHAMPOO",
                salts=[SaltEntity(salt="coal tar", strength="1%"), SaltEntity(salt="salicylic acid", strength="3%")],
                posology="Massage gently into wet scalp, leave on for 3-5 minutes, and rinse 2-3 times weekly.",
                is_photosensitizing=True,
                pregnancy_contraindicated=True,
            ),
            TreatmentOption(
                option_id="psor_otc_coal_tar_soap",
                display_name="Coal Tar 1% + Salicylic Acid 3% Medicated Soap / Cleanser",
                salts=[
                    SaltEntity(salt="coal tar", strength="1%"),
                    SaltEntity(salt="salicylic acid", strength="3%"),
                ],
                vehicle="SOAP",
                regulatory_tier=RegulatoryTier.TIER1_OTC,
                posology="Lather gently on thick plaques during bath for 2 minutes, rinse, and moisturize immediately.",
                is_photosensitizing=True,
                is_barrier_harsh=True,
                min_age_years=12,
                pregnancy_safe=False,
                lactation_safe=True,
            ),
            TreatmentOption(
                option_id="psor_otc_emollient",
                display_name="Thick Fragrance-Free Emollient Ointment",
                salts=[SaltEntity(salt="white soft paraffin", strength="10%")],
                vehicle="OINTMENT",
                regulatory_tier=RegulatoryTier.TIER1_OTC,
                posology="Apply generously twice daily to soften scales and reduce fissuring.",
                min_age_years=1,
                pregnancy_safe=True,
                lactation_safe=True,
            ),
        ],
        rx_discussion_options=[
            TreatmentOption(
                option_id="psor_rx_clobetasol_scalp_lotion",
                display_name="Clobetasol 0.05% / Calcipotriol Scalp Solution or Lotion",
                regulatory_tier=RegulatoryTier.TIER2_SCHEDULE_H,
                vehicle="LOTION",
                salts=[SaltEntity(salt="clobetasol", strength="0.05%")],
                posology="Short-pulse dermatologist-guided scalp lotion for thick psoriatic scalp plaques.",
                steroid_potency=SteroidPotency.SUPER_POTENT,
                pregnancy_contraindicated=True,
            ),
            TreatmentOption(
                option_id="psor_rx_calcipotriol_clobetasol",
                display_name="Calcipotriol 0.005% + Clobetasol 0.05% Ointment",
                salts=[
                    SaltEntity(salt="calcipotriol", strength="0.005%"),
                    SaltEntity(salt="clobetasol", strength="0.05%"),
                ],
                vehicle="OINTMENT",
                regulatory_tier=RegulatoryTier.TIER2_SCHEDULE_H,
                posology="Super-potent prescription combo evaluated by dermatologists strictly for thick body plaques (never face/groin).",
                steroid_potency=SteroidPotency.SUPER_POTENT,
                min_age_years=18,
                pregnancy_safe=False,
                lactation_safe=False,
            ),
        ],
        do_not_apply_defaults=[
            "Do NOT pick or forcefully peel silvery scales (triggers Koebner phenomenon and new plaques).",
            "Do NOT start oral steroids for psoriasis (abrupt withdrawal can trigger severe pustular flare).",
        ],
        lifestyle_plan=[
            "Manage stress and sleep consistency; limit alcohol and smoking, which worsen flares.",
            "Report any morning joint stiffness or nail pitting to a dermatologist (psoriatic arthritis screening).",
        ],
        trigger_lrs={
            "t04_high_stress": 2.3,
            "t04_poor_irregular_sleep": 1.08,
            "t05_cold_dry_windburn": 2.2,
            "t08_cut_scratch_koebner": 2.8,
            "t12_sore_throat_strep": 2.9,
            "t03_low_hydration_alcohol": 1.8,
        },
    ),
    "melasma_pih": ConditionKnowledge(
        condition_id="melasma_pih",
        display_name="Melasma / Post-Inflammatory Hyperpigmentation (PIH)",
        tier1_family=Tier1Family.PIGMENTARY,
        summary="Symmetric brown-grey facial patches (melasma) or dark marks following skin inflammation (PIH).",
        otc_options=[
            TreatmentOption(
                option_id="pigment_otc_spf",
                display_name="Broad-Spectrum Tinted Sunscreen SPF 50+ PA++++",
                salts=[SaltEntity(salt="zinc oxide", strength="10%")],
                vehicle="GEL",
                regulatory_tier=RegulatoryTier.TIER1_OTC,
                posology="Apply 2 finger-lengths every morning and reapply every 3 hours outdoors (iron-oxide tint protects against visible light).",
                min_age_years=6,
                pregnancy_safe=True,
                lactation_safe=True,
            ),
            TreatmentOption(
                option_id="pigment_otc_niacinamide",
                display_name="Niacinamide 4% / Azelaic Acid 10% Gel",
                salts=[SaltEntity(salt="nicotinamide", strength="4%")],
                vehicle="GEL",
                regulatory_tier=RegulatoryTier.TIER1_OTC,
                posology="Apply thinly once or twice daily under sunscreen.",
                min_age_years=12,
                pregnancy_safe=True,
                lactation_safe=True,
            ),
        ],
        rx_discussion_options=[
            TreatmentOption(
                option_id="pigment_rx_azelaic20",
                display_name="Azelaic Acid 20% Cream",
                salts=[SaltEntity(salt="azelaic acid", strength="20%")],
                vehicle="CREAM",
                regulatory_tier=RegulatoryTier.TIER2_SCHEDULE_H,
                posology="Prescription-strength pigment regulator evaluated by dermatologists (pregnancy-compatible option).",
                is_barrier_harsh=True,
                min_age_years=12,
                pregnancy_safe=True,
                lactation_safe=True,
            ),
            TreatmentOption(
                option_id="pigment_rx_hydroquinone",
                display_name="Hydroquinone 2% / Triple-Combination Depigmenting Cream",
                salts=[SaltEntity(salt="hydroquinone", strength="2%")],
                vehicle="CREAM",
                regulatory_tier=RegulatoryTier.TIER2_SCHEDULE_H,
                posology="Strictly time-limited prescription therapy under dermatologist monitoring (risk of ochronosis if misused).",
                is_photosensitizing=True,
                is_barrier_harsh=True,
                min_age_years=18,
                pregnancy_safe=False,
                lactation_safe=False,
            ),
        ],
        do_not_apply_defaults=[
            "Do NOT use unprescribed triple-combo steroid/hydroquinone creams long-term (causes skin thinning, rebound darkening, and ochronosis).",
        ],
        lifestyle_plan=[
            "Daily strict photoprotection (tinted SPF 50+, hat/shade) is mandatory—UV and visible light immediately darken melasma.",
            "Avoid harsh scrubs or lemon/DIY bleaching remedies that inflame Fitzpatrick IV-VI skin.",
        ],
        trigger_lrs={
            "t05_intense_uv_sun": 3.2,
            "t11_pregnancy_postpartum": 3.0,
            "t11_thyroid_insulin_shift": 2.0,
            "t01_fairness_cream": 2.4,
            "t01_home_remedy_irritant": 1.9,
            "t13_vitamin_mineral_def": 1.6,
        },
    ),
    "seborrheic_dermatitis_scalp": ConditionKnowledge(
        condition_id="seborrheic_dermatitis_scalp",
        display_name="Seborrheic Dermatitis / Scalp Dandruff",
        tier1_family=Tier1Family.SCALP_FOLLICULAR,
        summary="Flaky, itchy scalp (and occasionally eyebrows/nasolabial folds) driven by Malassezia yeast and sebum.",
        otc_options=[
            TreatmentOption(
                option_id="scalp_otc_keto2",
                display_name="Ketoconazole 2% Anti-Dandruff Shampoo",
                salts=[SaltEntity(salt="ketoconazole", strength="2%")],
                vehicle="SHAMPOO",
                regulatory_tier=RegulatoryTier.TIER1_OTC,
                posology="Apply on wet scalp, lather and leave on for 5 minutes before rinsing; use 2-3 times per week for 4 weeks.",
                min_age_years=12,
                pregnancy_safe=True,
                lactation_safe=True,
            ),
            TreatmentOption(
                option_id="scalp_otc_selenium",
                display_name="Selenium Sulfide 2.5% + Clotrimazole 1% Suspension/Shampoo",
                salts=[
                    SaltEntity(salt="selenium", strength="2.5%"),
                    SaltEntity(salt="clotrimazole", strength="1%"),
                ],
                vehicle="SHAMPOO",
                regulatory_tier=RegulatoryTier.TIER1_OTC,
                posology="Massage onto scalp, wait 3-5 minutes, rinse thoroughly twice weekly.",
                min_age_years=12,
                pregnancy_safe=False,
                lactation_safe=True,
            ),
        ],
        rx_discussion_options=[
            TreatmentOption(
                option_id="scalp_rx_clobetasol_lotion",
                display_name="Short-Pulse Topical Steroid Scalp Lotion",
                salts=[SaltEntity(salt="clobetasol", strength="0.05%")],
                vehicle="LOTION",
                regulatory_tier=RegulatoryTier.TIER2_SCHEDULE_H,
                posology="Evaluated by dermatologist briefly for severe inflamed/crusting scalp flares.",
                steroid_potency=SteroidPotency.SUPER_POTENT,
                min_age_years=18,
                pregnancy_safe=False,
                lactation_safe=True,
            ),
        ],
        do_not_apply_defaults=[
            "Avoid leaving heavy coconut/mustard hair oils overnight on a flaky scalp—oils feed Malassezia yeast and worsen dandruff.",
        ],
        lifestyle_plan=[
            "Rinse sweat promptly after workouts or helmet use; avoid sleeping with wet hair.",
            "If using hard/borewell water, consider a shower filter or final filtered-water rinse.",
        ],
        trigger_lrs={
            "t07_heavy_hair_oil": 2.9,
            "t06_hard_borewell_water": 2.1,
            "t09_damp_clothes_wet_hair": 2.4,
            "t08_helmet_cap_strap": 2.2,
            "t04_high_stress": 1.9,
            "t04_poor_irregular_sleep": 1.10,
            "t05_cold_dry_windburn": 1.7,
        },
    ),
    "tinea_capitis": ConditionKnowledge(
        condition_id="tinea_capitis",
        display_name="Tinea Capitis (Scalp Ringworm)",
        tier1_family=Tier1Family.SCALP_FOLLICULAR,
        is_infectious=True,
        summary="Dermatophyte infection of scalp hair shafts causing patchy hair loss, broken black-dot hairs, and scaling.",
        otc_options=[
            TreatmentOption(
                option_id="tcap_otc_keto_shampoo",
                display_name="Ketoconazole 2% Shampoo (Adjunct Spore Control Only)",
                salts=[SaltEntity(salt="ketoconazole", strength="2%")],
                vehicle="SHAMPOO",
                regulatory_tier=RegulatoryTier.TIER1_OTC,
                posology="Reduces surface shedding twice weekly—oral prescription antifungal from a dermatologist is mandatory to cure hair-shaft infection.",
                min_age_years=5,
                pregnancy_safe=True,
                lactation_safe=True,
            ),
        ],
        rx_discussion_options=[
            TreatmentOption(
                option_id="tcap_rx_terbinafine_oral",
                display_name="Oral Terbinafine 250mg Tablet (Weight-Adjusted Course)",
                salts=[SaltEntity(salt="terbinafine", strength="250mg")],
                vehicle="TABLET",
                regulatory_tier=RegulatoryTier.TIER2_SCHEDULE_H,
                posology="Dermatologist-prescribed systemic course (4-6 weeks) required for deep hair-follicle clearance.",
                is_systemic=True,
                min_age_years=12,
                pregnancy_safe=False,
                lactation_safe=False,
                contraindicated_systemic=["liver_disease"],
                max_adult_daily_mg=250.0,
                max_mg_per_kg_day=5.0,
            ),
        ],
        do_not_apply_defaults=[
            "Do NOT apply topical steroids or oil on patchy scalp hair loss.",
            "Do NOT share combs, caps, helmets, or pillowcases.",
        ],
        lifestyle_plan=[
            "Disinfect combs/brushes and wash pillowcases frequently; inspect household pets for ringworm patches.",
        ],
        trigger_lrs={
            "t09_shared_towels_combs": 3.0,
            "t10_pet_contact": 2.8,
            "t10_household_rash_itch": 2.4,
            "t08_helmet_cap_strap": 1.8,
        },
    ),
    "urticaria": ConditionKnowledge(
        condition_id="urticaria",
        display_name="Urticaria (Hives / Wheals)",
        tier1_family=Tier1Family.URTICARIAL_REACTIVE,
        summary="Transient, migratory itchy raised wheals resolving within 24 hours without scarring.",
        otc_options=[
            TreatmentOption(
                option_id="urt_otc_cetirizine",
                display_name="Cetirizine 10mg Tablet (Non-Sedating Antihistamine)",
                salts=[SaltEntity(salt="cetirizine", strength="10mg")],
                vehicle="TABLET",
                regulatory_tier=RegulatoryTier.TIER1_OTC,
                posology="1 tablet once daily at bedtime for acute uncomplicated hives (seek ER care immediately for lip/throat swelling or breathing difficulty).",
                is_systemic=True,
                min_age_years=12,
                pregnancy_safe=True,
                lactation_safe=True,
                contraindicated_systemic=["kidney_disease"],
                max_adult_daily_mg=10.0,
                max_mg_per_kg_day=0.25,
            ),
            TreatmentOption(
                option_id="urt_otc_calamine",
                display_name="Calamine / Cooling Menthol Lotion",
                salts=[SaltEntity(salt="calamine", strength="8%")],
                vehicle="LOTION",
                regulatory_tier=RegulatoryTier.TIER1_OTC,
                posology="Apply gently to cool itchy wheals as needed.",
                min_age_years=2,
                pregnancy_safe=True,
                lactation_safe=True,
            ),
        ],
        rx_discussion_options=[
            TreatmentOption(
                option_id="urt_rx_bilastine_fexo",
                display_name="Second-Generation Antihistamine Up-Dosing (Doctor Review)",
                salts=[SaltEntity(salt="levocetirizine", strength="5mg")],
                vehicle="TABLET",
                regulatory_tier=RegulatoryTier.TIER2_SCHEDULE_H,
                posology="Evaluated by a dermatologist/allergist if hives persist >6 weeks (chronic spontaneous urticaria).",
                is_systemic=True,
                min_age_years=12,
                pregnancy_safe=True,
                lactation_safe=True,
                contraindicated_systemic=["kidney_disease"],
                max_adult_daily_mg=10.0,
                max_mg_per_kg_day=0.2,
            ),
        ],
        do_not_apply_defaults=[
            "Avoid NSAIDs (ibuprofen/aspirin) if they triggered or worsened the hives.",
            "Seek EMERGENCY care immediately if lip/eyelid/throat swelling or breathing trouble occurs.",
        ],
        lifestyle_plan=[
            "Track a 7-day food, medication, and temperature diary to isolate triggers.",
            "Avoid hot showers and tight waistbands/straps during flares.",
        ],
        trigger_lrs={
            "t02_recent_antibiotic_nsaid": 3.0,
            "t03_food_allergen_trigger": 3.2,
            "t12_recent_viral_fever": 2.4,
            "t10_insect_bites": 2.1,
            "t05_cold_dry_windburn": 2.0,
            "t04_high_stress": 1.8,
        },
    ),
    "benign_growth_nevus": ConditionKnowledge(
        condition_id="benign_growth_nevus",
        display_name="Benign Skin Growth (Seborrheic Keratosis / Skin Tag / Benign Nevus)",
        tier1_family=Tier1Family.BENIGN_GROWTHS,
        summary="Stable, non-cancerous cutaneous growth requiring routine monitoring for any ABCDE changes.",
        otc_options=[],
        rx_discussion_options=[
            TreatmentOption(
                option_id="benign_rx_dermoscopy",
                display_name="In-Clinic Dermoscopy Evaluation (No Self-Treatment)",
                salts=[],
                vehicle="PROCEDURE",
                regulatory_tier=RegulatoryTier.TIER2_SCHEDULE_H,
                posology="Have a dermatologist confirm benign status via dermoscopy before any cosmetic removal.",
                min_age_years=1,
                pregnancy_safe=True,
                lactation_safe=True,
            ),
        ],
        do_not_apply_defaults=[
            "Do NOT apply acid wart removers, caustic pastes, or sharp tools to moles or skin growths.",
        ],
        lifestyle_plan=[
            "Monitor monthly using the ABCDE rule (Asymmetry, Border, Color, Diameter >6mm, Evolving) and use daily SPF.",
        ],
        trigger_lrs={
            "t05_intense_uv_sun": 1.6,
            "t08_tight_synthetic_clothes": 1.4,
        },
    ),
    "suspected_neoplastic_lesion": ConditionKnowledge(
        condition_id="suspected_neoplastic_lesion",
        display_name="Suspected Neoplastic / High-Risk Pigmented or Ulcerated Lesion",
        tier1_family=Tier1Family.NEOPLASTIC_RED_FLAG,
        is_neoplastic_or_red_flag=True,
        summary="High-priority lesion requiring direct in-person dermoscopy and potential biopsy.",
        otc_options=[],
        rx_discussion_options=[],
        do_not_apply_defaults=[
            "Do NOT apply any OTC cream, steroid, home remedy, or caustic agent to this lesion.",
        ],
        lifestyle_plan=[
            "Schedule an urgent in-person dermatologist consultation for dermoscopic examination.",
        ],
        trigger_lrs={
            "t05_intense_uv_sun": 2.2,
        },
    ),
}

# Extensible alias map for Phase 2 merged dataset labels (DermaCon-IN, SCIN, Fitzpatrick17k, DermNet)
TIER2_ALIAS_MAP: dict[str, str] = {
    "acne": "acne_vulgaris",
    "acne_vulgaris": "acne_vulgaris",
    "comedonal_acne": "acne_vulgaris",
    "nodulocystic_acne": "acne_vulgaris",
    "tinea": "tinea_corporis_cruris",
    "tinea_corporis": "tinea_corporis_cruris",
    "tinea_cruris": "tinea_corporis_cruris",
    "ringworm": "tinea_corporis_cruris",
    "dermatophytosis": "tinea_corporis_cruris",
    "tinea_incognito": "tinea_incognito",
    "candidiasis": "cutaneous_candidiasis",
    "intertrigo": "cutaneous_candidiasis",
    "folliculitis": "bacterial_folliculitis_impetigo",
    "impetigo": "bacterial_folliculitis_impetigo",
    "pyoderma": "bacterial_folliculitis_impetigo",
    "scabies": "scabies",
    "herpes_simplex": "herpes_viral_eruption",
    "herpes_zoster": "herpes_viral_eruption",
    "eczema": "atopic_dermatitis_eczema",
    "atopic_dermatitis": "atopic_dermatitis_eczema",
    "contact_dermatitis": "contact_dermatitis",
    "allergic_contact_dermatitis": "contact_dermatitis",
    "psoriasis": "psoriasis_vulgaris",
    "plaque_psoriasis": "psoriasis_vulgaris",
    "melasma": "melasma_pih",
    "pih": "melasma_pih",
    "hyperpigmentation": "melasma_pih",
    "seborrheic_dermatitis": "seborrheic_dermatitis_scalp",
    "dandruff": "seborrheic_dermatitis_scalp",
    "tinea_capitis": "tinea_capitis",
    "urticaria": "urticaria",
    "hives": "urticaria",
    "melanocytic_nevus": "benign_growth_nevus",
    "seborrheic_keratosis": "benign_growth_nevus",
    "melanoma": "suspected_neoplastic_lesion",
    "basal_cell_carcinoma": "suspected_neoplastic_lesion",
    "squamous_cell_carcinoma": "suspected_neoplastic_lesion",
}


def resolve_condition(raw_id: str) -> ConditionKnowledge | None:
    key = raw_id.strip().lower().replace(" ", "_").replace("-", "_")
    canonical = TIER2_ALIAS_MAP.get(key, key)
    return CONDITION_GRAPH.get(canonical)


# ==============================================================================
# 3. PILLAR 3 — DYNAMIC CARD 2-4 TRIGGER BUCKET SELECTOR
# ==============================================================================
def select_dynamic_trigger_buckets(
    stage1_output: ModelInferenceOutput,
    profile: PatientProfile,
    top_k_conditions: int = 3,
    n_cards: int = 3,
) -> list[TriggerBucket]:
    """
    Deterministically select the top 3-4 highest-value trigger buckets for Cards 2-4
    (Bucket 1 is always displayed on Card 1 alongside Red-Flag Triage).
    """
    scores: dict[int, float] = defaultdict(float)

    sorted_conds = sorted(
        stage1_output.condition_probs.items(), key=lambda kv: kv[1], reverse=True
    )[:top_k_conditions]

    for cond_id, prob in sorted_conds:
        ck = resolve_condition(cond_id)
        if not ck:
            continue
        priority_list = FAMILY_PRIORITY_BUCKETS.get(ck.tier1_family, [])
        for rank, bid in enumerate(priority_list):
            if bid == 1:
                continue
            scores[bid] += prob * (1.0 / (rank + 1.0))

    if profile.sleep_hours < 6.0 or profile.sleep_pattern != "regular" or profile.stress_level == "high":
        scores[4] += 0.25
    if any(t in {"humid", "cold", "high_uv"} for t in profile.climate_tags):
        scores[5] += 0.20
    if "hard_water" in profile.climate_tags:
        scores[6] += 0.25
    if profile.ongoing_oral_meds:
        scores[2] += 0.20
    if any(c in {"pcos", "thyroid"} for c in profile.systemic_conditions) or (14 <= profile.age <= 19):
        scores[11] += 0.25
    if any(c in {"anemia", "vitamin_deficiency"} for c in profile.systemic_conditions):
        scores[13] += 0.20

    ranked_ids = [bid for bid, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]
    for fallback_bid in (4, 7, 5, 9):
        if fallback_bid not in ranked_ids:
            ranked_ids.append(fallback_bid)

    return [BUCKET_BY_ID[bid] for bid in ranked_ids[:n_cards]]


# ==============================================================================
# 4. PILLAR 4 — STAGE 2 TRANSPARENT BAYESIAN CLINICAL MODIFIER
# ==============================================================================
MAX_BUCKET_LOG_LR = math.log(3.5)  # Per-bucket dampening cap prevents double-counting



def _symptom_compatibility_lrs(case: Any | None) -> dict[str, tuple[float, str]]:
    """Compute bounded symptom & anatomical-site likelihood ratios for Stage-2 refinement (_REDTEAM_HARDENING_V1)."""
    lrs: dict[str, tuple[float, str]] = {}
    if case is None:
        return lrs

    itch = int(getattr(case, "itch_score", 0) or 0)
    pain = int(getattr(case, "pain_burning_score", 0) or 0)
    scaling = bool(getattr(case, "has_scaling", False))
    barrier = bool(getattr(case, "raw_cracked_barrier", False))
    duration = int(getattr(case, "duration_days", 0) or 0)
    site_raw = getattr(case, "anatomical_site", "")
    site = site_raw.value if hasattr(site_raw, "value") else str(site_raw or "").lower()
    notes_txt = " ".join([
        str(getattr(case, "patient_notes", "") or ""),
        " ".join(str(t) for t in (getattr(case, "selected_triggers", None) or [])),
    ]).lower()

    # Scalp-specific condition evaluated on non-scalp / non-beard anatomical sites
    non_hair_sites = {"trunk", "arms_legs", "palms_soles", "groin_folds", "face", "eyelids", "lips_mucosa"}
    if site in non_hair_sites:
        lrs["seborrheic_dermatitis_scalp"] = (
            0.05,
            f"Site mismatch: scalp seborrheic dermatitis less likely on {site}",
        )

    # Completely asymptomatic profile (no itch, no pain, no scaling, no barrier break)
    if itch == 0 and pain == 0 and not scaling and not barrier:
        for cid, reason in [
            ("scabies", "Symptom discordance: scabies typically presents with prominent pruritus (itch=0)"),
            ("urticaria", "Symptom discordance: urticarial wheals typically cause itch/welts (itch=0)"),
            ("atopic_dermatitis_eczema", "Symptom discordance: no itch, pain, or scaling reported"),
        ]:
            lrs[cid] = (0.45, reason)

    # Acute painful eruption (<= 7 days) supports viral vesicular / acute bacterial differential
    if duration <= 7 and pain >= 4:
        lrs["herpes_viral_eruption"] = (1.35, "Acute painful eruption (<=7d)")

    # Free-text or trigger mention of prior steroid cream misuse
    if any(tok in notes_txt for tok in ("steroid", "betamethasone", "clobetasol", "mometasone")):
        lrs["tinea_incognito"] = (1.30, "Prior topical steroid exposure noted")

    return lrs


def apply_bayesian_modifier(
    stage1_output: ModelInferenceOutput | dict[str, float],
    active_triggers: list[str] | PatientProfile = (),
    top_n: int | Any = 5,
    case: Any | None = None,
) -> list[ConditionPrediction] | dict[str, float]:
    """
    Stage 2 Bayesian Clinical Modifier:
        P_post(d_i | E) proportional to P_NN(d_i) * prod_j LR(e_j | d_i)

    Executed in log-space with per-bucket dampening caps so checking multiple items
    within the same trigger bucket never over-inflates the posterior.
    Supports both `(ModelInferenceOutput, list[str], top_n, case)` -> `list[ConditionPrediction]`
    and `(dict[str, float], PatientProfile, CaseContext)` -> `dict[str, float]`.
    """
    return_dict = isinstance(stage1_output, dict)
    if return_dict:
        raw_probs = stage1_output
        if case is None and not isinstance(top_n, int):
            case = top_n
            top_n = len(raw_probs) or 5
        if isinstance(active_triggers, PatientProfile):
            trigger_list: list[str] = list(getattr(case, "active_triggers", []) or [])
        else:
            trigger_list = list(active_triggers or [])
    else:
        raw_probs = stage1_output.condition_probs
        trigger_list = list(active_triggers or [])

    priors: dict[str, float] = defaultdict(float)
    for raw_id, prob in raw_probs.items():
        ck = resolve_condition(raw_id)
        if ck is None:
            continue
        priors[ck.condition_id] += max(0.0, float(prob))

    total_prior = sum(priors.values())
    if total_prior <= 0.0:
        return {} if return_dict else []
    for k in priors:
        priors[k] /= total_prior

    unique_triggers = [t for t in dict.fromkeys(trigger_list) if t in TRIGGER_INDEX]

    unnorm_log_posts: dict[str, float] = {}
    explain_map: dict[str, list[dict[str, Any]]] = {}

    for cond_id, prior_p in priors.items():
        ck = CONDITION_GRAPH[cond_id]
        bucket_log_sums: dict[int, float] = defaultdict(float)
        bucket_items: dict[int, list[dict[str, Any]]] = defaultdict(list)

        site_val = str(getattr(getattr(case, "anatomical_site", None), "value", getattr(case, "anatomical_site", "")) or "").lower()
        body_fold_only = {"t09_damp_clothes_wet_hair__damp_clothes", "t08_tight_synthetic_clothes", "t08_tight_synthetic_clothes__tight_clothes", "t08_tight_synthetic_clothes__synthetic_fabric"}
        applied_parents: set[str] = set()
        for tid in unique_triggers:
            if site_val in {"eyelids", "lips_mucosa"} and tid in body_fold_only:
                continue
            parent_id = base_trigger_id(tid)
            lr = ck.trigger_lrs.get(tid, ck.trigger_lrs.get(parent_id, 1.0))
            if abs(lr - 1.0) < 1e-6:
                continue
            trig = TRIGGER_INDEX[tid]
            # Apply parent LR at most ONCE even when multiple split sibling chips are checked
            if parent_id not in applied_parents:
                applied_parents.add(parent_id)
                bucket_log_sums[trig.bucket_id] += math.log(max(lr, 1e-3))
            bucket_items[trig.bucket_id].append({
                "trigger_id": tid,
                "bucket_id": trig.bucket_id,
                "label": trig.label,
                "lr": lr,
            })

        total_capped_log_lr = 0.0
        contributors: list[dict[str, Any]] = []
        for bid, raw_log_sum in bucket_log_sums.items():
            capped = max(-MAX_BUCKET_LOG_LR, min(MAX_BUCKET_LOG_LR, raw_log_sum))
            total_capped_log_lr += capped
            contributors.extend(bucket_items[bid])

        sym_entry = _symptom_compatibility_lrs(case).get(cond_id)
        sym_log_lr = 0.0
        if sym_entry is not None:
            sym_lr, sym_note = sym_entry
            sym_log_lr = math.log(max(sym_lr, 1e-3))
            if sym_lr > 1.05:
                contributors.append({"trigger_id": "clinical_symptom_profile", "bucket_id": 0, "label": sym_note, "lr": sym_lr})
        unnorm_log_posts[cond_id] = math.log(max(prior_p, 1e-9)) + total_capped_log_lr + sym_log_lr
        explain_map[cond_id] = sorted(contributors, key=lambda x: -x["lr"])

    max_log = max(unnorm_log_posts.values())
    exp_scores = {k: math.exp(v - max_log) for k, v in unnorm_log_posts.items()}
    denom = sum(exp_scores.values())

    predictions: list[ConditionPrediction] = []
    for cond_id, exp_val in exp_scores.items():
        ck = CONDITION_GRAPH[cond_id]
        predictions.append(
            ConditionPrediction(
                condition_id=cond_id,
                display_name=ck.display_name,
                tier1_family=ck.tier1_family,
                prior_prob=round(priors[cond_id], 4),
                posterior_prob=round(exp_val / denom, 4),
                contributing_triggers=explain_map[cond_id],
            )
        )

    predictions.sort(key=lambda p: p.posterior_prob, reverse=True)
    if return_dict:
        return {p.condition_id: p.posterior_prob for p in predictions}
    return predictions[:top_n]


# ==============================================================================
# 5. DEFENSIVE PMBJP + 1MG CATALOG LOADER (FOR GATES 1-3 & DDI GATE 11)
# ==============================================================================
def _find_col(cols: list[str], candidates: list[str]) -> str | None:
    lmap = {re.sub(r"[\s\-]+", "_", c.lower().strip()): c for c in cols}
    for cand in candidates:
        key = re.sub(r"[\s\-]+", "_", cand.lower().strip())
        if key in lmap:
            return lmap[key]
    return None


def _parse_entities(raw: Any) -> list[SaltEntity]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return []
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, str):
        txt = raw.strip()
        if not txt or txt in {"[]", "nan", "None", "null"}:
            return []
        try:
            items = json.loads(txt)
        except json.JSONDecodeError:
            try:
                items = ast.literal_eval(txt)
            except (ValueError, SyntaxError):
                return []
    else:
        return []

    out: list[SaltEntity] = []
    for entry in items if isinstance(items, list) else []:
        if isinstance(entry, dict):
            salt = re.sub(r"\s+na$", "", re.sub(r"\s+", " ", str(entry.get("salt", "")).strip().lower())).strip()
            strength = str(entry.get("strength", "N/A")).strip()
            if not strength or strength.lower() in {"none", "nan", "na"}:
                strength = "N/A"
            if salt and salt != "na":
                out.append(SaltEntity(salt=salt, strength=strength))
    return out


def _parse_price(raw: Any) -> float | None:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None
    s = re.sub(r"[^\d.]", "", str(raw))
    if not s:
        return None
    try:
        return round(float(s), 2)
    except ValueError:
        return None


class MedicineCatalog:
    """Loads PMBJP generics and 1mg commercial prices/DDI entries defensively."""

    def __init__(
        self,
        pmbjp_csv: Path | None = None,
        onemg_csv: Path | None = None,
    ) -> None:
        self.pmbjp_rows: list[dict[str, Any]] = []
        self.commercial_avg_price_by_salt: dict[str, float] = {}
        self.ddi_index: dict[str, dict[str, str]] = defaultdict(dict)

        if pmbjp_csv and pmbjp_csv.exists():
            self._load_pmbjp(pmbjp_csv)
        if onemg_csv and onemg_csv.exists():
            self._load_onemg(onemg_csv)

    def _load_pmbjp(self, path: Path) -> None:
        df = pd.read_csv(path)
        cols = list(df.columns)
        c_name = _find_col(cols, ["generic_name", "product_name", "name", "drug_name"])
        c_code = _find_col(cols, ["drug_code", "generic_code", "code", "sr_no"])
        c_mrp = _find_col(cols, ["mrp", "product_price", "price"])
        c_veh = _find_col(cols, ["vehicle", "dosage_form", "form"])
        c_ent = _find_col(cols, ["parsed_entities", "entities", "salts"])

        if not c_name:
            return
        for _, row in df.iterrows():
            salts = _parse_entities(row[c_ent]) if c_ent else []
            if not salts:
                continue
            self.pmbjp_rows.append({
                "product_name": str(row[c_name]).strip(),
                "generic_code": str(row[c_code]).strip() if c_code and pd.notna(row[c_code]) else None,
                "mrp_inr": _parse_price(row[c_mrp]) if c_mrp else None,
                "vehicle": str(row[c_veh]).strip().upper() if c_veh and pd.notna(row[c_veh]) else "UNKNOWN",
                "salts": salts,
            })

    def _load_onemg(self, path: Path) -> None:
        df = pd.read_csv(path)
        cols = list(df.columns)
        c_price = _find_col(cols, ["product_price", "price", "mrp"])
        c_ent = _find_col(cols, ["parsed_entities", "entities"])
        c_ddi = _find_col(cols, ["drug_interactions"])

        price_buckets: dict[str, list[float]] = defaultdict(list)
        for _, row in df.iterrows():
            salts = _parse_entities(row[c_ent]) if c_ent else []
            if not salts:
                continue
            salt_key = "+".join(sorted(s.salt for s in salts))
            price = _parse_price(row[c_price]) if c_price else None
            if price is not None:
                price_buckets[salt_key].append(price)

            if c_ddi and pd.notna(row[c_ddi]):
                try:
                    ddi_obj = json.loads(str(row[c_ddi]))
                    drugs = ddi_obj.get("drug", [])
                    effects = ddi_obj.get("effect", [])
                    for s in salts:
                        for other_d, eff in zip(drugs, effects):
                            other_norm = str(other_d).strip().lower()
                            eff_norm = str(eff).strip().upper()
                            if other_norm:
                                self.ddi_index[s.salt][other_norm] = eff_norm
                except Exception:
                    pass

        for sk, vals in price_buckets.items():
            self.commercial_avg_price_by_salt[sk] = round(sum(vals) / len(vals), 2)