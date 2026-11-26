"""
Atomic trigger splitting so multi-concept triggers (e.g. "Low daily water intake,
heavy alcohol, or smoking") appear as separate, accurate chips in the UI while
preserving backward compatibility with parent trigger_ids and Bayesian LRs.
"""
from __future__ import annotations

from app.core.schemas import TriggerBucket, TriggerItem

# Map parent trigger_id -> list of (suffix, atomic_label, atomic_description)
ATOMIC_SPLIT_BY_ID: dict[str, list[tuple[str, str, str]]] = {
    # Bucket 1: Indian OTC / Home Remedies
    "t01_home_remedy_irritant": [
        ("acidic_diy", "Applied lemon, baking soda, or toothpaste", "Harsh alkaline or acidic DIY home remedies"),
        ("herbal_paste", "Applied raw garlic, turmeric, or herbal paste", "Unrefined topical pastes that can irritate skin"),
    ],
    # Bucket 2: Systemic Medications
    "t02_recent_antibiotic_nsaid": [
        ("antibiotic", "Recent oral antibiotic course", "Systemic antibiotics in the last 2–4 weeks"),
        ("nsaid", "Recent painkiller / NSAID (ibuprofen, aspirin)", "NSAID or fever painkiller exposure"),
    ],
    # Bucket 3: Diet, Gut & Hydration
    "t03_low_hydration_alcohol": [
        ("low_water", "Low daily water intake", "Consistently drinking less water than needed"),
        ("alcohol", "Regular or heavy alcohol intake", "Frequent alcohol consumption"),
        ("smoking", "Smoking or tobacco use", "Cigarettes, vaping, or tobacco use"),
    ],
    "t03_food_allergen_trigger": [
        ("allergen_foods", "Flare after seafood, nuts, or eggs", "Hives or itching soon after common food allergens"),
        ("spicy_fermented", "Flare after spicy or fermented foods", "Flushing or flare after spicy/fermented meals"),
    ],
    # Bucket 4: Stress, Sleep & Habits
    "t04_poor_irregular_sleep": [
        ("short_sleep", "Sleeping less than 6 hours regularly", "Chronic sleep deprivation"),
        ("irregular_schedule", "Irregular sleep timing or late nights", "Shifting sleep schedule or night shifts"),
    ],
    "t04_picking_scratching": [
        ("picking", "Picking or squeezing spots", "Mechanical manipulation of pimples or scabs"),
        ("scratching", "Repeated scratching or rubbing", "Habitual itch-scratch cycle"),
    ],
    # Bucket 5: Climate & Weather
    "t05_cold_dry_windburn": [
        ("cold_dry_air", "Cold or dry winter air", "Low outdoor humidity drying the skin barrier"),
        ("ac_windburn", "Windburn or long hours in dry AC", "Dry indoor air-conditioning or harsh wind"),
    ],
    "t05_intense_uv_sun": [
        ("heavy_uv", "Heavy daily outdoor sun / UV exposure", "Prolonged midday sunlight exposure"),
        ("sunburn", "Recent sunburn without sunscreen", "Acute UV burn or tanning flare"),
    ],
    # Bucket 7: Cosmetics, Topicals & Cultural Products
    "t07_new_skincare_makeup": [
        ("new_cleanser_moist", "Started new cleanser or moisturizer", "Recently changed core wash or cream"),
        ("new_serum_spf", "Started new serum, active, or sunscreen", "Recently added leave-on treatment or SPF"),
        ("new_makeup", "Started new makeup or foundation", "Cosmetic product introduced recently"),
    ],
    "t07_heavy_hair_oil": [
        ("hair_oil", "Frequent heavy hair oiling (coconut / mustard)", "Overnight or heavy scalp oiling"),
        ("pomade_wax", "Hair wax, gel, or occlusive pomades", "Heavy styling products near hairline/forehead"),
    ],
    "t07_hair_dye_ppd": [
        ("hair_dye", "Recent hair dye (PPD / box color)", "Chemical hair coloring exposure"),
        ("henna_mehndi", "Recent henna or mehndi application", "Natural or black henna exposure"),
        ("salon_chemical", "Salon chemical hair treatment", "Keratin, rebonding, or smoothing treatment"),
    ],
    "t07_perfume_bindi_detergent": [
        ("perfume", "New perfume or deodorant", "Fragrance spray or roll-on contact"),
        ("detergent", "New laundry detergent or fabric softener", "Residue from clothes or pillowcase wash"),
        ("bindi_kumkum", "Bindi or kumkum adhesive", "Forehead adhesive or pigment contact"),
    ],
    # Bucket 8: Mechanical Friction
    "t08_helmet_cap_strap": [
        ("helmet", "Regular two-wheeler helmet use", "Sweat and strap friction from helmets"),
        ("cap_mask", "Tight cap, headband, or face mask", "Localized pressure or friction"),
    ],
    "t08_tight_synthetic_clothes": [
        ("tight_clothes", "Tight-fitting clothing", "Friction along waist, thighs, or collar"),
        ("synthetic_fabric", "Synthetic or non-breathable fabrics", "Heat and moisture trapping against skin"),
    ],
    "t08_cut_scratch_koebner": [
        ("minor_injury", "Recent minor cut, graze, or abrasion", "Skin trauma at the rash site"),
        ("tattoo_friction", "Recent shave nick, wax, or tattoo", "Localized mechanical skin injury"),
    ],
    # Bucket 9: Sweat, Occlusion & Hygiene
    "t09_damp_clothes_wet_hair": [
        ("damp_clothes", "Staying in sweaty or damp clothes/socks", "Prolonged moisture against body folds or feet"),
        ("wet_hair", "Sleeping with wet hair", "Damp scalp overnight encouraging yeast growth"),
    ],
    "t09_overwashing_barrier_strip": [
        ("harsh_soap", "Overwashing with harsh soaps", "Frequent washing stripping natural skin lipids"),
        ("daily_scrub", "Daily physical scrubs or exfoliation", "Abrasive scrubs irritating the barrier"),
    ],
    "t09_shared_towels_combs": [
        ("shared_towels", "Shared towels or bedsheets", "Household linen sharing"),
        ("shared_grooming", "Shared combs, razors, or helmets", "Personal grooming item sharing"),
        ("shared_footwear", "Shared footwear or socks", "Footwear sharing"),
    ],
    # Bucket 11: Hormonal & Endocrine
    "t11_pregnancy_postpartum": [
        ("pregnant", "Current pregnancy hormonal shift", "Pregnancy-related endocrine changes"),
        ("postpartum", "Postpartum hormonal shift", "Recent childbirth hormonal shift"),
    ],
    "t11_thyroid_insulin_shift": [
        ("thyroid", "Known thyroid imbalance", "Hypothyroid or hyperthyroid history"),
        ("weight_insulin", "Rapid weight change or insulin resistance", "Metabolic or insulin fluctuation"),
    ],
    # Bucket 13: Nutritional
    "t13_vitamin_mineral_def": [
        ("vit_d_b12", "Low Vitamin D or B12", "Known Vitamin D or B12 deficiency"),
        ("iron_anemia", "Low iron or anemia", "Known low hemoglobin or ferritin"),
    ],
    # Bucket 14: Occupational / Contact
    "t14_wet_work_detergents": [
        ("wet_work", "Frequent dishwashing or wet-hand work", "Prolonged water immersion of hands"),
        ("cleaners", "Direct contact with household cleaners or detergents", "Chemical surfactant irritation"),
    ],
    "t14_nickel_metal_latex": [
        ("metal_nickel", "Metal watch strap, jewelry, or belt buckle", "Nickel / metal contact allergy"),
        ("latex_rubber", "Rubber or latex gloves / bands", "Rubber or latex contact"),
    ],
    "t14_cement_solvents_plants": [
        ("cement_solvents", "Cement, paint, or chemical solvents", "Construction or workshop chemical contact"),
        ("plants", "Gardening or plant sap exposure", "Outdoor foliage or sap contact"),
    ],
}

# Label-substring fallback splits for any trigger whose ID differs
ATOMIC_SPLIT_BY_LABEL: list[tuple[str, list[tuple[str, str, str]]]] = [
    (
        "pre-menstrual flares",
        [
            ("pms", "Pre-menstrual flares", "Breakouts or flares before periods"),
            ("pcos", "Irregular periods or PCOS", "Irregular menstrual cycle or PCOS"),
        ],
    ),
    (
        "high-glycemic",
        [
            ("sweets", "Frequent sweets or sugary drinks", "High added sugar intake"),
            ("junk_carbs", "Refined carbs or fried junk food", "White bread, pastries, or fast food"),
        ],
    ),
    (
        "heavy daily dairy",
        [
            ("dairy_food", "Heavy milk, paneer, or ice cream intake", "High daily whole dairy consumption"),
            ("whey_shakes", "Whey protein or dairy shakes", "Daily whey or milk-based protein shakes"),
        ],
    ),
    (
        "hot, humid, or monsoon",
        [
            ("hot_humid", "Hot and humid weather exposure", "High heat and ambient humidity"),
            ("monsoon", "Monsoon rain or damp weather exposure", "Persistent rainy-season dampness"),
        ],
    ),
    (
        "heavy gym/outdoor sweating",
        [
            ("gym_sweat", "Heavy gym or workout sweating", "Frequent intense exercise sweating"),
            ("outdoor_sweat", "Outdoor heat sweating without prompt wash", "Sweat sitting on skin in hot weather"),
        ],
    ),
]


def base_trigger_id(trigger_id: str) -> str:
    return trigger_id.split("__", 1)[0]


def expand_trigger_item(item: TriggerItem) -> list[TriggerItem]:
    splits = ATOMIC_SPLIT_BY_ID.get(item.trigger_id)
    if splits is None:
        lower_lbl = item.label.lower()
        for needle, candidates in ATOMIC_SPLIT_BY_LABEL:
            if needle in lower_lbl:
                splits = candidates
                break
    if not splits:
        return [item]
    return [
        TriggerItem(
            trigger_id=f"{item.trigger_id}__{suffix}",
            bucket_id=item.bucket_id,
            label=label,
            description=desc,
        )
        for suffix, label, desc in splits
    ]


def expand_trigger_bucket(bucket: TriggerBucket) -> TriggerBucket:
    atomic_items: list[TriggerItem] = []
    for tr in bucket.triggers:
        atomic_items.extend(expand_trigger_item(tr))
    return bucket.model_copy(update={"triggers": atomic_items})
