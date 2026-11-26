import type { AnatomicalSite, CaseContextInput, PatientProfileInput, RedFlagChecklist } from "./api";

export const ANATOMICAL_SITES: { value: AnatomicalSite; label: string }[] = [
  { value: "face", label: "Face" },
  { value: "eyelids", label: "Eyelids" },
  { value: "lips_mucosa", label: "Lips or mouth lining" },
  { value: "scalp", label: "Scalp" },
  { value: "beard", label: "Beard area" },
  { value: "trunk", label: "Chest or back" },
  { value: "arms_legs", label: "Arms or legs" },
  { value: "hands_feet", label: "Hands or feet" },
  { value: "groin", label: "Groin" },
  { value: "underarms", label: "Underarms" },
  { value: "nails", label: "Nails" },
  { value: "generalized", label: "Widespread, many areas" },
];

export const CLIMATE_TAGS = [
  { value: "humid", label: "Hot and humid" },
  { value: "monsoon", label: "Monsoon" },
  { value: "high_uv", label: "High UV" },
  { value: "dry", label: "Dry winter" },
  { value: "cold", label: "Cold" },
  { value: "urban_dust", label: "Urban dust" },
  { value: "hard_water", label: "Hard or borewell water" },
];

// Official Monk Skin Tone palette hex values, scale 1 (lightest) to 10 (darkest).
export const MONK_SWATCHES: { tone: number; hex: string }[] = [
  { tone: 1, hex: "#f6ede4" },
  { tone: 2, hex: "#f3e7db" },
  { tone: 3, hex: "#f7ead0" },
  { tone: 4, hex: "#eadaba" },
  { tone: 5, hex: "#d7bd96" },
  { tone: 6, hex: "#a07e56" },
  { tone: 7, hex: "#825c43" },
  { tone: 8, hex: "#604134" },
  { tone: 9, hex: "#3a312a" },
  { tone: 10, hex: "#292420" },
];

// Plain-English sun reactions that map onto the Fitzpatrick scale (I to VI).
export const SUN_REACTIONS: { fitzpatrick: number; title: string; detail: string }[] = [
  { fitzpatrick: 1, title: "Always burns", detail: "Burns within minutes and never tans" },
  { fitzpatrick: 2, title: "Usually burns", detail: "Burns easily and tans very little" },
  { fitzpatrick: 3, title: "Sometimes burns", detail: "Burns after long exposure, tans gradually" },
  { fitzpatrick: 4, title: "Rarely burns", detail: "Tans easily and fairly quickly" },
  { fitzpatrick: 5, title: "Very rarely burns", detail: "Tans deeply and often" },
  { fitzpatrick: 6, title: "Never burns, tans deeply", detail: "Rarely or never burns in the sun" },
];

export const PRESETS = {
  drugAllergies: ["penicillin", "sulfonamide", "nsaid", "lidocaine"],
  oralMeds: ["isotretinoin", "ketoconazole", "metformin", "oral contraceptive"],
  systemic: ["diabetes", "thyroid", "pcos", "anemia", "liver_disease", "kidney_disease", "vitamin_deficiency"],
  routineActives: ["adapalene", "tretinoin", "salicylic acid", "glycolic acid", "benzoyl peroxide"],
} as const;

export const RED_FLAGS: { key: keyof RedFlagChecklist; label: string }[] = [
  { key: "fever", label: "Fever or chills" },
  { key: "rapidly_spreading_redness_warmth", label: "Rapidly spreading warmth or redness" },
  { key: "pus_oozing_bleeding", label: "Pus, active oozing, or bleeding" },
  { key: "severe_pain", label: "Severe or rapidly worsening pain" },
  { key: "eye_lip_throat_swelling", label: "Swelling of eyes, lips, or throat" },
  { key: "breathing_trouble", label: "Shortness of breath or wheezing" },
  { key: "changing_bleeding_dark_mole", label: "Changing or bleeding dark spot or mole" },
];

export const SLOT_META = [
  { label: "Close-Up", hint: "Lesion fills most of the frame and is in focus" },
  { label: "Regional Context", hint: "Wider view showing where on the body it sits" },
  { label: "45° Angle", hint: "Taken at about 45 degrees to the skin" },
] as const;

export const DEFAULT_PROFILE: PatientProfileInput = {
  age: 25,
  biological_sex: "male",
  weight_kg: 60,
  fitzpatrick_tone: 4,
  monk_skin_tone: 6,
  drug_allergies: [],
  ongoing_oral_meds: [],
  systemic_conditions: [],
  is_pregnant: false,
  is_lactating: false,
  sleep_hours: 7,
  sleep_pattern: "regular",
  stress_level: "moderate",
  climate_tags: [],
  current_routine_actives: [],
};

export const EMPTY_FLAGS: RedFlagChecklist = {
  fever: false,
  rapidly_spreading_redness_warmth: false,
  pus_oozing_bleeding: false,
  severe_pain: false,
  eye_lip_throat_swelling: false,
  breathing_trouble: false,
  changing_bleeding_dark_mole: false,
};

export const DEFAULT_CASE: CaseContextInput = {
  anatomical_site: "face",
  duration_days: 7,
  itch_score: 0,
  pain_burning_score: 0,
  has_scaling: false,
  compromised_barrier: false,
  red_flags: EMPTY_FLAGS,
  active_triggers: [],
  other_trigger_note: "",
};
