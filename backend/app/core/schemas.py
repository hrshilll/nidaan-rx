"""
Nidaan-Rx v2.0 — Core Data Schemas (Pillar 2, Pillar 3, Pillar 4).

Strict Pydantic v2 models shared across the Knowledge Graph, Bayesian Modifier,
12 Deterministic Safety Gates, and FastAPI endpoints.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class Tier1Family(str, Enum):
    INFLAMMATORY_ACNEIFORM = "inflammatory_acneiform"
    FUNGAL_INFECTIONS = "fungal_infections"
    BACTERIAL_VIRAL_PARASITIC = "bacterial_viral_parasitic"
    ECZEMATOUS_ALLERGIC = "eczematous_allergic"
    PAPULOSQUAMOUS = "papulosquamous"
    PIGMENTARY = "pigmentary"
    SCALP_FOLLICULAR = "scalp_follicular"
    URTICARIAL_REACTIVE = "urticarial_reactive"
    BENIGN_GROWTHS = "benign_growths"
    NEOPLASTIC_RED_FLAG = "neoplastic_red_flag"


class RegulatoryTier(str, Enum):
    TIER1_OTC = "TIER1_OTC"
    TIER2_SCHEDULE_H = "TIER2_SCHEDULE_H"
    TIER2_SCHEDULE_H1 = "TIER2_SCHEDULE_H1"


class AnatomicalSite(str, Enum):
    FACE = "face"
    EYELIDS = "eyelids"
    LIPS_MUCOSA = "lips_mucosa"
    SCALP = "scalp"
    BEARD = "beard"
    TRUNK = "trunk"
    ARMS_LEGS = "arms_legs"
    HANDS_FEET = "hands_feet"
    GROIN = "groin"
    UNDERARMS = "underarms"
    NAILS = "nails"
    GENERALIZED = "generalized"


class FollowUpTrend(str, Enum):
    INITIAL = "initial"
    IMPROVING = "improving"
    UNCHANGED = "unchanged"
    WORSENING = "worsening"


class SteroidPotency(str, Enum):
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    POTENT = "potent"
    SUPER_POTENT = "super_potent"


# ---------------------------------------------------------------------------
# Pillar 2: Static Baseline User Profile
# ---------------------------------------------------------------------------
class PatientProfile(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Patient age in years")
    biological_sex: Literal["male", "female", "other"] = "other"
    weight_kg: float = Field(..., gt=2.0, le=300.0, description="Body weight in kg for dosing safety")
    weight_updated_at: date | None = Field(default=None, description="Last confirmed weight date (90-day refresh)")

    skin_type: Literal["oily", "dry", "combination", "sensitive", "normal"] = "combination"
    fitzpatrick_tone: int = Field(default=4, ge=1, le=6)
    monk_skin_tone: int = Field(default=6, ge=1, le=10)

    drug_allergies: list[str] = Field(default_factory=list, description="Known drug/salt allergies (lowercase)")
    ongoing_oral_meds: list[str] = Field(default_factory=list, description="Current oral medications for DDI check")
    systemic_conditions: list[str] = Field(
        default_factory=list,
        description="e.g. pcos, thyroid, diabetes, anemia, liver_disease, kidney_disease, vitamin_deficiency",
    )
    is_pregnant: bool = False
    is_lactating: bool = False

    sleep_hours: float = Field(default=7.0, ge=0.0, le=24.0)
    sleep_pattern: Literal["regular", "late_sleeper", "night_shift", "irregular"] = "regular"
    stress_level: Literal["low", "moderate", "high"] = "moderate"
    climate_tags: list[str] = Field(default_factory=list, description="humid, dry, high_uv, hard_water, cold")
    current_routine_actives: list[str] = Field(
        default_factory=list,
        description="Current skincare actives e.g. tretinoin, adapalene, salicylic acid, glycolic acid, benzoyl peroxide",
    )

    @field_validator(
        "drug_allergies",
        "ongoing_oral_meds",
        "systemic_conditions",
        "climate_tags",
        "current_routine_actives",
        mode="before",
    )
    @classmethod
    def _clean_tag_lists(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError("Must be a list of text items.")
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in v:
            s = str(item).strip().lower()
            if not s or s in ("none", "nil", "n/a", "na", "-"):
                continue
            if len(s) > 80:
                raise ValueError("Each entry must be 80 characters or fewer.")
            if s not in seen:
                seen.add(s)
                cleaned.append(s)
        if len(cleaned) > 30:
            raise ValueError("At most 30 items are allowed per field.")
        return cleaned

    @model_validator(mode="after")
    def _normalize_pregnancy_flags(self) -> "PatientProfile":
        if self.biological_sex != "female":
            self.is_pregnant = False
            self.is_lactating = False
        return self

    def needs_weight_refresh(self, today: date | None = None) -> bool:
        if self.weight_updated_at is None:
            return True
        ref = today or date.today()
        return (ref - self.weight_updated_at).days >= 90


# ---------------------------------------------------------------------------
# Pillar 3: Active Case Context & Dynamic Q&A Inputs
# ---------------------------------------------------------------------------
class RedFlagChecklist(BaseModel):
    fever: bool = False
    rapidly_spreading_redness_warmth: bool = False
    pus_oozing_bleeding: bool = False
    severe_pain: bool = False
    eye_lip_throat_swelling: bool = False
    breathing_trouble: bool = False
    changing_bleeding_dark_mole: bool = False

    def active_flags(self) -> list[str]:
        return [k for k, v in self.model_dump().items() if v]


class CaseContext(BaseModel):
    anatomical_site: AnatomicalSite = AnatomicalSite.FACE
    duration_days: int = Field(default=7, ge=0, le=3650)
    itch_score: int = Field(default=0, ge=0, le=10)
    pain_burning_score: int = Field(default=0, ge=0, le=10)
    has_scaling: bool = False
    compromised_barrier: bool = False

    red_flags: RedFlagChecklist = Field(default_factory=RedFlagChecklist)
    active_triggers: list[str] = Field(default_factory=list, description="Active trigger IDs across the 14 buckets")
    other_trigger_note: str = ""

    followup_day: int = Field(default=0, ge=0, description="Days elapsed since case opened (0 = initial visit)")
    followup_trend: FollowUpTrend = FollowUpTrend.INITIAL


# ---------------------------------------------------------------------------
# Vision / Fusion & Bayesian Posterior Payloads
# ---------------------------------------------------------------------------
class ConditionPrediction(BaseModel):
    condition_id: str
    display_name: str
    tier1_family: Tier1Family
    prior_prob: float = Field(..., ge=0.0, le=1.0)
    posterior_prob: float = Field(..., ge=0.0, le=1.0)
    contributing_triggers: list[dict[str, Any]] = Field(default_factory=list)


class ModelInferenceOutput(BaseModel):
    condition_probs: dict[str, float] = Field(
        ..., description="Calibrated Stage-1 neural probabilities keyed by condition_id"
    )
    family_probs: dict[str, float] = Field(default_factory=dict)
    is_ood: bool = False
    ood_score: float = 0.0


# ---------------------------------------------------------------------------
# Medication / Knowledge Graph Entities
# ---------------------------------------------------------------------------
class SaltEntity(BaseModel):
    salt: str
    strength: str = "N/A"


class TreatmentOption(BaseModel):
    option_id: str
    display_name: str
    salts: list[SaltEntity]
    vehicle: str
    regulatory_tier: RegulatoryTier
    posology: str
    steroid_potency: SteroidPotency = SteroidPotency.NONE
    is_photosensitizing: bool = False
    is_barrier_harsh: bool = False
    is_systemic: bool = False
    min_age_years: int = 12
    pregnancy_safe: bool = True
    lactation_safe: bool = True
    contraindicated_systemic: list[str] = Field(default_factory=list)
    max_adult_daily_mg: float | None = None
    max_mg_per_kg_day: float | None = None
    notes: str = ""


class PMBJPGenericMatch(BaseModel):
    product_name: str
    generic_code: str | None = None
    mrp_inr: float | None = None
    vehicle: str
    salts: list[SaltEntity]
    equivalence_verified: bool = True
    commercial_avg_inr: float | None = None
    estimated_savings_pct: float | None = None


class TriggerItem(BaseModel):
    trigger_id: str
    bucket_id: int = Field(..., ge=1, le=14)
    label: str
    description: str = ""


class TriggerBucket(BaseModel):
    bucket_id: int = Field(..., ge=1, le=14)
    slug: str
    title: str
    india_critical: bool = False
    triggers: list[TriggerItem]


class ConditionKnowledge(BaseModel):
    condition_id: str
    display_name: str
    tier1_family: Tier1Family
    is_infectious: bool = False
    is_neoplastic_or_red_flag: bool = False
    summary: str
    otc_options: list[TreatmentOption] = Field(default_factory=list)
    rx_discussion_options: list[TreatmentOption] = Field(default_factory=list)
    do_not_apply_defaults: list[str] = Field(default_factory=list)
    lifestyle_plan: list[str] = Field(default_factory=list)
    trigger_lrs: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Safety Gate Audit & Final Clinical Output Contract
# ---------------------------------------------------------------------------
class GateEvent(BaseModel):
    gate_id: str
    gate_name: str
    triggered: bool
    action: Literal["PASS", "WARN", "BLOCK", "MODIFY", "HARD_STOP"]
    message: str
    affected_option_id: str | None = None


class EvaluatedTreatmentCard(BaseModel):
    option: TreatmentOption
    regulatory_label: str
    inline_badges: list[str] = Field(default_factory=list)
    pmbjp_matches: list[PMBJPGenericMatch] = Field(default_factory=list)
    dosage_note: str | None = None


class ClinicalAssessmentResponse(BaseModel):
    disclaimer_banner: str
    escalation_hard_stop: bool = False
    escalation_reason: str | None = None
    minor_guardian_notice: str | None = None
    weight_refresh_prompt: bool = False

    top_matches: list[ConditionPrediction] = Field(default_factory=list)
    identified_triggers: list[str] = Field(default_factory=list)
    do_not_apply_warnings: list[str] = Field(default_factory=list)

    tier1_otc_care: list[EvaluatedTreatmentCard] = Field(default_factory=list)
    tier2_dermatologist_prep: list[EvaluatedTreatmentCard] = Field(default_factory=list)
    lifestyle_action_plan: list[str] = Field(default_factory=list)

    safety_audit_log: list[GateEvent] = Field(default_factory=list)