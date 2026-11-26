"use client";

import React, { useEffect, useState } from "react";
import { ArrowRight, CheckCircle2, Loader2, LogOut, Save, ShieldCheck, Sparkles, User } from "lucide-react";
import {
  saveUserProfile,
  type AuthUser,
  type PatientProfileInput,
  type SleepPattern,
  type StressLevel,
} from "@/lib/api";
import { CLIMATE_TAGS } from "@/lib/constants";
import { SkinTonePicker } from "./SkinTonePicker";
import { TagChipField } from "./TagChipField";

interface ProfileSettingsViewProps {
  user: AuthUser;
  profile: PatientProfileInput;
  weightUpdatedAt: string | null;
  isOnboarding?: boolean;
  onProfileSaved: (nextProfile: PatientProfileInput, nextWeightUpdatedAt: string) => void;
  onSignOut: () => void;
}

interface ProfileDraft {
  age: string;
  weight_kg: string;
  biological_sex: "male" | "female" | "other" | null;
  is_pregnant: boolean;
  is_lactating: boolean;
  sleep_hours: string;
  sleep_pattern: SleepPattern | null;
  stress_level: StressLevel | null;
  monk_skin_tone: number | null;
  fitzpatrick_tone: number | null;
  climate_tags: string[];
  drug_allergies: string[];
  ongoing_oral_meds: string[];
  systemic_conditions: string[];
  current_routine_actives: string[];
}

const SLEEP_OPTIONS: { value: SleepPattern; label: string }[] = [
  { value: "regular", label: "Regular" },
  { value: "late_sleeper", label: "Late night" },
  { value: "night_shift", label: "Night shift" },
  { value: "irregular", label: "Irregular" },
];

function makeDraft(profile: PatientProfileInput, blank: boolean): ProfileDraft {
  if (blank) {
    return {
      age: "",
      weight_kg: "",
      biological_sex: null,
      is_pregnant: false,
      is_lactating: false,
      sleep_hours: "",
      sleep_pattern: null,
      stress_level: null,
      monk_skin_tone: null,
      fitzpatrick_tone: null,
      climate_tags: [],
      drug_allergies: [],
      ongoing_oral_meds: [],
      systemic_conditions: [],
      current_routine_actives: [],
    };
  }
  return {
    age: String(profile.age ?? ""),
    weight_kg: String(profile.weight_kg ?? ""),
    biological_sex: profile.biological_sex ?? null,
    is_pregnant: Boolean(profile.is_pregnant),
    is_lactating: Boolean(profile.is_lactating),
    sleep_hours: String(profile.sleep_hours ?? ""),
    sleep_pattern: profile.sleep_pattern ?? null,
    stress_level: profile.stress_level ?? null,
    monk_skin_tone: profile.monk_skin_tone ?? null,
    fitzpatrick_tone: profile.fitzpatrick_tone ?? null,
    climate_tags: profile.climate_tags ?? [],
    drug_allergies: profile.drug_allergies ?? [],
    ongoing_oral_meds: profile.ongoing_oral_meds ?? [],
    systemic_conditions: profile.systemic_conditions ?? [],
    current_routine_actives: profile.current_routine_actives ?? [],
  };
}

export function ProfileSettingsView({
  user,
  profile,
  weightUpdatedAt,
  isOnboarding = false,
  onProfileSaved,
  onSignOut,
}: ProfileSettingsViewProps) {
  const [draft, setDraft] = useState<ProfileDraft>(() => makeDraft(profile, isOnboarding));
  const [saving, setSaving] = useState(false);
  const [savedBanner, setSavedBanner] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setDraft(makeDraft(profile, isOnboarding));
  }, [profile, isOnboarding]);

  const patch = (p: Partial<ProfileDraft>) => {
    setDraft((prev) => ({ ...prev, ...p }));
    setSavedBanner(false);
    setError(null);
  };

  const toggleClimate = (tag: string) => {
    const exists = draft.climate_tags.includes(tag);
    patch({
      climate_tags: exists
        ? draft.climate_tags.filter((t) => t !== tag)
        : [...draft.climate_tags, tag],
    });
  };

  const handleSave = async () => {
    setError(null);
    setSavedBanner(false);

    const ageNum = Number(draft.age);
    const weightNum = Number(draft.weight_kg);
    const sleepNum = Number(draft.sleep_hours);

    if (!draft.age.trim() || !Number.isInteger(ageNum) || ageNum < 1 || ageNum > 120) {
      setError("Please enter a valid whole-number age between 1 and 120 years.");
      return;
    }
    if (!draft.weight_kg.trim() || Number.isNaN(weightNum) || weightNum < 2.5 || weightNum > 300) {
      setError("Please enter a valid weight between 2.5 and 300 kg.");
      return;
    }
    if (!draft.biological_sex) {
      setError("Please select your biological sex.");
      return;
    }
    if (!draft.sleep_hours.trim() || Number.isNaN(sleepNum) || sleepNum < 1 || sleepNum > 24) {
      setError("Please enter average sleep hours between 1 and 24.");
      return;
    }
    if (!draft.sleep_pattern) {
      setError("Please select your sleep pattern.");
      return;
    }
    if (!draft.stress_level) {
      setError("Please select your stress level.");
      return;
    }
    if (draft.monk_skin_tone === null || draft.fitzpatrick_tone === null) {
      setError("Please select your Monk Skin Tone and sun reaction (Fitzpatrick type).");
      return;
    }

    const payload: PatientProfileInput = {
      age: ageNum,
      weight_kg: weightNum,
      biological_sex: draft.biological_sex,
      is_pregnant: draft.biological_sex === "female" ? draft.is_pregnant : false,
      is_lactating: draft.biological_sex === "female" ? draft.is_lactating : false,
      sleep_hours: sleepNum,
      sleep_pattern: draft.sleep_pattern,
      stress_level: draft.stress_level,
      monk_skin_tone: draft.monk_skin_tone,
      fitzpatrick_tone: draft.fitzpatrick_tone,
      climate_tags: draft.climate_tags,
      drug_allergies: draft.drug_allergies,
      ongoing_oral_meds: draft.ongoing_oral_meds,
      systemic_conditions: draft.systemic_conditions,
      current_routine_actives: draft.current_routine_actives,
    };

    setSaving(true);
    try {
      const res = await saveUserProfile(payload);
      onProfileSaved(res.profile, res.weight_updated_at);
      setSavedBanner(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save profile.");
    } finally {
      setSaving(false);
    }
  };

  const formattedWeightDate =
    !isOnboarding && weightUpdatedAt
      ? new Date(weightUpdatedAt).toLocaleDateString("en-IN", {
          day: "numeric",
          month: "short",
          year: "numeric",
        })
      : null;

  return (
    <div className="space-y-6">
      {isOnboarding && (
        <div className="flex items-start gap-3 rounded-2xl bg-[#ECFEFF] px-4 py-3.5 text-xs text-[#0E7490] ring-1 ring-[#A5F3FC]">
          <Sparkles className="mt-0.5 h-4 w-4 shrink-0 text-[#0891B2]" />
          <div>
            <p className="font-semibold text-[#164E63]">
              Step 2 of 2 · Complete your baseline health profile
            </p>
            <p className="mt-0.5 leading-relaxed text-[#475569]">
              Welcome, {user.full_name.split(" ")[0]}! Enter your own age, weight, skin tone, and health details once so Nidaan-Rx can calibrate every skin assessment and 12-gate safety check accurately.
            </p>
          </div>
        </div>
      )}

      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-[28px] sm:text-[32px] font-bold tracking-tight text-[#164E63]">
            {isOnboarding ? "Set Up Your Health Profile" : "Health Profile"}
          </h1>
          <p className="mt-1 text-sm text-[#475569]">
            {isOnboarding
              ? "Required once after sign up — no preset values are filled in."
              : "Saved once and auto-filled into every new skin assessment."}
          </p>
        </div>

        <button
          type="button"
          onClick={onSignOut}
          className="inline-flex cursor-pointer items-center gap-1.5 rounded-full bg-white px-4 py-2 text-xs font-semibold text-[#DC2626] ring-1 ring-[#FECACA] transition hover:bg-[#FEF2F2]"
        >
          <LogOut className="h-3.5 w-3.5" /> Sign Out
        </button>
      </div>

      {/* Account Header Card */}
      <section className="clinical-card flex flex-wrap items-center justify-between gap-4 p-5 sm:px-7">
        <div className="flex items-center gap-3.5">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-[#0891B2] text-base font-bold text-white shadow-xs">
            {user.full_name.slice(0, 2).toUpperCase()}
          </div>
          <div>
            <p className="text-[16px] font-bold text-[#164E63]">{user.full_name}</p>
            <p className="text-xs text-[#475569]">
              {user.username ? `@${user.username} · ` : ""}
              {user.email}
            </p>
          </div>
        </div>
        {formattedWeightDate && (
          <span className="inline-flex items-center gap-1.5 rounded-full bg-[#ECFEFF] px-3 py-1 text-xs font-medium text-[#0E7490] ring-1 ring-[#A5F3FC]">
            <ShieldCheck className="h-3.5 w-3.5" />
            Weight verified {formattedWeightDate}
          </span>
        )}
      </section>

      {error && (
        <div className="rounded-2xl bg-[#FEF2F2] px-4 py-3 text-xs font-medium text-[#DC2626] ring-1 ring-[#FECACA]">
          {error}
        </div>
      )}

      {savedBanner && (
        <div className="flex items-center gap-2 rounded-2xl bg-[#ECFDF5] px-4 py-3 text-xs font-semibold text-[#047857] ring-1 ring-[#A7F3D0]">
          <CheckCircle2 className="h-4 w-4" />
          Profile saved — your next skin assessment is automatically pre-filled.
        </div>
      )}

      {/* Baseline Demographics */}
      <section className="clinical-card p-6 sm:p-7 space-y-5">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-[#ECFEFF] text-[#0891B2] ring-1 ring-[#A5F3FC]">
            <User className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-[17px] font-bold text-[#164E63]">Baseline Measurements</h2>
            <p className="text-xs text-[#475569]">Powers pediatric dosing checks and trigger filtering.</p>
          </div>
        </div>

        <div className="grid gap-4 sm:grid-cols-12">
          <div className="sm:col-span-3">
            <label className="block text-xs font-semibold text-[#475569]">Age (1–120 yrs) *</label>
            <input
              type="number"
              required
              min={1}
              max={120}
              step={1}
              placeholder="e.g. 24"
              value={draft.age}
              onChange={(e) => patch({ age: e.target.value })}
              className="mt-1.5 h-11 w-full rounded-xl bg-[#F3FAFC] px-3.5 font-mono text-sm font-semibold text-[#164E63] ring-1 ring-[#CBEBF2] focus:bg-white focus:ring-2 focus:ring-[#0891B2] focus:outline-none"
            />
          </div>
          <div className="sm:col-span-3">
            <label className="block text-xs font-semibold text-[#475569]">Weight (2.5–300 kg) *</label>
            <input
              type="number"
              required
              min={2.5}
              max={300}
              step="0.5"
              placeholder="e.g. 62"
              value={draft.weight_kg}
              onChange={(e) => patch({ weight_kg: e.target.value })}
              className="mt-1.5 h-11 w-full rounded-xl bg-[#F3FAFC] px-3.5 font-mono text-sm font-semibold text-[#164E63] ring-1 ring-[#CBEBF2] focus:bg-white focus:ring-2 focus:ring-[#0891B2] focus:outline-none"
            />
          </div>
          <div className="sm:col-span-6">
            <label className="block text-xs font-semibold text-[#475569]">Biological Sex *</label>
            <div className="mt-1.5 grid h-11 grid-cols-3 gap-1 rounded-xl bg-[#E8F1F6] p-1">
              {(["male", "female", "other"] as const).map((sex) => (
                <button
                  key={sex}
                  type="button"
                  onClick={() =>
                    patch({
                      biological_sex: sex,
                      ...(sex !== "female" ? { is_pregnant: false, is_lactating: false } : {}),
                    })
                  }
                  className={`cursor-pointer rounded-lg text-xs font-semibold capitalize transition ${
                    draft.biological_sex === sex
                      ? "bg-white text-[#164E63] shadow-xs"
                      : "text-[#475569] hover:text-[#164E63]"
                  }`}
                >
                  {sex}
                </button>
              ))}
            </div>
          </div>
        </div>

        {draft.biological_sex === "female" && (
          <div className="flex flex-wrap gap-4 rounded-xl bg-[#F3FAFC] px-4 py-3 ring-1 ring-[#CBEBF2]">
            <label className="inline-flex cursor-pointer items-center gap-2 text-xs font-semibold text-[#164E63]">
              <input
                type="checkbox"
                checked={draft.is_pregnant}
                onChange={(e) => patch({ is_pregnant: e.target.checked })}
                className="h-4 w-4 accent-[#0891B2]"
              />
              Currently pregnant
            </label>
            <label className="inline-flex cursor-pointer items-center gap-2 text-xs font-semibold text-[#164E63]">
              <input
                type="checkbox"
                checked={draft.is_lactating}
                onChange={(e) => patch({ is_lactating: e.target.checked })}
                className="h-4 w-4 accent-[#0891B2]"
              />
              Currently breastfeeding
            </label>
          </div>
        )}

        <div className="grid gap-4 border-t border-[#E8F1F6] pt-5 sm:grid-cols-12">
          <div className="sm:col-span-3">
            <label className="block text-xs font-semibold text-[#475569]">Sleep (1–24 hrs) *</label>
            <input
              type="number"
              required
              min={1}
              max={24}
              step="0.5"
              placeholder="e.g. 7"
              value={draft.sleep_hours}
              onChange={(e) => patch({ sleep_hours: e.target.value })}
              className="mt-1.5 h-10 w-full rounded-xl bg-[#F3FAFC] px-3.5 font-mono text-sm font-semibold text-[#164E63] ring-1 ring-[#CBEBF2] focus:bg-white focus:ring-2 focus:ring-[#0891B2] focus:outline-none"
            />
          </div>
          <div className="sm:col-span-5">
            <label className="block text-xs font-semibold text-[#475569]">Sleep Pattern *</label>
            <div className="mt-1.5 grid grid-cols-2 gap-1.5">
              {SLEEP_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  type="button"
                  onClick={() => patch({ sleep_pattern: opt.value })}
                  className={`cursor-pointer rounded-xl px-2.5 py-2 text-xs font-semibold transition ${
                    draft.sleep_pattern === opt.value
                      ? "bg-[#0891B2] text-white"
                      : "bg-[#E8F1F6] text-[#334155] hover:bg-[#D8E7F0]"
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>
          <div className="sm:col-span-4">
            <label className="block text-xs font-semibold text-[#475569]">Stress Level *</label>
            <div className="mt-1.5 grid h-10 grid-cols-3 gap-1 rounded-xl bg-[#E8F1F6] p-1">
              {(["low", "moderate", "high"] as StressLevel[]).map((lvl) => (
                <button
                  key={lvl}
                  type="button"
                  onClick={() => patch({ stress_level: lvl })}
                  className={`cursor-pointer rounded-lg text-xs font-semibold capitalize transition ${
                    draft.stress_level === lvl
                      ? "bg-white text-[#164E63] shadow-xs"
                      : "text-[#475569]"
                  }`}
                >
                  {lvl === "moderate" ? "Med" : lvl}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="border-t border-[#E8F1F6] pt-5">
          <p className="text-xs font-semibold text-[#475569]">Local Climate & Water (optional)</p>
          <div className="mt-2.5 flex flex-wrap gap-2">
            {CLIMATE_TAGS.map((tag) => {
              const active = draft.climate_tags.includes(tag.value);
              return (
                <button
                  key={tag.value}
                  type="button"
                  onClick={() => toggleClimate(tag.value)}
                  className={`cursor-pointer rounded-full px-3.5 py-1.5 text-xs font-semibold transition ${
                    active
                      ? "bg-[#0891B2] text-white"
                      : "bg-[#E8F1F6] text-[#334155] hover:bg-[#D7E7F0]"
                  }`}
                >
                  {tag.label}
                </button>
              );
            })}
          </div>
        </div>
      </section>

      <SkinTonePicker
        monkTone={draft.monk_skin_tone}
        fitzpatrickTone={draft.fitzpatrick_tone}
        onChangeMonk={(monk_skin_tone) => patch({ monk_skin_tone })}
        onChangeFitzpatrick={(fitzpatrick_tone) => patch({ fitzpatrick_tone })}
      />

      <section className="clinical-card p-6 sm:p-7 space-y-5">
        <div>
          <h2 className="text-[17px] font-bold text-[#164E63]">
            Permanent Safety & Medication Record
          </h2>
          <p className="text-xs text-[#475569]">
            Optional — add any allergies, ongoing oral medicines, long-term conditions, or active skincare ingredients.
          </p>
        </div>
        <TagChipField
          label="Drug Allergies"
          hint="Blocks matching salts and cross-reactive drug classes."
          values={draft.drug_allergies}
          presets={[]}
          placeholder="Type allergy & press Enter (leave blank if none)"
          onChange={(drug_allergies) => patch({ drug_allergies })}
        />
        <TagChipField
          label="Ongoing Oral Medicines"
          hint="Screened for drug-drug interactions."
          values={draft.ongoing_oral_meds}
          presets={[]}
          placeholder="Type oral medicine & press Enter (leave blank if none)"
          onChange={(ongoing_oral_meds) => patch({ ongoing_oral_meds })}
        />
        <TagChipField
          label="Long-Term Conditions"
          hint="e.g., diabetes, thyroid, kidney or liver disease."
          values={draft.systemic_conditions}
          presets={[]}
          placeholder="Type condition & press Enter (leave blank if none)"
          onChange={(systemic_conditions) => patch({ systemic_conditions })}
        />
        <TagChipField
          label="Current Skincare Actives"
          hint="Prevents duplicate actives or harsh barrier stacking."
          values={draft.current_routine_actives}
          presets={[]}
          placeholder="Type active ingredient & press Enter (leave blank if none)"
          onChange={(current_routine_actives) => patch({ current_routine_actives })}
        />

        <div className="flex justify-end border-t border-[#E8F1F6] pt-4">
          <button
            type="button"
            disabled={saving}
            onClick={handleSave}
            className="inline-flex h-11 cursor-pointer items-center gap-2 rounded-full bg-[#059669] px-6 text-sm font-semibold text-white shadow-xs transition hover:bg-[#047857] disabled:opacity-60"
          >
            {saving ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : isOnboarding ? (
              <ArrowRight className="h-4 w-4" />
            ) : (
              <Save className="h-4 w-4" />
            )}
            {saving
              ? "Saving..."
              : isOnboarding
              ? "Save & Continue to Assessment"
              : "Save Health Profile"}
          </button>
        </div>
      </section>
    </div>
  );
}
