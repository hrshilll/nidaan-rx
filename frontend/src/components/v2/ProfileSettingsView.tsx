"use client";

import React, { useState } from "react";
import { CheckCircle2, Loader2, LogOut, Save, ShieldCheck, User } from "lucide-react";
import {
  saveUserProfile,
  type AuthUser,
  type PatientProfileInput,
  type SleepPattern,
  type StressLevel,
} from "@/lib/api";
import { CLIMATE_TAGS, PRESETS } from "@/lib/constants";
import { SkinTonePicker } from "./SkinTonePicker";
import { TagChipField } from "./TagChipField";

interface ProfileSettingsViewProps {
  user: AuthUser;
  profile: PatientProfileInput;
  weightUpdatedAt: string | null;
  onProfileSaved: (nextProfile: PatientProfileInput, nextWeightUpdatedAt: string) => void;
  onSignOut: () => void;
}

const SLEEP_OPTIONS: { value: SleepPattern; label: string }[] = [
  { value: "regular", label: "Regular" },
  { value: "late_sleeper", label: "Late night" },
  { value: "night_shift", label: "Night shift" },
  { value: "irregular", label: "Irregular" },
];

export function ProfileSettingsView({
  user,
  profile,
  weightUpdatedAt,
  onProfileSaved,
  onSignOut,
}: ProfileSettingsViewProps) {
  const [draft, setDraft] = useState<PatientProfileInput>(profile);
  const [saving, setSaving] = useState(false);
  const [savedBanner, setSavedBanner] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const patch = (p: Partial<PatientProfileInput>) => {
    setDraft((prev) => ({ ...prev, ...p }));
    setSavedBanner(false);
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
    setSaving(true);
    setError(null);
    setSavedBanner(false);
    try {
      const res = await saveUserProfile(draft);
      onProfileSaved(res.profile, res.weight_updated_at);
      setSavedBanner(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save profile.");
    } finally {
      setSaving(false);
    }
  };

  const formattedWeightDate = weightUpdatedAt
    ? new Date(weightUpdatedAt).toLocaleDateString("en-IN", {
        day: "numeric",
        month: "short",
        year: "numeric",
      })
    : null;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-[28px] sm:text-[32px] font-bold tracking-tight text-[#164E63]">
            Health Profile
          </h1>
          <p className="mt-1 text-sm text-[#475569]">
            Saved once and auto-filled into every new skin assessment.
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
            <p className="text-xs text-[#475569]">{user.email}</p>
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
            <label className="block text-xs font-semibold text-[#475569]">Age (years)</label>
            <input
              type="number"
              min={1}
              max={110}
              value={draft.age}
              onChange={(e) => patch({ age: Math.max(1, Number(e.target.value) || 1) })}
              className="mt-1.5 h-11 w-full rounded-xl bg-[#F3FAFC] px-3.5 font-mono text-sm font-semibold text-[#164E63] ring-1 ring-[#CBEBF2] focus:bg-white focus:ring-2 focus:ring-[#0891B2] focus:outline-none"
            />
          </div>
          <div className="sm:col-span-3">
            <label className="block text-xs font-semibold text-[#475569]">Weight (kg)</label>
            <input
              type="number"
              min={2}
              max={250}
              step="0.5"
              value={draft.weight_kg}
              onChange={(e) => patch({ weight_kg: Math.max(2, Number(e.target.value) || 2) })}
              className="mt-1.5 h-11 w-full rounded-xl bg-[#F3FAFC] px-3.5 font-mono text-sm font-semibold text-[#164E63] ring-1 ring-[#CBEBF2] focus:bg-white focus:ring-2 focus:ring-[#0891B2] focus:outline-none"
            />
          </div>
          <div className="sm:col-span-6">
            <label className="block text-xs font-semibold text-[#475569]">Biological Sex</label>
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

        <div className="grid gap-4 border-t border-[#E8F1F6] pt-5 sm:grid-cols-12">
          <div className="sm:col-span-3">
            <label className="block text-xs font-semibold text-[#475569]">Sleep (hrs)</label>
            <input
              type="number"
              min={2}
              max={14}
              step="0.5"
              value={draft.sleep_hours}
              onChange={(e) => patch({ sleep_hours: Number(e.target.value) || 7 })}
              className="mt-1.5 h-10 w-full rounded-xl bg-[#F3FAFC] px-3.5 font-mono text-sm font-semibold text-[#164E63] ring-1 ring-[#CBEBF2] focus:bg-white focus:ring-2 focus:ring-[#0891B2] focus:outline-none"
            />
          </div>
          <div className="sm:col-span-5">
            <label className="block text-xs font-semibold text-[#475569]">Sleep Pattern</label>
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
            <label className="block text-xs font-semibold text-[#475569]">Stress Level</label>
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
          <p className="text-xs font-semibold text-[#475569]">Local Climate & Water</p>
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
            Checked automatically against all 12 clinical safety gates on every triage.
          </p>
        </div>
        <TagChipField
          label="Drug Allergies"
          hint="Blocks matching salts and cross-reactive drug classes."
          values={draft.drug_allergies}
          presets={PRESETS.drugAllergies}
          onChange={(drug_allergies) => patch({ drug_allergies })}
        />
        <TagChipField
          label="Ongoing Oral Medicines"
          hint="Screened for drug-drug interactions."
          values={draft.ongoing_oral_meds}
          presets={PRESETS.oralMeds}
          onChange={(ongoing_oral_meds) => patch({ ongoing_oral_meds })}
        />
        <TagChipField
          label="Long-Term Conditions"
          hint="e.g., diabetes, thyroid, kidney or liver disease."
          values={draft.systemic_conditions}
          presets={PRESETS.systemic}
          onChange={(systemic_conditions) => patch({ systemic_conditions })}
        />
        <TagChipField
          label="Current Skincare Actives"
          hint="Prevents duplicate actives or harsh barrier stacking."
          values={draft.current_routine_actives}
          presets={PRESETS.routineActives}
          onChange={(current_routine_actives) => patch({ current_routine_actives })}
        />

        <div className="flex justify-end border-t border-[#E8F1F6] pt-4">
          <button
            type="button"
            disabled={saving}
            onClick={handleSave}
            className="inline-flex h-11 cursor-pointer items-center gap-2 rounded-full bg-[#059669] px-6 text-sm font-semibold text-white shadow-xs transition hover:bg-[#047857] disabled:opacity-60"
          >
            {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
            {saving ? "Saving..." : "Save Health Profile"}
          </button>
        </div>
      </section>
    </div>
  );
}
