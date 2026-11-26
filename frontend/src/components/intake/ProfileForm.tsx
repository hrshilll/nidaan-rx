"use client";

import { useId } from "react";
import { Field, Segmented, Checkbox, inputClass } from "@/components/ui/primitives";
import { PRESETS, CLIMATE_TAGS } from "@/lib/constants";
import type { PatientProfileInput } from "@/lib/api";
import { SkinToneStep } from "./SkinToneStep";
import { TagInput, asPresets } from "./TagInput";

export function ProfileForm({
  profile,
  onChange,
  weightError,
  ageError,
}: {
  profile: PatientProfileInput;
  onChange: (patch: Partial<PatientProfileInput>) => void;
  weightError: string | null;
  ageError: string | null;
}) {
  const ageId = useId();
  const weightId = useId();
  const sleepId = useId();
  const patternId = useId();
  const showPregnancy = profile.biological_sex === "female";

  return (
    <section aria-labelledby="profile-heading" className="flex flex-col gap-8">
      <h2 id="profile-heading" className="text-base font-semibold tracking-tight">
        Patient profile
      </h2>

      <div className="grid gap-5 sm:grid-cols-3">
        <Field id={ageId} label="Age (years)" error={ageError}>
          <input
            id={ageId}
            type="number"
            inputMode="numeric"
            min={1}
            max={120}
            value={profile.age}
            onChange={(e) => onChange({ age: Number(e.target.value) })}
            className={inputClass}
          />
        </Field>
        <Field label="Sex at birth">
          <Segmented
            label="Sex at birth"
            value={profile.biological_sex}
            options={[
              { value: "female", label: "Female" },
              { value: "male", label: "Male" },
              { value: "other", label: "Other" },
            ]}
            onChange={(v) =>
              onChange({
                biological_sex: v,
                ...(v !== "female" ? { is_pregnant: false, is_lactating: false } : {}),
              })
            }
          />
        </Field>
        <Field id={weightId} label="Weight (kg)" error={weightError} hint="Used for dose checks. Confirm it is current.">
          <input
            id={weightId}
            type="number"
            inputMode="decimal"
            step="0.1"
            min={2.1}
            max={300}
            value={profile.weight_kg}
            onChange={(e) => onChange({ weight_kg: Number(e.target.value) })}
            className={inputClass}
          />
        </Field>
      </div>

      {showPregnancy && (
        <div className="flex flex-wrap gap-5">
          <Checkbox checked={profile.is_pregnant} onChange={(v) => onChange({ is_pregnant: v })} label="Pregnant" />
          <Checkbox checked={profile.is_lactating} onChange={(v) => onChange({ is_lactating: v })} label="Breastfeeding" />
        </div>
      )}

      <div className="flex flex-col gap-3">
        <h3 className="text-sm font-medium">Skin tone</h3>
        <SkinToneStep
          monk={profile.monk_skin_tone}
          fitzpatrick={profile.fitzpatrick_tone}
          onMonk={(t) => onChange({ monk_skin_tone: t })}
          onFitzpatrick={(t) => onChange({ fitzpatrick_tone: t })}
        />
      </div>

      <div className="grid gap-5 sm:grid-cols-2">
        <Field id={sleepId} label="Sleep (hours a night)">
          <input
            id={sleepId}
            type="number"
            step="0.5"
            min={0}
            max={24}
            value={profile.sleep_hours}
            onChange={(e) => onChange({ sleep_hours: Number(e.target.value) })}
            className={inputClass}
          />
        </Field>
        <Field id={patternId} label="Sleep pattern">
          <select
            id={patternId}
            value={profile.sleep_pattern}
            onChange={(e) => onChange({ sleep_pattern: e.target.value as PatientProfileInput["sleep_pattern"] })}
            className={inputClass}
          >
            <option value="regular">Regular</option>
            <option value="late_sleeper">Late sleeper</option>
            <option value="night_shift">Night shift</option>
            <option value="irregular">Irregular</option>
          </select>
        </Field>
      </div>

      <Field label="Stress level">
        <Segmented
          label="Stress level"
          value={profile.stress_level}
          options={[
            { value: "low", label: "Low" },
            { value: "moderate", label: "Moderate" },
            { value: "high", label: "High" },
          ]}
          onChange={(v) => onChange({ stress_level: v })}
        />
      </Field>

      <TagInput
        label="Where you live"
        hint="Local climate and water can trigger flares."
        values={profile.climate_tags}
        presets={CLIMATE_TAGS}
        allowCustom={false}
        onChange={(v) => onChange({ climate_tags: v })}
      />

      <div className="flex flex-col gap-6 border-t border-line pt-7">
        <h3 className="text-sm font-medium">Medicines and conditions</h3>
        <TagInput
          label="Drug allergies"
          hint="Safety gates check these first."
          values={profile.drug_allergies}
          presets={asPresets(["penicillin", "sulfonamide", "nsaid", "lidocaine"])}
          onChange={(v) => onChange({ drug_allergies: v })}
        />
        <TagInput
          label="Oral medicines you take now"
          hint="Checked for interactions with the treatment list."
          values={profile.ongoing_oral_meds}
          presets={asPresets(PRESETS.oralMeds)}
          onChange={(v) => onChange({ ongoing_oral_meds: v })}
        />
        <TagInput
          label="Long-term conditions"
          values={profile.systemic_conditions}
          presets={asPresets(PRESETS.systemic)}
          onChange={(v) => onChange({ systemic_conditions: v })}
        />
        <TagInput
          label="Skincare actives you already use"
          hint="Stacking strong actives raises irritation risk."
          values={profile.current_routine_actives}
          presets={asPresets(PRESETS.routineActives)}
          onChange={(v) => onChange({ current_routine_actives: v })}
        />
      </div>
    </section>
  );
}
