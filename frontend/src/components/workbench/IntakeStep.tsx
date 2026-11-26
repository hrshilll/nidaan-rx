"use client";

import type { CaseContextInput, PatientProfileInput } from "@/lib/api";
import type { SlotResult } from "@/components/intake/PhotoSlots";
import { PhotoSlots } from "@/components/intake/PhotoSlots";
import { ProfileForm } from "@/components/intake/ProfileForm";
import { SymptomsForm } from "@/components/intake/SymptomsForm";
import { Button } from "@/components/ui/primitives";

export function validateIntake(files: (File | null)[], profile: PatientProfileInput, symptoms: CaseContextInput) {
  const photos = files.filter(Boolean).length;
  const ageOk = Number.isInteger(profile.age) && profile.age >= 1 && profile.age <= 120;
  const weightOk = Number.isFinite(profile.weight_kg) && profile.weight_kg > 2 && profile.weight_kg <= 300;
  const durationOk = Number.isInteger(symptoms.duration_days) && symptoms.duration_days >= 0 && symptoms.duration_days <= 3650;
  return {
    photos: photos >= 1 ? null : "Add at least one photograph.",
    age: ageOk ? null : "Enter an age between 1 and 120.",
    weight: weightOk ? null : "Enter a weight above 2 kg and up to 300 kg.",
    duration: durationOk ? null : "Enter a duration between 0 and 3650 days.",
    valid: photos >= 1 && ageOk && weightOk && durationOk,
  };
}

export function IntakeStep({
  files,
  onFile,
  results,
  qualityGate,
  onQualityGate,
  profile,
  onProfile,
  symptoms,
  onSymptoms,
  busy,
  error,
  onSubmit,
  rerun,
  stale,
}: {
  files: (File | null)[];
  onFile: (i: number, f: File | null) => void;
  results: (SlotResult | null)[];
  qualityGate: boolean;
  onQualityGate: (v: boolean) => void;
  profile: PatientProfileInput;
  onProfile: (p: Partial<PatientProfileInput>) => void;
  symptoms: CaseContextInput;
  onSymptoms: (p: Partial<CaseContextInput>) => void;
  busy: boolean;
  error: string | null;
  onSubmit: () => void;
  rerun: boolean;
  stale: boolean;
}) {
  const v = validateIntake(files, profile, symptoms);

  return (
    <div className="grid gap-12 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] lg:gap-16">
      <div className="flex flex-col gap-12">
        <PhotoSlots
          files={files}
          onChange={onFile}
          results={results}
          qualityGate={qualityGate}
          onQualityGate={onQualityGate}
          disabled={busy}
        />
        {v.photos && <p className="-mt-8 text-xs text-ink-soft">{v.photos}</p>}
        <ProfileForm profile={profile} onChange={onProfile} ageError={v.age} weightError={v.weight} />
      </div>

      <div className="flex flex-col gap-10">
        <SymptomsForm value={symptoms} onChange={onSymptoms} durationError={v.duration} />

        <div className="sticky bottom-4 flex flex-col gap-3 rounded-xl border border-line bg-surface/95 p-4 shadow-sm backdrop-blur">
          {error && (
            <p role="alert" className="text-sm text-danger">
              {error}
            </p>
          )}
          {stale && !busy && (
            <p className="text-xs text-warn">Inputs changed since the last analysis. Run it again to refresh results.</p>
          )}
          <Button onClick={onSubmit} disabled={!v.valid || busy} className="w-full">
            {busy ? "Analyzing photos…" : rerun ? "Analyze again" : "Analyze photos and profile"}
          </Button>
          {busy && (
            <div className="h-1 overflow-hidden rounded-full bg-surface-2" aria-hidden="true">
              <div className="sweep h-full w-1/3 rounded-full bg-accent" />
            </div>
          )}
          {!busy && !v.valid && (
            <p className="text-xs text-ink-soft">Complete the required fields above to continue.</p>
          )}
        </div>
      </div>
    </div>
  );
}

