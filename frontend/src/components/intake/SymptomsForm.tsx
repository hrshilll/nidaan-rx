"use client";

import { useId } from "react";
import { Checkbox, Field, inputClass } from "@/components/ui/primitives";
import { ANATOMICAL_SITES, RED_FLAGS } from "@/lib/constants";
import type { AnatomicalSite, CaseContextInput, RedFlagChecklist } from "@/lib/api";
import { cx } from "@/lib/format";

function Slider({
  id,
  label,
  value,
  onChange,
}: {
  id: string;
  label: string;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between">
        <label htmlFor={id} className="text-sm font-medium">
          {label}
        </label>
        <span className="tabular-nums text-sm text-ink-soft">
          <span className="font-semibold text-ink">{value}</span> / 10
        </span>
      </div>
      <input
        id={id}
        type="range"
        min={0}
        max={10}
        step={1}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-11 w-full cursor-pointer accent-accent"
      />
    </div>
  );
}

export function SymptomsForm({
  value,
  onChange,
  durationError,
}: {
  value: CaseContextInput;
  onChange: (patch: Partial<CaseContextInput>) => void;
  durationError: string | null;
}) {
  const siteId = useId();
  const durationId = useId();
  const itchId = useId();
  const painId = useId();
  const flags = value.red_flags;
  const activeFlags = Object.values(flags).filter(Boolean).length;

  const setFlag = (key: keyof RedFlagChecklist, v: boolean) =>
    onChange({ red_flags: { ...flags, [key]: v } });

  return (
    <section aria-labelledby="symptoms-heading" className="flex flex-col gap-8">
      <h2 id="symptoms-heading" className="text-base font-semibold tracking-tight">
        Symptoms
      </h2>

      <div className="grid gap-5 sm:grid-cols-2">
        <Field id={siteId} label="Where is it?">
          <select
            id={siteId}
            value={value.anatomical_site}
            onChange={(e) => onChange({ anatomical_site: e.target.value as AnatomicalSite })}
            className={inputClass}
          >
            {ANATOMICAL_SITES.map((s) => (
              <option key={s.value} value={s.value}>
                {s.label}
              </option>
            ))}
          </select>
        </Field>
        <Field id={durationId} label="How long has it lasted (days)" error={durationError}>
          <input
            id={durationId}
            type="number"
            min={0}
            max={3650}
            value={value.duration_days}
            onChange={(e) => onChange({ duration_days: Number(e.target.value) })}
            className={inputClass}
          />
        </Field>
      </div>

      <div className="grid gap-6 sm:grid-cols-2">
        <Slider id={itchId} label="Itch" value={value.itch_score} onChange={(v) => onChange({ itch_score: v })} />
        <Slider
          id={painId}
          label="Pain or burning"
          value={value.pain_burning_score}
          onChange={(v) => onChange({ pain_burning_score: v })}
        />
      </div>

      <div className="flex flex-wrap gap-x-8 gap-y-2">
        <Checkbox checked={value.has_scaling} onChange={(v) => onChange({ has_scaling: v })} label="Visible scaling or flaking" />
        <Checkbox
          checked={value.compromised_barrier}
          onChange={(v) => onChange({ compromised_barrier: v })}
          label="Skin feels raw or cracked"
        />
      </div>

      <fieldset
        className={cx(
          "rounded-xl border p-5 transition-colors duration-300",
          activeFlags ? "border-danger/40 bg-danger-tint" : "border-line bg-surface",
        )}
      >
        <legend className="px-1 text-sm font-semibold text-ink">Urgent red-flag screen</legend>
        <p className="mb-3 text-sm text-ink-soft">
          {activeFlags === 0
            ? "Check any of these that apply. Checked items go to the safety gates with the case."
            : `${activeFlags} checked. These are included with the case for escalation review.`}
        </p>
        <div className="grid gap-x-6 sm:grid-cols-2">
          {RED_FLAGS.map((f) => (
            <Checkbox
              key={f.key}
              checked={flags[f.key]}
              onChange={(v) => setFlag(f.key, v)}
              label={f.label}
              tone="danger"
            />
          ))}
        </div>
      </fieldset>
    </section>
  );
}
