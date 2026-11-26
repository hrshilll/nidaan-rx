"use client";

import { useId, useState } from "react";
import { Button, Chip, inputClass } from "@/components/ui/primitives";

export interface Preset {
  value: string;
  label: string;
}

export function asPresets(values: readonly string[]): Preset[] {
  return values.map((v) => ({ value: v, label: v.replace(/_/g, " ") }));
}

export function TagInput({
  label,
  hint,
  values,
  presets,
  onChange,
  allowCustom = true,
  placeholder = "Type and press Enter",
}: {
  label: string;
  hint?: string;
  values: string[];
  presets: Preset[];
  onChange: (next: string[]) => void;
  allowCustom?: boolean;
  placeholder?: string;
}) {
  const [draft, setDraft] = useState("");
  const inputId = useId();
  const presetValues = new Set(presets.map((p) => p.value));
  const custom = values.filter((v) => !presetValues.has(v));

  const toggle = (value: string) =>
    onChange(values.includes(value) ? values.filter((v) => v !== value) : [...values, value]);

  const add = (raw: string) => {
    const v = raw.trim().toLowerCase();
    if (v && !values.includes(v)) onChange([...values, v]);
    setDraft("");
  };

  return (
    <div className="flex min-w-0 flex-col gap-2">
      <div>
        <label htmlFor={allowCustom ? inputId : undefined} className="text-sm font-medium">
          {label}
        </label>
        {hint && <p className="text-xs text-ink-soft">{hint}</p>}
      </div>

      <div className="flex flex-wrap gap-2">
        {presets.map((p) => (
          <Chip key={p.value} pressed={values.includes(p.value)} onClick={() => toggle(p.value)}>
            {p.label}
          </Chip>
        ))}
        {custom.map((v) => (
          <Chip key={v} pressed onClick={() => toggle(v)} onRemove={() => toggle(v)}>
            {v}
          </Chip>
        ))}
      </div>

      {allowCustom && (
        <div className="flex gap-2">
          <input
            id={inputId}
            value={draft}
            placeholder={placeholder}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === ",") {
                e.preventDefault();
                add(draft);
              }
            }}
            className={inputClass}
          />
          <Button variant="secondary" onClick={() => add(draft)} disabled={!draft.trim()} className="shrink-0">
            Add
          </Button>
        </div>
      )}
    </div>
  );
}
