"use client";

import React, { useState } from "react";
import { Plus, X } from "lucide-react";

const IGNORED_EMPTY_TOKENS = new Set(["none", "nil", "n/a", "na", "no", "nothing", "-"]);

interface TagChipFieldProps {
  label: string;
  hint: string;
  values: string[];
  presets: readonly string[];
  placeholder?: string;
  onChange: (next: string[]) => void;
}

export function TagChipField({
  label,
  hint,
  values,
  presets,
  placeholder = "Add custom item & press Enter",
  onChange,
}: TagChipFieldProps) {
  const [draft, setDraft] = useState("");

  const addToken = (raw: string) => {
    const clean = raw.trim().toLowerCase();
    if (!clean || IGNORED_EMPTY_TOKENS.has(clean)) {
      setDraft("");
      return;
    }
    if (!values.includes(clean)) {
      onChange([...values, clean]);
    }
    setDraft("");
  };

  const toggleToken = (token: string) => {
    const clean = token.trim().toLowerCase();
    if (values.includes(clean)) {
      onChange(values.filter((v) => v !== clean));
    } else {
      onChange([...values, clean]);
    }
  };

  return (
    <div className="space-y-2.5">
      <div>
        <p className="text-[13px] font-semibold text-[#164E63]">{label}</p>
        <p className="text-xs text-[#475569]">{hint}</p>
      </div>

      <div className="flex flex-wrap gap-2">
        {presets.map((p) => {
          const active = values.includes(p.toLowerCase());
          return (
            <button
              key={p}
              type="button"
              aria-pressed={active}
              onClick={() => toggleToken(p)}
              className={`cursor-pointer rounded-full px-3 py-1.5 text-xs font-medium transition ${
                active
                  ? "bg-[#0891B2] text-white shadow-2xs"
                  : "bg-[#E8F1F6] text-[#334155] hover:bg-[#D7E7F0]"
              }`}
            >
              {p.replaceAll("_", " ")}
            </button>
          );
        })}

        {values
          .filter((v) => !presets.map((p) => p.toLowerCase()).includes(v))
          .map((custom) => (
            <span
              key={custom}
              className="inline-flex items-center gap-1 rounded-full bg-[#ECFEFF] px-3 py-1.5 text-xs font-medium text-[#0E7490] ring-1 ring-[#A5F3FC]"
            >
              {custom}
              <button
                type="button"
                onClick={() => onChange(values.filter((v) => v !== custom))}
                className="cursor-pointer rounded-full hover:text-[#FF3B30]"
                aria-label={`Remove ${custom}`}
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </span>
          ))}
      </div>

      <div className="flex gap-2">
        <input
          type="text"
          aria-label={label}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={() => { if (draft.trim()) addToken(draft); }}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              addToken(draft);
            }
          }}
          placeholder={placeholder}
          className="h-10 flex-1 rounded-xl bg-[#F3FAFC] px-3.5 text-xs text-[#164E63] ring-1 ring-[#CBEBF2] placeholder:text-[#64748B] focus:bg-white focus:ring-2 focus:ring-[#0891B2] focus:outline-none"
        />
        <button
          type="button"
          onClick={() => addToken(draft)}
          className="inline-flex h-10 cursor-pointer items-center gap-1 rounded-xl bg-[#E8F1F6] px-3.5 text-xs font-semibold text-[#164E63] transition hover:bg-[#E5E5EA]"
        >
          <Plus className="h-3.5 w-3.5" /> Add
        </button>
      </div>
    </div>
  );
}
