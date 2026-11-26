"use client";

import React from "react";
import { AlertCircle, Sun } from "lucide-react";
import { MONK_SWATCHES, SUN_REACTIONS } from "@/lib/constants";

interface SkinTonePickerProps {
  monkTone: number;
  fitzpatrickTone: number;
  onChangeMonk: (tone: number) => void;
  onChangeFitzpatrick: (tone: number) => void;
}

const FITZ_ROMAN = ["", "I", "II", "III", "IV", "V", "VI"];

function suggestedFitzpatrickForMonk(monk: number): number {
  if (monk <= 1) return 1;
  if (monk <= 2) return 2;
  if (monk <= 4) return 3;
  if (monk <= 6) return 4;
  if (monk <= 8) return 5;
  return 6;
}

export function SkinTonePicker({
  monkTone,
  fitzpatrickTone,
  onChangeMonk,
  onChangeFitzpatrick,
}: SkinTonePickerProps) {
  const expectedFitz = suggestedFitzpatrickForMonk(monkTone);
  const mismatch = Math.abs(fitzpatrickTone - expectedFitz) >= 2;

  const handleSelectMonk = (tone: number) => {
    onChangeMonk(tone);
    onChangeFitzpatrick(suggestedFitzpatrickForMonk(tone));
  };

  return (
    <section className="clinical-card p-6 sm:p-7 space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-start gap-3.5">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-[#ECFEFF] text-[#0891B2] ring-1 ring-[#A5F3FC]">
            <Sun className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-[17px] font-bold tracking-tight text-[#164E63]">
              Skin Tone Calibration
            </h2>
            <p className="mt-0.5 text-[13px] text-[#475569]">
              Compare the swatches below against the{" "}
              <strong className="font-semibold text-[#164E63]">inside of your forearm</strong> in daylight.
            </p>
          </div>
        </div>

        <span className="rounded-full bg-[#ECFEFF] px-3 py-1 text-xs font-semibold text-[#0E7490] ring-1 ring-[#A5F3FC] tabular-nums">
          Monk {monkTone} · Fitzpatrick {FITZ_ROMAN[fitzpatrickTone] || fitzpatrickTone}
        </span>
      </div>

      <div className="grid grid-cols-5 gap-2.5 sm:grid-cols-10">
        {MONK_SWATCHES.map((sw) => {
          const active = monkTone === sw.tone;
          const lightText = sw.tone >= 6;
          return (
            <button
              key={sw.tone}
              type="button"
              onClick={() => handleSelectMonk(sw.tone)}
              style={{ backgroundColor: sw.hex }}
              className={`relative flex h-12 cursor-pointer items-center justify-center rounded-2xl text-xs font-semibold tabular-nums transition-all duration-150 ${
                lightText ? "text-white" : "text-[#292420]"
              } ${
                active
                  ? "ring-[2.5px] ring-[#0891B2] ring-offset-2 ring-offset-white scale-[1.06] shadow-sm"
                  : "ring-1 ring-black/[0.08] hover:scale-[1.04]"
              }`}
              aria-label={`Monk Skin Tone ${sw.tone}`}
              aria-pressed={active}
            >
              {sw.tone}
            </button>
          );
        })}
      </div>

      <div className="border-t border-[#E8F1F6] pt-5">
        <p className="text-[13px] font-semibold text-[#164E63]">
          How does your skin react to 30–45 minutes of midday sun?
        </p>

        {mismatch && (
          <div className="mt-2.5 flex items-start gap-2 rounded-xl bg-[#FFFBEB] px-3.5 py-2.5 text-xs text-[#92400E] ring-1 ring-[#FDE68A]">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-[#D97706]" />
            <span>
              Monk {monkTone} usually aligns with Fitzpatrick Type {FITZ_ROMAN[expectedFitz]}. Double-check both selections so skin-tone calibration stays accurate.
            </span>
          </div>
        )}

        <div className="mt-3 grid grid-cols-1 gap-2.5 sm:grid-cols-3">
          {SUN_REACTIONS.map((item) => {
            const active = fitzpatrickTone === item.fitzpatrick;
            return (
              <button
                key={item.fitzpatrick}
                type="button"
                aria-pressed={active}
                onClick={() => onChangeFitzpatrick(item.fitzpatrick)}
                className={`cursor-pointer rounded-2xl p-3.5 text-left transition-all ${
                  active
                    ? "bg-[#ECFEFF] ring-2 ring-[#0891B2] shadow-2xs"
                    : "bg-[#F3FAFC] ring-1 ring-[#DCECF1] hover:bg-[#E8F1F6]"
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span
                    className={`text-[13px] font-semibold ${
                      active ? "text-[#164E63]" : "text-[#164E63]"
                    }`}
                  >
                    {item.title}
                  </span>
                  <span
                    className={`rounded-full px-2 py-0.5 text-[10px] font-bold ${
                      active
                        ? "bg-[#0891B2] text-white"
                        : "bg-white text-[#475569] ring-1 ring-[#CBEBF2]"
                    }`}
                  >
                    {FITZ_ROMAN[item.fitzpatrick]}
                  </span>
                </div>
                <p className="mt-1 text-[11px] leading-snug text-[#475569]">{item.detail}</p>
              </button>
            );
          })}
        </div>
      </div>
    </section>
  );
}
