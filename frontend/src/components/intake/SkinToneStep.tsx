"use client";

import { MONK_SWATCHES, SUN_REACTIONS } from "@/lib/constants";
import { cx } from "@/lib/format";

export function SkinToneStep({
  monk,
  fitzpatrick,
  onMonk,
  onFitzpatrick,
}: {
  monk: number;
  fitzpatrick: number;
  onMonk: (tone: number) => void;
  onFitzpatrick: (tone: number) => void;
}) {
  const sun = SUN_REACTIONS.find((r) => r.fitzpatrick === fitzpatrick);
  return (
    <div className="flex flex-col gap-7">
      <div>
        <h3 className="text-sm font-medium">Match your skin tone</h3>
        <p className="mt-1 text-sm text-ink-soft">
          Compare your inner forearm in daylight with the swatches. Avoid your face, which may be tanned or treated.
        </p>
        <div role="radiogroup" aria-label="Monk skin tone" className="mt-4 grid grid-cols-5 gap-2 sm:grid-cols-10">
          {MONK_SWATCHES.map((s) => {
            const on = s.tone === monk;
            return (
              <button
                key={s.tone}
                type="button"
                role="radio"
                aria-checked={on}
                aria-label={`Monk ${s.tone}`}
                onClick={() => onMonk(s.tone)}
                style={{ backgroundColor: s.hex }}
                className={cx(
                  "relative aspect-square min-h-11 rounded-lg border border-black/10 transition-[transform,box-shadow] duration-200 ease-out hover:-translate-y-px",
                  on && "ring-2 ring-accent ring-offset-2 ring-offset-surface",
                  "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent",
                )}
              >
                <span
                  className={cx(
                    "absolute inset-0 grid place-items-center text-xs font-semibold",
                    s.tone >= 6 ? "text-white/90" : "text-black/60",
                  )}
                >
                  {s.tone}
                </span>
              </button>
            );
          })}
        </div>
        <p className="mt-3 text-sm text-ink-soft">
          Selected: <span className="font-medium text-ink">Monk {monk}</span>
        </p>
      </div>

      <div>
        <h3 className="text-sm font-medium">How does your skin react to a long afternoon in the sun?</h3>
        <div role="radiogroup" aria-label="Sun reaction" className="mt-3 grid gap-2 sm:grid-cols-2">
          {SUN_REACTIONS.map((r) => {
            const on = r.fitzpatrick === fitzpatrick;
            return (
              <button
                key={r.fitzpatrick}
                type="button"
                role="radio"
                aria-checked={on}
                onClick={() => onFitzpatrick(r.fitzpatrick)}
                className={cx(
                  "flex min-h-11 flex-col items-start gap-0.5 rounded-lg border px-4 py-3 text-left transition-[background-color,border-color] duration-200 ease-out",
                  "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent",
                  on ? "border-accent bg-accent-tint" : "border-line-strong bg-surface hover:border-ink/40",
                )}
              >
                <span className="text-sm font-medium">{r.title}</span>
                <span className="text-xs text-ink-soft">{r.detail}</span>
              </button>
            );
          })}
        </div>
        {sun && (
          <p className="mt-3 text-xs text-ink-soft">
            Recorded as Fitzpatrick type {toRoman(sun.fitzpatrick)} for dosing and safety checks.
          </p>
        )}
      </div>
    </div>
  );
}

function toRoman(n: number) {
  return ["I", "II", "III", "IV", "V", "VI"][n - 1] ?? String(n);
}
