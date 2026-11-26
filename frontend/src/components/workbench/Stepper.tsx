"use client";

import { cx } from "@/lib/format";

export type StepId = "intake" | "review" | "report";

const STEPS: { id: StepId; label: string }[] = [
  { id: "intake", label: "Photos and profile" },
  { id: "review", label: "Review triggers" },
  { id: "report", label: "Clinical report" },
];

export function Stepper({
  current,
  reachable,
  onGo,
}: {
  current: StepId;
  reachable: Record<StepId, boolean>;
  onGo: (s: StepId) => void;
}) {
  const index = STEPS.findIndex((s) => s.id === current);
  return (
    <nav aria-label="Triage progress" className="flex flex-col gap-3">
      <ol className="grid grid-cols-3 gap-2">
        {STEPS.map((s, i) => {
          const done = i < index;
          const active = i === index;
          const enabled = reachable[s.id];
          return (
            <li key={s.id}>
              <button
                type="button"
                disabled={!enabled}
                aria-current={active ? "step" : undefined}
                onClick={() => onGo(s.id)}
                className={cx(
                  "flex min-h-11 w-full flex-col items-start justify-center gap-1 border-t-2 pt-2 text-left transition-[border-color,color] duration-300 ease-(--ease-settle)",
                  "disabled:cursor-default focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent",
                  active ? "border-accent" : done ? "border-accent/50" : "border-line",
                )}
              >
                <span className={cx("text-xs", active ? "font-semibold text-accent-strong" : "text-ink-soft")}>
                  {done ? "Done" : active ? "In progress" : "Next"}
                </span>
                <span className={cx("text-sm", active ? "font-medium text-ink" : "text-ink-soft")}>{s.label}</span>
              </button>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
