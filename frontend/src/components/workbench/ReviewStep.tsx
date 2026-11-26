"use client";

import type { TriageStageResponse } from "@/lib/api";
import { Stage1Results } from "@/components/review/Stage1Results";
import { TriggerSelector } from "@/components/review/TriggerSelector";
import { Button } from "@/components/ui/primitives";

export function ReviewStep({
  triage,
  selected,
  onToggle,
  onBack,
  onAssess,
  busy,
  error,
}: {
  triage: TriageStageResponse;
  selected: string[];
  onToggle: (id: string) => void;
  onBack: () => void;
  onAssess: () => void;
  busy: boolean;
  error: string | null;
}) {
  return (
    <div className="grid gap-12 lg:grid-cols-[minmax(0,5fr)_minmax(0,6fr)] lg:gap-16">
      <Stage1Results triage={triage} />
      <div className="flex flex-col gap-10">
        <TriggerSelector buckets={triage.dynamic_trigger_buckets} selected={selected} onToggle={onToggle} />
        <div className="flex flex-col gap-3 border-t border-line pt-6">
          {error && (
            <p role="alert" className="text-sm text-danger">
              {error}
            </p>
          )}
          <div className="flex flex-col-reverse gap-3 sm:flex-row sm:justify-between">
            <Button variant="ghost" onClick={onBack} disabled={busy}>
              Edit photos and profile
            </Button>
            <Button onClick={onAssess} disabled={busy} className="sm:min-w-64">
              {busy ? "Preparing report…" : "Generate clinical report"}
            </Button>
          </div>
          {busy && (
            <div className="h-1 overflow-hidden rounded-full bg-surface-2" aria-hidden="true">
              <div className="sweep h-full w-1/3 rounded-full bg-accent" />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
