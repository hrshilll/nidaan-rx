"use client";

import type { TriageStageResponse } from "@/lib/api";
import { delay, humanize, pct } from "@/lib/format";
import { Meter } from "@/components/ui/primitives";

export function Stage1Results({ triage }: { triage: TriageStageResponse }) {
  const out = triage.stage1_output;
  const top = Object.entries(out.condition_probs)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5);

  return (
    <section aria-labelledby="stage1-heading" className="flex flex-col gap-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 id="stage1-heading" className="text-base font-semibold tracking-tight">
            Vision and fusion probabilities
          </h2>
          <p className="mt-1 text-sm text-ink-soft">
            {triage.backbone} · {Math.round(triage.latency_ms)} ms · OOD distance {out.ood_score.toFixed(2)}
          </p>
        </div>
        {out.is_ood && (
          <span className="rounded-full border border-warn/30 bg-warn-tint px-3 py-1 text-xs font-medium text-warn">
            Outside the validated range
          </span>
        )}
      </div>

      {out.is_ood && (
        <p className="text-sm text-ink-soft">
          This image sits outside the training distribution. The uncertainty gate will apply to the report.
        </p>
      )}

      <ul className="flex flex-col gap-4">
        {top.map(([id, p], i) => (
          <li key={id} className="rise flex flex-col gap-1.5" style={delay(i * 60)}>
            <div className="flex items-baseline justify-between gap-3 text-sm">
              <span className="capitalize">{humanize(id)}</span>
              <span className="tabular-nums font-medium text-ink">{pct(p)}</span>
            </div>
            <Meter value={p} label={humanize(id)} />
          </li>
        ))}
      </ul>

      <p className="text-xs text-ink-soft">
        {triage.all_views_passed_quality
          ? "All photos passed the quality check."
          : "One or more photos failed the quality check. Consider retaking them before you rely on this result."}
      </p>
    </section>
  );
}
