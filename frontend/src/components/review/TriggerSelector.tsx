"use client";

import { Chip, SectionHeading } from "@/components/ui/primitives";
import { delay } from "@/lib/format";
import type { TriggerBucket } from "@/lib/api";

export function TriggerSelector({
  buckets,
  selected,
  onToggle,
}: {
  buckets: TriggerBucket[];
  selected: string[];
  onToggle: (id: string) => void;
}) {
  return (
    <section className="flex flex-col gap-6">
      <SectionHeading
        title="Possible triggers"
        detail="Select anything that fits this patient's history or exposures. Each choice updates the probabilities."
        action={<span className="text-sm tabular-nums text-ink-soft">{selected.length} selected</span>}
      />
      {buckets.map((b, bi) => (
        <div key={b.bucket_id} className="rise flex flex-col gap-3 border-t border-line pt-5" style={delay(bi * 50)}>
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-medium">{b.title}</h3>
            {b.india_critical && <span className="text-xs text-accent-strong">Common in India</span>}
          </div>
          <div className="flex flex-wrap gap-2">
            {b.triggers.map((t) => (
              <Chip key={t.trigger_id} pressed={selected.includes(t.trigger_id)} onClick={() => onToggle(t.trigger_id)}>
                {t.label}
              </Chip>
            ))}
          </div>
        </div>
      ))}
    </section>
  );
}
