"use client";

import { useEffect, useState } from "react";
import { fetchMetrics } from "@/lib/api";
import { errorText, humanize } from "@/lib/format";
import { Meter, SectionHeading } from "@/components/ui/primitives";

const BACKBONE_NAMES: Record<string, string> = {
  panderm_base: "PanDerm-Base",
  dinov2_small: "DINOv2-Small",
};

type Flat = Record<string, number>;

function flatten(node: unknown, prefix = "", out: Flat = {}): Flat {
  if (typeof node === "number" && Number.isFinite(node)) {
    out[prefix] = node;
  } else if (node && typeof node === "object" && !Array.isArray(node)) {
    for (const [k, v] of Object.entries(node as Record<string, unknown>)) {
      flatten(v, prefix ? `${prefix}.${k}` : k, out);
    }
  }
  return out;
}

export function Benchmarks() {
  const [reports, setReports] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchMetrics()
      .then((m) => setReports(m.ablation_reports))
      .catch((e) => setError(errorText(e)));
  }, []);

  if (error) return <p role="alert" className="text-sm text-danger">{error}</p>;
  if (!reports) return <p className="text-sm text-ink-soft">Loading benchmark reports…</p>;

  const backbones = Object.keys(reports);
  if (backbones.length === 0) {
    return <p className="text-sm text-ink-soft">No ablation reports found yet.</p>;
  }

  const flats = Object.fromEntries(backbones.map((b) => [b, flatten(reports[b])])) as Record<string, Flat>;
  const paths = Array.from(new Set(backbones.flatMap((b) => Object.keys(flats[b])))).sort();

  const groups = new Map<string, string[]>();
  for (const p of paths) {
    const [group] = p.split(".");
    groups.set(group, [...(groups.get(group) ?? []), p]);
  }

  return (
    <div className="flex flex-col gap-10">
      <SectionHeading
        title="Backbone comparison"
        detail="Each bar is scaled to the larger value on its row. Metric direction is shown as reported by the ablation run."
      />

      {Array.from(groups.entries()).map(([group, rows]) => (
        <section key={group} className="flex flex-col gap-4 border-t border-line pt-6">
          <h3 className="text-sm font-semibold">{humanize(group)}</h3>
          <div className="grid gap-x-8 gap-y-5 md:grid-cols-2">
            {rows.map((path) => {
              const leaf = path.slice(group.length + 1) || path;
              const values = backbones.map((b) => flats[b][path]);
              const max = Math.max(...values.filter((v) => v !== undefined), 1e-9);
              return (
                <div key={path} className="flex flex-col gap-2.5">
                  <p className="text-sm text-ink-soft">{humanize(leaf)}</p>
                  {backbones.map((b, i) => {
                    const v = flats[b][path];
                    return (
                      <div key={b} className="flex items-center gap-3">
                        <span className="w-28 shrink-0 truncate text-xs text-ink-soft">
                          {BACKBONE_NAMES[b] ?? humanize(b)}
                        </span>
                        <div className="flex-1">
                          {v === undefined ? (
                            <span className="text-xs text-ink-soft">Not reported</span>
                          ) : (
                            <Meter value={v / max} tone={i === 0 ? "accent" : "muted"} />
                          )}
                        </div>
                        <span className="w-16 shrink-0 text-right text-sm tabular-nums">
                          {v === undefined ? "" : v.toFixed(3)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              );
            })}
          </div>
        </section>
      ))}
    </div>
  );
}
