import type { ConditionPrediction } from "@/lib/api";
import { cx, delay, pct } from "@/lib/format";

export function PosteriorShift({ matches }: { matches: ConditionPrediction[] }) {
  if (!matches.length) return <p className="text-sm text-ink-soft">No condition matches were returned.</p>;
  return (
    <ul className="flex flex-col gap-6">
      {matches.map((m, i) => {
        const delta = m.posterior_prob - m.prior_prob;
        const up = delta >= 0;
        return (
          <li key={m.condition_id} className="rise flex flex-col gap-2" style={delay(i * 70)}>
            <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
              <span className="text-sm font-medium">{m.display_name}</span>
              <span className="text-xs text-ink-soft">{m.tier1_family.replace(/_/g, " ")}</span>
            </div>
            <Bar label="Before triggers" value={m.prior_prob} tone="muted" />
            <Bar label="After triggers" value={m.posterior_prob} tone="accent" />
            <p className="text-xs tabular-nums text-ink-soft">
              {pct(m.prior_prob)} to <span className="text-ink">{pct(m.posterior_prob)}</span>{" "}
              <span className={cx("font-medium", up ? "text-accent-strong" : "text-warn")}>
                ({up ? "+" : ""}
                {(delta * 100).toFixed(1)} pts)
              </span>
            </p>
          </li>
        );
      })}
    </ul>
  );
}

function Bar({ label, value, tone }: { label: string; value: number; tone: "muted" | "accent" }) {
  const width = Math.max(0, Math.min(100, value * 100));
  return (
    <div className="flex items-center gap-3">
      <span className="w-28 shrink-0 text-xs text-ink-soft">{label}</span>
      <div className="h-2 flex-1 overflow-hidden rounded-full bg-surface-2">
        <div
          className={cx(
            "h-full rounded-full transition-[width] duration-700 ease-(--ease-rubber)",
            tone === "accent" ? "bg-accent" : "bg-line-strong",
          )}
          style={{ width: `${width}%` }}
        />
      </div>
    </div>
  );
}
