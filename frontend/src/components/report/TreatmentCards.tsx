import type { EvaluatedTreatmentCard, PMBJPGenericMatch } from "@/lib/api";
import { delay, inr } from "@/lib/format";

export function TreatmentCards({
  cards,
  emptyText,
}: {
  cards: EvaluatedTreatmentCard[];
  emptyText: string;
}) {
  if (!cards.length) return <p className="text-sm text-ink-soft">{emptyText}</p>;
  return (
    <ul className="flex flex-col divide-y divide-line">
      {cards.map((c, i) => (
        <li key={c.option.option_id} className="rise flex flex-col gap-4 py-6 first:pt-0 last:pb-0" style={delay(i * 60)}>
          <div className="flex flex-col gap-2">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <h4 className="text-sm font-semibold">{c.option.display_name}</h4>
              <span className="text-xs text-ink-soft">{c.regulatory_label}</span>
            </div>
            <p className="text-sm text-ink-soft">{c.option.posology}</p>
            {c.dosage_note && <p className="text-sm text-ink">{c.dosage_note}</p>}
            {c.inline_badges.length > 0 && (
              <p className="text-xs text-ink-soft">{c.inline_badges.join(" · ")}</p>
            )}
          </div>
          {c.pmbjp_matches.length > 0 ? (
            <div className="flex flex-col gap-3">
              {c.pmbjp_matches.map((m) => (
                <PriceCompare key={`${m.product_name}-${m.generic_code ?? ""}`} match={m} />
              ))}
            </div>
          ) : (
            <p className="text-xs text-ink-soft">No Jan Aushadhi generic is listed for this option.</p>
          )}
        </li>
      ))}
    </ul>
  );
}

function PriceCompare({ match }: { match: PMBJPGenericMatch }) {
  const hasSavings = match.estimated_savings_pct != null && match.estimated_savings_pct > 0;
  return (
    <div className="grid gap-4 rounded-lg bg-surface-2 p-4 sm:grid-cols-[minmax(0,1fr)_auto_auto] sm:items-center">
      <div className="min-w-0">
        <p className="text-sm font-medium">{match.product_name}</p>
        <p className="text-xs text-ink-soft">
          {match.generic_code ? `PMBJP code ${match.generic_code}` : "PMBJP generic"} · {match.vehicle}
        </p>
        <p className="mt-1 text-xs text-ink-soft">
          {match.salts.map((s) => `${s.salt}${s.strength && s.strength !== "N/A" ? ` ${s.strength}` : ""}`).join(", ")}
        </p>
      </div>
      <div className="flex gap-6 sm:contents">
        <Price label="Commercial average" value={match.commercial_avg_inr} muted />
        <Price label="Jan Aushadhi MRP" value={match.mrp_inr} />
      </div>
      {hasSavings && (
        <span className="w-fit rounded-full bg-accent-tint px-3 py-1 text-xs font-medium text-accent-strong">
          About {Math.round(match.estimated_savings_pct ?? 0)}% lower
        </span>
      )}
    </div>
  );
}

function Price({ label, value, muted }: { label: string; value: number | null; muted?: boolean }) {
  return (
    <div className="flex flex-col">
      <span className="text-xs text-ink-soft">{label}</span>
      <span className={muted ? "text-sm tabular-nums text-ink-soft line-through decoration-ink-soft/50" : "text-base font-semibold tabular-nums"}>
        {value == null ? "Not listed" : inr(value)}
      </span>
    </div>
  );
}
