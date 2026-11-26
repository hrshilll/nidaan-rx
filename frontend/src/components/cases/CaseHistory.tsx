"use client";

import { useCallback, useEffect, useState } from "react";
import { fetchRecentCases, type CaseSummary } from "@/lib/api";
import { delay, errorText, humanize, pct } from "@/lib/format";
import { Button, SectionHeading } from "@/components/ui/primitives";
import { CaseDrawer } from "./CaseDrawer";

export function CaseHistory() {
  const [cases, setCases] = useState<CaseSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [openId, setOpenId] = useState<string | null>(null);

  const load = useCallback(() => {
    setError(null);
    fetchRecentCases(20)
      .then(setCases)
      .catch((e) => setError(errorText(e)));
  }, []);

  useEffect(() => load(), [load]);

  const close = useCallback(() => setOpenId(null), []);

  return (
    <div className="flex flex-col gap-8">
      <SectionHeading
        title="Saved cases"
        detail="Select a case to open its full report."
        action={
          <Button variant="secondary" onClick={load} className="min-h-10">
            Refresh
          </Button>
        }
      />

      {error && <p role="alert" className="text-sm text-danger">{error}</p>}
      {!cases && !error && <p className="text-sm text-ink-soft">Loading cases…</p>}
      {cases && cases.length === 0 && <p className="text-sm text-ink-soft">No saved cases yet.</p>}

      {cases && cases.length > 0 && (
        <div className="overflow-x-auto rounded-xl border border-line bg-surface">
          <table className="w-full min-w-[40rem] text-left text-sm">
            <thead className="bg-surface-2 text-xs text-ink-soft">
              <tr>
                <th className="px-4 py-3 font-medium">Created</th>
                <th className="px-4 py-3 font-medium">Top match</th>
                <th className="px-4 py-3 font-medium">Confidence</th>
                <th className="px-4 py-3 font-medium">Flags</th>
              </tr>
            </thead>
            <tbody>
              {cases.map((c, i) => (
                <tr
                  key={c.case_id}
                  onClick={() => setOpenId(c.case_id)}
                  className="rise cursor-pointer border-t border-line transition-colors duration-150 hover:bg-surface-2 focus-within:bg-surface-2"
                  style={delay(Math.min(i, 10) * 35)}
                >
                  <td className="px-4 py-3.5 tabular-nums text-ink-soft">
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setOpenId(c.case_id);
                      }}
                      className="min-h-9 text-left focus-visible:outline-2 focus-visible:outline-accent"
                    >
                      {new Date(c.created_at).toLocaleString("en-IN", { dateStyle: "medium", timeStyle: "short" })}
                    </button>
                  </td>
                  <td className="px-4 py-3.5 font-medium">
                    {c.top_condition_name ?? (c.top_condition_id ? humanize(c.top_condition_id) : "None")}
                  </td>
                  <td className="px-4 py-3.5 tabular-nums">{c.top_confidence == null ? "–" : pct(c.top_confidence)}</td>
                  <td className="px-4 py-3.5">
                    <div className="flex flex-wrap gap-1.5 text-xs">
                      {c.escalation_hard_stop && <span className="rounded-full bg-danger-tint px-2 py-0.5 text-danger">Escalated</span>}
                      {c.is_ood && <span className="rounded-full bg-warn-tint px-2 py-0.5 text-warn">Out of range</span>}
                      {!c.escalation_hard_stop && !c.is_ood && <span className="text-ink-soft">None</span>}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {openId && <CaseDrawer caseId={openId} onClose={close} />}
    </div>
  );
}
