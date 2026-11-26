"use client";

import { useState } from "react";
import type { GateEvent } from "@/lib/api";
import { cx } from "@/lib/format";

const ACTION_LABEL: Record<GateEvent["action"], string> = {
  PASS: "Passed",
  WARN: "Warning",
  BLOCK: "Blocked",
  MODIFY: "Adjusted",
  HARD_STOP: "Hard stop",
};

export function AuditLog({ events }: { events: GateEvent[] }) {
  const [open, setOpen] = useState(false);
  const triggered = events.filter((e) => e.triggered).length;
  return (
    <section aria-labelledby="audit-heading" className="flex flex-col gap-3">
      <button
        type="button"
        aria-expanded={open}
        aria-controls="audit-body"
        onClick={() => setOpen((v) => !v)}
        className="flex min-h-11 w-full items-center justify-between gap-3 rounded-lg text-left transition-colors duration-200 hover:bg-surface-2 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
      >
        <span id="audit-heading" className="text-sm font-semibold">
          Safety gate audit log
        </span>
        <span className="text-xs text-ink-soft">
          {events.length} entries · {triggered} triggered · {open ? "Hide" : "Show"}
        </span>
      </button>
      <div
        id="audit-body"
        hidden={!open}
        className="rise overflow-x-auto rounded-lg border border-line"
      >
        {events.length === 0 ? (
          <p className="p-4 text-sm text-ink-soft">No gate events were recorded.</p>
        ) : (
          <table className="w-full min-w-[32rem] text-left text-sm">
            <thead className="bg-surface-2 text-xs text-ink-soft">
              <tr>
                <th className="px-4 py-2.5 font-medium">Gate</th>
                <th className="px-4 py-2.5 font-medium">Result</th>
                <th className="px-4 py-2.5 font-medium">Detail</th>
              </tr>
            </thead>
            <tbody>
              {events.map((e, i) => (
                <tr key={`${e.gate_id}-${i}`} className="border-t border-line align-top">
                  <td className="px-4 py-3">
                    <span className="block font-medium">{e.gate_name}</span>
                    <span className="text-xs text-ink-soft">{e.gate_id}</span>
                  </td>
                  <td className="px-4 py-3">
                    <span
                      className={cx(
                        "whitespace-nowrap text-xs font-medium",
                        e.action === "HARD_STOP" || e.action === "BLOCK" ? "text-danger" : e.triggered ? "text-warn" : "text-accent-strong",
                      )}
                    >
                      {ACTION_LABEL[e.action]}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-ink-soft">{e.message}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </section>
  );
}
