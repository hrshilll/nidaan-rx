"use client";

import { useEffect, useState } from "react";
import { fetchHealth, type HealthResponse } from "@/lib/api";
import { cx } from "@/lib/format";
import { TriageWorkbench } from "@/components/workbench/TriageWorkbench";
import { Benchmarks } from "@/components/benchmarks/Benchmarks";
import { CaseHistory } from "@/components/cases/CaseHistory";

type Tab = "triage" | "benchmarks" | "cases";
const TABS: { id: Tab; label: string }[] = [
  { id: "triage", label: "Triage" },
  { id: "benchmarks", label: "Benchmarks" },
  { id: "cases", label: "Cases" },
];

export function AppShell() {
  const [tab, setTab] = useState<Tab>("triage");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthFailed, setHealthFailed] = useState(false);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() => setHealthFailed(true));
  }, []);

  return (
    <div className="min-h-dvh">
      <header className="border-b border-line bg-surface">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-4 px-4 py-4 sm:px-6">
          <div className="flex items-center gap-3">
            <div aria-hidden="true" className="grid size-9 place-items-center rounded-lg bg-accent text-sm font-semibold text-white">
              NR
            </div>
            <div>
              <h1 className="text-base font-semibold leading-tight">Nidaan-Rx</h1>
              <p className="text-xs text-ink-soft">Dermatology triage for Indian skin tones</p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <span className="rounded-full border border-line bg-canvas px-3 py-1 text-xs text-ink-soft" aria-live="polite">
              {health
                ? `Model: ${health.active_backbone}`
                : healthFailed
                  ? "Backend unreachable"
                  : "Connecting…"}
            </span>
            <nav aria-label="Sections" role="tablist" className="flex rounded-lg bg-surface-2 p-1">
              {TABS.map((t) => (
                <button
                  key={t.id}
                  role="tab"
                  aria-selected={tab === t.id}
                  onClick={() => setTab(t.id)}
                  className={cx(
                    "min-h-9 rounded-md px-3.5 text-sm transition-[background-color,color,box-shadow] duration-200",
                    tab === t.id ? "bg-surface font-medium text-ink shadow-sm" : "text-ink-soft hover:text-ink",
                  )}
                >
                  {t.label}
                </button>
              ))}
            </nav>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-6 sm:px-6 sm:py-8">
        <p className="mb-6 max-w-3xl text-sm text-ink-soft">
          Nidaan-Rx supports structured triage, trigger identification, and generic medicine awareness. It does not
          replace an in-person examination by a registered dermatologist or physician.
        </p>
        {tab === "triage" && <TriageWorkbench />}
        {tab === "benchmarks" && <Benchmarks />}
        {tab === "cases" && <CaseHistory />}
      </main>
    </div>
  );
}
