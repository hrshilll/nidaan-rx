"use client";

import { useEffect, useRef, useState } from "react";
import { fetchCase, type CaseDetail } from "@/lib/api";
import { errorText } from "@/lib/format";
import { Button } from "@/components/ui/primitives";
import { ClinicalReport } from "@/components/report/ClinicalReport";

export function CaseDrawer({ caseId, onClose }: { caseId: string; onClose: () => void }) {
  const [detail, setDetail] = useState<CaseDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    let live = true;
    fetchCase(caseId)
      .then((d) => live && setDetail(d))
      .catch((e) => live && setError(errorText(e)));
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => {
      live = false;
      window.removeEventListener("keydown", onKey);
    };
  }, [caseId, onClose]);

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <button type="button" aria-label="Close case" onClick={onClose} className="absolute inset-0 bg-ink/30" />
      <aside
        role="dialog"
        aria-modal="true"
        aria-labelledby="case-drawer-title"
        className="rise relative flex h-full w-full max-w-2xl flex-col overflow-y-auto bg-surface shadow-xl"
      >
        <div className="sticky top-0 flex items-center justify-between gap-4 border-b border-line bg-surface px-6 py-4">
          <div>
            <h2 id="case-drawer-title" className="text-base font-semibold">
              Saved case
            </h2>
            <p className="text-xs tabular-nums text-ink-soft">{caseId}</p>
          </div>
          <Button ref={closeRef} variant="secondary" onClick={onClose} className="min-h-10">
            Close
          </Button>
        </div>

        <div className="flex flex-col gap-8 p-6">
          {error && <p role="alert" className="text-sm text-danger">{error}</p>}
          {!detail && !error && <p className="text-sm text-ink-soft">Loading case…</p>}
          {detail && (
            <>
              <dl className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm sm:grid-cols-4">
                <Item label="Age" value={String(detail.profile.age)} />
                <Item label="Sex" value={detail.profile.biological_sex} />
                <Item label="Weight" value={`${detail.profile.weight_kg} kg`} />
                <Item label="Monk tone" value={String(detail.profile.monk_skin_tone)} />
                <Item label="Fitzpatrick" value={String(detail.profile.fitzpatrick_tone)} />
                <Item label="Site" value={detail.case_context.anatomical_site.replace(/_/g, " ")} />
                <Item label="Duration" value={`${detail.case_context.duration_days} days`} />
                <Item label="Out of range" value={detail.inference.is_ood ? "Yes" : "No"} />
              </dl>
              <ClinicalReport assessment={detail.assessment} caseId={detail.case_id ?? caseId} />
            </>
          )}
        </div>
      </aside>
    </div>
  );
}

function Item({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col">
      <dt className="text-xs text-ink-soft">{label}</dt>
      <dd className="font-medium capitalize">{value}</dd>
    </div>
  );
}
