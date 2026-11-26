import type { ClinicalAssessmentResponse } from "@/lib/api";
import { HardStopBanner } from "./HardStopBanner";
import { PosteriorShift } from "./PosteriorShift";
import { TreatmentCards } from "./TreatmentCards";
import { AuditLog } from "./AuditLog";
import { humanize } from "@/lib/format";

export function ClinicalReport({ assessment, caseId }: { assessment: ClinicalAssessmentResponse; caseId?: string | null }) {
  const a = assessment;
  return (
    <article aria-label="Clinical report" className="flex flex-col gap-12">
      <header className="flex flex-col gap-2">
        <h2 className="text-lg font-semibold tracking-tight">Clinical triage report</h2>
        {caseId && <p className="text-xs tabular-nums text-ink-soft">Case {caseId}</p>}
        <p className="text-sm text-ink-soft">{a.disclaimer_banner}</p>
      </header>

      {a.escalation_hard_stop && <HardStopBanner reason={a.escalation_reason} />}

      {a.minor_guardian_notice && (
        <p className="rounded-lg bg-warn-tint p-4 text-sm text-warn">{a.minor_guardian_notice}</p>
      )}
      {a.weight_refresh_prompt && (
        <p className="text-sm text-ink-soft">Your weight has not been confirmed in 90 days. Update it before dosing decisions.</p>
      )}

      <section aria-labelledby="likely-heading" className="flex flex-col gap-5">
        <h3 id="likely-heading" className="text-sm font-semibold">
          Likely conditions and how triggers shifted them
        </h3>
        <PosteriorShift matches={a.top_matches} />
        {a.identified_triggers.length > 0 && (
          <p className="text-sm text-ink-soft">
            Triggers used: <span className="text-ink">{a.identified_triggers.map(humanize).join(", ")}</span>
          </p>
        )}
      </section>

      {a.do_not_apply_warnings.length > 0 && (
        <section aria-labelledby="dna-heading" className="flex flex-col gap-3">
          <h3 id="dna-heading" className="text-sm font-semibold">
            Do not apply
          </h3>
          <ul className="flex list-disc flex-col gap-2 pl-5 text-sm text-ink">
            {a.do_not_apply_warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </section>
      )}

      <section aria-labelledby="tier1-heading" className="flex flex-col gap-4">
        <div>
          <h3 id="tier1-heading" className="text-sm font-semibold">
            Over the counter options
          </h3>
          <p className="mt-1 text-sm text-ink-soft">Lower-risk starting points. Confirm with a pharmacist before use.</p>
        </div>
        <TreatmentCards cards={a.tier1_otc_care} emptyText="No over the counter option fits this profile." />
      </section>

      <section aria-labelledby="tier2-heading" className="flex flex-col gap-4">
        <div>
          <h3 id="tier2-heading" className="text-sm font-semibold">
            To discuss with a dermatologist
          </h3>
          <p className="mt-1 text-sm text-ink-soft">Prescription-only options. Bring this list to the appointment.</p>
        </div>
        <TreatmentCards cards={a.tier2_dermatologist_prep} emptyText="No prescription options to discuss." />
      </section>

      {a.lifestyle_action_plan.length > 0 && (
        <section aria-labelledby="life-heading" className="flex flex-col gap-3">
          <h3 id="life-heading" className="text-sm font-semibold">
            Daily habits that help
          </h3>
          <ul className="flex list-disc flex-col gap-2 pl-5 text-sm text-ink">
            {a.lifestyle_action_plan.map((x, i) => (
              <li key={i}>{x}</li>
            ))}
          </ul>
        </section>
      )}

      <AuditLog events={a.safety_audit_log} />

      <p className="border-t border-line pt-6 text-sm text-ink">
        {a.escalation_hard_stop
          ? "Take this report to a doctor now. Do not begin treatment on your own."
          : "Share this report with a dermatologist before starting any prescription treatment."}
      </p>
    </article>
  );
}
