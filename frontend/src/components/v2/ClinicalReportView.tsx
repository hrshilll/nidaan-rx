"use client";

import React, { useState } from "react";
import {
  AlertOctagon,
  AlertTriangle,
  ArrowRight,
  BadgeCheck,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Info,
  Download,
  Loader2,
  ShieldAlert,
  ShieldCheck,
  Stethoscope,
} from "lucide-react";
import { downloadCasePdf, type ClinicalAssessmentResponse, type EvaluatedTreatmentCard, type GateEvent } from "@/lib/api";

interface ClinicalReportViewProps {
  assessment: ClinicalAssessmentResponse;
  caseId?: string | null;
  backbone?: string;
  patientAge?: number;
}

function gateBadge(action: GateEvent["action"]): string {
  switch (action) {
    case "HARD_STOP":
    case "BLOCK":
      return "bg-[#FEF2F2] text-[#DC2626] ring-1 ring-[#FECACA]";
    case "MODIFY":
    case "WARN":
      return "bg-[#FFFBEB] text-[#B45309] ring-1 ring-[#FDE68A]";
    default:
      return "bg-[#ECFDF5] text-[#047857] ring-1 ring-[#A7F3D0]";
  }
}

function TreatmentSection({
  title,
  subtitle,
  cards,
  tone,
}: {
  title: string;
  subtitle: string;
  cards: EvaluatedTreatmentCard[];
  tone: "green" | "cyan";
}) {
  if (!cards || cards.length === 0) return null;

  return (
    <section className="clinical-card p-6 sm:p-7 space-y-4">
      <div>
        <h3 className="text-[17px] font-bold tracking-tight text-[#164E63]">{title}</h3>
        <p className="mt-0.5 text-[13px] text-[#475569]">{subtitle}</p>
      </div>

      <div className="space-y-4">
        {cards.map((card) => (
          <div
            key={card.option.option_id}
            className="rounded-2xl bg-[#F3FAFC] p-4 sm:p-5 ring-1 ring-[#CBEBF2]"
          >
            <div className="flex flex-wrap items-start justify-between gap-2">
              <div>
                <h4 className="text-[15px] font-bold text-[#164E63]">
                  {card.option.display_name}
                </h4>
                <p className="mt-0.5 text-xs text-[#475569]">
                  <span className="font-semibold text-[#0E7490]">{card.option.vehicle}</span> ·{" "}
                  {card.option.posology}
                </p>
              </div>
              <span
                className={`rounded-full px-3 py-1 text-[11px] font-semibold ${
                  tone === "green"
                    ? "bg-[#ECFDF5] text-[#047857] ring-1 ring-[#A7F3D0]"
                    : "bg-[#ECFEFF] text-[#0E7490] ring-1 ring-[#A5F3FC]"
                }`}
              >
                {card.regulatory_label}
              </span>
            </div>

            {card.option.notes && (
              <p className="mt-2.5 text-[13px] leading-relaxed text-[#334155]">
                {card.option.notes}
              </p>
            )}

            {card.dosage_note && (
              <div className="mt-3 flex items-start gap-2 rounded-xl bg-[#FFFBEB] px-3.5 py-2.5 text-xs text-[#92400E] ring-1 ring-[#FDE68A]">
                <Info className="mt-0.5 h-4 w-4 shrink-0" />
                <span>{card.dosage_note}</span>
              </div>
            )}

            {card.inline_badges.length > 0 && (
              <div className="mt-3 flex flex-wrap gap-1.5">
                {card.inline_badges.map((b, i) => (
                  <span
                    key={i}
                    className="rounded-full bg-white px-2.5 py-0.5 text-[11px] font-medium text-[#334155] ring-1 ring-[#CBEBF2]"
                  >
                    {b}
                  </span>
                ))}
              </div>
            )}

            {card.pmbjp_matches.length > 0 && (
              <div className="mt-4 rounded-2xl bg-[#ECFDF5] p-4 ring-1 ring-[#A7F3D0]">
                <div className="mb-2.5 flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wider text-[#047857]">
                  <BadgeCheck className="h-4 w-4" />
                  Jan Aushadhi (PMBJP) Verified Equivalents
                </div>
                <div className="grid gap-2.5 sm:grid-cols-2">
                  {card.pmbjp_matches.map((m, idx) => (
                    <div
                      key={`${m.generic_code ?? m.product_name}-${idx}`}
                      className="rounded-xl bg-white p-3 ring-1 ring-[#A7F3D0]"
                    >
                      <div className="flex items-start justify-between gap-2">
                        <span className="text-xs font-semibold text-[#164E63]">
                          {m.product_name}
                        </span>
                        {m.mrp_inr != null && (
                          <span className="shrink-0 font-mono text-sm font-bold text-[#059669] tabular-nums">
                            ₹{m.mrp_inr.toFixed(0)}
                          </span>
                        )}
                      </div>
                      <div className="mt-1.5 flex flex-wrap items-center gap-2 text-[11px] text-[#475569]">
                        {m.generic_code && (
                          <span className="font-mono text-[10px]">#{m.generic_code}</span>
                        )}
                        <span>{m.vehicle}</span>
                        {m.commercial_avg_inr != null && (
                          <span className="line-through text-[#64748B] tabular-nums">
                            ₹{m.commercial_avg_inr.toFixed(0)}
                          </span>
                        )}
                        {m.estimated_savings_pct != null && m.estimated_savings_pct > 0 && (
                          <span className="rounded-full bg-[#059669] px-2 py-0.5 text-[10px] font-semibold text-white tabular-nums">
                            Save {Math.round(m.estimated_savings_pct)}%
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}

export function ClinicalReportView({
  assessment,
  caseId,
  backbone,
  patientAge,
}: ClinicalReportViewProps) {
  const [auditOpen, setAuditOpen] = useState(false);
  const [downloadingPdf, setDownloadingPdf] = useState(false);
  const [pdfError, setPdfError] = useState<string | null>(null);
  const [showMinorMatches, setShowMinorMatches] = useState(false);

  const primaryMatches = assessment.top_matches.filter((m, idx) => idx === 0 || m.posterior_prob >= 0.02);
  const lowProbMatches = assessment.top_matches.filter((m, idx) => idx > 0 && m.posterior_prob < 0.02);
  const visibleMatches = showMinorMatches ? assessment.top_matches : primaryMatches;

  const triggered = assessment.safety_audit_log.filter((g) => g.triggered);
  const blockedCount = triggered.filter(
    (g) => g.action === "BLOCK" || g.action === "HARD_STOP"
  ).length;
  const modifiedCount = triggered.filter(
    (g) => g.action === "MODIFY" || g.action === "WARN"
  ).length;

  const isPediatric = patientAge != null && patientAge < 18;

  return (
    <div className="space-y-6">
      {/* Urgent Hard-Stop Banner */}
      {assessment.escalation_hard_stop && (
        <div className="rounded-[22px] bg-[#FEF2F2] p-5 ring-2 ring-[#DC2626] shadow-xs">
          <div className="flex items-start gap-3.5">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-[#DC2626] text-white">
              <AlertOctagon className="h-5 w-5" />
            </div>
            <div>
              <h3 className="text-[15px] font-bold text-[#991B1B]">
                Urgent In-Person Dermatologist Evaluation Advised
              </h3>
              <p className="mt-1 text-[13px] leading-relaxed text-[#7F1D1D]">
                {assessment.escalation_reason ||
                  "Red-flag clinical signs or high-risk features were detected. Self-treatment is suppressed—please consult a doctor promptly."}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Gate 8 Longitudinal Escalation Banner (Non-Hard-Stop Escalation) */}
      {!assessment.escalation_hard_stop && assessment.escalation_reason && (
        <div className="rounded-[22px] bg-[#FFFBEB] p-5 ring-2 ring-[#D97706] shadow-xs">
          <div className="flex items-start gap-3.5">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-[#D97706] text-white">
              <AlertTriangle className="h-5 w-5" />
            </div>
            <div>
              <h3 className="text-[15px] font-bold text-[#92400E]">
                Longitudinal Escalation Notice (Gate 8)
              </h3>
              <p className="mt-1 text-[13px] leading-relaxed text-[#78350F]">
                {assessment.escalation_reason}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Guardian / Weight Verification Notices */}
      {(assessment.minor_guardian_notice || (assessment.weight_refresh_prompt && isPediatric)) && (
        <div className="grid gap-3 sm:grid-cols-2">
          {assessment.minor_guardian_notice && (
            <div className="rounded-2xl bg-[#FFFBEB] p-4 text-xs text-[#92400E] ring-1 ring-[#FDE68A]">
              <strong className="font-semibold">Guardian Oversight: </strong>
              {assessment.minor_guardian_notice}
            </div>
          )}
          {assessment.weight_refresh_prompt && isPediatric && (
            <div className="rounded-2xl bg-[#FFFBEB] p-4 text-xs text-[#92400E] ring-1 ring-[#FDE68A]">
              <strong className="font-semibold">Pediatric Weight Check: </strong>
              Confirm the weight was measured within the last 90 days before dosing.
            </div>
          )}
        </div>
      )}

      {/* Differential Assessment Card */}
      <section className="clinical-card p-6 sm:p-7">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-[17px] font-bold tracking-tight text-[#164E63]">
              Differential Assessment
            </h3>
            <p className="mt-0.5 text-[13px] text-[#475569]">
              Bayesian posterior probabilities refined with your selected clinical triggers.
            </p>
          </div>
          <div className="flex items-center gap-2">
            {caseId && (
              <span className="rounded-full bg-[#E8F1F6] px-3 py-1 font-mono text-[11px] font-semibold text-[#164E63]">
                {caseId}
              </span>
            )}
            {caseId && (
              <button
                type="button"
                disabled={downloadingPdf}
                onClick={async () => {
                  setPdfError(null);
                  setDownloadingPdf(true);
                  try {
                    await downloadCasePdf(caseId);
                  } catch (err) {
                    setPdfError(err instanceof Error ? err.message : "Could not download PDF.");
                  } finally {
                    setDownloadingPdf(false);
                  }
                }}
                className="no-print inline-flex cursor-pointer items-center gap-1.5 rounded-full bg-[#059669] px-4 py-1.5 text-xs font-semibold text-white shadow-xs transition hover:bg-[#047857] disabled:opacity-60"
              >
                {downloadingPdf ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Download className="h-3.5 w-3.5" />
                )}
                {downloadingPdf ? "Building PDF..." : "Download Summary PDF"}
              </button>
            )}
          </div>
        </div>

        {pdfError && (
          <div className="mt-3 rounded-xl bg-[#FEF2F2] px-3.5 py-2 text-xs font-medium text-[#DC2626] ring-1 ring-[#FECACA]">
            {pdfError}
          </div>
        )}

        <div className="mt-5 space-y-3">
          {visibleMatches.map((m, idx) => {
            const delta = (m.posterior_prob - m.prior_prob) * 100;
            const triggers = (m.contributing_triggers || [])
              .map((t) =>
                typeof t.label === "string"
                  ? t.label
                  : typeof t.trigger_id === "string"
                  ? t.trigger_id.replaceAll("_", " ")
                  : ""
              )
              .filter(Boolean);

            return (
              <div
                key={m.condition_id}
                className={`rounded-2xl p-4 ${
                  idx === 0
                    ? "bg-[#ECFEFF] ring-1 ring-[#67E8F9]"
                    : "bg-[#F3FAFC] ring-1 ring-[#DCECF1]"
                }`}
              >
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-[15px] font-bold text-[#164E63]">
                        {m.display_name}
                      </span>
                      {idx === 0 && (
                        <span className="rounded-full bg-[#0891B2] px-2.5 py-0.5 text-[10px] font-semibold text-white">
                          Top Match
                        </span>
                      )}
                    </div>
                    <p className="mt-0.5 text-xs capitalize text-[#475569]">
                      {m.tier1_family.replaceAll("_", " ")}
                    </p>
                  </div>
                  <span className="font-mono text-lg font-bold text-[#0891B2] tabular-nums">
                    {(m.posterior_prob * 100).toFixed(1)}%
                  </span>
                </div>

                <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-[#DCECF1]">
                  <div
                    className="h-full rounded-full bg-[#0891B2] transition-all duration-300"
                    style={{ width: `${Math.min(100, m.posterior_prob * 100)}%` }}
                  />
                </div>

                <div className="mt-2.5 flex flex-wrap items-center justify-between gap-2 text-xs text-[#475569] tabular-nums">
                  <span className="inline-flex items-center gap-1.5">
                    Stage-1 {(m.prior_prob * 100).toFixed(1)}%
                    <ArrowRight className="h-3 w-3" />
                    Refined {(m.posterior_prob * 100).toFixed(1)}%
                  </span>
                  {Math.abs(delta) >= 0.2 && (
                    <span
                      className={`rounded-full px-2 py-0.5 text-[11px] font-semibold ${
                        delta > 0
                          ? "bg-[#ECFDF5] text-[#047857]"
                          : "bg-[#FFFBEB] text-[#B45309]"
                      }`}
                    >
                      {delta > 0 ? `+${delta.toFixed(1)}%` : `${delta.toFixed(1)}%`}
                    </span>
                  )}
                </div>

                {triggers.length > 0 && (
                  <div className="mt-2.5 flex flex-wrap gap-1.5">
                    {triggers.map((tr, i) => (
                      <span
                        key={i}
                        className="rounded-full bg-white px-2.5 py-0.5 text-[11px] font-medium text-[#164E63] ring-1 ring-[#CBEBF2]"
                      >
                        {tr}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {lowProbMatches.length > 0 && (
          <button
            type="button"
            onClick={() => setShowMinorMatches(!showMinorMatches)}
            className="no-print mt-3 inline-flex cursor-pointer items-center gap-1 text-xs font-semibold text-[#0891B2] hover:underline"
          >
            {showMinorMatches
              ? "Hide low-probability differentials"
              : `Show ${lowProbMatches.length} low-probability differential${lowProbMatches.length > 1 ? "s" : ""} (<2%)`}
          </button>
        )}
      </section>

      {/* Do-Not-Apply Precautions */}
      {assessment.do_not_apply_warnings.length > 0 && (
        <section className="rounded-[22px] bg-[#FEF2F2] p-6 ring-1 ring-[#FECACA]">
          <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-[#DC2626]">
            <ShieldAlert className="h-4 w-4" />
            Do-Not-Apply Precautions
          </div>
          <ul className="mt-3 space-y-2 text-[13px] text-[#7F1D1D]">
            {assessment.do_not_apply_warnings.map((w, i) => (
              <li key={i} className="flex items-start gap-2.5">
                <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-[#DC2626]" />
                <span>{w}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* Treatments */}
      <TreatmentSection
        title="Tier-1 OTC & Supportive Care"
        subtitle="Safety-checked over-the-counter options tailored to site, barrier, and profile."
        cards={assessment.tier1_otc_care}
        tone="green"
      />

      <TreatmentSection
        title="Tier-2 Dermatologist Discussion Options"
        subtitle="Prescription therapies to review with a registered dermatologist—never self-prescribe."
        cards={assessment.tier2_dermatologist_prep}
        tone="cyan"
      />

      {assessment.tier1_otc_care.length === 0 &&
        assessment.tier2_dermatologist_prep.length === 0 && (
          <div className="clinical-card p-8 text-center">
            <Stethoscope className="mx-auto h-7 w-7 text-[#0891B2]" />
            <p className="mt-2.5 text-[15px] font-bold text-[#164E63]">
              In-Person Dermatologist Evaluation Required
            </p>
            <p className="mt-1 text-[13px] text-[#475569]">
              Treatment options were intentionally withheld by the clinical safety gates.
            </p>
          </div>
        )}

      {/* Lifestyle Plan */}
      {assessment.lifestyle_action_plan.length > 0 && (
        <section className="clinical-card p-6 sm:p-7">
          <h3 className="text-[17px] font-bold tracking-tight text-[#164E63]">
            Daily Skin & Environment Plan
          </h3>
          <ul className="mt-3.5 space-y-2.5 text-[13px] text-[#334155]">
            {assessment.lifestyle_action_plan.map((item, i) => (
              <li key={i} className="flex items-start gap-2.5">
                <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-[#059669]" />
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* 12-Gate Safety Audit */}
      <section className="clinical-card overflow-hidden">
        <button
          type="button"
          onClick={() => setAuditOpen(!auditOpen)}
          className="flex w-full cursor-pointer items-center justify-between gap-3 p-5 sm:px-7 text-left transition hover:bg-[#F3FAFC]"
        >
          <div className="flex items-center gap-3.5">
            <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-[#ECFDF5] text-[#059669] ring-1 ring-[#A7F3D0]">
              <ShieldCheck className="h-5 w-5" />
            </div>
            <div>
              <p className="text-[15px] font-bold text-[#164E63]">12-Gate Clinical Safety Audit</p>
              <p className="text-xs text-[#475569]">
                {assessment.safety_audit_log.length} checks · {blockedCount} blocked ·{" "}
                {modifiedCount} modified/warned
              </p>
            </div>
          </div>
          {auditOpen ? (
            <ChevronUp className="h-5 w-5 text-[#64748B]" />
          ) : (
            <ChevronDown className="h-5 w-5 text-[#64748B]" />
          )}
        </button>

        {auditOpen && (
          <div className="divide-y divide-[#E8F1F6] border-t border-[#E8F1F6] px-5 pb-4 sm:px-7">
            {assessment.safety_audit_log.map((g, idx) => (
              <div key={`${g.gate_id}-${idx}`} className="py-3 text-xs">
                <div className="flex items-center justify-between gap-2">
                  <span className="font-semibold text-[#164E63]">
                    {g.gate_id} · {g.gate_name}
                  </span>
                  <span
                    className={`rounded-full px-2.5 py-0.5 text-[10px] font-semibold ${gateBadge(
                      g.action
                    )}`}
                  >
                    {g.action}
                  </span>
                </div>
                <p className="mt-1 leading-relaxed text-[#475569]">{g.message}</p>
              </div>
            ))}
          </div>
        )}
      </section>

      <p className="px-2 text-center text-[11px] leading-relaxed text-[#64748B]">
        {assessment.disclaimer_banner}
        {backbone ? ` · Backbone: ${backbone}.` : ""}
      </p>
    </div>
  );
}
