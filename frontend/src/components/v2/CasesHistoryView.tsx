"use client";

import React, { useState } from "react";
import {
  AlertTriangle,
  CalendarClock,
  CheckCircle2,
  Clock,
  Eye,
  Loader2,
  Minus,
  Plus,
  Trash2,
  TrendingDown,
  TrendingUp,
  X,
} from "lucide-react";
import {
  deleteCase,
  deleteCaseFollowup,
  fetchCase,
  logCaseFollowup,
  type CaseDetail,
  type CaseSummary,
  type FollowUpTrendValue,
} from "@/lib/api";
import { ClinicalReportView } from "./ClinicalReportView";

interface CasesHistoryViewProps {
  cases: CaseSummary[];
  isAuthenticated: boolean;
  onCaseDeleted?: (caseId: string) => void;
  onCasesRefresh?: () => void;
}

const TREND_OPTIONS: Array<{
  value: FollowUpTrendValue;
  label: string;
  hint: string;
  activeClass: string;
}> = [
  {
    value: "improving",
    label: "Improving",
    hint: "Fading, less itch",
    activeClass: "bg-[#059669] text-white",
  },
  {
    value: "unchanged",
    label: "Unchanged",
    hint: "Same size / color",
    activeClass: "bg-[#D97706] text-white",
  },
  {
    value: "worsening",
    label: "Worsening",
    hint: "Spreading or more intense",
    activeClass: "bg-[#DC2626] text-white",
  },
];

function trendBadgeClass(trend: string, escalated: boolean): string {
  if (escalated || trend === "worsening") {
    return "bg-[#FEF2F2] text-[#DC2626] ring-1 ring-[#FECACA]";
  }
  if (trend === "unchanged") {
    return "bg-[#FFFBEB] text-[#B45309] ring-1 ring-[#FDE68A]";
  }
  return "bg-[#ECFDF5] text-[#047857] ring-1 ring-[#A7F3D0]";
}

export function CasesHistoryView({
  cases,
  isAuthenticated,
  onCaseDeleted,
  onCasesRefresh,
}: CasesHistoryViewProps) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<CaseDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Follow-up form state
  const [formOpen, setFormOpen] = useState(false);
  const [fuDay, setFuDay] = useState<number>(7);
  const [fuTrend, setFuTrend] = useState<FollowUpTrendValue>("improving");
  const [fuItch, setFuItch] = useState<number>(3);
  const [fuPain, setFuPain] = useState<number>(0);
  const [fuNotes, setFuNotes] = useState("");
  const [submittingFu, setSubmittingFu] = useState(false);
  const [deletingFuId, setDeletingFuId] = useState<string | null>(null);
  const [fuError, setFuError] = useState<string | null>(null);

  const openCase = async (id: string) => {
    setSelectedId(id);
    setDetail(null);
    setError(null);
    setFormOpen(false);
    setFuError(null);
    setLoading(true);
    try {
      const data = await fetchCase(id);
      setDetail(data);
      const maxDay = (data.followups || []).reduce((m, f) => Math.max(m, f.followup_day), 0);
      const lastFu = data.followups?.[data.followups.length - 1];
      const nextDay = maxDay > 0 ? maxDay + 7 : 7;
      setFuDay(nextDay);
      setFuItch(lastFu?.itch_score ?? data.case_context?.itch_score ?? 3);
      setFuPain(lastFu?.pain_burning_score ?? data.case_context?.pain_burning_score ?? 0);
      setFuTrend("improving");
      setFuNotes("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load case.");
    } finally {
      setLoading(false);
    }
  };

  const handleLogFollowup = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedId || !detail) return;
    setFuError(null);
    setSubmittingFu(true);
    try {
      const res = await logCaseFollowup(selectedId, {
        followup_day: Math.max(1, Number(fuDay) || 1),
        followup_trend: fuTrend,
        itch_score: fuItch,
        pain_burning_score: fuPain,
        notes: fuNotes.trim(),
      });
      setDetail({
        ...detail,
        case_context: res.case_context,
        assessment: res.assessment,
        followups: res.followups,
      });
      setFuNotes("");
      setFormOpen(false);
      const newMax = res.followups.reduce((m, f) => Math.max(m, f.followup_day), 0);
      setFuDay(newMax + 7);
      onCasesRefresh?.();
    } catch (err) {
      setFuError(err instanceof Error ? err.message : "Could not log follow-up.");
    } finally {
      setSubmittingFu(false);
    }
  };

  const handleDeleteFollowup = async (followupId: string) => {
    if (!selectedId || !detail) return;
    setDeletingFuId(followupId);
    setFuError(null);
    try {
      const res = await deleteCaseFollowup(selectedId, followupId);
      setDetail({
        ...detail,
        case_context: res.case_context,
        assessment: res.assessment,
        followups: res.followups,
      });
      const newMax = res.followups.reduce((m, f) => Math.max(m, f.followup_day), 0);
      setFuDay(newMax > 0 ? newMax + 7 : 7);
      onCasesRefresh?.();
    } catch (err) {
      setFuError(err instanceof Error ? err.message : "Could not delete follow-up.");
    } finally {
      setDeletingFuId(null);
    }
  };

  const maxLoggedDay = (detail?.followups || []).reduce((m, f) => Math.max(m, f.followup_day), 0);
  const minNextDay = maxLoggedDay + 1;
  const dayInvalid = fuDay < minNextDay;

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (!isAuthenticated) return;
    setDeletingId(id);
    try {
      await deleteCase(id);
      if (selectedId === id) setSelectedId(null);
      onCaseDeleted?.(id);
    } catch {
      // Ignore silent failure
    } finally {
      setDeletingId(null);
    }
  };

  const willEscalatePreview =
    (fuDay >= 7 && fuTrend === "worsening") ||
    (fuDay >= 14 && (fuTrend === "unchanged" || fuTrend === "worsening"));

  return (
    <>
      <div className="space-y-5">
        <div>
          <h1 className="text-[28px] sm:text-[32px] font-bold tracking-tight text-[#164E63]">
            Your Private Cases & Follow-Ups
          </h1>
          <p className="mt-1 text-sm text-[#475569]">
            Open any saved case to log Day 7 / Day 14 recovery check-ins and run Gate 8 longitudinal escalation checks.
          </p>
        </div>

        <section className="clinical-card p-5 sm:p-6">
          {cases.length === 0 ? (
            <div className="py-12 text-center">
              <Clock className="mx-auto h-7 w-7 text-[#64748B]" />
              <p className="mt-2.5 text-sm font-semibold text-[#164E63]">
                No triage assessments saved yet
              </p>
              <p className="mt-1 text-xs text-[#475569]">
                Complete a clinical assessment to store its report, safety audit log, and follow-up timeline here.
              </p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[660px] border-collapse text-left text-xs">
                <thead>
                  <tr className="border-b border-[#E8F1F6] text-[#475569]">
                    <th className="py-2.5 pr-4 font-semibold">Case ID</th>
                    <th className="px-3 py-2.5 font-semibold">Top Condition</th>
                    <th className="px-3 py-2.5 font-semibold">Confidence</th>
                    <th className="px-3 py-2.5 font-semibold">Follow-Up (Gate 8)</th>
                    <th className="px-3 py-2.5 font-semibold">Status</th>
                    <th className="py-2.5 pl-3 text-right font-semibold">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#E8F1F6]">
                  {cases.map((c) => (
                    <tr
                      key={c.case_id}
                      onClick={() => openCase(c.case_id)}
                      className="cursor-pointer transition hover:bg-[#F3FAFC]"
                    >
                      <td className="py-3.5 pr-4 font-mono font-semibold text-[#164E63]">
                        {c.case_id}
                      </td>
                      <td className="px-3 py-3.5 font-semibold text-[#164E63]">
                        {c.top_condition_name || c.top_condition_id || "—"}
                      </td>
                      <td className="px-3 py-3.5 font-mono font-bold text-[#0891B2] tabular-nums">
                        {c.top_confidence != null ? `${(c.top_confidence * 100).toFixed(1)}%` : "—"}
                      </td>
                      <td className="px-3 py-3.5">
                        {c.followup_count && c.followup_count > 0 ? (
                          <span
                            className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-[10px] font-semibold capitalize ${trendBadgeClass(
                              c.latest_followup_trend || "improving",
                              Boolean(c.longitudinal_escalated)
                            )}`}
                          >
                            Day {c.latest_followup_day} · {c.latest_followup_trend}
                            {c.longitudinal_escalated ? " (G8)" : ""}
                          </span>
                        ) : (
                          <span className="text-[11px] text-[#64748B]">Baseline</span>
                        )}
                      </td>
                      <td className="px-3 py-3.5">
                        {c.escalation_hard_stop ? (
                          <span className="rounded-full bg-[#FEF2F2] px-2.5 py-0.5 text-[10px] font-semibold text-[#DC2626]">
                            Hard Stop
                          </span>
                        ) : c.longitudinal_escalated ? (
                          <span className="rounded-full bg-[#FFFBEB] px-2.5 py-0.5 text-[10px] font-semibold text-[#B45309]">
                            Escalated (G8)
                          </span>
                        ) : c.is_ood ? (
                          <span className="rounded-full bg-[#FFFBEB] px-2.5 py-0.5 text-[10px] font-semibold text-[#B45309]">
                            OOD Flag
                          </span>
                        ) : (
                          <span className="rounded-full bg-[#ECFDF5] px-2.5 py-0.5 text-[10px] font-semibold text-[#047857]">
                            Complete
                          </span>
                        )}
                      </td>
                      <td className="py-3.5 pl-3 text-right">
                        <div className="inline-flex items-center gap-3">
                          <span className="inline-flex items-center gap-1 font-semibold text-[#0891B2]">
                            <Eye className="h-3.5 w-3.5" /> Open
                          </span>
                          {isAuthenticated && (
                            <button
                              type="button"
                              disabled={deletingId === c.case_id}
                              onClick={(e) => handleDelete(e, c.case_id)}
                              aria-label={`Delete ${c.case_id}`}
                              className="cursor-pointer rounded-lg p-1 text-[#64748B] transition hover:bg-[#FEF2F2] hover:text-[#DC2626]"
                            >
                              {deletingId === c.case_id ? (
                                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                              ) : (
                                <Trash2 className="h-3.5 w-3.5" />
                              )}
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>

      {/* Case Detail + Longitudinal Follow-Up Drawer */}
      {selectedId && (
        <div className="fixed inset-0 z-50 flex justify-end bg-[#0F172A]/40 backdrop-blur-[2px]">
          <div className="flex h-full w-full max-w-2xl flex-col bg-[#F3FAFC] shadow-2xl">
            <div className="flex items-center justify-between border-b border-[#CBEBF2] bg-white px-5 py-4">
              <div>
                <p className="font-mono text-sm font-bold text-[#164E63]">{selectedId}</p>
                <p className="text-xs text-[#475569]">Clinical triage record & Gate 8 follow-up timeline</p>
              </div>
              <button
                type="button"
                onClick={() => setSelectedId(null)}
                aria-label="Close drawer"
                className="flex h-9 w-9 cursor-pointer items-center justify-center rounded-full bg-[#E8F1F6] text-[#475569] hover:text-[#164E63]"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-5 space-y-5">
              {loading && (
                <div className="flex items-center justify-center py-16 text-xs text-[#475569]">
                  <Loader2 className="mr-2 h-4 w-4 animate-spin text-[#0891B2]" />
                  Loading case details...
                </div>
              )}
              {error && (
                <div className="rounded-xl bg-[#FEF2F2] p-4 text-xs text-[#DC2626] ring-1 ring-[#FECACA]">
                  {error}
                </div>
              )}

              {detail && (
                <>
                  {/* Longitudinal Follow-Up Tracker (Gate 8) */}
                  <section className="clinical-card p-5 space-y-4">
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <div className="flex items-center gap-3">
                        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-[#ECFEFF] text-[#0891B2] ring-1 ring-[#A5F3FC]">
                          <CalendarClock className="h-5 w-5" />
                        </div>
                        <div>
                          <h3 className="text-sm font-bold text-[#164E63]">
                            Longitudinal Follow-Up Tracker (Gate 8)
                          </h3>
                          <p className="text-xs text-[#475569]">
                            Escalates automatically if worsening at Day 7+ or unchanged at Day 14+.
                          </p>
                        </div>
                      </div>

                      <button
                        type="button"
                        onClick={() => {
                          setFormOpen((v) => !v);
                          setFuError(null);
                        }}
                        className="inline-flex cursor-pointer items-center gap-1.5 rounded-full bg-[#0891B2] px-3.5 py-1.5 text-xs font-semibold text-white shadow-xs transition hover:bg-[#0E7490]"
                      >
                        {formOpen ? <X className="h-3.5 w-3.5" /> : <Plus className="h-3.5 w-3.5" />}
                        {formOpen ? "Cancel" : "Log Follow-Up"}
                      </button>
                    </div>

                    {/* Timeline */}
                    <div className="space-y-2.5 border-l-2 border-[#CBEBF2] pl-4">
                      <div className="relative rounded-xl bg-[#F3FAFC] px-3.5 py-2.5 ring-1 ring-[#DCECF1]">
                        <span className="absolute -left-[22px] top-3.5 h-2.5 w-2.5 rounded-full bg-[#0891B2] ring-4 ring-white" />
                        <div className="flex items-center justify-between gap-2 text-xs">
                          <span className="font-bold text-[#164E63]">Day 0 · Initial Triage</span>
                          <span className="font-mono text-[11px] text-[#475569]">
                            Itch {detail.case_context?.itch_score ?? 0}/10 · Pain{" "}
                            {detail.case_context?.pain_burning_score ?? 0}/10
                          </span>
                        </div>
                      </div>

                      {(detail.followups || []).map((fu) => (
                        <div
                          key={fu.followup_id}
                          className={`relative rounded-xl px-3.5 py-2.5 ring-1 ${
                            fu.escalated
                              ? "bg-[#FFFBEB] ring-[#FDE68A]"
                              : "bg-white ring-[#DCECF1]"
                          }`}
                        >
                          <span
                            className={`absolute -left-[22px] top-3.5 h-2.5 w-2.5 rounded-full ring-4 ring-white ${
                              fu.escalated
                                ? "bg-[#D97706]"
                                : fu.followup_trend === "improving"
                                ? "bg-[#059669]"
                                : "bg-[#0891B2]"
                            }`}
                          />
                          <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                            <div className="flex items-center gap-2">
                              <span className="font-bold text-[#164E63]">
                                Day {fu.followup_day} Check-In
                              </span>
                              <span
                                className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold capitalize ${trendBadgeClass(
                                  fu.followup_trend,
                                  fu.escalated
                                )}`}
                              >
                                {fu.followup_trend === "improving" && (
                                  <TrendingUp className="h-3 w-3" />
                                )}
                                {fu.followup_trend === "worsening" && (
                                  <TrendingDown className="h-3 w-3" />
                                )}
                                {fu.followup_trend === "unchanged" && (
                                  <Minus className="h-3 w-3" />
                                )}
                                {fu.followup_trend}
                              </span>
                              {fu.escalated && (
                                <span className="inline-flex items-center gap-1 rounded-full bg-[#D97706] px-2 py-0.5 text-[10px] font-semibold text-white">
                                  <AlertTriangle className="h-3 w-3" /> Gate 8 Escalated
                                </span>
                              )}
                            </div>
                            <div className="flex items-center gap-2">
                              <span className="font-mono text-[11px] text-[#475569]">
                                {fu.itch_score != null ? `Itch ${fu.itch_score}/10` : ""}
                                {fu.itch_score != null && fu.pain_burning_score != null ? " · " : ""}
                                {fu.pain_burning_score != null ? `Pain ${fu.pain_burning_score}/10` : ""}
                              </span>
                              <button
                                type="button"
                                disabled={deletingFuId === fu.followup_id}
                                onClick={() => handleDeleteFollowup(fu.followup_id)}
                                title={`Delete Day ${fu.followup_day} check-in`}
                                aria-label={`Delete Day ${fu.followup_day} check-in`}
                                className="cursor-pointer rounded-md p-1 text-[#64748B] transition hover:bg-[#FEF2F2] hover:text-[#DC2626]"
                              >
                                {deletingFuId === fu.followup_id ? (
                                  <Loader2 className="h-3 w-3 animate-spin" />
                                ) : (
                                  <Trash2 className="h-3 w-3" />
                                )}
                              </button>
                            </div>
                          </div>
                          {fu.notes && (
                            <p className="mt-1 text-xs text-[#475569]">{fu.notes}</p>
                          )}
                        </div>
                      ))}
                    </div>

                    {/* Log Follow-Up Form */}
                    {formOpen && (
                      <form
                        onSubmit={handleLogFollowup}
                        className="space-y-4 rounded-2xl bg-[#F3FAFC] p-4 ring-1 ring-[#CBEBF2]"
                      >
                        {fuError && (
                          <div className="rounded-xl bg-[#FEF2F2] px-3 py-2 text-xs font-medium text-[#DC2626] ring-1 ring-[#FECACA]">
                            {fuError}
                          </div>
                        )}

                        <div className="grid gap-3 sm:grid-cols-12">
                          <div className="sm:col-span-4">
                            <label className="block text-xs font-semibold text-[#164E63]">
                              Follow-Up Day
                            </label>
                            <input
                              type="number"
                              min={minNextDay}
                              max={365}
                              required
                              value={fuDay}
                              onChange={(e) => setFuDay(Number(e.target.value) || minNextDay)}
                              className={`mt-1.5 h-10 w-full rounded-xl bg-white px-3 font-mono text-sm font-semibold text-[#164E63] ring-1 focus:ring-2 focus:outline-none ${
                                dayInvalid
                                  ? "ring-[#DC2626] focus:ring-[#DC2626]"
                                  : "ring-[#CBEBF2] focus:ring-[#0891B2]"
                              }`}
                            />
                            {maxLoggedDay > 0 && (
                              <p className={`mt-1 text-[11px] ${dayInvalid ? "font-semibold text-[#DC2626]" : "text-[#475569]"}`}>
                                Must be Day {minNextDay} or later
                              </p>
                            )}
                          </div>

                          <div className="sm:col-span-8">
                            <label className="block text-xs font-semibold text-[#164E63]">
                              Clinical Progress Trend
                            </label>
                            <div className="mt-1.5 grid grid-cols-3 gap-1.5">
                              {TREND_OPTIONS.map((opt) => (
                                <button
                                  key={opt.value}
                                  type="button"
                                  onClick={() => setFuTrend(opt.value)}
                                  className={`cursor-pointer rounded-xl px-2.5 py-2 text-center text-xs font-semibold transition ${
                                    fuTrend === opt.value
                                      ? opt.activeClass
                                      : "bg-white text-[#164E63] ring-1 ring-[#CBEBF2] hover:bg-[#E8F1F6]"
                                  }`}
                                >
                                  {opt.label}
                                </button>
                              ))}
                            </div>
                          </div>
                        </div>

                        <div className="grid gap-4 sm:grid-cols-2">
                          <div>
                            <div className="flex justify-between text-xs">
                              <span className="font-semibold text-[#475569]">Current Itch</span>
                              <span className="font-mono font-bold text-[#0891B2]">
                                {fuItch}/10
                              </span>
                            </div>
                            <input
                              type="range"
                              min={0}
                              max={10}
                              value={fuItch}
                              onChange={(e) => setFuItch(Number(e.target.value))}
                              className="clinical-slider mt-2.5 w-full"
                            />
                          </div>

                          <div>
                            <div className="flex justify-between text-xs">
                              <span className="font-semibold text-[#475569]">
                                Current Pain / Burning
                              </span>
                              <span className="font-mono font-bold text-[#B45309]">
                                {fuPain}/10
                              </span>
                            </div>
                            <input
                              type="range"
                              min={0}
                              max={10}
                              value={fuPain}
                              onChange={(e) => setFuPain(Number(e.target.value))}
                              className="clinical-slider mt-2.5 w-full"
                            />
                          </div>
                        </div>

                        <div>
                          <label className="block text-xs font-semibold text-[#164E63]">
                            Progress Notes (Optional)
                          </label>
                          <input
                            type="text"
                            maxLength={500}
                            value={fuNotes}
                            onChange={(e) => setFuNotes(e.target.value)}
                            placeholder="e.g., Border still expanding after 7 days of OTC cream..."
                            className="mt-1.5 h-10 w-full rounded-xl bg-white px-3.5 text-xs text-[#164E63] ring-1 ring-[#CBEBF2] placeholder:text-[#64748B] focus:ring-2 focus:ring-[#0891B2] focus:outline-none"
                          />
                        </div>

                        {willEscalatePreview && (
                          <div className="flex items-start gap-2 rounded-xl bg-[#FFFBEB] px-3.5 py-2.5 text-xs text-[#92400E] ring-1 ring-[#FDE68A]">
                            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-[#D97706]" />
                            <span>
                              Gate 8 will trigger an in-person dermatologist escalation notice for Day {fuDay} ({fuTrend}).
                            </span>
                          </div>
                        )}

                        <div className="flex justify-end">
                          <button
                            type="submit"
                            disabled={submittingFu || dayInvalid}
                            className="inline-flex h-10 cursor-pointer items-center gap-2 rounded-full bg-[#059669] px-5 text-xs font-semibold text-white shadow-xs transition hover:bg-[#047857] disabled:opacity-60"
                          >
                            {submittingFu ? (
                              <Loader2 className="h-4 w-4 animate-spin" />
                            ) : (
                              <CheckCircle2 className="h-4 w-4" />
                            )}
                            {submittingFu ? "Running Gate 8..." : "Save & Run Gate 8 Check"}
                          </button>
                        </div>
                      </form>
                    )}
                  </section>

                  <ClinicalReportView
                    assessment={detail.assessment}
                    caseId={selectedId}
                    patientAge={detail.profile?.age}
                  />
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
