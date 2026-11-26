"use client";

import { useMemo, useState } from "react";
import {
  runStage1Triage,
  runStage2Assessment,
  type AssessResponse,
  type CaseContextInput,
  type PatientProfileInput,
  type TriageStageResponse,
} from "@/lib/api";
import { DEFAULT_CASE, DEFAULT_PROFILE } from "@/lib/constants";
import { errorText } from "@/lib/format";
import type { SlotResult } from "@/components/intake/PhotoSlots";
import { ClinicalReport } from "@/components/report/ClinicalReport";
import { Button } from "@/components/ui/primitives";
import { IntakeStep } from "./IntakeStep";
import { ReviewStep } from "./ReviewStep";
import { Stepper, type StepId } from "./Stepper";

export function TriageWorkbench() {
  const [step, setStep] = useState<StepId>("intake");
  const [files, setFiles] = useState<(File | null)[]>([null, null, null]);
  const [qualityGate, setQualityGate] = useState(true);
  const [profile, setProfile] = useState<PatientProfileInput>(DEFAULT_PROFILE);
  const [symptoms, setSymptoms] = useState<CaseContextInput>(DEFAULT_CASE);

  const [triage, setTriage] = useState<TriageStageResponse | null>(null);
  const [slotIndexes, setSlotIndexes] = useState<number[]>([]);
  const [triggers, setTriggers] = useState<string[]>([]);
  const [report, setReport] = useState<AssessResponse | null>(null);

  const [busy, setBusy] = useState<"triage" | "assess" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [stale, setStale] = useState(false);

  // Map Stage-1 per-view outputs back onto their guided slots.
  const slotResults = useMemo<(SlotResult | null)[]>(() => {
    const out: (SlotResult | null)[] = [null, null, null];
    if (!triage) return out;
    slotIndexes.forEach((slot, viewIndex) => {
      const q = triage.quality_reports[viewIndex];
      out[slot] = {
        passed: typeof q?.passed === "boolean" ? q.passed : null,
        weight: triage.view_attention_weights[viewIndex] ?? null,
      };
    });
    return out;
  }, [triage, slotIndexes]);

  const markStale = () => {
    if (triage) setStale(true);
  };

  const setFile = (i: number, f: File | null) => {
    setFiles((prev) => prev.map((x, j) => (j === i ? f : x)));
    markStale();
  };
  const patchProfile = (p: Partial<PatientProfileInput>) => {
    setProfile((prev) => ({ ...prev, ...p }));
    markStale();
  };
  const patchSymptoms = (p: Partial<CaseContextInput>) => {
    setSymptoms((prev) => ({ ...prev, ...p }));
    markStale();
  };
  const toggleGate = (v: boolean) => {
    setQualityGate(v);
    markStale();
  };

  async function analyze() {
    const picked = files.flatMap((f, i) => (f ? [{ f, i }] : []));
    setBusy("triage");
    setError(null);
    try {
      const res = await runStage1Triage({
        files: picked.map((p) => p.f),
        profile,
        caseContext: { ...symptoms, active_triggers: [] },
        enforceQualityGate: qualityGate,
      });
      setTriage(res);
      setSlotIndexes(picked.map((p) => p.i));
      setTriggers([]);
      setReport(null);
      setStale(false);
      setStep("review");
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(null);
    }
  }

  async function assess() {
    if (!triage) return;
    setBusy("assess");
    setError(null);
    try {
      const res = await runStage2Assessment({
        profile,
        caseContext: {
          ...symptoms,
          active_triggers: triggers,
          other_trigger_note: symptoms.other_trigger_note ?? "",
        },
        stage1Output: triage.stage1_output,
        viewCount: slotIndexes.length,
        saveCase: true,
      });
      setReport(res);
      setStep("report");
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(null);
    }
  }

  function startNewCase() {
    setFiles([null, null, null]);
    setProfile(DEFAULT_PROFILE);
    setSymptoms(DEFAULT_CASE);
    setTriage(null);
    setSlotIndexes([]);
    setTriggers([]);
    setReport(null);
    setStale(false);
    setError(null);
    setStep("intake");
  }

  const reachable: Record<StepId, boolean> = {
    intake: true,
    review: triage !== null,
    report: report !== null,
  };

  return (
    <div className="flex flex-col gap-10">
      <Stepper current={step} reachable={reachable} onGo={setStep} />

      <div key={step} className="rise">
        {step === "intake" && (
          <IntakeStep
            files={files}
            onFile={setFile}
            results={slotResults}
            qualityGate={qualityGate}
            onQualityGate={toggleGate}
            profile={profile}
            onProfile={patchProfile}
            symptoms={symptoms}
            onSymptoms={patchSymptoms}
            busy={busy === "triage"}
            error={error}
            onSubmit={analyze}
            rerun={triage !== null}
            stale={stale}
          />
        )}

        {step === "review" && triage && (
          <ReviewStep
            triage={triage}
            selected={triggers}
            onToggle={(id) =>
              setTriggers((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]))
            }
            onBack={() => setStep("intake")}
            onAssess={assess}
            busy={busy === "assess"}
            error={error}
          />
        )}

        {step === "report" && report && (
          <div className="flex flex-col gap-10">
            <ClinicalReport assessment={report.assessment} caseId={report.case_id} />
            <div className="flex flex-col-reverse gap-3 border-t border-line pt-6 sm:flex-row sm:justify-between">
              <Button variant="secondary" onClick={() => setStep("review")}>
                Adjust triggers
              </Button>
              <Button onClick={startNewCase}>Start a new case</Button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
