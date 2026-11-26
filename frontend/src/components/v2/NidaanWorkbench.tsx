"use client";

import React, { useCallback, useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  ArrowRight,
  Check,
  ChevronDown,
  ChevronUp,
  HeartPulse,
  Loader2,
  LogOut,
  MapPin,
  RefreshCw,
  RotateCcw,
  Save,
  ShieldAlert,
  SlidersHorizontal,
  User,
  UserCheck,
  WifiOff,
} from "lucide-react";
import {
  type AuthResponse,
  type AuthUser,
  type CaseContextInput,
  type CaseSummary,
  type ClinicalAssessmentResponse,
  type HealthResponse,
  type MetricsResponse,
  type PatientProfileInput,
  type SleepPattern,
  type StressLevel,
  type TriageStageResponse,
  clearAuthToken,
  fetchHealth,
  fetchMe,
  fetchMetrics,
  fetchRecentCases,
  getAuthToken,
  logoutUser,
  runStage1Triage,
  runStage2Assessment,
  saveUserProfile,
} from "@/lib/api";
import {
  ANATOMICAL_SITES,
  CLIMATE_TAGS,
  DEFAULT_CASE,
  DEFAULT_PROFILE,
  PRESETS,
  RED_FLAGS,
} from "@/lib/constants";
import { AuthScreen } from "./AuthScreen";
import { BenchmarksView } from "./BenchmarksView";
import { CasesHistoryView } from "./CasesHistoryView";
import { ClinicalReportView } from "./ClinicalReportView";
import { PhotoUploader } from "./PhotoUploader";
import { ProfileSettingsView } from "./ProfileSettingsView";
import { RxScannerView } from "./RxScannerView";
import { SkinTonePicker } from "./SkinTonePicker";
import { TagChipField } from "./TagChipField";
import { ThemeToggle } from "./ThemeToggle";

type TopTab = "triage" | "rx" | "profile" | "benchmarks" | "cases";
type TriageStep = 1 | 2 | 3;

const SLEEP_OPTIONS: { value: SleepPattern; label: string }[] = [
  { value: "regular", label: "Regular" },
  { value: "late_sleeper", label: "Late night" },
  { value: "night_shift", label: "Night shift" },
  { value: "irregular", label: "Irregular" },
];

function ClinicalSwitch({
  checked,
  onChange,
  label,
  accent = "cyan",
}: {
  checked: boolean;
  onChange: (next: boolean) => void;
  label: string;
  accent?: "cyan" | "green" | "red";
}) {
  const onBg =
    accent === "red"
      ? "bg-[#DC2626]"
      : accent === "green"
      ? "bg-[#059669]"
      : "bg-[#0891B2]";
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      onClick={() => onChange(!checked)}
      className={`relative inline-flex h-[28px] w-[48px] shrink-0 cursor-pointer items-center rounded-full transition-colors duration-200 ${
        checked ? onBg : "bg-[#CBD5E1]"
      }`}
    >
      <span
        className={`inline-block h-[22px] w-[22px] transform rounded-full bg-white shadow-sm transition-transform duration-200 ${
          checked ? "translate-x-[23px]" : "translate-x-[3px]"
        }`}
      />
    </button>
  );
}

export function NidaanWorkbench() {
  const [tab, setTab] = useState<TopTab>("triage");
  const [step, setStep] = useState<TriageStep>(1);

  // User account state
  const [user, setUser] = useState<AuthUser | null>(null);
  const [authChecked, setAuthChecked] = useState(false);
  const [weightUpdatedAt, setWeightUpdatedAt] = useState<string | null>(null);
  const [profileCompleted, setProfileCompleted] = useState<boolean>(true);
  const [profileDirtyForSync, setProfileDirtyForSync] = useState(false);
  const [syncingProfile, setSyncingProfile] = useState(false);

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [backendOffline, setBackendOffline] = useState(false);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [cases, setCases] = useState<CaseSummary[]>([]);

  const [slots, setSlots] = useState<Array<File | null>>([null, null, null]);
  const [enforceQualityGate, setEnforceQualityGate] = useState(true);
  const [profile, setProfile] = useState<PatientProfileInput>(DEFAULT_PROFILE);
  const [caseCtx, setCaseCtx] = useState<CaseContextInput>(DEFAULT_CASE);
  const [historyOpen, setHistoryOpen] = useState(false);

  const [triageRes, setTriageRes] = useState<TriageStageResponse | null>(null);
  const [assessmentRes, setAssessmentRes] = useState<ClinicalAssessmentResponse | null>(null);
  const [savedCaseId, setSavedCaseId] = useState<string | null>(null);
  const [staleIntake, setStaleIntake] = useState(false);

  const [loadingStage1, setLoadingStage1] = useState(false);
  const [loadingStage2, setLoadingStage2] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const loadMeta = useCallback(async () => {
    try {
      const h = await fetchHealth();
      setHealth(h);
      setBackendOffline(false);

      let signedIn = false;
      if (getAuthToken()) {
        try {
          const me = await fetchMe();
          const completed = me.profile_completed ?? true;
          setUser(me.user);
          setProfile(me.profile);
          setWeightUpdatedAt(me.weight_updated_at);
          setProfileCompleted(completed);
          setProfileDirtyForSync(false);
          if (!completed) {
            setTab("profile");
          }
          signedIn = true;
        } catch {
          clearAuthToken();
          setUser(null);
        }
      }

      const [m, c] = await Promise.all([
        fetchMetrics().catch(() => null),
        signedIn ? fetchRecentCases(20).catch(() => []) : Promise.resolve([]),
      ]);
      if (m) setMetrics(m);
      setCases(c);
    } catch {
      setHealth(null);
      setBackendOffline(true);
    } finally {
      setAuthChecked(true);
    }
  }, []);

  useEffect(() => {
    loadMeta();
  }, [loadMeta]);

  const handleAuthSuccess = (res: AuthResponse) => {
    const completed = res.profile_completed ?? true;
    setUser(res.user);
    setProfile(res.profile);
    setWeightUpdatedAt(res.weight_updated_at);
    setProfileCompleted(completed);
    setProfileDirtyForSync(false);
    setStep(1);
    setTab(completed ? "triage" : "profile");
    window.scrollTo({ top: 0, behavior: "smooth" });
    fetchRecentCases(20).then(setCases).catch(() => null);
  };

  useEffect(() => {
    if (user && !profileCompleted && tab !== "profile") {
      setTab("profile");
    }
  }, [user, profileCompleted, tab]);

  const handleSignOut = async () => {
    await logoutUser();
    setUser(null);
    setWeightUpdatedAt(null);
    setProfileCompleted(true);
    setSlots([null, null, null]);
    setEnforceQualityGate(true);
    setProfile(DEFAULT_PROFILE);
    setCaseCtx(DEFAULT_CASE);
    setHistoryOpen(false);
    setTriageRes(null);
    setAssessmentRes(null);
    setSavedCaseId(null);
    setStaleIntake(false);
    setProfileDirtyForSync(false);
    setErrorMsg(null);
    setCases([]);
    setStep(1);
    setTab("triage");
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleQuickSyncProfile = async () => {
    if (!user) return;
    setSyncingProfile(true);
    setErrorMsg(null);
    try {
      const res = await saveUserProfile(profile);
      setWeightUpdatedAt(res.weight_updated_at);
      setProfileDirtyForSync(false);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Could not update saved profile.");
    } finally {
      setSyncingProfile(false);
    }
  };

  const markStale = () => {
    if (triageRes) setStaleIntake(true);
  };

  const updateProfile = (patch: Partial<PatientProfileInput>) => {
    setProfile((prev) => ({ ...prev, ...patch }));
    if (user) setProfileDirtyForSync(true);
    markStale();
  };

  const updateCase = (patch: Partial<CaseContextInput>) => {
    setCaseCtx((prev) => ({ ...prev, ...patch }));
    markStale();
  };

  const toggleClimate = (tag: string) => {
    const exists = profile.climate_tags.includes(tag);
    updateProfile({
      climate_tags: exists
        ? profile.climate_tags.filter((t) => t !== tag)
        : [...profile.climate_tags, tag],
    });
  };

  const toggleTrigger = (triggerId: string) => {
    setCaseCtx((prev) => {
      const exists = prev.active_triggers.includes(triggerId);
      return {
        ...prev,
        active_triggers: exists
          ? prev.active_triggers.filter((id) => id !== triggerId)
          : [...prev.active_triggers, triggerId],
      };
    });
  };

  const activeFiles = slots.filter((f): f is File => f !== null);
  const activeRedFlags = Object.values(caseCtx.red_flags).filter(Boolean).length;
  const safetyExtrasCount =
    profile.drug_allergies.length +
    profile.ongoing_oral_meds.length +
    profile.systemic_conditions.length +
    profile.current_routine_actives.length;

  const siteLabel =
    ANATOMICAL_SITES.find((s) => s.value === caseCtx.anatomical_site)?.label ||
    caseCtx.anatomical_site;

  const handleRunStage1 = async () => {
    setErrorMsg(null);
    if (activeFiles.length === 0) {
      setErrorMsg("Add at least 1 close-up lesion photo before continuing.");
      window.scrollTo({ top: 0, behavior: "smooth" });
      return;
    }

    setLoadingStage1(true);
    try {
      const res = await runStage1Triage({
        files: activeFiles,
        profile,
        caseContext: { ...caseCtx, active_triggers: [], other_trigger_note: "" },
        enforceQualityGate,
      });
      setTriageRes(res);
      setAssessmentRes(null);
      setSavedCaseId(null);
      setStaleIntake(false);
      setBackendOffline(false);
      setStep(2);
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Stage-1 triage failed.";
      if (msg.toLowerCase().includes("failed to fetch")) {
        setBackendOffline(true);
        setErrorMsg(
          "Cannot reach the FastAPI server on port 8000. Start it with: PYTHONPATH=backend uvicorn app.main:app --reload --port 8000"
        );
      } else {
        setErrorMsg(msg);
      }
      window.scrollTo({ top: 0, behavior: "smooth" });
    } finally {
      setLoadingStage1(false);
    }
  };

  const handleRunStage2 = async () => {
    if (!triageRes) return;
    setErrorMsg(null);
    setLoadingStage2(true);
    try {
      const res = await runStage2Assessment({
        profile,
        caseContext: { ...caseCtx, other_trigger_note: caseCtx.other_trigger_note || "" },
        stage1Output: triageRes.stage1_output,
        viewCount: activeFiles.length || 1,
        saveCase: true,
      });
      setAssessmentRes(res.assessment);
      setSavedCaseId(res.case_id);
      setStep(3);
      fetchRecentCases(20).then(setCases).catch(() => null);
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Stage-2 assessment failed.");
    } finally {
      setLoadingStage2(false);
    }
  };

  const resetForNewAssessment = () => {
    setSlots([null, null, null]);
    setCaseCtx(DEFAULT_CASE);
    if (!user) setProfile(DEFAULT_PROFILE);
    setTriageRes(null);
    setAssessmentRes(null);
    setSavedCaseId(null);
    setStaleIntake(false);
    setErrorMsg(null);
    setStep(1);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const severityLabel = (val: number) =>
    val === 0 ? "None" : val <= 3 ? "Mild" : val <= 6 ? "Moderate" : "Severe";

  const navItems: Array<[TopTab, string]> = user
    ? [
        ["triage", "Assessment"],
        ["rx", "Scan Rx"],
        ["profile", "My Profile"],
        ["cases", `Cases (${cases.length})`],
        ["benchmarks", "Benchmarks"],
      ]
    : [
        ["triage", "Assessment"],
        ["cases", `Cases (${cases.length})`],
        ["benchmarks", "Benchmarks"],
      ];

  if (!authChecked) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[var(--color-canvas)] text-sm text-[#475569]">
        <Loader2 className="mr-2 h-5 w-5 animate-spin text-[#0891B2]" />
        Checking session...
      </div>
    );
  }

  if (!user) {
    return (
      <AuthScreen
        backendOffline={backendOffline}
        onRetryHealth={loadMeta}
        onSuccess={handleAuthSuccess}
      />
    );
  }

  return (
    <div className="min-h-screen bg-[var(--color-canvas)] text-[#164E63] pb-28">
      {/* Translucent Clinical Header */}
      <header className="no-print sticky top-0 z-30 border-b border-[#CBEBF2]/80 bg-[#F3FAFC]/85 backdrop-blur-2xl">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-3 px-4 py-3 sm:px-6">
          <div className="flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#0891B2] text-xs font-bold tracking-tight text-white shadow-xs">
              NRx
            </div>
            <div>
              <span className="block font-heading text-[15px] font-bold tracking-tight text-[#164E63]">
                Nidaan-Rx
              </span>
              <span className="hidden text-[11px] text-[#475569] sm:block">
                {health?.active_backbone
                  ? `${health.active_backbone.replace("_", " ")} · ${health.device}`
                  : "Indian Derm Triage"}
              </span>
            </div>
          </div>

          <nav
            aria-label="Primary navigation"
            className="no-scrollbar order-3 flex w-full items-center gap-1 overflow-x-auto rounded-full bg-[#E8F1F6] p-1 text-xs font-semibold sm:order-2 sm:w-auto sm:justify-center"
          >
            {navItems.map(([key, label]) => {
              const locked = Boolean(user && !profileCompleted && key !== "profile");
              return (
                <button
                  key={key}
                  type="button"
                  disabled={locked}
                  title={locked ? "Complete your health profile first" : undefined}
                  aria-current={tab === key ? "page" : undefined}
                  onClick={() => {
                    if (!locked) setTab(key);
                  }}
                  className={`shrink-0 whitespace-nowrap rounded-full px-3.5 py-1.5 transition ${
                    locked
                      ? "cursor-not-allowed text-[#94A3B8] opacity-60"
                      : tab === key
                      ? "cursor-pointer bg-white text-[#0891B2] shadow-xs"
                      : "cursor-pointer text-[#475569] hover:text-[#164E63]"
                  }`}
                >
                  {label}
                </button>
              );
            })}
          </nav>

          <div className="order-2 flex items-center gap-2 sm:order-3">
            <ThemeToggle />
            {backendOffline ? (
              <button
                type="button"
                onClick={loadMeta}
                className="inline-flex cursor-pointer items-center gap-1.5 rounded-full bg-[#FEF2F2] px-3 py-1.5 text-xs font-semibold text-[#DC2626] ring-1 ring-[#FECACA]"
              >
                <WifiOff className="h-3.5 w-3.5" /> Offline
              </button>
            ) : user ? (
              <div className="flex items-center gap-1.5">
                <button
                  type="button"
                  onClick={() => setTab("profile")}
                  className="inline-flex cursor-pointer items-center gap-2 rounded-full bg-white py-1 pl-1.5 pr-3 text-xs font-semibold text-[#164E63] ring-1 ring-[#CBEBF2] transition hover:ring-[#0891B2]"
                >
                  <span className="flex h-6 w-6 items-center justify-center rounded-full bg-[#0891B2] text-[10px] font-bold text-white">
                    {user.full_name.slice(0, 2).toUpperCase()}
                  </span>
                  <span className="max-w-[100px] truncate">{user.full_name.split(" ")[0]}</span>
                </button>
                <button
                  type="button"
                  onClick={handleSignOut}
                  title="Sign out and clear form"
                  aria-label="Sign out"
                  className="flex h-8 w-8 cursor-pointer items-center justify-center rounded-full bg-white text-[#64748B] ring-1 ring-[#CBEBF2] transition hover:bg-[#FEF2F2] hover:text-[#DC2626] hover:ring-[#FECACA]"
                >
                  <LogOut className="h-3.5 w-3.5" />
                </button>
              </div>
            ) : null}
          </div>
        </div>
      </header>

      {/* Main Centered Single-Column Container */}
      <main className="mx-auto max-w-[820px] px-4 pt-8 sm:px-6">
        {backendOffline && (
          <div className="mb-6 flex flex-wrap items-center justify-between gap-3 rounded-2xl bg-[#FFFBEB] px-4 py-3.5 text-xs text-[#92400E] ring-1 ring-[#FDE68A]">
            <div className="flex items-start gap-2.5">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-[#D97706]" />
              <div>
                <p className="font-semibold">Backend server offline on port 8000</p>
                <code className="mt-0.5 block font-mono text-[11px]">
                  PYTHONPATH=backend uvicorn app.main:app --reload --port 8000
                </code>
              </div>
            </div>
            <button
              type="button"
              onClick={loadMeta}
              className="inline-flex cursor-pointer items-center gap-1.5 rounded-full bg-[#D97706] px-3.5 py-1.5 font-semibold text-white"
            >
              <RefreshCw className="h-3.5 w-3.5" /> Reconnect
            </button>
          </div>
        )}

        {tab === "triage" && (
          <div className="space-y-6">
            {/* Step Heading + Progress Rail */}
            <div className="space-y-4">
              <div className="flex flex-wrap items-end justify-between gap-3">
                <div>
                  <h1 className="text-[28px] sm:text-[34px] font-bold tracking-tight text-[#164E63] leading-tight">
                    {step === 1 && "Skin Assessment"}
                    {step === 2 && "Triggers & Review"}
                    {step === 3 && "Clinical Summary"}
                  </h1>
                  <p className="mt-1 text-sm text-[#475569]">
                    {step === 1 &&
                      "Calibrated for Indian skin tones · 12-gate safety stack · Jan Aushadhi generics."}
                    {step === 2 &&
                      "Tap only the exact exposures that apply to refine Stage-1 probabilities."}
                    {step === 3 &&
                      "Safety-verified differential, precautions, and affordable generic matches."}
                  </p>
                </div>

                {step === 1 && user && (
                  <div className="flex items-center gap-2">
                    <span className="inline-flex items-center gap-1.5 rounded-full bg-[#ECFDF5] px-3 py-1 text-xs font-semibold text-[#047857] ring-1 ring-[#A7F3D0]">
                      <UserCheck className="h-3.5 w-3.5" /> Profile loaded
                    </span>
                    {profileDirtyForSync && (
                      <button
                        type="button"
                        disabled={syncingProfile}
                        onClick={handleQuickSyncProfile}
                        className="inline-flex cursor-pointer items-center gap-1 rounded-full bg-[#0891B2] px-3 py-1 text-xs font-semibold text-white transition hover:bg-[#0E7490]"
                      >
                        <Save className="h-3 w-3" />
                        {syncingProfile ? "Saving..." : "Update saved profile"}
                      </button>
                    )}
                  </div>
                )}
              </div>

              <div className="no-print grid grid-cols-3 gap-2.5">
                {(
                  [
                    { id: 1 as TriageStep, label: "1. Intake", enabled: true },
                    { id: 2 as TriageStep, label: "2. Triggers", enabled: Boolean(triageRes) },
                    { id: 3 as TriageStep, label: "3. Report", enabled: Boolean(assessmentRes) },
                  ]
                ).map((item) => {
                  const active = step === item.id;
                  const done =
                    (item.id === 1 && Boolean(triageRes)) ||
                    (item.id === 2 && Boolean(assessmentRes));
                  return (
                    <button
                      key={item.id}
                      type="button"
                      disabled={!item.enabled}
                      onClick={() => item.enabled && setStep(item.id)}
                      className={`flex flex-col gap-1.5 text-left ${
                        item.enabled ? "cursor-pointer" : "cursor-not-allowed opacity-50"
                      }`}
                    >
                      <div
                        className={`h-1.5 w-full rounded-full transition-colors ${
                          active
                            ? "bg-[#0891B2]"
                            : done
                            ? "bg-[#059669]"
                            : "bg-[#DCECF1]"
                        }`}
                      />
                      <span
                        className={`inline-flex items-center gap-1 text-xs font-semibold ${
                          active ? "text-[#0891B2]" : "text-[#475569]"
                        }`}
                      >
                        {done && !active && <Check className="h-3 w-3 text-[#059669]" />}
                        {item.label}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>

            {errorMsg && (
              <div
                role="alert"
                className="rounded-2xl bg-[#FEF2F2] px-4 py-3.5 text-[13px] font-medium text-[#DC2626] ring-1 ring-[#FECACA]"
              >
                {errorMsg}
              </div>
            )}

            {/* STEP 1: SINGLE-COLUMN SCROLLABLE INTAKE */}
            {step === 1 && (
              <div className="space-y-6">
                <PhotoUploader
                  slots={slots}
                  onChange={(next) => {
                    setSlots(next);
                    markStale();
                  }}
                  enforceQualityGate={enforceQualityGate}
                  onToggleQualityGate={setEnforceQualityGate}
                  qualityReports={triageRes?.quality_reports}
                  attentionWeights={triageRes?.view_attention_weights}
                />

                {/* Affected Site & Symptoms */}
                <section className="clinical-card p-6 sm:p-7 space-y-6">
                  <div className="flex items-start gap-3.5">
                    <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-[#ECFEFF] text-[#0891B2] ring-1 ring-[#A5F3FC]">
                      <MapPin className="h-5 w-5" />
                    </div>
                    <div>
                      <h2 className="text-[17px] font-bold tracking-tight text-[#164E63]">
                        Location & Symptoms
                      </h2>
                      <p className="mt-0.5 text-[13px] text-[#475569]">
                        Body site controls topical steroid potency and vehicle safety automatically.
                      </p>
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-2">
                    {ANATOMICAL_SITES.map((s) => {
                      const active = caseCtx.anatomical_site === s.value;
                      return (
                        <button
                          key={s.value}
                          type="button"
                          onClick={() => updateCase({ anatomical_site: s.value })}
                          className={`cursor-pointer rounded-full px-3.5 py-2 text-xs font-semibold transition ${
                            active
                              ? "bg-[#0891B2] text-white shadow-xs"
                              : "bg-[#E8F1F6] text-[#334155] hover:bg-[#D7E7F0]"
                          }`}
                        >
                          {s.label}
                        </button>
                      );
                    })}
                  </div>

                  <div className="grid gap-5 border-t border-[#E8F1F6] pt-5 sm:grid-cols-3">
                    <div>
                      <label htmlFor="intake-duration-days" className="block text-xs font-semibold text-[#475569]">
                        Duration (days)
                      </label>
                      <input
                        id="intake-duration-days"
                        type="number"
                        min={1}
                        max={3650}
                        value={caseCtx.duration_days}
                        onChange={(e) =>
                          updateCase({ duration_days: Math.max(1, Number(e.target.value) || 1) })
                        }
                        className="mt-2 h-11 w-full rounded-xl bg-[#F3FAFC] px-4 font-mono text-[15px] font-semibold text-[#164E63] ring-1 ring-[#CBEBF2] focus:bg-white focus:ring-2 focus:ring-[#0891B2] focus:outline-none"
                      />
                    </div>

                    <div>
                      <div className="flex items-center justify-between text-xs">
                        <label htmlFor="intake-itch-score" className="font-semibold text-[#475569]">
                          Itch
                        </label>
                        <span className="rounded-full bg-[#ECFEFF] px-2 py-0.5 font-mono text-[11px] font-semibold text-[#0E7490] tabular-nums">
                          {caseCtx.itch_score}/10 · {severityLabel(caseCtx.itch_score)}
                        </span>
                      </div>
                      <input
                        id="intake-itch-score"
                        type="range"
                        min={0}
                        max={10}
                        value={caseCtx.itch_score}
                        onChange={(e) => updateCase({ itch_score: Number(e.target.value) })}
                        className="clinical-slider mt-4 w-full"
                      />
                    </div>

                    <div>
                      <div className="flex items-center justify-between text-xs">
                        <label htmlFor="intake-pain-score" className="font-semibold text-[#475569]">
                          Pain / Burning
                        </label>
                        <span className="rounded-full bg-[#FFFBEB] px-2 py-0.5 font-mono text-[11px] font-semibold text-[#B45309] tabular-nums">
                          {caseCtx.pain_burning_score}/10 ·{" "}
                          {severityLabel(caseCtx.pain_burning_score)}
                        </span>
                      </div>
                      <input
                        id="intake-pain-score"
                        type="range"
                        min={0}
                        max={10}
                        value={caseCtx.pain_burning_score}
                        onChange={(e) => updateCase({ pain_burning_score: Number(e.target.value) })}
                        className="clinical-slider mt-4 w-full"
                      />
                    </div>
                  </div>

                  <div className="divide-y divide-[#DCECF1] rounded-2xl bg-[#F3FAFC] px-4 ring-1 ring-[#DCECF1]">
                    <div className="flex items-center justify-between gap-4 py-3.5">
                      <div>
                        <p className="text-[13px] font-semibold text-[#164E63]">
                          Visible scaling, flaking, or crusting
                        </p>
                        <p className="text-xs text-[#475569]">
                          Helps distinguish fungal, eczematous, and papulosquamous patterns.
                        </p>
                      </div>
                      <ClinicalSwitch
                        label="Visible scaling, flaking, or crusting"
                        checked={caseCtx.has_scaling}
                        onChange={(has_scaling) => updateCase({ has_scaling })}
                        accent="green"
                      />
                    </div>

                    <div className="flex items-center justify-between gap-4 py-3.5">
                      <div>
                        <p className="text-[13px] font-semibold text-[#164E63]">
                          Skin feels raw, cracked, or stinging
                        </p>
                        <p className="text-xs text-[#475569]">
                          Automatically blocks harsh actives & prioritizes barrier repair.
                        </p>
                      </div>
                      <ClinicalSwitch
                        label="Skin feels raw, cracked, or stinging"
                        checked={caseCtx.compromised_barrier}
                        onChange={(compromised_barrier) => updateCase({ compromised_barrier })}
                      />
                    </div>
                  </div>
                </section>

                <SkinTonePicker
                  monkTone={profile.monk_skin_tone}
                  fitzpatrickTone={profile.fitzpatrick_tone}
                  onChangeMonk={(tone) => updateProfile({ monk_skin_tone: tone })}
                  onChangeFitzpatrick={(tone) => updateProfile({ fitzpatrick_tone: tone })}
                />

                {/* Patient Profile & Climate */}
                <section className="clinical-card p-6 sm:p-7 space-y-6">
                  <div className="flex items-start gap-3.5">
                    <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-[#ECFEFF] text-[#0891B2] ring-1 ring-[#A5F3FC]">
                      <User className="h-5 w-5" />
                    </div>
                    <div>
                      <h2 className="text-[17px] font-bold tracking-tight text-[#164E63]">
                        Patient Profile & Local Environment
                      </h2>
                      <p className="mt-0.5 text-[13px] text-[#475569]">
                        {user
                          ? "Pre-filled from your account—adjust anything for today's check."
                          : "Sign in anytime to save these defaults permanently."}
                      </p>
                    </div>
                  </div>

                  <div className="grid gap-4 sm:grid-cols-12">
                    <div className="sm:col-span-3">
                      <label htmlFor="intake-age" className="block text-xs font-semibold text-[#475569]">
                        Age (years)
                      </label>
                      <input
                        id="intake-age"
                        type="number"
                        min={1}
                        max={110}
                        value={profile.age}
                        onChange={(e) =>
                          updateProfile({ age: Math.max(1, Number(e.target.value) || 1) })
                        }
                        className="mt-1.5 h-11 w-full rounded-xl bg-[#F3FAFC] px-3.5 font-mono text-[15px] font-semibold text-[#164E63] ring-1 ring-[#CBEBF2] focus:bg-white focus:ring-2 focus:ring-[#0891B2] focus:outline-none"
                      />
                    </div>

                    <div className="sm:col-span-3">
                      <label htmlFor="intake-weight" className="block text-xs font-semibold text-[#475569]">
                        Weight (kg)
                      </label>
                      <input
                        id="intake-weight"
                        type="number"
                        min={2}
                        max={250}
                        step="0.5"
                        value={profile.weight_kg}
                        onChange={(e) =>
                          updateProfile({ weight_kg: Math.max(2, Number(e.target.value) || 2) })
                        }
                        className="mt-1.5 h-11 w-full rounded-xl bg-[#F3FAFC] px-3.5 font-mono text-[15px] font-semibold text-[#164E63] ring-1 ring-[#CBEBF2] focus:bg-white focus:ring-2 focus:ring-[#0891B2] focus:outline-none"
                      />
                    </div>

                    <div className="sm:col-span-6">
                      <label className="block text-xs font-semibold text-[#475569]">
                        Biological Sex
                      </label>
                      <div className="mt-1.5 grid h-11 grid-cols-3 gap-1 rounded-xl bg-[#E8F1F6] p-1">
                        {(["male", "female", "other"] as const).map((sex) => (
                          <button
                            key={sex}
                            type="button"
                            onClick={() =>
                              updateProfile({
                                biological_sex: sex,
                                ...(sex !== "female"
                                  ? { is_pregnant: false, is_lactating: false }
                                  : {}),
                              })
                            }
                            className={`cursor-pointer rounded-lg text-xs font-semibold capitalize transition ${
                              profile.biological_sex === sex
                                ? "bg-white text-[#164E63] shadow-xs"
                                : "text-[#475569] hover:text-[#164E63]"
                            }`}
                          >
                            {sex}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>

                  {profile.biological_sex === "female" && (
                    <div className="divide-y divide-[#FDE68A] rounded-2xl bg-[#FFFBEB] px-4 ring-1 ring-[#FDE68A]">
                      <div className="flex items-center justify-between py-3">
                        <span className="text-[13px] font-medium text-[#92400E]">
                          Currently pregnant
                        </span>
                        <ClinicalSwitch
                          label="Currently pregnant"
                          checked={profile.is_pregnant}
                          onChange={(is_pregnant) => updateProfile({ is_pregnant })}
                        />
                      </div>
                      <div className="flex items-center justify-between py-3">
                        <span className="text-[13px] font-medium text-[#92400E]">
                          Currently breastfeeding / lactating
                        </span>
                        <ClinicalSwitch
                          label="Currently breastfeeding or lactating"
                          checked={profile.is_lactating}
                          onChange={(is_lactating) => updateProfile({ is_lactating })}
                        />
                      </div>
                    </div>
                  )}

                  <div className="grid gap-4 border-t border-[#E8F1F6] pt-5 sm:grid-cols-12">
                    <div className="sm:col-span-3">
                      <label htmlFor="intake-sleep-hours" className="block text-xs font-semibold text-[#475569]">
                        Sleep (hrs)
                      </label>
                      <input
                        id="intake-sleep-hours"
                        type="number"
                        min={2}
                        max={14}
                        step="0.5"
                        value={profile.sleep_hours}
                        onChange={(e) =>
                          updateProfile({ sleep_hours: Number(e.target.value) || 7 })
                        }
                        className="mt-1.5 h-10 w-full rounded-xl bg-[#F3FAFC] px-3.5 font-mono text-sm font-semibold text-[#164E63] ring-1 ring-[#CBEBF2] focus:bg-white focus:ring-2 focus:ring-[#0891B2] focus:outline-none"
                      />
                    </div>

                    <div className="sm:col-span-5">
                      <label className="block text-xs font-semibold text-[#475569]">
                        Sleep Pattern
                      </label>
                      <div className="mt-1.5 grid grid-cols-2 gap-1.5">
                        {SLEEP_OPTIONS.map((opt) => (
                          <button
                            key={opt.value}
                            type="button"
                            onClick={() => updateProfile({ sleep_pattern: opt.value })}
                            className={`cursor-pointer rounded-xl px-2.5 py-2 text-xs font-semibold transition ${
                              profile.sleep_pattern === opt.value
                                ? "bg-[#0891B2] text-white"
                                : "bg-[#E8F1F6] text-[#334155] hover:bg-[#D7E7F0]"
                            }`}
                          >
                            {opt.label}
                          </button>
                        ))}
                      </div>
                    </div>

                    <div className="sm:col-span-4">
                      <label className="block text-xs font-semibold text-[#475569]">
                        Stress Level
                      </label>
                      <div className="mt-1.5 grid h-10 grid-cols-3 gap-1 rounded-xl bg-[#E8F1F6] p-1">
                        {(["low", "moderate", "high"] as StressLevel[]).map((lvl) => (
                          <button
                            key={lvl}
                            type="button"
                            onClick={() => updateProfile({ stress_level: lvl })}
                            className={`cursor-pointer rounded-lg text-xs font-semibold capitalize transition ${
                              profile.stress_level === lvl
                                ? "bg-white text-[#164E63] shadow-xs"
                                : "text-[#475569]"
                            }`}
                          >
                            {lvl === "moderate" ? "Med" : lvl}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div className="border-t border-[#E8F1F6] pt-5">
                    <p className="text-xs font-semibold text-[#475569]">
                      Local Climate & Water Conditions
                    </p>
                    <div className="mt-2.5 flex flex-wrap gap-2">
                      {CLIMATE_TAGS.map((tag) => {
                        const active = profile.climate_tags.includes(tag.value);
                        return (
                          <button
                            key={tag.value}
                            type="button"
                            onClick={() => toggleClimate(tag.value)}
                            className={`cursor-pointer rounded-full px-3.5 py-1.5 text-xs font-semibold transition ${
                              active
                                ? "bg-[#0891B2] text-white"
                                : "bg-[#E8F1F6] text-[#334155] hover:bg-[#D7E7F0]"
                            }`}
                          >
                            {tag.label}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                </section>

                {/* Collapsible Allergies, Meds & Actives */}
                <section className="clinical-card overflow-hidden">
                  <button
                    type="button"
                    onClick={() => setHistoryOpen(!historyOpen)}
                    className="flex w-full cursor-pointer items-center justify-between gap-3 p-6 sm:px-7 text-left transition hover:bg-[#F3FAFC]"
                  >
                    <div className="flex items-center gap-3.5">
                      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-[#ECFDF5] text-[#059669] ring-1 ring-[#A7F3D0]">
                        <SlidersHorizontal className="h-5 w-5" />
                      </div>
                      <div>
                        <p className="text-[16px] font-bold text-[#164E63]">
                          Allergies, Oral Meds & Skincare Actives
                        </p>
                        <p className="mt-0.5 text-xs text-[#475569]">
                          {safetyExtrasCount === 0
                            ? "Optional — leave collapsed if none apply"
                            : `${safetyExtrasCount} item${safetyExtrasCount > 1 ? "s" : ""} active for safety gates`}
                        </p>
                      </div>
                    </div>
                    {historyOpen ? (
                      <ChevronUp className="h-5 w-5 text-[#64748B]" />
                    ) : (
                      <ChevronDown className="h-5 w-5 text-[#64748B]" />
                    )}
                  </button>

                  {historyOpen && (
                    <div className="space-y-5 border-t border-[#E8F1F6] p-6 sm:px-7">
                      <TagChipField
                        label="Drug Allergies"
                        hint="Blocks matching salts & cross-reactive classes (Gate 7)."
                        values={profile.drug_allergies}
                        presets={PRESETS.drugAllergies}
                        onChange={(drug_allergies) => updateProfile({ drug_allergies })}
                      />
                      <TagChipField
                        label="Ongoing Oral Medicines"
                        hint="Screens for drug-drug interactions (Gate 11)."
                        values={profile.ongoing_oral_meds}
                        presets={PRESETS.oralMeds}
                        onChange={(ongoing_oral_meds) => updateProfile({ ongoing_oral_meds })}
                      />
                      <TagChipField
                        label="Long-Term Conditions"
                        hint="Screens renal, hepatic, diabetic, or endocrine contraindications."
                        values={profile.systemic_conditions}
                        presets={PRESETS.systemic}
                        onChange={(systemic_conditions) =>
                          updateProfile({ systemic_conditions })
                        }
                      />
                      <TagChipField
                        label="Current Skincare Actives"
                        hint="Prevents duplicate actives or over-exfoliation (Gate 7 & 12)."
                        values={profile.current_routine_actives}
                        presets={PRESETS.routineActives}
                        onChange={(current_routine_actives) =>
                          updateProfile({ current_routine_actives })
                        }
                      />
                    </div>
                  )}
                </section>

                {/* Urgent Red-Flag Screen */}
                <section className="clinical-card p-6 sm:p-7">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="flex items-start gap-3.5">
                      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-[#FEF2F2] text-[#DC2626] ring-1 ring-[#FECACA]">
                        <ShieldAlert className="h-5 w-5" />
                      </div>
                      <div>
                        <h2 className="text-[17px] font-bold tracking-tight text-[#164E63]">
                          Urgent Red-Flag Screen
                        </h2>
                        <p className="mt-0.5 text-[13px] text-[#475569]">
                          Toggle any warning sign below to trigger immediate escalation.
                        </p>
                      </div>
                    </div>

                    <span
                      className={`rounded-full px-3 py-1 text-xs font-semibold tabular-nums ${
                        activeRedFlags > 0
                          ? "bg-[#DC2626] text-white"
                          : "bg-[#E8F1F6] text-[#475569]"
                      }`}
                    >
                      {activeRedFlags} active
                    </span>
                  </div>

                  <div className="mt-5 grid gap-2.5 sm:grid-cols-2">
                    {RED_FLAGS.map((rf) => {
                      const on = caseCtx.red_flags[rf.key];
                      return (
                        <div
                          key={rf.key}
                          className={`flex items-center justify-between gap-3 rounded-2xl px-4 py-3 transition ${
                            on
                              ? "bg-[#FEF2F2] ring-1 ring-[#FECACA]"
                              : "bg-[#F3FAFC] ring-1 ring-[#DCECF1]"
                          }`}
                        >
                          <span
                            className={`text-[13px] font-medium ${
                              on ? "text-[#991B1B]" : "text-[#164E63]"
                            }`}
                          >
                            {rf.label}
                          </span>
                          <ClinicalSwitch
                            label={rf.label}
                            checked={on}
                            accent="red"
                            onChange={(v) =>
                              updateCase({
                                red_flags: { ...caseCtx.red_flags, [rf.key]: v },
                              })
                            }
                          />
                        </div>
                      );
                    })}
                  </div>
                </section>
              </div>
            )}

            {/* STEP 2: STAGE-1 REVIEW + ATOMIC TRIGGERS */}
            {step === 2 && triageRes && (
              <div className="space-y-6">
                <section className="clinical-card p-6 sm:p-7">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div className="flex items-start gap-3.5">
                      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-[#ECFEFF] text-[#0891B2] ring-1 ring-[#A5F3FC]">
                        <HeartPulse className="h-5 w-5" />
                      </div>
                      <div>
                        <h2 className="text-[17px] font-bold tracking-tight text-[#164E63]">
                          Initial Visual & Context Triage
                        </h2>
                        <p className="mt-0.5 font-mono text-xs text-[#475569]">
                          {triageRes.backbone} · {triageRes.latency_ms.toFixed(0)} ms · OOD{" "}
                          {triageRes.stage1_output.ood_score.toFixed(2)}
                        </p>
                      </div>
                    </div>

                    <span
                      className={`rounded-full px-3 py-1 text-xs font-semibold ${
                        triageRes.stage1_output.is_ood
                          ? "bg-[#FFFBEB] text-[#B45309] ring-1 ring-[#FDE68A]"
                          : "bg-[#ECFDF5] text-[#047857] ring-1 ring-[#A7F3D0]"
                      }`}
                    >
                      {triageRes.stage1_output.is_ood ? "Out-of-Distribution" : "In-Distribution"}
                    </span>
                  </div>

                  <div className="mt-5 grid gap-3 sm:grid-cols-2">
                    {Object.entries(triageRes.stage1_output.condition_probs)
                      .sort((a, b) => b[1] - a[1])
                      .slice(0, 4)
                      .map(([condId, prob], idx) => (
                        <div
                          key={condId}
                          className={`rounded-2xl p-4 ${
                            idx === 0
                              ? "bg-[#ECFEFF] ring-1 ring-[#67E8F9]"
                              : "bg-[#F3FAFC] ring-1 ring-[#DCECF1]"
                          }`}
                        >
                          <div className="flex items-center justify-between gap-2 text-[13px]">
                            <span className="font-semibold capitalize text-[#164E63]">
                              {condId.replaceAll("_", " ")}
                            </span>
                            <span className="font-mono font-bold text-[#0891B2] tabular-nums">
                              {(prob * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-[#DCECF1]">
                            <div
                              className="h-full rounded-full bg-[#0891B2]"
                              style={{ width: `${Math.min(100, prob * 100)}%` }}
                            />
                          </div>
                        </div>
                      ))}
                  </div>

                  <div className="mt-4 flex items-center justify-between border-t border-[#E8F1F6] pt-3.5 text-xs text-[#475569]">
                    <span className="inline-flex items-center gap-1.5">
                      <Activity className="h-3.5 w-3.5 text-[#0891B2]" />
                      MIL weights:{" "}
                      {triageRes.view_attention_weights
                        .map((w) => `${(w * 100).toFixed(0)}%`)
                        .join(" · ")}
                    </span>
                    <button
                      type="button"
                      onClick={() => setStep(1)}
                      className="cursor-pointer font-semibold text-[#0891B2] hover:underline"
                    >
                      Edit photos
                    </button>
                  </div>
                </section>

                <section className="clinical-card p-6 sm:p-7 space-y-5">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div>
                      <h2 className="text-[17px] font-bold tracking-tight text-[#164E63]">
                        Possible Contributing Triggers
                      </h2>
                      <p className="mt-0.5 text-[13px] text-[#475569]">
                        Select only the exact factors that apply to you.
                      </p>
                    </div>
                    <span className="rounded-full bg-[#ECFEFF] px-3 py-1 text-xs font-semibold text-[#0E7490] ring-1 ring-[#A5F3FC] tabular-nums">
                      {caseCtx.active_triggers.length} selected
                    </span>
                  </div>

                  <div className="space-y-4">
                    {triageRes.dynamic_trigger_buckets.map((bucket) => (
                      <div
                        key={bucket.bucket_id}
                        className="rounded-2xl bg-[#F3FAFC] p-4 ring-1 ring-[#DCECF1] space-y-3"
                      >
                        <div className="flex items-center gap-2">
                          <h3 className="text-[13px] font-bold text-[#164E63]">{bucket.title}</h3>
                          {bucket.india_critical && (
                            <span className="rounded-full bg-[#FFFBEB] px-2.5 py-0.5 text-[10px] font-semibold text-[#B45309] ring-1 ring-[#FDE68A]">
                              India Context
                            </span>
                          )}
                        </div>
                        <div className="flex flex-wrap gap-2">
                          {bucket.triggers.map((tr) => {
                            const active = caseCtx.active_triggers.includes(tr.trigger_id);
                            return (
                              <button
                                key={tr.trigger_id}
                                type="button"
                                title={tr.description}
                                onClick={() => toggleTrigger(tr.trigger_id)}
                                className={`cursor-pointer rounded-full px-3.5 py-2 text-left text-xs font-medium transition ${
                                  active
                                    ? "bg-[#0891B2] text-white shadow-xs"
                                    : "bg-white text-[#164E63] ring-1 ring-[#CBEBF2] hover:ring-[#0891B2]"
                                }`}
                              >
                                {tr.label}
                              </button>
                            );
                          })}
                        </div>
                      </div>
                    ))}
                  </div>

                  <div>
                    <label htmlFor="intake-other-trigger" className="block text-xs font-semibold text-[#475569]">
                      Other recent cream, trigger, or note (Optional)
                    </label>
                    <input
                      id="intake-other-trigger"
                      type="text"
                      value={caseCtx.other_trigger_note}
                      onChange={(e) =>
                        setCaseCtx((prev) => ({ ...prev, other_trigger_note: e.target.value }))
                      }
                      placeholder="e.g., Used an OTC steroid cream for 10 days..."
                      className="mt-2 h-11 w-full rounded-xl bg-[#F3FAFC] px-4 text-xs text-[#164E63] ring-1 ring-[#CBEBF2] placeholder:text-[#64748B] focus:bg-white focus:ring-2 focus:ring-[#0891B2] focus:outline-none"
                    />
                  </div>
                </section>
              </div>
            )}

            {/* STEP 3: CLINICAL REPORT */}
            {step === 3 && assessmentRes && (
              <ClinicalReportView
                assessment={assessmentRes}
                caseId={savedCaseId}
                backbone={triageRes?.backbone}
                patientAge={profile.age}
              />
            )}
          </div>
        )}

        {tab === "profile" && user && (
          <ProfileSettingsView
            user={user}
            profile={profile}
            weightUpdatedAt={weightUpdatedAt}
            isOnboarding={!profileCompleted}
            onProfileSaved={(nextP, nextW) => {
              const wasOnboarding = !profileCompleted;
              setProfile(nextP);
              setWeightUpdatedAt(nextW);
              setProfileCompleted(true);
              setProfileDirtyForSync(false);
              if (wasOnboarding) {
                setTab("triage");
                setStep(1);
                window.scrollTo({ top: 0, behavior: "smooth" });
              }
            }}
            onSignOut={handleSignOut}
          />
        )}

        {tab === "rx" && <RxScannerView />}

        {tab === "benchmarks" && <BenchmarksView metrics={metrics} />}

        {tab === "cases" && (
          <CasesHistoryView
            cases={cases}
            isAuthenticated={Boolean(user)}
            onCaseDeleted={(id) => setCases((prev) => prev.filter((c) => c.case_id !== id))}
            onCasesRefresh={() => fetchRecentCases(20).then(setCases).catch(() => null)}
          />
        )}
      </main>

      {/* Floating Frosted Bottom Action Dock */}
      {tab === "triage" && user && (
        <div className="no-print fixed inset-x-0 bottom-4 z-30 px-4">
          <div className="mx-auto flex max-w-[780px] items-center justify-between gap-3 rounded-full border border-[#CBEBF2] bg-white/90 px-4 py-2.5 shadow-[0_12px_32px_rgba(8,145,178,0.16)] backdrop-blur-2xl">
            {step === 1 && (
              <>
                <div className="hidden pl-2 text-xs text-[#475569] sm:block">
                  <span className="font-semibold text-[#164E63]">
                    {activeFiles.length} photo{activeFiles.length === 1 ? "" : "s"}
                  </span>
                  {" · "}
                  {siteLabel}
                  {" · "}
                  Monk {profile.monk_skin_tone}
                  {staleIntake && (
                    <span className="ml-2 font-semibold text-[#D97706]">· Updated</span>
                  )}
                </div>
                <button
                  type="button"
                  disabled={loadingStage1}
                  onClick={handleRunStage1}
                  className="ml-auto inline-flex h-11 w-full cursor-pointer items-center justify-center gap-2 rounded-full bg-[#059669] px-6 text-sm font-semibold text-white shadow-xs transition hover:bg-[#047857] disabled:opacity-60 sm:w-auto"
                >
                  {loadingStage1 ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      {staleIntake ? "Re-Analyze Assessment" : "Continue to Triggers"}
                      <ArrowRight className="h-4 w-4" />
                    </>
                  )}
                </button>
              </>
            )}

            {step === 2 && (
              <>
                <button
                  type="button"
                  onClick={() => setStep(1)}
                  className="inline-flex h-10 cursor-pointer items-center gap-1.5 rounded-full bg-[#E8F1F6] px-4 text-xs font-semibold text-[#164E63] hover:bg-[#D5E7F0]"
                >
                  <ArrowLeft className="h-3.5 w-3.5" /> Back
                </button>
                <button
                  type="button"
                  disabled={loadingStage2}
                  onClick={handleRunStage2}
                  className="inline-flex h-11 cursor-pointer items-center justify-center gap-2 rounded-full bg-[#059669] px-6 text-sm font-semibold text-white shadow-xs transition hover:bg-[#047857] disabled:opacity-60"
                >
                  {loadingStage2 ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Generating Report...
                    </>
                  ) : (
                    <>
                      Generate Clinical Report
                      <ArrowRight className="h-4 w-4" />
                    </>
                  )}
                </button>
              </>
            )}

            {step === 3 && (
              <>
                <button
                  type="button"
                  onClick={() => setStep(2)}
                  className="inline-flex h-10 cursor-pointer items-center gap-1.5 rounded-full bg-[#E8F1F6] px-4 text-xs font-semibold text-[#164E63] hover:bg-[#D5E7F0]"
                >
                  <ArrowLeft className="h-3.5 w-3.5" /> Adjust Triggers
                </button>
                <button
                  type="button"
                  onClick={resetForNewAssessment}
                  className="inline-flex h-10 cursor-pointer items-center gap-1.5 rounded-full bg-[#164E63] px-5 text-xs font-semibold text-white hover:bg-[#0E3A4A]"
                >
                  <RotateCcw className="h-3.5 w-3.5" /> New Assessment
                </button>
              </>
            )}
          </div>
        </div>
      )}


    </div>
  );
}
