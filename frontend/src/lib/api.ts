// Mirrors backend/app/core/schemas.py and backend/app/main.py.

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") || "http://127.0.0.1:8000";

const TOKEN_STORAGE_KEY = "nidaan_rx_v2_token";

export function getAuthToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.sessionStorage.getItem(TOKEN_STORAGE_KEY);
}

export function setAuthToken(token: string): void {
  if (typeof window !== "undefined") {
    window.localStorage.removeItem(TOKEN_STORAGE_KEY);
    window.sessionStorage.setItem(TOKEN_STORAGE_KEY, token);
  }
}

export function clearAuthToken(): void {
  if (typeof window !== "undefined") {
    window.sessionStorage.removeItem(TOKEN_STORAGE_KEY);
    window.localStorage.removeItem(TOKEN_STORAGE_KEY);
  }
}

function authHeaders(extra?: Record<string, string>): Record<string, string> {
  const token = getAuthToken();
  return {
    ...(extra ?? {}),
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
}

export type AnatomicalSite =
  | "face"
  | "eyelids"
  | "lips_mucosa"
  | "scalp"
  | "beard"
  | "trunk"
  | "arms_legs"
  | "hands_feet"
  | "groin"
  | "underarms"
  | "nails"
  | "generalized";

export type BiologicalSex = "male" | "female" | "other";
export type SleepPattern = "regular" | "late_sleeper" | "night_shift" | "irregular";
export type StressLevel = "low" | "moderate" | "high";
export type GateAction = "PASS" | "WARN" | "BLOCK" | "MODIFY" | "HARD_STOP";

export interface AuthUser {
  user_id: string;
  email: string;
  username?: string | null;
  full_name: string;
  created_at: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: AuthUser;
  profile: PatientProfileInput;
  weight_updated_at: string;
  profile_completed?: boolean;
}

export interface MeResponse {
  user: AuthUser;
  profile: PatientProfileInput;
  weight_updated_at: string;
  updated_at: string;
  profile_completed?: boolean;
}

export interface SavedProfileResponse {
  profile: PatientProfileInput;
  weight_updated_at: string;
  updated_at: string;
  profile_completed?: boolean;
}

export interface RedFlagChecklist {
  fever: boolean;
  rapidly_spreading_redness_warmth: boolean;
  pus_oozing_bleeding: boolean;
  severe_pain: boolean;
  eye_lip_throat_swelling: boolean;
  breathing_trouble: boolean;
  changing_bleeding_dark_mole: boolean;
}

export interface PatientProfileInput {
  age: number;
  biological_sex: BiologicalSex;
  weight_kg: number;
  fitzpatrick_tone: number;
  monk_skin_tone: number;
  drug_allergies: string[];
  ongoing_oral_meds: string[];
  systemic_conditions: string[];
  is_pregnant: boolean;
  is_lactating: boolean;
  sleep_hours: number;
  sleep_pattern: SleepPattern;
  stress_level: StressLevel;
  climate_tags: string[];
  current_routine_actives: string[];
}

export interface CaseContextInput {
  anatomical_site: AnatomicalSite;
  duration_days: number;
  itch_score: number;
  pain_burning_score: number;
  has_scaling: boolean;
  compromised_barrier: boolean;
  red_flags: RedFlagChecklist;
  active_triggers: string[];
  other_trigger_note: string;
}

export interface ModelInferenceOutput {
  condition_probs: Record<string, number>;
  family_probs: Record<string, number>;
  is_ood: boolean;
  ood_score: number;
}

export interface QualityReport {
  passed?: boolean;
  [key: string]: unknown;
}

export interface TriggerItem {
  trigger_id: string;
  bucket_id: number;
  label: string;
  description: string;
}

export interface TriggerBucket {
  bucket_id: number;
  slug: string;
  title: string;
  india_critical: boolean;
  triggers: TriggerItem[];
}

export interface TriageStageResponse {
  backbone: string;
  latency_ms: number;
  all_views_passed_quality: boolean;
  quality_reports: QualityReport[];
  view_attention_weights: number[];
  stage1_output: ModelInferenceOutput;
  dynamic_trigger_buckets: TriggerBucket[];
}

export interface HealthResponse {
  status: string;
  version: string;
  active_backbone: string;
  device: string;
  canonical_classes: number;
  tier1_families: number;
  vision_encoder_loaded: boolean;
}

export interface MetricsResponse {
  ablation_reports: Record<string, unknown>;
}

export interface GateEvent {
  gate_id: string;
  gate_name: string;
  triggered: boolean;
  action: GateAction;
  message: string;
  affected_option_id: string | null;
}

export interface PMBJPGenericMatch {
  product_name: string;
  generic_code: string | null;
  mrp_inr: number | null;
  vehicle: string;
  salts: Array<{ salt: string; strength: string }>;
  equivalence_verified: boolean;
  commercial_avg_inr: number | null;
  estimated_savings_pct: number | null;
}

export interface TreatmentOption {
  option_id: string;
  display_name: string;
  vehicle: string;
  regulatory_tier: string;
  posology: string;
  notes: string;
}

export interface EvaluatedTreatmentCard {
  option: TreatmentOption;
  regulatory_label: string;
  inline_badges: string[];
  pmbjp_matches: PMBJPGenericMatch[];
  dosage_note: string | null;
}

export interface ConditionPrediction {
  condition_id: string;
  display_name: string;
  tier1_family: string;
  prior_prob: number;
  posterior_prob: number;
  contributing_triggers: Array<Record<string, unknown>>;
}

export interface ClinicalAssessmentResponse {
  disclaimer_banner: string;
  escalation_hard_stop: boolean;
  escalation_reason: string | null;
  minor_guardian_notice: string | null;
  weight_refresh_prompt: boolean;
  top_matches: ConditionPrediction[];
  identified_triggers: string[];
  do_not_apply_warnings: string[];
  tier1_otc_care: EvaluatedTreatmentCard[];
  tier2_dermatologist_prep: EvaluatedTreatmentCard[];
  lifestyle_action_plan: string[];
  safety_audit_log: GateEvent[];
}

export interface AssessResponse {
  case_id: string | null;
  backbone: string;
  assessment: ClinicalAssessmentResponse;
}

export type FollowUpTrendValue = "improving" | "unchanged" | "worsening";

export interface FollowUpEntry {
  followup_id: string;
  case_id: string;
  created_at: string;
  followup_day: number;
  followup_trend: FollowUpTrendValue | string;
  itch_score: number | null;
  pain_burning_score: number | null;
  notes: string;
  escalated: boolean;
  escalation_reason: string | null;
}

export interface FollowUpResponse {
  followup: FollowUpEntry;
  assessment: ClinicalAssessmentResponse;
  case_context: CaseContextInput;
  followups: FollowUpEntry[];
}

export interface CaseSummary {
  case_id: string;
  user_id?: string | null;
  created_at: string;
  backbone: string;
  view_count: number;
  top_condition_id: string | null;
  top_condition_name: string | null;
  top_confidence: number | null;
  is_ood: boolean;
  ood_score: number | null;
  escalation_hard_stop: boolean;
  followup_count?: number;
  latest_followup_day?: number | null;
  latest_followup_trend?: string | null;
  longitudinal_escalated?: boolean;
}

export interface CaseDetail {
  case_id?: string;
  created_at?: string;
  profile: PatientProfileInput;
  case_context: CaseContextInput;
  inference: ModelInferenceOutput;
  assessment: ClinicalAssessmentResponse;
  followups?: FollowUpEntry[];
}

function errorMessage(body: unknown, fallback: string): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      const msgs = detail
        .map((d) => (d && typeof d === "object" && "msg" in d ? String((d as { msg: unknown }).msg) : ""))
        .filter(Boolean);
      if (msgs.length) return msgs.join("; ");
    }
  }
  return fallback;
}

async function parse<T>(res: Response, fallback: string): Promise<T> {
  const body: unknown = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(errorMessage(body, fallback));
  return body as T;
}

export async function registerUser(params: {
  email: string;
  username?: string;
  password: string;
  full_name: string;
}): Promise<AuthResponse> {
  const res = await fetch(`${API_BASE}/api/v2/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return parse<AuthResponse>(res, `Registration failed (${res.status})`);
}

export async function loginUser(params: {
  identifier?: string;
  email?: string;
  username?: string;
  login_by?: "auto" | "email" | "username";
  password: string;
}): Promise<AuthResponse> {
  const res = await fetch(`${API_BASE}/api/v2/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return parse<AuthResponse>(res, `Login failed (${res.status})`);
}

export async function fetchMe(): Promise<MeResponse> {
  const res = await fetch(`${API_BASE}/api/v2/auth/me`, {
    headers: authHeaders(),
    cache: "no-store",
  });
  return parse<MeResponse>(res, `Session check failed (${res.status})`);
}

export async function saveUserProfile(profile: PatientProfileInput): Promise<SavedProfileResponse> {
  const res = await fetch(`${API_BASE}/api/v2/profile`, {
    method: "PUT",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(profile),
  });
  return parse<SavedProfileResponse>(res, `Profile save failed (${res.status})`);
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/api/v2/health`, { cache: "no-store" });
  return parse<HealthResponse>(res, `Health check failed (${res.status})`);
}

export async function fetchMetrics(): Promise<MetricsResponse> {
  const res = await fetch(`${API_BASE}/api/v2/metrics`, { cache: "no-store" });
  return parse<MetricsResponse>(res, `Metrics fetch failed (${res.status})`);
}

export async function runStage1Triage(params: {
  files: File[];
  profile: PatientProfileInput;
  caseContext: CaseContextInput;
  enforceQualityGate: boolean;
}): Promise<TriageStageResponse> {
  const fd = new FormData();
  for (const file of params.files) fd.append("images", file);
  fd.append("profile_json", JSON.stringify(params.profile));
  fd.append("case_json", JSON.stringify(params.caseContext));
  fd.append("enforce_quality_gate", String(params.enforceQualityGate));

  const res = await fetch(`${API_BASE}/api/v2/triage`, {
    method: "POST",
    headers: authHeaders(),
    body: fd,
  });
  return parse<TriageStageResponse>(res, `Triage request failed (${res.status})`);
}

export async function runStage2Assessment(params: {
  profile: PatientProfileInput;
  caseContext: CaseContextInput;
  stage1Output: ModelInferenceOutput;
  viewCount: number;
  saveCase?: boolean;
}): Promise<AssessResponse> {
  const res = await fetch(`${API_BASE}/api/v2/assess-from-triage`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({
      profile: params.profile,
      case_context: params.caseContext,
      stage1_output: params.stage1Output,
      view_count: params.viewCount,
      save_case: params.saveCase ?? true,
    }),
  });
  return parse<AssessResponse>(res, `Assessment failed (${res.status})`);
}

export async function fetchRecentCases(limit = 20): Promise<CaseSummary[]> {
  const res = await fetch(`${API_BASE}/api/v2/cases?limit=${limit}`, {
    headers: authHeaders(),
    cache: "no-store",
  });
  const data = await parse<{ cases: CaseSummary[] }>(res, `Case list failed (${res.status})`);
  return data.cases ?? [];
}

export async function fetchCase(caseId: string): Promise<CaseDetail> {
  const res = await fetch(`${API_BASE}/api/v2/cases/${encodeURIComponent(caseId)}`, {
    headers: authHeaders(),
    cache: "no-store",
  });
  return parse<CaseDetail>(res, `Case fetch failed (${res.status})`);
}

export async function deleteCase(caseId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/v2/cases/${encodeURIComponent(caseId)}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  await parse<{ deleted: boolean }>(res, `Case deletion failed (${res.status})`);
}

export async function fetchImagePreviewUrl(file: File): Promise<string> {
  const fd = new FormData();
  fd.append("image", file);
  const res = await fetch(`${API_BASE}/api/v2/preview-image`, {
    method: "POST",
    headers: authHeaders(),
    body: fd,
  });
  if (!res.ok) {
    throw new Error(`Preview conversion failed (${res.status})`);
  }
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

export async function logoutUser(): Promise<void> {
  const token = getAuthToken();
  if (!token) return;
  try {
    await fetch(`${API_BASE}/api/v2/auth/logout`, {
      method: "POST",
      headers: authHeaders(),
    });
  } catch {
  } finally {
    clearAuthToken();
  }
}

export async function downloadCasePdf(caseId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/v2/cases/${encodeURIComponent(caseId)}/report.pdf`, {
    method: "GET",
    headers: authHeaders(),
    cache: "no-store",
  });
  if (!res.ok) {
    const body: unknown = await res.json().catch(() => ({}));
    throw new Error(errorMessage(body, `PDF download failed (${res.status})`));
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `Nidaan-Rx-${caseId}.pdf`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export interface RxSafetyAlert {
  gate_id: string;
  severity: "BLOCK" | "WARN";
  message: string;
}

export interface RxPmbjpMatch {
  product_name: string;
  generic_code: string | null;
  mrp_inr: number | null;
  vehicle: string;
  salts: Array<{ salt: string; strength: string }>;
  commercial_ref_inr: number | null;
  savings_inr: number | null;
  savings_pct: number | null;
  equivalence_status: string;
  vehicle_exact: boolean;
}

export interface RxScannedItem {
  ocr_line: string;
  ocr_confidence?: number;
  matched_name: string;
  match_source: "generic_salt" | "1mg_brand";
  match_confidence: number;
  vehicle: string;
  salts: Array<{ salt: string; strength: string }>;
  commercial_price_inr: number | null;
  pmbjp_matches: RxPmbjpMatch[];
  safety_alerts: RxSafetyAlert[];
}

export interface RxScanResponse {
  is_prescription: boolean;
  latency_ms: number;
  raw_ocr_lines: Array<{ text: string; confidence: number }>;
  items: RxScannedItem[];
  cross_rx_ddi_alerts: RxSafetyAlert[];
  summary: {
    detected_medicines: number;
    pmbjp_matched_medicines: number;
    safety_alert_count: number;
    commercial_total_inr: number | null;
    pmbjp_total_inr: number | null;
    total_savings_inr: number | null;
    total_savings_pct: number | null;
  };
}

export async function scanPrescriptionImage(file: File): Promise<RxScanResponse> {
  const fd = new FormData();
  fd.append("image", file);
  const res = await fetch(`${API_BASE}/api/v2/rx/scan`, {
    method: "POST",
    headers: authHeaders(),
    body: fd,
  });
  return parse<RxScanResponse>(res, `Prescription scan failed (${res.status})`);
}

export async function searchPrescriptionText(query: string): Promise<RxScanResponse> {
  const res = await fetch(`${API_BASE}/api/v2/rx/search`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ query }),
  });
  return parse<RxScanResponse>(res, `Medicine search failed (${res.status})`);
}

export async function logCaseFollowup(
  caseId: string,
  payload: {
    followup_day: number;
    followup_trend: FollowUpTrendValue;
    itch_score?: number;
    pain_burning_score?: number;
    notes?: string;
  }
): Promise<FollowUpResponse> {
  const res = await fetch(`${API_BASE}/api/v2/cases/${encodeURIComponent(caseId)}/followups`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload),
  });
  return parse<FollowUpResponse>(res, `Follow-up check-in failed (${res.status})`);
}

export async function deleteCaseFollowup(
  caseId: string,
  followupId: string
): Promise<{
  deleted: boolean;
  followup_id: string;
  assessment: ClinicalAssessmentResponse;
  case_context: CaseContextInput;
  followups: FollowUpEntry[];
}> {
  const res = await fetch(
    `${API_BASE}/api/v2/cases/${encodeURIComponent(caseId)}/followups/${encodeURIComponent(followupId)}`,
    {
      method: "DELETE",
      headers: authHeaders(),
    }
  );
  return parse(res, `Follow-up deletion failed (${res.status})`);
}
