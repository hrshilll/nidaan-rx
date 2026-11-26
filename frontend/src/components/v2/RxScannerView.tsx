"use client";

import React, { useEffect, useRef, useState } from "react";
import {
  AlertOctagon,
  AlertTriangle,
  BadgeCheck,
  ChevronDown,
  ChevronUp,
  FileText,
  IndianRupee,
  Loader2,
  Pill,
  ScanLine,
  Search,
  ShieldAlert,
  ShieldCheck,
  Upload,
  X,
} from "lucide-react";
import {
  fetchImagePreviewUrl,
  scanPrescriptionImage,
  searchPrescriptionText,
  type RxScanResponse,
} from "@/lib/api";

function needsServerPreview(file: File): boolean {
  const name = file.name.toLowerCase();
  const type = (file.type || "").toLowerCase();
  if (/\.(heic|heif|tif|tiff|bmp|avif)$/i.test(name)) return true;
  if (type && !["image/jpeg", "image/jpg", "image/png", "image/webp"].includes(type)) return true;
  return false;
}

const SAMPLE_QUERIES = [
  "Candid Cream 1%",
  "Terbinafine 1% Cream\nItraconazole 100mg Capsule",
  "Ketoconazole 2% Shampoo\nCetirizine 10mg Tablet",
];

export function RxScannerView() {
  const [mode, setMode] = useState<"photo" | "search">("photo");
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [convertingPreview, setConvertingPreview] = useState(false);
  const [queryText, setQueryText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RxScanResponse | null>(null);
  const [ocrOpen, setOcrOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    let cancelled = false;
    let objUrl: string | null = null;

    const loadPreview = async () => {
      if (needsServerPreview(file)) {
        setConvertingPreview(true);
        try {
          objUrl = await fetchImagePreviewUrl(file);
          if (!cancelled) setPreviewUrl(objUrl);
        } catch {
          if (!cancelled) setPreviewUrl(null);
        } finally {
          if (!cancelled) setConvertingPreview(false);
        }
      } else {
        objUrl = URL.createObjectURL(file);
        setPreviewUrl(objUrl);
      }
    };
    loadPreview();
    return () => {
      cancelled = true;
      if (objUrl) URL.revokeObjectURL(objUrl);
    };
  }, [file]);

  const handleScanPhoto = async () => {
    if (!file) {
      setError("Please upload a prescription photo first.");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const res = await scanPrescriptionImage(file);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prescription OCR scan failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleSearchText = async (overrideQuery?: string) => {
    const q = (overrideQuery ?? queryText).trim();
    if (q.length < 2) {
      setError("Enter at least one medicine brand or salt name.");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const res = await searchPrescriptionText(q);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Medicine lookup failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-[28px] sm:text-[32px] font-bold tracking-tight text-[#164E63]">
          Prescription Scanner & Jan Aushadhi Lookup
        </h1>
        <p className="mt-1 text-sm text-[#475569]">
          PaddleOCR + exact-salt PMBJP matching (Gates 1–3) + automated Allergy, Pregnancy & DDI screening against your saved profile.
        </p>
      </div>

      {/* Mode Switcher */}
      <div className="clinical-card p-6 sm:p-7 space-y-5">
        <div className="grid grid-cols-2 gap-1 rounded-full bg-[#E8F1F6] p-1 text-xs font-semibold">
          <button
            type="button"
            onClick={() => {
              setMode("photo");
              setError(null);
            }}
            className={`inline-flex cursor-pointer items-center justify-center gap-1.5 rounded-full py-2.5 transition ${
              mode === "photo"
                ? "bg-white text-[#0891B2] shadow-xs"
                : "text-[#475569] hover:text-[#164E63]"
            }`}
          >
            <ScanLine className="h-4 w-4" /> Scan Prescription Photo
          </button>
          <button
            type="button"
            onClick={() => {
              setMode("search");
              setError(null);
            }}
            className={`inline-flex cursor-pointer items-center justify-center gap-1.5 rounded-full py-2.5 transition ${
              mode === "search"
                ? "bg-white text-[#0891B2] shadow-xs"
                : "text-[#475569] hover:text-[#164E63]"
            }`}
          >
            <Search className="h-4 w-4" /> Search Brand / Salt
          </button>
        </div>

        {error && (
          <div className="rounded-xl bg-[#FEF2F2] px-4 py-3 text-xs font-medium text-[#DC2626] ring-1 ring-[#FECACA]">
            {error}
          </div>
        )}

        {mode === "photo" ? (
          <div className="space-y-4">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*,.heic,.heif,.avif,.tif,.tiff,.bmp"
              onChange={(e) => {
                const picked = e.target.files?.[0] ?? null;
                setFile(picked);
                setResult(null);
                setError(null);
                e.target.value = "";
              }}
              className="hidden"
            />

            {!file ? (
              <div
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  const dropped = e.dataTransfer.files?.[0] ?? null;
                  if (dropped) {
                    setFile(dropped);
                    setResult(null);
                  }
                }}
                onClick={() => fileInputRef.current?.click()}
                className="flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-[#A5F3FC] bg-[#F3FAFC] px-6 py-12 text-center transition hover:border-[#0891B2] hover:bg-[#ECFEFF]"
              >
                <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-white text-[#0891B2] shadow-xs ring-1 ring-[#CBEBF2]">
                  <Upload className="h-6 w-6" />
                </div>
                <p className="mt-3 text-sm font-bold text-[#164E63]">
                  Drop a prescription photo or click to browse
                </p>
                <p className="mt-1 text-xs text-[#475569]">
                  Supports printed & smartphone prescriptions (JPG, PNG, WebP, HEIC).
                </p>
              </div>
            ) : (
              <div className="overflow-hidden rounded-2xl bg-[#F3FAFC] ring-1 ring-[#CBEBF2]">
                <div className="relative max-h-[360px] overflow-hidden bg-[#0F172A]/5">
                  {convertingPreview ? (
                    <div className="flex h-52 items-center justify-center text-xs text-[#475569]">
                      <Loader2 className="mr-2 h-4 w-4 animate-spin text-[#0891B2]" />
                      Preparing preview...
                    </div>
                  ) : previewUrl ? (
                    <img
                      src={previewUrl}
                      alt="Uploaded prescription"
                      className="mx-auto max-h-[340px] object-contain"
                    />
                  ) : (
                    <div className="flex h-40 items-center justify-center text-xs text-[#475569]">
                      <FileText className="mr-2 h-4 w-4" /> {file.name}
                    </div>
                  )}
                  <button
                    type="button"
                    onClick={() => {
                      setFile(null);
                      setResult(null);
                    }}
                    aria-label="Remove prescription image"
                    className="absolute right-3 top-3 flex h-8 w-8 cursor-pointer items-center justify-center rounded-full bg-black/60 text-white backdrop-blur-sm hover:bg-[#DC2626]"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>

                <div className="flex flex-wrap items-center justify-between gap-3 border-t border-[#CBEBF2] bg-white px-4 py-3">
                  <span className="truncate text-xs font-medium text-[#475569]">{file.name}</span>
                  <button
                    type="button"
                    disabled={loading}
                    onClick={handleScanPhoto}
                    className="inline-flex h-10 cursor-pointer items-center gap-2 rounded-full bg-[#059669] px-5 text-xs font-semibold text-white shadow-xs transition hover:bg-[#047857] disabled:opacity-60"
                  >
                    {loading ? (
                      <>
                        <Loader2 className="h-4 w-4 animate-spin" /> Running PaddleOCR...
                      </>
                    ) : (
                      <>
                        <ScanLine className="h-4 w-4" /> Extract & Match Generics
                      </>
                    )}
                  </button>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-4">
            <div>
              <label className="block text-xs font-semibold text-[#164E63]">
                Enter branded medicines or salts (one per line)
              </label>
              <textarea
                rows={3}
                value={queryText}
                onChange={(e) => setQueryText(e.target.value)}
                placeholder={"e.g.\nCandid Cream 1%\nItraconazole 100mg Capsule\nKetoconazole 2% Shampoo"}
                className="mt-1.5 w-full rounded-xl bg-[#F3FAFC] p-3.5 text-sm text-[#164E63] ring-1 ring-[#CBEBF2] placeholder:text-[#64748B] focus:bg-white focus:ring-2 focus:ring-[#0891B2] focus:outline-none"
              />
            </div>

            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex flex-wrap gap-1.5">
                {SAMPLE_QUERIES.map((sq, i) => (
                  <button
                    key={i}
                    type="button"
                    onClick={() => {
                      setQueryText(sq);
                      handleSearchText(sq);
                    }}
                    className="cursor-pointer rounded-full bg-[#E8F1F6] px-3 py-1 text-[11px] font-medium text-[#164E63] transition hover:bg-[#D5E7F0]"
                  >
                    Try: {sq.split("\n")[0]}
                  </button>
                ))}
              </div>
              <button
                type="button"
                disabled={loading}
                onClick={() => handleSearchText()}
                className="inline-flex h-10 cursor-pointer items-center gap-2 rounded-full bg-[#059669] px-5 text-xs font-semibold text-white shadow-xs transition hover:bg-[#047857] disabled:opacity-60"
              >
                {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
                {loading ? "Matching..." : "Find Jan Aushadhi Equivalents"}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Results Area */}
      {result && (
        <div className="space-y-5">
          {/* Summary KPI Strip */}
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="clinical-card p-5">
              <p className="text-xs font-semibold text-[#475569]">Medicines Resolved</p>
              <p className="mt-1 font-mono text-2xl font-bold text-[#164E63] tabular-nums">
                {result.summary.detected_medicines}
                <span className="ml-1.5 text-xs font-normal text-[#475569]">
                  ({result.summary.pmbjp_matched_medicines} PMBJP matched)
                </span>
              </p>
            </div>

            <div className="clinical-card bg-[#ECFDF5] p-5 ring-1 ring-[#A7F3D0]">
              <p className="text-xs font-semibold text-[#047857]">Estimated Jan Aushadhi Savings</p>
              <p className="mt-1 flex items-baseline gap-2 font-mono text-2xl font-bold text-[#059669] tabular-nums">
                {result.summary.total_savings_inr != null ? (
                  <>
                    <span>₹{result.summary.total_savings_inr.toFixed(0)}</span>
                    <span className="rounded-full bg-[#059669] px-2 py-0.5 text-xs font-semibold text-white">
                      {Math.round(result.summary.total_savings_pct ?? 0)}% off
                    </span>
                  </>
                ) : (
                  <span className="text-base text-[#475569]">—</span>
                )}
              </p>
            </div>

            <div className="clinical-card p-5">
              <p className="text-xs font-semibold text-[#475569]">Patient Safety Flags</p>
              <p
                className={`mt-1 font-mono text-2xl font-bold tabular-nums ${
                  result.summary.safety_alert_count > 0 ? "text-[#DC2626]" : "text-[#059669]"
                }`}
              >
                {result.summary.safety_alert_count}
                <span className="ml-1.5 text-xs font-normal text-[#475569]">
                  · {result.latency_ms.toFixed(0)} ms
                </span>
              </p>
            </div>
          </div>

          {/* Cross-Prescription DDI Banner */}
          {result.cross_rx_ddi_alerts.length > 0 && (
            <div className="rounded-[22px] bg-[#FEF2F2] p-5 ring-2 ring-[#DC2626]">
              <div className="flex items-start gap-3">
                <AlertOctagon className="mt-0.5 h-5 w-5 shrink-0 text-[#DC2626]" />
                <div className="space-y-1.5">
                  <h3 className="text-sm font-bold text-[#991B1B]">
                    Cross-Prescription Drug Interaction Detected (Gate 11)
                  </h3>
                  {result.cross_rx_ddi_alerts.map((al, i) => (
                    <p key={i} className="text-xs text-[#7F1D1D]">
                      {al.message}
                    </p>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Hard-Negative / Empty State */}
          {!result.is_prescription && (
            <div className="clinical-card p-8 text-center">
              <ShieldCheck className="mx-auto h-8 w-8 text-[#0891B2]" />
              <p className="mt-2.5 text-base font-bold text-[#164E63]">
                No Prescription Medicines Detected
              </p>
              <p className="mt-1 text-xs text-[#475569]">
                Hard-negative guard prevented hallucinated drug matches. Try a sharper, well-lit photo or use the Search Brand / Salt tab.
              </p>
            </div>
          )}

          {/* Extracted Drug Cards */}
          {result.items.map((item, idx) => (
            <section key={`${item.matched_name}-${idx}`} className="clinical-card p-5 sm:p-6 space-y-4">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="flex items-start gap-3">
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-[#ECFEFF] text-[#0891B2] ring-1 ring-[#A5F3FC]">
                    <Pill className="h-5 w-5" />
                  </div>
                  <div>
                    <h3 className="text-base font-bold text-[#164E63]">{item.matched_name}</h3>
                    <p className="mt-0.5 text-xs text-[#475569]">
                      Detected: <span className="font-mono text-[#164E63]">{item.ocr_line}</span> ·{" "}
                      {item.vehicle} · Match {(item.match_confidence * 100).toFixed(0)}%
                    </p>
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {item.salts.map((s, i) => (
                        <span
                          key={i}
                          className="rounded-full bg-[#E8F1F6] px-2.5 py-0.5 text-[11px] font-semibold capitalize text-[#164E63]"
                        >
                          {s.salt} {s.strength !== "N/A" ? `(${s.strength})` : ""}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>

                {item.commercial_price_inr != null && (
                  <div className="text-right">
                    <p className="text-[11px] text-[#475569]">Commercial Avg</p>
                    <p className="font-mono text-sm font-bold text-[#475569] tabular-nums">
                      ₹{item.commercial_price_inr.toFixed(0)}
                    </p>
                  </div>
                )}
              </div>

              {/* Per-drug Safety Alerts */}
              {item.safety_alerts.length > 0 && (
                <div className="space-y-2">
                  {item.safety_alerts.map((al, i) => (
                    <div
                      key={i}
                      className={`flex items-start gap-2.5 rounded-xl px-3.5 py-2.5 text-xs font-medium ${
                        al.severity === "BLOCK"
                          ? "bg-[#FEF2F2] text-[#991B1B] ring-1 ring-[#FECACA]"
                          : "bg-[#FFFBEB] text-[#92400E] ring-1 ring-[#FDE68A]"
                      }`}
                    >
                      {al.severity === "BLOCK" ? (
                        <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0 text-[#DC2626]" />
                      ) : (
                        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-[#D97706]" />
                      )}
                      <span>
                        <strong className="font-bold">[{al.gate_id}] </strong>
                        {al.message}
                      </span>
                    </div>
                  ))}
                </div>
              )}

              {/* PMBJP Equivalents */}
              {item.pmbjp_matches.length > 0 ? (
                <div className="rounded-2xl bg-[#ECFDF5] p-4 ring-1 ring-[#A7F3D0]">
                  <div className="mb-2.5 flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wider text-[#047857]">
                    <BadgeCheck className="h-4 w-4" />
                    Verified Jan Aushadhi (PMBJP) Equivalents
                  </div>
                  <div className="grid gap-2.5 sm:grid-cols-2">
                    {item.pmbjp_matches.map((m, mi) => (
                      <div key={mi} className="rounded-xl bg-white p-3.5 ring-1 ring-[#A7F3D0]">
                        <div className="flex items-start justify-between gap-2">
                          <span className="text-xs font-bold text-[#164E63]">{m.product_name}</span>
                          {m.mrp_inr != null && (
                            <span className="inline-flex shrink-0 items-center font-mono text-sm font-bold text-[#059669] tabular-nums">
                              <IndianRupee className="h-3.5 w-3.5" />
                              {m.mrp_inr.toFixed(0)}
                            </span>
                          )}
                        </div>
                        <p className="mt-1 text-[11px] text-[#047857]">{m.equivalence_status}</p>
                        <div className="mt-1.5 flex flex-wrap items-center gap-2 text-[11px] text-[#475569]">
                          {m.generic_code && <span className="font-mono">#{m.generic_code}</span>}
                          <span>{m.vehicle}</span>
                          {m.commercial_ref_inr != null && (
                            <span className="line-through text-[#64748B]">
                              ₹{m.commercial_ref_inr.toFixed(0)}
                            </span>
                          )}
                          {m.savings_pct != null && m.savings_pct > 0 && (
                            <span className="rounded-full bg-[#059669] px-2 py-0.5 text-[10px] font-semibold text-white">
                              Save {Math.round(m.savings_pct)}%
                            </span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="rounded-xl bg-[#F3FAFC] px-4 py-3 text-xs text-[#475569] ring-1 ring-[#DCECF1]">
                  No exact-salt Jan Aushadhi (PMBJP) generic SKU matched Gates 1–3 for this formulation.
                </div>
              )}
            </section>
          ))}

          {/* Raw OCR Disclosure */}
          {result.raw_ocr_lines.length > 0 && (
            <section className="clinical-card overflow-hidden">
              <button
                type="button"
                onClick={() => setOcrOpen(!ocrOpen)}
                className="flex w-full cursor-pointer items-center justify-between px-6 py-4 text-left text-xs font-bold text-[#164E63] hover:bg-[#F3FAFC]"
              >
                <span>Raw OCR Lines ({result.raw_ocr_lines.length})</span>
                {ocrOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
              </button>
              {ocrOpen && (
                <div className="divide-y divide-[#E8F1F6] border-t border-[#E8F1F6] px-6 py-2 font-mono text-[11px] text-[#475569]">
                  {result.raw_ocr_lines.map((ln, i) => (
                    <div key={i} className="flex items-center justify-between gap-3 py-1.5">
                      <span>{ln.text}</span>
                      <span className="shrink-0 text-[#0891B2]">
                        {(ln.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </section>
          )}
        </div>
      )}
    </div>
  );
}
