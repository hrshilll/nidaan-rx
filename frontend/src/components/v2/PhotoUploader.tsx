"use client";

import React, { useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  Camera,
  CheckCircle2,
  Loader2,
  Plus,
  RefreshCw,
  ShieldCheck,
  X,
} from "lucide-react";
import { fetchImagePreviewUrl, type QualityReport } from "@/lib/api";
import { SLOT_META } from "@/lib/constants";

interface PhotoUploaderProps {
  slots: Array<File | null>;
  onChange: (next: Array<File | null>) => void;
  enforceQualityGate: boolean;
  onToggleQualityGate: (v: boolean) => void;
  qualityReports?: QualityReport[];
  attentionWeights?: number[];
}

function needsServerPreview(file: File): boolean {
  const name = file.name.toLowerCase();
  const type = (file.type || "").toLowerCase();
  if (/\.(heic|heif|tif|tiff|bmp|dng|cr2|nef|avif)$/i.test(name)) return true;
  if (type && !["image/jpeg", "image/jpg", "image/png", "image/webp", "image/gif"].includes(type)) {
    return true;
  }
  return false;
}

export function PhotoUploader({
  slots,
  onChange,
  enforceQualityGate,
  onToggleQualityGate,
  qualityReports,
  attentionWeights,
}: PhotoUploaderProps) {
  const [previews, setPreviews] = useState<Array<string | null>>([null, null, null]);
  const [converting, setConverting] = useState<Array<boolean>>([false, false, false]);
  const [previewErr, setPreviewErr] = useState<Array<string | null>>([null, null, null]);
  const [dragIndex, setDragIndex] = useState<number | null>(null);
  const inputsRef = useRef<Array<HTMLInputElement | null>>([null, null, null]);

  useEffect(() => {
    let cancelled = false;
    const createdUrls: string[] = [];

    const resolveAll = async () => {
      const nextUrls: Array<string | null> = [null, null, null];
      const nextConv = [false, false, false];
      const nextErr: Array<string | null> = [null, null, null];

      // First pass: instant URLs for native formats, mark server-converted ones as loading
      slots.forEach((f, idx) => {
        if (!f) return;
        if (needsServerPreview(f)) {
          nextConv[idx] = true;
        } else {
          const u = URL.createObjectURL(f);
          createdUrls.push(u);
          nextUrls[idx] = u;
        }
      });

      setPreviews([...nextUrls]);
      setConverting([...nextConv]);
      setPreviewErr([...nextErr]);

      // Second pass: convert HEIC/HEIF/TIFF/etc. via backend Pillow+heif endpoint
      await Promise.all(
        slots.map(async (f, idx) => {
          if (!f || !needsServerPreview(f)) return;
          try {
            const converted = await fetchImagePreviewUrl(f);
            if (cancelled) {
              URL.revokeObjectURL(converted);
              return;
            }
            createdUrls.push(converted);
            setPreviews((prev) => {
              const copy = [...prev];
              copy[idx] = converted;
              return copy;
            });
          } catch {
            if (!cancelled) {
              setPreviewErr((prev) => {
                const copy = [...prev];
                copy[idx] = "Preview unavailable (will still analyze)";
                return copy;
              });
            }
          } finally {
            if (!cancelled) {
              setConverting((prev) => {
                const copy = [...prev];
                copy[idx] = false;
                return copy;
              });
            }
          }
        })
      );
    };

    resolveAll();

    return () => {
      cancelled = true;
      createdUrls.forEach((u) => URL.revokeObjectURL(u));
    };
  }, [slots]);

  // If browser still fails to decode an image tag, automatically convert via server
  const handleImgDecodeError = async (idx: number) => {
    const f = slots[idx];
    if (!f || converting[idx]) return;
    setConverting((prev) => {
      const copy = [...prev];
      copy[idx] = true;
      return copy;
    });
    try {
      const converted = await fetchImagePreviewUrl(f);
      setPreviews((prev) => {
        const copy = [...prev];
        copy[idx] = converted;
        return copy;
      });
    } catch {
      setPreviewErr((prev) => {
        const copy = [...prev];
        copy[idx] = "Preview unavailable";
        return copy;
      });
    } finally {
      setConverting((prev) => {
        const copy = [...prev];
        copy[idx] = false;
        return copy;
      });
    }
  };

  const filledOrderMap = new Map<number, number>();
  let cursor = 0;
  slots.forEach((f, idx) => {
    if (f) {
      filledOrderMap.set(idx, cursor);
      cursor += 1;
    }
  });

  const setSlotFile = (idx: number, file: File | null) => {
    const next = [...slots];
    next[idx] = file;
    onChange(next);
  };

  const handleMultiPick = (startIdx: number, files: FileList | null) => {
    if (!files || files.length === 0) return;
    const picked = Array.from(files).filter(
      (f) =>
        f.type.startsWith("image/") ||
        /\.(heic|heif|jpg|jpeg|png|webp|bmp|tif|tiff|avif|gif)$/i.test(f.name)
    );
    if (picked.length === 0) return;
    const next = [...slots];
    if (picked.length === 1) {
      next[startIdx] = picked[0];
    } else {
      let pIdx = 0;
      for (let i = startIdx; i < 3 && pIdx < picked.length; i++) {
        next[i] = picked[pIdx++];
      }
    }
    onChange(next);
  };

  const filledCount = slots.filter(Boolean).length;

  return (
    <section className="clinical-card p-6 sm:p-7">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="flex items-start gap-3.5">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-[#ECFEFF] text-[#0891B2] ring-1 ring-[#A5F3FC]">
            <Camera className="h-5 w-5" />
          </div>
          <div>
            <div className="flex items-center gap-2.5">
              <h2 className="text-[17px] font-semibold tracking-[-0.015em] text-[#164E63]">
                Lesion Photographs
              </h2>
              <span className="rounded-full bg-[#E8F1F6] px-2.5 py-0.5 text-[11px] font-semibold text-[#475569] tabular-nums">
                {filledCount} of 3
              </span>
            </div>
            <p className="mt-0.5 text-[13px] leading-relaxed text-[#475569]">
              Upload 1–3 tight close-ups in daylight. Supports JPG, PNG, WebP, HEIC, and more.
            </p>
          </div>
        </div>

        <button
          type="button"
          onClick={() => onToggleQualityGate(!enforceQualityGate)}
          className={`inline-flex cursor-pointer items-center gap-1.5 rounded-full px-3.5 py-1.5 text-xs font-medium transition ${
            enforceQualityGate
              ? "bg-[#ECFDF5] text-[#047857] ring-1 ring-[#A7F3D0]"
              : "bg-[#E8F1F6] text-[#475569] ring-1 ring-black/[0.06]"
          }`}
        >
          <ShieldCheck className="h-3.5 w-3.5" />
          Quality Gate {enforceQualityGate ? "On" : "Off"}
        </button>
      </div>

      <div className="mt-5 grid grid-cols-1 gap-4 sm:grid-cols-3">
        {SLOT_META.map((meta, idx) => {
          const file = slots[idx];
          const preview = previews[idx];
          const isConv = converting[idx];
          const pErr = previewErr[idx];
          const apiPos = filledOrderMap.get(idx);
          const qReport = apiPos != null ? qualityReports?.[apiPos] : undefined;
          const attn = apiPos != null ? attentionWeights?.[apiPos] : undefined;
          const qPassed = qReport ? qReport.passed !== false : undefined;

          return (
            <div
              key={meta.label}
              onDragOver={(e) => {
                e.preventDefault();
                setDragIndex(idx);
              }}
              onDragLeave={() => setDragIndex(null)}
              onDrop={(e) => {
                e.preventDefault();
                setDragIndex(null);
                handleMultiPick(idx, e.dataTransfer.files);
              }}
              className={`group relative overflow-hidden rounded-[18px] transition-all duration-200 ${
                dragIndex === idx
                  ? "bg-[#E8F4F2]/70 ring-2 ring-[#0891B2]"
                  : file
                  ? "bg-[#F9F9FB] ring-1 ring-black/[0.07]"
                  : "bg-[#F5F5F7]/80 ring-1 ring-dashed ring-black/[0.14] hover:bg-[#EFEFF4] hover:ring-[#0891B2]/50"
              }`}
            >
              <input
                ref={(el) => {
                  inputsRef.current[idx] = el;
                }}
                type="file"
                accept="image/*,.heic,.heif,.avif,.tif,.tiff,.bmp"
                multiple={idx === 0 && filledCount === 0}
                onChange={(e) => {
                  handleMultiPick(idx, e.target.files);
                  e.target.value = "";
                }}
                className="hidden"
              />

              {file ? (
                <div className="flex flex-col">
                  <div className="relative aspect-[4/3] w-full overflow-hidden bg-[#1D1D1F]">
                    {isConv ? (
                      <div className="flex h-full w-full flex-col items-center justify-center gap-2 bg-[#E8F1F6] text-[#475569]">
                        <Loader2 className="h-5 w-5 animate-spin text-[#0891B2]" />
                        <span className="text-[11px] font-medium">Converting preview...</span>
                      </div>
                    ) : preview ? (
                      <img
                        src={preview}
                        alt={meta.label}
                        onError={() => handleImgDecodeError(idx)}
                        className="h-full w-full object-cover"
                      />
                    ) : (
                      <div className="flex h-full w-full flex-col items-center justify-center p-3 text-center bg-[#E8F1F6] text-[#475569]">
                        <Camera className="h-5 w-5 opacity-60" />
                        <span className="mt-1.5 text-[11px] font-medium truncate max-w-full">
                          {file.name}
                        </span>
                        {pErr && <span className="mt-0.5 text-[10px] text-[#C25E24]">{pErr}</span>}
                      </div>
                    )}

                    <div className="absolute inset-x-0 top-0 flex items-center justify-between bg-gradient-to-b from-black/50 to-transparent p-2.5">
                      <span className="rounded-full bg-black/50 px-2.5 py-0.5 text-[11px] font-medium text-white backdrop-blur-md">
                        {meta.label}
                      </span>
                      <button
                        type="button"
                        onClick={() => setSlotFile(idx, null)}
                        aria-label={`Remove ${meta.label}`}
                        className="flex h-7 w-7 cursor-pointer items-center justify-center rounded-full bg-black/50 text-white backdrop-blur-md transition hover:bg-[#FF3B30]"
                      >
                        <X className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  </div>

                  <div className="space-y-1.5 px-3 py-2.5">
                    <div className="flex items-center justify-between text-[11px]">
                      {qPassed === true && (
                        <span className="inline-flex items-center gap-1 font-semibold text-[#157A3E]">
                          <CheckCircle2 className="h-3.5 w-3.5" /> Verified
                        </span>
                      )}
                      {qPassed === false && (
                        <span className="inline-flex items-center gap-1 font-semibold text-[#D97706]">
                          <AlertTriangle className="h-3.5 w-3.5" /> Retake
                        </span>
                      )}
                      {qPassed === undefined && (
                        <span className="truncate max-w-[110px] text-[#475569]">
                          {file.name.split(".").pop()?.toUpperCase() || "IMG"} ready
                        </span>
                      )}

                      <button
                        type="button"
                        onClick={() => inputsRef.current[idx]?.click()}
                        className="inline-flex cursor-pointer items-center gap-1 font-semibold text-[#0891B2] hover:underline"
                      >
                        <RefreshCw className="h-3 w-3" /> Replace
                      </button>
                    </div>

                    {attn != null && (
                      <div className="space-y-1 pt-0.5">
                        <div className="flex items-center justify-between text-[10px] text-[#475569]">
                          <span>Attention weight</span>
                          <span className="font-mono font-semibold text-[#0891B2] tabular-nums">
                            {(attn * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="h-1.5 w-full overflow-hidden rounded-full bg-[#E5E5EA]">
                          <div
                            className="h-full rounded-full bg-[#0891B2]"
                            style={{ width: `${Math.min(100, attn * 100)}%` }}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                <button
                  type="button"
                  onClick={() => inputsRef.current[idx]?.click()}
                  className="flex aspect-[4/3] w-full cursor-pointer flex-col items-center justify-center p-4 text-center"
                >
                  <div className="flex h-11 w-11 items-center justify-center rounded-full bg-white text-[#0891B2] shadow-sm ring-1 ring-black/[0.06] transition-transform duration-200 group-hover:scale-105">
                    <Plus className="h-5 w-5" />
                  </div>
                  <p className="mt-3 text-[13px] font-semibold text-[#164E63]">{meta.label}</p>
                  <p className="mt-1 line-clamp-2 text-[11px] leading-snug text-[#475569]">
                    {meta.hint}
                  </p>
                </button>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
