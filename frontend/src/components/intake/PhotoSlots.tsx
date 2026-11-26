"use client";

import { useEffect, useId, useRef, useState } from "react";
import { Upload } from "lucide-react";
import { Button, Meter } from "@/components/ui/primitives";
import { SLOT_META } from "@/lib/constants";
import { cx, pct } from "@/lib/format";

export interface SlotResult {
  passed: boolean | null; // null until Stage-1 has returned
  weight: number | null; // Gated Attention-MIL weight, 0 to 1
}

function useObjectUrl(file: File | null) {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!file) {
      setUrl(null);
      return;
    }
    const next = URL.createObjectURL(file);
    setUrl(next);
    return () => URL.revokeObjectURL(next);
  }, [file]);
  return url;
}

function PhotoSlot({
  index,
  file,
  onFile,
  result,
  disabled,
}: {
  index: number;
  file: File | null;
  onFile: (f: File | null) => void;
  result: SlotResult | null;
  disabled?: boolean;
}) {
  const meta = SLOT_META[index];
  const url = useObjectUrl(file);
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const inputId = useId();

  const choose = () => inputRef.current?.click();

  return (
    <div className="flex min-w-0 flex-col gap-2">
      <div className="flex items-baseline justify-between gap-2">
        <h3 className="text-sm font-medium">{meta.label}</h3>
        <span className="text-xs text-ink-soft">{index + 1} of 3</span>
      </div>

      {file && url ? (
        <div className="overflow-hidden rounded-xl border border-line bg-surface-2">
          <img src={url} alt={`${meta.label} preview`} className="aspect-[4/5] w-full object-cover" />
        </div>
      ) : (
        <button
          type="button"
          disabled={disabled}
          onClick={choose}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            const f = e.dataTransfer.files?.[0];
            if (f && f.type.startsWith("image/")) onFile(f);
          }}
          className={cx(
            "flex aspect-[4/5] flex-col items-center justify-center gap-2 rounded-xl border border-dashed p-4 text-center",
            "transition-[background-color,border-color] duration-200 ease-out disabled:opacity-50",
            "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent",
            dragging ? "border-accent bg-accent-tint" : "border-line-strong bg-surface hover:border-accent/60",
          )}
        >
          <Upload aria-hidden="true" size={20} className="text-ink-soft" />
          <span className="text-sm font-medium">Drop a photo or choose</span>
          <span className="max-w-[20ch] text-xs text-ink-soft">{meta.hint}</span>
        </button>
      )}

      <input
        id={inputId}
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        className="sr-only"
        tabIndex={-1}
        aria-label={`${meta.label} photo`}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
          e.target.value = "";
        }}
      />

      {file && (
        <>
          <div className="flex items-center justify-between gap-2 text-xs">
            <span className={cx("font-medium", result?.passed === false ? "text-danger" : "text-ink-soft")}>
              {result == null || result.passed == null
                ? "Quality check after triage"
                : result.passed
                  ? "Quality passed"
                  : "Quality failed, retake"}
            </span>
            {result?.weight != null && <span className="tabular-nums text-accent-strong">{pct(result.weight)}</span>}
          </div>
          {result?.weight != null && <Meter value={result.weight} label={`${meta.label} attention weight`} />}
          <div className="flex gap-2">
            <Button variant="secondary" onClick={choose} disabled={disabled} className="min-h-9 flex-1 px-3 text-xs">
              Replace
            </Button>
            <Button variant="ghost" onClick={() => onFile(null)} disabled={disabled} className="min-h-9 flex-1 px-3 text-xs">
              Remove
            </Button>
          </div>
        </>
      )}
    </div>
  );
}

export function PhotoSlots({
  files,
  onChange,
  results,
  qualityGate,
  onQualityGate,
  disabled,
}: {
  files: (File | null)[];
  onChange: (index: number, file: File | null) => void;
  results: (SlotResult | null)[];
  qualityGate: boolean;
  onQualityGate: (v: boolean) => void;
  disabled?: boolean;
}) {
  const count = files.filter(Boolean).length;
  return (
    <section aria-labelledby="photos-heading" className="flex flex-col gap-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 id="photos-heading" className="text-base font-semibold tracking-tight">
            Photographs
          </h2>
          <p className="mt-1 text-sm text-ink-soft">Add one to three photos. Natural daylight works best.</p>
        </div>
        <label className="flex min-h-9 cursor-pointer items-center gap-2 text-sm text-ink-soft">
          <input
            type="checkbox"
            checked={qualityGate}
            onChange={(e) => onQualityGate(e.target.checked)}
            className="size-4 accent-accent"
          />
          Quality gate
        </label>
      </div>
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-3">
        {SLOT_META.map((_, i) => (
          <PhotoSlot
            key={i}
            index={i}
            file={files[i] ?? null}
            onFile={(f) => onChange(i, f)}
            result={results[i] ?? null}
            disabled={disabled}
          />
        ))}
      </div>
      <p className="text-xs text-ink-soft">{count === 0 ? "No photos added yet." : `${count} of 3 photos added.`}</p>
    </section>
  );
}
