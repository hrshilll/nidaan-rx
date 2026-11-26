"use client";

import React from "react";
import { BarChart3 } from "lucide-react";
import type { MetricsResponse } from "@/lib/api";

interface StageRow {
  stage: string;
  tier1_acc: number;
  tier1_macro_f1: number;
  canonical_top1_acc: number;
  canonical_top3_acc: number;
  canonical_macro_f1: number;
  canonical_ece: number;
}

interface ParsedBackbone {
  name: string;
  stages: StageRow[];
}

function parseReports(metrics: MetricsResponse | null): ParsedBackbone[] {
  if (!metrics?.ablation_reports) return [];
  const out: ParsedBackbone[] = [];
  for (const [key, val] of Object.entries(metrics.ablation_reports)) {
    if (
      val &&
      typeof val === "object" &&
      "stages" in val &&
      Array.isArray((val as { stages: unknown }).stages)
    ) {
      out.push({ name: key, stages: (val as { stages: StageRow[] }).stages });
    }
  }
  return out;
}

export function BenchmarksView({ metrics }: { metrics: MetricsResponse | null }) {
  const reports = parseReports(metrics);

  if (reports.length === 0) {
    return (
      <div className="clinical-card p-10 text-center">
        <BarChart3 className="mx-auto h-7 w-7 text-[#0891B2]" />
        <p className="mt-2.5 text-sm font-bold text-[#164E63]">No benchmark metrics loaded</p>
        <p className="mt-1 text-xs text-[#475569]">
          Ensure FastAPI is running on port 8000 with ablation JSON files in backend/artifacts.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-[28px] sm:text-[32px] font-bold tracking-tight text-[#164E63]">
          Model Benchmarks
        </h1>
        <p className="mt-1 text-sm text-[#475569]">
          4-stage ablation evaluation across PanDerm-Base and DINOv2-Small on held-out test cases.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {reports.map((rep) => {
          const finalStage = rep.stages[rep.stages.length - 1];
          return (
            <div key={rep.name} className="clinical-card p-5 sm:p-6">
              <div className="flex items-center justify-between">
                <h2 className="text-base font-bold capitalize text-[#164E63]">
                  {rep.name.replaceAll("_", " ")}
                </h2>
                <span className="rounded-full bg-[#ECFEFF] px-3 py-0.5 text-[11px] font-semibold text-[#0E7490] ring-1 ring-[#A5F3FC]">
                  Stage 4 Final
                </span>
              </div>

              {finalStage && (
                <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <div className="rounded-xl bg-[#F3FAFC] p-3 ring-1 ring-[#DCECF1]">
                    <p className="text-[11px] text-[#475569]">Tier-1</p>
                    <p className="mt-1 font-mono text-base font-bold text-[#164E63] tabular-nums">
                      {(finalStage.tier1_acc * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="rounded-xl bg-[#F3FAFC] p-3 ring-1 ring-[#DCECF1]">
                    <p className="text-[11px] text-[#475569]">Top-1</p>
                    <p className="mt-1 font-mono text-base font-bold text-[#164E63] tabular-nums">
                      {(finalStage.canonical_top1_acc * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="rounded-xl bg-[#ECFDF5] p-3 ring-1 ring-[#A7F3D0]">
                    <p className="text-[11px] text-[#047857]">Top-3</p>
                    <p className="mt-1 font-mono text-base font-bold text-[#059669] tabular-nums">
                      {(finalStage.canonical_top3_acc * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="rounded-xl bg-[#ECFEFF] p-3 ring-1 ring-[#A5F3FC]">
                    <p className="text-[11px] text-[#0E7490]">ECE</p>
                    <p className="mt-1 font-mono text-base font-bold text-[#0891B2] tabular-nums">
                      {finalStage.canonical_ece.toFixed(4)}
                    </p>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {reports.map((rep) => (
        <section key={`${rep.name}-table`} className="clinical-card p-5 sm:p-6">
          <h3 className="text-sm font-bold capitalize text-[#164E63]">
            {rep.name.replaceAll("_", " ")} — 4-Stage Ablation Table
          </h3>
          <div className="mt-3 overflow-x-auto">
            <table className="w-full min-w-[600px] border-collapse text-left text-xs">
              <thead>
                <tr className="border-b border-[#E8F1F6] text-[#475569]">
                  <th className="py-2.5 pr-4 font-semibold">Stage</th>
                  <th className="px-3 py-2.5 font-semibold">Tier-1 Acc</th>
                  <th className="px-3 py-2.5 font-semibold">Tier-1 F1</th>
                  <th className="px-3 py-2.5 font-semibold">Top-1 Acc</th>
                  <th className="px-3 py-2.5 font-semibold">Top-3 Acc</th>
                  <th className="px-3 py-2.5 font-semibold">Macro F1</th>
                  <th className="py-2.5 pl-3 font-semibold">ECE</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#E8F1F6] font-mono tabular-nums">
                {rep.stages.map((s, idx) => (
                  <tr
                    key={s.stage}
                    className={idx === rep.stages.length - 1 ? "bg-[#ECFEFF] font-semibold" : ""}
                  >
                    <td className="py-2.5 pr-4 font-sans font-semibold text-[#164E63]">
                      {s.stage}
                    </td>
                    <td className="px-3 py-2.5">{(s.tier1_acc * 100).toFixed(2)}%</td>
                    <td className="px-3 py-2.5">{(s.tier1_macro_f1 * 100).toFixed(2)}%</td>
                    <td className="px-3 py-2.5">{(s.canonical_top1_acc * 100).toFixed(2)}%</td>
                    <td className="px-3 py-2.5 text-[#059669]">
                      {(s.canonical_top3_acc * 100).toFixed(2)}%
                    </td>
                    <td className="px-3 py-2.5 text-[#0891B2]">
                      {(s.canonical_macro_f1 * 100).toFixed(2)}%
                    </td>
                    <td className="py-2.5 pl-3">{s.canonical_ece.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ))}
    </div>
  );
}
