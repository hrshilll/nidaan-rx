import { ShieldAlert } from "lucide-react";

export function HardStopBanner({ reason }: { reason: string | null }) {
  return (
    <div role="alert" className="rise flex gap-4 rounded-xl border-2 border-danger bg-danger-tint p-5">
      <ShieldAlert aria-hidden="true" size={24} className="mt-0.5 shrink-0 text-danger" />
      <div className="flex flex-col gap-2">
        <p className="text-sm font-semibold uppercase tracking-wide text-danger">Urgent escalation</p>
        <p className="text-sm text-ink">{reason ?? "Escalate this case to a doctor before any treatment."}</p>
        <p className="text-sm font-medium text-ink">Do not start self-treatment before a doctor has examined this.</p>
      </div>
    </div>
  );
}
