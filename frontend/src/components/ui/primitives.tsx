import { forwardRef, useId, type ButtonHTMLAttributes, type ReactNode } from "react";
import { cx } from "@/lib/format";

type Variant = "primary" | "secondary" | "ghost" | "danger";

// Fitts's law: every interactive target is at least 44px tall.
export const Button = forwardRef<HTMLButtonElement, ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }>(
  function Button({ variant = "primary", className, ...rest }, ref) {
    return (
      <button
        ref={ref}
        type="button"
        {...rest}
        className={cx(
          "inline-flex min-h-11 items-center justify-center gap-2 rounded-lg px-5 text-sm font-medium",
          "transition-[background-color,border-color,color,transform] duration-200 ease-(--ease-settle)",
          "active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-45",
          "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent",
          variant === "primary" && "bg-accent text-white hover:bg-accent-strong",
          variant === "secondary" && "border border-line-strong bg-surface text-ink hover:border-ink/40",
          variant === "ghost" && "text-ink-soft hover:bg-surface-2 hover:text-ink",
          variant === "danger" && "bg-danger text-white hover:brightness-110",
          className,
        )}
      />
    );
  },
);

export function Field({
  label,
  hint,
  error,
  children,
  id,
}: {
  label: string;
  hint?: string;
  error?: string | null;
  children: ReactNode;
  id?: string;
}) {
  return (
    <div className="flex min-w-0 flex-col gap-1.5">
      <label htmlFor={id} className="text-sm font-medium text-ink">
        {label}
      </label>
      {children}
      {error ? (
        <p className="text-xs text-danger">{error}</p>
      ) : hint ? (
        <p className="text-xs text-ink-soft">{hint}</p>
      ) : null}
    </div>
  );
}

export const inputClass =
  "min-h-11 w-full rounded-lg border border-line-strong bg-surface px-3 text-sm text-ink " +
  "placeholder:text-ink-soft/70 transition-colors duration-200 focus:border-accent focus:outline-2 focus:outline-accent/30";

export function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div role="radiogroup" aria-label={label} className="grid auto-cols-fr grid-flow-col gap-1 rounded-lg bg-surface-2 p-1">
      {options.map((o) => {
        const on = o.value === value;
        return (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={on}
            onClick={() => onChange(o.value)}
            className={cx(
              "min-h-10 rounded-md px-2 text-sm transition-[background-color,color,box-shadow] duration-200",
              on ? "bg-surface font-medium text-ink shadow-sm" : "text-ink-soft hover:text-ink",
            )}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

export function Chip({
  pressed,
  children,
  onClick,
  onRemove,
  tone = "accent",
}: {
  pressed: boolean;
  children: ReactNode;
  onClick: () => void;
  onRemove?: () => void;
  tone?: "accent" | "danger";
}) {
  return (
    <button
      type="button"
      aria-pressed={pressed}
      onClick={onClick}
      className={cx(
        "inline-flex min-h-9 items-center gap-1.5 rounded-full border px-3 text-sm transition-[background-color,border-color,color] duration-200 ease-(--ease-settle)",
        "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent",
        pressed
          ? tone === "danger"
            ? "border-danger bg-danger-tint text-danger"
            : "border-accent bg-accent-tint text-accent-strong"
          : "border-line-strong bg-surface text-ink hover:border-ink/40",
      )}
    >
      {children}
      {onRemove && (
        <span
          role="button"
          tabIndex={0}
          aria-label="Remove"
          onClick={(e) => {
            e.stopPropagation();
            onRemove();
          }}
          className="-mr-1 rounded-full px-1 text-ink-soft hover:text-ink"
        >
          ×
        </span>
      )}
    </button>
  );
}

export function Meter({ value, tone = "accent", label }: { value: number; tone?: "accent" | "muted"; label?: string }) {
  const width = Math.max(0, Math.min(100, value * 100));
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-surface-2" role={label ? "meter" : undefined} aria-label={label} aria-valuenow={Math.round(width)} aria-valuemin={0} aria-valuemax={100}>
      <div
        className={cx(
          "h-full rounded-full transition-[width] duration-700 ease-(--ease-rubber)",
          tone === "accent" ? "bg-accent" : "bg-line-strong",
        )}
        style={{ width: `${width}%` }}
      />
    </div>
  );
}

export function SectionHeading({ title, detail, action }: { title: string; detail?: string; action?: ReactNode }) {
  const id = useId();
  return (
    <div className="flex flex-wrap items-end justify-between gap-3">
      <div>
        <h2 id={id} className="text-base font-semibold tracking-tight text-ink">
          {title}
        </h2>
        {detail && <p className="mt-1 text-sm text-ink-soft">{detail}</p>}
      </div>
      {action}
    </div>
  );
}

export function Checkbox({
  checked,
  onChange,
  label,
  tone = "accent",
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
  tone?: "accent" | "danger";
}) {
  return (
    <label className="flex min-h-11 cursor-pointer items-center gap-3 text-sm text-ink">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className={cx("size-5 rounded", tone === "danger" ? "accent-danger" : "accent-accent")}
      />
      <span>{label}</span>
    </label>
  );
}
