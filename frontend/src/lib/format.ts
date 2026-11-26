export const inr = (n: number) =>
  new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 }).format(n);

export const pct = (p: number, digits = 1) => `${(p * 100).toFixed(digits)}%`;

export const humanize = (id: string) =>
  id.replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim();

export const cx = (...classes: (string | false | null | undefined)[]) =>
  classes.filter(Boolean).join(" ");

export const errorText = (e: unknown) => (e instanceof Error ? e.message : "Something went wrong.");

import type { CSSProperties } from "react";

// Staggered entrance delay for the .rise utility (CSS custom property).
export const delay = (ms: number) => ({ "--delay": `${ms}ms` }) as CSSProperties;
