"use client";

import React, { useState } from "react";
import {
  AlertTriangle,
  AtSign,
  BadgeCheck,
  Eye,
  EyeOff,
  Loader2,
  Lock,
  Mail,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  User,
} from "lucide-react";
import {
  loginUser,
  registerUser,
  setAuthToken,
  type AuthResponse,
} from "@/lib/api";
import { ThemeToggle } from "./ThemeToggle";

interface AuthScreenProps {
  onSuccess: (res: AuthResponse) => void;
  backendOffline?: boolean;
  onRetryHealth?: () => void;
}

const EMAIL_REGEX = /^[^@\s]+@[^@\s]+\.[^@\s]+$/;
const USERNAME_REGEX = /^[a-zA-Z0-9_.-]{3,30}$/;

export function AuthScreen({
  onSuccess,
  backendOffline = false,
  onRetryHealth,
}: AuthScreenProps) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [loginBy, setLoginBy] = useState<"email" | "username">("email");
  const [fullName, setFullName] = useState("");
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const switchMode = (next: "login" | "register") => {
    setMode(next);
    setError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    const cleanName = fullName.trim();
    const cleanEmail = email.trim().toLowerCase();
    const cleanUsername = username.trim().toLowerCase();

    if (mode === "register") {
      if (cleanName.length < 2 || cleanName.length > 100) {
        setError("Full name must be between 2 and 100 characters.");
        return;
      }
      if (!USERNAME_REGEX.test(cleanUsername)) {
        setError("Username must be 3–30 characters using only letters, numbers, '.', '_' or '-'.");
        return;
      }
      if (!EMAIL_REGEX.test(cleanEmail)) {
        setError("Please enter a valid email address.");
        return;
      }
      if (password.length < 6 || password.length > 128 || !password.trim()) {
        setError("Password must be 6–128 characters and cannot be blank.");
        return;
      }
    } else {
      if (loginBy === "email") {
        if (!EMAIL_REGEX.test(cleanEmail)) {
          setError("Please enter a valid email address.");
          return;
        }
      } else {
        if (!USERNAME_REGEX.test(cleanUsername)) {
          setError("Please enter a valid username (3–30 characters, no spaces).");
          return;
        }
      }
      if (!password) {
        setError("Please enter your password.");
        return;
      }
    }

    setLoading(true);
    try {
      const res =
        mode === "register"
          ? await registerUser({
              email: cleanEmail,
              username: cleanUsername,
              password,
              full_name: cleanName,
            })
          : await loginUser({
              identifier: loginBy === "email" ? cleanEmail : cleanUsername,
              login_by: loginBy,
              password,
            });

      setAuthToken(res.access_token);
      setFullName("");
      setUsername("");
      setEmail("");
      setPassword("");
      onSuccess(res);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Authentication failed.";
      if (msg.toLowerCase().includes("failed to fetch")) {
        setError(
          "Cannot reach the Nidaan-Rx API server on port 8000. Start FastAPI and retry."
        );
      } else {
        setError(msg);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen flex-col justify-between bg-[var(--color-canvas)] px-4 py-8 text-[#164E63] sm:px-6 lg:px-8">
      <div className="mx-auto flex w-full max-w-4xl justify-end pb-3">
        <ThemeToggle />
      </div>

      <div className="mx-auto flex w-full max-w-4xl flex-1 flex-col justify-center">
        {backendOffline && (
          <div className="mb-6 flex flex-wrap items-center justify-between gap-3 rounded-2xl bg-[#FFFBEB] px-4 py-3 text-xs text-[#92400E] ring-1 ring-[#FDE68A]">
            <div className="flex items-start gap-2.5">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-[#D97706]" />
              <div>
                <p className="font-semibold">Backend server offline on port 8000</p>
                <code className="mt-0.5 block font-mono text-[11px]">
                  PYTHONPATH=backend uvicorn app.main:app --reload --port 8000
                </code>
              </div>
            </div>
            {onRetryHealth && (
              <button
                type="button"
                onClick={onRetryHealth}
                className="inline-flex cursor-pointer items-center gap-1.5 rounded-full bg-[#D97706] px-3.5 py-1.5 font-semibold text-white transition hover:bg-[#B45309]"
              >
                <RefreshCw className="h-3.5 w-3.5" /> Reconnect
              </button>
            )}
          </div>
        )}

        <div className="clinical-card overflow-hidden">
          <div className="grid grid-cols-1 md:grid-cols-12">
            <div className="flex flex-col justify-between bg-gradient-to-br from-[#0E7490] via-[#0891B2] to-[#047857] p-7 text-white sm:p-9 md:col-span-5">
              <div>
                <div className="inline-flex items-center gap-2.5">
                  <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-white/15 text-sm font-bold tracking-tight text-white ring-1 ring-white/25 backdrop-blur-sm">
                    NRx
                  </div>
                  <span className="font-heading text-lg font-bold tracking-tight">
                    Nidaan-Rx
                  </span>
                </div>

                <h1 className="mt-7 font-heading text-2xl font-bold leading-snug tracking-tight sm:text-[26px]">
                  Clinical skin triage built for Indian skin tones.
                </h1>
                <p className="mt-2.5 text-[13px] leading-relaxed text-white/85">
                  Sign in to access your private health profile, multi-angle skin triage, and downloadable clinical summaries.
                </p>

                <ul className="mt-7 space-y-3.5 text-xs text-white/95">
                  <li className="flex items-start gap-2.5">
                    <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-[#A7F3D0]" />
                    <span>
                      <strong className="font-semibold">12-Gate Clinical Safety:</strong>{" "}
                      Automated allergy, DDI, site-sensitivity, and urgent red-flag checks.
                    </span>
                  </li>
                  <li className="flex items-start gap-2.5">
                    <BadgeCheck className="mt-0.5 h-4 w-4 shrink-0 text-[#A7F3D0]" />
                    <span>
                      <strong className="font-semibold">Jan Aushadhi (PMBJP):</strong>{" "}
                      Exact-salt verified generic equivalents with transparent MRP savings.
                    </span>
                  </li>
                  <li className="flex items-start gap-2.5">
                    <Sparkles className="mt-0.5 h-4 w-4 shrink-0 text-[#A7F3D0]" />
                    <span>
                      <strong className="font-semibold">Private Health Vault:</strong>{" "}
                      Your skin tone, allergies, meds, and triage history stay isolated to your account.
                    </span>
                  </li>
                </ul>
              </div>

              <p className="mt-8 border-t border-white/15 pt-4 text-[11px] leading-relaxed text-white/75">
                Decision-support prototype for educational triage — not a substitute for an in-person dermatologist.
              </p>
            </div>

            <div className="p-7 sm:p-9 md:col-span-7">
              <div>
                <h2 className="font-heading text-2xl font-bold tracking-tight text-[#164E63]">
                  {mode === "login" ? "Welcome back" : "Create your account"}
                </h2>
                <p className="mt-1 text-xs text-[#475569]">
                  {mode === "login"
                    ? "Sign in with your email or username to continue."
                    : "Register once, then complete your baseline health profile."}
                </p>
              </div>

              <div
                role="tablist"
                aria-label="Authentication mode"
                className="mt-6 grid grid-cols-2 gap-1 rounded-full bg-[#E8F1F6] p-1 text-xs font-semibold"
              >
                <button
                  type="button"
                  role="tab"
                  aria-selected={mode === "login"}
                  onClick={() => switchMode("login")}
                  className={`cursor-pointer rounded-full py-2.5 transition ${
                    mode === "login"
                      ? "bg-white text-[#164E63] shadow-xs"
                      : "text-[#475569] hover:text-[#164E63]"
                  }`}
                >
                  Sign In
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={mode === "register"}
                  onClick={() => switchMode("register")}
                  className={`cursor-pointer rounded-full py-2.5 transition ${
                    mode === "register"
                      ? "bg-white text-[#164E63] shadow-xs"
                      : "text-[#475569] hover:text-[#164E63]"
                  }`}
                >
                  Create Account
                </button>
              </div>

              {error && (
                <div
                  role="alert"
                  className="mt-5 rounded-xl bg-[#FEF2F2] px-4 py-3 text-xs font-medium text-[#DC2626] ring-1 ring-[#FECACA]"
                >
                  {error}
                </div>
              )}

              <form onSubmit={handleSubmit} className="mt-6 space-y-4">
                {mode === "login" && (
                  <div>
                    <span className="block text-xs font-semibold text-[#164E63]">Sign in using</span>
                    <div className="mt-1.5 grid grid-cols-2 gap-1 rounded-xl bg-[#E8F1F6] p-1 text-xs font-semibold">
                      <button
                        type="button"
                        onClick={() => { setLoginBy("email"); setError(null); }}
                        className={`cursor-pointer rounded-lg py-1.5 transition ${
                          loginBy === "email"
                            ? "bg-white text-[#164E63] shadow-xs"
                            : "text-[#475569] hover:text-[#164E63]"
                        }`}
                      >
                        Email
                      </button>
                      <button
                        type="button"
                        onClick={() => { setLoginBy("username"); setError(null); }}
                        className={`cursor-pointer rounded-lg py-1.5 transition ${
                          loginBy === "username"
                            ? "bg-white text-[#164E63] shadow-xs"
                            : "text-[#475569] hover:text-[#164E63]"
                        }`}
                      >
                        Username
                      </button>
                    </div>
                  </div>
                )}

                {mode === "register" && (
                  <div>
                    <label htmlFor="auth-fullname" className="block text-xs font-semibold text-[#164E63]">
                      Full Name
                    </label>
                    <div className="relative mt-1.5">
                      <User className="pointer-events-none absolute left-3.5 top-3 h-4 w-4 text-[#64748B]" />
                      <input
                        id="auth-fullname"
                        type="text"
                        required
                        minLength={2}
                        maxLength={100}
                        autoComplete="name"
                        value={fullName}
                        onChange={(e) => setFullName(e.target.value)}
                        placeholder="Enter your full name"
                        className="h-11 w-full rounded-xl bg-[#F3FAFC] pl-10 pr-3.5 text-sm text-[#164E63] ring-1 ring-[#CBEBF2] placeholder:text-[#64748B] focus:bg-white focus:ring-2 focus:ring-[#0891B2] focus:outline-none"
                      />
                    </div>
                  </div>
                )}

                {(mode === "register" || loginBy === "username") && (
                  <div>
                    <label htmlFor="auth-username" className="block text-xs font-semibold text-[#164E63]">
                      Username
                    </label>
                    <div className="relative mt-1.5">
                      <AtSign className="pointer-events-none absolute left-3.5 top-3 h-4 w-4 text-[#64748B]" />
                      <input
                        id="auth-username"
                        type="text"
                        required
                        minLength={3}
                        maxLength={30}
                        pattern="[a-zA-Z0-9_.-]{3,30}"
                        title="3–30 characters: letters, numbers, '.', '_' or '-'"
                        autoComplete="username"
                        value={username}
                        onChange={(e) => setUsername(e.target.value.replace(/\s+/g, ""))}
                        placeholder="e.g. harshil_01"
                        className="h-11 w-full rounded-xl bg-[#F3FAFC] pl-10 pr-3.5 text-sm text-[#164E63] ring-1 ring-[#CBEBF2] placeholder:text-[#64748B] focus:bg-white focus:ring-2 focus:ring-[#0891B2] focus:outline-none"
                      />
                    </div>
                  </div>
                )}

                {(mode === "register" || loginBy === "email") && (
                  <div>
                    <label htmlFor="auth-email" className="block text-xs font-semibold text-[#164E63]">
                      Email Address
                    </label>
                    <div className="relative mt-1.5">
                      <Mail className="pointer-events-none absolute left-3.5 top-3 h-4 w-4 text-[#64748B]" />
                      <input
                        id="auth-email"
                        type="email"
                        required
                        maxLength={160}
                        autoComplete="email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        placeholder="you@example.com"
                        className="h-11 w-full rounded-xl bg-[#F3FAFC] pl-10 pr-3.5 text-sm text-[#164E63] ring-1 ring-[#CBEBF2] placeholder:text-[#64748B] focus:bg-white focus:ring-2 focus:ring-[#0891B2] focus:outline-none"
                      />
                    </div>
                  </div>
                )}

                <div>
                  <label htmlFor="auth-password" className="block text-xs font-semibold text-[#164E63]">
                    Password
                  </label>
                  <div className="relative mt-1.5">
                    <Lock className="pointer-events-none absolute left-3.5 top-3 h-4 w-4 text-[#64748B]" />
                    <input
                      id="auth-password"
                      type={showPassword ? "text" : "password"}
                      required
                      minLength={6}
                      maxLength={128}
                      autoComplete={mode === "login" ? "current-password" : "new-password"}
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      placeholder={mode === "register" ? "Minimum 6 characters" : "Enter your password"}
                      className="h-11 w-full rounded-xl bg-[#F3FAFC] pl-10 pr-11 text-sm text-[#164E63] ring-1 ring-[#CBEBF2] placeholder:text-[#64748B] focus:bg-white focus:ring-2 focus:ring-[#0891B2] focus:outline-none"
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword((v) => !v)}
                      aria-label={showPassword ? "Hide password" : "Show password"}
                      className="absolute right-2.5 top-2 flex h-7 w-7 cursor-pointer items-center justify-center rounded-lg text-[#64748B] transition hover:bg-[#E8F1F6] hover:text-[#164E63]"
                    >
                      {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </button>
                  </div>
                </div>

                <button
                  type="submit"
                  disabled={loading}
                  className="mt-2 flex h-11 w-full cursor-pointer items-center justify-center gap-2 rounded-full bg-[#059669] text-sm font-semibold text-white shadow-xs transition hover:bg-[#047857] disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {loading ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      {mode === "login" ? "Signing in..." : "Creating account..."}
                    </>
                  ) : mode === "login" ? (
                    "Sign In"
                  ) : (
                    "Create Account"
                  )}
                </button>
              </form>

              <p className="mt-6 text-center text-xs text-[#475569]">
                {mode === "login" ? (
                  <>
                    New to Nidaan-Rx?{" "}
                    <button
                      type="button"
                      onClick={() => switchMode("register")}
                      className="cursor-pointer font-semibold text-[#0891B2] hover:underline"
                    >
                      Create an account
                    </button>
                  </>
                ) : (
                  <>
                    Already have an account?{" "}
                    <button
                      type="button"
                      onClick={() => switchMode("login")}
                      className="cursor-pointer font-semibold text-[#0891B2] hover:underline"
                    >
                      Sign in instead
                    </button>
                  </>
                )}
              </p>
            </div>
          </div>
        </div>
      </div>

      <footer className="mt-8 text-center text-[11px] text-[#64748B]">
        Nidaan-Rx v2.0 · Session-Scoped Auth · Private Clinical Workspace
      </footer>
    </div>
  );
}
