"use client";

import React, { useState } from "react";
import { Loader2, Lock, Mail, ShieldCheck, User } from "lucide-react";
import { loginUser, registerUser, setAuthToken, type AuthResponse } from "@/lib/api";

interface AuthRequiredCardProps {
  onSuccess: (res: AuthResponse) => void;
}

export function AuthRequiredCard({ onSuccess }: AuthRequiredCardProps) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const res =
        mode === "register"
          ? await registerUser({ email, password, full_name: fullName })
          : await loginUser({ email, password });
      setAuthToken(res.access_token);
      setFullName("");
      setEmail("");
      setPassword("");
      onSuccess(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Authentication failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-md py-6">
      <div className="clinical-card p-7 sm:p-8">
        <div className="flex items-center gap-3.5">
          <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-[#ECFEFF] text-[#0891B2] ring-1 ring-[#A5F3FC]">
            <ShieldCheck className="h-6 w-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-[#164E63]">
              {mode === "login" ? "Sign in to Nidaan-Rx" : "Create your health account"}
            </h1>
            <p className="mt-0.5 text-xs text-[#475569]">
              Authentication is required before uploading skin photos or viewing case history.
            </p>
          </div>
        </div>

        <div className="mt-6 grid grid-cols-2 gap-1 rounded-full bg-[#E8F1F6] p-1 text-xs font-semibold">
          <button
            type="button"
            onClick={() => {
              setMode("login");
              setError(null);
            }}
            className={`cursor-pointer rounded-full py-2 transition ${
              mode === "login"
                ? "bg-white text-[#164E63] shadow-xs"
                : "text-[#475569] hover:text-[#164E63]"
            }`}
          >
            Sign In
          </button>
          <button
            type="button"
            onClick={() => {
              setMode("register");
              setError(null);
            }}
            className={`cursor-pointer rounded-full py-2 transition ${
              mode === "register"
                ? "bg-white text-[#164E63] shadow-xs"
                : "text-[#475569] hover:text-[#164E63]"
            }`}
          >
            Create Account
          </button>
        </div>

        {error && (
          <div className="mt-4 rounded-xl bg-[#FEF2F2] px-3.5 py-2.5 text-xs font-medium text-[#DC2626] ring-1 ring-[#FECACA]">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="mt-5 space-y-4">
          {mode === "register" && (
            <div>
              <label className="block text-xs font-semibold text-[#164E63]">Full Name</label>
              <div className="relative mt-1.5">
                <User className="pointer-events-none absolute left-3.5 top-3 h-4 w-4 text-[#64748B]" />
                <input
                  type="text"
                  required
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  placeholder="Your full name"
                  className="h-11 w-full rounded-xl bg-[#F3FAFC] pl-10 pr-3 text-sm text-[#164E63] ring-1 ring-[#CBEBF2] focus:bg-white focus:ring-2 focus:ring-[#0891B2] focus:outline-none"
                />
              </div>
            </div>
          )}

          <div>
            <label className="block text-xs font-semibold text-[#164E63]">Email Address</label>
            <div className="relative mt-1.5">
              <Mail className="pointer-events-none absolute left-3.5 top-3 h-4 w-4 text-[#64748B]" />
              <input
                type="email"
                required
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                className="h-11 w-full rounded-xl bg-[#F3FAFC] pl-10 pr-3 text-sm text-[#164E63] ring-1 ring-[#CBEBF2] focus:bg-white focus:ring-2 focus:ring-[#0891B2] focus:outline-none"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-[#164E63]">Password</label>
            <div className="relative mt-1.5">
              <Lock className="pointer-events-none absolute left-3.5 top-3 h-4 w-4 text-[#64748B]" />
              <input
                type="password"
                required
                minLength={6}
                autoComplete={mode === "login" ? "current-password" : "new-password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder={mode === "register" ? "At least 6 characters" : "Enter password"}
                className="h-11 w-full rounded-xl bg-[#F3FAFC] pl-10 pr-3 text-sm text-[#164E63] ring-1 ring-[#CBEBF2] focus:bg-white focus:ring-2 focus:ring-[#0891B2] focus:outline-none"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="flex h-11 w-full cursor-pointer items-center justify-center gap-2 rounded-full bg-[#059669] text-sm font-semibold text-white shadow-xs transition hover:bg-[#047857] disabled:opacity-60"
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
      </div>
    </div>
  );
}
