"use client";

import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Loader2 } from "lucide-react";
import { clearAuthToken, fetchHealth, fetchMe, getAuthToken } from "@/lib/api";
import { AuthScreen } from "@/components/v2/AuthScreen";

export default function LoginPage() {
  const router = useRouter();
  const [checking, setChecking] = useState(true);
  const [offline, setOffline] = useState(false);

  const checkSession = async () => {
    try {
      await fetchHealth();
      setOffline(false);
      if (getAuthToken()) {
        try {
          await fetchMe();
          router.replace("/");
          return;
        } catch {
          clearAuthToken();
        }
      }
    } catch {
      setOffline(true);
    } finally {
      setChecking(false);
    }
  };

  useEffect(() => {
    checkSession();
  }, []);

  if (checking) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[var(--color-canvas)] text-sm text-[#475569]">
        <Loader2 className="mr-2 h-5 w-5 animate-spin text-[#0891B2]" />
        Checking session...
      </div>
    );
  }

  return (
    <AuthScreen
      backendOffline={offline}
      onRetryHealth={checkSession}
      onSuccess={() => router.replace("/")}
    />
  );
}
