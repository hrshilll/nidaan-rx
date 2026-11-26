"use client";

import React, { useEffect, useState } from "react";
import { Moon, Sun } from "lucide-react";

const THEME_KEY = "nidaan_rx_theme";

export function ThemeToggle({ className = "" }: { className?: string }) {
  const [dark, setDark] = useState(false);

  useEffect(() => {
    const saved = window.localStorage.getItem(THEME_KEY);
    const prefersDark =
      saved === "dark" ||
      (!saved && window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
    document.documentElement.classList.toggle("dark", prefersDark);
    setDark(prefersDark);
  }, []);

  const toggle = () => {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle("dark", next);
    window.localStorage.setItem(THEME_KEY, next ? "dark" : "light");
  };

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}
      title={dark ? "Switch to light mode" : "Switch to dark mode"}
      className={`relative inline-flex h-9 w-9 cursor-pointer items-center justify-center overflow-hidden rounded-full bg-white text-[#164E63] ring-1 ring-[#CBEBF2] transition-all duration-200 hover:ring-[#0891B2] active:scale-90 ${className}`}
    >
      <Sun
        className={`absolute h-4 w-4 text-[#F59E0B] transition-all duration-300 ${
          dark ? "rotate-0 scale-100 opacity-100" : "-rotate-90 scale-0 opacity-0"
        }`}
      />
      <Moon
        className={`absolute h-4 w-4 text-[#0891B2] transition-all duration-300 ${
          dark ? "rotate-90 scale-0 opacity-0" : "rotate-0 scale-100 opacity-100"
        }`}
      />
    </button>
  );
}
