import type { Metadata } from "next";
import { Figtree, JetBrains_Mono, Noto_Sans } from "next/font/google";
import "./globals.css";

const figtree = Figtree({
  subsets: ["latin"],
  variable: "--font-figtree",
  weight: ["400", "500", "600", "700"],
});

const notoSans = Noto_Sans({
  subsets: ["latin"],
  variable: "--font-noto",
  weight: ["400", "500", "600", "700"],
});

const jetbrains = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains",
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: "Nidaan-Rx — Clinical Skin Triage",
  description:
    "Multimodal dermatology triage calibrated for Indian skin tones with personal health profiles, 12-gate safety checks, and Jan Aushadhi generics.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning className={`${figtree.variable} ${notoSans.variable} ${jetbrains.variable}`}>
      <body className="font-sans antialiased bg-[var(--color-canvas)] text-[var(--color-foreground)] selection:bg-[#0891B2]/15 selection:text-[#164E63]">
        {children}
      </body>
    </html>
  );
}
