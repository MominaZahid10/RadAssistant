/**
 * RadAssistant — Root Layout
 *
 * The top-level wrapper for every page. Fonts and metadata are defined once
 * here; the shell (sidebar + main column) and the conversation state live in
 * AppShell, which is a client component because both depend on browser
 * storage and on the current route.
 */

import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import AppShell from "@/components/AppShell";
import "./globals.css";

// Next.js loads Google Fonts at BUILD time, not runtime: no layout shift,
// no extra network request.
const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "RadAssistant — Radiology Reporting Assistant",
  description:
    "An explainable, multimodal RAG platform for radiology report generation and clinical decision support.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="h-full">
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
