/**
 * RadAssist AI — Root Layout
 * 
 * This is the TOP-LEVEL wrapper for every page in the app.
 * 
 * In Next.js App Router:
 * - layout.tsx wraps ALL pages (like a picture frame)
 * - page.tsx is the actual content inside the frame
 * - {children} is where the page content gets inserted
 * 
 * WHY IS THIS IMPORTANT?
 * The sidebar, fonts, and metadata are defined ONCE here
 * and automatically apply to every page in the app.
 * You never have to import the sidebar in individual pages.
 */

import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Sidebar from "@/components/Sidebar";
import "./globals.css";

// ── Font Loading ────────────────────────────────────────────
// Next.js loads Google Fonts at BUILD time (not runtime).
// This means: no layout shift, no extra network requests.
const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

// ── SEO Metadata ────────────────────────────────────────────
// This sets the <title> and <meta> tags in the HTML <head>.
// Important for search engines and browser tabs.
export const metadata: Metadata = {
  title: "RadAssist AI — Radiology Reporting Assistant",
  description:
    "An explainable, multimodal RAG platform for radiology report generation and clinical decision support.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex bg-background">
        {/* Sidebar — always visible on the left */}
        <Sidebar />

        {/* Main content area — offset by sidebar width */}
        <main className="flex-1 ml-[260px] min-h-screen transition-all duration-300">
          {children}
        </main>
      </body>
    </html>
  );
}
