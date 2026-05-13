import type { Metadata } from "next";
import { Geist, Geist_Mono, Outfit } from "next/font/google";
import "./globals.css";
import { AuthProvider } from "@/context/AuthContext";
import { CursorGlow } from "@/components/landing/CursorGlow";
import Script from "next/script";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const outfit = Outfit({
  variable: "--font-outfit",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "ML Guard | Enterprise ML Quality Governance",
  description: "Automated quality gates, drift telemetry, and regulatory compliance auditing for production ML models.",
};

import { headers } from "next/headers";

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const nonce = (await headers()).get('x-nonce') || "";

  return (
    <html lang="en" className="dark scroll-smooth">
      <body
        className={`${geistSans.variable} ${geistMono.variable} ${outfit.variable} antialiased font-sans bg-[#090A0C]`}
      >
        {/* Global safeJson polyfill — used across many components without local imports */}
        <Script id="safeJson-polyfill" strategy="beforeInteractive">{`
          window.safeJson = async function safeJson(res) {
            if (res.status === 204) return {};
            var text = await res.text();
            return text ? JSON.parse(text) : {};
          };
        `}</Script>
        <AuthProvider>
          <CursorGlow />
          {children}
        </AuthProvider>
      </body>
    </html>
  );
}
