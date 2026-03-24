import type { Metadata } from "next";
import { Geist, Geist_Mono, Outfit } from "next/font/google";
import "./globals.css";
import { AuthProvider } from "@/context/AuthContext";
import { CursorGlow } from "@/components/landing/CursorGlow";

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

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark scroll-smooth">
      <body
        className={`${geistSans.variable} ${geistMono.variable} ${outfit.variable} antialiased font-sans bg-[#090A0C]`}
      >
        <AuthProvider>
          <CursorGlow />
          {children}
        </AuthProvider>
      </body>
    </html>
  );
}
