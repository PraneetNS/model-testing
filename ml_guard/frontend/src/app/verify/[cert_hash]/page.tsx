"use client";
import React, { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import {
  Shield, ShieldCheck, ShieldAlert,
  CheckCircle2, XCircle, AlertTriangle,
  Activity,
} from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

interface VerifyData {
  cert_hash: string;
  valid: boolean;
  still_compliant: boolean;
  issued_at: string | null;
  verdict: string;
  overall_score: number | null;
  is_revoked: boolean;
  revocation_reason: string | null;
  drift_events_since_issue: number;
  message: string;
  verified_at: string;
}

export default function VerifyPage() {
  const params = useParams();
  const cert_hash = params?.cert_hash as string;
  const [data, setData] = useState<VerifyData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!cert_hash) return;
    // Public endpoint — no X-API-Key required
    fetch(`${API_BASE}/api/v1/governance/verify/${cert_hash}`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return safeJson(r);
      })
      .then((d) => setData(d))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [cert_hash]);

  const verdictColor =
    data?.verdict === "CERTIFIED"
      ? "text-emerald-400"
      : data?.verdict === "CONDITIONAL"
      ? "text-amber-400"
      : "text-red-400";

  const verdictBg =
    data?.verdict === "CERTIFIED"
      ? "bg-emerald-500/10 border-emerald-500/20"
      : data?.verdict === "CONDITIONAL"
      ? "bg-amber-500/10 border-amber-500/20"
      : "bg-red-500/10 border-red-500/20";

  const VerdictIcon =
    data?.verdict === "CERTIFIED"
      ? ShieldCheck
      : data?.verdict === "CONDITIONAL"
      ? ShieldAlert
      : Shield;

  const details = data
    ? [
        {
          label: "Valid",
          value: data.valid ? "Yes" : "No",
          color: data.valid ? "text-emerald-400" : "text-red-400",
        },
        {
          label: "Still Compliant",
          value: data.still_compliant ? "Yes" : "No",
          color: data.still_compliant ? "text-emerald-400" : "text-amber-400",
        },
        {
          label: "Score",
          value: data.overall_score != null ? `${data.overall_score.toFixed(1)}/100` : "N/A",
          color: "text-white",
        },
        {
          label: "Issued",
          value: data.issued_at ? new Date(data.issued_at).toLocaleDateString() : "N/A",
          color: "text-slate-300",
        },
        {
          label: "Drift Events Since Issue",
          value: String(data.drift_events_since_issue ?? 0),
          color: (data.drift_events_since_issue ?? 0) > 0 ? "text-amber-400" : "text-emerald-400",
        },
        {
          label: "Revoked",
          value: data.is_revoked ? "Yes" : "No",
          color: data.is_revoked ? "text-red-400" : "text-emerald-400",
        },
      ]
    : [];

  return (
    <main className="min-h-screen bg-[#090A0C] flex items-center justify-center p-6">
      {/* Ambient glow */}
      <div className="fixed top-0 right-0 w-1/2 h-1/2 bg-orange-500/5 blur-[120px] pointer-events-none rounded-full" />

      <div className="w-full max-w-lg space-y-6 relative z-10">
        {/* Header */}
        <div className="text-center space-y-2">
          <div className="flex items-center justify-center gap-2.5 mb-5">
            <div className="w-10 h-10 rounded-2xl bg-orange-600 flex items-center justify-center shadow-lg shadow-orange-500/20">
              <Shield className="w-5 h-5 text-black" />
            </div>
            <span className="text-white font-black text-xl tracking-tight">ML Guard</span>
          </div>
          <h1 className="text-white font-black text-2xl">Certificate Verification</h1>
          <p className="text-slate-500 text-sm leading-relaxed">
            Verify the authenticity and compliance status of an ML governance certificate
          </p>
        </div>

        {/* Certificate hash display */}
        <div className="bg-[#0E1014] border border-white/[0.06] rounded-2xl p-4">
          <p className="text-[9px] font-black uppercase tracking-widest text-slate-600 mb-1">Certificate Hash</p>
          <p className="font-mono text-xs text-slate-400 break-all">{cert_hash}</p>
        </div>

        {/* Loading */}
        {loading && (
          <div className="bg-[#0E1014] border border-white/[0.06] rounded-2xl p-10 text-center">
            <div className="w-8 h-8 border-2 border-orange-400 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
            <p className="text-slate-500 text-sm">Verifying certificate…</p>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="bg-red-500/10 border border-red-500/20 rounded-2xl p-4 flex items-start gap-3 text-red-400 text-sm">
            <XCircle className="w-4 h-4 shrink-0 mt-0.5" />
            {error}
          </div>
        )}

        {/* Result */}
        {data && !loading && (
          <div className="space-y-4">
            {/* Main verdict card */}
            <div className={`border rounded-2xl p-8 text-center ${verdictBg}`}>
              <VerdictIcon className={`w-14 h-14 mx-auto mb-4 ${verdictColor}`} />
              <p className={`text-3xl font-black ${verdictColor}`}>{data.verdict || "INVALID"}</p>
              <p className="text-slate-400 text-sm mt-2 leading-relaxed">{data.message}</p>
            </div>

            {/* Details table */}
            <div className="bg-[#0E1014] border border-white/[0.06] rounded-2xl p-5 space-y-0">
              {details.map(({ label, value, color }, i) => (
                <div
                  key={label}
                  className={`flex justify-between items-center py-3 ${
                    i < details.length - 1 ? "border-b border-white/[0.04]" : ""
                  }`}
                >
                  <p className="text-[10px] font-black uppercase tracking-widest text-slate-600">{label}</p>
                  <p className={`text-sm font-black ${color}`}>{value}</p>
                </div>
              ))}
            </div>

            {/* Drift warning */}
            {(data.drift_events_since_issue ?? 0) > 0 && (
              <div className="bg-amber-500/10 border border-amber-500/20 rounded-2xl p-4 flex items-start gap-3">
                <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                <p className="text-amber-400 text-xs leading-relaxed">
                  <span className="font-black">{data.drift_events_since_issue} production drift event(s)</span>{" "}
                  detected since this certificate was issued. Re-audit is recommended to maintain compliance.
                </p>
              </div>
            )}

            {/* Revoked warning */}
            {data.is_revoked && (
              <div className="bg-red-500/10 border border-red-500/20 rounded-2xl p-4 flex items-start gap-3">
                <XCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                <div>
                  <p className="text-red-400 text-xs font-black">Certificate Revoked</p>
                  {data.revocation_reason && (
                    <p className="text-red-400/80 text-xs mt-1">{data.revocation_reason}</p>
                  )}
                </div>
              </div>
            )}

            {/* Validity check indicator */}
            <div className="bg-[#0E1014] border border-white/[0.06] rounded-2xl p-4 flex items-center gap-3">
              <Activity className="w-4 h-4 text-slate-600" />
              <div>
                <p className="text-[9px] font-black uppercase tracking-widest text-slate-600">Verified At</p>
                <p className="text-xs text-slate-400 mt-0.5">
                  {data.verified_at
                    ? new Date(data.verified_at).toLocaleString()
                    : new Date().toLocaleString()}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="text-center space-y-1.5 pt-2">
          <p className="text-slate-700 text-[10px] uppercase tracking-widest">
            Verified by ML Guard v7.2
          </p>
          <a
            href="/"
            className="text-orange-400/40 text-[10px] hover:text-orange-400 transition-colors inline-block"
          >
            ← Return to ML Guard
          </a>
        </div>
      </div>
    </main>
  );
}
