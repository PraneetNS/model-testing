"use client";
import React, { useState, useEffect } from "react";
import { 
    ShieldCheck, AlertCircle, XCircle, CheckCircle2, 
    Calendar, Hash, Briefcase, ExternalLink, ShieldAlert
} from "lucide-react";

export default function VerificationPage({ cert_hash }: { cert_hash: string }) {
    const [report, setReport] = useState<any>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchVerify = async () => {
            try {
                const res = await fetch(`${process.env.NEXT_PUBLIC_API_BASE}/api/v1/reports/verify/${cert_hash}`);
                const data = await res.json();
                setReport(data);
            } catch (err) {
                console.error("Verification failed", err);
            } finally {
                setLoading(false);
            }
        };
        fetchVerify();
    }, [cert_hash]);

    if (loading) return (
        <div className="min-h-screen bg-[#0E1014] flex items-center justify-center p-8">
            <div className="w-12 h-12 border-4 border-emerald-500/20 border-t-emerald-500 rounded-full animate-spin" />
        </div>
    );

    if (!report || !report.valid && !report.revoked) return (
        <div className="min-h-screen bg-[#0E1014] flex flex-col items-center justify-center p-8">
            <XCircle className="w-16 h-16 text-red-500 mb-6" />
            <h1 className="text-2xl font-black text-white uppercase tracking-widest">Invalid Certificate</h1>
            <p className="text-slate-500 text-sm mt-2 text-center max-w-md">
                The certificate hash provided could not be found in the ML Guard v7.2 verification registry. 
                Please ensure you have the correct link.
            </p>
        </div>
    );

    const isRevoked = report.revoked;

    return (
        <div className="min-h-screen bg-[#0E1014] flex items-center justify-center p-8 selection:bg-emerald-500 selection:text-white">
            <div className="max-w-2xl w-full">
                {/* Branding */}
                <div className="flex flex-col items-center mb-12 animate-in fade-in slide-in-from-top-4 duration-1000">
                    <div className="w-16 h-16 rounded-3xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center mb-6 shadow-2xl shadow-emerald-500/10">
                        <ShieldCheck className="w-8 h-8 text-emerald-500" />
                    </div>
                    <h1 className="text-3xl font-black text-white flex items-center gap-3">
                        ML GUARD v7.2
                    </h1>
                    <p className="text-slate-500 text-[10px] font-black uppercase tracking-[0.3em] mt-2">
                        Official Governance Registry
                    </p>
                </div>

                {/* Revocation Banner */}
                {isRevoked && (
                    <div className="bg-red-500/10 border border-red-500/20 rounded-2xl p-6 mb-8 flex items-start gap-4 animate-in zoom-in duration-500">
                        <ShieldAlert className="w-6 h-6 text-red-500 shrink-0" />
                        <div>
                            <h4 className="text-red-500 font-black text-[11px] uppercase tracking-widest leading-none mb-2">Certificate Revoked</h4>
                            <p className="text-red-400 text-xs font-bold leading-relaxed">
                                This certificate is no longer valid. {report.revocation_reason}
                            </p>
                        </div>
                    </div>
                )}

                {/* Main Card */}
                <div className="bg-[#14171C] border border-white/[0.05] rounded-3xl p-10 shadow-3xl shadow-black relative overflow-hidden animate-in fade-in slide-in-from-bottom-4 duration-1000">
                    {/* Visual Proof */}
                    <div className="absolute top-0 right-0 -mt-10 -mr-10 w-40 h-40 bg-emerald-500/5 rounded-full blur-3xl" />

                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-10 mb-12">
                        <div className="space-y-4">
                            <div className="space-y-1">
                                <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Model Identification</p>
                                <p className="text-2xl font-black text-white">{report.model_name}</p>
                            </div>
                            <div className="flex items-center gap-6">
                                <div className="space-y-1">
                                    <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Issued Date</p>
                                    <div className="flex items-center gap-2 text-[11px] font-black text-slate-300 uppercase">
                                        <Calendar className="w-3.5 h-3.5" />
                                        {new Date(report.issued_at).toLocaleDateString()}
                                    </div>
                                </div>
                                <div className="space-y-1">
                                    <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Governance Score</p>
                                    <div className={`text-xl font-black ${isRevoked ? "text-slate-500 line-through" : "text-emerald-400"}`}>
                                        {report.overall_score}%
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div className="flex flex-col items-center">
                            <div className={`w-32 h-32 rounded-full border-[10px] flex items-center justify-center relative ${
                                isRevoked ? "border-red-500/10" : "border-emerald-500/20"
                            }`}>
                                <div className={`absolute inset-0 rounded-full border-[2px] transition-all duration-1000 ${
                                    isRevoked ? "border-red-500 scale-105 opacity-50" : "border-emerald-500 scale-105 shadow-2xl shadow-emerald-500/20"
                                }`} style={{ clipPath: `inset(0 0 ${100 - report.overall_score}% 0)` }} />
                                <div className="text-center">
                                    <p className={`text-[10px] font-black uppercase tracking-widest mb-1 ${isRevoked ? "text-red-500" : "text-emerald-500"}`}>
                                        {isRevoked ? "REVOKED" : report.verdict}
                                    </p>
                                    {!isRevoked && <CheckCircle2 className="w-6 h-6 text-emerald-500 mx-auto" />}
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="pt-10 border-t border-white/5 grid grid-cols-1 md:grid-cols-2 gap-8">
                        <div className="space-y-2">
                            <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest flex items-center gap-2">
                                <Hash className="w-3.5 h-3.5" /> Certificate Signature
                            </p>
                            <div className="bg-black/40 border border-white/5 p-4 rounded-xl text-[10px] font-mono text-slate-400 break-all leading-snug font-bold">
                                {cert_hash}
                            </div>
                        </div>
                        <div className="space-y-2">
                            <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest flex items-center gap-2">
                                <ShieldCheck className="w-3.5 h-3.5" /> Compliance Integrity
                            </p>
                            <p className="text-xs font-bold text-slate-400 leading-relaxed italic">
                                "This certificate was issued by the ML Guard v7.2 Governance Autopilot system following a comprehensive audit session."
                            </p>
                        </div>
                    </div>
                </div>

                <p className="mt-8 text-center text-[11px] font-bold text-slate-600 flex items-center justify-center gap-6">
                    <span>TRUSTED GOVERNANCE v7.2</span>
                    <span className="w-1.5 h-1.5 rounded-full bg-slate-800" />
                    <span>SECURE BLOCKCHAIN HASHING</span>
                    <span className="w-1.5 h-1.5 rounded-full bg-slate-800" />
                    <span>ISO/IEC 42001 ALIGNMENT</span>
                </p>
            </div>
        </div>
    );
}
