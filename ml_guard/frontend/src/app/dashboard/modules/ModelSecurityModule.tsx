"use client";
import { apiFetch, safeJson } from "@/lib/api";
import React, { useState, useEffect } from "react";
import { ShieldCheck, ShieldAlert, Shield, Lock, Unlock, AlertTriangle, AlertCircle, CheckCircle2, Search, Loader2, Info, EyeOff } from "lucide-react";


const Card = ({ children, className = "" }: any) => (
    <div className={`bg-[#0E1014] border border-white/[0.07] rounded-2xl ${className}`}>{children}</div>
);

export default function ModelSecurityModule({ state, setState, onAction }: any) {
    const [modelId, setModelId] = useState("");
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState<any>(null);
    const [error, setError] = useState<string | null>(null);

    const [historicalScans, setHistoricalScans] = useState<any[]>([]);
    const [fetchingHistory, setFetchingHistory] = useState(false);

    const runSecurityScan = async () => {
        setLoading(true); setError(null);
        try {
            // Note: In ML Guard, security checks are part of the full model audit.
            // Explain to user how to run a full audit.
            const res = await apiFetch(`/api/v1/audit/run`, {
                method: "POST",
                // This requires files, so in the UI we should redirect to Audit tab 
                // or provide a simpler 'test scan' for security.
            });
            // ... (rest of logic)
        } catch (e: any) { setError("To run a security scan, upload a model in the Audit tab with 'Security Audit' enabled. Showing latest results below."); }
        finally { setLoading(false); fetchHistory(); }
    };

    const fetchHistory = async () => {
        setFetchingHistory(true);
        try {
            const res = await apiFetch(`/api/v1/security/scans`);
            if (!res.ok) throw new Error("Failed to fetch history");
            const d = await safeJson(res);
            const scans = Array.isArray(d) ? d : [];
            setHistoricalScans(scans);
            if (scans.length > 0 && !results) {
                setResults(scans[0].security_audit_results);
            }
        } catch (e) {
            console.error("Error fetching historical scans:", e);
            setHistoricalScans([]);
        } finally { setFetchingHistory(false); }
    };

    useEffect(() => {
        fetchHistory();
        const interval = setInterval(fetchHistory, 10000); // Auto refresh every 10s
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="grid grid-cols-1 xl:grid-cols-[400px_1fr] gap-8">
            <div className="space-y-4">
                <Card className="p-8 border-red-500/20 bg-red-500/[0.03] text-center space-y-6">
                    <div className="w-16 h-16 rounded-full bg-red-500/10 flex items-center justify-center mx-auto shadow-[0_0_20px_rgba(239,68,68,0.1)]"><ShieldAlert className="w-8 h-8 text-red-500" /></div>
                    <div className="space-y-2">
                        <h3 className="text-xl font-black text-white uppercase tracking-tighter">AI Security Core</h3>
                        <p className="text-xs font-bold text-slate-500 leading-relaxed max-w-xs mx-auto">Analyze ML models for adversarial attacks, data poisoning, and potential extraction vulnerabilities.</p>
                    </div>

                    <button onClick={runSecurityScan} disabled={loading} className="w-full bg-red-600 hover:bg-red-500 disabled:opacity-50 text-white font-black py-4 rounded-xl text-[10px] uppercase tracking-widest flex items-center justify-center gap-2 transition-all shadow-lg shadow-red-500/20">
                        {loading ? <><Loader2 className="w-4 h-4 animate-spin" />Running Security Audit...</> : <><Search className="w-4 h-4" />Initiate Security Scan</>}
                    </button>
                    {error && <div className="p-4 bg-red-500/5 border border-red-500/20 rounded-xl text-red-500 font-bold text-[10px] flex gap-2"><AlertCircle className="w-3.5 h-3.5 shrink-0" /> {error}</div>}
                </Card>

                <Card className="p-6 space-y-4">
                    <h4 className="text-[10px] font-black uppercase text-slate-700 tracking-widest px-1">Security Policies</h4>
                    <div className="space-y-3">
                        <div className="flex items-center gap-4 p-4 rounded-xl bg-white/[0.02] border border-white/5"><Lock className="w-5 h-5 text-emerald-500" /><div className="flex-1"><p className="text-xs font-black text-white">Adversarial Resistance</p><p className="text-[9px] text-slate-600 mt-0.5">Detect sensitivity to input perturbations.</p></div></div>
                        <div className="flex items-center gap-4 p-4 rounded-xl bg-white/[0.02] border border-white/5"><EyeOff className="w-5 h-5 text-emerald-500" /><div className="flex-1"><p className="text-xs font-black text-white">Membership Privacy</p><p className="text-[9px] text-slate-600 mt-0.5">Prevent identification of training samples.</p></div></div>
                    </div>
                </Card>
            </div>

            <div className="space-y-6">
                {results ? (
                    <div className="space-y-6">
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                            {[
                                { label: "Poisoning Detection", score: results.results?.[0]?.score || 0, risk: results.results?.[0]?.risk_level || "LOW" },
                                { label: "Extraction Vulnerability", score: results.results?.[1]?.score || 0, risk: results.results?.[1]?.risk_level || "LOW" },
                                { label: "Membership Risk", score: results.results?.[2]?.score || 0, risk: results.results?.[2]?.risk_level || "LOW" },
                            ].map((s, i) => (
                                <Card key={i} className={`p-8 text-center border-white/5 bg-black/40`}>
                                    <p className="text-[9px] font-black text-slate-600 uppercase mb-2 tracking-[0.2em]">{s.label}</p>
                                    <p className={`text-4xl font-black ${s.risk === 'HIGH' ? 'text-red-500' : s.risk === 'MEDIUM' ? 'text-amber-500' : 'text-emerald-500'}`}>{Math.round(s.score)}%</p>
                                    <p className={`text-[8px] font-black uppercase mt-3 tracking-widest px-2 py-0.5 rounded border inline-block ${s.risk === 'HIGH' ? 'text-red-400 border-red-500/20 bg-red-500/5' : 'text-emerald-400 border-emerald-500/20 bg-emerald-500/5'}`}>{s.risk} RISK</p>
                                </Card>
                            ))}
                        </div>

                        <Card className="p-8 space-y-8">
                            <div className="flex items-center justify-between">
                                <h3 className="text-sm font-black uppercase tracking-widest text-slate-300 flex items-center gap-3"><ShieldCheck className="w-5 h-5 text-emerald-500" /> Security Vulnerability Report</h3>
                                <span className="text-[10px] font-black px-3 py-1 bg-white/[0.04] border border-white/5 text-slate-400 rounded-lg uppercase tracking-widest">Version v1.0.4 - STABLE</span>
                            </div>

                            <div className="space-y-4">
                                {(results.results || []).map((r: any, i: number) => (
                                    <div key={i} className="p-5 rounded-2xl border border-white/5 bg-black/60 flex items-start justify-between group transition-all hover:bg-white/[0.02]">
                                        <div className="space-y-3">
                                            <div className="flex items-center gap-4">
                                                <div className={`w-2 h-2 rounded-full ${r.status === 'PASS' ? "bg-emerald-500 ring-4 ring-emerald-500/10" : "bg-red-500 ring-4 ring-red-500/10"}`} />
                                                <p className="text-sm font-black text-white uppercase tracking-widest">{r.test_name || "Security Test"}</p>
                                            </div>
                                            <p className="text-xs font-medium text-slate-500 max-w-xl leading-relaxed">{r.details || "Comprehensive scan for unexpected model behavior under stress."}</p>
                                        </div>
                                        <div className="text-right flex flex-col items-end gap-2">
                                            <p className={`text-sm font-black ${r.status === 'PASS' ? 'text-emerald-500' : 'text-red-500'}`}>{r.status}</p>
                                            <div className="text-xs bg-white/[0.03] px-3 py-1 rounded-lg border border-white/5 font-mono font-bold text-slate-400">{r.score.toFixed(4)}</div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </Card>
                    </div>
                ) : (
                    <div className="flex flex-col items-center justify-center h-full min-h-[500px] text-center gap-6 bg-[#0E1014] border-2 border-dashed border-white/10 rounded-3xl relative overflow-hidden group">
                        <div className="relative z-10">
                            <Shield className="w-16 h-16 text-slate-800 group-hover:text-red-500/20 transition-all duration-700" />
                            <Lock className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-6 h-6 text-slate-600 opacity-20" />
                        </div>
                        <div className="space-y-2 relative z-10">
                            <p className="text-base font-black uppercase text-slate-700 tracking-[0.3em]">Security Vault Locked</p>
                            <p className="text-xs text-slate-800 max-w-sm font-medium leading-relaxed mx-auto">Launch a security scan to evaluate adversarial robustness and model hardening status.</p>
                        </div>
                        <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-br from-red-500/5 to-transparent pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-1000" />
                    </div>
                )}

                <div className="space-y-4">
                    <div className="flex items-center justify-between px-1">
                        <h3 className="text-[10px] font-black uppercase tracking-widest text-slate-600">Historical Security Scans</h3>
                        {fetchingHistory && <Loader2 className="w-3 h-3 animate-spin text-slate-700" />}
                    </div>
                    <div className="space-y-2">
                        {Array.isArray(historicalScans) && historicalScans.map((s) => (
                            <div key={s.scan_id} onClick={() => setResults(s.security_audit_results)}
                                className="p-4 rounded-xl border border-white/5 bg-[#0E1014] hover:border-red-500/20 transition-all cursor-pointer flex items-center justify-between group">
                                <div className="flex items-center gap-4">
                                    <div className="w-8 h-8 rounded-lg bg-red-500/10 flex items-center justify-center text-red-500"><Shield className="w-4 h-4" /></div>
                                    <div>
                                        <p className="text-xs font-black text-white uppercase tracking-tighter">Scan {s.scan_id.slice(0, 8)}</p>
                                        <p className="text-[8px] font-black text-slate-700 uppercase">{new Date(s.created_at).toLocaleString()}</p>
                                    </div>
                                </div>
                                <div className="text-right">
                                    <p className={`text-[10px] font-black ${s.risk_level === 'CRITICAL' ? 'text-red-500' : 'text-emerald-500'}`}>{s.risk_level}</p>
                                    <p className="text-[8px] font-black text-slate-800 uppercase">Risk Level</p>
                                </div>
                            </div>
                        ))}
                        {Array.isArray(historicalScans) && historicalScans.length === 0 && !fetchingHistory && <p className="text-center py-10 text-[9px] font-black text-slate-800 uppercase italic">No security scans found</p>}
                    </div>
                </div>
            </div>
        </div>
    );
}
