"use client";
import React, { useState, useEffect } from "react";
import { apiFetch, safeJson } from "@/lib/api";
import { FileText, ShieldCheck, Download, AlertCircle, CheckCircle2, Loader2, Link } from "lucide-react";

export default function ComplianceModule({ modelId = "" }) {
    const [selectedModel, setSelectedModel] = useState(modelId);
    const [pack, setPack] = useState("sr_11_7");
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState<any>(null);
    const [error, setError] = useState<string | null>(null);
    const [availableModels, setAvailableModels] = useState<any[]>([]);

    useEffect(() => {
        const fetchModels = async () => {
            try {
                const res = await apiFetch("/api/inventory");
                const data = await safeJson<any>(res);
                const list = Array.isArray(data) ? data : (data?.items || []);
                setAvailableModels(list);
            } catch (err) {
                console.error("Compliance fetch error:", err);
                setError("Failed to load models list. Please try refreshing.");
            }
        };
        fetchModels();
    }, []);

    const runPack = async () => {
        if (!selectedModel) {
            setError("Please select a model first.");
            return;
        }
        setLoading(true);
        setError(null);
        setResults(null);
        try {
            const res = await apiFetch(`/api/compliance/${selectedModel}/pack/${pack}`);
            const data = await safeJson<any>(res);
            if (!res.ok) {
                if (data.detail?.error === "usage_limit_reached") {
                    throw new Error(`Usage limit reached for ${data.detail.event_type}. Please upgrade your plan.`);
                }
                const msg = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail || data);
                throw new Error(msg);
            }
            setResults(data);
        } catch (e: any) {
            setError(e.message || String(e));
        } finally {
            setLoading(false);
        }
    };

    const downloadPDF = async () => {
        if (!selectedModel || !pack) return;
        try {
            const url = `${process.env.NEXT_PUBLIC_API_BASE || ""}/api/compliance/${selectedModel}/pack/${pack}/pdf`;
            window.open(url, "_blank");
        } catch (e) {
            console.error(e);
        }
    };

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-xl font-black text-white">Vertical Compliance Packs</h2>
                    <p className="text-sm text-slate-400">Run regulatory checks and generate enterprise reports</p>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-[300px_1fr] gap-6">
                <div className="space-y-4">
                    <div className="bg-[#0E1014] border border-white/[0.07] rounded-xl p-5 space-y-4">
                        <div>
                            <label className="text-[10px] font-black uppercase text-slate-500 mb-1 block">Select Model</label>
                            <select
                                value={selectedModel}
                                onChange={(e) => setSelectedModel(e.target.value)}
                                className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-orange-500/50"
                            >
                                <option value="">-- Choose Model --</option>
                                {availableModels.map(m => (
                                    <option key={m.model_id || m.id} value={m.model_id || m.id}>{m.name} (v{m.latest_version ?? '1.0'})</option>
                                ))}
                            </select>
                        </div>
                        
                        <div>
                            <label className="text-[10px] font-black uppercase text-slate-500 mb-1 block">Compliance Pack</label>
                            <div className="space-y-2">
                                {[
                                    { id: "sr_11_7", label: "SR 11-7 (US Fed)" },
                                    { id: "eu_ai_act", label: "EU AI Act" },
                                    { id: "rbi_mlrg", label: "RBI MLRG (India)" },
                                    { id: "fda_ai", label: "FDA AI Guidance" }
                                ].map(p => (
                                    <div 
                                        key={p.id}
                                        onClick={() => setPack(p.id)}
                                        className={`px-3 py-2 border rounded-lg cursor-pointer transition-colors ${pack === p.id ? "bg-orange-500/10 border-orange-500/30 text-orange-400 font-bold" : "bg-black/20 border-white/5 text-slate-400 hover:bg-white/5"}`}
                                    >
                                        <span className="text-xs">{p.label}</span>
                                    </div>
                                ))}
                            </div>
                        </div>

                        <button
                            onClick={runPack}
                            disabled={loading}
                            className="w-full py-3 bg-orange-600 hover:bg-orange-500 disabled:opacity-50 text-black font-black uppercase tracking-widest text-[10px] rounded-lg transition-all flex justify-center items-center gap-2"
                        >
                            {loading ? <><Loader2 className="w-4 h-4 animate-spin" /> Running</> : <><ShieldCheck className="w-4 h-4" /> Run Pack</>}
                        </button>
                        
                        {error && (
                            <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg flex items-start gap-2">
                                <AlertCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                                <p className="text-xs text-red-400">{error}</p>
                            </div>
                        )}
                    </div>
                </div>

                <div>
                    {results ? (
                        <div className="space-y-6">
                            <div className="flex items-center justify-between bg-[#0E1014] border border-white/[0.07] rounded-xl p-6">
                                <div className="flex items-center gap-6">
                                    <div className="relative">
                                        <svg width="80" height="80" viewBox="0 0 120 120" className="-rotate-90">
                                            <circle cx="60" cy="60" r="50" fill="none" stroke="#ffffff08" strokeWidth="10" />
                                            <circle 
                                                cx="60" cy="60" r="50" fill="none" 
                                                stroke={results.score >= 80 ? "#22c55e" : results.score >= 60 ? "#f59e0b" : "#ef4444"} 
                                                strokeWidth="10" 
                                                strokeDasharray={2 * Math.PI * 50} 
                                                strokeDashoffset={2 * Math.PI * 50 * (1 - results.score / 100)} 
                                                strokeLinecap="round"
                                            />
                                        </svg>
                                        <div className="absolute inset-0 flex flex-col items-center justify-center">
                                            <span className={`text-xl font-black ${results.score >= 80 ? "text-emerald-400" : results.score >= 60 ? "text-amber-400" : "text-red-400"}`}>{Math.round(results.score)}</span>
                                        </div>
                                    </div>
                                    <div>
                                        <p className="text-[10px] uppercase font-black tracking-widest text-slate-500">Compliance Score</p>
                                        <p className="text-lg font-bold text-white">{results.status === 'compliant' ? 'FULLY COMPLIANT' : results.status === 'partial' ? 'PARTIALLY COMPLIANT' : 'NON-COMPLIANT'}</p>
                                    </div>
                                </div>
                                <button
                                    onClick={downloadPDF}
                                    className="px-4 py-2 border border-white/10 hover:border-white/30 hover:bg-white/5 rounded-lg text-xs font-bold text-white flex items-center gap-2 transition-all"
                                >
                                    <Download className="w-4 h-4" /> Download PDF Report
                                </button>
                            </div>

                            <div className="space-y-3">
                                {results.checks.map((c: any, i: number) => (
                                    <div key={i} className={`bg-[#0E1014] border rounded-xl p-5 ${c.status === "pass" ? "border-emerald-500/20 bg-emerald-500/5" : "border-red-500/20 bg-red-500/5"}`}>
                                        <div className="flex items-start justify-between mb-3">
                                            <div className="flex items-center gap-3">
                                                {c.status === "pass" ? <CheckCircle2 className="w-5 h-5 text-emerald-400" /> : <AlertCircle className="w-5 h-5 text-red-400" />}
                                                <div>
                                                    <p className="text-[10px] font-black uppercase text-slate-500 tracking-widest">{c.article}</p>
                                                    <h4 className="text-sm font-bold text-white">{c.title}</h4>
                                                </div>
                                            </div>
                                            <span className={`px-2 py-1 rounded text-[9px] font-black uppercase ${c.status === "pass" ? "bg-emerald-500/20 text-emerald-400" : "bg-red-500/20 text-red-400"}`}>
                                                {c.status}
                                            </span>
                                        </div>
                                        
                                        <div className="space-y-2 text-xs">
                                            <div className="flex gap-2">
                                                <span className="text-slate-500 w-24 shrink-0 font-medium">Evidence:</span>
                                                <span className="text-slate-300 font-mono bg-black/20 px-2 py-0.5 rounded break-all">
                                                    {typeof c.evidence === 'string' ? c.evidence : JSON.stringify(c.evidence)}
                                                </span>
                                            </div>
                                            {c.status !== "pass" && c.remediation && (
                                                <div className="flex gap-2">
                                                    <span className="text-slate-500 w-24 shrink-0 font-medium">Remediation:</span>
                                                    <span className="text-amber-400">
                                                        {typeof c.remediation === 'string' ? c.remediation : JSON.stringify(c.remediation)}
                                                    </span>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ) : (
                        <div className="h-full min-h-[400px] flex flex-col items-center justify-center border border-white/[0.05] rounded-xl bg-[#0E1014]/50 border-dashed">
                            <ShieldCheck className="w-12 h-12 text-slate-800 mb-4" />
                            <p className="text-slate-500 text-sm font-medium">Select a model and compliance pack to run checks.</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
