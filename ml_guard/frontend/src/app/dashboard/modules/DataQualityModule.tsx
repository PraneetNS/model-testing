"use client";
import React, { useState } from "react";
import { ShieldCheck, AlertCircle, CheckCircle2, Search, Database, Upload, Loader2, BarChart3, Info } from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

const Card = ({ children, className = "" }: any) => (
    <div className={`bg-[#0E1014] border border-white/[0.07] rounded-2xl ${className}`}>{children}</div>
);

export default function DataQualityModule({ state, setState, onAction }: any) {
    const [dataFile, setDataFile] = useState<File | null>(null);
    const [refFile, setRefFile] = useState<File | null>(null);
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState<any>(null);
    const [error, setError] = useState<string | null>(null);

    const runValidation = async () => {
        if (!dataFile) { setError("Dataset file required."); return; }
        setLoading(true); setError(null);
        const fd = new FormData();
        fd.append("dataset_file", dataFile);
        if (refFile) fd.append("reference_file", refFile);

        try {
            const res = await fetch(`${API_BASE}/api/v1/data-quality/validate`, { method: "POST", body: fd });
            const d = await res.json();
            if (!res.ok) throw new Error(d.detail || "Validation failed.");
            setResults(d);
        } catch (e: any) { setError(e.message); } finally { setLoading(false); }
    };

    return (
        <div className="grid grid-cols-1 xl:grid-cols-[400px_1fr] gap-8">
            <div className="space-y-4">
                <Card className="p-6 space-y-6">
                    <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-lg bg-emerald-500/10 flex items-center justify-center"><Search className="w-4 h-4 text-emerald-400" /></div>
                        <h3 className="text-xs font-black uppercase tracking-widest text-white">Quality Scan</h3>
                    </div>

                    <div className="space-y-4">
                        <div className="space-y-2">
                            <p className="text-[9px] font-black uppercase tracking-widest text-slate-500">Target Dataset (CSV)</p>
                            <label className="block p-4 rounded-xl border border-white/5 bg-black/20 cursor-pointer hover:border-emerald-500/20 transition-all">
                                <div className="flex items-center gap-3">
                                    <Database className={`w-4 h-4 ${dataFile ? "text-emerald-400" : "text-slate-600"}`} />
                                    <span className="text-xs font-bold text-slate-400 truncate">{dataFile ? dataFile.name : "Select CSV to scan"}</span>
                                </div>
                                <input type="file" className="hidden" onChange={e => e.target.files?.[0] && setDataFile(e.target.files[0])} />
                            </label>
                        </div>
                        <div className="space-y-2">
                            <p className="text-[9px] font-black uppercase tracking-widest text-slate-500">Reference (Optional)</p>
                            <label className="block p-4 rounded-xl border border-white/5 bg-black/20 cursor-pointer hover:border-emerald-500/20 transition-all opacity-60">
                                <div className="flex items-center gap-3">
                                    <Database className={`w-4 h-4 ${refFile ? "text-emerald-400" : "text-slate-600"}`} />
                                    <span className="text-xs font-bold text-slate-400 truncate">{refFile ? refFile.name : "Baseline for drift"}</span>
                                </div>
                                <input type="file" className="hidden" onChange={e => e.target.files?.[0] && setRefFile(e.target.files[0])} />
                            </label>
                        </div>
                    </div>

                    {error && <div className="p-4 bg-red-500/5 border border-red-500/20 rounded-xl text-red-400 font-bold text-xs flex gap-2"><AlertCircle className="w-4 h-4 shrink-0" /> {error}</div>}

                    <button onClick={runValidation} disabled={loading} className="w-full bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-black font-black py-4 rounded-xl text-[10px] uppercase tracking-widest flex items-center justify-center gap-2 transition-all shadow-lg shadow-emerald-500/10">
                        {loading ? <><Loader2 className="w-4 h-4 animate-spin" />Scanning Quality...</> : <><ShieldCheck className="w-4 h-4" />Run Validation</>}
                    </button>
                </Card>
            </div>

            <div className="space-y-6">
                {results ? (
                    <div className="space-y-6">
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <Card className="p-6 text-center border-emerald-500/20 bg-emerald-500/[0.03]">
                                <p className="text-[9px] font-black text-slate-600 uppercase mb-1">Quality Score</p>
                                <p className="text-5xl font-black text-emerald-400">{Math.round(results.quality_score)}</p>
                                <p className="text-[9px] font-black text-emerald-600 uppercase mt-2 tracking-widest">{results.status}</p>
                            </Card>
                            <Card className="p-6 text-center">
                                <p className="text-[9px] font-black text-slate-600 uppercase mb-1">Checks Passed</p>
                                <p className="text-5xl font-black text-white">{results.checks_passed}/{results.total_checks}</p>
                                <p className="text-[9px] font-black text-slate-700 uppercase mt-2 tracking-widest">Validation Coverage</p>
                            </Card>
                            <Card className="p-6 text-center">
                                <p className="text-[9px] font-black text-slate-600 uppercase mb-1">Critical Issues</p>
                                <p className={`text-5xl font-black ${results.critical_issues > 0 ? "text-red-500" : "text-slate-400"}`}>{results.critical_issues}</p>
                                <p className="text-[9px] font-black text-slate-700 uppercase mt-2 tracking-widest">Immediate Attention</p>
                            </Card>
                        </div>

                        <Card className="p-8 space-y-6">
                            <h3 className="text-sm font-black uppercase tracking-widest text-slate-300">Detailed Report</h3>
                            <div className="grid grid-cols-1 gap-3">
                                {results.report && Object.entries(results.report).map(([check, data]: any) => (
                                    <div key={check} className={`p-4 rounded-xl border flex items-center justify-between transition-all ${data.status === 'PASS' ? 'bg-emerald-500/[0.02] border-emerald-500/10' : 'bg-red-500/[0.02] border-red-500/10'}`}>
                                        <div className="flex items-center gap-4">
                                            {data.status === 'PASS' ? <CheckCircle2 className="w-5 h-5 text-emerald-500" /> : <AlertCircle className="w-5 h-5 text-red-500" />}
                                            <div>
                                                <p className="text-xs font-black text-white uppercase tracking-widest">{check.replace(/_/g, " ")}</p>
                                                <p className="text-[10px] text-slate-500 mt-0.5">{data.message}</p>
                                            </div>
                                        </div>
                                        <div className="text-right">
                                            <p className={`text-xs font-black ${data.status === 'PASS' ? 'text-emerald-400' : 'text-red-400'}`}>{data.status}</p>
                                            {data.score !== undefined && <p className="text-[9px] font-black text-slate-700">Score: {data.score.toFixed(2)}</p>}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </Card>
                    </div>
                ) : (
                    <div className="flex flex-col items-center justify-center h-full min-h-[500px] text-center gap-6 bg-[#0E1014] border-2 border-dashed border-white/5 rounded-3xl">
                        <Database className="w-16 h-16 text-slate-800" />
                        <div className="space-y-2">
                            <p className="text-base font-black uppercase text-slate-700 tracking-[0.2em]">Ready for Validation</p>
                            <p className="text-xs text-slate-800 max-w-sm font-medium leading-relaxed">Execute advanced data quality checks including missing value detection, schema matching, and feature drift analysis.</p>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
