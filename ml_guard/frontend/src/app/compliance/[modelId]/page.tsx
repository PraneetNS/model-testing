"use client";

import { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import { apiFetch, safeJson } from "@/lib/api";
import { ShieldCheck, ChevronDown, ChevronRight, Download, AlertTriangle, XCircle, CheckCircle2 } from "lucide-react";

export default function CompliancePage() {
    const { modelId } = useParams() as { modelId: string };
    const router = useRouter();
    const [filter, setFilter] = useState("all");
    const [results, setResults] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const [expandedIds, setExpandedIds] = useState<Record<string, boolean>>({});

    useEffect(() => {
        setLoading(true);
        apiFetch(`/api/v1/report/${modelId}/compliance?framework=${filter}`)
            .then(res => safeJson(res))
            .then(data => {
                setResults(data.results || []);
                setLoading(false);
            })
            .catch(() => setLoading(false));
    }, [modelId, filter]);

    const toggleExpand = (id: string) => {
        setExpandedIds(prev => ({ ...prev, [id]: !prev[id] }));
    };

    const downloadReport = async () => {
        alert("Downloading PDF generated via back-end!");
        // Simulated or actual implementation could be linked to existing dashboard feature
    };

    return (
        <div className="p-8 max-w-5xl mx-auto space-y-8">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-black text-white flex items-center gap-3">
                        <ShieldCheck className="w-8 h-8 text-emerald-400" />
                        Regulatory Compliance
                    </h1>
                    <p className="text-slate-400 mt-2 text-sm">Model ID: <span className="text-white font-mono">{modelId}</span></p>
                </div>
                <button
                    onClick={downloadReport}
                    className="flex items-center gap-2 bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/20 border border-emerald-500/30 px-4 py-2 rounded-xl text-sm font-bold transition-colors"
                >
                    <Download className="w-4 h-4" /> Download compliance report
                </button>
            </div>

            <div className="flex gap-2">
                {["all", "eu_ai_act", "nist_rmf"].map(f => (
                    <button
                        key={f}
                        onClick={() => setFilter(f)}
                        className={`px-4 py-2 text-xs font-black uppercase tracking-wider rounded-lg transition-colors border ${
                            filter === f ? "bg-orange-500 text-white border-orange-500" : "bg-black/20 text-slate-400 border-white/10 hover:border-white/30"
                        }`}
                    >
                        {f.replace("_", " ")}
                    </button>
                ))}
            </div>

            {loading ? (
                <div className="text-center py-20 text-slate-500 animate-pulse">Checking compliance constraints...</div>
            ) : results.length === 0 ? (
                <div className="text-center py-20 text-slate-500 border border-white/10 rounded-2xl bg-black/20">
                    No compliance data mapped yet. Generate a governance report card first.
                </div>
            ) : (
                <div className="space-y-4">
                    {results.map((r, i) => {
                        const isPass = r.status === "pass";
                        const isFail = r.status === "fail";
                        const isPartial = r.status === "partial";
                        const id = `${r.framework}-${r.control}`;
                        const expanded = expandedIds[id];

                        return (
                            <div key={id} className="border border-white/10 rounded-xl bg-[#0E1014] overflow-hidden">
                                <button
                                    onClick={() => toggleExpand(id)}
                                    className="w-full flex items-center justify-between p-4 bg-white/[0.02] hover:bg-white/[0.04] transition-colors"
                                >
                                    <div className="flex items-center gap-4">
                                        {isPass ? <CheckCircle2 className="w-5 h-5 text-emerald-400" /> :
                                         isFail ? <XCircle className="w-5 h-5 text-red-500" /> :
                                         <AlertTriangle className="w-5 h-5 text-orange-400" />}
                                        <div className="text-left">
                                            <p className="text-xs text-slate-500 font-mono mb-1">{r.framework.toUpperCase()} - {r.control}</p>
                                            <p className="text-sm font-bold text-slate-200">{r.title}</p>
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-4">
                                        <span className={`text-[10px] uppercase font-black px-2 py-0.5 rounded border ${
                                            isPass ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400" :
                                            isFail ? "bg-red-500/10 border-red-500/30 text-red-400" :
                                            "bg-orange-500/10 border-orange-500/30 text-orange-400"
                                        }`}>{r.status}</span>
                                        {expanded ? <ChevronDown className="w-4 h-4 text-slate-600" /> : <ChevronRight className="w-4 h-4 text-slate-600" />}
                                    </div>
                                </button>
                                
                                {expanded && (
                                    <div className="p-4 border-t border-white/5 bg-black/20 text-sm space-y-4">
                                        <div>
                                            <h4 className="text-[10px] uppercase font-black text-slate-500 tracking-wider mb-1">Requirement</h4>
                                            <p className="text-slate-300">{r.description}</p>
                                        </div>
                                        <div className="grid grid-cols-2 gap-4">
                                            <div>
                                                <h4 className="text-[10px] uppercase font-black text-slate-500 tracking-wider mb-1">Evidence</h4>
                                                <p className="text-slate-300 font-mono text-xs">{r.evidence}</p>
                                            </div>
                                            {(r.gap || isFail || isPartial) && (
                                                <div>
                                                    <h4 className="text-[10px] uppercase font-black text-orange-500 tracking-wider mb-1">Identified Gaps</h4>
                                                    <p className="text-orange-400/80 font-mono text-xs">{r.gap || "Details unavailable"}</p>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                )}
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
}
