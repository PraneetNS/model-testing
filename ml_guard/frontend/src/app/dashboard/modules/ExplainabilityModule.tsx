"use client";
import { apiFetch } from "@/lib/api";
import React, { useState, useEffect, useCallback } from "react";
import { Eye, Zap, Info, ArrowRight, BarChart3, AlertCircle, FileText, Upload, Database, Loader2, History } from "lucide-react";

const Card = ({ children, className = "" }: any) => (
    <div className={`bg-[#0E1014] border border-white/[0.07] rounded-2xl ${className}`}>{children}</div>
);

const Badge = ({ label, color }: any) => (
    <span className={`px-2 py-0.5 rounded-full text-[8px] font-black uppercase ${color === "green" ? "bg-emerald-500/10 text-emerald-500" : "bg-rose-500/10 text-rose-500"}`}>{label}</span>
);

export default function ExplainabilityModule({ state, setState, onAction }: any) {
    const [modelId, setModelId] = useState("");
    const [results, setResults] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [historyScans, setHistoryScans] = useState<any[]>([]);

    const [modelFile, setModelFile] = useState<File | null>(null);
    const [dataFile, setDataFile] = useState<File | null>(null);

    const fetchHistory = useCallback(async () => {
        try {
            const res = await apiFetch(`/v1/history?scan_type=explainability&limit=10`);
            const d = await res.json();
            if (Array.isArray(d)) setHistoryScans(d);
        } catch (e) { console.error("History fetch failed:", e); }
    }, []);

    useEffect(() => { fetchHistory(); }, [fetchHistory]);

    const loadPastResult = async (scanId: string) => {
        setLoading(true);
        try {
            // Get full detail from history endpoint
            const res = await apiFetch(`/v1/history/${scanId}`);
            const d = await res.json();
            
            if (d.results_json) {
                // Map ScanRecord format to the UI's results format
                const metrics = d.results_json.metrics || {};
                setResults({
                    interpretability_score: metrics.interpretability_score || d.governance_score,
                    feature_importance: d.results_json.feature_importance || {},
                    local_explanations: d.results_json.local_explanations || [],
                    method: d.results_json.method || "shap"
                });
            }
        } catch (e) { console.error("Load failed:", e); }
        setLoading(false);
    };

    const runExplanation = async () => {
        if (!modelFile || !dataFile) { setError("Model and Dataset files required."); return; }
        setLoading(true); setError(null); setResults(null);
        const fd = new FormData();
        fd.append("model_file", modelFile);
        fd.append("dataset_file", dataFile);
        fd.append("model_id", modelId || ""); 
        fd.append("max_samples", "100");

        try {
            const res = await apiFetch(`/v1/explainability/compute`, { method: "POST", body: fd });
            const d = await res.json();
            if (!res.ok) throw new Error(d.detail || "Computation failed.");

            const mid = d.model_id; 
            if (!mid) throw new Error("Backend did not return a valid model_id.");

            let pollCount = 0;
            const poll = setInterval(async () => {
                pollCount++;
                try {
                    const r2 = await apiFetch(`/v1/explainability/${mid}`);
                    if (r2.status === 404) {
                        if (pollCount > 30) { clearInterval(poll); setLoading(false); setError("Timed out waiting for results."); }
                        return;
                    }
                    const d2 = await r2.json();
                    if (d2.results && d2.results.length > 0) {
                        const latest = d2.results[d2.results.length - 1];
                        setResults({
                            ...latest.summary_metrics,
                            feature_importance: latest.global_importance,
                            local_explanations: latest.local_explanations,
                            method: latest.method
                        });
                        setLoading(false);
                        clearInterval(poll);
                        fetchHistory(); // Refresh history
                    }
                } catch (pollErr: any) {
                    console.warn("Poll attempt failed (will retry):", pollErr?.message);
                }
            }, 3000);

        } catch (e: any) { setError(e.message); setLoading(false); }
    };

    return (
        <div className="grid grid-cols-1 xl:grid-cols-[400px_1fr] gap-8">
            <div className="space-y-4">
                <Card className="p-6 space-y-6">
                    <h3 className="text-xs font-black uppercase tracking-widest text-slate-400">Compute Explanation</h3>
                    <div className="space-y-4">
                        <div className="space-y-2">
                            <p className="text-[9px] font-black uppercase tracking-widest text-slate-500">Model File (.pkl/.joblib/.onnx)</p>
                            <label className="block p-4 rounded-xl border border-white/5 bg-black/20 cursor-pointer hover:border-orange-500/20 transition-all">
                                <div className="flex items-center gap-3">
                                    <Upload className={`w-4 h-4 ${modelFile ? "text-emerald-400" : "text-slate-600"}`} />
                                    <span className="text-xs font-bold text-slate-400 truncate">{modelFile ? modelFile.name : "Select model"}</span>
                                </div>
                                <input type="file" accept=".pkl,.joblib,.onnx" className="hidden" onChange={e => {
                                    const f = e.target.files?.[0];
                                    if (f) setModelFile(f);
                                }} />
                            </label>
                        </div>
                        <div className="space-y-2">
                            <p className="text-[9px] font-black uppercase tracking-widest text-slate-500">Dataset File (CSV/Parquet)</p>
                            <label className="block p-4 rounded-xl border border-white/5 bg-black/20 cursor-pointer hover:border-orange-500/20 transition-all">
                                <div className="flex items-center gap-3">
                                    <Database className={`w-4 h-4 ${dataFile ? "text-emerald-400" : "text-slate-600"}`} />
                                    <span className="text-xs font-bold text-slate-400 truncate">{dataFile ? dataFile.name : "Select dataset"}</span>
                                </div>
                                <input type="file" accept=".csv,.parquet" className="hidden" onChange={e => {
                                    const f = e.target.files?.[0];
                                    if (f) setDataFile(f);
                                }} />
                            </label>
                        </div>
                    </div>
                    {error && <div className="p-4 bg-red-500/5 border border-red-500/20 rounded-xl text-red-400 font-bold text-xs flex gap-2"><AlertCircle className="w-4 h-4 shrink-0" /> {error}</div>}
                    <button onClick={runExplanation} disabled={loading} className="w-full bg-orange-600 hover:bg-orange-500 disabled:opacity-50 text-black font-black py-4 rounded-xl text-[10px] uppercase tracking-widest flex items-center justify-center gap-2 transition-all">
                        {loading ? <><Loader2 className="w-4 h-4 animate-spin" />Computing SHAP...</> : <><Zap className="w-4 h-4" />Run Explainability</>}
                    </button>
                    {results && (
                        <button onClick={() => setResults(null)} className="w-full text-[9px] font-black uppercase tracking-widest text-slate-600 hover:text-white transition-colors">
                            Clear Results
                        </button>
                    )}
                </Card>

                <Card className="p-6 border-blue-500/10 bg-blue-500/[0.02] space-y-4">
                    <div className="flex items-center gap-3">
                        <Info className="w-4 h-4 text-blue-400" />
                        <h4 className="text-[10px] font-black uppercase tracking-widest text-slate-400">Governance Context</h4>
                    </div>
                    <div className="space-y-3">
                        <div className="space-y-1">
                            <p className="text-[10px] text-white font-black uppercase">SHAP Values</p>
                            <p className="text-[9px] text-slate-500 leading-relaxed">
                                Game theoretic attribution of each feature's impact on the model prediction.
                            </p>
                        </div>
                        <div className="space-y-1">
                            <p className="text-[10px] text-white font-black uppercase">Interpretability</p>
                            <p className="text-[9px] text-slate-500 leading-relaxed">
                                Higher transparency indicates fewer features are needed to explain model behavior.
                            </p>
                        </div>
                    </div>
                </Card>
            </div>

            <div className="space-y-6">
                {results ? (
                    <>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                            <Card className="p-8 border-orange-500/20 bg-orange-500/[0.03] text-center">
                                <p className="text-[9px] font-black uppercase tracking-widest text-slate-600 mb-2">Interpretability Score</p>
                                <div className="text-7xl font-black text-orange-400">{Math.round(results.interpretability_score || 0)}</div>
                                <div className="mt-4 flex flex-col items-center gap-1">
                                    <p className={`text-[10px] font-black uppercase tracking-widest ${results.interpretability_score >= 70 ? "text-emerald-500" : results.interpretability_score >= 40 ? "text-amber-500" : "text-rose-500"}`}>
                                        {results.interpretability_score >= 70 ? "✓ Highly Transparent" : results.interpretability_score >= 40 ? "⚠ Moderate Complexity" : "✖ Black Box Model"}
                                    </p>
                                    <p className="text-[8px] text-slate-500 max-w-[200px]">
                                        {results.interpretability_score >= 70 ? "The model logic is concentrated into very few understandable features." : "Decision paths are complex with high entropy across many features."}
                                    </p>
                                </div>
                            </Card>

                            <Card className="col-span-2 p-8 space-y-6 bg-gradient-to-br from-[#0E1014] to-[#121418]">
                                <h3 className="text-sm font-black uppercase tracking-widest text-slate-300">Feature Influence (SHAP Global)</h3>
                                <div className="space-y-4">
                                    {Object.entries(results.feature_importance || {}).sort((a: any, b: any) => b[1] - a[1]).slice(0, 10).map(([k, v]: any) => (
                                        <div key={k} className="space-y-1.5">
                                            <div className="flex justify-between text-[11px] font-black uppercase tracking-tighter">
                                                <span className="text-slate-400">{k}</span>
                                                <span className="text-white">{(v * 100).toFixed(1)}%</span>
                                            </div>
                                            <div className="h-2 bg-white/5 rounded-full overflow-hidden">
                                                <div className="h-full bg-orange-500/80 rounded-full transition-all" style={{ width: `${Math.min(v * 100, 100)}%` }} />
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </Card>
                        </div>

                        {results.local_explanations?.length > 0 && (
                            <div className="space-y-4">
                                <h3 className="text-xs font-black uppercase tracking-widest text-orange-500 flex items-center gap-2"><Info className="w-4 h-4" /> Local Feature Impact</h3>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    {results.local_explanations.slice(0, 4).map((lx: any, i: number) => (
                                        <Card key={i} className="p-5 space-y-4">
                                            <p className="text-[10px] font-black uppercase text-slate-600">Sample Row #{i + 1}</p>
                                            <div className="space-y-2">
                                                {Object.entries(lx).slice(0, 5).map(([k, v]: any) => (
                                                    <div key={k} className="flex justify-between text-[10px]">
                                                        <span className="text-slate-500 font-mono truncate mr-4">{k}</span>
                                                        <span className={v > 0 ? "text-emerald-400" : "text-rose-400"}>{v > 0 ? "+" : ""}{v.toFixed(4)}</span>
                                                    </div>
                                                ))}
                                            </div>
                                        </Card>
                                    ))}
                                </div>
                            </div>
                        )}
                    </>
                ) : (
                    <div className="flex flex-col md:flex-row gap-6 h-full">
                        <div className="flex-1 flex flex-col items-center justify-center min-h-[500px] text-center gap-6 bg-[#0E1014] border-2 border-dashed border-white/5 rounded-3xl">
                            <div className="relative">
                                <Eye className="w-16 h-16 text-slate-800" />
                                <Zap className="absolute -top-2 -right-2 w-6 h-6 text-orange-500 animate-pulse" />
                            </div>
                            <div className="space-y-2">
                                <p className="text-base font-black uppercase text-slate-700 tracking-[0.2em]">Ready for Analysis</p>
                                <p className="text-xs text-slate-800 max-w-sm font-medium">Upload artifacts or select a past scan from the history sidebar to begin.</p>
                            </div>
                        </div>
                        
                        <div className="w-full md:w-72 space-y-4 overflow-y-auto max-h-[600px] pr-2 custom-scrollbar">
                            <div className="flex items-center gap-2 px-1 py-1 border-b border-white/5">
                                <History className="w-4 h-4 text-slate-600" />
                                <h3 className="text-[10px] font-black uppercase tracking-widest text-slate-600">Scan History</h3>
                            </div>
                            <div className="space-y-2">
                                {historyScans.length > 0 ? historyScans.map((s: any) => (
                                    <button 
                                        key={s.id} 
                                        onClick={() => loadPastResult(s.id)} 
                                        className="w-full text-left p-4 rounded-xl border border-white/5 bg-black/20 hover:border-orange-500/40 hover:bg-white/[0.02] transition-all group"
                                    >
                                        <div className="flex justify-between items-center mb-1">
                                            <p className="text-[10px] font-black text-white">{new Date(s.created_at).toLocaleDateString()}</p>
                                            <Badge label={s.gate_status} color={s.gate_status === "PASSED" ? "green" : "red"} />
                                        </div>
                                        <p className="text-[9px] text-slate-500 font-mono truncate">{s.id.slice(0, 14)}...</p>
                                        <div className="mt-2 flex items-center justify-between">
                                            <span className="text-[9px] text-slate-600">Interpretability</span>
                                            <span className="text-[11px] font-black text-orange-400">{Math.round(s.governance_score)}</span>
                                        </div>
                                    </button>
                                )) : (
                                    <div className="p-8 text-center border border-white/5 bg-black/10 rounded-xl">
                                        <p className="text-[9px] font-black text-slate-800 uppercase tracking-widest italic">No scans found</p>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

