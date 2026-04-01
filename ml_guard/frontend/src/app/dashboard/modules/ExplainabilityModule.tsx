"use client";
import { apiFetch } from "@/lib/api";
import React, { useState, useEffect } from "react";
import { Eye, Zap, Info, ArrowRight, BarChart3, AlertCircle, FileText, Upload, Database, Loader2 } from "lucide-react";


const Card = ({ children, className = "" }: any) => (
    <div className={`bg-[#0E1014] border border-white/[0.07] rounded-2xl ${className}`}>{children}</div>
);

const Tile = ({ label, value, sub, accent = false }: any) => (
    <div className="bg-black/20 rounded-xl p-4 space-y-1 transition-all hover:bg-black/30">
        <p className="text-[9px] uppercase font-black tracking-[0.2em] text-slate-700">{label}</p>
        <p className={`text-base font-black truncate ${accent ? "text-orange-400" : "text-white"}`}>{value ?? "—"}</p>
    </div>
);

export default function ExplainabilityModule({ state, setState, onAction }: any) {
    const [modelId, setModelId] = useState("");
    const [results, setResults] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const [modelFile, setModelFile] = useState<File | null>(null);
    const [dataFile, setDataFile] = useState<File | null>(null);

    const getToken = () => {
        try { return JSON.parse(localStorage.getItem("mlguard_session") || "{}").token || ""; } catch { return ""; }
    };

    const runExplanation = async () => {
        if (!modelFile || !dataFile) { setError("Model and Dataset files required."); return; }
        setLoading(true); setError(null); setResults(null);
        const token = getToken();
        const fd = new FormData();
        fd.append("model_file", modelFile);
        fd.append("dataset_file", dataFile);
        fd.append("model_id", modelId || ""); // Let backend assign a proper UUID
        fd.append("max_samples", "100");

        try {
            const headers: Record<string, string> = {};
            if (token) headers["Authorization"] = `Bearer ${token}`;
            const res = await apiFetch(`/api/v1/explainability/compute`, { method: "POST", headers, body: fd });
            const d = await res.json();
            if (!res.ok) throw new Error(d.detail || "Computation failed.");

            const mid = d.model_id; // This is now always a valid UUID from backend
            if (!mid) throw new Error("Backend did not return a valid model_id.");

            let pollCount = 0;
            const poll = setInterval(async () => {
                pollCount++;
                try {
                    const pollHeaders: Record<string, string> = {};
                    if (token) pollHeaders["Authorization"] = `Bearer ${token}`;
                    const r2 = await apiFetch(`/api/v1/explainability/${mid}`, { headers: pollHeaders });
                    if (r2.status === 404) {
                        // Still computing — keep polling
                        if (pollCount > 20) { clearInterval(poll); setLoading(false); setError("Timed out waiting for results."); }
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
                </Card>

                <Card className="p-6 border-blue-500/10 bg-blue-500/[0.02] space-y-4">
                    <div className="flex items-center gap-3">
                        <Info className="w-4 h-4 text-blue-400" />
                        <h4 className="text-[10px] font-black uppercase tracking-widest text-slate-400">Glossary of Terms</h4>
                    </div>
                    <div className="space-y-3">
                        <div className="space-y-1">
                            <p className="text-[10px] text-white font-black uppercase">SHAP (Shapley Values)</p>
                            <p className="text-[9px] text-slate-500 leading-relaxed">
                                A method from Game Theory that assigns each feature a value representing its contribution to the final prediction. <span className="text-blue-400 italic">Positive</span> values increase the output, <span className="text-rose-400 italic">Negative</span> decrease it.
                            </p>
                        </div>
                        <div className="space-y-1">
                            <p className="text-[10px] text-white font-black uppercase">Interpretability Score</p>
                            <p className="text-[9px] text-slate-500 leading-relaxed">
                                Measures how "dense" vs "sparse" the model's logic is. A high score means the model relies on a few key, consistent features (Transparent). A low score means complex, high-entropy logic (Black Box).
                            </p>
                        </div>
                        <div className="space-y-1">
                            <p className="text-[10px] text-white font-black uppercase">Global vs Local</p>
                            <p className="text-[9px] text-slate-500 leading-relaxed">
                                <b>Global</b> shows the general behavior of the model across all data. <b>Local</b> shows exactly why the model made a specific prediction for one single individual.
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
                                                <div className="h-full bg-orange-500/80 rounded-full transition-all duration-1000 shadow-[0_0_8px_rgba(249,115,22,0.3)]" style={{ width: `${Math.min(v * 100, 100)}%` }} />
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </Card>
                        </div>

                        {results.local_explanations?.length > 0 && (
                            <div className="space-y-4">
                                <h3 className="text-xs font-black uppercase tracking-widest text-orange-500 flex items-center gap-2"><Info className="w-4 h-4" /> Sample Local Explanations</h3>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    {results.local_explanations.slice(0, 4).map((lx: any, i: number) => (
                                        <Card key={i} className="p-5 space-y-4">
                                            <div className="flex items-center justify-between"><p className="text-[10px] font-black uppercase text-slate-600">Sample #{i + 1}</p><p className="text-[10px] font-black uppercase text-white">Prediction: {results.method}</p></div>
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
                    <div className="flex flex-col items-center justify-center h-full min-h-[500px] text-center gap-6 bg-[#0E1014] border-2 border-dashed border-white/5 rounded-3xl">
                        <div className="relative">
                            <Eye className="w-16 h-16 text-slate-800" />
                            <Zap className="absolute -top-2 -right-2 w-6 h-6 text-orange-500 animate-pulse" />
                        </div>
                        <div className="space-y-2">
                            <p className="text-base font-black uppercase text-slate-700 tracking-[0.2em]">Explanation Core Pending</p>
                            <p className="text-xs text-slate-800 max-w-sm font-medium">Upload a model and sample dataset to perform SHAP-based feature attribution and compute the interpretability score.</p>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
