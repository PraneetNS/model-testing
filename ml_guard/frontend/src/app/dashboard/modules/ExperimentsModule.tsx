"use client";
import React, { useState, useEffect } from "react";
import { FlaskConical, Beaker, BarChart3, Clock, ChevronRight, Zap, Target, Sliders } from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

const Card = ({ children, className = "" }: any) => (
    <div className={`bg-[#0E1014] border border-white/[0.07] rounded-2xl ${className}`}>{children}</div>
);

const Tile = ({ label, value, sub, accent = false }: any) => (
    <div className="bg-black/20 rounded-xl p-4 space-y-1 transition-all hover:bg-black/30">
        <p className="text-[9px] uppercase font-black tracking-[0.2em] text-slate-700">{label}</p>
        <p className={`text-base font-black truncate ${accent ? "text-purple-400" : "text-white"}`}>{value ?? "—"}</p>
    </div>
);

export default function ExperimentsModule({ state, setState, onAction }: any) {
    const [experiments, setExperiments] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const [selectedExp, setSelectedExp] = useState<any>(null);

    const fetchExperiments = async () => {
        setLoading(true);
        try {
            const res = await fetch(`${API_BASE}/api/v1/experiments`);
            const d = await res.json();
            setExperiments(d.items || []);
        } catch (e) { } finally { setLoading(false); }
    };

    useEffect(() => { fetchExperiments(); }, []);

    if (loading && experiments.length === 0) return (
        <div className="flex flex-col items-center justify-center py-32 gap-5 text-center">
            <div className="w-14 h-14 rounded-full border-2 border-purple-500/20 border-t-purple-500 animate-spin" />
            <p className="text-[10px] uppercase tracking-[0.4em] font-black text-slate-700">Tracking Training Runs...</p>
        </div>
    );

    return (
        <div className="grid grid-cols-1 xl:grid-cols-[1fr_450px] gap-8">
            <div className="space-y-4">
                <div className="flex items-center justify-between mb-4 px-2">
                    <h3 className="text-sm font-black uppercase tracking-[0.2em] text-slate-400">All Experiments</h3>
                    <button onClick={fetchExperiments} className="text-[10px] uppercase font-black text-slate-600 hover:text-white transition-all">↻ Sync Tracker</button>
                </div>

                <div className="space-y-2.5">
                    {experiments.map(exp => (
                        <div key={exp.experiment_id} onClick={() => setSelectedExp(exp)}
                            className={`p-4 rounded-2xl border transition-all cursor-pointer flex items-center justify-between ${selectedExp?.experiment_id === exp.experiment_id ? "border-purple-500/40 bg-purple-500/5 shadow-lg shadow-purple-500/5" : "border-white/5 bg-[#0E1014] hover:border-white/10"}`}>
                            <div className="flex items-center gap-5">
                                <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${exp.status === 'COMPLETED' ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20" : "bg-purple-500/10 text-purple-400 border border-purple-500/20 animate-pulse"}`}>
                                    <FlaskConical className="w-6 h-6" />
                                </div>
                                <div>
                                    <h4 className="text-sm font-black text-white leading-tight">{exp.name}</h4>
                                    <div className="flex items-center gap-3 mt-1.5 font-black uppercase tracking-widest text-[8px] text-slate-600">
                                        <span className="text-slate-400">{exp.model_name || "Unknown Model"}</span>
                                        <span className="opacity-30">•</span>
                                        <span>{exp.framework || "N/A"}</span>
                                        <span className="opacity-30">•</span>
                                        <span className={exp.status === 'COMPLETED' ? "text-emerald-500" : "text-purple-400"}>{exp.status}</span>
                                    </div>
                                </div>
                            </div>

                            <div className="flex items-center gap-6">
                                {exp.metrics && (
                                    <div className="hidden md:flex items-center gap-4">
                                        {Object.entries(exp.metrics).slice(0, 2).map(([k, v]: any) => (
                                            <div key={k} className="text-right">
                                                <p className="text-[8px] uppercase font-black text-slate-700 tracking-tighter">{k}</p>
                                                <p className="text-xs font-black text-slate-300">{typeof v === 'number' ? v.toFixed(4) : v}</p>
                                            </div>
                                        ))}
                                    </div>
                                )}
                                <div className="text-right w-24">
                                    <p className="text-[8px] uppercase font-black text-slate-700">{new Date(exp.started_at).toLocaleDateString()}</p>
                                    <p className="text-[8px] font-mono text-slate-800">#{exp.experiment_id.slice(0, 8)}</p>
                                </div>
                                <ChevronRight className="w-4 h-4 text-slate-800" />
                            </div>
                        </div>
                    ))}
                    {experiments.length === 0 && <p className="text-center py-24 text-slate-700 text-[10px] uppercase font-black tracking-widest">No experiments recorded yet</p>}
                </div>
            </div>

            <div className="space-y-6">
                {selectedExp ? (
                    <>
                        <div className="p-8 rounded-2xl border border-purple-500/20 bg-purple-500/[0.03] space-y-6">
                            <div className="flex items-start justify-between">
                                <div className="space-y-1">
                                    <h2 className="text-xl font-black text-white tracking-tighter">{selectedExp.name}</h2>
                                    <p className="text-[9px] font-black uppercase tracking-widest text-slate-600">Run ID: {selectedExp.experiment_id}</p>
                                </div>
                                <span className={`text-[10px] font-black px-3 py-1 rounded-lg border ${selectedExp.status === 'COMPLETED' ? "bg-emerald-500/5 text-emerald-400 border-emerald-500/20" : "bg-purple-500/5 text-purple-400 border-purple-500/20"}`}>
                                    {selectedExp.status}
                                </span>
                            </div>

                            <div className="grid grid-cols-3 gap-3">
                                <Tile label="Status" value={selectedExp.status} />
                                <Tile label="Duration" value={selectedExp.training_time_ms ? `${Math.round(selectedExp.training_time_ms / 1000)}s` : "—"} />
                                <Tile label="Runtime" value={selectedExp.framework} />
                            </div>
                        </div>

                        {selectedExp.metrics && Object.keys(selectedExp.metrics).length > 0 && (
                            <div className="space-y-4">
                                <div className="flex items-center gap-2 px-1">
                                    <Target className="w-4 h-4 text-purple-400" />
                                    <h3 className="text-xs font-black uppercase tracking-widest text-slate-400">Target Metrics</h3>
                                </div>
                                <div className="grid grid-cols-2 gap-3">
                                    {Object.entries(selectedExp.metrics).map(([k, v]: any) => (
                                        <div key={k} className="p-4 rounded-xl bg-[#0E1014] border border-white/5 flex items-center justify-between transition-all hover:border-white/10 group">
                                            <p className="text-[10px] uppercase font-black text-slate-500 group-hover:text-slate-400 transition-colors uppercase tracking-widest">{k.replace(/_/g, " ")}</p>
                                            <p className="text-sm font-black text-white">{typeof v === 'number' ? v.toFixed(6) : v}</p>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {selectedExp.parameters && Object.keys(selectedExp.parameters).length > 0 && (
                            <div className="space-y-4">
                                <div className="flex items-center gap-2 px-1">
                                    <Sliders className="w-4 h-4 text-purple-400" />
                                    <h3 className="text-xs font-black uppercase tracking-widest text-slate-400">Hyperparameters</h3>
                                </div>
                                <div className="grid grid-cols-1 gap-1.5">
                                    {Object.entries(selectedExp.parameters).map(([k, v]: any) => (
                                        <div key={k} className="flex items-center justify-between px-4 py-2 bg-black/40 rounded-lg border border-white/[0.03]">
                                            <span className="text-[10px] font-mono text-slate-600">{k}</span>
                                            <span className="text-[10px] font-black text-purple-300">{String(v)}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </>
                ) : (
                    <div className="flex flex-col items-center justify-center h-full min-h-[400px] text-center gap-4 bg-[#0E1014] border-2 border-dashed border-white/5 rounded-3xl">
                        <FlaskConical className="w-14 h-14 text-slate-800" />
                        <p className="text-xs font-black uppercase text-slate-700 tracking-widest">Select an experiment to view full run details</p>
                    </div>
                )}
            </div>
        </div>
    );
}
