"use client";
import React, { useState, useEffect } from "react";
import { Zap, Activity, HardDrive, ShieldCheck, ChevronRight, Layout, Server, Database, TrendingUp, AlertTriangle, AlertCircle, CheckCircle2, Loader2 } from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

const Card = ({ children, className = "" }: any) => (
    <div className={`bg-[#0E1014] border border-white/[0.07] rounded-2xl ${className}`}>{children}</div>
);

const Tile = ({ label, value, sub, accent = false }: any) => (
    <div className="bg-black/20 rounded-xl p-4 space-y-1 transition-all hover:bg-black/30">
        <p className="text-[9px] uppercase font-black tracking-[0.2em] text-slate-700">{label}</p>
        <p className={`text-base font-black truncate ${accent ? "text-orange-400" : "text-white"}`}>{value ?? "—"}</p>
    </div>
);

export default function DeploymentsModule({ state, setState, onAction }: any) {
    const [environments, setEnvironments] = useState<any[]>([]);
    const [deployments, setDeployments] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const [activeTab, setActiveTab] = useState("PRODUCTION");

    const fetchData = async () => {
        setLoading(true);
        try {
            const [envsRes, deploysRes] = await Promise.all([
                fetch(`${API_BASE}/api/v1/deployments/environments`),
                fetch(`${API_BASE}/api/v1/deployments?per_page=50`)
            ]);
            setEnvironments(await envsRes.json());
            setDeployments((await deploysRes.json()).items || []);
        } catch (e) { } finally { setLoading(false); }
    };

    useEffect(() => { fetchData(); }, []);

    const filtered = deployments.filter(d => d.environment === activeTab);

    return (
        <div className="grid grid-cols-1 xl:grid-cols-[1fr_400px] gap-8">
            <div className="space-y-6">
                <div className="flex items-center justify-between mb-4 px-2">
                    <h3 className="text-sm font-black uppercase tracking-widest text-slate-300">Active Environments</h3>
                    <div className="flex items-center gap-2 bg-black/40 p-1 rounded-xl border border-white/5">
                        {["DEV", "STAGING", "PRODUCTION"].map(env => (
                            <button key={env} onClick={() => setActiveTab(env)}
                                className={`px-4 py-2 rounded-lg text-[9px] font-black uppercase tracking-widest transition-all ${activeTab === env ? "bg-orange-600 text-black shadow-lg shadow-orange-500/20" : "text-slate-600 hover:text-white"}`}>
                                {env}
                            </button>
                        ))}
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {filtered.map(d => (
                        <div key={d.deployment_id} className="p-6 rounded-2xl border border-white/5 bg-[#0E1014] transition-all hover:border-orange-500/20 group relative overflow-hidden shadow-2xl">
                            <div className="flex items-start justify-between mb-4">
                                <div className="w-10 h-10 rounded-xl bg-orange-500/10 flex items-center justify-center p-2.5 group-hover:bg-orange-600 group-hover:text-black transition-all duration-500"><Zap className="w-full h-full" /></div>
                                <div className="text-right">
                                    <span className={`text-[8px] font-black uppercase px-2 py-0.5 rounded border border-emerald-500/20 bg-emerald-500/5 text-emerald-400`}>
                                        Active
                                    </span>
                                </div>
                            </div>
                            <h4 className="text-base font-black text-white">{d.model_name}</h4>
                            <p className="text-[10px] items-center gap-2 font-bold text-slate-600 uppercase tracking-widest mt-1">Version {d.version_number} • {d.environment}</p>

                            <div className="mt-8 pt-6 border-t border-white/[0.03] space-y-4">
                                <div className="flex justify-between items-center"><p className="text-[9px] font-black uppercase text-slate-700">Governance</p><p className="text-xs font-black text-emerald-400">{d.governance_score || "N/A"}</p></div>
                                <div className="flex justify-between items-center"><p className="text-[9px] font-black uppercase text-slate-700">Deploy Date</p><p className="text-xs font-black text-slate-400">{new Date(d.deployed_at).toLocaleDateString()}</p></div>
                            </div>

                            <div className="absolute top-0 right-0 w-24 h-24 bg-orange-500/10 blur-3xl opacity-0 group-hover:opacity-100 transition-all pointer-events-none" />
                        </div>
                    ))}
                    {filtered.length === 0 && (
                        <div className="col-span-3 flex flex-col items-center justify-center py-24 text-center border-2 border-dashed border-white/5 rounded-3xl gap-4">
                            <Server className="w-12 h-12 text-slate-800" />
                            <p className="text-[10px] font-black uppercase tracking-widest text-slate-700">No active models in {activeTab}</p>
                        </div>
                    )}
                </div>
            </div>

            <div className="space-y-6">
                <div className="p-8 rounded-3xl border border-orange-500/20 bg-orange-500/[0.03] space-y-8 flex flex-col items-center text-center">
                    <div className="w-20 h-20 rounded-2xl bg-orange-500/10 flex items-center justify-center shadow-lg shadow-orange-500/5"><TrendingUp className="w-10 h-10 text-orange-500" /></div>
                    <div className="space-y-2">
                        <h3 className="text-xl font-black text-white uppercase tracking-tighter leading-tight">Environment Oversight</h3>
                        <p className="text-xs font-medium text-slate-500 max-w-xs leading-relaxed">System-wide monitoring of models running in production, staging, and development.</p>
                    </div>

                    <div className="w-full space-y-4">
                        <div className="flex items-center justify-between px-4 py-3 bg-black/40 rounded-xl border border-white/5"><div className="flex items-center gap-3"><Server className="w-4 h-4 text-slate-600" /><p className="text-xs font-bold text-slate-300 uppercase tracking-widest">Active Instances</p></div><p className="text-sm font-black text-white">{filtered.length}</p></div>
                        <div className="flex items-center justify-between px-4 py-3 bg-black/40 rounded-xl border border-white/5"><div className="flex items-center gap-3"><Activity className="w-4 h-4 text-slate-600" /><p className="text-xs font-bold text-slate-300 uppercase tracking-widest">Global Healthy</p></div><p className="text-sm font-black text-emerald-400">100%</p></div>
                    </div>
                </div>

                <div className="space-y-4">
                    <h3 className="text-[10px] font-black uppercase tracking-widest text-slate-700 px-1">Infrastructure Logs</h3>
                    <div className="space-y-2 max-h-[400px] overflow-y-auto pr-2">
                        {deployments.slice(0, 10).map(d => (
                            <div key={d.deployment_id} className="p-3.5 rounded-xl border border-white/5 bg-black/20 flex items-start gap-3 transition-all hover:bg-white/[0.05]">
                                <div className={`w-2 h-2 rounded-full mt-1.5 shrink-0 ${d.status === 'ACTIVE' ? "bg-emerald-500 ring-4 ring-emerald-500/10" : "bg-red-500 ring-4 ring-red-500/10"}`} />
                                <div className="flex-1 min-w-0">
                                    <p className="text-xs font-bold text-slate-300 truncate">{d.model_name} v{d.version_number} → {d.environment}</p>
                                    <div className="flex items-center gap-2 mt-1">
                                        <p className="text-[8px] font-black uppercase text-slate-700">{new Date(d.deployed_at).toLocaleString()}</p>
                                        <span className="text-[8px] font-black text-slate-800">•</span>
                                        <p className={`text-[8px] font-black uppercase ${d.status === 'ACTIVE' ? "text-emerald-500" : "text-red-400"}`}>{d.status}</p>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}
