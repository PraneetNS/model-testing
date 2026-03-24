"use client";
import React, { useState, useEffect } from "react";
import { Activity, TrendingUp, TrendingDown, Target, Clock, BarChart3, AlertCircle, ShieldCheck, Zap, Server, ChevronRight, Info } from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

const Card = ({ children, className = "" }: any) => (
    <div className={`bg-[#0E1014] border border-white/[0.07] rounded-2xl ${className} relative overflow-hidden`}>{children}</div>
);

const Metric = ({ label, value, trend, icon: Icon, color }: any) => (
    <Card className="p-6 space-y-4">
        <div className="flex items-center justify-between">
            <div className={`p-2.5 rounded-xl bg-white/[0.02] border border-white/5`}>
                <Icon className={`w-5 h-5 ${color}`} />
            </div>
            {trend !== undefined && (
                <div className={`text-[9px] font-black uppercase px-2 py-0.5 rounded border ${trend >= 0 ? "text-emerald-400 border-emerald-500/20 bg-emerald-500/5" : "text-red-400 border-red-500/20 bg-red-500/5"}`}>
                    {trend >= 0 ? "+" : ""}{trend}%
                </div>
            )}
        </div>
        <div>
            <p className="text-[9px] font-black uppercase tracking-widest text-slate-600 mb-1">{label}</p>
            <p className="text-3xl font-black text-white tracking-tighter tabular-nums">{value}</p>
        </div>
        <div className="absolute -bottom-4 -right-4 w-20 h-20 bg-white/[0.01] blur-2xl pointer-events-none" />
    </Card>
);

export default function PerformanceModule({ state, setState, onAction }: any) {
    const [stats, setStats] = useState<any>(null);
    const [logs, setLogs] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);

    const fetchData = async () => {
        setLoading(true);
        try {
            const [statsRes, logsRes] = await Promise.all([
                fetch(`${API_BASE}/api/v1/predictions/stats`),
                fetch(`${API_BASE}/api/v1/predictions/logs?limit=50`)
            ]);
            setStats(await statsRes.json());
            setLogs((await logsRes.json()).logs || []);
        } catch (e) { } finally { setLoading(false); }
    };

    useEffect(() => {
        fetchData();
        const interval = setInterval(fetchData, 5000); // 5s Auto-refresh
        return () => clearInterval(interval);
    }, []);

    if (loading && !stats) return (
        <div className="flex flex-col items-center justify-center py-32 gap-5">
            <div className="w-14 h-14 rounded-full border-2 border-orange-500/20 border-t-orange-500 animate-spin" />
            <p className="text-[10px] uppercase tracking-[0.4em] font-black text-slate-700">Analyzing Performance...</p>
        </div>
    );

    return (
        <div className="space-y-8">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <Metric label="Live Inferences" value={stats?.total_inferences?.toLocaleString() || "0"} trend={stats?.volume_drift} icon={Zap} color="text-orange-400" />
                <Metric label="Avg Latency" value={stats?.avg_latency_ms?.toFixed(2) + "ms" || "0ms"} trend={stats?.latency_drift} icon={Clock} color="text-blue-400" />
                <Metric label="Error Rate" value={stats?.error_rate?.toFixed(4) + "%" || "0%"} trend={stats?.error_trend} icon={AlertCircle} color="text-red-400" />
                <Metric label="Drift Confidence" value={stats?.drift_confidence?.toFixed(1) + "%" || "0%"} icon={TrendingUp} color="text-emerald-400" />
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-[1fr_400px] gap-8">
                <div className="space-y-4">
                    <div className="flex items-center justify-between mb-2">
                        <h3 className="text-sm font-black uppercase tracking-widest text-white">Live Prediction Streams</h3>
                        <button onClick={fetchData} className="text-[10px] font-black uppercase text-slate-600 hover:text-white transition-all">↻ Live Update</button>
                    </div>

                    <Card className="overflow-hidden bg-[#0E1014] border-white/5">
                        <table className="w-full text-xs text-left">
                            <thead className="bg-white/[0.03] border-b border-white/5">
                                <tr>
                                    <th className="px-6 py-4 text-[9px] font-black uppercase text-slate-700 tracking-widest">Prediction ID</th>
                                    <th className="px-6 py-4 text-[9px] font-black uppercase text-slate-700 tracking-widest">Timestamp</th>
                                    <th className="px-6 py-4 text-[9px] font-black uppercase text-slate-700 tracking-widest">Latency</th>
                                    <th className="px-6 py-4 text-[9px] font-black uppercase text-slate-700 tracking-widest">Audit State</th>
                                    <th className="px-6 py-4 text-[9px] font-black uppercase text-slate-700 tracking-widest">Status</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-white/[0.03]">
                                {logs.map(log => (
                                    <tr key={log.id} className="hover:bg-white/[0.02] cursor-pointer transition-all">
                                        <td className="px-6 py-4 font-mono font-bold text-slate-400 tabular-nums">#{log.id.slice(0, 8)}</td>
                                        <td className="px-6 py-4 text-[8px] font-black text-slate-600 uppercase tabular-nums">{new Date(log.created_at).toLocaleTimeString()}</td>
                                        <td className="px-6 py-4 font-black text-slate-300 tabular-nums">{log.latency_ms?.toFixed(2)}ms</td>
                                        <td className="px-6 py-4">
                                            <span className={`text-[8px] font-black uppercase px-2 py-0.5 rounded border ${log.audit_result === 'CLEAN' ? 'text-emerald-400 border-emerald-500/20 bg-emerald-500/5' : 'text-red-400 border-red-500/20 bg-red-500/5'}`}>
                                                {log.audit_result || "N/A"}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4">
                                            <div className="flex items-center gap-1.5"><div className={`w-1.5 h-1.5 rounded-full ${log.status === 'SUCCESS' ? "bg-emerald-500" : "bg-red-500"}`} /><span className="text-[9px] font-black text-slate-500 uppercase">{log.status}</span></div>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                        {logs.length === 0 && <p className="text-center py-20 text-[10px] font-black uppercase text-slate-800 tracking-widest">Listening for inference data...</p>}
                    </Card>
                </div>

                <div className="space-y-6">
                    <Card className="p-8 border-orange-500/20 bg-orange-500/[0.03] space-y-6">
                        <div className="flex items-center gap-4">
                            <div className="w-12 h-12 rounded-2xl bg-orange-600/20 flex items-center justify-center p-3 text-orange-400 shadow-lg shadow-orange-500/10"><TrendingUp className="w-full h-full" /></div>
                            <div><p className="text-xl font-black text-white tracking-tighter uppercase leading-tight">Performance Delta</p><p className="text-[9px] font-black uppercase tracking-widest text-slate-600">Real-time Drift Detection</p></div>
                        </div>

                        <div className="space-y-4">
                            <div className="p-4 rounded-xl bg-black/40 border border-white/5 space-y-3">
                                <div className="flex items-center justify-between"><p className="text-[10px] font-black uppercase text-slate-500 tracking-widest">Data Drift Prob</p><p className="text-sm font-black text-emerald-400">Low</p></div>
                                <div className="h-2 bg-white/5 rounded-full overflow-hidden"><div className="h-full bg-emerald-500 w-[12%] rounded-full shadow-[0_0_10px_rgba(16,185,129,0.3)]" /></div>
                            </div>
                            <div className="p-4 rounded-xl bg-black/40 border border-white/5 space-y-3">
                                <div className="flex items-center justify-between"><p className="text-[10px] font-black uppercase text-slate-500 tracking-widest">Label Drift Prob</p><p className="text-sm font-black text-slate-400">4.2%</p></div>
                                <div className="h-2 bg-white/5 rounded-full overflow-hidden"><div className="h-full bg-blue-500 w-[4%] rounded-full shadow-[0_0_10px_rgba(59,130,246,0.3)]" /></div>
                            </div>
                        </div>
                    </Card>

                    <Card className="p-6 border-blue-500/10 bg-blue-500/[0.02] space-y-4">
                        <div className="flex items-center gap-3">
                            <Info className="w-4 h-4 text-blue-400" />
                            <h4 className="text-[10px] font-black uppercase tracking-widest text-slate-400">Mathematical Insight</h4>
                        </div>
                        <div className="space-y-3">
                            <div>
                                <p className="text-[9px] font-black text-white uppercase tracking-tighter">PSI (Population Stability Index)</p>
                                <p className="text-[9px] text-slate-500 mt-1">Formula: Σ(Actual% - Expected%) * ln(Actual% / Expected%). A value &gt; 0.25 indicates significant population shift requiring retraining.</p>
                            </div>
                            <div>
                                <p className="text-[9px] font-black text-white uppercase tracking-tighter">JSD (Jensen-Shannon Divergence)</p>
                                <p className="text-[9px] text-slate-500 mt-1">A symmetric and smoothed version of KL divergence. It measures the similarity between production and baseline probability distributions.</p>
                            </div>
                        </div>
                    </Card>

                    <div className="space-y-4">
                        <h3 className="text-[10px] font-black uppercase tracking-widest text-slate-700">System Capacity</h3>
                        <div className="grid grid-cols-2 gap-3">
                            <Card className="p-4 text-center items-center justify-center flex flex-col gap-1 border-white/5"><p className="text-[8px] uppercase font-black text-slate-800">Requests/sec</p><p className="text-xl font-black text-white">{stats?.throughput || "0"}</p></Card>
                            <Card className="p-4 text-center items-center justify-center flex flex-col gap-1 border-white/5"><p className="text-[8px] uppercase font-black text-slate-800">Healthy Nodes</p><p className="text-xl font-black text-emerald-400">1/1</p></Card>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
