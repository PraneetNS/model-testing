"use client";
import React, { useState, useEffect } from "react";
import { 
    ShieldAlert, Zap, Skull, Terminal, Clipboard, 
    Download, PieChart, TrendingUp, AlertCircle, CheckCircle2, 
    Activity, Play, Loader2, Search, FileText
} from "lucide-react";
import { 
    PieChart as RePieChart, Pie, Cell, ResponsiveContainer, 
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend 
} from "recharts";
import { apiPost, apiGet, safeJson } from "@/lib/api";

interface RedTeamSummary {
    total_attacks: number;
    success_rate: number;
    critical_vulnerabilities: number;
}

interface RedTeamFinding {
    id: string;
    category: string;
    severity: string;
    rounds: number;
    is_successful: boolean;
    prompt: string;
    response: string | null;
    reasoning: string;
    timestamp: string;
}

interface RedTeamReport {
    session_id: string;
    status: string;
    summary: RedTeamSummary;
    findings: RedTeamFinding[];
}

export default function RedTeamModule({ model_id }: { model_id: string }) {
    const [report, setReport] = useState<RedTeamReport | null>(null);
    const [loading, setLoading] = useState(false);
    const [running, setRunning] = useState(false);
    const [activeFinding, setActiveFinding] = useState<RedTeamFinding | null>(null);

    const startCampaign = async () => {
        setLoading(true);
        try {
            const data = await apiPost<any>(`/api/v1/redteam/start?model_id=${model_id}`);
            setRunning(true);
            pollStatus(data.session_id);
        } catch (err) {
            console.error("Failed to start red team campaign", err);
        } finally {
            setLoading(false);
        }
    };

    const pollStatus = async (sessionId: string) => {
        const fetchReport = async () => {
            const data = await apiGet<any>(`/api/v1/redteam/${sessionId}/report`);
            setReport(data);
            if (data.status === "COMPLETED" || data.status === "FAILED") {
                setRunning(false);
                clearInterval(interval);
            }
        };
        const interval = setInterval(fetchReport, 5000);
        fetchReport();
    };

    const copyToClipboard = (text: string) => {
        navigator.clipboard.writeText(text);
        alert("Attack prompt copied to clipboard!");
    };

    const severityColors: any = {
        CRITICAL: "#ef4444",
        HIGH: "#f97316",
        MEDIUM: "#eab308",
        LOW: "#22c55e"
    };

    const pieData = report ? [
        { name: "Critical", value: report.findings.filter(f => f.severity === "CRITICAL").length },
        { name: "High", value: report.findings.filter(f => f.severity === "HIGH").length },
        { name: "Medium", value: report.findings.filter(f => f.severity === "MEDIUM").length },
        { name: "Low", value: report.findings.filter(f => f.severity === "LOW").length },
    ].filter(d => d.value > 0) : [];

    return (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-1000">
            {/* Header / Trigger */}
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-2xl font-black text-white flex items-center gap-3">
                        <Skull className="w-8 h-8 text-red-500" />
                        Autopilot Red-Teaming
                    </h2>
                    <p className="text-slate-500 text-xs font-bold uppercase tracking-widest mt-1">
                        AI-Augmented Stress Testing & Jailbreak Simulation
                    </p>
                </div>
                <button 
                    onClick={startCampaign}
                    disabled={loading || running}
                    className="bg-red-500 hover:bg-red-600 disabled:bg-slate-800 text-white px-8 py-3 rounded-xl font-black text-[11px] uppercase tracking-widest flex items-center gap-2 transition-all shadow-xl shadow-red-500/10 active:scale-95"
                >
                    {running ? (
                        <> <Loader2 className="w-4 h-4 animate-spin" /> Campaign in Progress... </>
                    ) : (
                        <> <Play className="w-4 h-4" /> Launch Attack Engine </>
                    )}
                </button>
            </div>

            {report && (
                <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                    {/* Stats Card */}
                    <div className="lg:col-span-1 space-y-6">
                        <div className="bg-[#0E1014] border border-white/[0.07] rounded-2xl p-6">
                            <p className="text-[9px] font-black text-slate-600 uppercase tracking-widest mb-6">Severity Breakdown</p>
                            <div className="h-[200px]">
                                <ResponsiveContainer width="100%" height="100%">
                                    <RePieChart>
                                        <Pie
                                            data={pieData}
                                            innerRadius={60}
                                            outerRadius={80}
                                            paddingAngle={5}
                                            dataKey="value"
                                        >
                                            {pieData.map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={severityColors[entry.name.toUpperCase()]} />
                                            ))}
                                        </Pie>
                                        <Tooltip />
                                    </RePieChart>
                                </ResponsiveContainer>
                            </div>
                            <div className="grid grid-cols-2 gap-4 mt-6">
                                <div className="text-center">
                                    <p className="text-[24px] font-black text-white">{report.summary.total_attacks}</p>
                                    <p className="text-[8px] font-black text-slate-600 uppercase">Total Attacks</p>
                                </div>
                                <div className="text-center">
                                    <p className="text-[24px] font-black text-red-500">{report.summary.critical_vulnerabilities}</p>
                                    <p className="text-[8px] font-black text-slate-600 uppercase">Critical</p>
                                </div>
                            </div>
                        </div>

                        <div className="bg-red-500/5 border border-red-500/10 rounded-2xl p-6">
                            <p className="text-[9px] font-black text-red-500 uppercase tracking-widest mb-2 flex items-center gap-2">
                                <AlertCircle className="w-3 h-3" /> Security Posture
                            </p>
                            <p className="text-xs font-bold text-white leading-relaxed">
                                {report.summary.success_rate > 0.3 
                                    ? "CRITICAL: Multiple successful jailbreaks detected. Immediate hardening required."
                                    : report.summary.success_rate > 0 
                                        ? "WARNING: Isolated vulnerabilities found in edge cases."
                                        : "SECURE: Target successfully resisted all 20 standardized attack vectors."}
                            </p>
                            <a 
                                href={`${process.env.NEXT_PUBLIC_API_BASE}/api/v1/redteam/${report.session_id}/report/pdf`}
                                className="mt-4 w-full bg-white/5 border border-white/10 hover:bg-white/10 text-white rounded-lg py-2.5 text-[9px] font-black uppercase tracking-widest flex items-center justify-center gap-2"
                            >
                                <Download className="w-3.5 h-3.5" /> Export PDF Report
                            </a>
                        </div>
                    </div>

                    {/* Findings Table */}
                    <div className="lg:col-span-3 bg-[#0E1014] border border-white/[0.07] rounded-2xl overflow-hidden">
                        <table className="w-full text-left border-collapse">
                            <thead>
                                <tr className="border-b border-white/5 bg-white/[0.02]">
                                    <th className="p-4 text-[9px] font-black text-slate-500 uppercase tracking-widest border-r border-white/5">Finding Category</th>
                                    <th className="p-4 text-[9px] font-black text-slate-500 uppercase tracking-widest border-r border-white/5">Severity</th>
                                    <th className="p-4 text-[9px] font-black text-slate-500 uppercase tracking-widest border-r border-white/5">Rounds</th>
                                    <th className="p-4 text-[9px] font-black text-slate-500 uppercase tracking-widest">Outcome</th>
                                </tr>
                            </thead>
                            <tbody>
                                {report.findings.map((f) => (
                                    <tr 
                                        key={f.id} 
                                        onClick={() => setActiveFinding(f)}
                                        className={`border-b border-white/5 hover:bg-white/[0.02] cursor-pointer transition-colors ${activeFinding?.id === f.id ? "bg-orange-500/5" : ""}`}
                                    >
                                        <td className="p-4 flex items-center gap-3">
                                            <Terminal className="w-4 h-4 text-slate-600" />
                                            <span className="text-xs font-black text-slate-300 uppercase">{f.category.replace('_', ' ')}</span>
                                        </td>
                                        <td className="p-4">
                                            <span className={`text-[9px] font-black px-2 py-0.5 rounded border ${
                                                f.severity === "CRITICAL" ? "bg-red-500/10 text-red-400 border-red-500/20" :
                                                f.severity === "HIGH" ? "bg-orange-500/10 text-orange-400 border-orange-500/20" :
                                                "bg-slate-500/10 text-slate-400 border-white/10"
                                            }`}>
                                                {f.severity}
                                            </span>
                                        </td>
                                        <td className="p-4 text-xs font-bold text-slate-400">{f.rounds}</td>
                                        <td className="p-4">
                                            {f.is_successful ? (
                                                <div className="flex items-center gap-1.5 text-red-500 animate-pulse text-[10px] font-black uppercase">
                                                    <Zap className="w-3.5 h-3.5" /> BREACHED
                                                </div>
                                            ) : (
                                                <div className="flex items-center gap-1.5 text-emerald-500 text-[10px] font-black uppercase">
                                                    <CheckCircle2 className="w-3.5 h-3.5" /> REFUSED
                                                </div>
                                            )}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>

                        {/* Finding Detail Sidebar */}
                        {activeFinding && (
                            <div className="p-6 border-t border-white/5 bg-black/40 animate-in slide-in-from-right-4 duration-500">
                                <div className="flex items-center justify-between mb-4">
                                    <h5 className="text-xs font-black text-white uppercase tracking-widest">Finding Investigation</h5>
                                    <button 
                                        onClick={() => copyToClipboard(activeFinding.prompt)}
                                        className="text-[9px] font-black text-orange-400 flex items-center gap-1.5 uppercase hover:text-orange-500"
                                    >
                                        <Clipboard className="w-3.5 h-3.5" /> Copy Payload
                                    </button>
                                </div>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    <div className="space-y-2">
                                        <p className="text-[8px] font-black text-slate-600 uppercase">Adversarial Prompt</p>
                                        <div className="bg-black/50 border border-white/5 p-4 rounded-xl text-[11px] font-mono text-orange-300/80 leading-relaxed max-h-[150px] overflow-y-auto">
                                            {activeFinding.prompt}
                                        </div>
                                    </div>
                                    <div className="space-y-2">
                                        <p className="text-[8px] font-black text-slate-600 uppercase">Target Response</p>
                                        <div className="bg-black/50 border border-white/5 p-4 rounded-xl text-[11px] font-mono text-emerald-300/80 leading-relaxed max-h-[150px] overflow-y-auto">
                                            {activeFinding.response || "No response received."}
                                        </div>
                                    </div>
                                </div>
                                <div className="mt-6 p-4 bg-orange-500/5 border border-orange-500/10 rounded-xl">
                                    <p className="text-[9px] font-black text-orange-500 uppercase mb-1">Judge Reasoning</p>
                                    <p className="text-[11px] font-bold text-slate-300 italic">"{activeFinding.reasoning}"</p>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
