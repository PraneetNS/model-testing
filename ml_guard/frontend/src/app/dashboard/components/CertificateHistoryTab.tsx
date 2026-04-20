"use client";
import React, { useState, useEffect } from "react";
import { 
    Activity, FileText, Download, ShieldCheck, 
    AlertTriangle, XCircle, Search, Copy, 
    ExternalLink, CheckCircle2, Loader2, Gauge
} from "lucide-react";
import { 
    LineChart, Line, XAxis, YAxis, CartesianGrid, 
    Tooltip, ResponsiveContainer, ReferenceLine 
} from "recharts";

interface Report {
    cert_hash: string;
    overall_score: number;
    verdict: string;
    issued_at: string;
    is_revoked: boolean;
}

export default function CertificateHistoryTab({ model_id }: { model_id: string }) {
    const [reports, setReports] = useState<Report[]>([]);
    const [loading, setLoading] = useState(true);

    const fetchHistory = async () => {
        try {
            const res = await fetch(`${process.env.NEXT_PUBLIC_API_BASE}/api/v1/reports/${model_id}/history`);
            const data = await safeJson(res);
            setReports(data);
        } catch (err) {
            console.error("Failed to fetch certificate history", err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => { fetchHistory(); }, [model_id]);

    const chartData = [...reports].reverse().map(r => ({
        date: new Date(r.issued_at).toLocaleDateString(),
        score: r.overall_score
    }));

    const copyVerifyLink = (hash: string) => {
        const link = `https://mlguard.io/verify/${hash}`;
        navigator.clipboard.writeText(link);
        alert("Verification Link Copied!");
    };

    return (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            {/* Score Trend Card */}
            <div className="bg-[#0E1014] border border-white/[0.07] rounded-3xl p-8 relative overflow-hidden">
                <div className="flex items-center justify-between mb-8">
                    <div>
                        <h4 className="text-sm font-black text-white flex items-center gap-2">
                            <Gauge className="w-4 h-4 text-emerald-500" />
                            Governance Compliance Trajectory
                        </h4>
                        <p className="text-[10px] text-slate-500 font-bold uppercase mt-1">
                            Longitudinal Analysis of Certified Audit Sessions
                        </p>
                    </div>
                </div>

                <div className="h-[250px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
                            <XAxis dataKey="date" stroke="#475569" fontSize={8} tickLine={false} axisLine={false} />
                            <YAxis domain={[0, 100]} stroke="#475569" fontSize={8} tickLine={false} axisLine={false} />
                            <Tooltip content={({ active, payload }) => {
                                if (active && payload && payload.length) {
                                    return (
                                        <div className="bg-black/90 border border-white/10 p-2 rounded text-[10px] font-black">
                                            Score: <span className="text-emerald-400">{payload[0].value}%</span>
                                        </div>
                                    );
                                }
                                return null;
                            }} />
                            <Line 
                                type="monotone" 
                                dataKey="score" 
                                stroke="#10b981" 
                                strokeWidth={3} 
                                dot={{ fill: '#10b981', r: 4 }}
                                activeDot={{ r: 6, stroke: '#10b981', strokeWidth: 2, fill: '#000' }}
                            />
                            <ReferenceLine y={80} stroke="#f59e0b" strokeDasharray="3 3" label={{ position: 'right', value: 'Certified Threshold', fill: '#f59e0b', fontSize: 8 }} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* History Table */}
            <div className="bg-[#0E1014] border border-white/[0.07] rounded-3xl overflow-hidden shadow-2xl">
                <table className="w-full text-left border-collapse">
                    <thead>
                        <tr className="border-b border-white/5 bg-white/[0.02]">
                            <th className="p-6 text-[9px] font-black text-slate-500 uppercase tracking-widest border-r border-white/5">Certificate Hash</th>
                            <th className="p-6 text-[9px] font-black text-slate-500 uppercase tracking-widest border-r border-white/5">Issued At</th>
                            <th className="p-6 text-[9px] font-black text-slate-500 uppercase tracking-widest border-r border-white/5">Score</th>
                            <th className="p-6 text-[9px] font-black text-slate-500 uppercase tracking-widest border-r border-white/5">Verdict</th>
                            <th className="p-6 text-[9px] font-black text-slate-500 uppercase tracking-widest">Controls</th>
                        </tr>
                    </thead>
                    <tbody>
                        {reports.map((r) => (
                            <tr key={r.cert_hash} className={`border-b border-white/5 hover:bg-white/[0.02] transition-colors ${r.is_revoked ? "opacity-50 grayscale" : ""}`}>
                                <td className="p-6">
                                    <div className="flex items-center gap-3">
                                        <div className="w-8 h-8 rounded-lg bg-white/5 flex items-center justify-center border border-white/10">
                                            <ShieldCheck className={`w-4 h-4 ${r.is_revoked ? "text-red-500" : "text-emerald-500"}`} />
                                        </div>
                                        <div>
                                            <p className="text-[10px] font-black text-slate-300 font-mono tracking-tighter">{r.cert_hash.slice(0, 16)}...</p>
                                            <button onClick={() => copyVerifyLink(r.cert_hash)} className="text-[8px] font-black text-emerald-500 uppercase hover:underline">Copy Verify Link</button>
                                        </div>
                                    </div>
                                </td>
                                <td className="p-6 text-[10px] font-bold text-slate-400">{new Date(r.issued_at).toLocaleString()}</td>
                                <td className="p-6">
                                    <span className="text-sm font-black text-white">{r.overall_score}%</span>
                                </td>
                                <td className="p-6">
                                    <div className={`flex items-center gap-2 text-[9px] font-black uppercase tracking-widest px-3 py-1 rounded-full border ${
                                        r.is_revoked ? "bg-red-500/10 text-red-500 border-red-500/20" :
                                        r.verdict === "CERTIFIED" ? "bg-emerald-500/10 text-emerald-500 border-emerald-500/20" :
                                        "bg-orange-500/10 text-orange-500 border-orange-500/20"
                                    }`}>
                                        {r.is_revoked ? "REVOKED" : r.verdict}
                                    </div>
                                </td>
                                <td className="p-6">
                                    <div className="flex items-center gap-4">
                                        <button className="text-white bg-white/5 border border-white/10 p-2 rounded-lg hover:bg-white/10 transition-colors shadow-lg">
                                            <Download className="w-4 h-4" />
                                        </button>
                                        {!r.is_revoked && (
                                            <button className="text-[9px] font-black text-red-500 bg-red-500/10 px-3 py-2 rounded-lg border border-red-500/20 hover:bg-red-500 hover:text-white transition-all uppercase">
                                                Revoke
                                            </button>
                                        )}
                                    </div>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
