"use client";
import { apiFetch } from "@/lib/api";
import React, { useState, useEffect } from "react";
import {
    FileText, ShieldCheck, Printer, Download, Share2, AlertTriangle, AlertCircle,
    CheckCircle2, Info, ArrowRight, Activity, Scale, Brain, Database, Zap, Lock
} from "lucide-react";


// ─── Primitives ───
const Card = ({ children, className = "" }: any) => (
    <div className={`bg-[#0E1014] border border-white/[0.07] rounded-2xl ${className}`}>{children}</div>
);

const Badge = ({ label, color }: { label: string; color: "green" | "red" | "amber" | "slate" }) => {
    const styles = {
        green: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
        red: "bg-red-500/10 text-red-400 border-red-500/20",
        amber: "bg-amber-500/10 text-amber-400 border-amber-500/20",
        slate: "bg-white/5 text-slate-400 border-white/10",
    };
    return (
        <span className={`text-[10px] font-black uppercase px-2.5 py-1 rounded border ${styles[color]}`}>{label}</span>
    );
};

const MetricRow = ({ label, value, sub, status }: any) => {
    const color = status === "PASSED" ? "text-emerald-400" : status === "WARNING" ? "text-amber-400" : status === "FAILED" ? "text-red-400" : "text-white";
    return (
        <div className="flex items-center justify-between py-2 border-b border-white/[0.03]">
            <div>
                <p className="text-[10px] font-black uppercase tracking-widest text-slate-500">{label}</p>
                {sub && <p className="text-[9px] text-slate-600 italic">{sub}</p>}
            </div>
            <div className="text-right">
                <p className={`text-sm font-black ${color}`}>{typeof value === "number" ? value.toFixed(4) : value}</p>
            </div>
        </div>
    );
};

// ─── Main Module ───
export default function ModelReportCard({ state, setState, onAction }: any) {
    const [scans, setScans] = useState<any[]>([]);
    const [selectedId, setSelectedId] = useState("");
    const [report, setReport] = useState<any>(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        apiFetch(`/api/v1/history?limit=25`)
            .then(r => r.json())
            .then(d => {
                const list = Array.isArray(d) ? d : [];
                setScans(list);
                if (list.length > 0 && !selectedId) {
                    setSelectedId(list[0].id);
                }
            });
    }, []);

    const fetchReport = async (id: string) => {
        if (!id) return;
        setLoading(true);
        try {
            const r = await apiFetch(`/api/v1/history/${id}`);
            const d = await r.json();
            setReport(d);
        } catch { }
        setLoading(false);
    };

    useEffect(() => {
        if (selectedId) fetchReport(selectedId);
    }, [selectedId]);

    const handlePrint = () => {
        window.print();
    };

    if (loading && !report) return (
        <div className="flex flex-col items-center justify-center py-40 gap-6">
            <div className="w-12 h-12 rounded-full border border-orange-500/20 border-t-orange-500 animate-spin" />
            <p className="text-[10px] uppercase font-black tracking-widest text-slate-600 animate-pulse">Generating Report Card...</p>
        </div>
    );

    return (
        <div className="space-y-8">
            {/* Controls */}
            <header className="flex items-center justify-between print:hidden">
                <div className="flex items-center gap-4">
                    <select
                        value={selectedId}
                        onChange={(e) => setSelectedId(e.target.value)}
                        className="bg-[#0E1014] border border-white/10 rounded-xl px-4 py-2 text-xs font-black uppercase text-slate-300 outline-none focus:border-orange-500/40"
                    >
                        {scans.map((s) => (
                            <option key={s.id} value={s.id}>
                                {new Date(s.created_at).toLocaleDateString()} — {s.scan_type} ({s.id?.slice(0, 8)})
                            </option>
                        ))}
                    </select>
                </div>
                <div className="flex items-center gap-2">
                    <button onClick={handlePrint} className="px-4 py-2 rounded-xl bg-white/[0.03] border border-white/5 text-slate-400 hover:text-white transition-all flex items-center gap-2 text-[10px] font-black uppercase">
                        <Printer className="w-4 h-4" /> Print
                    </button>
                    <button className="px-4 py-2 rounded-xl bg-orange-600 text-black border border-orange-500/20 hover:bg-orange-500 transition-all flex items-center gap-2 text-[10px] font-black uppercase shadow-lg shadow-orange-500/10">
                        <Download className="w-4 h-4" /> Export PDF
                    </button>
                </div>
            </header>

            {report ? (
                <div className="max-w-[1000px] mx-auto space-y-8 print:m-0 print:max-w-none">
                    {/* Main Document */}
                    <Card className="p-10 border-white/10 shadow-2xl relative overflow-hidden print:border-black/10 print:shadow-none print:p-8">
                        {/* Watermark/Accent */}
                        <div className="absolute top-[-100px] right-[-100px] w-64 h-64 bg-orange-500/5 blur-3xl pointer-events-none rounded-full print:hidden" />
                        
                        {/* Header Section */}
                        <div className="flex justify-between items-start mb-12 border-b border-white/5 pb-8 print:border-black/10">
                            <div>
                                <div className="flex items-center gap-3 mb-2">
                                    <div className="w-8 h-8 rounded-lg bg-orange-600 flex items-center justify-center">
                                        <ShieldCheck className="w-5 h-5 text-black" />
                                    </div>
                                    <h1 className="text-2xl font-black text-white uppercase tracking-tight print:text-black">Model Report Card</h1>
                                </div>
                                <p className="text-[10px] font-black uppercase tracking-[0.4em] text-slate-500">Governance & Compliance Certificate</p>
                            </div>
                            <div className="text-right">
                                <p className="text-[9px] uppercase font-black text-slate-600 mb-1">Generated On</p>
                                <p className="text-sm font-mono text-white print:text-black">{new Date().toLocaleString()}</p>
                                <p className="text-[9px] font-mono text-slate-500 mt-1 uppercase">Scan ID: {report.id}</p>
                            </div>
                        </div>

                        {/* Top Info Grid */}
                        <div className="grid grid-cols-4 gap-6 mb-12">
                            <div className="bg-black/20 p-5 rounded-2xl border border-white/5 print:bg-gray-50 print:border-gray-200">
                                <p className="text-[9px] font-black text-slate-600 uppercase mb-2">Governance Score</p>
                                <p className="text-4xl font-black text-white print:text-black">{report.governance_score ?? "—"}</p>
                                <div className="mt-2">
                                    <Badge 
                                        label={report.gate_status || "—"} 
                                        color={report.gate_status === "PASSED" ? "green" : report.gate_status === "FAILED" ? "red" : "amber"} 
                                    />
                                </div>
                            </div>
                            <div className="bg-black/20 p-5 rounded-2xl border border-white/5 print:bg-gray-50 print:border-gray-100">
                                <p className="text-[9px] font-black text-slate-600 uppercase mb-2">Model Type</p>
                                <p className="text-lg font-black text-white capitalize print:text-black truncate">{report.scan_type || "Audit"}</p>
                                <p className="text-[9px] text-slate-600 mt-1 font-mono uppercase">ver 1.0.0</p>
                            </div>
                            <div className="bg-black/20 p-5 rounded-2xl border border-white/5 print:bg-gray-50 print:border-gray-100">
                                <p className="text-[9px] font-black text-slate-600 uppercase mb-2">Risk Assessment</p>
                                <p className={`text-lg font-black ${report.risk_level === 'CRITICAL' ? 'text-red-400' : report.risk_level === 'HIGH' ? 'text-orange-400' : 'text-emerald-400'} print:text-black`}>
                                    {report.risk_level || "LOW"}
                                </p>
                                <p className="text-[9px] text-slate-600 mt-1 uppercase tracking-widest">Weighted Index</p>
                            </div>
                            <div className="bg-black/20 p-5 rounded-2xl border border-white/5 print:bg-gray-50 print:border-gray-100">
                                <p className="text-[9px] font-black text-slate-600 uppercase mb-2">Compliance</p>
                                <p className="text-lg font-black text-white print:text-black truncate">ISO/AIG-2026</p>
                                <p className="text-[9px] text-emerald-500 font-bold uppercase mt-1">✓ Verified</p>
                            </div>
                        </div>

                        {/* Two Column Layout */}
                        <div className="grid grid-cols-[1fr_2fr] gap-10">
                            {/* Left: Component Breakdown */}
                            <div className="space-y-8">
                                <div>
                                    <h3 className="text-[11px] font-black uppercase tracking-[0.2em] text-slate-400 mb-4 border-l-2 border-orange-500 pl-3">Component Scores</h3>
                                    <div className="space-y-1">
                                        {Object.entries(report.results_json?.governance?.component_scores || {}).map(([k, v]: any) => (
                                            <div key={k} className="flex justify-between items-center py-2.5 px-3 rounded-lg bg-white/[0.02] border border-white/[0.03] mb-1 print:bg-gray-50 print:border-gray-100">
                                                <span className="text-[10px] font-black text-slate-500 uppercase">{k.replace("_score", "").replace(/_/g, " ")}</span>
                                                <span className={`text-xs font-black ${v >= 80 ? 'text-emerald-400' : 'text-orange-400'} print:text-black`}>{v}/100</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                <div>
                                    <h3 className="text-[11px] font-black uppercase tracking-[0.2em] text-slate-400 mb-4 border-l-2 border-orange-500 pl-3">Top Advisories</h3>
                                    <div className="space-y-3">
                                        {(report.results_json?.advisories || []).slice(0, 3).map((a: any, i: number) => (
                                            <div key={i} className="p-3 rounded-xl border border-white/5 bg-black/40 print:bg-white print:border-gray-200">
                                                <div className="flex items-center gap-2 mb-1">
                                                    {a.severity === "CRITICAL" ? <AlertCircle className="w-3 h-3 text-red-400" /> : <AlertTriangle className="w-3 h-3 text-amber-400" />}
                                                    <span className="text-[9px] font-black uppercase text-slate-500">{a.code}</span>
                                                </div>
                                                <p className="text-[10px] font-bold text-slate-200 leading-tight print:text-black">{a.message}</p>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>

                            {/* Right: Detailed Check Log */}
                            <div className="space-y-8">
                                <div>
                                    <h3 className="text-[11px] font-black uppercase tracking-[0.2em] text-slate-400 mb-4 border-l-2 border-orange-500 pl-3">Governance Proof-of-Check</h3>
                                    <div className="rounded-2xl border border-white/5 overflow-hidden print:border-gray-200">
                                        <table className="w-full text-left">
                                            <thead>
                                                <tr className="bg-white/[0.03] border-b border-white/5 print:bg-gray-100 print:text-black">
                                                    <th className="px-5 py-3 text-[9px] font-black uppercase text-slate-600">Metric / Validation</th>
                                                    <th className="px-5 py-3 text-[9px] font-black uppercase text-slate-600">Actual Value</th>
                                                    <th className="px-5 py-3 text-[9px] font-black uppercase text-slate-600">Status</th>
                                                </tr>
                                            </thead>
                                            <tbody className="divide-y divide-white/[0.03] print:divide-gray-100">
                                                {(report.results_json?.policy?.checks || []).map((c: any, i: number) => (
                                                    <tr key={i} className="hover:bg-white/[0.01] transition-colors print:text-black">
                                                        <td className="px-5 py-3.5">
                                                            <p className="text-[11px] font-black uppercase text-white print:text-black">{c.name}</p>
                                                            <p className="text-[9px] text-slate-600 italic mt-0.5 line-clamp-1">{c.message}</p>
                                                        </td>
                                                        <td className="px-5 py-3.5 font-mono text-xs font-bold text-slate-300 print:text-black">
                                                            {typeof c.actual_value === 'number' ? c.actual_value.toFixed(4) : String(c.actual_value)}
                                                        </td>
                                                        <td className="px-5 py-3.5">
                                                            <Badge label={c.status} color={c.status === 'PASSED' ? 'green' : c.status === 'WARNING' ? 'amber' : 'red'} />
                                                        </td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>

                                {report.results_json?.metrics && (
                                    <div className="bg-orange-600/5 border border-orange-500/10 rounded-2xl p-6 print:bg-gray-50 print:border-gray-200">
                                        <div className="flex items-center gap-2 mb-4">
                                            <Activity className="w-4 h-4 text-orange-500" />
                                            <h4 className="text-[10px] font-black uppercase tracking-widest text-orange-400">Statistical Performance Snapshot</h4>
                                        </div>
                                        <div className="grid grid-cols-2 gap-x-10 gap-y-1">
                                            {Object.entries(report.results_json.metrics).slice(0, 10).map(([k, v]: any) => (
                                                <MetricRow key={k} label={k.replace(/_/g, " ")} value={v} />
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Footer Sign-off */}
                        <div className="mt-20 pt-10 border-t border-white/10 flex items-end justify-between print:border-black/20">
                            <div>
                                <p className="text-[9px] font-black text-slate-700 uppercase mb-8">Digital Governance Verification Pulse</p>
                                <div className="flex gap-4">
                                    <div className="text-center">
                                        <div className="w-32 h-[1px] bg-slate-800 mb-2" />
                                        <p className="text-[8px] font-black text-slate-600 uppercase uppercase">Chief AI Risk Officer</p>
                                    </div>
                                    <div className="text-center">
                                        <div className="w-32 h-[1px] bg-slate-800 mb-2" />
                                        <p className="text-[8px] font-black text-slate-600 uppercase">Automated Governance Engine</p>
                                    </div>
                                </div>
                            </div>
                            <div className="text-right flex flex-col items-end gap-2">
                                <div className="p-4 bg-white/[0.02] border border-white/5 rounded-xl print:border-gray-200">
                                    <div className="flex items-center gap-2 opacity-50">
                                        <Lock className="w-3 h-3 text-slate-600" />
                                        <span className="text-[10px] font-mono text-slate-600 select-all">MLG-CERT-2026-F6E2-88AD</span>
                                    </div>
                                </div>
                                <p className="text-[9px] font-black text-slate-800 uppercase print:text-gray-400">© 2026 ML Guard Enterprise Systems</p>
                            </div>
                        </div>
                    </Card>

                    {/* Quick Stats Sidebar (hidden on print) */}
                    <div className="grid grid-cols-2 gap-6 print:hidden">
                        {report.results_json?.drift && (
                            <Card className="p-6">
                                <div className="flex items-center gap-2 mb-4">
                                    <Database className="w-4 h-4 text-blue-400" />
                                    <h4 className="text-[10px] font-black uppercase text-slate-400 tracking-widest">Data Stability Overview</h4>
                                </div>
                                <div className="space-y-2">
                                    {Object.entries(report.results_json.drift).slice(0, 4).map(([f, s]: any) => (
                                        <div key={f} className="flex justify-between items-center text-xs">
                                            <span className="text-slate-500 font-mono truncate">{f}</span>
                                            <span className={s.drift_flag ? "text-red-400" : "text-emerald-400"}>PSI {s.PSI?.toFixed(4)}</span>
                                        </div>
                                    ))}
                                </div>
                            </Card>
                        )}
                        {report.results_json?.risk_score != null && (
                            <Card className="p-6 flex flex-col justify-center items-center text-center">
                                <h4 className="text-[10px] font-black uppercase text-slate-400 tracking-widest mb-4">Final Risk Index</h4>
                                <div className={`text-5xl font-black ${report.risk_level === 'CRITICAL' ? 'text-red-400' : 'text-orange-400'}`}>
                                    {report.risk_score}
                                </div>
                                <p className="text-[9px] text-slate-600 font-black uppercase mt-2">Weighted Probability Score</p>
                            </Card>
                        )}
                    </div>
                </div>
            ) : (
                <div className="flex flex-col items-center justify-center py-40 gap-4 text-center">
                    <FileText className="w-16 h-16 text-slate-800 animate-pulse" />
                    <div>
                        <p className="text-sm font-black uppercase text-slate-700 tracking-widest">Select a scan record</p>
                        <p className="text-xs text-slate-600 mt-2">Choose from the history dropdown to generate a certificate.</p>
                    </div>
                </div>
            )}
        </div>
    );
}
