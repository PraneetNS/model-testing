"use client";
import React, { useState, useEffect, useCallback } from "react";
import {
    History, GitCompare, TrendingUp, TrendingDown, Minus, RefreshCw,
    ChevronDown, ChevronUp, CheckCircle2, AlertCircle, AlertTriangle, Clock, Filter,
    ArrowUpRight, ArrowDownRight
} from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "";

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
        <span className={`text-[9px] font-black uppercase px-2 py-0.5 rounded border ${styles[color]}`}>{label}</span>
    );
};

const ScoreBar = ({ score }: { score: number | null }) => {
    if (score == null) return <span className="text-slate-600 text-xs">—</span>;
    const color = score >= 80 ? "#22c55e" : score >= 60 ? "#f59e0b" : "#ef4444";
    return (
        <div className="flex items-center gap-2">
            <div className="w-16 h-1.5 bg-white/5 rounded-full overflow-hidden">
                <div className="h-full rounded-full transition-all" style={{ width: `${Math.min(score, 100)}%`, background: color }} />
            </div>
            <span className="text-xs font-black tabular-nums" style={{ color }}>{Math.round(score)}</span>
        </div>
    );
};

// Simple SVG sparkline
const Sparkline = ({ points, color = "#f97316" }: { points: number[]; color?: string }) => {
    if (!points || points.length < 2) return null;
    const w = 80; const h = 28;
    const min = Math.min(...points); const max = Math.max(...points);
    const range = max - min || 1;
    const xs = points.map((_, i) => (i / (points.length - 1)) * w);
    const ys = points.map(v => h - ((v - min) / range) * h);
    const d = xs.map((x, i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${ys[i].toFixed(1)}`).join(" ");
    const trend = points[points.length - 1] - points[0];
    const tColor = trend > 0 ? "#22c55e" : trend < 0 ? "#ef4444" : "#f59e0b";
    return (
        <div className="flex items-center gap-2">
            <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
                <path d={d} fill="none" stroke={tColor} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            {trend > 0 ? <ArrowUpRight className="w-3 h-3 text-emerald-400" /> : trend < 0 ? <ArrowDownRight className="w-3 h-3 text-red-400" /> : <Minus className="w-3 h-3 text-amber-400" />}
        </div>
    );
};

// ─── Compare Panel ───
function ComparePanel({ scanA, scanB }: { scanA: any; scanB: any }) {
    const [comparison, setComparison] = useState<any>(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        if (!scanA || !scanB) return;
        setLoading(true);
        fetch(`${API_BASE}/api/v1/compare?scan_a=${scanA.id}&scan_b=${scanB.id}`)
            .then(r => r.json())
            .then(d => { setComparison(d); setLoading(false); })
            .catch(() => setLoading(false));
    }, [scanA, scanB]);

    if (!scanA || !scanB) return (
        <div className="flex items-center justify-center h-40 text-slate-600 text-xs font-bold uppercase tracking-widest">
            Select two scans to compare
        </div>
    );

    if (loading) return (
        <div className="flex items-center justify-center h-40">
            <div className="w-8 h-8 rounded-full border border-orange-500/20 border-t-orange-500 animate-spin" />
        </div>
    );

    if (!comparison) return null;

    const delta = comparison.governance_delta;
    const deltaColor = delta > 0 ? "text-emerald-400" : delta < 0 ? "text-red-400" : "text-amber-400";

    return (
        <div className="space-y-4">
            {/* Header */}
            <div className="grid grid-cols-3 gap-2">
                <Card className="p-4 text-center">
                    <p className="text-[9px] uppercase font-black text-slate-600 mb-1">Scan A</p>
                    <p className="text-2xl font-black text-white">{comparison.scan_a.score != null ? Math.round(comparison.scan_a.score) : "—"}</p>
                    <p className="text-[9px] text-slate-500 mt-1 font-mono truncate">{comparison.scan_a.id?.slice(0, 12)}...</p>
                    <p className="text-[8px] text-slate-600 mt-1">{new Date(comparison.scan_a.created_at).toLocaleDateString()}</p>
                </Card>
                <Card className="p-4 text-center border-orange-500/20 flex flex-col items-center justify-center">
                    <p className="text-[9px] uppercase font-black text-slate-600 mb-1">Score Delta</p>
                    <p className={`text-2xl font-black ${deltaColor}`}>{delta != null ? `${delta > 0 ? "+" : ""}${delta.toFixed(1)}` : "—"}</p>
                    <p className="text-[9px] text-slate-500 mt-1">vs Baseline</p>
                </Card>
                <Card className="p-4 text-center">
                    <p className="text-[9px] uppercase font-black text-slate-600 mb-1">Scan B</p>
                    <p className="text-2xl font-black text-white">{comparison.scan_b.score != null ? Math.round(comparison.scan_b.score) : "—"}</p>
                    <p className="text-[9px] text-slate-500 mt-1 font-mono truncate">{comparison.scan_b.id?.slice(0, 12)}...</p>
                    <p className="text-[8px] text-slate-600 mt-1">{new Date(comparison.scan_b.created_at).toLocaleDateString()}</p>
                </Card>
            </div>

            {/* Metrics comparison */}
            {comparison.metrics_comparison && Object.keys(comparison.metrics_comparison).length > 0 && (
                <Card className="overflow-hidden">
                    <div className="px-5 py-3 border-b border-white/5 bg-white/[0.02]">
                        <p className="text-[10px] font-black uppercase tracking-widest text-slate-400">Metrics Comparison</p>
                    </div>
                    <div className="divide-y divide-white/[0.03]">
                        {Object.entries(comparison.metrics_comparison).map(([key, val]: any) => (
                            <div key={key} className="grid grid-cols-3 px-5 py-2.5 text-xs">
                                <span className="font-mono text-slate-400 capitalize">{key.replace(/_/g, " ")}</span>
                                <span className="text-white font-black text-center">{val.scan_a != null ? (typeof val.scan_a === "number" ? val.scan_a.toFixed(4) : val.scan_a) : "—"}</span>
                                <div className="flex items-center justify-end gap-2">
                                    <span className="text-white font-black">{val.scan_b != null ? (typeof val.scan_b === "number" ? val.scan_b.toFixed(4) : val.scan_b) : "—"}</span>
                                    {val.delta != null && (
                                        <span className={`text-[9px] font-black ${val.delta > 0 ? "text-emerald-400" : val.delta < 0 ? "text-red-400" : "text-slate-500"}`}>
                                            {val.delta > 0 ? "+" : ""}{val.delta.toFixed(4)}
                                        </span>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                </Card>
            )}
        </div>
    );
}

// ─── Main Module ───
export default function ScanHistoryPage({ state, setState, onAction }: any) {
    const [scans, setScans] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const [selectedA, setSelectedA] = useState<any>(null);
    const [selectedB, setSelectedB] = useState<any>(null);
    const [tab, setTab] = useState<"history" | "compare">("history");
    const [typeFilter, setTypeFilter] = useState("");
    const [gateFilter, setGateFilter] = useState("");
    const [trajectories, setTrajectories] = useState<Record<string, number[]>>({});
    const [expandedScan, setExpandedScan] = useState<string | null>(null);
    const [scanDetails, setScanDetails] = useState<Record<string, any>>({});

    const fetchHistory = useCallback(async () => {
        setLoading(true);
        try {
            const params = new URLSearchParams({ limit: "50" });
            if (typeFilter) params.set("scan_type", typeFilter);
            const r = await fetch(`${API_BASE}/api/v1/history?${params}`);
            const d = await r.json();
            const list = Array.isArray(d) ? d : [];
            setScans(list);

            // Fetch trajectories for unique model_ids
            const modelIds = [...new Set(list.map((s: any) => s.model_id).filter(Boolean))];
            const trajMap: Record<string, number[]> = {};
            await Promise.all(modelIds.map(async (mid: string) => {
                try {
                    const tr = await fetch(`${API_BASE}/api/v1/history/trajectory/${mid}`);
                    const td = await tr.json();
                    if (td.data_points) {
                        trajMap[mid] = td.data_points.map((p: any) => p.score).filter((s: any) => s != null);
                    }
                } catch { }
            }));
            setTrajectories(trajMap);
        } catch { }
        setLoading(false);
    }, [typeFilter]);

    useEffect(() => { fetchHistory(); }, [fetchHistory]);

    const fetchScanDetail = async (scanId: string) => {
        if (scanDetails[scanId]) {
            setExpandedScan(s => s === scanId ? null : scanId);
            return;
        }
        try {
            const r = await fetch(`${API_BASE}/api/v1/history/${scanId}`);
            const d = await r.json();
            setScanDetails(prev => ({ ...prev, [scanId]: d }));
            setExpandedScan(scanId);
        } catch { }
    };

    const toggleSelectForCompare = (scan: any) => {
        if (selectedA?.id === scan.id) { setSelectedA(null); return; }
        if (selectedB?.id === scan.id) { setSelectedB(null); return; }
        if (!selectedA) { setSelectedA(scan); return; }
        if (!selectedB) { setSelectedB(scan); return; }
        // Replace B with the oldest one
        setSelectedA(selectedB);
        setSelectedB(scan);
    };

    const isSelected = (scan: any) => selectedA?.id === scan.id || selectedB?.id === scan.id;
    const gateColor = (g: string) => g === "PASSED" ? "green" : g === "FAILED" ? "red" : g === "WARNING" ? "amber" : "slate";

    const filtered = gateFilter ? scans.filter(s => s.gate_status === gateFilter) : scans;

    return (
        <div className="space-y-6">
            {/* Header Controls */}
            <div className="flex items-center justify-between">
                <div className="flex bg-black p-1 rounded-xl border border-white/5 gap-1">
                    <button onClick={() => setTab("history")} className={`px-4 py-1.5 rounded-lg text-[10px] font-black uppercase transition-all ${tab === "history" ? "bg-orange-600 text-black" : "text-slate-500 hover:text-white"}`}>
                        <span className="flex items-center gap-2"><History className="w-3 h-3" />History</span>
                    </button>
                    <button onClick={() => setTab("compare")} className={`px-4 py-1.5 rounded-lg text-[10px] font-black uppercase transition-all ${tab === "compare" ? "bg-orange-600 text-black" : "text-slate-500 hover:text-white"}`}>
                        <span className="flex items-center gap-2"><GitCompare className="w-3 h-3" />Compare {selectedA && selectedB ? "(2 loaded)" : selectedA ? "(1 loaded)" : ""}</span>
                    </button>
                </div>
                <div className="flex items-center gap-3">
                    <select value={gateFilter} onChange={e => setGateFilter(e.target.value)}
                        className="bg-black/40 border border-white/5 rounded-lg px-3 py-1.5 text-[10px] text-slate-400 font-black uppercase outline-none focus:border-orange-500/40">
                        <option value="">All Gates</option>
                        <option value="PASSED">Passed</option>
                        <option value="FAILED">Failed</option>
                        <option value="WARNING">Warning</option>
                    </select>
                    <select value={typeFilter} onChange={e => setTypeFilter(e.target.value)}
                        className="bg-black/40 border border-white/5 rounded-lg px-3 py-1.5 text-[10px] text-slate-400 font-black uppercase outline-none focus:border-orange-500/40">
                        <option value="">All Types</option>
                        <option value="governance">Governance</option>
                        <option value="behavior">Behavior</option>
                        <option value="explainability">Explainability</option>
                    </select>
                    <button onClick={fetchHistory} className="p-2 rounded-lg bg-white/[0.03] border border-white/5 text-slate-500 hover:text-white transition-colors">
                        <RefreshCw className="w-3.5 h-3.5" />
                    </button>
                </div>
            </div>

            {tab === "compare" && (
                <div className="space-y-4">
                    {(selectedA || selectedB) && (
                        <div className="flex gap-3 p-4 rounded-xl border border-orange-500/20 bg-orange-500/[0.03]">
                            <div className="flex-1">
                                <p className="text-[9px] uppercase font-black text-slate-600 mb-1">Scan A (Baseline)</p>
                                {selectedA ? (
                                    <p className="text-xs font-black text-white">{selectedA.scan_type} — Score: {selectedA.governance_score ?? "—"} <span className="text-slate-600 font-mono">({selectedA.id?.slice(0, 10)}...)</span></p>
                                ) : <p className="text-xs text-slate-600">Not selected — click a row below</p>}
                            </div>
                            <div className="flex-1">
                                <p className="text-[9px] uppercase font-black text-slate-600 mb-1">Scan B (Candidate)</p>
                                {selectedB ? (
                                    <p className="text-xs font-black text-white">{selectedB.scan_type} — Score: {selectedB.governance_score ?? "—"} <span className="text-slate-600 font-mono">({selectedB.id?.slice(0, 10)}...)</span></p>
                                ) : <p className="text-xs text-slate-600">Not selected — click a row below</p>}
                            </div>
                        </div>
                    )}
                    <ComparePanel scanA={selectedA} scanB={selectedB} />
                    <p className="text-[9px] text-slate-600 text-center">Select rows from the table below to populate comparison</p>
                </div>
            )}

            {/* Scan History Table */}
            {loading ? (
                <div className="flex items-center justify-center h-48">
                    <div className="w-10 h-10 rounded-full border border-orange-500/20 border-t-orange-500 animate-spin" />
                </div>
            ) : filtered.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-48 gap-3 text-center">
                    <History className="w-10 h-10 text-slate-800" />
                    <p className="text-sm font-black uppercase text-slate-700 tracking-widest">No Scan History</p>
                    <p className="text-xs text-slate-600">Run a Model Audit to generate scan records.</p>
                </div>
            ) : (
                <Card className="overflow-hidden">
                    <div className="flex items-center justify-between px-5 py-3 border-b border-white/5 bg-white/[0.02]">
                        <p className="text-[10px] font-black uppercase tracking-widest text-slate-400">
                            {filtered.length} Scan Records {tab === "compare" ? "— Click to select for comparison" : "— Click to expand details"}
                        </p>
                        <div className="flex items-center gap-2 text-[9px] font-black uppercase text-slate-600">
                            <span className="w-3 h-3 bg-orange-500/20 rounded border border-orange-500/20 inline-block" />
                            Selected for Compare
                        </div>
                    </div>
                    <div className="divide-y divide-white/[0.03] max-h-[500px] overflow-y-auto">
                        {filtered.map((s: any) => (
                            <div key={s.id}>
                                <div
                                    onClick={() => tab === "compare" ? toggleSelectForCompare(s) : fetchScanDetail(s.id)}
                                    className={`grid grid-cols-[1fr_auto_auto_auto_auto_auto] items-center gap-4 px-5 py-3 cursor-pointer transition-all hover:bg-white/[0.02] ${isSelected(s) ? "bg-orange-500/[0.04] border-l-2 border-orange-500/40" : ""}`}
                                >
                                    <div>
                                        <p className="text-[10px] font-black text-white capitalize">{s.scan_type || "audit"}</p>
                                        <p className="text-[9px] font-mono text-slate-600">{s.id?.slice(0, 14)}...</p>
                                    </div>
                                    <ScoreBar score={s.governance_score} />
                                    <Badge label={s.gate_status || "—"} color={gateColor(s.gate_status) as any} />
                                    <div className="text-right">
                                        {trajectories[s.model_id] && <Sparkline points={trajectories[s.model_id]} />}
                                    </div>
                                    <div className="text-right">
                                        <p className="text-[9px] text-slate-500">{s.duration_ms != null ? `${s.duration_ms}ms` : "—"}</p>
                                        <p className="text-[8px] text-slate-700">{new Date(s.created_at).toLocaleDateString()}</p>
                                    </div>
                                    {tab !== "compare" && (expandedScan === s.id ? <ChevronUp className="w-3.5 h-3.5 text-slate-600" /> : <ChevronDown className="w-3.5 h-3.5 text-slate-600" />)}
                                </div>
                                {/* Expanded details */}
                                {tab !== "compare" && expandedScan === s.id && scanDetails[s.id] && (
                                    <div className="px-5 pb-5 bg-black/20 border-t border-white/5">
                                        <div className="pt-4 grid grid-cols-2 gap-4">
                                            <div>
                                                <p className="text-[9px] uppercase font-black text-slate-600 mb-2">Checks Run</p>
                                                <div className="flex flex-wrap gap-1">
                                                    {(scanDetails[s.id]?.checks_run ?? []).map((c: string) => (
                                                        <span key={c} className="text-[8px] font-bold px-2 py-0.5 rounded bg-white/5 text-slate-400 border border-white/10">{c}</span>
                                                    ))}
                                                </div>
                                            </div>
                                            <div>
                                                <p className="text-[9px] uppercase font-black text-slate-600 mb-2">Key Metrics</p>
                                                <div className="space-y-1">
                                                    {Object.entries(scanDetails[s.id]?.results_json?.metrics ?? {}).slice(0, 4).map(([k, v]: any) => (
                                                        <div key={k} className="flex justify-between text-[10px]">
                                                            <span className="text-slate-500 capitalize">{k.replace(/_/g, " ")}</span>
                                                            <span className="font-black text-white">{typeof v === "number" ? v.toFixed(4) : String(v)}</span>
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>
                                        </div>
                                        <div className="mt-3 pt-3 border-t border-white/5">
                                            <p className="text-[9px] text-slate-600 font-mono">Model ID: {scanDetails[s.id]?.model_id} · Trigger: {scanDetails[s.id]?.trigger_source || "manual"}</p>
                                        </div>
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                </Card>
            )}
        </div>
    );
}
