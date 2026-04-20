"use client";
import { apiFetch, safeJson } from "@/lib/api";
import React, { useState, useEffect, useCallback } from "react";
import {
    Activity, TrendingDown, TrendingUp, Minus, AlertTriangle, CheckCircle2,
    RefreshCw, Database, Clock, Zap, Filter, ChevronDown, Eye, BarChart3,
    ArrowUpRight, ArrowDownRight, Shield, Cpu, Download
} from "lucide-react";
import {
    LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip,
    ResponsiveContainer, ReferenceLine, CartesianGrid, Cell
} from "recharts";


function UmapScatterPlot({ data }: { data: { reference_points: number[][], current_points: number[][] } }) {
    const canvasRef = React.useRef<HTMLCanvasElement>(null);
    React.useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        
        const width = canvas.width;
        const height = canvas.height;
        ctx.clearRect(0, 0, width, height);
        
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        const allPoints = [...(data.reference_points || []), ...(data.current_points || [])];
        if (allPoints.length === 0) return;
        
        allPoints.forEach(([x, y]) => {
            if (x < minX) minX = x; if (x > maxX) maxX = x;
            if (y < minY) minY = y; if (y > maxY) maxY = y;
        });
        
        const padding = 15;
        const scaleX = (width - padding * 2) / (maxX - minX || 1);
        const scaleY = (height - padding * 2) / (maxY - minY || 1);
        
        const drawPoints = (pts: number[][], color: string) => {
            ctx.fillStyle = color;
            pts.forEach(([x, y]) => {
                const cx = padding + (x - minX) * scaleX;
                const cy = padding + (y - minY) * scaleY;
                ctx.beginPath();
                ctx.arc(cx, cy, 3, 0, Math.PI * 2);
                ctx.fill();
            });
        };
        
        drawPoints(data.reference_points || [], '#4ade80');
        drawPoints(data.current_points || [], '#f97316');
    }, [data]);
    
    return <canvas ref={canvasRef} width={800} height={300} className="w-full h-[300px] rounded-xl bg-black/20" />;
}



// ─── Primitives ───────────────────────────────────────────────────────────────
const Badge = ({ label, variant = "neutral" }: { label: string; variant?: string }) => {
    const cls = variant === "critical" ? "bg-red-500/10 text-red-400 border-red-500/30"
        : variant === "high" ? "bg-orange-500/10 text-orange-400 border-orange-500/30"
            : variant === "medium" ? "bg-yellow-500/10 text-yellow-400 border-yellow-500/30"
                : variant === "ok" ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                    : "bg-white/5 text-slate-400 border-white/10";
    return <span className={`text-[9px] font-black uppercase px-2 py-0.5 rounded border ${cls}`}>{label}</span>;
};

const StatCard = ({ label, value, sub, trend, icon: Icon, accent }: any) => {
    const TrendIcon = trend === "up" ? ArrowUpRight : trend === "down" ? ArrowDownRight : Minus;
    const trendColor = trend === "up" ? "text-emerald-400" : trend === "down" ? "text-red-400" : "text-slate-600";
    return (
        <div className="bg-[#0E1014] border border-white/[0.06] rounded-2xl p-5 flex flex-col gap-3">
            <div className="flex items-center justify-between">
                <p className="text-[9px] font-black uppercase tracking-[0.2em] text-slate-600">{label}</p>
                {Icon && <Icon className="w-4 h-4 text-slate-700" />}
            </div>
            <div className="flex items-end gap-2">
                <p className={`text-2xl font-black ${accent ? "text-orange-400" : "text-white"}`}>{value ?? "—"}</p>
                {trend && <TrendIcon className={`w-4 h-4 mb-1 ${trendColor}`} />}
            </div>
            {sub && <div className="text-[10px] text-slate-600 font-medium">{sub}</div>}
        </div>
    );
};

const severityVariant = (s: string) =>
    s === "CRITICAL" ? "critical" : s === "HIGH" ? "high" : s === "MEDIUM" ? "medium" : "ok";

// ─── Feature Drift Table ──────────────────────────────────────────────────────
function FeatureDriftTable({ features, onSelect }: { features: any[]; onSelect: (f: string) => void }) {
    return (
        <div className="rounded-xl border border-white/[0.05] overflow-hidden">
            <div className="px-4 py-3 bg-white/[0.02] border-b border-white/5 flex items-center justify-between">
                <p className="text-[10px] font-black uppercase tracking-widest text-slate-400">Feature Drift Analysis</p>
                <p className="text-[9px] text-slate-600">{features.length} features monitored</p>
            </div>
            <table className="w-full text-xs">
                <thead>
                    <tr className="border-b border-white/[0.04]">
                        {["Feature", "Type", "Drift Score", "Method", "Status", "Severity"].map(h => (
                            <th key={h} className="text-left px-4 py-2.5 text-[9px] font-black uppercase text-slate-600">{h}</th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {features.map((f, i) => (
                        <tr
                            key={f.feature_name}
                            className={`border-b border-white/[0.03] hover:bg-white/[0.02] cursor-pointer transition-colors ${i % 2 === 0 ? "bg-black/10" : ""}`}
                            onClick={() => onSelect(f.feature_name)}
                        >
                            <td className="px-4 py-3 font-mono font-bold text-slate-200">{f.feature_name}</td>
                            <td className="px-4 py-3 text-slate-500 text-[10px]">{f.type}</td>
                            <td className="px-4 py-3 font-black tabular-nums"
                                style={{ color: f.drift_score > 0.3 ? "#f87171" : f.drift_score > 0.15 ? "#fb923c" : "#4ade80" }}>
                                {(f.drift_score || 0).toFixed(4)}
                            </td>
                            <td className="px-4 py-3 text-[10px] text-slate-500 font-mono">{f.method}</td>
                            <td className="px-4 py-3">
                                {f.drift_detected
                                    ? <span className="text-[9px] text-orange-400 font-black flex items-center gap-1"><AlertTriangle className="w-3 h-3" /> Drift</span>
                                    : <span className="text-[9px] text-emerald-400 font-black flex items-center gap-1"><CheckCircle2 className="w-3 h-3" /> Stable</span>}
                            </td>
                            <td className="px-4 py-3">
                                <Badge label={f.severity || "NONE"} variant={severityVariant(f.severity || "NONE")} />
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}

// ─── Performance Timeline Chart ───────────────────────────────────────────────
function PerformanceChart({ timeline, baseline }: { timeline: any[]; baseline?: any }) {
    const data = [...timeline].reverse().map((s, i) => ({
        time: new Date(s.computed_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        accuracy: s.metrics?.accuracy,
        f1: s.metrics?.f1,
        roc_auc: s.metrics?.roc_auc,
    }));

    return (
        <div className="bg-[#0E1014] border border-white/[0.06] rounded-2xl p-5">
            <div className="flex items-center justify-between mb-5">
                <p className="text-[10px] font-black uppercase tracking-widest text-slate-400">Performance Timeline</p>
                <div className="flex items-center gap-4 text-[9px] text-slate-500">
                    <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-orange-400 inline-block rounded" /> Accuracy</span>
                    <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-blue-400 inline-block rounded" /> F1</span>
                    <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-emerald-400 inline-block rounded" /> ROC-AUC</span>
                    {baseline?.accuracy && <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-red-500 border-t border-dashed border-red-500 inline-block" /> Baseline</span>}
                </div>
            </div>
            <ResponsiveContainer width="100%" height={220}>
                <LineChart data={data}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#ffffff08" />
                    <XAxis dataKey="time" tick={{ fill: "#475569", fontSize: 9 }} tickLine={false} axisLine={false} />
                    <YAxis domain={[0, 1]} tick={{ fill: "#475569", fontSize: 9 }} tickLine={false} axisLine={false} tickFormatter={(v: number) => v.toFixed(2)} />
                    <Tooltip
                        contentStyle={{ background: "#0E1014", border: "1px solid #ffffff10", borderRadius: 8, fontSize: 11 }}
                        labelStyle={{ color: "#94a3b8" }}
                    />
                    <Line type="monotone" dataKey="accuracy" stroke="#f97316" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="f1" stroke="#60a5fa" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="roc_auc" stroke="#34d399" strokeWidth={2} dot={false} />
                    {baseline?.accuracy && <ReferenceLine y={baseline.accuracy} stroke="#ef4444" strokeDasharray="6 3" strokeWidth={1} />}
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
}

// ─── Prediction Volume Sparkline ──────────────────────────────────────────────
function SparklineBar({ data }: { data: { hour: number; count: number }[] }) {
    return (
        <ResponsiveContainer width="100%" height={48}>
            <BarChart data={data} barSize={4} margin={{ top: 0, bottom: 0, left: 0, right: 0 }}>
                <Bar dataKey="count" radius={[2, 2, 0, 0]}>
                    {data.map((_, i) => (
                        <Cell key={i} fill={i === data.length - 1 ? "#f97316" : "#ffffff15"} />
                    ))}
                </Bar>
            </BarChart>
        </ResponsiveContainer>
    );
}

// ─── Live Governance Score Widget ─────────────────────────────────────────────
function LiveGovScore({ score, decay, driftPenalty }: { score: number; decay: number; driftPenalty: number }) {
    const color = score >= 80 ? "#34d399" : score >= 60 ? "#fbbf24" : "#f87171";
    return (
        <div className="bg-[#0E1014] border border-white/[0.06] rounded-2xl p-5 flex flex-col gap-4">
            <p className="text-[9px] font-black uppercase tracking-[0.2em] text-slate-600">Live Governance Score</p>
            <div className="flex items-end gap-3">
                <span className="text-5xl font-black tabular-nums" style={{ color }}>{Math.round(score)}</span>
                <span className="text-slate-600 text-sm mb-1">/100</span>
            </div>
            <div className="space-y-2">
                <div className="flex justify-between text-[9px] text-slate-600">
                    <span>Drift Penalty</span><span className="text-red-400 font-black">{(driftPenalty * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full h-1.5 bg-white/5 rounded-full overflow-hidden">
                    <div className="h-full rounded-full transition-all duration-700" style={{ width: `${score}%`, backgroundColor: color }} />
                </div>
            </div>
            <p className="text-[9px] text-slate-600">Decays live with drift · recovers on re-audit</p>
        </div>
    );
}

// ─── Main Observability Module ────────────────────────────────────────────────
export default function ObservabilityModule({ state, setState, onAction }: any) {
    const [modelId, setModelId] = useState(state.modelId || "");
    const [overview, setOverview] = useState<any>(null);
    const [driftReport, setDriftReport] = useState<any>(null);
    const [embeddingReport, setEmbeddingReport] = useState<any>(null);
    const [perfTimeline, setPerfTimeline] = useState<any[]>([]);
    const [selectedFeature, setSelectedFeature] = useState<string | null>(null);
    const [feedData, setFeedData] = useState<any[]>([]);
    const [activeTab, setActiveTab] = useState<"overview" | "drift" | "performance" | "feed">("feed");
    const [loading, setLoading] = useState(false);
    const [models, setModels] = useState<any[]>([]);

    // Load global feed always
    const loadFeed = useCallback(async () => {
        try {
            const r = await apiFetch(`/api/v1/observe/feed`);
            const d = await safeJson<any>(r);
            setFeedData(d.models || []);
            if (!modelId && d.models?.[0]) setModelId(d.models[0].model_id);
        } catch { }
    }, []);

    // Load model list
    const loadModels = useCallback(async () => {
        try {
            const r = await apiFetch(`/api/v1/models`);
            const d = await safeJson<any>(r);
            // Handle both flat array and paginated items object
            const items = Array.isArray(d) ? d : (d.items || []);
            setModels(items);
        } catch { }
    }, []);

    useEffect(() => {
        loadFeed();
        loadModels();
    }, [loadFeed, loadModels]);

    const loadModelData = useCallback(async () => {
        if (!modelId) return;
        setLoading(true);
        try {
            const [ov, dr, pt, er] = await Promise.allSettled([
                apiFetch(`/api/v1/observe/${modelId}/overview`).then(r => safeJson<any>(r)),
                apiFetch(`/api/v1/observe/drift/${modelId}/report`).then(r => safeJson<any>(r)),
                apiFetch(`/api/v1/observe/performance/${modelId}/timeline?limit=24`).then(r => safeJson<any>(r)),
                apiFetch(`/api/v1/drift/${modelId}/embedding-report`).then(r => safeJson<any>(r))
            ]);
            if (ov.status === "fulfilled") setOverview(ov.value);
            if (dr.status === "fulfilled") setDriftReport(dr.value);
            if (pt.status === "fulfilled") setPerfTimeline(pt.value?.timeline || []);
            if (er.status === "fulfilled" && !er.value.detail) setEmbeddingReport(er.value); else setEmbeddingReport(null);
        } finally {
            setLoading(false);
        }
    }, [modelId]);

    useEffect(() => {
        if (modelId && activeTab !== "feed") loadModelData();
    }, [modelId, activeTab, loadModelData]);

    const tabs = [
        { id: "feed", label: "Global Feed", icon: Eye },
        { id: "overview", label: "Model Overview", icon: Activity },
        { id: "drift", label: "Drift Analysis", icon: BarChart3 },
        { id: "performance", label: "Performance", icon: Cpu },
    ] as const;

    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-xl font-black text-white tracking-tight uppercase">Production Observability</h2>
                    <p className="text-[10px] text-slate-600 font-bold uppercase tracking-widest mt-1">
                        Governance-First · Evidently + Arize Equivalent
                    </p>
                </div>
                {activeTab !== "feed" && (
                    <div className="flex items-center gap-3">
                        <select
                            value={modelId}
                            onChange={e => setModelId(e.target.value)}
                            className="bg-[#0E1014] border border-white/10 text-white text-xs font-bold rounded-xl px-4 py-2.5 focus:outline-none focus:border-orange-500/50"
                        >
                            <option value="">Select model...</option>
                            {models.map((m: any) => (
                                <option key={m.id || m} value={m.id || m}>{m.name || m.id || m}</option>
                            ))}
                            {feedData.map(f => (
                                <option key={f.model_id} value={f.model_id}>{f.model_id}</option>
                            ))}
                        </select>
                        <button
                            onClick={loadModelData}
                            disabled={loading}
                            className="p-2.5 rounded-xl bg-orange-500/10 border border-orange-500/20 text-orange-400 hover:bg-orange-500/20 transition-colors"
                        >
                            <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
                        </button>
                    </div>
                )}
            </div>

            {/* Tab navigation */}
            <div className="flex gap-1 p-1 bg-black/20 rounded-xl border border-white/5">
                {tabs.map(t => (
                    <button
                        key={t.id}
                        onClick={() => setActiveTab(t.id)}
                        className={`flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-[10px] font-black uppercase tracking-wider transition-all ${activeTab === t.id
                            ? "bg-orange-500/10 text-orange-400 border border-orange-500/20"
                            : "text-slate-600 hover:text-slate-400"
                            }`}
                    >
                        <t.icon className="w-3.5 h-3.5" />
                        {t.label}
                    </button>
                ))}
            </div>

            {/* ── GLOBAL FEED ───────────────────────────── */}
            {activeTab === "feed" && (
                <div className="space-y-4">
                    <div className="flex items-center justify-between">
                        <p className="text-[10px] font-black uppercase tracking-widest text-slate-500">
                            {feedData.length} models monitored
                        </p>
                        <button onClick={loadFeed} className="text-[9px] text-orange-400 hover:text-orange-300 font-black flex items-center gap-1">
                            <RefreshCw className="w-3 h-3" /> Refresh
                        </button>
                    </div>
                    {feedData.length === 0 ? (
                        <div className="flex flex-col items-center py-24 text-center gap-4">
                            <Eye className="w-12 h-12 text-slate-800" />
                            <p className="text-slate-500 text-sm font-bold">No models being monitored yet.</p>
                            <p className="text-slate-700 text-xs">Ingest predictions via <code className="text-orange-400">/api/v1/ingest/predict</code> to start.</p>
                        </div>
                    ) : (
                        <div className="grid grid-cols-1 gap-3">
                            {feedData.map(model => (
                                <div
                                    key={model.model_id}
                                    className="bg-[#0E1014] border border-white/[0.06] rounded-2xl p-5 flex items-center gap-6 hover:border-orange-500/20 transition-all cursor-pointer"
                                    onClick={() => { setModelId(model.model_id); setActiveTab("overview"); }}
                                >
                                    {/* Score */}
                                    <div className="text-center shrink-0 w-16">
                                        <p className="text-2xl font-black"
                                            style={{ color: model.live_governance_score >= 80 ? "#34d399" : model.live_governance_score >= 60 ? "#fbbf24" : "#f87171" }}>
                                            {Math.round(model.live_governance_score || 0)}
                                        </p>
                                        <p className="text-[8px] text-slate-600 font-black uppercase">Live Score</p>
                                    </div>

                                    <div className="w-px h-10 bg-white/5 shrink-0" />

                                    {/* Info */}
                                    <div className="flex-1 min-w-0">
                                        <div className="flex items-center gap-2 mb-1">
                                            <p className="text-sm font-black text-white truncate">{model.model_id}</p>
                                            <Badge label={model.drift_status || "NONE"} variant={severityVariant(model.drift_status)} />
                                            {model.alert_triggered && <Badge label="Alert" variant="critical" />}
                                        </div>
                                        <div className="flex items-center gap-4 text-[9px] text-slate-600">
                                            <span className="flex items-center gap-1"><Activity className="w-3 h-3" />{model.predictions_24h || 0} preds/24h</span>
                                            {model.last_drift_score != null && (
                                                <span className="flex items-center gap-1"><BarChart3 className="w-3 h-3" />PSI {model.last_drift_score.toFixed(3)}</span>
                                            )}
                                            {model.days_since_last_audit != null && (
                                                <span className="flex items-center gap-1"><Clock className="w-3 h-3" />Audited {model.days_since_last_audit}d ago</span>
                                            )}
                                        </div>
                                    </div>

                                    <ArrowUpRight className="w-4 h-4 text-slate-700 shrink-0" />
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}

            {/* ── MODEL OVERVIEW ────────────────────────── */}
            {activeTab === "overview" && overview && (
                <div className="space-y-6">
                    {/* Stats row */}
                    <div className="grid grid-cols-4 gap-4">
                        <StatCard
                            label="Predictions 24h"
                            value={overview.predictions_24h?.toLocaleString()}
                            icon={Activity}
                            sub={<SparklineBar data={overview.sparkline || []} />}
                        />
                        <StatCard
                            label="Avg Latency"
                            value={overview.avg_latency_ms ? `${overview.avg_latency_ms}ms` : "N/A"}
                            icon={Zap}
                            sub="P50 inference time"
                        />
                        <div className="col-span-1">
                            <LiveGovScore
                                score={overview.live_governance_score || 0}
                                decay={overview.drift_penalty || 0}
                                driftPenalty={overview.drift_penalty || 0}
                            />
                        </div>
                        <StatCard
                            label="Drift Status"
                            value={overview.drift_status || "N/A"}
                            icon={BarChart3}
                            accent={overview.drift_status !== "NONE"}
                            sub="Current window"
                        />
                    </div>

                    {/* Feature drift summary */}
                    {overview.feature_drift_summary?.length > 0 && (
                        <div className="bg-[#0E1014] border border-white/[0.06] rounded-2xl p-5">
                            <p className="text-[10px] font-black uppercase tracking-widest text-slate-400 mb-4">Top Drifted Features</p>
                            <FeatureDriftTable
                                features={overview.feature_drift_summary}
                                onSelect={(f) => { setSelectedFeature(f); setActiveTab("drift"); }}
                            />
                        </div>
                    )}

                    {perfTimeline.length > 0 && (
                        <PerformanceChart timeline={perfTimeline} />
                    )}
                </div>
            )}

            {/* ── DRIFT ANALYSIS ────────────────────────── */}
            {activeTab === "drift" && (
                <div className="space-y-6">
                    <div className="flex items-center justify-between">
                        <div>
                            {driftReport?.drift_detected
                                ? <Badge label={`Drift Detected — ${driftReport.max_severity}`} variant={severityVariant(driftReport.max_severity)} />
                                : <Badge label="Stable — No Significant Drift" variant="ok" />}
                        </div>
                        <div className="flex gap-3">
                            <button
                                onClick={() => apiFetch(`/api/v1/observe/drift/${modelId}/set-baseline`, { method: "POST" }).then(loadModelData)}
                                className="text-[9px] font-black text-slate-400 hover:text-white border border-white/10 rounded-lg px-3 py-2 transition-colors"
                            >
                                Set Current as Baseline
                            </button>
                        </div>
                    </div>

                    {embeddingReport && embeddingReport.umap_snapshot && (
                        <div className="bg-[#0E1014] border border-white/[0.06] rounded-2xl p-5 mb-6">
                            <h3 className="text-sm font-bold text-white mb-2">Embedding Drift UMAP</h3>
                            <div className="flex justify-between items-center mb-4 text-xs font-mono">
                                <span className="text-emerald-400">Baseline</span>
                                <span className="text-slate-400">cosine: {embeddingReport.cosine_drift?.toFixed(4)} | mmd: {embeddingReport.mmd_score?.toFixed(4)}</span>
                                <span className="text-orange-400">Current</span>
                            </div>
                            <UmapScatterPlot data={embeddingReport.umap_snapshot} />
                        </div>
                    )}

                    {driftReport?.feature_results?.length > 0 ? (
                        <FeatureDriftTable features={driftReport.feature_results} onSelect={setSelectedFeature} />
                    ) : (
                        <div className="flex flex-col items-center py-24 text-center gap-4">
                            <BarChart3 className="w-12 h-12 text-slate-800" />
                            <p className="text-slate-500 text-sm font-bold">No drift report available.</p>
                            <p className="text-slate-700 text-xs">Ingest at least 30 predictions and wait for the hourly scan, or click Refresh.</p>
                        </div>
                    )}

                    {driftReport && (
                        <div className="grid grid-cols-3 gap-4">
                            <StatCard label="Overall Drift Score" value={(driftReport.overall_drift_score || 0).toFixed(4)} icon={BarChart3} />
                            <StatCard label="Method" value={driftReport.method?.toUpperCase() || "KS"} icon={Cpu} />
                            <StatCard label="Sample Count" value={driftReport.sample_count?.toLocaleString()} icon={Database} />
                        </div>
                    )}
                </div>
            )}

            {/* ── PERFORMANCE ───────────────────────────── */}
            {activeTab === "performance" && (
                <div className="space-y-6">
                    {perfTimeline.length > 0 ? (
                        <>
                            <PerformanceChart timeline={perfTimeline} />
                            <div className="grid grid-cols-3 gap-4">
                                {Object.entries(perfTimeline[0]?.metrics || {}).map(([k, v]: any) => (
                                    <StatCard
                                        key={k}
                                        label={k.toUpperCase()}
                                        value={typeof v === "number" ? v.toFixed(4) : String(v)}
                                        icon={TrendingUp}
                                    />
                                ))}
                            </div>
                        </>
                    ) : (
                        <div className="flex flex-col items-center py-24 text-center gap-4">
                            <Cpu className="w-12 h-12 text-slate-800" />
                            <p className="text-slate-500 text-sm font-bold">No performance snapshots yet.</p>
                            <p className="text-slate-700 text-xs">Add ground truth labels via <code className="text-orange-400">/api/v1/ingest/label</code> to unlock metrics.</p>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
