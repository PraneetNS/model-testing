"use client";
import React, { useState, useEffect } from "react";
import {
    ShieldCheck, Activity, Wifi, FileText, AlertCircle, CheckCircle2,
    Loader2, Upload, ChevronDown, ChevronUp, AlertTriangle, Info,
    Building2, Users, FolderOpen, KeyRound, Bell, GitBranch, BarChart3,
    Clock, ArrowUpDown, TrendingUp, TrendingDown, Minus, Shield, Eye, LogOut, User,
    Scale, Brain, Zap, Package, FlaskConical, Target, Sliders, Database, Layout,
    Search, ShieldAlert, MonitorCheck
} from "lucide-react";
import { useAuth } from "@/context/AuthContext";
import { LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip as ReTooltip, ResponsiveContainer } from "recharts";

// Lifecycle Modules
import ModelRegistryPage from "./modules/RegistryModule";
import DatasetsPage from "./modules/DatasetsModule";
import ExperimentsPage from "./modules/ExperimentsModule";
import ExplainabilityPage from "./modules/ExplainabilityModule";
import CIModulePage from "./modules/CIModule";
import DeploymentsPage from "./modules/DeploymentsModule";
import DataQualityPage from "./modules/DataQualityModule";
import PerformancePage from "./modules/PerformanceModule";
import ModelSecurityPage from "./modules/ModelSecurityModule";
import ScanHistoryPage from "./modules/ScanHistoryModule";
import ModelReportCardModule from "./modules/ModelReportCardModule";
import ObservabilityModule from "./modules/ObservabilityModule";
import GovernanceModule from "./modules/GovernanceModule";
import NotificationsBell from "./components/NotificationsBell";
import { apiFetch } from "@/lib/api";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "";

// ─────────────── PRIMITIVES ───────────────
const Card = ({ children, className = "" }: any) => (
    <div className={`bg-[#0E1014] border border-white/[0.07] rounded-2xl ${children ? "" : ""} ${className}`}>{children}</div>
);

const CardHeader = ({ title, badge }: any) => {
    const color = badge === "PASSED" || badge === "APPROVED" || badge === "STABLE" || badge === "active" || badge === "enterprise"
        ? "text-emerald-400 border-emerald-500/20 bg-emerald-500/5"
        : badge === "WARNING" || badge === "pro"
            ? "text-amber-400 border-amber-500/20 bg-amber-500/5"
            : badge === "CRITICAL" || badge === "FAILED" || badge === "FRAGILE"
                ? "text-red-400 border-red-500/20 bg-red-500/5"
                : "text-slate-400 border-white/10 bg-white/[0.03]";
    return (
        <div className="flex items-center justify-between mb-5">
            <h4 className="text-[11px] font-black uppercase tracking-[0.15em] text-slate-300">{title}</h4>
            {badge && <span className={`text-[9px] font-black uppercase px-2.5 py-1 rounded-lg border ${color}`}>{badge}</span>}
        </div>
    );
};

const Tile = ({ label, value, sub, accent = false }: any) => (
    <div className="bg-black/20 rounded-xl p-4 space-y-1">
        <p className="text-[9px] uppercase font-black tracking-widest text-slate-600">{label}</p>
        <p className={`text-base font-black truncate ${accent ? "text-orange-400" : "text-white"}`}>{value ?? "—"}</p>
        {sub && <p className="text-[9px] text-slate-600">{sub}</p>}
    </div>
);

const FileUpload = ({ label, accept, file, onFile }: any) => (
    <label className="block cursor-pointer">
        <p className="text-[9px] font-black uppercase tracking-widest text-slate-500 mb-1.5">{label}</p>
        <div className={`flex items-center gap-3 p-4 rounded-xl border transition-all ${file ? "border-emerald-500/30 bg-emerald-500/5" : "border-white/5 bg-black/20 hover:border-orange-500/30"}`}>
            <Upload className={`w-4 h-4 shrink-0 ${file ? "text-emerald-400" : "text-slate-600"}`} />
            <span className={`text-xs font-bold truncate ${file ? "text-emerald-300" : "text-slate-500"}`}>{file ? (typeof file === 'string' ? file : file.name) : "Click to upload"}</span>
        </div>
        <input type="file" accept={accept} className="hidden" onChange={e => e.target.files?.[0] && onFile(e.target.files[0])} />
    </label>
);

const ErrBanner = ({ msg }: any) => msg ? (
    <div className="flex items-start gap-3 p-4 bg-red-500/5 border border-red-500/15 rounded-xl text-red-400 text-xs font-bold">
        <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />{msg}
    </div>
) : null;

const Spinner = ({ label = "Computing..." }: any) => (
    <div className="flex flex-col items-center justify-center py-32 gap-5">
        <div className="w-14 h-14 rounded-full border-2 border-orange-500/20 border-t-orange-500 animate-spin" />
        <p className="text-[10px] uppercase tracking-[0.4em] font-black text-slate-600 animate-pulse">{label}</p>
    </div>
);

const GovScore = ({ score, allowed }: any) => {
    const color = allowed ? "text-emerald-400" : score >= 50 ? "text-amber-400" : "text-red-400";
    const ring = allowed ? "border-emerald-500/20 shadow-emerald-500/5" : score >= 50 ? "border-amber-500/20" : "border-red-500/20";
    return (
        <div className={`p-8 rounded-2xl border text-center shadow-xl ${ring} bg-[#0E1014]`}>
            <p className="text-[9px] uppercase font-black tracking-[0.35em] text-slate-600 mb-1">Governance Score</p>
            <div className={`text-[80px] leading-none font-black ${color}`}>{typeof score === "number" ? Math.round(score) : "—"}</div>
            <p className={`text-[10px] font-black uppercase mt-2 tracking-widest ${color}`}>
                {allowed ? "✓ DEPLOYMENT APPROVED" : "✗ DEPLOYMENT BLOCKED"}
            </p>
        </div>
    );
};

const Advisory = ({ a }: any) => {
    const cls = a.severity === "CRITICAL" ? "border-red-500/20 bg-red-500/5 text-red-400"
        : a.severity === "WARNING" ? "border-amber-500/20 bg-amber-500/5 text-amber-400"
            : "border-white/10 bg-white/[0.02] text-slate-400";
    const Icon = a.severity === "CRITICAL" ? AlertCircle : a.severity === "WARNING" ? AlertTriangle : Info;
    return (
        <div className={`p-4 rounded-xl border ${cls} space-y-1.5`}>
            <div className="flex items-center gap-2">
                <Icon className="w-3.5 h-3.5 shrink-0" />
                <span className="text-[10px] font-black uppercase">[{a.code}] {a.severity}</span>
            </div>
            <p className="text-xs font-medium leading-relaxed opacity-90">{a.message}</p>
            <p className="text-[10px] opacity-70 italic">→ {a.recommendation}</p>
        </div>
    );
};

const AUDIT_CHECKS = [
    { id: "accuracy", label: "Accuracy Evaluation" },
    { id: "f1", label: "F1 Score" },
    { id: "overfitting_check", label: "Overfitting Check" },
    { id: "psi_drift", label: "PSI Drift" },
    { id: "ks_drift", label: "KS Drift" },
    { id: "jsd_drift", label: "Jensen–Shannon Divergence" },
    { id: "target_drift", label: "Target Drift Detection" },
    { id: "calibration_check", label: "Calibration (Brier / ECE)" },
    { id: "leakage_detection", label: "Feature Leakage Detection" },
    { id: "data_quality_check", label: "Data Quality Check" },
    { id: "security_audit", label: "Model Security (Feature 9)" },
    { id: "explainability", label: "Explainability Core (Feature 4)" },
];

const BEHAVIOR_SCENARIOS = [
    { id: "sensitivity_analysis", label: "Sensitivity Analysis", sub: "Finite-difference Δy/Δx per feature" },
    { id: "monte_carlo_stability", label: "Monte Carlo Stability", sub: "100 noisy runs → flip rate & stability score" },
    { id: "ood_boundary_test", label: "OOD Boundary Test", sub: "min−3σ / max+3σ synthetic extremes" },
    { id: "adversarial_permutation", label: "Adversarial Permutation", sub: "Permutation importance — fragile feature detection" },
    { id: "noise_perturbation", label: "Noise Perturbation", sub: "σ = 0.1 × feature_std" },
    { id: "extreme_values", label: "Extreme Values", sub: "Feature min / max as uniform rows" },
    { id: "missing_data_injection", label: "Missing Data Injection", sub: "30% NaN → mean imputation" },
    { id: "boundary_inputs", label: "Boundary Inputs", sub: "5th and 95th percentile rows" },
    { id: "adversarial_shifts", label: "Adversarial Shifts", sub: "+2σ shift across all features" },
];

// ═══════════════════════════════════════════════
// RISK SCORE CIRCULAR INDICATOR
// ═══════════════════════════════════════════════
const RiskScoreIndicator = ({ score, level }: { score: number; level: string }) => {
    const R = 48;
    const circumference = 2 * Math.PI * R;
    const progress = circumference - (score / 100) * circumference;
    const color =
        level === "CRITICAL" ? "#ef4444" :
            level === "HIGH" ? "#f97316" :
                level === "MEDIUM" ? "#eab308" :
                    "#22c55e";
    const textColor =
        level === "CRITICAL" ? "text-red-400" :
            level === "HIGH" ? "text-orange-400" :
                level === "MEDIUM" ? "text-yellow-400" :
                    "text-emerald-400";
    const bgBorder =
        level === "CRITICAL" ? "border-red-500/20 shadow-red-500/5" :
            level === "HIGH" ? "border-orange-500/20 shadow-orange-500/5" :
                level === "MEDIUM" ? "border-yellow-500/20 shadow-yellow-500/5" :
                    "border-emerald-500/20 shadow-emerald-500/5";
    return (
        <div className={`p-6 rounded-2xl border bg-[#0E1014] ${bgBorder} flex items-center gap-8`}>
            <div className="relative shrink-0">
                <svg width="120" height="120" viewBox="0 0 120 120" className="-rotate-90">
                    <circle cx="60" cy="60" r={R} fill="none" stroke="#ffffff08" strokeWidth="8" />
                    <circle
                        cx="60" cy="60" r={R} fill="none" stroke={color} strokeWidth="8"
                        strokeDasharray={circumference}
                        strokeDashoffset={progress}
                        strokeLinecap="round"
                        style={{ transition: "stroke-dashoffset 0.8s ease", filter: `drop-shadow(0 0 6px ${color}60)` }}
                    />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <span className={`text-3xl font-black ${textColor}`}>{score}</span>
                    <span className="text-[8px] font-black text-slate-600 uppercase tracking-widest">Risk</span>
                </div>
            </div>
            <div className="flex-1">
                <p className="text-[9px] uppercase font-black tracking-[0.2em] text-slate-600 mb-1">Model Risk Score</p>
                <p className={`text-2xl font-black ${textColor}`}>{level}</p>
                <p className="text-[10px] text-slate-600 mt-1">Deterministic weighted risk assessment</p>
                <div className="mt-3 flex gap-2 flex-wrap">
                    {["LOW", "MEDIUM", "HIGH", "CRITICAL"].map(l => (
                        <span key={l} className={`text-[8px] font-black px-1.5 py-0.5 rounded border ${l === level
                            ? l === "CRITICAL" ? "bg-red-500/10 text-red-400 border-red-500/30"
                                : l === "HIGH" ? "bg-orange-500/10 text-orange-400 border-orange-500/30"
                                    : l === "MEDIUM" ? "bg-yellow-500/10 text-yellow-400 border-yellow-500/30"
                                        : "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                            : "text-slate-700 border-white/5 bg-transparent"
                            }`}>{l}</span>
                    ))}
                </div>
            </div>
        </div>
    );
};

// ═══════════════════════════════════════════════
// DRIFT FEATURE TABLE
// ═══════════════════════════════════════════════
const DriftFeatureTable = ({ features }: { features: Array<{ feature: string; psi: number; severity: string }> }) => {
    const [sortDir, setSortDir] = React.useState<"desc" | "asc">("desc");
    const sorted = [...features].sort((a, b) => sortDir === "desc" ? b.psi - a.psi : a.psi - b.psi);
    const severityStyle = (s: string) =>
        s === "CRITICAL" ? "bg-red-500/10 text-red-400 border-red-500/30" :
            s === "WARNING" ? "bg-amber-500/10 text-amber-400 border-amber-500/30" :
                "bg-emerald-500/10 text-emerald-400 border-emerald-500/30";
    return (
        <div className="rounded-xl overflow-hidden border border-white/[0.05]">
            <div className="flex items-center justify-between px-4 py-2.5 bg-white/[0.02] border-b border-white/5">
                <p className="text-[9px] font-black uppercase tracking-[0.15em] text-slate-400">Top Drifted Features (PSI Ranked)</p>
                <button onClick={() => setSortDir(d => d === "desc" ? "asc" : "desc")}
                    className="text-[8px] font-black text-slate-600 hover:text-white flex items-center gap-1 transition-colors">
                    PSI {sortDir === "desc" ? "▼" : "▲"}
                </button>
            </div>
            <table className="w-full text-xs">
                <thead>
                    <tr className="border-b border-white/5">
                        <th className="text-left px-4 py-2 text-[9px] font-black uppercase text-slate-600">Feature</th>
                        <th className="text-right px-4 py-2 text-[9px] font-black uppercase text-slate-600">PSI</th>
                        <th className="text-right px-4 py-2 text-[9px] font-black uppercase text-slate-600">Severity</th>
                    </tr>
                </thead>
                <tbody>
                    {sorted.map((row, i) => (
                        <tr key={row.feature} className={`border-b border-white/[0.03] ${i % 2 === 0 ? "bg-black/10" : ""}`}>
                            <td className="px-4 py-2.5 font-mono font-bold text-slate-300">{row.feature}</td>
                            <td className="px-4 py-2.5 text-right font-black"
                                style={{ color: row.psi > 0.25 ? "#f87171" : row.psi > 0.15 ? "#fbbf24" : "#4ade80" }}>
                                {row.psi.toFixed(4)}
                            </td>
                            <td className="px-4 py-2.5 text-right">
                                <span className={`text-[8px] font-black px-2 py-0.5 rounded-lg border ${severityStyle(row.severity)}`}>
                                    {row.severity}
                                </span>
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
};

// ═══════════════════════════════════════════════
// POLICY VIEWER
// ═══════════════════════════════════════════════
const PolicyViewer = ({ policy }: { policy: any }) => {
    if (!policy) return null;
    const displayConfig = policy.config ?? policy.rules ?? {};
    return (
        <div className="rounded-2xl border border-purple-500/20 bg-[#0E1014] overflow-hidden">
            <div className="flex items-center justify-between px-5 py-3 border-b border-purple-500/10 bg-purple-500/[0.04]">
                <div className="flex items-center gap-2">
                    <Shield className="w-3.5 h-3.5 text-purple-400" />
                    <p className="text-[10px] font-black uppercase tracking-[0.15em] text-purple-300">Active Governance Policy</p>
                </div>
                <span className="text-[8px] font-black text-purple-400 border border-purple-500/20 px-2 py-0.5 rounded bg-purple-500/5">
                    {policy.name ?? "Policy"} v{policy.version ?? "default"}
                </span>
            </div>
            <div className="p-4 space-y-3">
                <p className="text-[9px] font-bold uppercase tracking-widest text-slate-600">All Active Thresholds</p>
                <div className="grid grid-cols-2 gap-2">
                    {Object.entries(displayConfig).map(([k, v]: any) => (
                        <div key={k} className="bg-white/[0.02] rounded-lg px-3 py-2 border border-white/5">
                            <p className="text-[9px] text-slate-500 font-mono">{k}</p>
                            <p className="text-xs font-black text-purple-300">{typeof v === "number" ? v : String(v)}</p>
                        </div>
                    ))}
                </div>
                {policy.rules && Object.keys(policy.rules).length > 0 && Object.keys(policy.rules).length < Object.keys(displayConfig).length && (
                    <div className="border-t border-purple-500/10 pt-2 mt-2">
                        <p className="text-[8px] text-slate-600 font-mono">Custom overrides: {JSON.stringify(policy.rules)}</p>
                    </div>
                )}
            </div>
        </div>
    );
};

function Section({ title, badge, children, defaultOpen = true }: any) {
    const [open, setOpen] = useState(defaultOpen);
    return (
        <Card className="overflow-hidden">
            <button className="w-full flex items-center justify-between p-5 text-left" onClick={() => setOpen(!open)}>
                <CardHeader title={title} badge={badge} />
                {open ? <ChevronUp className="w-4 h-4 text-slate-600 shrink-0" /> : <ChevronDown className="w-4 h-4 text-slate-600 shrink-0" />}
            </button>
            {open && <div className="px-5 pb-5">{children}</div>}
        </Card>
    );
}

// ═══════════════════════════════════════════════
// MODULE 1 — MODEL AUDIT
// ═══════════════════════════════════════════════
function ModelAuditPage({ state, setState, onAction }: any) {
    const {
        modelFile, trainFile, valFile, labelCol, checks,
        modelMeta, trainSum, results, loading, error, activePolicy
    } = state;

    const setAuditState = (chunk: any) => setState((prev: any) => {
        const next = { ...prev };
        Object.keys(chunk).forEach(k => { next[k] = typeof chunk[k] === 'function' ? chunk[k](prev[k]) : chunk[k]; });
        return next;
    });

    useEffect(() => {
        if (!activePolicy) {
            apiFetch(`/api/v1/policies/active`)
                .then(r => r.json())
                .then(d => setAuditState({ activePolicy: d }))
                .catch(() => { });
        }
    }, [activePolicy]);

    const onModelUpload = async (f: File) => {
        setAuditState({ modelFile: f, modelMeta: null, error: null });
        const fd = new FormData(); fd.append("model_file", f);
        try { const res = await apiFetch(`/api/v1/audit/inspect-model`, { method: "POST", body: fd }); const d = await res.json(); if (!res.ok) { setAuditState({ error: d.detail }); return; } setAuditState({ modelMeta: d }); } catch (e: any) { setAuditState({ error: e.message }); }
    };

    const [trainSrc, setTrainSrc] = useState<"upload" | "url">("upload");
    const [valSrc, setValSrc] = useState<"upload" | "url">("upload");
    const [trainUrl, setTrainUrl] = useState("");
    const [valUrl, setValUrl] = useState("");

    const onTrainUpload = async (f: File) => {
        setAuditState({ trainFile: f, trainSum: null });
        const fd = new FormData(); fd.append("csv_file", f);
        try { const res = await apiFetch(`/api/v1/audit/dataset-summary`, { method: "POST", body: fd }); const d = await res.json(); if (res.ok) setAuditState({ trainSum: d.dataset_summary }); } catch { }
    };
    const setValFile = (f: File) => setAuditState({ valFile: f });
    const setLabelCol = (v: string) => setAuditState({ labelCol: v });
    const toggle = (id: string) => setState((prev: any) => ({ ...prev, checks: { ...prev.checks, [id]: !prev.checks[id] } }));
    const setError = (v: any) => setAuditState({ error: v });
    const setResults = (v: any) => setAuditState({ results: v });
    const setLoading = (v: boolean) => setAuditState({ loading: v });

    const resetAudit = () => {
        setAuditState({
            modelFile: null, trainFile: null, valFile: null,
            modelMeta: null, trainSum: null, results: null,
            error: null, loading: false
        });
        setTrainUrl(""); setValUrl("");
    };

    const selected = Object.entries(checks).filter(([, v]) => v).map(([k]) => k);
    const runAudit = async () => {
        const missing = [];
        if (!modelFile) missing.push("Model Artifact");
        if (trainSrc === "upload" && !trainFile) missing.push("Training Data");
        if (trainSrc === "url" && !trainUrl) missing.push("Training URL");
        if (valSrc === "upload" && !valFile) missing.push("Validation Data");
        if (valSrc === "url" && !valUrl) missing.push("Validation URL");
        
        if (missing.length > 0) {
            setError(`Please provide: ${missing.join(", ")}`);
            return;
        }
        if (selected.length === 0) { setError("Select at least one governance check."); return; }
        setLoading(true); setError(null); setResults(null);
        
        const fd = new FormData();
        fd.append("model_file", modelFile);
        fd.append("model_name", modelFile.name);
        fd.append("label_col", labelCol);
        selected.forEach(c => fd.append("selected", c));

        if (trainSrc === "upload" && trainFile) fd.append("train_file", trainFile);
        if (valSrc === "upload" && valFile) fd.append("val_file", valFile);
        
        // Note: The standard /audit/run endpoint may need backend updates for URLs,
        // but for now we follow the user's request for UI integration.
        if (trainSrc === "url") fd.append("train_dataset_url", trainUrl);
        if (valSrc === "url") fd.append("val_dataset_url", valUrl);

        try {
            const res = await apiFetch(`/api/v1/audit/run`, { method: "POST", body: fd });
            const d = await res.json();
            if (!res.ok) throw new Error(d.detail || "Audit failed.");

            if (d.status === "pending" && d.job_id) {
                // Poll for results
                let attempts = 0;
                const poll = setInterval(async () => {
                    attempts++;
                    if (attempts > 30) {
                        clearInterval(poll);
                        setLoading(false);
                        setError("Audit timed out in background. Check history later.");
                        return;
                    }
                    try {
                        const jobRes = await apiFetch(`/api/v1/gate/result/${d.submission_token}`);
                        const jobData = await jobRes.json();
                        if (jobData.status === "COMPLETED") {
                            clearInterval(poll);
                            setResults(jobData);
                            setLoading(false);
                            onAction();
                        } else if (jobData.status === "FAILED") {
                            clearInterval(poll);
                            setLoading(false);
                            setError(`Audit failed: ${jobData.error}`);
                        }
                    } catch (e) {
                        console.error("Polling error", e);
                    }
                }, 3000);
            } else {
                setResults(d);
                onAction();
                setLoading(false);
            }
        } catch (e: any) { setError(e.message); setLoading(false); }
    };

    return (
        <div className="grid grid-cols-1 xl:grid-cols-[400px_1fr] gap-8">
            <div className="space-y-4">
                <FileUpload label="1. Model Artifact (.pkl/.joblib/.onnx)" accept=".pkl,.joblib,.onnx" file={modelFile} onFile={onModelUpload} />
                {modelMeta?.model_metadata && (
                    <Card className="p-5 border-orange-500/20">
                        <CardHeader title="Detected Model" />
                        <div className="grid grid-cols-2 gap-3">
                            <Tile label="Class" value={modelMeta.model_metadata.model_class} />
                            <Tile label="Framework" value={modelMeta.model_metadata.framework} />
                            <Tile label="Task" value={modelMeta.model_metadata.task} />
                            <Tile label="Features" value={modelMeta.model_metadata.n_features_in ?? "—"} />
                            {modelMeta.complexity?.proxy_score != null && <Tile label="Complexity" value={modelMeta.complexity.proxy_score} accent />}
                        </div>
                        {modelMeta.fingerprint && <p className="text-[9px] text-slate-600 font-mono mt-3 truncate">SHA256: {modelMeta.fingerprint}</p>}
                    </Card>
                )}
                <div className="space-y-2">
                    <div className="flex items-center justify-between px-1">
                        <p className="text-[9px] font-black uppercase tracking-widest text-slate-500">2. Training Data</p>
                        <div className="flex bg-black p-0.5 rounded-lg border border-white/5">
                            <button onClick={() => setTrainSrc("upload")} className={`px-2 py-0.5 rounded text-[8px] font-black uppercase transition-all ${trainSrc === "upload" ? "bg-orange-600 text-black" : "text-slate-600"}`}>Upload</button>
                            <button onClick={() => setTrainSrc("url")} className={`px-2 py-0.5 rounded text-[8px] font-black uppercase transition-all ${trainSrc === "url" ? "bg-orange-600 text-black" : "text-slate-600"}`}>MinIO</button>
                        </div>
                    </div>
                    {trainSrc === "upload" ? (
                        <FileUpload label="" accept=".csv,.parquet" file={trainFile} onFile={onTrainUpload} />
                    ) : (
                        <input value={trainUrl} onChange={e => setTrainUrl(e.target.value)} placeholder="minio://bucket/train.parquet" className="w-full bg-black/40 border border-white/5 rounded-xl px-4 py-3 text-xs text-white focus:border-orange-500/40 outline-none" />
                    )}
                </div>
                {trainSum && <Card className="p-5"><CardHeader title="Training Data Summary" /><div className="grid grid-cols-3 gap-2"><Tile label="Rows" value={trainSum.n_rows?.toLocaleString()} /><Tile label="Cols" value={trainSum.n_cols} /><Tile label="Missing" value={`${trainSum.missing_pct_global}%`} accent={trainSum.missing_pct_global > 5} /></div></Card>}
                
                <div className="space-y-2">
                    <div className="flex items-center justify-between px-1">
                        <p className="text-[9px] font-black uppercase tracking-widest text-slate-500">3. Validation Data</p>
                        <div className="flex bg-black p-0.5 rounded-lg border border-white/5">
                            <button onClick={() => setValSrc("upload")} className={`px-2 py-0.5 rounded text-[8px] font-black uppercase transition-all ${valSrc === "upload" ? "bg-orange-600 text-black" : "text-slate-600"}`}>Upload</button>
                            <button onClick={() => setValSrc("url")} className={`px-2 py-0.5 rounded text-[8px] font-black uppercase transition-all ${valSrc === "url" ? "bg-orange-600 text-black" : "text-slate-600"}`}>MinIO</button>
                        </div>
                    </div>
                    {valSrc === "upload" ? (
                        <FileUpload label="" accept=".csv,.parquet" file={valFile} onFile={setValFile} />
                    ) : (
                        <input value={valUrl} onChange={e => setValUrl(e.target.value)} placeholder="minio://bucket/val.parquet" className="w-full bg-black/40 border border-white/5 rounded-xl px-4 py-3 text-xs text-white focus:border-orange-500/40 outline-none" />
                    )}
                </div>
                <Card className="p-4 space-y-2"><p className="text-[9px] font-black uppercase tracking-widest text-slate-500">Label Column</p><input value={labelCol} onChange={e => setLabelCol(e.target.value)} className="w-full bg-black/40 border border-white/5 rounded-lg px-3 py-2 text-sm text-white font-bold" /></Card>
                <Card className="p-5 space-y-3"><p className="text-[9px] font-black uppercase tracking-widest text-slate-500 mb-1">4. Select Governance Checks</p>{AUDIT_CHECKS.map(c => (<label key={c.id} className="flex items-center gap-3 cursor-pointer" onClick={() => toggle(c.id)}><div className={`w-4 h-4 rounded border flex items-center justify-center transition-all ${checks[c.id] ? "bg-orange-500 border-orange-500" : "border-white/10"}`}>{checks[c.id] && <CheckCircle2 className="w-2.5 h-2.5 text-black" />}</div><span className={`text-[11px] font-bold ${checks[c.id] ? "text-white" : "text-slate-600"}`}>{c.label}</span></label>))}</Card>
                {/* Feature 3: Policy Viewer */}
                {activePolicy && <PolicyViewer policy={activePolicy} />}
                <ErrBanner msg={error} />
                <div className="flex gap-2">
                    <button onClick={runAudit} disabled={loading} className="flex-1 bg-orange-600 hover:bg-orange-500 disabled:opacity-50 text-black font-black py-4 rounded-xl text-[10px] uppercase tracking-widest flex items-center justify-center gap-2 transition-all shadow-lg shadow-orange-500/10">
                        {loading ? <><Loader2 className="w-4 h-4 animate-spin" />Running...</> : <><ShieldCheck className="w-4 h-4" />Analyze</>}
                    </button>
                    {results && (
                        <button onClick={resetAudit} className="bg-white/5 hover:bg-white/10 text-slate-400 border border-white/5 font-black px-5 rounded-xl text-[10px] uppercase transition-all">New</button>
                    )}
                </div>
            </div>
            <div className="space-y-6 min-h-[400px]">
                {loading && <Spinner label="Running Governance Analysis..." />}
                {results && !loading && (
                    <div className="space-y-5">
                        {/* Governance Score */}
                        <GovScore score={results.governance?.governance_score} allowed={results.governance?.deployment_allowed} />
                        {/* Feature 1: Risk Score Circular Indicator */}
                        {results.risk_score != null && (
                            <RiskScoreIndicator score={results.risk_score} level={results.risk_level ?? "LOW"} />
                        )}
                        {results.governance?.component_scores && <Section title="Governance Score Breakdown"><div className="grid grid-cols-2 gap-3">{Object.entries(results.governance.component_scores).map(([k, v]: any) => <Tile key={k} label={k.replace("_score", "").replace(/_/g, " ")} value={`${v}/100`} accent={v < 70} />)}</div></Section>}
                        {results.policy && <Section title="Policy Gate" badge={results.policy.gate_status}><div className="space-y-2">{(results.policy.checks ?? []).map((c: any, i: number) => <div key={i} className={`flex items-start gap-3 p-3 rounded-xl text-xs font-bold ${c.status === "PASSED" ? "bg-emerald-500/5 text-emerald-400" : c.status === "WARNING" ? "bg-amber-500/5 text-amber-400" : "bg-red-500/5 text-red-400"}`}>{c.status === "PASSED" ? <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5" /> : <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />}<div><p className="font-black">{c.name}: {c.actual_value != null ? (typeof c.actual_value === "number" ? c.actual_value.toFixed(4) : c.actual_value) : "N/A"}</p><p className="opacity-80 font-medium">{c.message}</p></div></div>)}</div>{results.policy.policy_used && <div className="mt-3 border-t border-white/5 pt-3"><p className="text-[9px] font-bold uppercase text-slate-600 mb-2">Thresholds Used</p><div className="grid grid-cols-2 gap-1">{Object.entries(results.policy.policy_used).map(([k, v]: any) => <div key={k} className="flex justify-between text-[10px] px-2 py-1 bg-white/[0.02] rounded"><span className="text-slate-500 font-mono">{k}</span><span className="text-slate-300 font-black">{v}</span></div>)}</div></div>}<div className="mt-3 flex items-center gap-2 px-3 py-2 rounded-lg bg-white/[0.02]"><span className={`text-[10px] font-black ${results.policy.deployment_allowed ? "text-emerald-400" : "text-red-400"}`}>{results.policy.deployment_allowed ? "✓ DEPLOYMENT ALLOWED" : "✗ DEPLOYMENT BLOCKED"}</span>{results.policy.policy_name && <span className="text-[8px] text-slate-600 ml-auto">Policy: {results.policy.policy_name}</span>}</div></Section>}
                        {results.metrics && Object.keys(results.metrics).length > 0 && <Section title="Computed Metrics"><div className="grid grid-cols-2 gap-3">{Object.entries(results.metrics).map(([k, v]: any) => <Tile key={k} label={k.replace(/_/g, " ")} value={typeof v === "number" ? v.toFixed(4) : String(v)} />)}</div></Section>}
                        {results.overfitting_gap && Object.keys(results.overfitting_gap).length > 0 && <Section title="Overfitting Gaps" badge={Object.values(results.overfitting_gap).some((v: any) => Math.abs(v) > 0.08) ? "CRITICAL" : "PASSED"}><div className="grid grid-cols-2 gap-3">{Object.entries(results.overfitting_gap).map(([k, v]: any) => <Tile key={k} label={k.replace(/_/g, " ")} value={v > 0 ? `+${v.toFixed(4)}` : v.toFixed(4)} accent={Math.abs(v) > 0.08} sub={Math.abs(v) > 0.08 ? "⚠ Overfit risk" : "✓ OK"} />)}</div></Section>}
                        {results.drift && Object.keys(results.drift).length > 0 && <Section title={`Feature Drift (Top: ${results.top5_drifted_features?.join(", ") || "none"})`}><div className="space-y-2.5 max-h-[300px] overflow-y-auto pr-1">{Object.entries(results.drift).map(([feat, s]: any) => <div key={feat} className={`flex justify-between items-center py-2 px-3 rounded-lg text-xs ${s.drift_flag ? "bg-red-500/5 border border-red-500/10" : "bg-white/[0.02]"}`}><span className="font-mono font-bold text-slate-300">{feat}</span><div className="flex gap-5 text-[10px] font-black text-slate-500"><span className={s.PSI > 0.25 ? "text-red-400" : s.PSI > 0.1 ? "text-amber-400" : "text-emerald-400"}>PSI {s.PSI?.toFixed(4)}</span><span className={s.JSD > 0.1 ? "text-red-400" : "text-slate-500"}>JSD {s.JSD?.toFixed(4)}</span></div></div>)}</div></Section>}
                        {/* Feature 2: Top Drifted Feature Table */}
                        {results.top_drifted_ranked?.length > 0 && (
                            <Section title={`Top Drifted Features (Ranked)`} badge={results.top_drifted_ranked[0]?.severity}>
                                <DriftFeatureTable features={results.top_drifted_ranked} />
                            </Section>
                        )}
                        {results.target_drift && <Section title="Target Drift" badge={results.target_drift.drifted ? "DRIFTED" : "STABLE"}><div className="grid grid-cols-3 gap-3"><Tile label="Test" value={results.target_drift.test?.toUpperCase()} /><Tile label="Statistic" value={results.target_drift.statistic?.toFixed(4)} /><Tile label="p-value" value={results.target_drift.p_value?.toFixed(4)} accent={results.target_drift.drifted} /></div></Section>}
                        {results.calibration && <Section title="Calibration" badge={results.calibration.overconfident_flag ? "OVERCONFIDENT" : "OK"}><div className="grid grid-cols-3 gap-3"><Tile label="Brier" value={results.calibration.brier_score?.toFixed(4)} accent={results.calibration.brier_score > 0.2} sub="Lower=better" /><Tile label="ECE" value={results.calibration.ece?.toFixed(4)} /><Tile label="Overconfident" value={results.calibration.overconfident_flag ? "YES" : "NO"} accent={results.calibration.overconfident_flag} /></div></Section>}
                        {results.leakage && <Section title="Feature Leakage" badge={results.leakage.risk_level}>{results.leakage.leakage_suspects && Object.keys(results.leakage.leakage_suspects).length > 0 && <ErrBanner msg={`Suspects: ${Object.keys(results.leakage.leakage_suspects).join(", ")}`} />}<div className="space-y-1.5 max-h-[200px] overflow-y-auto mt-2">{Object.entries(results.leakage.mi_scores || {}).slice(0, 10).map(([k, v]: any) => <div key={k} className="flex justify-between text-xs font-bold py-1 border-b border-white/5"><span className="text-slate-400 font-mono">{k}</span><span className={v >= 0.85 ? "text-red-400" : "text-slate-500"}>MI {(v * 100).toFixed(1)}%</span></div>)}</div></Section>}
                        {results.advisories?.length > 0 && <Section title={`Advisories (${results.advisories.length})`}><div className="space-y-3">{results.advisories.map((a: any, i: number) => <Advisory key={i} a={a} />)}</div></Section>}
                        {results.scan_id && <Card className="p-4 text-center"><p className="text-[9px] text-slate-600 font-mono">Scan ID: {results.scan_id}</p></Card>}
                    </div>
                )}
                {!loading && !results && <div className="flex flex-col items-center justify-center h-full min-h-[400px] text-center gap-4"><ShieldCheck className="w-14 h-14 text-slate-800" /><p className="text-sm font-black uppercase text-slate-700 tracking-widest">No Audit Run Yet</p></div>}
            </div>
        </div>
    );
}

// ═══════════════════════════════════════════════
// MODULE 2 — BEHAVIOR TESTING (compact, same as before)
// ═══════════════════════════════════════════════
// ═══════════════════════════════════════════════
// MODULE 2 — BEHAVIOR TESTING
// ═══════════════════════════════════════════════
function BehaviorTestingPage({ state, setState, onAction }: any) {
    const { modelFile, refFile, scenarios, labelCol, results, loading, error } = state;
    const [refSrc, setRefSrc] = useState<"upload" | "url">("upload");
    const [refUrl, setRefUrl] = useState("");

    const setBState = (chunk: any) => setState((prev: any) => {
        const next = { ...prev };
        Object.keys(chunk).forEach(k => { next[k] = typeof chunk[k] === 'function' ? chunk[k](prev[k]) : chunk[k]; });
        return next;
    });

    const setModelFile = (f: File) => setBState({ modelFile: f });
    const setRefFile = (f: File) => setBState({ refFile: f });
    const setLabelCol = (v: string) => setBState({ labelCol: v });
    const setError = (v: any) => setBState({ error: v });
    const setResults = (v: any) => setBState({ results: v });
    const setLoading = (v: boolean) => setBState({ loading: v });
    const toggle = (id: string) => setState((prev: any) => ({ ...prev, scenarios: { ...prev.scenarios, [id]: !prev.scenarios[id] } }));

    const selected = Object.entries(scenarios).filter(([, v]) => v).map(([k]) => k);
    const run = async () => {
        if (!modelFile) { setError("Upload model artifact."); return; }
        if (refSrc === "upload" && !refFile) { setError("Upload reference data."); return; }
        if (refSrc === "url" && !refUrl) { setError("Provide reference data URL."); return; }
        
        if (selected.length === 0) { setError("Select at least one scenario."); return; }
        setLoading(true); setError(null); setResults(null);
        
        const fd = new FormData();
        fd.append("model_file", modelFile);
        if (refSrc === "upload" && refFile) fd.append("ref_file", refFile);
        if (refSrc === "url") fd.append("ref_dataset_url", refUrl);
        
        fd.append("scenarios", selected.join(","));
        fd.append("label_col", labelCol);
        
        try {
            const res = await apiFetch(`/api/v1/behavior/test`, { method: "POST", body: fd });
            const d = await res.json();
            if (!res.ok) throw new Error(d.detail || "Failed.");
            setResults(d);
            onAction();
        } catch (e: any) { setError(e.message); } finally { setLoading(false); }
    };
    return (
        <div className="grid grid-cols-1 xl:grid-cols-[360px_1fr] gap-8">
            <div className="space-y-4">
                <FileUpload label="Model (.pkl/.joblib/.onnx)" accept=".pkl,.joblib,.onnx" file={modelFile} onFile={setModelFile} />
                
                <div className="space-y-2">
                    <div className="flex items-center justify-between px-1">
                        <p className="text-[9px] font-black uppercase tracking-widest text-slate-500">Reference Data</p>
                        <div className="flex bg-black p-0.5 rounded-lg border border-white/5">
                            <button onClick={() => setRefSrc("upload")} className={`px-2 py-0.5 rounded text-[8px] font-black uppercase transition-all ${refSrc === "upload" ? "bg-orange-600 text-black" : "text-slate-600"}`}>Upload</button>
                            <button onClick={() => setRefSrc("url")} className={`px-2 py-0.5 rounded text-[8px] font-black uppercase transition-all ${refSrc === "url" ? "bg-orange-600 text-black" : "text-slate-600"}`}>MinIO</button>
                        </div>
                    </div>
                    {refSrc === "upload" ? (
                        <FileUpload label="" accept=".csv,.parquet" file={refFile} onFile={setRefFile} />
                    ) : (
                        <input value={refUrl} onChange={e => setRefUrl(e.target.value)} placeholder="minio://bucket/ref.parquet" className="w-full bg-black/40 border border-white/5 rounded-xl px-4 py-3 text-xs text-white focus:border-orange-500/40 outline-none" />
                    )}
                </div>
                <Card className="p-4 space-y-2"><p className="text-[9px] font-black uppercase tracking-widest text-slate-500">Label Column</p><input value={labelCol} onChange={e => setLabelCol(e.target.value)} className="w-full bg-black/40 border border-white/5 rounded-lg px-3 py-2 text-sm text-white font-bold" /></Card>
                <Card className="p-5 space-y-3"><p className="text-[9px] font-black uppercase tracking-widest text-slate-500 mb-1">Test Scenarios</p>{BEHAVIOR_SCENARIOS.map(s => (<label key={s.id} className="flex items-start gap-3 cursor-pointer" onClick={() => toggle(s.id)}><div className={`mt-0.5 w-4 h-4 rounded border flex items-center justify-center shrink-0 transition-all ${scenarios[s.id] ? "bg-orange-500 border-orange-500" : "border-white/10"}`}>{scenarios[s.id] && <CheckCircle2 className="w-2.5 h-2.5 text-black" />}</div><div><p className={`text-[11px] font-bold ${scenarios[s.id] ? "text-white" : "text-slate-600"}`}>{s.label}</p><p className="text-[9px] text-slate-700">{s.sub}</p></div></label>))}</Card>
                <ErrBanner msg={error} />
                <button onClick={run} disabled={loading} className="w-full bg-orange-600 hover:bg-orange-500 disabled:opacity-50 text-black font-black py-4 rounded-xl text-[10px] uppercase tracking-widest flex items-center justify-center gap-2">{loading ? <><Loader2 className="w-4 h-4 animate-spin" />Running...</> : <><Activity className="w-4 h-4" />Run Tests</>}</button>
            </div>
            <div className="space-y-5">
                {loading && <Spinner label="Running Behavioral Tests..." />}
                {results && !loading && (<>
                    <Card className={`p-8 text-center border ${results.robustness_score >= 60 ? "border-emerald-500/20" : "border-red-500/20"}`}><p className="text-[9px] uppercase font-black tracking-widest text-slate-600 mb-1">Robustness Score</p><div className={`text-7xl font-black ${results.robustness_score >= 80 ? "text-emerald-400" : results.robustness_score >= 60 ? "text-amber-400" : "text-red-400"}`}>{results.robustness_score}</div></Card>
                    {Object.entries(results.stress_results || {}).map(([id, data]: any) => {
                        if (data?.error) return <Section key={id} title={id} badge="ERROR"><ErrBanner msg={data.error} /></Section>;
                        if (id === "monte_carlo_stability") return <Section key={id} title="Monte Carlo Stability" badge={data.status}><div className="grid grid-cols-2 gap-3"><Tile label="Runs" value={data.n_runs} /><Tile label="Flip Rate" value={`${(data.flip_rate * 100).toFixed(1)}%`} accent={data.flip_rate > 0.1} /><Tile label="Stability" value={data.stability_score?.toFixed(4)} accent={data.stability_score < 0.9} /></div></Section>;
                        if (id === "sensitivity_analysis") return <Section key={id} title="Sensitivity Analysis"><div className="space-y-2">{Object.entries(data.sensitivity_scores || {}).slice(0, 8).map(([k, v]: any) => <div key={k} className="flex justify-between items-center"><span className="text-xs font-mono text-slate-400">{k}</span><div className="flex items-center gap-3"><div className="w-24 h-1.5 rounded-full bg-white/5"><div className="h-full rounded-full bg-orange-500" style={{ width: `${v * 100}%` }} /></div><span className={`text-[10px] font-black ${v >= 0.8 ? "text-red-400" : "text-slate-500"}`}>{(v * 100).toFixed(1)}%</span></div></div>)}</div></Section>;
                        if (id === "ood_boundary_test") return <Section key={id} title="OOD Boundary Test" badge={data.extreme_high?.status}><div className="flex flex-col gap-2"><p className="text-xs text-white">Low Boundary NaNs: {data.extreme_low?.has_nan ? 'Yes' : 'No'}</p><p className="text-xs text-white">High Boundary NaNs: {data.extreme_high?.has_nan ? 'Yes' : 'No'}</p></div></Section>;
                        if (id === "adversarial_permutation") return <Section key={id} title="Adversarial Permutation" badge={data.warning ? "WARNING" : "STABLE"}><div className="space-y-2"><p className="text-xs text-amber-500">{data.warning}</p>{Object.entries(data.permutation_importances || {}).slice(0, 5).map(([k, v]: any) => <div key={k} className="flex justify-between text-xs items-center bg-black/20 p-2 rounded"><span className="text-slate-400 font-mono">{k}</span><span className="text-white font-bold">{(v.fraction_of_total * 100).toFixed(1)}% drop</span></div>)}</div></Section>;
                        return <Section key={id} title={id} badge={data.stability_flag}><div className="grid grid-cols-2 gap-3">{data.output_variance != null && <Tile label="Variance" value={data.output_variance?.toFixed(6)} />}{data.variance_change != null && <Tile label="Var Change" value={data.variance_change?.toFixed(6)} accent={data.variance_change > 0.1} />}</div></Section>;
                    })}
                </>)}
                {!loading && !results && <div className="flex flex-col items-center justify-center h-full min-h-[400px] text-center gap-4"><Activity className="w-14 h-14 text-slate-800" /><p className="text-sm font-black uppercase text-slate-700 tracking-widest">No Tests Run</p></div>}
            </div>
        </div>
    );
}

// ═══════════════════════════════════════════════
// MODULE 3 — LIVE MONITORING (same)
// ═══════════════════════════════════════════════
// ═══════════════════════════════════════════════
// MODULE 3 — LIVE MONITORING
// ═══════════════════════════════════════════════
function LiveMonitoringPage({ state, setState, onAction }: any) {
    const { endpointUrl, probeFile, results, loading, error } = state;
    const setLState = (chunk: any) => setState((prev: any) => {
        const next = { ...prev };
        Object.keys(chunk).forEach(k => { next[k] = typeof chunk[k] === 'function' ? chunk[k](prev[k]) : chunk[k]; });
        return next;
    });

    const setEndpointUrl = (v: string) => setLState({ endpointUrl: v });
    const setProbeFile = (f: File) => setLState({ probeFile: f });
    const setError = (v: any) => setLState({ error: v });
    const setResults = (v: any) => setLState({ results: v });
    const setLoading = (v: boolean) => setLState({ loading: v });

    const run = async () => {
        if (!endpointUrl) { setError("Endpoint URL required."); return; }
        if (!probeFile) { setError("Upload probe CSV."); return; }
        setLoading(true); setError(null); setResults(null);
        try {
            const rawCsv = await probeFile.text(); const lines = rawCsv.split("\n").filter(Boolean); const headers = lines[0].split(","); const rows = lines.slice(1, 11); const latencies: number[] = []; const responses: any[] = [];
            for (const row of rows) { const vals = row.split(","); const payload = Object.fromEntries(headers.map((h: string, i: number) => [h.trim(), vals[i]?.trim()])); const t0 = Date.now(); try { const r = await fetch(endpointUrl, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }); const lat = Date.now() - t0; latencies.push(lat); responses.push({ status: r.status, ok: r.ok, latency_ms: lat }); } catch (ex: any) { responses.push({ status: "ERR", ok: false, latency_ms: Date.now() - t0, error: ex.message }); } }
            const avg = latencies.length ? latencies.reduce((a, b) => a + b) / latencies.length : 0; const sorted = [...latencies].sort((a, b) => a - b); const p95 = sorted[Math.floor(sorted.length * 0.95)] ?? 0; const variance = latencies.length ? latencies.reduce((a, b) => a + (b - avg) ** 2, 0) / latencies.length : 0; const cv = avg > 0 ? Math.sqrt(variance) / avg : 0; const errRate = responses.filter(r => !r.ok).length / responses.length;
            const resData = { probe_count: rows.length, avg_latency_ms: Math.round(avg), p95_latency_ms: Math.round(p95), cv_latency: parseFloat(cv.toFixed(4)), error_rate_pct: Math.round(errRate * 100), status: errRate > 0.1 ? "DEGRADED" : cv > 0.5 ? "UNSTABLE" : "HEALTHY", responses };
            setResults(resData);
            // Log to server for Enterprise Stream
            fetch(`${API_BASE}/api/v1/monitoring/log`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    endpoint_url: endpointUrl,
                    status: resData.status,
                    avg_latency_ms: resData.avg_latency_ms,
                    p95_latency_ms: resData.p95_latency_ms,
                    error_rate_pct: resData.error_rate_pct,
                    probe_count: resData.probe_count
                })
            }).then(() => onAction()).catch(() => { });
        } catch (e: any) { setError(e.message); } finally { setLoading(false); }
    };
    return (
        <div className="grid grid-cols-1 xl:grid-cols-[360px_1fr] gap-8">
            <div className="space-y-4">
                <Card className="p-5 space-y-4"><div className="space-y-2"><p className="text-[9px] font-black uppercase tracking-widest text-slate-500">Inference Endpoint URL</p><input type="url" value={endpointUrl} placeholder="https://api.youmodel.com/predict" onChange={e => setEndpointUrl(e.target.value)} className="w-full bg-black/40 border border-white/5 rounded-lg px-3 py-2 text-sm text-white font-bold" /></div></Card>
                <FileUpload label="Probe Data CSV" accept=".csv" file={probeFile} onFile={setProbeFile} />
                <ErrBanner msg={error} />
                <button onClick={run} disabled={loading} className="w-full bg-orange-600 hover:bg-orange-500 disabled:opacity-50 text-black font-black py-4 rounded-xl text-[10px] uppercase tracking-widest flex items-center justify-center gap-2">{loading ? <><Loader2 className="w-4 h-4 animate-spin" />Probing...</> : <><Wifi className="w-4 h-4" />Probe</>}</button>
            </div>
            <div className="space-y-5">
                {loading && <Spinner label="Probing..." />}
                {results && !loading && <><Card className={`p-8 text-center border ${results.status === "HEALTHY" ? "border-emerald-500/20" : "border-red-500/20"}`}><p className="text-[9px] uppercase font-black tracking-widest text-slate-600 mb-1">Status</p><div className={`text-5xl font-black mb-4 ${results.status === "HEALTHY" ? "text-emerald-400" : "text-red-400"}`}>{results.status}</div><div className="grid grid-cols-4 gap-3"><Tile label="Avg" value={`${results.avg_latency_ms}ms`} /><Tile label="P95" value={`${results.p95_latency_ms}ms`} /><Tile label="CV" value={results.cv_latency} accent={results.cv_latency > 0.5} /><Tile label="Errors" value={`${results.error_rate_pct}%`} accent={results.error_rate_pct > 10} /></div></Card></>}
                {!loading && !results && <div className="flex flex-col items-center justify-center h-full min-h-[400px] text-center gap-4"><Wifi className="w-14 h-14 text-slate-800" /><p className="text-sm font-black uppercase text-slate-700 tracking-widest">No Monitor Active</p></div>}
            </div>
        </div>
    );
}

// ═══════════════════════════════════════════════
// ENTERPRISE INTELLIGENCE STREAM
// ═══════════════════════════════════════════════
const EnterpriseIntelligence = ({ scanHistory, alertEvents, auditLogs }: any) => {
    const activities = [
        ...scanHistory.map((s: any) => ({
            type: "SCAN", time: s.created_at, msg: `Governance Audit: ${s.gate_status}`,
            score: s.governance_score, level: s.risk_level, id: s.id, icon: Activity, color: "blue"
        })),
        ...alertEvents.map((a: any) => ({
            type: "ALERT", time: a.created_at, msg: a.message,
            severity: a.severity, id: a.id, icon: AlertTriangle, color: "red"
        })),
        ...auditLogs.map((l: any) => {
            let icon = Clock; let color = "slate"; let msg = l.action;
            if (l.action === "behavior.test") { icon = Activity; color = "amber"; msg = `Robustness Test: ${l.details?.score || 0}%`; }
            else if (l.action?.startsWith("advisory")) { icon = Eye; color = "purple"; msg = l.details?.question || "AI Advisory Generation"; }
            else if (l.action === "stream.persist") { icon = TrendingUp; color = "emerald"; msg = `Stream Checkpoint (${l.details?.window_size} events)`; }
            else if (l.action === "monitor.probe") { icon = Wifi; color = "cyan"; msg = `Probe: ${l.details?.status || "HEALTHY"}`; }
            else if (l.action === "policy.create") { icon = Shield; color = "orange"; msg = `New Policy: ${l.details?.name}`; }
            else if (l.action === "policy.update") { icon = Shield; color = "purple"; msg = `Policy Updated`; }
            else if (l.action === "fairness.analyze") { icon = Scale; color = "purple"; msg = `Fairness Analysis`; }
            else if (l.action === "llm.evaluate") { icon = Brain; color = "cyan"; msg = `LLM Evaluation`; }
            return { type: "AUDIT", time: l.created_at, msg, id: l.id, icon, color };
        })
    ].sort((a, b) => new Date(b.time).getTime() - new Date(a.time).getTime()).slice(0, 25);

    const colors: any = {
        blue: "border-blue-500/20 bg-blue-500/5 text-blue-400",
        red: "border-red-500/20 bg-red-500/5 text-red-400",
        amber: "border-amber-500/20 bg-amber-500/5 text-amber-400",
        purple: "border-purple-500/20 bg-purple-500/5 text-purple-400",
        emerald: "border-emerald-500/20 bg-emerald-500/5 text-emerald-400",
        cyan: "border-cyan-500/20 bg-cyan-500/5 text-cyan-400",
        orange: "border-orange-500/20 bg-orange-500/5 text-orange-400",
        slate: "border-white/5 bg-white/[0.02] text-slate-400"
    };

    return (
        <Section title="Enterprise Intelligence Stream" badge="REAL-TIME" defaultOpen>
            <div className="space-y-3">
                {activities.length === 0 ? <p className="text-xs text-slate-600 text-center py-8 font-black uppercase tracking-widest">No global activity detected</p> :
                    activities.map((act: any, idx: number) => {
                        const Icon = act.icon;
                        const isScan = act.type === "SCAN";
                        return (
                            <div key={idx} className={`flex items-start gap-4 p-4 rounded-xl border ${colors[act.color]} transition-all hover:bg-white/[0.05]`}>
                                <div className="shrink-0 mt-1"><Icon className="w-4 h-4" /></div>
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center justify-between mb-1">
                                        <span className="text-[10px] font-black uppercase tracking-widest">{act.type}</span>
                                        <span className="text-[9px] opacity-50 font-mono">{act.time?.split('.')[0]}</span>
                                    </div>
                                    <p className="text-xs font-bold truncate pr-2">{act.msg}</p>
                                    {isScan && act.level && (
                                        <div className="mt-2 flex gap-2 overflow-hidden">
                                            <span className="text-[8px] font-black px-1.5 py-0.5 rounded border border-blue-400/20 bg-blue-500/5 uppercase">Score: {Math.round(act.score)}</span>
                                            <span className={`text-[8px] font-black px-1.5 py-0.5 rounded border uppercase ${act.level === 'CRITICAL' ? 'border-red-500/40 text-red-400' : 'border-emerald-500/40 text-emerald-400'}`}>{act.level} Risk</span>
                                        </div>
                                    )}
                                </div>
                                <div className="shrink-0"><span className="text-[8px] font-mono opacity-30">#{act.id?.slice(0, 6)}</span></div>
                            </div>
                        );
                    })
                }
            </div>
            {activities.length > 0 && (
                <div className="mt-4 pt-4 border-t border-white/5 text-center">
                    <p className="text-[9px] font-black uppercase text-slate-600 tracking-[0.2em] animate-pulse">Analyzing live governance events...</p>
                </div>
            )}
        </Section>
    );
};

// ═══════════════════════════════════════════════
// MODULE 4 — ENTERPRISE ADMIN (v7.0 — fully dynamic)
// ═══════════════════════════════════════════════
function EnterprisePage({ state, setState, onAction }: any) {
    const {
        orgs, policies, scanHistory, alertRules, alertEvents,
        auditLogs, models, loading, scanA, scanB, comparison
    } = state;

    const setEState = (chunk: any) => setState((prev: any) => {
        const next = { ...prev };
        Object.keys(chunk).forEach(k => { next[k] = typeof chunk[k] === 'function' ? chunk[k](prev[k]) : chunk[k]; });
        return next;
    });

    // ─── Enterprise Summary (real DB data) ───
    const [summary, setSummary] = useState<any>(null);
    const [scanPage, setScanPage] = useState(1);
    const [scansData, setScansData] = useState<any>(null);
    const [modelsData, setModelsData] = useState<any>(null);
    const [modelsPage, setModelsPage] = useState(1);
    const [logsData, setLogsData] = useState<any>(null);
    const [logsPage, setLogsPage] = useState(1);
    const [policiesData, setPoliciesData] = useState<any[]>([]);
    const [editingPolicy, setEditingPolicy] = useState<string | null>(null);
    const [editConfig, setEditConfig] = useState<string>("");
    const [dbHealth, setDbHealth] = useState<any>(null);
    const [autoRefresh, setAutoRefresh] = useState(true);

    const setScanA = (v: string) => setEState({ scanA: v });
    const setScanB = (v: string) => setEState({ scanB: v });
    const setComparison = (v: any) => setEState({ comparison: v });

    // ─── Fetch all enterprise data from backend ───
    const fetchAll = async (silent = false) => {
        if (!silent) setEState({ loading: true });
        try {
            const [summRes, scansRes, modelsRes, policiesRes, logsRes, dbRes,
                orgsRes, rulesRes, eventsRes] = await Promise.all([
                    apiFetch(`/api/v1/enterprise/summary`).then(r => r.json()),
                    apiFetch(`/api/v1/enterprise/scans?page=${scanPage}&per_page=15`).then(r => r.json()),
                    apiFetch(`/api/v1/enterprise/models?page=${modelsPage}&per_page=15`).then(r => r.json()),
                    apiFetch(`/api/v1/enterprise/policies`).then(r => r.json()),
                    apiFetch(`/api/v1/enterprise/audit-logs?page=${logsPage}&per_page=25`).then(r => r.json()),
                    apiFetch(`/api/v1/health/db`).then(r => r.json()),
                    apiFetch(`/api/v1/orgs`).then(r => r.json()),
                    apiFetch(`/api/v1/alerts/rules`).then(r => r.json()),
                    apiFetch(`/api/v1/alerts/events`).then(r => r.json()),
                ]);
            setSummary(summRes);
            setScansData(scansRes);
            setModelsData(modelsRes);
            setPoliciesData(Array.isArray(policiesRes) ? policiesRes : []);
            setLogsData(logsRes);
            setDbHealth(dbRes);
            const isArr = (v: any) => Array.isArray(v) ? v : [];

            setEState({
                orgs: isArr(orgsRes),
                policies: isArr(policiesRes),
                scanHistory: scansRes?.items || [],
                alertRules: isArr(rulesRes),
                alertEvents: isArr(eventsRes),
                auditLogs: logsRes?.items || [],
                models: modelsRes?.items || [],
                loading: false
            });
        } catch { setEState({ loading: false }); }
    };

    useEffect(() => { fetchAll(); }, [scanPage, modelsPage, logsPage]);

    // Auto-refresh every 15 seconds
    useEffect(() => {
        if (!autoRefresh) return;
        const interval = setInterval(() => fetchAll(true), 15000);
        return () => clearInterval(interval);
    }, [autoRefresh, scanPage, modelsPage, logsPage]);

    const runCompare = async () => {
        if (!scanA || !scanB) return;
        try { const r = await apiFetch(`/api/v1/compare?scan_a=${scanA}&scan_b=${scanB}`); setComparison(await r.json()); } catch { }
    };

    // ─── PATCH policy ───
    const savePolicy = async (policyId: string) => {
        try {
            const parsed = JSON.parse(editConfig);
            const r = await fetch(`${API_BASE}/api/v1/policies/${policyId}`, {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ config: parsed }),
            });
            if (r.ok) {
                setEditingPolicy(null);
                fetchAll(true);
            }
        } catch { }
    };

    if (loading && !summary) return <Spinner label="Loading Enterprise Platform..." />;

    const s = summary || {};
    const totalScans = s.total_scans || 0;
    const gd = s.gate_distribution || {};
    const passedPct = totalScans > 0 ? Math.round((gd.passed / totalScans) * 100) : 0;

    return (
        <div className="space-y-6">
            {/* DB Status + Auto Refresh */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                    {dbHealth && (
                        <span className={`text-[8px] font-black uppercase px-2 py-1 rounded-lg border ${dbHealth.status === "connected" ? "text-emerald-400 border-emerald-500/20 bg-emerald-500/5" : "text-red-400 border-red-500/20 bg-red-500/5"}`}>
                            DB: {dbHealth.db} — {dbHealth.status}
                        </span>
                    )}
                    <span className="text-[8px] text-slate-600 font-black uppercase">v7.0 Enterprise</span>
                </div>
                <div className="flex items-center gap-2">
                    <button onClick={() => fetchAll()} className="text-[8px] font-black uppercase text-slate-500 hover:text-white border border-white/5 px-3 py-1.5 rounded-lg transition-all">
                        ↻ Refresh
                    </button>
                    <button onClick={() => setAutoRefresh(!autoRefresh)}
                        className={`text-[8px] font-black uppercase px-3 py-1.5 rounded-lg border transition-all ${autoRefresh ? "text-emerald-400 border-emerald-500/20 bg-emerald-500/5" : "text-slate-600 border-white/5"}`}>
                        {autoRefresh ? "● Live" : "○ Paused"}
                    </button>
                </div>
            </div>

            {/* ──── Summary Cards (real data from /enterprise/summary) ──── */}
            <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-6 gap-4">
                <Card className="p-5 flex items-center gap-4">
                    <div className="w-10 h-10 rounded-xl bg-orange-500/10 flex items-center justify-center"><BarChart3 className="w-5 h-5 text-orange-400" /></div>
                    <div><p className="text-[9px] uppercase font-black tracking-widest text-slate-600">Total Scans</p><p className="text-2xl font-black text-white">{s.total_scans ?? 0}</p></div>
                </Card>
                <Card className="p-5 flex items-center gap-4">
                    <div className="w-10 h-10 rounded-xl bg-blue-500/10 flex items-center justify-center"><ShieldCheck className="w-5 h-5 text-blue-400" /></div>
                    <div><p className="text-[9px] uppercase font-black tracking-widest text-slate-600">Total Models</p><p className="text-2xl font-black text-white">{s.total_models ?? 0}</p></div>
                </Card>
                <Card className="p-5 flex items-center gap-4">
                    <div className="w-10 h-10 rounded-xl bg-red-500/10 flex items-center justify-center"><AlertTriangle className="w-5 h-5 text-red-400" /></div>
                    <div><p className="text-[9px] uppercase font-black tracking-widest text-slate-600">High Risk</p><p className="text-2xl font-black text-red-400">{s.high_risk_models ?? 0}</p></div>
                </Card>
                <Card className="p-5 flex items-center gap-4">
                    <div className="w-10 h-10 rounded-xl bg-emerald-500/10 flex items-center justify-center"><TrendingUp className="w-5 h-5 text-emerald-400" /></div>
                    <div><p className="text-[9px] uppercase font-black tracking-widest text-slate-600">Avg Score</p><p className={`text-2xl font-black ${(s.average_governance_score || 0) >= 70 ? "text-emerald-400" : "text-red-400"}`}>{s.average_governance_score ?? "—"}</p></div>
                </Card>
                <Card className="p-5 flex items-center gap-4">
                    <div className="w-10 h-10 rounded-xl bg-purple-500/10 flex items-center justify-center"><Shield className="w-5 h-5 text-purple-400" /></div>
                    <div><p className="text-[9px] uppercase font-black tracking-widest text-slate-600">Policies</p><p className="text-2xl font-black text-white">{s.active_policies ?? 0}<span className="text-sm text-slate-600">/{s.total_policies ?? 0}</span></p></div>
                </Card>
                <Card className="p-5 flex items-center gap-4">
                    <div className="w-10 h-10 rounded-xl bg-cyan-500/10 flex items-center justify-center"><Brain className="w-5 h-5 text-cyan-400" /></div>
                    <div><p className="text-[9px] uppercase font-black tracking-widest text-slate-600">LLM Scans</p><p className="text-2xl font-black text-white">{s.total_llm_scans ?? 0}</p></div>
                </Card>
            </div>

            {/* ──── Gate Distribution Bar ──── */}
            {totalScans > 0 && (
                <Card className="p-5">
                    <p className="text-[9px] font-black uppercase tracking-widest text-slate-600 mb-3">Gate Status Distribution</p>
                    <div className="flex h-4 rounded-full overflow-hidden bg-white/5">
                        {gd.passed > 0 && <div className="bg-emerald-500 transition-all" style={{ width: `${(gd.passed / totalScans) * 100}%` }} title={`Passed: ${gd.passed}`} />}
                        {gd.warning > 0 && <div className="bg-amber-500 transition-all" style={{ width: `${(gd.warning / totalScans) * 100}%` }} title={`Warning: ${gd.warning}`} />}
                        {gd.critical > 0 && <div className="bg-red-500 transition-all" style={{ width: `${(gd.critical / totalScans) * 100}%` }} title={`Critical: ${gd.critical}`} />}
                    </div>
                    <div className="flex justify-between mt-2 text-[9px] font-black">
                        <span className="text-emerald-400">✓ Passed: {gd.passed}</span>
                        <span className="text-amber-400">⚠ Warning: {gd.warning}</span>
                        <span className="text-red-400">✗ Critical: {gd.critical}</span>
                        <span className="text-slate-500">{passedPct}% pass rate</span>
                    </div>
                </Card>
            )}

            {/* ──── Organizations ──── */}
            <Section title={`Organizations (${orgs.length})`} defaultOpen={orgs.length <= 5}>
                {orgs.length === 0 ? <p className="text-xs text-slate-600">No organizations yet.</p> :
                    <div className="space-y-2">{orgs.map((o: any) => <div key={o.id} className="flex justify-between items-center p-3 rounded-xl bg-white/[0.02]"><div className="flex items-center gap-3"><Building2 className="w-4 h-4 text-blue-400" /><span className="text-xs font-black text-white">{o.name}</span><span className="text-[9px] font-mono text-slate-600">{o.slug}</span></div><span className="text-[9px] font-black uppercase text-emerald-400 border border-emerald-500/20 px-2 py-0.5 rounded-lg bg-emerald-500/5">{o.plan}</span></div>)}</div>}
            </Section>

            {/* ──── Model Registry (paginated, enriched) ──── */}
            <Section title={`Model Registry (${modelsData?.total ?? models.length})`} defaultOpen={(modelsData?.total ?? models.length) <= 10}>
                {(modelsData?.items || models).length === 0 ? <p className="text-xs text-slate-600">No models registered. Run an audit to register models.</p> :
                    <div className="space-y-2">{(modelsData?.items || models).map((m: any) => <div key={m.id} className="flex justify-between items-center p-3 rounded-xl bg-white/[0.02]"><div><p className="text-xs font-bold text-white">{m.name}</p><p className="text-[9px] font-mono text-slate-600 truncate max-w-xs">{m.fingerprint?.slice(0, 32)}...</p></div><div className="text-right flex items-center gap-3">
                        {m.latest_scan && (
                            <span className={`text-[8px] font-black uppercase px-2 py-0.5 rounded-lg border ${m.latest_scan.gate_status === "PASSED" ? "text-emerald-400 border-emerald-500/20 bg-emerald-500/5" : m.latest_scan.gate_status === "WARNING" ? "text-amber-400 border-amber-500/20 bg-amber-500/5" : "text-red-400 border-red-500/20 bg-red-500/5"}`}>
                                {m.latest_scan.risk_level || m.latest_scan.gate_status} · {m.latest_scan.governance_score != null ? Math.round(m.latest_scan.governance_score) : "—"}
                            </span>
                        )}
                        <div><p className="text-[9px] font-black text-slate-500">v{m.version}</p><p className="text-[9px] text-slate-600">{m.created_at?.split('.')[0]}</p></div>
                    </div></div>)}</div>}
                {modelsData && modelsData.total_pages > 1 && (
                    <div className="flex justify-center gap-2 mt-3">{Array.from({ length: modelsData.total_pages }, (_, i) => (
                        <button key={i} onClick={() => setModelsPage(i + 1)}
                            className={`w-8 h-8 rounded-lg text-[10px] font-black ${modelsPage === i + 1 ? "bg-orange-600 text-black" : "bg-white/5 text-slate-500 hover:bg-white/10"}`}>{i + 1}</button>
                    ))}</div>
                )}
            </Section>

            {/* ──── Governance Policies (with inline edit) ──── */}
            <Section title={`Governance Policies (${policiesData.length})`} defaultOpen>
                {policiesData.length === 0 ? <p className="text-xs text-slate-600">No policies defined.</p> :
                    <div className="space-y-3">{policiesData.map((p: any) => (
                        <Card key={p.id} className="p-4">
                            <div className="flex justify-between items-center mb-2">
                                <div className="flex items-center gap-2">
                                    <p className="text-xs font-bold text-white">{p.name} <span className="text-slate-600">v{p.version}</span></p>
                                    {p.notes && <p className="text-[9px] italic text-slate-600 ml-2">{p.notes}</p>}
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className={`text-[9px] font-black uppercase px-2 py-0.5 rounded-lg border ${p.is_active ? "text-emerald-400 bg-emerald-500/5 border-emerald-500/20" : "text-slate-600 border-white/5"}`}>{p.is_active ? "ACTIVE" : "INACTIVE"}</span>
                                    <button onClick={() => { setEditingPolicy(editingPolicy === p.id ? null : p.id); setEditConfig(JSON.stringify(p.config || {}, null, 2)); }}
                                        className="text-[8px] font-black text-slate-500 hover:text-orange-400 border border-white/5 px-2 py-1 rounded-lg transition-all">
                                        {editingPolicy === p.id ? "Cancel" : "Edit"}
                                    </button>
                                </div>
                            </div>
                            {editingPolicy === p.id ? (
                                <div className="space-y-2 mt-2">
                                    <textarea value={editConfig} onChange={(e: any) => setEditConfig(e.target.value)} rows={8}
                                        className="w-full bg-black/60 border border-white/10 rounded-lg px-3 py-2 text-xs text-white font-mono resize-none" />
                                    <button onClick={() => savePolicy(p.id)}
                                        className="bg-orange-600 hover:bg-orange-500 text-black font-black px-4 py-2 rounded-lg text-[10px] uppercase">Save Changes</button>
                                </div>
                            ) : (
                                <div className="grid grid-cols-3 gap-1.5 mt-1">
                                    {Object.entries(p.config || {}).slice(0, 12).map(([k, v]: any) => (
                                        <div key={k} className="flex justify-between text-[10px] px-2 py-1 bg-white/[0.02] rounded">
                                            <span className="text-slate-500 font-mono">{k}</span>
                                            <span className="text-slate-300 font-black">{typeof v === "number" ? v : String(v)}</span>
                                        </div>
                                    ))}
                                </div>
                            )}
                            <p className="text-[8px] text-slate-700 mt-2">Created: {p.created_at?.split('.')[0]}</p>
                        </Card>
                    ))}</div>}
            </Section>

            {/* ──── Scan History (paginated, sortable) ──── */}
            <Section title={`Scan History (${scansData?.total ?? scanHistory.length})`} defaultOpen>
                {(scansData?.items || scanHistory).length === 0 ? <p className="text-xs text-slate-600">No scans completed yet.</p> :
                    <>
                        <div className="overflow-x-auto">
                            <table className="w-full text-xs">
                                <thead>
                                    <tr className="text-[9px] font-black uppercase tracking-widest text-slate-600 border-b border-white/5">
                                        <th className="text-left py-2 px-2">Scan ID</th>
                                        <th className="text-left py-2 px-2">Model</th>
                                        <th className="text-center py-2 px-2">Score</th>
                                        <th className="text-center py-2 px-2">Risk</th>
                                        <th className="text-center py-2 px-2">Gate</th>
                                        <th className="text-right py-2 px-2">Time</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {(scansData?.items || scanHistory).map((s: any) => {
                                        const gateColor = s.gate_status === "PASSED" ? "text-emerald-400 bg-emerald-500/5 border-emerald-500/20" : s.gate_status === "WARNING" ? "text-amber-400 bg-amber-500/5 border-amber-500/20" : "text-red-400 bg-red-500/5 border-red-500/20";
                                        const riskColor = s.risk_level === "CRITICAL" || s.risk_level === "HIGH" ? "text-red-400" : s.risk_level === "MEDIUM" ? "text-amber-400" : "text-emerald-400";
                                        return (
                                            <tr key={s.id} className="border-b border-white/[0.03] hover:bg-white/[0.02] transition-all">
                                                <td className="py-2.5 px-2 font-mono text-slate-500">{s.id?.slice(0, 8)}...</td>
                                                <td className="py-2.5 px-2 font-bold text-white">{s.model_name || s.scan_type?.toUpperCase() || "—"}</td>
                                                <td className="py-2.5 px-2 text-center"><span className={`font-black ${(s.governance_score || 0) >= 70 ? "text-emerald-400" : "text-red-400"}`}>{s.governance_score != null ? Math.round(s.governance_score) : "—"}</span></td>
                                                <td className={`py-2.5 px-2 text-center font-black ${riskColor}`}>{s.risk_level || "—"}</td>
                                                <td className="py-2.5 px-2 text-center"><span className={`text-[8px] font-black uppercase px-2 py-0.5 rounded-lg border ${gateColor}`}>{s.gate_status}</span></td>
                                                <td className="py-2.5 px-2 text-right text-slate-600 font-mono text-[10px]">{s.created_at?.split('.')[0]}</td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>
                        {scansData && scansData.total_pages > 1 && (
                            <div className="flex justify-center gap-2 mt-3">{Array.from({ length: Math.min(scansData.total_pages, 10) }, (_, i) => (
                                <button key={i} onClick={() => setScanPage(i + 1)}
                                    className={`w-8 h-8 rounded-lg text-[10px] font-black ${scanPage === i + 1 ? "bg-orange-600 text-black" : "bg-white/5 text-slate-500 hover:bg-white/10"}`}>{i + 1}</button>
                            ))}</div>
                        )}
                    </>
                }
            </Section>

            {/* ──── Model Comparison ──── */}
            <Section title="Model Comparison" defaultOpen={false}>
                <div className="flex gap-3 mb-4">
                    <input value={scanA} onChange={e => setScanA(e.target.value)} placeholder="Scan ID A" className="flex-1 bg-black/40 border border-white/5 rounded-lg px-3 py-2 text-xs text-white font-mono" />
                    <span className="text-slate-600 self-center"><ArrowUpDown className="w-4 h-4" /></span>
                    <input value={scanB} onChange={e => setScanB(e.target.value)} placeholder="Scan ID B" className="flex-1 bg-black/40 border border-white/5 rounded-lg px-3 py-2 text-xs text-white font-mono" />
                    <button onClick={runCompare} className="px-4 py-2 bg-orange-600 text-black font-black text-[10px] uppercase rounded-lg">Compare</button>
                </div>
                {comparison && (
                    <div className="space-y-3">
                        <div className="grid grid-cols-3 gap-3">
                            <Tile label="Score A" value={comparison.scan_a?.score} />
                            <Tile label="Score B" value={comparison.scan_b?.score} />
                            <Tile label="Delta" value={comparison.governance_delta != null ? `${comparison.governance_delta > 0 ? "+" : ""}${comparison.governance_delta.toFixed(2)}` : "—"} accent={comparison.governance_delta != null && comparison.governance_delta < 0} sub={comparison.governance_delta > 0 ? "↑ Improved" : "↓ Degraded"} />
                        </div>
                        {comparison.metrics_comparison && Object.keys(comparison.metrics_comparison).length > 0 && <div className="space-y-1.5">{Object.entries(comparison.metrics_comparison).map(([k, v]: any) => <div key={k} className="flex justify-between text-xs font-bold py-1 border-b border-white/5"><span className="text-slate-400">{k}</span><div className="flex gap-4"><span className="text-slate-500">A: {v.scan_a?.toFixed(4)}</span><span className="text-slate-500">B: {v.scan_b?.toFixed(4)}</span><span className={v.delta > 0 ? "text-emerald-400" : v.delta < 0 ? "text-red-400" : "text-slate-600"}>{v.delta != null ? `Δ ${v.delta > 0 ? "+" : ""}${v.delta.toFixed(4)}` : "—"}</span></div></div>)}</div>}
                    </div>
                )}
            </Section>

            {/* ──── Alert Rules ──── */}
            <Section title={`Alert Rules (${alertRules.length})`} defaultOpen={false}>
                {alertRules.length === 0 ? <p className="text-xs text-slate-600">No alert rules defined.</p> :
                    <div className="space-y-2">{alertRules.map((r: any) => <div key={r.id} className="flex justify-between items-center p-3 rounded-xl bg-white/[0.02]"><div><p className="text-xs font-bold text-white">{r.name}</p><p className="text-[9px] text-slate-600 font-mono">{JSON.stringify(r.condition)}</p></div><div className="flex gap-2">{r.channels?.map((c: string) => <span key={c} className="text-[8px] font-black uppercase text-slate-500 border border-white/5 px-2 py-0.5 rounded">{c}</span>)}</div></div>)}</div>}
            </Section>

            {/* ──── Audit Trail (paginated) ──── */}
            <Section title={`Audit Trail (${logsData?.total ?? auditLogs.length})`} defaultOpen={false}>
                {(logsData?.items || auditLogs).length === 0 ? <p className="text-xs text-slate-600">No actions recorded.</p> :
                    <div className="space-y-1.5 max-h-[400px] overflow-y-auto">{(logsData?.items || auditLogs).map((l: any) => <div key={l.id} className="flex justify-between py-2 px-3 rounded-lg bg-white/[0.01] text-xs"><div className="flex items-center gap-2"><Clock className="w-3 h-3 text-slate-700" /><span className="font-bold text-slate-400">{l.action}</span>{l.resource_type && <span className="text-[8px] text-slate-600 border border-white/5 px-1.5 py-0.5 rounded">{l.resource_type}</span>}</div><span className="text-[9px] text-slate-600">{l.created_at?.split('.')[0]}</span></div>)}</div>}
                {logsData && logsData.total_pages > 1 && (
                    <div className="flex justify-center gap-2 mt-3">{Array.from({ length: Math.min(logsData.total_pages, 10) }, (_, i) => (
                        <button key={i} onClick={() => setLogsPage(i + 1)}
                            className={`w-8 h-8 rounded-lg text-[10px] font-black ${logsPage === i + 1 ? "bg-orange-600 text-black" : "bg-white/5 text-slate-500 hover:bg-white/10"}`}>{i + 1}</button>
                    ))}</div>
                )}
            </Section>

            {/* ──── Enterprise Intelligence Stream ──── */}
            <EnterpriseIntelligence scanHistory={scansData?.items || scanHistory} alertEvents={alertEvents} auditLogs={logsData?.items || auditLogs} />
        </div>
    );
}

// ═══════════════════════════════════════════════
// MODULE 5 — STREAMING DRIFT MONITOR
// ═══════════════════════════════════════════════
// ═══════════════════════════════════════════════
// MODULE 5 — STREAMING DRIFT MONITOR
// ═══════════════════════════════════════════════
function StreamingMonitorPage({ state, setState, onAction }: any) {
    const { modelId, metrics, alerts, eventCount, wsStatus, streamModels } = state;
    const [ws, setWs] = useState<WebSocket | null>(null);
    const setSState = (chunk: any) => setState((prev: any) => {
        const next = { ...prev };
        Object.keys(chunk).forEach(k => { next[k] = typeof chunk[k] === 'function' ? chunk[k](prev[k]) : chunk[k]; });
        return next;
    });

    const setModelId = (v: string) => setSState({ modelId: v });
    const setMetrics = (v: any) => setSState({ metrics: v });
    const setAlerts = (v: any) => setSState({ alerts: v });
    const setEventCount = (v: any) => setSState({ eventCount: v });
    const setWsStatus = (v: any) => setSState({ wsStatus: v });
    const setStreamModels = (v: any) => setSState({ streamModels: v });

    useEffect(() => {
        apiFetch(`/api/v1/stream/models`).then(r => r.json()).then(d => setStreamModels(d.streaming_models || [])).catch(() => { });
    }, [eventCount]);

    const connect = () => {
        setWsStatus("disconnected");

        // Robust construction of the WebSocket URL
        // Strip trailing slash first, then normalize protocol
        const base = (API_BASE || "http://127.0.0.1:8090").replace(/\/$/, "");
        let wsUrl = base.replace(/^http/, "ws");

        // Ensure /api/v1 is present exactly once
        if (!wsUrl.includes("/api/v1")) {
            wsUrl += "/api/v1";
        }

        // Construct finally - avoiding double slashes
        const fullWsUrl = `${wsUrl.replace(/\/$/, "")}/ws/stream?model_id=${modelId || "default"}`;

        console.log("🛠️ Initializing ML Guard WebSocket System...");
        console.log("📍 API Base:", API_BASE);
        console.log("🔗 WebSocket Target:", fullWsUrl);

        const socket = new WebSocket(fullWsUrl);
        socket.onopen = () => {
            console.log("WebSocket connected successfully");
            setWsStatus("connected");
            setWs(socket);
        };
        socket.onclose = (event) => {
            console.log("WebSocket closed:", event.code, event.reason);
            // Only set to disconnected if we weren't already in an error state
            setWsStatus((prev: string) => prev === "error" ? "error" : "disconnected");
            setWs(null);
        };
        socket.onerror = (err) => {
            console.error("WebSocket error:", err);
            setWsStatus("error");
        };
        socket.onmessage = (ev) => {
            try {
                const data = JSON.parse(ev.data);
                if (data.type === "metrics") { setMetrics(data); }
                if (data.type === "alert") { setAlerts((prev: any[]) => [data, ...prev].slice(0, 20)); }
                if (data.type === "ack") { setEventCount(data.window_size); }
            } catch { }
        };
    };

    const disconnect = () => { ws?.close(); setWs(null); setWsStatus("disconnected"); };

    const sendTestEvent = () => {
        if (!ws) return;
        const prediction = 0.5 + (Math.random() - 0.5) * 0.6;
        ws.send(JSON.stringify({ prediction, confidence: Math.random(), actual: Math.round(Math.random()), features: {} }));
    };

    const sendBaseline = () => {
        if (!ws) return;
        const preds = Array.from({ length: 200 }, () => 0.5 + (Math.random() - 0.5) * 0.3);
        ws.send(JSON.stringify({ type: "baseline", predictions: preds }));
    };

    const pollStatus = async () => {
        try { const r = await apiFetch(`/api/v1/stream/status/${modelId}`); setMetrics(await r.json()); } catch { }
    };

    const statusColor = wsStatus === "connected" ? "text-emerald-400" : wsStatus === "error" ? "text-red-400" : "text-slate-600";

    return (
        <div className="grid grid-cols-1 xl:grid-cols-[360px_1fr] gap-8">
            <div className="space-y-4">
                <Card className="p-5 space-y-3">
                    <p className="text-[9px] font-black uppercase tracking-widest text-slate-500">Model ID</p>
                    <input value={modelId} onChange={e => setModelId(e.target.value)} className="w-full bg-black/40 border border-white/5 rounded-lg px-3 py-2 text-sm text-white font-bold" />
                    <div className="flex items-center gap-2">
                        <div className={`w-2 h-2 rounded-full ${wsStatus === "connected" ? "bg-emerald-400 animate-pulse" : "bg-slate-700"}`} />
                        <span className={`text-[10px] font-black uppercase ${statusColor}`}>{wsStatus}</span>
                        {eventCount > 0 && <span className="text-[9px] text-slate-600 ml-auto">{eventCount} events</span>}
                    </div>
                </Card>
                <div className="grid grid-cols-2 gap-2">
                    {wsStatus !== "connected" ? (
                        <button onClick={connect} className="col-span-2 bg-orange-600 hover:bg-orange-500 text-black font-black py-3 rounded-xl text-[10px] uppercase tracking-widest">Connect WebSocket</button>
                    ) : (
                        <>
                            <button onClick={disconnect} className="bg-red-600/20 text-red-400 font-black py-3 rounded-xl text-[10px] uppercase">Disconnect</button>
                            <button onClick={sendBaseline} className="bg-blue-600/20 text-blue-400 font-black py-3 rounded-xl text-[10px] uppercase">Set Baseline</button>
                            <button onClick={sendTestEvent} className="col-span-2 bg-emerald-600/20 text-emerald-400 font-black py-3 rounded-xl text-[10px] uppercase">Send Test Event</button>
                        </>
                    )}
                </div>
                <button onClick={pollStatus} className="w-full bg-white/5 text-slate-400 font-black py-3 rounded-xl text-[10px] uppercase tracking-widest hover:bg-white/10">Poll HTTP Status</button>

                {streamModels.length > 0 && (
                    <Section title="Active Streams" defaultOpen={true}>
                        <div className="space-y-1.5">{streamModels.map((m: any, i: number) => (
                            <div key={i} className="flex justify-between text-xs py-1">
                                <span className="font-mono text-slate-400">{m.model_id}</span>
                                <span className="text-slate-600">{m.window_size} events</span>
                            </div>
                        ))}</div>
                    </Section>
                )}
            </div>
            <div className="space-y-5">
                {metrics ? (
                    <>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <Card className="p-5 text-center"><p className="text-[9px] uppercase font-black tracking-widest text-slate-600">Window</p><p className="text-2xl font-black text-white">{metrics.window_size ?? 0}</p></Card>
                            <Card className={`p-5 text-center border ${(metrics.rolling_psi ?? 0) > 0.25 ? "border-red-500/20" : "border-white/[0.07]"}`}><p className="text-[9px] uppercase font-black tracking-widest text-slate-600">Rolling PSI</p><p className={`text-2xl font-black ${(metrics.rolling_psi ?? 0) > 0.25 ? "text-red-400" : "text-emerald-400"}`}>{metrics.rolling_psi?.toFixed(4) ?? "—"}</p></Card>
                            <Card className={`p-5 text-center border ${(metrics.rolling_jsd ?? 0) > 0.15 ? "border-amber-500/20" : "border-white/[0.07]"}`}><p className="text-[9px] uppercase font-black tracking-widest text-slate-600">Rolling JSD</p><p className={`text-2xl font-black ${(metrics.rolling_jsd ?? 0) > 0.15 ? "text-amber-400" : "text-emerald-400"}`}>{metrics.rolling_jsd?.toFixed(4) ?? "—"}</p></Card>
                            <Card className="p-5 text-center"><p className="text-[9px] uppercase font-black tracking-widest text-slate-600">Confidence</p><p className="text-2xl font-black text-white">{metrics.mean_confidence?.toFixed(3) ?? "—"}</p></Card>
                        </div>
                        {metrics.rolling_stability && (
                            <Section title="Rolling Stability">
                                <div className="grid grid-cols-3 gap-3">
                                    <Tile label="Variance" value={metrics.rolling_stability.variance?.toFixed(6)} />
                                    <Tile label="CV" value={metrics.rolling_stability.cv?.toFixed(4)} accent={metrics.rolling_stability.cv > 0.5} />
                                    <Tile label="Stability" value={metrics.rolling_stability.stability_score?.toFixed(4)} accent={metrics.rolling_stability.stability_score < 0.8} />
                                </div>
                            </Section>
                        )}
                        {metrics.rolling_calibration && (
                            <Section title="Rolling Calibration">
                                <div className="grid grid-cols-2 gap-3">
                                    <Tile label="Brier Score" value={metrics.rolling_calibration.brier_score?.toFixed(4)} accent={metrics.rolling_calibration.brier_score > 0.2} />
                                    <Tile label="Labeled Samples" value={metrics.rolling_calibration.n_labeled} />
                                </div>
                            </Section>
                        )}
                    </>
                ) : (
                    <div className="flex flex-col items-center justify-center h-full min-h-[400px] text-center gap-4">
                        <BarChart3 className="w-14 h-14 text-slate-800" />
                        <p className="text-sm font-black uppercase text-slate-700 tracking-widest">No Streaming Data</p>
                        <p className="text-xs text-slate-600">Connect via WebSocket or poll HTTP status</p>
                    </div>
                )}
                {alerts.length > 0 && (
                    <Section title={`Active Alerts (${alerts.length})`}>
                        <div className="space-y-2">{alerts.map((a: any, i: number) => (
                            <div key={i} className={`flex items-start gap-3 p-3 rounded-xl text-xs font-bold ${a.severity === "CRITICAL" ? "bg-red-500/5 text-red-400 border border-red-500/10" : "bg-amber-500/5 text-amber-400 border border-amber-500/10"}`}>
                                <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                                <div><p className="font-black">{a.metric}</p><p className="opacity-80">Value: {a.value} (threshold: {a.threshold})</p></div>
                            </div>
                        ))}</div>
                    </Section>
                )}
            </div>
        </div>
    );
}

// ═══════════════════════════════════════════════
// MODULE 6 — AI ADVISORY COPILOT
// ═══════════════════════════════════════════════
function AIAdvisoryPage({ state, setState, onAction }: any) {
    const { scanId, question, advisory, loading, error, useLLM } = state;
    const setAState = (chunk: any) => setState((prev: any) => {
        const next = { ...prev };
        Object.keys(chunk).forEach(k => { next[k] = typeof chunk[k] === 'function' ? chunk[k](prev[k]) : chunk[k]; });
        return next;
    });

    const setScanId = (v: string) => setAState({ scanId: v });
    const setQuestion = (v: string) => setAState({ question: v });
    const setAdvisory = (v: any) => setAState({ advisory: v });
    const setError = (v: any) => setAState({ error: v });
    const setLoading = (v: boolean) => setAState({ loading: v });
    const setUseLLM = (v: boolean) => setAState({ useLLM: v });

    const run = async () => {
        setLoading(true); setError(null); setAdvisory(null);
        const endpoint = useLLM ? "advisory/explain-with-llm" : "advisory/explain";
        const body: any = { question };
        if (scanId) body.scan_id = scanId;
        try {
            const r = await apiFetch(`/api/v1/${endpoint}`, {
                method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
            });
            const d = await r.json();
            if (!r.ok) throw new Error(d.detail || "Advisory failed.");
            setAdvisory(d);
            onAction();
        } catch (e: any) { setError(e.message); }
        finally { setLoading(false); }
    };

    return (
        <div className="grid grid-cols-1 xl:grid-cols-[360px_1fr] gap-8">
            <div className="space-y-4">
                <Card className="p-5 space-y-3">
                    <p className="text-[9px] font-black uppercase tracking-widest text-slate-500">Scan ID (from audit)</p>
                    <input value={scanId} onChange={e => setScanId(e.target.value)} placeholder="Paste scan ID..." className="w-full bg-black/40 border border-white/5 rounded-lg px-3 py-2 text-sm text-white font-mono" />
                </Card>
                <Card className="p-5 space-y-3">
                    <p className="text-[9px] font-black uppercase tracking-widest text-slate-500">Your Question</p>
                    <textarea value={question} onChange={e => setQuestion(e.target.value)} rows={3} className="w-full bg-black/40 border border-white/5 rounded-lg px-3 py-2 text-sm text-white resize-none" />
                </Card>
                <Card className="p-4">
                    <label className="flex items-center gap-3 cursor-pointer" onClick={() => setUseLLM(!useLLM)}>
                        <div className={`w-4 h-4 rounded border flex items-center justify-center transition-all ${useLLM ? "bg-purple-500 border-purple-500" : "border-white/10"}`}>
                            {useLLM && <CheckCircle2 className="w-2.5 h-2.5 text-black" />}
                        </div>
                        <div>
                            <p className={`text-[11px] font-bold ${useLLM ? "text-purple-400" : "text-slate-600"}`}>Use AI Advisor (Groq)</p>
                            <p className="text-[9px] text-slate-700">Requires GROQ_API_KEY on backend. Falls back to structured analysis.</p>
                        </div>
                    </label>
                </Card>
                <ErrBanner msg={error} />
                <button onClick={run} disabled={loading || !scanId} className="w-full bg-purple-600 hover:bg-purple-500 disabled:opacity-50 text-white font-black py-4 rounded-xl text-[10px] uppercase tracking-widest flex items-center justify-center gap-2 transition-all shadow-lg shadow-purple-500/10">
                    {loading ? <><Loader2 className="w-4 h-4 animate-spin" />Analyzing...</> : <><Eye className="w-4 h-4" />Get Advisory</>}
                </button>
            </div>
            <div className="space-y-5">
                {loading && <Spinner label="Generating Advisory..." />}
                {advisory && !loading && (
                    <div className="space-y-4">
                        <Card className="p-5 flex items-center gap-4">
                            <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${advisory.advisory_type === "llm" ? "bg-purple-500/10" : "bg-blue-500/10"}`}>
                                <Eye className={`w-5 h-5 ${advisory.advisory_type === "llm" ? "text-purple-400" : "text-blue-400"}`} />
                            </div>
                            <div>
                                <p className="text-xs font-black text-white">{advisory.advisory_type === "llm" ? "LLM Advisory" : "Structured Analysis"}</p>
                                <p className="text-[9px] text-slate-600">{advisory.disclaimer}</p>
                            </div>
                            {advisory.governance_score != null && (
                                <div className="ml-auto text-right">
                                    <p className={`text-2xl font-black ${advisory.governance_score >= 70 ? "text-emerald-400" : advisory.governance_score >= 50 ? "text-amber-400" : "text-red-400"}`}>{Math.round(advisory.governance_score)}</p>
                                    <p className="text-[8px] font-black text-slate-600">SCORE</p>
                                </div>
                            )}
                        </Card>
                        <Card className="p-6">
                            <div className="prose prose-invert prose-sm max-w-none text-slate-300 leading-relaxed" style={{ whiteSpace: "pre-wrap" }}>
                                {advisory.explanation}
                            </div>
                        </Card>
                        {advisory.fallback_reason && (
                            <Card className="p-4 border-amber-500/20">
                                <p className="text-[10px] font-bold text-amber-400">⚠ LLM Fallback: {advisory.fallback_reason}</p>
                            </Card>
                        )}
                    </div>
                )}
                {!loading && !advisory && (
                    <div className="flex flex-col items-center justify-center h-full min-h-[400px] text-center gap-4">
                        <Eye className="w-14 h-14 text-slate-800" />
                        <p className="text-sm font-black uppercase text-slate-700 tracking-widest">AI Advisory</p>
                        <p className="text-xs text-slate-600 max-w-sm">Enter a scan ID from a completed audit and ask a question about your governance results.</p>
                    </div>
                )}
            </div>
        </div>
    );
}

// ═══════════════════════════════════════════════
// MODULE 7 — FAIRNESS ANALYSIS
// ═══════════════════════════════════════════════
function FairnessAnalysisPage({ state, setState, onAction }: any) {
    const { modelFile, dataFile, sensitiveCol, labelCol, results, loading, error } = state;
    const [dataSrc, setDataSrc] = useState<"upload" | "url">("upload");
    const [dataUrl, setDataUrl] = useState("");

    const setFState = (chunk: any) => setState((prev: any) => {
        const next = { ...prev };
        Object.keys(chunk).forEach(k => { next[k] = typeof chunk[k] === 'function' ? chunk[k](prev[k]) : chunk[k]; });
        return next;
    });

    const run = async () => {
        if (!modelFile) { setFState({ error: "Upload model artifact." }); return; }
        if (dataSrc === "upload" && !dataFile) { setFState({ error: "Upload evaluation data." }); return; }
        if (dataSrc === "url" && !dataUrl) { setFState({ error: "Provide evaluation data URL." }); return; }
        
        if (!sensitiveCol.trim()) { setFState({ error: "Specify the sensitive feature column." }); return; }
        setFState({ loading: true, error: null, results: null });
        const fd = new FormData();
        fd.append("model_file", modelFile);
        if (dataSrc === "upload" && dataFile) fd.append("data_file", dataFile);
        if (dataSrc === "url") fd.append("dataset_url", dataUrl);
        
        fd.append("sensitive_column", sensitiveCol); fd.append("label_col", labelCol);
        try {
            const r = await apiFetch(`/api/v1/fairness/analyze`, { method: "POST", body: fd });
            const d = await r.json();
            if (!r.ok) throw new Error(d.detail || "Fairness analysis failed.");
            setFState({ results: d });
            onAction();
        } catch (e: any) { setFState({ error: e.message }); }
        finally { setFState({ loading: false }); }
    };

    const fairness = results?.fairness;
    const policy = results?.policy;

    return (
        <div className="grid grid-cols-1 xl:grid-cols-[400px_1fr] gap-8">
            <div className="space-y-4">
                <FileUpload label="1. Model Artifact (.pkl/.joblib/.onnx)" accept=".pkl,.joblib,.onnx" file={modelFile}
                    onFile={(f: File) => setFState({ modelFile: f })} />
                
                <div className="space-y-2">
                    <div className="flex items-center justify-between px-1">
                        <p className="text-[9px] font-black uppercase tracking-widest text-slate-500">2. Evaluation Data</p>
                        <div className="flex bg-black p-0.5 rounded-lg border border-white/5">
                            <button onClick={() => setDataSrc("upload")} className={`px-2 py-0.5 rounded text-[8px] font-black uppercase transition-all ${dataSrc === "upload" ? "bg-orange-600 text-black" : "text-slate-600"}`}>Upload</button>
                            <button onClick={() => setDataSrc("url")} className={`px-2 py-0.5 rounded text-[8px] font-black uppercase transition-all ${dataSrc === "url" ? "bg-orange-600 text-black" : "text-slate-600"}`}>MinIO</button>
                        </div>
                    </div>
                    {dataSrc === "upload" ? (
                        <FileUpload label="" accept=".csv,.parquet" file={dataFile} onFile={(f: File) => setFState({ dataFile: f })} />
                    ) : (
                        <input value={dataUrl} onChange={e => setDataUrl(e.target.value)} placeholder="minio://bucket/eval.parquet" className="w-full bg-black/40 border border-white/5 rounded-xl px-4 py-3 text-xs text-white focus:border-orange-500/40 outline-none" />
                    )}
                </div>
                <Card className="p-4 space-y-2">
                    <p className="text-[9px] font-black uppercase tracking-widest text-slate-500">Sensitive Feature Column</p>
                    <input value={sensitiveCol} onChange={(e: any) => setFState({ sensitiveCol: e.target.value })}
                        placeholder="e.g., gender, race, age_group"
                        className="w-full bg-black/40 border border-white/5 rounded-lg px-3 py-2 text-sm text-white font-bold" />
                </Card>
                <Card className="p-4 space-y-2">
                    <p className="text-[9px] font-black uppercase tracking-widest text-slate-500">Label Column</p>
                    <input value={labelCol} onChange={(e: any) => setFState({ labelCol: e.target.value })}
                        className="w-full bg-black/40 border border-white/5 rounded-lg px-3 py-2 text-sm text-white font-bold" />
                </Card>
                <ErrBanner msg={error} />
                <button onClick={run} disabled={loading}
                    className="w-full bg-purple-600 hover:bg-purple-500 disabled:opacity-50 text-white font-black py-4 rounded-xl text-[10px] uppercase tracking-widest flex items-center justify-center gap-2 transition-all shadow-lg shadow-purple-500/10">
                    {loading ? <><Loader2 className="w-4 h-4 animate-spin" />Analyzing...</> : <><Scale className="w-4 h-4" />Run Fairness Analysis</>}
                </button>
            </div>
            <div className="space-y-6 min-h-[400px]">
                {loading && <Spinner label="Computing Fairness Metrics..." />}
                {fairness && !loading && (
                    <div className="space-y-5">
                        {/* Hero Metrics */}
                        <div className="grid grid-cols-3 gap-4">
                            <Card className={`p-6 text-center border ${fairness.violations?.spd_violated ? "border-red-500/20" : "border-emerald-500/20"}`}>
                                <p className="text-[9px] uppercase font-black tracking-widest text-slate-600">Statistical Parity Diff</p>
                                <p className={`text-3xl font-black ${fairness.violations?.spd_violated ? "text-red-400" : "text-emerald-400"}`}>
                                    {fairness.statistical_parity_diff?.toFixed(4)}
                                </p>
                                <p className="text-[8px] text-slate-600 mt-1">Threshold: ≤ {fairness.thresholds_used?.max_spd}</p>
                            </Card>
                            <Card className={`p-6 text-center border ${fairness.violations?.dir_violated ? "border-red-500/20" : "border-emerald-500/20"}`}>
                                <p className="text-[9px] uppercase font-black tracking-widest text-slate-600">Disparate Impact Ratio</p>
                                <p className={`text-3xl font-black ${fairness.violations?.dir_violated ? "text-red-400" : "text-emerald-400"}`}>
                                    {fairness.disparate_impact_ratio?.toFixed(4)}
                                </p>
                                <p className="text-[8px] text-slate-600 mt-1">80% Rule: ≥ {fairness.thresholds_used?.min_dir}</p>
                            </Card>
                            <Card className={`p-6 text-center border ${fairness.violations?.eod_violated ? "border-red-500/20" : "border-emerald-500/20"}`}>
                                <p className="text-[9px] uppercase font-black tracking-widest text-slate-600">Equal Opportunity Diff</p>
                                <p className={`text-3xl font-black ${fairness.violations?.eod_violated ? "text-red-400" : "text-emerald-400"}`}>
                                    {fairness.equal_opportunity_diff?.toFixed(4)}
                                </p>
                                <p className="text-[8px] text-slate-600 mt-1">Threshold: ≤ {fairness.thresholds_used?.max_eod}</p>
                            </Card>
                        </div>

                        {/* Fairness Badge */}
                        <Card className={`p-5 flex items-center justify-between ${fairness.fairness_flag ? "border-emerald-500/20" : "border-red-500/20"}`}>
                            <div className="flex items-center gap-3">
                                {fairness.fairness_flag ? <CheckCircle2 className="w-6 h-6 text-emerald-400" /> : <AlertCircle className="w-6 h-6 text-red-400" />}
                                <div>
                                    <p className={`text-sm font-black ${fairness.fairness_flag ? "text-emerald-400" : "text-red-400"}`}>
                                        {fairness.fairness_flag ? "✓ FAIRNESS COMPLIANCE PASSED" : "✗ BIAS DETECTED — FAIRNESS VIOLATION"}
                                    </p>
                                    <p className="text-[9px] text-slate-600">Subscore: {(fairness.fairness_subscore * 100).toFixed(1)}/100 — contributes 15% to Governance Score</p>
                                </div>
                            </div>
                            <span className={`text-[9px] font-black uppercase px-3 py-1.5 rounded-lg border ${fairness.fairness_flag ? "text-emerald-400 border-emerald-500/20 bg-emerald-500/5" : "text-red-400 border-red-500/20 bg-red-500/5"}`}>
                                {fairness.fairness_flag ? "COMPLIANT" : "NON-COMPLIANT"}
                            </span>
                        </Card>

                        {/* Group Performance Breakdown */}
                        {fairness.group_metrics && (
                            <Section title={`Group Performance Breakdown (${results.n_groups} groups)`} badge={fairness.fairness_flag ? "FAIR" : "BIASED"}>
                                <div className="space-y-3">
                                    {Object.entries(fairness.group_metrics).map(([group, metrics]: any) => (
                                        <Card key={group} className="p-4">
                                            <div className="flex items-center justify-between mb-3">
                                                <span className="text-xs font-black text-white">{sensitiveCol} = {group}</span>
                                                <span className="text-[9px] text-slate-500">{metrics.count} samples</span>
                                            </div>
                                            <div className="grid grid-cols-5 gap-2">
                                                <div className="text-center">
                                                    <p className="text-[8px] uppercase font-black text-slate-600">Accuracy</p>
                                                    <p className="text-sm font-black text-white">{(metrics.accuracy * 100).toFixed(1)}%</p>
                                                </div>
                                                <div className="text-center">
                                                    <p className="text-[8px] uppercase font-black text-slate-600">Precision</p>
                                                    <p className="text-sm font-black text-white">{(metrics.precision * 100).toFixed(1)}%</p>
                                                </div>
                                                <div className="text-center">
                                                    <p className="text-[8px] uppercase font-black text-slate-600">Recall</p>
                                                    <p className="text-sm font-black text-white">{(metrics.recall * 100).toFixed(1)}%</p>
                                                </div>
                                                <div className="text-center">
                                                    <p className="text-[8px] uppercase font-black text-slate-600">F1</p>
                                                    <p className="text-sm font-black text-white">{(metrics.f1 * 100).toFixed(1)}%</p>
                                                </div>
                                                <div className="text-center">
                                                    <p className="text-[8px] uppercase font-black text-slate-600">Pos Rate</p>
                                                    <p className={`text-sm font-black ${Math.abs(metrics.positive_rate - 0.5) > 0.2 ? "text-amber-400" : "text-white"}`}>
                                                        {(metrics.positive_rate * 100).toFixed(1)}%
                                                    </p>
                                                </div>
                                            </div>
                                            {/* Mini bar */}
                                            <div className="mt-2 h-1.5 bg-white/5 rounded-full overflow-hidden">
                                                <div className="h-full bg-purple-500 rounded-full" style={{ width: `${metrics.accuracy * 100}%` }} />
                                            </div>
                                        </Card>
                                    ))}
                                </div>
                            </Section>
                        )}

                        {/* Policy Gate */}
                        {policy && (
                            <Section title="Fairness Policy Gate" badge={policy.gate_status}>
                                <div className="space-y-2">{(policy.checks ?? []).map((c: any, i: number) => (
                                    <div key={i} className={`flex items-start gap-3 p-3 rounded-xl text-xs font-bold ${c.status === "PASSED" ? "bg-emerald-500/5 text-emerald-400" : c.status === "WARNING" ? "bg-amber-500/5 text-amber-400" : "bg-red-500/5 text-red-400"}`}>
                                        {c.status === "PASSED" ? <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5" /> : <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />}
                                        <div><p className="font-black">{c.name}: {c.actual_value?.toFixed(4)}</p><p className="opacity-80 font-medium">{c.message}</p></div>
                                    </div>
                                ))}</div>
                            </Section>
                        )}
                    </div>
                )}
                {!loading && !results && (
                    <div className="flex flex-col items-center justify-center h-full min-h-[400px] text-center gap-4">
                        <Scale className="w-14 h-14 text-slate-800" />
                        <p className="text-sm font-black uppercase text-slate-700 tracking-widest">No Fairness Analysis Yet</p>
                        <p className="text-xs text-slate-600 max-w-md">Upload a model and dataset with a sensitive feature column to detect bias and measure equitable performance across demographic groups.</p>
                    </div>
                )}
            </div>
        </div>
    );
}

// ═══════════════════════════════════════════════
// MODULE 8 — LLM GOVERNANCE
// ═══════════════════════════════════════════════
function LLMGovernancePage({ state, setState, onAction }: any) {
    const { prompt, response, additionalResponses, referenceFacts, retrievedChunks, modelName, results, ragReport, history, loading, error } = state;
    const setLState = (chunk: any) => setState((prev: any) => {
        const next = { ...prev };
        Object.keys(chunk).forEach(k => { next[k] = typeof chunk[k] === 'function' ? chunk[k](prev[k]) : chunk[k]; });
        return next;
    });

    const run = async () => {
        if (!prompt.trim() || !response.trim()) { setLState({ error: "Provide both prompt and response." }); return; }
        setLState({ loading: true, error: null, results: null });
        const body: any = { prompt, response, model_name: modelName };
        if (additionalResponses?.trim()) body.additional_responses = additionalResponses.split("\n---\n").filter(Boolean);
        if (referenceFacts?.trim()) body.reference_facts = referenceFacts.split("\n").filter(Boolean);
        try {
            const r = await fetch(`${API_BASE}/api/v1/llm/evaluate`, {
                method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
            });
            const d = await r.json();
            if (!r.ok) throw new Error(d.detail || "LLM evaluation failed.");
            setLState({ results: d });

            // Also log RAG if chunks provided
            if (retrievedChunks?.trim()) {
                const ragChunks = retrievedChunks.split("\n---\n").filter(Boolean);
                await fetch(`${API_BASE}/api/v1/rag-eval/${modelName || 'default'}/log`, {
                    method: "POST", headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ query: prompt, answer: response, retrieved_chunks: ragChunks, retrieved_doc_ids: [] })
                });
            }

            onAction();
            fetchRag();
            apiFetch(`/api/v1/llm/history`).then(r => r.json()).then(h => setLState({ history: h })).catch(() => { });
        } catch (e: any) { setLState({ error: e.message }); }
        finally { setLState({ loading: false }); }
    };

    const fetchRag = () => {
        if (!modelName) return;
        apiFetch(`/api/v1/rag-eval/${modelName}/report`)
            .then(r => r.json())
            .then(d => setLState({ ragReport: d.error ? null : d }))
            .catch(() => setLState({ ragReport: null }));
    };

    useEffect(() => {
        apiFetch(`/api/v1/llm/history`).then(r => r.json()).then(h => setLState({ history: h })).catch(() => { });
        fetchRag();
    }, [modelName]);

    const ev = results?.evaluation;
    const riskColor = ev?.llm_risk_level === "HIGH" ? "text-red-400" : ev?.llm_risk_level === "MEDIUM" ? "text-amber-400" : "text-emerald-400";
    const riskBorder = ev?.llm_risk_level === "HIGH" ? "border-red-500/20" : ev?.llm_risk_level === "MEDIUM" ? "border-amber-500/20" : "border-emerald-500/20";

    return (
        <div className="grid grid-cols-1 xl:grid-cols-[400px_1fr] gap-8">
            <div className="space-y-4">
                <Card className="p-4 space-y-2">
                    <p className="text-[9px] font-black uppercase tracking-widest text-slate-500">Model Name</p>
                    <input value={modelName} onChange={(e: any) => setLState({ modelName: e.target.value })}
                        className="w-full bg-black/40 border border-white/5 rounded-lg px-3 py-2 text-sm text-white font-bold" />
                </Card>
                <Card className="p-4 space-y-2">
                    <p className="text-[9px] font-black uppercase tracking-widest text-slate-500">Prompt / Query</p>
                    <textarea value={prompt} onChange={(e: any) => setLState({ prompt: e.target.value })} rows={3}
                        placeholder="Enter the prompt sent to the LLM..."
                        className="w-full bg-black/40 border border-white/5 rounded-lg px-3 py-2 text-sm text-white resize-none" />
                </Card>
                <Card className="p-4 space-y-2">
                    <p className="text-[9px] font-black uppercase tracking-widest text-slate-500">Retrieved Chunks (Separate by ---)</p>
                    <textarea value={retrievedChunks} onChange={(e: any) => setLState({ retrievedChunks: e.target.value })} rows={3}
                        placeholder="Retrieved context chunks..."
                        className="w-full bg-black/40 border border-white/5 rounded-lg px-3 py-2 text-xs text-white resize-none" />
                </Card>
                <Card className="p-4 space-y-2">
                    <p className="text-[9px] font-black uppercase tracking-widest text-slate-500">LLM Answer</p>
                    <textarea value={response} onChange={(e: any) => setLState({ response: e.target.value })} rows={3}
                        placeholder="Enter the LLM response to evaluate..."
                        className="w-full bg-black/40 border border-white/5 rounded-lg px-3 py-2 text-sm text-white resize-none" />
                </Card>
                <ErrBanner msg={error} />
                <button onClick={run} disabled={loading}
                    className="w-full bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-black font-black py-4 rounded-xl text-[10px] uppercase tracking-widest flex items-center justify-center gap-2 transition-all shadow-lg shadow-cyan-500/10">
                    {loading ? <><Loader2 className="w-4 h-4 animate-spin" />Evaluating...</> : <><Brain className="w-4 h-4" />Evaluate Guardrails & RAG</>}
                </button>

                {/* History sidebar */}
                {history.length > 0 && (
                    <Section title={`Scan History (${history.length})`} defaultOpen={false}>
                        <div className="space-y-1.5 max-h-[200px] overflow-y-auto">{history.map((h: any, i: number) => (
                            <div key={i} className={`flex items-center justify-between text-[10px] py-1.5 px-2 rounded-lg bg-white/[0.02] ${h.llm_risk_level === "HIGH" ? "border border-red-500/10" : ""}`}>
                                <span className="text-slate-500 font-mono truncate max-w-[120px]">{h.prompt_hash}</span>
                                <span className={`font-black ${h.llm_risk_level === "HIGH" ? "text-red-400" : h.llm_risk_level === "MEDIUM" ? "text-amber-400" : "text-emerald-400"}`}>
                                    {h.llm_risk_level} ({h.llm_risk_score?.toFixed(0)})
                                </span>
                            </div>
                        ))}</div>
                    </Section>
                )}
            </div>
            <div className="space-y-6 min-h-[400px]">
                {loading && <Spinner label="Evaluating LLM Governance..." />}
                {ev && !loading && (
                    <div className="space-y-5">
                        {/* Risk Score Hero */}
                        <Card className={`p-8 text-center border ${riskBorder} bg-[#0E1014]`}>
                            <p className="text-[9px] uppercase font-black tracking-[0.35em] text-slate-600 mb-1">LLM Risk Score</p>
                            <div className={`text-[72px] leading-none font-black ${riskColor}`}>{Math.round(ev.llm_risk_score)}</div>
                            <p className={`text-[10px] font-black uppercase mt-2 tracking-widest ${riskColor}`}>{ev.llm_risk_level} RISK</p>
                        </Card>

                        {/* 4 Metric Cards */}
                        <div className="grid grid-cols-2 gap-4">
                            <Card className={`p-5 border ${ev.prompt_injection_flag ? "border-red-500/20" : "border-emerald-500/20"}`}>
                                <div className="flex items-center justify-between mb-2">
                                    <p className="text-[9px] uppercase font-black tracking-widest text-slate-600">Prompt Injection</p>
                                    <span className={`text-[8px] font-black uppercase px-2 py-0.5 rounded-lg border ${ev.prompt_injection_flag ? "text-red-400 border-red-500/20 bg-red-500/5" : "text-emerald-400 border-emerald-500/20 bg-emerald-500/5"}`}>
                                        {ev.prompt_injection_flag ? "DETECTED" : "CLEAN"}
                                    </span>
                                </div>
                                <p className={`text-2xl font-black ${ev.prompt_injection.suspicion_score > 0.3 ? "text-red-400" : "text-emerald-400"}`}>
                                    {(ev.prompt_injection.suspicion_score * 100).toFixed(0)}%
                                </p>
                                <p className="text-[9px] text-slate-600">Suspicion Score • {ev.prompt_injection.matched_patterns} patterns matched</p>
                            </Card>

                            <Card className={`p-5 border ${ev.toxicity_score > 0.3 ? "border-red-500/20" : "border-emerald-500/20"}`}>
                                <div className="flex items-center justify-between mb-2">
                                    <p className="text-[9px] uppercase font-black tracking-widest text-slate-600">Toxicity</p>
                                    <span className={`text-[8px] font-black uppercase px-2 py-0.5 rounded-lg border ${ev.toxicity_response.severity === "HIGH" ? "text-red-400 border-red-500/20 bg-red-500/5" : ev.toxicity_response.severity === "MEDIUM" ? "text-amber-400 border-amber-500/20 bg-amber-500/5" : "text-emerald-400 border-emerald-500/20 bg-emerald-500/5"}`}>
                                        {ev.toxicity_response.severity}
                                    </span>
                                </div>
                                <p className={`text-2xl font-black ${ev.toxicity_score > 0.3 ? "text-red-400" : "text-emerald-400"}`}>
                                    {(ev.toxicity_score * 100).toFixed(0)}%
                                </p>
                                <p className="text-[9px] text-slate-600">{ev.toxicity_response.toxic_keywords} toxic • {ev.toxicity_response.profanity_count} profane</p>
                                {/* Toxicity bar */}
                                <div className="mt-2 h-2 bg-white/5 rounded-full overflow-hidden">
                                    <div className={`h-full rounded-full transition-all ${ev.toxicity_score > 0.5 ? "bg-red-500" : ev.toxicity_score > 0.2 ? "bg-amber-500" : "bg-emerald-500"}`}
                                        style={{ width: `${Math.min(ev.toxicity_score * 100, 100)}%` }} />
                                </div>
                            </Card>

                            <Card className={`p-5 border ${ev.hallucination_risk > 0.5 ? "border-amber-500/20" : "border-emerald-500/20"}`}>
                                <p className="text-[9px] uppercase font-black tracking-widest text-slate-600 mb-2">Hallucination Risk</p>
                                <p className={`text-2xl font-black ${ev.hallucination_risk > 0.5 ? "text-amber-400" : "text-emerald-400"}`}>
                                    {(ev.hallucination_risk * 100).toFixed(0)}%
                                </p>
                                <p className="text-[9px] text-slate-600">Overconfidence: {(ev.hallucination.overconfidence_ratio * 100).toFixed(0)}%</p>
                                <div className="flex items-center gap-3 mt-2 text-[9px]">
                                    <span className="text-slate-500">Hedging: {ev.hallucination.hedge_phrases}</span>
                                    <span className="text-slate-500">Claims: {ev.hallucination.specific_claims_count}</span>
                                </div>
                            </Card>

                            <Card className={`p-5 border ${ev.stability_score < 0.7 ? "border-amber-500/20" : "border-emerald-500/20"}`}>
                                <p className="text-[9px] uppercase font-black tracking-widest text-slate-600 mb-2">Response Stability</p>
                                <p className={`text-2xl font-black ${ev.stability_score < 0.7 ? "text-amber-400" : "text-emerald-400"}`}>
                                    {(ev.stability_score * 100).toFixed(0)}%
                                </p>
                                <p className="text-[9px] text-slate-600">{ev.stability.n_responses} response(s) analyzed</p>
                                <p className="text-[9px] text-slate-600">Length CV: {ev.stability.length_cv?.toFixed(3)}</p>
                            </Card>
                        </div>

                        {/* Fingerprints */}
                        <Card className="p-4 flex items-center justify-between">
                            <div className="flex items-center gap-4">
                                <div className="text-center">
                                    <p className="text-[8px] uppercase font-black text-slate-600">Prompt Hash</p>
                                    <p className="text-[10px] font-mono text-slate-400">{ev.prompt_hash}</p>
                                </div>
                                <div className="text-center">
                                    <p className="text-[8px] uppercase font-black text-slate-600">Response Hash</p>
                                    <p className="text-[10px] font-mono text-slate-400">{ev.response_hash}</p>
                                </div>
                            </div>
                            {results.scan_id && <p className="text-[9px] text-slate-600 font-mono">Scan: {results.scan_id.slice(0, 12)}...</p>}
                        </Card>

                        {/* Policy */}
                        {results.policy && (
                            <Section title="LLM Policy Gate" badge={results.policy.gate_status}>
                                <div className="space-y-2">{(results.policy.checks ?? []).map((c: any, i: number) => (
                                    <div key={i} className={`flex items-start gap-3 p-3 rounded-xl text-xs font-bold ${c.status === "PASSED" ? "bg-emerald-500/5 text-emerald-400" : c.status === "WARNING" ? "bg-amber-500/5 text-amber-400" : "bg-red-500/5 text-red-400"}`}>
                                        {c.status === "PASSED" ? <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5" /> : <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />}
                                        <div><p className="font-black">{c.name}: {c.actual_value?.toFixed(4)}</p><p className="opacity-80 font-medium">{c.message}</p></div>
                                    </div>
                                ))}</div>
                            </Section>
                        )}
                    </div>
                )}
                {!loading && !results && !ragReport && (
                    <div className="flex flex-col items-center justify-center h-full min-h-[400px] text-center gap-4">
                        <Brain className="w-14 h-14 text-slate-800" />
                        <p className="text-sm font-black uppercase text-slate-700 tracking-widest">No LLM Evaluation Yet</p>
                        <p className="text-xs text-slate-600 max-w-md">Enter a prompt and response to evaluate for injection attacks, toxicity, hallucination risk, and response stability.</p>
                    </div>
                )}

                {!loading && ragReport && (
                    <Section title="RAG System Observability">
                        <div className="grid grid-cols-2 xl:grid-cols-4 gap-4 mb-6">
                            <Card className="p-4 border border-white/5 bg-[#0E1014]">
                                <p className="text-[9px] uppercase font-black tracking-widest text-slate-500 mb-1">Context Relevance</p>
                                <p className="text-2xl font-black text-white">{(ragReport.avg_context_relevance * 100 || 0).toFixed(1)}%</p>
                            </Card>
                            <Card className="p-4 border border-white/5 bg-[#0E1014]">
                                <p className="text-[9px] uppercase font-black tracking-widest text-slate-500 mb-1">Grounding Fidelity</p>
                                <p className="text-2xl font-black text-emerald-400">{(ragReport.avg_grounding_fidelity * 100 || 0).toFixed(1)}%</p>
                            </Card>
                            <Card className="p-4 border border-white/5 bg-[#0E1014]">
                                <p className="text-[9px] uppercase font-black tracking-widest text-slate-500 mb-1">Retrieval Hit Rate</p>
                                <p className="text-2xl font-black text-cyan-400">{(ragReport.retrieval_hit_rate * 100 || 0).toFixed(1)}%</p>
                            </Card>
                            <Card className="p-4 border border-white/5 bg-[#0E1014]">
                                <p className="text-[9px] uppercase font-black tracking-widest text-slate-500 mb-1">High Risk Ratio</p>
                                <p className="text-2xl font-black text-red-400">{((ragReport.hallucination_risk_distribution?.high || 0) * 100).toFixed(1)}%</p>
                            </Card>
                        </div>
                        <Card className="p-5 border border-white/5 h-[300px] bg-[#0E1014]">
                            <p className="text-[10px] font-black uppercase text-slate-400 mb-4 tracking-widest">Grounding Fidelity Time-Series</p>
                            <ResponsiveContainer width="100%" height="85%">
                                <LineChart data={ragReport.time_series || []}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                                    <XAxis dataKey="time" stroke="#475569" tick={{fill: '#475569', fontSize: 10}} tickFormatter={(v) => new Date(v).toLocaleTimeString()} />
                                    <YAxis stroke="#475569" tick={{fill: '#475569', fontSize: 10}} domain={[0, 1]} />
                                    <ReTooltip contentStyle={{backgroundColor: '#0E1014', borderColor: 'rgba(255,255,255,0.1)'}} />
                                    <Line type="stepAfter" dataKey="grounding_fidelity" stroke="#10b981" strokeWidth={2} dot={{fill: '#10b981', r: 3}} activeDot={{r: 5}} />
                                </LineChart>
                            </ResponsiveContainer>
                        </Card>
                    </Section>
                )}
            </div>
        </div>
    );
}

// ═══════════════════════════════════════════════
// DASHBOARD SHELL
// ═══════════════════════════════════════════════
const NAV_CATEGORIES = [
    {
        id: "governance",
        label: "Governance",
        icon: ShieldCheck,
        items: [
            { id: "audit", label: "Model Audit", sub: "Core compliance & risk", icon: ShieldCheck },
            { id: "governance-score", label: "Governance Score", sub: "Live score · cert · gate", icon: ShieldCheck },
            { id: "report", label: "Report Card", sub: "Compliance certificates", icon: FileText },
            { id: "fairness", label: "Fairness", sub: "Bias & equity metrics", icon: Scale },
            { id: "explainability", label: "Explainability", sub: "SHAP & importance", icon: Zap },
            { id: "behavior", label: "Behavior Test", sub: "Scenario robustness", icon: Activity },
        ]
    },
    {
        id: "tracking",
        label: "Asset tracking",
        icon: Package,
        items: [
            { id: "registry", label: "Model Registry", sub: "Version control", icon: Package },
            { id: "datasets", label: "Datasets", sub: "Lineage & assets", icon: Database },
            { id: "experiments", label: "Experiments", sub: "Training tracker", icon: FlaskConical },
            { id: "history", label: "Scan History", sub: "Past audits & compare", icon: Clock },
        ]
    },
    {
        id: "monitoring",
        label: "Live Guard",
        icon: MonitorCheck,
        items: [
            { id: "observe", label: "Observability", sub: "Live drift · perf · feed", icon: Eye },
            { id: "streaming", label: "Stream Drift", sub: "Real-time detection", icon: BarChart3 },
            { id: "performance", label: "Performance", sub: "Drift & stats", icon: MonitorCheck },
            { id: "monitoring", label: "Production Probe", sub: "Active inference testing", icon: Wifi },
            { id: "health", label: "Data Quality", sub: "Validation scans", icon: Search },
        ]
    },
    {
        id: "ops",
        label: "Operations",
        icon: GitBranch,
        items: [
            { id: "ci", label: "CI/CD Gate", sub: "Pipeline governance", icon: GitBranch },
            { id: "deployments", label: "Deployments", sub: "Environments · Promo", icon: Layout },
            { id: "security", label: "Security", sub: "Vulnerability audit", icon: ShieldAlert },
        ]
    },
    {
        id: "safety",
        label: "AI Safety",
        icon: Brain,
        items: [
            { id: "llm", label: "LLM Guard", sub: "Prompt & response safety", icon: Brain },
            { id: "advisory", label: "AI Advisor", sub: "Governance copilot", icon: Eye },
        ]
    },
    {
        id: "admin",
        label: "Administration",
        icon: Building2,
        items: [
            { id: "enterprise", label: "Enterprise Hub", sub: "Org · Policies · Audit", icon: Building2 },
        ]
    }
];

const ALL_NAV_ITEMS = NAV_CATEGORIES.flatMap(c => c.items);

export default function DashboardPage() {
    const [active, setActive] = useState("audit");
    const { user, logout } = useAuth();

    // Lifted States
    const [auditState, setAuditState] = useState({
        modelFile: null, trainFile: null, valFile: null, labelCol: "target",
        checks: { accuracy: true, f1: true, psi_drift: true, overfitting_check: true },
        modelMeta: null, trainSum: null, results: null, loading: false, error: null, activePolicy: null
    });
    const [behaviorState, setBehaviorState] = useState({
        modelFile: null, refFile: null, scenarios: { monte_carlo_stability: true, ood_boundary_test: true },
        labelCol: "target", results: null, loading: false, error: null
    });
    const [monitorState, setMonitorState] = useState({
        endpointUrl: "", probeFile: null, results: null, loading: false, error: null
    });
    const [streamState, setStreamState] = useState({
        modelId: "default", metrics: null, alerts: [], eventCount: 0, wsStatus: "disconnected", streamModels: []
    });
    const [advisoryState, setAdvisoryState] = useState({
        scanId: "", question: "Why is my governance score low?", advisory: null, loading: false, error: null, useLLM: false
    });
    const [fairnessState, setFairnessState] = useState({
        modelFile: null as File | null, dataFile: null as File | null,
        sensitiveCol: "gender", labelCol: "target",
        results: null as any, loading: false, error: null as string | null
    });
    const [llmState, setLlmState] = useState({
        prompt: "", response: "", additionalResponses: "" as string,
        referenceFacts: "" as string, modelName: "gpt-4",
        results: null as any, history: [] as any[], loading: false, error: null as string | null
    });
    const [enterpriseState, setEnterpriseState] = useState({
        orgs: [] as any[], policies: [] as any[], scanHistory: [] as any[], alertRules: [] as any[], alertEvents: [] as any[],
        auditLogs: [] as any[], models: [] as any[], loading: true, scanA: "", scanB: "", comparison: null as any
    });
    const [registryState, setRegistryState] = useState({});
    const [datasetsState, setDatasetsState] = useState({});
    const [experimentsState, setExperimentsState] = useState({});
    const [explainabilityState, setExplainabilityState] = useState({});
    const [ciState, setCiState] = useState({});
    const [deploymentsState, setDeploymentsState] = useState({});
    const [healthState, setHealthState] = useState({});
    const [performanceState, setPerformanceState] = useState({});
    const [securityState, setSecurityState] = useState({});
    const [historyState, setHistoryState] = useState({});
    const [reportCardState, setReportCardState] = useState({});
    const [observabilityState, setObservabilityState] = useState({ modelId: "" });
    const [governanceState, setGovernanceState] = useState<{ selectedModelId: string }>({ selectedModelId: "" });

    const refreshEnterprise = async () => {
        try {
            const [ro, rp, rh, rar, rae, ral, rm] = await Promise.all([
                apiFetch(`/api/v1/orgs`).then(r => r.json()),
                apiFetch(`/api/v1/policies`).then(r => r.json()),
                apiFetch(`/api/v1/history`).then(r => r.json()),
                apiFetch(`/api/v1/alerts/rules`).then(r => r.json()),
                apiFetch(`/api/v1/alerts/events`).then(r => r.json()),
                apiFetch(`/api/v1/audit-logs?limit=40`).then(r => r.json()),
                apiFetch(`/api/v1/models`).then(r => r.json()),
            ]);
            const isArr = (v: any) => Array.isArray(v) ? v : [];
            setEnterpriseState(prev => ({
                ...prev,
                orgs: isArr(ro),
                policies: isArr(rp),
                scanHistory: isArr(rh),
                alertRules: isArr(rar),
                alertEvents: isArr(rae),
                auditLogs: isArr(ral),
                models: isArr(rm),
                loading: false
            }));
        } catch { }
    };

    useEffect(() => {
        refreshEnterprise();
    }, []);

    const nav = ALL_NAV_ITEMS.find(n => n.id === active)!;

    return (
        <div className="flex h-screen bg-[#050608] text-slate-200 overflow-hidden font-inter selection:bg-orange-500/30">
            {/* ════ Sidebar Navigation ════ */}
            <aside className="w-72 flex-shrink-0 border-r border-white/5 bg-[#08090B] flex flex-col h-full shadow-[20px_0_40px_rgba(0,0,0,0.5)] z-[60]">
                {/* Brand */}
                <div className="p-8 pb-4 flex items-center gap-4">
                    <div className="w-10 h-10 rounded-2xl bg-orange-600 flex items-center justify-center shadow-lg shadow-orange-500/20 group cursor-pointer hover:rotate-12 transition-transform">
                        <ShieldCheck className="w-5 h-5 text-black" />
                    </div>
                    <div>
                        <h1 className="text-sm font-black tracking-tight text-white uppercase flex items-center gap-1.5 leading-none">
                            ML Guard <span className="px-1.5 py-0.5 rounded-md bg-white/5 text-[8px] text-orange-400 font-mono tracking-normal">V7.2</span>
                        </h1>
                        <p className="text-[8px] font-black uppercase tracking-[0.2em] text-slate-600 mt-1">Enterprise Governance</p>
                    </div>
                </div>

                {/* Nav Zones */}
                <div className="flex-1 overflow-y-auto px-4 py-6 space-y-8 no-scrollbar">
                    {NAV_CATEGORIES.map(cat => (
                        <div key={cat.id} className="space-y-3">
                            <h3 className="px-4 text-[9px] font-black uppercase tracking-[0.25em] text-slate-700 flex items-center gap-2">
                                <cat.icon className="w-3 h-3 text-slate-800" />
                                {cat.label}
                            </h3>
                            <div className="space-y-1">
                                {cat.items.map(n => (
                                    <button
                                        key={n.id}
                                        onClick={() => setActive(n.id)}
                                        className={`w-full group flex items-center gap-3 px-4 py-2.5 rounded-xl transition-all duration-300 relative overflow-hidden ${active === n.id
                                            ? "bg-white/[0.04] text-white shadow-[0_0_20px_rgba(255,255,255,0.02)]"
                                            : "text-slate-500 hover:text-slate-300 hover:bg-white/[0.02]"
                                            }`}
                                    >
                                        {active === n.id && (
                                            <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-1/2 bg-orange-500 rounded-r-full shadow-[0_0_10px_rgba(249,115,22,0.8)]" />
                                        )}
                                        <n.icon className={`w-4 h-4 transition-transform group-hover:scale-110 ${active === n.id ? "text-orange-400" : "text-slate-700"}`} />
                                        <div className="text-left">
                                            <p className="text-[11px] font-black uppercase tracking-tight">{n.label}</p>
                                            <p className="text-[8px] font-medium text-slate-600 line-clamp-1 truncate w-40">{n.sub}</p>
                                        </div>
                                    </button>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>

                {/* Identity & Org */}
                <div className="p-6 mt-auto border-t border-white/5 bg-black/20">
                    {user && (
                        <div className="flex flex-col gap-4">
                            <div className="flex items-center gap-3">
                                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-orange-400 to-red-500 p-0.5 shadow-lg shadow-orange-500/10">
                                    <div className="w-full h-full rounded-full bg-[#08090B] flex items-center justify-center">
                                        <User className="w-3.5 h-3.5 text-orange-400" />
                                    </div>
                                </div>
                                <div className="min-w-0">
                                    <p className="text-[10px] font-black text-white uppercase truncate">{user.displayName || user.email?.split('@')[0]}</p>
                                    <p className="text-[8px] font-bold text-slate-600 uppercase tracking-widest leading-none mt-1">{user.role || 'Operator'}</p>
                                </div>
                                <button
                                    onClick={logout}
                                    className="ml-auto p-2 rounded-lg bg-red-500/10 text-red-500 hover:bg-red-500 hover:text-white transition-all border border-red-500/10"
                                    title="Sign Out"
                                >
                                    <LogOut className="w-3 h-3" />
                                </button>
                            </div>
                        </div>
                    )}
                </div>
            </aside>

            {/* ════ Content Area ════ */}
            <main className="flex-1 flex flex-col h-full overflow-hidden relative">
                {/* Header Shadow Overlay */}
                <div className="absolute top-0 left-0 right-0 h-24 bg-gradient-to-b from-[#050608] to-transparent pointer-events-none z-10 opactiy-50" />

                {/* Top Toolbar */}
                <header className="px-10 h-24 flex items-center justify-between z-20 shrink-0">
                    <div className="flex items-center gap-6">
                        <div className="flex items-center gap-2">
                            <div className="p-2.5 rounded-xl bg-orange-600/10 border border-orange-500/10">
                                <nav.icon className="w-5 h-5 text-orange-500" />
                            </div>
                            <div>
                                <div className="flex items-center gap-2">
                                    <h2 className="text-xl font-black text-white tracking-tighter uppercase">{nav.label}</h2>
                                    <div className="px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/20 text-[8px] font-black text-emerald-500">SYSTEM READY</div>
                                </div>
                                <p className="text-[10px] font-bold text-slate-600 uppercase tracking-widest mt-0.5">{nav.sub}</p>
                            </div>
                        </div>
                    </div>

                    <div className="flex items-center gap-3">
                        <div className="px-4 py-2 rounded-xl bg-white/[0.02] border border-white/5 flex items-center gap-3">
                            <Clock className="w-3.5 h-3.5 text-slate-700" />
                            <span className="text-[11px] font-mono text-slate-500 tabular-nums">LIVE MONITOR</span>
                        </div>
                        <NotificationsBell />
                    </div>
                </header>

                {/* Scrollable Viewport */}
                <div className="flex-1 overflow-y-auto px-10 pb-20 no-scrollbar relative z-20">
                    <div className="max-w-[1400px]">
                        <div className="animate-in fade-in slide-in-from-bottom-6 duration-700">
                            {active === "audit" && <ModelAuditPage state={auditState} setState={setAuditState} onAction={refreshEnterprise} />}
                            {active === "behavior" && <BehaviorTestingPage state={behaviorState} setState={setBehaviorState} onAction={refreshEnterprise} />}
                            {active === "monitoring" && <LiveMonitoringPage state={monitorState} setState={setMonitorState} onAction={refreshEnterprise} />}
                            {active === "streaming" && <StreamingMonitorPage state={streamState} setState={setStreamState} onAction={refreshEnterprise} />}
                            {active === "advisory" && <AIAdvisoryPage state={advisoryState} setState={setAdvisoryState} onAction={refreshEnterprise} />}
                            {active === "fairness" && <FairnessAnalysisPage state={fairnessState} setState={setFairnessState} onAction={refreshEnterprise} />}
                            {active === "llm" && <LLMGovernancePage state={llmState} setState={setLlmState} onAction={refreshEnterprise} />}
                            {active === "enterprise" && <EnterprisePage state={enterpriseState} setState={setEnterpriseState} onAction={refreshEnterprise} />}

                            {active === "registry" && <ModelRegistryPage state={registryState} setState={setRegistryState} onAction={refreshEnterprise} />}
                            {active === "datasets" && <DatasetsPage state={datasetsState} setState={setDatasetsState} onAction={refreshEnterprise} />}
                            {active === "experiments" && <ExperimentsPage state={experimentsState} setState={setExperimentsState} onAction={refreshEnterprise} />}
                            {active === "explainability" && <ExplainabilityPage state={explainabilityState} setState={setExplainabilityState} onAction={refreshEnterprise} />}
                            {active === "ci" && <CIModulePage state={ciState} setState={setCiState} onAction={refreshEnterprise} />}
                            {active === "deployments" && <DeploymentsPage state={deploymentsState} setState={setDeploymentsState} onAction={refreshEnterprise} />}
                            {active === "health" && <DataQualityPage state={healthState} setState={setHealthState} onAction={refreshEnterprise} />}
                            {active === "performance" && <PerformancePage state={performanceState} setState={setPerformanceState} onAction={refreshEnterprise} />}
                            {active === "security" && <ModelSecurityPage state={securityState} setState={setSecurityState} onAction={refreshEnterprise} />}
                            {active === "history" && <ScanHistoryPage state={historyState} setState={setHistoryState} onAction={refreshEnterprise} />}
                            {active === "report" && <ModelReportCardModule state={reportCardState} setState={setReportCardState} onAction={refreshEnterprise} />}
                            {active === "observe" && <ObservabilityModule state={observabilityState} setState={setObservabilityState} onAction={refreshEnterprise} />}
                            {active === "governance-score" && <GovernanceModule state={governanceState} setState={setGovernanceState} onAction={refreshEnterprise} />}
                        </div>
                    </div>
                </div>

                {/* Ambient Glows */}
                <div className="absolute top-[-10%] right-[-10%] w-[50%] h-[50%] bg-orange-500/5 blur-[120px] pointer-events-none rounded-full" />
                <div className="absolute bottom-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-500/5 blur-[120px] pointer-events-none rounded-full" />
            </main>
        </div>
    );
}
