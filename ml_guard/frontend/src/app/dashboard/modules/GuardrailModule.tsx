"use client";
import { apiFetch, safeJson } from "@/lib/api";
import React, { useState, useEffect, useCallback } from "react";
import {
    Shield, ShieldAlert, ShieldCheck, Zap, Activity, Clock, 
    RefreshCw, Play, Search, AlertCircle, CheckCircle2, XCircle,
    ChevronDown, ChevronUp, Lock, Unlock, Eye, Trash2
} from "lucide-react";

const Badge = ({ label, variant = "neutral" }: { label: string; variant?: string }) => {
    const cls = variant === "block" ? "bg-red-500/10 text-red-400 border-red-500/30"
        : variant === "flag" ? "bg-orange-500/10 text-orange-400 border-orange-500/30"
            : variant === "allow" ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                : variant === "redact" ? "bg-blue-500/10 text-blue-400 border-blue-500/30"
                    : "bg-white/5 text-slate-400 border-white/10";
    return <span className={`text-[9px] font-black uppercase px-2 py-0.5 rounded border ${cls}`}>{label}</span>;
};

const StatCard = ({ label, value, sub, icon: Icon, accent }: any) => {
    return (
        <div className="bg-[#0E1014] border border-white/[0.06] rounded-2xl p-5 flex flex-col gap-3">
            <div className="flex items-center justify-between">
                <p className="text-[9px] font-black uppercase tracking-[0.2em] text-slate-600">{label}</p>
                {Icon && <Icon className="w-4 h-4 text-slate-700" />}
            </div>
            <div className="flex items-end gap-2">
                <p className={`text-2xl font-black ${accent ? "text-orange-400" : "text-white"}`}>{value ?? "—"}</p>
            </div>
            {sub && <div className="text-[10px] text-slate-600 font-medium">{sub}</div>}
        </div>
    );
};

export default function GuardrailModule({ state }: any) {
    const [models, setModels] = useState<any[]>([]);
    const [selectedModelId, setSelectedModelId] = useState<string>(state?.modelId || "");
    const [config, setConfig] = useState<any>(null);
    const [stats, setStats] = useState<any>(null);
    const [traces, setTraces] = useState<any[]>([]);
    const [loading, setLoading] = useState(false);
    
    // Test section state
    const [testPrompt, setTestPrompt] = useState("");
    const [testResponse, setTestResponse] = useState("");
    const [testResult, setTestResult] = useState<any>(null);
    const [testing, setTesting] = useState(false);

    const loadModels = useCallback(async () => {
        try {
            const r = await apiFetch(`/api/v1/models`);
            const d = await safeJson<any>(r);
            setModels(Array.isArray(d) ? d : (d.items || []));
        } catch { }
    }, []);

    const loadGuardrail = useCallback(async (modelId: string) => {
        if (!modelId) return;
        setLoading(true);
        try {
            // First try to find or create config
            const r = await apiFetch(`/api/guardrail`, {
                method: "POST",
                body: JSON.stringify({ model_id: modelId, name: `Guardrail for ${modelId}` })
            });
            const cfg = await safeJson<any>(r);
            setConfig(cfg);

            if (cfg?.id) {
                // Load stats and traces
                const [s, t] = await Promise.all([
                    apiFetch(`/api/guardrail/${cfg.id}/stats`).then(r => safeJson<any>(r)),
                    apiFetch(`/api/guardrail/${cfg.id}/traces`).then(r => safeJson<any>(r))
                ]);
                setStats(s);
                setTraces(t || []);
            }
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        loadModels();
    }, [loadModels]);

    useEffect(() => {
        if (selectedModelId) loadGuardrail(selectedModelId);
    }, [selectedModelId, loadGuardrail]);

    const updateConfig = async (updates: any) => {
        if (!config) return;
        const newConfig = { ...config, ...updates };
        setConfig(newConfig);
        try {
            await apiFetch(`/api/guardrail`, {
                method: "POST",
                body: JSON.stringify({ ...newConfig, model_id: selectedModelId })
            });
        } catch (e) {
            console.error("Failed to save config", e);
        }
    };

    const toggleCheck = (type: "input" | "output", check: string) => {
        const field = type === "input" ? "enabled_input_checks" : "enabled_output_checks";
        const current = config[field] || [];
        const next = current.includes(check) 
            ? current.filter((c: string) => c !== check)
            : [...current, check];
        updateConfig({ [field]: next });
    };

    const [testError, setTestError] = useState<string | null>(null);

    const runTest = async () => {
        if (!config || !testPrompt) return;
        setTesting(true);
        setTestResult(null);
        setTestError(null);
        try {
            const r = await apiFetch(`/api/guardrail/${config.id}/evaluate`, {
                method: "POST",
                body: JSON.stringify({ prompt: testPrompt, response: testResponse })
            });
            const res = await safeJson<any>(r);
            // Validate the response has the expected shape before setting state
            if (!r.ok || res?.detail || res?.error) {
                setTestError(res?.detail || res?.error || `Server error (${r.status})`);
            } else if (!res?.action) {
                setTestError("Unexpected response from guardrail engine.");
            } else {
                setTestResult(res);
                // Refresh traces
                loadGuardrail(selectedModelId);
            }
        } catch (e: any) {
            setTestError(e?.message || "Network error during evaluation.");
        } finally {
            setTesting(false);
        }
    };


    return (
        <div className="space-y-8">
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-xl font-black text-white tracking-tight uppercase">Real-Time Guardrails</h2>
                    <p className="text-[10px] text-slate-600 font-bold uppercase tracking-widest mt-1">
                        Active Proxy · <span className="text-orange-500/80">Latency &lt; 300ms</span> · Input/Output Filtering
                    </p>
                </div>
                <div className="flex items-center gap-3">
                    <select
                        value={selectedModelId}
                        onChange={e => setSelectedModelId(e.target.value)}
                        className="bg-[#0E1014] border border-white/10 text-white text-xs font-bold rounded-xl px-4 py-2.5 focus:outline-none focus:border-orange-500/50"
                    >
                        <option value="">Select model...</option>
                        {models.map((m: any) => (
                            <option key={m.id} value={m.id}>{m.name}</option>
                        ))}
                    </select>
                    <button
                        onClick={() => loadGuardrail(selectedModelId)}
                        disabled={loading}
                        className="p-2.5 rounded-xl bg-orange-500/10 border border-orange-500/20 text-orange-400 hover:bg-orange-500/20 transition-colors"
                    >
                        <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
                    </button>
                </div>
            </div>

            {config ? (
                <div className="grid grid-cols-12 gap-6">
                    {/* Left: Settings & Stats */}
                    <div className="col-span-12 lg:col-span-4 space-y-6">
                        <div className="grid grid-cols-1 gap-4">
                            <StatCard 
                                label="Total Evaluated" 
                                value={stats?.total_evaluated?.toLocaleString() || "0"} 
                                icon={Activity}
                            />
                            <StatCard 
                                label="Block Rate" 
                                value={stats?.blocked_pct ? `${stats.blocked_pct}%` : "0%"} 
                                icon={ShieldAlert}
                                accent={stats?.blocked_pct > 5}
                            />
                            <StatCard 
                                label="Avg Latency" 
                                value={stats?.avg_latency_ms ? `${stats.avg_latency_ms}ms` : "0ms"} 
                                icon={Zap}
                            />
                        </div>

                        <div className="bg-[#0E1014] border border-white/[0.06] rounded-2xl p-6 space-y-6">
                            <p className="text-[10px] font-black uppercase tracking-widest text-slate-500">Guardrail Engine Settings</p>
                            
                            <div className="space-y-4">
                                <p className="text-[9px] font-black uppercase text-slate-700">Input Checks (Prompt)</p>
                                {[
                                    { id: "injection", label: "Injection Detection", desc: "100+ patterns" },
                                    { id: "pii", label: "PII Scrubber", desc: "Email, SSN, PAN, Card" },
                                    { id: "jailbreak", label: "Jailbreak Heuristics", desc: "DAN, role-play bypass" },
                                    { id: "topic_policy", label: "Topic Control", desc: "TF-IDF similarity" }
                                ].map(c => (
                                    <div key={c.id} className="flex items-center justify-between p-3 rounded-xl bg-white/[0.02] border border-white/5 hover:border-white/10 transition-colors">
                                        <div>
                                            <p className="text-xs font-bold text-white">{c.label}</p>
                                            <p className="text-[9px] text-slate-500">{c.desc}</p>
                                        </div>
                                        <button 
                                            onClick={() => toggleCheck("input", c.id)}
                                            className={`w-10 h-5 rounded-full relative transition-colors ${config.enabled_input_checks.includes(c.id) ? "bg-orange-500" : "bg-slate-800"}`}
                                        >
                                            <div className={`absolute top-1 w-3 h-3 rounded-full bg-white transition-all ${config.enabled_input_checks.includes(c.id) ? "right-1" : "left-1"}`} />
                                        </button>
                                    </div>
                                ))}
                            </div>

                            <div className="space-y-4 pt-2 border-t border-white/5">
                                <p className="text-[9px] font-black uppercase text-slate-700">Output Checks (Response)</p>
                                {[
                                    { id: "toxicity", label: "Toxicity Filter", desc: "Hate, violence, explicit" },
                                    { id: "hallucination", label: "Hallucination Risk", desc: "Grounding fidelity" },
                                    { id: "pii", label: "Output PII Redaction", desc: "Privacy leak prevent" }
                                ].map(c => (
                                    <div key={c.id} className="flex items-center justify-between p-3 rounded-xl bg-white/[0.02] border border-white/5 hover:border-white/10 transition-colors">
                                        <div>
                                            <p className="text-xs font-bold text-white">{c.label}</p>
                                            <p className="text-[9px] text-slate-500">{c.desc}</p>
                                        </div>
                                        <button 
                                            onClick={() => toggleCheck("output", c.id)}
                                            className={`w-10 h-5 rounded-full relative transition-colors ${config.enabled_output_checks.includes(c.id) ? "bg-orange-500" : "bg-slate-800"}`}
                                        >
                                            <div className={`absolute top-1 w-3 h-3 rounded-full bg-white transition-all ${config.enabled_output_checks.includes(c.id) ? "right-1" : "left-1"}`} />
                                        </button>
                                    </div>
                                ))}
                            </div>

                            <div className="space-y-2 pt-2 border-t border-white/5">
                                <p className="text-[9px] font-black uppercase text-slate-700">Action on Block</p>
                                <select 
                                    value={config.action_on_block}
                                    onChange={e => updateConfig({ action_on_block: e.target.value })}
                                    className="w-full bg-black/40 border border-white/10 text-white text-[10px] font-bold rounded-lg px-3 py-2"
                                >
                                    <option value="return_error">Return API Error</option>
                                    <option value="return_fallback_response">Return Fallback Response</option>
                                </select>
                            </div>
                        </div>
                    </div>

                    {/* Right: Test & Traces */}
                    <div className="col-span-12 lg:col-span-8 space-y-6">
                        {/* Test Guardrail */}
                        <div className="bg-[#0E1014] border border-white/[0.06] rounded-2xl p-6">
                            <div className="flex items-center justify-between mb-4">
                                <p className="text-[10px] font-black uppercase tracking-widest text-slate-400">Test Guardrail Engine</p>
                                <button 
                                    onClick={runTest}
                                    disabled={testing || !testPrompt}
                                    className="px-4 py-2 bg-orange-500 hover:bg-orange-600 disabled:bg-slate-800 disabled:text-slate-600 rounded-xl text-[10px] font-black uppercase transition-all flex items-center gap-2"
                                >
                                    {testing ? <RefreshCw className="w-3 h-3 animate-spin" /> : <Play className="w-3 h-3 fill-current" />}
                                    Run Evaluation
                                </button>
                            </div>
                            
                            <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-2">
                                    <p className="text-[9px] font-black uppercase text-slate-700">Prompt Input</p>
                                    <textarea 
                                        value={testPrompt}
                                        onChange={e => setTestPrompt(e.target.value)}
                                        placeholder="Type something risky (e.g. DAN jailbreak or credit card)..."
                                        className="w-full h-32 bg-black/40 border border-white/10 rounded-xl p-3 text-xs text-white focus:outline-none focus:border-orange-500/50 resize-none font-mono"
                                    />
                                </div>
                                <div className="space-y-2">
                                    <p className="text-[9px] font-black uppercase text-slate-700">Response (Optional)</p>
                                    <textarea 
                                        value={testResponse}
                                        onChange={e => setTestResponse(e.target.value)}
                                        placeholder="Simulate an LLM response to check output filters..."
                                        className="w-full h-32 bg-black/40 border border-white/10 rounded-xl p-3 text-xs text-white focus:outline-none focus:border-orange-500/50 resize-none font-mono"
                                    />
                                </div>
                            </div>

                            {testError && (
                                <div className="mt-4 p-4 rounded-xl bg-red-500/10 border border-red-500/20 flex items-center gap-3">
                                    <div className="w-2 h-2 rounded-full bg-red-400 shrink-0" />
                                    <p className="text-xs font-bold text-red-400">{testError}</p>
                                </div>
                            )}

                            {testResult && (
                                <div className="mt-4 p-4 rounded-xl bg-orange-500/5 border border-orange-500/10 animate-in fade-in slide-in-from-top-2 duration-300">
                                    <div className="flex items-center justify-between mb-3">
                                        <div className="flex items-center gap-3">
                                            <Badge label={testResult.action} variant={testResult.action} />
                                            <p className="text-xs font-bold text-white">{testResult.blocked_reason || "Check Passed"}</p>
                                        </div>
                                        <p className="text-[10px] font-mono text-slate-600">Latency: {testResult.latency_ms}ms</p>
                                    </div>
                                    <div className="grid grid-cols-2 gap-3 text-[10px]">
                                        <div className="space-y-1">
                                            <p className="font-black text-slate-700 uppercase">Input Results</p>
                                            {Object.entries(testResult.input_checks || {}).map(([k, v]: any) => (
                                                <div key={k} className="flex justify-between border-b border-white/5 py-1">
                                                    <span className="text-slate-500 capitalize">{k}</span>
                                                    <span className={v.flagged ? "text-red-400 font-bold" : "text-emerald-400"}>{v.flagged ? "FLAGGED" : "OK"}</span>
                                                </div>
                                            ))}
                                        </div>
                                        <div className="space-y-1">
                                            <p className="font-black text-slate-700 uppercase">Output Results</p>
                                            {Object.entries(testResult.output_checks || {}).map(([k, v]: any) => (
                                                <div key={k} className="flex justify-between border-b border-white/5 py-1">
                                                    <span className="text-slate-500 capitalize">{k}</span>
                                                    <span className={v.flagged ? "text-red-400 font-bold" : "text-emerald-400"}>{v.flagged ? "FLAGGED" : "OK"}</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Live Trace Feed */}
                        <div className="bg-[#0E1014] border border-white/[0.06] rounded-2xl overflow-hidden">
                            <div className="px-6 py-4 border-b border-white/5 flex items-center justify-between">
                                <p className="text-[10px] font-black uppercase tracking-widest text-slate-400">Live Evaluation Traces</p>
                                <p className="text-[9px] text-slate-600 font-mono">Last 50 events</p>
                            </div>
                            <div className="max-h-[500px] overflow-y-auto">
                                <table className="w-full text-left">
                                    <thead>
                                        <tr className="border-b border-white/5 text-[9px] font-black uppercase text-slate-600">
                                            <th className="px-6 py-3">Timestamp</th>
                                            <th className="px-6 py-3">Trace ID</th>
                                            <th className="px-6 py-3">Action</th>
                                            <th className="px-6 py-3">Latency</th>
                                            <th className="px-6 py-3 text-right">Violations</th>
                                        </tr>
                                    </thead>
                                    <tbody className="text-xs">
                                        {traces.map((t, i) => (
                                            <tr key={t.id} className="border-b border-white/[0.03] hover:bg-white/[0.01] transition-colors">
                                                <td className="px-6 py-4 text-slate-500 font-mono text-[10px]">
                                                    {new Date(t.timestamp).toLocaleTimeString()}
                                                </td>
                                                <td className="px-6 py-4 text-slate-400 font-mono text-[10px]">
                                                    {t.trace_id.split('-')[0]}...
                                                </td>
                                                <td className="px-6 py-4">
                                                    <Badge label={t.action} variant={t.action} />
                                                </td>
                                                <td className="px-6 py-4">
                                                    <span className={`font-mono text-[10px] ${t.latency_ms > 200 ? "text-orange-400" : "text-slate-500"}`}>
                                                        {t.latency_ms}ms
                                                    </span>
                                                </td>
                                                <td className="px-6 py-4 text-right space-x-1">
                                                    {Object.entries(t.checks_summary?.input || {}).map(([k, v]) => !!v && (
                                                        <span key={k} className="text-[8px] bg-red-500/10 text-red-400 px-1.5 py-0.5 rounded uppercase font-black">In:{k}</span>
                                                    ))}
                                                    {Object.entries(t.checks_summary?.output || {}).map(([k, v]) => !!v && (
                                                        <span key={k} className="text-[8px] bg-orange-500/10 text-orange-400 px-1.5 py-0.5 rounded uppercase font-black">Out:{k}</span>
                                                    ))}
                                                    {(!t.checks_summary || (!Object.values(t.checks_summary.input || {}).some(v => v) && !Object.values(t.checks_summary.output || {}).some(v => v))) && (
                                                        <span className="text-[8px] text-slate-700 uppercase font-black">None</span>
                                                    )}
                                                </td>
                                            </tr>
                                        ))}
                                        {traces.length === 0 && (
                                            <tr>
                                                <td colSpan={5} className="px-6 py-12 text-center text-slate-600 italic">No traces recorded yet</td>
                                            </tr>
                                        )}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            ) : (
                <div className="flex flex-col items-center py-32 text-center gap-6">
                    <div className="p-6 rounded-full bg-white/5 border border-white/10">
                        <Shield className="w-16 h-16 text-slate-800" />
                    </div>
                    <div className="space-y-2">
                        <p className="text-slate-500 text-lg font-bold">No Guardrail Configured</p>
                        <p className="text-slate-700 text-sm max-w-md mx-auto">
                            Select a model above to automatically initialize a real-time guardrail proxy for it.
                        </p>
                    </div>
                </div>
            )}
        </div>
    );
}
