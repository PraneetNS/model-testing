"use client";
import React, { useState, useEffect } from "react";
import { apiFetch, safeJson } from "@/lib/api";
import { Play, ShieldCheck, CheckCircle2, AlertCircle, Save, Settings, FileText, Bell, RefreshCw } from "lucide-react";

export default function RetrainingModule({ modelId = "" }) {
    const [selectedModel, setSelectedModel] = useState(modelId);
    const [availableModels, setAvailableModels] = useState<any[]>([]);
    
    const [policy, setPolicy] = useState<any>(null);
    const [events, setEvents] = useState<any[]>([]);
    
    const [enabled, setEnabled] = useState(false);
    const [psiThreshold, setPsiThreshold] = useState(0.2);
    const [ksThreshold, setKsThreshold] = useState(0.1);
    const [degPct, setDegPct] = useState(15);
    const [minDays, setMinDays] = useState(7);
    const [requireAll, setRequireAll] = useState(false);
    
    const [actionType, setActionType] = useState("notify_only");
    const [webhookUrl, setWebhookUrl] = useState("");
    const [githubRepo, setGithubRepo] = useState("");
    const [githubWf, setGithubWf] = useState("");
    const [githubToken, setGithubToken] = useState("");
    
    const [loading, setLoading] = useState(false);
    const [saving, setSaving] = useState(false);
    const [simulateResult, setSimulateResult] = useState<any>(null);

    useEffect(() => {
        apiFetch("/api/inventory")
            .then(res => safeJson<any[]>(res))
            .then(data => {
                if (Array.isArray(data)) setAvailableModels(data);
            })
            .catch(console.error);
    }, []);
    
    useEffect(() => {
        if (selectedModel) {
            loadPolicy();
            loadEvents();
        } else {
            setPolicy(null);
            setEvents([]);
        }
    }, [selectedModel]);

    const loadPolicy = async () => {
        setLoading(true);
        try {
            const res = await apiFetch(`/api/v1/models/${selectedModel}/retraining-policy`);
            if (res.ok) {
                const data = await safeJson(res);
                setPolicy(data);
                setEnabled(data.enabled);
                setPsiThreshold(data.trigger_conditions?.psi_threshold || 0.2);
                setKsThreshold(data.trigger_conditions?.ks_stat_threshold || 0.1);
                setDegPct(data.trigger_conditions?.performance_degradation_pct || 15);
                setMinDays(data.trigger_conditions?.min_days_since_last_retrain || 7);
                setRequireAll(data.trigger_conditions?.require_all_conditions || false);
                
                setActionType(data.retrain_action?.action_type || "notify_only");
                setWebhookUrl(data.retrain_action?.webhook_url || "");
                setGithubRepo(data.retrain_action?.github_repo || "");
                setGithubWf(data.retrain_action?.github_workflow_file || "");
                // Token isn't sent back usually, leave blank unless provided
            } else {
                setPolicy(null);
                setEnabled(false);
            }
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    };
    
    const loadEvents = async () => {
        try {
            const res = await apiFetch(`/api/v1/models/${selectedModel}/retraining-events`);
            if (res.ok) {
                setEvents(await safeJson(res));
            }
        } catch (e) {
            console.error(e);
        }
    };

    const savePolicy = async () => {
        setSaving(true);
        try {
            const payload = {
                enabled,
                trigger_conditions: {
                    psi_threshold: psiThreshold,
                    ks_stat_threshold: ksThreshold,
                    performance_degradation_pct: degPct,
                    min_days_since_last_retrain: minDays,
                    require_all_conditions: requireAll
                },
                retrain_action: {
                    action_type: actionType,
                    webhook_url: webhookUrl,
                    github_repo: githubRepo,
                    github_workflow_file: githubWf,
                    github_token_encrypted: githubToken || undefined
                }
            };
            
            await apiFetch(`/api/v1/models/${selectedModel}/retraining-policy`, {
                method: "POST",
                body: JSON.stringify(payload)
            });
            await loadPolicy();
        } catch (e) {
            console.error(e);
        } finally {
            setSaving(false);
        }
    };

    const simulateTrigger = async () => {
        try {
            const res = await apiFetch(`/api/v1/models/${selectedModel}/retraining-policy/simulate`, { method: "POST" });
            if (res.ok) {
                setSimulateResult(await safeJson(res));
            }
        } catch (e) {
            console.error(e);
        }
    };
    
    const triggerNow = async () => {
        try {
            await apiFetch(`/api/v1/models/${selectedModel}/retraining-policy/trigger-now`, { method: "POST" });
            await loadEvents();
        } catch (e) {
            console.error(e);
        }
    };

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-xl font-black text-white">Automated Retraining</h2>
                    <p className="text-sm text-slate-400">Configure triggers and dispatches to close the lineage loop.</p>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-[400px_1fr] gap-6">
                <div className="space-y-4">
                    <div className="bg-[#0E1014] border border-white/[0.07] rounded-xl p-5 space-y-4">
                        <div>
                            <label className="text-[10px] font-black uppercase text-slate-500 mb-1 block">Select Model</label>
                            <select
                                value={selectedModel}
                                onChange={(e) => setSelectedModel(e.target.value)}
                                className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-orange-500/50"
                            >
                                <option value="">-- Choose Model --</option>
                                {availableModels.map(m => (
                                    <option key={m.id} value={m.id}>{m.name} (Tier: {m.risk_tier || 'N/A'})</option>
                                ))}
                            </select>
                        </div>
                        
                        {selectedModel && (
                            <>
                                <div className="flex items-center justify-between py-3 border-y border-white/5">
                                    <div>
                                        <p className="text-sm font-bold text-white">Enable Automated Retraining</p>
                                        <p className="text-[10px] text-slate-500">Evaluates triggers every hour</p>
                                    </div>
                                    <label className="relative inline-flex items-center cursor-pointer">
                                        <input type="checkbox" checked={enabled} onChange={e => setEnabled(e.target.checked)} className="sr-only peer" />
                                        <div className="w-11 h-6 bg-slate-800 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-emerald-500"></div>
                                    </label>
                                </div>
                                
                                <div className="space-y-3">
                                    <p className="text-[10px] font-black uppercase text-slate-500 tracking-widest">Trigger Conditions</p>
                                    
                                    <div>
                                        <div className="flex justify-between text-xs mb-1">
                                            <span className="text-slate-300">PSI Threshold</span>
                                            <span className="text-slate-400 font-mono">{psiThreshold}</span>
                                        </div>
                                        <input type="range" min="0" max="1" step="0.05" value={psiThreshold} onChange={e => setPsiThreshold(parseFloat(e.target.value))} className="w-full" />
                                    </div>
                                    <div>
                                        <div className="flex justify-between text-xs mb-1">
                                            <span className="text-slate-300">KS Stat Threshold</span>
                                            <span className="text-slate-400 font-mono">{ksThreshold}</span>
                                        </div>
                                        <input type="range" min="0" max="1" step="0.05" value={ksThreshold} onChange={e => setKsThreshold(parseFloat(e.target.value))} className="w-full" />
                                    </div>
                                    <div>
                                        <div className="flex justify-between text-xs mb-1">
                                            <span className="text-slate-300">Performance Degradation %</span>
                                            <span className="text-slate-400 font-mono">{degPct}%</span>
                                        </div>
                                        <input type="range" min="1" max="50" step="1" value={degPct} onChange={e => setDegPct(parseFloat(e.target.value))} className="w-full" />
                                    </div>
                                    <div>
                                        <label className="text-xs text-slate-300 mb-1 block">Min Days Since Last Retrain</label>
                                        <input type="number" min="0" value={minDays} onChange={e => setMinDays(parseInt(e.target.value))} className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-1.5 text-sm text-white" />
                                    </div>
                                    <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                                        <input type="checkbox" checked={requireAll} onChange={e => setRequireAll(e.target.checked)} className="rounded border-white/10 bg-black/40" />
                                        Require ALL conditions to be met (AND logic instead of OR)
                                    </label>
                                </div>
                                
                                <div className="space-y-3 pt-3 border-t border-white/5">
                                    <p className="text-[10px] font-black uppercase text-slate-500 tracking-widest">Dispatch Action</p>
                                    <select value={actionType} onChange={e => setActionType(e.target.value)} className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-orange-500/50">
                                        <option value="notify_only">Notify Only</option>
                                        <option value="webhook">Webhook</option>
                                        <option value="github_actions">GitHub Actions</option>
                                        <option value="mlflow_run">MLflow Run</option>
                                    </select>
                                    
                                    {actionType === "webhook" && (
                                        <input type="url" placeholder="https://..." value={webhookUrl} onChange={e => setWebhookUrl(e.target.value)} className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2 text-sm text-white" />
                                    )}
                                    {actionType === "github_actions" && (
                                        <div className="space-y-2">
                                            <input type="text" placeholder="owner/repo" value={githubRepo} onChange={e => setGithubRepo(e.target.value)} className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2 text-sm text-white" />
                                            <input type="text" placeholder="workflow_file.yml" value={githubWf} onChange={e => setGithubWf(e.target.value)} className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2 text-sm text-white" />
                                            <input type="password" placeholder="GitHub Token" value={githubToken} onChange={e => setGithubToken(e.target.value)} className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2 text-sm text-white" />
                                        </div>
                                    )}
                                </div>
                                
                                <button onClick={savePolicy} disabled={saving} className="w-full py-3 bg-white/10 hover:bg-white/20 text-white font-black uppercase tracking-widest text-[10px] rounded-lg transition-all flex justify-center items-center gap-2">
                                    <Save className="w-4 h-4" /> Save Policy
                                </button>
                                
                                <div className="grid grid-cols-2 gap-2 pt-2">
                                    <button onClick={simulateTrigger} className="py-2.5 bg-blue-600/20 text-blue-400 hover:bg-blue-600/30 font-bold text-xs rounded-lg transition-all flex justify-center items-center gap-2">
                                        <Settings className="w-3.5 h-3.5" /> Simulate
                                    </button>
                                    <button onClick={triggerNow} className="py-2.5 bg-red-600/20 text-red-400 hover:bg-red-600/30 font-bold text-xs rounded-lg transition-all flex justify-center items-center gap-2">
                                        <Play className="w-3.5 h-3.5" /> Force Trigger
                                    </button>
                                </div>
                            </>
                        )}
                    </div>
                </div>

                <div className="space-y-6">
                    {simulateResult && (
                        <div className={`p-4 rounded-xl border ${simulateResult.should_trigger ? "border-amber-500/30 bg-amber-500/5" : "border-emerald-500/30 bg-emerald-500/5"}`}>
                            <div className="flex items-center gap-3 mb-2">
                                {simulateResult.should_trigger ? <AlertCircle className="w-5 h-5 text-amber-400" /> : <CheckCircle2 className="w-5 h-5 text-emerald-400" />}
                                <h3 className="font-bold text-white">Simulation Result: {simulateResult.should_trigger ? "Would Trigger" : "Would Not Trigger"}</h3>
                            </div>
                            {simulateResult.suppressed && (
                                <p className="text-xs text-blue-400 mb-2 font-medium">{simulateResult.suppression_reason}</p>
                            )}
                            <div className="text-xs text-slate-400 space-y-1">
                                {simulateResult.triggered_conditions?.map((c: string, i: number) => (
                                    <div key={i} className="flex items-center gap-2">
                                        <div className="w-1 h-1 rounded-full bg-slate-500" /> {c}
                                    </div>
                                ))}
                                {simulateResult.triggered_conditions?.length === 0 && <p>No conditions breached.</p>}
                            </div>
                        </div>
                    )}
                    
                    <div className="bg-[#0E1014] border border-white/[0.07] rounded-xl overflow-hidden">
                        <div className="p-4 border-b border-white/5 flex items-center justify-between">
                            <h3 className="text-sm font-bold text-white flex items-center gap-2"><RefreshCw className="w-4 h-4 text-slate-400" /> Event History</h3>
                            <button onClick={loadEvents} className="text-xs text-slate-500 hover:text-white">Refresh</button>
                        </div>
                        {events.length > 0 ? (
                            <table className="w-full text-left text-xs text-slate-300">
                                <thead>
                                    <tr className="border-b border-white/5 bg-black/20 text-slate-500 uppercase font-black text-[9px] tracking-wider">
                                        <th className="p-4">Time</th>
                                        <th className="p-4">Action</th>
                                        <th className="p-4">Result</th>
                                        <th className="p-4">Conditions Met</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {events.map(e => (
                                        <tr key={e.id} className="border-b border-white/5 hover:bg-white/[0.02]">
                                            <td className="p-4 whitespace-nowrap">{new Date(e.triggered_at).toLocaleString()}</td>
                                            <td className="p-4 font-mono">{e.action_type}</td>
                                            <td className="p-4">
                                                <span className={`px-2 py-0.5 rounded text-[9px] font-black uppercase ${e.action_result === "success" ? "bg-emerald-500/20 text-emerald-400" : "bg-red-500/20 text-red-400"}`}>
                                                    {e.action_result}
                                                </span>
                                            </td>
                                            <td className="p-4">
                                                {e.triggered_conditions?.join(", ")}
                                                {e.action_error && <p className="text-red-400 text-[10px] mt-1">{e.action_error}</p>}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        ) : (
                            <div className="p-8 text-center text-slate-500 text-sm">No trigger events found.</div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
