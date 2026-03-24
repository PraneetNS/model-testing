"use client";
import React, { useState, useEffect } from "react";
import { GitBranch, GitPullRequest, ShieldCheck, Activity, Terminal, ExternalLink, Play, CheckCircle2, AlertTriangle, AlertCircle, Loader2, Key, Copy, Github, Sliders, Info, Server } from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export default function CIModule({ state, setState, onAction }: any) {
    const [integrations, setIntegrations] = useState<any[]>([]);
    const [apiKeys, setApiKeys] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const [selectedIntegration, setSelectedIntegration] = useState<any>(null);
    const [activeTab, setActiveTab] = useState("pipelines"); // "pipelines" | "keys" | "setup"

    const fetchData = async () => {
        setLoading(true);
        try {
            const [intRes, keysRes] = await Promise.all([
                fetch(`${API_BASE}/api/v1/ci/integrations`),
                fetch(`${API_BASE}/api/v1/auth/apikeys`)
            ]);
            setIntegrations(await intRes.json());
            setApiKeys(await keysRes.json());
        } catch (e) { } finally { setLoading(false); }
    };

    useEffect(() => { fetchData(); }, []);

    const [newKeyLabel, setNewKeyLabel] = useState("CI/CD Pipeline Key");
    const [generatedKey, setGeneratedKey] = useState<string | null>(null);
    const [creatingKey, setCreatingKey] = useState(false);

    const createKey = async () => {
        setCreatingKey(true);
        try {
            const res = await fetch(`${API_BASE}/api/v1/auth/apikey?label=${newKeyLabel}`, { method: "POST" });
            const d = await res.json();
            setGeneratedKey(d.api_key);
            fetchData();
        } catch (e) { } finally { setCreatingKey(false); }
    };

    const [testModel, setTestModel] = useState("CreditRiskPredictor");
    const [testScore, setTestScore] = useState(85);
    const [testResult, setTestResult] = useState<any>(null);
    const [testing, setTesting] = useState(false);

    const triggerTest = async () => {
        setTesting(true); setTestResult(null);
        try {
            const res = await fetch(`${API_BASE}/api/v1/ci/audit?model_name=${testModel}&governance_score_override=${testScore}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    pipeline_metadata: { repo: "ml-guard-demo", branch: "main", commit: "a1b2c3d4" }
                })
            });
            const d = await res.json();
            setTestResult(d);
        } catch (e) { } finally { setTesting(false); }
    };

    return (
        <div className="space-y-8 animate-in fade-in duration-700">
            {/* ─── Header Navigation ─── */}
            <div className="flex items-center justify-between px-2">
                <div className="flex gap-1 bg-black/40 p-1 rounded-2xl border border-white/5">
                    {[
                        { id: "pipelines", label: "Pipelines", icon: GitBranch },
                        { id: "keys", label: "API Keys", icon: Key },
                        { id: "setup", label: "Setup Guide", icon: Info }
                    ].map(t => (
                        <button key={t.id} onClick={() => setActiveTab(t.id)}
                            className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all ${activeTab === t.id ? "bg-orange-600 text-black shadow-lg shadow-orange-500/20" : "text-slate-500 hover:text-white"}`}>
                            <t.icon className="w-3.5 h-3.5" />
                            {t.label}
                        </button>
                    ))}
                </div>
                <button onClick={fetchData} className="text-[10px] uppercase font-black text-slate-700 hover:text-orange-400 transition-all">↻ Sync System</button>
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-[1fr_420px] gap-8">
                <div className="space-y-6">
                    {activeTab === "pipelines" && (
                        <div className="space-y-6">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                {integrations.map(int => (
                                    <div key={int.id} onClick={() => setSelectedIntegration(int)}
                                        className={`p-6 rounded-3xl border transition-all cursor-pointer group ${selectedIntegration?.id === int.id ? "border-orange-500/40 bg-orange-500/5 shadow-2xl shadow-orange-500/5" : "border-white/5 bg-[#0E1014] hover:border-white/10"}`}>
                                        <div className="flex items-start justify-between mb-5">
                                            <div className="p-3 rounded-2xl bg-white/[0.03] border border-white/5 group-hover:scale-110 transition-transform">
                                                {int.provider === 'GITHUB' ? <Github className="w-5 h-5 text-slate-400" /> : <Server className="w-5 h-5 text-slate-400" />}
                                            </div>
                                            <span className={`text-[9px] font-black uppercase px-2.5 py-1 rounded-lg border ${int.is_active ? "text-emerald-400 border-emerald-500/20 bg-emerald-500/5 shadow-[0_0_15px_rgba(16,185,129,0.1)]" : "text-slate-600 border-white/5 bg-white/[0.02]"}`}>
                                                {int.is_active ? "Active" : "Disabled"}
                                            </span>
                                        </div>
                                        <h4 className="text-lg font-black text-white tracking-tight">{int.repo_name}</h4>
                                        <p className="text-[10px] font-bold text-slate-600 uppercase tracking-widest mt-1.5 flex items-center gap-2">
                                            <span className="w-1.5 h-1.5 rounded-full bg-orange-500" />
                                            {int.provider} • {int.branch_pattern || "main"}
                                        </p>

                                        <div className="mt-8 pt-6 border-t border-white/[0.03] flex items-center justify-between">
                                            <div className="flex flex-col"><span className="text-[8px] font-black text-slate-700 uppercase tracking-[0.2em] mb-1">Latest Compliance</span><span className="text-[11px] font-black text-emerald-400">PASSED · 92/100</span></div>
                                            <div className="flex flex-col text-right"><span className="text-[8px] font-black text-slate-700 uppercase tracking-[0.2em] mb-1">Last Run</span><span className="text-[11px] font-black text-slate-500">{int.last_run_at ? new Date(int.last_run_at).toLocaleDateString() : "Never"}</span></div>
                                        </div>
                                    </div>
                                ))}
                                {integrations.length === 0 && <p className="text-center py-24 text-slate-700 text-[10px] uppercase font-black tracking-widest border-2 border-dashed border-white/5 rounded-3xl col-span-2">No active integrations found</p>}
                            </div>

                            <div className="p-10 rounded-[32px] border border-white/5 bg-black/40 space-y-8 shadow-2xl relative overflow-hidden group">
                                <div className="absolute top-0 right-0 w-64 h-64 bg-orange-500/5 blur-[100px] -mr-32 -mt-32 pointer-events-none group-hover:bg-orange-500/10 transition-colors" />
                                <div className="flex items-center gap-4">
                                    <div className="p-3 rounded-2xl bg-orange-500/10 border border-orange-500/20"><Terminal className="w-6 h-6 text-orange-500" /></div>
                                    <div>
                                        <h3 className="text-lg font-black uppercase tracking-tighter text-white">Pipeline Gate Simulator</h3>
                                        <p className="text-xs font-medium text-slate-600">Mirror the behavior of a CI environment governance audit.</p>
                                    </div>
                                </div>
                                <div className="grid grid-cols-2 gap-6">
                                    <div className="space-y-3"><p className="text-[9px] font-black text-slate-600 uppercase tracking-widest px-1">Mock Target Model</p><input value={testModel} onChange={e => setTestModel(e.target.value)} className="w-full bg-black/60 border border-white/5 rounded-2xl px-5 py-4 text-xs font-bold text-white focus:border-orange-500/40 outline-none transition-all" /></div>
                                    <div className="space-y-3"><p className="text-[9px] font-black text-slate-600 uppercase tracking-widest px-1">Governance Score %</p><input type="number" value={testScore} onChange={e => setTestScore(Number(e.target.value))} className="w-full bg-black/60 border border-white/5 rounded-2xl px-5 py-4 text-xs font-bold text-white focus:border-orange-500/40 outline-none transition-all" /></div>
                                </div>
                                <button onClick={triggerTest} disabled={testing} className="w-full bg-white/5 hover:bg-orange-600 hover:text-black hover:shadow-2xl hover:shadow-orange-500/20 disabled:opacity-50 text-slate-300 font-black py-5 rounded-[20px] text-[11px] uppercase tracking-[0.2em] flex items-center justify-center gap-3 transition-all">
                                    {testing ? <><Loader2 className="w-5 h-5 animate-spin" />Running Pipeline Analysis...</> : <><Play className="w-5 h-5 fill-current" />Simulate Governance Gate</>}
                                </button>

                                {testResult && (
                                    <div className={`p-8 rounded-3xl border bg-black/80 flex items-start gap-6 shadow-2xl animate-in zoom-in-95 duration-500 ${testResult.deployment_allowed ? "border-emerald-500/20" : "border-red-500/20"}`}>
                                        <div className={`p-4 rounded-2xl ${testResult.deployment_allowed ? "bg-emerald-500/10 text-emerald-400" : "bg-red-500/10 text-red-500"}`}>
                                            {testResult.deployment_allowed ? <CheckCircle2 className="w-8 h-8" /> : <ShieldCheck className="w-8 h-8" />}
                                        </div>
                                        <div className="flex-1">
                                            <div className="flex items-center justify-between">
                                                <h4 className={`text-xl font-black ${testResult.deployment_allowed ? "text-emerald-400" : "text-red-400"}`}>{testResult.deployment_allowed ? "Deployment Approved" : "Deployment Blocked"}</h4>
                                                <span className="text-[10px] font-mono text-slate-700">EXIT CODE: {testResult.deployment_allowed ? "0" : "1"}</span>
                                            </div>
                                            <p className="text-[13px] font-medium text-slate-500 mt-2 leading-relaxed">{testResult.message}</p>
                                            <div className="mt-8 grid grid-cols-2 gap-8 border-t border-white/5 pt-6">
                                                <div><p className="text-[9px] font-black text-slate-700 uppercase tracking-[0.2em] mb-1">Score Result</p><p className={`text-3xl font-black ${testResult.governance_score >= 70 ? "text-white" : "text-red-400"}`}>{testResult.governance_score}%</p></div>
                                                <div><p className="text-[9px] font-black text-slate-700 uppercase tracking-[0.2em] mb-1">Risk Classification</p><p className={`text-3xl font-black ${testResult.risk_level === 'CRITICAL' ? "text-red-400" : testResult.risk_level === 'MEDIUM' ? "text-orange-400" : "text-emerald-400"}`}>{testResult.risk_level}</p></div>
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}

                    {activeTab === "keys" && (
                        <div className="space-y-6">
                            <div className="p-8 rounded-3xl border border-white/5 bg-[#0E1014] space-y-6">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-3"><Key className="w-5 h-5 text-orange-500" /><h3 className="text-sm font-black uppercase tracking-widest text-white">API Credentials</h3></div>
                                    <p className="text-[10px] font-bold text-slate-700 uppercase">Service Accounts</p>
                                </div>

                                <div className="space-y-2">
                                    <p className="text-[9px] font-black text-slate-600 uppercase tracking-widest px-1">Create New Key</p>
                                    <div className="flex gap-3">
                                        <input value={newKeyLabel} onChange={e => setNewKeyLabel(e.target.value)} placeholder="Key label (e.g., Jenkins Lead)" className="flex-1 bg-black/60 border border-white/5 rounded-2xl px-5 py-4 text-xs font-bold text-white focus:border-orange-500/40 outline-none" />
                                        <button onClick={createKey} disabled={creatingKey} className="bg-orange-600 hover:bg-orange-500 text-black px-8 rounded-2xl text-[10px] font-black uppercase tracking-widest disabled:opacity-50 transition-all">
                                            {creatingKey ? "Generating..." : "Generate"}
                                        </button>
                                    </div>
                                </div>

                                {generatedKey && (
                                    <div className="p-5 rounded-2xl bg-orange-500/5 border border-orange-500/20 space-y-3 animate-in fade-in slide-in-from-top-4 duration-500">
                                        <div className="flex items-center justify-between"><p className="text-[10px] font-black text-orange-400 uppercase">New Key Generated</p><AlertTriangle className="w-4 h-4 text-orange-500" /></div>
                                        <div className="flex gap-3 bg-black/40 p-4 rounded-xl border border-white/5">
                                            <code className="flex-1 text-xs font-mono text-white/80 select-all truncate">{generatedKey}</code>
                                            <button onClick={() => navigator.clipboard.writeText(generatedKey)} className="text-orange-400 hover:text-white transition-colors"><Copy className="w-4 h-4" /></button>
                                        </div>
                                        <p className="text-[9px] text-orange-500/60 font-medium">⚠️ Copy this now. For security reasons, it cannot be displayed later.</p>
                                    </div>
                                )}
                            </div>

                            <div className="space-y-3">
                                <h3 className="text-[10px] font-black text-slate-700 uppercase tracking-[0.2em] px-2">Active API Keys</h3>
                                <div className="grid grid-cols-1 gap-3">
                                    {apiKeys.map(k => (
                                        <div key={k.id} className="p-5 rounded-2xl border border-white/5 bg-[#0E1014] flex items-center justify-between">
                                            <div className="flex items-center gap-4">
                                                <div className="w-10 h-10 rounded-xl bg-white/[0.03] flex items-center justify-center text-slate-600"><Key className="w-5 h-5" /></div>
                                                <div><p className="text-xs font-black text-white">{k.label}</p><p className="text-[9px] font-bold text-slate-700 uppercase tracking-widest mt-0.5">Created {new Date(k.created_at).toLocaleDateString()}</p></div>
                                            </div>
                                            <div className="text-right"><p className="text-[8px] font-black text-slate-800 uppercase tracking-widest mb-1">Last Used</p><p className="text-[10px] font-black text-slate-500">{k.last_used ? new Date(k.last_used).toLocaleString() : "Never"}</p></div>
                                        </div>
                                    ))}
                                    {apiKeys.length === 0 && <p className="text-center py-12 text-[10px] font-black uppercase text-slate-800 tracking-widest">No keys configured</p>}
                                </div>
                            </div>
                        </div>
                    )}

                    {activeTab === "setup" && (
                        <div className="space-y-6">
                            <div className="p-10 rounded-[40px] border border-white/5 bg-[#0E1014] space-y-10 shadow-2xl">
                                <div className="space-y-4">
                                    <div className="flex items-center gap-4"><div className="p-3 rounded-2xl bg-blue-500/10"><Github className="w-6 h-6 text-blue-400" /></div><h4 className="text-xl font-black text-white tracking-tight">GitHub Actions Integration</h4></div>
                                    <p className="text-sm font-medium text-slate-500 leading-relaxed max-w-2xl">Integrate ML Guard governance checks into your Pull Request lifecycle. Automatically block merges if the model fails safety audits.</p>
                                    <div className="relative group">
                                        <div className="absolute top-4 right-4 flex gap-2"><button onClick={() => navigator.clipboard.writeText("Copying...")} className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-slate-500 transition-all opacity-0 group-hover:opacity-100"><Copy className="w-4 h-4" /></button></div>
                                        <pre className="bg-black/60 p-6 rounded-3xl border border-white/5 text-[11px] font-mono text-slate-400 overflow-x-auto selection:bg-orange-500/30">
                                            {`# .github/workflows/ml-guard-audit.yml
name: ML Governance Audit
on: [pull_request]

jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run ML Guard Compliance
        run: |
          pip install requests
          python .github/scripts/ml_guard_ci.py \\
            --api-url "https://api.your-org.ml-guard.io" \\
            --api-key "\${{ secrets.MLG_API_KEY }}" \\
            --model-name "CreditRiskModel" \\
            --model-path "models/latest.pkl" \\
            --data-path "data/validation.csv"`}
                                        </pre>
                                    </div>
                                </div>

                                <div className="space-y-4 pt-4 border-t border-white/5">
                                    <div className="flex items-center gap-4"><div className="p-3 rounded-2xl bg-orange-500/10"><Server className="w-6 h-6 text-orange-500" /></div><h4 className="text-xl font-black text-white tracking-tight">Jenkins Pipeline Integration</h4></div>
                                    <pre className="bg-black/60 p-6 rounded-3xl border border-white/5 text-[11px] font-mono text-slate-400 overflow-x-auto">
                                        {`// Jenkinsfile
pipeline {
    agent any
    environment {
        MLG_KEY = credentials('ml-guard-api-key')
    }
    stages {
        stage('ML Governance') {
            steps {
                sh "python ml_guard_ci.py --api-url http://api.local --api-key $MLG_KEY ..."
            }
        }
    }
}`}
                                    </pre>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                <div className="space-y-6">
                    <div className="p-8 rounded-[32px] border border-orange-500/20 bg-orange-500/[0.02] space-y-8 flex flex-col items-center text-center">
                        <div className="w-20 h-20 rounded-3xl bg-orange-500/10 flex items-center justify-center shadow-lg shadow-orange-500/5 rotate-3"><Sliders className="w-10 h-10 text-orange-500" /></div>
                        <div className="space-y-2">
                            <h3 className="text-xl font-black text-white uppercase tracking-tighter leading-tight">Fireflink Governance System</h3>
                            <p className="text-xs font-medium text-slate-500 max-w-xs leading-relaxed italic">"Trust but verify." — Automated model auditing integrated directly into the engineering loop.</p>
                        </div>

                        <div className="w-full space-y-4">
                            {[
                                { label: "Automatic Blocking", icon: ShieldCheck, sub: "Stop low-score deployments" },
                                { label: "Standardized CLI", icon: Terminal, sub: "One script for all providers" },
                                { label: "Audit Immortality", icon: Activity, sub: "Permanent scan trail in registry" }
                            ].map((v, i) => (
                                <div key={i} className="flex items-center gap-4 px-4 py-4 bg-black/40 rounded-2xl border border-white/5 text-left">
                                    <v.icon className="w-5 h-5 text-slate-700 shrink-0" />
                                    <div><p className="text-[11px] font-black text-slate-300 uppercase leading-none">{v.label}</p><p className="text-[9px] font-bold text-slate-600 uppercase mt-1">{v.sub}</p></div>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="p-8 rounded-[32px] border border-white/5 bg-[#0E1014] space-y-6">
                        <p className="text-[10px] font-black uppercase tracking-widest text-slate-700 px-1">Infrastructure Monitoring</p>
                        <div className="space-y-3">
                            {[1, 2, 3].map(i => (
                                <div key={i} className="flex items-center justify-between p-4 rounded-2xl bg-white/[0.02] border border-white/5">
                                    <div className="flex items-center gap-3"><div className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)]" /><div><p className="text-xs font-black text-white">Hub-Pipeline-{i}</p><p className="text-[9px] font-bold text-slate-700 uppercase">Latency: {12 * i}ms</p></div></div>
                                    <div className="text-[10px] font-black text-slate-500 uppercase">RUNNING</div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
