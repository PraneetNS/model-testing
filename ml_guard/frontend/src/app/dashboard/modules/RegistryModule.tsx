"use client";
import React, { useState, useEffect } from "react";
import { Package, ShieldCheck, ChevronRight, Activity, Clock, User, HardDrive } from "lucide-react";
import { apiFetch } from "@/lib/api";

const Card = ({ children, className = "" }: any) => (
    <div className={`bg-[#0E1014] border border-white/[0.07] rounded-2xl ${className}`}>{children}</div>
);

export default function ModelRegistryModule({ state, setState, onAction }: any) {
    const [models, setModels] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const [selectedModel, setSelectedModel] = useState<any>(null);
    const [versions, setVersions] = useState<any[]>([]);
    const [showRegister, setShowRegister] = useState(false);
    const [newModel, setNewModel] = useState({ name: "", provider: "", description: "" });
    const [registering, setRegistering] = useState(false);

    const fetchModels = async () => {
        setLoading(true);
        try {
            const res = await apiFetch(`/api/v1/models`);
            const d = await res.json();
            setModels(d.items || []);
        } catch (e) { } finally { setLoading(false); }
    };

    const fetchVersions = async (modelId: string) => {
        try {
            const res = await apiFetch(`/api/v1/models/${modelId}/versions`);
            const d = await res.json();
            setVersions(d.versions || []);
        } catch (e) { }
    };

    const handleRegister = async (e: React.FormEvent) => {
        e.preventDefault();
        setRegistering(true);
        try {
            const res = await apiFetch(`/api/v1/models/register?model_name=${encodeURIComponent(newModel.name)}&owner=${encodeURIComponent(newModel.provider)}&description=${encodeURIComponent(newModel.description)}`, {
                method: "POST"
            });
            if (res.ok) {
                setShowRegister(false);
                setNewModel({ name: "", provider: "", description: "" });
                fetchModels();
            }
        } catch (e) { } finally { setRegistering(false); }
    };

    const handleDeploy = async (versionId: string) => {
        try {
            const res = await apiFetch(`/api/v1/deployments/promote?version_id=${versionId}&target_environment=DEV`, { method: "POST" });
            const d = await res.json();
            if (!res.ok) throw new Error(d.detail || "Promotion failed");
            alert("Model version promoted to DEV environment");
            if (selectedModel) fetchVersions(selectedModel.model_id);
        } catch (e: any) { alert(e.message); }
    };

    useEffect(() => { fetchModels(); }, []);

    if (loading && models.length === 0) return (
        <div className="flex flex-col items-center justify-center py-32 gap-5">
            <div className="w-14 h-14 rounded-full border-2 border-orange-500/20 border-t-orange-500 animate-spin" />
            <p className="text-[10px] uppercase tracking-[0.4em] font-black text-slate-600">Syncing Registry...</p>
        </div>
    );

    return (
        <div className="grid grid-cols-1 xl:grid-cols-[1fr_400px] gap-8">
            <div className="space-y-4">
                <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-black uppercase tracking-widest text-slate-300">Registered Models</h3>
                    <div className="flex items-center gap-4">
                        <button onClick={() => setShowRegister(true)} className="px-4 py-1.5 bg-orange-600 hover:bg-orange-500 text-white rounded-lg text-[10px] uppercase font-black tracking-widest transition-all">Register Model</button>
                        <button onClick={fetchModels} className="text-[10px] uppercase font-black text-slate-500 hover:text-white transition-all">↻ Sync</button>
                    </div>
                </div>

                {showRegister && (
                    <Card className="p-6 mb-6 border-orange-500/20 bg-orange-500/[0.02]">
                        <div className="flex items-center justify-between mb-4">
                            <h4 className="text-[10px] font-black uppercase tracking-widest text-orange-400">Register New Model</h4>
                            <button onClick={() => setShowRegister(false)} className="text-slate-600 hover:text-white">✕</button>
                        </div>
                        <form onSubmit={handleRegister} className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <input value={newModel.name} onChange={e => setNewModel({ ...newModel, name: e.target.value })} placeholder="Model Name (e.g. SalesForecaster)" className="bg-black/40 border border-white/5 rounded-xl px-4 py-3 text-xs text-white focus:border-orange-500/40 outline-none" required />
                            <input value={newModel.provider} onChange={e => setNewModel({ ...newModel, provider: e.target.value })} placeholder="Owner / Team" className="bg-black/40 border border-white/5 rounded-xl px-4 py-3 text-xs text-white focus:border-orange-500/40 outline-none" required />
                            <input value={newModel.description} onChange={e => setNewModel({ ...newModel, description: e.target.value })} placeholder="Description (Optional)" className="bg-black/40 border border-white/5 rounded-xl px-4 py-3 text-xs text-white focus:border-orange-500/40 outline-none" />
                            <button type="submit" disabled={registering} className="md:col-span-3 bg-orange-600 text-white font-black py-3 rounded-xl text-[10px] uppercase tracking-widest disabled:opacity-50">
                                {registering ? "Registering..." : "Confirm Registration"}
                            </button>
                        </form>
                    </Card>
                )}

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {models.map(m => (
                        <div key={m.model_id || m.id} onClick={() => { setSelectedModel(m); fetchVersions(m.model_id || m.id); }}
                            className={`p-5 rounded-2xl border cursor-pointer transition-all ${selectedModel?.model_id === (m.model_id || m.id) ? "border-orange-500/40 bg-orange-500/5 shadow-lg shadow-orange-500/5" : "border-white/5 bg-[#0E1014] hover:border-white/10"}`}>
                            <div className="flex items-start justify-between mb-3">
                                <div className="p-2.5 rounded-xl bg-white/[0.03] border border-white/5"><Package className="w-5 h-5 text-slate-400" /></div>
                                <div className="text-right">
                                    <span className={`text-[9px] font-black uppercase px-2 py-0.5 rounded border ${m.latest_governance_score >= 70 ? "text-emerald-400 border-emerald-500/20 bg-emerald-500/5" : "text-red-400 border-red-500/20 bg-red-500/5"}`}>
                                        Score: {m.latest_governance_score ?? "N/A"}
                                    </span>
                                </div>
                            </div>
                            <h4 className="text-base font-black text-white">{m.name}</h4>
                            <p className="text-[10px] font-bold text-slate-600 uppercase tracking-widest mt-1">{m.provider} • v{m.latest_version}</p>

                            <div className="mt-5 pt-4 border-t border-white/[0.03] flex items-center justify-between">
                                <div className="flex items-center gap-4">
                                    <div className="flex flex-col"><span className="text-[8px] font-black text-slate-600 uppercase">Risk</span><span className={`text-[10px] font-black ${m.latest_risk_class === "CRITICAL" ? "text-red-400" : "text-emerald-400"}`}>{m.latest_risk_class || "LOW"}</span></div>
                                    <div className="flex flex-col"><span className="text-[8px] font-black text-slate-600 uppercase">Versions</span><span className="text-[10px] font-black text-slate-300">{m.version_count}</span></div>
                                </div>
                                <ChevronRight className="w-4 h-4 text-slate-700" />
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            <div className="space-y-6">
                {selectedModel ? (
                    <>
                        <div className="p-6 rounded-2xl border border-white/5 bg-[#0E1014] space-y-5">
                            <h3 className="text-xs font-black uppercase tracking-widest text-slate-400">Model Details</h3>
                            <div className="space-y-4">
                                <div className="flex items-center gap-4">
                                    <div className="w-10 h-10 rounded-xl bg-orange-500/10 flex items-center justify-center"><Package className="w-5 h-5 text-orange-400" /></div>
                                    <div><p className="text-lg font-black text-white leading-tight">{selectedModel.name}</p><p className="text-[9px] font-black uppercase tracking-widest text-slate-600">{selectedModel.model_id}</p></div>
                                </div>
                                <div className="grid grid-cols-2 gap-3">
                                    <div className="bg-white/[0.02] p-3 rounded-xl border border-white/[0.03]"><p className="text-[8px] uppercase font-black text-slate-700 mb-1">Provider</p><p className="text-xs font-bold text-slate-300">{selectedModel.provider}</p></div>
                                    <div className="bg-white/[0.02] p-3 rounded-xl border border-white/[0.03]"><p className="text-[8px] uppercase font-black text-slate-700 mb-1">Created</p><p className="text-xs font-bold text-slate-300">{new Date(selectedModel.created_at).toLocaleDateString()}</p></div>
                                </div>
                            </div>
                        </div>

                        <div className="space-y-3">
                            <h3 className="text-[10px] font-black uppercase tracking-widest text-slate-600 ml-1">Version History</h3>
                            {versions.length === 0 ? <p className="text-xs text-slate-700 text-center py-10 italic">No versions detected</p> :
                                versions.map(v => (
                                    <div key={v.version_id} className="p-4 rounded-xl border border-white/5 bg-black/20 flex items-center justify-between group transition-all hover:border-white/10">
                                        <div className="flex items-center gap-4">
                                            <div className="flex flex-col items-center justify-center w-8 h-8 rounded-lg bg-white/[0.04] text-[10px] font-black text-slate-500">v{v.version_number}</div>
                                            <div>
                                                <p className="text-xs font-black text-white">{v.framework || "SCIKIT-LEARN"}</p>
                                                <div className="flex items-center gap-3 mt-0.5">
                                                    <span className={`text-[8px] font-black uppercase ${v.governance_score >= 70 ? 'text-emerald-500' : 'text-red-400'}`}>Score: {v.governance_score || "N/A"}</span>
                                                    <span className="text-[8px] font-black text-slate-700 uppercase">{v.risk_class || "LOW"} Risk</span>
                                                </div>
                                            </div>
                                        </div>
                                        <div className="text-right">
                                            {v.deployments?.length > 0 ? (
                                                <span className="text-[8px] font-black uppercase px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">Active in {v.deployments[0].environment}</span>
                                            ) : (
                                                <button onClick={() => handleDeploy(v.version_id)} className="text-[8px] font-black uppercase px-2 py-0.5 rounded-full bg-orange-500 hover:bg-orange-400 text-black border border-orange-600/20 transition-all">Deploy to DEV</button>
                                            )}
                                        </div>
                                    </div>
                                ))
                            }
                        </div>
                    </>
                ) : (
                    <div className="flex flex-col items-center justify-center py-24 text-center border-2 border-dashed border-white/5 rounded-3xl">
                        <Package className="w-12 h-12 text-slate-800 mb-4" />
                        <p className="text-[10px] font-black uppercase tracking-widest text-slate-700">Select a model to view full lineage & versions</p>
                    </div>
                )}
            </div>
        </div>
    );
}
