"use client";
import { apiFetch } from "@/lib/api";
import React, { useState, useEffect } from "react";
import { FileText, Database, ShieldCheck, ChevronRight, HardDrive, Filter, Clock } from "lucide-react";


const Card = ({ children, className = "" }: any) => (
    <div className={`bg-[#0E1014] border border-white/[0.07] rounded-2xl ${className}`}>{children}</div>
);

const Tile = ({ label, value, sub, accent = false }: any) => (
    <div className="bg-black/20 rounded-xl p-4 space-y-1">
        <p className="text-[9px] uppercase font-black tracking-widest text-slate-600">{label}</p>
        <p className={`text-base font-black truncate ${accent ? "text-orange-400" : "text-white"}`}>{value ?? "—"}</p>
        {sub && <p className="text-[9px] text-slate-600">{sub}</p>}
    </div>
);

export default function DatasetsModule({ state, setState, onAction }: any) {
    const [datasets, setDatasets] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const [selectedDataset, setSelectedDataset] = useState<any>(null);
    const [lineage, setLineage] = useState<any[]>([]);
    const [showRegister, setShowRegister] = useState(false);
    const [models, setModels] = useState<any[]>([]);
    const [newDataset, setNewDataset] = useState<any>({ 
        model_id: "", name: "", type: "training", source: "local",
        config: {} 
    });
    const [openmlQuery, setOpenmlQuery] = useState("");
    const [openmlResults, setOpenmlResults] = useState<any[]>([]);
    const [searchingOpenml, setSearchingOpenml] = useState(false);

    const searchOpenML = async () => {
        if (!openmlQuery) return;
        setSearchingOpenml(true);
        try {
            const res = await apiFetch(`/api/plugins/openml/search?query=${encodeURIComponent(openmlQuery)}&limit=5`);
            const d = await res.json();
            setOpenmlResults(d);
        } catch (e) { } finally { setSearchingOpenml(false); }
    };

    const handleRegister = async (e: React.FormEvent) => {
        e.preventDefault();
        setRegistering(true);
        try {
            let res;
            if (newDataset.source === "local") {
                // Legacy registration
                res = await apiFetch(`/api/v1/datasets/register`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        model_id: newDataset.model_id,
                        dataset_type: newDataset.type,
                        dataset_name: newDataset.name,
                    })
                });
            } else {
                // Plugin-based fetch
                const endpoint = newDataset.source === "huggingface" 
                    ? `/api/plugins/huggingface/fetch`
                    : `/api/plugins/${newDataset.source}/fetch`;
                
                res = await apiFetch(endpoint, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        source_uri: newDataset.source === "openml" ? `openml://${newDataset.config.dataset_id}` : newDataset.name,
                        config: newDataset.config,
                        model_id: newDataset.model_id,
                        dataset_type: newDataset.type,
                        dataset_name: newDataset.name
                    })
                });
            }
            
            if (res.ok) {
                setShowRegister(false);
                setNewDataset({ model_id: "", name: "", type: "training", source: "local", config: {} });
                fetchDatasets();
            } else {
                const err = await res.json();
                alert(`Error: ${JSON.stringify(err.detail || err)}`);
            }
        } catch (e) { 
            console.error(e);
        } finally { setRegistering(false); }
    };

    const updateConfig = (key: string, value: any) => {
        setNewDataset({
            ...newDataset,
            config: { ...newDataset.config, [key]: value }
        });
    };

    useEffect(() => {
        fetchDatasets();
        fetchModels();
    }, []);

    if (loading && datasets.length === 0) return (
        <div className="flex flex-col items-center justify-center py-32 gap-5 text-center">
            <div className="w-14 h-14 rounded-full border-2 border-emerald-500/20 border-t-emerald-500 animate-spin" />
            <p className="text-[10px] uppercase tracking-[0.4em] font-black text-slate-600">Discovering Data Assets...</p>
        </div>
    );

    return (
        <div className="grid grid-cols-1 xl:grid-cols-[1fr_400px] gap-8">
            <div className="space-y-4">
                <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-black uppercase tracking-widest text-slate-300">Registered Datasets</h3>
                    <div className="flex items-center gap-4">
                        <div className="flex items-center gap-2 px-3 py-1 bg-emerald-500/10 border border-emerald-500/20 rounded-full">
                            <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse" />
                            <span className="text-[8px] font-black text-emerald-500 uppercase">Automated Discovery Active</span>
                        </div>
                        <button onClick={() => setShowRegister(true)} className="text-[9px] uppercase font-black px-4 py-2 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600 transition-all">+ Add Dataset</button>
                        <button onClick={fetchDatasets} className="text-[9px] uppercase font-black text-slate-500 hover:text-white transition-all">↻ Sync</button>
                    </div>
                </div>

                {showRegister && (
                    <Card className="p-6 mb-6 border-emerald-500/20 bg-emerald-500/[0.02]">
                        <div className="flex items-center justify-between mb-4">
                            <h4 className="text-[10px] font-black uppercase tracking-widest text-emerald-400">Register New Dataset</h4>
                            <button onClick={() => setShowRegister(false)} className="text-slate-600 hover:text-white">✕</button>
                        </div>
                        <form onSubmit={handleRegister} className="grid grid-cols-1 gap-4">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <select value={newDataset.model_id} onChange={e => setNewDataset({ ...newDataset, model_id: e.target.value })} className="bg-black/40 border border-white/5 rounded-xl px-4 py-3 text-xs text-white focus:border-emerald-500/40 outline-none" required>
                                    <option value="" className="bg-[#0E1014]">Select Source Model</option>
                                    {models.map(m => <option key={m.model_id || m.id} value={m.model_id || m.id} className="bg-[#0E1014]">{m.name}</option>)}
                                </select>
                                <input value={newDataset.name} onChange={e => setNewDataset({ ...newDataset, name: e.target.value })} placeholder="Dataset Display Name" className="bg-black/40 border border-white/5 rounded-xl px-4 py-3 text-xs text-white focus:border-emerald-500/40 outline-none" required />
                                <select value={newDataset.type} onChange={e => setNewDataset({ ...newDataset, type: e.target.value })} className="bg-black/40 border border-white/5 rounded-xl px-4 py-3 text-xs text-white focus:border-emerald-500/40 outline-none" required>
                                    <option value="training" className="bg-[#0E1014]">Training Data</option>
                                    <option value="validation" className="bg-[#0E1014]">Validation Data</option>
                                    <option value="test" className="bg-[#0E1014]">Test Data</option>
                                </select>
                                <select value={newDataset.source} onChange={e => setNewDataset({ ...newDataset, source: e.target.value, config: {} })} className="bg-black/40 border border-white/5 rounded-xl px-4 py-3 text-xs text-white focus:border-emerald-500/40 outline-none" required>
                                    <option value="local" className="bg-[#0E1014]">Local upload</option>
                                    <option value="huggingface" className="bg-[#0E1014]">HuggingFace</option>
                                    <option value="kaggle" className="bg-[#0E1014]">Kaggle</option>
                                    <option value="openml" className="bg-[#0E1014]">OpenML</option>
                                    <option value="roboflow" className="bg-[#0E1014]">Roboflow</option>
                                    <option value="s3" className="bg-[#0E1014]">S3/Cloud</option>
                                </select>
                            </div>
                            
                            {/* Inline Form Additions Based on Source Selection */}
                            <div className="bg-black/20 p-4 rounded-xl border border-white/[0.03]">
                                {newDataset.source === "kaggle" && (
                                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                                        <input value={newDataset.config.kaggle_username || ""} onChange={e => updateConfig("kaggle_username", e.target.value)} placeholder="Kaggle Username" className="bg-black/40 border border-white/5 rounded-md px-3 py-2 text-xs text-white outline-none" required />
                                        <input value={newDataset.config.kaggle_key || ""} onChange={e => updateConfig("kaggle_key", e.target.value)} type="password" placeholder="Kaggle API Key" className="bg-black/40 border border-white/5 rounded-md px-3 py-2 text-xs text-white outline-none" required />
                                        <input value={newDataset.config.dataset_slug || ""} onChange={e => updateConfig("dataset_slug", e.target.value)} placeholder="Dataset Slug" className="bg-black/40 border border-white/5 rounded-md px-3 py-2 text-xs text-white outline-none" required />
                                        <input value={newDataset.config.file_name || ""} onChange={e => updateConfig("file_name", e.target.value)} placeholder="File Name (Optional)" className="bg-black/40 border border-white/5 rounded-md px-3 py-2 text-xs text-white outline-none" />
                                    </div>
                                )}
                                {newDataset.source === "openml" && (
                                    <div className="space-y-4">
                                        <div className="flex gap-2">
                                            <input value={openmlQuery} onChange={e => setOpenmlQuery(e.target.value)} placeholder="Search OpenML datasets..." className="flex-1 bg-black/40 border border-white/5 rounded-md px-3 py-2 text-xs text-white outline-none" />
                                            <button type="button" onClick={searchOpenML} className="px-4 py-2 bg-blue-600 rounded-md text-[9px] font-black uppercase text-white hover:bg-blue-500 transition-all">Search</button>
                                        </div>
                                        {openmlResults.length > 0 && (
                                            <div className="grid grid-cols-1 gap-2">
                                                {openmlResults.map(r => (
                                                    <div key={r.id} onClick={() => { updateConfig("dataset_id", r.id); setNewDataset({...newDataset, name: r.name, config: {dataset_id: r.id}}); setOpenmlResults([]); }} 
                                                        className={`p-3 rounded-lg border cursor-pointer transition-all ${newDataset.config.dataset_id === r.id ? "border-emerald-500 bg-emerald-500/10" : "border-white/5 bg-black/40 hover:border-white/20"}`}>
                                                        <div className="flex justify-between items-center">
                                                            <p className="text-xs font-bold text-white">{r.name}</p>
                                                            <span className="text-[8px] font-black text-slate-600">ID: {r.id}</span>
                                                        </div>
                                                        <div className="flex gap-4 mt-2">
                                                            <p className="text-[9px] text-slate-500">Rows: {r.n_rows}</p>
                                                            <p className="text-[9px] text-slate-500">Cols: {r.n_cols}</p>
                                                            <p className="text-[9px] text-slate-500">Classes: {r.n_classes}</p>
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        )}
                                        {newDataset.config.dataset_id && !openmlResults.length && (
                                            <div className="text-[10px] text-emerald-400 font-bold italic">Selected OpenML Dataset ID: {newDataset.config.dataset_id}</div>
                                        )}
                                    </div>
                                )}
                                {newDataset.source === "roboflow" && (
                                    <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
                                        <input value={newDataset.config.api_key || ""} onChange={e => updateConfig("api_key", e.target.value)} type="password" placeholder="Roboflow API Key" className="bg-black/40 border border-white/5 rounded-md px-3 py-2 text-xs text-white outline-none" required />
                                        <input value={newDataset.config.workspace || ""} onChange={e => updateConfig("workspace", e.target.value)} placeholder="Workspace" className="bg-black/40 border border-white/5 rounded-md px-3 py-2 text-xs text-white outline-none" required />
                                        <input value={newDataset.config.project || ""} onChange={e => updateConfig("project", e.target.value)} placeholder="Project" className="bg-black/40 border border-white/5 rounded-md px-3 py-2 text-xs text-white outline-none" required />
                                        <input value={newDataset.config.version || ""} onChange={e => updateConfig("version", e.target.value)} placeholder="Version" className="bg-black/40 border border-white/5 rounded-md px-3 py-2 text-xs text-white outline-none" required />
                                        <select value={newDataset.config.format || "yolov8"} onChange={e => updateConfig("format", e.target.value)} className="bg-black/40 border border-white/5 rounded-md px-3 py-2 text-xs text-white outline-none" required>
                                            <option value="yolov8">YOLOv8</option>
                                            <option value="coco">COCO</option>
                                            <option value="csv">CSV</option>
                                        </select>
                                    </div>
                                )}
                                {newDataset.source === "s3" && (
                                    <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
                                        <input value={newDataset.config.url || ""} onChange={e => updateConfig("url", e.target.value)} placeholder="S3/Blob URL" className="bg-black/40 border border-white/5 rounded-md px-3 py-2 text-xs text-white outline-none col-span-2" required />
                                        <input value={newDataset.config.access_key || ""} onChange={e => updateConfig("access_key", e.target.value)} placeholder="Access Key" className="bg-black/40 border border-white/5 rounded-md px-3 py-2 text-xs text-white outline-none" />
                                        <input value={newDataset.config.secret_key || ""} onChange={e => updateConfig("secret_key", e.target.value)} type="password" placeholder="Secret Key" className="bg-black/40 border border-white/5 rounded-md px-3 py-2 text-xs text-white outline-none" />
                                    </div>
                                )}
                                {newDataset.source === "huggingface" && (
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                        <input value={newDataset.config.repo_id || ""} onChange={e => updateConfig("repo_id", e.target.value)} placeholder="HuggingFace Repo ID" className="bg-black/40 border border-white/5 rounded-md px-3 py-2 text-xs text-white outline-none" required />
                                        <input value={newDataset.config.token || ""} onChange={e => updateConfig("token", e.target.value)} type="password" placeholder="HF Token (Optional)" className="bg-black/40 border border-white/5 rounded-md px-3 py-2 text-xs text-white outline-none" />
                                    </div>
                                )}
                                {newDataset.source === "local" && (
                                    <div className="text-xs text-slate-500 italic p-2">Upload a file directly from your computer (Coming Soon).</div>
                                )}
                            </div>
                            
                            <button type="submit" disabled={registering} className="bg-emerald-600 text-white font-black py-3 rounded-xl text-[10px] uppercase tracking-widest disabled:opacity-50 mt-2">
                                {registering ? "Registering..." : "Confirm & Fetch Data"}
                            </button>
                        </form>
                    </Card>
                )}

                <Card className="overflow-hidden border-emerald-500/5">
                    <table className="w-full text-xs text-left">
                        <thead>
                            <tr className="bg-white/[0.02] border-b border-white/5 transition-all">
                                <th className="px-6 py-4 text-[9px] font-black uppercase tracking-widest text-slate-600">Dataset Name</th>
                                <th className="px-6 py-4 text-[9px] font-black uppercase tracking-widest text-slate-600">Type</th>
                                <th className="px-6 py-4 text-[9px] font-black uppercase tracking-widest text-slate-600">Records</th>
                                <th className="px-6 py-4 text-[9px] font-black uppercase tracking-widest text-slate-600">Source Model</th>
                                <th className="px-6 py-4 text-[9px] font-black uppercase tracking-widest text-slate-600">Created At</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-white/[0.03]">
                            {datasets.map(d => (
                                <tr key={d.dataset_id} onClick={() => { setSelectedDataset(d); fetchLineage(d.dataset_id); }}
                                    className={`cursor-pointer transition-all hover:bg-emerald-500/[0.02] ${selectedDataset?.dataset_id === d.dataset_id ? "bg-emerald-500/5" : ""}`}>
                                    <td className="px-6 py-4 font-mono font-black text-slate-300 flex items-center gap-3">
                                        <div className={`w-8 h-8 rounded-lg flex items-center justify-center p-1.5 ${selectedDataset?.dataset_id === d.dataset_id ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20" : "bg-white/[0.03] text-slate-600 border border-white/5"}`}>
                                            <Database className="w-full h-full" />
                                        </div>
                                        {d.name}
                                    </td>
                                    <td className="px-6 py-4">
                                        <span className={`text-[8px] font-black uppercase px-2 py-0.5 rounded border ${d.type === 'training' ? 'text-blue-400 border-blue-500/20 bg-blue-500/5' : 'text-amber-400 border-amber-500/20 bg-amber-500/5'}`}>
                                            {d.type || "Training"}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4 font-black text-slate-400">{d.row_count?.toLocaleString() || "—"}</td>
                                    <td className="px-6 py-4 text-xs font-bold text-slate-300">{d.model_name || "N/A"}</td>
                                    <td className="px-6 py-4 text-[9px] font-bold text-slate-600 uppercase tabular-nums">{new Date(d.created_at).toLocaleDateString()}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </Card>
            </div>

            <div className="space-y-6">
                {selectedDataset ? (
                    <>
                        <div className="p-6 rounded-2xl border border-white/5 bg-[#0E1014] space-y-5">
                            <h3 className="text-xs font-black uppercase tracking-widest text-slate-400">Resource Summary</h3>
                            <div className="grid grid-cols-2 gap-3">
                                <Tile label="Records" value={selectedDataset.row_count?.toLocaleString() || "—"} />
                                <Tile label="Versions" value={selectedDataset.version_count || 0} accent={selectedDataset.version_count > 1} />
                            </div>
                            <div className="p-4 rounded-xl border border-white/[0.03] bg-black/20 space-y-2">
                                <p className="text-[8px] uppercase font-black text-slate-700">Storage Location</p>
                                <p className="text-xs font-mono font-bold text-emerald-500 truncate">{selectedDataset.storage_url || "minio://mlguard/datasets/" + selectedDataset.dataset_id}</p>
                            </div>
                        </div>

                        <div className="space-y-3">
                            <div className="flex items-center gap-2 mb-2 p-1">
                                <Clock className="w-3.5 h-3.5 text-slate-700" />
                                <h3 className="text-[10px] font-black uppercase tracking-widest text-slate-600">Dataset Lineage</h3>
                            </div>
                            {lineage.length === 0 ? <p className="text-xs text-slate-700 text-center py-10 italic">No historical lineage data found</p> :
                                lineage.map(v => (
                                    <div key={v.version_id} className="p-4 rounded-xl border border-white/5 bg-black/20 space-y-4 group transition-all hover:border-white/10">
                                        <div className="flex items-center justify-between">
                                            <div className="flex items-center gap-3">
                                                <div className="w-8 h-8 rounded-lg bg-emerald-500/10 flex items-center justify-center font-black text-[10px] text-emerald-400">v{v.version_number}</div>
                                                <div>
                                                    <p className="text-xs font-black text-white">{v.row_count?.toLocaleString() || 0} Rows</p>
                                                    <p className="text-[8px] font-black text-slate-700 uppercase">{v.created_at?.split('T')[0]}</p>
                                                </div>
                                            </div>
                                            <ChevronRight className="w-4 h-4 text-slate-800" />
                                        </div>
                                        {v.linked_models?.length > 0 && (
                                            <div className="border-t border-white/[0.03] pt-3 space-y-2">
                                                <p className="text-[8px] font-black uppercase tracking-widest text-slate-700">Consumed By</p>
                                                {v.linked_models.map((m: any, i: number) => (
                                                    <div key={i} className="flex flex-col p-2.5 rounded-lg border border-white/5 bg-white/[0.02]">
                                                        <div className="flex items-center justify-between">
                                                            <p className="text-xs font-bold text-slate-300">{m.model_name}</p>
                                                            <span className="text-[8px] font-black text-slate-600">v{m.model_version}</span>
                                                        </div>
                                                        <p className="text-[8px] font-black text-slate-700 uppercase mt-1">Usage: {m.link_type || "Training"}</p>
                                                    </div>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                ))
                            }
                        </div>
                    </>
                ) : (
                    <div className="flex flex-col items-center justify-center py-24 text-center border-2 border-dashed border-white/5 rounded-3xl">
                        <Database className="w-12 h-12 text-slate-800 mb-4" />
                        <p className="text-[10px] font-black uppercase tracking-widest text-slate-700">Select a dataset to view full lineage</p>
                    </div>
                )}
            </div>
        </div>
    );
}
