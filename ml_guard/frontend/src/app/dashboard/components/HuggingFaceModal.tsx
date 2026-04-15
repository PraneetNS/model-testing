"use client";
import React, { useState, useEffect, useRef } from "react";
import { Search, Globe, ShieldAlert, ShieldCheck, Loader2, Download, Zap, X, Info, ExternalLink } from "lucide-react";
import { apiFetch } from "@/lib/api";

interface HFModel {
  repo_id: string;
  downloads: number;
  likes: number;
  pipeline_tag: string;
  license: string;
  has_model_card: boolean;
}

interface CardRisks {
  has_model_card: boolean;
  license: string;
  has_limitations_section: boolean;
  has_bias_section: boolean;
  risk_flags: string[];
  pipeline_tag?: string;
}

const Card = ({ children, className = "" }: any) => (
  <div className={`bg-[#0A0C10] border border-white/[0.05] rounded-xl ${className}`}>{children}</div>
);

export default function HuggingFacePluginModal({ isOpen, onClose, onModelSelected, onAuditStarted }: any) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<HFModel[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedRepo, setSelectedRepo] = useState<string | null>(null);
  const [risks, setRisks] = useState<CardRisks | null>(null);
  const [fetchingRisks, setFetchingRisks] = useState(false);
  const [hfToken, setHfToken] = useState("");
  const [revision, setRevision] = useState("main");
  const [filename, setFilename] = useState("");
  const [view, setView] = useState<"search" | "details">("search");
  
  // Dataset field for "Direct Audit"
  const [datasetRepoId, setDatasetRepoId] = useState("");
  const [datasetSplit, setDatasetSplit] = useState("test");

  const searchTimeout = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (query.length < 2) {
      setResults([]);
      return;
    }

    if (searchTimeout.current) clearTimeout(searchTimeout.current);
    
    searchTimeout.current = setTimeout(async () => {
      setLoading(true);
      try {
        const res = await apiFetch(`/api/plugins/huggingface/search?query=${encodeURIComponent(query)}&limit=8`);
        if (res.ok) {
          const data = await res.json();
          setResults(data);
        }
      } catch (e) {
        console.error("HF Search failed", e);
      } finally {
        setLoading(false);
      }
    }, 400);

    return () => {
      if (searchTimeout.current) clearTimeout(searchTimeout.current);
    };
  }, [query]);

  const fetchRisks = async (repoId: string) => {
    setFetchingRisks(true);
    setRisks(null);
    try {
      const res = await apiFetch(`/api/plugins/huggingface/model-card-risks?repo_id=${encodeURIComponent(repoId)}`);
      if (res.ok) {
        setRisks(await res.json());
      }
    } catch (e) {
      console.error("HF Risks failed", e);
    } finally {
      setFetchingRisks(false);
    }
  };

  const handleSelect = (repo: string) => {
    setSelectedRepo(repo);
    setView("details");
    fetchRisks(repo);
  };

  const handlePullModel = async () => {
    setLoading(true);
    try {
      const res = await apiFetch(`/api/plugins/huggingface/pull-model`, {
        method: "POST",
        body: JSON.stringify({
          repo_id: selectedRepo,
          revision,
          filename: filename || null,
          hf_token: hfToken || null
        })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Pull failed");
      
      onModelSelected?.(data);
      onClose();
    } catch (e: any) {
      alert(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDirectAudit = async () => {
    if (!datasetRepoId) {
      alert("Please specify a Dataset Repo ID for auditing.");
      return;
    }
    setLoading(true);
    try {
      const res = await apiFetch(`/api/plugins/huggingface/audit-from-hub`, {
        method: "POST",
        body: JSON.stringify({
          model_repo_id: selectedRepo,
          dataset_repo_id: datasetRepoId,
          split: datasetSplit,
          hf_token: hfToken || null
        })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Audit failed");
      
      onAuditStarted?.(data);
      onClose();
    } catch (e: any) {
      alert(e.message);
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
      <Card className="w-full max-w-2xl max-h-[90vh] flex flex-col overflow-hidden shadow-2xl border-white/10 ring-1 ring-white/5">
        {/* Header */}
        <div className="p-6 border-b border-white/5 flex items-center justify-between bg-gradient-to-r from-orange-600/10 to-transparent">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-orange-600/20 text-orange-400">
              <Globe className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-sm font-black uppercase tracking-[0.2em] text-white">HuggingFace Hub Integration</h2>
              <p className="text-[10px] text-slate-500 font-bold uppercase">Zero-Upload Model Governance</p>
            </div>
          </div>
          <button onClick={onClose} className="p-2 text-slate-500 hover:text-white transition-colors">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {view === "search" ? (
            <div className="space-y-4">
              <div className="relative">
                <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                <input
                  autoFocus
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Search repository (e.g. microsoft/resnet-50)..."
                  className="w-full bg-white/[0.03] border border-white/10 rounded-xl pl-11 pr-4 py-4 text-sm text-white focus:border-orange-500/50 outline-none transition-all"
                />
                {loading && <Loader2 className="absolute right-4 top-1/2 -translate-y-1/2 w-4 h-4 text-orange-500 animate-spin" />}
              </div>

              <div className="grid grid-cols-1 gap-2">
                {results.length > 0 ? (
                  results.map((r) => (
                    <button
                      key={r.repo_id}
                      onClick={() => handleSelect(r.repo_id)}
                      className="flex items-center justify-between p-4 rounded-xl bg-white/[0.02] border border-white/5 hover:bg-white/[0.05] hover:border-white/10 transition-all text-left"
                    >
                      <div>
                        <p className="text-xs font-black text-white">{r.repo_id}</p>
                        <div className="flex items-center gap-3 mt-1.5">
                          <span className="text-[9px] font-bold text-slate-500">{r.pipeline_tag || "Unknown"}</span>
                          <span className="text-[9px] font-bold text-slate-500">• {r.downloads.toLocaleString()} DLs</span>
                        </div>
                      </div>
                      <Zap className="w-3 h-3 text-orange-400 opacity-0 group-hover:opacity-100 transition-all" />
                    </button>
                  ))
                ) : query.length >= 2 && !loading ? (
                  <div className="py-12 text-center">
                    <p className="text-xs text-slate-600 font-bold uppercase tracking-widest text-center">No results found</p>
                  </div>
                ) : (
                  <div className="py-12 text-center opacity-40">
                    <Globe className="w-10 h-10 text-slate-700 mx-auto mb-3" />
                    <p className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-700">Type to search 700k+ models</p>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              {/* Repo Detail & Risks */}
              <div className="flex items-start justify-between">
                <div>
                  <button onClick={() => setView("search")} className="text-[10px] font-black uppercase text-slate-500 hover:text-white mb-2 flex items-center gap-1">
                    ← Back to Search
                  </button>
                  <h3 className="text-lg font-black text-white">{selectedRepo}</h3>
                  <a href={`https://huggingface.co/${selectedRepo}`} target="_blank" rel="noreferrer" className="text-[10px] text-orange-400 hover:underline flex items-center gap-1 mt-1">
                    View on Hub <ExternalLink className="w-2.5 h-2.5" />
                  </a>
                </div>
                
                {fetchingRisks ? (
                  <div className="animate-pulse flex items-center gap-2 text-[10px] text-slate-500 font-black uppercase">
                    <Loader2 className="w-3 h-3 animate-spin" /> Scanning Risks...
                  </div>
                ) : risks && (
                  <div className="flex gap-2">
                    {risks.risk_flags.length === 0 ? (
                      <span className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-[9px] font-black uppercase">
                        <ShieldCheck className="w-3 h-3" /> Governance Compliant
                      </span>
                    ) : (
                      <span className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-500/10 text-red-400 border border-red-500/20 text-[9px] font-black uppercase">
                        <ShieldAlert className="w-3 h-3" /> {risks.risk_flags.length} Risk Flags
                      </span>
                    )}
                  </div>
                )}
              </div>

              {/* Risk Details Grid */}
              {risks && (
                <div className="grid grid-cols-2 gap-3">
                  <RiskItem icon={<Info className="w-3 h-3" />} label="Model Card" value={risks.has_model_card ? "Present" : "Missing"} status={risks.has_model_card ? "pass" : "fail"} />
                  <RiskItem icon={<ShieldCheck className="w-3 h-3" />} label="License" value={risks.license || "None"} status={risks.license ? (risks.risk_flags.includes("restrictive_license") ? "warn" : "pass") : "fail"} />
                  <RiskItem icon={<ShieldAlert className="w-3 h-3" />} label="Bias Disclosure" value={risks.has_bias_section ? "Present" : "Missing"} status={risks.has_bias_section ? "pass" : "fail"} />
                  <RiskItem icon={<Info className="w-3 h-3" />} label="Limitations Info" value={risks.has_limitations_section ? "Present" : "Missing"} status={risks.has_limitations_section ? "pass" : "fail"} />
                </div>
              )}

              {/* Advanced Controls */}
              <div className="space-y-4 pt-4 border-t border-white/5">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1.5">
                    <p className="text-[9px] font-black uppercase tracking-widest text-slate-500">HF Access Token (Optional)</p>
                    <input type="password" value={hfToken} onChange={e => setHfToken(e.target.value)} placeholder="hf_..." className="w-full bg-white/[0.03] border border-white/10 rounded-lg px-3 py-2 text-xs text-white focus:border-orange-500 outline-none" />
                  </div>
                  <div className="space-y-1.5">
                    <p className="text-[9px] font-black uppercase tracking-widest text-slate-500">Revision / Branch</p>
                    <input value={revision} onChange={e => setRevision(e.target.value)} className="w-full bg-white/[0.03] border border-white/10 rounded-lg px-3 py-2 text-xs text-white focus:border-orange-500 outline-none" />
                  </div>
                </div>

                <div className="p-4 rounded-xl bg-orange-600/[0.03] border border-orange-600/10 space-y-4">
                  <div className="flex items-center gap-2 mb-2">
                    <Zap className="w-4 h-4 text-orange-400" />
                    <p className="text-[10px] font-black uppercase tracking-widest text-orange-400">Zero-Upload Direct Audit</p>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <div className="space-y-1.5">
                      <p className="text-[8px] font-black uppercase text-slate-600">Evaluation Dataset (Repo ID)</p>
                      <input value={datasetRepoId} onChange={e => setDatasetRepoId(e.target.value)} placeholder="e.g. glue / titanic" className="w-full bg-black/40 border border-white/5 rounded-lg px-3 py-2 text-xs text-white focus:border-orange-500 outline-none" />
                    </div>
                    <div className="space-y-1.5">
                      <p className="text-[8px] font-black uppercase text-slate-600">Dataset Split</p>
                      <input value={datasetSplit} onChange={e => setDatasetSplit(e.target.value)} className="w-full bg-black/40 border border-white/5 rounded-lg px-3 py-2 text-xs text-white focus:border-orange-500 outline-none" />
                    </div>
                  </div>
                  <button onClick={handleDirectAudit} disabled={loading} className="w-full py-3 bg-orange-600 hover:bg-orange-500 text-black font-black text-[10px] uppercase tracking-widest rounded-xl transition-all flex items-center justify-center gap-2">
                    {loading ? <Loader2 className="w-3 h-3 animate-spin" /> : <><Globe className="w-3 h-3" /> Start One-Shot Hub Audit</>}
                  </button>
                </div>

                <button onClick={handlePullModel} disabled={loading} className="w-full py-3 bg-white/5 hover:bg-white/10 text-white font-black text-[10px] uppercase tracking-widest rounded-xl transition-all flex items-center justify-center gap-2">
                  <Download className="w-3 h-3" /> Pull to Registry Only
                </button>
              </div>
            </div>
          )}
        </div>
      </Card>
    </div>
  );
}

function RiskItem({ icon, label, value, status }: any) {
  const colorClass = status === "pass" ? "text-emerald-400" : status === "warn" ? "text-amber-400" : "text-red-400";
  const bgClass = status === "pass" ? "bg-emerald-500/5" : status === "warn" ? "bg-amber-500/5" : "bg-red-500/5";
  return (
    <div className={`p-3 rounded-lg border border-white/5 flex items-center gap-3 ${bgClass}`}>
      <div className={colorClass}>{icon}</div>
      <div>
        <p className="text-[8px] font-black text-slate-600 uppercase mb-0.5">{label}</p>
        <p className={`text-[10px] font-black truncate max-w-[120px] ${colorClass}`}>{value}</p>
      </div>
    </div>
  );
}
