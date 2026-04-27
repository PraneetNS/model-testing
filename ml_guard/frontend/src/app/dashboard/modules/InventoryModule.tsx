"use client";
import { apiFetch, safeJson } from "@/lib/api";
import React, { useState, useEffect, useCallback } from "react";
import {
    LayoutGrid, ShieldAlert, Clock, UserCheck, 
    Download, Filter, Search, ChevronRight,
    AlertTriangle, CheckCircle2, Shield,
    ArrowUpRight, BarChart3, Globe, Briefcase
} from "lucide-react";

const Badge = ({ label, variant = "neutral" }: { label: string; variant?: string }) => {
    const cls = 
        variant === "critical" ? "bg-red-500/10 text-red-400 border-red-500/30" :
        variant === "high" ? "bg-orange-500/10 text-orange-400 border-orange-500/30" :
        variant === "medium" ? "bg-blue-500/10 text-blue-400 border-blue-500/30" :
        variant === "low" ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30" :
        "bg-white/5 text-slate-400 border-white/10";
    
    return <span className={`text-[10px] font-black uppercase px-2.5 py-0.5 rounded-full border ${cls}`}>{label}</span>;
};

const StatCard = ({ label, value, icon: Icon, sub, color }: any) => (
    <div className="bg-[#0E1014] border border-white/[0.06] rounded-2xl p-6 relative overflow-hidden group">
        <div className="absolute top-0 right-0 p-8 opacity-[0.03] group-hover:opacity-[0.06] transition-opacity">
            {Icon && <Icon className="w-16 h-16" />}
        </div>
        <div className="flex items-center gap-3 mb-4">
            <div className={`p-2 rounded-lg ${color || "bg-white/5 text-slate-400"}`}>
                {Icon && <Icon className="w-4 h-4" />}
            </div>
            <p className="text-[10px] font-black uppercase tracking-widest text-slate-500">{label}</p>
        </div>
        <div className="flex items-baseline gap-2">
            <h3 className="text-3xl font-black text-white">{value}</h3>
            {sub && <span className="text-[10px] font-bold text-slate-600">{sub}</span>}
        </div>
    </div>
);

export default function InventoryModule() {
    const [models, setModels] = useState<any[]>([]);
    const [stats, setStats] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [filters, setFilters] = useState({
        risk_tier: "",
        environment: "",
        overdue: false
    });
    const [search, setSearch] = useState("");

    const loadData = useCallback(async () => {
        setLoading(true);
        try {
            const query = new URLSearchParams();
            if (filters.risk_tier) query.append("risk_tier", filters.risk_tier);
            if (filters.environment) query.append("environment", filters.environment);
            if (filters.overdue) query.append("overdue_validation", "true");

            const [mResp, sResp] = await Promise.all([
                apiFetch(`/api/inventory?${query.toString()}`),
                apiFetch(`/api/inventory/dashboard`)
            ]);
            
            setModels(await safeJson(mResp, []));
            setStats(await safeJson(sResp, null));
        } catch (e) {
            console.error("Inventory load failed", e);
        } finally {
            setLoading(false);
        }
    }, [filters]);

    useEffect(() => {
        loadData();
    }, [loadData]);

    const handleExport = () => {
        window.open("/api/inventory/export", "_blank");
    };

    const filteredModels = models.filter(m => 
        m.name.toLowerCase().includes(search.toLowerCase()) ||
        m.business_owner?.toLowerCase().includes(search.toLowerCase())
    );

    return (
        <div className="space-y-8 animate-in fade-in duration-700">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-2xl font-black text-white tracking-tight uppercase">Model Risk Inventory</h2>
                    <p className="text-[11px] text-slate-500 font-bold uppercase tracking-[0.2em] mt-1">
                        SR 11-7 Compliance · Actuarial Lineage · Risk Tiering
                    </p>
                </div>
                <button 
                    onClick={handleExport}
                    className="flex items-center gap-2 px-5 py-2.5 bg-orange-500 hover:bg-orange-600 text-black text-[11px] font-black uppercase rounded-xl transition-all"
                >
                    <Download className="w-4 h-4" />
                    Export CSV for Audit
                </button>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <StatCard 
                    label="Active Models" 
                    value={stats?.total_models || 0} 
                    icon={LayoutGrid} 
                    sub="Total Registry"
                />
                <StatCard 
                    label="Critical/High Risk" 
                    value={(stats?.by_risk_tier?.critical || 0) + (stats?.by_risk_tier?.high || 0)} 
                    icon={ShieldAlert} 
                    color="bg-red-500/10 text-red-500"
                    sub="Priority Attention"
                />
                <StatCard 
                    label="Validation Overdue" 
                    value={stats?.overdue_validations_count || 0} 
                    icon={Clock} 
                    color="bg-orange-500/10 text-orange-500"
                    sub="Action Required"
                />
                <StatCard 
                    label="Needs Owner" 
                    value={stats?.models_without_owner || 0} 
                    icon={UserCheck} 
                    color="bg-blue-500/10 text-blue-500"
                    sub="Governance Gap"
                />
            </div>

            {/* Filter Bar */}
            <div className="bg-[#0E1014] border border-white/[0.06] rounded-2xl p-4 flex flex-wrap items-center gap-4">
                <div className="relative flex-1 min-w-[300px]">
                    <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-600" />
                    <input 
                        type="text"
                        placeholder="Search by model name or owner..."
                        value={search}
                        onChange={e => setSearch(e.target.value)}
                        className="w-full bg-black/40 border border-white/10 rounded-xl py-2.5 pl-12 pr-4 text-xs text-white focus:outline-none focus:border-orange-500/50 transition-colors"
                    />
                </div>
                
                <div className="flex items-center gap-3">
                    <select 
                        value={filters.risk_tier}
                        onChange={e => setFilters({...filters, risk_tier: e.target.value})}
                        className="bg-black/40 border border-white/10 text-slate-400 text-[11px] font-bold uppercase rounded-xl px-4 py-2.5 outline-none focus:border-white/20"
                    >
                        <option value="">All Risk Tiers</option>
                        <option value="critical">Critical</option>
                        <option value="high">High</option>
                        <option value="medium">Medium</option>
                        <option value="low">Low</option>
                    </select>

                    <select 
                        value={filters.environment}
                        onChange={e => setFilters({...filters, environment: e.target.value})}
                        className="bg-black/40 border border-white/10 text-slate-400 text-[11px] font-bold uppercase rounded-xl px-4 py-2.5 outline-none focus:border-white/20"
                    >
                        <option value="">All Environments</option>
                        <option value="production">Production</option>
                        <option value="staging">Staging</option>
                        <option value="development">Development</option>
                    </select>

                    <button 
                        onClick={() => setFilters({...filters, overdue: !filters.overdue})}
                        className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-[11px] font-bold uppercase transition-all border ${
                            filters.overdue 
                                ? "bg-orange-500/10 border-orange-500/50 text-orange-500" 
                                : "bg-black/40 border-white/10 text-slate-400"
                        }`}
                    >
                        <AlertTriangle className="w-4 h-4" />
                        Overdue Only
                    </button>
                </div>
            </div>

            {/* Inventory Table */}
            <div className="bg-[#0E1014] border border-white/[0.06] rounded-2xl overflow-hidden">
                <table className="w-full text-left">
                    <thead>
                        <tr className="border-b border-white/5 bg-white/[0.01]">
                            <th className="px-6 py-4 text-[10px] font-black uppercase tracking-widest text-slate-600">Model Name</th>
                            <th className="px-6 py-4 text-[10px] font-black uppercase tracking-widest text-slate-600">Risk Tier</th>
                            <th className="px-6 py-4 text-[10px] font-black uppercase tracking-widest text-slate-600">Environment</th>
                            <th className="px-6 py-4 text-[10px] font-black uppercase tracking-widest text-slate-600">Gov Score</th>
                            <th className="px-6 py-4 text-[10px] font-black uppercase tracking-widest text-slate-600">Owner</th>
                            <th className="px-6 py-4 text-[10px] font-black uppercase tracking-widest text-slate-600">Next Validation</th>
                            <th className="px-6 py-4 text-right"></th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-white/[0.03]">
                        {filteredModels.map((m) => (
                            <tr key={m.id} className="group hover:bg-white/[0.02] transition-colors">
                                <td className="px-6 py-5">
                                    <div className="flex flex-col">
                                        <span className="text-sm font-bold text-white group-hover:text-orange-400 transition-colors">{m.name}</span>
                                        <span className="text-[10px] text-slate-600 font-mono mt-0.5">{m.id.split('-')[0]}...</span>
                                    </div>
                                </td>
                                <td className="px-6 py-5">
                                    <Badge label={m.risk_tier || "Unassigned"} variant={m.risk_tier} />
                                </td>
                                <td className="px-6 py-5">
                                    <div className="flex items-center gap-2">
                                        <div className={`w-1.5 h-1.5 rounded-full ${
                                            m.deployment_environment === "production" ? "bg-emerald-500" : "bg-slate-700"
                                        }`} />
                                        <span className="text-[11px] font-bold text-slate-400 uppercase">{m.deployment_environment}</span>
                                    </div>
                                </td>
                                <td className="px-6 py-5">
                                    <div className="flex items-center gap-2">
                                        <span className={`text-xs font-black ${
                                            m.governance_score > 80 ? "text-emerald-400" :
                                            m.governance_score > 60 ? "text-orange-400" : "text-red-400"
                                        }`}>
                                            {m.governance_score ? `${Math.round(m.governance_score)}%` : "N/A"}
                                        </span>
                                    </div>
                                </td>
                                <td className="px-6 py-5">
                                    <div className="flex flex-col">
                                        <span className="text-[11px] font-bold text-slate-300">{m.business_owner || "Unassigned"}</span>
                                        <span className="text-[9px] text-slate-600 font-medium uppercase tracking-tight">{m.technical_owner}</span>
                                    </div>
                                </td>
                                <td className="px-6 py-5">
                                    <div className="flex flex-col">
                                        <span className={`text-[11px] font-bold ${
                                            new Date(m.next_validation_due_at) < new Date() ? "text-red-500" : "text-slate-400"
                                        }`}>
                                            {m.next_validation_due_at ? new Date(m.next_validation_due_at).toLocaleDateString() : "Never"}
                                        </span>
                                        {new Date(m.next_validation_due_at) < new Date() && (
                                            <span className="text-[8px] text-red-500/50 font-black uppercase mt-0.5">Overdue</span>
                                        )}
                                    </div>
                                </td>
                                <td className="px-6 py-5 text-right">
                                    <button 
                                        onClick={() => window.location.href = `/dashboard/models/${m.id}`}
                                        className="p-2 rounded-lg bg-white/5 text-slate-600 hover:text-white hover:bg-orange-500 transition-all"
                                    >
                                        <ChevronRight className="w-4 h-4" />
                                    </button>
                                </td>
                            </tr>
                        ))}
                        {loading && (
                            <tr>
                                <td colSpan={7} className="px-6 py-20 text-center">
                                    <div className="flex flex-col items-center gap-3">
                                        <div className="w-8 h-8 border-4 border-orange-500/20 border-t-orange-500 rounded-full animate-spin" />
                                        <p className="text-[10px] font-black uppercase text-slate-600 tracking-widest">Loading Inventory...</p>
                                    </div>
                                </td>
                            </tr>
                        )}
                        {!loading && filteredModels.length === 0 && (
                            <tr>
                                <td colSpan={7} className="px-6 py-20 text-center">
                                    <p className="text-slate-600 italic text-sm">No models found in inventory.</p>
                                </td>
                            </tr>
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
