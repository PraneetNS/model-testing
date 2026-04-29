"use client";
import React, { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { Package, ShieldCheck, ChevronLeft, Activity, Clock, User, HardDrive, AlertTriangle, ArrowRight, Shield, Database, Zap, Scale } from "lucide-react";
import { apiFetch, safeJson } from "@/lib/api";

const Card = ({ children, className = "" }: any) => (
    <div className={`bg-[#0E1014] border border-white/[0.07] rounded-2xl ${className}`}>{children}</div>
);

const Badge = ({ label, variant = "neutral" }: { label: string; variant?: string }) => {
    const cls = 
        variant === "critical" ? "bg-red-500/10 text-red-400 border-red-500/30" :
        variant === "high" ? "bg-orange-500/10 text-orange-400 border-orange-500/30" :
        variant === "medium" ? "bg-blue-500/10 text-blue-400 border-blue-500/30" :
        variant === "low" ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30" :
        "bg-white/5 text-slate-400 border-white/10";
    
    return <span className={`text-[10px] font-black uppercase px-2.5 py-0.5 rounded-full border ${cls}`}>{label}</span>;
};

export default function ModelDetailPage() {
    const params = useParams();
    const router = useRouter();
    const id = params.id as string;
    
    const [model, setModel] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (!id) return;
        
        const fetchModel = async () => {
            setLoading(true);
            try {
                const res = await apiFetch(`/api/v1/models/${id}`);
                if (!res.ok) {
                    if (res.status === 404) throw new Error("Model not found");
                    throw new Error("Failed to fetch model details");
                }
                const data = await safeJson(res);
                setModel(data);
            } catch (e: any) {
                setError(e.message);
            } finally {
                setLoading(false);
            }
        };
        
        fetchModel();
    }, [id]);

    if (loading) {
        return (
            <div className="min-h-screen bg-[#050608] flex flex-col items-center justify-center gap-4">
                <div className="w-10 h-10 border-2 border-orange-500/20 border-t-orange-500 rounded-full animate-spin" />
                <p className="text-[10px] font-black uppercase text-slate-600 tracking-widest">Loading Model Lineage...</p>
            </div>
        );
    }

    if (error || !model) {
        return (
            <div className="min-h-screen bg-[#050608] flex flex-col items-center justify-center gap-6 p-10 text-center">
                <AlertTriangle className="w-12 h-12 text-red-500 opacity-50" />
                <div>
                    <h2 className="text-xl font-black text-white uppercase tracking-tight">Access Denied or Not Found</h2>
                    <p className="text-slate-600 text-sm mt-2 max-w-md">{error || "The requested model could not be located in the governance registry."}</p>
                </div>
                <button 
                    onClick={() => router.push("/dashboard")}
                    className="px-6 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl text-[10px] font-black uppercase text-slate-400 transition-all"
                >
                    Return to Terminal
                </button>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-[#050608] text-slate-200 p-8 lg:p-12">
            <div className="max-w-6xl mx-auto space-y-12 animate-in fade-in slide-in-from-bottom-4 duration-700">
                {/* Header */}
                <header className="flex items-center justify-between">
                    <div className="flex items-center gap-6">
                        <button 
                            onClick={() => router.back()}
                            className="p-3 rounded-2xl bg-white/[0.03] border border-white/5 hover:bg-white/[0.08] transition-all group"
                        >
                            <ChevronLeft className="w-5 h-5 text-slate-400 group-hover:text-white" />
                        </button>
                        <div>
                            <div className="flex items-center gap-3">
                                <h1 className="text-3xl font-black text-white tracking-tighter uppercase">{model.name}</h1>
                                <Badge label={model.risk_tier || "Unassigned"} variant={model.risk_tier} />
                            </div>
                            <p className="text-[10px] font-black text-slate-500 uppercase tracking-[0.3em] mt-1.5 flex items-center gap-2">
                                <Database className="w-3 h-3" /> Model ID: {model.id}
                            </p>
                        </div>
                    </div>
                    
                    <div className="flex items-center gap-3">
                        <div className="text-right">
                            <p className="text-[9px] font-black text-slate-600 uppercase tracking-widest">Governance Score</p>
                            <p className={`text-2xl font-black ${model.governance_score >= 80 ? "text-emerald-400" : "text-orange-400"}`}>
                                {model.governance_score ? `${Math.round(model.governance_score)}%` : "N/A"}
                            </p>
                        </div>
                    </div>
                </header>

                {/* Main Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                    {/* Left: Summary Stats */}
                    <div className="space-y-6">
                        <Card className="p-6 space-y-6">
                            <h3 className="text-[11px] font-black uppercase tracking-widest text-slate-500 border-b border-white/5 pb-4">Identity & Ownership</h3>
                            
                            <div className="space-y-4">
                                <div className="space-y-1">
                                    <p className="text-[9px] font-black text-slate-700 uppercase">Business Owner</p>
                                    <p className="text-sm font-bold text-white">{model.business_owner || "Not Assigned"}</p>
                                </div>
                                <div className="space-y-1">
                                    <p className="text-[9px] font-black text-slate-700 uppercase">Technical Owner</p>
                                    <p className="text-sm font-bold text-slate-300">{model.technical_owner || "Not Assigned"}</p>
                                </div>
                                <div className="space-y-1">
                                    <p className="text-[9px] font-black text-slate-700 uppercase">Environment</p>
                                    <div className="flex items-center gap-2">
                                        <div className={`w-1.5 h-1.5 rounded-full ${model.deployment_environment === "production" ? "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" : "bg-slate-700"}`} />
                                        <p className="text-xs font-black uppercase text-slate-400">{model.deployment_environment || "Unknown"}</p>
                                    </div>
                                </div>
                            </div>
                        </Card>

                        <Card className="p-6 space-y-4">
                            <h3 className="text-[11px] font-black uppercase tracking-widest text-slate-500 border-b border-white/5 pb-4">Compliance Status</h3>
                            <div className="flex items-center justify-between">
                                <span className="text-[11px] font-bold text-slate-400">SR 11-7</span>
                                <span className="text-[10px] font-black text-emerald-500 uppercase">Compliant</span>
                            </div>
                            <div className="flex items-center justify-between">
                                <span className="text-[11px] font-bold text-slate-400">EU AI Act</span>
                                <span className="text-[10px] font-black text-amber-500 uppercase">Reviewing</span>
                            </div>
                            <div className="flex items-center justify-between">
                                <span className="text-[11px] font-bold text-slate-400">Model Inventory</span>
                                <span className="text-[10px] font-black text-emerald-500 uppercase">Synced</span>
                            </div>
                        </Card>
                    </div>

                    {/* Right: History & Actions */}
                    <div className="lg:col-span-2 space-y-6">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <Card className="p-8 flex flex-col items-center justify-center text-center gap-4 group hover:border-orange-500/30 transition-all cursor-pointer">
                                <div className="p-4 rounded-2xl bg-orange-500/10 border border-orange-500/10 group-hover:scale-110 transition-transform">
                                    <Shield className="w-8 h-8 text-orange-500" />
                                </div>
                                <div>
                                    <h4 className="text-sm font-black text-white uppercase tracking-tight">Run New Audit</h4>
                                    <p className="text-[10px] text-slate-600 font-bold uppercase mt-1">Full governance validation</p>
                                </div>
                            </Card>

                            <Card className="p-8 flex flex-col items-center justify-center text-center gap-4 group hover:border-blue-500/30 transition-all cursor-pointer">
                                <div className="p-4 rounded-2xl bg-blue-500/10 border border-blue-500/10 group-hover:scale-110 transition-transform">
                                    <Zap className="w-8 h-8 text-blue-500" />
                                </div>
                                <div>
                                    <h4 className="text-sm font-black text-white uppercase tracking-tight">Explainability</h4>
                                    <p className="text-[10px] text-slate-600 font-bold uppercase mt-1">Feature importance & SHAP</p>
                                </div>
                            </Card>
                        </div>

                        <Card className="overflow-hidden">
                            <div className="p-6 border-b border-white/5 bg-white/[0.01] flex items-center justify-between">
                                <h3 className="text-[11px] font-black uppercase tracking-widest text-slate-400">Recent Scan History</h3>
                                <button className="text-[9px] font-black text-orange-500 uppercase hover:text-orange-400 transition-colors">View All</button>
                            </div>
                            <div className="divide-y divide-white/[0.03]">
                                {[1, 2, 3].map(i => (
                                    <div key={i} className="p-5 flex items-center justify-between hover:bg-white/[0.01] transition-colors">
                                        <div className="flex items-center gap-4">
                                            <div className="w-10 h-10 rounded-xl bg-white/[0.03] flex items-center justify-center">
                                                <Activity className="w-5 h-5 text-slate-500" />
                                            </div>
                                            <div>
                                                <p className="text-xs font-bold text-white uppercase">Comprehensive Audit</p>
                                                <p className="text-[10px] text-slate-600 font-mono mt-0.5">APR {29-i}, 2026 • 11:1{i} AM</p>
                                            </div>
                                        </div>
                                        <div className="flex items-center gap-6">
                                            <div className="text-right">
                                                <p className="text-[9px] font-black text-slate-600 uppercase">Score</p>
                                                <p className="text-xs font-black text-emerald-400">8{i}%</p>
                                            </div>
                                            <ArrowRight className="w-4 h-4 text-slate-700" />
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </Card>
                    </div>
                </div>
            </div>

            {/* Background Effects */}
            <div className="fixed top-0 right-0 w-1/3 h-1/3 bg-orange-500/5 blur-[120px] pointer-events-none rounded-full" />
            <div className="fixed bottom-0 left-0 w-1/4 h-1/4 bg-blue-500/5 blur-[120px] pointer-events-none rounded-full" />
        </div>
    );
}
