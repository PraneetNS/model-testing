"use client";
import React, { useState, useEffect } from "react";
import { 
    CreditCard, Zap, ShieldCheck, FileText, AlertCircle, 
    ArrowUpRight, Check, Loader2, Gauge, Activity, 
    Layers, Package, DollarSign, Clock, Shield
} from "lucide-react";
import { apiFetch, safeJson } from "@/lib/api";

const Card = ({ children, className = "" }: any) => (
    <div className={`bg-[#0E1014] border border-white/[0.07] rounded-2xl ${className}`}>{children}</div>
);

const Badge = ({ children, variant = "default" }: any) => {
    const colors = {
        default: "bg-white/5 text-slate-400 border-white/10",
        pro: "bg-amber-500/10 text-amber-400 border-amber-500/20",
        free: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
        enterprise: "bg-blue-500/10 text-blue-400 border-blue-500/20"
    };
    return (
        <span className={`px-2 py-0.5 rounded text-[9px] font-black uppercase tracking-widest border ${colors[variant as keyof typeof colors]}`}>
            {children}
        </span>
    );
};

const Progress = ({ value, max, color = "bg-orange-500" }: any) => {
    const pct = Math.min(100, (value / max) * 100);
    return (
        <div className="w-full h-1.5 bg-white/5 rounded-full overflow-hidden">
            <div 
                className={`h-full transition-all duration-1000 ${color}`} 
                style={{ width: `${pct}%` }} 
            />
        </div>
    );
};

export default function BillingModule() {
    const [usage, setUsage] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [submitting, setSubmitting] = useState(false);

    const fetchUsage = async () => {
        try {
            setLoading(true);
            const res = await apiFetch("/api/v1/billing/usage");
            const data = await safeJson(res);
            if (!res.ok) throw new Error(data.detail || "Failed to fetch usage");
            setUsage(data);
        } catch (e: any) {
            setError(e.message);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchUsage();
    }, []);

    const handleUpgrade = async (planSlug: string) => {
        try {
            setSubmitting(true);
            const res = await apiFetch("/api/v1/billing/subscribe", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ plan_slug: planSlug })
            });
            const data = await safeJson(res);
            if (!res.ok) throw new Error(data.detail || "Subscription failed");
            
            if (data.checkout_url) {
                window.location.href = data.checkout_url;
            } else {
                fetchUsage();
            }
        } catch (e: any) {
            alert(e.message);
        } finally {
            setSubmitting(false);
        }
    };

    if (loading) return (
        <div className="flex flex-col items-center justify-center py-40 gap-4">
            <Loader2 className="w-10 h-10 text-orange-500 animate-spin" />
            <p className="text-[10px] uppercase font-black tracking-[0.3em] text-slate-600">Syncing Ledger...</p>
        </div>
    );

    const plan = usage?.plan || { name: "Free", slug: "free", limits: {} };
    const stats = usage?.usage || {};

    const METER_MAP = [
        { key: "predictions", label: "Predictions", icon: Zap, limit: plan.limits?.predictions || 1000, current: stats.predictions || 0 },
        { key: "models", label: "Registered Models", icon: Package, limit: plan.limits?.models || 2, current: stats.models || 0 },
        { key: "reports", label: "Governance Reports", icon: FileText, limit: plan.limits?.reports || 1, current: stats.reports || 0 },
        { key: "compliance", label: "Compliance Packs", icon: ShieldCheck, limit: plan.limits?.compliance || 0, current: stats.compliance || 0 },
        { key: "guardrail", label: "Guardrail Evals", icon: Shield, limit: plan.limits?.guardrail || 0, current: stats.guardrail || 0 },
    ];

    return (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
            {/* Header Section */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
                <div>
                    <h2 className="text-2xl font-black text-white tracking-tighter uppercase flex items-center gap-3">
                        <CreditCard className="w-6 h-6 text-orange-500" />
                        Billing & Usage
                    </h2>
                    <p className="text-xs font-bold text-slate-500 uppercase tracking-widest mt-1">
                        Manage your subscription, limits, and metered consumption.
                    </p>
                </div>
                <div className="flex items-center gap-4 bg-white/[0.02] border border-white/5 p-4 rounded-2xl">
                    <div className="text-right">
                        <p className="text-[9px] font-black text-slate-600 uppercase tracking-widest">Current Plan</p>
                        <p className="text-lg font-black text-white uppercase">{plan.name}</p>
                    </div>
                    <div className="h-8 w-px bg-white/10" />
                    <Badge variant={plan.slug}>{plan.slug}</Badge>
                </div>
            </div>

            {/* Usage Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <Card className="lg:col-span-2 p-8">
                    <div className="flex items-center justify-between mb-8">
                        <h3 className="text-xs font-black uppercase tracking-widest text-slate-300 flex items-center gap-2">
                            <Gauge className="w-4 h-4 text-orange-400" />
                            Usage Meters
                        </h3>
                        <p className="text-[10px] font-bold text-slate-600 uppercase">Billing Period: {new Date().toLocaleString('default', { month: 'long' })}</p>
                    </div>
                    
                    <div className="space-y-8">
                        {METER_MAP.map(meter => {
                            const isOver = meter.limit !== -1 && meter.current >= meter.limit;
                            const progressColor = isOver ? "bg-red-500" : (meter.current / meter.limit > 0.8) ? "bg-amber-500" : "bg-orange-500";
                            
                            return (
                                <div key={meter.key} className="space-y-3">
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center gap-3">
                                            <div className="p-2 rounded-lg bg-white/5">
                                                <meter.icon className="w-4 h-4 text-slate-400" />
                                            </div>
                                            <span className="text-xs font-black text-slate-300 uppercase tracking-tight">{meter.label}</span>
                                        </div>
                                        <div className="text-right">
                                            <span className={`text-xs font-black ${isOver ? "text-red-400" : "text-white"}`}>
                                                {meter.current.toLocaleString()}
                                            </span>
                                            <span className="text-[10px] font-bold text-slate-600 ml-1">
                                                / {meter.limit === -1 ? "∞" : meter.limit.toLocaleString()}
                                            </span>
                                        </div>
                                    </div>
                                    <Progress value={meter.current} max={meter.limit === -1 ? meter.current * 1.5 : meter.limit} color={progressColor} />
                                </div>
                            );
                        })}
                    </div>
                </Card>

                <div className="space-y-6">
                    <Card className="p-6 border-orange-500/20 bg-orange-500/[0.02]">
                        <h3 className="text-xs font-black uppercase tracking-widest text-orange-400 mb-4 flex items-center gap-2">
                            <Zap className="w-4 h-4" />
                            Overage Alerts
                        </h3>
                        <p className="text-[11px] text-slate-400 leading-relaxed">
                            Predictions over your plan limit are billed at <span className="text-white font-bold">$0.0001</span> per unit. 
                            Compliance certificates are fixed at <span className="text-white font-bold">$500</span> per issuance.
                        </p>
                        <div className="mt-6 space-y-2">
                            <div className="flex justify-between text-[10px] font-bold py-2 border-b border-white/5">
                                <span className="text-slate-600 uppercase">Current Overage</span>
                                <span className="text-white">$0.00</span>
                            </div>
                            <div className="flex justify-between text-[10px] font-bold py-2 border-b border-white/5">
                                <span className="text-slate-600 uppercase">Estimated Total</span>
                                <span className="text-orange-400 font-black">$299.00</span>
                            </div>
                        </div>
                    </Card>

                    <Card className="p-6">
                        <h3 className="text-xs font-black uppercase tracking-widest text-slate-300 mb-4 flex items-center gap-2">
                            <Clock className="w-4 h-4 text-slate-500" />
                            Billing History
                        </h3>
                        <div className="space-y-4">
                            <div className="flex items-center justify-between group cursor-pointer">
                                <div>
                                    <p className="text-[11px] font-black text-white uppercase">INV-2024-001</p>
                                    <p className="text-[9px] text-slate-600">April 01, 2024</p>
                                </div>
                                <div className="text-right">
                                    <p className="text-[11px] font-black text-white">$299.00</p>
                                    <p className="text-[9px] text-emerald-500 font-bold uppercase">PAID</p>
                                </div>
                            </div>
                            <button className="w-full py-2 bg-white/5 hover:bg-white/10 rounded-lg text-[9px] font-black uppercase tracking-widest text-slate-400 transition-all">
                                View All Invoices
                            </button>
                        </div>
                    </Card>
                </div>
            </div>

            {/* Plans Section */}
            <div className="space-y-6">
                <h3 className="text-xs font-black uppercase tracking-[0.2em] text-slate-500 text-center">Available Tiers</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    {/* Free */}
                    <Card className={`p-8 flex flex-col ${plan.slug === 'free' ? 'border-emerald-500/30 ring-1 ring-emerald-500/20' : ''}`}>
                        <div className="mb-6">
                            <p className="text-[10px] font-black text-emerald-400 uppercase tracking-widest mb-1">Standard</p>
                            <h4 className="text-2xl font-black text-white uppercase">Free</h4>
                            <p className="text-3xl font-black text-white mt-4">$0 <span className="text-xs font-bold text-slate-600">/mo</span></p>
                        </div>
                        <ul className="space-y-4 flex-1">
                            {[
                                "1,000 Predictions /mo",
                                "2 Registered Models",
                                "1 Governance Report",
                                "Community Support"
                            ].map((feat, i) => (
                                <li key={i} className="flex items-center gap-3 text-xs font-bold text-slate-400">
                                    <Check className="w-4 h-4 text-emerald-500 shrink-0" />
                                    {feat}
                                </li>
                            ))}
                        </ul>
                        <button 
                            disabled={plan.slug === 'free' || submitting}
                            className={`mt-8 w-full py-4 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all ${
                                plan.slug === 'free' 
                                ? "bg-white/5 text-slate-600 cursor-not-allowed" 
                                : "bg-white/10 text-white hover:bg-white/20"
                            }`}
                        >
                            {plan.slug === 'free' ? "Active Plan" : "Downgrade"}
                        </button>
                    </Card>

                    {/* Pro */}
                    <Card className={`p-8 flex flex-col relative overflow-hidden ${plan.slug === 'pro' ? 'border-orange-500/50 ring-2 ring-orange-500/20' : 'border-orange-500/20'}`}>
                        <div className="absolute top-0 right-0 px-4 py-1 bg-orange-600 text-black text-[9px] font-black uppercase tracking-widest transform translate-x-[25%] translate-y-[50%] rotate-45 shadow-xl">
                            Popular
                        </div>
                        <div className="mb-6">
                            <p className="text-[10px] font-black text-orange-400 uppercase tracking-widest mb-1">Professional</p>
                            <h4 className="text-2xl font-black text-white uppercase">Pro</h4>
                            <p className="text-3xl font-black text-white mt-4">$299 <span className="text-xs font-bold text-slate-600">/mo</span></p>
                        </div>
                        <ul className="space-y-4 flex-1">
                            {[
                                "100,000 Predictions /mo",
                                "Unlimited Models",
                                "Unlimited Reports",
                                "2 Compliance Packs",
                                "10,000 Guardrail Evals",
                                "Priority Email Support"
                            ].map((feat, i) => (
                                <li key={i} className="flex items-center gap-3 text-xs font-bold text-slate-300">
                                    <Check className="w-4 h-4 text-orange-500 shrink-0" />
                                    {feat}
                                </li>
                            ))}
                        </ul>
                        <button 
                            onClick={() => handleUpgrade('pro')}
                            disabled={plan.slug === 'pro' || submitting}
                            className={`mt-8 w-full py-4 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all ${
                                plan.slug === 'pro' 
                                ? "bg-orange-600/10 text-orange-400 cursor-not-allowed border border-orange-500/20" 
                                : "bg-orange-600 hover:bg-orange-500 text-black shadow-lg shadow-orange-500/20"
                            }`}
                        >
                            {submitting ? <Loader2 className="w-4 h-4 animate-spin mx-auto" /> : (plan.slug === 'pro' ? "Active Plan" : "Upgrade to Pro")}
                        </button>
                    </Card>

                    {/* Enterprise */}
                    <Card className={`p-8 flex flex-col ${plan.slug === 'enterprise' ? 'border-blue-500/30 ring-1 ring-blue-500/20' : ''}`}>
                        <div className="mb-6">
                            <p className="text-[10px] font-black text-blue-400 uppercase tracking-widest mb-1">Institutional</p>
                            <h4 className="text-2xl font-black text-white uppercase">Enterprise</h4>
                            <p className="text-3xl font-black text-white mt-4">Custom <span className="text-xs font-bold text-slate-600">/year</span></p>
                        </div>
                        <ul className="space-y-4 flex-1">
                            {[
                                "Unlimited Predictions",
                                "Air-gapped Deployment",
                                "SLA Guarantees",
                                "Unlimited Compliance Packs",
                                "Dedicated Technical Account Manager",
                                "SSO & SCIM Integration"
                            ].map((feat, i) => (
                                <li key={i} className="flex items-center gap-3 text-xs font-bold text-slate-400">
                                    <Check className="w-4 h-4 text-blue-500 shrink-0" />
                                    {feat}
                                </li>
                            ))}
                        </ul>
                        <button 
                            disabled={plan.slug === 'enterprise' || submitting}
                            className={`mt-8 w-full py-4 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all ${
                                plan.slug === 'enterprise' 
                                ? "bg-blue-600/10 text-blue-400 cursor-not-allowed border border-blue-500/20" 
                                : "bg-white/10 text-white hover:bg-white/20"
                            }`}
                        >
                            {plan.slug === 'enterprise' ? "Active Plan" : "Contact Sales"}
                        </button>
                    </Card>
                </div>
            </div>

            {/* Additional Paid Products */}
            <div className="pt-8">
                <Card className="p-10 border-blue-500/20 bg-gradient-to-br from-blue-500/[0.03] to-transparent">
                    <div className="flex flex-col lg:flex-row items-center justify-between gap-10">
                        <div className="space-y-4 text-center lg:text-left">
                            <h3 className="text-xl font-black text-white uppercase tracking-tight flex items-center justify-center lg:justify-start gap-3">
                                <Shield className="w-6 h-6 text-blue-400" />
                                Professional Certification
                            </h3>
                            <p className="text-sm text-slate-400 max-w-xl leading-relaxed">
                                Need a cryptographic seal for your model's compliance? Issuing a professional 
                                governance certificate involves independent automated verification and 
                                permanent ledger archival.
                            </p>
                        </div>
                        <div className="bg-black/40 p-8 rounded-3xl border border-white/5 text-center min-w-[280px]">
                            <p className="text-[10px] font-black text-slate-600 uppercase tracking-widest mb-2">One-time Fee</p>
                            <p className="text-4xl font-black text-white mb-6">$500 <span className="text-xs text-slate-700">/cert</span></p>
                            <button className="w-full py-4 bg-blue-600 hover:bg-blue-500 text-white font-black rounded-xl text-[10px] uppercase tracking-widest transition-all shadow-lg shadow-blue-500/20">
                                Purchase Certificate
                            </button>
                        </div>
                    </div>
                </Card>
            </div>
        </div>
    );
}
