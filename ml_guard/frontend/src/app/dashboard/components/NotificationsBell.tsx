"use client";
import React, { useState, useEffect, useCallback } from "react";
import { Bell, AlertTriangle, AlertCircle, Info, X, Clock, CheckCircle2 } from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "";

export default function NotificationsBell() {
    const [isOpen, setIsOpen] = useState(false);
    const [alerts, setAlerts] = useState<any[]>([]);
    const [unreadCount, setUnreadCount] = useState(0);
    const [lastChecked, setLastChecked] = useState<Date>(new Date());

    const fetchAlerts = useCallback(async () => {
        try {
            const r = await fetch(`${API_BASE}/api/v1/alerts/events?limit=10`);
            const d = await r.json();
            const list = Array.isArray(d) ? d : [];
            setAlerts(list);
            
            // Count new since last toggle or last known
            if (!isOpen) {
                const newCount = list.filter((a: any) => new Date(a.created_at) > lastChecked).length;
                setUnreadCount(newCount);
            }
        } catch (e) {
            console.error("Failed to fetch alerts", e);
        }
    }, [isOpen, lastChecked]);

    useEffect(() => {
        fetchAlerts();
        const interval = setInterval(fetchAlerts, 30000); // 30s polling
        return () => clearInterval(interval);
    }, [fetchAlerts]);

    const toggleOpen = () => {
        if (!isOpen) {
            setUnreadCount(0);
            setLastChecked(new Date());
        }
        setIsOpen(!isOpen);
    };

    const getIcon = (severity: string) => {
        switch (severity) {
            case "CRITICAL": return <AlertCircle className="w-4 h-4 text-red-400" />;
            case "WARNING": return <AlertTriangle className="w-4 h-4 text-amber-400" />;
            case "INFO": return <Info className="w-4 h-4 text-blue-400" />;
            default: return <Bell className="w-4 h-4 text-slate-400" />;
        }
    };

    const getColor = (severity: string) => {
        switch (severity) {
            case "CRITICAL": return "border-red-500/20 bg-red-500/5";
            case "WARNING": return "border-amber-500/20 bg-amber-500/5";
            case "INFO": return "border-blue-500/20 bg-blue-500/5";
            default: return "border-white/5 bg-white/[0.02]";
        }
    };

    return (
        <div className="relative">
            <button
                onClick={toggleOpen}
                className={`w-10 h-10 rounded-xl border flex items-center justify-center transition-all cursor-pointer group relative ${
                    isOpen ? "bg-orange-600/20 border-orange-500/40 text-orange-400" : "bg-white/[0.02] border-white/5 text-slate-500 hover:text-white hover:bg-white/[0.05]"
                }`}
            >
                <Bell className="w-4 h-4" />
                {unreadCount > 0 && (
                    <span className="absolute -top-1 -right-1 w-4 h-4 bg-orange-600 text-black text-[9px] font-black rounded-full flex items-center justify-center animate-pulse border-2 border-[#050608]">
                        {unreadCount}
                    </span>
                )}
            </button>

            {isOpen && (
                <>
                    <div className="fixed inset-0 z-[100]" onClick={() => setIsOpen(false)} />
                    <div className="absolute right-0 mt-3 w-80 bg-[#0E1014] border border-white/10 rounded-2xl shadow-2xl z-[101] overflow-hidden animate-in fade-in zoom-in-95 duration-200 origin-top-right">
                        <div className="px-5 py-4 border-b border-white/5 flex items-center justify-between bg-white/[0.02]">
                            <h3 className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-400">Live Notifications</h3>
                            <button onClick={() => setIsOpen(false)} className="text-slate-600 hover:text-white transition-colors">
                                <X className="w-4 h-4" />
                            </button>
                        </div>
                        <div className="max-h-[400px] overflow-y-auto no-scrollbar divide-y divide-white/[0.03]">
                            {alerts.length === 0 ? (
                                <div className="p-10 text-center flex flex-col items-center gap-3">
                                    <CheckCircle2 className="w-8 h-8 text-slate-800" />
                                    <p className="text-[9px] font-black uppercase text-slate-700 tracking-widest leading-relaxed">System is stable.<br/>No active alerts.</p>
                                </div>
                            ) : (
                                alerts.map((alert) => (
                                    <div key={alert.id} className={`p-4 hover:bg-white/[0.02] transition-colors relative group`}>
                                        <div className="flex items-start gap-3">
                                            <div className="mt-1 shrink-0">{getIcon(alert.severity)}</div>
                                            <div className="flex-1 min-w-0">
                                                <div className="flex items-center justify-between mb-1">
                                                    <span className={`text-[8px] font-black uppercase px-1.5 py-0.5 rounded border ${getColor(alert.severity)}`}>
                                                        {alert.severity}
                                                    </span>
                                                    <span className="text-[8px] font-mono text-slate-600 flex items-center gap-1">
                                                        <Clock className="w-2 h-2" />
                                                        {new Date(alert.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                                    </span>
                                                </div>
                                                <p className="text-xs font-bold text-slate-200 leading-tight mb-1">{alert.message}</p>
                                                <p className="text-[9px] text-slate-500 font-medium truncate">{alert.rule_name || "System Monitor"}</p>
                                            </div>
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                        {alerts.length > 0 && (
                            <div className="px-5 py-3 bg-black/40 border-t border-white/5 text-center">
                                <button className="text-[9px] font-black uppercase tracking-widest text-orange-500 hover:text-orange-400 transition-colors">
                                    View Alert Manager
                                </button>
                            </div>
                        )}
                    </div>
                </>
            )}
        </div>
    );
}
