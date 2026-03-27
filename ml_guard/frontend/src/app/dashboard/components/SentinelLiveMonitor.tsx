"use client";
import React, { useState, useEffect, useRef } from "react";
import { 
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, 
    ResponsiveContainer, ReferenceLine, AreaChart, Area
} from "recharts";
import { 
    Activity, Zap, AlertTriangle, ShieldCheck, 
    Wifi, WifiOff, RefreshCw, AlertCircle
} from "lucide-react";

interface SentinelPoint {
    timestamp: string;
    avg_psi: number;
    is_breached: boolean;
}

export default function SentinelLiveMonitor({ model_id }: { model_id: string }) {
    const [points, setPoints] = useState<SentinelPoint[]>([]);
    const [connected, setConnected] = useState(false);
    const [lastPsi, setLastPsi] = useState<number | null>(null);
    const wsRef = useRef<WebSocket | null>(null);

    // Initial Fetch
    useEffect(() => {
        const fetchRecent = async () => {
            try {
                const res = await fetch(`${process.env.NEXT_PUBLIC_API_BASE}/api/v1/sentinel/${model_id}/live`);
                const data = await res.json();
                setPoints(data);
                if (data.length > 0) setLastPsi(data[data.length - 1].avg_psi);
            } catch (err) {
                console.error("Failed to load sentinel history:", err);
            }
        };
        fetchRecent();
    }, [model_id]);

    // WebSocket Connection
    useEffect(() => {
        const connect = () => {
            const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
            const url = `${protocol}//${process.env.NEXT_PUBLIC_API_BASE?.replace(/^https?:\/\//, "")}/api/v1/sentinel/live/ws/${model_id}`;
            
            const ws = new WebSocket(url);
            ws.onopen = () => setConnected(true);
            ws.onclose = () => {
                setConnected(false);
                setTimeout(connect, 3000); // Reconnect
            };
            ws.onmessage = (event) => {
                const update = JSON.parse(event.data);
                if (update.type === "SENTINEL_UPDATE") {
                    setPoints(prev => [...prev.slice(-49), {
                        timestamp: new Date().toISOString(),
                        avg_psi: update.avg_psi,
                        is_breached: update.is_breached
                    }]);
                    setLastPsi(update.avg_psi);
                }
            };
            wsRef.current = ws;
        };
        connect();
        return () => wsRef.current?.close();
    }, [model_id]);

    const isBreached = lastPsi !== null && lastPsi > 0.2;

    return (
        <div className="bg-[#0E1014] border border-white/[0.07] rounded-2xl p-6 overflow-hidden relative">
            {/* Background Grain/Visual */}
            <div className="absolute top-0 right-0 p-4">
                {connected ? (
                    <div className="flex items-center gap-1.5 text-emerald-500 bg-emerald-500/10 px-2 py-0.5 rounded-full border border-emerald-500/20 text-[8px] font-black uppercase">
                        <Wifi className="w-2.5 h-2.5" /> LIVE STREAMING
                    </div>
                ) : (
                    <div className="flex items-center gap-1.5 text-red-500 bg-red-500/10 px-2 py-0.5 rounded-full border border-red-500/20 text-[8px] font-black uppercase">
                        <WifiOff className="w-2.5 h-2.5" /> DISCONNECTED
                    </div>
                )}
            </div>

            <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6 mb-8">
                <div className="space-y-1">
                    <h4 className="text-sm font-black text-white flex items-center gap-2">
                        <Activity className="w-4 h-4 text-orange-500" />
                        Real-Time Drift Sentinel
                    </h4>
                    <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">
                        Sliding-Window PSI Surveillance (Window: 5k Samples)
                    </p>
                </div>

                <div className="flex items-center gap-8 bg-black/30 p-4 rounded-xl border border-white/5">
                    <div className="text-center">
                        <p className="text-[8px] font-black text-slate-600 uppercase mb-1">Current PSI</p>
                        <p className={`text-2xl font-black ${isBreached ? "text-red-500 animate-pulse" : "text-emerald-400"}`}>
                            {lastPsi?.toFixed(4) || "—"}
                        </p>
                    </div>
                    <div className="h-8 w-px bg-white/10" />
                    <div>
                        {isBreached ? (
                            <div className="flex flex-col items-end">
                                <span className="text-[10px] font-black text-red-500 flex items-center gap-1">
                                    <AlertTriangle className="w-3 h-3" /> DRIFT BREACHED
                                </span>
                                <span className="text-[8px] text-slate-500 font-bold uppercase mt-1">Rollback Recommended</span>
                            </div>
                        ) : (
                            <div className="flex flex-col items-end">
                                <span className="text-[10px] font-black text-emerald-400 flex items-center gap-1">
                                    <ShieldCheck className="w-3 h-3" /> STEADY STATE
                                </span>
                                <span className="text-[8px] text-slate-500 font-bold uppercase mt-1">Within Policy Threshold</span>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            <div className="h-[200px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={points}>
                        <defs>
                            <linearGradient id="livePsiGradient" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor={isBreached ? "#ef4444" : "#10b981"} stopOpacity={0.1}/>
                                <stop offset="95%" stopColor={isBreached ? "#ef4444" : "#10b981"} stopOpacity={0}/>
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#ffffff03" vertical={false} />
                        <XAxis hide dataKey="timestamp" />
                        <YAxis 
                            domain={[0, (dataMax: any) => Math.max(dataMax * 1.5, 0.4)]} 
                            stroke="#475569" 
                            fontSize={8} 
                            tickLine={false}
                            axisLine={false}
                            orientation="right"
                        />
                        <Tooltip 
                            content={({ active, payload }: any) => {
                                if (active && payload && payload.length) {
                                    return (
                                        <div className="bg-black border border-white/10 p-2 rounded text-[10px] font-black">
                                            PSI: <span className="text-orange-400">{payload[0].value.toFixed(4)}</span>
                                        </div>
                                    );
                                }
                                return null;
                            }}
                        />
                        <Area
                            type="stepAfter"
                            dataKey="avg_psi"
                            stroke={isBreached ? "#ef4444" : "#10b981"}
                            strokeWidth={2}
                            fillOpacity={1}
                            fill="url(#livePsiGradient)"
                            isAnimationActive={false}
                        />
                        <ReferenceLine 
                            y={0.2} 
                            stroke="#ef4444" 
                            strokeDasharray="4 4" 
                            strokeWidth={1}
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </div>

            <div className="mt-4 flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-1.5">
                        <div className="w-2 h-2 rounded-full bg-orange-500/20 border border-orange-500" />
                        <span className="text-[9px] font-black text-slate-600 uppercase">Input Sampling</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                        <div className="w-2 h-2 rounded-full bg-emerald-500/20 border border-emerald-500" />
                        <span className="text-[9px] font-black text-slate-600 uppercase">PSI Stream</span>
                    </div>
                </div>
                <button className="text-[9px] font-black text-slate-500 uppercase hover:text-white transition-colors flex items-center gap-1">
                    <RefreshCw className="w-2.5 h-2.5" /> Force Baseline Refresh
                </button>
            </div>
        </div>
    );
}
