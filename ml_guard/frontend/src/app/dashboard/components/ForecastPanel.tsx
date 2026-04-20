"use client";
import React, { useState, useEffect } from "react";
import { 
    AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, 
    ResponsiveContainer, ReferenceLine, Label 
} from "recharts";
import { 
    TrendingUp, TrendingDown, Minus, AlertCircle, 
    Clock, Bell, ShieldAlert, LineChart
} from "lucide-react";

interface ForecastPoint {
    date: string;
    value: number;
    lower: number;
    upper: number;
}

interface ForecastResult {
    metric: string;
    forecast_points: ForecastPoint[];
    breach_date: string | null;
    breach_confidence: number;
    trend: string;
    recommendation: string;
    status: string;
}

interface ForecastSummary {
    model_id: string;
    summary: string;
    forecasts: Record<string, ForecastResult>;
}

export const TrendBadge = ({ trend }: { trend: string }) => {
    if (trend === "IMPROVING") return (
        <span className="flex items-center gap-1 text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded-full text-[9px] font-black uppercase tracking-tighter">
            <TrendingUp className="w-3 h-3" /> Improving
        </span>
    );
    if (trend === "DEGRADING") return (
        <span className="flex items-center gap-1 text-red-500 bg-red-500/10 border border-red-500/20 px-2 py-0.5 rounded-full text-[9px] font-black uppercase tracking-tighter">
            <TrendingDown className="w-3 h-3" /> Degrading
        </span>
    );
    return (
        <span className="flex items-center gap-1 text-slate-500 bg-white/5 border border-white/10 px-2 py-0.5 rounded-full text-[9px] font-black uppercase tracking-tighter">
            <Minus className="w-3 h-3" /> Stable
        </span>
    );
};

const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
        return (
            <div className="bg-[#0E1014] border border-white/10 p-3 rounded-xl shadow-2xl">
                <p className="text-[10px] text-slate-500 font-black mb-1.5 uppercase tracking-widest">{label}</p>
                <div className="space-y-1">
                    <p className="text-sm font-black text-orange-400">
                        Value: {payload[0].value.toFixed(4)}
                    </p>
                    <p className="text-[9px] text-slate-400 font-bold">
                        Range: {payload[0].payload.lower.toFixed(4)} — {payload[0].payload.upper.toFixed(4)}
                    </p>
                </div>
            </div>
        );
    }
    return null;
};

export default function ForecastPanel({ model_id }: { model_id: string }) {
    const [data, setData] = useState<ForecastSummary | null>(null);
    const [loading, setLoading] = useState(true);
    const [activeMetric, setActiveMetric] = useState("psi");

    useEffect(() => {
        const fetchForecast = async () => {
            try {
                const res = await fetch(`${process.env.NEXT_PUBLIC_API_BASE}/api/v1/forecast/${model_id}`);
                const json = await safeJson(res);
                setData(json);
                // Set first successful metric as active
                const firstMetric = Object.keys(json.forecasts).find(k => json.forecasts[k].status === "SUCCESS");
                if (firstMetric) setActiveMetric(firstMetric);
            } catch (err) {
                console.error("Failed to fetch forecast:", err);
            } finally {
                setLoading(false);
            }
        };
        if (model_id) fetchForecast();
    }, [model_id]);

    if (loading) return (
        <div className="flex flex-col items-center justify-center py-20 gap-4">
            <Clock className="w-10 h-10 text-orange-500/20 animate-pulse" />
            <p className="text-[10px] font-black text-slate-600 uppercase tracking-widest">Predicting Risk Trajectory...</p>
        </div>
    );

    if (!data || Object.values(data.forecasts).every(f => f.status === "INSUFFICIENT_DATA")) {
        return (
            <div className="p-8 border border-white/5 bg-black/20 rounded-2xl flex flex-col items-center text-center gap-3">
                <AlertCircle className="w-8 h-8 text-slate-700" />
                <p className="text-xs font-bold text-slate-400">Insufficient Audit History</p>
                <p className="text-[10px] text-slate-600 uppercase font-black">Requires minimum 3 audit cycles for trajectory forecasting.</p>
            </div>
        );
    }

    const currentForecast = data.forecasts[activeMetric];

    return (
        <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                {/* Metric Selector Sidebar */}
                <div className="lg:col-span-1 space-y-3">
                    <p className="text-[9px] font-black uppercase text-slate-600 tracking-[0.2em] mb-4">Governing Metrics</p>
                    {Object.entries(data.forecasts).map(([key, f]) => (
                        <button
                            key={key}
                            onClick={() => setActiveMetric(key)}
                            disabled={f.status !== "SUCCESS"}
                            className={`w-full text-left p-4 rounded-xl border transition-all ${
                                activeMetric === key 
                                    ? "bg-orange-500/5 border-orange-500/30 ring-1 ring-orange-500/20" 
                                    : f.status === "SUCCESS"
                                        ? "bg-white/[0.02] border-white/5 hover:border-white/10"
                                        : "opacity-40 cursor-not-allowed bg-transparent border-white/5"
                            }`}
                        >
                            <div className="flex items-center justify-between mb-1">
                                <span className="text-[10px] font-black uppercase text-slate-300">{key.replace('_', ' ')}</span>
                                {f.status === "SUCCESS" && <TrendBadge trend={f.trend} />}
                            </div>
                            {f.status === "SUCCESS" ? (
                                <div className="flex items-end gap-2">
                                    <span className="text-xl font-black text-white">
                                        {f.forecast_points[0]?.value.toFixed(3)}
                                    </span>
                                    {f.breach_date && (
                                        <span className="text-[9px] text-red-400 font-black mb-1 animate-pulse">
                                            BREACH IMMINENT
                                        </span>
                                    )}
                                </div>
                            ) : (
                                <span className="text-[9px] font-bold text-slate-600 uppercase">Insufficient History</span>
                            )}
                        </button>
                    ))}
                </div>

                {/* Main Forecast Chart Area */}
                <div className="lg:col-span-3 space-y-6">
                    <div className="bg-[#0E1014] border border-white/[0.07] rounded-2xl p-6">
                        <div className="flex items-center justify-between mb-8">
                            <div>
                                <h4 className="text-sm font-black text-white flex items-center gap-2">
                                    <LineChart className="w-4 h-4 text-orange-500" />
                                    {activeMetric.toUpperCase()} Risk Trajectory
                                </h4>
                                <p className="text-[10px] text-slate-500 font-bold mt-1">
                                    AI-powered 30-day forecast with 95% confidence interval
                                </p>
                            </div>
                            {currentForecast?.breach_date && (
                                <div className="bg-red-500/10 border border-red-500/20 px-4 py-2 rounded-xl">
                                    <p className="text-[8px] font-black text-red-400 uppercase tracking-widest">Breach Prediction</p>
                                    <p className="text-xs font-black text-red-500">{currentForecast.breach_date}</p>
                                </div>
                            )}
                        </div>

                        <div className="h-[300px] w-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={currentForecast.forecast_points}>
                                    <defs>
                                        <linearGradient id="colorVal" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#f97316" stopOpacity={0.3}/>
                                            <stop offset="95%" stopColor="#f97316" stopOpacity={0}/>
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
                                    <XAxis 
                                        dataKey="date" 
                                        stroke="#475569" 
                                        fontSize={9} 
                                        tickLine={false}
                                        axisLine={false}
                                        interval="preserveStartEnd"
                                    />
                                    <YAxis 
                                        stroke="#475569" 
                                        fontSize={9} 
                                        tickLine={false}
                                        axisLine={false}
                                        tickFormatter={(v: number) => v.toFixed(2)}
                                    />
                                    <Tooltip content={<CustomTooltip />} />
                                    <Area
                                        type="monotone"
                                        dataKey="value"
                                        stroke="#f97316"
                                        strokeWidth={3}
                                        fillOpacity={1}
                                        fill="url(#colorVal)"
                                    />
                                    <Area
                                        type="monotone"
                                        dataKey="upper"
                                        stroke="transparent"
                                        fill="#f9731610"
                                    />
                                    <Area
                                        type="monotone"
                                        dataKey="lower"
                                        stroke="transparent"
                                        fill="#f9731610"
                                    />
                                    {currentForecast.breach_date && (
                                        <ReferenceLine 
                                            x={currentForecast.breach_date} 
                                            stroke="#ef4444" 
                                            strokeDasharray="5 5"
                                            strokeWidth={2}
                                        >
                                            <Label 
                                                value="THRESHOLD BREACH" 
                                                position="top" 
                                                fill="#ef4444" 
                                                fontSize={8} 
                                                fontWeight="900" 
                                            />
                                        </ReferenceLine>
                                    )}
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Summary Card */}
                    <div className="bg-orange-500/5 border border-orange-500/10 rounded-2xl p-6 flex items-start gap-4">
                        <div className="bg-orange-500/10 p-3 rounded-xl">
                            <ShieldAlert className="w-5 h-5 text-orange-500" />
                        </div>
                        <div>
                            <p className="text-[10px] font-black text-orange-500 uppercase tracking-[0.2em] mb-1.5">Governance Advisory</p>
                            <p className="text-xs font-bold text-white mb-3 leading-relaxed">
                                {data.summary}
                            </p>
                            <div className="flex items-center gap-4">
                                <span className="text-[9px] font-black text-slate-500 uppercase flex items-center gap-1.5">
                                    <Bell className="w-3.5 h-3.5" /> Auto-Alerting Enabled
                                </span>
                                <span className="text-[9px] font-black text-slate-500 uppercase flex items-center gap-1.5">
                                    <Clock className="w-3.5 h-3.5" /> Updated 6h ago
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
