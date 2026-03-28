"use client";
import React, { useState } from "react";
import { 
    FileText, Play, Loader2, CheckCircle, 
    Download, ExternalLink, AlertTriangle 
} from "lucide-react";

export default function ReportCardTrigger({ model_id }: { model_id: string }) {
    const [status, setStatus] = useState<"IDLE" | "PENDING" | "SUCCESS" | "FAILED">("IDLE");
    const [taskId, setTaskId] = useState<string | null>(null);
    const [certHash, setCertHash] = useState<string | null>(null);

    const startGeneration = async () => {
        setStatus("PENDING");
        try {
            const res = await fetch(`${process.env.NEXT_PUBLIC_API_BASE}/api/v1/reports/${model_id}/generate`, { method: "POST" });
            const data = await res.json();
            setTaskId(data.task_id);
            pollStatus(data.task_id);
        } catch (err) {
            console.error(err);
            setStatus("FAILED");
        }
    };

    const pollStatus = async (id: string) => {
        const interval = setInterval(async () => {
            const res = await fetch(`${process.env.NEXT_PUBLIC_API_BASE}/api/v1/reports/status/${id}`);
            const data = await res.json();
            if (data.status === "SUCCESS") {
                setCertHash(data.cert_hash);
                setStatus("SUCCESS");
                clearInterval(interval);
            } else if (data.status === "FAILED") {
                setStatus("FAILED");
                clearInterval(interval);
            }
        }, 3000);
    };

    return (
        <div className="bg-[#0E1014] border border-white/[0.07] rounded-3xl p-8 flex flex-col items-center text-center">
            <div className="w-16 h-16 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center mb-6">
                <FileText className="w-8 h-8 text-emerald-500" />
            </div>
            
            <h3 className="text-xl font-black text-white uppercase tracking-tight mb-2">
                Certified Report Card
            </h3>
            <p className="text-slate-500 text-xs font-bold leading-relaxed max-w-[280px] mb-8">
                Generate a multi-page PDF compliance certificate synthesizing all recent audit snapshots.
            </p>

            {status === "IDLE" && (
                <button 
                    onClick={startGeneration}
                    className="w-full bg-emerald-500 hover:bg-emerald-600 text-white font-black py-4 rounded-2xl text-[11px] uppercase tracking-widest flex items-center justify-center gap-2 transition-all shadow-xl shadow-emerald-500/10"
                >
                    <Play className="w-4 h-4" /> Synthesize Report
                </button>
            )}

            {status === "PENDING" && (
                <div className="w-full bg-white/5 border border-white/10 p-4 rounded-2xl flex items-center justify-center gap-3">
                    <Loader2 className="w-5 h-5 text-emerald-500 animate-spin" />
                    <span className="text-[10px] font-black text-slate-300 uppercase tracking-widest">Generating Certificate...</span>
                </div>
            )}

            {status === "SUCCESS" && (
                <div className="w-full space-y-4">
                    <div className="bg-emerald-500/10 border border-emerald-500/20 p-4 rounded-2xl flex items-center justify-center gap-3">
                        <CheckCircle className="w-5 h-5 text-emerald-500" />
                        <span className="text-[10px] font-black text-emerald-500 uppercase tracking-widest">Synthesis Complete</span>
                    </div>
                    <div className="flex gap-4">
                        <a 
                            href={`${process.env.NEXT_PUBLIC_API_BASE}/api/v1/reports/download/${cert_hash}`}
                            target="_blank"
                            className="flex-1 bg-white text-black font-black py-3 rounded-xl text-[10px] uppercase tracking-widest flex items-center justify-center gap-2"
                        >
                            <Download className="w-4 h-4" /> Download PDF
                        </a>
                        <a 
                            href={`/verify/${cert_hash}`}
                            target="_blank"
                            className="flex-1 bg-white/5 border border-white/10 text-white font-black py-3 rounded-xl text-[10px] uppercase tracking-widest flex items-center justify-center gap-2"
                        >
                            <ExternalLink className="w-4 h-4" /> Verify Link
                        </a>
                    </div>
                </div>
            )}

            {status === "FAILED" && (
                <div className="w-full bg-red-500/10 border border-red-500/20 p-4 rounded-2xl flex items-center justify-center gap-3 text-red-500 text-[10px] font-black uppercase tracking-widest">
                    <AlertTriangle className="w-5 h-5" /> Synthesis Failed
                </div>
            )}
        </div>
    );
}
