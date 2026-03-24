"use client";

import React from 'react';
import { motion } from 'framer-motion';
import { Cpu, Database, BarChart3, AlertTriangle, CheckCircle2 } from 'lucide-react';

export const Frameworks = () => {
    const logos = [
        "scikit-learn", "TensorFlow", "PyTorch", "XGBoost", "MLflow", "ONNX"
    ];

    return (
        <section className="py-24 bg-[#090A0C] border-y border-white/5">
            <div className="max-w-7xl mx-auto px-6">
                <p className="text-center text-[10px] font-black text-slate-500 uppercase tracking-[0.5em] mb-12">
                    Native Framework Compatibility
                </p>
                <div className="flex flex-wrap justify-center items-center gap-12 md:gap-20">
                    {logos.map((logo, i) => (
                        <motion.span
                            key={i}
                            initial={{ opacity: 0 }}
                            whileInView={{ opacity: 1 }}
                            viewport={{ once: true }}
                            transition={{ delay: i * 0.1 }}
                            className="text-2xl font-black text-white/20 hover:text-white/60 transition-colors cursor-default"
                        >
                            {logo}
                        </motion.span>
                    ))}
                </div>
            </div>
        </section>
    );
};

export const DashboardPreview = () => {
    return (
        <section className="py-32 bg-[#090A0C] overflow-hidden">
            <div className="max-w-7xl mx-auto px-6">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-20 items-center">
                    <div>
                        <h2 className="text-4xl md:text-6xl font-black text-white tracking-tighter mb-8 italic">
                            Operational <br /> Intelligence.
                        </h2>
                        <p className="text-slate-400 text-lg font-medium mb-12">
                            Visualize your entire model fleet's health in a single glass pane.
                            From risk scores to detailed drift logs, governance has never been this intuitive.
                        </p>

                        <div className="space-y-6">
                            {[
                                { icon: BarChart3, title: "Aggregated Quality Scoring" },
                                { icon: AlertTriangle, title: "Automated Risk Classification" },
                                { icon: CheckCircle2, title: "Real-time Deployment Telemetry" }
                            ].map((item, i) => (
                                <motion.div
                                    key={i}
                                    initial={{ opacity: 0, x: -20 }}
                                    whileInView={{ opacity: 1, x: 0 }}
                                    transition={{ delay: i * 0.1 }}
                                    className="flex items-center gap-4 bg-white/5 border border-white/5 p-4 rounded-2xl"
                                >
                                    <item.icon className="w-5 h-5 text-orange-500" />
                                    <span className="text-sm font-black text-white uppercase tracking-widest">{item.title}</span>
                                </motion.div>
                            ))}
                        </div>
                    </div>

                    <div className="relative">
                        {/* Mock Dashboard Animation */}
                        <motion.div
                            initial={{ opacity: 0, scale: 0.9, rotateY: -20 }}
                            whileInView={{ opacity: 1, scale: 1, rotateY: 0 }}
                            viewport={{ once: true }}
                            transition={{ duration: 1 }}
                            className="bg-[#0F1115] border border-white/10 rounded-[3rem] p-8 shadow-2xl relative z-10"
                        >
                            <div className="flex justify-between items-center mb-10">
                                <div>
                                    <p className="text-orange-500 text-[9px] font-black uppercase tracking-widest">Model Health</p>
                                    <h4 className="text-white font-black text-xl">Governance Overview</h4>
                                </div>
                                <div className="text-right">
                                    <span className="text-4xl font-black text-orange-500">98</span>
                                    <p className="text-[9px] text-slate-500 font-bold uppercase tracking-widest">Quality Index</p>
                                </div>
                            </div>

                            <div className="grid grid-cols-2 gap-4 mb-8">
                                <div className="bg-white/5 rounded-2xl p-6 border border-white/5">
                                    <p className="text-[8px] text-slate-500 font-black uppercase mb-2">Drift Level</p>
                                    <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden">
                                        <motion.div
                                            initial={{ width: 0 }}
                                            whileInView={{ width: '65%' }}
                                            className="h-full bg-orange-500"
                                        />
                                    </div>
                                    <p className="text-white font-black mt-2">Nominal</p>
                                </div>
                                <div className="bg-white/5 rounded-2xl p-6 border border-white/5">
                                    <p className="text-[8px] text-slate-500 font-black uppercase mb-2">Bias Status</p>
                                    <p className="text-green-500 font-black">Passed ✓</p>
                                </div>
                            </div>

                            <div className="space-y-3">
                                {[1, 2, 3].map(i => (
                                    <div key={i} className="h-3 bg-white/5 rounded-full w-full relative">
                                        <motion.div
                                            initial={{ width: 0 }}
                                            whileInView={{ width: `${Math.random() * 80 + 20}%` }}
                                            className="h-full bg-blue-500/20 rounded-full"
                                        />
                                    </div>
                                ))}
                            </div>
                        </motion.div>

                        {/* Background Blobs */}
                        <div className="absolute -top-20 -right-20 w-64 h-64 bg-orange-500/20 rounded-full blur-[80px] animate-pulse"></div>
                        <div className="absolute -bottom-20 -left-20 w-64 h-64 bg-blue-500/10 rounded-full blur-[80px]"></div>
                    </div>
                </div>
            </div>
        </section>
    );
};
