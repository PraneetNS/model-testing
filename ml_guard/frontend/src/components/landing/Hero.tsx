"use client";

import React, { useRef } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import { ChevronRight, Play, Database, ShieldCheck, Zap } from 'lucide-react';
import Link from 'next/link';

export const Hero = () => {
    const containerRef = useRef(null);
    const { scrollYProgress } = useScroll({
        target: containerRef,
        offset: ["start start", "end start"]
    });

    const yValue = useTransform(scrollYProgress, [0, 1], ["0%", "50%"]);
    const opacityValue = useTransform(scrollYProgress, [0, 0.5], [1, 0]);

    return (
        <section
            ref={containerRef}
            className="relative min-h-screen flex items-center justify-center pt-20 overflow-hidden"
        >
            {/* Animated Grid Background */}
            <div className="absolute inset-0 z-0">
                <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:40px_40px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_50%,#000_70%,transparent_100%)]"></div>
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[600px] bg-orange-500/10 rounded-full blur-[120px] animate-pulse"></div>
            </div>

            <motion.div
                style={{ y: yValue, opacity: opacityValue }}
                className="relative z-10 text-center max-w-5xl px-6"
            >
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8 }}
                    className="inline-flex items-center gap-2 bg-orange-500/10 text-orange-500 text-[10px] font-black tracking-[0.4em] uppercase px-6 py-2 rounded-full border border-orange-500/20 mb-10 shadow-[0_0_20px_rgba(249,115,22,0.1)]"
                >
                    <span className="w-1.5 h-1.5 rounded-full bg-orange-500 animate-ping"></span>
                    Enterprise ML Governance v2.0
                </motion.div>

                <motion.h1
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 1, delay: 0.2 }}
                    className="text-6xl md:text-8xl font-black text-white tracking-tighter leading-[0.9] mb-8"
                >
                    Enterprise ML Quality <br />
                    <span className="text-transparent bg-clip-text bg-gradient-to-r from-orange-600 via-orange-400 to-white italic">
                        Starts Here.
                    </span>
                </motion.h1>

                <motion.p
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 1, delay: 0.4 }}
                    className="text-slate-400 text-lg md:text-xl max-w-2xl mx-auto mb-12 font-medium leading-relaxed"
                >
                    Ensure your AI models are production-ready with automated quality gates,
                    deep-drift telemetry, and regulatory compliance auditing.
                </motion.p>

                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 1, delay: 0.6 }}
                    className="flex flex-col sm:flex-row items-center justify-center gap-6"
                >
                    <Link
                        href="/login?signup=true"
                        className="w-full sm:w-auto bg-orange-500 text-black px-10 py-5 rounded-2xl text-[13px] font-black uppercase tracking-widest hover:bg-orange-400 hover:scale-105 active:scale-95 transition-all flex items-center justify-center gap-3 shadow-[0_20px_40px_rgba(249,115,22,0.2)]"
                    >
                        Initalize Deployment
                        <ChevronRight className="w-5 h-5" />
                    </Link>
                    <button className="w-full sm:w-auto bg-white/5 border border-white/10 text-white px-10 py-5 rounded-2xl text-[13px] font-black uppercase tracking-widest hover:bg-white/10 transition-all flex items-center justify-center gap-3 group">
                        <Play className="w-4 h-4 text-orange-500 fill-current group-hover:scale-110 transition-transform" />
                        Explore Telemetry
                    </button>
                </motion.div>

                {/* KPI Ribbon */}
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 2, delay: 1 }}
                    className="mt-20 flex flex-wrap justify-center gap-10 md:gap-20 opacity-50 grayscale hover:grayscale-0 transition-all duration-700"
                >
                    {[
                        { icon: Database, val: "10B+", label: "Features Analyzed" },
                        { icon: ShieldCheck, val: "99.9%", label: "Model Reliability" },
                        { icon: Zap, val: "<10ms", label: "Latency Impact" }
                    ].map((item, i) => (
                        <div key={i} className="flex items-center gap-3 text-left">
                            <item.icon className="w-5 h-5 text-orange-500" />
                            <div>
                                <p className="text-white font-black text-xl">{item.val}</p>
                                <p className="text-[9px] font-bold uppercase tracking-widest text-slate-500">{item.label}</p>
                            </div>
                        </div>
                    ))}
                </motion.div>
            </motion.div>

            {/* Scroll Progress Indicator */}
            <motion.div
                style={{ scaleY: scrollYProgress }}
                className="fixed left-0 top-0 bottom-0 w-1 bg-orange-500 origin-top z-50 rounded-r-full shadow-[0_0_20px_rgba(249,115,22,0.5)]"
            />
        </section>
    );
};
