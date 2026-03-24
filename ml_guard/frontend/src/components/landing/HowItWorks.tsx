"use client";

import React from 'react';
import { motion } from 'framer-motion';
import { Box, Search, Rocket } from 'lucide-react';

const steps = [
    {
        title: "Register Model",
        desc: "Upload your model artifacts (.pkl, .h5, ONNX) and baseline data via our SDK or CLI. We fingerprint the architecture for traceability.",
        icon: Box,
        status: "Deployment Step 01"
    },
    {
        title: "Run Intelligent Test Suite",
        desc: "ML Guard automatically selects the optimal test battery based on your model profile—including bias, drift, and adversarial checks.",
        icon: Search,
        status: "Security Protocol 02"
    },
    {
        title: "Deploy with Confidence",
        desc: "Get an instant quality certificate. Only models passing your enterprise guardrails are permitted for production rollout.",
        icon: Rocket,
        status: "Live Production 03"
    }
];

export const HowItWorks = () => {
    return (
        <section id="how-it-works" className="py-32 bg-[#090A0C] relative">
            <div className="max-w-5xl mx-auto px-6">
                <div className="text-center mb-24">
                    <h2 className="text-4xl md:text-6xl font-black text-white tracking-tighter mb-4">
                        Zero Trust <br /> <span className="text-orange-500 italic">Deployment Pipeline</span>
                    </h2>
                    <p className="text-slate-500 font-medium">Three steps to algorithmic integrity.</p>
                </div>

                <div className="relative">
                    {/* Visual Connector Line */}
                    <div className="absolute left-1/2 top-0 bottom-0 w-px bg-white/5 -translate-x-1/2 hidden md:block" />

                    <div className="space-y-24">
                        {steps.map((step, i) => (
                            <motion.div
                                key={i}
                                initial={{ opacity: 0, x: i % 2 === 0 ? -50 : 50 }}
                                whileInView={{ opacity: 1, x: 0 }}
                                viewport={{ once: true }}
                                transition={{ duration: 0.8, delay: i * 0.2 }}
                                className={`flex flex-col md:flex-row items-center gap-10 md:gap-20 ${i % 2 !== 0 ? 'md:flex-row-reverse' : ''}`}
                            >
                                <div className="flex-1 text-center md:text-left">
                                    <span className="text-orange-500 text-[10px] font-black uppercase tracking-[0.4em] mb-4 block">
                                        {step.status}
                                    </span>
                                    <h3 className="text-3xl font-black text-white mb-6 tracking-tight">
                                        {step.title}
                                    </h3>
                                    <p className="text-slate-400 font-medium leading-relaxed">
                                        {step.desc}
                                    </p>
                                </div>

                                <div className="relative">
                                    <div className="w-20 h-20 bg-orange-500 rounded-3xl flex items-center justify-center relative z-10 shadow-[0_20px_40px_rgba(249,115,22,0.3)]">
                                        <step.icon className="w-10 h-10 text-black fill-current" />
                                    </div>
                                    <div className="absolute inset-0 bg-orange-500/20 blur-3xl animate-pulse"></div>
                                </div>

                                <div className="flex-1 hidden md:block" />
                            </motion.div>
                        ))}
                    </div>
                </div>
            </div>
        </section>
    );
};
