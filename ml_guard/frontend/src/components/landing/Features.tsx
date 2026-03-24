"use client";

import React from 'react';
import { motion } from 'framer-motion';
import {
    Zap,
    Activity,
    Scale,
    Lock,
    FileCheck,
    GitBranch
} from 'lucide-react';

const features = [
    {
        title: "Pre-Deployment Quality Gates",
        desc: "Automated verification protocols that stop low-quality models before they impact production environments.",
        icon: Zap,
        color: "text-orange-500",
        bg: "bg-orange-500/10"
    },
    {
        title: "Drift Detection & Monitoring",
        desc: "Time-series telemetry tracking feature drift, label shift, and concept decay with sub-second latency.",
        icon: Activity,
        color: "text-blue-500",
        bg: "bg-blue-500/10"
    },
    {
        title: "Bias & Fairness Validation",
        desc: "Algorithmic scanning for protected attributes and disparate impact across demographic segments.",
        icon: Scale,
        color: "text-green-500",
        bg: "bg-green-500/10"
    },
    {
        title: "Robustness Testing",
        desc: "Stress-testing model integrity against adversarial perturbations and edge-case distributional shifts.",
        icon: Lock,
        color: "text-purple-500",
        bg: "bg-purple-500/10"
    },
    {
        title: "Compliance & Audit Reports",
        desc: "Exportable governance documentation mapped to global regulatory frameworks (EU AI Act, NIST).",
        icon: FileCheck,
        color: "text-red-500",
        bg: "bg-red-500/10"
    },
    {
        title: "CI/CD Integration",
        desc: "Native connectors for GitHub Actions, GitLab CI, and Jenkins to embed quality in your SDLC.",
        icon: GitBranch,
        color: "text-amber-500",
        bg: "bg-amber-500/10"
    }
];

export const Features = () => {
    return (
        <section id="features" className="py-32 px-6 relative overflow-hidden bg-[#090A0C]">
            <div className="max-w-7xl mx-auto">
                <div className="text-center mb-24">
                    <motion.p
                        initial={{ opacity: 0, y: 10 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        className="text-orange-500 text-[10px] font-black uppercase tracking-[0.5em] mb-4"
                    >
                        Core Capabilities
                    </motion.p>
                    <motion.h2
                        initial={{ opacity: 0, y: 10 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        transition={{ delay: 0.1 }}
                        className="text-4xl md:text-6xl font-black text-white tracking-tighter"
                    >
                        Governance for the <br />
                        <span className="italic">Agentic Era.</span>
                    </motion.h2>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                    {features.map((f, i) => (
                        <motion.div
                            key={i}
                            initial={{ opacity: 0, y: 30 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ duration: 0.5, delay: i * 0.1 }}
                            whileHover={{ y: -10 }}
                            className="bg-[#0F1115] border border-white/5 p-10 rounded-[2.5rem] hover:border-orange-500/30 transition-all group relative overflow-hidden"
                        >
                            {/* Gradient Border Mask */}
                            <div className="absolute inset-0 bg-gradient-to-br from-orange-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />

                            <div className={`w-14 h-14 ${f.bg} rounded-2xl flex items-center justify-center mb-8 group-hover:scale-110 transition-transform duration-500`}>
                                <f.icon className={`w-7 h-7 ${f.color}`} />
                            </div>

                            <h3 className="text-xl font-black text-white mb-4 tracking-tight group-hover:text-orange-500 transition-colors">
                                {f.title}
                            </h3>
                            <p className="text-slate-400 text-sm leading-relaxed font-medium">
                                {f.desc}
                            </p>

                            <div className="mt-8 pt-8 border-t border-white/5 flex items-center gap-2 text-[10px] font-black text-slate-500 uppercase tracking-widest opacity-0 group-hover:opacity-100 transition-all duration-500 translate-y-2 group-hover:translate-y-0">
                                Documentation <GitBranch className="w-3 h-3" />
                            </div>
                        </motion.div>
                    ))}
                </div>
            </div>
        </section>
    );
};
